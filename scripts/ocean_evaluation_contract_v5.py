"""Pure evaluation contract helpers; no datasets, models, fitting, or answer-file I/O.

Only the CLI reads the frozen config. All numerical helpers consume explicit caller
inputs. Synthetic PASS does not establish real-data support or feature leakage safety.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATUS = "CONFIG_FROZEN_PENDING_TRAIN_ONLY_SUPPORT_AUDIT"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/evaluation/ocean_forward_v5.json"


def _aware(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("timestamps must be nonmissing and timezone-aware")
    return stamp.tz_convert("UTC")


def _times(values):
    return pd.DatetimeIndex([_aware(value) for value in values])


def _time_key_guard(frame, column, other_keys):
    canonical = frame[list(other_keys)].reset_index(drop=True).copy()
    canonical[column] = _times(frame[column])
    if canonical.duplicated([*other_keys, column]).any():
        raise ValueError("duplicate normalized timestamp keys")


def _keys(frame, keys):
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("explicit unique key columns are required")
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("empty table or duplicate columns")
    if set(keys) - set(frame.columns):
        raise ValueError("missing key columns")
    if frame[list(keys)].isna().any().any() or frame.duplicated(list(keys)).any():
        raise ValueError("missing or duplicate keys")
    for column in keys:
        if pd.api.types.is_numeric_dtype(frame[column]):
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError("nonfinite keys")
        elif frame[column].map(lambda value: isinstance(value, str) and not value.strip()).any():
            raise ValueError("empty keys")


def _values(values, binary=False):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("evaluation values must be nonempty, one-dimensional and finite")
    if binary and not np.isin(array, [0, 1]).all():
        raise ValueError("F1 requires binary labels and predictions")
    return array


def pooled_metric(truth, prediction, metric):
    if metric not in {"f1", "rmse"}:
        raise ValueError("unknown metric")
    y = _values(truth, binary=metric == "f1")
    p = _values(prediction, binary=metric == "f1")
    if y.shape != p.shape:
        raise ValueError("evaluation value lengths differ")
    if metric == "rmse":
        value = float(np.sqrt(np.mean(np.square(p - y))))
    else:
        tp = np.sum((y == 1) & (p == 1))
        fp = np.sum((y == 0) & (p == 1))
        fn = np.sum((y == 1) & (p == 0))
        denominator = 2 * tp + fp + fn
        value = 0.0 if denominator == 0 else float(2 * tp / denominator)
    if not np.isfinite(value):
        raise ValueError("nonfinite metric result")
    return value


def aligned_values(truth, control, candidate, keys, *, p3=False):
    """Reject differences instead of inner-joining away difficult/missing rows."""
    for frame in (truth, control, candidate):
        _keys(frame, keys)
        if "value" not in frame:
            raise ValueError("missing evaluation value column")
        _values(frame["value"])
    expected = truth[list(keys)].reset_index(drop=True)
    for frame in (control, candidate):
        if not expected.equals(frame[list(keys)].reset_index(drop=True)):
            raise ValueError("evaluation key/order mismatch")
    if p3:
        if not {"station", "anchor_time", "lead_hours"}.issubset(keys):
            raise ValueError("P3 requires station/anchor_time/lead_hours keys")
        for _, rows in truth.groupby(["station", "anchor_time"], sort=False):
            if sorted(rows["lead_hours"].tolist()) != [3, 6, 9, 12, 18, 24]:
                raise ValueError("P3 requires all six leads for every anchor")
    return tuple(frame["value"].to_numpy(dtype=float) for frame in (truth, control, candidate))


def validate_contract(contract):
    if contract.get("contract_id") != "ocean_forward_v5" or contract.get("status") != STATUS:
        raise ValueError("wrong contract identity or unsupported readiness claim")
    execution = contract["execution"]
    if execution != {
        "fit_budget": 0,
        "official_input_access": False,
        "csv_generation": False,
        "upload": False,
        "support_audit": "NOT_RUN",
        "model_execution_ready": False,
    }:
        raise ValueError("this contract does not authorize real-data execution")
    common = contract["common"]
    boot = common["bootstrap"]
    if (
        boot["resamples"],
        boot["seed"],
        boot["ci_quantiles"],
        boot["paired"],
        boot["ties_improve"],
        boot["minimum_clusters"],
    ) != (2000, 20260906, [0.05, 0.95], True, False, 2):
        raise ValueError("bootstrap contract changed")
    selection = common["selection"]
    if (
        selection["bootstrap_probability_hard_gate"] is not None
        or selection["automatic_promotion_or_submission"]
        or selection["candidate_retention"] != "strict_primary_pooled_mean_improvement"
    ):
        raise ValueError("unsupported selection or promotion rule")
    expected = {
        "P1": ("forward", "Asia/Seoul", 4, 1),
        "P2": ("two_sided_purged", "Asia/Seoul", 8, 0),
        "P3": ("forward", "UTC", 6, 1),
    }
    exact_dates = {
        "P1": "2024-01-01/2024-07-01 2024-07-01/2025-01-01 "
        "2025-01-01/2025-07-01 2025-07-01/2026-01-01",
        "P2": "2024-05-01/2024-07-01 2024-07-01/2024-09-01 "
        "2024-09-01/2024-11-01 2024-11-01/2025-01-01 "
        "2025-04-01/2025-06-01 2025-06-01/2025-08-01 "
        "2025-08-01/2025-09-01 2025-11-01/2026-01-01",
        "P3": "2024-01-01/2024-04-01 2024-04-01/2024-07-01 "
        "2024-07-01/2024-10-01 2024-10-01/2025-01-01 "
        "2025-01-01/2025-04-01 2025-04-01/2025-07-01",
    }
    for problem, (direction, zone, count, warmup) in expected.items():
        part = contract[problem]
        folds = part["folds"]
        if (part["direction"], part["timezone"], len(folds)) != (direction, zone, count):
            raise ValueError("wrong direction, timezone or fold count")
        if len({fold["id"] for fold in folds}) != count:
            raise ValueError("duplicate fold IDs")
        if [fold["warmup"] for fold in folds] != [bool(warmup)] + [False] * (count - 1):
            raise ValueError("wrong warm-up allocation")
        previous_end = None
        for fold in folds:
            start, end = _aware(fold["start"]), _aware(fold["end"])
            if start >= end or (previous_end is not None and start < previous_end):
                raise ValueError("invalid temporal order or overlapping folds")
            previous_end = end
        actual_dates = [
            "/".join(_aware(fold[key]).tz_convert(zone).isoformat() for key in ("start", "end"))
            for fold in folds
        ]
        expected_dates = [
            "/".join(pd.Timestamp(day, tz=zone).isoformat() for day in pair.split("/"))
            for pair in exact_dates[problem].split()
        ]
        if actual_dates != expected_dates:
            raise ValueError("frozen fold dates changed")
    if (
        contract["P1"]["purge_days"],
        contract["P2"]["purge_days"],
        contract["P3"]["purge_hours"],
    ) != (21, 7, 78):
        raise ValueError("purge contract changed")
    if (
        contract["P1"]["primary"]["name"] != "calendar_H1_2025_pooled_f1"
        or contract["P2"]["primary"]["fold"] != "B3"
        or contract["P3"]["primary"]["name"] != "all_forward_oof_unweighted_pooled_rmse"
        or contract["P2"]["outage"]["last_days"] != 17
        or contract["P3"]["diagnostics"]["onset_weighted_rmse"] != "NOT_ENABLED"
        or contract["P3"]["diagnostics"]["greedy"] != "NOT_ENABLED"
    ):
        raise ValueError("primary or disabled diagnostic contract changed")
    return contract


def load_contract(path=DEFAULT_CONFIG):
    return validate_contract(json.loads(Path(path).read_text(encoding="utf-8")))


def _fold(contract, problem, fold_id):
    rows = [fold for fold in contract[problem]["folds"] if fold["id"] == fold_id]
    if len(rows) != 1 or rows[0]["warmup"]:
        raise ValueError("unknown fold or warm-up has no OOF")
    return rows[0], _aware(rows[0]["start"]), _aware(rows[0]["end"])


def _split_result(train, validation):
    train, validation = np.asarray(train, dtype=bool), np.asarray(validation, dtype=bool)
    if not train.any() or not validation.any():
        raise ValueError("empty training or validation support")
    if np.any(train & validation):
        raise ValueError("train/validation overlap")
    return {"train": train, "validation": validation}


def p1_run_metadata(frame, contract):
    """Positive run crosses year/fold boundaries but never a true cadence gap."""
    _keys(frame, ["station", "layer", "time"])
    _time_key_guard(frame, "time", ["station", "layer"])
    if "label" not in frame:
        raise ValueError("P1 split metadata requires train-only labels")
    labels = _values(frame["label"], binary=True)
    times = _times(frame["time"])
    work = frame[["station", "layer"]].reset_index(drop=True).copy()
    work["time"] = times
    run_id = np.full(len(frame), -1, dtype=int)
    owners = np.full(len(frame), "", dtype=object)
    ends = np.full(len(frame), np.iinfo(np.int64).min, dtype=np.int64)
    next_id = 0
    cadence = pd.Timedelta(minutes=contract["P1"]["positive_runs"]["cadence_minutes"])
    for _, group in work.groupby(["station", "layer"], sort=False):
        positions = group.sort_values("time").index.to_numpy()
        active = []
        previous = None
        runs = []
        for pos in positions:
            if labels[pos] == 0 or (previous is not None and times[pos] - previous != cadence):
                if active:
                    runs.append(active)
                    active = []
            if labels[pos] == 1:
                active.append(pos)
            previous = times[pos]
        if active:
            runs.append(active)
        for run in runs:
            start, end = times[run[0]], times[run[-1]]
            owner = "OUTSIDE_CONTRACT"
            for fold in contract["P1"]["folds"]:
                if _aware(fold["start"]) <= start < _aware(fold["end"]):
                    owner = fold["id"]
                    break
            run_id[run], owners[run], ends[run] = next_id, owner, end.value
            next_id += 1
    return {"run_id": run_id, "owner": owners, "end_ns": ends}


def p1_split(frame, fold_id, contract):
    _, start, end = _fold(contract, "P1", fold_id)
    meta = p1_run_metadata(frame, contract)
    times = _times(frame["time"])
    labels = _values(frame["label"], binary=True)
    cutoff = start - pd.Timedelta(days=contract["P1"]["purge_days"])
    train = (times < cutoff) & ((labels == 0) | (meta["end_ns"] < cutoff.value))
    validation = ((labels == 0) & (times >= start) & (times < end)) | (
        (labels == 1) & (meta["owner"] == fold_id)
    )
    return {**_split_result(train, validation), **meta}


def p2_split(frame, fold_id, contract, keys):
    _keys(frame, keys)
    if "time" not in keys:
        raise ValueError("P2 keys must include time")
    _time_key_guard(frame, "time", [key for key in keys if key != "time"])
    _, start, end = _fold(contract, "P2", fold_id)
    times = _times(frame["time"])
    purge = pd.Timedelta(days=contract["P2"]["purge_days"])
    train = (times < start - purge) | (times >= end + purge)
    validation = (times >= start) & (times < end)
    outage = validation & (times >= end - pd.Timedelta(days=17))
    return {**_split_result(train, validation), "outage": np.asarray(outage)}


def p2_joint_t5_mask(public_channels, fold_id, contract):
    """Long-form public input: time/channel/temp/psal. Never mutates the caller.

    This only masks T5. Target-layer masking and recomputing every derived feature
    remain mandatory adapter duties, not a claim made by this helper.
    """
    _keys(public_channels, ["time", "channel"])
    _time_key_guard(public_channels, "time", ["channel"])
    if not {"temp", "psal"}.issubset(public_channels):
        raise ValueError("both temp and psal public channels are required")
    _, _, end = _fold(contract, "P2", fold_id)
    times = _times(public_channels["time"])
    mask = (times >= end - pd.Timedelta(days=17)) & (times < end)
    mask &= public_channels["channel"].eq("T5").to_numpy()
    result = public_channels.copy(deep=True)
    result.loc[mask, ["temp", "psal"]] = np.nan
    return result


def p3_split(anchors, fold_id, contract):
    _keys(anchors, ["station", "anchor_time"])
    _time_key_guard(anchors, "anchor_time", ["station"])
    if "episode_id" not in anchors or anchors["episode_id"].isna().any():
        raise ValueError("train-only episode metadata is required")
    if anchors["episode_id"].astype(str).str.strip().eq("").any():
        raise ValueError("unknown episode ID")
    _episode_id_guard(anchors["episode_id"])
    _, start, end = _fold(contract, "P3", fold_id)
    times = _times(anchors["anchor_time"])
    validation = (times >= start) & (times < end)
    cutoff = start - pd.Timedelta(hours=contract["P3"]["purge_hours"])
    episode_keys = list(zip(anchors["station"], anchors["episode_id"], strict=True))
    held_out = {key for key, include in zip(episode_keys, validation, strict=True) if include}
    train = (times < cutoff) & np.array([key not in held_out for key in episode_keys])
    result = _split_result(train, validation)
    for station in anchors.loc[validation, "station"].unique():
        same = anchors["station"].eq(station).to_numpy()
        if np.any(train & same):
            last_training_target = times[train & same].max() + pd.Timedelta(hours=24)
            first_validation_context = times[validation & same].min() - pd.Timedelta(hours=48)
            if last_training_target >= first_validation_context:
                raise ValueError("P3 context/target footprint overlap")
    return result


def primary_mask(frame, problem, contract):
    if frame.empty:
        raise ValueError("empty OOF table")
    time_column = "anchor_time" if problem == "P3" else "time"
    times = _times(frame[time_column])
    if problem == "P1":
        part = contract[problem]["primary"]
        mask = (times >= _aware(part["start"])) & (times < _aware(part["end"]))
    elif problem == "P2":
        _, start, end = _fold(contract, "P2", "B3")
        mask = (times >= start) & (times < end)
    elif problem == "P3":
        mask = np.zeros(len(frame), dtype=bool)
        for fold in contract["P3"]["folds"]:
            if not fold["warmup"]:
                mask |= (times >= _aware(fold["start"])) & (times < _aware(fold["end"]))
    else:
        raise ValueError("unknown problem")
    if not mask.any():
        raise ValueError("primary evaluation values missing")
    return np.asarray(mask)


def bootstrap_groups(frame, problem, contract):
    if frame.empty:
        raise ValueError("empty bootstrap metadata")
    if problem in {"P1", "P2"}:
        times = _times(frame["time"]).tz_convert("Asia/Seoul")
        if problem == "P1":
            return times.strftime("%Y-%m-%d").tolist()
        origin = _aware(contract["P2"]["bootstrap_origin"])
        return ((times.tz_convert("UTC") - origin) // pd.Timedelta(days=7)).tolist()
    if problem == "P3":
        if not {"station", "episode_id"}.issubset(frame):
            raise ValueError("P3 bootstrap episode metadata missing")
        if frame[["station", "episode_id"]].isna().any().any():
            raise ValueError("P3 bootstrap episode metadata missing")
        _episode_id_guard(frame["episode_id"])
        return list(zip(frame["station"], frame["episode_id"], strict=True))
    raise ValueError("unknown problem")


def _episode_id_guard(values):
    for value in values:
        if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
            raise ValueError("unknown episode ID")
        if isinstance(value, int | float | np.integer | np.floating) and not np.isfinite(value):
            raise ValueError("nonfinite episode ID")


def _cluster_statistics(y, prediction, codes, count, metric):
    """Sufficient statistics, not repeatedly concatenated observation arrays."""
    if metric == "f1":
        values = [
            (y == 1) & (prediction == 1),
            (y == 0) & (prediction == 1),
            (y == 1) & (prediction == 0),
        ]
    else:
        values = [np.square(prediction - y), np.ones(len(y))]
    stats = np.column_stack([np.bincount(codes, weights=v, minlength=count) for v in values])
    if not np.isfinite(stats).all():
        raise ValueError("nonfinite cluster sufficient statistics")
    return stats


def _metric_from_statistics(stats, metric):
    if metric == "rmse":
        return float(np.sqrt(stats[0] / stats[1]))
    denominator = 2 * stats[0] + stats[1] + stats[2]
    return 0.0 if denominator == 0 else float(2 * stats[0] / denominator)


def paired_bootstrap(truth, control, candidate, groups, metric, contract):
    y, b, c = (_values(values, binary=metric == "f1") for values in (truth, control, candidate))
    if not (len(y) == len(b) == len(c) == len(groups)):
        raise ValueError("paired bootstrap length mismatch")
    if any(group is None or str(group).strip() == "" for group in groups):
        raise ValueError("missing bootstrap group")
    base = pooled_metric(y, b, metric)
    challenger = pooled_metric(y, c, metric)
    delta = challenger - base
    improve = delta > 0 if metric == "f1" else delta < 0
    group_codes, unique = pd.factorize(pd.Series(list(groups), dtype=object), sort=False)
    if (group_codes < 0).any():
        raise ValueError("missing bootstrap group")
    report = {
        "control": base,
        "candidate": challenger,
        "delta_candidate_minus_control": delta,
        "candidate_retained": bool(improve),
        "automatic_promotion": False,
        "n_rows": len(y),
        "n_clusters": len(unique),
        "probability_hard_gate": None,
        "risk_assessment": "REQUIRED_SEPARATELY_worst_block_and_slices",
    }
    settings = contract["common"]["bootstrap"]
    if len(unique) < settings["minimum_clusters"]:
        return {
            **report,
            "bootstrap_status": "NOT_ESTIMABLE",
            "ci90": None,
            "p_improve": None,
            "resamples": 0,
        }
    base_stats = _cluster_statistics(y, b, group_codes, len(unique), metric)
    candidate_stats = _cluster_statistics(y, c, group_codes, len(unique), metric)
    rng = np.random.default_rng(settings["seed"])
    deltas = []
    for _ in range(settings["resamples"]):
        sampled = rng.integers(0, len(unique), len(unique))
        deltas.append(
            _metric_from_statistics(candidate_stats[sampled].sum(axis=0), metric)
            - _metric_from_statistics(base_stats[sampled].sum(axis=0), metric)
        )
    deltas = np.asarray(deltas)
    improves = deltas > 0 if metric == "f1" else deltas < 0
    return {
        **report,
        "bootstrap_status": "COMPUTED_DESCRIPTIVE_NOT_OFFICIAL_PROBABILITY",
        "ci90": np.quantile(deltas, settings["ci_quantiles"]).tolist(),
        "p_improve": float(improves.mean()),
        "resamples": len(deltas),
        "seed": settings["seed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    contract = load_contract(args.config)
    print(
        json.dumps(
            {
                "contract_id": contract["contract_id"],
                "status": STATUS,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "config_structure_valid": True,
                "real_data_support_audit": "NOT_RUN",
                "fits": 0,
                "official_inputs_read": 0,
                "model_execution_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
