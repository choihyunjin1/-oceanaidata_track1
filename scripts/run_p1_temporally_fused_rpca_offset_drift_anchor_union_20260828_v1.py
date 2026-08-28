"""Execute the sealed P1 temporally fused RPCA diagnostic exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1_qc.temporally_fused_rpca import (  # noqa: E402
    duration_mask,
    temporally_fused_rpca,
)

EXPERIMENT_ID = "p1_temporally_fused_rpca_offset_drift_anchor_union_20260828_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
KEYS = ["station", "year", "layer", "time"]
CADENCE_NS = int(pd.Timedelta(minutes=10).value)


class ContractError(RuntimeError):
    """Raised when the preregistered experiment cannot be executed exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int8).tobytes()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment ID drifted")
    if value["decoder"]["result_based_retry"] is not False:
        raise ContractError("result-based retry must remain disabled")
    policy = value["execution_policy"]
    if any(
        (
            policy["official_test_sample_submission_read_allowed"],
            policy["submission_csv_generation_allowed"],
            policy["official_upload_authorized"],
        )
    ):
        raise ContractError("official data or upload access is forbidden")
    for record in value["immutable_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ContractError(f"immutable input changed: {path}")
    return value


def _prefix_labels(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(
        columns=[*KEYS, "label"],
        filter=ds.field("time") < cutoff.isoformat(),
    )
    frame = table.to_pandas()
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul")
    return frame


def _robust_scale(
    period: pd.DataFrame,
    labels: pd.DataFrame,
    layers: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    merged = period.merge(labels, on=KEYS, how="left", validate="one_to_one")
    centers: list[float] = []
    scales: list[float] = []
    for layer in layers:
        values = merged.loc[(merged["layer"] == layer) & (merged["label"] == 0), "temp"]
        array = values.to_numpy(dtype=np.float64)
        center = float(np.nanmedian(array))
        mad = float(1.4826 * np.nanmedian(np.abs(array - center)))
        if not np.isfinite(mad) or mad < 1e-6:
            mad = float(np.nanstd(array))
        centers.append(center if np.isfinite(center) else 0.0)
        scales.append(mad if np.isfinite(mad) and mad >= 1e-6 else 1.0)
    return np.asarray(centers), np.asarray(scales)


def _window_starts(length: int, window: int, hop: int, minimum: int) -> list[tuple[int, int]]:
    if length < minimum:
        return []
    if length <= window:
        return [(0, length)]
    last = length - window
    starts = list(range(0, last + 1, hop))
    if starts[-1] != last:
        starts.append(last)
    return [(start, start + window) for start in starts]


def score_period(
    period: pd.DataFrame,
    layers: list[int],
    center: np.ndarray,
    scale: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    decomposition = config["decomposition"]
    pivot = period.pivot(index="time", columns="layer", values="temp").sort_index()
    pivot = pivot.reindex(columns=layers)
    observed = pivot.notna()
    pivot = pivot.interpolate(
        method="linear",
        limit=int(config["scope"]["small_gap_fill_rows"]),
        limit_area="inside",
    )
    valid = pivot.notna().all(axis=1).to_numpy()
    valid_positions = np.flatnonzero(valid)
    score_lists: dict[tuple[pd.Timestamp, int], list[float]] = defaultdict(list)
    convergence: list[bool] = []
    iterations: list[int] = []
    if len(valid_positions):
        times = pd.DatetimeIndex(pivot.index)
        split_at = np.flatnonzero(np.diff(times.asi8[valid_positions]) != CADENCE_NS) + 1
        for positions in np.split(valid_positions, split_at):
            for local_start, local_stop in _window_starts(
                len(positions),
                int(decomposition["window_rows"]),
                int(decomposition["hop_rows"]),
                int(decomposition["minimum_segment_rows"]),
            ):
                chosen = positions[local_start:local_stop]
                matrix = pivot.iloc[chosen].to_numpy(dtype=np.float64)
                matrix = np.clip((matrix - center) / scale, -12.0, 12.0)
                result = temporally_fused_rpca(
                    matrix,
                    maximum_iterations=int(decomposition["maximum_iterations"]),
                    tolerance=float(decomposition["tolerance"]),
                    proximal_iterations=int(decomposition["proximal_iterations"]),
                )
                convergence.append(result.converged)
                iterations.append(result.iterations)
                scores = np.abs(result.sparse)
                for row_index, pivot_position in enumerate(chosen):
                    timestamp = times[pivot_position]
                    for column_index, layer in enumerate(layers):
                        if observed.iloc[pivot_position, column_index]:
                            score_lists[(timestamp, layer)].append(float(scores[row_index, column_index]))
    rows = [
        {"time": timestamp, "layer": layer, "score": float(np.median(values))}
        for (timestamp, layer), values in score_lists.items()
    ]
    frame = pd.DataFrame(rows, columns=["time", "layer", "score"])
    diagnostics = {
        "available_rows": int(observed.to_numpy().sum()),
        "scored_rows": int(len(frame)),
        "coverage": float(len(frame) / max(int(observed.to_numpy().sum()), 1)),
        "windows": int(len(convergence)),
        "convergence_rate": float(np.mean(convergence)) if convergence else 0.0,
        "median_iterations": float(np.median(iterations)) if iterations else 0.0,
    }
    return frame, diagnostics


def _decode(
    scores: pd.DataFrame,
    thresholds: dict[int, float],
    decoder: dict[str, Any],
) -> pd.DataFrame:
    output = scores.sort_values(["layer", "time"]).copy()
    output["candidate"] = False
    for layer, group in output.groupby("layer", sort=True):
        positions = group.index.to_numpy()
        above = group["score"].to_numpy() > thresholds.get(int(layer), float("inf"))
        times = pd.DatetimeIndex(group["time"])
        contiguous = np.ones(len(group), dtype=bool)
        if len(group) > 1:
            contiguous[1:] = np.diff(times.asi8) == CADENCE_NS
        marked = np.zeros(len(group), dtype=bool)
        start = 0
        while start < len(group):
            stop = start + 1
            while stop < len(group) and contiguous[stop]:
                stop += 1
            marked[start:stop] = duration_mask(
                above[start:stop],
                int(decoder["minimum_duration_rows"]),
                int(decoder["maximum_duration_rows"]),
            )
            start = stop
        output.loc[positions, "candidate"] = marked
    return output


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=bool)
    guess = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(truth & guess))
    fp = int(np.sum(~truth & guess))
    fn = int(np.sum(truth & ~guess))
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _bootstrap(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    gates = config["gates"]
    work = frame.copy()
    work["block"] = (
        work["fold"].astype(str)
        + "|"
        + work["station"].astype(str)
        + "|"
        + (pd.to_datetime(work["time"], utc=True).dt.floor(f"{gates['bootstrap_blocks_days']}D")).astype(str)
    )
    groups = [group.index.to_numpy() for _, group in work.groupby("block", sort=True)]
    rng = np.random.default_rng(int(gates["bootstrap_seed"]))
    deltas = np.empty(int(gates["bootstrap_replicates"]), dtype=np.float64)
    for index in range(len(deltas)):
        selected = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[position] for position in selected])
        sample = work.loc[rows]
        deltas[index] = _f1(sample["label"], sample["candidate_prediction"]) - _f1(
            sample["label"], sample["current_router_prediction"]
        )
    return {
        "probability_delta_positive": float(np.mean(deltas > 0.0)),
        "ci90_lower": float(np.quantile(deltas, 0.05)),
        "ci90_upper": float(np.quantile(deltas, 0.95)),
    }


def execute(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    output_directory = ROOT / config["artifact_directory"]
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    raw_path = ROOT / config["immutable_inputs"]["training_rows"]["path"]
    anchor = pd.read_parquet(ROOT / config["immutable_inputs"]["anchor"]["path"])
    raw = pd.read_parquet(raw_path, columns=[*KEYS, "temp"])
    for frame in (anchor, raw):
        frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul")
    predictions = anchor.copy()
    predictions["rpca_candidate"] = np.int8(0)
    fold_diagnostics: dict[str, Any] = {}
    for fold in config["split"]["folds"]:
        fold_anchor = anchor.loc[anchor["fold"] == fold]
        validation_start = fold_anchor["time"].min()
        validation_stop = fold_anchor["time"].max()
        cutoff = validation_start - pd.Timedelta(days=int(config["split"]["purge_days"]))
        calibration_start = cutoff - pd.Timedelta(days=int(config["split"]["calibration_days"]))
        prefix_labels = _prefix_labels(raw_path, cutoff)
        fold_diagnostics[fold] = {}
        for station in config["scope"]["stations"]:
            station_raw = raw.loc[raw["station"] == station]
            calibration = station_raw.loc[
                (station_raw["time"] >= calibration_start) & (station_raw["time"] < cutoff)
            ].copy()
            validation = station_raw.loc[
                (station_raw["time"] >= validation_start) & (station_raw["time"] <= validation_stop)
            ].copy()
            calibration_pivot = calibration.pivot(index="time", columns="layer", values="temp")
            coverage = calibration_pivot.notna().mean()
            layers = [
                int(layer)
                for layer, value in coverage.items()
                if float(value) >= float(config["scope"]["minimum_layer_coverage"])
            ]
            if len(layers) < int(config["scope"]["minimum_layers"]):
                fold_diagnostics[fold][station] = {
                    "status": "INSUFFICIENT_LAYERS",
                    "layers": layers,
                }
                continue
            center, scale = _robust_scale(calibration, prefix_labels, layers)
            calibration_scores, calibration_diag = score_period(
                calibration, layers, center, scale, config
            )
            calibration_keys = calibration.loc[:, KEYS].merge(
                prefix_labels, on=KEYS, how="left", validate="one_to_one"
            )
            calibration_scores = calibration_scores.merge(
                calibration_keys[["time", "layer", "label"]],
                on=["time", "layer"],
                how="left",
                validate="one_to_one",
            )
            thresholds = {
                int(layer): float(
                    group.loc[group["label"] == 0, "score"].quantile(
                        float(config["decoder"]["normal_score_quantile"])
                    )
                )
                for layer, group in calibration_scores.groupby("layer")
            }
            validation_scores, validation_diag = score_period(
                validation, layers, center, scale, config
            )
            decoded = _decode(validation_scores, thresholds, config["decoder"])
            candidate_keys = decoded.loc[decoded["candidate"], ["time", "layer"]]
            if not candidate_keys.empty:
                key_index = pd.MultiIndex.from_frame(candidate_keys.astype({"layer": int}))
                target_mask = (
                    (predictions["fold"] == fold)
                    & (predictions["station"] == station)
                    & pd.MultiIndex.from_frame(predictions[["time", "layer"]]).isin(key_index)
                )
                predictions.loc[target_mask, "rpca_candidate"] = np.int8(1)
            fold_diagnostics[fold][station] = {
                "status": "SCORED",
                "layers": layers,
                "layer_prefix_coverage": {str(k): float(v) for k, v in coverage.items()},
                "thresholds": {str(k): v for k, v in thresholds.items()},
                "calibration": calibration_diag,
                "validation": validation_diag,
                "decoded_rows": int(decoded["candidate"].sum()),
            }
    predictions["candidate_prediction"] = np.maximum(
        predictions["current_router_prediction"].to_numpy(dtype=np.int8),
        predictions["rpca_candidate"].to_numpy(dtype=np.int8),
    ).astype(np.int8)
    canonical = predictions.sort_values(["fold", *KEYS]).reset_index(drop=True)
    prediction_path = output_directory / "sealed_candidate_predictions.parquet"
    atomic_parquet(prediction_path, canonical)
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_rows": int(len(canonical)),
        "prediction_sha256": prediction_sha(canonical["candidate_prediction"].to_numpy()),
        "candidate_sha256": prediction_sha(canonical["rpca_candidate"].to_numpy()),
        "truth_rows_read_before_commitment": 0,
        "official_rows_read": 0,
    }
    atomic_json(output_directory / "prediction_commitment.json", commitment)

    truth = pd.read_parquet(ROOT / config["immutable_inputs"]["truth_after_commit"]["path"])
    truth["time"] = pd.to_datetime(truth["time"], utc=True).dt.tz_convert("Asia/Seoul")
    evaluation = canonical.merge(
        truth[[*KEYS, "fold", "label", "anomaly_type"]],
        on=[*KEYS, "fold"],
        how="left",
        validate="one_to_one",
    )
    if evaluation["label"].isna().any():
        raise ContractError("truth binding failed")
    incumbent_f1 = _f1(evaluation["label"], evaluation["current_router_prediction"])
    candidate_f1 = _f1(evaluation["label"], evaluation["candidate_prediction"])
    added = (evaluation["rpca_candidate"] == 1) & (evaluation["current_router_prediction"] == 0)
    added_count = int(added.sum())
    added_tp = int((added & (evaluation["label"] == 1)).sum())
    added_precision = float(added_tp / max(added_count, 1))
    fold_metrics: dict[str, Any] = {}
    for fold, group in evaluation.groupby("fold", sort=True):
        old = _f1(group["label"], group["current_router_prediction"])
        new = _f1(group["label"], group["candidate_prediction"])
        fold_added = (group["rpca_candidate"] == 1) & (group["current_router_prediction"] == 0)
        fold_metrics[str(fold)] = {
            "incumbent_f1": old,
            "candidate_f1": new,
            "delta_f1": new - old,
            "added_rows": int(fold_added.sum()),
            "added_precision": float(
                (fold_added & (group["label"] == 1)).sum() / max(int(fold_added.sum()), 1)
            ),
        }
    relevant = evaluation.loc[
        (evaluation["label"] == 1)
        & evaluation["anomaly_type"].fillna("").str.contains("offset|drift", regex=True)
    ].sort_values(["fold", "station", "layer", "time"])
    recovered_events: list[dict[str, Any]] = []
    for (fold, station, layer), group in relevant.groupby(["fold", "station", "layer"]):
        times = pd.DatetimeIndex(group["time"])
        boundary = np.ones(len(group), dtype=bool)
        if len(group) > 1:
            boundary[1:] = np.diff(times.asi8) != CADENCE_NS
        event_ids = np.cumsum(boundary)
        for event_id, event in group.groupby(event_ids):
            event_added = (event["rpca_candidate"] == 1) & (
                event["current_router_prediction"] == 0
            )
            if event_added.any():
                recovered_events.append(
                    {
                        "fold": str(fold),
                        "station": str(station),
                        "layer": int(layer),
                        "event_id": int(event_id),
                        "added_true_rows": int(event_added.sum()),
                    }
                )
    normal_exposure = evaluation.loc[evaluation["label"] == 0].copy()
    normal_exposure["date"] = normal_exposure["time"].dt.date
    cell_days = normal_exposure[["station", "layer", "date"]].drop_duplicates().shape[0]
    false_positive_rate = float(
        (added & (evaluation["label"] == 0)).sum() / max(cell_days, 1)
    )
    bootstrap = _bootstrap(evaluation, config)
    station_records = [record for fold in fold_diagnostics.values() for record in fold.values()]
    scored_records = [record for record in station_records if record.get("status") == "SCORED"]
    minimum_coverage = min(
        (record["validation"]["coverage"] for record in scored_records), default=0.0
    )
    convergence_rate = float(
        np.mean([record["validation"]["convergence_rate"] for record in scored_records])
    ) if scored_records else 0.0
    gates = config["gates"]
    checks = {
        "panel_coverage": minimum_coverage >= float(gates["minimum_panel_coverage"]),
        "convergence": convergence_rate >= float(gates["minimum_convergence_rate"]),
        "pooled_delta": candidate_f1 - incumbent_f1 > 0.0,
        "positive_folds": sum(value["delta_f1"] > 0.0 for value in fold_metrics.values())
        >= int(gates["minimum_positive_folds"]),
        "fold_floor": min((value["delta_f1"] for value in fold_metrics.values()), default=-1.0)
        >= float(gates["minimum_fold_delta_f1"]),
        "added_precision": added_precision > float(gates["minimum_added_precision"]),
        "recovered_events": len(recovered_events)
        >= int(gates["minimum_recovered_offset_drift_events"]),
        "recovery_folds": len({value["fold"] for value in recovered_events})
        >= int(gates["minimum_recovery_folds"]),
        "recovery_cells": len({(value["station"], value["layer"]) for value in recovered_events})
        >= int(gates["minimum_recovery_cells"]),
        "normal_false_positive_rate": false_positive_rate
        <= float(gates["maximum_normal_false_positive_rows_per_cell_day"]),
        "bootstrap_probability": bootstrap["probability_delta_positive"]
        >= float(gates["minimum_probability_delta_positive"]),
        "bootstrap_ci_lower": bootstrap["ci90_lower"]
        >= float(gates["minimum_ci90_lower"]),
    }
    result = {
        "schema_version": "p1.temporally_fused_rpca.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "GO_LOCAL_ONLY" if all(checks.values()) else "NO_GO_EXACT_NO_OUTPUT",
        "incumbent_f1": incumbent_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": candidate_f1 - incumbent_f1,
        "added_rows": added_count,
        "added_true_rows": added_tp,
        "added_precision": added_precision,
        "recovered_offset_drift_events": recovered_events,
        "normal_false_positive_rows_per_cell_day": false_positive_rate,
        "fold_metrics": fold_metrics,
        "bootstrap": bootstrap,
        "diagnostics": fold_diagnostics,
        "minimum_panel_coverage": minimum_coverage,
        "mean_convergence_rate": convergence_rate,
        "gate_checks": checks,
        "commitment": commitment,
        "truth_rows_read_after_commitment": int(len(truth)),
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output_directory / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.check == args.execute:
        raise SystemExit("choose exactly one of --check or --execute")
    config = load_config()
    if args.check:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "PASS",
                    "mode": "CHECK_ONLY",
                    "official_rows_read": 0,
                    "submission_generated_or_uploaded": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(json.dumps(execute(config), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
