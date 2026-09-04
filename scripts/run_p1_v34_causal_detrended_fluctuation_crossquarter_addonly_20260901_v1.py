"""Exactly-once causal detrended-fluctuation P1 cross-quarter audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V30 = ROOT / "scripts/run_p1_v30_causal_backward_teager_energy_crossquarter_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v34_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V30)
base = shared.base


def _rolling_linear_detrended_rms(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling RMS after least-squares linear detrending of an integrated profile."""

    values = np.asarray(values, dtype=np.float64)
    output = np.zeros(len(values), dtype=np.float64)
    if len(values) < window:
        return output
    profile = np.cumsum(values, dtype=np.float64)
    index = np.arange(len(values), dtype=np.float64)

    def cumulative(value: np.ndarray) -> np.ndarray:
        return np.r_[0.0, np.cumsum(value, dtype=np.float64)]

    sum_y = cumulative(profile)
    sum_y2 = cumulative(np.square(profile))
    sum_xy = cumulative(index * profile)
    ends = np.arange(window, len(values) + 1, dtype=np.int64)
    starts = ends - window
    sy = sum_y[ends] - sum_y[starts]
    sy2 = sum_y2[ends] - sum_y2[starts]
    sxy = sum_xy[ends] - sum_xy[starts]
    first = starts.astype(np.float64)
    last = (ends - 1).astype(np.float64)
    sx = window * (first + last) / 2.0
    def prefix_square(value: np.ndarray) -> np.ndarray:
        return value * (value + 1.0) * (2.0 * value + 1.0) / 6.0

    sx2 = prefix_square(last) - prefix_square(first - 1.0)
    denominator = window * sx2 - np.square(sx)
    slope = np.divide(window * sxy - sx * sy, denominator, out=np.zeros_like(sy), where=denominator > 0)
    intercept = (sy - slope * sx) / window
    sse = np.maximum(0.0, sy2 - intercept * sy - slope * sxy)
    output[ends - 1] = np.sqrt(sse / window)
    return output


def rolling_dfa(values: np.ndarray, times_ns: np.ndarray, window: int) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.float64)
    gaps = np.r_[True, np.diff(times_ns) != CADENCE_NS]
    starts = np.flatnonzero(gaps)
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends, strict=True):
        output[start:end] = _rolling_linear_detrended_rms(values[start:end], window)
    return output


def dfa_features(frame: pd.DataFrame, train_boundary_ns: int, representation: dict[str, Any]) -> np.ndarray:
    windows = np.asarray(representation["rolling_rows"], dtype=np.int64)
    log_windows = np.log(windows.astype(np.float64))
    centered_log_windows = log_windows - log_windows.mean()
    slope_denominator = float(np.square(centered_log_windows).sum())
    output = np.zeros((len(frame), 8), dtype=np.float32)
    for _key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = base._time_ns(ordered["_time"])
        raw = ordered["temp"].to_numpy(np.float64)
        prefix = raw[(times <= train_boundary_ns) & np.isfinite(raw)]
        if not len(prefix):
            continue
        center = float(np.median(prefix))
        scale = float(1.4826 * np.median(np.abs(prefix - center)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(prefix))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        values = np.clip(np.nan_to_num((raw - center) / scale, nan=0.0), -12.0, 12.0)
        fluctuations = np.column_stack([rolling_dfa(values, times, int(window)) for window in windows])
        log_fluctuations = np.log(fluctuations + 1e-9)
        alpha = (log_fluctuations * centered_log_windows[None, :]).sum(axis=1) / slope_denominator
        alpha[~np.all(fluctuations > 0.0, axis=1)] = 0.0
        ratio = np.log((fluctuations[:, -1] + 1e-9) / (fluctuations[:, 0] + 1e-9))
        ratio[~np.all(fluctuations > 0.0, axis=1)] = 0.0
        difference = np.r_[0.0, np.diff(values)]
        difference[np.r_[True, np.diff(times) != CADENCE_NS]] = 0.0
        support = (fluctuations[:, -1] > 0.0).astype(np.float64)
        feature = np.column_stack(
            [values, difference, np.log1p(fluctuations), np.clip(alpha, -12.0, 12.0), np.clip(ratio, -12.0, 12.0), support]
        )
        if feature.shape[1] != 8 or not np.isfinite(feature).all():
            raise RuntimeError("DFA features invalid")
        output[positions] = feature.astype(np.float32)
    return output


def _configure() -> None:
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = dfa_features
    base.VIBClassifier = shared.LinearProbeClassifier


def _synthetic_guards(representation: dict[str, Any]) -> dict[str, bool]:
    rows = 240
    cadence = pd.Timedelta(minutes=10)
    times = pd.date_range("2024-01-01", periods=rows, freq=cadence, tz="UTC")
    times_ns = base._time_ns(times)
    constant = _rolling_linear_detrended_rms(np.ones(rows, dtype=np.float64), 24)
    fluctuating = _rolling_linear_detrended_rms(np.tile([1.0, -1.0], rows // 2), 24)

    gap_times = times_ns.copy()
    gap_times[120:] += CADENCE_NS
    gap_feature = rolling_dfa(np.sin(np.arange(rows, dtype=np.float64)), gap_times, 24)

    group_rows = 120
    group_times = pd.date_range("2024-02-01", periods=group_rows, freq=cadence, tz="UTC")
    values = np.sin(np.arange(group_rows, dtype=np.float64) / 7.0)
    frame = pd.DataFrame(
        {
            "station": np.repeat(["S-A", "S-B"], group_rows),
            "layer": np.repeat(["L1", "L2"], group_rows),
            "_time": np.tile(group_times, 2),
            "temp": np.tile(values, 2),
        }
    )
    boundary = int(group_times[59].value)
    original = dfa_features(frame, boundary, representation)
    perturbed_frame = frame.copy()
    future = base._time_ns(perturbed_frame["_time"]) > boundary
    perturbed_frame.loc[future, "temp"] += np.linspace(100.0, 500.0, int(future.sum()))
    perturbed = dfa_features(perturbed_frame, boundary, representation)
    prefix = ~future
    support_index = 7
    return {
        "linear_profile_fluctuation_zero": bool(np.max(np.abs(constant[23:])) < 1e-5),
        "injected_fluctuation_positive": bool(np.max(fluctuating[23:]) > 0.01),
        "cadence_gap_resets_window": bool(np.all(gap_feature[120:143] == 0.0)),
        "station_layer_group_reset": bool(
            np.all(original[:95, support_index] == 0.0)
            and np.all(original[group_rows : group_rows + 95, support_index] == 0.0)
            and original[95, support_index] == original[group_rows + 95, support_index] == 1.0
        ),
        "prefix_future_invariant": bool(np.array_equal(original[prefix], perturbed[prefix])),
        "ns_cutoff_distinct": bool(
            base._time_ns(group_times).dtype == np.dtype("int64")
            and int(group_times[58].value) < boundary < int(group_times[60].value)
        ),
        "shape_finite": bool(original.shape == (2 * group_rows, 8) and np.isfinite(original).all()),
    }


def preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    ready = base.preflight(data_dir)
    config = base._read(CONFIG)
    amendment = config["auditability_amendment"]
    if base._sha(ROOT / amendment["path"]) != amendment["sha256"]:
        raise RuntimeError("auditability amendment drifted")
    ready["auditability"] = {
        "amendment_sha256": amendment["sha256"],
        "preserve_all_pre_q2_threshold_q2_label_blind_actions": amendment[
            "preserve_all_pre_q2_threshold_q2_label_blind_actions"
        ],
        "q2_target_reads": 0,
    }
    guards = _synthetic_guards(config["representation"])
    if not all(guards.values()):
        raise RuntimeError(f"synthetic DFA guard failed: {guards}")
    ready["synthetic_guards"] = guards
    return ready


def _preserve_q2_auditability(
    config: dict[str, Any],
    ensemble_scores: np.ndarray,
    threshold_selection: dict[str, Any],
    model_hashes: list[str],
) -> dict[str, Any]:
    part = pd.read_parquet(ROOT / config["parts"]["2025_q2"]["path"], columns=list(base.PART_COLUMNS))
    positions = part["row_position"].to_numpy(np.int64)
    incumbent = part["baseline_prediction"].to_numpy(np.int8)
    q2_scores = ensemble_scores[positions]
    arrays: dict[str, np.ndarray] = {"positions": positions, "incumbent": incumbent, "scores": q2_scores}
    candidates = []
    for index, candidate in enumerate(threshold_selection["candidates"]):
        key = f"actions_candidate_{index}"
        actions = base._fixed_threshold_additions(
            q2_scores,
            incumbent,
            float(candidate["threshold"]),
            float(config["selection"]["maximum_addition_share"]),
        )
        arrays[key] = actions
        candidates.append(
            {
                "index": index,
                "action_key": key,
                "quantile": candidate["quantile"],
                "threshold": candidate["threshold"],
                "q2_label_blind_action_count": int(actions.sum()),
                "q2_label_blind_action_share": float(actions.mean()),
            }
        )
    bundle = ARTIFACT / "q2_all_threshold_label_blind_actions_sealed.npz"
    np.savez_compressed(bundle, **arrays)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "bundle_path": str(bundle.relative_to(ROOT)).replace("\\", "/"),
        "bundle_sha256": base._sha(bundle),
        "rows": int(len(positions)),
        "model_state_hashes": model_hashes,
        "candidates": candidates,
        "q2_target_reads_before_seal": 0,
        "q3_q4_target_reads": 0,
        "diagnostic_only_no_promotion": True,
    }
    base._write(ARTIFACT / "q2_all_threshold_auditability_manifest.json", manifest)
    return {**manifest, "manifest_sha256": base._sha(ARTIFACT / "q2_all_threshold_auditability_manifest.json")}


def execute(data_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    ready = preflight(data_dir)
    config = base._read(CONFIG)
    base._write(LOCK, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_EXACTLY_ONCE", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT.mkdir(exist_ok=False)
    base._write(ARTIFACT / "preflight.json", ready)
    frame = pd.read_csv(ready["source"]["train"], usecols=["station", "year", "layer", "time", "temp", "label", "anomaly_type"])
    frame["_time"] = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    times_ns = base._time_ns(frame["_time"])
    labels = frame["label"].to_numpy(np.int8)
    boundary = pd.Timestamp(ready["pre_q2"]["fit_boundary"]).value
    cutoff = pd.Timestamp(ready["pre_q2"]["calibration_cutoff"]).value
    features = dfa_features(frame, boundary, config["representation"])
    train_mask = times_ns <= boundary
    calibration_mask = (times_ns > boundary) & (times_ns <= cutoff)
    scaler = base.StandardScaler().fit(features[train_mask])
    scaled = scaler.transform(features).astype(np.float32)
    probabilities = []
    model_hashes = []
    for seed in config["model"]["seeds"]:
        model = shared.LinearProbeClassifier(scaled.shape[1], config["model"], int(seed)).fit(scaled[train_mask], labels[train_mask])
        probabilities.append(model.predict_score(scaled))
        model_hashes.append(base._model_hash(model, scaler))
    ensemble_scores = np.mean(np.stack(probabilities), axis=0)
    threshold_selection = base._select_pre_q2_threshold(
        ensemble_scores[calibration_mask],
        labels[calibration_mask],
        frame.loc[calibration_mask, list(base.KEY_COLUMNS)].reset_index(drop=True),
        config["selection"],
    )
    base._write(
        ARTIFACT / "pre_q2_candidate_threshold_seal.json",
        {"experiment_id": EXPERIMENT_ID, "fit_boundary": ready["pre_q2"]["fit_boundary"], "calibration_cutoff": ready["pre_q2"]["calibration_cutoff"], "selection": threshold_selection, "model_hashes": model_hashes, "fits": len(model_hashes), "q2_q3_q4_target_reads_before_seal": 0},
    )
    auditability = _preserve_q2_auditability(config, ensemble_scores, threshold_selection, model_hashes)
    threshold_selection = {**threshold_selection, "auditability": auditability}
    chosen = threshold_selection["chosen"]
    if chosen is None:
        return base._terminal_result(ready, config, started, len(model_hashes), threshold_selection, [], failed_stage="PRE_Q2_CALIBRATION_GATE")
    transport_receipts = []
    for fold in ("2025_q2", "2025_q3"):
        part = pd.read_parquet(ROOT / config["parts"][fold]["path"], columns=list(base.PART_COLUMNS))
        positions = part["row_position"].to_numpy(np.int64)
        incumbent = part["baseline_prediction"].to_numpy(np.int8)
        window_scores = ensemble_scores[positions]
        additions = base._fixed_threshold_additions(window_scores, incumbent, float(chosen["threshold"]), float(config["selection"]["maximum_addition_share"]))
        action_path = ARTIFACT / f"{fold}_transport_actions_sealed.npz"
        np.savez_compressed(action_path, positions=positions, incumbent=incumbent, additions=additions, scores=window_scores)
        base._write(ARTIFACT / f"{fold}_transport_action_seal.json", {"fold": fold, "path": str(action_path.relative_to(ROOT)).replace("\\", "/"), "sha256": base._sha(action_path), "threshold": chosen["threshold"], "target_reads_before_seal": 0})
        truth = labels[positions]
        metadata = frame.iloc[positions].loc[:, list(base.KEY_COLUMNS)].reset_index(drop=True)
        checked, receipt = base._window_gate(window_scores, incumbent, truth, metadata, float(chosen["threshold"]), config["selection"])
        if not np.array_equal(additions, checked):
            raise RuntimeError("transport action drifted")
        receipt = {"fold": fold, **receipt, "same_threshold": float(chosen["threshold"]), "refits_after_pre_q2": 0}
        base._write(ARTIFACT / f"{fold}_transport_gate.json", receipt)
        transport_receipts.append(receipt)
        if not receipt["passed"]:
            return base._terminal_result(ready, config, started, len(model_hashes), threshold_selection, transport_receipts, failed_stage=f"{fold}_TRANSPORT_GATE")
    base._write(ARTIFACT / "cross_quarter_gate_pass.json", {"experiment_id": EXPERIMENT_ID, "threshold": chosen["threshold"], "transport_windows": ["2025_q2", "2025_q3"], "both_passed": True, "q4_target_reads_before_gate_pass_receipt": 0})
    q4_part = pd.read_parquet(ROOT / config["parts"]["2025_q4"]["path"], columns=list(base.PART_COLUMNS))
    positions = q4_part["row_position"].to_numpy(np.int64)
    incumbent = q4_part["baseline_prediction"].to_numpy(np.int8)
    q4_scores = ensemble_scores[positions]
    additions = base._fixed_threshold_additions(q4_scores, incumbent, float(chosen["threshold"]), float(config["selection"]["maximum_addition_share"]))
    candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
    q4_path = ARTIFACT / "2025_q4_sealed.npz"
    np.savez_compressed(q4_path, positions=positions, incumbent=incumbent, additions=additions, candidate=candidate, scores=q4_scores)
    q4_seal = {"fold": "2025_q4", "path": str(q4_path.relative_to(ROOT)).replace("\\", "/"), "sha256": base._sha(q4_path), "threshold": chosen["threshold"], "model_hashes": model_hashes, "fits": len(model_hashes), "q4_target_reads_before_seal": 0}
    base._write(ARTIFACT / "2025_q4_seal.json", q4_seal)
    base._write(ARTIFACT / "predictions_complete.json", {"experiment_id": EXPERIMENT_ID, "performance_window_opened": True, "transport_windows_read": 2, "q4_target_reads_before_action_seal": 0, "q4_actions": int(additions.sum()), "seals": [q4_seal]})
    truth = labels[positions]
    metadata = frame.iloc[positions].loc[:, list(base.KEY_COLUMNS)].reset_index(drop=True)
    types = frame.iloc[positions]["anomaly_type"].reset_index(drop=True)
    performance = base.scorer._score_surface(truth, incumbent, candidate, additions, types, metadata)
    bootstrap = base.scorer._paired_cluster_bootstrap(truth, incumbent, candidate, metadata, replicates=config["decision"]["bootstrap_replicates"], seed=config["decision"]["seed"])
    passed = performance["delta_f1"] > 0 and bootstrap["ci90"][0] >= 0
    result = {"schema_version": config["result_schema_version"], "experiment_id": EXPERIMENT_ID, "surface": config["surface"], "decision": config["decision"]["pass"] if passed else config["decision"]["performance_fail"], "threshold_selection": threshold_selection, "transport_receipts": transport_receipts, "performance": performance, "performance_window": "2025_q4", "performance_window_opened": True, "block_bootstrap": bootstrap, "long_event_interior": base.base._long_event_interior(truth, incumbent, candidate, metadata), "long_event_boundary": base.shared.shared._boundary_recall(truth, incumbent, candidate, metadata), "worst_slices": sorted(performance["station_layer_diagnostics"], key=lambda item: item["delta_f1"])[:10], "action_slices": base.base._action_slices(additions, metadata), "points": {"nominal": performance["delta_f1"] * base.core.POINTS_PER_F1, "transport_adjusted": performance["delta_f1"] * base.core.POINTS_PER_F1 * base.core.TRANSPORT_FACTOR}, "counters": {"fits": len(model_hashes), "transport_target_windows_read": 2, "q4_target_reads": 1, "q4_actions": int(additions.sum()), "anchor_removals": performance["incumbent_positive_removals"], "official": 0, "csv": 0, "uploads": 0}, "runtime_seconds": time.monotonic() - started, "hashes": {"config": ready["config_sha256"], "runner": ready["runner_sha256"], "guard": ready["cross_quarter_guard_sha256"], "completion": base._sha(ARTIFACT / "predictions_complete.json"), "lock": base._sha(LOCK)}}
    base._write(ARTIFACT / "result.json", result)
    return result


def qa(data_dir: Path) -> dict[str, Any]:
    _configure()
    result_path = ARTIFACT / "result.json"
    if not result_path.exists():
        value = base.qa(data_dir)
        config = base._read(CONFIG)
        amendment = config["auditability_amendment"]
        value["checks"]["auditability_amendment"] = (
            amendment["preserve_all_pre_q2_threshold_q2_label_blind_actions"]
            and base._sha(ROOT / amendment["path"]) == amendment["sha256"]
        )
        value["checks"]["synthetic_guards"] = all(_synthetic_guards(config["representation"]).values())
        value["verdict"] = "PASS" if all(value["checks"].values()) else "FAIL"
        return value
    value = base.qa(data_dir)
    config = base._read(CONFIG)
    result = base._read(result_path)
    manifest_path = ARTIFACT / "q2_all_threshold_auditability_manifest.json"
    manifest = base._read(manifest_path)
    bundle = ROOT / manifest["bundle_path"]
    checks = value["checks"]
    checks["auditability_manifest"] = result["threshold_selection"]["auditability"]["manifest_sha256"] == base._sha(manifest_path)
    checks["auditability_bundle"] = manifest["bundle_sha256"] == base._sha(bundle)
    checks["all_threshold_actions"] = len(manifest["candidates"]) == len(result["threshold_selection"]["candidates"])
    checks["auditability_target0"] = manifest["q2_target_reads_before_seal"] == manifest["q3_q4_target_reads"] == 0
    amendment = config["auditability_amendment"]
    checks["auditability_amendment"] = base._sha(ROOT / amendment["path"]) == amendment["sha256"]
    checks["synthetic_guards"] = all(_synthetic_guards(config["representation"]).values())
    value["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    value["result_sha256"] = base._sha(result_path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = preflight(args.data_dir) if args.preflight else qa(args.data_dir) if args.qa else execute(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
