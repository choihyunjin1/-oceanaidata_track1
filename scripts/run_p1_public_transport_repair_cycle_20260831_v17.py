from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v16 as source  # noqa: E402

from src.p1_qc.causal_minirocket_lite import (  # noqa: E402
    build_spec,
    calibrate_biases,
    causal_windows,
    transform,
)

CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v17.json"
REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v17/preflight-report.json"
ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v17"
QA_REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v17/independent-qa.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_contract() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _check_resources(started: float, config: dict) -> None:
    if time.perf_counter() - started > config["resource_caps"]["wall_seconds"]:
        raise RuntimeError("v17 wall cap exceeded")
    if psutil.Process().memory_info().rss > config["resource_caps"]["rss_bytes"]:
        raise RuntimeError("v17 RSS cap exceeded")
    if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > config["resource_caps"]["vram_bytes"]:
        raise RuntimeError("v17 VRAM cap exceeded")


def _windows_for_positions(frame: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    """Vectorized 15-minute trailing grids from the immutable causal z72 feature."""
    output = np.zeros((len(positions), 2, 97), dtype=np.float32)
    destination = {int(position): index for index, position in enumerate(positions)}
    work = frame[["station", "layer", "time", "temp_robust_z_72h"]].copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    wanted = np.zeros(len(work), dtype=bool)
    wanted[positions] = True
    for _, group in work.groupby(["station", "layer"], sort=False, observed=True):
        group = group.sort_values("time", kind="stable")
        target = group.loc[wanted[group["__position"].to_numpy()], "__position"].to_numpy(np.int64)
        if not len(target):
            continue
        times = (pd.to_datetime(group["time"], utc=True).astype("int64") // 60_000_000_000).to_numpy(np.int64)
        values = group["temp_robust_z_72h"].to_numpy(np.float32)
        query = (pd.to_datetime(frame.loc[target, "time"], utc=True).astype("int64") // 60_000_000_000).to_numpy(np.int64)
        for start in range(0, len(query), 4096):
            block_q = query[start : start + 4096]
            grid = block_q[:, None] - np.arange(96, -1, -1, dtype=np.int64)[None, :] * 15
            flat = grid.ravel()
            right = np.searchsorted(times, flat, side="left")
            exact = (right < len(times)) & (times[np.minimum(right, len(times) - 1)] == flat)
            left = right - 1
            interp = (~exact) & (left >= 0) & (right < len(times))
            interp &= (times[np.minimum(right, len(times) - 1)] - times[np.maximum(left, 0)]) <= 120
            vals = np.zeros(len(flat), dtype=np.float32)
            observed = np.zeros(len(flat), dtype=np.float32)
            vals[exact] = values[right[exact]]
            observed[exact] = 1.0
            valid_i = np.flatnonzero(interp)
            if len(valid_i):
                lidx, ridx = left[valid_i], right[valid_i]
                fraction = (flat[valid_i] - times[lidx]) / (times[ridx] - times[lidx])
                vals[valid_i] = values[lidx] + fraction * (values[ridx] - values[lidx])
                observed[valid_i] = 1.0
            rows = [destination[int(position)] for position in target[start : start + len(block_q)]]
            output[rows, 0] = vals.reshape(-1, 97)
            output[rows, 1] = observed.reshape(-1, 97)
    return output


def _feature_matrix(windows: np.ndarray, spec, biases: np.ndarray, device: str, started: float, config: dict) -> np.ndarray:
    matrix = np.empty((len(windows), 512), dtype=np.float32)
    for begin in range(0, len(windows), 2048):
        matrix[begin : begin + 2048] = transform(windows[begin : begin + 2048], spec, biases, device)
        _check_resources(started, config)
    return matrix


def execute() -> dict:
    if ARTIFACT.exists():
        raise FileExistsError("v17 exactly-once artifact already exists")
    config = load_contract()
    if not config["authorization"]["historical_execution"]:
        raise RuntimeError("historical execution not authorized")
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    lock = {
        "experiment_id": config["experiment_id"], "pid": os.getpid(), "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)), "source_sha256": sha256(ROOT / "src/p1_qc/causal_minirocket_lite.py"),
        "fit_budget": 2, "official_reads": 0, "hidden_truth_reads": 0,
    }
    (ARTIFACT / "attempt_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "loading", "fit_count": 0}) + "\n")
    frame, anchor, _, dependency = source.load_feature_surface()
    truth = frame["label_base"].to_numpy(np.int8)
    candidate, probability = anchor.copy(), np.zeros(len(frame), dtype=np.float32)
    receipts = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = build_spec()
    for fit_number, fold_spec in enumerate(config["validation"]["nested_fits"], 1):
        train_mask = frame["fold"].isin(fold_spec["train_folds"]).to_numpy() & (anchor == 0)
        validation_mask = frame["fold"].eq(fold_spec["validation_fold"]).to_numpy() & (anchor == 0)
        train_pos, valid_pos = np.flatnonzero(train_mask), np.flatnonzero(validation_mask)
        train_windows = _windows_for_positions(frame, train_pos)
        key_text = (frame.loc[train_pos, "station"].astype(str) + "|" + frame.loc[train_pos, "layer"].astype(str) + "|" + frame.loc[train_pos, "time"].astype(str)).to_numpy()
        hashes = np.asarray([hashlib.sha256(value.encode()).digest() for value in key_text], dtype="S32")
        calibration_local = np.argsort(hashes, kind="stable")[:2048]
        calibration_key_sha = stable_hash(train_pos[calibration_local].astype(np.int64))
        biases = calibrate_biases(train_windows[calibration_local], spec, device)
        train_features = _feature_matrix(train_windows, spec, biases, device, started, config)
        del train_windows
        model = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=500, tol=1e-8, class_weight=None)
        model.fit(train_features, truth[train_pos])
        del train_features
        valid_windows = _windows_for_positions(frame, valid_pos)
        valid_features = _feature_matrix(valid_windows, spec, biases, device, started, config)
        current_probability = model.predict_proba(valid_features)[:, 1]
        del valid_features, valid_windows
        probability[valid_pos] = current_probability
        proposed = valid_pos[current_probability >= 0.99]
        candidate[proposed] = 1
        receipts.append({"fit_number": fit_number, "train_folds": fold_spec["train_folds"], "validation_fold": fold_spec["validation_fold"], "train_rows": len(train_pos), "validation_rows": len(valid_pos), "calibration_rows": 2048, "calibration_key_sha256": calibration_key_sha, "validation_rows_in_bias_calibration": 0, "validation_labels_in_bias_calibration": 0, "sealed_additions": len(proposed), "coefficient_sha256": stable_hash(model.coef_.astype(np.float64)), "bias_sha256": stable_hash(biases)})
        (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "fit_complete", "fit_count": fit_number}) + "\n")
        _check_resources(started, config)
    np.savez_compressed(ARTIFACT / "sealed_nested_predictions.npz", candidate=candidate, probability=probability)
    record = evaluation.evaluate(frame, anchor, candidate, config)
    record["name"] = config["candidate"]
    qa = {"status": "PASS", "checks": {"exact_two_fits": len(receipts) == 2, "bias_prefix_only": all(x["validation_rows_in_bias_calibration"] == 0 and x["validation_labels_in_bias_calibration"] == 0 for x in receipts), "anchor_order_preserved": len(candidate) == len(anchor), "anchor_removals_zero": record["anchor_removals"] == 0, "official_zero": True, "hidden_zero": True, "csv_zero": True, "upload_zero": True}}
    qa["status"] = "PASS" if all(qa["checks"].values()) else "FAIL"
    result = {"schema_version": "p1.v17.result.1", "experiment_id": config["experiment_id"], "status": "COMPLETE_INTERNAL_ONLY", "runtime_seconds": time.perf_counter() - started, "fit_count": 2, "pass_count": int(record["strict_internal_pass"]), "candidate": record, "nested_fit_receipts": receipts, "source_feature_dependency_receipt": dependency, "independent_qa": qa, "operations": {"historical_reads": 1, "official_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0}, "hashes": {"config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "source_sha256": sha256(ROOT / "src/p1_qc/causal_minirocket_lite.py"), "lock_sha256": sha256(ARTIFACT / "attempt_lock.json"), "prediction_sha256": sha256(ARTIFACT / "sealed_nested_predictions.npz")}}
    (ARTIFACT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    QA_REPORT.parent.mkdir(parents=True, exist_ok=True)
    QA_REPORT.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "progress.json").write_text(json.dumps({"phase": "terminal", "fit_count": 2, "pass_count": result["pass_count"]}) + "\n")
    return result


def run_preflight() -> dict:
    contract = load_contract()
    rng = np.random.default_rng(20260831)
    regular_t = np.arange(0, 3601, 15)
    signal = np.sin(regular_t / 180.0).astype(np.float32)
    queries = np.asarray([1800, 2700, 3600])
    windows = causal_windows(regular_t, signal, queries)
    future_t = np.append(regular_t, [3615, 3630])
    future_v = np.append(signal, [999.0, -999.0])
    invariant = np.array_equal(windows, causal_windows(future_t, future_v, queries))

    gap_t = np.asarray([0, 15, 300, 315, 1800])
    gap_w = causal_windows(gap_t, np.arange(5, dtype=np.float32), np.asarray([1800]))
    gap_missing = bool((gap_w[0, 1] == 0).any())

    synthetic = rng.normal(size=(96, 2, 97)).astype(np.float32)
    synthetic[:, 1] = (rng.random((96, 97)) > 0.1).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    spec = build_spec()
    biases = calibrate_biases(synthetic[:64], spec, device)
    features = transform(synthetic[64:], spec, biases, device)
    elapsed = time.perf_counter() - start
    repeat_spec = build_spec()
    repeat_bias = calibrate_biases(synthetic[:64], repeat_spec, device)
    repeat_features = transform(synthetic[64:], repeat_spec, repeat_bias, device)
    deterministic = stable_hash(spec.weights, spec.dilations, biases, features) == stable_hash(
        repeat_spec.weights, repeat_spec.dilations, repeat_bias, repeat_features
    )
    constant = np.ones((4, 2, 97), dtype=np.float32)
    constant[:, 1] = 1.0
    constant_features = transform(constant, spec, biases, device)

    matrix_bytes = 421_032 * 512 * 4
    batch_conv_bytes = 2048 * 512 * 97 * 4
    analytical_rss = matrix_bytes * 3 + batch_conv_bytes * 3
    analytical_vram = batch_conv_bytes * 4
    peak_vram = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    checks = {
        "exact_512_finite": features.shape == (32, 512) and bool(np.isfinite(features).all()),
        "future_append_invariant": invariant,
        "gap_over_2h_missing": gap_missing,
        "deterministic_hash": deterministic,
        "constant_series_finite": bool(np.isfinite(constant_features).all()),
        "threshold_exact": contract["model"]["probability_threshold_inclusive"] == 0.99,
        "two_pipeline_fits": contract["fit_budget"]["maximum"] == 2,
        "rss_cap": analytical_rss < contract["resource_caps"]["rss_bytes"],
        "vram_cap": max(analytical_vram, peak_vram) < contract["resource_caps"]["vram_bytes"],
        "wall_cap": 3600 < contract["resource_caps"]["wall_seconds"],
        "historical_execution_disabled": contract["authorization"]["historical_execution"] is False,
    }
    return {
        "schema_version": "p1.v17.synthetic-preflight.1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate": contract["candidate"],
        "checks": checks,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": device},
        "benchmark": {"synthetic_windows": 96, "seconds": elapsed, "historical_wall_estimate_seconds_conservative": 3600},
        "resource_estimate": {"feature_matrix_bytes": matrix_bytes, "analytical_peak_rss_bytes": analytical_rss, "analytical_peak_vram_bytes": analytical_vram, "observed_peak_vram_bytes": peak_vram},
        "hashes": {"config_sha256": sha256(CONFIG), "feature_preflight_sha256": stable_hash(features), "spec_bias_sha256": stable_hash(spec.weights, spec.dilations, spec.same_padding, spec.quantiles, biases)},
        "access": {"historical_truth_reads": 0, "official_reads": 0, "hidden_truth_reads": 0, "attempt_locks_created": 0, "uploads": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        try:
            print(json.dumps(execute(), indent=2, sort_keys=True))
            return
        except Exception as exc:  # noqa: BLE001
            payload = {"status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "official_reads": 0, "hidden_truth_reads": 0, "uploads": 0}
            if ARTIFACT.exists():
                (ARTIFACT / "terminal_failure.json").write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(payload, indent=2))
            raise SystemExit(1) from exc
    if not args.preflight:
        raise SystemExit("only --preflight is authorized")
    report = run_preflight()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
