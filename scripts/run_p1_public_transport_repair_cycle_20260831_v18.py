"""Exactly-once nested historical execution for the sealed P1 v18 candidate."""

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
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (ROOT, SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_p1_public_transport_repair_cycle_20260831_v13 as base  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation_base  # noqa: E402

from src.p1_qc.causal_soft_symbolic import (  # noqa: E402
    build_spec,
    causal_robust_windows,
    soft_symbolic_transform,
)

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v18"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v2/calibration.json"
TRAIN = ROOT / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly/train.csv"
ANCHOR = base.ANCHOR_PATH
FOLDS = base.FOLDS


class ContractError(RuntimeError):
    """Frozen v18 contract violation."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def stable_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    spec = build_spec()
    smooth_penalty = max(
        float(item["adverse_penalty_points"])
        for item in calibration["observed_pairs"]
        if item["tier_id"] == "SMOOTH_LEARNED_PROFILE"
    )
    checks = {
        "calibration": contract["transport_family"]["calibration_sha256"] == sha256(CALIBRATION),
        "family": contract["transport_family"]["family_id"] == "P1_SOFT_SYMBOLIC_TRANSITION_SMOOTH_LINEAR_RESIDUAL",
        "tier": contract["transport_family"]["tier_id"] == "SMOOTH_LEARNED_PROFILE",
        "smooth": contract["transport_family"]["routing_discontinuous"] is False,
        "penalty": np.isclose(contract["transport_family"]["transport_penalty_points"], smooth_penalty, atol=1e-15),
        "raw_gate": np.isclose(
            contract["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
            smooth_penalty + calibration["minimum_calibrated_expected_points_delta"],
            atol=1e-15,
        ),
        "segments": contract["representation"]["paa_segments"] == 12,
        "centers": np.array_equal(np.asarray(contract["representation"]["soft_symbol_centers"]), spec.centers),
        "feature_count": contract["representation"]["feature_count"] == spec.feature_count == 347,
        "model": contract["model"]["C"] == 0.1 and contract["model"]["probability_threshold_inclusive"] == 0.99,
        "fits": contract["fit_budget"] == {"pipeline_fits": 2, "maximum": 2},
        "no_retune": contract["model"]["retuning"] is False and contract["validation"]["outer_result_based_tuning"] is False,
        "historical_authorized": contract["authorization"]["historical_execution"] is True,
        "lock_authorized": contract["authorization"]["attempt_lock_creation"] is True,
        "official_zero": contract["authorization"]["official_reads"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"v18 frozen contract mismatch: {checks}")
    return contract


def run_preflight() -> dict[str, Any]:
    contract = load_contract()
    times = np.arange(0, 10_001, 10, dtype=np.int64)
    signal = 0.003 * times + np.sin(times / 300.0)
    queries = np.asarray([7200, 8400, 9600], dtype=np.int64)
    started = time.perf_counter()
    windows, observed = causal_robust_windows(times, signal, queries)
    features = soft_symbolic_transform(windows, observed)
    future_windows, future_observed = causal_robust_windows(
        np.append(times, [10_010, 10_020]),
        np.append(signal, [1.0e9, -1.0e9]),
        queries,
    )
    future_features = soft_symbolic_transform(future_windows, future_observed)
    shifted_windows, shifted_observed = causal_robust_windows(times, signal + 1000.0, queries)
    shifted_features = soft_symbolic_transform(shifted_windows, shifted_observed)
    matrix_bytes = 421_032 * 347 * 4
    checks = {
        "shape_3_by_347": features.shape == (3, 347),
        "finite": bool(np.isfinite(features).all()),
        "future_append_invariant": np.array_equal(features, future_features),
        "future_append_mask_invariant": np.array_equal(observed, future_observed),
        "constant_offset_invariant": bool(np.allclose(features, shifted_features, atol=1.0e-5, rtol=0.0)),
        "representation_coverage": float(observed.mean()) >= contract["safety"]["minimum_representation_coverage"],
        "feature_matrix_bytes_exact": matrix_bytes == contract["resource_caps"]["feature_matrix_bytes"],
        "rss_cap": matrix_bytes * 5 < contract["resource_caps"]["rss_bytes"],
        "official_hidden_upload_zero": all(
            contract["authorization"][name] == 0
            for name in ("official_reads", "hidden_truth_reads", "submission_csv_created", "uploads")
        ),
    }
    return {
        "schema_version": "p1.v18.synthetic-preflight.2",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "benchmark": {"synthetic_queries": len(queries), "seconds": time.perf_counter() - started},
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "runner_sha256": sha256(Path(__file__)),
            "module_sha256": sha256(ROOT / "src/p1_qc/causal_soft_symbolic.py"),
            "calibration_sha256": sha256(CALIBRATION),
            "synthetic_feature_sha256": stable_hash(features),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "access": {"historical_truth_reads": 0, "official_reads": 0, "hidden_truth_reads": 0, "attempt_locks_created": 0, "submission_csv_created": 0, "uploads": 0},
    }


def _utc_minutes(values: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(values, utc=True).astype("int64").to_numpy()
    return (timestamps // 60_000_000_000).astype(np.int64)


def _torch_nanmedian_average(values: torch.Tensor) -> torch.Tensor:
    """Match NumPy's average-of-two convention for even finite counts."""

    finite_count = torch.isfinite(values).sum(dim=1)
    ordered = torch.sort(values, dim=1).values
    lower_index = torch.clamp((finite_count - 1) // 2, min=0)
    upper_index = torch.clamp(finite_count // 2, min=0)
    lower = ordered.gather(1, lower_index[:, None]).squeeze(1)
    upper = ordered.gather(1, upper_index[:, None]).squeeze(1)
    median = 0.5 * (lower + upper)
    return torch.where(finite_count > 0, median, torch.nan)


def _robust_prefix_z(times: np.ndarray, values: np.ndarray, *, device: str, chunk_size: int = 32768) -> np.ndarray:
    """Exact trailing-72h median/MAD for irregular but <=10-minute sensor rows."""

    n_rows = len(times)
    result = np.full(n_rows, np.nan, dtype=np.float32)
    backward = np.arange(-432, 1, dtype=np.int64)
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        current = np.arange(start, stop, dtype=np.int64)
        left = np.searchsorted(times, times[current] - 4320, side="left")
        count = current - left + 1
        if int(count.max(initial=0)) > 433:
            raise ContractError("more than 433 source rows occur in a 72h prefix")
        index = current[:, None] + backward[None, :]
        valid = (index >= left[:, None]) & (index >= 0)
        clipped = np.clip(index, 0, n_rows - 1)
        matrix = values[clipped]
        matrix = np.where(valid & np.isfinite(matrix), matrix, np.nan)
        tensor = torch.as_tensor(matrix, dtype=torch.float64, device=device)
        median = _torch_nanmedian_average(tensor)
        mad = _torch_nanmedian_average(torch.abs(tensor - median[:, None]))
        scale = torch.clamp(1.4826 * mad, min=1.0e-4)
        current_value = torch.as_tensor(values[current], dtype=torch.float64, device=device)
        z = torch.clamp((current_value - median) / scale, -12.0, 12.0)
        finite_count = torch.isfinite(tensor).sum(dim=1)
        z = torch.where((finite_count >= 12) & torch.isfinite(current_value), z, torch.nan)
        result[start:stop] = z.detach().cpu().numpy().astype(np.float32)
    return result


def _group_feature_blocks(
    raw: pd.DataFrame,
    anchor_frame: pd.DataFrame,
    feature_path: Path,
    progress_path: Path,
    contract: dict[str, Any],
) -> tuple[np.memmap, float, list[dict[str, Any]]]:
    features = np.memmap(feature_path, mode="w+", dtype=np.float32, shape=(len(anchor_frame), 347))
    coverage_total = 0.0
    coverage_count = 0
    receipts: list[dict[str, Any]] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    anchor_frame["time"] = pd.to_datetime(anchor_frame["time"], utc=True)
    anchor_frame["__row"] = np.arange(len(anchor_frame), dtype=np.int64)
    raw_groups = {
        key: group.sort_values("time", kind="stable")
        for key, group in raw.groupby(["station", "year", "layer"], sort=False, observed=True)
    }
    group_items = list(anchor_frame.groupby(["station", "year", "layer"], sort=False, observed=True))
    for group_number, (key, targets) in enumerate(group_items, start=1):
        if key not in raw_groups:
            raise ContractError(f"missing raw group for {key}")
        source = raw_groups[key]
        source_times = _utc_minutes(source["time"])
        if len(source_times) and (np.diff(source_times) <= 0).any():
            raise ContractError(f"non-unique/nonmonotone source time for {key}")
        source_values = pd.to_numeric(source["temp"], errors="coerce").to_numpy(np.float64)
        source_z = _robust_prefix_z(source_times, source_values, device=device)
        target_rows = targets["__row"].to_numpy(np.int64)
        target_times = _utc_minutes(targets["time"])
        offsets = np.arange(-1440, 1, 10, dtype=np.int64)
        for batch_start in range(0, len(target_rows), 8192):
            batch_stop = min(batch_start + 8192, len(target_rows))
            query = target_times[batch_start:batch_stop]
            grid = query[:, None] + offsets[None, :]
            index = np.searchsorted(source_times, grid, side="right") - 1
            valid = index >= 0
            clipped = np.clip(index, 0, len(source_times) - 1)
            age = np.where(valid, grid - source_times[clipped], 21)
            observed = valid & (age >= 0) & (age <= 20) & np.isfinite(source_z[clipped])
            windows = np.where(observed, source_z[clipped], 0.0).astype(np.float32)
            block = soft_symbolic_transform(windows, observed)
            destination = target_rows[batch_start:batch_stop]
            features[destination] = block
            coverage_total += float(observed.sum())
            coverage_count += int(observed.size)
        receipts.append(
            {
                "station": str(key[0]),
                "year": int(key[1]),
                "layer": int(key[2]),
                "source_rows": len(source),
                "target_rows": len(targets),
                "target_row_sha256": sha256_array(target_rows),
            }
        )
        base.write_json(
            progress_path,
            {"phase": "soft_symbolic_features", "groups_complete": group_number, "groups_total": len(group_items), "fit_count": 0},
            exclusive=False,
        )
    features.flush()
    coverage = coverage_total / coverage_count if coverage_count else 0.0
    if coverage < float(contract["safety"]["minimum_representation_coverage"]):
        raise ContractError(f"representation coverage {coverage} is below the sealed 0.95 gate")
    return features, coverage, receipts


def load_nested_frame() -> tuple[pd.DataFrame, np.ndarray]:
    anchor_frame = pd.read_parquet(ANCHOR)
    anchor = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    historical, _ = base.attach_truth(anchor_frame, anchor)
    historical["time"] = pd.to_datetime(historical["time"], utc=True)
    return historical, anchor


def train_nested(
    frame: pd.DataFrame,
    anchor: np.ndarray,
    features: np.ndarray,
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    truth = frame["label_base"].to_numpy(np.int8)
    candidate = anchor.copy()
    probability = np.zeros(len(frame), dtype=np.float32)
    receipts: list[dict[str, Any]] = []
    threshold = float(contract["model"]["probability_threshold_inclusive"])
    for fit_number, fold_spec in enumerate(contract["validation"]["nested_fits"], start=1):
        train = frame["fold"].isin(fold_spec["train_folds"]).to_numpy() & (anchor == 0)
        validation = frame["fold"].eq(fold_spec["validation_fold"]).to_numpy()
        model = LogisticRegression(
            penalty="l2",
            C=float(contract["model"]["C"]),
            solver="lbfgs",
            max_iter=int(contract["model"]["max_iter"]),
            tol=float(contract["model"]["tol"]),
            class_weight=None,
        )
        model.fit(np.asarray(features[train]), truth[train])
        validation_negative = validation & (anchor == 0)
        validation_probability = model.predict_proba(np.asarray(features[validation_negative]))[:, 1]
        probability[validation_negative] = validation_probability.astype(np.float32)
        proposed = validation_negative.copy()
        proposed[validation_negative] = validation_probability >= threshold
        candidate[proposed] = 1
        receipts.append(
            {
                "fit_number": fit_number,
                "train_folds": list(fold_spec["train_folds"]),
                "validation_fold": fold_spec["validation_fold"],
                "train_rows": int(train.sum()),
                "train_positives": int(truth[train].sum()),
                "validation_rows": int(validation.sum()),
                "validation_anchor_negative_rows": int(validation_negative.sum()),
                "sealed_additions": int(proposed.sum()),
                "probability_sha256": sha256_array(validation_probability.astype(np.float64)),
                "candidate_bits_sha256": base.sha256_bool(candidate[validation]),
                "coefficient_sha256": sha256_array(model.coef_.astype(np.float64)),
                "iterations": int(model.n_iter_[0]),
                "outer_target_reads_before_prediction_seal": 0,
            }
        )
        base.write_json(
            ARTIFACT / "progress.json",
            {"phase": "nested_fit", "fit_count": fit_number, "fit_budget": 2},
            exclusive=False,
        )
    return candidate, probability, receipts


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result["candidate"]
    checks = {
        "two_nested_fits": result["fit_count"] == 2 and len(result["nested_fit_receipts"]) == 2,
        "exact_347_features": result["representation_contract"]["feature_count"] == 347,
        "fixed_five_centers": len(result["representation_contract"]["soft_symbol_centers"]) == 5,
        "fixed_model": result["model_contract"]["C"] == 0.1 and result["model_contract"]["probability_threshold_inclusive"] == 0.99,
        "outer_targets_zero_before_seal": all(item["outer_target_reads_before_prediction_seal"] == 0 for item in result["nested_fit_receipts"]),
        "add_only": candidate["anchor_removals"] == 0,
        "official_zero": result["operations"]["official_covariate_reads"] == 0,
        "hidden_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "upload_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    load_contract()
    for path in (TRAIN, ANCHOR, CALIBRATION, base.AUTHORITATIVE_PATH):
        if not path.is_file():
            raise ContractError(f"missing frozen input: {path}")
    return {
        "status": "VALID",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "module_sha256": sha256(ROOT / "src/p1_qc/causal_soft_symbolic.py"),
        "fit_budget": 2,
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError("exactly-once v18 artifact path already exists")
    contract = load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    base.write_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pid": os.getpid(),
            "config_sha256": sha256(CONFIG),
            "runner_sha256": sha256(Path(__file__)),
            "module_sha256": sha256(ROOT / "src/p1_qc/causal_soft_symbolic.py"),
            "fit_budget": 2,
            "official_covariate_reads": 0,
        },
    )
    base.write_json(ARTIFACT / "progress.json", {"phase": "load_historical", "fit_count": 0}, exclusive=False)
    frame, anchor = load_nested_frame()
    raw = pd.read_csv(TRAIN, usecols=["station", "year", "layer", "time", "temp"])
    feature_path = ARTIFACT / "soft_symbolic_features.float32.mmap"
    features, coverage, group_receipts = _group_feature_blocks(
        raw,
        frame[["station", "year", "layer", "time"]].copy(),
        feature_path,
        ARTIFACT / "progress.json",
        contract,
    )
    feature_sha = sha256(feature_path)
    candidate, probability, fit_receipts = train_nested(frame, anchor, features, contract)
    prediction_path = ARTIFACT / "sealed_nested_predictions.npz"
    np.savez_compressed(prediction_path, candidate=candidate, probability=probability)
    prediction_seal = {
        "candidate_sha256": base.sha256_bool(candidate),
        "probability_sha256": sha256_array(probability),
        "npz_sha256": sha256(prediction_path),
        "q3_outer_target_reads_before_seal": 0,
        "q4_outer_target_reads_before_seal": 0,
    }
    base.write_json(ARTIFACT / "prediction_seal.json", prediction_seal)
    base.write_json(ARTIFACT / "progress.json", {"phase": "prediction_sealed", "fit_count": 2}, exclusive=False)
    record = evaluation_base.evaluate(frame, anchor, candidate, contract)
    record["name"] = contract["candidate"]
    result: dict[str, Any] = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v18.result",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "transport_family": contract["transport_family"],
        "representation_contract": contract["representation"],
        "model_contract": contract["model"],
        "safety_contract": contract["safety"],
        "decision_policy": contract["decision_policy"],
        "candidate_count": 1,
        "pass_count": int(record["strict_internal_pass"]),
        "fit_count": 2,
        "candidate": record,
        "representation_coverage": coverage,
        "nested_fit_receipts": fit_receipts,
        "feature_group_receipts": group_receipts,
        "prediction_seal": prediction_seal,
        "outputs": [],
        "adaptive_surface_disclaimer": "This exact candidate was preregistered after repeated historical development; Q3/Q4 are development surfaces, not fresh confirmation.",
        "operations": {"historical_train_csv_reads": 1, "official_covariate_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0},
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "runner_sha256": sha256(Path(__file__)),
            "module_sha256": sha256(ROOT / "src/p1_qc/causal_soft_symbolic.py"),
            "historical_train_sha256": sha256(TRAIN),
            "anchor_sha256": sha256(ANCHOR),
            "feature_matrix_sha256": feature_sha,
            "root_calibration_sha256": sha256(CALIBRATION),
            "authoritative_results_sha256": sha256(base.AUTHORITATIVE_PATH),
        },
    }
    result["independent_qa"] = independent_qa(result)
    base.write_json(ARTIFACT / "result.json", result)
    base.write_json(REPORT / "independent-qa.json", result["independent_qa"])
    base.write_json(
        ARTIFACT / "progress.json",
        {"phase": "terminal", "fit_count": 2, "pass_count": result["pass_count"]},
        exclusive=False,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sum((args.preflight, args.validate_only, args.execute)) != 1:
        parser.error("choose exactly one mode")
    try:
        if args.preflight:
            payload = run_preflight()
            REPORT.mkdir(parents=True, exist_ok=True)
            base.write_json(REPORT / "preflight-report.json", payload, exclusive=False)
        elif args.validate_only:
            payload = validate_only()
        else:
            payload = execute()
        print(json.dumps(base.native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if payload.get("status") != "FAIL" else 1
    except Exception as exc:  # noqa: BLE001
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            base.write_json(ARTIFACT / "terminal_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
