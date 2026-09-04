"""Exactly-once P3 fixed heterogeneous simplex stacking cycle."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

import run_p3_kma_wind_work_residual_axis_cycle_20260901_v20 as v20  # noqa: E402
import run_p3_path_signature_residual_cycle_20260901_v23 as v23  # noqa: E402
from run_p3_parallel_candidate_cycle_20260831_v4 import load_historical, rmse  # noqa: E402
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    POINTS_PER_RMSE_M,
    bootstrap,
)

EXPERIMENT_ID = "p3_fixed_simplex_stacking_cycle_20260901_v25"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
ENSEMBLE = ROOT / "artifacts/p3/final_ensemble_validation/oof.parquet"
BASE_NAMES = ("uniform_0p425", "catboost_single", "catboost_multi", "persistence", "kma_calibrated_source")
BLOCKS = v23.BLOCKS
LEADS = v23.LEADS
TRANSPORT_PENALTY_POINTS = 0.04958605409228893
OFFICIAL_CHAMPION_POINTS = 24.203599


class ContractError(RuntimeError):
    """Raised when the sealed v25 contract differs."""


@dataclass(frozen=True)
class Spec:
    name: str
    shrink: float


SPECS = (
    Spec("P3_1_FIXED_SIMPLEX_SHRINK10", 0.10),
    Spec("P3_2_FIXED_SIMPLEX_SHRINK20", 0.20),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "schema": config["schema_version"] == "p3.fixed_simplex_stacking.config.v25",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "revin_stop": config["duplication_audit"]["revin_style_joint_residual"] == "STOP_SEMANTIC_DUPLICATE",
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_ARCHITECTURE_AXIS",
        "bases": tuple(config["stacking"]["base_order"]) == BASE_NAMES,
        "specs": tuple((item["name"], float(item["champion_residual_weight"])) for item in config["stacking"]["candidates"])
        == tuple((item.name, item.shrink) for item in SPECS),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
    }
    if not all(checks.values()):
        raise ContractError(f"v25 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def simplex_fit(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(BASE_NAMES) or y.shape != (len(x),):
        raise ContractError("simplex fit inputs differ")
    best: tuple[float, tuple[int, ...], np.ndarray] | None = None
    for width in range(1, x.shape[1] + 1):
        for active in itertools.combinations(range(x.shape[1]), width):
            local = x[:, active]
            gram = local.T @ local
            rhs = local.T @ y
            kkt = np.block([[gram + 1e-10 * np.eye(width), np.ones((width, 1))], [np.ones((1, width)), np.zeros((1, 1))]])
            solution = np.linalg.lstsq(kkt, np.r_[rhs, 1.0], rcond=None)[0][:-1]
            if np.min(solution) < -1e-9:
                continue
            solution = np.maximum(solution, 0.0)
            solution /= solution.sum()
            weights = np.zeros(x.shape[1], dtype=np.float64)
            weights[list(active)] = solution
            sse = float(np.sum(np.square(x @ weights - y)))
            key = (sse, active, weights)
            if best is None or (sse, active) < (best[0], best[1]):
                best = key
    if best is None:
        raise ContractError("no feasible simplex solution")
    weights = best[2]
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ContractError("simplex weights violate constraints")
    return weights, {"weights": {name: float(value) for name, value in zip(BASE_NAMES, weights, strict=True)}, "active_set": [BASE_NAMES[index] for index in best[1]], "train_sse": best[0], "fit_count": 1, "row_deletion": 0}


def base_surface() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cases, targets, reference, profile = v23.case_surface()
    long = v23.long_frame(cases, targets, reference)
    long["anchor_id"] = np.repeat(cases["anchor_id"].to_numpy(), len(LEADS))
    historical, _ = load_historical()
    source = historical[["anchor_id", "station", "lead_h", "calibrated_source"]].copy()
    old = pd.read_parquet(ENSEMBLE, columns=["anchor_id", "station", "lead_h", "target_hs", "single_prediction", "multi_prediction", "persistence"])
    if old.duplicated(["anchor_id", "station", "lead_h"]).any() or source.duplicated(["anchor_id", "station", "lead_h"]).any():
        raise ContractError("base OOF keys are duplicated")
    merged = long.merge(old, on=["anchor_id", "station", "lead_h"], how="left", validate="one_to_one", suffixes=("", "_old"))
    merged = merged.merge(source, on=["anchor_id", "station", "lead_h"], how="left", validate="one_to_one")
    if len(merged) != 1092 or not np.allclose(merged["target_hs"], merged["target_hs_old"], rtol=0.0, atol=1e-12):
        raise ContractError("base OOF target/key alignment differs")
    matrix = np.column_stack([merged["reference"], merged["single_prediction"], merged["multi_prediction"], merged["persistence"], merged["calibrated_source"]]).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ContractError("base prediction matrix is non-finite")
    receipt = {"rows": len(matrix), "columns": matrix.shape[1], "base_order": list(BASE_NAMES), "matrix_sha256": hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest(), "pair_keys_unique": True, "target_alignment_max_abs": float(np.max(np.abs(merged["target_hs"] - merged["target_hs_old"])))}
    return cases, targets, reference, matrix, {"historical": profile, "base_matrix": receipt}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT_DIR.exists() or LOCK.exists():
        raise ContractError("exactly-once namespace is already consumed")
    x = np.column_stack([np.linspace(0, 1, 20), np.linspace(1, 0, 20), np.ones(20), np.zeros(20), np.linspace(0.2, 0.8, 20)])
    y = 0.25 * x[:, 0] + 0.75 * x[:, 1]
    weights, receipt = simplex_fit(x, y)
    payload = {"schema_version": "p3.fixed_simplex_stacking.preflight.v25", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12, "synthetic_weights": weights.tolist(), "synthetic_receipt": receipt, "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def crossfit(cases: pd.DataFrame, targets: np.ndarray, reference: np.ndarray, matrix: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    truth = targets.reshape(-1)
    outputs = {spec.name: reference.copy() for spec in SPECS}
    receipts: list[dict[str, Any]] = []
    case_positions = np.repeat(np.arange(len(cases)), len(LEADS))
    for block in BLOCKS:
        valid_cases = cases["block"].eq(block).to_numpy()
        train_cases = v23.purged_train_indices(cases, valid_cases)
        train_rows = np.isin(case_positions, train_cases)
        valid_rows = np.isin(case_positions, np.flatnonzero(valid_cases))
        weights, receipt = simplex_fit(matrix[train_rows], truth[train_rows])
        stack = matrix[valid_rows] @ weights
        reference_valid = reference[valid_cases].reshape(-1)
        for spec in SPECS:
            candidate = np.clip(reference_valid + spec.shrink * (stack - reference_valid), 0.0, 30.0)
            outputs[spec.name][valid_cases] = candidate.reshape(int(valid_cases.sum()), len(LEADS))
        receipt.update({"block": block, "train_cases": len(train_cases), "valid_cases": int(valid_cases.sum())})
        receipts.append(receipt)
    if len(receipts) != 6 or sum(item["fit_count"] for item in receipts) != 6:
        raise ContractError("simplex fit budget differs")
    return outputs, receipts


def score(frame: pd.DataFrame, prediction: np.ndarray, spec: Spec) -> dict[str, Any]:
    flat, truth, reference = prediction.reshape(-1), frame["target_hs"].to_numpy(float), frame["reference"].to_numpy(float)
    before, after = rmse(truth, reference), rmse(truth, flat)
    delta = after - before
    by_block = v20.group_deltas(frame, flat, reference, ["block"])
    by_station = v20.group_deltas(frame, flat, reference, ["station"])
    by_lead = v20.group_deltas(frame, flat, reference, ["lead_h"])
    station_lead = v20.group_deltas(frame, flat, reference, ["station", "lead_h"])
    improved = sum(item["delta_rmse_m"] < 0 for item in by_block.values())
    worst_block = max(item["delta_rmse_m"] for item in by_block.values())
    worst_slice = max(item["delta_rmse_m"] for item in station_lead.values())
    offset = SPECS.index(spec) * 100
    episode_ci = bootstrap(frame, flat, ("episode_id",), 20261011 + offset)
    group_ci = bootstrap(frame, flat, ("block", "station"), 20261012 + offset)
    stable_checks = {"delta_rmse_negative": delta < 0, "minimum_four_improved_blocks": improved >= 4, "episode_ci90_upper_below_zero": episode_ci["ci90_m"][1] < 0, "block_station_ci90_upper_below_zero": group_ci["ci90_m"][1] < 0, "worst_station_lead_at_most_0p01m": worst_slice <= 0.01, "finite_predictions": bool(np.isfinite(flat).all())}
    high_risk_checks = {"delta_rmse_at_most_minus_0p005m": delta <= -0.005, "worst_station_lead_at_most_0p02m": worst_slice <= 0.02, "finite_predictions": stable_checks["finite_predictions"]}
    stable = all(stable_checks.values())
    high_risk = not stable and all(high_risk_checks.values())
    points = -delta * POINTS_PER_RMSE_M
    return {"name": spec.name, "decision": "PASS_STABLE" if stable else "PRESERVE_HIGH_RISK" if high_risk else "NO_GO", "champion_residual_weight": spec.shrink, "rmse_m": {"uniform_0p425": before, "candidate": after, "delta_candidate_minus_uniform": delta}, "expected_points": {"raw_gain": points, "transport_penalty": TRANSPORT_PENALTY_POINTS, "transport_adjusted_gain": points - TRANSPORT_PENALTY_POINTS, "nominal_official_score": OFFICIAL_CHAMPION_POINTS + points}, "improved_blocks": int(improved), "by_block": by_block, "station": by_station, "lead": by_lead, "station_lead": station_lead, "worst_block_delta_m": worst_block, "worst_station_lead_delta_m": worst_slice, "episode_bootstrap": episode_ci, "block_station_bootstrap": group_ci, "stable_checks": stable_checks, "high_risk_checks": high_risk_checks}


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 fixed simplex stacking cycle v25", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- RevIN was stopped at 0 fits as an exact semantic duplicate of the already executed and harmful robust RevIN PatchTST family.", "- The executed stacker fits non-negative sum-to-one weights only on each purged outer-training fold, then applies a presealed small residual from the exact uniform champion.", "- This is EXPLORATORY_ONLY on the repeatedly exposed 182-case surface, not a Public transport guarantee."]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; raw {points['raw_gain']:+.6f} points; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; worst station-lead {item['worst_station_lead_delta_m']:+.9f}m.")
        lines.append(f"  - episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.")
    lines.extend(["", "No official test/sample/submission/hidden value was read. No CSV was materialized and no upload occurred. No row was deleted and no outer result changed a base model, weight rule, or shrink strength."])
    return "\n".join(lines) + "\n"


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, matrix, profile = base_surface()
    predictions, receipts = crossfit(cases, targets, reference, matrix)
    frame = v23.long_frame(cases, targets, reference)
    scored = [score(frame, predictions[spec.name], spec) for spec in SPECS]
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {"schema_version": "p3.fixed_simplex_stacking.result.v25", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "PASS_CANDIDATE_AVAILABLE" if passing else "NO_GO_ALL_SIMPLEX_CANDIDATES", "surface_claim": config["validation"]["surface"], "reference": config["reference"], "duplication_audit": config["duplication_audit"], "base_order": list(BASE_NAMES), "candidates": scored, "fit_receipts": receipts, "fit_count": 6, "data_profile": profile, "data_access": {"historical_target_rows": 1092, "official_test_rows": 0, "official_sample_rows": 0, "official_submission_rows": 0, "hidden_truth_rows": 0, "csv_materializations": 0, "uploads": 0}, "execution": {"python": platform.python_version(), "elapsed_seconds": time.perf_counter() - started, "candidate_count": 2, "result_based_tuning": False, "outer_result_parameter_changes": 0, "row_deletion": 0}}
    arrays = {"truth": targets, "uniform": reference, "candidate_1": predictions[SPECS[0].name], "candidate_2": predictions[SPECS[1].name], "anchor_id": cases["anchor_id"].to_numpy(np.int32), "lead_h": np.asarray(LEADS, dtype=np.int16), "block": cases["block"].to_numpy(dtype="U5"), "station": cases["station"].to_numpy(dtype="U5"), "episode": cases["episode_id"].to_numpy(dtype="U32")}
    return result, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT_DIR.exists() or REPORT_DIR.exists() or LOCK.exists():
        raise ContractError("v25 exactly-once namespace already exists")
    config = load_config()
    preflight = preflight_payload()
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "official_access": 0}))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    result, arrays = execute(config)
    array_path = ARTIFACT_DIR / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {"runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "evaluation_arrays_sha256": sha256(array_path), "preflight_receipt_sha256": preflight["receipt_sha256"], "input_sha256": config["inputs"]}
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, render_report(result).encode())
    write_new(REPORT_DIR / "result.json", canonical(result))
    write_new(REPORT_DIR / "gap-matrix.md", b"# Gap matrix\n\n| Axis | Verdict | Reason |\n|---|---|---|\n| RevIN joint residual | STOP_SEMANTIC_DUPLICATE | Exact casewise robust normalization and de-normalization already executed and harmed |\n| Equal-weight ensemble | PRIOR_PRECEDENT | Fixed 50:50 single/multi CatBoost exists |\n| Purged train-only heterogeneous simplex | EXECUTED | No prior non-negative sum-to-one outer-train fit found |\n")
    write_new(REPORT_DIR / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| RevIN exact family exists | `src/p3_wave/revin_patch.py` | Duplicate stop |\n| RevIN candidate harmed | `reports/promotion_retroaudit_20260827_v1/report-source.md` | Duplicate-family negative evidence |\n| Equal-weight ensemble exists | `artifacts/p3/final_ensemble_validation/metrics.json` | Separates prior fixed averaging from fitted simplex |\n")
    write_new(REPORT_DIR / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "result_sha256": sha256(result_path), "arrays_sha256": sha256(array_path), "report_sha256": sha256(report_path), "fit_count": result["fit_count"], "official_access": 0, "csv_materializations": 0, "uploads": 0}))
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": result["fit_count"], "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
