"""Independent post-run QA for P3 v25 fixed simplex stacking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_fixed_simplex_stacking_cycle_20260901_v25"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RUNNER = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
RESULT = ARTIFACT / "result.json"
ARRAYS = ARTIFACT / "evaluation-arrays.npz"
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
OUTPUT = REPORT / "independent-qa.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    arrays = np.load(ARRAYS, allow_pickle=False)
    truth = arrays["truth"].astype(np.float64)
    reference = arrays["uniform"].astype(np.float64)
    predictions = [arrays["candidate_1"].astype(np.float64), arrays["candidate_2"].astype(np.float64)]
    checks: dict[str, bool] = {}
    checks["terminal_complete"] = result["status"] == "COMPLETE"
    checks["terminal_no_go"] = result["decision"] == "NO_GO_ALL_SIMPLEX_CANDIDATES"
    checks["shape_182_by_6"] = truth.shape == reference.shape == predictions[0].shape == predictions[1].shape == (182, 6)
    checks["finite_arrays"] = all(np.isfinite(item).all() for item in (truth, reference, *predictions))
    checks["fit_count_6"] = result["fit_count"] == 6 and len(result["fit_receipts"]) == 6
    checks["six_outer_blocks"] = {item["block"] for item in result["fit_receipts"]} == {"01_02", "03_04", "05_06", "07_08", "09_10", "11_12"}
    checks["simplex_constraints"] = all(abs(sum(item["weights"].values()) - 1.0) < 1e-10 and min(item["weights"].values()) >= 0.0 for item in result["fit_receipts"])
    checks["zero_row_deletion"] = all(item["row_deletion"] == 0 for item in result["fit_receipts"])
    checks["official_access_zero"] = all(value == 0 for key, value in result["data_access"].items() if key != "historical_target_rows")
    checks["config_official_zero"] = all(value == 0 for value in config["official_policy"].values())
    checks["lock_consumed_once"] = lock["status"] == "ATTEMPT_CONSUMED_ONE_SHOT"
    checks["runner_hash"] = result["provenance"]["runner_sha256"] == sha256(RUNNER)
    checks["config_hash"] = result["provenance"]["config_sha256"] == sha256(CONFIG)
    checks["arrays_hash"] = result["provenance"]["evaluation_arrays_sha256"] == sha256(ARRAYS)
    metric_match = True
    decision_match = True
    baseline = rmse(truth, reference)
    for candidate, prediction in zip(result["candidates"], predictions, strict=True):
        after = rmse(truth, prediction)
        delta = after - baseline
        stored = candidate["rmse_m"]
        metric_match &= abs(stored["uniform_0p425"] - baseline) < 1e-12
        metric_match &= abs(stored["candidate"] - after) < 1e-12
        metric_match &= abs(stored["delta_candidate_minus_uniform"] - delta) < 1e-12
        stable = all(candidate["stable_checks"].values())
        high_risk = (not stable) and all(candidate["high_risk_checks"].values())
        expected = "PASS_STABLE" if stable else "PRESERVE_HIGH_RISK" if high_risk else "NO_GO"
        decision_match &= candidate["decision"] == expected
    checks["independent_metric_recalculation"] = metric_match
    checks["decision_recalculation"] = decision_match
    payload = {"schema_version": "p3.fixed_simplex_stacking.independent_qa.v25", "experiment_id": EXPERIMENT_ID, "decision": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "check_count": len(checks), "passed": sum(checks.values()), "failed": sum(not value for value in checks.values()), "model_fits": result["fit_count"], "official_rows": 0, "csv_materializations": 0, "uploads": 0, "hashes": {"result": sha256(RESULT), "arrays": sha256(ARRAYS), "runner": sha256(RUNNER), "config": sha256(CONFIG)}}
    if OUTPUT.exists():
        raise RuntimeError("independent QA already exists")
    OUTPUT.write_bytes(canonical(payload))
    print(json.dumps({"decision": payload["decision"], "checks": payload["check_count"], "passed": payload["passed"], "failed": payload["failed"], "model_fits": payload["model_fits"], "official_rows": 0, "csv_materializations": 0, "uploads": 0}, ensure_ascii=False))
    return 0 if payload["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
