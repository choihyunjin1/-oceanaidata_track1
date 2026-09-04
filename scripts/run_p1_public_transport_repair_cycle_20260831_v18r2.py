"""Metric-only recovery for the immutable P1 v18r1 prediction seal."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation_base  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v18 as original  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v18r2"
RECOVERY_CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
BASE_CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v18.json"
SOURCE_ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v18r1"
PREDICTION_NPZ = SOURCE_ARTIFACT / "sealed_nested_predictions.npz"
PREDICTION_SEAL = SOURCE_ARTIFACT / "prediction_seal.json"
SOURCE_FAILURE = SOURCE_ARTIFACT / "terminal_failure.json"
SOURCE_LOCK = SOURCE_ARTIFACT / "attempt_lock.json"
FEATURE_MATRIX = SOURCE_ARTIFACT / "soft_symbolic_features.float32.mmap"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RECEIPT = REPORT / "technical-recovery-receipt.json"


def load_recovery_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    recovery = json.loads(RECOVERY_CONFIG.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    method = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    chain = receipt["hash_chain"]
    seal = json.loads(PREDICTION_SEAL.read_text(encoding="utf-8"))
    checks = {
        "base_method": recovery["base_method_config_sha256"] == original.sha256(BASE_CONFIG),
        "lock": chain["v18r1_attempt_lock_sha256"] == original.sha256(SOURCE_LOCK),
        "failure": chain["v18r1_terminal_failure_sha256"] == original.sha256(SOURCE_FAILURE),
        "seal_file": chain["v18r1_prediction_seal_sha256"] == original.sha256(PREDICTION_SEAL),
        "npz_file": recovery["sealed_prediction_npz_sha256"] == original.sha256(PREDICTION_NPZ),
        "npz_seal": recovery["sealed_prediction_npz_sha256"] == seal["npz_sha256"],
        "candidate_seal": recovery["sealed_candidate_sha256"] == seal["candidate_sha256"],
        "probability_seal": recovery["sealed_probability_sha256"] == seal["probability_sha256"],
        "feature_matrix": chain["v18r1_feature_matrix_sha256"] == original.sha256(FEATURE_MATRIX),
        "fits": recovery["inherited_fit_count"] == 2 and recovery["additional_fit_budget"] == 0,
        "prediction_immutable": recovery["prediction_changes"] == 0,
        "method_immutable": all(
            recovery[name] == 0
            for name in ("method_changes", "feature_changes", "model_changes", "threshold_changes", "gate_changes")
        ),
        "external_zero": all(
            recovery[name] == 0
            for name in ("official_reads", "hidden_truth_reads", "submission_csv_created", "uploads")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v18r2 recovery contract mismatch: {checks}")
    return recovery, receipt, method


def evaluation_contract() -> dict[str, Any]:
    _, _, method = load_recovery_contract()
    contract = copy.deepcopy(method)
    penalty = contract["transport_family"]["transport_penalty_points"]
    contract["decision_policy"]["transport_penalty_points"] = penalty
    return contract


def validate_only() -> dict[str, Any]:
    contract = evaluation_contract()
    with np.load(PREDICTION_NPZ) as sealed:
        candidate = sealed["candidate"]
        probability = sealed["probability"]
    checks = {
        "candidate_bits": original.base.sha256_bool(candidate) == json.loads(PREDICTION_SEAL.read_text())["candidate_sha256"],
        "probability_bits": original.sha256_array(probability) == json.loads(PREDICTION_SEAL.read_text())["probability_sha256"],
        "penalty_alias_exact": contract["decision_policy"]["transport_penalty_points"]
        == contract["transport_family"]["transport_penalty_points"],
        "additional_fits_zero": json.loads(RECOVERY_CONFIG.read_text())["additional_fit_budget"] == 0,
    }
    return {
        "status": "VALID" if all(checks.values()) else "INVALID",
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "runner_sha256": original.sha256(Path(__file__)),
        "recovery_config_sha256": original.sha256(RECOVERY_CONFIG),
        "receipt_sha256": original.sha256(RECEIPT),
    }


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    record = result["candidate"]
    checks = {
        "inherited_two_fits": result["inherited_fit_count"] == 2,
        "additional_fits_zero": result["additional_fit_count"] == 0,
        "prediction_changes_zero": result["prediction_changes"] == 0,
        "candidate_seal_match": result["prediction_seal"]["candidate_sha256"]
        == result["recovery_contract"]["sealed_candidate_sha256"],
        "probability_seal_match": result["prediction_seal"]["probability_sha256"]
        == result["recovery_contract"]["sealed_probability_sha256"],
        "penalty_alias_exact": result["penalty_alias_value"] == result["transport_family"]["transport_penalty_points"],
        "add_only": record["anchor_removals"] == 0,
        "official_zero": result["operations"]["official_covariate_reads"] == 0,
        "hidden_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "upload_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def execute() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError("exactly-once v18r2 artifact path already exists")
    recovery, _, _ = load_recovery_contract()
    contract = evaluation_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    original.base.write_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pid": os.getpid(),
            "mode": "metric_only_recovery",
            "inherited_fit_count": 2,
            "additional_fit_budget": 0,
            "prediction_changes": 0,
            "sealed_prediction_npz_sha256": original.sha256(PREDICTION_NPZ),
            "runner_sha256": original.sha256(Path(__file__)),
            "official_covariate_reads": 0,
        },
    )
    frame, anchor = original.load_nested_frame()
    with np.load(PREDICTION_NPZ) as sealed:
        candidate = sealed["candidate"].astype(np.int8, copy=True)
        probability = sealed["probability"].copy()
    prediction_seal = json.loads(PREDICTION_SEAL.read_text(encoding="utf-8"))
    if original.base.sha256_bool(candidate) != prediction_seal["candidate_sha256"]:
        raise RuntimeError("sealed candidate changed")
    if original.sha256_array(probability) != prediction_seal["probability_sha256"]:
        raise RuntimeError("sealed probability changed")
    feature_matrix = np.memmap(FEATURE_MATRIX, mode="r", dtype=np.float32, shape=(421_032, 347))
    representation_coverage = float(np.mean(feature_matrix[:, -12:], dtype=np.float64))
    record = evaluation_base.evaluate(frame, anchor, candidate, contract)
    record["name"] = contract["candidate"]
    result: dict[str, Any] = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v18r2.result",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "transport_family": contract["transport_family"],
        "representation_contract": contract["representation"],
        "model_contract": contract["model"],
        "safety_contract": contract["safety"],
        "decision_policy": contract["decision_policy"],
        "penalty_alias_value": contract["decision_policy"]["transport_penalty_points"],
        "candidate_count": 1,
        "pass_count": int(record["strict_internal_pass"]),
        "inherited_fit_count": 2,
        "additional_fit_count": 0,
        "prediction_changes": 0,
        "candidate": record,
        "representation_coverage": representation_coverage,
        "prediction_seal": prediction_seal,
        "recovery_contract": recovery,
        "outputs": [],
        "operations": {"historical_train_csv_reads": 1, "official_covariate_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0},
        "hashes": {
            "base_method_config_sha256": original.sha256(BASE_CONFIG),
            "runner_sha256": original.sha256(Path(__file__)),
            "recovery_config_sha256": original.sha256(RECOVERY_CONFIG),
            "technical_receipt_sha256": original.sha256(RECEIPT),
            "sealed_prediction_npz_sha256": original.sha256(PREDICTION_NPZ),
            "prediction_seal_sha256": original.sha256(PREDICTION_SEAL),
            "source_terminal_failure_sha256": original.sha256(SOURCE_FAILURE),
        },
    }
    result["independent_qa"] = independent_qa(result)
    original.base.write_json(ARTIFACT / "result.json", result)
    original.base.write_json(REPORT / "independent-qa.json", result["independent_qa"])
    original.base.write_json(
        ARTIFACT / "progress.json",
        {"phase": "terminal", "inherited_fit_count": 2, "additional_fit_count": 0, "pass_count": result["pass_count"]},
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate_only == args.execute:
        parser.error("choose exactly one mode")
    try:
        payload = validate_only() if args.validate_only else execute()
        print(json.dumps(original.base.native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if payload.get("status") != "INVALID" else 1
    except Exception as exc:  # noqa: BLE001
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "additional_fit_count": 0,
            "prediction_changes": 0,
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            original.base.write_json(ARTIFACT / "terminal_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
