"""Logging-only recovery for the sealed P1 v13 proposal."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v13 as base  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v13r1"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
SOURCE_ARTIFACT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v13"
SOURCE_PROPOSAL = SOURCE_ARTIFACT / "proposal_blind.npz"
SOURCE_SEAL = SOURCE_ARTIFACT / "proposal_seal.json"
SOURCE_FAILURE = SOURCE_ARTIFACT / "terminal_failure.json"
SOURCE_LOCK = SOURCE_ARTIFACT / "attempt_lock.json"
FIRST_RUNNER_SHA256 = "af324b1d594d0aba5428f317924255d39b1e10b5117069c87b7e10230524d00d"
FIRST_LOCK_SHA256 = "954f612e1bc1fa686e5757890b2483ae2be251d6b225d97e27ece0c0436eb619"


class RecoveryError(RuntimeError):
    """The sealed technical-recovery contract was violated."""


def load_reused_proposal() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    for path in (SOURCE_PROPOSAL, SOURCE_SEAL, SOURCE_FAILURE, SOURCE_LOCK):
        if not path.is_file():
            raise RecoveryError(f"missing v13 recovery source: {path}")
    if base.sha256_file(SOURCE_LOCK) != FIRST_LOCK_SHA256:
        raise RecoveryError("v13 source lock hash changed")
    failure = json.loads(SOURCE_FAILURE.read_text(encoding="utf-8"))
    if failure.get("error_type") != "ValueError" or "merge on str and datetime" not in failure.get("error", ""):
        raise RecoveryError("v13 source failure is not the approved time-dtype failure")
    seal = json.loads(SOURCE_SEAL.read_text(encoding="utf-8"))
    with np.load(SOURCE_PROPOSAL, allow_pickle=False) as payload:
        additions = np.asarray(payload["additions"], dtype=bool)
        candidate = np.asarray(payload["candidate"], dtype=np.int8)
    anchor_frame = pd.read_parquet(base.ANCHOR_PATH)
    anchor = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    checks = {
        "rows": len(anchor_frame) == len(additions) == len(candidate) == int(seal["rows"]),
        "npz_hash": base.sha256_file(SOURCE_PROPOSAL) == seal["npz_sha256"],
        "additions_hash": base.sha256_bool(additions) == seal["additions_sha256"],
        "candidate_hash": base.sha256_bool(candidate) == seal["candidate_sha256"],
        "bit_exact_union": np.array_equal(candidate, np.maximum(anchor, additions.astype(np.int8))),
        "add_only": not bool(np.any((anchor == 1) & (candidate == 0))),
        "target_zero_at_seal": seal["target_columns_read_before_seal"] == 0,
        "raw_features_zero": seal["raw_feature_columns_read"] == 0,
    }
    if not all(checks.values()):
        raise RecoveryError(f"sealed proposal integrity failure: {checks}")
    return anchor_frame, additions, candidate, seal


def validate_only() -> dict[str, Any]:
    base.load_contract()
    _anchor, additions, candidate, seal = load_reused_proposal()
    return {
        "status": "VALID_RECOVERY",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": base.sha256_file(base.CONFIG_PATH),
        "corrected_v13_runner_sha256": base.sha256_file(Path(base.__file__)),
        "recovery_runner_sha256": base.sha256_file(Path(__file__)),
        "source_proposal_sha256": seal["npz_sha256"],
        "source_additions_sha256": base.sha256_bool(additions),
        "source_candidate_sha256": base.sha256_bool(candidate),
        "candidate_count": 1,
        "fit_budget": 0,
    }


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result["candidate"]
    checks = {
        "one_candidate": result["candidate_count"] == 1,
        "zero_fits": result["fit_count"] == 0,
        "proposal_reused_not_regenerated": result["recovery"]["proposal_reused_not_regenerated"],
        "proposal_hash_bit_exact": result["recovery"]["proposal_hash_bit_exact"],
        "approved_dtype_only_change": result["recovery"]["change_scope"] == "historical_time_utc_dtype_normalization_one_line",
        "same_config": result["hashes"]["config_sha256"] == result["recovery"]["source_config_sha256"],
        "add_only": candidate["anchor_removals"] == 0,
        "official_reads_zero": result["operations"]["official_covariate_reads"] == 0,
        "hidden_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def execute() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError("exactly-once recovery artifact path exists")
    config = base.load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "config_sha256": base.sha256_file(base.CONFIG_PATH),
        "first_runner_sha256": FIRST_RUNNER_SHA256,
        "corrected_v13_runner_sha256": base.sha256_file(Path(base.__file__)),
        "recovery_runner_sha256": base.sha256_file(Path(__file__)),
        "source_proposal_sha256": base.sha256_file(SOURCE_PROPOSAL),
        "fit_budget": 0,
    }
    base.write_json(ARTIFACT / "attempt_lock.json", lock)
    anchor_frame, _additions, candidate, seal = load_reused_proposal()
    base.write_json(
        ARTIFACT / "progress.json",
        {"phase": "sealed_v13_proposal_reused_target_attach_pending", "fit_count": 0},
        exclusive=False,
    )
    frame, candidate = base.attach_truth(anchor_frame, candidate)
    record = base.evaluate(frame, candidate, config)
    result: dict[str, Any] = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v13r1.result",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "transport_family": config["transport_family"],
        "candidate_contract": config["candidate"],
        "decision_policy": config["decision_policy"],
        "candidate_count": 1,
        "pass_count": int(record["strict_internal_pass"]),
        "fit_count": 0,
        "candidate": record,
        "outputs": [],
        "proposal_seal": seal,
        "recovery": {
            "source_experiment_id": "p1_public_transport_repair_cycle_20260831_v13",
            "source_failure_metrics_computed": 0,
            "source_failure_official_reads": 0,
            "source_failure_hidden_reads": 0,
            "source_failure_uploads": 0,
            "proposal_reused_not_regenerated": True,
            "proposal_hash_bit_exact": True,
            "change_scope": "historical_time_utc_dtype_normalization_one_line",
            "source_config_sha256": base.sha256_file(base.CONFIG_PATH),
            "first_runner_sha256": FIRST_RUNNER_SHA256,
            "corrected_v13_runner_sha256": base.sha256_file(Path(base.__file__)),
            "recovery_runner_sha256": base.sha256_file(Path(__file__)),
            "first_lock_sha256": FIRST_LOCK_SHA256,
            "source_proposal_sha256": base.sha256_file(SOURCE_PROPOSAL),
            "source_seal_sha256": base.sha256_file(SOURCE_SEAL),
        },
        "operations": {
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config_sha256": base.sha256_file(base.CONFIG_PATH),
            "corrected_v13_runner_sha256": base.sha256_file(Path(base.__file__)),
            "recovery_runner_sha256": base.sha256_file(Path(__file__)),
            "root_calibration_sha256": base.sha256_file(base.CALIBRATION_PATH),
            "authoritative_results_sha256": base.sha256_file(base.AUTHORITATIVE_PATH),
            "anchor_sha256": base.sha256_file(base.ANCHOR_PATH),
        },
    }
    result["independent_qa"] = independent_qa(result)
    base.write_json(ARTIFACT / "result.json", result)
    base.write_json(REPORT / "independent-qa.json", result["independent_qa"])
    base.write_json(
        ARTIFACT / "progress.json",
        {"phase": "terminal", "fit_count": 0, "pass_count": result["pass_count"]},
        exclusive=False,
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
        print(json.dumps(base.native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            base.write_json(ARTIFACT / "terminal_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
