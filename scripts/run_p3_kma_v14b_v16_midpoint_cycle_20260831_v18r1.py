"""Evaluation-only recovery for v18 after a dtype-only materialization failure."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

from run_p3_kma_v14b_v16_midpoint_cycle_20260831_v18 import (  # noqa: E402
    attach_energy,
    canonical,
    evaluate,
    load_energy_history,
    load_historical,
    sha256,
    write_new,
)

EXPERIMENT_ID = "p3_kma_v14b_v16_midpoint_cycle_20260831_v18r1"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CONFIG = ROOT / "configs/experiments/p3_kma_v14b_v16_midpoint_cycle_20260831_v18r1.json"
SOURCE_RUNNER = ROOT / "scripts/run_p3_kma_v14b_v16_midpoint_cycle_20260831_v18.py"
SOURCE_CONFIG = ROOT / "configs/experiments/p3_kma_v14b_v16_midpoint_cycle_20260831_v18.json"
SOURCE_LOCK = ROOT / "artifacts/p3_kma_v14b_v16_midpoint_cycle_20260831_v18.ATTEMPT_LOCK.json"
SOURCE_STDERR = ROOT / "artifacts/p3_kma_v14b_v16_midpoint_cycle_20260831_v18.execute.stderr.log"
SOURCE_RUNNER_SHA = "268b7e71f75b61efa5dab75c17376b12be6a2eb26c55158b218ce246d5808cad"
SOURCE_CONFIG_SHA = "441a1fc102c28e51fb31979196ae83d2d9ca35bfc4d87f48f6742592ba75a406"
SOURCE_LOCK_SHA = "90f653fdeee3e2c399d921a1442a4faf2ff09b845ada8d9869585d9b1adb073d"


class ContractError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "RECOVERY_READY"}))
        return 0
    if ARTIFACT_DIR.exists() or LOCK.exists():
        raise ContractError("recovery artifact or lock exists")
    if sha256(SOURCE_RUNNER) != SOURCE_RUNNER_SHA or sha256(SOURCE_CONFIG) != SOURCE_CONFIG_SHA or sha256(SOURCE_LOCK) != SOURCE_LOCK_SHA:
        raise ContractError("source v18 provenance changed")
    if "official key/order/energy contract failed" not in SOURCE_STDERR.read_text(encoding="utf-8"):
        raise ContractError("source failure signature changed")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runner_hash = sha256(Path(__file__))
    started = time.perf_counter()
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_EVALUATION_ONLY_RECOVERY", "created_at_utc": datetime.now(UTC).isoformat(), "runner_sha256": runner_hash, "config_sha256": sha256(CONFIG), "source_runner_sha256": SOURCE_RUNNER_SHA, "source_lock_sha256": SOURCE_LOCK_SHA, "official_reads": 0}))
    frame, profile = load_historical()
    history = load_energy_history()
    candidate = evaluate(attach_energy(frame, history), history)
    result = {"schema_version": "p3.kma_v14b_v16_midpoint.recovery_result.v18r1", "experiment_id": EXPERIMENT_ID, "source_experiment": "p3_kma_v14b_v16_midpoint_cycle_20260831_v18", "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "PASS_INTERNAL_AWAITING_SEPARATE_MATERIALIZER" if candidate["passed"] else "NO_GO_RECOVERED_INTERNAL_GATE", "passing_candidate_count": int(candidate["passed"]), "candidate": candidate, "fit_budget": {"prefix_ridge_solves": 6, "model_fits": 0}, "data_profile": profile, "outputs": [], "data_access": {"official_test_index_rows_read": 0, "official_test_context_rows_read": 0, "official_feature_rows_read": 0, "official_component_prediction_rows_read": 0, "hidden_truth_rows_read": 0, "uploads": 0}, "provenance": {"runner_sha256": runner_hash, "config_sha256": sha256(CONFIG), "source_runner_sha256": SOURCE_RUNNER_SHA, "source_config_sha256": SOURCE_CONFIG_SHA, "source_lock_sha256": SOURCE_LOCK_SHA, "source_stderr_sha256": sha256(SOURCE_STDERR)}, "execution": {"elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(), "candidate_or_gate_changed": False, "official_materialization_attempted": False, "result_based_tuning_or_retry": False}}
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report = "# P3 v18r1 evaluation-only recovery\n\n" + f"- Decision: **{result['decision']}**; official/hidden/upload 0.\n" + f"- pooled delta {candidate['delta_candidate_minus_reference_rmse_m']:+.9f}m; central calibrated {candidate['expected_points']['central_calibrated']:.9f} points.\n" + "- Candidate, midpoint weight, theta calculation, split, and gates are bit-identical to sealed v18.\n"
    write_new(REPORT_DIR / "report-source.md", report.encode("utf-8"))
    print(json.dumps({"status": "COMPLETE", "passing": result["passing_candidate_count"], "official_reads": 0}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
