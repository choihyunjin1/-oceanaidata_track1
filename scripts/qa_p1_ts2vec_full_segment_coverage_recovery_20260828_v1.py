"""Independent aggregate-only QA for P1 full-segment coverage recovery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_ts2vec_full_segment_coverage_recovery_20260828_v1.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p1_coverage_recovery_runner_qa", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_runner()
    config = runner._json(runner.CONFIG)
    contract = runner.validate_contract(config)
    output = ROOT / config["artifacts"]["directory"]
    result_path = output / "result.json"
    manifest_path = output / "manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_files = [path for path in output.rglob("*") if path.is_file()]
    forbidden = [
        str(path.relative_to(output))
        for path in artifact_files
        if path.suffix.lower() == ".csv" or "submission" in path.name.lower()
    ]
    gate_pass = all(result.get("gate", {}).values())
    checks = {
        "contract_pass": contract["status"] == "PASS",
        "parent_hash_frozen": manifest["parent_encoder_sha256"]
        == config["parent_experiment"]["encoder_sha256"],
        "result_hash_matches_manifest": manifest["result_sha256"] == runner.sha256_file(result_path),
        "coverage_complete": float(result["coverage_audit"]["coverage"]) >= float(config["historical_gate"]["coverage_min"]),
        "every_historical_row_covered": result["coverage_audit"]["eligible_rows"] == result["coverage_audit"]["covered_rows"],
        "new_model_fit_count_zero": result["new_model_fit_count"] == 0,
        "status_matches_gate": result["status"]
        == ("READY_FOR_SEPARATE_Q2_OUTER_RESEARCH_ONLY" if gate_pass else "NO_GO_HISTORICAL_GATE"),
        "q2_truth_rows_zero": result["q2_truth_rows_read"] == 0,
        "q3_q4_truth_rows_zero": result["q3_q4_truth_rows_read"] == 0,
        "official_rows_zero": result["official_rows_read"] == 0,
        "no_submission_artifact": not forbidden,
        "submission_false": result["submission_generated_or_uploaded"] is False,
    }
    qa = {
        "schema_version": "p1.ts2vec_full_segment_coverage_recovery.qa.v1",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_decision": result["status"],
        "checks": checks,
        "artifacts": {
            "result.json": {"bytes": result_path.stat().st_size, "sha256": runner.sha256_file(result_path)},
            "manifest.json": {"bytes": manifest_path.stat().st_size, "sha256": runner.sha256_file(manifest_path)},
        },
    }
    qa_path = output / "independent_qa.json"
    runner._atomic_json(qa_path, qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
