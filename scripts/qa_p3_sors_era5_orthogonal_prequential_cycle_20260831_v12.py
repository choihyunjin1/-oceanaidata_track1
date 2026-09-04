"""Independent terminal QA for P3 v12."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = "p3_sors_era5_orthogonal_prequential_cycle_20260831_v12"
RESULT = ROOT / "artifacts" / EXP / "result.json"
REPORT = ROOT / "reports" / EXP / "report-source.md"
MANIFEST = ROOT / "reports" / EXP / "run-manifest.json"
RUNNER = ROOT / "scripts" / f"run_{EXP}.py"
OUTPUT = ROOT / "reports" / EXP / "independent-qa.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks = {
        "complete_zero_pass": result["status"] == "COMPLETE" and result["passing_candidate_count"] == 0,
        "three_candidates": result["candidate_count"] == 3,
        "three_actual_fits": result["fit_budget"]["actual_historical"] == 3 and result["fit_budget"]["actual_full"] == 0,
        "runner_hash": digest(RUNNER) == result["provenance"]["runner_sha256"],
        "result_hash": digest(RESULT) == manifest["result_sha256"],
        "report_hash": digest(REPORT) == manifest["report_sha256"],
        "official_hidden_upload_zero": all(value == 0 for value in result["data_access"].values()),
        "no_outputs": result["outputs"] == [],
        "all_gates_recomputed": all(item["passed"] == all(item["gate_checks"].values()) for item in result["candidates"]),
        "inactive_exact_noop": all(item["gate_checks"]["inactive_exact_noop"] for item in result["candidates"]),
        "raw_gate_exact": result["transport"]["equivalent_ci_upper_m"] == -0.020913058224751535,
    }
    payload = {"schema_version": "p3.v12.independent_qa", "created_at_utc": datetime.now(UTC).isoformat(), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "result_sha256": digest(RESULT), "runner_sha256": digest(RUNNER), "hidden_truth_rows_read": 0, "upload_attempt_count": 0}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
