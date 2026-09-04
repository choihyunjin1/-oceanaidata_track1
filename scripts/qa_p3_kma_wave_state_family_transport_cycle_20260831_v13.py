"""Independent aggregate QA for P3 v13."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = "p3_kma_wave_state_family_transport_cycle_20260831_v13"
RESULT = ROOT / "artifacts" / EXP / "result.json"
REPORT = ROOT / "reports" / EXP / "report-source.md"
MANIFEST = ROOT / "reports" / EXP / "run-manifest.json"
RUNNER = ROOT / "scripts" / f"run_{EXP}.py"
CONFIG = ROOT / "configs/experiments" / f"{EXP}.json"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v2/calibration.json"
OUTPUT = ROOT / "reports" / EXP / "independent-qa.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks = {
        "complete_zero_pass": result["status"] == "COMPLETE" and result["passing_candidate_count"] == 0,
        "three_candidates": result["candidate_count"] == 3,
        "twelve_train_only_calibrations": result["fit_budget"]["train_only_calibration_fits"] == 12,
        "runner_hash": digest(RUNNER) == result["provenance"]["runner_sha256"],
        "config_hash": digest(CONFIG) == result["provenance"]["config_sha256"],
        "calibration_hash": digest(CALIBRATION) == "1a1d2c96cbe6c2c69b753fb4a130eb092922cc46524286cabcc0f9fce150441f",
        "result_hash": digest(RESULT) == manifest["result_sha256"],
        "report_hash": digest(REPORT) == manifest["report_sha256"],
        "official_hidden_upload_zero": all(value == 0 for value in result["data_access"].values()),
        "no_outputs": result["outputs"] == [],
        "all_gates_recomputed": all(item["passed"] == all(item["gate_checks"].values()) for item in result["candidates"]),
        "family_penalties_pre_registered": result["candidates"][0]["spec"]["penalty"] == 0.04958605409228893 and all(item["spec"]["penalty"] == 0.3219056897594759 for item in result["candidates"][1:]),
        "exact_comparator_registered": all(item["spec"]["exact_comparator"] == "uniform_kma_alpha_0.425" for item in result["candidates"]),
    }
    payload = {"schema_version": "p3.v13.independent_qa", "created_at_utc": datetime.now(UTC).isoformat(), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "result_sha256": digest(RESULT), "runner_sha256": digest(RUNNER), "config_sha256": digest(CONFIG), "calibration_sha256": digest(CALIBRATION), "hidden_truth_rows_read": 0, "upload_attempt_count": 0}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
