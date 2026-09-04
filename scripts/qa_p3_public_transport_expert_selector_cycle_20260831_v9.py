"""Independent aggregate QA for terminal P3 v9."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = "p3_public_transport_expert_selector_cycle_20260831_v9"
RESULT = ROOT / "artifacts" / EXP / "result.json"
REPORT = ROOT / "reports" / EXP / "report-source.md"
MANIFEST = ROOT / "reports" / EXP / "run-manifest.json"
RUNNER = ROOT / "scripts" / f"run_{EXP}.py"
OUTPUT = ROOT / "reports" / EXP / "independent-qa.json"
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_PUBLIC_TRANSPORT_SELECTOR_V9"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "terminal_no_go": result["status"] == "COMPLETE"
        and result["decision"] == "NO_GO_PUBLIC_TRANSPORT_GATE",
        "candidate_pass_count": result["candidate_count"] == 3
        and result["passing_candidate_count"] == 0,
        "fit_count": result["fit_budget"]["actual_historical"] == 6
        and result["fit_budget"]["actual_full"] == 0,
        "selection_surface": result["data_profile"]["selection_matched_cases"] == 135
        and result["data_profile"]["selection_matched_rows"] == 810,
        "runner_hash": sha256(RUNNER) == result["provenance"]["runner_sha256"],
        "result_hash": sha256(RESULT) == manifest["result_sha256"],
        "report_hash": sha256(REPORT) == manifest["report_sha256"],
        "outputs_zero": result["outputs"] == [] and manifest["outputs"] == [],
        "delivery_absent": not DELIVERY.exists(),
        "official_hidden_upload_zero": all(value == 0 for value in result["data_access"].values()),
    }
    for item in result["candidates"]:
        name = item["spec"]["name"]
        checks[f"{name}_all_predictions_finite"] = item["gate_checks"]["finite_predictions"]
        checks[f"{name}_gate_recomputed"] = item["passed"] == all(item["gate_checks"].values())
        checks[f"{name}_strict_fail"] = not item["passed"]
        checks[f"{name}_point_math"] = abs(
            item["expected_points"]["raw_conservative"]
            - item["expected_points"]["public_reversal_penalty"]
            - item["expected_points"]["calibrated_conservative"]
        ) < 1e-12
    payload = {
        "schema_version": "p3.public_transport_expert_selector.independent_qa.v9",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "result_sha256": sha256(RESULT),
        "runner_sha256": sha256(RUNNER),
        "official_hidden_upload_rows": 0,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
