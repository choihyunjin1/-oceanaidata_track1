"""Independent structural QA for the P2 v13 fresh-surface audit."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "p2_v13_fresh_authenticated_surface_audit_20260901_v13c"
RESULT = ROOT / "reports" / ID / "result.json"
OUTPUT = ROOT / "reports" / ID / "independent-qa.json"


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks = {
        "terminal_hold": result["status"] == "HOLD_NO_FRESH_AUTHENTICATED_SURFACE",
        "zero_fit": result["model_fits"] == 0,
        "zero_predictions": result["candidate_predictions"] == 0,
        "zero_target_values": result["operation_counters"]["target_value_rows_read"] == 0,
        "zero_official": result["operation_counters"]["official_rows_read"] == 0,
        "zero_hidden": result["operation_counters"]["hidden_rows_read"] == 0,
        "zero_csv": result["operation_counters"]["submission_csv_created"] == 0,
        "zero_upload": result["operation_counters"]["uploads"] == 0,
        "no_fresh_frame": result["finding"]["fresh_authenticated_scoring_frame_count"] == 0,
        "no_repartition": result["finding"]["post_v13_repartition_forbidden"] is True,
        "commitment_preserved": result["decision"]["v13_commitment_preserved"] is True,
        "no_posthoc_routing": result["decision"]["v13_posthoc_routing"] is False,
    }
    payload = {
        "experiment_id": ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
