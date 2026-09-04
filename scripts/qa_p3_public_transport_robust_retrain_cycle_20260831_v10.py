"""Independent aggregate QA for the terminal P3 v10 one-shot."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = "p3_public_transport_robust_retrain_cycle_20260831_v10"
RESULT = ROOT / "artifacts" / EXP / "result.json"
REPORT = ROOT / "reports" / EXP / "report-source.md"
MANIFEST = ROOT / "reports" / EXP / "run-manifest.json"
OUTPUT = ROOT / "reports" / EXP / "independent-qa.json"
RUNNER = ROOT / "scripts" / f"run_{EXP}.py"
PRIOR = ROOT / "reports" / EXP / "prior-results-audit.json"
SLOPE = 15.870739046986959


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    penalty = float(result["transport"]["penalty_points"])
    checks: dict[str, bool] = {
        "terminal_complete": result["status"] == "COMPLETE",
        "decision_no_go": result["decision"] == "NO_GO_PUBLIC_TRANSPORT_GATE",
        "three_candidates_zero_pass": result["candidate_count"] == 3
        and result["passing_candidate_count"] == 0,
        "six_historical_fits_zero_full_fits": result["fit_budget"]["actual_historical"] == 6
        and result["fit_budget"]["actual_full"] == 0
        and result["fit_budget"]["actual_total"] == 6,
        "runner_hash_matches_terminal": sha256(RUNNER) == result["provenance"]["runner_sha256"],
        "result_hash_matches_manifest": sha256(RESULT) == manifest["result_sha256"],
        "report_hash_matches_manifest": sha256(REPORT) == manifest["report_sha256"],
        "no_outputs": result["outputs"] == [] and manifest["outputs"] == [],
        "official_access_zero": all(value == 0 for value in result["data_access"].values()),
        "hidden_upload_zero": result["execution"]["hidden_truth_rows_read"] == 0
        and result["execution"]["upload_attempt_count"] == 0,
        "prior_audit_excluded_sensitive_paths": len(prior["excluded_official_hidden_submission_paths"]) > 0,
    }
    translations: list[dict[str, object]] = []
    for item in result["candidates"]:
        low, high = item["episode_bootstrap"]["ci90_m"]
        central = -float(item["delta_candidate_minus_reference_rmse_m"]) * SLOPE
        conservative = max(0.0, -float(high) * SLOPE)
        optimistic = max(0.0, -float(low) * SLOPE)
        name = item["spec"]["name"]
        checks[f"{name}_point_math"] = (
            abs(central - item["expected_points"]["raw_central"]) < 1e-12
            and abs(conservative - item["expected_points"]["raw_conservative"]) < 1e-12
            and abs(conservative - penalty - item["expected_points"]["calibrated_conservative"])
            < 1e-12
        )
        checks[f"{name}_gate_recomputed"] = item["passed"] == all(item["gate_checks"].values())
        checks[f"{name}_strict_fail"] = not item["passed"]
        checks[f"{name}_validation_rows_preserved"] = all(
            receipt.get("validation_rows_deleted", 0) == 0
            for receipt in item["fit_receipts"]
        )
        translations.append(
            {
                "candidate": name,
                "raw_points_delta": {
                    "conservative": conservative,
                    "central": central,
                    "optimistic": optimistic,
                },
                "public_calibrated_points_delta": {
                    "conservative": conservative - penalty,
                    "central": central - penalty,
                    "optimistic": optimistic - penalty,
                },
            }
        )
    payload = {
        "schema_version": "p3.public_transport_robust_retrain.independent_qa.v10",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "score_translation": translations,
        "result_sha256": sha256(RESULT),
        "runner_sha256": sha256(RUNNER),
        "hidden_truth_rows_read": 0,
        "upload_attempt_count": 0,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
