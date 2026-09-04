from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "reports/public_transport_calibration_20260831_v2/calibration.json"
LEDGER = ROOT / "reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json"
OUTPUT = ROOT / "reports/public_transport_calibration_20260831_v3/calibration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    p1 = next(item for item in v2["observed_pairs"] if item["problem"] == "P1")
    p1_ledger = next(item for item in ledger["submissions"] if item["problem"] == "P1")
    residual = max(0.0, float(p1["expected_central_points_delta"]) - float(p1["actual_points_delta"]))
    if p1_ledger["candidate"] != p1["candidate"] or p1_ledger["official_points_delta_vs_best"] != p1["actual_points_delta"]:
        raise RuntimeError("P1 official ledger and v2 pair disagree")
    minimum = float(v2["minimum_calibrated_expected_points_delta"])
    return {
        "schema_version": "public_transport_calibration.20260831.v3",
        "status": "PROSPECTIVE_PROBLEM_SPECIFIC_GUARDRAIL_READY",
        "effective_scope": "newly preregistered experiments after this v3 only",
        "minimum_calibrated_expected_points_delta": minimum,
        "method": "Same-problem-first empirical transport guardrail. Use same-problem exact family worst adverse residual; otherwise same-problem tier worst when available; otherwise same-problem worst observed adverse residual. Only when a problem has no official pair may the v2 cross-problem tier/global fallback be used.",
        "precedence": ["same_problem_exact_family", "same_problem_tier", "same_problem_worst_observed", "cross_problem_v2_fallback_only_if_problem_has_zero_pairs"],
        "p1": {
            "observed_pair_count": 1,
            "observed_candidate": p1["candidate"],
            "observed_family_id": p1["family_id"],
            "observed_tier_id": p1["tier_id"],
            "observed_expected_central_points_delta": p1["expected_central_points_delta"],
            "observed_actual_points_delta": p1["actual_points_delta"],
            "observed_adverse_residual_points": residual,
            "prospective_exact_family_penalty_points": residual,
            "prospective_unseen_family_or_tier_penalty_points": residual,
            "prospective_minimum_raw_expected_points_delta": residual + minimum,
            "uncertainty_status": "PROVISIONAL_N1_NOT_A_CONFIDENCE_INTERVAL",
            "relaxation_review_minimum_same_problem_pairs": 3,
            "metric": "F1 classification score mapped to competition points",
            "mechanism": "anchor-preserving add-only intervention"
        },
        "p2_p3": {"policy": "unchanged_from_v2", "reason": "v3 audits P1 only and does not retroactively alter other problem gates"},
        "policy": {
            "retroactive_reclassification_forbidden": True,
            "existing_v5_v22_results_unchanged": True,
            "minimum_plus_0_01_points_maintained": True,
            "public_score_as_row_label_forbidden": True,
            "candidate_family_registered_before_internal_results": True,
            "nested_or_frozen_validation_required": True,
            "single_pair_penalty_is_empirical_floor_not_interval": True,
            "no_future_penalty_reduction_before_three_same_problem_pairs": True
        },
        "provenance": {"v2_sha256": sha256(V2), "official_ledger_sha256": sha256(LEDGER), "official_row_reads": 0, "hidden_truth_reads": 0, "uploads": 0}
    }


def main() -> None:
    result = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
