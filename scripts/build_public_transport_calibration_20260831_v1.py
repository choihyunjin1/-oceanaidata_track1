from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_RESULTS = (
    ROOT
    / "reports"
    / "parallel_internal_pass_registry_20260831_v1"
    / "official-submission-results-20260831.json"
)
PASS_REGISTRY = (
    ROOT
    / "reports"
    / "parallel_internal_pass_registry_20260831_v1"
    / "pass-registry.json"
)
LEGACY_RECEIPTS = (
    ROOT
    / "reports"
    / "submission_ladders_internal_validation_20260831_v1"
    / "p2_3_official_submission_receipt.json",
    ROOT
    / "reports"
    / "submission_ladders_internal_validation_20260831_v1"
    / "p3_2_official_submission_receipt.json",
)
OUTPUT = (
    ROOT
    / "reports"
    / "public_transport_calibration_20260831_v1"
    / "calibration.json"
)

MIN_POINT_GAIN = 0.01


def build_calibration(official: dict, registry: dict, legacy_receipts: list[dict]) -> dict:
    slopes = {
        "P1": abs(float(registry["empirical_score_mapping"]["P1_points_per_f1"])),
        "P2": abs(float(registry["empirical_score_mapping"]["P2_points_per_rmse_c"])),
        "P3": abs(float(registry["empirical_score_mapping"]["P3_points_per_rmse_m"])),
    }
    residuals: dict[str, list[dict]] = defaultdict(list)
    for submission in official["submissions"]:
        expected = float(submission["expected_points_delta"]["central"])
        actual = float(submission["official_points_delta_vs_best"])
        residuals[submission["problem"]].append(
            {
                "candidate": submission["candidate"],
                "expected_central_points_delta": expected,
                "actual_points_delta": actual,
                "transport_residual": actual - expected,
            }
        )

    for receipt in legacy_receipts:
        problem = receipt["problem"].replace("OCN-0", "P")
        expected = float(receipt["pre_submit_conditional_estimate"]["point_delta"])
        actual = float(receipt["actual_delta_vs_incumbent"]["points"])
        residuals[problem].append(
            {
                "candidate": receipt["candidate"],
                "expected_central_points_delta": expected,
                "actual_points_delta": actual,
                "transport_residual": actual - expected,
            }
        )

    gates = {}
    for problem in ("P1", "P2", "P3"):
        rows = residuals[problem]
        worst_residual = min(row["transport_residual"] for row in rows)
        penalty = max(0.0, -worst_residual)
        required_raw = MIN_POINT_GAIN + penalty
        gates[problem] = {
            "observations": len(rows),
            "observed_pairs": rows,
            "worst_observed_transport_residual_points": worst_residual,
            "transport_penalty_points": penalty,
            "minimum_calibrated_expected_points_delta": MIN_POINT_GAIN,
            "minimum_uncalibrated_expected_points_delta": required_raw,
            "full_transport_metric_improvement_equivalent": required_raw / slopes[problem],
            "pass_rule": (
                "candidate_raw_expected_points_delta + "
                "worst_observed_transport_residual_points >= 0.01"
            ),
        }

    return {
        "schema_version": "public_transport_calibration.20260831.v1",
        "status": "CALIBRATED_GATE_READY",
        "minimum_point_gain": MIN_POINT_GAIN,
        "calibration_method": (
            "Worst observed candidate-specific residual between pre-submission central "
            "expected point delta and official Public point delta. This is an empirical "
            "guardrail, not a confidence bound."
        ),
        "gates": gates,
        "invariants": {
            "strictly_time_ordered_validation": True,
            "group_or_episode_blocking_required": True,
            "nested_selection_or_frozen_candidate_required": True,
            "official_public_score_not_used_as_training_label": True,
            "candidates_per_problem_min": 1,
            "candidates_per_problem_max": 3,
        },
    }


def main() -> None:
    official = json.loads(OFFICIAL_RESULTS.read_text(encoding="utf-8"))
    registry = json.loads(PASS_REGISTRY.read_text(encoding="utf-8"))
    legacy_receipts = [
        json.loads(path.read_text(encoding="utf-8")) for path in LEGACY_RECEIPTS
    ]
    result = build_calibration(official, registry, legacy_receipts)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
