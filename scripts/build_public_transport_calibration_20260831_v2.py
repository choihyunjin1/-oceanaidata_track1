from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_CALIBRATION = (
    ROOT
    / "reports"
    / "public_transport_calibration_20260831_v1"
    / "calibration.json"
)
PASS_REGISTRY = (
    ROOT
    / "reports"
    / "parallel_internal_pass_registry_20260831_v1"
    / "pass-registry.json"
)
OUTPUT = (
    ROOT
    / "reports"
    / "public_transport_calibration_20260831_v2"
    / "calibration.json"
)

MIN_POINT_GAIN = 0.01
GLOBAL_FALLBACK_PENALTY = 0.3219056897594759


PAIR_CLASSES = {
    "P1_2_HIST_GBDT_OOF_STACK_UNION": (
        "P1_FIXED_ADD_ONLY_UNION",
        "LOW_DOF_FIXED",
    ),
    "P2_2_HGB_ABSOLUTE_PROFILE": (
        "P2_HGB_ABSOLUTE_PROFILE",
        "SMOOTH_LEARNED_PROFILE",
    ),
    "P2_1_PUBLIC_PROFILE_RESIDUAL_SHALLOW": (
        "P2_SHALLOW_RESIDUAL_PROFILE",
        "SMOOTH_LEARNED_PROFILE",
    ),
    "P2_3_BIN17_DROP_LAYER4": ("P2_FIXED_DROP_LAYER4", "LOW_DOF_FIXED"),
    "P3_2_EXTRATREES_HARD_PHYSICAL_ROUTER": (
        "P3_EXTRATREES_HARD_ROUTER",
        "HARD_CONDITIONAL_ROUTER",
    ),
    "P3_2_KMA_A18_0200_A24_0425": (
        "P3_FIXED_KMA_LONGLEAD_FACTOR",
        "LOW_DOF_FIXED",
    ),
}


def _negative_residual_penalty(row: dict) -> float:
    return max(0.0, -float(row["transport_residual"]))


def build_calibration(v1: dict, registry: dict) -> dict:
    slopes = {
        "P1": abs(float(registry["empirical_score_mapping"]["P1_points_per_f1"])),
        "P2": abs(
            float(registry["empirical_score_mapping"]["P2_points_per_rmse_c"])
        ),
        "P3": abs(
            float(registry["empirical_score_mapping"]["P3_points_per_rmse_m"])
        ),
    }
    observed = []
    for problem, gate in v1["gates"].items():
        for row in gate["observed_pairs"]:
            candidate = row["candidate"]
            family_id, tier_id = PAIR_CLASSES[candidate]
            observed.append(
                {
                    "problem": problem,
                    **row,
                    "family_id": family_id,
                    "tier_id": tier_id,
                    "adverse_penalty_points": _negative_residual_penalty(row),
                }
            )

    family_penalties: dict[str, float] = {}
    tier_penalties: dict[str, float] = {}
    for row in observed:
        family = row["family_id"]
        tier = row["tier_id"]
        penalty = float(row["adverse_penalty_points"])
        family_penalties[family] = max(family_penalties.get(family, 0.0), penalty)
        tier_penalties[tier] = max(tier_penalties.get(tier, 0.0), penalty)

    tier_penalties["UNKNOWN_OR_COMPOUND"] = GLOBAL_FALLBACK_PENALTY
    for tier in (
        "LOW_DOF_FIXED",
        "SMOOTH_LEARNED_PROFILE",
        "HARD_CONDITIONAL_ROUTER",
    ):
        tier_penalties.setdefault(tier, GLOBAL_FALLBACK_PENALTY)

    family_gates = {
        family: {
            "transport_penalty_points": penalty,
            "minimum_raw_expected_points_delta": penalty + MIN_POINT_GAIN,
        }
        for family, penalty in sorted(family_penalties.items())
    }
    tier_gates = {
        tier: {
            "transport_penalty_points": penalty,
            "minimum_raw_expected_points_delta": penalty + MIN_POINT_GAIN,
        }
        for tier, penalty in sorted(tier_penalties.items())
    }

    examples = {
        "P1_FIXED_ADD_ONLY_UNION": {
            "problem": "P1",
            "metric_improvement_equivalent": (
                family_gates["P1_FIXED_ADD_ONLY_UNION"][
                    "minimum_raw_expected_points_delta"
                ]
                / slopes["P1"]
            ),
        },
        "P2_FIXED_DROP_LAYER4": {
            "problem": "P2",
            "metric_improvement_equivalent": (
                family_gates["P2_FIXED_DROP_LAYER4"][
                    "minimum_raw_expected_points_delta"
                ]
                / slopes["P2"]
            ),
        },
        "P3_FIXED_KMA_LONGLEAD_FACTOR": {
            "problem": "P3",
            "metric_improvement_equivalent": (
                family_gates["P3_FIXED_KMA_LONGLEAD_FACTOR"][
                    "minimum_raw_expected_points_delta"
                ]
                / slopes["P3"]
            ),
        },
        "P3_UNSEEN_LOW_DOF_FIXED": {
            "problem": "P3",
            "metric_improvement_equivalent": (
                tier_gates["LOW_DOF_FIXED"]["minimum_raw_expected_points_delta"]
                / slopes["P3"]
            ),
        },
    }

    return {
        "schema_version": "public_transport_calibration.20260831.v2",
        "status": "FAMILY_AWARE_GUARDRAIL_READY",
        "minimum_calibrated_expected_points_delta": MIN_POINT_GAIN,
        "method": (
            "Pre-registered hierarchical empirical guardrail. Use the worst adverse "
            "official residual from an exact intervention family; otherwise use the "
            "worst residual in a pre-registered complexity tier; otherwise use the "
            "global worst residual. This is not a confidence interval."
        ),
        "precedence": [
            "exact_family",
            "pre_registered_complexity_tier",
            "global_unknown_or_compound_fallback",
        ],
        "observed_pairs": observed,
        "family_gates": family_gates,
        "tier_gates": tier_gates,
        "global_fallback": {
            "transport_penalty_points": GLOBAL_FALLBACK_PENALTY,
            "minimum_raw_expected_points_delta": (
                GLOBAL_FALLBACK_PENALTY + MIN_POINT_GAIN
            ),
        },
        "metric_equivalent_examples": examples,
        "policy": {
            "retroactive_reclassification_forbidden": True,
            "effective_for_newly_registered_experiments_only": True,
            "public_score_as_row_or_event_label_forbidden": True,
            "candidate_family_must_be_registered_before_internal_results": True,
            "representation_and_router_both_changed_uses_global_fallback": True,
            "same_family_adverse_residual_updates_by_max_only": True,
            "minimum_three_same_family_pairs_before_any_future_relaxation_review": 3,
            "inclusive_pass_rule": (
                "raw_expected_points_delta - selected_transport_penalty_points >= 0.01"
            ),
        },
        "required_registration_fields": [
            "family_id",
            "tier_id",
            "representation_changed",
            "routing_discontinuous",
            "active_share_rule",
            "exact_comparator",
            "local_lcb_raw_points",
            "selected_penalty_provenance_sha256",
        ],
        "validation_invariants": v1["invariants"],
    }


def select_penalty(
    calibration: dict,
    *,
    family_id: str,
    tier_id: str,
    representation_changed: bool,
    routing_discontinuous: bool,
) -> dict:
    if representation_changed and routing_discontinuous:
        source = "global_unknown_or_compound_fallback"
        gate = calibration["global_fallback"]
    elif family_id in calibration["family_gates"]:
        source = f"exact_family:{family_id}"
        gate = calibration["family_gates"][family_id]
    elif tier_id in calibration["tier_gates"]:
        source = f"complexity_tier:{tier_id}"
        gate = calibration["tier_gates"][tier_id]
    else:
        source = "global_unknown_or_compound_fallback"
        gate = calibration["global_fallback"]
    return {"source": source, **gate}


def main() -> None:
    v1 = json.loads(V1_CALIBRATION.read_text(encoding="utf-8"))
    registry = json.loads(PASS_REGISTRY.read_text(encoding="utf-8"))
    result = build_calibration(v1, registry)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
