"""Dry-run validation for the P2 authoritative nested-surrogate contract.

The validator consumes aggregate JSON evidence only.  It never opens training
data, OOF prediction values, official test data, or submission artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EXACTNESS_DIMENSIONS = (
    "population and row keys",
    "outer folds",
    "training cutoffs",
    "complete-pipeline seeds",
    "prefix mask semantics",
    "component OOF availability",
    "epoch and meta-refit semantics",
    "postprocess opportunity and metric aggregation",
)
BLOCKING_GAPS = (
    "DEEP_PREFIX_MASK_API_MISSING",
    "PREFIX_LOCAL_COMPONENT_OOF_RECIPE_MISSING",
    "PREFIX_LOCAL_EPOCH_SELECTION_RECIPE_MISSING",
    "COMPLETE_REFERENCE_SEED_REPLICATES_MISSING",
    "REFERENCE_SEED_FROZEN_OOF_EQUALITY_IMPOSSIBLE_ON_REQUIRED_KEYS",
    "META_PARAMETER_REFIT_SEMANTICS_UNDEFINED",
)
FOLDS = (
    "outer_2024_sep_oct",
    "outer_2025_may_jun",
    "outer_2025_jul_aug",
)
PREFIX_FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
SEEDS = (20260823, 20260824, 20260825)
SEMANTIC_ONLY_DIMENSIONS = {
    "prefix mask semantics",
    "component OOF availability",
    "epoch and meta-refit semantics",
}


@dataclass(frozen=True)
class ContractValidation:
    decision: dict[str, Any]
    state_matrix: dict[str, Any]
    comparison_preregistration: dict[str, Any]
    qa: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ids(values: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str(value["id"]) for value in values)


def _validate_recon_preregistration(value: dict[str, Any]) -> None:
    _require(
        value.get("status") == "SEALED_BEFORE_SECONDARY_RECON_RESULTS",
        "decisive reconstruction preregistration is not sealed",
    )
    p2 = value["problem_contracts"]["P2"]
    _require(
        tuple(p2["exactness_dimensions"]) == EXACTNESS_DIMENSIONS,
        "P2 exactness dimensions changed",
    )
    _require(
        set(p2["verdicts"])
        == {
            "EXACT_COMPARISON_REPRODUCIBLE",
            "NEW_AUTHORITATIVE_SURROGATE_REQUIRED",
            "CENTRAL_CONTRACT_REVISION_REQUIRED",
        },
        "P2 verdict set changed",
    )
    _require(not p2["promotion_from_current_surrogate_allowed"], "promotion gate changed")
    _require(
        not p2["public_score_based_recipe_selection_allowed"],
        "Public-score quarantine changed",
    )


def _validate_exact_audit(value: dict[str, Any]) -> None:
    _require(
        value.get("verdict")
        == "NOT_CURRENTLY_REPRODUCIBLE_AS_EXACT_SAME_PREFIX_FOLD_TRAIN_ONLY_INCUMBENT",
        "exact audit verdict changed",
    )
    curve = value["required_curve"]
    _require(tuple(curve["folds"]) == FOLDS, "exact-audit fold surface changed")
    _require(
        tuple(float(item) for item in curve["prefix_fractions"]) == PREFIX_FRACTIONS,
        "exact-audit prefix surface changed",
    )
    _require(int(curve["required_exact_prefix_cells"]) == 15, "required cell count changed")
    _require(int(curve["currently_exact_cells"]) == 0, "exact cell availability changed")
    _require(int(curve["embargo_days"]) == 7, "embargo changed")
    _require(
        int(curve["fixed_complete_pipeline_seed_count"]) == 3,
        "complete-pipeline seed budget changed",
    )
    observed_gaps = tuple(str(item["id"]) for item in value["blocking_recipe_gaps"])
    _require(observed_gaps == BLOCKING_GAPS, "blocking exact-recipe gaps changed")
    _require(not value["gen2_fit_started"], "unexpected exact refit has started")


def _validate_exact_seal(value: dict[str, Any], audit_sha256: str) -> None:
    _require(
        value.get("audit_sha256") == audit_sha256,
        "exact audit seal does not pin the audit",
    )
    _require(
        value.get("verdict")
        == "NOT_CURRENTLY_REPRODUCIBLE_AS_EXACT_SAME_PREFIX_FOLD_TRAIN_ONLY_INCUMBENT",
        "exact audit seal verdict changed",
    )
    _require(not value["gen2_fit_started"], "exact seal reports a started fit")
    _require(int(value["official_upload_count"]) == 0, "exact seal reports an upload")
    _require(all(int(item) == 0 for item in value["mutations"].values()), "exact audit mutated state")


def _validate_matched_common(value: dict[str, Any]) -> None:
    _require(
        value.get("status") == "SEALED_BEFORE_MATCHED_COMPARISON_RESULTS",
        "matched common protocol is not sealed",
    )
    quarantine = value["selection_quarantine"]
    _require(
        not quarantine["official_public_scores_may_be_used_for_candidate_selection"],
        "matched common protocol permits Public selection",
    )
    _require(
        value["problem_contracts"]["P2"]["mandatory_caveat"]
        == "surrogate comparators must not be described as the exact official incumbent",
        "matched P2 surrogate caveat changed",
    )
    _require(
        not value["interpretation_rules"]["promotion_authorized"],
        "matched protocol authorizes promotion",
    )


def _validate_matched_evidence(
    config: dict[str, Any],
    result: dict[str, Any],
    qa: dict[str, Any],
    manifest: dict[str, Any],
    preregistration_seal: dict[str, Any],
    technical_resume_seal: dict[str, Any],
) -> None:
    surface = config["surfaces"]["forward_causal_surrogate"]
    _require(tuple(surface["folds"]) == FOLDS, "matched folds changed")
    _require(
        tuple(float(item) for item in surface["prefix_fractions"]) == PREFIX_FRACTIONS,
        "matched cutoffs changed",
    )
    expected_seed_columns = tuple(f"seed_{seed}" for seed in SEEDS)
    _require(tuple(surface["seed_columns"]) == expected_seed_columns, "matched seeds changed")
    _require(not surface["exact_official_incumbent_architecture"], "surrogate mislabeled exact")
    _require(surface["causal_forward_unbiased"], "surrogate causal flag changed")

    panel = result["panels"]["forward_causal_surrogate"]
    _require(int(panel["rows"]) == 390780, "matched repeated population changed")
    _require(tuple(panel["folds"]) == tuple(sorted(FOLDS)), "result folds changed")
    _require(tuple(panel["seed_budget"]) == expected_seed_columns, "result seeds changed")
    _require(
        tuple(float(item) for item in panel["prefix_fractions"]) == PREFIX_FRACTIONS,
        "result prefix fractions changed",
    )
    prefix_diagnostics = panel["materialization_diagnostics"]["by_prefix_fraction"]
    _require(set(prefix_diagnostics) == {"040", "055", "070", "085", "100"}, "prefix keys changed")
    _require(
        all(int(item["rows"]) == 78156 for item in prefix_diagnostics.values()),
        "per-prefix population differs",
    )
    _require(
        all(int(item["base_seed_count"]) == 3 for item in prefix_diagnostics.values()),
        "per-prefix seed budget differs",
    )
    policy = result["selection_policy"]
    _require(policy["candidate_grid_sealed_before_score"], "matched grid was not sealed")
    _require(not policy["official_public_score_used_for_selection"], "Public selected recipe")
    _require(not policy["official_public_score_used_for_tuning"], "Public tuned recipe")
    _require(int(policy["new_model_fits"]) == 0, "unexpected matched-budget fit")

    _require(
        qa.get("status") == "PASS_WITH_MANDATORY_SURROGATE_CAVEAT",
        "matched QA status changed",
    )
    for key in (
        "official_test_reads",
        "sample_submission_reads",
        "submission_candidate_reads",
        "submission_files_generated",
        "uploads",
        "p3_era5_process_mutations",
        "new_model_fits",
        "candidate_or_grid_changes_after_first_score",
    ):
        _require(int(qa[key]) == 0, f"matched QA external action is nonzero: {key}")
    _require(int(qa["surrogate_seed_count"]) == 3, "QA seed count changed")
    _require(int(qa["surrogate_training_cutoff_count"]) == 5, "QA cutoff count changed")
    _require(int(qa["surrogate_outer_fold_count"]) == 3, "QA fold count changed")

    _require(
        manifest["config"]["sha256"]
        == preregistration_seal["config"]["sha256"],
        "matched manifest and preregistration config differ",
    )
    _require(
        technical_resume_seal["prior_seal_sha256"]
        == manifest["outputs"]["preregistration_seal.json"]["sha256"],
        "technical-resume chain does not pin the original seal",
    )
    _require(
        int(technical_resume_seal["candidate_or_grid_changes_after_first_score"]) == 0,
        "technical resume changed the grid",
    )


def _validate_recipe_contract(recipe: dict[str, Any], matched: dict[str, Any]) -> None:
    _require(
        recipe.get("status") == "SEALED_DRY_RUN_BEFORE_ANY_NEW_SCORE_OR_FIT",
        "recipe is not sealed before scores/fits",
    )
    decision = recipe["chosen_decision"]
    _require(
        decision["verdict"] == "NEW_AUTHORITATIVE_SURROGATE_REQUIRED",
        "recipe chooses the wrong P2 verdict",
    )
    _require(not decision["central_contract_revision_required"], "central revision selected")
    _require(not recipe["exact_official_incumbent_claimed"], "surrogate claims exactness")
    _require(tuple(recipe["exactness_dimensions"]) == EXACTNESS_DIMENSIONS, "recipe dimensions changed")
    historical = recipe["historical_exactness_verdict"]
    _require(tuple(historical["blocking_recipe_gap_ids"]) == BLOCKING_GAPS, "recipe blockers changed")
    _require(int(historical["required_exact_prefix_cells"]) == 15, "recipe cell count changed")
    _require(int(historical["currently_exact_cells"]) == 0, "recipe exact cells changed")

    nested = recipe["authoritative_nested_surrogate_recipe"]
    _require("never the exact official incumbent" in nested["claim"], "mandatory caveat absent")
    _require(
        tuple(item["id"] for item in nested["outer_fold_contract"]["folds"]) == FOLDS,
        "nested fold contract changed",
    )
    _require(int(nested["outer_fold_contract"]["embargo_days"]) == 7, "nested embargo changed")
    _require(
        tuple(float(item) for item in nested["chronological_prefix_contract"]["fractions"])
        == PREFIX_FRACTIONS,
        "nested prefix contract changed",
    )
    _require(
        tuple(int(item) for item in nested["complete_pipeline_seed_contract"]["seeds"])
        == SEEDS,
        "nested complete seeds changed",
    )
    _require(
        int(nested["nested_component_oof_contract"]["inner_fold_count"]) == 3,
        "nested inner-fold count changed",
    )
    _require(
        not nested["nested_component_oof_contract"][
            "frozen_official_stack_or_gate_reuse_allowed"
        ],
        "frozen exposed meta-parameters are reusable",
    )

    comparison = recipe["preregistered_comparison"]
    _require(int(comparison["required_outer_prefix_cells"]) == 15, "comparison cells changed")
    _require(int(comparison["required_seeded_pipeline_cells"]) == 45, "seeded cells changed")
    _require(comparison["same_population_folds_cutoffs_seeds_postprocess_metric_required"], "common surface not required")
    _require(comparison["official_public_score_use"] == "post-hoc transport audit only", "Public role changed")
    _require(not comparison["promotion_from_existing_or_future_surrogate_allowed"], "surrogate promotion enabled")
    for family, matched_family in (
        ("incumbent_surrogate", "incumbent"),
        ("conservative_stack", "conservative_stack"),
        ("round_b", "round_b"),
    ):
        expected = _ids(matched["families"][matched_family]["settings"])
        _require(tuple(comparison["families"][family]["settings"]) == expected, f"{family} grid changed")
        _require(
            comparison["families"][family]["default"]
            == matched["families"][matched_family]["default_setting"],
            f"{family} default changed",
        )
    external = recipe["resource_and_external_action_contract"]
    for key, value in external.items():
        if key != "cpu_threads_if_future_training_is_separately_authorized":
            _require(int(value) == 0, f"recipe external action is nonzero: {key}")


def _execution_plan(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    nested = recipe["authoritative_nested_surrogate_recipe"]
    for fold in nested["outer_fold_contract"]["folds"]:
        for fraction in nested["chronological_prefix_contract"]["fractions"]:
            for seed in nested["complete_pipeline_seed_contract"]["seeds"]:
                plan.append(
                    {
                        "cell_id": f"{fold['id']}__p{int(float(fraction) * 100):03d}__s{seed}",
                        "outer_fold": fold["id"],
                        "prefix_fraction": float(fraction),
                        "complete_pipeline_seed": int(seed),
                        "score_access_authorized": False,
                        "fit_authorized": False,
                    }
                )
    return plan


def validate_contract(
    recipe: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
) -> ContractValidation:
    """Validate the sealed decision and build aggregate-only dry-run outputs."""

    _validate_recon_preregistration(evidence["decisive_recon_preregistration"])
    _validate_exact_audit(evidence["exact_audit"])
    _validate_exact_seal(
        evidence["exact_audit_seal"],
        recipe["evidence"]["exact_audit"]["sha256"],
    )
    _validate_matched_common(evidence["matched_common_protocol"])
    _validate_matched_evidence(
        evidence["matched_config"],
        evidence["matched_result"],
        evidence["matched_qa"],
        evidence["matched_manifest"],
        evidence["matched_preregistration_seal"],
        evidence["matched_technical_resume_seal"],
    )
    _validate_recipe_contract(recipe, evidence["matched_config"])

    exact_audit = evidence["exact_audit"]
    matched_result = evidence["matched_result"]
    exact_panel = matched_result["panels"]["exact_frozen_lineage"]
    surrogate_panel = matched_result["panels"]["forward_causal_surrogate"]
    exact_fallback = float(
        exact_panel["metrics_by_setting"]["FALLBACK_BLEND50_A0625"][
            "fold_equal_layer_equal_rmse_c"
        ]
    )
    exact_incumbent = float(
        exact_panel["metrics_by_setting"]["INCUMBENT_NOOP"][
            "fold_equal_layer_equal_rmse_c"
        ]
    )
    surrogate_fallback = float(
        surrogate_panel["metrics_by_setting"]["FALLBACK_BLEND50_A0625"][
            "fold_equal_layer_equal_rmse_c"
        ]
    )
    surrogate_incumbent = float(
        surrogate_panel["metrics_by_setting"]["INCUMBENT_NOOP"][
            "fold_equal_layer_equal_rmse_c"
        ]
    )
    full_prefix = surrogate_panel["metrics_by_prefix_fraction"]["100"]
    full_prefix_gain = float(
        full_prefix["INCUMBENT_NOOP"]["fold_equal_layer_equal_rmse_c"]
        - full_prefix["FALLBACK_BLEND50_A0625"]["fold_equal_layer_equal_rmse_c"]
    )

    state_rows: list[dict[str, Any]] = []
    for dimension in EXACTNESS_DIMENSIONS:
        item = recipe["exactness_dimensions"][dimension]
        semantic_only = dimension in SEMANTIC_ONLY_DIMENSIONS
        state_rows.append(
            {
                "dimension": dimension,
                "historical_exact_state": item["historical_exact_state"],
                "historical_exact_owned": False,
                "surrogate_contract_owner": item["surrogate_owner"],
                "surrogate_semantics_pinned": True,
                "current_implementation_conformance": (
                    "PENDING_SEPARATE_IMPLEMENTATION_AND_AUTHORIZATION"
                    if semantic_only
                    else "AGGREGATE_DRY_RUN_EVIDENCE_AVAILABLE"
                ),
                "execution_requirement": item["execution_requirement"],
            }
        )
    execution_plan = _execution_plan(recipe)
    _require(len(execution_plan) == 45, "execution plan is not 45 seeded cells")

    decision = {
        "schema_version": "p2_authoritative_nested_surrogate_decision.v1",
        "verdict": "NEW_AUTHORITATIVE_SURROGATE_REQUIRED",
        "chosen_path": "A_NEW_AUTHORITATIVE_NESTED_SURROGATE_RECIPE",
        "central_contract_revision_required": False,
        "exact_comparison_reproducible": False,
        "historical_exact_cells": int(exact_audit["required_curve"]["currently_exact_cells"]),
        "historical_blocking_gap_count": len(exact_audit["blocking_recipe_gaps"]),
        "surrogate_contract_dimensions_owned": len(EXACTNESS_DIMENSIONS),
        "implementation_conformance_dimensions_pending": len(SEMANTIC_ONLY_DIMENSIONS),
        "dry_run_contract_status": "PASS",
        "new_training_status": "BLOCKED_PENDING_SEPARATE_IMPLEMENTATION_AND_AUTHORIZATION",
        "claim_boundary": "The recipe is authoritative for a future common train-only surrogate comparison, never an exact reconstruction of the official incumbent.",
        "matched_evidence": {
            "exact_frozen_fallback_gain_c": exact_incumbent - exact_fallback,
            "surrogate_all_cutoff_fallback_gain_c": surrogate_incumbent
            - surrogate_fallback,
            "surrogate_full_prefix_fallback_gain_c": full_prefix_gain,
            "direction_conflict_confirmed": (
                exact_incumbent - exact_fallback < 0.0
                and surrogate_incumbent - surrogate_fallback > 0.0
                and full_prefix_gain < 0.0
            ),
            "causal_correction_supported_rows": int(
                exact_panel["materialization_diagnostics"]["causal_correction"][
                    "supported_rows"
                ]
            ),
        },
        "why_not_central_revision": recipe["chosen_decision"]["why_not_central_revision"],
    }
    state_matrix = {
        "schema_version": "p2_authoritative_nested_surrogate_state_matrix.v1",
        "exactness_dimensions": state_rows,
        "blocking_recipe_gaps": exact_audit["blocking_recipe_gaps"],
        "historical_exact_fold_state": exact_audit["missing_recipe_by_fold"],
        "reproducible_without_refit": exact_audit["what_is_reproducible"],
    }
    comparison = {
        "schema_version": "p2_authoritative_nested_surrogate_comparison_preregistration.v1",
        "status": recipe["preregistered_comparison"]["status"],
        "recipe_identity": recipe["authoritative_nested_surrogate_recipe"]["identity"],
        "claim": recipe["authoritative_nested_surrogate_recipe"]["claim"],
        "population": recipe["authoritative_nested_surrogate_recipe"][
            "evaluation_population"
        ],
        "folds": recipe["authoritative_nested_surrogate_recipe"]["outer_fold_contract"],
        "cutoffs": recipe["authoritative_nested_surrogate_recipe"][
            "chronological_prefix_contract"
        ],
        "seeds": recipe["authoritative_nested_surrogate_recipe"][
            "complete_pipeline_seed_contract"
        ],
        "families": recipe["preregistered_comparison"]["families"],
        "metrics": {
            "primary": recipe["preregistered_comparison"]["primary_metric"],
            "secondary": recipe["preregistered_comparison"]["secondary_metrics"],
            "uncertainty": recipe["preregistered_comparison"]["uncertainty"],
        },
        "seeded_execution_plan": execution_plan,
        "fit_authorized": False,
        "score_access_authorized": False,
    }
    qa = {
        "schema_version": "p2_authoritative_nested_surrogate_contract_qa.v1",
        "status": "PASS_DESIGN_ONLY_FIT_BLOCKED",
        "decisive_recon_preregistration_pass": True,
        "exact_audit_verdict_pass": True,
        "exactness_dimension_count": len(EXACTNESS_DIMENSIONS),
        "blocking_gap_count": len(BLOCKING_GAPS),
        "matched_population_rows_per_cutoff": 78156,
        "matched_repeated_population_rows": 390780,
        "outer_fold_count": len(FOLDS),
        "prefix_fraction_count": len(PREFIX_FRACTIONS),
        "complete_pipeline_seed_count": len(SEEDS),
        "outer_prefix_cell_count": len(FOLDS) * len(PREFIX_FRACTIONS),
        "seeded_pipeline_cell_count": len(execution_plan),
        "surrogate_exact_claims": 0,
        "central_contract_mutations": 0,
        "new_model_fits": 0,
        "new_score_reads": 0,
        "official_public_score_used_for_decision": False,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
        "submission_files_generated": 0,
        "uploads": 0,
        "p3_era5_process_mutations": 0,
    }
    return ContractValidation(decision, state_matrix, comparison, qa)
