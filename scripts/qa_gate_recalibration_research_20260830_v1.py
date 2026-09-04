from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "reports/gate_recalibration_research_20260830_v1/independent-qa.json"
)

PATHS = {
    "policy": ROOT
    / "configs/goals/metric_aligned_gate_recalibration_20260830_v1.json",
    "gate_replay": ROOT
    / "reports/gate_recalibration_research_20260830_v1/gate-replay.json",
    "gate_replay_runner": ROOT / "scripts/audit_metric_aligned_gates_20260830_v1.py",
    "p1_result": ROOT
    / "reports/p1_addonly_hierarchical_event_precision_lcb_20260830_v1/result.json",
    "p1_attempt_lock": ROOT
    / "reports/p1_addonly_hierarchical_event_precision_lcb_20260830_v1/attempt_lock.json",
    "p1_config": ROOT
    / "configs/experiments/p1_addonly_hierarchical_event_precision_lcb_20260830_v1.json",
    "p1_core": ROOT
    / "src/p1_qc/p1_addonly_hierarchical_event_precision_lcb_20260830_v1.py",
    "p1_runner": ROOT
    / "scripts/run_p1_addonly_hierarchical_event_precision_lcb_20260830_v1.py",
    "p2_v1_failure": ROOT
    / "reports/p2_availability_aware_continuous_sparse_copula_20260830_v1/failure-receipt.json",
    "p2_v1_guard": ROOT
    / "reports/p2_availability_aware_continuous_sparse_copula_20260830_v1/guard-diagnostic.json",
    "p2_result": ROOT
    / "reports/p2_availability_aware_continuous_sparse_copula_20260830_v2/result.json",
    "p2_qa": ROOT
    / "reports/p2_availability_aware_continuous_sparse_copula_20260830_v2/independent-qa.json",
    "p2_config": ROOT
    / "configs/experiments/p2_availability_aware_continuous_sparse_copula_20260830_v2.json",
    "p2_core": ROOT
    / "src/p2_restore/p2_availability_aware_continuous_sparse_copula_20260830_v2.py",
    "p2_runner": ROOT
    / "scripts/run_p2_availability_aware_continuous_sparse_copula_20260830_v2.py",
    "p3_result": ROOT
    / "reports/p3_selection_matched_sparse_gp_abstention_20260830_v1/result.json",
    "p3_attempt_lock": ROOT
    / "reports/p3_selection_matched_sparse_gp_abstention_20260830_v1/ATTEMPT_LOCK.json",
    "p3_qa": ROOT
    / "reports/p3_selection_matched_sparse_gp_abstention_20260830_v1/independent-qa.json",
    "p3_config": ROOT
    / "configs/experiments/p3_selection_matched_sparse_gp_abstention_20260830_v1.json",
    "p3_core": ROOT
    / "src/p3_wave/p3_selection_matched_sparse_gp_abstention_20260830_v1.py",
    "p3_runner": ROOT
    / "scripts/run_p3_selection_matched_sparse_gp_abstention_20260830_v1.py",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _f1(counts: dict[str, Any]) -> float:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    return (2.0 * tp) / (2.0 * tp + fp + fn)


def _hash_matches(path: Path, expected: str) -> bool:
    return _sha256(path) == expected


def _seal(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_qa() -> dict[str, Any]:
    policy = _read_json(PATHS["policy"])
    replay = _read_json(PATHS["gate_replay"])
    p1 = _read_json(PATHS["p1_result"])
    p1_attempt_lock = _read_json(PATHS["p1_attempt_lock"])
    p2_v1_failure = _read_json(PATHS["p2_v1_failure"])
    p2_v1_guard = _read_json(PATHS["p2_v1_guard"])
    p2 = _read_json(PATHS["p2_result"])
    p2_qa = _read_json(PATHS["p2_qa"])
    p3 = _read_json(PATHS["p3_result"])
    p3_qa = _read_json(PATHS["p3_qa"])

    policy_sha = _sha256(PATHS["policy"])

    replay_candidates = replay["candidates"]
    replay_provenance = replay["provenance"]
    replay_inputs = {
        name: _read_json(ROOT / item["path"])
        for name, item in replay_provenance["inputs"].items()
    }
    replay_p1_benefit = float(
        replay_inputs["p1_supcon"]["pooled"]["candidate"]["f1"]
    ) - float(replay_inputs["p1_supcon"]["pooled"]["control"]["f1"])
    replay_p2g_delta = float(
        replay_inputs["p2_gaussian_copula"]["metrics"]["aggregate"]["delta_rmse"]
    )
    replay_p2g_delta_ci = replay_inputs["p2_gaussian_copula"]["bootstrap"]
    replay_p2g_benefit_ci = [
        -float(replay_p2g_delta_ci["ci90_high"]),
        -float(replay_p2g_delta_ci["ci90_low"]),
    ]
    replay_p2s_delta = float(
        replay_inputs["p2_state_copula"]["metrics"]["pooled"]["delta_rmse"]
    )
    replay_p2s_delta_ci = replay_inputs["p2_state_copula"]["bootstrap"]
    replay_p2s_benefit_ci = [
        -float(replay_p2s_delta_ci["ci90_high"]),
        -float(replay_p2s_delta_ci["ci90_low"]),
    ]
    replay_p3c = replay_inputs["p3_catboost_confirmation"]["confirmation"]
    replay_p3c_delta = float(replay_p3c["metrics"]["delta_rmse_m"])
    replay_p3c_delta_ci = replay_p3c["paired_case_bootstrap"]
    replay_p3c_benefit_ci = [
        -float(replay_p3c_delta_ci["ci90_upper_m"]),
        -float(replay_p3c_delta_ci["ci90_lower_m"]),
    ]
    replay_p3m = replay_inputs["p3_masked_ssl"]["paired_comparison"]
    replay_p3m_delta = float(
        replay_p3m["metrics"]["overall"]["delta_candidate_minus_incumbent_m"]
    )
    replay_p3m_delta_ci = replay_p3m["paired_case_bootstrap"][
        "delta_candidate_minus_incumbent_ci90_m"
    ]
    replay_p3m_benefit_ci = [
        -float(replay_p3m_delta_ci[1]),
        -float(replay_p3m_delta_ci[0]),
    ]
    replay_checks = {
        "zero_fit_aggregate_only": replay["summary"]["model_fits"] == 0
        and replay["summary"]["prediction_rows_read"] == 0
        and replay["summary"]["raw_training_rows_read"] == 0,
        "two_p2_candidates_reclassified": replay["summary"][
            "reclassified_legacy_no_go_to_high_value_challenger"
        ]
        == 2
        and replay_candidates["P2_gaussian_copula_conditional_mean"]["new_state"]
        == "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
        and replay_candidates["P2_state_conditioned_copula"]["new_state"]
        == "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY",
        "three_primary_harm_states_unchanged": replay["summary"][
            "primary_harm_conclusions_unchanged"
        ]
        == 3,
        "policy_hash_matches": replay_provenance["policy"]["sha256"]
        == policy_sha,
        "runner_hash_matches": replay_provenance["runner"]["sha256"]
        == _sha256(PATHS["gate_replay_runner"]),
        "all_input_hashes_match": all(
            _hash_matches(ROOT / item["path"], item["sha256"])
            for item in replay_provenance["inputs"].values()
        ),
        "all_five_primary_values_recomputed": _close(
            replay_candidates["P1_event_balanced_supcon"]["benefit_f1"],
            replay_p1_benefit,
        )
        and _close(
            replay_candidates["P2_gaussian_copula_conditional_mean"]["benefit_c"],
            -replay_p2g_delta,
        )
        and all(
            _close(left, right)
            for left, right in zip(
                replay_candidates["P2_gaussian_copula_conditional_mean"][
                    "benefit_ci90_c"
                ],
                replay_p2g_benefit_ci,
                strict=True,
            )
        )
        and _close(
            replay_candidates["P2_state_conditioned_copula"]["benefit_c"],
            -replay_p2s_delta,
        )
        and all(
            _close(left, right)
            for left, right in zip(
                replay_candidates["P2_state_conditioned_copula"]["benefit_ci90_c"],
                replay_p2s_benefit_ci,
                strict=True,
            )
        )
        and _close(
            replay_candidates["P3_catboost_confirmation"]["benefit_m"],
            -replay_p3c_delta,
        )
        and all(
            _close(left, right)
            for left, right in zip(
                replay_candidates["P3_catboost_confirmation"]["benefit_ci90_m"],
                replay_p3c_benefit_ci,
                strict=True,
            )
        )
        and _close(
            replay_candidates["P3_selection_matched_masked_ssl"]["benefit_m"],
            -replay_p3m_delta,
        )
        and all(
            _close(left, right)
            for left, right in zip(
                replay_candidates["P3_selection_matched_masked_ssl"][
                    "benefit_ci90_m"
                ],
                replay_p3m_benefit_ci,
                strict=True,
            )
        ),
        "interval_states_recomputed_without_legacy_vetoes": (
            -replay_p2g_delta > 0.0
            and replay_p2g_benefit_ci[0] > 0.0
            and -replay_p2s_delta > 0.0
            and replay_p2s_benefit_ci[0] > 0.0
            and -replay_p3c_delta < 0.0
            and replay_p3c_benefit_ci[1] < 0.0
            and -replay_p3m_delta < 0.0
            and replay_p3m_benefit_ci[1] < 0.0
        ),
        "official_csv_upload_zero": replay["summary"][
            "official_test_sample_submission_hidden_rows_read"
        ]
        == 0
        and replay["summary"]["csv_created"] == 0
        and replay["summary"]["uploads"] == 0,
    }

    p1_primary = p1["pooled_primary"]
    p1_interval = p1["paired_uncertainty"]
    p1_exec = p1["execution_audit"]
    p1_anchor_f1 = _f1(p1_primary["anchor_counts"])
    p1_candidate_f1 = _f1(p1_primary["candidate_counts"])
    p1_delta = p1_candidate_f1 - p1_anchor_f1
    p1_checks = {
        "pooled_f1_recomputed": _close(p1_anchor_f1, p1_primary["anchor_f1"])
        and _close(p1_candidate_f1, p1_primary["candidate_f1"])
        and _close(p1_delta, p1_primary["candidate_minus_anchor_f1"]),
        "interval_crosses_zero": p1_interval["lower_one_sided_95"] < 0.0
        < p1_interval["upper_one_sided_95"],
        "state_recomputed": p1["status"] == "INCONCLUSIVE_RESEARCH_ONLY",
        "all_level_0_gates_pass": all(p1["level_0_hard_validity"].values()),
        "pure_add_only_algebra_pass": p1["pooled_f1_over_2_hard_sanity"]["pass"]
        and p1["pooled_f1_over_2_hard_sanity"]["anchor_positive_removed_rows"]
        == 0,
        "implementation_hashes_match": _hash_matches(
            PATHS["p1_config"], p1["code_and_registration_sha256"]["config"]
        )
        and _hash_matches(
            PATHS["p1_core"], p1["code_and_registration_sha256"]["core_module"]
        )
        and _hash_matches(
            PATHS["p1_runner"], p1["code_and_registration_sha256"]["runner"]
        ),
        "exclusive_attempt_lock_matches": p1_attempt_lock["exclusive_create"]
        and p1_attempt_lock["attempt"] == 1
        and p1_attempt_lock["authorized_attempts"] == 1
        and p1_attempt_lock["retry_authorized"] is False
        and p1_attempt_lock["config_sha256"]
        == p1["code_and_registration_sha256"]["config"],
        "one_shot_no_search_retry": p1_exec["attempts_executed"] == 1
        and p1_exec["threshold_search_count"] == 0
        and p1_exec["hyperparameter_or_feature_search_count"] == 0
        and p1_exec["retry_or_tuning_count"] == 0,
        "official_csv_upload_deletion_zero": p1_exec["official_interface_rows_read"]
        == 0
        and p1_exec["prediction_csv_count"] == 0
        and p1_exec["upload_count"] == 0
        and p1_exec["outlier_hard_deleted_rows"] == 0
        and p1_exec["anchor_positive_removed_rows"] == 0,
    }

    p2_guard = p2_v1_guard["guard_operands"]
    p2_predicates = p2_guard["guard_predicates"]
    p2_pooled = p2["metrics"]["pooled"]
    p2_bootstrap = p2["dependence_aware_bootstrap"]
    p2_access = p2["access_receipt"]
    p2_checks = {
        "v1_is_sealed_technical_failure_without_metrics": p2_v1_failure["status"]
        == "INVALID_TECHNICAL_NO_PERFORMANCE_RESULT"
        and not p2_v1_failure["artifact_receipt"]["performance_metrics_computed"]
        and not p2_v1_failure["execution"]["technical_failure_retry"],
        "v1_failure_hash_bound_to_guard": p2_v1_guard["sealed_failure_receipt"][
            "sha256"
        ]
        == _sha256(PATHS["p2_v1_failure"]),
        "guard_failure_was_preexisting_exact_noop": p2_predicates[
            "reference_outside_count"
        ]
        == 18
        and p2_predicates["candidate_outside_count"] == 18
        and p2_predicates["inactive_candidate_outside_count"] == 18
        and p2_predicates["new_candidate_outside_count"] == 0
        and p2_predicates["active_candidate_outside_count"] == 0
        and p2_guard["minimal_relative_domain_repair_diagnostic"][
            "changed_rows_from_raw_candidate"
        ]
        == 0,
        "v2_is_guard_only_overlay": p2["technical_overlay"]["model_changed"]
        is False
        and p2["technical_overlay"]["prediction_values_changed_by_overlay"]
        is False
        and p2["technical_overlay"]["changed_component"]
        == "post_prediction_physical_domain_guard_only",
        "pooled_rmse_recomputed": _close(
            p2_pooled["candidate_rmse"] - p2_pooled["reference_rmse"],
            p2_pooled["delta_rmse"],
        ),
        "wholly_unfavorable_interval": p2_pooled["delta_rmse"] > 0.0
        and p2_bootstrap["ci90_low"] > 0.0
        and p2_bootstrap["ci90_high"] > 0.0,
        "state_recomputed": p2["decision"] == "PRIMARY_HARM_RESEARCH_ONLY",
        "individual_qa_28_of_28": p2_qa["qa_status"] == "PASS"
        and p2_qa["check_count"] == 28
        and all(p2_qa["checks"].values()),
        "result_hash_matches_individual_qa": _sha256(PATHS["p2_result"])
        == p2_qa["result"]["sha256"],
        "implementation_hashes_match": _hash_matches(
            PATHS["p2_config"], p2["config_sha256"]
        )
        and _hash_matches(PATHS["p2_core"], p2["implementation_hashes"]["core"])
        and _hash_matches(
            PATHS["p2_runner"], p2["implementation_hashes"]["runner"]
        ),
        "official_query_csv_upload_deletion_zero": p2_access[
            "official_interface_rows_read"
        ]
        == 0
        and p2_access["query_support_rows_read"] == 0
        and p2_access["csv_output_count"] == 0
        and p2_access["upload_count"] == 0
        and p2_access["hard_deleted_training_profiles"] == 0,
    }

    p3_primary = p3["primary_evaluation"]["metrics"]["overall"]
    p3_interval = p3["primary_evaluation"]["dependence_aware_interval"]
    p3_access = p3["data_access"]
    p3_source = p3["source_receipt"]
    p3_checks = {
        "pooled_rmse_recomputed": _close(
            p3_primary["paired_incumbent_rmse_m"]
            - p3_primary["candidate_rmse_m"],
            p3_primary["benefit_incumbent_minus_candidate_rmse_m"],
        ),
        "interval_crosses_zero": p3_interval["benefit_ci90_m"][0] < 0.0
        < p3_interval["benefit_ci90_m"][1],
        "state_recomputed": p3["evidence_state"]
        == "INCONCLUSIVE_RESEARCH_ONLY",
        "all_fatal_hard_gates_pass": p3["validity"][
            "all_fatal_hard_gates_pass"
        ],
        "individual_qa_14_of_14": p3_qa["all_checks_pass"]
        and len(p3_qa["checks"]) == 14
        and all(p3_qa["checks"].values()),
        "result_hash_matches_individual_qa": _sha256(PATHS["p3_result"])
        == p3_qa["result_sha256"],
        "attempt_lock_hash_matches_individual_qa": _sha256(PATHS["p3_attempt_lock"])
        == p3_qa["attempt_lock_sha256"],
        "implementation_hashes_match": _hash_matches(
            PATHS["p3_config"], p3["provenance"]["implementation_sha256"]["config"]
        )
        and _hash_matches(
            PATHS["p3_core"],
            p3["provenance"]["implementation_sha256"]["implementation"],
        )
        and _hash_matches(
            PATHS["p3_runner"], p3["provenance"]["implementation_sha256"]["runner"]
        ),
        "official_csv_upload_deletion_zero": all(
            p3_access[name] == 0
            for name in (
                "official_test_rows_read",
                "official_context_rows_read",
                "test_index_rows_read",
                "sample_rows_read",
                "baseline_rows_read",
                "score_rows_read",
                "submission_rows_read_or_written",
                "hidden_or_answer_rows_read",
                "csv_output_count",
                "upload_attempt_count",
                "source_rows_modified_or_deleted",
            )
        )
        and p3["sensor_error_flags"]["rows_deleted_or_masked"] == 0,
        "training_source_access_disclosed": p3_source["train_wave.csv"]["rows"]
        == 118_152
        and p3_source["train_atmos.csv"]["rows"] == 130_896,
    }

    policy_checks = {
        "future_only_nonretroactive": policy["non_retroactivity"][
            "historical_configs_and_results_are_immutable"
        ]
        and policy["non_retroactivity"]["historical_one_shot_attempts_are_not_reopened"],
        "unsupported_fixed_thresholds_removed": policy[
            "unsupported_fixed_threshold_policy"
        ]["cross_problem_raw_metric_thresholds"]
        == "forbidden",
        "outlier_hard_deletion_not_authorized": policy["official_and_data_boundary"][
            "outlier_hard_deletion_authorized"
        ]
        is False,
        "official_action_not_authorized": policy["official_and_data_boundary"][
            "official_upload_authorized"
        ]
        is False,
    }

    checks = {
        "policy": policy_checks,
        "legacy_gate_replay": replay_checks,
        "P1": p1_checks,
        "P2": p2_checks,
        "P3": p3_checks,
    }
    failed_checks = [
        f"{group}.{name}"
        for group, group_checks in checks.items()
        for name, passed in group_checks.items()
        if not passed
    ]

    payload: dict[str, Any] = {
        "schema_version": "gate_recalibration_research.independent_qa.20260830.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if not failed_checks else "FAIL",
        "conclusion": (
            "Unsupported fixed tolerances changed two historical P2 interpretations, "
            "but none of the three new one-shot candidates is a submission-ready champion."
        ),
        "checks": checks,
        "failed_checks": failed_checks,
        "recomputed_results": {
            "legacy_replay": {
                "high_value_p2_challengers": 2,
                "unchanged_primary_harms": 3,
            },
            "P1": {
                "rows": p1_primary["rows"],
                "anchor_f1": p1_anchor_f1,
                "candidate_f1": p1_candidate_f1,
                "benefit_candidate_minus_anchor_f1": p1_delta,
                "uncertainty_bounds": [
                    p1_interval["lower_one_sided_95"],
                    p1_interval["upper_one_sided_95"],
                ],
                "state": "INCONCLUSIVE_RESEARCH_ONLY",
                "model_fits": p1_exec["model_fit_count"],
            },
            "P2": {
                "rows": p2_pooled["rows"],
                "reference_rmse_c": p2_pooled["reference_rmse"],
                "candidate_rmse_c": p2_pooled["candidate_rmse"],
                "delta_candidate_minus_reference_rmse_c": p2_pooled["delta_rmse"],
                "delta_ci90_c": [
                    p2_bootstrap["ci90_low"],
                    p2_bootstrap["ci90_high"],
                ],
                "state": "PRIMARY_HARM_RESEARCH_ONLY",
                "outer_model_fits": p2["fit_counts"]["outer_dependence_model_fits"],
                "edge_estimations": p2["fit_counts"]["continuous_edge_estimations"],
                "preexisting_reference_extreme_exact_noop_rows": 18,
            },
            "P3": {
                "cases": p3_primary["cases"],
                "rows": p3_primary["rows"],
                "incumbent_rmse_m": p3_primary["paired_incumbent_rmse_m"],
                "candidate_rmse_m": p3_primary["candidate_rmse_m"],
                "benefit_incumbent_minus_candidate_rmse_m": p3_primary[
                    "benefit_incumbent_minus_candidate_rmse_m"
                ],
                "benefit_ci90_m": p3_interval["benefit_ci90_m"],
                "state": "INCONCLUSIVE_RESEARCH_ONLY",
                "candidate_fits": p3["fit_budget"]["actual_candidate_fits"],
            },
        },
        "data_and_action_boundary": {
            "qa_reads_aggregate_receipts_and_code_only": True,
            "qa_raw_training_rows_read": 0,
            "qa_official_rows_read": 0,
            "P1_runner_raw_training_rows_read": p1_exec["raw_training_rows_read"],
            "P2_observations_source": {
                "row_cardinality": p2["source"]["rows"],
                "v1_main_inferred_opens": p2_v1_failure["access_receipt"][
                    "source_open_counts_inferred_from_sealed_control_flow"
                ]["observations.csv"],
                "guard_diagnostic_opens": p2_v1_guard["source_open_counts"][
                    "observations.csv"
                ],
                "v2_main_opens": p2["source_open_counts"]["observations.csv"],
            },
            "P3_runner_training_rows_read": {
                "train_wave.csv": p3_source["train_wave.csv"]["rows"],
                "train_atmos.csv": p3_source["train_atmos.csv"]["rows"],
            },
            "official_test_sample_submission_hidden_rows_read": 0,
            "prediction_csv_created": 0,
            "uploads": 0,
            "hard_deleted_or_masked_source_rows": 0,
        },
        "outlier_finding": {
            "P2_preexisting_reference_extreme_rows": 18,
            "P2_new_or_active_extreme_rows": 0,
            "P2_rows_changed_by_guard_repair": 0,
            "P3_jump_return_flags": p3["sensor_error_flags"]["jump_return_hs"][
                "total_flag_count"
            ],
            "interpretation": (
                "Flags are diagnostics, not proof of sensor error. No hard deletion was "
                "performed; the P2 guard distinguishes inherited exact no-ops from new "
                "candidate violations."
            ),
        },
        "artifact_sha256": {name: _sha256(path) for name, path in PATHS.items()},
        "caveats": [
            "All evaluated surfaces are historically exposed, so every state is research-only.",
            "The gate replay is provenance-hashed but is not an exclusive immutable receipt.",
            "P1 has an exclusive attempt lock and code hashes, but its result has no separate external immutable seal.",
            "P2 v2 has an execution receipt and independent QA but no pre-fit attempt-lock or sealed result payload; do not overstate cryptographic one-shot proof.",
            "No official hidden-mixture score transform is available, so no raw-unit practical margin was invented.",
        ],
    }
    payload["seal"] = {
        "algorithm": "sha256",
        "payload_without_seal_sha256": _seal(payload),
    }
    return payload


def main() -> None:
    payload = build_qa()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "failed_checks": payload["failed_checks"],
                "output": str(OUTPUT.relative_to(ROOT)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
