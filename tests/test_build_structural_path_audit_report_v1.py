from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_structural_path_audit_report_v1.py"
)
SPEC = importlib.util.spec_from_file_location("build_structural_path_audit_report_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _zero_counters(*names: str) -> dict[str, int]:
    return {name: 0 for name in names}


def _evidence() -> dict[str, object]:
    p1_diag = {
        "structural_feasibility": {"original_rolling_split_identifiable": False},
        "expanding_fit_scopes": {"fit_1_through_2024_may23": {"composite_super_events": 0}},
        "operation_counts": _zero_counters(
            "inner_score",
            "model_fit",
            "outer_score",
            "prediction",
            "test_read",
            "submission",
            "upload",
        ),
    }
    p1_v1 = {
        "decision": "NO_GO_PRECHECK",
        "inner_blocks_executed": 0,
        **_zero_counters(
            "outer_validation_or_scoring_count",
            "test_prediction_count",
            "submission_generation_count",
            "upload_count",
        ),
    }
    p1_v2 = {
        "decision": "NO_GO_EXACT_CONFIGURATION",
        "inner_blocks_executed": 2,
        **_zero_counters(
            "outer_score_count", "test_prediction_count", "submission_count", "upload_count"
        ),
        "aggregate": {
            "gate_passed": False,
            "micro_f1_delta": 0.0024336282407236842,
            "recall_delta_by_type": {
                "spike": -0.47058823529411764,
                "offset": 0.018011527377521624,
            },
            "worst_station_layer_f1_delta": -0.6666666666666666,
        },
    }
    p2_diag = {
        "aggregate_only": True,
        "decision": {"v1_failure_is_denominator_artifact": True},
        "forbidden_operations": _zero_counters(
            "truth_reads",
            "prediction_reads",
            "model_reads_or_fits",
            "alpha_computations",
            "gate5_runs",
            "outer_scores",
            "test_index_reads",
            "submission_rows",
            "upload_attempts",
        ),
        "blocks": {
            "2024_sep_oct": {
                "full_grid_support_share": 0.9882741347905283,
                "full_grid_times": 8784,
                "full_grid_supported_times": 8681,
                "key_aligned_support_share": 0.9888369973801117,
                "unique_scored_times": 8779,
                "key_aligned_supported_times": 8681,
            },
            "2025_jul_aug_61d": {
                "full_grid_support_share": 0.9997723132969034,
                "full_grid_times": 8784,
                "full_grid_supported_times": 8782,
                "key_aligned_support_share": 0.9997723132969034,
                "unique_scored_times": 8784,
                "key_aligned_supported_times": 8782,
            },
            "2025_nov_dec": {
                "full_grid_support_share": 0.6257969034608379,
                "full_grid_times": 8784,
                "full_grid_supported_times": 5497,
                "key_aligned_support_share": 0.9762031610726336,
                "unique_scored_times": 5631,
                "key_aligned_supported_times": 5497,
            },
        },
    }

    def p2_fold(alpha: float, delta: float) -> dict[str, object]:
        return {"alpha": alpha, "pooled_delta_rmse": delta, "pass": False}

    p2_v2 = {
        "aggregate_only": True,
        "gates_1_to_4": {"pass": True},
        "gate_5_inner_only": {
            "executed": True,
            "all_outer_inner_gates_pass": False,
            "folds": {
                "2024_sep_oct": p2_fold(0.06382863999889778, -0.003042942187990394),
                "2025_jul_aug_61d": p2_fold(0.01741333531983693, -0.00023858679414312522),
                "2025_nov_dec": p2_fold(0.0, 0.0),
            },
        },
        "forbidden_operations": _zero_counters(
            "hidden_target_value_reads",
            "outer_truth_scores",
            "outer_prediction_rows",
            "test_index_reads",
            "submission_rows",
            "upload_attempts",
        ),
    }

    def b_gate(passing_folds: int, ratios: tuple[float, float, float]) -> dict[str, object]:
        names = ("2024_h2_storm", "2025_h1", "winter_transition")
        cases = (9, 33, 15)
        return {
            "passing_folds": passing_folds,
            "by_fold": {
                name: {
                    "ratio": ratio,
                    "eligible_cases": case_count,
                    "pass": ratio <= 0.9,
                }
                for name, ratio, case_count in zip(names, ratios, cases, strict=True)
            },
        }

    p3_v1 = {
        "decision": "NO_GO_B_NEAREST_NOT_BETTER_THAN_RANDOM",
        "b_gate": b_gate(1, (0.6597429801517992, 0.9040461930746312, 0.9310587704066647)),
        "c_gate_executed": False,
        **_zero_counters(
            "model_fit_count",
            "outer_membership_read_count",
            "outer_designated_scoring_open_count",
            "test_context_read_count",
            "submission_write_count",
        ),
    }
    p3_v2 = {
        "decision": "PASS_B_ADAPTIVE_INNER_ONLY_STOP",
        "B_gate": b_gate(2, (0.6528683911306801, 0.914923824174085, 0.8753118074594218)),
        "adaptive_research": True,
        "independent_confirmation": False,
        **_zero_counters(
            "model_fit_count",
            "C_execution_count",
            "outer_membership_read_count",
            "outer_designated_scoring_open_count",
            "test_context_read_count",
            "submission_write_count",
        ),
    }
    p3_v2_precheck = {
        "search": {
            "2024_h2_storm": {"eligible_queries": 9, "forcing_conditioned_queries": 2},
            "2025_h1": {"eligible_queries": 33, "forcing_conditioned_queries": 12},
            "winter_transition": {"eligible_queries": 15, "forcing_conditioned_queries": 5},
        }
    }
    decision = "PASS_C_ADAPTIVE_INNER_STOP_BEFORE_OUTER"
    gate = {
        "pass": True,
        "control_rmse_m": 0.8145113965364116,
        "candidate_rmse_m": 0.8085858943887596,
        "pooled_delta_m": -0.005925502147651973,
        "improved_folds": 2,
        "maximum_station_delta_m": 0.004368704763641151,
        "checks": {"lead_18_non_degrading": True, "lead_24_non_degrading": True},
    }
    p3_v3 = {
        "decision": decision,
        "C_gate": gate,
        "adaptive_research": True,
        "independent_confirmation": False,
        "model_fit_count": 3,
        **_zero_counters(
            "outer_membership_read_count",
            "outer_designated_scoring_open_count",
            "test_context_read_count",
            "submission_write_count",
        ),
    }
    p3_v3_manifest = {
        "decision": decision,
        "same_split_reused": True,
        "independent_confirmation": False,
        "output_sha256": {
            "artifacts\\p3_causal_forcing_analog_inner_predictive_v3\\C_one_shot\\result.json": report.EXPECTED_SHA256[
                "p3_predictive_v3_result"
            ]
        },
        **_zero_counters(
            "outer_membership_or_target_read_count",
            "outer_designated_scoring_open_count",
            "test_context_read_count",
            "submission_write_count",
        ),
    }
    p3_v4_qa = {
        "decision": "QA_GO_OUTER_V4",
        "P0_finding_count": 0,
        "P1_finding_count": 0,
        **_zero_counters(
            "outer_model_execution_count",
            "incumbent_prediction_read_count",
            "designated_target_read_count",
            "test_context_read_count",
            "submission_write_count",
        ),
    }
    p3_outer_gate = {
        "pass": False,
        "incumbent_rmse_m": 0.7801609198910191,
        "candidate_rmse_m": 0.7834329274214491,
        "candidate_minus_incumbent_rmse_m": 0.003272007530429999,
        "by_fold": {
            "2024_h2_storm": {"delta_m": 0.0012524301243984626},
            "2025_h1": {"delta_m": 0.0034960779477826165},
            "winter_transition": {"delta_m": 0.004261038159333874},
        },
        "by_station": {
            "G-ORS": {"delta_m": -0.004233173079980279},
            "I-ORS": {"delta_m": 0.007243123515021432},
            "S-ORS": {"delta_m": 0.0073650273180262404},
        },
        "by_lead": {
            "18": {"delta_m": 0.009504234924817778},
            "24": {"delta_m": 0.00473010464479251},
        },
        "paired_case_bootstrap": {
            "ci90_lower_m": -0.0011743923930318146,
            "ci90_upper_m": 0.007779987367889003,
        },
        "rows": 1092,
        "cases": 182,
    }
    p3_v4 = {
        "decision": "NO_GO_KEEP_FROZEN_INCUMBENT",
        "outer_gate": p3_outer_gate,
        "required_action": "permanent_stop_keep_frozen_incumbent",
        "promotion_performed": False,
        "rerun_prohibited": True,
        "outer_key_membership_read_count": 1,
        "incumbent_prediction_read_count": 1,
        "designated_target_read_count": 1,
        **_zero_counters(
            "model_fit_count", "test_context_read_count", "submission_write_count", "upload_count"
        ),
    }
    p3_v4_manifest = {
        "decision": "NO_GO_KEEP_FROZEN_INCUMBENT",
        "QA_GO_receipt_sha256": report.EXPECTED_SHA256["p3_outer_v4_qa_go"],
        "promotion_performed": False,
        "output_sha256": {
            "artifacts\\p3_causal_forcing_analog_outer_research_v4\\outer_one_shot\\result.json": report.EXPECTED_SHA256[
                "p3_outer_v4_result"
            ]
        },
        **_zero_counters(
            "model_fit_count", "test_context_read_count", "submission_write_count", "upload_count"
        ),
    }
    return {
        "p1_factorial_diagnostic": p1_diag,
        "p1_factorial_result": p1_v1,
        "p1_duration_result": p1_v2,
        "p2_denominator_diagnostic": p2_diag,
        "p2_sigmoid_v2_result": p2_v2,
        "p3_raw_shape_v1_result": p3_v1,
        "p3_forcing_v2_precheck": p3_v2_precheck,
        "p3_forcing_v2_result": p3_v2,
        "p3_predictive_v3_result": p3_v3,
        "p3_predictive_v3_manifest": p3_v3_manifest,
        "p3_outer_v4_qa_go": p3_v4_qa,
        "p3_outer_v4_result": p3_v4,
        "p3_outer_v4_manifest": p3_v4_manifest,
        "hashes": dict(report.EXPECTED_SHA256),
    }


def test_builds_executive_report_with_required_visuals_and_exact_table() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-22T02:00:00+09:00")

    assert artifact["surface"] == "report"
    assert artifact["manifest"]["blocks"][1]["body"].startswith("## Executive Summary")
    assert {chart["id"] for chart in artifact["manifest"]["charts"]} == {
        "p2_support_basis",
        "p3_b_fold_ratios",
    }
    assert len(artifact["manifest"]["tables"]) == 1
    decisions = artifact["snapshot"]["datasets"]["exact_decision_table"]
    assert len(decisions) == 8
    assert decisions[3]["decision"] == "NO-GO_GATE5_0_OF_3"
    assert decisions[6]["decision"] == "조건부 PASS — outer 미개방"
    assert decisions[7]["decision"] == "NO_GO_KEEP_FROZEN_INCUMBENT"


def test_chart_contracts_use_grouped_bars_and_fixed_reference_lines() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-22T02:00:00+09:00")
    charts = {chart["id"]: chart for chart in artifact["manifest"]["charts"]}

    assert charts["p2_support_basis"]["settings"]["groupMode"] == "grouped"
    assert charts["p2_support_basis"]["referenceLines"][0]["value"] == 0.8
    assert charts["p3_b_fold_ratios"]["settings"]["groupMode"] == "grouped"
    assert charts["p3_b_fold_ratios"]["referenceLines"][0]["value"] == 0.9
    assert charts["p3_b_fold_ratios"]["encodings"]["color"]["field"] == "version"


def test_p2_denominator_chart_preserves_full_grid_and_scored_key_support() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-22T02:00:00+09:00")
    rows = artifact["snapshot"]["datasets"]["p2_support_basis"]
    nov_dec = [row for row in rows if row["block"] == "2025년 11–12월"]

    assert {row["support_basis"] for row in nov_dec} == {"전체 61일 격자", "채점키 시각"}
    support = {row["support_basis"]: row["support_share"] for row in nov_dec}
    assert support["전체 61일 격자"] == pytest.approx(0.6257969034608379)
    assert support["채점키 시각"] == pytest.approx(0.9762031610726336)


def test_p3_outer_failure_closes_the_exact_candidate_and_keeps_incumbent() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-22T02:00:00+09:00")
    serialized = json.dumps(artifact, ensure_ascii=False)
    decision = artifact["snapshot"]["datasets"]["exact_decision_table"][-1]

    assert decision["decision"] == "NO_GO_KEEP_FROZEN_INCUMBENT"
    assert "재튜닝·재실행 금지" in decision["implication"]
    assert "P3 동결 incumbent를 유지" in serialized
    assert "공식 hidden score가 아닙니다" in serialized


def test_report_is_aggregate_only_and_records_html_fallback_source_note() -> None:
    artifact = report.build_artifact(_evidence(), generated_at="2026-08-22T02:00:00+09:00")
    serialized = json.dumps(artifact, ensure_ascii=False)
    source_ids = {source["id"] for source in artifact["manifest"]["sources"]}

    assert "report_delivery_note" in source_ids
    assert "MCP report tools were unavailable" in serialized
    assert "정책·점수 환산 상수" in serialized
    assert "C:\\Users\\" not in serialized
    assert ".parquet" not in serialized
    assert "test_context.parquet" not in serialized
    assert all(path.suffix == ".json" for path in report.RELATIVE_PATHS.values())
    assert all(not path.is_absolute() for path in report.RELATIVE_PATHS.values())


def test_pending_v3_pin_fails_closed_before_any_report_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        report,
        "RELATIVE_PATHS",
        {"p3_predictive_v3_result": Path("artifacts/p3_v3/result.json")},
    )
    monkeypatch.setattr(
        report,
        "EXPECTED_SHA256",
        {"p3_predictive_v3_result": report.PENDING_P3_V3_SHA256},
    )

    with pytest.raises(report.ReportEvidenceError, match="unsealed SHA-256 pin"):
        report.collect_evidence(tmp_path)


def test_sha_mismatch_fails_before_json_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = Path("artifacts/safe/result.json")
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("not even valid JSON", encoding="utf-8")
    monkeypatch.setattr(report, "RELATIVE_PATHS", {"safe": relative})
    monkeypatch.setattr(report, "EXPECTED_SHA256", {"safe": "0" * 64})

    with pytest.raises(report.ReportEvidenceError, match="sealed SHA mismatch"):
        report.collect_evidence(tmp_path)
