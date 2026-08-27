"""Build the append-only validation-system correction report.

The builder consumes only SHA-pinned aggregate audit JSON.  It intentionally
supersedes the earlier structural-path report's validation endorsement while
preserving frozen submissions as immutable baselines/risk controls.  Portable
HTML packaging remains a separate, one-shot step after all three audit pins are
final.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
REPORT_TITLE = "검증 시스템 감사 — 동결은 정답 확정이 아니다"
REPORT_ID = "validation-system-audit-2026-08-22-r1"
DEFAULT_OUTPUT = Path("reports/generated/validation_system_audit_2026-08-22_r1/artifact.json")

EXPECTED_SHA256 = {
    "p1_audit": "0752941c2d11d5e052307fd7eaf83d60c39f7e28768c06cc2da285814ed9cd40",
    "p2_audit": "ea09df2b874248ddc7466406da1908d137d9025b126a5de43482317e11b30b17",
    "p3_audit": "4e2f2a74612621f5659c9fd5e9c4996ed8322d2f6cd1a19104f620490568b12b",
    "cross_problem_policy": (
        "9529aa4aad806799dd3ed410e55bd4d3563d857358ae4e7c99f59789db4a27e7"
    ),
    "superseded_report_artifact": (
        "18cdfe378a72862498b445c6c9f569cf6dc69a7077b178f7eeab591713d6055c"
    ),
}
RELATIVE_PATHS = {
    "p1_audit": Path("artifacts/validation_system_audit_20260822/p1.json"),
    "p2_audit": Path("artifacts/validation_system_audit_20260822/p2.json"),
    "p3_audit": Path("artifacts/validation_system_audit_20260822/p3.json"),
    "cross_problem_policy": Path(
        "artifacts/validation_system_audit_20260822/cross_problem_policy.json"
    ),
    "superseded_report_artifact": Path(
        "reports/generated/structural_path_audit_2026-08-22_r1/artifact.json"
    ),
}


class ValidationReportError(RuntimeError):
    """Raised when aggregate evidence or report structure is not sealed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationReportError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"aggregate JSON must be an object: {path}")
    return payload


def _validate_path_contract() -> None:
    _require(set(RELATIVE_PATHS) == set(EXPECTED_SHA256), "pin/path set drifted")
    for name, relative in RELATIVE_PATHS.items():
        _require(not relative.is_absolute(), f"absolute evidence path forbidden: {name}")
        _require(relative.suffix.lower() == ".json", f"non-JSON evidence forbidden: {name}")
        lowered = relative.as_posix().lower()
        for token in (".csv", ".parquet", "test_context", "test_index", "submission.csv"):
            _require(token not in lowered, f"row-level evidence class forbidden: {name}")


def collect_evidence(root: Path) -> dict[str, Any]:
    """Load only final aggregate JSON and fail closed on any pending/drifted pin."""

    _validate_path_contract()
    evidence: dict[str, Any] = {}
    for name, relative in RELATIVE_PATHS.items():
        expected = EXPECTED_SHA256[name]
        _require(
            re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            f"unsealed evidence pin: {name}",
        )
        path = root / relative
        _require(path.is_file(), f"missing aggregate evidence: {relative.as_posix()}")
        actual = _sha256(path)
        _require(actual == expected, f"SHA mismatch for {name}: {actual} != {expected}")
        evidence[name] = _read_json(path)
    evidence["hashes"] = dict(EXPECTED_SHA256)
    return evidence


def _trust_text(item: dict[str, Any]) -> str:
    score = item.get("score")
    rating = str(item.get("rating", "")).strip()
    _require(isinstance(score, (int, float)) and 0 <= float(score) <= 100, "invalid trust score")
    _require(bool(rating), "missing trust rating")
    return f"{float(score):.0f}/100 · {rating}"


def _adapt_p1(payload: dict[str, Any]) -> dict[str, Any]:
    _require(payload.get("decision") == "relative ranking", "P1 audit decision drifted")
    confidence = payload["confidence"]
    _require(confidence["level"] == "moderate", "P1 confidence label drifted")
    _require(float(confidence["score_0_to_1"]) == 0.7, "P1 confidence score drifted")
    conclusion = payload["executive_conclusion"]
    _require(conclusion["trust_frozen_absolute_score"] is False, "P1 hidden trust drifted")
    _require(conclusion["trust_local_relative_ranking"] is True, "P1 local rank drifted")
    _require(conclusion["absolute_hidden_score_estimate"] is None, "P1 hidden estimate drifted")

    operations = payload["scope_and_access_counters"]
    for key in (
        "official_test_label_accesses",
        "incumbent_model_inference_calls",
        "incumbent_model_training_calls",
        "new_prediction_rows_generated",
        "submission_files_modified",
        "submission_uploads",
        "source_files_modified",
        "existing_artifacts_modified",
        "raw_rows_emitted",
        "paths_or_secrets_emitted",
    ):
        _require(int(operations[key]) == 0, f"P1 forbidden operation drifted: {key}")

    provenance = payload["provenance_and_model"]
    _require(provenance["saved_model_reproduction_sha_identical"] is True, "P1 replay drifted")
    _require(
        provenance["training_manifest_git_tree_contains_p1_core_code"] is False,
        "P1 source-linkage limitation drifted",
    )
    splits = payload["split_chronology_and_leakage_audit"]
    for key in (
        "fold_membership_mismatches",
        "cross_fold_duplicate_keys",
        "outer_train_validation_overlap_rows",
        "positive_event_overlap_inner_fit_calibration",
        "positive_event_overlap_calibration_outer_validation",
    ):
        _require(int(splits[key]) == 0, f"P1 split-integrity finding drifted: {key}")
    _require(
        splits["centered_feature_reach_is_inside_inner_and_outer_embargo"] is True,
        "P1 embargo finding drifted",
    )

    rank = payload["local_relative_ranking_evidence"]
    delta = float(rank["weighted_f1_delta"])
    ci_low, ci_high = (float(value) for value in rank["paired_event_day_block_delta_ci90"])
    _require(delta > 0 and ci_low > 0 and ci_high > ci_low, "P1 paired rank evidence drifted")
    _require(
        all(float(value) > 0 for value in rank["station_f1_deltas"].values()),
        "P1 station rank evidence drifted",
    )
    alignment = payload["oof_to_official_test_population_alignment"]
    unsupported = float(alignment["total_test_share_without_same_season_q2_station_layer_month_support"])
    domain = payload["label_free_domain_classifier"]
    domain_auc = float(domain["all_period_value_and_operational"]["pooled_oof_roc_auc"])
    exposure = payload["outer_label_exposure_and_selection_multiplicity"]
    _require(exposure["outer_is_independent_holdout_after_exposure_history"] is False, "P1 exposure drifted")
    _require(exposure["multiplicity_adjusted_confidence_interval_available"] is False, "P1 multiplicity drifted")
    absolute = payload["absolute_hidden_score_calibration"]
    _require(absolute["valid_hidden_score_prediction_interval"] is None, "P1 hidden interval drifted")
    _require(absolute["official_baseline_directly_comparable"] is False, "P1 baseline comparison drifted")

    return {
        "problem": "P1",
        "implementation_integrity": (
            "중상 — key/split·embargo·saved-model 재현은 일치; 다만 training manifest의 core-code "
            "연결은 cryptographic pin이 아니라 사후 temporal linkage"
        ),
        "local_relative_ranking": (
            f"중상 — XGBoost {float(rank['xgboost_test_share_weighted_f1']):.6f} vs "
            f"LightGBM {float(rank['lightgbm_test_share_weighted_f1']):.6f}; Δ {delta:+.6f}, "
            f"paired CI90 [{ci_low:+.6f}, {ci_high:+.6f}]"
        ),
        "hidden_calibration": (
            f"낮음·미확립 — numeric point/interval 없음; same-season support 없는 test mass "
            f"{unsupported:.2%}, label-free domain AUC {domain_auc:.3f}"
        ),
        "adaptive_exposure": (
            f"높음 — outer-result exposure {int(exposure['outer_evaluations_or_closed_runs_with_outer_result'])}회, "
            f"family {int(exposure['distinct_recorded_families'])}개, virgin local tail "
            f"{int(exposure['virgin_local_tail_rows_after_fixed_q4'])}, multiplicity-adjusted CI 없음"
        ),
        "action": (
            "exact frozen candidate를 rollback baseline으로 보존; untouched 2026-matched labels 또는 "
            "official score 없이는 hidden F1/최적성 주장 금지"
        ),
        "decision": "LOCAL_RELATIVE_RANKING_SUPPORTED__ABSOLUTE_HIDDEN_UNCALIBRATED",
        "source_id": "p1_validation_audit",
    }


def _adapt_p2(payload: dict[str, Any]) -> dict[str, Any]:
    decision = payload["decision"]
    _require(decision["overall_trust_rating"] == "LOW", "P2 overall trust drifted")
    local = decision["local_candidate_ranking"]
    hidden = decision["absolute_hidden_calibration"]
    _require(int(local["score_out_of_5"]) == 2, "P2 local trust score drifted")
    _require(int(hidden["score_out_of_5"]) == 1, "P2 hidden trust score drifted")
    _require(
        decision["promotion_recommendation"] == "DO_NOT_PROMOTE_FROM_THIS_VALIDATION_SYSTEM_ALONE",
        "P2 promotion decision drifted",
    )

    safety = payload["safety_attestation"]
    for key in (
        "hidden_target_layer_temp_values_accessed_or_retained",
        "hidden_target_layer_psal_values_accessed_or_retained",
        "model_or_candidate_execution_count",
        "submission_or_upload_actions",
    ):
        _require(int(safety[key]) == 0, f"P2 forbidden operation drifted: {key}")
    _require(safety["row_level_values_emitted"] is False, "P2 row-level boundary drifted")
    _require(safety["hidden_public_layers_used"] == [1, 5, 6, 7, 8], "P2 public-layer boundary drifted")

    validation = payload["validation_contract_audit"]
    _require(
        int(validation["oof_grain"]["duplicate_time_layer_keys"]) == 0,
        "P2 OOF key integrity drifted",
    )
    _require(
        int(validation["hidden_official_grain"]["duplicate_time_layer_keys"]) == 0,
        "P2 hidden key integrity drifted",
    )
    _require(validation["masking"]["direct_target_feature_leakage_found"] is False, "P2 leakage drifted")
    selection = validation["purge_and_selection"]
    _require(int(selection["explicit_temporal_purge_hours"]) == 0, "P2 purge finding drifted")
    _require(selection["deep_checkpoint_outer_label_use"].startswith("FOUND:"), "P2 checkpoint finding drifted")
    _require(selection["v1_v2_extrapolation"].startswith("NOT NESTED:"), "P2 nesting finding drifted")

    coverage = payload["denominator_and_coverage"]
    local_coverage = float(coverage["pooled"]["coverage"])
    hidden_coverage = float(coverage["hidden"]["row_coverage"])
    uncertainty = payload["recomputed_uncertainty"]
    overall = uncertainty["overall"]
    same_season = uncertainty["by_block"]["2024_sep_oct"]
    later = uncertainty["by_block"]["2025_nov_dec"]
    overall_delta = float(overall["delta_rmse_c"])
    same_ci_low, same_ci_high = (float(value) for value in same_season["delta_ci90_c"])
    later_ci_low, later_ci_high = (float(value) for value in later["delta_ci90_c"])
    _require(overall_delta < 0, "P2 pooled direction drifted")
    _require(same_ci_low < 0 < same_ci_high, "P2 same-season uncertainty drifted")
    _require(float(later["delta_rmse_c"]) > 0 and later_ci_low > 0, "P2 Nov-Dec failure drifted")

    exposure = payload["adaptive_exposure_audit"]
    _require(exposure["v2_adaptive_provenance"]["config_flags_adaptive_after_outer_exposure"] is True, "P2 adaptive flag drifted")
    _require(exposure["v2_adaptive_provenance"]["fresh_holdout_claimed"] is False, "P2 holdout status drifted")

    return {
        "problem": "P2",
        "implementation_integrity": (
            "중간 — key·mask·direct-target leakage 검사는 일치; explicit purge 0h이며 deep checkpoint가 "
            "동일 outer block RMSE로 선택됨"
        ),
        "local_relative_ranking": (
            f"2/5 · {local['rating']} — pooled Δ {overall_delta:+.6f}°C; same-season CI90 "
            f"[{same_ci_low:+.6f}, {same_ci_high:+.6f}]는 0 포함, Nov–Dec는 "
            f"{float(later['delta_rmse_c']):+.6f}°C 악화"
        ),
        "hidden_calibration": (
            f"1/5 · {hidden['rating']} — local row coverage {local_coverage:.2%} vs hidden "
            f"{hidden_coverage:.2%}; 0.768367°C는 hidden estimate가 아님"
        ),
        "adaptive_exposure": (
            f"매우 높음 — executed generation ≥{int(exposure['executed_generation_lower_bound'])}, "
            f"same-block result artifacts {int(exposure['result_artifacts_explicitly_containing_same_three_blocks_or_69850_rows'])}, "
            f"fresh holdout 없음, stack optimism gap "
            f"{float(exposure['deep_stack_optimism_indicator']['optimism_gap_c']):.6f}°C"
        ),
        "action": (
            "이 validation system만으로 승격 금지; frozen candidate는 adaptive research/rollback "
            "artifact로만 보존하고 untouched seasonal validation을 봉인"
        ),
        "decision": str(decision["promotion_recommendation"]),
        "source_id": "p2_validation_audit",
    }


def _adapt_p3(payload: dict[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("decision")
        == "KEEP_FROZEN_INCUMBENT_AS_RISK_CONTROL_ONLY__HIDDEN_GENERALIZATION_UNVERIFIED",
        "P3 audit decision drifted",
    )
    _require(
        payload["executive_conclusion"]["official_hidden_score_known"] is False,
        "P3 hidden status drifted",
    )
    operations = payload["operation_counters"]
    for key in (
        "model_fits",
        "prediction_generations",
        "prediction_writes",
        "official_test_target_reads",
        "submission_reads",
        "submission_writes",
        "uploads",
        "source_mutations",
    ):
        _require(int(operations[key]) == 0, f"P3 forbidden operation drifted: {key}")
    sampling = payload["sampling_and_event_audit"]
    _require(
        sampling["implementation"]["fold_membership_exactly_recomputed"] is True,
        "P3 membership drifted",
    )
    _require(
        int(sampling["spacing"]["context48_plus_target24_footprint_overlap_pairs_below_72h"]) == 1,
        "P3 footprint-overlap finding drifted",
    )
    calibration = payload["official_baseline_calibration"]
    _require(calibration["official_T"]["is_hidden_model_score"] is False, "P3 T correction drifted")
    trust = payload["trust"]
    exposure = payload["adaptive_exposure"]
    return {
        "problem": "P3",
        "implementation_integrity": (
            "중간 — 182-case membership·RMSE 재계산은 일치하지만 window reset으로 "
            f"global min gap {sampling['spacing']['all_folds_station_min_gap_hours']:.3f}h, "
            "72h footprint overlap 1쌍"
        ),
        "local_relative_ranking": _trust_text(trust["relative_local_ranking"]),
        "hidden_calibration": (
            f"{_trust_text(trust['absolute_hidden_score'])}; local B−official B "
            f"{calibration['local_minus_official_B_m']:+.6f}m"
        ),
        "adaptive_exposure": (
            f"높음 — exact-key target-bearing OOF ≥"
            f"{exposure['persisted_same_key_oof_lower_bound']['target_bearing_exact_same_key_artifact_count']}, "
            f"same-grain RMSE 문서 {exposure['same_1092_grain_rmse_json_document_count']}, "
            f"ledger scoring {exposure['central_outer_truth_ledger']['explicit_designated_scoring_events']}회"
        ),
        "action": (
            "동결 incumbent를 immutable risk-control baseline으로만 유지; 새 episode-disjoint·global-78h "
            "검증면 없이는 hidden 최적성/승격을 주장하지 않음"
        ),
        "decision": str(payload["decision"]),
        "source_id": "p3_validation_audit",
    }


def _validate_cross_problem_policy(payload: dict[str, Any]) -> None:
    _require(
        payload.get("schema_version") == "cross_problem_official_scoring_preregistration.v1",
        "cross-problem policy schema drifted",
    )
    _require(
        payload.get("status") == "PREREGISTERED__P2_P3_POOLS_PINNED__P1_POOL_INCOMPLETE",
        "cross-problem policy status drifted",
    )
    non_actions = payload["non_actions_in_this_policy_run"]
    for key in (
        "model_fits",
        "prediction_generations",
        "prediction_values_decoded_or_inspected",
        "prediction_writes",
        "submission_files_created_or_modified",
        "uploads",
        "existing_artifacts_modified",
    ):
        _require(int(non_actions[key]) == 0, f"policy forbidden operation drifted: {key}")

    audit_pins = payload["evidence_pin"]["audits"]
    for problem, source_id in (("P1", "p1_audit"), ("P2", "p2_audit"), ("P3", "p3_audit")):
        _require(
            audit_pins[problem]["sha256"] == EXPECTED_SHA256[source_id],
            f"policy audit pin drifted: {problem}",
        )
    rules = payload["governing_rules"]
    _require(int(rules["maximum_scored_candidates_per_problem"]) == 3, "policy score cap drifted")
    _require(rules["one_score_per_structural_family"] is True, "policy family cap drifted")
    _require(rules["within_family_leaderboard_tuning"] is False, "policy tuning rule drifted")
    _require(rules["score_derived_blends"] is False, "policy blend rule drifted")
    _require(rules["score_derived_thresholds"] is False, "policy threshold rule drifted")
    _require(rules["same_split_comparisons_only"] is True, "policy split rule drifted")

    for problem in ("P2", "P3"):
        benchmark = payload["problem_policies"][problem]["official_benchmark"]["path"]
        _require(not Path(benchmark).is_absolute(), f"policy benchmark path is absolute: {problem}")
        _require("downloads" not in benchmark.lower(), f"policy personal path leaked: {problem}")
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    for forbidden in ("c:/users/", "c:\\users\\"):
        _require(forbidden not in serialized, "policy contains a personal absolute path")


def normalize_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        _adapt_p1(evidence["p1_audit"]),
        _adapt_p2(evidence["p2_audit"]),
        _adapt_p3(evidence["p3_audit"]),
    ]
    _require([row["problem"] for row in rows] == ["P1", "P2", "P3"], "problem order drifted")
    return rows


def _source(source_id: str, label: str, path: str, sha256: str, *, note: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "sha256": sha256,
        "note": note,
    }


def _sql_text(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_artifact(
    evidence: dict[str, Any], *, generated_at: str, rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build the correction artifact; charts stay absent without comparable metrics."""

    _validate_cross_problem_policy(evidence["cross_problem_policy"])
    normalized = rows if rows is not None else normalize_rows(evidence)
    _require([row["problem"] for row in normalized] == ["P1", "P2", "P3"], "row order drifted")
    hashes = evidence["hashes"]
    sources = [
        _source(
            "p1_validation_audit",
            "P1 validation-system audit",
            RELATIVE_PATHS["p1_audit"].as_posix(),
            hashes["p1_audit"],
            note="Final aggregate-only audit; no row-level values are consumed by this report.",
        ),
        _source(
            "p2_validation_audit",
            "P2 validation-system audit",
            RELATIVE_PATHS["p2_audit"].as_posix(),
            hashes["p2_audit"],
            note="Final aggregate-only audit; no row-level values are consumed by this report.",
        ),
        _source(
            "p3_validation_audit",
            "P3 validation-system audit",
            RELATIVE_PATHS["p3_audit"].as_posix(),
            hashes["p3_audit"],
            note="Final aggregate-only audit; official test comparison is label-free.",
        ),
        _source(
            "cross_problem_policy",
            "Cross-problem official-score preregistration policy",
            RELATIVE_PATHS["cross_problem_policy"].as_posix(),
            hashes["cross_problem_policy"],
            note=(
                "Aggregate-only action policy; exact audit pins, immutable-baseline rule, candidate "
                "caps, split branches, and leaderboard stop rules are preregistered."
            ),
        ),
        _source(
            "superseded_structural_report",
            "Superseded structural-path audit artifact",
            RELATIVE_PATHS["superseded_report_artifact"].as_posix(),
            hashes["superseded_report_artifact"],
            note="Supersession target only; it is not evidence for corrected validation claims.",
        ),
        {
            "id": "method_note",
            "label": "Technical synthesis, delivery, and comparability note",
            "note": (
                "The trust matrix is a deterministic field-level synthesis of the three SHA-pinned "
                "aggregate audits. MCP report tools were unavailable, so the validated canonical "
                "artifact is delivered through the technical portable HTML fallback. No chart is "
                "shown because P1 uses F1/population-shift diagnostics, P2 uses temperature RMSE and "
                "seasonal block uncertainty, and P3 uses water-level RMSE/case sampling; these do not "
                "form a quantitatively comparable hidden-representativeness or uncertainty scale."
            ),
        },
    ]

    decisions = {row["problem"]: row["decision"] for row in normalized}
    p2_uncertainty = evidence["p2_audit"]["recomputed_uncertainty"]
    p2_scopes = [
        ("Pooled", p2_uncertainty["overall"]),
        ("2024 Sep–Oct", p2_uncertainty["by_block"]["2024_sep_oct"]),
        ("2025 Jul–Aug", p2_uncertainty["by_block"]["2025_jul_aug"]),
        ("2025 Nov–Dec", p2_uncertainty["by_block"]["2025_nov_dec"]),
    ]
    p2_chart_rows = []
    for sequence, (scope, metrics) in enumerate(p2_scopes, start=1):
        delta = float(metrics["delta_rmse_c"])
        ci_low, ci_high = (float(value) for value in metrics["delta_ci90_c"])
        p2_chart_rows.append(
            {
                "sequence": sequence,
                "scope": scope,
                "delta_rmse_c": delta,
                "signed_delta_label": f"{delta:+.6f}°C",
                "ci90_lower_c": ci_low,
                "ci90_upper_c": ci_high,
                "kst_days": int(metrics["kst_days"]),
                "rows": int(metrics["rows"]),
                "baseline_rmse_c": float(metrics["baseline_rmse_c"]),
                "candidate_rmse_c": float(metrics["candidate_rmse_c"]),
                "direction": "개선 (Δ<0)" if delta < 0 else "악화 (Δ>0)",
            }
        )
    p2_chart_sql = " UNION ALL ".join(
        (
            f"SELECT {row['sequence']} AS sequence, {_sql_text(row['scope'])} AS scope, "
            f"{row['delta_rmse_c']!r} AS delta_rmse_c, "
            f"{_sql_text(row['signed_delta_label'])} AS signed_delta_label, "
            f"{row['ci90_lower_c']!r} AS ci90_lower_c, "
            f"{row['ci90_upper_c']!r} AS ci90_upper_c, "
            f"{row['kst_days']} AS kst_days, {row['rows']} AS rows, "
            f"{row['baseline_rmse_c']!r} AS baseline_rmse_c, "
            f"{row['candidate_rmse_c']!r} AS candidate_rmse_c, "
            f"{_sql_text(row['direction'])} AS direction"
        )
        for row in p2_chart_rows
    )
    trust_table_sql = " UNION ALL ".join(
        (
            f"SELECT {sequence} AS sequence, {_sql_text(row['problem'])} AS problem, "
            f"{_sql_text(row['implementation_integrity'])} AS implementation_integrity, "
            f"{_sql_text(row['local_relative_ranking'])} AS local_relative_ranking, "
            f"{_sql_text(row['hidden_calibration'])} AS hidden_calibration, "
            f"{_sql_text(row['adaptive_exposure'])} AS adaptive_exposure, "
            f"{_sql_text(row['action'])} AS action"
        )
        for sequence, row in enumerate(normalized, start=1)
    )
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Technical Summary\n\n"
                "**정정 결론:** freeze = immutable baseline/risk control, not validation endorsement. "
                "동결은 기존 산출물을 되돌릴 수 있게 보존하고 검증되지 않은 challenger의 승격 위험을 막는 "
                "운영 조치입니다. 검증 시스템이 충분하거나 frozen incumbent가 official hidden에서 최적이라는 "
                "승인은 아닙니다.\n\n"
                "기존 구조적 해법 감사의 **‘세 문제 모두 동결 유지가 맞다’**는 문장을 정답 확정 근거로 "
                "사용하는 해석을 이 보고서가 명시적으로 대체합니다. 허용되는 논리는 `challenger가 reused "
                "local surface에서 실패` → `그 exact challenger를 승격하지 않는다`까지입니다. "
                "`incumbent의 hidden score가 검증됐다` 또는 `현 제출이 최적이다`는 결론은 나오지 않습니다.\n\n"
                "P1은 local 상대 순위 근거가 세 문제 중 가장 강하지만 absolute hidden calibration은 없습니다. "
                "P2는 pooled 개선이 계절 블록에 따라 뒤집히고 동일 outer surface가 반복 사용됐습니다. P3는 "
                "182-case membership 재현에도 global spacing·episode 독립성 결함과 높은 adaptive exposure가 있어 "
                "local 값의 hidden 대표성이 낮습니다. 따라서 세 문제 모두 baseline은 보존하되, 승격 판단은 "
                "새 untouched/forward validation surface에서 다시 해야 합니다."
            ),
        },
        {
            "id": "key_findings",
            "type": "markdown",
            "sourceId": "method_note",
            "body": (
                "## Key Findings — 신뢰 축을 분리하면 결론이 달라진다\n\n"
                "아래 표는 각 문제를 implementation integrity, local relative ranking, hidden calibration, "
                "adaptive exposure, action의 다섯 축으로 분리합니다. 점수·등급은 각 감사 내부의 판단 척도이며 "
                "문제 간 확률 비교가 아닙니다. 세 문제의 단위와 불확실성 정의가 달라 비교 차트는 의도적으로 "
                "생략하고 exact lookup table을 사용했습니다."
            ),
        },
        {"id": "trust_matrix", "type": "table", "tableId": "trust_matrix"},
        {
            "id": "p1_finding",
            "type": "markdown",
            "sourceId": "p1_validation_audit",
            "body": (
                "### P1 — local rank는 지지되지만 hidden F1은 추정할 수 없다\n\n"
                f"판정: `{decisions['P1']}`. XGBoost의 test-share-weighted local F1은 0.813316, "
                "LightGBM은 0.768804로 Δ +0.044511이며 paired event/day CI90은 "
                "[+0.019929, +0.070360]입니다. 그러나 official test mass의 43.61%는 같은 계절 Q2 "
                "station-layer-month support가 없고, label-free domain AUC는 0.880입니다. outer-result "
                "exposure 13회·10 family·virgin local tail 0이므로 0.813316은 conditional local ranking "
                "statistic이지 expected leaderboard score가 아닙니다."
            ),
        },
        {
            "id": "p2_finding",
            "type": "markdown",
            "sourceId": "p2_validation_audit",
            "body": (
                "### P2 — pooled gain은 seasonal robustness를 통과하지 못했다\n\n"
                f"판정: `{decisions['P2']}`. pooled ΔRMSE는 −0.006050°C이지만 same-season Sep–Oct "
                "CI90은 [−0.004239, +0.001759]로 0을 포함하고, Nov–Dec는 +0.001257°C로 "
                "유의하게 악화됩니다. local coverage 87.87%와 hidden coverage 98.90%도 다릅니다. "
                "executed generation은 최소 18개, 같은 three-block/69,850-row 결과 artifact는 17개이며 "
                "fresh holdout이 없습니다. Pooled 163일 CI90은 [−0.009841, −0.002163], "
                "Jul–Aug 62일은 [−0.018483, −0.004282], Nov–Dec 40일은 "
                "[+0.000018, +0.002663]입니다. 따라서 0.768367°C는 hidden RMSE 추정치가 아닙니다."
            ),
        },
        {"id": "p2_delta_chart", "type": "chart", "chartId": "p2_delta_rmse"},
        {
            "id": "p3_finding",
            "type": "markdown",
            "sourceId": "p3_validation_audit",
            "body": (
                "### P3 — reproduced OOF도 official hidden 대표성을 보장하지 않는다\n\n"
                f"판정: `{decisions['P3']}`. local incumbent RMSE 0.780161m은 reused 182-case OOF의 "
                "값입니다. Original splitter는 각 window 내부 78h spacing을 재현하지만 union의 same-station "
                "minimum gap은 34.667h이고 72h context+target footprint overlap이 1쌍 있습니다. Local "
                "persistence 0.862977m은 official B 0.769455m보다 +0.093522m 높습니다. Official "
                "`T=0.624165`는 hidden model score가 아니라 공개 policy/scoring constant입니다."
            ),
        },
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## Scope, Data, and Metric Definitions\n\n"
                "- **Freeze:** exact bytes를 보존하는 rollback/risk-control 상태입니다. 성능 승인 상태가 아닙니다.\n"
                "- **Implementation integrity:** key/grain, split, 재현, leakage boundary와 금지 연산 준수입니다.\n"
                "- **Local relative ranking:** 같은 관측 local labels와 비교 기준 안에서 후보 순서가 유지되는지입니다.\n"
                "- **Hidden calibration:** local absolute metric이 official hidden population·denominator·season을 "
                "대표하는지입니다. hidden label이 없으면 point estimate와 prediction interval은 별도 근거 없이는 "
                "산출하지 않습니다.\n"
                "- **Adaptive exposure:** 동일 label surface가 tuning, checkpoint, family selection, diagnostics에 "
                "반복 사용된 정도입니다.\n"
                "- **단위:** P1은 weighted F1, P2는 temperature RMSE(°C), P3는 water-level RMSE(m)입니다. "
                "서로 다른 metric·grain·sampling contract라 크기나 신뢰 점수를 직접 비교하지 않습니다."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "method_note",
            "body": (
                "## Methodology — aggregate-only, SHA-pinned, fail-closed synthesis\n\n"
                "보고서 builder는 P1/P2/P3 final aggregate audit JSON, cross-problem preregistration policy, "
                "superseded report artifact만 exact "
                "SHA-256으로 pin해 읽습니다. P1/P2/P3 schema별 adapter가 decision, 금지 연산 0, split/key "
                "integrity, uncertainty sign, exposure 상태와 calibration limitation을 assertion으로 검증하며, "
                "pin 또는 schema가 drift하면 artifact 생성을 중단합니다. Raw/OOF/test/submission row와 model "
                "prediction은 builder input contract에서 금지됩니다.\n\n"
                "시각화 검토에서는 세 감사의 hidden-representativeness/uncertainty를 공통 수치 축으로 "
                "정규화할 근거가 없다고 판정했습니다. 따라서 오도 가능한 cross-problem chart 대신 exact trust "
                "matrix를 선택했습니다. MCP report tools were unavailable; canonical artifact는 technical portable "
                "HTML fallback으로 package되며 delivery layer는 새로운 계산을 추가하지 않습니다."
            ),
        },
        {
            "id": "limitations_robustness",
            "type": "markdown",
            "body": (
                "## Limitations, Uncertainty, and Robustness Checks\n\n"
                "P1의 paired CI는 observed·already-selected 2025 OOF blocks에 conditional하며 2026 shift와 "
                "winner’s curse를 포함하지 않습니다. P2의 day-cluster uncertainty는 row autocorrelation을 "
                "완화하지만 세 seasonal block뿐이고 same-season interval은 0을 포함합니다. P3의 case bootstrap도 "
                "already-reused 182 cases에 conditional하며, 한 episode 중복과 context footprint overlap이 있습니다.\n\n"
                "모든 hidden comparison은 label-free covariate/index·public baseline aggregate에 한정됩니다. "
                "따라서 이 보고서는 absolute hidden score나 최종 제출 순위의 확률을 추정하지 않습니다. "
                "동일 label surface의 exposure count는 filesystem/ledger lower bound라 실제 adaptive selection은 "
                "더 클 수 있습니다. 이전 report/builder/artifact는 수정하지 않고 append-only generation으로 "
                "정정하므로 historical experiment 판정은 그대로 보존됩니다."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended Next Steps — 새 검증면 없이는 승격하지 않는다\n\n"
                "1. Frozen artifacts를 byte-level rollback 기준으로 유지하고, 이미 노출된 outer surfaces는 "
                "development data로 재분류합니다.\n"
                "2. P1은 untouched 2026-matched labeled slice 또는 organizer score로 absolute calibration을 "
                "검증합니다.\n"
                "3. P2는 season-matched untouched block을 먼저 seal하고 checkpoint·factor·routing selection을 "
                "그 바깥에서 끝낸 뒤 one-shot 평가합니다.\n"
                "4. P3는 episode-disjoint + global 78h + 72h footprint-disjoint case selection을 고정하고, "
                "후보 선택과 최종 평가를 분리합니다.\n"
                "5. 새 후보 승격에는 leakage/overlap 검사, selection-adjusted uncertainty, predeclared action "
                "threshold를 동시에 요구합니다. 새 검증면이 없으면 action은 ‘현재 baseline 유지’이지 "
                "‘정답 확정’이 아닙니다."
            ),
        },
        {
            "id": "official_score_policy",
            "type": "markdown",
            "sourceId": "cross_problem_policy",
            "body": (
                "### Preregistered official-score boundary\n\n"
                "현재 ready CSV 세 개는 exact-byte immutable baselines로 유지합니다. 첫 official score 전에만 "
                "eligible candidate identity와 SHA를 봉인할 수 있고, 각 문제의 첫 score가 나오면 새 candidate "
                "생성을 영구 중단합니다. 문제당 scored candidate 상한은 3개, structural family당 1회이며 "
                "score-derived blend·threshold와 within-family leaderboard tuning은 금지됩니다. 비교는 동일 split에서만 "
                "허용하고, split identity가 불명확하거나 바뀌면 중단합니다. P2/P3 pool은 pin됐지만 P1 pool은 "
                "현재 incomplete이므로 첫 upload 전에 pool 완결 또는 one-candidate terminal waiver가 필요합니다. "
                "어떤 upload도 별도 SHA/schema/daily-slot 확인과 명시적 사용자 승인을 요구합니다."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further Questions\n\n"
                "- 각 문제에서 완전히 untouched인 label period·episode를 실제로 확보할 수 있는가?\n"
                "- organizer score가 공개되면 local metric과의 차이를 어떤 population/denominator decomposition으로 "
                "설명할 것인가?\n"
                "- 반복 outer exposure를 반영할 selection-adjusted interval 또는 nested evaluation budget을 "
                "사전에 어떻게 고정할 것인가?\n"
                "- ‘baseline 유지’, ‘새 candidate 연구’, ‘submission 교체’를 분리하는 최소 effect와 downside "
                "threshold는 무엇인가?"
            ),
        },
    ]

    chart = {
        "id": "p2_delta_rmse",
        "title": "P2 ΔRMSE by validation scope",
        "subtitle": (
            "°C; candidate − physical-projection base; 음수=개선, 양수=악화; exposed 3-block OOF"
        ),
        "showDescription": True,
        "intent": "comparison",
        "question": "P2 candidate improvement가 pooled 및 세 seasonal validation block에서 일관적인가?",
        "rationale": (
            "같은 °C 단위의 네 aggregate ΔRMSE를 signed horizontal bar로 비교한다. "
            "CI는 bar geometry로 흉내내지 않고 reviewed dataset과 인접 문단에 보존한다. "
            "Direction은 최대 두 categorical roots로 제한하며 signed labels와 neutral zero line을 "
            "함께 사용해 color alone을 피한다."
        ),
        "type": "horizontalBar",
        "dataset": "p2_delta_rmse",
        "sourceId": "p2_validation_audit",
        "source": {
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": p2_chart_sql,
                "description": (
                    "Deterministic four-row projection of the SHA-pinned P2 aggregate uncertainty audit."
                ),
                "tables_used": [RELATIVE_PATHS["p2_audit"].as_posix()],
                "filters": [
                    "candidate minus physical-projection baseline",
                    "paired KST-calendar-day bootstrap",
                    "three exposed validation blocks plus pooled aggregate",
                    "raw/test-target/prediction/submission rows = 0",
                ],
                "metric_definitions": {
                    "delta_rmse_c": (
                        "candidate RMSE minus physical-projection baseline RMSE in degrees Celsius; "
                        "negative means improvement"
                    ),
                    "ci90_lower_c": "lower endpoint of paired KST-day bootstrap 90% interval",
                    "ci90_upper_c": "upper endpoint of paired KST-day bootstrap 90% interval",
                    "kst_days": "number of KST calendar-day resampling units",
                },
            }
        },
        "valueFormat": "number",
        "unit": "°C",
        "layout": "full",
        "maxRows": 4,
        "settings": {
            "orientation": "horizontal",
            "groupMode": "single",
            "showValues": True,
        },
        "referenceLines": [
            {
                "axis": "y",
                "value": 0,
                "label": "no change",
                "color": "neutral",
                "lineStyle": "solid",
            }
        ],
        "encodings": {
            "x": {"field": "scope", "type": "nominal", "label": "Validation scope"},
            "y": {
                "field": "delta_rmse_c",
                "type": "quantitative",
                "label": "ΔRMSE (°C)",
                "format": "number",
            },
            "color": {"field": "direction", "type": "nominal", "label": "Direction"},
            "label": {
                "field": "signed_delta_label",
                "type": "nominal",
                "label": "Signed ΔRMSE",
            },
            "tooltip": [
                {"field": "ci90_lower_c", "type": "quantitative", "label": "CI90 lower", "unit": "°C"},
                {"field": "ci90_upper_c", "type": "quantitative", "label": "CI90 upper", "unit": "°C"},
                {"field": "kst_days", "type": "quantitative", "label": "KST days"},
                {"field": "rows", "type": "quantitative", "label": "Rows"},
                {
                    "field": "baseline_rmse_c",
                    "type": "quantitative",
                    "label": "Baseline RMSE",
                    "unit": "°C",
                },
                {
                    "field": "candidate_rmse_c",
                    "type": "quantitative",
                    "label": "Candidate RMSE",
                    "unit": "°C",
                },
            ],
        },
    }
    table = {
        "id": "trust_matrix",
        "title": "Validation trust and permitted action",
        "subtitle": "Freeze = immutable baseline/risk control, not validation endorsement",
        "dataset": "trust_matrix",
        "sourceId": "method_note",
        "source": {
            "label": "P1–P3 final aggregate validation audits",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": trust_table_sql,
                "description": (
                    "Deterministic field-level trust synthesis from three SHA-pinned aggregate audits."
                ),
                "tables_used": [
                    RELATIVE_PATHS["p1_audit"].as_posix(),
                    RELATIVE_PATHS["p2_audit"].as_posix(),
                    RELATIVE_PATHS["p3_audit"].as_posix(),
                ],
                "filters": [
                    "aggregate audit fields only",
                    "raw/OOF/test/submission rows = 0",
                    "problem order P1, P2, P3",
                ],
                "metric_definitions": {
                    "implementation_integrity": "reproduction, key/grain, split, and leakage-boundary trust",
                    "local_relative_ranking": "relative candidate ordering on the audited local surface",
                    "hidden_calibration": "support for mapping a local absolute metric to official hidden performance",
                    "adaptive_exposure": "lower-bound evidence of repeated use of the same labeled validation surface",
                    "action": "operational decision permitted by the evidence, not a hidden-score endorsement",
                },
            },
        },
        "density": "spacious",
        "defaultSort": {"field": "problem", "direction": "asc"},
        "columns": [
            {"field": "problem", "label": "문제", "type": "text"},
            {
                "field": "implementation_integrity",
                "label": "Implementation integrity",
                "type": "text",
            },
            {"field": "local_relative_ranking", "label": "Local relative ranking", "type": "text"},
            {"field": "hidden_calibration", "label": "Hidden calibration", "type": "text"},
            {"field": "adaptive_exposure", "label": "Adaptive exposure", "type": "text"},
            {"field": "action", "label": "Action", "type": "text"},
        ],
    }
    dataset = [
        {
            "sequence": index,
            **{
                key: row[key]
                for key in (
                    "problem",
                    "implementation_integrity",
                    "local_relative_ranking",
                    "hidden_calibration",
                    "adaptive_exposure",
                    "action",
                )
            },
        }
        for index, row in enumerate(normalized, start=1)
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "P1–P3 validation-system correction for a technical audience",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [],
            "charts": [chart],
            "tables": [table],
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"trust_matrix": dataset, "p2_delta_rmse": p2_chart_rows},
            "accessIssues": [],
        },
        "sources": [
            {
                "id": source["id"],
                "label": source["label"],
                **({"path": source["path"]} if "path" in source else {}),
                **({"sha256": source["sha256"]} if "sha256" in source else {}),
            }
            for source in sources
        ],
        "package_info": {
            "originUrl": f"artifact://{REPORT_ID}",
            "controls": {"edit": False, "refresh": False},
            "delivery": "technical portable HTML fallback",
        },
    }
    _validate_artifact(artifact)
    return artifact


def _validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact["surface"] == "report", "surface drifted")
    manifest = artifact["manifest"]
    _require(manifest["blocks"][0]["body"] == f"# {REPORT_TITLE}", "title drifted")
    _require(
        manifest["blocks"][1]["body"].startswith("## Technical Summary"),
        "technical summary order drifted",
    )
    _require(len(manifest["charts"]) == 1, "exactly one within-P2 chart is required")
    chart = manifest["charts"][0]
    _require(chart["id"] == "p2_delta_rmse", "cross-problem or unexpected chart forbidden")
    _require(chart["type"] == "horizontalBar", "P2 chart type drifted")
    _require(chart["sourceId"] == "p2_validation_audit", "P2 chart source drifted")
    _require(chart["settings"]["showValues"] is True, "signed bar values must be shown")
    _require(
        chart["referenceLines"]
        == [
            {
                "axis": "y",
                "value": 0,
                "label": "no change",
                "color": "neutral",
                "lineStyle": "solid",
            }
        ],
        "neutral zero reference drifted",
    )
    chart_rows = artifact["snapshot"]["datasets"]["p2_delta_rmse"]
    _require(len(chart_rows) == 4, "P2 chart must contain pooled plus three blocks")
    _require(
        [row["scope"] for row in chart_rows]
        == ["Pooled", "2024 Sep–Oct", "2025 Jul–Aug", "2025 Nov–Dec"],
        "P2 chart scope order drifted",
    )
    _require(
        all(
            isinstance(row["ci90_lower_c"], float)
            and isinstance(row["ci90_upper_c"], float)
            and int(row["kst_days"]) > 0
            and row["signed_delta_label"].startswith(("+", "-"))
            for row in chart_rows
        ),
        "P2 chart uncertainty/context fields drifted",
    )
    _require(len(manifest["tables"]) == 1, "trust matrix count drifted")
    rows = artifact["snapshot"]["datasets"]["trust_matrix"]
    _require(len(rows) == 3, "trust matrix must contain P1/P2/P3")
    _require([row["problem"] for row in rows] == ["P1", "P2", "P3"], "problem order drifted")
    columns = [column["field"] for column in manifest["tables"][0]["columns"]]
    _require(
        columns
        == [
            "problem",
            "implementation_integrity",
            "local_relative_ranking",
            "hidden_calibration",
            "adaptive_exposure",
            "action",
        ],
        "trust matrix column contract drifted",
    )
    serialized = json.dumps(artifact, ensure_ascii=False)
    for phrase in (
        "freeze = immutable baseline/risk control, not validation endorsement",
        "명시적으로 대체",
        "정답 확정",
        "MCP report tools were unavailable",
        "technical portable HTML fallback",
        "T=0.624165",
    ):
        _require(phrase in serialized, f"required correction phrase missing: {phrase}")
    required_roles = (
        "## Technical Summary",
        "## Key Findings",
        "## Scope, Data, and Metric Definitions",
        "## Methodology",
        "## Limitations, Uncertainty, and Robustness Checks",
        "## Recommended Next Steps",
        "## Further Questions",
    )
    for heading in required_roles:
        _require(heading in serialized, f"technical report role missing: {heading}")
    for forbidden in (
        "C:/Users/",
        "C:\\Users\\",
        ".parquet",
        ".csv",
        "test_context",
        "test_index",
        "submission.csv",
    ):
        _require(forbidden not in serialized, f"forbidden source reference leaked: {forbidden}")
    source_ids = {source["id"] for source in manifest["sources"]}
    _require(len(source_ids) == len(manifest["sources"]), "duplicate source id")
    for block in manifest["blocks"]:
        if "sourceId" in block:
            _require(block["sourceId"] in source_ids, f"block source missing: {block['id']}")
    _require(manifest["tables"][0]["sourceId"] in source_ids, "table source missing")
    _require(manifest["charts"][0]["sourceId"] in source_ids, "chart source missing")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate pinned inputs and the complete artifact without writing output.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    expected = (root / DEFAULT_OUTPUT).resolve()
    _require(output.resolve() == expected, f"output is frozen at {DEFAULT_OUTPUT.as_posix()}")
    evidence = collect_evidence(root)
    generated_at = args.generated_at or datetime.now(KST).isoformat()
    artifact = build_artifact(evidence, generated_at=generated_at)
    if args.check_only:
        print("PASS: validated pinned aggregate inputs and complete technical report artifact")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(f"PASS: wrote canonical aggregate report artifact to {DEFAULT_OUTPUT.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
