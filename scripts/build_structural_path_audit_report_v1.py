"""Build the aggregate-only structural-path audit report.

This append-only report generation reads only SHA-pinned aggregate JSON
receipts.  It never opens raw observations, row-level OOF predictions, test
contexts, test indices, or submission values.  The portable HTML packaging
step is intentionally separate from this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
REPORT_TITLE = "세 문제 구조적 해법 검증 감사"
DEFAULT_OUTPUT = Path("reports/generated/structural_path_audit_2026-08-22_r1/artifact.json")

# collect_evidence() fails closed while any pin is not an exact SHA-256.
PENDING_P3_V3_SHA256 = "PENDING_P3_V3_RESULT_SHA256"
EXPECTED_SHA256 = {
    "p1_factorial_diagnostic": ("5ac69957e8b3d0a04fb9fdda9cb7f5ba514eef94a806e8e315aa41e1d5e10ec3"),
    "p1_factorial_result": ("7eaadb615c9015deca9beb8b8d22995de74582c5697fced85f444316f9a29a89"),
    "p1_duration_result": ("357059936f6eb123af177a555d90f1a4ce4d2aeddd0de6d271e346af8fb573b3"),
    "p2_denominator_diagnostic": (
        "c858e3ea01b29e19d5eb60f65561439158050ba6ec5a77b3a44454610b575078"
    ),
    "p2_sigmoid_v2_result": ("b541942005646cd070b47aa5855660d7998425a9fed9ecc50ab2a2c36de74bf8"),
    "p3_raw_shape_v1_result": ("2b5b548ae17f9a0487eb53015e600731d4a2f45f5d4c621a5a7eebdb7ece513d"),
    "p3_forcing_v2_precheck": ("4c4132359771dbe0f64e610b2894652ad9c066dc4ff4c96daba87a22f710acb2"),
    "p3_forcing_v2_result": ("e938df0e7f69372ae5a21b94671017ffe15c5d0894e3ab0d8a0dc80027eef3a4"),
    "p3_predictive_v3_result": ("2571a8819650c02039a4c93a57004d8e99708914f2129599a5df0c2b4017160b"),
    "p3_predictive_v3_manifest": (
        "a662ed011c02d82377ddfd9b990533d0855062489ea318751a5670d206b1a632"
    ),
    "p3_outer_v4_qa_go": ("f0f5e018faf311260294fed02f642f3eaec01394bf731ad94cdc083f0e6316f6"),
    "p3_outer_v4_result": ("229617694e174e6a00a8d146c4b2bdec921692704e9330179fbf744aaf356323"),
    "p3_outer_v4_manifest": ("18c9ba775db8c0c0c5eb9a9b3bf40f7e0678164d86f020e92a91bd3242c511d2"),
}

RELATIVE_PATHS = {
    "p1_factorial_diagnostic": Path(
        "artifacts/p1_typed_factorial_semimarkov_v1/grammar_identifiability_diagnostic.json"
    ),
    "p1_factorial_result": Path("artifacts/p1_typed_factorial_semimarkov_v1/result.json"),
    "p1_duration_result": Path("artifacts/p1_typed_duration_semimarkov_v2/result.json"),
    "p2_denominator_diagnostic": Path(
        "artifacts/p2_dynamic_sigmoid_profile_v1_support_denominator_audit/diagnostic.json"
    ),
    "p2_sigmoid_v2_result": Path(
        "artifacts/p2_dynamic_sigmoid_profile_v2_key_aligned_precheck_20260822/precheck.json"
    ),
    "p3_raw_shape_v1_result": Path(
        "artifacts/p3_episode_distinct_raw_shape_analog_v1/inner_one_shot/result.json"
    ),
    "p3_forcing_v2_precheck": Path(
        "artifacts/p3_causal_forcing_conditioned_episode_analog_v2/B_one_shot/b_precheck.json"
    ),
    "p3_forcing_v2_result": Path(
        "artifacts/p3_causal_forcing_conditioned_episode_analog_v2/B_one_shot/result.json"
    ),
    "p3_predictive_v3_result": Path(
        "artifacts/p3_causal_forcing_analog_inner_predictive_v3/C_one_shot/result.json"
    ),
    "p3_predictive_v3_manifest": Path(
        "artifacts/p3_causal_forcing_analog_inner_predictive_v3/C_one_shot/manifest.json"
    ),
    "p3_outer_v4_qa_go": Path("artifacts/p3_causal_forcing_analog_outer_research_v4/qa/QA_GO.json"),
    "p3_outer_v4_result": Path(
        "artifacts/p3_causal_forcing_analog_outer_research_v4/outer_one_shot/result.json"
    ),
    "p3_outer_v4_manifest": Path(
        "artifacts/p3_causal_forcing_analog_outer_research_v4/outer_one_shot/manifest.json"
    ),
}

OFFICIAL_P3_URL = "https://oceanaidata.org/app/problems/7"
OFFICIAL_LEADERBOARD_URL = "https://oceanaidata.org/app/leaderboard"


class ReportEvidenceError(RuntimeError):
    """Raised when aggregate evidence no longer matches the sealed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportEvidenceError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"aggregate JSON must be an object: {path}")
    return payload


def _aggregate_input_contract() -> None:
    _require(set(RELATIVE_PATHS) == set(EXPECTED_SHA256), "evidence pin set drifted")
    for name, relative in RELATIVE_PATHS.items():
        _require(not relative.is_absolute(), f"absolute evidence path forbidden: {name}")
        _require(relative.suffix.lower() == ".json", f"non-JSON input forbidden: {name}")
        _require(relative.parts[0] == "artifacts", f"non-artifact input forbidden: {name}")
        lowered = relative.as_posix().lower()
        for token in (".parquet", ".csv", "test_context", "test_index", "submission"):
            _require(token not in lowered, f"forbidden input class in {name}: {token}")


def collect_evidence(root: Path) -> dict[str, Any]:
    """Read only pinned aggregate JSON receipts and fail closed on drift."""

    _aggregate_input_contract()
    evidence: dict[str, Any] = {}
    for name, relative in RELATIVE_PATHS.items():
        expected = EXPECTED_SHA256[name]
        _require(
            re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            f"unsealed SHA-256 pin: {name}",
        )
        resolved = root / relative
        _require(resolved.is_file(), f"missing aggregate evidence: {relative}")
        actual = _sha256(resolved)
        _require(actual == expected, f"sealed SHA mismatch for {name}: {actual} != {expected}")
        evidence[name] = _read_json(resolved)
    evidence["hashes"] = dict(EXPECTED_SHA256)
    _validate_evidence(evidence)
    return evidence


def _all_zero(mapping: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return all(int(mapping.get(key, -1)) == 0 for key in keys)


def _validate_evidence(evidence: dict[str, Any]) -> None:
    p1_v1_diag = evidence["p1_factorial_diagnostic"]
    p1_v1 = evidence["p1_factorial_result"]
    p1_v2 = evidence["p1_duration_result"]
    p2_diag = evidence["p2_denominator_diagnostic"]
    p2_v2 = evidence["p2_sigmoid_v2_result"]
    p3_v1 = evidence["p3_raw_shape_v1_result"]
    p3_v2_precheck = evidence["p3_forcing_v2_precheck"]
    p3_v2 = evidence["p3_forcing_v2_result"]
    p3_v3 = evidence["p3_predictive_v3_result"]
    p3_v3_manifest = evidence["p3_predictive_v3_manifest"]
    p3_v4_qa = evidence["p3_outer_v4_qa_go"]
    p3_v4 = evidence["p3_outer_v4_result"]
    p3_v4_manifest = evidence["p3_outer_v4_manifest"]

    feasibility = p1_v1_diag["structural_feasibility"]
    _require(feasibility["original_rolling_split_identifiable"] is False, "P1 v1 ID drift")
    _require(
        p1_v1_diag["expanding_fit_scopes"]["fit_1_through_2024_may23"]["composite_super_events"]
        == 0,
        "P1 v1 early composite support drifted",
    )
    _require(p1_v1["decision"] == "NO_GO_PRECHECK", "P1 v1 decision drifted")
    _require(p1_v1["inner_blocks_executed"] == 0, "P1 v1 inner score unexpectedly ran")
    _require(
        _all_zero(
            p1_v1,
            (
                "outer_validation_or_scoring_count",
                "test_prediction_count",
                "submission_generation_count",
                "upload_count",
            ),
        ),
        "P1 v1 safety counter drifted",
    )
    _require(
        _all_zero(
            p1_v1_diag["operation_counts"],
            (
                "inner_score",
                "model_fit",
                "outer_score",
                "prediction",
                "test_read",
                "submission",
                "upload",
            ),
        ),
        "P1 v1 diagnostic operation counter drifted",
    )

    _require(p1_v2["decision"] == "NO_GO_EXACT_CONFIGURATION", "P1 v2 decision drifted")
    _require(p1_v2["aggregate"]["gate_passed"] is False, "P1 v2 gate drifted")
    _require(p1_v2["inner_blocks_executed"] == 2, "P1 v2 block count drifted")
    _require(
        _all_zero(
            p1_v2,
            ("outer_score_count", "test_prediction_count", "submission_count", "upload_count"),
        ),
        "P1 v2 safety counter drifted",
    )

    _require(p2_diag["aggregate_only"] is True, "P2 denominator diagnostic is not aggregate-only")
    _require(
        p2_diag["decision"]["v1_failure_is_denominator_artifact"] is True,
        "P2 denominator decision drifted",
    )
    _require(
        _all_zero(
            p2_diag["forbidden_operations"],
            (
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
        ),
        "P2 denominator diagnostic safety counter drifted",
    )
    _require(p2_v2["aggregate_only"] is True, "P2 v2 is not aggregate-only")
    _require(p2_v2["gates_1_to_4"]["pass"] is True, "P2 v2 gates 1-4 drifted")
    p2_gate5 = p2_v2["gate_5_inner_only"]
    _require(p2_gate5["executed"] is True, "P2 v2 Gate 5 was not executed")
    _require(p2_gate5["all_outer_inner_gates_pass"] is False, "P2 v2 Gate 5 drifted")
    _require(
        sum(bool(fold["pass"]) for fold in p2_gate5["folds"].values()) == 0,
        "P2 v2 expected Gate 5 pass count is 0/3",
    )
    _require(
        _all_zero(
            p2_v2["forbidden_operations"],
            (
                "hidden_target_value_reads",
                "outer_truth_scores",
                "outer_prediction_rows",
                "test_index_reads",
                "submission_rows",
                "upload_attempts",
            ),
        ),
        "P2 v2 safety counter drifted",
    )

    _require(
        p3_v1["decision"] == "NO_GO_B_NEAREST_NOT_BETTER_THAN_RANDOM",
        "P3 v1 decision drifted",
    )
    _require(p3_v1["b_gate"]["passing_folds"] == 1, "P3 v1 expected 1/3 B folds")
    _require(p3_v1["c_gate_executed"] is False, "P3 v1 C gate unexpectedly ran")
    _require(
        _all_zero(
            p3_v1,
            (
                "model_fit_count",
                "outer_membership_read_count",
                "outer_designated_scoring_open_count",
                "test_context_read_count",
                "submission_write_count",
            ),
        ),
        "P3 v1 safety counter drifted",
    )
    _require(p3_v2["decision"] == "PASS_B_ADAPTIVE_INNER_ONLY_STOP", "P3 v2 decision drifted")
    _require(p3_v2["B_gate"]["passing_folds"] == 2, "P3 v2 expected 2/3 B folds")
    _require(p3_v2["adaptive_research"] is True, "P3 v2 adaptive disclosure drifted")
    _require(p3_v2["independent_confirmation"] is False, "P3 v2 independence drifted")
    conditioned_cases = sum(
        int(fold["forcing_conditioned_queries"]) for fold in p3_v2_precheck["search"].values()
    )
    eligible_cases = sum(
        int(fold["eligible_queries"]) for fold in p3_v2_precheck["search"].values()
    )
    _require(conditioned_cases == 19 and eligible_cases == 57, "P3 v2 coverage drifted")
    _require(
        _all_zero(
            p3_v2,
            (
                "model_fit_count",
                "C_execution_count",
                "outer_membership_read_count",
                "outer_designated_scoring_open_count",
                "test_context_read_count",
                "submission_write_count",
            ),
        ),
        "P3 v2 safety counter drifted",
    )

    _require(
        p3_v3["decision"] in {"PASS_C_ADAPTIVE_INNER_STOP_BEFORE_OUTER", "NO_GO_C_INNER_GATE"},
        "P3 v3 decision is not final",
    )
    _require(p3_v3["adaptive_research"] is True, "P3 v3 adaptive disclosure drifted")
    _require(p3_v3["independent_confirmation"] is False, "P3 v3 independence drifted")
    _require(p3_v3["model_fit_count"] == 3, "P3 v3 model fit count drifted")
    _require(
        bool(p3_v3["C_gate"]["pass"]) == p3_v3["decision"].startswith("PASS"),
        "P3 v3 decision/gate mismatch",
    )
    _require(
        _all_zero(
            p3_v3,
            (
                "outer_membership_read_count",
                "outer_designated_scoring_open_count",
                "test_context_read_count",
                "submission_write_count",
            ),
        ),
        "P3 v3 safety counter drifted",
    )
    _require(p3_v3_manifest["decision"] == p3_v3["decision"], "P3 v3 manifest decision drifted")
    _require(p3_v3_manifest["same_split_reused"] is True, "P3 v3 same-split disclosure drifted")
    _require(
        p3_v3_manifest["independent_confirmation"] is False,
        "P3 v3 manifest independence drifted",
    )
    _require(
        _all_zero(
            p3_v3_manifest,
            (
                "outer_membership_or_target_read_count",
                "outer_designated_scoring_open_count",
                "test_context_read_count",
                "submission_write_count",
            ),
        ),
        "P3 v3 manifest safety counter drifted",
    )
    result_hashes = [
        value
        for key, value in p3_v3_manifest["output_sha256"].items()
        if str(key).replace("\\", "/").endswith("/result.json")
    ]
    _require(
        result_hashes == [EXPECTED_SHA256["p3_predictive_v3_result"]],
        "P3 v3 manifest/result hash binding drifted",
    )

    _require(
        p3_v3["decision"] == "PASS_C_ADAPTIVE_INNER_STOP_BEFORE_OUTER",
        "P3 v4 could not exist after a failed v3 C gate",
    )
    _require(p3_v4_qa["decision"] == "QA_GO_OUTER_V4", "P3 v4 QA decision drifted")
    _require(p3_v4_qa["P0_finding_count"] == 0, "P3 v4 QA P0 count drifted")
    _require(p3_v4_qa["P1_finding_count"] == 0, "P3 v4 QA P1 count drifted")
    _require(
        _all_zero(
            p3_v4_qa,
            (
                "outer_model_execution_count",
                "incumbent_prediction_read_count",
                "designated_target_read_count",
                "test_context_read_count",
                "submission_write_count",
            ),
        ),
        "P3 v4 pre-run QA safety counter drifted",
    )
    _require(p3_v4["decision"] == "NO_GO_KEEP_FROZEN_INCUMBENT", "P3 v4 decision drifted")
    _require(p3_v4["outer_gate"]["pass"] is False, "P3 v4 gate drifted")
    _require(
        p3_v4["required_action"] == "permanent_stop_keep_frozen_incumbent", "P3 v4 action drifted"
    )
    _require(p3_v4["promotion_performed"] is False, "P3 v4 unexpectedly promoted")
    _require(p3_v4["rerun_prohibited"] is True, "P3 v4 rerun contract drifted")
    _require(p3_v4["outer_key_membership_read_count"] == 1, "P3 v4 key read count drifted")
    _require(p3_v4["incumbent_prediction_read_count"] == 1, "P3 v4 incumbent read drifted")
    _require(p3_v4["designated_target_read_count"] == 1, "P3 v4 target read drifted")
    _require(
        _all_zero(
            p3_v4,
            (
                "model_fit_count",
                "test_context_read_count",
                "submission_write_count",
                "upload_count",
            ),
        ),
        "P3 v4 safety counter drifted",
    )
    _require(p3_v4["outer_gate"]["rows"] == 1092, "P3 v4 row grain drifted")
    _require(p3_v4["outer_gate"]["cases"] == 182, "P3 v4 case grain drifted")
    _require(p3_v4_manifest["decision"] == p3_v4["decision"], "P3 v4 manifest decision drifted")
    _require(
        p3_v4_manifest["QA_GO_receipt_sha256"] == EXPECTED_SHA256["p3_outer_v4_qa_go"],
        "P3 v4 QA binding drifted",
    )
    _require(p3_v4_manifest["promotion_performed"] is False, "P3 v4 manifest promotion drifted")
    _require(
        _all_zero(
            p3_v4_manifest,
            (
                "model_fit_count",
                "test_context_read_count",
                "submission_write_count",
                "upload_count",
            ),
        ),
        "P3 v4 manifest safety counter drifted",
    )
    outer_result_hashes = [
        value
        for key, value in p3_v4_manifest["output_sha256"].items()
        if str(key).replace("\\", "/").endswith("/result.json")
    ]
    _require(
        outer_result_hashes == [EXPECTED_SHA256["p3_outer_v4_result"]],
        "P3 v4 manifest/result hash binding drifted",
    )


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), "non-finite value cannot enter report SQL")
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _values_sql(rows: list[dict[str, object]], columns: list[str]) -> str:
    return " UNION ALL ".join(
        "SELECT " + ", ".join(f"{_sql_literal(row.get(column))} AS {column}" for column in columns)
        for row in rows
    )


def _source(
    *,
    source_id: str,
    label: str,
    path: str,
    rows: list[dict[str, object]],
    columns: list[str],
    tables_used: list[str],
    description: str,
    filters: list[str] | None = None,
    metric_definitions: dict[str, str] | None = None,
) -> dict[str, object]:
    query: dict[str, object] = {
        "engine": "sqlite",
        "language": "sql",
        "sql": _values_sql(rows, columns),
        "description": description,
        "tables_used": tables_used,
    }
    if filters:
        query["filters"] = filters
    if metric_definitions:
        query["metric_definitions"] = metric_definitions
    return {"id": source_id, "label": label, "path": path, "query": query}


def _chart_source(source: dict[str, object]) -> dict[str, object]:
    return {"query": source["query"]}


def _table(
    *,
    table_id: str,
    title: str,
    subtitle: str,
    rows: list[dict[str, object]],
    columns: list[dict[str, object]],
    source: dict[str, object],
) -> dict[str, object]:
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": table_id,
        "sourceId": source["id"],
        "source": _chart_source(source),
        "density": "spacious",
        "defaultSort": {"field": "sequence", "direction": "asc"},
        "columns": columns,
    }


def _block_label(name: str) -> str:
    return {
        "2024_sep_oct": "2024년 9–10월",
        "2025_jul_aug_61d": "2025년 7–8월",
        "2025_nov_dec": "2025년 11–12월",
        "2024_h2_storm": "2024년 하반기 폭풍",
        "2025_h1": "2025년 상반기",
        "winter_transition": "겨울 전이기",
    }[name]


def build_artifact(evidence: dict[str, Any], *, generated_at: str) -> dict[str, Any]:
    """Build one canonical report artifact from validated aggregate evidence."""

    _validate_evidence(evidence)
    p1_v1 = evidence["p1_factorial_result"]
    p1_v2 = evidence["p1_duration_result"]
    p2_diag = evidence["p2_denominator_diagnostic"]
    p2_v2 = evidence["p2_sigmoid_v2_result"]
    p3_v1 = evidence["p3_raw_shape_v1_result"]
    p3_v2_precheck = evidence["p3_forcing_v2_precheck"]
    p3_v2 = evidence["p3_forcing_v2_result"]
    p3_v3 = evidence["p3_predictive_v3_result"]
    p3_v4 = evidence["p3_outer_v4_result"]

    p2_support_rows: list[dict[str, object]] = []
    for name, block in p2_diag["blocks"].items():
        label = _block_label(name)
        p2_support_rows.extend(
            [
                {
                    "block": label,
                    "support_basis": "전체 61일 격자",
                    "support_share": float(block["full_grid_support_share"]),
                    "denominator_times": int(block["full_grid_times"]),
                    "supported_times": int(block["full_grid_supported_times"]),
                    "gate_threshold": 0.8,
                },
                {
                    "block": label,
                    "support_basis": "채점키 시각",
                    "support_share": float(block["key_aligned_support_share"]),
                    "denominator_times": int(block["unique_scored_times"]),
                    "supported_times": int(block["key_aligned_supported_times"]),
                    "gate_threshold": 0.8,
                },
            ]
        )

    p3_ratio_rows: list[dict[str, object]] = []
    for version, payload in (
        ("v1 원시 파형", p3_v1["b_gate"]),
        ("v2 강제력 조건", p3_v2["B_gate"]),
    ):
        for name, block in payload["by_fold"].items():
            p3_ratio_rows.append(
                {
                    "fold": _block_label(name),
                    "version": version,
                    "nearest_to_random_mse_ratio": float(block["ratio"]),
                    "eligible_cases": int(block["eligible_cases"]),
                    "gate_threshold": 0.9,
                    "fold_pass": bool(block["pass"]),
                }
            )

    p1_v2_agg = p1_v2["aggregate"]
    p2_gate5_folds = p2_v2["gate_5_inner_only"]["folds"]
    p3_c = p3_v3["C_gate"]
    p3_c_pass = bool(p3_c["pass"])
    p3_conditioned_cases = sum(
        int(fold["forcing_conditioned_queries"]) for fold in p3_v2_precheck["search"].values()
    )
    p3_eligible_cases = sum(
        int(fold["eligible_queries"]) for fold in p3_v2_precheck["search"].values()
    )
    p3_c_pass_margin_m = -0.005 - float(p3_c["pooled_delta_m"])
    p3_outer = p3_v4["outer_gate"]
    p3_c_status = "조건부 PASS — outer 미개방" if p3_c_pass else "NO-GO — exact 후보 종료"
    p3_c_implication = (
        "별도 승인된 독립 검증 전까지 연구 경로만 유지"
        if p3_c_pass
        else "재튜닝·재실행 없이 exact 후보를 영구 종료"
    )

    decision_rows = [
        {
            "sequence": 1,
            "problem": "P1",
            "generation": "typed factorial v1",
            "stage": "식별가능성 precheck",
            "evidence": "초기 expanding fit의 composite super-event 0건; inner 평가 0회",
            "threshold": "각 fold가 composite grammar를 식별 가능",
            "decision": "NO-GO_PRECHECK",
            "implication": "factorial decoder 성능 비교 자체를 열지 않음",
        },
        {
            "sequence": 2,
            "problem": "P1",
            "generation": "typed duration v2",
            "stage": "2개 historical inner block",
            "evidence": (
                f"micro F1 Δ {p1_v2_agg['micro_f1_delta']:+.6f}; "
                f"spike recall Δ {p1_v2_agg['recall_delta_by_type']['spike']:+.6f}; "
                f"worst group Δ {p1_v2_agg['worst_station_layer_f1_delta']:+.6f}"
            ),
            "threshold": "ΔF1≥+0.005 및 모든 안전 gate",
            "decision": "NO_GO_EXACT_CONFIGURATION",
            "implication": "duration decoder만으로는 unary 오류를 해결하지 못함",
        },
        {
            "sequence": 3,
            "problem": "P2",
            "generation": "dynamic sigmoid v1 진단",
            "stage": "Gate 1 분모 감사",
            "evidence": "2025년 11–12월 support 62.58%→97.62% (채점키 정렬)",
            "threshold": "support≥80%",
            "decision": "분모 artifact 확인",
            "implication": "단일 변경 v2 재검증은 정당화됨",
        },
        {
            "sequence": 4,
            "problem": "P2",
            "generation": "dynamic sigmoid v2",
            "stage": "Gates 1–5 inner-only",
            "evidence": "Gates 1–4 3/3 통과; Gate 5 0/3 통과",
            "threshold": "각 outer의 2-inner gate 모두 통과",
            "decision": "NO-GO_GATE5_0_OF_3",
            "implication": "프로파일 적합성은 residual 예측력의 충분조건이 아님",
        },
        {
            "sequence": 5,
            "problem": "P3",
            "generation": "raw-shape analog v1",
            "stage": "B 유사도 precheck",
            "evidence": f"통과 fold {p3_v1['b_gate']['passing_folds']}/3",
            "threshold": "nearest/random MSE≤0.90 in ≥2/3 folds",
            "decision": "NO_GO_B",
            "implication": "hs 파형만으로는 episode 이웃이 안정적이지 않음",
        },
        {
            "sequence": 6,
            "problem": "P3",
            "generation": "forcing-conditioned v2",
            "stage": "adaptive B precheck",
            "evidence": f"통과 fold {p3_v2['B_gate']['passing_folds']}/3; 독립 확인 아님",
            "threshold": "nearest/random MSE≤0.90 in ≥2/3 folds",
            "decision": "PASS_B_ADAPTIVE_INNER_ONLY_STOP",
            "implication": "C predictive gate 1회만 정당화",
        },
        {
            "sequence": 7,
            "problem": "P3",
            "generation": "forcing predictive v3",
            "stage": "adaptive C inner-only",
            "evidence": (
                f"pooled ΔRMSE {p3_c['pooled_delta_m']:+.6f}m; "
                f"improved folds {p3_c['improved_folds']}/3; "
                f"worst station Δ {p3_c['maximum_station_delta_m']:+.6f}m"
            ),
            "threshold": "pooled≤−0.005m, ≥2 folds, station≤+0.010m, +18/+24h 비악화",
            "decision": p3_c_status,
            "implication": p3_c_implication,
        },
        {
            "sequence": 8,
            "problem": "P3",
            "generation": "forcing outer v4",
            "stage": "sealed 182-case outer one-shot",
            "evidence": (
                f"pooled ΔRMSE {p3_outer['candidate_minus_incumbent_rmse_m']:+.6f}m; "
                "improved folds 0/3; "
                f"CI90 upper {p3_outer['paired_case_bootstrap']['ci90_upper_m']:+.6f}m"
            ),
            "threshold": "pooled≤−0.010m, CI90 upper<0, ≥2 folds, station/+18/+24h 안전",
            "decision": "NO_GO_KEEP_FROZEN_INCUMBENT",
            "implication": "exact forcing-analog 경로 영구 종료; 재튜닝·재실행 금지",
        },
    ]

    p1_factorial_rows = [
        {
            "decision": p1_v1["decision"],
            "original_split_identifiable": False,
            "early_fit_composite_events": 0,
            "inner_blocks_executed": int(p1_v1["inner_blocks_executed"]),
            "outer_scores": int(p1_v1["outer_validation_or_scoring_count"]),
        }
    ]
    p1_duration_rows = [
        {
            "decision": p1_v2["decision"],
            "micro_f1_delta": float(p1_v2_agg["micro_f1_delta"]),
            "spike_recall_delta": float(p1_v2_agg["recall_delta_by_type"]["spike"]),
            "offset_recall_delta": float(p1_v2_agg["recall_delta_by_type"]["offset"]),
            "worst_group_f1_delta": float(p1_v2_agg["worst_station_layer_f1_delta"]),
            "outer_scores": int(p1_v2["outer_score_count"]),
        }
    ]
    p2_gate_rows = [
        {
            "outer_fold": _block_label(name),
            "alpha": float(fold["alpha"]),
            "pooled_delta_rmse_c": float(fold["pooled_delta_rmse"]),
            "pass": bool(fold["pass"]),
        }
        for name, fold in p2_gate5_folds.items()
    ]
    p3_c_rows = [
        {
            "decision": p3_v3["decision"],
            "pass": p3_c_pass,
            "control_rmse_m": float(p3_c["control_rmse_m"]),
            "candidate_rmse_m": float(p3_c["candidate_rmse_m"]),
            "pooled_delta_m": float(p3_c["pooled_delta_m"]),
            "improved_folds": int(p3_c["improved_folds"]),
            "maximum_station_delta_m": float(p3_c["maximum_station_delta_m"]),
            "pass_margin_m": p3_c_pass_margin_m,
            "forcing_conditioned_cases": p3_conditioned_cases,
            "B_eligible_cases": p3_eligible_cases,
            "lead_18_non_degrading": bool(p3_c["checks"]["lead_18_non_degrading"]),
            "lead_24_non_degrading": bool(p3_c["checks"]["lead_24_non_degrading"]),
        }
    ]
    p3_outer_rows = [
        {
            "decision": p3_v4["decision"],
            "pass": bool(p3_outer["pass"]),
            "incumbent_rmse_m": float(p3_outer["incumbent_rmse_m"]),
            "candidate_rmse_m": float(p3_outer["candidate_rmse_m"]),
            "delta_rmse_m": float(p3_outer["candidate_minus_incumbent_rmse_m"]),
            "improved_folds": sum(
                float(fold["delta_m"]) < 0.0 for fold in p3_outer["by_fold"].values()
            ),
            "maximum_station_delta_m": max(
                float(station["delta_m"]) for station in p3_outer["by_station"].values()
            ),
            "lead_18_delta_m": float(p3_outer["by_lead"]["18"]["delta_m"]),
            "lead_24_delta_m": float(p3_outer["by_lead"]["24"]["delta_m"]),
            "ci90_lower_m": float(p3_outer["paired_case_bootstrap"]["ci90_lower_m"]),
            "ci90_upper_m": float(p3_outer["paired_case_bootstrap"]["ci90_upper_m"]),
            "cases": int(p3_outer["cases"]),
        }
    ]

    sources = [
        _source(
            source_id="p1_factorial_evidence",
            label="P1 typed-factorial 식별가능성 집계 영수증",
            path=RELATIVE_PATHS["p1_factorial_diagnostic"].as_posix(),
            rows=p1_factorial_rows,
            columns=list(p1_factorial_rows[0]),
            tables_used=[
                RELATIVE_PATHS["p1_factorial_diagnostic"].as_posix(),
                RELATIVE_PATHS["p1_factorial_result"].as_posix(),
            ],
            description="SHA-pinned aggregate grammar diagnostic and terminal precheck result only.",
            filters=["raw rows = 0", "inner scores = 0", "outer/test/submission operations = 0"],
            metric_definitions={
                "original_split_identifiable": "whether the original rolling fit scopes contain enough composite grammar support for the factorial comparison",
                "early_fit_composite_events": "composite super-events available to the earliest expanding fit",
            },
        ),
        _source(
            source_id="p1_duration_evidence",
            label="P1 typed-duration v2 집계 결과",
            path=RELATIVE_PATHS["p1_duration_result"].as_posix(),
            rows=p1_duration_rows,
            columns=list(p1_duration_rows[0]),
            tables_used=[RELATIVE_PATHS["p1_duration_result"].as_posix()],
            description="SHA-pinned aggregate two-block inner result; no row-level prediction values.",
            filters=["outer score count = 0", "test prediction count = 0", "upload count = 0"],
            metric_definitions={
                "micro_f1_delta": "duration decoder candidate minus same-unary rowwise-union control micro F1",
                "spike_recall_delta": "candidate minus control spike recall",
                "worst_group_f1_delta": "minimum station-layer candidate-minus-control F1",
            },
        ),
        _source(
            source_id="p2_denominator_evidence",
            label="P2 dynamic-sigmoid v1 support 분모 감사",
            path=RELATIVE_PATHS["p2_denominator_diagnostic"].as_posix(),
            rows=p2_support_rows,
            columns=list(p2_support_rows[0]),
            tables_used=[RELATIVE_PATHS["p2_denominator_diagnostic"].as_posix()],
            description="SHA-pinned aggregate support counts comparing full-grid and scored-key denominators.",
            filters=["public-layer support only", "truth/prediction/model reads = 0"],
            metric_definitions={
                "support_share": "timestamps with at least four public-temperature depths spanning at least 30 m divided by the stated timestamp denominator",
                "gate_threshold": "pre-registered Gate 1 minimum support share of 0.80",
            },
        ),
        _source(
            source_id="p2_sigmoid_v2_evidence",
            label="P2 dynamic-sigmoid v2 Gate 5 집계 결과",
            path=RELATIVE_PATHS["p2_sigmoid_v2_result"].as_posix(),
            rows=p2_gate_rows,
            columns=list(p2_gate_rows[0]),
            tables_used=[RELATIVE_PATHS["p2_sigmoid_v2_result"].as_posix()],
            description="SHA-pinned aggregate key-aligned precheck and inner-only Gate 5 result.",
            filters=["outer truth scores = 0", "hidden target reads = 0", "submission rows = 0"],
            metric_definitions={
                "pooled_delta_rmse_c": "sigmoid blend minus incumbent pooled RMSE over the two allowed inner blocks for each outer fold",
                "pass": "all fixed pooled, block, layer, and positive-alpha safeguards passed",
            },
        ),
        _source(
            source_id="p3_b_derivation",
            label="P3 raw-shape v1 및 forcing-conditioned v2 B gate",
            path=RELATIVE_PATHS["p3_forcing_v2_result"].as_posix(),
            rows=p3_ratio_rows,
            columns=list(p3_ratio_rows[0]),
            tables_used=[
                RELATIVE_PATHS["p3_raw_shape_v1_result"].as_posix(),
                RELATIVE_PATHS["p3_forcing_v2_precheck"].as_posix(),
                RELATIVE_PATHS["p3_forcing_v2_result"].as_posix(),
            ],
            description="Deterministic aggregate comparison of the two sealed B-gate receipts.",
            filters=["same three inner folds", "outer/test/submission operations = 0"],
            metric_definitions={
                "nearest_to_random_mse_ratio": "nearest distinct-episode future-residual normalized MSE divided by the matched random distinct-episode MSE; lower is better",
                "fold_pass": "ratio at or below 0.90",
            },
        ),
        _source(
            source_id="p3_predictive_v3_evidence",
            label="P3 forcing analog v3 predictive C gate",
            path=RELATIVE_PATHS["p3_predictive_v3_result"].as_posix(),
            rows=p3_c_rows,
            columns=list(p3_c_rows[0]),
            tables_used=[
                RELATIVE_PATHS["p3_predictive_v3_result"].as_posix(),
                RELATIVE_PATHS["p3_predictive_v3_manifest"].as_posix(),
            ],
            description="SHA-pinned aggregate adaptive inner-only predictive result; no outer/test/submission values.",
            filters=[
                "three fixed proxy fits",
                "outer membership read count = 0",
                "test context read count = 0",
            ],
            metric_definitions={
                "pooled_delta_m": "candidate minus fixed inner-control pooled six-lead RMSE in metres",
                "maximum_station_delta_m": "largest station-level candidate-minus-control RMSE delta",
                "pass_margin_m": "distance beyond the fixed -0.005 m pooled-delta threshold; positive means the gate cleared",
            },
        ),
        _source(
            source_id="p3_outer_v4_evidence",
            label="P3 forcing analog v4 outer one-shot",
            path=RELATIVE_PATHS["p3_outer_v4_result"].as_posix(),
            rows=p3_outer_rows,
            columns=list(p3_outer_rows[0]),
            tables_used=[
                RELATIVE_PATHS["p3_outer_v4_qa_go"].as_posix(),
                RELATIVE_PATHS["p3_outer_v4_result"].as_posix(),
                RELATIVE_PATHS["p3_outer_v4_manifest"].as_posix(),
            ],
            description="SHA-pinned aggregate pre-run QA, one-shot outer result, and manifest; no row-level values.",
            filters=[
                "182 sealed research cases",
                "incumbent read once after component seal",
                "designated target read once after final seal and locks",
                "test/submission/upload operations = 0",
            ],
            metric_definitions={
                "delta_rmse_m": "forcing-analog candidate minus frozen incumbent pooled RMSE in metres",
                "ci90_upper_m": "upper endpoint of the fixed 5,000-replicate paired case-bootstrap 90% interval",
                "improved_folds": "number of the three fixed folds with negative candidate-minus-incumbent RMSE",
            },
        ),
        _source(
            source_id="official_context",
            label="공식 P3 문제 페이지 및 공개 리더보드 상태",
            path=OFFICIAL_P3_URL,
            rows=[
                {
                    "fact": "P3 T=0.624165",
                    "interpretation": "정책·점수 환산 상수이며 숨은 모델 점수가 아님",
                    "as_of_kst": "2026-08-22",
                },
                {
                    "fact": "대학부 공개 리더보드 8개 항목",
                    "interpretation": "모두 심사 중으로 표시되어 모델 비교 점수 없음",
                    "as_of_kst": "2026-08-22",
                },
            ],
            columns=["fact", "interpretation", "as_of_kst"],
            tables_used=[OFFICIAL_P3_URL, OFFICIAL_LEADERBOARD_URL],
            description="Official UI facts inspected read-only on August 22, 2026 KST.",
            filters=["university division", "visible public entries only", "snapshot not live"],
        ),
    ]

    decision_source = _source(
        source_id="decision_derivation",
        label="구조 검증 exact 판정 도출",
        path=RELATIVE_PATHS["p3_predictive_v3_result"].as_posix(),
        rows=decision_rows,
        columns=list(decision_rows[0]),
        tables_used=[relative.as_posix() for relative in RELATIVE_PATHS.values()],
        description=(
            f"Deterministic decision table over {len(RELATIVE_PATHS)} exact SHA-pinned aggregate JSON artifacts."
        ),
        filters=[
            "raw observations read = 0",
            "row-level OOF values read = 0",
            "test values read = 0",
            "submission values read = 0",
        ],
        metric_definitions={
            "decision": "terminal or conditional action stated by the corresponding pre-registered gate",
            "evidence": "aggregate-only observation used to apply that gate",
        },
    )
    sources.append(decision_source)

    delivery_note_rows = [
        {
            "delivery_mode": "portable HTML",
            "audience": "product stakeholders",
            "fallback_reason": "MCP report tools unavailable in the current runtime tool inventory",
            "visual_1": "grouped bar: P2 full-grid versus scored-key support with 0.80 reference",
            "visual_2": "grouped bar: P3 v1 versus v2 fold B ratios with 0.90 reference",
            "table": "exact eight-row structural decision audit",
            "omitted_visuals": (
                "P1 heterogeneous safety gates and P3 outer single candidate-versus-incumbent lookup remain in "
                "the exact decision table to avoid duplicative or mixed-scale charts"
            ),
        }
    ]
    delivery_source = _source(
        source_id="report_delivery_note",
        label="보고서 전달 모드·차트 맵 source note",
        path="scripts/build_structural_path_audit_report_v1.py",
        rows=delivery_note_rows,
        columns=list(delivery_note_rows[0]),
        tables_used=["scripts/build_structural_path_audit_report_v1.py"],
        description=(
            "One delivery mode selected: portable HTML fallback because MCP report tools were unavailable. "
            "Audience contract: product stakeholders / executive. The two grouped bars use spatial grouping, "
            "legends, exact tooltips, and neutral threshold lines so interpretation does not rely on color alone."
        ),
        filters=["append-only generation", "no prior report or builder mutated"],
    )
    sources.append(delivery_source)

    p2_chart = {
        "id": "p2_support_basis",
        "title": "P2 공개 프로파일 support",
        "subtitle": "세 고정 61일 블록; Gate 1 기준 80%; 비율이 높을수록 좋음",
        "showDescription": True,
        "intent": "comparison",
        "question": "v1의 support 실패가 모델 한계였는가, 분모 정의 오류였는가?",
        "rationale": "세 블록의 두 분모 정의를 같은 축에서 비교하는 grouped bar.",
        "type": "bar",
        "dataset": "p2_support_basis",
        "sourceId": "p2_denominator_evidence",
        "source": _chart_source(sources[2]),
        "valueFormat": "percent",
        "layout": "full",
        "settings": {"groupMode": "grouped", "showValues": True},
        "referenceLines": [
            {
                "axis": "y",
                "value": 0.8,
                "label": "Gate 1 80%",
                "color": "neutral",
                "lineStyle": "dashed",
            }
        ],
        "encodings": {
            "x": {"field": "block", "type": "nominal", "label": "검증 블록"},
            "y": {
                "field": "support_share",
                "type": "quantitative",
                "label": "support 비율",
                "format": "percent",
            },
            "color": {"field": "support_basis", "type": "nominal", "label": "분모"},
            "tooltip": [
                {"field": "supported_times", "type": "quantitative", "label": "지원 시각"},
                {"field": "denominator_times", "type": "quantitative", "label": "분모 시각"},
            ],
        },
    }

    p3_chart = {
        "id": "p3_b_fold_ratios",
        "title": "P3 fold별 nearest/random MSE 비율",
        "subtitle": "동일 세 inner fold; 90% 이하면 해당 fold 통과; 낮을수록 좋음",
        "showDescription": True,
        "intent": "comparison",
        "question": "과거 hs 파형에 causal forcing을 더하면 episode 이웃의 미래 유사성이 안정화되는가?",
        "rationale": "v1/v2와 고정 임계값을 fold별로 직접 비교하는 grouped bar.",
        "type": "bar",
        "dataset": "p3_b_fold_ratios",
        "sourceId": "p3_b_derivation",
        "source": _chart_source(sources[4]),
        "valueFormat": "percent",
        "layout": "full",
        "settings": {"groupMode": "grouped", "showValues": True},
        "referenceLines": [
            {
                "axis": "y",
                "value": 0.9,
                "label": "B gate 90%",
                "color": "neutral",
                "lineStyle": "dashed",
            }
        ],
        "encodings": {
            "x": {"field": "fold", "type": "nominal", "label": "inner fold"},
            "y": {
                "field": "nearest_to_random_mse_ratio",
                "type": "quantitative",
                "label": "nearest / random MSE",
                "format": "percent",
            },
            "color": {"field": "version", "type": "nominal", "label": "구조"},
            "tooltip": [
                {"field": "eligible_cases", "type": "quantitative", "label": "eligible 사례"},
                {"field": "fold_pass", "type": "nominal", "label": "fold 통과"},
            ],
        },
    }

    decision_table = _table(
        table_id="exact_decision_table",
        title="구조 검증 exact 판정표",
        subtitle="공식 hidden score가 아니라 각 generation의 사전 고정 inner/precheck 기준",
        rows=decision_rows,
        source=decision_source,
        columns=[
            {"field": "sequence", "label": "#", "type": "number"},
            {"field": "problem", "label": "문제", "type": "text"},
            {"field": "generation", "label": "세대", "type": "text"},
            {"field": "stage", "label": "검증 단계", "type": "text"},
            {"field": "evidence", "label": "관측", "type": "text"},
            {"field": "threshold", "label": "기준", "type": "text"},
            {"field": "decision", "label": "판정", "type": "text"},
            {"field": "implication", "label": "의미", "type": "text"},
        ],
    )

    p3_v3_summary = "v3 predictive gate는 통과했지만 sealed outer one-shot에서 incumbent보다 악화해 exact 경로가 종료됐습니다."
    p3_v3_detail = (
        "**당시 판정:** 구조 precheck와 predictive inner gate의 연속 통과는 봉인된 outer one-shot 1회를 "
        "정당화했을 뿐, 모델 승격이나 제출 변경을 정당화하지는 않았습니다."
    )

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "sourceId": "decision_derivation",
            "body": (
                "## Executive Summary\n\n"
                "- **길은 있었지만, 통과 기준을 충족한 길은 거의 없었습니다.** 새 구조를 실제로 구현해 보니 P1은 "
                "식별가능성과 안전 gate에서, P2는 residual 예측 gate에서 멈췄습니다.\n"
                "- **P2의 첫 실패 원인 하나는 모델이 아니라 분모였습니다.** 채점키 정렬로 support 오류를 고쳤지만, "
                "수정한 v2도 Gate 5를 통과하지 못해 현재 sigmoid 구조는 승격하지 않습니다.\n"
                "- **P3만 구조적 신호가 단계적으로 살아났습니다.** raw-shape v1은 B gate 1/3, forcing v2는 adaptive "
                f"2/3이었고, {p3_v3_summary}\n"
                "- **현재 동결 제출은 바꾸지 않습니다.** 모든 결론은 aggregate-only inner/precheck 증거이며 outer, test, "
                "submission 값은 보고서 입력에 사용하지 않았습니다."
            ),
        },
        {
            "id": "why_it_looked_stuck",
            "type": "markdown",
            "sourceId": "official_context",
            "body": (
                "## 왜 ‘길을 못 찾은 것처럼’ 보였는가\n\n"
                "**공식 피드백이 아직 없고, 실패를 조기에 멈추는 검증 계약을 지켰기 때문입니다.** 2026년 8월 22일 "
                "KST에 확인한 대학부 공개 리더보드는 보이는 8개 항목이 모두 ‘심사 중’이라 모델별 공식 비교 점수가 "
                "없었습니다. P3의 `T=0.624165`도 숨은 우수 모델 점수가 아니라 문제지에 고정된 정책·점수 환산 "
                "상수입니다. 따라서 로컬 결과가 그 값보다 높다는 이유만으로 접근이 틀렸다고 판단할 근거는 없습니다.\n\n"
                "이번 감사는 headline 하나가 좋아 보여도 식별가능성, 분모, fold 안정성, 비악화 조건이 깨지면 다음 "
                "단계를 열지 않았습니다. 이것이 탐색 속도는 느려 보여도 hidden 결과에 맞춘 사후 튜닝을 막는 이유입니다."
            ),
        },
        {
            "id": "definitions",
            "type": "markdown",
            "sourceId": "decision_derivation",
            "body": (
                "## 같은 ‘통과’라도 검증 단계가 다릅니다\n\n"
                "- **P1:** 같은 unary 확률을 둔 decoder 비교입니다. F1은 높을수록 좋지만, pooled 개선뿐 아니라 유형별 "
                "recall과 최악 station-layer 손실을 함께 제한했습니다.\n"
                "- **P2:** 공식 목적은 temperature RMSE 최소화입니다. Gates 1–4는 sigmoid 적합·관측가능 조건이고, "
                "Gate 5가 incumbent보다 실제 inner RMSE를 낮추는지를 판정합니다.\n"
                "- **P3 B gate:** nearest/random normalized future-residual MSE 비율이 0.90 이하인 fold가 2개 이상이어야 "
                "합니다. **C gate:** six-lead pooled RMSE Δ가 −0.005m 이하이고 fold·station·+18h·+24h 안전 조건을 "
                "동시에 만족해야 합니다.\n\n"
                "각 수치는 2024–2025 historical 또는 inner block의 로컬 근거이며 공식 hidden score가 아닙니다."
            ),
        },
        {
            "id": "p1_finding",
            "type": "markdown",
            "sourceId": "p1_duration_evidence",
            "body": (
                "## P1 — decoder를 복잡하게 해도 unary 정보 부족은 해결되지 않았습니다\n\n"
                "Factorial v1의 원래 rolling fit 첫 구간에는 composite super-event가 0건이어서 overlap grammar를 "
                "식별할 수 없었습니다. 그래서 inner score를 열지 않고 종료했습니다. Composite 의존성을 제거한 duration "
                f"v2는 두 block 모두 F1이 소폭 올랐지만 pooled ΔF1은 {p1_v2_agg['micro_f1_delta']:+.6f}로 "
                f"+0.005 기준에 못 미쳤습니다. Spike recall은 {p1_v2_agg['recall_delta_by_type']['spike']:+.6f}, "
                f"최악 station-layer F1은 {p1_v2_agg['worst_station_layer_f1_delta']:+.6f} 떨어졌습니다.\n\n"
                "**의미:** 남은 유망 경로는 decoder 재튜닝이 아니라 offset/drift를 정상 해양 변동과 구분하는 새로운 "
                "관측 신호 또는 unary 학습 구조입니다. 현재 typed decoder 두 세대는 닫고 동결 incumbent를 유지합니다."
            ),
        },
        {
            "id": "p2_finding",
            "type": "markdown",
            "sourceId": "p2_denominator_evidence",
            "body": (
                "## P2 — 분모 오류는 고쳤지만 sigmoid residual은 일반화되지 않았습니다\n\n"
                "2025년 11–12월 전체 61일 격자는 실제 채점키에 없는 3,153개 시각을 분모에 넣었습니다. 지원되는 "
                "시각은 하나도 제거되지 않은 채 support가 62.58%로 낮아졌고, 채점키 시각으로 정렬하면 97.62%입니다. "
                "아래 막대는 세 블록에서 두 분모를 같은 80% 기준과 비교합니다. **따라서 v1 Gate 1 실패는 분모 "
                "artifact였고 v2 재실행은 정당했습니다.**"
            ),
        },
        {"id": "p2_chart", "type": "chart", "chartId": "p2_support_basis"},
        {
            "id": "p2_interpretation",
            "type": "markdown",
            "sourceId": "p2_sigmoid_v2_evidence",
            "body": (
                "### 교정 후에도 예측 gate는 0/3이었습니다\n\n"
                "Key-aligned v2는 Gates 1–4를 세 fold 모두 통과했지만 Gate 5는 0/3이었습니다. 한 fold는 pooled "
                "ΔRMSE가 −0.003043°C였어도 개별 inner block과 layer 비악화 조건을 위반했고, 다른 fold는 개선 폭이 "
                "부족했으며 마지막 fold는 α=0 exact no-op을 선택했습니다.\n\n"
                "**의미:** sigmoid가 공개 프로파일을 잘 적합한다는 사실과 숨은 중층 residual을 안정적으로 예측한다는 "
                "사실은 다릅니다. 다음 P2 후보는 같은 sigmoid 경계·폭을 다시 튜닝하기보다, 세 inner fold 모두에서 "
                "독립적인 residual 예측력을 먼저 보여 주는 새 신호여야 합니다."
            ),
        },
        {
            "id": "p3_finding",
            "type": "markdown",
            "sourceId": "p3_b_derivation",
            "body": (
                "## P3 — 과거 파형에 causal forcing을 더하자 구조 precheck가 1/3에서 2/3으로 회복됐습니다\n\n"
                "Raw-shape v1은 pooled ratio가 0.8316으로 좋아 보여도 fold 기준은 1/3만 통과했습니다. v2는 바람 "
                "입력·방향 정렬·기압과 주기 기울기를 이용할 수 있는 경우에만 거리 계산을 조건화하고, 불가능하면 "
                "v1로 정확히 돌아갔습니다. 그 결과 겨울 전이기가 0.9311→0.8753으로 넘어오며 2/3 기준을 "
                "통과했습니다. 아래 막대에서 90% 점선 아래가 fold 통과입니다.\n\n"
                "**주의:** v2는 같은 inner split을 본 adaptive 연구이고 57개 eligible 사례 중 forcing 조건화가 가능한 "
                f"사례는 {p3_conditioned_cases}개뿐이었으며 가용성은 주로 G-ORS가 이끌었습니다. 이 B gate는 유사한 "
                "이웃을 찾았다는 증거일 뿐 실제 예측 개선 증거는 아닙니다."
            ),
        },
        {"id": "p3_chart", "type": "chart", "chartId": "p3_b_fold_ratios"},
        {
            "id": "p3_v3_result",
            "type": "markdown",
            "sourceId": "p3_predictive_v3_evidence",
            "body": (
                "### P3 v3가 유사도 신호를 실제 예측으로 검증했습니다\n\n"
                f"고정 591-feature CatBoost inner control에 +12/+18/+24h만 α=0.2로 섞은 v3의 pooled "
                f"ΔRMSE는 {p3_c['pooled_delta_m']:+.6f}m, 개선 fold는 {p3_c['improved_folds']}/3, 최악 station "
                f"Δ는 {p3_c['maximum_station_delta_m']:+.6f}m였습니다. +18h 비악화는 "
                f"{'통과' if p3_c['checks']['lead_18_non_degrading'] else '실패'}, +24h는 "
                f"{'통과' if p3_c['checks']['lead_24_non_degrading'] else '실패'}입니다. 고정 pooled 기준을 넘긴 "
                f"여유는 {p3_c_pass_margin_m:.6f}m입니다.\n\n"
                f"{p3_v3_detail}"
            ),
        },
        {
            "id": "p3_outer_result",
            "type": "markdown",
            "sourceId": "p3_outer_v4_evidence",
            "body": (
                "## P3 — inner에서 살아난 신호가 sealed outer에서 역전됐습니다\n\n"
                f"182-case outer one-shot에서 incumbent RMSE {p3_outer['incumbent_rmse_m']:.6f}m 대비 후보는 "
                f"{p3_outer['candidate_rmse_m']:.6f}m로 {p3_outer['candidate_minus_incumbent_rmse_m']:+.6f}m "
                "악화했습니다. 세 fold가 모두 악화했고, +18h는 "
                f"{p3_outer['by_lead']['18']['delta_m']:+.6f}m, +24h는 "
                f"{p3_outer['by_lead']['24']['delta_m']:+.6f}m 악화했습니다. Paired case-bootstrap CI90은 "
                f"[{p3_outer['paired_case_bootstrap']['ci90_lower_m']:+.6f}, "
                f"{p3_outer['paired_case_bootstrap']['ci90_upper_m']:+.6f}]m로 0을 포함합니다.\n\n"
                "**판정:** inner PASS는 공식 점수도, outer 일반화 보증도 아니었습니다. 사전 계약대로 exact forcing-analog "
                "경로를 재튜닝·재실행 없이 영구 종료하고 P3 동결 incumbent를 유지합니다."
            ),
        },
        {
            "id": "decision_intro",
            "type": "markdown",
            "sourceId": "decision_derivation",
            "body": (
                "## 정확한 판정은 세대별 gate에 묶여 있습니다\n\n"
                "아래 표는 ‘좋아 보이는 평균’이 아니라 각 세대가 실제로 열린 단계, 사전 기준, terminal action을 "
                "한 줄씩 고정합니다. NO-GO는 전체 물리 가설을 부정한다는 뜻이 아니라 해당 exact configuration을 "
                "결과에 맞춰 재탐색하지 않는다는 뜻입니다."
            ),
        },
        {"id": "decision_table", "type": "table", "tableId": "exact_decision_table"},
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "decision_derivation",
            "body": (
                "## Recommended Next Steps\n\n"
                "1. **P1은 typed decoder 세대를 닫고 incumbent를 유지합니다.** 새 시도는 composite grammar가 아니라 "
                "offset/drift unary 분리력의 독립 사전검증으로 시작합니다.\n"
                "2. **P2는 dynamic sigmoid v2를 승격하지 않습니다.** 분모 문제는 해결됐으므로 같은 Gate 1을 다시 "
                "논쟁하지 말고, 세 inner fold 모두에서 residual 예측성을 보이는 새 signal/operator가 있을 때만 재개합니다.\n"
                "3. **P3 forcing-analog exact 경로를 종료합니다.** Inner gate를 다시 보거나 distance·k·alpha를 "
                "재탐색하지 않고 현재 동결 incumbent를 유지합니다.\n"
                "4. **공식 리더보드 점수가 공개되면 먼저 검증 분포 이동을 점검합니다.** 로컬 gate를 점수에 맞춰 "
                "사후 조정하기 전에 문제별 incumbent와 새 구조의 예상 실패 모드를 비교합니다."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further Questions\n\n"
                "- P1의 offset/drift false negative를 줄이면서 spike recall과 희소 station-layer를 보호할 독립 신호가 있는가?\n"
                "- P2에서 공개 프로파일 적합도가 아니라 중층 곡률 residual의 fold 간 부호·크기를 예측할 정보가 있는가?\n"
                "- P3 forcing 가용성이 낮은 I-ORS·S-ORS에서도 G-ORS 중심의 adaptive 이득을 재현할 수 있는가?\n"
                "- 첫 공식 점수가 공개될 때 local-to-hidden 난이도 차이가 문제별로 어느 방향인지 확인할 수 있는가?"
            ),
        },
        {
            "id": "caveats",
            "type": "markdown",
            "sourceId": "decision_derivation",
            "body": (
                "## Caveats and Assumptions\n\n"
                "- 모든 성능·gate 수치는 historical 또는 inner validation 집계이며 공식 hidden score가 아닙니다.\n"
                "- P3 v2와 v3는 같은 inner split을 순차적으로 사용한 adaptive 연구이며 독립 확인이 아닙니다.\n"
                "- P3 v4 outer도 이전 연구 계보에서 지정된 182-case adaptive research 평가이며 공식 hidden score가 "
                "아닙니다. 다만 봉인된 one-shot에서 세 fold 모두 악화해 exact 후보 종료 근거로는 충분합니다.\n"
                "- P2 v2도 이전 outer 노출 이후의 adaptive 진단이므로 fresh holdout 주장에 사용할 수 없습니다.\n"
                f"- 보고서 빌더는 정확히 {len(RELATIVE_PATHS)}개 SHA-pinned aggregate JSON만 읽습니다. 원자료, "
                "row-level OOF, test, submission 값은 읽지 않으며 SHA가 하나라도 달라지면 fail-close합니다.\n"
                "- 공식 리더보드 상태는 2026년 8월 22일 KST의 읽기 전용 snapshot이며 이후 변경될 수 있습니다.\n"
                "- 이번 감사로 기존 제출 파일, 모델, 원자료, 이전 보고서 또는 이전 builder를 변경하지 않았습니다."
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "P1–P3 새 구조의 식별가능성, inner gate, 남은 의사결정을 정리한 aggregate-only executive audit",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [],
            "charts": [p2_chart, p3_chart],
            "tables": [decision_table],
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "p2_support_basis": p2_support_rows,
                "p3_b_fold_ratios": p3_ratio_rows,
                "exact_decision_table": decision_rows,
            },
            "accessIssues": [],
        },
        "sources": [
            {
                "id": source["id"],
                "label": source["label"],
                **({"path": source["path"]} if "path" in source else {}),
            }
            for source in sources
        ],
        "package_info": {
            "originUrl": "artifact://structural-path-audit-2026-08-22-r1",
            "controls": {"edit": False, "refresh": False},
        },
    }
    _validate_artifact_shape(artifact)
    return artifact


def _validate_artifact_shape(artifact: dict[str, Any]) -> None:
    _require(artifact["surface"] == "report", "surface must be report")
    manifest = artifact["manifest"]
    blocks = manifest["blocks"]
    _require(blocks[0]["body"] == f"# {REPORT_TITLE}", "title block mismatch")
    _require(blocks[1]["body"].startswith("## Executive Summary"), "summary must follow title")
    _require(len(manifest["charts"]) == 2, "report must have exactly two charts")
    _require(len(manifest["tables"]) == 1, "report must have exactly one decision table")
    datasets = artifact["snapshot"]["datasets"]
    _require(len(datasets["p2_support_basis"]) == 6, "P2 chart row count drifted")
    _require(len(datasets["p3_b_fold_ratios"]) == 6, "P3 chart row count drifted")
    _require(len(datasets["exact_decision_table"]) == 8, "decision row count drifted")
    _require(
        [row["sequence"] for row in datasets["exact_decision_table"]] == list(range(1, 9)),
        "decision sequence drifted",
    )
    source_ids = {source["id"] for source in manifest["sources"]}
    _require(len(source_ids) == len(manifest["sources"]), "duplicate source id")
    for chart in manifest["charts"]:
        _require(chart["sourceId"] in source_ids, f"chart source missing: {chart['id']}")
        _require(chart["dataset"] in datasets, f"chart dataset missing: {chart['id']}")
        _require("source" in chart, f"chart inline provenance missing: {chart['id']}")
    for table in manifest["tables"]:
        _require(table["sourceId"] in source_ids, f"table source missing: {table['id']}")
        _require(table["dataset"] in datasets, f"table dataset missing: {table['id']}")
        _require(table["defaultSort"]["field"] == "sequence", "table default sort drifted")
    serialized = json.dumps(artifact, ensure_ascii=False)
    for forbidden in (
        "C:/Users/",
        "C:\\Users\\",
        ".parquet",
        "train.csv",
        "test_context.parquet",
        "test_index.csv",
    ):
        _require(forbidden not in serialized, f"forbidden data reference leaked: {forbidden}")
    _require("정책·점수 환산 상수" in serialized, "P3 T correction missing")
    _require("MCP report tools were unavailable" in serialized, "HTML fallback note missing")
    for source in artifact["sources"]:
        location = source.get("path")
        if location and not str(location).startswith("https://"):
            _require(not Path(str(location)).is_absolute(), "absolute source path leaked")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    generated_at = args.generated_at or datetime.now(KST).isoformat(timespec="seconds")
    evidence = collect_evidence(root)
    artifact = build_artifact(evidence, generated_at=generated_at)
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(output), "sha256": _sha256(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
