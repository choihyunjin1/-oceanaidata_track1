"""Build the aggregate-only canonical artifact for the P1 R1 technical report.

This module deliberately produces only ``artifact.json``.  The packaged
Data Analytics report builder owns HTML rendering and browser verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class R1ReportArtifactError(ValueError):
    """Raised when report inputs cannot support an auditable comparison."""


def _read_json(path: str | Path, *, role: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise R1ReportArtifactError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise R1ReportArtifactError(f"{role} must contain a JSON object")
    return source, value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R1ReportArtifactError(f"{role} must be an object")
    return value


def _sequence(value: Any, *, role: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise R1ReportArtifactError(f"{role} must be a list")
    return value


def _number(value: Any, *, role: str) -> float:
    if isinstance(value, bool):
        raise R1ReportArtifactError(f"{role} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise R1ReportArtifactError(f"{role} must be numeric") from exc
    if not math.isfinite(result):
        raise R1ReportArtifactError(f"{role} must be finite")
    return result


def _f1(report: Mapping[str, Any], *, role: str, family: str) -> float:
    metric = _mapping(report.get(family), role=f"{role}.{family}")
    return _number(metric.get("f1"), role=f"{role}.{family}.f1")


def _assert_close(left: float, right: float, *, role: str, tolerance: float = 1.0e-9) -> None:
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise R1ReportArtifactError(
            f"{role} does not reconcile across saved evidence: {left:.12g} != {right:.12g}"
        )


def _logical_path(path: Path, fallback: str) -> str:
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    for marker in ("artifacts", "reports", "configs", "scripts"):
        if marker in lowered:
            start = lowered.index(marker)
            logical = "/".join(parts[start:])
            if ".." not in Path(logical).parts:
                return logical
    return fallback


def _source(
    source_id: str,
    label: str,
    logical_path: str,
    digest: str,
    description: str,
    *,
    executed_at: str | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "id": digest,
        "language": "json",
        "description": description,
        "filters": [
            "Aggregate metrics only",
            "No external observations",
            "No hidden-test labels",
        ],
    }
    if executed_at:
        query["executed_at"] = executed_at
    return {
        "id": source_id,
        "label": label,
        "path": logical_path,
        "query": query,
    }


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        numeric = _number(value, role="SQL materialization value")
        return repr(numeric) if isinstance(value, float) else str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _materialization_sql(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        raise R1ReportArtifactError("cannot materialize an empty report dataset")
    selects: list[str] = []
    for index, row in enumerate(rows):
        values = []
        for column in columns:
            literal = _sql_literal(row.get(column))
            values.append(f"{literal} AS {column}" if index == 0 else literal)
        selects.append("SELECT " + ", ".join(values))
    return " UNION ALL ".join(selects)


def _fold_records(
    metrics: Mapping[str, Any], independent: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_folds = _sequence(metrics.get("folds"), role="metrics.folds")
    independent_folds = _sequence(independent.get("by_fold"), role="independent.by_fold")
    independent_by_name: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(independent_folds):
        record = _mapping(raw, role=f"independent.by_fold[{index}]")
        name = str(record.get("fold", "")).strip()
        if not name or name in independent_by_name:
            raise R1ReportArtifactError("independent fold names must be non-empty and unique")
        independent_by_name[name] = record

    chart_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(metric_folds):
        record = _mapping(raw, role=f"metrics.folds[{index}]")
        fold = str(record.get("fold", record.get("name", ""))).strip()
        if not fold or fold not in independent_by_name:
            raise R1ReportArtifactError(f"metrics fold {fold!r} is absent from independent QA")
        candidate = _mapping(record.get("candidate"), role=f"metrics fold {fold}.candidate")
        baseline = _mapping(record.get("base"), role=f"metrics fold {fold}.base")
        independent_fold = independent_by_name[fold]
        independent_candidate = _mapping(
            independent_fold.get("candidate"), role=f"independent fold {fold}.candidate"
        )
        independent_baseline = _mapping(
            independent_fold.get("baseline"), role=f"independent fold {fold}.baseline"
        )
        candidate_micro = _number(
            independent_candidate.get("f1"), role=f"independent fold {fold} candidate f1"
        )
        baseline_micro = _number(
            independent_baseline.get("f1"), role=f"independent fold {fold} baseline f1"
        )
        _assert_close(
            candidate_micro,
            _f1(candidate, role=f"metrics fold {fold}.candidate", family="micro"),
            role=f"fold {fold} candidate micro F1",
        )
        _assert_close(
            baseline_micro,
            _f1(baseline, role=f"metrics fold {fold}.base", family="micro"),
            role=f"fold {fold} baseline micro F1",
        )
        candidate_weighted = _f1(
            candidate, role=f"metrics fold {fold}.candidate", family="weighted"
        )
        baseline_weighted = _f1(baseline, role=f"metrics fold {fold}.base", family="weighted")
        rows = int(_number(independent_fold.get("rows"), role=f"fold {fold}.rows"))
        positives = int(
            _number(independent_fold.get("positive_rows"), role=f"fold {fold}.positive_rows")
        )
        for metric, baseline_f1, candidate_f1 in (
            ("Row micro F1", baseline_micro, candidate_micro),
            ("Test-share weighted F1", baseline_weighted, candidate_weighted),
        ):
            delta = candidate_f1 - baseline_f1
            category = f"{fold} · {metric}"
            for model, value in (("Frozen baseline", baseline_f1), ("R1", candidate_f1)):
                chart_rows.append(
                    {
                        "fold_metric": category,
                        "fold": fold,
                        "metric": metric,
                        "model": model,
                        "f1": value,
                        "delta_f1": delta,
                        "rows": rows,
                        "positive_rows": positives,
                    }
                )
        audit_rows.append(
            {
                "fold": fold,
                "micro_delta": candidate_micro - baseline_micro,
                "weighted_delta": candidate_weighted - baseline_weighted,
            }
        )
    if set(independent_by_name) != {record["fold"] for record in audit_rows}:
        raise R1ReportArtifactError("metrics and independent QA fold sets differ")
    return chart_rows, audit_rows


def _gate_rows(
    independent: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    fold_audit: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    outer = _mapping(
        preregistration.get("outer_evaluation"), role="preregistration.outer_evaluation"
    )
    gate = _mapping(outer.get("promotion_gate"), role="preregistration promotion_gate")
    official = _mapping(
        independent.get("official_row_metrics"), role="independent official metrics"
    )
    weighted = _mapping(
        independent.get("test_share_weighted_metrics"), role="independent weighted metrics"
    )
    bootstrap = _mapping(independent.get("paired_block_bootstrap"), role="independent bootstrap")
    group_rows = _sequence(independent.get("by_station_layer"), role="independent.by_station_layer")
    fp = _mapping(independent.get("normal_station_layer_day_fp"), role="independent normal-day FP")

    micro_delta = _number(
        _mapping(official.get("delta"), role="official delta").get("f1"),
        role="official micro F1 delta",
    )
    # Reading this validates the report's secondary aggregate even though the
    # preregistered outer gate is intentionally micro-F1 based.
    _number(
        _mapping(weighted.get("delta"), role="weighted delta").get("f1"),
        role="weighted F1 delta",
    )
    ci = _sequence(bootstrap.get("difference_ci90"), role="bootstrap difference_ci90")
    if len(ci) != 2:
        raise R1ReportArtifactError("bootstrap difference_ci90 must contain two values")
    ci_lower = _number(ci[0], role="bootstrap CI lower")
    nondegrading = sum(
        _number(row["micro_delta"], role="fold micro delta") >= 0 for row in fold_audit
    )

    worst_drop = 0.0
    for index, raw in enumerate(group_rows):
        record = _mapping(raw, role=f"station-layer group {index}")
        candidate = _mapping(record.get("candidate"), role=f"group {index}.candidate")
        baseline = _mapping(record.get("baseline"), role=f"group {index}.baseline")
        drop = _number(baseline.get("f1"), role=f"group {index} baseline F1") - _number(
            candidate.get("f1"), role=f"group {index} candidate F1"
        )
        worst_drop = max(worst_drop, drop)

    fp_candidate = _number(
        _mapping(fp.get("candidate"), role="FP candidate").get(
            "false_positive_rows_per_normal_station_layer_day"
        ),
        role="candidate FP/day",
    )
    fp_baseline = _number(
        _mapping(fp.get("baseline"), role="FP baseline").get(
            "false_positive_rows_per_normal_station_layer_day"
        ),
        role="baseline FP/day",
    )
    fp_relative = (
        (fp_candidate - fp_baseline) / fp_baseline
        if fp_baseline > 0
        else (0.0 if fp_candidate == 0 else None)
    )

    micro_min = _number(gate.get("micro_f1_delta_min"), role="micro delta gate")
    ci_min = _number(gate.get("bootstrap_90pct_delta_lower_gt"), role="CI gate")
    folds_min = int(_number(gate.get("folds_non_degrading_min"), role="fold gate"))
    group_max = _number(gate.get("station_group_f1_drop_max"), role="group-drop gate")
    fp_max = _number(gate.get("normal_fp_day_relative_increase_lt"), role="normal FP relative gate")
    specs = [
        (
            1,
            "Overall micro F1 uplift",
            micro_delta,
            f">= {micro_min:.4f}",
            micro_delta >= micro_min,
        ),
        (2, "Paired block bootstrap 90% CI lower", ci_lower, f"> {ci_min:.4f}", ci_lower > ci_min),
        (
            3,
            "Non-degrading outer folds",
            float(nondegrading),
            f">= {folds_min}",
            nondegrading >= folds_min,
        ),
        (
            4,
            "Worst station-layer F1 drop",
            worst_drop,
            f"<= {group_max:.4f}",
            worst_drop <= group_max,
        ),
        (
            5,
            "Normal FP/day relative increase",
            fp_relative,
            f"< {fp_max:.1%}",
            fp_relative is not None and fp_relative < fp_max,
        ),
    ]
    rows = [
        {
            "rank": rank,
            "gate": label,
            "observed": ("not evaluable" if observed is None else f"{observed:.6f}"),
            "requirement": requirement,
            "result": "PASS" if passed else "FAIL",
        }
        for rank, label, observed, requirement, passed in specs
    ]
    return rows, all(bool(spec[-1]) for spec in specs)


def _artifact_timestamp(manifest: Mapping[str, Any], preregistration: Mapping[str, Any]) -> str:
    for value in (
        manifest.get("finished_at"),
        manifest.get("created_at"),
        preregistration.get("created_at_kst"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise R1ReportArtifactError("manifest/preregistration has no report timestamp")


def build_artifact(
    metrics_path: str | Path,
    independent_validation_path: str | Path,
    manifest_path: str | Path,
    preregistration_path: str | Path,
    baseline_metrics_path: str | Path,
) -> dict[str, Any]:
    """Create a canonical, aggregate-only Data Analytics report artifact."""

    metrics_file, metrics = _read_json(metrics_path, role="R1 metrics")
    independent_file, independent = _read_json(
        independent_validation_path, role="independent validation"
    )
    manifest_file, run_manifest = _read_json(manifest_path, role="R1 manifest")
    prereg_file, prereg = _read_json(preregistration_path, role="R1 preregistration")
    baseline_file, baseline_metrics = _read_json(
        baseline_metrics_path, role="frozen baseline metrics"
    )

    if independent.get("status") != "passed":
        raise R1ReportArtifactError("independent validation status must be 'passed'")
    use_policy = _mapping(independent.get("use_policy"), role="independent use_policy")
    if use_policy.get("candidate_selection_allowed") is not False:
        raise R1ReportArtifactError("independent validation must forbid candidate selection")
    if metrics.get("outer_labels_used_for_selection") is not False:
        raise R1ReportArtifactError(
            "R1 metrics must prove outer labels were not used for selection"
        )

    official = _mapping(independent.get("official_row_metrics"), role="official row metrics")
    weighted = _mapping(independent.get("test_share_weighted_metrics"), role="weighted row metrics")
    candidate_official = _mapping(official.get("candidate"), role="official candidate")
    baseline_official = _mapping(official.get("baseline"), role="official baseline")
    candidate_weighted = _mapping(weighted.get("candidate"), role="weighted candidate")
    baseline_weighted = _mapping(weighted.get("baseline"), role="weighted baseline")
    candidate_micro = _number(candidate_official.get("f1"), role="candidate micro F1")
    baseline_micro = _number(baseline_official.get("f1"), role="baseline micro F1")
    candidate_weighted_f1 = _number(candidate_weighted.get("f1"), role="candidate weighted F1")
    baseline_weighted_f1 = _number(baseline_weighted.get("f1"), role="baseline weighted F1")

    r1_aggregate = _mapping(metrics.get("aggregate"), role="metrics.aggregate")
    r1_base = _mapping(metrics.get("base_aggregate"), role="metrics.base_aggregate")
    frozen_aggregate = _mapping(baseline_metrics.get("aggregate"), role="baseline.aggregate")
    for label, observed, expected in (
        (
            "candidate aggregate micro",
            candidate_micro,
            _f1(r1_aggregate, role="R1 aggregate", family="micro"),
        ),
        (
            "candidate aggregate weighted",
            candidate_weighted_f1,
            _f1(r1_aggregate, role="R1 aggregate", family="weighted"),
        ),
        ("baseline aggregate micro", baseline_micro, _f1(r1_base, role="R1 base", family="micro")),
        (
            "baseline aggregate weighted",
            baseline_weighted_f1,
            _f1(r1_base, role="R1 base", family="weighted"),
        ),
        (
            "frozen baseline micro",
            baseline_micro,
            _f1(frozen_aggregate, role="frozen aggregate", family="micro"),
        ),
        (
            "frozen baseline weighted",
            baseline_weighted_f1,
            _f1(frozen_aggregate, role="frozen aggregate", family="weighted"),
        ),
    ):
        _assert_close(observed, expected, role=label)

    fold_rows, fold_audit = _fold_records(metrics, independent)
    gate_rows, promotion_passed = _gate_rows(independent, prereg, fold_audit)
    enabled_candidate_count = 0
    candidates_with_proposals = 0
    proposed_rows_max = 0
    proposal_diagnostics_available = True
    metric_folds = _sequence(metrics.get("folds"), role="metrics.folds")
    for fold_index, raw_fold in enumerate(metric_folds):
        fold = _mapping(raw_fold, role=f"metrics fold {fold_index}")
        diagnostics = fold.get("boundary_diagnostics")
        if not isinstance(diagnostics, Mapping):
            proposal_diagnostics_available = False
            continue
        candidates = _sequence(
            diagnostics.get("candidates"), role=f"metrics fold {fold_index} candidates"
        )
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate = _mapping(
                raw_candidate,
                role=f"metrics fold {fold_index} candidate {candidate_index}",
            )
            parameters = _mapping(
                candidate.get("parameters"),
                role=f"metrics fold {fold_index} candidate {candidate_index} parameters",
            )
            if parameters.get("enabled") is not True:
                continue
            enabled_candidate_count += 1
            proposal_rows = int(
                _number(
                    candidate.get("proposal_rows"),
                    role=f"metrics fold {fold_index} candidate {candidate_index} proposal rows",
                )
            )
            proposed_rows_max = max(proposed_rows_max, proposal_rows)
            candidates_with_proposals += int(proposal_rows > 0)
    bootstrap = _mapping(independent.get("paired_block_bootstrap"), role="bootstrap")
    ci = _sequence(bootstrap.get("difference_ci90"), role="bootstrap CI")
    ci_lower = _number(ci[0], role="bootstrap CI lower")
    ci_upper = _number(ci[1], role="bootstrap CI upper")
    probability_improved = _number(
        bootstrap.get("probability_improved"), role="bootstrap probability improved"
    )
    long_event = _mapping(independent.get("long_positive_events"), role="long events")
    long_delta = long_event.get("delta_row_recall")
    long_delta_value = (
        None if long_delta is None else _number(long_delta, role="long-event recall delta")
    )
    scope = _mapping(independent.get("scope"), role="independent scope")
    timestamp = _artifact_timestamp(run_manifest, prereg)

    experiment_id = str(prereg.get("experiment_id", "R1 boundary completion")).strip()
    baseline_info = _mapping(prereg.get("baseline"), role="preregistration baseline")
    baseline_run = str(baseline_info.get("run_id", "frozen_baseline")).strip()
    metrics_logical = _logical_path(metrics_file, "artifacts/r1/metrics.json")
    independent_logical = _logical_path(
        independent_file, "artifacts/r1/independent_validation.json"
    )
    manifest_logical = _logical_path(manifest_file, "artifacts/r1/manifest.json")
    prereg_logical = _logical_path(prereg_file, "artifacts/r1/preregistration.json")
    baseline_logical = _logical_path(baseline_file, f"artifacts/runs/{baseline_run}/metrics.json")
    micro_delta = candidate_micro - baseline_micro
    weighted_delta = candidate_weighted_f1 - baseline_weighted_f1
    added_tp = _number(candidate_official.get("tp"), role="candidate TP") - _number(
        baseline_official.get("tp"), role="baseline TP"
    )
    added_fp = _number(candidate_official.get("fp"), role="candidate FP") - _number(
        baseline_official.get("fp"), role="baseline FP"
    )
    added_positive_rows = added_tp + added_fp
    marginal_precision = added_tp / added_positive_rows if added_positive_rows > 0 else None
    passed_count = sum(row["result"] == "PASS" for row in gate_rows)
    headline_row = {
        "r1_micro_f1": candidate_micro,
        "baseline_micro_f1": baseline_micro,
        "micro_delta": micro_delta,
        "r1_weighted_f1": candidate_weighted_f1,
        "baseline_weighted_f1": baseline_weighted_f1,
        "weighted_delta": weighted_delta,
        "bootstrap_ci90_lower": ci_lower,
        "bootstrap_ci90_upper": ci_upper,
        "bootstrap_probability_improved": probability_improved,
        "passed_gate_count": passed_count,
        "total_gate_count": len(gate_rows),
        "promotion_passed": promotion_passed,
        "enabled_inner_candidates": (
            enabled_candidate_count if proposal_diagnostics_available else None
        ),
        "candidates_with_proposals": (
            candidates_with_proposals if proposal_diagnostics_available else None
        ),
        "added_true_positive_rows": added_tp,
        "added_false_positive_rows": added_fp,
        "marginal_precision": marginal_precision,
    }
    sources = [
        _source(
            "r1_metrics",
            "R1 nested-CV aggregate metrics",
            metrics_logical,
            _sha256(metrics_file),
            "Fold and aggregate R1-versus-base metrics produced by the preregistered nested CV.",
            executed_at=timestamp,
        ),
        _source(
            "independent_validation",
            "Independent R1 OOF validation",
            independent_logical,
            _sha256(independent_file),
            "Alignment checks, independently recalculated metrics, subgroup checks, and paired block bootstrap.",
            executed_at=timestamp,
        ),
        _source(
            "run_manifest",
            "R1 execution manifest",
            manifest_logical,
            _sha256(manifest_file),
            "Execution provenance, input hashes, environment record, and output hashes.",
            executed_at=timestamp,
        ),
        _source(
            "preregistration",
            "R1 preregistration",
            prereg_logical,
            _sha256(prereg_file),
            "Frozen hypothesis, finite grid, inner-label selection contract, and promotion gates.",
        ),
        _source(
            "baseline_metrics",
            "Frozen XGBoost baseline metrics",
            baseline_logical,
            _sha256(baseline_file),
            "Saved aggregate and fold metrics for the frozen comparison baseline.",
        ),
        {
            "id": "report_derivation",
            "label": "R1 technical report derivation",
            "path": "scripts/build_r1_report_artifact.py",
            "query": {
                "language": "python",
                "description": "Reconciles five aggregate JSON inputs and derives fold-comparison and promotion-gate datasets without exporting observation rows.",
                "tables_used": [
                    metrics_logical,
                    independent_logical,
                    manifest_logical,
                    prereg_logical,
                    baseline_logical,
                ],
                "filters": [
                    "Aggregate-only inputs",
                    "Outer labels excluded from model and boundary selection",
                    "No competition upload decision",
                ],
                "metric_definitions": {
                    "row_micro_f1": "2TP / (2TP + FP + FN) across aligned outer-validation rows",
                    "test_share_weighted_f1": "Station-layer confusion totals weighted by 2026 test row shares before F1",
                    "paired_delta_ci90": "5th and 95th percentiles of paired positive-event and normal-day block bootstrap F1 differences",
                },
            },
        },
        {
            "id": "headline_metrics_sql",
            "label": "Reconciled R1 headline metrics",
            "path": "scripts/build_r1_report_artifact.py",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": _materialization_sql([headline_row], list(headline_row)),
                "description": "Materializes reconciled aggregate R1, baseline, bootstrap, and promotion-count values.",
                "tables_used": [],
                "filters": ["Aggregate-only inputs", "Cross-source reconciliation passed"],
                "metric_definitions": {
                    "micro_delta": "R1 row micro F1 minus frozen-baseline row micro F1",
                    "weighted_delta": "R1 test-share weighted F1 minus frozen-baseline weighted F1",
                },
            },
        },
        {
            "id": "fold_comparison_sql",
            "label": "Reconciled outer-fold F1 comparison",
            "path": "scripts/build_r1_report_artifact.py",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": _materialization_sql(fold_rows, list(fold_rows[0])),
                "description": "Materializes fold-by-metric baseline and R1 F1 rows after independent micro-F1 reconciliation.",
                "tables_used": [],
                "filters": [
                    "Purged rolling-origin outer folds",
                    "Micro and test-share weighted F1",
                ],
                "metric_definitions": {
                    "delta_f1": "R1 F1 minus frozen-baseline F1 within the same outer fold and metric"
                },
            },
        },
        {
            "id": "promotion_gates_sql",
            "label": "Preregistered promotion-gate evaluation",
            "path": "scripts/build_r1_report_artifact.py",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": _materialization_sql(gate_rows, list(gate_rows[0])),
                "description": "Materializes exact observed values and pass/fail outcomes for the frozen promotion contract.",
                "tables_used": [],
                "filters": ["Five preregistered outer gates", "No post-hoc threshold changes"],
                "metric_definitions": {
                    "result": "PASS only when the observed value satisfies the preregistered comparator"
                },
            },
        },
    ]

    decision = (
        "R1은 사전등록된 로컬 승격 기준을 모두 통과했다. 다음 단계 후보로 동결할 수 있지만, 공식 점수나 제출 승인을 뜻하지 않는다."
        if promotion_passed
        else "R1은 사전등록된 로컬 승격 기준을 모두 통과하지 못했다. frozen baseline을 유지하고 이 outer 결과로 R1을 재조정하지 않는다."
    )
    if not proposal_diagnostics_available:
        proposal_sentence = "이 입력에는 fold별 proposal 생성 진단이 없어 proposal 적용 여부를 별도로 판정하지 않았다."
    elif candidates_with_proposals == 0:
        proposal_sentence = (
            f"세 fold의 enabled inner 후보 {enabled_candidate_count:,}개가 모두 빈 proposal을 반환해 "
            "안전한 no-op으로 후퇴했다. 따라서 이 결과는 R1 알고리즘 성능으로 해석할 수 없다."
        )
    else:
        precision_text = "계산 불가" if marginal_precision is None else f"{marginal_precision:.1%}"
        proposal_sentence = (
            f"세 fold의 enabled inner 후보 {enabled_candidate_count:,}개 중 "
            f"{candidates_with_proposals:,}개가 실제 proposal을 만들었고 후보당 최대 {proposed_rows_max:,}행이었다. "
            f"최종 union은 baseline 대비 TP {added_tp:+.0f}행과 FP {added_fp:+.0f}행을 추가해 "
            f"추가 행의 한계 정밀도는 {precision_text}였다."
        )
    long_sentence = (
        "48시간 이상 양성 이벤트가 없어 장기 이벤트 recall 변화는 평가하지 못했다."
        if long_delta_value is None
        else f"48시간 이상 양성 이벤트 row recall 변화는 {long_delta_value:+.6f}이다."
    )
    next_steps = (
        "1. 현재 R1 파라미터·가중치·입력 및 출력 해시를 동결한다.\n"
        "2. clean environment에서 동일 OOF와 보고서 해시를 재현한다.\n"
        "3. 추가 outer tuning 없이 R2 진입 여부를 inner-only 근거로 결정한다.\n"
        "4. test 추론과 제출 파일 생성은 별도 승인 단계로 유지하며 업로드하지 않는다."
        if promotion_passed
        else "1. frozen XGBoost baseline을 현 최선 후보로 유지한다.\n"
        "2. 경계 proposal을 곧바로 OR하지 말고, interval-level precision gate·segment reranker의 입력으로만 사용한다.\n"
        "3. 다음 가설과 finite grid를 별도 preregistration으로 동결한 뒤에만 새 outer 평가를 연다.\n"
        "4. 실패한 R1 결과로 test 추론이나 대회 업로드를 진행하지 않는다."
    )

    title = "P1 R1 경계 완성 실험 기술 검증"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Frozen XGBoost 대비 CAPA/PELT·CPOP-lite 경계 완성 R1의 누출 안전 nested-CV 및 독립 OOF 검증 보고서",
            "generatedAt": timestamp,
            "cards": [
                {
                    "id": "micro_card",
                    "description": "정렬된 outer-validation 전체 행의 공식식 micro F1",
                    "dataset": "headline",
                    "sourceId": "headline_metrics_sql",
                    "metrics": [
                        {"label": "R1 micro F1", "field": "r1_micro_f1"},
                        {"label": "Frozen baseline", "field": "baseline_micro_f1"},
                        {"label": "Delta", "field": "micro_delta", "signed": True},
                    ],
                },
                {
                    "id": "weighted_card",
                    "description": "2026 test station-layer 행 비중으로 confusion totals를 재가중한 F1",
                    "dataset": "headline",
                    "sourceId": "headline_metrics_sql",
                    "metrics": [
                        {"label": "R1 weighted F1", "field": "r1_weighted_f1"},
                        {"label": "Frozen baseline", "field": "baseline_weighted_f1"},
                        {"label": "Delta", "field": "weighted_delta", "signed": True},
                    ],
                },
                {
                    "id": "bootstrap_card",
                    "description": "양성 event와 정상 station-layer-day를 함께 재표집한 paired bootstrap",
                    "dataset": "headline",
                    "sourceId": "headline_metrics_sql",
                    "metrics": [
                        {"label": "90% CI lower", "field": "bootstrap_ci90_lower"},
                        {"label": "90% CI upper", "field": "bootstrap_ci90_upper"},
                        {
                            "label": "P(delta > 0)",
                            "field": "bootstrap_probability_improved",
                            "format": "percent",
                        },
                    ],
                },
                {
                    "id": "gate_card",
                    "description": "사전등록한 다섯 개 outer promotion gate의 통과 개수",
                    "dataset": "headline",
                    "sourceId": "headline_metrics_sql",
                    "metrics": [
                        {"label": "Passed gates", "field": "passed_gate_count"},
                        {"label": "Total gates", "field": "total_gate_count"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "fold_f1_chart",
                    "title": "Outer-fold baseline and R1 F1",
                    "subtitle": "Three purged rolling-origin folds; row micro and 2026 test-share weighted F1",
                    "type": "bar",
                    "intent": "comparison",
                    "question": "각 outer fold에서 frozen baseline 대비 R1의 micro 및 weighted F1은 어떻게 달라졌는가?",
                    "rationale": "세 개의 이산 fold와 두 개의 같은 단위 F1을 baseline/R1 두 계열로 직접 비교하므로 grouped bar가 가장 읽기 쉽다.",
                    "comparisonContext": {
                        "baseline": "Frozen XGBoost",
                        "grain": "outer fold × metric",
                        "unit": "F1 (0–1)",
                    },
                    "dataset": "fold_comparison",
                    "sourceId": "fold_comparison_sql",
                    "valueFormat": "number",
                    "palette": {"kind": "semantic", "name": "actual-vs-baseline"},
                    "legend": {"position": "bottom", "sort": "spec", "title": "Model"},
                    "labels": {"values": "auto"},
                    "encodings": {
                        "x": {
                            "field": "fold_metric",
                            "type": "nominal",
                            "label": "Outer fold and metric",
                        },
                        "y": {"field": "f1", "type": "quantitative", "label": "F1"},
                        "color": {"field": "model", "type": "nominal", "label": "Model"},
                        "tooltip": [
                            {"field": "delta_f1", "type": "quantitative", "label": "R1 − baseline"},
                            {"field": "rows", "type": "quantitative", "label": "Validation rows"},
                            {
                                "field": "positive_rows",
                                "type": "quantitative",
                                "label": "Positive rows",
                            },
                        ],
                    },
                }
            ],
            "tables": [
                {
                    "id": "promotion_gate_table",
                    "title": "Preregistered outer promotion gates",
                    "subtitle": "이벤트 보호용 fold membership 고정 뒤 모델·경계 선택은 outer 정답과 분리되며, 한 항목이라도 실패하면 승격하지 않는다.",
                    "dataset": "promotion_gates",
                    "sourceId": "promotion_gates_sql",
                    "defaultSort": {"field": "rank", "direction": "asc"},
                    "columns": [
                        {"field": "rank", "label": "Order", "type": "number"},
                        {"field": "gate", "label": "Gate", "type": "text"},
                        {"field": "observed", "label": "Observed", "type": "text"},
                        {"field": "requirement", "label": "Requirement", "type": "text"},
                        {"field": "result", "label": "Result", "type": "text"},
                    ],
                }
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "report_derivation",
                    "body": f"## Technical Summary\n\n**{decision}** R1 micro F1은 {candidate_micro:.6f}로 baseline {baseline_micro:.6f} 대비 {micro_delta:+.6f}, test-share weighted F1은 {candidate_weighted_f1:.6f}로 {weighted_delta:+.6f} 변했다. paired block bootstrap의 90% delta 구간은 [{ci_lower:+.6f}, {ci_upper:+.6f}]이다. {proposal_sentence} 이 수치는 hidden test의 공식 점수가 아닌 outer-label 1회 연구 검증이다.",
                },
                {
                    "id": "headline_strip",
                    "type": "metric-strip",
                    "cardIds": ["micro_card", "weighted_card", "bootstrap_card", "gate_card"],
                },
                {
                    "id": "key_findings",
                    "type": "markdown",
                    "sourceId": "report_derivation",
                    "body": f"## Fold Evidence Shows Where the Aggregate Result Came From\n\n아래 비교는 각 fold에서 동일한 validation population과 동일한 frozen baseline을 사용한다. 막대의 높이는 F1 절대값이고 tooltip의 delta는 R1−baseline이다. **{passed_count}/5개 promotion gate가 통과**했으므로 fold별 방향과 weighted 결과를 aggregate 하나와 함께 읽어야 한다.",
                },
                {"id": "fold_chart_block", "type": "chart", "chartId": "fold_f1_chart"},
                {
                    "id": "gate_interpretation",
                    "type": "markdown",
                    "sourceId": "report_derivation",
                    "body": f"## The Frozen Promotion Contract Decides the Outcome\n\n{decision} micro uplift, paired uncertainty, fold 일관성, 최악 station-layer 하락, 정상 FP/day 증가를 서로 대체할 수 없는 guardrail로 평가했다. 그 결과는 아래 표에 정확값과 함께 제시한다.",
                },
                {"id": "gate_table_block", "type": "table", "tableId": "promotion_gate_table"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "sourceId": "independent_validation",
                    "body": f"## Scope and Metric Definitions\n\n평가 grain은 **정렬된 outer-validation station-layer-time 1행**이며 총 {int(_number(scope.get('rows'), role='scope rows')):,}행, {int(_number(scope.get('folds'), role='scope folds'))}개 fold, {int(_number(scope.get('groups'), role='scope groups'))}개 station-layer group이다. row micro F1은 전체 TP·FP·FN을 합산한 뒤 `2TP / (2TP + FP + FN)`으로 계산한다. weighted F1은 2026 test의 station-layer 행 비중으로 group confusion totals를 재가중한 뒤 같은 식을 적용한다. anomaly prevalence를 4%에 맞추지 않았고 외부 관측값은 사용하지 않았다.",
                },
                {
                    "id": "nested_methodology",
                    "type": "markdown",
                    "sourceId": "preregistration",
                    "body": f"## Nested Selection and Provenance Prevent Outer-Label Tuning\n\n실험 `{experiment_id}`는 XGBoost score seed 주변의 paired mean/variance 및 continuous-linear slope change proposal을 시험한다. finite grid는 no-op을 포함해 {int(_number(_mapping(prereg.get('grid'), role='preregistration grid').get('total_candidates_including_no_op'), role='grid candidates'))}개이며, 각 outer train 내부의 과거방향 inner validation에서만 선택했다. 7일 purge와 event-boundary 제외 규칙을 적용하고 plateau와 spike singleton을 보호했다. 양성 이벤트가 fold 경계에서 잘리지 않도록 outer evaluation membership을 정할 때 label을 사용했지만, 그 뒤 outer truth 값은 모델·후처리·경계 파라미터 선택에 사용하지 않았다. 독립 validator가 candidate/base key·fold·label 정렬과 base prediction 동일성을 재검산했다.",
                },
                {
                    "id": "robustness",
                    "type": "markdown",
                    "sourceId": "independent_validation",
                    "body": f"## Robustness Checks Bound—but Do Not Remove—Uncertainty\n\npaired bootstrap은 {int(_number(bootstrap.get('replicates'), role='bootstrap replicates')):,}회 수행됐고 delta>0 비율은 {probability_improved:.1%}다. {long_sentence} station-layer 최악 하락과 정상 FP/day 증가는 별도 승격 gate로 제한했다. 다만 세 outer fold는 독립적인 세 번의 대회 test가 아니며 같은 2024–2025 train에서 나온 시간 분할이다. CAPA/PELT·CPOP-lite는 미래 문맥을 쓰는 offline QC이므로 causal 운영 성능을 입증하지 않으며, 이 보고서는 공식 검증기나 hidden-test 결과가 아니다.",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "sourceId": "report_derivation",
                    "body": f"## Recommended Next Steps\n\n{next_steps}",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": "## Further Questions\n\n- full-series 양방향 offline QC 허용 여부를 운영진에게 서면 확인해야 하는가?\n- 다음 연구는 R2 forecast/backcast residual인가, 아니면 실패한 gate에 대한 새 inner-only 가설인가?\n- 최종 후보 전 clean-environment 재현과 다중 seed 확인 범위를 어디까지 고정할 것인가?\n- 9월 7일 최종 모델 지정과 예측 업로드 잠금의 정확한 시각·순서는 무엇인가?",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": timestamp,
            "status": "ready",
            "datasets": {
                "headline": [headline_row],
                "fold_comparison": fold_rows,
                "promotion_gates": gate_rows,
            },
        },
        "sources": sources,
    }
    serialized = json.dumps(artifact, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if re.search(r"(?i)(?:[a-z]:[\\/]|file://|(?:^|[\\/])\.\.(?:[\\/]|$))", serialized):
        raise R1ReportArtifactError("portable artifact contains an absolute or parent path")
    return artifact


def write_artifact(artifact: Mapping[str, Any], output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    if destination.suffix.lower() != ".json":
        raise R1ReportArtifactError("--output must end in .json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    destination.write_text(payload + "\n", encoding="utf-8", newline="\n")
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--independent-validation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        artifact = build_artifact(
            args.metrics,
            args.independent_validation,
            args.manifest,
            args.preregistration,
            args.baseline_metrics,
        )
        output = write_artifact(artifact, args.output)
    except (OSError, KeyError, R1ReportArtifactError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: wrote aggregate-only canonical artifact to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
