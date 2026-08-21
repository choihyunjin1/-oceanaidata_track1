"""Build the aggregate-only portable report for the three sealed experiments.

The report deliberately reads only aggregate JSON receipts and hashes the three
frozen submissions.  It never reads source observations, row-level OOF files,
or hidden-label tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
REPORT_TITLE = "세 문제 최신 근거 실험 실행 결과"
DEFAULT_OUTPUT = Path("reports/generated/three_problem_execution_results_2026-08-21/artifact.json")

EXPECTED_SHA256 = {
    "p1_metrics": "4f53291d18ef94dcb6ef6f989cd8dc168a52fbcfabdf3dbf86d0bd42f552ed57",
    "p1_manifest": "1a2251a2000ed6dbdb975080085ba0f0722073a57d824abadbce0049476bc9c1",
    "p2_result": "3df6dd849cc244c087cf873196d6edef5fe9ad688606f9478a3d1ab43b5397fc",
    "p2_manifest": "09f2365f13e9d9c20461d41f7979560896bc313299dac1b6ed178f6dc6755a72",
    "p3_metrics": "a9423cbdb411d093067d4b4117a409825119e7b871b00f347f88d15a766734b2",
    "p3_receipt": "0799bfaa0d1492e08419d170fc63d211cf4f5fc22672b4177ede427aa5ef3f8d",
    "p1_submission": "28243fda9bc56e25a698366823dfab3198cda21bfaec04f30fda6a899eaf0cd3",
    "p2_submission": "1c959f818737850fd7fa9c6609ba3ae49dc9a470a269f7313119d840df1736bf",
    "p3_submission": "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
}

RELATIVE_PATHS = {
    "p1_metrics": Path("artifacts/p1_block_inpaint_v1/historical_metrics.json"),
    "p1_manifest": Path("artifacts/p1_block_inpaint_v1/manifest.json"),
    "p2_result": Path("artifacts/p2_tide_rts_v1/result.json"),
    "p2_manifest": Path("artifacts/p2_tide_rts_v1/manifest.json"),
    "p3_metrics": Path(
        "artifacts/p3_revin_patch_v1/recovery_181_sealed_blind/post_open/metrics.json"
    ),
    "p3_receipt": Path(
        "artifacts/p3_revin_patch_v1/recovery_181_sealed_blind/FINAL_IMMUTABLE_RECEIPT.json"
    ),
    "p1_submission": Path("submissions/frozen/P1_FROZEN_READY_TO_UPLOAD_28243fda.csv"),
    "p2_submission": Path("submissions/p2/P2_EXTRAPOLATED_SOFT_GATE_V2.csv"),
    "p3_submission": Path("submissions/p3_long_persistence_shrink/submission.csv"),
}


class ReportEvidenceError(RuntimeError):
    """Raised when a sealed aggregate receipt no longer matches its contract."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportEvidenceError(message)


def _close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ReportEvidenceError(f"{label} drifted: {actual!r} != {expected!r}")


def collect_evidence(root: Path) -> dict[str, Any]:
    """Load sealed aggregate receipts and verify the frozen submission hashes."""

    resolved = {name: root / path for name, path in RELATIVE_PATHS.items()}
    for name, path in resolved.items():
        _require(path.is_file(), f"missing required evidence: {name} ({RELATIVE_PATHS[name]})")
        actual = _sha256(path)
        _require(
            actual == EXPECTED_SHA256[name],
            f"sealed SHA mismatch for {name}: {actual} != {EXPECTED_SHA256[name]}",
        )

    p1_metrics = _read_json(resolved["p1_metrics"])
    p1_manifest = _read_json(resolved["p1_manifest"])
    p2_result = _read_json(resolved["p2_result"])
    p2_manifest = _read_json(resolved["p2_manifest"])
    p3_metrics = _read_json(resolved["p3_metrics"])
    p3_receipt = _read_json(resolved["p3_receipt"])

    _require(p1_manifest["decision"] == "failed_historical_gate", "unexpected P1 decision")
    _require(p1_metrics["outer_evaluation_count"] == 0, "P1 outer labels were opened")
    _require(p1_manifest["outer_evaluation_count"] == 0, "P1 manifest outer count drifted")
    _require(p2_result["decision"] == "NO_GO_PRECHECK", "unexpected P2 decision")
    _require(p2_result["precheck"]["passed"] is False, "P2 precheck state drifted")
    _require(p2_result["full_model_executed"] is False, "P2 full model unexpectedly ran")
    _require(p2_manifest["full_model_executed"] is False, "P2 manifest full-model drift")
    _require(p3_metrics["status"] == "gate_failed", "unexpected P3 decision")
    _require(
        p3_metrics["grain"] == {"rows": 1086, "cases": 181, "leads": [3, 6, 9, 12, 18, 24]},
        "P3 grain drifted",
    )
    _require(p3_receipt["family_cumulative_target_open_count"] == 2, "P3 open count drifted")
    _require(p3_receipt["metrics_generation_count"] == 1, "P3 metric count drifted")

    _close(p1_metrics["baseline"]["weighted"]["f1"], 0.5721820403993508, "P1 baseline")
    _close(p1_metrics["candidate"]["weighted"]["f1"], 0.5747732464951086, "P1 candidate")
    _close(
        p2_result["reference_metrics"]["adaptive_proxy_rmse"],
        0.7683674566216134,
        "P2 adaptive proxy",
    )
    _close(p3_metrics["gate"]["incumbent"]["rmse"], 0.779748041094144, "P3 incumbent")
    _close(p3_metrics["gate"]["candidate"]["rmse"], 0.7840617300585763, "P3 candidate")

    return {
        "p1": {"metrics": p1_metrics, "manifest": p1_manifest},
        "p2": {"result": p2_result, "manifest": p2_manifest},
        "p3": {"metrics": p3_metrics, "receipt": p3_receipt},
        "hashes": dict(EXPECTED_SHA256),
    }


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ReportEvidenceError("non-finite value cannot enter report SQL")
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _values_sql(rows: list[dict[str, object]], columns: list[str]) -> str:
    return " UNION ALL ".join(
        "SELECT " + ", ".join(f"{_sql_literal(row.get(column))} AS {column}" for column in columns)
        for row in rows
    )


def _inline_source(rows: list[dict[str, object]], columns: list[str]) -> dict[str, object]:
    return {
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": _values_sql(rows, columns),
        }
    }


def _table(
    *,
    table_id: str,
    title: str,
    subtitle: str,
    rows: list[dict[str, object]],
    columns: list[dict[str, object]],
) -> dict[str, object]:
    fields = [str(column["field"]) for column in columns]
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "dataset": table_id,
        "sourceId": "aggregate_receipt",
        "source": _inline_source(rows, fields),
        "density": "spacious",
        "columns": columns,
    }


def build_artifact(evidence: dict[str, Any], *, generated_at: str) -> dict[str, Any]:
    p1 = evidence["p1"]["metrics"]
    p2 = evidence["p2"]["result"]
    p3 = evidence["p3"]["metrics"]
    p1_base = float(p1["baseline"]["weighted"]["f1"])
    p1_candidate = float(p1["candidate"]["weighted"]["f1"])
    p3_base = float(p3["gate"]["incumbent"]["rmse"])
    p3_candidate = float(p3["gate"]["candidate"]["rmse"])
    p2_layer_r2 = p2["precheck"]["aggregate_residual_r2"]

    comparison_rows = [
        {
            "evidence": "P1 block-inpaint",
            "problem": "P1",
            "normalized_signed_improvement": (p1_candidate - p1_base) / p1_base,
            "metric": "weighted F1 (higher is better)",
            "incumbent_value": p1_base,
            "candidate_value": p1_candidate,
            "stage": "historical gate",
        },
        *[
            {
                "evidence": f"P2 tide-RTS {layer.upper()}",
                "problem": "P2",
                "normalized_signed_improvement": float(p2_layer_r2[f"temp_{layer}"]),
                "metric": "residual MSE skill R2 (higher is better)",
                "incumbent_value": 1.0,
                "candidate_value": 1.0 - float(p2_layer_r2[f"temp_{layer}"]),
                "stage": "recoverability precheck",
            }
            for layer in ("l2", "l3", "l4")
        ],
        {
            "evidence": "P3 RevIN/Patch",
            "problem": "P3",
            "normalized_signed_improvement": (p3_base - p3_candidate) / p3_base,
            "metric": "RMSE (lower is better)",
            "incumbent_value": p3_base,
            "candidate_value": p3_candidate,
            "stage": "181-case sealed-blind recovery",
        },
    ]

    summary_rows = [
        {
            "problem": "P1",
            "experiment": "block-inpaint",
            "evidence_stage": "historical gate only",
            "scope": "152805 rows / 3 chronological blocks",
            "metric": "test-share weighted F1",
            "incumbent": f"{p1_base:.15f}",
            "candidate": f"{p1_candidate:.15f}",
            "delta": f"{p1['gates']['weighted_f1_delta']:+.15f}",
            "decision": "REJECT; outer evaluation count = 0",
        },
        {
            "problem": "P2",
            "experiment": "tide-aware low-rank RTS",
            "evidence_stage": "recoverability precheck only",
            "scope": "3 jointly masked 61-day pseudo-blocks",
            "metric": "temperature residual R2 by layer",
            "incumbent": f"adaptive proxy RMSE {p2['reference_metrics']['adaptive_proxy_rmse']:.15f} C",
            "candidate": "not produced",
            "delta": "L2/L3/L4 R2 all below 0",
            "decision": "NO-GO; full-model count = 0",
        },
        {
            "problem": "P3",
            "experiment": "three-seed RevIN/Patch blend",
            "evidence_stage": "sealed-blind exact-key recovery",
            "scope": "181 cases / 1086 lead rows",
            "metric": "pooled RMSE (m)",
            "incumbent": f"{p3_base:.15f}",
            "candidate": f"{p3_candidate:.15f}",
            "delta": f"{p3['gate']['delta_rmse']:+.15f}",
            "decision": "REJECT; family opens = 2, metrics = 1",
        },
    ]

    p2_blocks = p2["precheck"]["by_block"]
    gate_rows = [
        {
            "problem": "P1",
            "stage": "historical gate",
            "gate": "weighted F1 delta",
            "requirement": ">= +0.005",
            "observed": f"{p1['gates']['weighted_f1_delta']:+.15f}",
            "status": "FAIL",
        },
        {
            "problem": "P1",
            "stage": "historical gate",
            "gate": "paired bootstrap CI90 lower",
            "requirement": "> 0",
            "observed": f"{p1['gates']['bootstrap_ci90_lower']:+.15f}",
            "status": "FAIL",
        },
        {
            "problem": "P1",
            "stage": "historical gate",
            "gate": "normal FP/day relative increase",
            "requirement": "< 0.10",
            "observed": f"{p1['gates']['normal_fp_day_relative_increase']:.15f}",
            "status": "FAIL",
        },
        {
            "problem": "P1",
            "stage": "historical gate",
            "gate": "worst station-layer F1 delta",
            "requirement": ">= -0.01",
            "observed": f"{p1['gates']['worst_station_layer_f1_delta']:+.15f}",
            "status": "FAIL",
        },
        {
            "problem": "P2",
            "stage": "precheck",
            "gate": "each block pooled temperature residual R2",
            "requirement": "all > 0",
            "observed": "; ".join(
                f"{name}={block['residual_skill']['pooled_temperature_r2']:+.15f}"
                for name, block in p2_blocks.items()
            ),
            "status": "FAIL",
        },
        {
            "problem": "P2",
            "stage": "precheck",
            "gate": "each aggregate temperature-layer residual R2",
            "requirement": "all > 0",
            "observed": "; ".join(
                f"{layer.upper()}={p2_layer_r2[f'temp_{layer}']:+.15f}"
                for layer in ("l2", "l3", "l4")
            ),
            "status": "FAIL",
        },
        {
            "problem": "P2",
            "stage": "precheck",
            "gate": "finite-horizon observability rank",
            "requirement": "rank = state dimension in all blocks",
            "observed": "3/3 blocks: 6/6",
            "status": "PASS",
        },
        {
            "problem": "P2",
            "stage": "precheck",
            "gate": "observability condition",
            "requirement": "all <= 100000000",
            "observed": f"max={max(block['observability']['condition'] for block in p2_blocks.values()):.15f}",
            "status": "PASS",
        },
        {
            "problem": "P2",
            "stage": "precheck",
            "gate": "public-state support share",
            "requirement": "all >= 0.80",
            "observed": f"min={min(block['support']['validation_supported_share'] for block in p2_blocks.values()):.15f}",
            "status": "FAIL",
        },
        {
            "problem": "P2",
            "stage": "precheck",
            "gate": "two-public-temperature coverage",
            "requirement": "all >= 0.95",
            "observed": f"min={min(block['coverage']['two_public_temperature_share'] for block in p2_blocks.values()):.15f}",
            "status": "FAIL",
        },
        {
            "problem": "P2",
            "stage": "conditional full model",
            "gate": "candidate RMSE / bootstrap / block / L4 promotion",
            "requirement": "RMSE <= 0.763367 and all preregistered safeguards",
            "observed": "not evaluated after terminal precheck NO-GO",
            "status": "NOT RUN",
        },
        {
            "problem": "P3",
            "stage": "181-case sealed-blind recovery",
            "gate": "delta RMSE",
            "requirement": "<= -0.010",
            "observed": f"{p3['gate']['delta_rmse']:+.15f}",
            "status": "FAIL",
        },
        {
            "problem": "P3",
            "stage": "181-case sealed-blind recovery",
            "gate": "case-bootstrap CI90 upper",
            "requirement": "< 0",
            "observed": f"{p3['case_bootstrap']['ci90'][1]:+.15f}",
            "status": "FAIL",
        },
        {
            "problem": "P3",
            "stage": "181-case sealed-blind recovery",
            "gate": "episode-bootstrap CI90 upper",
            "requirement": "< 0",
            "observed": f"{p3['episode_bootstrap']['ci90'][1]:+.15f}",
            "status": "FAIL",
        },
        {
            "problem": "P3",
            "stage": "181-case sealed-blind recovery",
            "gate": "+18h non-degrading",
            "requirement": "candidate RMSE <= incumbent RMSE",
            "observed": f"delta={p3['gate']['candidate']['by_lead']['18'] - p3['gate']['incumbent']['by_lead']['18']:+.15f}",
            "status": "FAIL",
        },
        {
            "problem": "P3",
            "stage": "181-case sealed-blind recovery",
            "gate": "+24h non-degrading",
            "requirement": "candidate RMSE <= incumbent RMSE",
            "observed": f"delta={p3['gate']['candidate']['by_lead']['24'] - p3['gate']['incumbent']['by_lead']['24']:+.15f}",
            "status": "PASS",
        },
        {
            "problem": "P3",
            "stage": "181-case sealed-blind recovery",
            "gate": "station degradation",
            "requirement": "each delta <= +0.010",
            "observed": f"worst={max(p3['gate']['candidate']['by_station'][s] - p3['gate']['incumbent']['by_station'][s] for s in p3['gate']['candidate']['by_station']):+.15f}",
            "status": "PASS",
        },
    ]

    frozen_rows = [
        {
            "problem": problem,
            "artifact": artifact,
            "sha256": evidence["hashes"][key],
            "unchanged": "YES",
        }
        for problem, artifact, key in (
            ("P1", "frozen QC submission", "p1_submission"),
            ("P2", "adaptive soft-gate v2 submission", "p2_submission"),
            ("P3", "long-lead persistence-shrink submission", "p3_submission"),
        )
    ]

    chart = {
        "id": "normalized_signed_improvement",
        "title": "승격 증거의 방향 통일 비교",
        "subtitle": (
            "양수는 개선. P1은 weighted F1 상대 증가, P3는 RMSE 상대 감소; "
            "P2는 최종 RMSE가 없어 layer별 residual-MSE R2를 표시"
        ),
        "type": "bar",
        "dataset": "normalized_signed_improvement",
        "sourceId": "aggregate_receipt",
        "source": _inline_source(
            comparison_rows,
            [
                "evidence",
                "problem",
                "normalized_signed_improvement",
                "metric",
                "incumbent_value",
                "candidate_value",
                "stage",
            ],
        ),
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": "evidence", "type": "nominal", "label": "근거"},
            "y": {
                "field": "normalized_signed_improvement",
                "type": "quantitative",
                "label": "정규화 signed improvement",
            },
            "tooltip": [
                {"field": "metric", "type": "nominal", "label": "지표"},
                {"field": "incumbent_value", "type": "quantitative", "label": "기준"},
                {"field": "candidate_value", "type": "quantitative", "label": "후보/정규화 MSE"},
                {"field": "stage", "type": "nominal", "label": "단계"},
            ],
        },
    }

    summary_table = _table(
        table_id="experiment_summary",
        title="세 실험의 실제 평가 단계",
        subtitle="서로 다른 검증 단계를 공식 hidden score처럼 혼합하지 않음",
        rows=summary_rows,
        columns=[
            {"field": "problem", "label": "문제", "type": "text"},
            {"field": "experiment", "label": "실험", "type": "text"},
            {"field": "evidence_stage", "label": "근거 단계", "type": "text"},
            {"field": "scope", "label": "범위", "type": "text"},
            {"field": "metric", "label": "지표", "type": "text"},
            {"field": "incumbent", "label": "기준", "type": "text"},
            {"field": "candidate", "label": "후보", "type": "text"},
            {"field": "delta", "label": "차이", "type": "text"},
            {"field": "decision", "label": "판정", "type": "text"},
        ],
    )
    gate_table = _table(
        table_id="exact_gate_table",
        title="사전 고정 gate — exact 판정표",
        subtitle="NOT RUN은 실패가 아니라 앞 단계 terminal gate로 평가 자체가 열리지 않았음을 뜻함",
        rows=gate_rows,
        columns=[
            {"field": "problem", "label": "문제", "type": "text"},
            {"field": "stage", "label": "단계", "type": "text"},
            {"field": "gate", "label": "검사", "type": "text"},
            {"field": "requirement", "label": "요구조건", "type": "text"},
            {"field": "observed", "label": "관측값", "type": "text"},
            {"field": "status", "label": "상태", "type": "text"},
        ],
    )
    frozen_table = _table(
        table_id="frozen_integrity",
        title="동결 제출 무결성",
        subtitle="보고서 작성 시점에 실제 파일 SHA-256을 재계산해 sealed 값과 대조",
        rows=frozen_rows,
        columns=[
            {"field": "problem", "label": "문제", "type": "text"},
            {"field": "artifact", "label": "동결 파일", "type": "text"},
            {"field": "sha256", "label": "SHA-256", "type": "text"},
            {"field": "unchanged", "label": "불변", "type": "text"},
        ],
    )

    sources = [
        {
            "id": "aggregate_receipt",
            "label": "세 실험 aggregate-only report receipt",
            "path": str(DEFAULT_OUTPUT).replace("\\", "/"),
            "query": {
                "description": "Sealed aggregate JSON receipts and frozen-file hashes only; no observation rows.",
                "tables_used": [str(path).replace("\\", "/") for path in RELATIVE_PATHS.values()],
                "filters": [
                    "raw observation rows = 0",
                    "official hidden labels = 0",
                    "external observation values = 0",
                    "experiment artifact mutation = 0",
                ],
            },
        },
        {
            "id": "p1_result",
            "label": "P1 block-inpaint historical gate",
            "path": str(RELATIVE_PATHS["p1_metrics"]).replace("\\", "/"),
            "query": {
                "description": "Aggregate historical metrics; three chronological folds and zero outer evaluations.",
                "tables_used": [str(RELATIVE_PATHS["p1_metrics"]).replace("\\", "/")],
                "filters": ["historical gate only", "outer_evaluation_count = 0"],
            },
        },
        {
            "id": "p2_result",
            "label": "P2 tide-RTS public-only recoverability precheck",
            "path": str(RELATIVE_PATHS["p2_result"]).replace("\\", "/"),
            "query": {
                "description": "Aggregate public-layer observability and residual-skill diagnostics.",
                "tables_used": [str(RELATIVE_PATHS["p2_result"]).replace("\\", "/")],
                "filters": ["target L2-L4 temp+psal jointly masked", "full_model_executed = false"],
            },
        },
        {
            "id": "p3_result",
            "label": "P3 181-case sealed-blind recovery",
            "path": str(RELATIVE_PATHS["p3_metrics"]).replace("\\", "/"),
            "query": {
                "description": "Aggregate exact-key recovery metrics and immutable target-access receipt.",
                "tables_used": [
                    str(RELATIVE_PATHS["p3_metrics"]).replace("\\", "/"),
                    str(RELATIVE_PATHS["p3_receipt"]).replace("\\", "/"),
                ],
                "filters": [
                    "181 exact-intersection cases",
                    "family opens = 2",
                    "metrics generations = 1",
                ],
            },
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "aggregate_receipt",
            "body": (
                "## Technical Summary\n\n"
                "- **어느 신규 family도 동결 제출을 대체하지 않습니다.** P1은 역사 구간에서 소폭 F1을 올렸지만 4개 승격 gate를 모두 실패했고, P2는 public-only recoverability precheck에서 중단됐으며, P3는 exact 181-case sealed-blind 평가에서 RMSE가 악화했습니다.\n"
                "- **평가 단계는 엄격히 분리합니다.** P1은 historical gate만 열었고 outer evaluation은 0회입니다. P2는 precheck만 실행했고 Kalman/RTS full model은 0회입니다. P3는 첫 key mismatch 뒤 승인된 181-case 복구를 거쳐 family target-open 누적 2회, metrics 생성 1회로 봉인됐습니다.\n"
                "- **독립 QA는 큰 구현 오류 정황을 찾지 못했습니다.** P1 headline·group·event·type·bootstrap·gate 재계산은 수치 오차 0이었고, P2 result/config SHA는 nonsemantic formatter 보정 뒤에도 유지됐으며, P3 exact-key·3-seed·protected short leads·frozen SHA가 receipt와 일치했습니다.\n"
                "- **공식 hidden 성능 주장이 아닙니다.** 세 결과는 각 로컬 validation 계약의 근거이며 제출 업로드, 외부 관측값 사용, 실험 artifact 수정은 없었습니다."
            ),
        },
        {
            "id": "visual_intro",
            "type": "markdown",
            "body": (
                "## 방향을 통일하면 P1의 작은 이득과 P2·P3의 실패가 함께 보입니다\n\n"
                "아래 한 개 차트는 양수가 개선이 되도록 방향만 맞춥니다. P1은 weighted F1이 높을수록 좋아 상대 증가율을 사용하고, P3는 RMSE가 낮을수록 좋아 상대 감소율을 사용합니다. P2도 최종 목표는 RMSE 최소화지만 precheck에서 종료되어 candidate RMSE가 없습니다. 따라서 P2 막대는 frozen residual을 1로 둔 후보 residual-MSE의 개선률, 즉 layer별 R²이며 세 층을 모두 표시합니다."
            ),
        },
        {"id": "comparison_chart", "type": "chart", "chartId": chart["id"]},
        {
            "id": "summary_intro",
            "type": "markdown",
            "body": "## 결과 범위와 생명주기\n\nP1의 historical 지표, P2의 구조적 precheck, P3의 sealed-blind 지표는 서로 다른 증거 단계입니다. 아래 표는 이 차이를 숨기지 않고 후보 생성 여부와 다음 단계 개방 횟수를 함께 표시합니다.",
        },
        {"id": "summary_table", "type": "table", "tableId": summary_table["id"]},
        {
            "id": "p1_findings",
            "type": "markdown",
            "sourceId": "p1_result",
            "body": (
                "## P1 — 탐지 민감도 증가는 false positive와 최악 그룹 손실을 상쇄하지 못했습니다\n\n"
                f"동일 역사 구간에서 micro F1은 {p1['baseline']['micro']['f1']:.6f}→{p1['candidate']['micro']['f1']:.6f}, weighted F1은 {p1_base:.6f}→{p1_candidate:.6f}로 올랐습니다. Spike·noise·offset·drift recall도 모두 증가했고 flatline recall은 1.0을 유지했습니다. 그러나 event F1은 {p1['baseline']['events']['f1']:.6f}→{p1['candidate']['events']['f1']:.6f}로 낮아졌고 정상 station-layer-day당 FP는 {p1['normal_station_layer_day_fp']['baseline']['false_positive_rows_per_normal_station_layer_day']:.6f}→{p1['normal_station_layer_day_fp']['candidate']['false_positive_rows_per_normal_station_layer_day']:.6f}로 27.87% 증가했습니다. 세 fold 중 두 fold가 weighted F1에서 악화했고 최악 그룹은 G-ORS layer 1의 -0.058874였습니다.\n\n"
                "**판정:** historical gate 4개가 모두 실패했으므로 outer label은 한 번도 열지 않았습니다. 이 결과는 block-inpaint family의 현재 고정 설정만 기각합니다."
            ),
        },
        {
            "id": "p2_findings",
            "type": "markdown",
            "sourceId": "p2_result",
            "body": (
                "## P2 — 관측가능 rank는 충분했지만 숨은 중층 residual의 예측가능성은 없었습니다\n\n"
                f"세 block의 finite-horizon Gramian은 모두 6/6 full rank이고 최대 condition은 {max(block['observability']['condition'] for block in p2_blocks.values()):.3f}로 수치상 안정적이었습니다. 그러나 aggregate temperature residual R²는 L2 {p2_layer_r2['temp_l2']:.6f}, L3 {p2_layer_r2['temp_l3']:.6f}, L4 {p2_layer_r2['temp_l4']:.6f}로 모두 음수였습니다. 2025-11~12 block은 support {p2_blocks['2025_nov_dec']['support']['validation_supported_share']:.3f}, two-public-temperature coverage {p2_blocks['2025_nov_dec']['coverage']['two_public_temperature_share']:.3f}로도 gate를 실패했습니다.\n\n"
                "**판정:** precheck가 terminal NO-GO여서 joint T/S factor + 12.42h resonator + RTS smoother를 학습·평가하지 않았습니다. 따라서 0.768367이라는 frozen adaptive proxy RMSE와 비교할 신규 candidate RMSE도 없습니다."
            ),
        },
        {
            "id": "p3_findings",
            "type": "markdown",
            "sourceId": "p3_result",
            "body": (
                "## P3 — +24h의 작은 이득보다 +12h·+18h 손실이 컸습니다\n\n"
                f"Exact 181-case intersection에서 incumbent RMSE {p3_base:.6f}m 대비 candidate는 {p3_candidate:.6f}m로 {p3['gate']['delta_rmse']:+.6f}m 악화했습니다. +3/+6/+9h는 bit-exact 보호됐고 +24h는 0.000821m 개선됐지만 +12h와 +18h는 각각 약 0.01185m 악화했습니다. Case bootstrap CI90은 [{p3['case_bootstrap']['ci90'][0]:+.6f}, {p3['case_bootstrap']['ci90'][1]:+.6f}], episode bootstrap은 [{p3['episode_bootstrap']['ci90'][0]:+.6f}, {p3['episode_bootstrap']['ci90'][1]:+.6f}]로 모두 악화 방향입니다.\n\n"
                "**판정:** 승격하지 않습니다. 첫 target open은 1개 outer key mismatch로 metric을 만들지 못했고, 승인된 exact-intersection 복구 1회만 추가했습니다. 누적 open 2회·metrics 생성 1회 이후 family는 봉인되어 세 번째 open이나 두 번째 metric 생성은 금지됩니다."
            ),
        },
        {
            "id": "definitions",
            "type": "markdown",
            "body": (
                "## Scope and Metric Definitions\n\n"
                "- **P1 weighted F1:** 동일 historical validation rows의 station-layer F1을 2026 test row share로 재가중한 값이며 높을수록 좋습니다. 공식 hidden F1이 아닙니다.\n"
                "- **P2 residual R²:** frozen adaptive temperature prediction이 남긴 residual을 public-only state로 예측했을 때, zero-correction residual MSE 대비 개선률입니다. 0 이하는 보정을 하지 않는 것보다 나쁨을 뜻합니다. P2의 공식 선택 지표는 여전히 pooled temperature RMSE이며 낮을수록 좋습니다.\n"
                "- **P3 RMSE:** 181 exact-intersection cases × six leads의 pooled significant-wave-height RMSE(m)이며 낮을수록 좋습니다. Delta와 bootstrap은 candidate minus incumbent라 음수가 개선입니다."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## Methodology and Validation Design\n\n"
                "1. P1은 7일 purge를 둔 세 historical chronological fold에서 normal-only dual-flank masked reconstruction을 학습했고, calibration labels는 고정 additive gate에만 사용했습니다. Historical gate 통과 전 outer labels를 열지 않는 계약을 지켰습니다.\n"
                "2. P2는 세 outer pseudo-mask block에서 target L2-L4 temp와 psal을 61일 동안 동시에 가리고 ±7일 purge를 적용했습니다. Public L1/L5-L8 T/S와 실제 depth만으로 observability, support, coverage, residual R²를 계산했습니다.\n"
                "3. P3는 candidate와 incumbent의 exact-key 181-case intersection만 봉인 평가했습니다. 세 seed 평균과 0.2 patch blend는 target open 전에 고정됐고 +3/+6/+9h는 incumbent와 bit-exact였습니다.\n"
                "4. 이 보고서 빌더는 aggregate JSON receipts만 읽고 세 동결 submission 파일은 SHA-256만 계산합니다. 원자료·row-level OOF·평가 행·외부 관측값은 보고서 입력과 출력에 0건입니다."
            ),
        },
        {
            "id": "gate_intro",
            "type": "markdown",
            "body": "## 정확한 승격 조건과 판정\n\n평균 headline 하나만으로 결론을 바꾸지 않도록 실행 전에 고정한 조건을 그대로 나열합니다. P2 full-model promotion rows는 실행되지 않았으므로 실패가 아니라 NOT RUN입니다.",
        },
        {"id": "gate_table", "type": "table", "tableId": gate_table["id"]},
        {
            "id": "qa_caveats",
            "type": "markdown",
            "sourceId": "aggregate_receipt",
            "body": (
                "## Independent QA Caveats\n\n"
                "- **P1:** 독립 read-only 재계산은 headline, group, event, anomaly-type recall, FP/day, paired bootstrap, 네 gate와 historical key grain을 모두 수치 오차 0으로 재현했습니다. 다만 별도 OS-level file-access audit log가 없어 프로세스 외부의 label read 부재까지 법의학적으로 증명하지는 못합니다. Status gauge의 elapsed 721.583초는 manifest 실제 runtime 153.818초와 다르며, 이전 gauge start 재사용에서 생긴 저위험 provenance 결함입니다.\n"
                "- **P2:** 2025-11~12 support/coverage는 official scored timestamps가 아니라 full 61-day grid에서 계산되어 보수적입니다. 그러나 세 temperature-layer R²와 세 fold pooled R²가 모두 gate 방향을 충족하지 못해 이 caveat와 독립적으로 NO-GO입니다. 실행 후 두 Python 파일에 Ruff format만 적용했고 result/config SHA는 불변이며 precheck를 재실행하지 않았다는 amendment가 manifest에 남아 있습니다.\n"
                "- **P3:** 첫 open은 one-key mismatch를 발견한 뒤 metric 0개로 끝났습니다. 현재 지표는 승인된 181-case exact intersection recovery의 유일한 metrics generation입니다. 따라서 182-case 절대 threshold는 reference-only이고 canonical 판정에는 사용할 수 없습니다. 세 번째 target open과 두 번째 metrics generation은 금지됩니다."
            ),
        },
        {"id": "frozen_table", "type": "table", "tableId": frozen_table["id"]},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations and Robustness\n\n"
                "- P1 historical blocks와 P2 pseudo-mask blocks는 공식 hidden 구간이 아니며 이전 연구 계보에서 관련 validation 구조가 알려진 자료입니다.\n"
                "- P1의 작은 aggregate 개선은 CI, FP/day, worst-group, fold 안정성을 통과하지 못했습니다. 따라서 모델 family 전체의 무효가 아니라 현재 고정 gate의 일반화 근거 부족으로 해석합니다.\n"
                "- P2 precheck는 final model 성능을 측정하지 않습니다. 음의 residual R²는 현재 public-only linear recoverability 설계의 비식별성을 보이지만 모든 state-space 또는 nonlinear operator를 이론적으로 기각하지 않습니다.\n"
                "- P3 recovery는 200-case 공식 test가 아니라 local 181-case sealed intersection입니다. Candidate-only/incumbent-only 한 case씩을 제외했으므로 결과를 182-case 이전 기준과 섞지 않습니다.\n"
                "- 세 frozen SHA 불변은 파일 무결성을 뜻할 뿐 공식 leaderboard 성능을 보증하지 않습니다."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended Next Steps\n\n"
                "- **세 동결 제출을 유지합니다.** 이번 세 실험으로 submission CSV, 모델, checkpoint, router를 바꾸지 않습니다.\n"
                "- **P1은 outer를 열지 않습니다.** 같은 block-inpaint family를 historical 결과에 맞춰 즉석 튜닝하지 말고, 새로운 사전등록 family 또는 공식 feedback 단위를 기다립니다.\n"
                "- **P2는 tide-RTS full model을 실행하지 않습니다.** Public-only recoverability가 양수가 되는 새로운 식별가능성 근거가 생기기 전에는 같은 factor/window/grid를 재탐색하지 않습니다.\n"
                "- **P3 RevIN/Patch family는 봉인합니다.** 세 번째 target open이나 두 번째 metric 생성을 하지 않고, +12/+18h 손실 원인은 aggregate evidence로만 보존합니다."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## Further Questions\n\n"
                "- P1에서 spike/offset/drift recall 이득을 유지하면서 normal FP/day와 G-ORS L1 손실을 동시에 제한할 새 관측가능 신호가 있는가?\n"
                "- P2에서 public profile만으로 61-day 중층 residual이 음의 R²라면, target boundary continuity를 독립적으로 검증할 새 masking 계약을 만들 수 있는가?\n"
                "- P3에서 +24h 이득을 보존하면서 +12/+18h 손실을 막을 구조를 target 재개방 없이 사전 고정할 충분한 train-only 증거가 있는가?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "P1 block-inpaint, P2 tide-RTS precheck, P3 sealed-blind RevIN/Patch recovery의 aggregate-only 기술 보고서",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [],
            "charts": [chart],
            "tables": [summary_table, gate_table, frozen_table],
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "normalized_signed_improvement": comparison_rows,
                "experiment_summary": summary_rows,
                "exact_gate_table": gate_rows,
                "frozen_integrity": frozen_rows,
            },
            "accessIssues": [],
        },
        "sources": [
            {"id": source["id"], "label": source["label"], "path": source["path"]}
            for source in sources
        ],
        "package_info": {
            "originUrl": "artifact://three-problem-execution-results-2026-08-21",
            "controls": {"edit": False, "refresh": False},
        },
    }
    _validate_aggregate_artifact(artifact)
    return artifact


def _validate_aggregate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact["surface"] == "report", "surface must be report")
    _require(
        artifact["manifest"]["blocks"][0]["body"] == f"# {REPORT_TITLE}", "title block mismatch"
    )
    _require(len(artifact["manifest"]["charts"]) == 1, "report must contain exactly one chart")
    _require(
        len(artifact["snapshot"]["datasets"]["normalized_signed_improvement"]) == 5,
        "comparison row count drifted",
    )
    serialized = json.dumps(artifact, ensure_ascii=False)
    _require(
        "C:/Users/" not in serialized and "C:\\Users\\" not in serialized, "personal path leaked"
    )
    for forbidden in ("station,time,temp", "case_id,station,lead_h", "target_hs"):
        _require(forbidden not in serialized, f"row-level schema leaked: {forbidden}")
    for source in artifact["sources"]:
        _require(not Path(source["path"]).is_absolute(), "absolute source path leaked")


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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(output), "sha256": _sha256(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
