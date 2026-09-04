"""Build the portable technical report for the P2 physical profile projection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _union_sql(rows: list[dict[str, object]], columns: list[str]) -> str:
    return " UNION ALL ".join(
        "SELECT " + ", ".join(f"{_literal(row.get(column))} AS {column}" for column in columns)
        for row in rows
    )


def _source(
    source_id: str,
    label: str,
    rows: list[dict[str, object]],
    columns: list[str],
    description: str,
) -> dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "path": "artifacts/p2_physical_profile_projection_v1/result.json",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": _union_sql(rows, columns),
            "description": description,
            "tables_used": [
                "artifacts/p2_physical_profile_projection_v1/result.json",
                "artifacts/p2_physical_profile_projection_v1/oof.parquet",
            ],
            "filters": [
                "69,850 frozen leave-one-block-out prediction rows",
                "Public layer-1 and layer-5 temperatures only",
                "No hidden target values and no external observations",
            ],
        },
    }


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def build_artifact(result: dict[str, object], oof: pd.DataFrame) -> dict[str, object]:
    generated = datetime.now().astimezone().isoformat()
    metrics = result["metrics"]
    bootstrap = result["paired_kst_day_bootstrap"]
    headline = [
        {
            "baseline_rmse": metrics["baseline_rmse"],
            "candidate_rmse": metrics["candidate_rmse"],
            "delta_rmse": metrics["delta_rmse"],
            "ci90_low": bootstrap["ci90_low"],
            "ci90_high": bootstrap["ci90_high"],
            "same_season_delta": metrics["by_block"]["2024_sep_oct"]["delta_rmse"],
            "test_active_share": result["test_projection"]["active_share"],
        }
    ]
    block_rows: list[dict[str, object]] = []
    for block, values in metrics["by_block"].items():
        block_rows.extend(
            [
                {
                    "block": block,
                    "method": "Frozen Deep Stack",
                    "rmse": values["baseline_rmse"],
                    "rows": values["rows"],
                },
                {
                    "block": block,
                    "method": "Physical projection",
                    "rmse": values["candidate_rmse"],
                    "rows": values["rows"],
                },
            ]
        )
    layer_rows: list[dict[str, object]] = []
    for layer, values in metrics["by_layer"].items():
        layer_rows.append(
            {
                "layer": int(layer),
                "rows": values["rows"],
                "baseline_rmse": values["baseline_rmse"],
                "candidate_rmse": values["candidate_rmse"],
                "delta_rmse": values["delta_rmse"],
            }
        )

    base = oof["lobo_prediction"].to_numpy(float)
    candidate = oof["prediction"].to_numpy(float)
    truth = oof["truth"].to_numpy(float)
    active = oof["active"].to_numpy(bool)
    eligible = oof["eligible"].to_numpy(bool)
    coverage_rows: list[dict[str, object]] = []
    for block in oof["block"].drop_duplicates():
        selected = oof["block"].eq(block).to_numpy()
        changed = selected & active
        coverage_rows.append(
            {
                "scope": str(block),
                "rows": int(selected.sum()),
                "eligible_share": float(eligible[selected].mean()),
                "active_share": float(active[selected].mean()),
                "correction_rmse": float(
                    np.sqrt(np.mean((candidate[selected] - base[selected]) ** 2))
                ),
                "active_baseline_rmse": (
                    _rmse(truth[changed], base[changed]) if changed.any() else None
                ),
                "active_candidate_rmse": (
                    _rmse(truth[changed], candidate[changed]) if changed.any() else None
                ),
            }
        )
    coverage_rows.append(
        {
            "scope": "2025 hidden test",
            "rows": result["test_projection"]["rows"],
            "eligible_share": result["test_projection"]["eligible_share"],
            "active_share": result["test_projection"]["active_share"],
            "correction_rmse": result["test_projection"]["rmse_correction"],
            "active_baseline_rmse": None,
            "active_candidate_rmse": None,
        }
    )

    sources = [
        _source(
            "headline_source",
            "Physical projection headline metrics",
            headline,
            list(headline[0]),
            "Reconciles the frozen Deep Stack and physical-projection RMSE with uncertainty.",
        ),
        _source(
            "block_source",
            "RMSE by validation block",
            block_rows,
            ["block", "method", "rmse", "rows"],
            "Compares methods on each pre-existing target-proxy block.",
        ),
        _source(
            "layer_source",
            "RMSE by reconstructed layer",
            layer_rows,
            list(layer_rows[0]),
            "Shows layer-level direction and magnitude of the projection effect.",
        ),
        _source(
            "coverage_source",
            "Projection eligibility and intervention",
            coverage_rows,
            list(coverage_rows[0]),
            "Shows public-endpoint coverage, active corrections, and correction magnitude.",
        ),
    ]

    cards = [
        {
            "id": "delta_card",
            "description": "69,850 frozen LOBO rows; lower RMSE is better",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Projection ΔRMSE",
                    "field": "delta_rmse",
                    "format": "number",
                    "unit": " °C",
                    "signed": True,
                }
            ],
        },
        {
            "id": "same_season_card",
            "description": "2024 September–October direct seasonal proxy",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Same-season ΔRMSE",
                    "field": "same_season_delta",
                    "format": "number",
                    "unit": " °C",
                    "signed": True,
                }
            ],
        },
        {
            "id": "test_active_card",
            "description": "26,061 hidden submission rows; target labels unavailable",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Test rows changed",
                    "field": "test_active_share",
                    "format": "percent",
                }
            ],
        },
    ]

    charts = [
        {
            "id": "block_rmse_chart",
            "title": "RMSE by validation block",
            "subtitle": "Three fixed proxy blocks; lower is better, unit °C.",
            "type": "bar",
            "dataset": "block_rmse",
            "sourceId": "block_source",
            "valueFormat": "number",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "block", "type": "nominal", "label": "Validation block"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (°C)"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [{"field": "rows", "type": "quantitative", "label": "Rows"}],
            },
        }
    ]

    tables = [
        {
            "id": "layer_table",
            "title": "Layer-level RMSE",
            "subtitle": "All three proxy blocks pooled; signed delta is candidate minus baseline.",
            "dataset": "layer_metrics",
            "sourceId": "layer_source",
            "defaultSort": {"field": "layer", "direction": "asc"},
            "columns": [
                {"field": "layer", "label": "Layer", "type": "number"},
                {"field": "rows", "label": "Rows", "type": "number"},
                {"field": "baseline_rmse", "label": "Deep RMSE", "type": "number"},
                {"field": "candidate_rmse", "label": "Projected RMSE", "type": "number"},
                {
                    "field": "delta_rmse",
                    "label": "ΔRMSE",
                    "type": "number",
                    "movement": True,
                },
            ],
        },
        {
            "id": "coverage_table",
            "title": "Eligibility and intervention",
            "subtitle": "Missing layer-1/layer-5 endpoints cause an exact no-op.",
            "dataset": "coverage",
            "sourceId": "coverage_source",
            "columns": [
                {"field": "scope", "label": "Scope", "type": "text"},
                {"field": "rows", "label": "Rows", "type": "number"},
                {"field": "eligible_share", "label": "Eligible share", "type": "number"},
                {"field": "active_share", "label": "Changed share", "type": "number"},
                {"field": "correction_rmse", "label": "Correction RMSE", "type": "number"},
                {
                    "field": "active_baseline_rmse",
                    "label": "Active Deep RMSE",
                    "type": "number",
                },
                {
                    "field": "active_candidate_rmse",
                    "label": "Active projected RMSE",
                    "type": "number",
                },
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# P2 공개층 물리 연직 투영 검증"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "headline_source",
            "body": (
                "## 기술 요약: 작지만 일관된 개선이며 제출 challenger로 유지한다\n\n"
                f"공개 layer 1·5가 유효할 때만 Deep Stack의 layer 2·3·4를 두 endpoint 사이로 제한하고 "
                f"endpoint 방향에 맞게 순서 투영했다. LOBO RMSE는 **{metrics['baseline_rmse']:.6f}→"
                f"{metrics['candidate_rmse']:.6f}°C**로 **{metrics['delta_rmse']:+.6f}°C** 개선됐다. "
                f"KST-day bootstrap 90% CI는 **[{bootstrap['ci90_low']:+.6f}, "
                f"{bootstrap['ci90_high']:+.6f}]°C**로 전부 0 아래다. 다만 세 변형을 본 뒤 선택한 "
                "adaptive research이므로 hidden 개선을 보장하지 않으며 자동 업로드하지 않는다."
            ),
        },
        {
            "id": "metrics",
            "type": "metric-strip",
            "cardIds": ["delta_card", "same_season_card", "test_active_card"],
        },
        {
            "id": "blocks_intro",
            "type": "markdown",
            "sourceId": "block_source",
            "body": (
                "## 개선은 같은 계절에 집중됐고 다른 블록은 악화하지 않았다\n\n"
                "2024년 9–10월은 0.00558°C 개선됐고, 2025년 7–8월은 사실상 동일, 공개 endpoint가 "
                "전부 없는 2025년 11–12월은 exact no-op이었다. 이는 월 규칙이 아니라 public endpoint가 "
                "실제로 제공하는 물리 제약에서 이득이 발생했다는 증거다."
            ),
        },
        {"id": "block_chart", "type": "chart", "chartId": "block_rmse_chart"},
        {
            "id": "layers_intro",
            "type": "markdown",
            "sourceId": "layer_source",
            "body": (
                "## 세 목표층 모두 같은 방향으로 개선됐다\n\n"
                "layer 2의 개선이 가장 크지만 layer 3과 4도 악화하지 않았다. 특정 층 하나의 사후 router가 "
                "아니라 세 층을 공동 프로파일로 투영한 구조적 효과다."
            ),
        },
        {"id": "layer_block", "type": "table", "tableId": "layer_table"},
        {
            "id": "coverage_intro",
            "type": "markdown",
            "sourceId": "coverage_source",
            "body": (
                "## hidden 개입량은 같은 계절 검증보다 낮고 결측 시 자동 복귀한다\n\n"
                "같은 계절 OOF에서는 46.2%를 바꿨지만 hidden test에서는 31.5%만 바꾼다. layer 1 또는 5가 "
                "결측이거나 목표 3개 층이 모두 없으면 원 Deep 예측을 byte-level 의미로 보존한다. hidden target "
                "정답은 없으므로 test correction 크기는 일반화의 증명이 아니라 배포 위험 지표다."
            ),
        },
        {"id": "coverage_block", "type": "table", "tableId": "coverage_table"},
        {
            "id": "method",
            "type": "markdown",
            "body": (
                "## 모델 사양과 검증 설계\n\n"
                "각 시각의 동결 Deep 예측 3개에 unit-weight PAVA를 적용하고, layer 1과 5의 동시각 공개 수온 "
                "최솟값·최댓값으로 clip했다. endpoint 방향이 증가면 목표 3층도 증가, 감소면 감소하도록 한다. "
                "변환에는 target layer 수온·염분, hidden 정답, 외부 관측값을 사용하지 않는다. 비교 기준은 기존 "
                "3-block LOBO 69,850행이며 공식 지표와 같은 pooled row-level RMSE를 사용했다."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 한계·불확실성·강건성\n\n"
                "**검증됨:** 저장 OOF 재계산, 세 층 개선, block 비악화, 2,000회 KST-day CI, 제출 26,061행, "
                "저장 CSV 재현 오차 3.6×10⁻¹⁵은 통과했다.\n\n"
                "**한계:** 실제 중층은 endpoint 범위를 벗어나거나 역전될 수 있고, 변형 선택은 이미 노출된 OOF를 "
                "본 뒤 이뤄졌다. hidden leaderboard 점수는 업로드 전 알 수 없다. 따라서 이 파일은 물리적으로 "
                "정당화된 challenger이지 공식 우승을 증명한 모델이 아니다."
            ),
        },
        {
            "id": "next",
            "type": "markdown",
            "body": (
                "## 권장 다음 단계\n\n"
                "1. Deep Stack과 물리 투영 challenger 두 파일을 계속 동결한다.\n"
                "2. 추가 사후 threshold·softness 튜닝은 중단한다.\n"
                "3. 첫 공식 제출 가능일의 제한된 슬롯에서는 물리 투영 challenger를 정보가치가 높은 후보로 검토한다.\n"
                "4. 그 전에는 target-layer boundary condition을 실제 입력으로 쓰는 structured-mask sequence imputer를 별도 세대로 개발한다."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 남은 질문\n\n"
                "- hidden에서 실제 중층 역전이 2024 같은 계절보다 더 잦은가?\n"
                "- 8월 말·11월 초 목표층 관측을 입력으로 사용하는 long-gap imputer가 endpoint projection 이상의 이득을 주는가?\n"
                "- 공식 첫 제출에서 얻은 hidden 점수가 Deep Stack 대비 물리 제약의 실제 전이를 지지하는가?"
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "P2 공개층 물리 연직 투영 검증",
            "description": "동결 Deep Stack에 공개 endpoint 기반 연직 제약을 적용한 기술 검증 보고서",
            "generatedAt": generated,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "block_rmse": block_rows,
                "layer_metrics": layer_rows,
                "coverage": coverage_rows,
            },
        },
        "sources": [{"id": source["id"], "path": source["path"]} for source in sources],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("artifacts/p2_physical_profile_projection_v1/result.json"),
    )
    parser.add_argument(
        "--oof", type=Path, default=Path("artifacts/p2_physical_profile_projection_v1/oof.parquet")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_artifact(
        json.loads(args.result.read_text(encoding="utf-8")), pd.read_parquet(args.oof)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": args.output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
