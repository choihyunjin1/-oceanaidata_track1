"""Append validated P2 deep-tournament results to the existing full HTML report artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _union_sql(rows: list[dict[str, object]], fields: tuple[str, ...]) -> str:
    statements = []
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, (int, float)):
                values.append(str(value))
            else:
                values.append("'" + str(value).replace("'", "''") + "'")
        statements.append(
            "SELECT "
            + ", ".join(f"{value} AS {field}" for value, field in zip(values, fields, strict=True))
        )
    return " UNION ALL ".join(statements)


def _replace_by_id(items: list[dict[str, object]], additions: list[dict[str, object]]) -> None:
    ids = {item["id"] for item in additions}
    items[:] = [item for item in items if item.get("id") not in ids]
    items.extend(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("reports/p2_method_scout_20260816/artifact.json"),
    )
    parser.add_argument(
        "--tournament",
        type=Path,
        default=Path("artifacts/p2_deep_model_tournament_v1/result.json"),
    )
    parser.add_argument(
        "--finalists", type=Path, default=Path("artifacts/p2_deep_finalists_v1/result.json")
    )
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    tournament = json.loads(args.tournament.read_text(encoding="utf-8"))
    finalists = json.loads(args.finalists.read_text(encoding="utf-8"))
    oof = pd.read_parquet(finalists["stacked_oof_path"])

    ranking_rows: list[dict[str, object]] = []
    for row in tournament["ranking"]:
        for variant, value in (("Standalone", row["rmse"]), ("Best tree blend", row["blend_rmse"])):
            ranking_rows.append(
                {
                    "model": row["model"],
                    "variant": variant,
                    "rmse": round(float(value), 6),
                    "parameters_m": round(float(row["parameter_count"]) / 1_000_000, 3),
                }
            )
    ranking_rows.append(
        {
            "model": "Frozen LightGBM router",
            "variant": "Incumbent",
            "rmse": round(float(finalists["incumbent_rmse"]), 6),
            "parameters_m": 0.0,
        }
    )
    weight_rows = [
        {
            "layer": int(layer),
            "model": model,
            "weight": round(float(weight), 6),
        }
        for layer, values in finalists["weights_by_layer"].items()
        for model, weight in values.items()
        if float(weight) > 1e-8
    ]
    block_rows = []
    for block, frame in oof.groupby("block", sort=False):
        for method, column in (
            ("Frozen router", "router_400"),
            ("Deep stack", "prediction"),
            ("LOBO stack", "lobo_prediction"),
        ):
            rmse = (
                (frame[column].to_numpy(float) - frame["truth"].to_numpy(float)) ** 2
            ).mean() ** 0.5
            block_rows.append(
                {
                    "block": block,
                    "method": method,
                    "rmse": round(float(rmse), 6),
                    "rows": len(frame),
                }
            )
    headline_rows = [
        {
            "stacked_rmse": float(finalists["stacked_oof_rmse"]),
            "delta_vs_incumbent": float(finalists["delta_vs_incumbent"]),
            "lobo_rmse": float(finalists["lobo_stacked_rmse"]),
            "submission_rows": int(finalists["submission"]["rows"]),
        }
    ]

    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]
    generated = datetime.now().astimezone().isoformat()
    manifest["title"] = "P2 연직 수온 복원 모델 검증"
    manifest["description"] = (
        "Tree 기준부터 8개 deep 계열, 복수 seed stack, 제출 재현성까지 검증한 P2 기술 보고서"
    )
    manifest["generatedAt"] = generated
    snapshot["generatedAt"] = generated
    snapshot["datasets"].update(
        {
            "deep_headline": headline_rows,
            "deep_model_ranking": ranking_rows,
            "deep_layer_weights": weight_rows,
            "deep_block_metrics": block_rows,
        }
    )

    sources = [
        {
            "id": "deep_tournament_result",
            "label": "P2 eight-family deep-model tournament",
            "path": "artifacts/p2_deep_model_tournament_v1/result.json",
        },
        {
            "id": "deep_finalist_result",
            "label": "P2 multi-seed finalist and frozen submission result",
            "path": "artifacts/p2_deep_finalists_v1/result.json",
        },
        {
            "id": "deep_headline_sql",
            "label": "Reviewed P2 finalist headline metrics",
            "path": "artifacts/p2_deep_finalists_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    headline_rows,
                    (
                        "stacked_rmse",
                        "delta_vs_incumbent",
                        "lobo_rmse",
                        "submission_rows",
                    ),
                ),
                "description": "Materializes the final deep-stack OOF, leave-one-block-out, incumbent delta, and submission row-count metrics.",
                "tables_used": [],
                "filters": ["Three frozen target-proxy blocks", "Validated local submission only"],
                "metric_definitions": {
                    "stacked_rmse": "Fitted layer-specific convex stack RMSE in degrees Celsius",
                    "delta_vs_incumbent": "Stack RMSE minus frozen router RMSE in degrees Celsius",
                    "lobo_rmse": "Leave-one-block-out layer-stack RMSE in degrees Celsius",
                    "submission_rows": "Rows passing the exact P2 submission schema and key-order validator",
                },
            },
        },
        {
            "id": "deep_ranking_sql",
            "label": "Reviewed P2 deep-model RMSE ranking",
            "path": "artifacts/p2_deep_model_tournament_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(ranking_rows, ("model", "variant", "rmse", "parameters_m")),
                "description": "Materializes identical-grain 69,850-row OOF RMSE for eight local architecture families and their best tree blends.",
                "tables_used": [],
                "filters": [
                    "Three frozen target-proxy blocks",
                    "No external values or pretrained weights",
                ],
                "metric_definitions": {
                    "rmse": "Row-level root mean squared temperature error in degrees Celsius"
                },
            },
        },
        {
            "id": "deep_weights_sql",
            "label": "Reviewed layer-specific convex stack weights",
            "path": "artifacts/p2_deep_finalists_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(weight_rows, ("layer", "model", "weight")),
                "description": "Materializes nonnegative sum-to-one OOF least-squares weights by target layer.",
                "tables_used": [],
                "filters": ["Weights below 1e-8 omitted from the visible table"],
                "metric_definitions": {
                    "weight": "Fractional prediction weight within one target layer"
                },
            },
        },
        {
            "id": "deep_blocks_sql",
            "label": "Reviewed P2 deep-stack blocked-validation metrics",
            "path": "artifacts/p2_deep_finalists_v1/stacked_oof.parquet",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(block_rows, ("block", "method", "rmse", "rows")),
                "description": "Materializes frozen-router, fitted-stack, and leave-one-block-out stack RMSE by seasonal block.",
                "tables_used": [],
                "filters": ["Target layers 2, 3, and 4", "Finite target-proxy rows only"],
                "metric_definitions": {
                    "rmse": "Row-level root mean squared temperature error in degrees Celsius"
                },
            },
        },
    ]
    _replace_by_id(manifest["sources"], sources)
    top_sources = [
        {
            "id": "deep_tournament_result",
            "path": "artifacts/p2_deep_model_tournament_v1/result.json",
        },
        {"id": "deep_finalist_result", "path": "artifacts/p2_deep_finalists_v1/result.json"},
    ]
    _replace_by_id(artifact["sources"], top_sources)

    cards = [
        {
            "id": "deep_stacked_rmse",
            "description": "Three target-proxy blocks, 69,850 rows",
            "dataset": "deep_headline",
            "sourceId": "deep_headline_sql",
            "metrics": [
                {
                    "label": "Deep stack OOF RMSE",
                    "field": "stacked_rmse",
                    "format": "number",
                    "unit": " °C",
                }
            ],
        },
        {
            "id": "deep_delta",
            "description": "Deep stack minus frozen 400-round router",
            "dataset": "deep_headline",
            "sourceId": "deep_headline_sql",
            "metrics": [
                {
                    "label": "RMSE change",
                    "field": "delta_vs_incumbent",
                    "format": "number",
                    "unit": " °C",
                    "signed": True,
                }
            ],
        },
        {
            "id": "deep_lobo_rmse",
            "description": "Weights fit on the other two blocks",
            "dataset": "deep_headline",
            "sourceId": "deep_headline_sql",
            "metrics": [
                {
                    "label": "LOBO stack RMSE",
                    "field": "lobo_rmse",
                    "format": "number",
                    "unit": " °C",
                }
            ],
        },
        {
            "id": "deep_submission_rows",
            "description": "Saved-weight reproduction and key-order validation passed",
            "dataset": "deep_headline",
            "sourceId": "deep_headline_sql",
            "metrics": [
                {"label": "Frozen candidate rows", "field": "submission_rows", "format": "number"}
            ],
        },
    ]
    _replace_by_id(manifest["cards"], cards)
    charts = [
        {
            "id": "deep_model_chart",
            "title": "P2 deep-family OOF RMSE",
            "subtitle": "Three frozen seasonal blocks and 69,850 rows; lower is better. Each family uses its development-selected learning rate.",
            "type": "bar",
            "dataset": "deep_model_ranking",
            "sourceId": "deep_ranking_sql",
            "valueFormat": "number",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model family"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (°C)"},
                "color": {"field": "variant", "type": "nominal", "label": "Prediction"},
                "tooltip": [
                    {
                        "field": "parameters_m",
                        "type": "quantitative",
                        "label": "Trainable parameters (M)",
                    }
                ],
            },
        },
        {
            "id": "deep_block_chart",
            "title": "P2 stack RMSE by validation block",
            "subtitle": "Same row-level metric by seasonal block; LOBO weights never use the held block.",
            "type": "bar",
            "dataset": "deep_block_metrics",
            "sourceId": "deep_blocks_sql",
            "valueFormat": "number",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "block", "type": "nominal", "label": "Validation block"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (°C)"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [{"field": "rows", "type": "quantitative", "label": "Rows"}],
            },
        },
    ]
    _replace_by_id(manifest["charts"], charts)
    tables = [
        {
            "id": "deep_layer_weights",
            "title": "Final layer-specific stack weights",
            "subtitle": "Nonnegative weights sum to one within each target layer; zero weights are omitted.",
            "dataset": "deep_layer_weights",
            "sourceId": "deep_weights_sql",
            "defaultSort": {"field": "layer", "direction": "asc"},
            "columns": [
                {"field": "layer", "label": "Layer", "type": "number"},
                {"field": "model", "label": "Model", "type": "text"},
                {"field": "weight", "label": "Weight", "type": "number"},
            ],
        }
    ]
    _replace_by_id(manifest["tables"], tables)

    replaced_block_ids = {
        "deep_results",
        "deep_headline_strip",
        "deep_model_chart_block",
        "deep_model_chart_note",
        "deep_weights",
        "deep_weights_table",
        "deep_robustness",
        "deep_block_chart_block",
        "deep_block_chart_note",
        "limitations",
        "next_steps",
        "questions",
    }
    blocks = [block for block in manifest["blocks"] if block.get("id") not in replaced_block_ids]
    blocks[0]["body"] = "# P2 연직 수온 복원 모델 검증"
    blocks[1]["body"] = (
        "## Technical Summary\n\n"
        "동일한 69,850개 target-proxy 행에서 8개 P2-adapted deep 계열을 비교했다. "
        "단독 최강은 3-seed LSTI 계열이었고, 최종 layer별 convex stack은 400-round tree, "
        "Depth-query BiTCN, LSTI, TimeMixer++, local patch-foundation을 결합해 OOF RMSE "
        f"**{finalists['stacked_oof_rmse']:.6f}°C**를 기록했다. 기존 router {finalists['incumbent_rmse']:.6f}°C보다 "
        f"{abs(finalists['delta_vs_incumbent']):.6f}°C 낮다. leave-one-block-out weight 진단도 "
        f"{finalists['lobo_stacked_rmse']:.6f}°C로 기준보다 낮았고, 2,000회 KST-day bootstrap 90% CI는 "
        f"[{finalists['bootstrap']['ci90_low']:.6f}, {finalists['bootstrap']['ci90_high']:.6f}]°C였다. "
        "CSDI와 SSSD-SSM은 posterior mean을 써도 tree 가중치가 100%로 선택되어 기각됐다. "
        "26,061행 제출 후보는 저장 가중치에서 동일 SHA로 재현됐으며 업로드하지 않았다."
    )
    blocks.extend(
        [
            {
                "id": "deep_results",
                "type": "markdown",
                "sourceId": "deep_finalist_result",
                "body": "## 구조와 용량을 함께 늘렸지만 승자는 혼합이었다\n\n단순히 파라미터 수가 큰 모델이 우승하지 않았다. 0.73M LSTI는 단독 RMSE 0.7718°C로 가장 낮았고, 2.51M TimeMixer++는 tree와 50:50일 때 0.7569°C로 가장 좋은 단일-family blend였다. 5.02M patch-foundation과 4.18M BiTCN도 서로 다른 오차를 제공해 최종 stack에 일부 남았지만, 4.32M FNO와 두 diffusion 계열은 최종 가중치가 0이었다. 이는 P2에서 모델 크기보다 수직 baseline residual과 계절별 오류 다양성이 더 중요함을 보여준다.",
            },
            {
                "id": "deep_headline_strip",
                "type": "metric-strip",
                "cardIds": [
                    "deep_stacked_rmse",
                    "deep_delta",
                    "deep_lobo_rmse",
                    "deep_submission_rows",
                ],
            },
            {"id": "deep_model_chart_block", "type": "chart", "chartId": "deep_model_chart"},
            {
                "id": "deep_model_chart_note",
                "type": "markdown",
                "sourceId": "deep_tournament_result",
                "body": "위 차트는 동일 OOF 행에서 standalone과 각 deep-family의 최적 tree blend를 비교한다. 확률적 CSDI·SSSD-SSM은 독립 오차를 제공하지 못해 blend가 incumbent와 같은 0.7889°C에 머물렀다. 반면 LSTI·TimeMixer++·BiTCN·patch-foundation은 모두 tree와 혼합할 때 standalone보다 낮아졌다.",
            },
            {
                "id": "deep_weights",
                "type": "markdown",
                "sourceId": "deep_finalist_result",
                "body": "## 목표 수심마다 유효한 모델이 달랐다\n\nLayer 2는 TimeMixer++ 비중이 가장 컸고, layer 3은 tree가 약 절반을 담당했다. 가장 깊은 layer 4는 LSTI와 TimeMixer++ 비중이 높고 BiTCN도 남았다. 따라서 하나의 전역 가중치보다 층별 비음수 합-1 제약이 수직 구조 차이를 더 잘 반영했다.",
            },
            {"id": "deep_weights_table", "type": "table", "tableId": "deep_layer_weights"},
            {
                "id": "deep_robustness",
                "type": "markdown",
                "sourceId": "deep_finalist_result",
                "body": "## 평균 개선은 강하지만 블록 간 차이는 남았다\n\nAll-OOF fitted stack은 0.7458°C, 다른 두 블록에서만 가중치를 정한 LOBO stack은 0.7757°C였다. 둘 다 incumbent 0.7889°C보다 낮다. 다만 2025년 11–12월은 deep 조합이 tree보다 불리해 계절 전이 방향에 따른 편차가 남는다. hidden 2025년 9–10월은 2024 동일계절과 2025년 7–8월 사이에 있으므로 개선 가능성은 있지만, 이는 공식 점수가 아니라 로컬 proxy다.",
            },
            {"id": "deep_block_chart_block", "type": "chart", "chartId": "deep_block_chart"},
            {
                "id": "deep_block_chart_note",
                "type": "markdown",
                "sourceId": "deep_blocks_sql",
                "body": "차단별 막대는 fitted stack과 LOBO stack을 구분한다. fitted 값은 최종 test 가중치 설계용이며, LOBO 값은 한 블록의 라벨을 보지 않은 가중치가 그 블록으로 이동하는지 확인하는 보조 일반화 진단이다.",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "deep_finalist_result",
                "body": "## 한계와 검증 경계\n\n세 outer block은 모델 선택에 반복 노출됐으므로 0.7458°C를 독립 추정치로 볼 수 없다. LOBO 0.7757°C와 KST-day bootstrap은 이 위험을 줄이지만 hidden test를 대체하지 않는다. MOMENT·UniTS 공식 pretrained weight는 외부 가중치 허용 여부가 불명확해 사용하지 않았고, 보고서의 patch-foundation은 로컬 from-scratch proxy다. CSDI·SSSD 역시 upstream 완전 재현이 아니라 동일 P2 계약에 맞춘 구조 계열 비교다.",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": "## 권장 다음 단계\n\n1. `P2_DEEP_STACK_V1.csv`를 현재 첫 공식 제출 후보로 유지한다.\n2. 제출 전 파일 경로와 SHA를 사용자에게 다시 제시하고 정확한 파일 승인을 받는다.\n3. 첫 공식 점수가 로컬 tree 기준보다 악화하면 구조를 더 늘리지 말고 tree-only와 deep-stack 차이를 점검한다.\n4. 공식 점수가 개선되면 모델 family를 추가하지 않고 현재 stack의 epoch·weight 재현성을 유지한다.\n5. 원본·가중치·제출 CSV는 Git에 올리지 않고 코드·설정·보고서만 백업한다.",
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": "## 남은 질문\n\n- 리더보드 분할이 층 또는 시계열 구간을 어떻게 나누는가?\n- 외부 pretrained time-series weight가 허용되는가?\n- 첫 공식 점수에서 layer별 오차 또는 public/private 분할 차이를 확인할 수 있는가?",
            },
        ]
    )
    manifest["blocks"] = blocks
    args.artifact.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
