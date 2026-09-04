"""Append the independently validated GBM tournament to the complete P2 report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _union_sql(rows: list[dict[str, object]], fields: tuple[str, ...]) -> str:
    statements: list[str] = []
    for row in rows:
        values: list[str] = []
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


def _replace(items: list[dict[str, object]], additions: list[dict[str, object]]) -> None:
    identifiers = {item["id"] for item in additions}
    items[:] = [item for item in items if item.get("id") not in identifiers]
    items.extend(additions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", type=Path, default=Path("reports/p2_method_scout_20260816/artifact.json")
    )
    parser.add_argument(
        "--result", type=Path, default=Path("artifacts/p2_gbm_family_tournament_v1/result.json")
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/p2_gbm_family_tournament_v1/independent_validation.json"),
    )
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise ValueError("GBM tournament independent validation did not pass")
    selected = result["selected_for_parameter_search"]
    selected_arm = result["arms"][selected]
    paired = pd.read_parquet(
        Path("artifacts/p2_gbm_family_tournament_v1") / selected / "paired_oof.parquet"
    )

    arm_rows: list[dict[str, object]] = []
    for rank, row in enumerate(result["ranking"], start=1):
        arm_rows.extend(
            [
                {
                    "arm": row["arm"],
                    "variant": "Standalone",
                    "rmse": round(float(row["standalone_rmse"]), 6),
                    "rank": rank,
                    "iterations": 400,
                },
                {
                    "arm": row["arm"],
                    "variant": "Fitted pair with deep",
                    "rmse": round(float(row["fitted_pair_rmse"]), 6),
                    "rank": rank,
                    "iterations": 400,
                },
                {
                    "arm": row["arm"],
                    "variant": "LOBO pair with deep",
                    "rmse": round(float(row["lobo_pair_rmse"]), 6),
                    "rank": rank,
                    "iterations": 400,
                },
            ]
        )
    exact_rows = [
        {
            "rank": rank,
            "arm": row["arm"],
            "standalone_rmse": round(float(row["standalone_rmse"]), 6),
            "fitted_pair_rmse": round(float(row["fitted_pair_rmse"]), 6),
            "lobo_pair_rmse": round(float(row["lobo_pair_rmse"]), 6),
            "lobo_delta": round(float(row["lobo_delta_vs_deep"]), 6),
        }
        for rank, row in enumerate(result["ranking"], start=1)
    ]
    weight_rows = [
        {
            "layer": int(layer),
            "deep_weight": round(1.0 - float(weight), 6),
            "catboost_weight": round(float(weight), 6),
        }
        for layer, weight in selected_arm["pair_with_deep"]["fitted_weights_by_layer"].items()
    ]
    block_rows: list[dict[str, object]] = []
    for block, frame in paired.groupby("block", sort=False):
        truth = frame["truth"].to_numpy(float)
        for method, column in (
            ("Deep fitted", "deep_prediction"),
            ("CatBoost layerwise", "gbm_prediction"),
            ("Deep + CatBoost fitted", "fitted_pair_prediction"),
            ("Deep LOBO", "deep_lobo_prediction"),
            ("Deep + CatBoost LOBO", "lobo_pair_prediction"),
        ):
            rmse = float(((frame[column].to_numpy(float) - truth) ** 2).mean() ** 0.5)
            block_rows.append(
                {"block": block, "method": method, "rmse": round(rmse, 6), "rows": len(frame)}
            )
    headline_rows = [
        {
            "best_standalone_rmse": float(min(row["standalone_rmse"] for row in result["ranking"])),
            "selected_fitted_pair_rmse": float(selected_arm["pair_with_deep"]["fitted_blend_rmse"]),
            "selected_lobo_pair_rmse": float(selected_arm["pair_with_deep"]["lobo_blend_rmse"]),
            "selected_lobo_delta": float(selected_arm["pair_with_deep"]["lobo_delta_vs_deep_lobo"]),
        }
    ]

    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]
    generated = datetime.now().astimezone().isoformat()
    manifest["description"] = (
        "P2 tree, deep, and six-family GBM structure comparison with blocked OOF and reproducible submissions"
    )
    manifest["generatedAt"] = generated
    snapshot["generatedAt"] = generated
    snapshot["datasets"].update(
        {
            "gbm_headline": headline_rows,
            "gbm_arm_ranking": arm_rows,
            "gbm_arm_exact": exact_rows,
            "gbm_selected_weights": weight_rows,
            "gbm_selected_blocks": block_rows,
        }
    )

    sources = [
        {
            "id": "gbm_tournament_result",
            "label": "P2 six-family fixed-budget GBM tournament",
            "path": "artifacts/p2_gbm_family_tournament_v1/result.json",
        },
        {
            "id": "gbm_independent_validation",
            "label": "Independent P2 GBM OOF and submission reconciliation",
            "path": "artifacts/p2_gbm_family_tournament_v1/independent_validation.json",
        },
        {
            "id": "gbm_headline_sql",
            "label": "Reviewed P2 GBM headline metrics",
            "path": "artifacts/p2_gbm_family_tournament_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    headline_rows,
                    (
                        "best_standalone_rmse",
                        "selected_fitted_pair_rmse",
                        "selected_lobo_pair_rmse",
                        "selected_lobo_delta",
                    ),
                ),
                "description": "Materializes fixed-budget standalone and deep-pair GBM results.",
                "tables_used": [],
                "filters": ["Three target-proxy blocks", "69,850 unique target rows"],
                "metric_definitions": {
                    "best_standalone_rmse": "Lowest row-level standalone GBM RMSE in degrees Celsius",
                    "selected_fitted_pair_rmse": "Layer-weighted deep plus CatBoost fitted OOF RMSE",
                    "selected_lobo_pair_rmse": "Layer weights fit on the other two blocks and applied to the held block",
                    "selected_lobo_delta": "LOBO pair RMSE minus frozen deep-stack LOBO RMSE",
                },
            },
        },
        {
            "id": "gbm_ranking_sql",
            "label": "Reviewed P2 GBM model-family ranking",
            "path": "artifacts/p2_gbm_family_tournament_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(arm_rows, ("arm", "variant", "rmse", "rank", "iterations")),
                "description": "Materializes the same-grain six-family RMSE comparison.",
                "tables_used": [],
                "filters": ["Fixed 400 boosting iterations", "Public-only phase features"],
                "metric_definitions": {
                    "rmse": "Row-level root mean squared temperature error in degrees Celsius"
                },
            },
        },
        {
            "id": "gbm_exact_sql",
            "label": "Reviewed P2 GBM exact comparison table",
            "path": "artifacts/p2_gbm_family_tournament_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(
                    exact_rows,
                    (
                        "rank",
                        "arm",
                        "standalone_rmse",
                        "fitted_pair_rmse",
                        "lobo_pair_rmse",
                        "lobo_delta",
                    ),
                ),
                "description": "Materializes exact arm-level scores and LOBO deltas.",
                "tables_used": [],
                "filters": ["Lower RMSE is better"],
                "metric_definitions": {
                    "lobo_delta": "LOBO pair RMSE minus frozen deep-stack LOBO RMSE"
                },
            },
        },
        {
            "id": "gbm_weights_sql",
            "label": "Reviewed P2 CatBoost layer blend weights",
            "path": "artifacts/p2_gbm_family_tournament_v1/result.json",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(weight_rows, ("layer", "deep_weight", "catboost_weight")),
                "description": "Materializes nonnegative sum-to-one fitted pair weights by target layer.",
                "tables_used": [],
                "filters": ["Selected arm: catboost_layerwise"],
                "metric_definitions": {
                    "catboost_weight": "CatBoost fraction in the fitted deep-plus-CatBoost pair"
                },
            },
        },
        {
            "id": "gbm_blocks_sql",
            "label": "Reviewed P2 GBM block-level metrics",
            "path": "artifacts/p2_gbm_family_tournament_v1/catboost_layerwise/paired_oof.parquet",
            "query": {
                "engine": "sqlite",
                "sql": _union_sql(block_rows, ("block", "method", "rmse", "rows")),
                "description": "Materializes deep, CatBoost, and fitted or LOBO pair RMSE by block.",
                "tables_used": [],
                "filters": ["Target layers 2, 3, and 4", "Finite target-proxy rows"],
                "metric_definitions": {
                    "rmse": "Row-level root mean squared temperature error in degrees Celsius"
                },
            },
        },
    ]
    _replace(manifest["sources"], sources)
    _replace(
        artifact["sources"],
        [
            {
                "id": "gbm_tournament_result",
                "path": "artifacts/p2_gbm_family_tournament_v1/result.json",
            },
            {
                "id": "gbm_independent_validation",
                "path": "artifacts/p2_gbm_family_tournament_v1/independent_validation.json",
            },
        ],
    )

    _replace(
        manifest["cards"],
        [
            {
                "id": "gbm_best_standalone",
                "description": "LightGBM ExtraTrees, fixed 400 iterations",
                "dataset": "gbm_headline",
                "sourceId": "gbm_headline_sql",
                "metrics": [
                    {
                        "label": "Best GBM standalone RMSE",
                        "field": "best_standalone_rmse",
                        "format": "number",
                        "unit": " °C",
                    }
                ],
            },
            {
                "id": "gbm_fitted_pair",
                "description": "Deep stack plus layerwise CatBoost",
                "dataset": "gbm_headline",
                "sourceId": "gbm_headline_sql",
                "metrics": [
                    {
                        "label": "Fitted pair RMSE",
                        "field": "selected_fitted_pair_rmse",
                        "format": "number",
                        "unit": " °C",
                    }
                ],
            },
            {
                "id": "gbm_lobo_pair",
                "description": "Pair weights learned without the held block",
                "dataset": "gbm_headline",
                "sourceId": "gbm_headline_sql",
                "metrics": [
                    {
                        "label": "LOBO pair RMSE",
                        "field": "selected_lobo_pair_rmse",
                        "format": "number",
                        "unit": " °C",
                    }
                ],
            },
            {
                "id": "gbm_lobo_delta",
                "description": "CatBoost pair minus frozen deep LOBO",
                "dataset": "gbm_headline",
                "sourceId": "gbm_headline_sql",
                "metrics": [
                    {
                        "label": "LOBO RMSE change",
                        "field": "selected_lobo_delta",
                        "format": "number",
                        "unit": " °C",
                        "signed": True,
                    }
                ],
            },
        ],
    )
    _replace(
        manifest["charts"],
        [
            {
                "id": "gbm_family_chart",
                "title": "P2 GBM-family RMSE comparison",
                "subtitle": "Six fixed 400-iteration structures on 69,850 blocked OOF rows; lower is better.",
                "type": "bar",
                "dataset": "gbm_arm_ranking",
                "sourceId": "gbm_ranking_sql",
                "valueFormat": "number",
                "settings": {"groupMode": "grouped"},
                "encodings": {
                    "x": {"field": "arm", "type": "nominal", "label": "GBM arm"},
                    "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (°C)"},
                    "color": {"field": "variant", "type": "nominal", "label": "Evaluation"},
                    "tooltip": [
                        {"field": "rank", "type": "quantitative", "label": "LOBO pair rank"},
                        {
                            "field": "iterations",
                            "type": "quantitative",
                            "label": "Boosting iterations",
                        },
                    ],
                },
            },
            {
                "id": "gbm_block_chart",
                "title": "P2 CatBoost-pair RMSE by validation block",
                "subtitle": "Deep, layerwise CatBoost, and pair variants across the three target-proxy periods.",
                "type": "bar",
                "dataset": "gbm_selected_blocks",
                "sourceId": "gbm_blocks_sql",
                "valueFormat": "number",
                "settings": {"groupMode": "grouped"},
                "encodings": {
                    "x": {"field": "block", "type": "nominal", "label": "Validation block"},
                    "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (°C)"},
                    "color": {"field": "method", "type": "nominal", "label": "Method"},
                    "tooltip": [{"field": "rows", "type": "quantitative", "label": "Rows"}],
                },
            },
        ],
    )
    _replace(
        manifest["tables"],
        [
            {
                "id": "gbm_exact_table",
                "title": "Exact GBM structure-screen metrics",
                "subtitle": "Same 69,850 OOF rows; rank follows LOBO pair RMSE.",
                "dataset": "gbm_arm_exact",
                "sourceId": "gbm_exact_sql",
                "defaultSort": {"field": "rank", "direction": "asc"},
                "columns": [
                    {"field": "rank", "label": "Rank", "type": "number"},
                    {"field": "arm", "label": "Arm", "type": "text"},
                    {
                        "field": "standalone_rmse",
                        "label": "Standalone RMSE",
                        "type": "number",
                    },
                    {
                        "field": "fitted_pair_rmse",
                        "label": "Fitted pair RMSE",
                        "type": "number",
                    },
                    {
                        "field": "lobo_pair_rmse",
                        "label": "LOBO pair RMSE",
                        "type": "number",
                    },
                    {
                        "field": "lobo_delta",
                        "label": "LOBO delta",
                        "type": "number",
                        "movement": True,
                    },
                ],
            },
            {
                "id": "gbm_weight_table",
                "title": "Deep and CatBoost fitted weights by layer",
                "subtitle": "Nonnegative weights sum to one within each target layer.",
                "dataset": "gbm_selected_weights",
                "sourceId": "gbm_weights_sql",
                "defaultSort": {"field": "layer", "direction": "asc"},
                "columns": [
                    {"field": "layer", "label": "Layer", "type": "number"},
                    {"field": "deep_weight", "label": "Deep weight", "type": "number"},
                    {
                        "field": "catboost_weight",
                        "label": "CatBoost weight",
                        "type": "number",
                    },
                ],
            },
        ],
    )

    replacement_ids = {
        "gbm_results",
        "gbm_headline_strip",
        "gbm_family_chart_block",
        "gbm_family_note",
        "gbm_exact_table_block",
        "gbm_diversity",
        "gbm_weight_table_block",
        "gbm_robustness",
        "gbm_block_chart_block",
        "gbm_block_note",
        "limitations",
        "next_steps",
        "questions",
    }
    blocks = [block for block in manifest["blocks"] if block.get("id") not in replacement_ids]
    for block in blocks:
        if block.get("id") == "technical_summary":
            block["body"] = (
                "## Technical Summary\n\n"
                "기존 8개 deep 계열의 layer별 stack은 동일 69,850개 target-proxy 행에서 "
                f"RMSE **{result['oof_contract']['deep_stack_rmse']:.6f}°C**, LOBO weight 진단 "
                f"{result['oof_contract']['deep_stack_lobo_rmse']:.6f}°C를 유지한다. 추가로 LightGBM GBDT·ExtraTrees·DART, "
                "XGBoost hist, CatBoost pooled·layerwise를 public-only phase 특징과 400 iteration으로 비교했다. "
                f"단독 최강은 ExtraTrees {min(row['standalone_rmse'] for row in result['ranking']):.6f}°C였지만, "
                f"deep stack에 LOBO 기준으로 보완 이득을 준 모델은 layerwise CatBoost뿐이었다 "
                f"({selected_arm['pair_with_deep']['lobo_blend_rmse']:.6f}°C, Δ "
                f"{selected_arm['pair_with_deep']['lobo_delta_vs_deep_lobo']:+.6f}°C). "
                f"다만 2,000회 KST-day bootstrap 90% CI "
                f"[{result['selected_lobo_bootstrap_vs_deep_lobo']['ci90_low']:.6f}, "
                f"{result['selected_lobo_bootstrap_vs_deep_lobo']['ci90_high']:.6f}]°C가 0을 포함하므로, "
                "현재 제출 1순위는 deep stack을 유지하고 다음 파라미터 최적화 대상만 layerwise CatBoost로 좁힌다."
            )

    insertion = next(
        (index for index, block in enumerate(blocks) if block.get("id") == "limitations"),
        len(blocks),
    )
    blocks[insertion:insertion] = [
        {
            "id": "gbm_results",
            "type": "markdown",
            "sourceId": "gbm_tournament_result",
            "body": "## 단독 성능과 앙상블 가치는 다른 모델이 이겼다\n\n고정 400 iteration에서 ExtraTrees형 LightGBM이 단독 RMSE 0.816096°C로 가장 낮았다. 그러나 기존 deep stack과의 중복 오차가 커서 LOBO pair는 오히려 0.793065°C로 악화했다. 반대로 layerwise CatBoost의 단독 RMSE는 0.833549°C였지만, layer 2·3에서 deep과 다른 오차를 제공해 LOBO pair를 0.774577°C로 낮췄다. 구조 선택 목적이 최종 앙상블 개선이라면 CatBoost 층별형이 다음 튜닝 대상이다.",
        },
        {
            "id": "gbm_headline_strip",
            "type": "metric-strip",
            "cardIds": [
                "gbm_best_standalone",
                "gbm_fitted_pair",
                "gbm_lobo_pair",
                "gbm_lobo_delta",
            ],
        },
        {"id": "gbm_family_chart_block", "type": "chart", "chartId": "gbm_family_chart"},
        {
            "id": "gbm_family_note",
            "type": "markdown",
            "sourceId": "gbm_ranking_sql",
            "body": "차트의 standalone은 각 GBM 자체 예측, fitted pair는 세 블록 전체에서 층별 가중치를 맞춘 낙관적 조합, LOBO pair는 다른 두 블록에서만 가중치를 정한 일반화 진단이다. LGBM GBDT의 fitted pair 0.742085°C는 가장 낮지만 LOBO 0.790544°C로 역전되므로 가중치 과적합 신호다.",
        },
        {"id": "gbm_exact_table_block", "type": "table", "tableId": "gbm_exact_table"},
        {
            "id": "gbm_diversity",
            "type": "markdown",
            "sourceId": "gbm_tournament_result",
            "body": "## CatBoost의 이득은 얕지만 층별로 해석 가능하다\n\n최종 fitted pair에서 CatBoost 가중치는 layer 2가 0.280840, layer 3이 0.279260, layer 4가 0이었다. 즉 현재 이득은 깊은 중간층 전체를 대체하는 것이 아니라 위쪽 두 목표층의 오차 다양성에서 나온다. layer 4는 deep stack을 그대로 유지하는 편이 낫다.",
        },
        {"id": "gbm_weight_table_block", "type": "table", "tableId": "gbm_weight_table"},
        {
            "id": "gbm_robustness",
            "type": "markdown",
            "sourceId": "gbm_tournament_result",
            "body": "## 현재 개선폭은 탐색 가치가 있지만 제출 승격 근거로는 약하다\n\nLayerwise CatBoost pair의 fitted 개선은 -0.001954°C였고 KST-day bootstrap 개선확률은 93.95%였지만 90% CI 상단이 +0.000120°C였다. 더 엄격한 LOBO 개선은 -0.001083°C, 개선확률 68.8%, 90% CI [-0.005133, +0.003215]°C다. 따라서 연구 challenger CSV는 생성했지만 현재 deep-stack 제출본을 교체하지 않는다.",
        },
        {"id": "gbm_block_chart_block", "type": "chart", "chartId": "gbm_block_chart"},
        {
            "id": "gbm_block_note",
            "type": "markdown",
            "sourceId": "gbm_blocks_sql",
            "body": "블록별 막대는 fitted 조합과 LOBO 조합을 구분한다. 세 블록이 이미 여러 구조 선택에 노출됐으므로 이 비교는 hidden 점수 추정치가 아니라 다음 탐색 방향을 고르는 진단으로 사용한다.",
        },
    ]
    blocks.extend(
        [
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "gbm_independent_validation",
                "body": "## 한계와 검증 경계\n\n여섯 GBM 계열은 구조 비교를 위해 400 iteration과 한 개의 공통 초기 파라미터 세트를 사용했으므로 각 backend의 최고 성능을 의미하지 않는다. 특히 XGBoost의 0.965441°C는 현재 초기값의 결과이지 XGBoost 계열의 절대 상한이 아니다. 세 target-proxy block은 반복 노출됐으며, fitted pair는 같은 OOF에서 가중치를 맞췄다. 독립 validator가 69,850개 키·정답·RMSE·가중치 투영과 7개 제출 CSV의 26,061개 키를 재검산했지만 hidden 정답은 사용하지 않았다.",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": "## 권장 다음 단계\n\n1. 현재 공식 제출 1순위는 `P2_DEEP_STACK_V1.csv`로 유지한다.\n2. 다음 세대는 layerwise CatBoost만 대상으로 iterations/learning rate, depth, L2, random strength, bootstrap과 feature subsampling을 blocked inner search한다.\n3. ExtraTrees형 LightGBM은 단독 기준 comparator로 유지하되 별도 대규모 탐색은 CatBoost 결과를 본 뒤 결정한다.\n4. CatBoost 최적화 후 동일 69,850행 standalone·fitted pair·LOBO pair와 KST-day bootstrap을 다시 계산한다.\n5. 연구 challenger와 모든 모델·OOF·제출 CSV는 업로드하지 않으며, 정확한 파일 승인 전에는 공식 제출을 실행하지 않는다.",
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": "## 남은 질문\n\n- CatBoost의 이득이 depth 7.04 m와 9.44 m에 안정적으로 집중되는가, 아니면 특정 계절 블록에 의존하는가?\n- layerwise CatBoost의 최적 boosting horizon이 층마다 다른가?\n- 첫 공식 점수에서 deep-only와 CatBoost challenger의 차이를 확인할 수 있는가?",
            },
        ]
    )
    manifest["blocks"] = blocks
    args.artifact.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
