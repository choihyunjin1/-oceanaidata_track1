"""Build a portable technical report for the P2 soft-gate failure diagnosis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


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
        "path": "artifacts/p2_soft_gate_failure_diagnostic/result.json",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": _union_sql(rows, columns),
            "description": description,
            "tables_used": ["artifacts/p2_soft_gate_failure_diagnostic/result.json"],
            "filters": [
                "69,850 frozen OOF rows",
                "Three predeclared seasonal proxy blocks",
                "No hidden target values and no external observations",
            ],
        },
    }


def build_artifact(result: dict[str, object]) -> dict[str, object]:
    generated = datetime.now().astimezone().isoformat()
    metric = result["metric_reconciliation"]
    blocks = result["error_decomposition_by_block"]
    layers = result["error_decomposition_by_layer"]
    headline = [
        {
            "delta_rmse": metric["delta_rmse"],
            "same_season_delta": next(
                row["delta_rmse"] for row in blocks if row["block"] == "2024_sep_oct"
            ),
            "layer4_delta": next(row["delta_rmse"] for row in layers if row["layer"] == 4),
            "stable_mapping": (
                f"{result['state_to_winner_mapping']['three_block_same_winner_cells']} / "
                f"{result['state_to_winner_mapping']['three_block_cells']}"
            ),
        }
    ]
    block_layer = []
    for row in result["state_cell_decomposition"]:
        key = f"{row['block']} · L{row['layer']}"
        # Aggregate the mutually exclusive state cells back to block-layer grain.
        current = next((item for item in block_layer if item["block_layer"] == key), None)
        if current is None:
            current = {
                "block_layer": key,
                "block": row["block"],
                "layer": row["layer"],
                "rows": 0,
                "baseline_sse": 0.0,
                "candidate_sse": 0.0,
                "alignment_sse": 0.0,
                "movement_sse": 0.0,
            }
            block_layer.append(current)
        current["rows"] += row["rows"]
        current["baseline_sse"] += row["rows"] * row["baseline_rmse"] ** 2
        current["candidate_sse"] += row["rows"] * row["candidate_rmse"] ** 2
        current["alignment_sse"] += row["rows"] * row["alignment_mse"]
        current["movement_sse"] += row["rows"] * row["movement_mse"]
    rmse_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    for row in block_layer:
        baseline_rmse = (row["baseline_sse"] / row["rows"]) ** 0.5
        candidate_rmse = (row["candidate_sse"] / row["rows"]) ** 0.5
        rmse_rows.extend(
            [
                {
                    "block_layer": row["block_layer"],
                    "block": row["block"],
                    "layer": row["layer"],
                    "method": "Frozen deep stack",
                    "rmse": baseline_rmse,
                    "rows": row["rows"],
                },
                {
                    "block_layer": row["block_layer"],
                    "block": row["block"],
                    "layer": row["layer"],
                    "method": "Public-state soft gate",
                    "rmse": candidate_rmse,
                    "rows": row["rows"],
                },
            ]
        )
        decomposition_rows.extend(
            [
                {
                    "block_layer": row["block_layer"],
                    "block": row["block"],
                    "layer": row["layer"],
                    "component": "Alignment term (2e·a)",
                    "mse_component": row["alignment_sse"] / row["rows"],
                    "delta_mse": (row["candidate_sse"] - row["baseline_sse"]) / row["rows"],
                    "rows": row["rows"],
                },
                {
                    "block_layer": row["block_layer"],
                    "block": row["block"],
                    "layer": row["layer"],
                    "component": "Movement cost (a²)",
                    "mse_component": row["movement_sse"] / row["rows"],
                    "delta_mse": (row["candidate_sse"] - row["baseline_sse"]) / row["rows"],
                    "rows": row["rows"],
                },
            ]
        )

    selection_rows = []
    for row in result["selection_diagnostic"]:
        selection_rows.append(
            {
                "outer_block": row["outer_block"],
                "selected_lambda": row["selected_regularization"],
                "inner_no_op_rmse": row["inner_no_op_rmse"],
                "inner_selected_rmse": row["inner_selected_gate_rmse"],
                "outer_no_op_rmse": row["outer_no_op_rmse"],
                "outer_selected_rmse": row["outer_selected_rmse"],
                "outer_delta": row["outer_selected_delta"],
                "posthoc_best": row["posthoc_best_candidate"],
                "selection_regret": row["outer_selection_regret"],
            }
        )

    state_harm = sorted(
        result["state_cell_decomposition"], key=lambda row: row["delta_sse"], reverse=True
    )[:8]
    state_harm_rows = [
        {
            "block": row["block"],
            "layer": row["layer"],
            "state": row["state"],
            "rows": row["rows"],
            "baseline_rmse": row["baseline_rmse"],
            "candidate_rmse": row["candidate_rmse"],
            "delta_rmse": row["delta_rmse"],
            "delta_sse_share": row["delta_sse_share"],
        }
        for row in state_harm
    ]

    state_support = []
    state_cells = result["state_cell_decomposition"]
    for block in sorted({row["block"] for row in state_cells}):
        current = [row for row in state_cells if row["block"] == block]
        total = sum(row["rows"] for row in current)
        for state in ("low", "transition", "high", "missing"):
            count = sum(row["rows"] for row in current if row["state"] == state)
            state_support.append(
                {"block": block, "state": state, "rows": count, "share": count / total}
            )

    gate_rows = [
        {
            "outer_block": row["outer_block"],
            "layer": row["layer"],
            "lambda": row["regularization"],
            "mean_l1_shift": row["mean_l1_weight_shift"],
            "p95_l1_shift": row["p95_l1_weight_shift"],
            "saturation_share": row["max_weight_above_0_8_share"],
            "floored_experts": len(row["floored_contributors"]),
            "floored_oracle_share": row["floored_contributor_is_row_oracle_share"],
        }
        for row in result["gate_dynamics"]
    ]
    shift_rows = sorted(
        result["feature_shift"],
        key=lambda row: row["normalized_wasserstein"] or -1,
        reverse=True,
    )[:10]

    sources = [
        _source(
            "headline_source",
            "Soft-gate headline reconciliation",
            headline,
            list(headline[0]),
            "Reconciles the frozen deep-stack and public-state soft-gate OOF metrics.",
        ),
        _source(
            "rmse_source",
            "Block-layer RMSE comparison",
            rmse_rows,
            ["block_layer", "block", "layer", "method", "rmse", "rows"],
            "Materializes exact block-layer RMSE for the incumbent and failed gate.",
        ),
        _source(
            "decomposition_source",
            "Exact gate-adjustment error decomposition",
            decomposition_rows,
            ["block_layer", "block", "layer", "component", "mse_component", "delta_mse", "rows"],
            "Decomposes ΔMSE into 2e·a alignment and a² movement terms.",
        ),
        _source(
            "selection_source",
            "Nested selection and posthoc diagnostic",
            selection_rows,
            list(selection_rows[0]),
            "Shows inner selection, outer result, and research-only posthoc best candidate.",
        ),
        _source(
            "state_harm_source",
            "Largest state-cell error contributions",
            state_harm_rows,
            list(state_harm_rows[0]),
            "Ranks mutually exclusive block-layer-state cells by excess SSE.",
        ),
        _source(
            "state_support_source",
            "Public-state support by proxy block",
            state_support,
            ["block", "state", "rows", "share"],
            "Shows the support mismatch for 2024 same-season, 2025 pre-gap, and post-gap blocks.",
        ),
        _source(
            "gate_source",
            "Gate weight dynamics and effective expert exclusions",
            gate_rows,
            list(gate_rows[0]),
            "Summarizes dynamic movement from each fold-train simplex prior.",
        ),
        _source(
            "shift_source",
            "Largest public-feature distribution shifts",
            shift_rows,
            [
                "outer_block",
                "feature",
                "normalized_wasserstein",
                "train_missing_rate",
                "outer_missing_rate",
                "missing_rate_delta",
            ],
            "Ranks normalized Wasserstein and missing-rate shifts across outer blocks.",
        ),
    ]

    cards = [
        {
            "id": "delta_card",
            "description": "69,850 frozen OOF rows; lower RMSE is better",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Soft-gate ΔRMSE",
                    "field": "delta_rmse",
                    "format": "number",
                    "unit": " °C",
                    "signed": True,
                }
            ],
        },
        {
            "id": "same_season_card",
            "description": "2024 September–October outer block",
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
            "id": "layer4_card",
            "description": "All three proxy blocks pooled",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Layer 4 ΔRMSE",
                    "field": "layer4_delta",
                    "format": "number",
                    "unit": " °C",
                    "signed": True,
                }
            ],
        },
        {
            "id": "mapping_card",
            "description": "Layer-state cells observed in all three blocks",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [{"label": "Same winning expert", "field": "stable_mapping"}],
        },
    ]

    charts = [
        {
            "id": "block_layer_rmse_chart",
            "title": "Block-layer RMSE",
            "subtitle": "Nine block-layer cells; lower is better, unit °C.",
            "type": "bar",
            "dataset": "block_layer_rmse",
            "sourceId": "rmse_source",
            "valueFormat": "number",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "block_layer", "type": "nominal", "label": "Outer block and layer"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (°C)"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "rows", "type": "quantitative", "label": "Rows"},
                    {"field": "block", "type": "nominal", "label": "Block"},
                    {"field": "layer", "type": "quantitative", "label": "Layer"},
                ],
            },
        },
        {
            "id": "decomposition_chart",
            "title": "Gate-adjustment ΔMSE decomposition",
            "subtitle": "2e·a below zero corrects baseline error; a² is the unavoidable movement cost.",
            "type": "bar",
            "dataset": "decomposition",
            "sourceId": "decomposition_source",
            "valueFormat": "number",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "block_layer", "type": "nominal", "label": "Outer block and layer"},
                "y": {
                    "field": "mse_component",
                    "type": "quantitative",
                    "label": "MSE contribution (°C²)",
                },
                "color": {"field": "component", "type": "nominal", "label": "Component"},
                "tooltip": [
                    {"field": "delta_mse", "type": "quantitative", "label": "Net ΔMSE"},
                    {"field": "rows", "type": "quantitative", "label": "Rows"},
                ],
            },
        },
        {
            "id": "state_support_chart",
            "title": "Public-state support by proxy block",
            "subtitle": "Shares use 2024 same-season |T1−T5| terciles plus an explicit missing state.",
            "type": "bar",
            "dataset": "state_support",
            "sourceId": "state_support_source",
            "valueFormat": "percent",
            "settings": {"groupMode": "stacked"},
            "encodings": {
                "x": {"field": "block", "type": "nominal", "label": "Proxy block"},
                "y": {"field": "share", "type": "quantitative", "label": "Row share"},
                "color": {"field": "state", "type": "nominal", "label": "Public state"},
                "tooltip": [{"field": "rows", "type": "quantitative", "label": "Rows"}],
            },
        },
    ]

    tables = [
        {
            "id": "selection_table",
            "title": "Nested selection versus outer behavior",
            "subtitle": "Posthoc best is diagnostic only and cannot be used for hidden selection.",
            "dataset": "selection",
            "sourceId": "selection_source",
            "defaultSort": {"field": "outer_block", "direction": "asc"},
            "columns": [
                {"field": "outer_block", "label": "Outer block", "type": "text"},
                {"field": "selected_lambda", "label": "Selected λ", "type": "number"},
                {"field": "inner_no_op_rmse", "label": "Inner no-op", "type": "number"},
                {"field": "inner_selected_rmse", "label": "Inner gate", "type": "number"},
                {"field": "outer_no_op_rmse", "label": "Outer no-op", "type": "number"},
                {"field": "outer_selected_rmse", "label": "Outer gate", "type": "number"},
                {
                    "field": "outer_delta",
                    "label": "Outer ΔRMSE",
                    "type": "number",
                    "movement": True,
                },
                {"field": "posthoc_best", "label": "Posthoc best", "type": "text"},
            ],
        },
        {
            "id": "state_harm_table",
            "title": "Largest excess-error state cells",
            "subtitle": "Ranked by excess SSE; shares can exceed 100% when other cells offset harm.",
            "dataset": "state_harm",
            "sourceId": "state_harm_source",
            "defaultSort": {"field": "delta_sse_share", "direction": "desc"},
            "columns": [
                {"field": "block", "label": "Block", "type": "text"},
                {"field": "layer", "label": "Layer", "type": "number"},
                {"field": "state", "label": "State", "type": "text"},
                {"field": "rows", "label": "Rows", "type": "number"},
                {"field": "baseline_rmse", "label": "Deep RMSE", "type": "number"},
                {"field": "candidate_rmse", "label": "Gate RMSE", "type": "number"},
                {
                    "field": "delta_rmse",
                    "label": "ΔRMSE",
                    "type": "number",
                    "movement": True,
                },
                {"field": "delta_sse_share", "label": "Excess SSE share", "type": "number"},
            ],
        },
        {
            "id": "gate_table",
            "title": "Weight movement and expert-floor diagnostics",
            "subtitle": "Nine outer-block × layer cells; L1 shift is relative to the fold-train simplex prior.",
            "dataset": "gate_dynamics",
            "sourceId": "gate_source",
            "defaultSort": {"field": "mean_l1_shift", "direction": "desc"},
            "columns": [
                {"field": "outer_block", "label": "Outer block", "type": "text"},
                {"field": "layer", "label": "Layer", "type": "number"},
                {"field": "lambda", "label": "λ", "type": "number"},
                {"field": "mean_l1_shift", "label": "Mean L1 shift", "type": "number"},
                {"field": "saturation_share", "label": "Max weight >0.8", "type": "number"},
                {"field": "floored_experts", "label": "Floored experts", "type": "number"},
                {
                    "field": "floored_oracle_share",
                    "label": "Floored expert row-oracle share",
                    "type": "number",
                },
            ],
        },
        {
            "id": "shift_table",
            "title": "Largest public-feature support shifts",
            "subtitle": "Wasserstein distance is normalized by the fold-train IQR.",
            "dataset": "feature_shift",
            "sourceId": "shift_source",
            "defaultSort": {"field": "normalized_wasserstein", "direction": "desc"},
            "columns": [
                {"field": "outer_block", "label": "Outer block", "type": "text"},
                {"field": "feature", "label": "Feature", "type": "text"},
                {
                    "field": "normalized_wasserstein",
                    "label": "Normalized W distance",
                    "type": "number",
                },
                {"field": "train_missing_rate", "label": "Train missing", "type": "number"},
                {"field": "outer_missing_rate", "label": "Outer missing", "type": "number"},
                {
                    "field": "missing_rate_delta",
                    "label": "Missing-rate Δ",
                    "type": "number",
                    "movement": True,
                },
            ],
        },
    ]

    blocks_manifest = [
        {"id": "title", "type": "markdown", "body": "# P2 Soft Gate 실패 원인 분석"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "headline_source",
            "body": (
                "## 결론: 상태 조건화 가설이 아니라 현재 구현 방식이 실패했다\n\n"
                "이번 실험은 공개층 물리상태가 전문가 선택에 유용한지를 일반적으로 검증하지 않았다. "
                "정확한 no-op 후보 없이, fold-train simplex prior에 고정된 선형 softmax gate를 강제로 적용했다. "
                f"그 결과 LOBO RMSE는 **{metric['baseline_rmse']:.6f}→{metric['candidate_rmse']:.6f}°C**로 "
                f"**{metric['delta_rmse']:+.6f}°C** 악화했다. 가장 큰 원인은 2024년 같은 계절 layer 4에서 "
                "gate가 기존 오차와 같은 방향으로 크게 움직인 것이다. 따라서 현재 결론은 `가설 기각`이 아니라 "
                "`forced prior-anchored gate 기각`이다."
            ),
        },
        {
            "id": "headline_strip",
            "type": "metric-strip",
            "cardIds": ["delta_card", "same_season_card", "layer4_card", "mapping_card"],
        },
        {
            "id": "where_failed",
            "type": "markdown",
            "sourceId": "rmse_source",
            "body": (
                "## 실패는 2024년 같은 계절의 layer 4에 집중됐다\n\n"
                "layer 2는 소폭 개선됐지만 layer 3과 4가 악화했다. 특히 2024년 9–10월 layer 4의 "
                "강성층·전이 상태가 전체 초과오차의 대부분을 만들었다. 이는 모든 계절에서 조금씩 나빠진 "
                "현상이 아니라, 특정 계절 궤적에서 gate의 방향이 뒤집힌 국소적 전이 실패다."
            ),
        },
        {"id": "rmse_chart_block", "type": "chart", "chartId": "block_layer_rmse_chart"},
        {
            "id": "adjustment_direction",
            "type": "markdown",
            "sourceId": "decomposition_source",
            "body": (
                "## 2024년 조정은 크기뿐 아니라 방향도 틀렸다\n\n"
                "gate 조정을 `a`, 기존 오차를 `e`라 하면 ΔMSE는 `2e·a + a²`로 정확히 분해된다. "
                "2024년 블록에서는 정렬항 `2e·a`가 양수여서 gate가 오차를 상쇄하지 않고 키웠고, 큰 조정량의 "
                "비용 `a²`까지 더해졌다. 반대로 2025년 7–8월에는 정렬항이 음수여서 같은 구조가 실제로 "
                "오차를 줄였다. 즉 상태 특징이 항상 무용한 것이 아니라, 상태에서 조정 방향으로 가는 매핑이 "
                "계절 간 보존되지 않았다."
            ),
        },
        {"id": "decomposition_chart_block", "type": "chart", "chartId": "decomposition_chart"},
        {
            "id": "state_harm_intro",
            "type": "markdown",
            "sourceId": "state_harm_source",
            "body": (
                "## 강성층 layer 4 한 셀이 실패를 지배했다\n\n"
                "2024년 강성층 layer 4는 RMSE가 1.0646→1.2381°C로 악화했고 전체 초과 SSE의 약 87%를 "
                "차지했다. 전이 상태 layer 4도 약 26%를 추가했다. 일부 다른 셀의 개선이 이를 상쇄했기 때문에 "
                "기여율 합은 100%를 넘을 수 있다."
            ),
        },
        {"id": "state_harm_block", "type": "table", "tableId": "state_harm_table"},
        {
            "id": "selection_failure",
            "type": "markdown",
            "sourceId": "selection_source",
            "body": (
                "## 정규화 선택은 계절 전이를 예측하지 못했고 no-op도 없었다\n\n"
                "2024 outer에서는 inner가 λ=0.001을 선택했지만 outer에서 +0.0661°C 악화했다. 사후 진단상 "
                "λ=10은 같은 outer에서 -0.0009°C였으므로, 모델 용량 자체보다 선택의 전이 실패가 컸다. "
                "또 두 outer가 grid 최대값 λ=10을 선택했고, 정확한 no-op은 후보군에 없었다. 이 endpoint 포화는 "
                "검증이 사실상 `거의 움직이지 말라`고 요구했다는 신호다."
            ),
        },
        {"id": "selection_block", "type": "table", "tableId": "selection_table"},
        {
            "id": "support_failure",
            "type": "markdown",
            "sourceId": "state_support_source",
            "body": (
                "## 세 검증 블록은 동일 상태를 반복 관측하지 않았다\n\n"
                "2024년 같은 계절은 low·transition·high가 고르게 존재했지만, 2025년 7–8월은 거의 전부 high, "
                "2025년 11–12월은 `|T1−T5|`가 전부 missing이었다. 따라서 gate는 동일 물리상태가 다른 "
                "계절에서 같은 전문가를 요구하는지 충분히 학습할 수 없었다. 세 블록 모두에서 관측된 "
                "layer-state 셀 3개 중 동일 승자가 유지된 셀은 0개였다."
            ),
        },
        {"id": "state_support_block", "type": "chart", "chartId": "state_support_chart"},
        {
            "id": "prior_floor",
            "type": "markdown",
            "sourceId": "gate_source",
            "body": (
                "## simplex prior가 일부 전문가를 사실상 영구 제외했다\n\n"
                "9개 outer-block×layer 셀 중 8개에서 최소 한 전문가의 prior가 10⁻⁶ 미만이었다. softmax는 "
                "`log(prior)`에서 시작하므로 이 전문가는 상태가 바뀌어도 거의 되살아나지 않았다. 실제로 floored "
                "전문가가 행별 최저오차 모델인 비율이 여러 셀에서 40–76%였다. 반면 2024 outer의 낮은 λ는 "
                "남은 전문가 사이에서 평균 L1 가중치 이동을 0.71–1.61까지 키워 포화를 만들었다. 이것은 "
                "전문가 선택 가설을 온전히 시험한 구조가 아니다."
            ),
        },
        {"id": "gate_block", "type": "table", "tableId": "gate_table"},
        {
            "id": "feature_shift",
            "type": "markdown",
            "sourceId": "shift_source",
            "body": (
                "## 공개 특징 자체도 outer support 밖으로 크게 이동했다\n\n"
                "정규화 Wasserstein 거리가 2를 넘는 특징이 있었고, `|T1−T5|` 관련 특징의 missing rate는 "
                "train과 outer 사이에서 약 39%p 이동했다. 10분 행 수는 많지만 inner 학습은 사실상 한 개의 "
                "61일 계절 궤적에 의존한다. 따라서 행 단위 표본 수가 gate의 독립적인 물리상태 표본 수를 "
                "과대평가한다."
            ),
        },
        {"id": "shift_block", "type": "table", "tableId": "shift_table"},
        {
            "id": "scope_methods",
            "type": "markdown",
            "body": (
                "## 범위와 검증 방법\n\n"
                "진단 대상은 사전 고정된 69,850개 OOF 행과 기존 5개 contributor다. 평가지표는 세 층 pooled "
                "RMSE이며, 2024년 9–10월·2025년 7–8월·2025년 11–12월을 outer block으로 사용했다. "
                "오차 분해, 모든 λ의 outer 반응, 상태별 승자와 feature support는 outer 정답을 사용한 사후 "
                "연구 진단이다. 이 값으로 새 파라미터를 고르거나 hidden 제출을 만들지 않았다. hidden target "
                "수온·염분과 외부 관측값은 사용하지 않았다."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 한계와 확실성\n\n"
                "**검증됨:** forced gate의 악화, 2024 layer 4 집중, 정렬항 방향 역전, no-op 부재, prior floor, "
                "상태 support 불균형은 재현됐다.\n\n"
                "**가능성이 높음:** 계절별 궤적과 상태가 교락되어 state→expert 매핑이 불안정했고, 낮은 λ가 "
                "이를 증폭했다.\n\n"
                "**미해결:** 정확한 no-op arm, 양의 expert weight floor, day-balanced residual correction을 가진 "
                "cross-fitted gate가 일반화하는지는 아직 시험하지 않았다. 따라서 공개층 상태 조건화 가설 전체를 "
                "기각할 근거는 없다."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 다음 실험은 가설을 분리해서 검증한다\n\n"
                "1. 기존 deep stack을 정확한 no-op arm으로 포함한다.\n"
                "2. contributor prior를 0으로 두지 않고 작은 양의 floor를 둔다.\n"
                "3. weight 자체 대신 `deep prediction + bounded residual correction`을 학습하고 correction=0을 기본값으로 둔다.\n"
                "4. 10분 행이 아니라 KST day를 균등 가중해 계절 궤적의 반복을 줄인다.\n"
                "5. 같은 상태가 두 계절 이상에 충분히 존재하는 셀만 학습하고, support 밖에서는 자동 no-op으로 복귀한다.\n"
                "6. 위 변경을 한꺼번에 튜닝하지 말고 `safe residual gate` 한 구조로 사전등록해 한 번 비교한다."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 남은 질문\n\n"
                "- hidden 구간의 missing 상태에서 어떤 contributor가 안정적인지는 target-blind 자료만으로 충분히 확인 가능한가?\n"
                "- layer 4 강성층에서 필요한 보정 방향을 2024와 2025가 공유하는 더 물리적인 특징이 있는가?\n"
                "- day-balanced correction이 2024 same-season 손실을 막으면서 2025 July 개선을 보존하는가?"
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "P2 Soft Gate 실패 원인 분석",
            "description": "Public-state soft gate의 계절 간 실패를 오차·선택·support·구조로 분해한 기술 보고서",
            "generatedAt": generated,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks_manifest,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "block_layer_rmse": rmse_rows,
                "decomposition": decomposition_rows,
                "selection": selection_rows,
                "state_harm": state_harm_rows,
                "state_support": state_support,
                "gate_dynamics": gate_rows,
                "feature_shift": shift_rows,
            },
        },
        "sources": [{"id": source["id"], "path": source["path"]} for source in sources],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=Path("artifacts/p2_soft_gate_failure_diagnostic/result.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    artifact = build_artifact(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": args.output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
