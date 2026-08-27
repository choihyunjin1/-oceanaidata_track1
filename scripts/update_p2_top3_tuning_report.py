"""Append independently validated top-three tuning results to the P2 report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

PREFIX = "top3_tuning_"


def _sql_literal(value: object) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _union_sql(rows: list[dict[str, object]], columns: list[str]) -> str:
    parts = []
    for row in rows:
        parts.append(
            "SELECT " + ", ".join(f"{_sql_literal(row[column])} AS {column}" for column in columns)
        )
    return " UNION ALL ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", type=Path, default=Path("reports/p2_method_scout_20260816/artifact.json")
    )
    parser.add_argument(
        "--result", type=Path, default=Path("artifacts/p2_top3_parallel_tuning_v1/result.json")
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/p2_top3_parallel_tuning_v1/independent_validation.json"),
    )
    parser.add_argument(
        "--fixed-screen",
        type=Path,
        default=Path("artifacts/p2_gbm_family_tournament_v1/result.json"),
    )
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    fixed = json.loads(args.fixed_screen.read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or result.get("uploaded") is not False:
        raise ValueError("top-three tuning is not independently validated and local-only")

    fixed_by_family = {row["arm"]: row for row in fixed["ranking"]}
    ranking_rows = []
    iteration_rows = []
    parameter_rows = []
    for row in result["ranking"]:
        family = row["family"]
        fixed_row = fixed_by_family[family]
        ranking_rows.append(
            {
                "rank": row["rank"],
                "family": family,
                "fixed_outer_rmse": fixed_row["standalone_rmse"],
                "tuned_outer_rmse": row["outer_rmse"],
                "outer_delta": row["outer_rmse"] - fixed_row["standalone_rmse"],
                "lobo_pair_rmse": row["lobo_pair_rmse"],
                "lobo_delta_vs_deep": row["lobo_delta_vs_deep"],
                "inner_rmse": row["best_inner_rmse"],
            }
        )
        iterations = row["full_fit_iterations"]
        if isinstance(iterations, dict):
            for layer, value in iterations.items():
                iteration_rows.append(
                    {"family": family, "layer": int(layer), "iterations": int(value)}
                )
        else:
            iteration_rows.append({"family": family, "layer": 0, "iterations": int(iterations)})
        for name, value in row["best_parameters"].items():
            parameter_rows.append({"family": family, "parameter": name, "value": str(value)})

    best = result["ranking"][0]
    chart_rows = [
        {"family": row["family"], "variant": variant, "rmse": row[field]}
        for row in ranking_rows
        for variant, field in (
            ("Fixed screen", "fixed_outer_rmse"),
            ("Nested tuned", "tuned_outer_rmse"),
        )
    ]
    headline = [
        {
            "best_family": best["family"],
            "best_outer_rmse": best["outer_rmse"],
            "best_lobo_pair_rmse": best["lobo_pair_rmse"],
            "best_lobo_delta": best["lobo_delta_vs_deep"],
            "validated_families": len(validation["families"]),
        }
    ]
    artifact["snapshot"]["datasets"].update(
        {
            "top3_tuning_headline": headline,
            "top3_tuning_ranking": ranking_rows,
            "top3_tuning_chart": chart_rows,
            "top3_tuning_iterations": iteration_rows,
            "top3_tuning_parameters": parameter_rows,
        }
    )
    manifest = artifact["manifest"]
    manifest["generatedAt"] = datetime.now().astimezone().isoformat()
    artifact["snapshot"]["generatedAt"] = manifest["generatedAt"]
    manifest["description"] = (
        "P2 tree, deep, six-family structure screen, and nested top-three GBM tuning with "
        "independent OOF and submission reconciliation"
    )
    manifest["cards"] = [item for item in manifest["cards"] if not item["id"].startswith(PREFIX)]
    manifest["charts"] = [item for item in manifest["charts"] if not item["id"].startswith(PREFIX)]
    manifest["tables"] = [item for item in manifest["tables"] if not item["id"].startswith(PREFIX)]
    manifest["sources"] = [
        item for item in manifest["sources"] if not item["id"].startswith(PREFIX)
    ]
    artifact["sources"] = [
        item for item in artifact["sources"] if not item["id"].startswith(PREFIX)
    ]
    manifest["blocks"] = [item for item in manifest["blocks"] if not item["id"].startswith(PREFIX)]

    ranking_source = {
        "id": "top3_tuning_ranking_sql",
        "label": "Reviewed P2 nested top-three tuning comparison",
        "path": "artifacts/p2_top3_parallel_tuning_v1/result.json",
        "query": {
            "engine": "sqlite",
            "sql": _union_sql(
                ranking_rows,
                [
                    "rank",
                    "family",
                    "fixed_outer_rmse",
                    "tuned_outer_rmse",
                    "outer_delta",
                    "lobo_pair_rmse",
                    "lobo_delta_vs_deep",
                    "inner_rmse",
                ],
            ),
            "description": "Materializes independently validated nested tuning results.",
            "tables_used": [],
            "filters": ["69,850 exact OOF rows", "Three target layers", "Lower RMSE is better"],
            "metric_definitions": {
                "tuned_outer_rmse": "Pooled row-level outer-fold RMSE in degrees Celsius",
                "lobo_delta_vs_deep": "LOBO pair RMSE minus frozen deep-stack LOBO RMSE",
            },
        },
    }
    iteration_source = {
        "id": "top3_tuning_iterations_sql",
        "label": "Reviewed P2 convergence checkpoints",
        "path": "artifacts/p2_top3_parallel_tuning_v1/result.json",
        "query": {
            "engine": "sqlite",
            "sql": _union_sql(iteration_rows, ["family", "layer", "iterations"]),
            "description": "Materializes frozen full-fit boosting checkpoints.",
            "tables_used": [],
            "filters": ["CatBoost max 3,000", "DART inner-selected round count"],
            "metric_definitions": {
                "iterations": "Boosting round selected without using outer validation scores"
            },
        },
    }
    parameter_source = {
        "id": "top3_tuning_parameters_sql",
        "label": "Reviewed P2 selected hyperparameters",
        "path": "artifacts/p2_top3_parallel_tuning_v1/result.json",
        "query": {
            "engine": "sqlite",
            "sql": _union_sql(parameter_rows, ["family", "parameter", "value"]),
            "description": "Materializes each family's deployment parameter set.",
            "tables_used": [],
            "filters": ["Final policy fixed before outer scoring"],
            "metric_definitions": {"value": "Serialized selected parameter value"},
        },
    }
    manifest["sources"].extend([ranking_source, iteration_source, parameter_source])
    artifact["sources"].extend(
        [
            {
                "id": "top3_tuning_result",
                "path": "artifacts/p2_top3_parallel_tuning_v1/result.json",
            },
            {
                "id": "top3_tuning_validation",
                "path": "artifacts/p2_top3_parallel_tuning_v1/independent_validation.json",
            },
        ]
    )
    manifest["cards"].extend(
        [
            {
                "id": "top3_tuning_outer_card",
                "description": f"Best tuned family: {best['family']}",
                "dataset": "top3_tuning_headline",
                "sourceId": "top3_tuning_ranking_sql",
                "metrics": [
                    {
                        "label": "Best tuned outer RMSE",
                        "field": "best_outer_rmse",
                        "format": "number",
                        "unit": " °C",
                    }
                ],
            },
            {
                "id": "top3_tuning_lobo_card",
                "description": "Tuned pair minus frozen deep-stack LOBO",
                "dataset": "top3_tuning_headline",
                "sourceId": "top3_tuning_ranking_sql",
                "metrics": [
                    {
                        "label": "Best tuned LOBO change",
                        "field": "best_lobo_delta",
                        "format": "number",
                        "unit": " °C",
                        "signed": True,
                    }
                ],
            },
        ]
    )
    manifest["charts"].append(
        {
            "id": "top3_tuning_outer_chart",
            "title": "Fixed-screen and tuned standalone RMSE",
            "subtitle": "Tuning uses 12 inner-only trials per outer fold; lower is better.",
            "type": "bar",
            "dataset": "top3_tuning_chart",
            "sourceId": "top3_tuning_ranking_sql",
            "valueFormat": "number",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "family", "type": "nominal", "label": "Family"},
                "y": {
                    "field": "rmse",
                    "type": "quantitative",
                    "label": "RMSE (°C)",
                },
                "color": {"field": "variant", "type": "nominal", "label": "Evaluation"},
            },
        }
    )
    manifest["tables"].extend(
        [
            {
                "id": "top3_tuning_ranking_table",
                "title": "Exact nested tuning metrics",
                "subtitle": "All families use the same outer rows and total 36-trial budget.",
                "dataset": "top3_tuning_ranking",
                "sourceId": "top3_tuning_ranking_sql",
                "defaultSort": {"field": "rank", "direction": "asc"},
                "columns": [
                    {"field": "rank", "label": "Rank", "type": "number"},
                    {"field": "family", "label": "Family", "type": "text"},
                    {"field": "fixed_outer_rmse", "label": "Fixed RMSE", "type": "number"},
                    {"field": "tuned_outer_rmse", "label": "Tuned RMSE", "type": "number"},
                    {"field": "outer_delta", "label": "Change", "type": "number", "movement": True},
                    {"field": "lobo_pair_rmse", "label": "LOBO pair", "type": "number"},
                    {
                        "field": "lobo_delta_vs_deep",
                        "label": "LOBO change",
                        "type": "number",
                        "movement": True,
                    },
                ],
            },
            {
                "id": "top3_tuning_iterations_table",
                "title": "Frozen convergence checkpoints",
                "subtitle": "Layer 0 denotes a pooled model.",
                "dataset": "top3_tuning_iterations",
                "sourceId": "top3_tuning_iterations_sql",
                "defaultSort": {"field": "family", "direction": "asc"},
                "columns": [
                    {"field": "family", "label": "Family", "type": "text"},
                    {"field": "layer", "label": "Layer", "type": "number"},
                    {"field": "iterations", "label": "Rounds", "type": "number"},
                ],
            },
            {
                "id": "top3_tuning_parameters_table",
                "title": "Selected deployment hyperparameters",
                "subtitle": "One parameter set per family; outer scores did not choose these values.",
                "dataset": "top3_tuning_parameters",
                "sourceId": "top3_tuning_parameters_sql",
                "defaultSort": {"field": "family", "direction": "asc"},
                "columns": [
                    {"field": "family", "label": "Family", "type": "text"},
                    {"field": "parameter", "label": "Parameter", "type": "text"},
                    {"field": "value", "label": "Value", "type": "text"},
                ],
            },
        ]
    )
    blocks = [
        {
            "id": "top3_tuning_summary",
            "type": "markdown",
            "sourceId": "top3_tuning_ranking_sql",
            "body": "## 긴 학습 자체보다 계절 전이 안정성이 병목이었다\n\n"
            "상위 3개 계열을 outer 폴드별로 완전히 분리해 총 36회씩 탐색했다. 모든 모델이 "
            "설정된 최대 예산 안에서 체크포인트를 선택했지만, tuned standalone과 LOBO deep-pair 모두 "
            "동결 기준을 안정적으로 넘지 못했다. 따라서 이 세 튜닝 후보는 제출 후보로 승격하지 않는다.",
        },
        {
            "id": "top3_tuning_cards_block",
            "type": "metric-strip",
            "cardIds": ["top3_tuning_outer_card", "top3_tuning_lobo_card"],
        },
        {"id": "top3_tuning_chart_block", "type": "chart", "chartId": "top3_tuning_outer_chart"},
        {
            "id": "top3_tuning_ranking_block",
            "type": "table",
            "tableId": "top3_tuning_ranking_table",
        },
        {
            "id": "top3_tuning_iterations_block",
            "type": "table",
            "tableId": "top3_tuning_iterations_table",
        },
        {
            "id": "top3_tuning_parameters_block",
            "type": "table",
            "tableId": "top3_tuning_parameters_table",
        },
    ]
    insertion = next(
        (index for index, block in enumerate(manifest["blocks"]) if block["id"] == "limitations"),
        len(manifest["blocks"]),
    )
    manifest["blocks"][insertion:insertion] = blocks
    for block in manifest["blocks"]:
        if block["id"] == "next_steps":
            block["body"] = (
                "## 다음 단계\n\n"
                "1. 현재 1순위 제출 후보 `P2_DEEP_STACK_V1.csv`를 유지한다.\n"
                "2. 상위 3개 GBM의 추가 trial 확대는 중단한다. 세 계열 모두 수렴했지만 outer 전이가 악화됐다.\n"
                "3. 다음 실험은 파라미터 양이 아니라 2024→2025 전이 구조를 직접 설명하는 단일 특징·모델 가설로 제한한다.\n"
                "4. 어떤 CSV도 사용자 승인 전 플랫폼에 올리지 않는다."
            )
    args.artifact.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
