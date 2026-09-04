"""Build the portable decision report for the P2 extrapolated soft-gate candidate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from build_p2_profile_projection_report import _source


def build_artifact(result: dict[str, object], oof: pd.DataFrame) -> dict[str, object]:
    metrics = result["metrics"]
    bootstrap = result["paired_kst_day_bootstrap"]
    headline = [
        {
            "baseline_rmse": metrics["baseline_rmse"],
            "candidate_rmse": metrics["candidate_rmse"],
            "delta_rmse": metrics["delta_rmse"],
            "ci90_low": bootstrap["ci90_low"],
            "ci90_high": bootstrap["ci90_high"],
            "probability_improved": bootstrap["probability_improved"],
            "layer2_factor": result["layer_factors"]["2"],
            "layer4_factor": result["layer_factors"]["4"],
        }
    ]
    block_rows = [
        {
            "block": block,
            "baseline_rmse": values["baseline_rmse"],
            "candidate_rmse": values["candidate_rmse"],
            "delta_rmse": values["delta_rmse"],
            "rows": values["rows"],
        }
        for block, values in metrics["by_block"].items()
    ]
    chart_rows = [
        {"block": row["block"], "method": method, "rmse": row[field]}
        for row in block_rows
        for method, field in (
            ("Physical projection", "baseline_rmse"),
            ("Extrapolated gate", "candidate_rmse"),
        )
    ]
    layer_rows = [
        {
            "layer": int(layer),
            "baseline_rmse": values["baseline_rmse"],
            "candidate_rmse": values["candidate_rmse"],
            "delta_rmse": values["delta_rmse"],
            "rows": values["rows"],
        }
        for layer, values in metrics["by_layer"].items()
    ]
    correction = oof["prediction"].to_numpy(float) - oof["baseline"].to_numpy(float)
    deployment = [
        {
            "oof_active_share": float(np.mean(np.abs(correction) > 1e-12)),
            "oof_correction_rmse": float(np.sqrt(np.mean(correction**2))),
            "oof_max_abs_correction": float(np.max(np.abs(correction))),
            "submission_sha256": result["artifacts"]["submission"]["sha256"],
            "submission_rows": result["artifacts"]["submission"]["rows"],
        }
    ]
    sources = [
        _source(
            "headline",
            "Candidate headline",
            headline,
            list(headline[0]),
            "Pooled RMSE and paired KST-day uncertainty.",
        ),
        _source(
            "blocks",
            "Block metrics",
            block_rows,
            list(block_rows[0]),
            "Fixed target-proxy block RMSE comparison.",
        ),
        _source(
            "chart",
            "Block chart",
            chart_rows,
            ["block", "method", "rmse"],
            "Grouped block RMSE chart rows.",
        ),
        _source(
            "layers",
            "Layer metrics",
            layer_rows,
            list(layer_rows[0]),
            "Layer-specific RMSE comparison.",
        ),
        _source(
            "deployment",
            "Deployment diagnostics",
            deployment,
            list(deployment[0]),
            "Correction magnitude and frozen submission identity.",
        ),
    ]
    for source in sources:
        source["path"] = "artifacts/p2_extrapolated_soft_gate_v2/result.json"
        source["query"]["tables_used"] = [
            "artifacts/p2_extrapolated_soft_gate_v2/result.json",
            "artifacts/p2_extrapolated_soft_gate_v2/oof.parquet",
        ]
        source["query"]["filters"] = [
            "69,850 exposed local OOF rows",
            "Layer 2/4 raw cross-fitted expert; layer 3 physical-projection base",
            "No hidden target labels and no external observations",
        ]
    cards = [
        {
            "id": "delta",
            "description": "Candidate minus current local primary; lower is better",
            "dataset": "headline",
            "sourceId": "headline",
            "metrics": [
                {
                    "label": "ΔRMSE",
                    "field": "delta_rmse",
                    "format": "number",
                    "unit": " °C",
                    "signed": True,
                }
            ],
        },
        {
            "id": "ci",
            "description": "2,000 paired KST-day bootstrap replicates",
            "dataset": "headline",
            "sourceId": "headline",
            "metrics": [
                {"label": "P(improved)", "field": "probability_improved", "format": "percent"}
            ],
        },
        {
            "id": "factor",
            "description": "Adaptive OOF-selected extrapolation strength",
            "dataset": "headline",
            "sourceId": "headline",
            "metrics": [
                {
                    "label": "Layer 2 factor",
                    "field": "layer2_factor",
                    "format": "number",
                    "unit": "×",
                }
            ],
        },
    ]
    charts = [
        {
            "id": "block_chart",
            "title": "RMSE by fixed validation block",
            "subtitle": "The November–December regression is small but real.",
            "type": "bar",
            "dataset": "chart",
            "sourceId": "chart",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "block", "type": "nominal", "label": "Block"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (°C)"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
            },
        }
    ]
    tables = [
        {
            "id": "layers",
            "title": "Layer-level effect",
            "subtitle": "Layer 3 is intentionally retained from the physical-projection base.",
            "dataset": "layers",
            "sourceId": "layers",
            "columns": [
                {"field": "layer", "label": "Layer", "type": "number"},
                {"field": "rows", "label": "Rows", "type": "number"},
                {"field": "baseline_rmse", "label": "Base RMSE", "type": "number"},
                {"field": "candidate_rmse", "label": "Candidate RMSE", "type": "number"},
                {"field": "delta_rmse", "label": "ΔRMSE", "type": "number", "movement": True},
            ],
        },
        {
            "id": "blocks_table",
            "title": "Block-level effect",
            "subtitle": "All blocks were fixed before this adaptive composition was created.",
            "dataset": "blocks",
            "sourceId": "blocks",
            "columns": [
                {"field": "block", "label": "Block", "type": "text"},
                {"field": "rows", "label": "Rows", "type": "number"},
                {"field": "baseline_rmse", "label": "Base RMSE", "type": "number"},
                {"field": "candidate_rmse", "label": "Candidate RMSE", "type": "number"},
                {"field": "delta_rmse", "label": "ΔRMSE", "type": "number", "movement": True},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# P2 extrapolated soft-gate v2 점수 후보"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "headline",
            "body": (
                "## 결론: 새 로컬 1순위로 동결하되 공식 점수는 아직 모른다\n\n"
                f"공식과 같은 pooled RMSE proxy는 **{metrics['baseline_rmse']:.6f}→{metrics['candidate_rmse']:.6f}°C**로 "
                f"**{metrics['delta_rmse']:+.6f}°C** 개선됐다. paired KST-day 90% CI는 "
                f"**[{bootstrap['ci90_low']:+.6f}, {bootstrap['ci90_high']:+.6f}]°C**다. 이 후보는 제출 형식과 SHA 재현을 통과했지만 업로드하지 않았다."
            ),
        },
        {"id": "cards", "type": "metric-strip", "cardIds": ["delta", "ci", "factor"]},
        {
            "id": "method",
            "type": "markdown",
            "body": (
                "## 구조\n\n동결 physical-projection Deep을 기준으로 layer 2·4만 raw cross-fitted public-state expert로 바꾸고 layer 3은 유지한다. "
                "그 profile을 공개 layer 1·5 envelope와 방향에 투영한 다음, 기준에서 그 방향으로 layer 2는 10배, layer 4는 2배 이동하고 다시 투영한다. "
                "target-layer hidden 값, target salinity, 외부 관측값은 쓰지 않는다."
            ),
        },
        {"id": "chart_block", "type": "chart", "chartId": "block_chart"},
        {"id": "block_table", "type": "table", "tableId": "blocks_table"},
        {"id": "layer_table", "type": "table", "tableId": "layers"},
        {
            "id": "uncertainty",
            "type": "markdown",
            "body": (
                "## 불확실성\n\n**검증됨:** 전체 proxy와 같은 계절 2024년 9–10월, layer 2·4가 개선됐고 bootstrap CI가 0 아래다. "
                "**미검증:** 층별 배율과 layer route는 이미 본 OOF에서 선택됐으며 fresh holdout이 없다. 2025년 11–12월은 약 +0.00126°C 악화했다. "
                "따라서 공식 leaderboard 한 번의 정보가 이 adaptive 선택의 실제 일반화를 판별한다."
            ),
        },
        {
            "id": "decision",
            "type": "markdown",
            "body": (
                "## 동결 순위\n\n1. `P2_EXTRAPOLATED_SOFT_GATE_V2.csv` — 점수 최적화 1순위\n"
                "2. `P2_PHYSICAL_PROFILE_PROJECTION_V1.csv` — 안정형 fallback\n"
                "3. `P2_DEEP_STACK_V1.csv` — 원모델 fallback\n\n정확한 파일 승인 전 플랫폼 업로드는 금지한다."
            ),
        },
    ]
    generated = datetime.now().astimezone().isoformat()
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "P2 extrapolated soft-gate v2 점수 후보",
            "description": "물리 투영 Deep과 cross-fitted public-state expert의 적응형 점수 후보 검증",
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
                "blocks": block_rows,
                "chart": chart_rows,
                "layers": layer_rows,
                "deployment": deployment,
            },
        },
        "sources": [{"id": source["id"], "path": source["path"]} for source in sources],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", type=Path, default=Path("artifacts/p2_extrapolated_soft_gate_v2/result.json")
    )
    parser.add_argument(
        "--oof", type=Path, default=Path("artifacts/p2_extrapolated_soft_gate_v2/oof.parquet")
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
