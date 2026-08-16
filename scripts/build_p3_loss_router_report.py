"""Build the canonical technical report for the P3 lead-long loss router."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _source(
    source_id: str,
    label: str,
    path: str,
    rows: list[dict[str, object]],
    description: str,
) -> dict[str, object]:
    columns = list(rows[0])
    sql = " UNION ALL ".join(
        "SELECT " + ", ".join(f"{_literal(row.get(column))} AS {column}" for column in columns)
        for row in rows
    )
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": description,
            "tables_used": [path],
            "filters": [
                "182 independent local validation cases",
                "Six official forecast leads pooled by row",
                "Router uses current and earlier folds only",
                "No external observations or hidden labels",
            ],
        },
    }


def build_artifact(root: Path) -> dict[str, object]:
    metrics_path = root / "artifacts/p3/lead_long_loss_router/metrics.json"
    validation_path = root / "artifacts/p3/lead_long_loss_router/independent_validation.json"
    manifest_path = root / "submissions/p3_lead_long_loss_router/manifest.json"
    metrics = _read(metrics_path)
    validation = _read(validation_path)
    manifest = _read(manifest_path)
    candidate = float(validation["metrics"]["candidate_rmse"])
    frozen = float(validation["metrics"]["frozen_rmse"])
    delta = float(validation["metrics"]["delta_rmse"])
    ci = [float(value) for value in validation["bootstrap"]["ci90"]]

    headline = [
        {
            "candidate_rmse": candidate,
            "frozen_rmse": frozen,
            "delta_rmse": delta,
            "relative_reduction_pct": 100.0 * (frozen - candidate) / frozen,
            "ci90_low": ci[0],
            "ci90_high": ci[1],
            "probability_improved": float(validation["bootstrap"]["probability_improved"]),
            "submission_rows": int(manifest["test_rows"]),
        }
    ]
    lead_rows: list[dict[str, object]] = []
    for lead, values in validation["metrics"]["by_lead"].items():
        for method, field in (
            ("Frozen ensemble", "frozen_rmse"),
            ("Soft router", "candidate_rmse"),
        ):
            lead_rows.append(
                {
                    "lead_h": int(lead),
                    "method": method,
                    "rmse": float(values[field]),
                    "delta_rmse": float(values["candidate_rmse"] - values["frozen_rmse"]),
                    "rows": int(values["rows"]),
                }
            )
    fold_rows = [
        {
            "fold": fold,
            "rows": int(values["rows"]),
            "frozen_rmse": float(values["frozen_rmse"]),
            "candidate_rmse": float(values["candidate_rmse"]),
            "delta_rmse": float(values["candidate_rmse"] - values["frozen_rmse"]),
        }
        for fold, values in validation["metrics"]["by_fold"].items()
    ]
    selection_rows = [
        {
            "fold": row["fold"],
            "past_cases": int(row["past_cases"]),
            "fit_cases": int(row["selection_fit_cases"]),
            "calibration_cases": int(row["selection_calibration_cases"]),
            "selected": row["selected"],
            "strength": float(row["config"]["strength"]),
            "temperature_multiplier": float(row["config"]["temperature_multiplier"]),
            "current_truth_used": bool(row["current_fold_truth_used_for_selection"]),
        }
        for row in metrics["selections"]
    ]
    weight_rows = [
        {
            "component": component,
            "local_oof_mean": float(values["mean"]),
            "hidden_test_mean": float(manifest["weight_summary"][component]["mean"]),
            "hidden_test_p10": float(manifest["weight_summary"][component]["p10"]),
            "hidden_test_p90": float(manifest["weight_summary"][component]["p90"]),
        }
        for component, values in metrics["weight_summary"].items()
    ]

    sources = [
        _source(
            "headline_source",
            "Independent lead-long router validation",
            "artifacts/p3/lead_long_loss_router/independent_validation.json",
            headline,
            "Independently reconciled pooled RMSE, paired case bootstrap, and output grain.",
        ),
        _source(
            "lead_source",
            "Lead-level frozen/router RMSE",
            "artifacts/p3/lead_long_loss_router/independent_validation.json",
            lead_rows,
            "Compares the two methods on each of the six official leads.",
        ),
        _source(
            "fold_source",
            "Chronological fold comparison",
            "artifacts/p3/lead_long_loss_router/independent_validation.json",
            fold_rows,
            "Reconciles candidate and frozen RMSE inside each time-extrapolation fold.",
        ),
        _source(
            "selection_source",
            "Past-only router selections",
            "artifacts/p3/lead_long_loss_router/metrics.json",
            selection_rows,
            "Shows the chronological fit/calibration population and selected fixed grid point.",
        ),
        _source(
            "weight_source",
            "OOF and hidden-input router weights",
            "submissions/p3_lead_long_loss_router/manifest.json",
            weight_rows,
            "Aggregate component-weight diagnostics; no hidden truth is available or used.",
        ),
    ]
    cards = [
        {
            "id": "candidate_card",
            "description": "182-case chronological local validation",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Router RMSE",
                    "field": "candidate_rmse",
                    "format": "number",
                    "unit": " m",
                },
                {"label": "Frozen", "field": "frozen_rmse", "format": "number", "unit": " m"},
            ],
        },
        {
            "id": "delta_card",
            "description": "Negative is better",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "ΔRMSE",
                    "field": "delta_rmse",
                    "format": "number",
                    "unit": " m",
                    "signed": True,
                },
                {
                    "label": "Relative reduction",
                    "field": "relative_reduction_pct",
                    "format": "number",
                    "unit": "%",
                },
            ],
        },
        {
            "id": "confidence_card",
            "description": "2,000 paired case resamples",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {"label": "P(improved)", "field": "probability_improved", "format": "percent"},
                {"label": "CI90 upper", "field": "ci90_high", "format": "number", "unit": " m"},
            ],
        },
    ]
    charts = [
        {
            "id": "lead_chart",
            "title": "RMSE by forecast lead",
            "subtitle": "182 cases per lead; lower is better; short leads are exact frozen no-ops.",
            "type": "bar",
            "dataset": "leads",
            "sourceId": "lead_source",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "lead_h", "type": "ordinal", "label": "Lead (hours)"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (m)"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "delta_rmse", "type": "quantitative", "label": "Router ΔRMSE"},
                    {"field": "rows", "type": "quantitative", "label": "Rows"},
                ],
            },
        }
    ]
    tables = [
        {
            "id": "fold_table",
            "title": "Chronological fold results",
            "subtitle": "First fold is an exact no-op; later folds use only earlier OOF cases.",
            "dataset": "folds",
            "sourceId": "fold_source",
            "columns": [
                {"field": "fold", "label": "Fold", "type": "text"},
                {"field": "rows", "label": "Rows", "type": "number"},
                {"field": "frozen_rmse", "label": "Frozen", "type": "number"},
                {"field": "candidate_rmse", "label": "Router", "type": "number"},
                {"field": "delta_rmse", "label": "ΔRMSE", "type": "number", "movement": True},
            ],
        },
        {
            "id": "selection_table",
            "title": "Past-only router selection",
            "subtitle": "Five fixed choices including no-op; current-fold truth is excluded.",
            "dataset": "selections",
            "sourceId": "selection_source",
            "columns": [
                {"field": "fold", "label": "Applied fold", "type": "text"},
                {"field": "past_cases", "label": "Past cases", "type": "number"},
                {"field": "fit_cases", "label": "Fit", "type": "number"},
                {"field": "calibration_cases", "label": "Calibration", "type": "number"},
                {"field": "selected", "label": "Selected", "type": "text"},
                {"field": "strength", "label": "Strength", "type": "number"},
            ],
        },
        {
            "id": "weight_table",
            "title": "Component-weight behavior",
            "subtitle": "Means include short leads, where persistence weight is fixed to zero.",
            "dataset": "weights",
            "sourceId": "weight_source",
            "columns": [
                {"field": "component", "label": "Component", "type": "text"},
                {"field": "local_oof_mean", "label": "OOF mean", "type": "number"},
                {"field": "hidden_test_mean", "label": "Test-input mean", "type": "number"},
                {"field": "hidden_test_p10", "label": "Test p10", "type": "number"},
                {"field": "hidden_test_p90", "label": "Test p90", "type": "number"},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# P3 장기 리드 손실 라우터 검증"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "headline_source",
            "body": (
                "## 기술 요약: 장기 리드만 동적으로 섞어 처음으로 불확실성 기준까지 통과했다\n\n"
                f"과거 case에서 cross-fit된 component loss만 학습한 soft router가 로컬 pooled RMSE를 "
                f"**{frozen:.6f} → {candidate:.6f}m**로 낮췄다(Δ **{delta:.6f}m**, 상대 "
                f"**{100.0 * (frozen - candidate) / frozen:.2f}%**). 2,000회 paired case bootstrap의 "
                f"90% CI는 **[{ci[0]:.6f}, {ci[1]:.6f}]m**로 0 아래다. +3·+6·+9시간은 기존 값을 "
                "정확히 유지하고 +12·+18·+24시간에만 router를 적용했다."
            ),
        },
        {
            "id": "cards",
            "type": "metric-strip",
            "cardIds": ["candidate_card", "delta_card", "confidence_card"],
        },
        {
            "id": "lead_finding",
            "type": "markdown",
            "sourceId": "lead_source",
            "body": (
                "## 개선은 문제에서 의도적으로 어렵게 만든 장기 리드에 집중됐다\n\n"
                "+12·+18·+24시간 RMSE가 각각 0.00390·0.00980·0.01461m 감소했다. 짧은 리드는 "
                "동일한 50:50 single/multi 예측을 사용하므로 변화가 없다. 이는 persistence를 포함한 "
                "동적 혼합이 긴 시간의 성장·감쇠 불확실성에는 도움이 되지만 단기에는 불필요하다는 결과다."
            ),
        },
        {"id": "lead_chart_block", "type": "chart", "chartId": "lead_chart"},
        {
            "id": "fold_finding",
            "type": "markdown",
            "sourceId": "fold_source",
            "body": (
                "## 실제 router가 적용된 두 시간 외삽 fold에서 모두 개선됐다\n\n"
                "첫 2024 H2 fold는 과거 cross-fit case가 없어 exact no-op이다. 이후 winter transition과 "
                "2025 H1은 각각 약 0.00756m와 0.00636m 개선했다. 현재 fold 정답은 router 선택이나 "
                "학습에 사용하지 않았다."
            ),
        },
        {"id": "fold_table_block", "type": "table", "tableId": "fold_table"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 평가 범위와 기준\n\n"
                "평가는 세 정점의 182개 독립 case×6 lead=1,092행에 대한 pooled RMSE다. frozen 기준은 "
                "single-output CatBoost와 six-output CatBoost의 50:50 앙상블이다. router 입력은 station, "
                "같은 case의 과거 48시간 관측 요약, 두 component가 예측한 6-lead trajectory뿐이다. "
                "외부 관측, test 절대시각, hidden target은 사용하지 않았다."
            ),
        },
        {
            "id": "method",
            "type": "markdown",
            "body": (
                "## 모델 사양과 누출 차단\n\n"
                "각 OOF case에서 single·multi·persistence의 lead별 squared error를 만든 뒤 작은 Ridge가 "
                "log loss를 예측한다. 예측 loss의 softmax를 기존 50:50과 절반만 혼합한다. 후보는 no-op "
                "포함 5개다. winter router는 2024 H2의 앞 29 case로 fit하고 뒤 20 case로 선택했으며, "
                "2025 H1 router는 2024 H2로 fit하고 winter 80 case로 선택했다. 최종 hidden 후보는 가장 "
                "최근 선택값을 고정하고 182 OOF case 전체로 router만 재학습했다."
            ),
        },
        {"id": "selection_block", "type": "table", "tableId": "selection_table"},
        {"id": "weight_block", "type": "table", "tableId": "weight_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "headline_source",
            "body": (
                "## 불확실성과 적응적 연구 한계\n\n"
                "독립 validator가 키·source 예측·convex weight·첫 fold no-op·짧은 lead no-op·RMSE·bootstrap을 "
                "재계산해 모두 일치시켰다. 그러나 이 router 가설은 동일 outer OOF의 실패 구조를 본 뒤 선택됐다. "
                "따라서 CI가 0 아래라는 사실은 계산상 강한 근거지만 완전히 새로운 virgin holdout의 증거는 아니다. "
                "공식 hidden 점수와 로컬 RMSE를 직접 비교해서도 안 된다."
            ),
        },
        {
            "id": "recommendation",
            "type": "markdown",
            "body": (
                "## 권장 다음 단계\n\n"
                "이 파일을 P3의 새 로컬 1순위 제출 후보로 보존한다. 첫 공식 제출 전에는 기존 frozen과 router "
                "두 파일의 SHA를 함께 제시하고 사용자가 하나를 승인해야 한다. 공식 점수가 확인되기 전 추가 "
                "router 파라미터 탐색은 중단해 같은 182 case에 대한 적응을 늘리지 않는다."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 남은 질문\n\n"
                "- 2025-07~2026-06 hidden 사례에서도 장기 lead의 persistence 혼합이 같은 방향으로 작동하는가?\n"
                "- 공개 점수가 개선될 경우 private 일반화를 위해 가중치를 그대로 동결할 것인가?\n"
                "- 공식 점수가 악화되면 입력 분포 이동과 component calibration 중 어느 쪽이 원인인가?"
            ),
        },
    ]
    generated = datetime.now().astimezone().isoformat()
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "P3 장기 리드 손실 라우터 검증",
            "description": "Cross-fitted component-loss soft router의 시간 외삽 성능과 제출 후보 검증",
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
                "leads": lead_rows,
                "folds": fold_rows,
                "selections": selection_rows,
                "weights": weight_rows,
            },
        },
        "sources": [{"id": source["id"], "path": source["path"]} for source in sources],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_artifact(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": args.output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
