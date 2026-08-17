"""Build the canonical technical report for P3 long-lead persistence shrinkage."""

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
                "182 local chronological validation cases",
                "Six official leads pooled by row",
                "Shrinkage active only for 12, 18, and 24 hours",
                "No external observations or hidden labels",
            ],
        },
    }


def build_artifact(root: Path) -> dict[str, object]:
    metrics_path = root / "artifacts/p3/long_persistence_shrink/metrics.json"
    validation_path = root / "artifacts/p3/long_persistence_shrink/independent_validation.json"
    submission_path = root / "submissions/p3_long_persistence_shrink/manifest.json"
    metrics = _read(metrics_path)
    validation = _read(validation_path)
    submission = _read(submission_path)
    candidate = float(validation["metrics"]["candidate_rmse"])
    incumbent = float(validation["metrics"]["incumbent_rmse"])
    delta = float(validation["metrics"]["delta_rmse"])
    ci = [float(value) for value in metrics["paired_case_bootstrap"]["ci90"]]
    probability = float(metrics["paired_case_bootstrap"]["probability_improved"])
    relative = 100.0 * (incumbent - candidate) / incumbent
    headline = [
        {
            "candidate_rmse": candidate,
            "incumbent_rmse": incumbent,
            "delta_rmse": delta,
            "relative_reduction_pct": relative,
            "ci90_low": ci[0],
            "ci90_high": ci[1],
            "probability_improved": probability,
            "submission_rows": int(submission["rows"]),
        }
    ]
    lead_rows: list[dict[str, object]] = []
    for lead, values in metrics["metrics"]["by_lead"].items():
        for method, field in (
            ("Incumbent router", "incumbent_rmse"),
            ("20% persistence shrink", "candidate_rmse"),
        ):
            lead_rows.append(
                {
                    "lead_h": int(lead),
                    "method": method,
                    "rmse": float(values[field]),
                    "delta_rmse": float(values["delta_rmse"]),
                    "rows": int(values["rows"]),
                }
            )
    fold_rows = [
        {
            "fold": fold,
            "rows": int(values["rows"]),
            "incumbent_rmse": float(values["incumbent_rmse"]),
            "candidate_rmse": float(values["candidate_rmse"]),
            "delta_rmse": float(values["delta_rmse"]),
        }
        for fold, values in metrics["metrics"]["by_fold"].items()
    ]
    station_rows = [
        {
            "station": station,
            "rows": int(values["rows"]),
            "incumbent_rmse": float(values["incumbent_rmse"]),
            "candidate_rmse": float(values["candidate_rmse"]),
            "delta_rmse": float(values["delta_rmse"]),
        }
        for station, values in metrics["metrics"]["by_station"].items()
    ]
    sensitivity_rows = [
        {
            "persistence_weight": float(weight),
            "rmse": float(score),
            "delta_vs_incumbent": float(score) - incumbent,
            "selected": weight == "0.20",
        }
        for weight, score in metrics["bounded_sensitivity_rmse"].items()
    ]
    sources = [
        _source(
            "headline_source",
            "Independent long-lead shrinkage validation",
            "artifacts/p3/long_persistence_shrink/independent_validation.json",
            headline,
            "Reconciled pooled RMSE and the fixed prediction reconstruction.",
        ),
        _source(
            "lead_source",
            "Lead-level shrinkage metrics",
            "artifacts/p3/long_persistence_shrink/metrics.json",
            lead_rows,
            "Incumbent and candidate RMSE for every official forecast lead.",
        ),
        _source(
            "fold_source",
            "Chronological fold metrics",
            "artifacts/p3/long_persistence_shrink/metrics.json",
            fold_rows,
            "Time-extrapolation fold comparison at the official row-pooled grain.",
        ),
        _source(
            "station_source",
            "Station metrics",
            "artifacts/p3/long_persistence_shrink/metrics.json",
            station_rows,
            "Station-level robustness comparison for G-ORS, I-ORS, and S-ORS.",
        ),
        _source(
            "sensitivity_source",
            "Bounded scalar sensitivity",
            "artifacts/p3/long_persistence_shrink/metrics.json",
            sensitivity_rows,
            "Three nearby fixed long-lead persistence weights; diagnostic only.",
        ),
    ]
    cards = [
        {
            "id": "candidate_card",
            "description": "182 cases × 6 leads; lower is better",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Candidate RMSE",
                    "field": "candidate_rmse",
                    "format": "number",
                    "unit": " m",
                },
                {
                    "label": "Incumbent",
                    "field": "incumbent_rmse",
                    "format": "number",
                    "unit": " m",
                },
            ],
        },
        {
            "id": "delta_card",
            "description": "Negative ΔRMSE is better",
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
            "description": "5,000 paired case resamples",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "P(improved)",
                    "field": "probability_improved",
                    "format": "percent",
                },
                {
                    "label": "CI90 upper",
                    "field": "ci90_high",
                    "format": "number",
                    "unit": " m",
                },
            ],
        },
    ]
    charts = [
        {
            "id": "lead_chart",
            "title": "RMSE by forecast lead",
            "subtitle": "182 cases per lead; lower is better; 3/6/9h are exact incumbent values.",
            "type": "bar",
            "dataset": "leads",
            "sourceId": "lead_source",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "lead_h", "type": "ordinal", "label": "Lead (hours)"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (m)"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "delta_rmse", "type": "quantitative", "label": "ΔRMSE"},
                    {"field": "rows", "type": "quantitative", "label": "Rows"},
                ],
            },
        }
    ]
    common_columns = [
        {"field": "rows", "label": "Rows", "type": "number"},
        {"field": "incumbent_rmse", "label": "Incumbent", "type": "number"},
        {"field": "candidate_rmse", "label": "Candidate", "type": "number"},
        {
            "field": "delta_rmse",
            "label": "ΔRMSE",
            "type": "number",
            "movement": True,
        },
    ]
    tables = [
        {
            "id": "fold_table",
            "title": "Chronological validation folds",
            "subtitle": "The same fixed 20% correction improves every observed period.",
            "dataset": "folds",
            "sourceId": "fold_source",
            "columns": [{"field": "fold", "label": "Fold", "type": "text"}, *common_columns],
        },
        {
            "id": "station_table",
            "title": "Station robustness",
            "subtitle": "All three stations improve at the same pooled-row definition.",
            "dataset": "stations",
            "sourceId": "station_source",
            "columns": [
                {"field": "station", "label": "Station", "type": "text"},
                *common_columns,
            ],
        },
        {
            "id": "sensitivity_table",
            "title": "Bounded weight sensitivity",
            "subtitle": "Nearby weights all improve; 0.20 is frozen for the submission candidate.",
            "dataset": "sensitivity",
            "sourceId": "sensitivity_source",
            "columns": [
                {"field": "persistence_weight", "label": "Weight", "type": "number"},
                {"field": "rmse", "label": "RMSE", "type": "number"},
                {
                    "field": "delta_vs_incumbent",
                    "label": "Δ vs incumbent",
                    "type": "number",
                    "movement": True,
                },
                {"field": "selected", "label": "Frozen", "type": "boolean"},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# P3 장기 리드 persistence 보정 검증"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "headline_source",
            "body": (
                "## 기술 요약: 단일 20% 장기 리드 보정이 검증 기준을 다시 낮췄다\n\n"
                f"현재 component-loss router의 +12·+18·+24시간 예측을 기준시각 유의파고 쪽으로 "
                f"20% 축소해 로컬 pooled RMSE를 **{incumbent:.6f} → {candidate:.6f}m**로 "
                f"낮췄다(Δ **{delta:.6f}m**, 상대 **{relative:.2f}%**). 5,000회 paired case "
                f"bootstrap 90% CI는 **[{ci[0]:.6f}, {ci[1]:.6f}]m**, 개선 확률은 "
                f"**{probability:.1%}**다. +3·+6·+9시간은 기존 후보와 bitwise 동일하다."
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
                "## 이득은 물리적으로 불확실성이 커지는 +12~24시간에만 나타났다\n\n"
                "+12·+18·+24시간 RMSE는 각각 0.00400·0.01407·0.01730m 감소했다. 짧은 "
                "리드는 변경하지 않았다. 미래 기상장이 없는 문제에서 장기 CatBoost 예측의 성장·감쇠 "
                "진폭을 지속성 쪽으로 보수화한 결과로 해석할 수 있다."
            ),
        },
        {"id": "lead_chart_block", "type": "chart", "chartId": "lead_chart"},
        {
            "id": "fold_finding",
            "type": "markdown",
            "sourceId": "fold_source",
            "body": (
                "## 세 시간 외삽 구간에서 같은 방향으로 개선됐다\n\n"
                "2024 H2 storm, winter transition, 2025 H1의 ΔRMSE는 각각 -0.01238, "
                "-0.00450, -0.00488m다. 특정 계절 하나에만 의존한 결과는 아니지만, 모든 구간은 "
                "동일한 182-case 연구 과정에 이미 노출됐다는 점을 함께 봐야 한다."
            ),
        },
        {"id": "fold_table_block", "type": "table", "tableId": "fold_table"},
        {
            "id": "station_finding",
            "type": "markdown",
            "sourceId": "station_source",
            "body": (
                "## G·I·S 세 기지 모두 개선돼 기지 특화 보정은 필요하지 않았다\n\n"
                "기지별 ΔRMSE는 G -0.00447, I -0.00699, S -0.00836m다. station별 "
                "파라미터를 추가하지 않고 하나의 장기 weight만 사용해 분산을 제한했다."
            ),
        },
        {"id": "station_table_block", "type": "table", "tableId": "station_table"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 평가 범위와 지표 정의\n\n"
                "로컬 검증 대상은 세 정점의 182개 독립 case×6 lead=1,092행이며, 공식과 같은 "
                "row-pooled RMSE를 계산했다. incumbent는 과거 fold component loss로 학습한 장기 "
                "soft router다. persistence는 각 case의 step_minute=0 hs를 모든 리드에 유지한 값이다."
            ),
        },
        {
            "id": "method",
            "type": "markdown",
            "body": (
                "## 모델 사양: 재학습 없는 단일 convex 보정\n\n"
                "예측식은 장기 리드에서 `0.8 × incumbent + 0.2 × persistence`이고, 짧은 리드는 "
                "incumbent 그대로다. target, test 절대시각, 외부 관측을 입력으로 사용하지 않는다. "
                "독립 validator는 1,092개 키·source 열·incumbent 재현·공식 수식·RMSE를 다시 계산했다."
            ),
        },
        {
            "id": "robustness",
            "type": "markdown",
            "sourceId": "sensitivity_source",
            "body": (
                "## 15~25%의 좁은 범위에서 결론이 뒤집히지 않았다\n\n"
                "0.15·0.20·0.25의 RMSE는 0.781161·0.780161·0.779594m로 모두 incumbent보다 "
                "낮다. 가장 낮은 0.25를 끝까지 추격하지 않고, bootstrap 상한이 0 아래인 0.20을 "
                "동결해 추가 적응을 제한했다."
            ),
        },
        {"id": "sensitivity_table_block", "type": "table", "tableId": "sensitivity_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "headline_source",
            "body": (
                "## 이 결과는 공식 점수가 아니라 적응적으로 선택된 로컬 추정치다\n\n"
                "가설과 20% 값은 기존 OOF 오류를 본 뒤 선택됐으므로 virgin holdout 증거가 아니다. "
                "bootstrap은 현재 182 case 안의 sampling uncertainty만 반영하며 2025-07~2026-06의 "
                "분포 이동은 측정하지 못한다. Public 66 case 점수만으로 weight를 다시 조정하면 "
                "Private 일반화가 악화될 수 있다."
            ),
        },
        {
            "id": "recommendation",
            "type": "markdown",
            "body": (
                "## 권장 다음 단계: 새 로컬 1순위로 보존하고 weight는 더 탐색하지 않는다\n\n"
                "1,200행 제출 파일과 SHA를 동결한다. 실제 업로드는 사용자가 정확한 파일을 승인한 "
                "뒤에만 수행한다. 공식 점수가 나와도 같은 날 추가 weight 탐색은 하지 않고, 이전 "
                "router 후보를 안전 대조군으로 유지한다."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 남은 질문\n\n"
                "- hidden 기간에도 장기 성장·감쇠 진폭이 같은 정도로 과대 추정되는가?\n"
                "- Public 개선이 확인돼도 Private용 weight를 0.20으로 그대로 동결할 것인가?\n"
                "- 악화 시 원인이 persistence 비중인지, component router의 기간 이동인지 어떻게 분리할 것인가?"
            ),
        },
    ]
    generated = datetime.now().astimezone().isoformat()
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "P3 장기 리드 persistence 보정 검증",
            "description": "장기 리드 20% persistence shrinkage의 로컬 시간 외삽 검증",
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
                "stations": station_rows,
                "sensitivity": sensitivity_rows,
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
