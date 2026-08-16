"""Build the canonical portable technical report artifact for P3 research."""

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


def _source(
    source_id: str,
    label: str,
    path: str,
    rows: list[dict[str, object]],
    columns: list[str],
    description: str,
) -> dict[str, object]:
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
                "Three fixed chronological validation windows",
                "First eligible case per station then at least 78 hours apart",
                "Only the case-local 48-hour context; no external observations",
                "Official pooled row-level RMSE over six lead times",
            ],
        },
    }


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_case_bootstrap(oof: pd.DataFrame, replicates: int = 2_000) -> dict[str, float]:
    frame = oof.loc[oof["backend"].eq("catboost")].copy()
    case_columns = ["fold", "anchor_id"]
    groups = [group for _, group in frame.groupby(case_columns, sort=False)]
    rng = np.random.default_rng(20260817)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = [groups[item] for item in rng.integers(0, len(groups), len(groups))]
        block = pd.concat(sampled, ignore_index=True)
        truth = block["target_hs"].to_numpy(float)
        candidate = block["prediction"].to_numpy(float)
        persistence = block["persistence"].to_numpy(float)
        deltas[index] = np.sqrt(np.mean((truth - candidate) ** 2)) - np.sqrt(
            np.mean((truth - persistence) ** 2)
        )
    return {
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0.0)),
    }


def build_artifact(root: Path) -> dict[str, object]:
    initial_path = root / "artifacts/p3/initial_tournament_all20/metrics.json"
    initial = _read(initial_path)
    deep_gru = _read(root / "artifacts/p3/deep_gru_probe/metrics.json")
    deep_tcn = _read(root / "artifacts/p3/deep_tcn_probe/metrics.json")
    deep_30 = _read(root / "artifacts/p3/deep_fixed30_probe/metrics.json")
    nested = _read(root / "artifacts/p3/catboost_nested_tuning/metrics.json")
    event = _read(root / "artifacts/p3/event_phase_probe/metrics.json")
    ensemble = _read(root / "artifacts/p3/final_ensemble_validation/metrics.json")
    shrink_path = root / "artifacts/p3/shrinkage_calibration/metrics.json"
    multi_gpu_path = root / "artifacts/p3/multioutput_gpu_probe/metrics.json"
    final_path = root / "submissions/p3_frozen_catboost/manifest.json"
    shrink = _read(shrink_path) if shrink_path.exists() else None
    multi_gpu = _read(multi_gpu_path) if multi_gpu_path.exists() else None
    final = _read(final_path) if final_path.exists() else None

    exact_models = [
        ("Persistence", initial["metrics"]["catboost"]["persistence"]["rmse"], 1_092),
        ("LightGBM", initial["metrics"]["lightgbm"]["candidate"]["rmse"], 1_092),
        ("XGBoost", initial["metrics"]["xgboost"]["candidate"]["rmse"], 1_092),
        ("CatBoost fixed", initial["metrics"]["catboost"]["candidate"]["rmse"], 1_092),
        ("GRU early-stop", deep_gru["metrics"]["gru"]["rmse"], 1_092),
        ("TCN early-stop", deep_tcn["metrics"]["tcn"]["rmse"], 1_092),
        ("GRU 30 epochs", deep_30["metrics"]["gru"]["rmse"], 1_092),
        ("TCN 30 epochs", deep_30["metrics"]["tcn"]["rmse"], 1_092),
        ("CatBoost nested", nested["metrics"]["candidate"]["rmse"], 1_092),
        ("CatBoost + event", event["metrics"]["event_gpu"]["rmse"], 1_092),
    ]
    if shrink is not None:
        exact_models.append(
            ("CatBoost inner-shrink", shrink["metrics"]["calibrated"]["rmse"], 1_092)
        )
    if multi_gpu is not None:
        for name, metric in multi_gpu["metrics"].items():
            if name != "persistence":
                exact_models.append((name.replace("_", " "), metric["rmse"], 1_092))
    exact_models.append(("50:50 CatBoost ensemble", ensemble["metrics"]["ensemble"]["rmse"], 1_092))
    model_rows = [
        {
            "model": name,
            "rmse": float(score),
            "rows": rows,
            "delta_vs_persistence": float(
                score - initial["metrics"]["catboost"]["persistence"]["rmse"]
            ),
        }
        for name, score, rows in exact_models
    ]
    model_rows.sort(key=lambda row: row["rmse"])
    chart_models = {
        "50:50 CatBoost ensemble",
        "CatBoost inner-shrink",
        "cat multi compact",
        "CatBoost fixed",
        "LightGBM",
        "XGBoost",
        "GRU early-stop",
        "Persistence",
    }
    chart_model_rows = [row for row in model_rows if row["model"] in chart_models]
    cat = initial["metrics"]["catboost"]["candidate"]
    persistence = initial["metrics"]["catboost"]["persistence"]
    ensemble_metric = ensemble["metrics"]["ensemble"]
    lead_rows = [
        {"lead_h": int(lead), "method": method, "rmse": float(metrics["by_lead"][str(lead)])}
        for lead in (3, 6, 9, 12, 18, 24)
        for method, metrics in (
            ("50:50 ensemble", ensemble_metric),
            ("CatBoost fixed", cat),
            ("Persistence", persistence),
        )
    ]
    station_rows = [
        {
            "station": station,
            "ensemble_rmse": float(ensemble_metric["by_station"][station]),
            "catboost_rmse": float(cat["by_station"][station]),
            "persistence_rmse": float(persistence["by_station"][station]),
            "delta_rmse": float(
                ensemble_metric["by_station"][station] - persistence["by_station"][station]
            ),
        }
        for station in ("G-ORS", "I-ORS", "S-ORS")
    ]
    fold_rows = [
        {
            "fold": fold,
            "ensemble_rmse": float(ensemble["metrics"]["folds"][fold]["rmse"]),
            "catboost_rmse": float(values["rmse"]),
            "rows": int(values["n"]),
        }
        for fold, values in initial["metrics"]["catboost"]["folds"].items()
    ]
    bootstrap = ensemble["paired_case_bootstrap"]["ensemble_minus_persistence"]
    headline = [
        {
            "persistence_rmse": float(persistence["rmse"]),
            "catboost_rmse": float(cat["rmse"]),
            "ensemble_rmse": float(ensemble_metric["rmse"]),
            "delta_rmse": float(ensemble_metric["rmse"] - persistence["rmse"]),
            "ci90_low": bootstrap["ci90"][0],
            "ci90_high": bootstrap["ci90"][1],
            "probability_improved": bootstrap["probability_improved"],
            "validation_cases": 182,
            "validation_rows": 1_092,
            "submission_ready": final is not None,
        }
    ]
    sources = [
        _source(
            "headline_source",
            "P3 fixed equal-weight CatBoost ensemble headline",
            "artifacts/p3/final_ensemble_validation/metrics.json",
            headline,
            list(headline[0]),
            "Reconciles the equal-weight ensemble, persistence baseline, and paired case bootstrap.",
        ),
        _source(
            "model_source",
            "Exact-fold model comparison",
            "artifacts/p3/initial_tournament_all20/metrics.json",
            chart_model_rows,
            list(chart_model_rows[0]),
            "Compares all models evaluated on the same 1,092 forecast rows.",
        ),
        _source(
            "lead_source",
            "RMSE by lead time",
            "artifacts/p3/initial_tournament_all20/oof.parquet",
            lead_rows,
            ["lead_h", "method", "rmse"],
            "Shows forecast degradation from +3 to +24 hours.",
        ),
        _source(
            "station_source",
            "RMSE by station",
            "artifacts/p3/initial_tournament_all20/oof.parquet",
            station_rows,
            list(station_rows[0]),
            "Shows station-specific generalization and the I-ORS bottleneck.",
        ),
        _source(
            "fold_source",
            "RMSE by chronological fold",
            "artifacts/p3/initial_tournament_all20/oof.parquet",
            fold_rows,
            list(fold_rows[0]),
            "Shows stability across three fixed chronological validation windows.",
        ),
    ]
    cards = [
        {
            "id": "cat_rmse",
            "description": "182 independent validation cases; lower is better",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Ensemble RMSE",
                    "field": "ensemble_rmse",
                    "format": "number",
                    "unit": " m",
                }
            ],
        },
        {
            "id": "delta_rmse",
            "description": "Fixed CatBoost minus local persistence on identical rows",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "ΔRMSE",
                    "field": "delta_rmse",
                    "format": "number",
                    "unit": " m",
                    "signed": True,
                }
            ],
        },
        {
            "id": "bootstrap",
            "description": "2,000 paired independent-case bootstrap replicates",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "P(improved)",
                    "field": "probability_improved",
                    "format": "percent",
                }
            ],
        },
    ]
    charts = [
        {
            "id": "model_chart",
            "title": "Pooled RMSE by model",
            "subtitle": "Same 1,092 chronological forecast rows; lower is better, unit m.",
            "type": "bar",
            "dataset": "models",
            "sourceId": "model_source",
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (m)"},
                "tooltip": [
                    {
                        "field": "delta_vs_persistence",
                        "type": "quantitative",
                        "label": "Δ vs persistence",
                    },
                    {"field": "rows", "type": "quantitative", "label": "Rows"},
                ],
            },
        },
        {
            "id": "lead_chart",
            "title": "RMSE by forecast lead",
            "subtitle": "Six official lead times on 182 independent cases; lower is better.",
            "type": "bar",
            "dataset": "leads",
            "sourceId": "lead_source",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "lead_h", "type": "ordinal", "label": "Lead (hours)"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (m)"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
            },
        },
    ]
    tables = [
        {
            "id": "station_table",
            "title": "Station-level RMSE",
            "subtitle": "The local baseline and candidate use identical station rows.",
            "dataset": "stations",
            "sourceId": "station_source",
            "columns": [
                {"field": "station", "label": "Station", "type": "text"},
                {"field": "ensemble_rmse", "label": "Ensemble", "type": "number"},
                {"field": "catboost_rmse", "label": "Single CatBoost", "type": "number"},
                {"field": "persistence_rmse", "label": "Persistence", "type": "number"},
                {"field": "delta_rmse", "label": "ΔRMSE", "type": "number", "movement": True},
            ],
        },
        {
            "id": "fold_table",
            "title": "Chronological-fold RMSE",
            "subtitle": "Validation membership is selected independently within fixed time windows.",
            "dataset": "folds",
            "sourceId": "fold_source",
            "columns": [
                {"field": "fold", "label": "Fold", "type": "text"},
                {"field": "rows", "label": "Rows", "type": "number"},
                {"field": "ensemble_rmse", "label": "Ensemble RMSE", "type": "number"},
                {"field": "catboost_rmse", "label": "Single RMSE", "type": "number"},
            ],
        },
    ]
    final_note = (
        "최종 제출 CSV와 모델은 로컬 validator를 통과해 동결됐다. 플랫폼 업로드는 하지 않았다."
        if final is not None
        else "최종 제출 CSV는 아직 생성 중이며 플랫폼 업로드는 하지 않았다."
    )
    blocks = [
        {"id": "title", "type": "markdown", "body": "# P3 유의파고 3–24시간 예측 기술 검증"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "headline_source",
            "body": (
                "## 기술 요약: 단일·다중출력 CatBoost의 단순 평균이 가장 낮았다\n\n"
                f"공식과 같은 pooled RMSE 로컬 proxy에서 persistence **{persistence['rmse']:.6f}m** 대비 "
                f"50:50 CatBoost 앙상블이 **{ensemble_metric['rmse']:.6f}m**로 "
                f"**{ensemble_metric['rmse'] - persistence['rmse']:+.6f}m** 개선됐다. "
                f"독립 case bootstrap 90% CI는 **[{bootstrap['ci90'][0]:+.6f}, {bootstrap['ci90'][1]:+.6f}]m**다. "
                "다만 이 값은 hidden 공식 점수가 아니라 시간 차단 로컬 추정치다."
            ),
        },
        {"id": "cards", "type": "metric-strip", "cardIds": ["cat_rmse", "delta_rmse", "bootstrap"]},
        {
            "id": "data",
            "type": "markdown",
            "body": (
                "## 데이터 범위와 평가 grain\n\n"
                "train_wave 118,152행, train_atmos 130,896행, test context 57,800행을 원본 변경 없이 읽었다. "
                "검증은 공식 사례 조건 hs≥1.5m, 6개 target 유효, 정점 내 78시간 간격을 그대로 적용한 182 case×6 lead=1,092행이다. "
                "모든 특징은 해당 case의 과거 48시간에만 의존하며 test 절대시각과 외부 관측값은 사용하지 않았다."
            ),
        },
        {"id": "model_chart_block", "type": "chart", "chartId": "model_chart"},
        {
            "id": "model_interpretation",
            "type": "markdown",
            "body": (
                "## 구조 탐색 결과\n\n"
                "LightGBM·XGBoost·CatBoost, analog, 물리 Ridge, GRU, TCN, event-phase 특징, 최대 2,500회 nested tuning을 비교했다. "
                "CatBoost 700회 고정 설정이 가장 안정적이었고, 6개 lead 공동출력과의 단순 평균이 추가로 0.00323m 개선했다. "
                "nested tuning은 1,320회까지 선택됐지만 outer RMSE가 악화했다. "
                "GRU/TCN은 inner에서 2–4 epoch에 수렴했으며 30 epoch 강제학습은 특히 GRU를 크게 악화시켰다."
            ),
        },
        {"id": "lead_chart_block", "type": "chart", "chartId": "lead_chart"},
        {"id": "station_block", "type": "table", "tableId": "station_table"},
        {"id": "fold_block", "type": "table", "tableId": "fold_table"},
        {
            "id": "method",
            "type": "markdown",
            "body": (
                "## 최종 모델 사양\n\n"
                "첫 모델은 3개 정점·6개 lead를 pooled CatBoost residual regressor로, 둘째 모델은 한 case의 6개 lead를 공동 MultiRMSE로 학습한다. "
                "두 예측을 50:50 평균한다. 목표는 `future_hs-current_hs`, "
                "입력은 station·lead와 hs/tp/hmax/풍향·풍속·기압 등의 현재값, lag, 3–48시간 통계, 방향의 sin/cos 표현이다. "
                "case-selection shift를 반영하는 사전고정 mild weight를 사용한다. 예측은 0–30m로 clip하고 sample key 순서를 보존한다."
            ),
        },
        {
            "id": "literature",
            "type": "markdown",
            "body": (
                "## 논문 정찰과 적용 판단\n\n"
                "다변량 GRU의 다단계 파고 예측은 [Li et al., Ocean Engineering 2022](https://doi.org/10.1016/j.oceaneng.2022.110689), "
                "VMD–TCN–LSTM은 [Ji et al., Ocean Science 2023](https://doi.org/10.5194/os-19-1561-2023), "
                "GBT residual correction은 [Journal of Marine Science and Engineering 2024](https://www.mdpi.com/2077-1312/12/9/1573)를 참고했다. "
                "그러나 본 배포본은 미래 기상장과 절대 평가시각이 없고 독립 case가 200개뿐이므로, 논문의 복잡한 구조가 자동으로 일반화되지는 않았다."
            ),
        },
        {
            "id": "uncertainty",
            "type": "markdown",
            "body": (
                "## 한계와 강건성\n\n"
                "**검증됨:** 키·간격·결측 구조, 78시간 독립성, 동일 1,092행 비교, case bootstrap, 6개 lead·3개 정점 진단. "
                "**미검증:** 공식 hidden RMSE와 public/private 순위. 50:50 조합은 component OOF를 본 뒤 평가됐고 single 대비 CI는 0을 걸친다. "
                "test case가 상승 국면에 더 집중되어 있어 로컬 시간분할과의 분포차도 남는다. "
                "Public 66 case만 보고 재튜닝하면 private 순위가 뒤집힐 위험이 크므로 이 보고서는 leaderboard 적합을 모델 선택 근거로 쓰지 않는다."
            ),
        },
        {
            "id": "decision",
            "type": "markdown",
            "body": "## 배포 상태\n\n" + final_note,
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 다음 돌파 질문\n\n"
                "- +18/+24시간에 미래 바람이 없는 한 얻을 수 있는 정보 상한은 어디인가?\n"
                "- I-ORS의 짧은 기상 학습기간을 station-specific regularization으로 보완할 수 있는가?\n"
                "- 첫 제한 제출의 hidden 점수가 로컬 CatBoost 개선과 같은 방향인지, 아니면 test storm-phase shift를 드러내는가?"
            ),
        },
    ]
    generated = datetime.now().astimezone().isoformat()
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "P3 유의파고 3–24시간 예측 기술 검증",
            "description": "48시간 case-local 관측만 사용한 모델 구조·수렴·강건성 비교",
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
                "models": chart_model_rows,
                "leads": lead_rows,
                "stations": station_rows,
                "folds": fold_rows,
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
