"""Add a target-blind seasonal-regime diagnostic to the durable P2 report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from p2_restore.data import load_p2_data

PREFIX = "regime_diag_"
PUBLIC_LAYERS = (1, 5, 6, 7, 8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_literal(value: object) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _union_sql(rows: list[dict[str, object]], columns: list[str]) -> str:
    return " UNION ALL ".join(
        "SELECT " + ", ".join(f"{_sql_literal(row.get(column))} AS {column}" for column in columns)
        for row in rows
    )


def _public_features(observations: pd.DataFrame) -> pd.DataFrame:
    public = observations.loc[
        observations["layer"].isin(PUBLIC_LAYERS), ["time", "layer", "temp", "psal"]
    ].copy()
    public["time"] = pd.to_datetime(public["time"], utc=True)
    temperature = public.pivot(index="time", columns="layer", values="temp").sort_index()
    salinity = public.pivot(index="time", columns="layer", values="psal").sort_index()
    result = pd.DataFrame(index=temperature.index)
    result["abs_t1_t5"] = (temperature[1] - temperature[5]).abs()
    result["public_temp_range"] = temperature.max(axis=1) - temperature.min(axis=1)
    result["public_psal_range"] = salinity.max(axis=1) - salinity.min(axis=1)
    gap = pd.Series(temperature.index, index=temperature.index).diff().dt.total_seconds().div(60)
    segment = gap.ne(10).cumsum().to_numpy()
    result["contrast_change_24h"] = result["abs_t1_t5"].groupby(segment).diff(144).abs()
    result["range_change_24h"] = result["public_temp_range"].groupby(segment).diff(144).abs()
    return result


def _quantiles(values: pd.Series) -> dict[str, object]:
    finite = values.dropna().to_numpy(float)
    result: dict[str, object] = {"n": int(len(finite))}
    if len(finite):
        result.update(
            {
                "q10": float(np.quantile(finite, 0.1)),
                "median": float(np.median(finite)),
                "q90": float(np.quantile(finite, 0.9)),
            }
        )
    return result


def _normalized_wasserstein(reference: pd.Series, hidden: pd.Series) -> float | None:
    left = reference.dropna().to_numpy(float)
    right = hidden.dropna().to_numpy(float)
    if not len(left) or not len(right):
        return None
    scale = float(np.subtract(*np.quantile(right, [0.75, 0.25])))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(right)) or 1.0
    return float(wasserstein_distance(left, right) / scale)


def _diagnose(data_dir: Path, deep_oof: Path, state_result: Path, tuning_result: Path) -> dict:
    data = load_p2_data(data_dir)
    features = _public_features(data.observations)
    oof = pd.read_parquet(deep_oof, columns=["time", "block"])
    oof["time"] = pd.to_datetime(oof["time"], utc=True)
    supports = {
        name: pd.Index(group["time"].unique()) for name, group in oof.groupby("block", sort=False)
    }
    test_index = data.test_index.copy()
    test_index["time"] = pd.to_datetime(test_index["time"], utc=True)
    supports["2025_sep_oct_hidden_public"] = pd.Index(test_index["time"].unique())
    aligned = {name: features.reindex(times) for name, times in supports.items()}

    reference = aligned["2024_sep_oct"]["abs_t1_t5"].dropna().to_numpy(float)
    q33, q67 = np.quantile(reference, [1 / 3, 2 / 3])
    hidden = aligned["2025_sep_oct_hidden_public"]
    block_rows: list[dict[str, object]] = []
    for name, frame in aligned.items():
        contrast = frame["abs_t1_t5"].dropna().to_numpy(float)
        contrast_summary = _quantiles(frame["abs_t1_t5"])
        range_summary = _quantiles(frame["public_temp_range"])
        change_summary = _quantiles(frame["contrast_change_24h"])
        block_rows.append(
            {
                "block": name,
                "timestamps": int(len(frame)),
                "contrast_coverage": float(len(contrast) / len(frame)),
                "contrast_median": contrast_summary.get("median"),
                "contrast_q90": contrast_summary.get("q90"),
                "temp_range_median": range_summary.get("median"),
                "contrast_change_24h_median": change_summary.get("median"),
                "distance_to_hidden_contrast": (
                    0.0
                    if name == "2025_sep_oct_hidden_public"
                    else _normalized_wasserstein(frame["abs_t1_t5"], hidden["abs_t1_t5"])
                ),
                "distance_to_hidden_temp_range": (
                    0.0
                    if name == "2025_sep_oct_hidden_public"
                    else _normalized_wasserstein(
                        frame["public_temp_range"], hidden["public_temp_range"]
                    )
                ),
                "reference_low_share": (float(np.mean(contrast <= q33)) if len(contrast) else None),
                "reference_middle_share": (
                    float(np.mean((contrast > q33) & (contrast < q67))) if len(contrast) else None
                ),
                "reference_high_share": (
                    float(np.mean(contrast >= q67)) if len(contrast) else None
                ),
            }
        )

    weekly_rows: list[dict[str, object]] = []
    for name in ("2024_sep_oct", "2025_sep_oct_hidden_public"):
        frame = aligned[name].copy()
        local = frame.index.tz_convert("Asia/Seoul")
        year = 2024 if name.startswith("2024") else 2025
        start = pd.Timestamp(f"{year}-09-01", tz="Asia/Seoul")
        frame["week"] = ((local.normalize() - start).days // 7 + 1).astype(int)
        for week, group in frame.groupby("week", sort=True):
            contrast = group["abs_t1_t5"]
            weekly_rows.append(
                {
                    "period": "2024 same season" if year == 2024 else "2025 hidden public",
                    "week": int(week),
                    "contrast_median": (
                        float(contrast.median()) if int(contrast.notna().sum()) else None
                    ),
                    "public_temp_range_median": float(group["public_temp_range"].median()),
                    "contrast_valid": int(contrast.notna().sum()),
                    "timestamps": int(len(group)),
                    "contrast_coverage": float(contrast.notna().mean()),
                }
            )

    state = json.loads(state_result.read_text(encoding="utf-8"))
    layerwise = json.loads(
        (tuning_result.parent / "catboost_layerwise" / "result.json").read_text(encoding="utf-8")
    )
    convergence_rows = [
        {"outer_block": block, "layer": int(layer), "iterations": int(value)}
        for block, by_layer in layerwise["tuning"]["best_iterations_by_outer_fold"].items()
        for layer, value in by_layer.items()
    ]
    ratios = {
        str(layer): float(
            max(row["iterations"] for row in convergence_rows if row["layer"] == layer)
            / min(row["iterations"] for row in convergence_rows if row["layer"] == layer)
        )
        for layer in (2, 3, 4)
    }
    hidden_row = next(row for row in block_rows if row["block"] == "2025_sep_oct_hidden_public")
    same_season = next(row for row in block_rows if row["block"] == "2024_sep_oct")
    pre_gap = next(row for row in block_rows if row["block"] == "2025_jul_aug")
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "aggregate target-blind public-layer regime diagnostic",
        "uploaded": False,
        "target_layer_temp_or_psal_used": False,
        "external_values_used": False,
        "support": {
            "oof_rows": int(len(oof)),
            "hidden_test_rows": int(len(data.test_index)),
            "hidden_timestamps": int(len(supports["2025_sep_oct_hidden_public"])),
        },
        "reference_2024_sep_oct_contrast_terciles": {"q33": float(q33), "q67": float(q67)},
        "block_rows": block_rows,
        "weekly_rows": weekly_rows,
        "convergence_rows": convergence_rows,
        "findings": {
            "hidden_contrast_coverage": hidden_row["contrast_coverage"],
            "hidden_low_share": hidden_row["reference_low_share"],
            "hidden_middle_share": hidden_row["reference_middle_share"],
            "hidden_high_share": hidden_row["reference_high_share"],
            "same_season_contrast_distance": same_season["distance_to_hidden_contrast"],
            "pre_gap_contrast_distance": pre_gap["distance_to_hidden_contrast"],
            "state_split_aggregate_delta_rmse": state["diagnostics"]["aggregate"]["delta_rmse"],
            "state_split_transition_delta_rmse": state["state_bins"]["transition_weight_025_075"][
                "delta_rmse"
            ],
            "iteration_max_min_ratio_by_layer": ratios,
            "frozen_deep_lobo_rmse": layerwise["deep_pair"]["deep_lobo_rmse"],
        },
        "source_hashes": {
            "observations.csv": _sha256(data_dir / "observations.csv"),
            "test_index.csv": _sha256(data_dir / "test_index.csv"),
            "deep_oof": _sha256(deep_oof),
            "state_result": _sha256(state_result),
            "tuning_result": _sha256(tuning_result),
        },
        "interpretation": (
            "Calendar season is useful context, but the hidden interval is a mixture of public-layer "
            "physical states. A low-capacity soft gate with missing-aware fallback is the next testable "
            "hypothesis; this diagnostic does not establish hidden-score improvement."
        ),
    }


def _update_report(artifact_path: Path, result: dict) -> None:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest = artifact["manifest"]
    for collection in ("cards", "charts", "tables", "sources"):
        manifest[collection] = [
            item for item in manifest.get(collection, []) if not item["id"].startswith(PREFIX)
        ]
    artifact["sources"] = [
        item for item in artifact.get("sources", []) if not item["id"].startswith(PREFIX)
    ]
    manifest["blocks"] = [item for item in manifest["blocks"] if not item["id"].startswith(PREFIX)]
    for key in list(artifact["snapshot"]["datasets"]):
        if key.startswith(PREFIX):
            del artifact["snapshot"]["datasets"][key]

    block_rows = result["block_rows"]
    weekly_rows = result["weekly_rows"]
    convergence_rows = result["convergence_rows"]
    artifact["snapshot"]["datasets"].update(
        {
            "regime_diag_blocks": block_rows,
            "regime_diag_weekly": weekly_rows,
            "regime_diag_convergence": convergence_rows,
        }
    )
    public_source = {
        "id": "regime_diag_public_sql",
        "label": "Reviewed P2 public-layer regime aggregates",
        "path": "artifacts/p2_regime_diagnostic_v1/result.json",
        "query": {
            "engine": "sqlite",
            "sql": _union_sql(
                block_rows,
                [
                    "block",
                    "timestamps",
                    "contrast_coverage",
                    "contrast_median",
                    "contrast_q90",
                    "temp_range_median",
                    "contrast_change_24h_median",
                    "distance_to_hidden_contrast",
                    "distance_to_hidden_temp_range",
                    "reference_low_share",
                    "reference_middle_share",
                    "reference_high_share",
                ],
            ),
            "description": "Materializes target-blind public-layer regime diagnostics.",
            "tables_used": ["P2_DATA_DIR/observations.csv", "P2_DATA_DIR/test_index.csv"],
            "filters": [
                "Public layers 1, 5, 6, 7, and 8 only",
                "Exact OOF time support for three proxy blocks",
                "Exact test_index time support for the hidden public interval",
            ],
            "metric_definitions": {
                "contrast_coverage": "Share of timestamps with finite absolute layer-1 minus layer-5 temperature contrast",
                "distance_to_hidden_contrast": "Wasserstein distance divided by hidden-block interquartile range",
            },
        },
    }
    weekly_source = {
        "id": "regime_diag_weekly_sql",
        "label": "Reviewed weekly public-layer stratification summaries",
        "path": "artifacts/p2_regime_diagnostic_v1/result.json",
        "query": {
            "engine": "sqlite",
            "sql": _union_sql(
                weekly_rows,
                [
                    "period",
                    "week",
                    "contrast_median",
                    "public_temp_range_median",
                    "contrast_valid",
                    "timestamps",
                    "contrast_coverage",
                ],
            ),
            "description": "Materializes weekly public-layer contrast for the same-season and hidden intervals.",
            "tables_used": ["P2_DATA_DIR/observations.csv", "P2_DATA_DIR/test_index.csv"],
            "filters": ["KST weeks beginning 1 September", "Target layers excluded"],
            "metric_definitions": {
                "contrast_median": "Weekly median absolute layer-1 minus layer-5 temperature in degrees Celsius"
            },
        },
    }
    convergence_source = {
        "id": "regime_diag_convergence_sql",
        "label": "Reviewed layerwise CatBoost fold convergence",
        "path": "artifacts/p2_top3_parallel_tuning_v1/catboost_layerwise/result.json",
        "query": {
            "engine": "sqlite",
            "sql": _union_sql(convergence_rows, ["outer_block", "layer", "iterations"]),
            "description": "Materializes inner-selected CatBoost checkpoints by outer fold and target layer.",
            "tables_used": [],
            "filters": [
                "12 independent inner-only trials per outer fold",
                "Outer labels unused for selection",
            ],
            "metric_definitions": {
                "iterations": "Best boosting checkpoint selected inside the outer training fold"
            },
        },
    }
    manifest["sources"].extend([public_source, weekly_source, convergence_source])
    artifact["sources"].append(
        {"id": "regime_diag_result", "path": "artifacts/p2_regime_diagnostic_v1/result.json"}
    )
    manifest["charts"].append(
        {
            "id": "regime_diag_weekly_chart",
            "title": "Weekly public-layer stratification contrast",
            "subtitle": "Median |T1−T5| by KST week from 1 September; target-layer values are excluded.",
            "type": "line",
            "dataset": "regime_diag_weekly",
            "sourceId": "regime_diag_weekly_sql",
            "palette": {"kind": "categorical"},
            "legend": {"position": "top", "title": "Period"},
            "encodings": {
                "x": {"field": "week", "type": "quantitative", "label": "Week from 1 September"},
                "y": {
                    "field": "contrast_median",
                    "type": "quantitative",
                    "label": "Median |T1−T5| (°C)",
                },
                "color": {"field": "period", "type": "nominal", "label": "Period"},
            },
        }
    )
    manifest["tables"].extend(
        [
            {
                "id": "regime_diag_blocks_table",
                "title": "Public-state similarity to the hidden interval",
                "subtitle": "Normalized distances use hidden-block IQR; lower is more similar.",
                "dataset": "regime_diag_blocks",
                "sourceId": "regime_diag_public_sql",
                "defaultSort": {"field": "block", "direction": "asc"},
                "columns": [
                    {"field": "block", "label": "Block", "type": "text"},
                    {"field": "timestamps", "label": "Times", "type": "number"},
                    {"field": "contrast_coverage", "label": "T1−T5 coverage", "type": "number"},
                    {"field": "contrast_median", "label": "Median |T1−T5|", "type": "number"},
                    {
                        "field": "temp_range_median",
                        "label": "Median public range",
                        "type": "number",
                    },
                    {
                        "field": "distance_to_hidden_contrast",
                        "label": "Contrast distance",
                        "type": "number",
                    },
                    {
                        "field": "distance_to_hidden_temp_range",
                        "label": "Range distance",
                        "type": "number",
                    },
                ],
            },
            {
                "id": "regime_diag_convergence_table",
                "title": "Layerwise CatBoost convergence by outer block",
                "subtitle": "The large checkpoint spread is diagnostic evidence of non-stationarity, not proof of a better gate.",
                "dataset": "regime_diag_convergence",
                "sourceId": "regime_diag_convergence_sql",
                "defaultSort": {"field": "outer_block", "direction": "asc"},
                "columns": [
                    {"field": "outer_block", "label": "Outer block", "type": "text"},
                    {"field": "layer", "label": "Layer", "type": "number"},
                    {"field": "iterations", "label": "Selected rounds", "type": "number"},
                ],
            },
        ]
    )
    findings = result["findings"]
    ratios = findings["iteration_max_min_ratio_by_layer"]
    blocks = [
        {
            "id": "regime_diag_public_finding",
            "type": "markdown",
            "sourceId": "regime_diag_public_sql",
            "body": (
                "## 계절 라벨보다 공개층의 물리 상태가 병목을 더 잘 설명한다\n\n"
                f"숨은 2025년 9–10월은 2024년 같은 계절의 공개층 분포와 가깝다. "
                f"`|T1−T5|` 정규화 거리는 `{findings['same_season_contrast_distance']:.3f}`로, "
                f"직전 2025년 7–8월의 `{findings['pre_gap_contrast_distance']:.3f}`보다 훨씬 작다. "
                "하지만 숨은 구간 자체도 한 상태가 아니다. 2024년 같은 계절의 삼분위 기준으로 "
                f"저성층 `{findings['hidden_low_share']:.1%}`, 전이 `{findings['hidden_middle_share']:.1%}`, "
                f"강성층 `{findings['hidden_high_share']:.1%}`가 섞여 있다. 따라서 월 단위 hard split은 "
                "전이 시점을 평균내며, 시점별 soft gate가 더 적합하다."
            ),
        },
        {"id": "regime_diag_weekly_block", "type": "chart", "chartId": "regime_diag_weekly_chart"},
        {
            "id": "regime_diag_weekly_note",
            "type": "markdown",
            "sourceId": "regime_diag_weekly_sql",
            "body": (
                "### 같은 9–10월 안에서도 성층 붕괴 경로가 달라진다\n\n"
                "주별 중앙값은 달력상 같은 계절이더라도 상태 전환의 시점과 속도가 다름을 보여준다. "
                f"또한 숨은 시각 중 `T1−T5`가 유효한 비율은 `{findings['hidden_contrast_coverage']:.1%}`뿐이다. "
                "따라서 이 한 변수만으로 gate를 만들면 나머지 시각이 모두 동일한 fallback으로 몰린다. "
                "공개 전 층의 온도 범위, 염분 범위, 24시간 변화, M2 성분과 결측 mask를 함께 써야 한다."
            ),
        },
        {"id": "regime_diag_blocks_block", "type": "table", "tableId": "regime_diag_blocks_table"},
        {
            "id": "regime_diag_experiment_finding",
            "type": "markdown",
            "body": (
                "## 기존 실험도 상태 조건화의 방향성은 지지하지만 충분하지 않았다\n\n"
                f"단순한 `|T1−T5|` 2전문가 실험은 전체 연구 구간 RMSE를 "
                f"`{abs(findings['state_split_aggregate_delta_rmse']):.5f}℃` 낮췄고, 전이 bin에서는 "
                f"`{abs(findings['state_split_transition_delta_rmse']):.5f}℃` 낮췄다. 반면 CatBoost의 "
                f"outer별 수렴 checkpoint 최대/최소 비율은 layer 2 `{ratios['2']:.1f}×`, "
                f"layer 3 `{ratios['3']:.1f}×`, layer 4 `{ratios['4']:.1f}×`였다. 이는 단일 전역 함수의 "
                "비정상성이 크다는 증거지만, 상태 분기만으로 hidden RMSE 개선이 보장된다는 뜻은 아니다."
            ),
        },
        {
            "id": "regime_diag_convergence_block",
            "type": "table",
            "tableId": "regime_diag_convergence_table",
        },
        {
            "id": "regime_diag_next_hypothesis",
            "type": "markdown",
            "body": (
                "## 다음 실험은 hard split이 아니라 저복잡도 soft mixture로 한정한다\n\n"
                "현재 deep stack의 구성 모델과 학습 결과는 그대로 두고, 공개층 상태에 따라 layer별 simplex "
                "가중치만 연속적으로 바꾸는 작은 gate를 우선 시험한다. 입력은 `|T1−T5|`, 공개층 온도·염분 "
                "범위, 두 지표의 24시간 변화, M2 amplitude/coherence, 결측 mask로 제한한다. 분할 경계는 "
                "target 정답이 아니라 fold-train 공개층에서만 학습하고, 계절 블록을 통째로 남기는 "
                "leave-one-block-out RMSE로 기존 deep stack `0.77566℃`와 비교한다. 별도 전문가를 처음부터 "
                "재학습하는 안은 이 저복잡도 gate가 실제로 개선된 뒤에만 진행한다."
            ),
        },
    ]
    insertion = next(
        (index for index, block in enumerate(manifest["blocks"]) if block["id"] == "limitations"),
        len(manifest["blocks"]),
    )
    manifest["blocks"][insertion:insertion] = blocks
    generated = datetime.now().astimezone().isoformat()
    manifest["generatedAt"] = generated
    artifact["snapshot"]["generatedAt"] = generated
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact", type=Path, default=Path("reports/p2_method_scout_20260816/artifact.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/p2_regime_diagnostic_v1/result.json")
    )
    parser.add_argument(
        "--deep-oof",
        type=Path,
        default=Path("artifacts/p2_deep_finalists_v1/stacked_oof.parquet"),
    )
    parser.add_argument(
        "--state-result",
        type=Path,
        default=Path("artifacts/p2_state_conditional_lean_v1/result.json"),
    )
    parser.add_argument(
        "--tuning-result",
        type=Path,
        default=Path("artifacts/p2_top3_parallel_tuning_v1/result.json"),
    )
    args = parser.parse_args()
    result = _diagnose(args.data_dir, args.deep_oof, args.state_result, args.tuning_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _update_report(args.artifact, result)
    print(json.dumps({"status": "passed", "result_sha256": _sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
