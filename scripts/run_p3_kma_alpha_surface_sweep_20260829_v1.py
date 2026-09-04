"""Run an exhaustive diagnostic sweep over the sealed P3 KMA correction axis.

The runner reads only previously sealed OOF predictions and frozen submission
vectors. It never reads the official P3 test context or hidden targets and it
does not upload anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.kma_alpha_surface import (
    ACTIVE_LEADS,
    apply_official_correction,
    crossfit_predictions,
    exhaustive_lead_surface,
    fold_robust_lead_surface,
    fit_alpha,
    fit_group_alphas,
    make_alpha_grid,
    metric_breakdown,
    paired_case_bootstrap,
    predict_with_mapping,
    prepare_oof_frame,
)


EXPERIMENT_ID = "p3_kma_alpha_surface_sweep_20260829_v1"
DEFAULT_CONFIG = ROOT / "configs/experiments/p3_kma_alpha_surface_sweep_20260829_v1.json"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/p3_kma_alpha_surface_sweep_20260829_v1"
DEFAULT_REPORT_DIR = ROOT / "reports/p3_kma_alpha_surface_sweep_20260829_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--current-submission",
        type=Path,
        required=True,
        help="Frozen current official champion CSV; supplied explicitly and never committed.",
    )
    parser.add_argument(
        "--candidate-output-dir",
        type=Path,
        required=True,
        help="New local-only directory for READY_NOT_UPLOADED candidate CSV files.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def official_curve(points: list[dict[str, float]]) -> dict[str, Any]:
    x = np.asarray([row["uniform_alpha"] for row in points], dtype=np.float64)
    rmse = np.asarray([row["rmse"] for row in points], dtype=np.float64)
    a, b, c = np.polyfit(x, np.square(rmse), 2)
    optimum = float(-b / (2.0 * a)) if a > 0.0 else float("nan")
    predicted = float(np.sqrt(a * optimum**2 + b * optimum + c)) if a > 0.0 else float("nan")
    first_gain = float(rmse[0] - rmse[1])
    second_gain = float(rmse[1] - rmse[2])
    return {
        "points": points,
        "quadratic_fit_on_mse": {"a": float(a), "b": float(b), "c": float(c)},
        "quadratic_optimum_alpha": optimum,
        "quadratic_predicted_rmse": predicted,
        "marginal_rmse_gain_0_to_0p2": first_gain,
        "marginal_rmse_gain_0p2_to_0p4": second_gain,
        "second_to_first_marginal_gain_ratio": second_gain / first_gain,
        "warning": "Three Public points estimate only the uniform combined axis; they do not identify 18h and 24h separately.",
    }


def strategy_result(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    fitted: dict[str, object],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        "metrics": metric_breakdown(frame, prediction),
        "paired_case_bootstrap": paired_case_bootstrap(
            frame,
            prediction,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "train_fold_fits_for_each_heldout_fold": fitted,
    }


def build_report(result: dict[str, Any]) -> str:
    local = result["crossfit"]
    ranked = result["crossfit_ranked"]
    best = ranked[0]
    surface = result["same_row_exhaustive"]
    official = result["official_public_curve"]
    menu = result["candidate_manifest"]["candidates"]
    stability_sentence = (
        "모든 cross-fit 구조가 incumbent보다 악화했으므로, 로컬 세분화 최적점을 그대로 "
        "제출 파라미터로 승격할 근거는 없다."
        if best["delta_rmse"] >= 0.0
        else "최상위 cross-fit 구조는 incumbent보다 개선했지만 폴드별 안정성도 함께 확인해야 한다."
    )
    lines = [
        "# P3 KMA 장기리드 보정축 전수 탐색 — 2026-08-29",
        "",
        "## 결론",
        "",
        (
            f"봉인된 과거 OOF {result['data_contract']['cases']}사례에서 α=-1.00~2.00을 0.01 간격으로 훑고, "
            f"18h×24h {surface['lead_surface']['evaluated_pairs']:,}조합과 정점×리드 분리축을 계산했다. "
            f"폴드 밖 예측 기준 최상위 구조는 **{best['strategy']}**이며 incumbent 대비 RMSE 변화는 "
            f"**{best['delta_rmse']:+.6f}m**이다."
        ),
        "",
        stability_sentence,
        "",
        "같은 행에서 고른 최적점은 탐색적 상한일 뿐 독립 검증이 아니다. 실제 판단은 각 폴드를 제외하고 α를 맞춘 cross-fit 결과와 사례 단위 bootstrap을 우선한다. 공식 Public 0/20/40% 곡선은 40%까지 개선됐지만 한계효율이 줄고 있어 균일 보정을 무작정 키우는 전략은 근거가 약하다. 18h와 24h를 분리하는 두 후보가 다음 제출 기회를 더 정보성 있게 쓴다.",
        "",
        "## 실제 탐색 범위",
        "",
        f"- 균일 α: {surface['uniform']['grid_points']}점",
        f"- 18h×24h: {surface['lead_surface']['evaluated_pairs']:,}쌍",
        f"- 정점×리드: {surface['station_lead']['groups']}개 독립 축; 제곱오차 분리성으로 각 301점 최적화가 전체 조합의 전역 격자 최적점과 동일",
        f"- hierarchical shrink: {len(result['crossfit']['hierarchical_grid'])}개 shrink 값, 각 외부 폴드에서 나머지 두 폴드만으로 α 적합",
        "",
        "## Cross-fit 결과",
        "",
        "| 구조 | pooled ΔRMSE(m) | 개선 폴드 | bootstrap P(개선) | 90% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ranked:
        entry = local[row["key"]]
        metrics = entry["metrics"]
        improved_folds = sum(
            slice_["delta_rmse"] < 0.0 for slice_ in metrics["by_fold"].values()
        )
        boot = entry["paired_case_bootstrap"]
        lines.append(
            f"| {row['strategy']} | {metrics['delta_rmse']:+.6f} | {improved_folds}/3 | "
            f"{boot['probability_improvement']:.3f} | [{boot['ci90_lower']:+.6f}, {boot['ci90_upper']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## 같은 OOF 행에서 본 탐색적 최적점",
            "",
            f"- 균일 α 최적: {surface['uniform']['best_alpha']:.2f}, RMSE {surface['uniform']['best_rmse']:.6f}",
            f"- 리드 분리 최적: α18={surface['lead_surface']['best_alpha_18']:.2f}, α24={surface['lead_surface']['best_alpha_24']:.2f}, RMSE {surface['lead_surface']['best_rmse']:.6f}",
            "- 정점×리드 최적 α는 result.json에 집계값만 기록했다. 같은 사례에 맞춘 값이므로 그대로 제출 파라미터로 쓰지 않는다.",
            "",
            "## 공식 Public 곡선",
            "",
            f"- 현재 champion α=0: RMSE {official['points'][0]['rmse']:.6f}",
            f"- 균일 20%: RMSE {official['points'][1]['rmse']:.6f}",
            f"- 균일 40%: RMSE {official['points'][2]['rmse']:.6f}",
            f"- MSE 2차 근사 정점: α={official['quadratic_optimum_alpha']:.3f}, 예측 RMSE {official['quadratic_predicted_rmse']:.6f}",
            f"- 0→20% 대비 20→40% 한계개선 비율: {official['second_to_first_marginal_gain_ratio']:.3f}",
            "",
            "## 생성된 후보와 권고",
            "",
            "모든 CSV는 스키마·키·유한값·0~30m·단기리드 exact no-op을 통과했으며 아직 업로드하지 않았다.",
            "",
            "| 목록 | 후보 | α18 | α24 | 역할 | SHA-256 |",
            "|---:|---|---:|---:|---|---|",
        ]
    )
    for index, candidate in enumerate(menu, start=1):
        lines.append(
            f"| {index} | {candidate['id']} | {candidate['alpha_18']:.3f} | "
            f"{candidate['alpha_24']:.3f} | {candidate['role']} | `{candidate['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "첫 검증 후보는 α18=0.4, α24=0.6이다. 기존 균일 0.4에서 24h만 움직이므로 공식 점수 변화가 24h 방향의 순효과를 알려준다. 다음 후보 α18=0.2, α24=0.6은 첫 후보와 비교해 18h만 달라져 18h 방향을 분리한다. 세 번째 기회는 두 공식 결과를 받은 뒤 선택하는 편이 정보가 가장 크다.",
            "",
            "## 한계",
            "",
            "- 로컬 OOF incumbent와 공식 현재 champion은 서로 다른 long-axis 계보다. 로컬 절대 개선량을 공식 개선량으로 환산하지 않는다.",
            "- Public은 66사례뿐이며 Private 일반화가 보장되지 않는다.",
            "- 정점×리드 동일행 최적점은 6개 자유도로 과적합될 수 있어 cross-fit과 shrink 결과를 함께 봐야 한다.",
            "- 이 실험은 기존 예측벡터를 분석했으며 KMA 모델 자체를 재학습하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("unexpected experiment id")
    if not all(config["prohibitions"].values()):
        raise RuntimeError("a diagnostic prohibition was disabled")
    artifact_dir = args.artifact_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    candidate_dir = args.candidate_output_dir.expanduser().resolve()
    for path in (artifact_dir, report_dir, candidate_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    inputs = config["inputs"]
    blind_path = ROOT / inputs["blind_predictions"]
    evaluated_path = ROOT / inputs["evaluated_incumbent_oof"]
    old_path = ROOT / inputs["old_official_submission"]
    kma_path = ROOT / inputs["kma_alpha40_official_submission"]
    current_path = args.current_submission.expanduser().resolve()
    for path in (blind_path, evaluated_path, old_path, kma_path, current_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    blind = pd.read_parquet(blind_path)
    evaluated = pd.read_parquet(evaluated_path)
    frame = prepare_oof_frame(blind, evaluated)
    grid_config = config["alpha_grid"]
    grid = make_alpha_grid(grid_config["start"], grid_config["stop"], grid_config["step"])

    active_mask = frame["lead_h"].isin(ACTIVE_LEADS)
    uniform_fit = fit_alpha(frame, grid, mask=active_mask)
    uniform_mapping = {(18,): uniform_fit.grid, (24,): uniform_fit.grid}
    uniform_same_prediction = predict_with_mapping(frame, uniform_mapping, ("lead_h",))
    lead_mapping, lead_diagnostics = fit_group_alphas(frame, grid, ("lead_h",))
    lead_same_prediction = predict_with_mapping(frame, lead_mapping, ("lead_h",))
    station_mapping, station_diagnostics = fit_group_alphas(
        frame, grid, ("station", "lead_h")
    )
    station_same_prediction = predict_with_mapping(
        frame, station_mapping, ("station", "lead_h")
    )
    lead_surface = exhaustive_lead_surface(frame, grid)
    robust_lead_surface = fold_robust_lead_surface(frame, grid)
    minimax = robust_lead_surface["minimax_pair"]
    minimax_prediction = predict_with_mapping(
        frame,
        {(18,): float(minimax["alpha_18"]), (24,): float(minimax["alpha_24"])},
        ("lead_h",),
    )
    robust_lead_surface["posthoc_minimax_metrics"] = metric_breakdown(
        frame, minimax_prediction
    )
    robust_lead_surface["posthoc_minimax_bootstrap"] = paired_case_bootstrap(
        frame,
        minimax_prediction,
        replicates=int(config["bootstrap"]["replicates"]),
        seed=int(config["bootstrap"]["seed"]),
    )

    bootstrap_config = config["bootstrap"]
    crossfit: dict[str, Any] = {}
    for strategy in ("uniform", "lead", "station_lead"):
        prediction, fitted = crossfit_predictions(frame, grid, strategy=strategy)
        crossfit[strategy] = strategy_result(
            frame,
            prediction,
            fitted,
            bootstrap_replicates=bootstrap_config["replicates"],
            bootstrap_seed=bootstrap_config["seed"],
        )
    hierarchical_entries: list[dict[str, Any]] = []
    for index, shrink in enumerate(config["hierarchical_shrink_grid"]):
        prediction, fitted = crossfit_predictions(
            frame, grid, strategy="hierarchical", shrink=float(shrink)
        )
        key = f"hierarchical_{float(shrink):.1f}"
        entry = strategy_result(
            frame,
            prediction,
            fitted,
            bootstrap_replicates=bootstrap_config["replicates"],
            bootstrap_seed=bootstrap_config["seed"] + index + 1,
        )
        crossfit[key] = entry
        hierarchical_entries.append(
            {"key": key, "shrink": float(shrink), "delta_rmse": entry["metrics"]["delta_rmse"]}
        )
    crossfit["hierarchical_grid"] = hierarchical_entries
    ranking_source = [
        {"key": "uniform", "strategy": "uniform", "delta_rmse": crossfit["uniform"]["metrics"]["delta_rmse"]},
        {"key": "lead", "strategy": "lead-specific", "delta_rmse": crossfit["lead"]["metrics"]["delta_rmse"]},
        {"key": "station_lead", "strategy": "station×lead", "delta_rmse": crossfit["station_lead"]["metrics"]["delta_rmse"]},
    ]
    ranking_source.extend(
        {
            "key": row["key"],
            "strategy": f"hierarchical λ={row['shrink']:.1f}",
            "delta_rmse": row["delta_rmse"],
        }
        for row in hierarchical_entries
    )
    ranked = sorted(ranking_source, key=lambda row: row["delta_rmse"])

    current = pd.read_csv(current_path)
    old = pd.read_csv(old_path)
    kma = pd.read_csv(kma_path)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    candidate_records: list[dict[str, Any]] = []
    for order, specification in enumerate(config["candidate_menu"], start=1):
        candidate = apply_official_correction(
            current,
            old,
            kma,
            alpha_by_lead={
                18: float(specification["alpha_18"]),
                24: float(specification["alpha_24"]),
            },
        )
        directory = candidate_dir / f"{order:02d}_{specification['id']}"
        directory.mkdir()
        csv_path = directory / "P3_submission.csv"
        candidate.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
        values = candidate["hs_pred"].to_numpy(dtype=np.float64)
        current_values = current["hs_pred"].to_numpy(dtype=np.float64)
        changed = np.abs(values - current_values) > 1e-12
        record = {
            **specification,
            "status": "READY_NOT_UPLOADED",
            "file": str(csv_path),
            "rows": int(len(candidate)),
            "changed_rows_vs_current": int(changed.sum()),
            "rms_change_vs_current": float(np.sqrt(np.mean(np.square(values - current_values)))),
            "minimum_m": float(values.min()),
            "maximum_m": float(values.max()),
            "sha256": sha256_file(csv_path),
            "uploaded": False,
        }
        (directory / "제출정보.txt").write_text(
            "제출물 제목: P3 KMA 리드별 보정 "
            f"18h {float(specification['alpha_18']):.3f} / 24h {float(specification['alpha_24']):.3f}\n"
            "한줄요약(접근방식): 공식 최고 예측은 유지하고 KMA 장기파고 보정축을 18h와 24h에 서로 다른 비율로 적용합니다.\n"
            f"파일 SHA-256: {record['sha256']}\n상태: READY_NOT_UPLOADED\n",
            encoding="utf-8-sig",
        )
        candidate_records.append(record)

    uniform_same_metrics = metric_breakdown(frame, uniform_same_prediction)
    same_row = {
        "uniform": {
            "grid_points": int(len(grid)),
            "analytic_alpha": uniform_fit.analytic,
            "best_alpha": uniform_fit.grid,
            "active_long_lead_only_rmse": uniform_fit.grid_rmse,
            "best_rmse": uniform_same_metrics["candidate_rmse"],
            "metrics": uniform_same_metrics,
        },
        "lead_surface": lead_surface,
        "fold_robust_lead_surface": robust_lead_surface,
        "lead_diagnostics": lead_diagnostics,
        "station_lead": {
            "groups": int(len(station_mapping)),
            "diagnostics": station_diagnostics,
            "metrics": metric_breakdown(frame, station_same_prediction),
        },
        "lead_metrics": metric_breakdown(frame, lead_same_prediction),
    }
    result = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE_READY_NOT_UPLOADED",
        "git": {
            "head_at_start": "493504f",
            "worktree_expected_dirty_from_this_experiment": True,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data_contract": {
            "rows": int(len(frame)),
            "cases": int(frame.groupby(["fold", "anchor_id"], observed=True).ngroups),
            "folds": sorted(str(value) for value in frame["fold"].unique()),
            "stations": sorted(str(value) for value in frame["station"].unique()),
            "leads": sorted(int(value) for value in frame["lead_h"].unique()),
            "official_test_context_read": False,
            "hidden_target_read": False,
            "new_model_fit_count": 0,
            "row_level_values_written_to_report": False,
        },
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "blind_predictions_sha256": sha256_file(blind_path),
            "evaluated_incumbent_oof_sha256": sha256_file(evaluated_path),
            "current_official_champion_sha256": sha256_file(current_path),
            "old_official_submission_sha256": sha256_file(old_path),
            "kma_alpha40_axis_submission_sha256": sha256_file(kma_path),
        },
        "alpha_grid": {
            **grid_config,
            "points": int(len(grid)),
        },
        "same_row_exhaustive": same_row,
        "crossfit": crossfit,
        "crossfit_ranked": ranked,
        "official_public_curve": official_curve(config["official_public_points"]),
        "candidate_manifest": {
            "status": "READY_NOT_UPLOADED",
            "output_directory": str(candidate_dir),
            "candidates": candidate_records,
        },
        "interpretation": {
            "same_row_optima_are_independent_evidence": False,
            "crossfit_is_primary_local_stability_evidence": True,
            "local_incumbent_matches_current_official_champion": False,
            "official_public_curve_is_primary_deployment_axis_evidence": True,
            "recommended_first_probe": "P3_KMA_L18_040_L24_060",
            "recommended_second_probe_after_first_result": "P3_KMA_L18_020_L24_060",
            "third_probe_should_be_adaptive_after_two_lead_isolation_results": True,
        },
    }
    artifact_dir.mkdir(parents=True, exist_ok=False)
    write_json(artifact_dir / "result.json", result)
    write_json(candidate_dir / "SET_MANIFEST.json", result["candidate_manifest"])
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "summary.md"
    report_path.write_text(build_report(result), encoding="utf-8")
    aggregate_result = {
        key: value
        for key, value in result.items()
        if key not in {"crossfit"}
    }
    aggregate_result["candidate_manifest"] = {
        "status": result["candidate_manifest"]["status"],
        "candidates": [
            {key: value for key, value in row.items() if key != "file"}
            for row in result["candidate_manifest"]["candidates"]
        ],
    }
    aggregate_result["crossfit"] = {
        key: value
        for key, value in crossfit.items()
        if key == "hierarchical_grid" or key in {row["key"] for row in ranked[:4]}
    }
    write_json(report_dir / "result.json", aggregate_result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["data_contract"]["rows"],
                "cases": result["data_contract"]["cases"],
                "evaluated_lead_pairs": lead_surface["evaluated_pairs"],
                "best_crossfit": ranked[0],
                "official_quadratic_alpha": result["official_public_curve"]["quadratic_optimum_alpha"],
                "candidate_count": len(candidate_records),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
