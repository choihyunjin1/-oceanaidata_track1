"""Select one persistence-shrinkage coefficient on inner cases and apply to frozen outer OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from p3_wave.data import select_independent_validation
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.validation import build_forecast_folds, expand_leads, metric_slices, rmse


def _model(seed: int, thread_count: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=700,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=8.0,
        random_strength=0.2,
        random_seed=seed,
        thread_count=thread_count,
        verbose=False,
        allow_writing_files=False,
    )


def _cat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["station"] = result["station"].astype(str)
    result["lead_h"] = result["lead_h"].astype(str)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--outer-oof", default="artifacts/p3/initial_tournament_all20/oof.parquet")
    parser.add_argument("--output-dir", default="artifacts/p3/shrinkage_calibration")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    outer = pd.read_parquet(args.outer_oof)
    outer = outer.loc[outer["backend"].eq("catboost")].copy()
    output_rows: list[pd.DataFrame] = []
    selections: dict[str, object] = {}
    alpha_grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    for fold_number, fold in enumerate(build_forecast_folds(anchors)):
        outer_train = anchors.set_index("anchor_id").loc[fold.train_ids]
        inner_end = outer_train["anchor_time"].max() + pd.Timedelta(minutes=20)
        inner_start = inner_end - pd.Timedelta(days=60)
        calibration_ids = select_independent_validation(
            anchors, start=inner_start, end=inner_end, gap_hours=78
        )
        calibration_ids = np.intersect1d(calibration_ids, fold.train_ids)
        fit_end = inner_start - pd.Timedelta(hours=78)
        fit_ids = outer_train.loc[outer_train["anchor_time"].lt(fit_end)].index.to_numpy(
            dtype=np.int64
        )
        x_fit, y_fit, fit_meta = expand_leads(features, anchors, fit_ids, columns)
        x_cal, _, cal_meta = expand_leads(features, anchors, calibration_ids, columns)
        x_fit = _cat_frame(x_fit)
        x_cal = _cat_frame(x_cal)
        model = _model(20260817 + fold_number, args.threads)
        model.fit(
            x_fit,
            y_fit,
            sample_weight=threshold_case_weights(fit_meta["current_hs"].to_numpy()),
            cat_features=[0, 1],
            verbose=False,
        )
        model_prediction = np.clip(
            cal_meta["current_hs"].to_numpy() + model.predict(x_cal), 0.0, 30.0
        )
        truth = cal_meta["target_hs"].to_numpy()
        persistence = cal_meta["current_hs"].to_numpy()
        scores = [
            rmse(truth, alpha * model_prediction + (1.0 - alpha) * persistence)
            for alpha in alpha_grid
        ]
        best_index = min(
            range(len(alpha_grid)), key=lambda index: (scores[index], alpha_grid[index])
        )
        alpha = float(alpha_grid[best_index])
        fold_outer = outer.loc[outer["fold"].eq(fold.name)].copy()
        fold_outer["raw_prediction"] = fold_outer["prediction"]
        fold_outer["prediction"] = (
            alpha * fold_outer["raw_prediction"] + (1.0 - alpha) * fold_outer["persistence"]
        )
        output_rows.append(fold_outer)
        selections[fold.name] = {
            "fit_rows": int(len(fit_ids)),
            "calibration_cases": int(len(calibration_ids)),
            "alpha": alpha,
            "inner_raw_rmse": rmse(truth, model_prediction),
            "inner_persistence_rmse": rmse(truth, persistence),
            "inner_blended_rmse": scores[best_index],
        }
    calibrated = pd.concat(output_rows, ignore_index=True)
    metrics = {
        "calibrated": metric_slices(calibrated, calibrated["prediction"].to_numpy()),
        "raw_catboost": metric_slices(calibrated, calibrated["raw_prediction"].to_numpy()),
        "persistence": metric_slices(calibrated, calibrated["persistence"].to_numpy()),
    }
    oof_path = output / "oof.parquet"
    calibrated.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_inner_only_global_shrinkage",
        "alpha_grid": alpha_grid.tolist(),
        "selections": selections,
        "metrics": metrics,
        "oof_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
