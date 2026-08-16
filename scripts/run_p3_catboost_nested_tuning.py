"""Nested inner-only CatBoost tuning for the official pooled P3 RMSE."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from p3_wave.data import select_independent_validation
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.validation import build_forecast_folds, expand_leads, metric_slices, rmse

PARAMETER_GRID = (
    {"depth": 5, "learning_rate": 0.020, "l2_leaf_reg": 10.0, "random_strength": 0.10},
    {"depth": 6, "learning_rate": 0.020, "l2_leaf_reg": 10.0, "random_strength": 0.10},
    {"depth": 7, "learning_rate": 0.020, "l2_leaf_reg": 15.0, "random_strength": 0.10},
    {"depth": 8, "learning_rate": 0.015, "l2_leaf_reg": 20.0, "random_strength": 0.05},
    {"depth": 6, "learning_rate": 0.040, "l2_leaf_reg": 15.0, "random_strength": 0.20},
    {"depth": 7, "learning_rate": 0.040, "l2_leaf_reg": 20.0, "random_strength": 0.20},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--output-dir", default="artifacts/p3/catboost_nested_tuning")
    parser.add_argument("--iterations", type=int, default=2500)
    return parser.parse_args()


def _cat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    value["station"] = value["station"].astype(str)
    value["lead_h"] = value["lead_h"].astype(str)
    return value


def _model(parameters: dict[str, float | int], iterations: int, seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        task_type="GPU",
        devices="0",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        **parameters,
    )


def main() -> int:
    args = parse_args()
    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    oof_rows: list[pd.DataFrame] = []
    selection: dict[str, object] = {}
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
        fit_weight = threshold_case_weights(fit_meta["current_hs"].to_numpy())
        train_pool = Pool(x_fit, y_fit, weight=fit_weight, cat_features=[0, 1])
        cal_delta = cal_meta["target_hs"].to_numpy() - cal_meta["current_hs"].to_numpy()
        cal_pool = Pool(x_cal, cal_delta, cat_features=[0, 1])
        candidates: list[dict[str, object]] = []
        best: tuple[float, int, int] | None = None
        for candidate_number, parameters in enumerate(PARAMETER_GRID):
            model = _model(
                parameters, args.iterations, 20260816 + fold_number * 100 + candidate_number
            )
            model.fit(
                train_pool,
                eval_set=cal_pool,
                early_stopping_rounds=120,
                use_best_model=True,
                verbose=False,
            )
            iteration = max(int(model.get_best_iteration()) + 1, 1)
            prediction = cal_meta["current_hs"].to_numpy() + model.predict(x_cal)
            score = rmse(cal_meta["target_hs"].to_numpy(), prediction)
            candidates.append(
                {
                    "candidate": candidate_number,
                    "parameters": parameters,
                    "best_iteration": iteration,
                    "inner_rmse": score,
                }
            )
            rank = (score, candidate_number, iteration)
            if best is None or rank < best:
                best = rank
        if best is None:
            raise RuntimeError("no CatBoost candidate completed")
        best_score, best_number, best_iteration = best
        selected = dict(PARAMETER_GRID[best_number])
        x_outer, y_outer, outer_meta = expand_leads(features, anchors, fold.train_ids, columns)
        x_valid, _, valid_meta = expand_leads(features, anchors, fold.validation_ids, columns)
        x_outer = _cat_frame(x_outer)
        x_valid = _cat_frame(x_valid)
        outer_weight = threshold_case_weights(outer_meta["current_hs"].to_numpy())
        final_model = _model(selected, best_iteration, 20260916 + fold_number)
        final_model.fit(
            Pool(x_outer, y_outer, weight=outer_weight, cat_features=[0, 1]), verbose=False
        )
        prediction = np.clip(
            valid_meta["current_hs"].to_numpy() + final_model.predict(x_valid), 0.0, 30.0
        )
        frame = valid_meta.copy()
        frame["fold"] = fold.name
        frame["prediction"] = prediction
        frame["persistence"] = frame["current_hs"]
        oof_rows.append(frame)
        selection[fold.name] = {
            "fit_rows": int(len(fit_ids)),
            "calibration_cases": int(len(calibration_ids)),
            "selected_candidate": best_number,
            "selected_parameters": selected,
            "selected_iteration": best_iteration,
            "selected_inner_rmse": best_score,
            "candidates": candidates,
        }
    oof = pd.concat(oof_rows, ignore_index=True)
    metrics = {
        "candidate": metric_slices(oof, oof["prediction"].to_numpy()),
        "persistence": metric_slices(oof, oof["persistence"].to_numpy()),
    }
    oof_path = output / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_nested_inner_selection",
        "gpu_nondeterministic": True,
        "maximum_iterations": args.iterations,
        "parameter_grid": PARAMETER_GRID,
        "selection": selection,
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
