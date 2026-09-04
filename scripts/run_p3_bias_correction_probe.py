"""Inner-only one-parameter bias correction applied to the frozen P3 ensemble OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from p3_wave.calibration import apply_global_bias_correction, estimate_global_bias_correction
from p3_wave.data import select_independent_validation
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.validation import build_forecast_folds, expand_leads, metric_slices, rmse


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _fold_bias(frame: pd.DataFrame, prediction: str) -> float:
    return float(np.mean(frame[prediction].to_numpy(float) - frame["target_hs"].to_numpy(float)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--outer-oof", default="artifacts/p3/final_ensemble_validation/oof.parquet")
    parser.add_argument("--output-dir", default="artifacts/p3/bias_correction_probe")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-absolute-correction", type=float, default=0.35)
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    frozen_oof = pd.read_parquet(args.outer_oof)
    columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )

    corrected_parts: list[pd.DataFrame] = []
    selections: dict[str, object] = {}
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
        model = _model(20260817 + fold_number, args.threads)
        model.fit(
            _cat_frame(x_fit),
            y_fit,
            sample_weight=threshold_case_weights(fit_meta["current_hs"].to_numpy()),
            cat_features=[0, 1],
            verbose=False,
        )
        inner_raw = np.clip(
            cal_meta["current_hs"].to_numpy(float) + model.predict(_cat_frame(x_cal)),
            0.0,
            30.0,
        )
        inner_truth = cal_meta["target_hs"].to_numpy(float)
        correction = estimate_global_bias_correction(
            inner_truth,
            inner_raw,
            max_absolute_correction=args.max_absolute_correction,
        )

        part = frozen_oof.loc[frozen_oof["fold"].eq(fold.name)].copy()
        part["frozen_prediction"] = part["prediction"]
        part["prediction"] = apply_global_bias_correction(
            part["frozen_prediction"].to_numpy(float), correction
        )
        corrected_parts.append(part)
        selections[fold.name] = {
            "fit_anchor_count": int(len(fit_ids)),
            "calibration_case_count": int(len(calibration_ids)),
            "calibration_row_count": int(len(inner_truth)),
            "correction_m": correction,
            "inner_raw_bias_m": float(np.mean(inner_raw - inner_truth)),
            "inner_raw_rmse_m": rmse(inner_truth, inner_raw),
            "inner_corrected_rmse_m": rmse(inner_truth, inner_raw + correction),
        }

    corrected = pd.concat(corrected_parts, ignore_index=True)
    if len(corrected) != len(frozen_oof):
        raise ValueError("corrected OOF row count does not match frozen OOF")
    metrics = {
        "corrected": metric_slices(corrected, corrected["prediction"].to_numpy(float)),
        "frozen_ensemble": metric_slices(corrected, corrected["frozen_prediction"].to_numpy(float)),
        "persistence": metric_slices(corrected, corrected["persistence"].to_numpy(float)),
        "folds": {
            name: {
                "corrected_rmse": rmse(part["target_hs"], part["prediction"]),
                "frozen_rmse": rmse(part["target_hs"], part["frozen_prediction"]),
                "corrected_bias_m": _fold_bias(part, "prediction"),
                "frozen_bias_m": _fold_bias(part, "frozen_prediction"),
            }
            for name, part in corrected.groupby("fold", sort=True)
        },
    }
    oof_path = output / "oof.parquet"
    corrected.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_inner_only_single_global_intercept",
        "selection_scope": "each outer train trailing 60 days only",
        "outer_labels_used_for_correction_selection": False,
        "model_parameters_changed": False,
        "max_absolute_correction_m": args.max_absolute_correction,
        "selections": selections,
        "metrics": metrics,
        "provenance": {
            "frozen_oof_sha256": _sha256(Path(args.outer_oof)),
            "train_features_sha256": _sha256(cache / "train_features.parquet"),
            "train_anchors_sha256": _sha256(cache / "train_anchors.parquet"),
            "output_oof_sha256": _sha256(oof_path),
            "raw_rows_written": int(len(corrected)),
            "external_observations_used": 0,
        },
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
