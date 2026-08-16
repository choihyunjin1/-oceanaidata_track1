"""Fit the frozen P3 CatBoost residual model and create a validated submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from p3_wave.data import LEADS, load_p3_data, resolve_p3_data_dir, select_independent_validation
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.submission import build_submission, write_submission
from p3_wave.validation import expand_leads, rmse


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _model(seed: int, threads: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=700,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=8.0,
        random_strength=0.2,
        random_seed=seed,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )


def _multi_model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=1200,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=10.0,
        random_strength=0.15,
        random_seed=seed,
        task_type="GPU",
        devices="0",
        boosting_type="Plain",
        verbose=False,
        allow_writing_files=False,
    )


def _cat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["station"] = result["station"].astype(str)
    result["lead_h"] = result["lead_h"].astype(str)
    return result


def _select_final_alpha(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    columns: list[str],
    *,
    threads: int,
) -> tuple[float, dict[str, object]]:
    inner_end = anchors["anchor_time"].max() + pd.Timedelta(minutes=20)
    inner_start = inner_end - pd.Timedelta(days=60)
    calibration_ids = select_independent_validation(
        anchors, start=inner_start, end=inner_end, gap_hours=78
    )
    fit_end = inner_start - pd.Timedelta(hours=78)
    fit_ids = anchors.loc[anchors["anchor_time"].lt(fit_end), "anchor_id"].to_numpy(dtype=np.int64)
    x_fit, y_fit, fit_meta = expand_leads(features, anchors, fit_ids, columns)
    x_cal, _, cal_meta = expand_leads(features, anchors, calibration_ids, columns)
    model = _model(20260817, threads)
    model.fit(
        _cat_frame(x_fit),
        y_fit,
        sample_weight=threshold_case_weights(fit_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    persistence = cal_meta["current_hs"].to_numpy(dtype=float)
    raw = np.clip(persistence + model.predict(_cat_frame(x_cal)), 0.0, 30.0)
    truth = cal_meta["target_hs"].to_numpy(dtype=float)
    grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    scores = [rmse(truth, alpha * raw + (1.0 - alpha) * persistence) for alpha in grid]
    best = min(range(len(grid)), key=lambda index: (scores[index], grid[index]))
    return float(grid[best]), {
        "selection_scope": "last_60_days_independent_cases_inside_full_training_only",
        "fit_end_exclusive": fit_end.isoformat(),
        "calibration_start": inner_start.isoformat(),
        "calibration_end_exclusive": inner_end.isoformat(),
        "fit_anchor_count": int(len(fit_ids)),
        "calibration_case_count": int(len(calibration_ids)),
        "alpha_grid": grid.tolist(),
        "selected_alpha": float(grid[best]),
        "raw_rmse": rmse(truth, raw),
        "persistence_rmse": rmse(truth, persistence),
        "selected_rmse": float(scores[best]),
    }


def _test_matrix(
    test_features: pd.DataFrame, test_index: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    source = test_features.set_index(["case_id", "station"])
    keys = pd.MultiIndex.from_frame(test_index[["case_id", "station"]])
    selected = source.loc[keys, columns].reset_index(drop=True)
    selected.insert(0, "lead_h", test_index["lead_h"].to_numpy())
    selected.insert(0, "station", test_index["station"].astype(str).to_numpy())
    selected.insert(
        2,
        "current_hs_for_residual",
        source.loc[keys, "hs_current"].to_numpy(dtype=float),
    )
    return _cat_frame(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--output-dir", default="submissions/p3_frozen_catboost")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--single-weight", type=float, default=0.5)
    parser.add_argument(
        "--alpha",
        type=float,
        help="Frozen CatBoost weight. Omit to select it on the final 60-day inner block.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.alpha is not None and not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be within 0..1")
    if not 0.0 <= args.single_weight <= 1.0:
        raise ValueError("--single-weight must be within 0..1")
    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    test_features = pd.read_parquet(cache / "test_features.parquet")
    feature_columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    data_root = resolve_p3_data_dir(args.data_dir)
    data = load_p3_data(data_root)
    test_index = data.test_index[["case_id", "station", "lead_h"]].copy()
    if args.alpha is None:
        alpha, alpha_selection = _select_final_alpha(
            features, anchors, feature_columns, threads=args.threads
        )
    else:
        alpha = float(args.alpha)
        alpha_selection = {
            "selection_scope": "explicit_frozen_cli_value",
            "selected_alpha": alpha,
        }

    train_ids = anchors["anchor_id"].to_numpy(dtype=np.int64)
    x_train, y_train, train_meta = expand_leads(features, anchors, train_ids, feature_columns)
    model = _model(20260817, args.threads)
    model.fit(
        _cat_frame(x_train),
        y_train,
        sample_weight=threshold_case_weights(train_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    matrix = _test_matrix(test_features, test_index, feature_columns)
    current = (
        test_features.set_index(["case_id", "station"])
        .loc[pd.MultiIndex.from_frame(test_index[["case_id", "station"]]), "hs_current"]
        .to_numpy(dtype=float)
    )
    raw_prediction = np.clip(current + model.predict(matrix), 0.0, 30.0)
    case_lookup = features.set_index("anchor_id")
    multi_train = case_lookup.loc[train_ids, ["station", *feature_columns]].reset_index(drop=True)
    multi_train["station"] = multi_train["station"].astype(str)
    anchor_lookup = anchors.set_index("anchor_id")
    multi_target = np.column_stack(
        [
            anchor_lookup.loc[train_ids, f"target_{lead}"].to_numpy(dtype=float)
            - anchor_lookup.loc[train_ids, "current_hs"].to_numpy(dtype=float)
            for lead in LEADS
        ]
    )
    multi = _multi_model(20260817)
    multi.fit(
        multi_train,
        multi_target,
        sample_weight=threshold_case_weights(
            anchor_lookup.loc[train_ids, "current_hs"].to_numpy(dtype=float)
        ),
        cat_features=[0],
        verbose=False,
    )
    multi_test = test_features[["station", *feature_columns]].copy()
    multi_test["station"] = multi_test["station"].astype(str)
    multi_delta = np.asarray(multi.predict(multi_test), dtype=float)
    multi_long = pd.DataFrame(
        {
            "case_id": np.repeat(test_features["case_id"].to_numpy(), len(LEADS)),
            "station": np.repeat(test_features["station"].to_numpy(), len(LEADS)),
            "lead_h": np.tile(np.asarray(LEADS), len(test_features)),
            "multi_prediction": np.clip(
                np.repeat(test_features["hs_current"].to_numpy(dtype=float), len(LEADS))
                + multi_delta.reshape(-1),
                0.0,
                30.0,
            ),
        }
    )
    ordered_multi = test_index.merge(
        multi_long, on=["case_id", "station", "lead_h"], how="left", validate="one_to_one"
    )["multi_prediction"].to_numpy(dtype=float)
    if not np.isfinite(ordered_multi).all():
        raise ValueError("multi-output prediction alignment failed")
    ensemble_prediction = np.clip(
        args.single_weight * raw_prediction + (1.0 - args.single_weight) * ordered_multi,
        0.0,
        30.0,
    )
    prediction = np.clip(alpha * ensemble_prediction + (1.0 - alpha) * current, 0.0, 30.0)
    submission = build_submission(test_index, prediction)
    submission_path = write_submission(submission, test_index, output / "submission.csv")
    raw_path = write_submission(
        build_submission(test_index, raw_prediction), test_index, output / "submission_raw.csv"
    )
    multi_path = write_submission(
        build_submission(test_index, ordered_multi),
        test_index,
        output / "submission_multi.csv",
    )
    model_path = output / "model.cbm"
    model.save_model(model_path)
    multi_model_path = output / "model_multi.cbm"
    multi.save_model(multi_model_path)
    feature_path = output / "feature_columns.json"
    feature_path.write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "frozen_local_candidate_not_uploaded",
        "problem": "P3_wave_forecast",
        "model": "50:50 single-output and six-output CatBoost residual ensemble",
        "parameters": {
            "single": model.get_params(),
            "multi": multi.get_params(),
            "single_weight": args.single_weight,
        },
        "feature_count": len(feature_columns),
        "alpha_selection": alpha_selection,
        "positive_prediction_summary": {
            "minimum": float(np.min(prediction)),
            "median": float(np.median(prediction)),
            "maximum": float(np.max(prediction)),
        },
        "git_sha": _git_sha(),
        "input_sha256": {
            name: _sha256(data_root / name)
            for name in [
                "train_wave.csv",
                "train_atmos.csv",
                "test_context.parquet",
                "test_index.csv",
                "sample_submission.csv",
                "baseline_persistence.csv",
            ]
        },
        "artifact_sha256": {
            "submission.csv": _sha256(submission_path),
            "submission_raw.csv": _sha256(raw_path),
            "submission_multi.csv": _sha256(multi_path),
            "model.cbm": _sha256(model_path),
            "model_multi.cbm": _sha256(multi_model_path),
            "feature_columns.json": _sha256(feature_path),
        },
        "uploaded": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
