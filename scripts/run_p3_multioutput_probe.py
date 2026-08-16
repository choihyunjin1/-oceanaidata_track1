"""Evaluate multi-output trajectory models for six correlated P3 lead times."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from p3_wave.data import LEADS
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.validation import build_forecast_folds, metric_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--output-dir", default="artifacts/p3/multioutput_probe")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--cat-only", action="store_true")
    return parser.parse_args()


def _metadata(anchors: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    lookup = anchors.set_index("anchor_id")
    blocks: list[pd.DataFrame] = []
    for lead in LEADS:
        blocks.append(
            pd.DataFrame(
                {
                    "anchor_id": ids,
                    "station": lookup.loc[ids, "station"].astype(str).to_numpy(),
                    "lead_h": lead,
                    "current_hs": lookup.loc[ids, "current_hs"].to_numpy(dtype=float),
                    "target_hs": lookup.loc[ids, f"target_{lead}"].to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def _targets(anchors: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    lookup = anchors.set_index("anchor_id")
    current = lookup.loc[ids, "current_hs"].to_numpy(dtype=float)
    return np.column_stack(
        [lookup.loc[ids, f"target_{lead}"].to_numpy(dtype=float) - current for lead in LEADS]
    )


def main() -> int:
    args = parse_args()
    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    feature_lookup = features.set_index("anchor_id")
    all_columns = [c for c in features.columns if c not in {"anchor_id", "station"}]
    compact = compact_feature_columns(all_columns)
    configurations = {
        "cat_multi_compact": ("catboost", compact),
        "cat_multi_full": ("catboost", all_columns),
        "extra_multi_compact": ("extra_trees", compact),
    }
    if args.cat_only:
        configurations = {
            name: value for name, value in configurations.items() if value[0] == "catboost"
        }
    rows: list[pd.DataFrame] = []
    for fold_number, fold in enumerate(build_forecast_folds(anchors)):
        train_current = anchors.set_index("anchor_id").loc[fold.train_ids, "current_hs"].to_numpy()
        weight = threshold_case_weights(train_current)
        target = _targets(anchors, fold.train_ids)
        valid_current = (
            anchors.set_index("anchor_id").loc[fold.validation_ids, "current_hs"].to_numpy()
        )
        metadata = _metadata(anchors, fold.validation_ids)
        for name, (backend, columns) in configurations.items():
            x_train = feature_lookup.loc[fold.train_ids, ["station", *columns]].reset_index(
                drop=True
            )
            x_valid = feature_lookup.loc[fold.validation_ids, ["station", *columns]].reset_index(
                drop=True
            )
            if backend == "catboost":
                x_train["station"] = x_train["station"].astype(str)
                x_valid["station"] = x_valid["station"].astype(str)
                parameters: dict[str, object] = {
                    "loss_function": "MultiRMSE",
                    "iterations": 1200,
                    "learning_rate": 0.03,
                    "depth": 7,
                    "l2_leaf_reg": 10.0,
                    "random_strength": 0.15,
                    "random_seed": 20260816 + fold_number,
                    "thread_count": 8,
                    "verbose": False,
                    "allow_writing_files": False,
                    "task_type": args.task_type,
                }
                if args.task_type == "GPU":
                    parameters["devices"] = "0"
                    parameters["boosting_type"] = "Plain"
                model = CatBoostRegressor(
                    **parameters,
                )
                model.fit(x_train, target, sample_weight=weight, cat_features=[0], verbose=False)
                delta = np.asarray(model.predict(x_valid), dtype=float)
            else:
                station = pd.concat([x_train.pop("station"), x_valid.pop("station")])
                station_matrix = pd.get_dummies(station.astype(str), dtype=float).to_numpy()
                imputer = SimpleImputer(strategy="median", add_indicator=True)
                numeric_train = imputer.fit_transform(x_train)
                numeric_valid = imputer.transform(x_valid)
                train_matrix = np.column_stack(
                    [numeric_train, station_matrix[: len(numeric_train)]]
                )
                valid_matrix = np.column_stack(
                    [numeric_valid, station_matrix[len(numeric_train) :]]
                )
                model = ExtraTreesRegressor(
                    n_estimators=600,
                    min_samples_leaf=12,
                    max_features=0.6,
                    n_jobs=8,
                    random_state=20260816 + fold_number,
                )
                model.fit(train_matrix, target, sample_weight=weight)
                delta = np.asarray(model.predict(valid_matrix), dtype=float)
            absolute = np.clip(delta + valid_current[:, None], 0.0, 30.0)
            frame = metadata.copy()
            frame["fold"] = fold.name
            frame["configuration"] = name
            frame["prediction"] = absolute.T.reshape(-1)
            rows.append(frame)
    oof = pd.concat(rows, ignore_index=True)
    metrics = {
        name: metric_slices(group, group["prediction"].to_numpy())
        for name, group in oof.groupby("configuration", observed=True)
    }
    persistence = oof.loc[oof["configuration"].eq(next(iter(configurations)))].copy()
    metrics["persistence"] = metric_slices(persistence, persistence["current_hs"].to_numpy())
    oof_path = output / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_multioutput_probe",
        "task_type": args.task_type,
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
