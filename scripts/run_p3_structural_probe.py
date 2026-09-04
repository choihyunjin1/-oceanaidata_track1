"""Probe intended P3 structures: lead-specific boosting and analog trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p3_wave.data import LEADS
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.validation import build_forecast_folds, metric_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/p3/features_v1")
    parser.add_argument("--output-dir", default="artifacts/p3/structural_probe")
    return parser.parse_args()


def _design(features: pd.DataFrame, anchor_ids: np.ndarray, columns: list[str]) -> pd.DataFrame:
    lookup = features.set_index("anchor_id")
    frame = lookup.loc[anchor_ids, ["station", *columns]].reset_index(drop=True)
    frame["station"] = frame["station"].astype("category")
    return frame


def _domain_weights(train: pd.DataFrame, reference: pd.DataFrame, columns: list[str]) -> np.ndarray:
    categorical = pd.concat([train["station"], reference["station"]], ignore_index=True)
    station = pd.get_dummies(categorical.astype(str), dtype=float)
    numeric = pd.concat([train[columns], reference[columns]], ignore_index=True)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=1000, random_state=20260816),
    )
    matrix = np.column_stack([numeric.to_numpy(dtype=float), station.to_numpy(dtype=float)])
    label = np.r_[np.zeros(len(train)), np.ones(len(reference))]
    model.fit(matrix, label)
    probability = model.predict_proba(matrix[: len(train)])[:, 1]
    ratio = probability / np.maximum(1.0 - probability, 1e-6)
    ratio *= len(train) / len(reference)
    ratio = np.clip(ratio, 0.1, 8.0)
    return ratio / ratio.mean()


def _lgb_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.55,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=8,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )


def _boost_predictions(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    columns: list[str],
    weighting: str,
    fold_number: int,
) -> np.ndarray:
    x_train = _design(features, train_ids, columns)
    x_valid = _design(features, validation_ids, columns)
    anchor_lookup = anchors.set_index("anchor_id")
    current_train = anchor_lookup.loc[train_ids, "current_hs"].to_numpy(dtype=float)
    if weighting == "threshold":
        weights = threshold_case_weights(current_train)
    elif weighting == "domain":
        domain_columns = [
            c
            for c in columns
            if c
            in {
                "hs_current",
                "hs_lag_1h",
                "hs_lag_3h",
                "hs_lag_6h",
                "hs_mean_6h",
                "hs_std_6h",
                "hs_slope_3h",
                "hs_slope_6h",
                "wspd_current",
                "wspd_mean_6h",
                "wspd_slope_6h",
                "gust_current",
                "caph_current",
                "caph_slope_6h",
                "tp_current",
                "wind_wave_alignment_current",
            }
        ]
        weights = _domain_weights(x_train, x_valid, domain_columns)
    elif weighting == "uniform":
        weights = np.ones(len(train_ids))
    else:
        raise ValueError(weighting)
    predictions: list[np.ndarray] = []
    for lead in LEADS:
        target = anchor_lookup.loc[train_ids, f"target_{lead}"].to_numpy(dtype=float)
        delta = target - current_train
        model = _lgb_model(20260816 + fold_number * 10 + lead)
        model.fit(x_train, delta, sample_weight=weights, categorical_feature=["station"])
        predictions.append(
            anchor_lookup.loc[validation_ids, "current_hs"].to_numpy(dtype=float)
            + model.predict(x_valid)
        )
    return np.concatenate(predictions)


def _analog_predictions(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    columns: list[str],
    *,
    neighbors: int,
    same_station: bool,
) -> np.ndarray:
    lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    selected = [
        c
        for c in columns
        if c
        in {
            "hs_current",
            "hs_lag_1h",
            "hs_lag_3h",
            "hs_lag_6h",
            "hs_lag_9h",
            "hs_lag_12h",
            "hs_lag_18h",
            "hs_lag_24h",
            "hs_lag_36h",
            "hs_lag_48h",
            "hs_std_3h",
            "hs_std_6h",
            "hs_std_12h",
            "hs_slope_3h",
            "hs_slope_6h",
            "hs_slope_12h",
            "tp_current",
            "tp_mean_6h",
            "hmax_hs_ratio_current",
            "wspd_current",
            "wspd_mean_3h",
            "wspd_mean_6h",
            "wspd_slope_3h",
            "wspd_slope_6h",
            "gust_current",
            "caph_current",
            "caph_slope_3h",
            "caph_slope_6h",
            "wind_wave_alignment_current",
            "wind_input_proxy_mean_6h",
        }
    ]
    output = np.empty((len(LEADS), len(validation_ids)), dtype=float)
    for station in sorted(lookup.loc[validation_ids, "station"].astype(str).unique()):
        valid_mask = lookup.loc[validation_ids, "station"].astype(str).to_numpy() == station
        valid_station_ids = validation_ids[valid_mask]
        if same_station:
            train_mask = lookup.loc[train_ids, "station"].astype(str).to_numpy() == station
            train_station_ids = train_ids[train_mask]
        else:
            train_station_ids = train_ids
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        train_matrix = imputer.fit_transform(lookup.loc[train_station_ids, selected])
        valid_matrix = imputer.transform(lookup.loc[valid_station_ids, selected])
        scaler = StandardScaler()
        train_matrix = scaler.fit_transform(train_matrix)
        valid_matrix = scaler.transform(valid_matrix)
        k = min(neighbors, len(train_station_ids))
        search = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=8)
        search.fit(train_matrix)
        distance, indices = search.kneighbors(valid_matrix)
        weight = 1.0 / np.maximum(distance, 0.2) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        current = anchor_lookup.loc[valid_station_ids, "current_hs"].to_numpy(dtype=float)
        for lead_number, lead in enumerate(LEADS):
            train_delta = anchor_lookup.loc[train_station_ids, f"target_{lead}"].to_numpy(
                dtype=float
            ) - anchor_lookup.loc[train_station_ids, "current_hs"].to_numpy(dtype=float)
            output[lead_number, valid_mask] = current + np.sum(
                weight * train_delta[indices], axis=1
            )
    return output.reshape(-1)


def _ridge_physics_predictions(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
) -> np.ndarray:
    lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    columns = [
        "hs_current",
        "hs_change_1h",
        "hs_change_3h",
        "hs_change_6h",
        "hs_change_12h",
        "hs_slope_3h",
        "hs_slope_6h",
        "hs_std_6h",
        "wspd_current",
        "wspd_change_3h",
        "wspd_change_6h",
        "caph_change_3h",
        "caph_change_6h",
        "wind_wave_alignment_current",
        "wind_input_proxy_mean_6h",
    ]
    station = pd.concat([lookup.loc[train_ids, "station"], lookup.loc[validation_ids, "station"]])
    one_hot = pd.get_dummies(station.astype(str), dtype=float)
    train_numeric = lookup.loc[train_ids, columns].to_numpy(dtype=float)
    valid_numeric = lookup.loc[validation_ids, columns].to_numpy(dtype=float)
    train_matrix = np.column_stack([train_numeric, one_hot.iloc[: len(train_ids)].to_numpy()])
    valid_matrix = np.column_stack([valid_numeric, one_hot.iloc[len(train_ids) :].to_numpy()])
    predictions: list[np.ndarray] = []
    current_train = anchor_lookup.loc[train_ids, "current_hs"].to_numpy(dtype=float)
    current_valid = anchor_lookup.loc[validation_ids, "current_hs"].to_numpy(dtype=float)
    for lead in LEADS:
        delta = anchor_lookup.loc[train_ids, f"target_{lead}"].to_numpy(dtype=float) - current_train
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=30.0),
        )
        model.fit(train_matrix, delta, ridge__sample_weight=threshold_case_weights(current_train))
        predictions.append(current_valid + model.predict(valid_matrix))
    return np.concatenate(predictions)


def main() -> int:
    args = parse_args()
    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    feature_columns = compact_feature_columns(
        [c for c in features.columns if c not in {"anchor_id", "station"}]
    )
    configurations = (
        "lgb_per_lead_uniform",
        "lgb_per_lead_threshold",
        "lgb_per_lead_domain",
        "analog_station_k20",
        "analog_station_k50",
        "analog_pooled_k50",
        "ridge_physics",
    )
    rows: list[pd.DataFrame] = []
    folds = build_forecast_folds(anchors)
    anchor_lookup = anchors.set_index("anchor_id")
    for fold_number, fold in enumerate(folds):
        valid_base: list[pd.DataFrame] = []
        for lead in LEADS:
            valid_base.append(
                pd.DataFrame(
                    {
                        "anchor_id": fold.validation_ids,
                        "station": anchor_lookup.loc[fold.validation_ids, "station"].to_numpy(),
                        "lead_h": lead,
                        "current_hs": anchor_lookup.loc[
                            fold.validation_ids, "current_hs"
                        ].to_numpy(),
                        "target_hs": anchor_lookup.loc[
                            fold.validation_ids, f"target_{lead}"
                        ].to_numpy(),
                    }
                )
            )
        metadata = pd.concat(valid_base, ignore_index=True)
        for configuration in configurations:
            if configuration.startswith("lgb_per_lead_"):
                weighting = configuration.rsplit("_", 1)[-1]
                prediction = _boost_predictions(
                    features,
                    anchors,
                    fold.train_ids,
                    fold.validation_ids,
                    feature_columns,
                    weighting,
                    fold_number,
                )
            elif configuration == "ridge_physics":
                prediction = _ridge_physics_predictions(
                    features, anchors, fold.train_ids, fold.validation_ids
                )
            else:
                same_station = "station" in configuration
                neighbors = int(configuration.rsplit("k", 1)[-1])
                prediction = _analog_predictions(
                    features,
                    anchors,
                    fold.train_ids,
                    fold.validation_ids,
                    feature_columns,
                    neighbors=neighbors,
                    same_station=same_station,
                )
            frame = metadata.copy()
            frame["fold"] = fold.name
            frame["configuration"] = configuration
            frame["prediction"] = np.clip(prediction, 0.0, 30.0)
            rows.append(frame)
    oof = pd.concat(rows, ignore_index=True)
    metrics = {
        name: metric_slices(group, group["prediction"].to_numpy())
        for name, group in oof.groupby("configuration", observed=True)
    }
    persistence = oof.loc[oof["configuration"].eq(configurations[0])].copy()
    metrics["persistence"] = metric_slices(persistence, persistence["current_hs"].to_numpy())
    oof_path = output / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_structural_probe",
        "configurations": configurations,
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
