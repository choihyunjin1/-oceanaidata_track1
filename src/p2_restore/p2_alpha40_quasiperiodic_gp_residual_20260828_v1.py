"""Bounded alpha40 plus uncertainty-gated quasi-periodic residual pilot for P2.

The residual model is a deterministic random-feature approximation to an
additive state and state-modulated M2-periodic Gaussian process.  Bayesian
linear regression supplies posterior predictive standard deviations used for
an exact no-op fallback.  This module has no deployment or submission path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import BayesianRidge

TARGET_LAYERS = (2, 3, 4)
M2_PERIOD_HOURS = 12.42


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if actual.shape != estimate.shape or not np.isfinite(actual).all() or not np.isfinite(estimate).all():
        raise ValueError("RMSE inputs must be aligned and finite")
    return float(np.sqrt(np.mean((estimate - actual) ** 2)))


def predict_forward_seasonal_oas(
    panel: pd.DataFrame,
    query: pd.DataFrame,
    *,
    train_stop: pd.Timestamp,
    exclude_start: pd.Timestamp | None = None,
    exclude_stop: pd.Timestamp | None = None,
    season_bin_days: int = 14,
    season_window_days: float = 60.0,
    minimum_season_rows: int = 100,
    fallback_nearest_complete_rows: int = 1000,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Predict layer-ID seasonal OAS with labels strictly before ``train_stop``.

    Optional exclusion is used only for residual-training cross-fits inside an
    already-past prefix.  Evaluation covariates may be later than train_stop;
    target temperature and salinity are never among the conditioning columns.
    """

    stop = pd.Timestamp(train_stop).tz_convert("UTC")
    x_columns = [
        column
        for column in panel
        if column.startswith(("temp_", "psal_", "doy_"))
        and not (
            column.startswith(("temp_", "psal_"))
            and int(column.rsplit("_", 1)[1]) in TARGET_LAYERS
        )
    ]
    y_columns = [
        f"{name}_{layer}" for layer in TARGET_LAYERS for name in ("temp", "psal")
    ]
    query_times = pd.DatetimeIndex(pd.to_datetime(query["time"], utc=True))
    unique_times = pd.DatetimeIndex(query_times.drop_duplicates().sort_values())
    if not unique_times.isin(panel.index).all():
        raise ValueError("query timestamp is absent from the public panel")
    evaluate = panel.loc[unique_times, x_columns]
    values_x = evaluate.to_numpy(np.float64)
    patterns = np.isfinite(values_x)
    prediction = np.full((len(evaluate), len(y_columns)), np.nan, dtype=np.float64)
    local = unique_times.tz_convert("Asia/Seoul")
    bins = ((local.dayofyear.to_numpy() - 1) // season_bin_days).astype(int)
    train_mask = panel.index < stop
    if exclude_start is not None or exclude_stop is not None:
        if exclude_start is None or exclude_stop is None:
            raise ValueError("both exclusion bounds are required")
        left = pd.Timestamp(exclude_start).tz_convert("UTC")
        right = pd.Timestamp(exclude_stop).tz_convert("UTC")
        train_mask &= (panel.index < left) | (panel.index >= right)
    train_index = panel.index[train_mask]
    train_doy = train_index.tz_convert("Asia/Seoul").dayofyear.to_numpy(np.float64)
    nx = len(x_columns)
    receipts: list[dict[str, float | int]] = []
    for season_bin in np.unique(bins):
        center = float(season_bin * season_bin_days + 7.5)
        distance = np.abs(train_doy - center)
        distance = np.minimum(distance, 365.2425 - distance)
        complete = panel.loc[train_index, x_columns + y_columns].dropna()
        complete_doy = complete.index.tz_convert("Asia/Seoul").dayofyear.to_numpy(np.float64)
        complete_distance = np.abs(complete_doy - center)
        complete_distance = np.minimum(complete_distance, 365.2425 - complete_distance)
        train = complete.loc[complete_distance <= season_window_days]
        fallback_used = False
        if len(train) < int(minimum_season_rows) and len(complete) >= int(minimum_season_rows):
            take = min(int(fallback_nearest_complete_rows), len(complete))
            nearest = np.argsort(complete_distance, kind="stable")[:take]
            train = complete.iloc[nearest]
            fallback_used = True
        if len(train) < int(minimum_season_rows):
            raise RuntimeError(
                f"insufficient prefix seasonal OAS rows for bin {season_bin}: {len(train)}"
            )
        matrix = train.to_numpy(np.float64)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        estimator = OAS(store_precision=False, assume_centered=False).fit(
            (matrix - mean) / scale
        )
        covariance = estimator.covariance_
        sigma_xx = covariance[:nx, :nx]
        sigma_yx = covariance[nx:, :nx]
        bin_rows = np.flatnonzero(bins == season_bin)
        for pattern in np.unique(patterns[bin_rows], axis=0):
            row_ids = bin_rows[np.all(patterns[bin_rows] == pattern, axis=1)]
            observed = np.flatnonzero(pattern)
            if len(observed):
                conditional = sigma_yx[:, observed] @ np.linalg.pinv(
                    sigma_xx[np.ix_(observed, observed)], rcond=1e-10
                )
                standardized = (
                    values_x[np.ix_(row_ids, observed)] - mean[observed]
                ) / scale[observed]
                conditional_y = standardized @ conditional.T
            else:
                conditional_y = np.zeros((len(row_ids), len(y_columns)))
            prediction[row_ids] = mean[nx:] + conditional_y * scale[nx:]
        receipts.append(
            {
                "season_bin": int(season_bin),
                "train_timestamps": int(len(train)),
                "oas_shrinkage": float(estimator.shrinkage_),
                "nearest_prefix_fallback_used": fallback_used,
            }
        )
    lookup = {
        (timestamp, layer): prediction[position, y_columns.index(f"temp_{layer}")]
        for position, timestamp in enumerate(unique_times)
        for layer in TARGET_LAYERS
    }
    result = np.asarray(
        [
            lookup[(timestamp, int(layer))]
            for timestamp, layer in zip(query_times, query["layer"], strict=True)
        ],
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise RuntimeError("forward OAS produced non-finite predictions")
    return result, receipts


@dataclass(frozen=True)
class QuasiPeriodicFeatureMap:
    columns: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    rff: RBFSampler
    gamma: float
    components: int
    seed: int

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        *,
        gamma: float,
        components: int,
        seed: int,
    ) -> QuasiPeriodicFeatureMap:
        excluded = {
            "annual_sin",
            "annual_cos",
            "m2_sin",
            "m2_cos",
            "public_temp_count",
            "public_psal_count",
            "public_depth_span",
        }
        columns = tuple(
            column
            for column in features.columns
            if column not in excluded and not column.startswith("depth_")
        )
        if not columns:
            raise ValueError("no public state columns")
        values = features.loc[:, columns].to_numpy(np.float64)
        center = np.nanmedian(values, axis=0)
        center[~np.isfinite(center)] = 0.0
        absolute = np.abs(values - center)
        scale = np.nanmedian(absolute, axis=0) * 1.4826
        fallback = np.nanstd(values, axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-6)] = fallback[
            ~np.isfinite(scale) | (scale < 1e-6)
        ]
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        standardized = np.where(np.isfinite(values), (values - center) / scale, 0.0)
        standardized = np.clip(standardized, -8.0, 8.0)
        rff = RBFSampler(gamma=float(gamma), n_components=int(components), random_state=int(seed))
        rff.fit(standardized)
        return cls(columns, center, scale, rff, float(gamma), int(components), int(seed))

    def transform(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = features.loc[:, self.columns].to_numpy(np.float64)
        standardized_raw = np.where(
            np.isfinite(values), (values - self.center) / self.scale, 0.0
        )
        maximum_absolute = np.max(np.abs(standardized_raw), axis=1)
        standardized = np.clip(standardized_raw, -8.0, 8.0)
        state = self.rff.transform(standardized)
        m2_sin = features["m2_sin"].to_numpy(np.float64)[:, None]
        m2_cos = features["m2_cos"].to_numpy(np.float64)[:, None]
        annual = features[["annual_sin", "annual_cos"]].to_numpy(np.float64)
        design = np.column_stack(
            (
                standardized,
                state,
                state * m2_sin,
                state * m2_cos,
                annual,
                m2_sin,
                m2_cos,
            )
        )
        if not np.isfinite(design).all():
            raise RuntimeError("quasi-periodic design contains non-finite values")
        return design, maximum_absolute

    def receipt(self) -> dict[str, object]:
        return {
            "public_state_columns": list(self.columns),
            "state_rff_gamma": self.gamma,
            "state_rff_components": self.components,
            "rff_seed": self.seed,
            "design": "linear_state + state_RFF * (1 + sin(M2) + cos(M2)) + annual_mean",
        }


@dataclass(frozen=True)
class FittedResidualLayer:
    feature_map: QuasiPeriodicFeatureMap
    model: BayesianRidge
    uncertainty_threshold: float
    train_rows: int

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        residual: np.ndarray,
        *,
        gamma: float,
        components: int,
        seed: int,
        uncertainty_quantile: float,
        max_iter: int,
        tolerance: float,
    ) -> FittedResidualLayer:
        target = np.asarray(residual, dtype=np.float64)
        if target.shape != (len(features),) or not np.isfinite(target).all() or len(target) < 500:
            raise ValueError("invalid residual training set")
        mapping = QuasiPeriodicFeatureMap.fit(
            features, gamma=gamma, components=components, seed=seed
        )
        design, _ = mapping.transform(features)
        model = BayesianRidge(
            max_iter=int(max_iter),
            tol=float(tolerance),
            compute_score=False,
            fit_intercept=True,
        )
        model.fit(design, target)
        _, standard_deviation = model.predict(design, return_std=True)
        threshold = float(np.quantile(standard_deviation, uncertainty_quantile))
        return cls(mapping, model, threshold, len(target))

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        design, maximum_absolute = self.feature_map.transform(features)
        mean, standard_deviation = self.model.predict(design, return_std=True)
        return (
            np.asarray(mean, dtype=np.float64),
            np.asarray(standard_deviation, dtype=np.float64),
            maximum_absolute,
        )

    def receipt(self) -> dict[str, object]:
        return {
            "train_rows": self.train_rows,
            "uncertainty_threshold": self.uncertainty_threshold,
            "posterior_alpha": float(self.model.alpha_),
            "posterior_lambda": float(self.model.lambda_),
            "feature_map": self.feature_map.receipt(),
        }


def bounded_profile_correction(
    raw_correction: np.ndarray,
    enabled: np.ndarray,
    *,
    rms_cap: float,
    p99_cap: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    raw = np.asarray(raw_correction, dtype=np.float64)
    gate = np.asarray(enabled, dtype=bool)
    if raw.shape != gate.shape or not np.isfinite(raw).all():
        raise ValueError("correction inputs differ")
    correction = np.zeros_like(raw)
    correction[gate] = np.clip(raw[gate], -p99_cap, p99_cap)
    current_rms = float(np.sqrt(np.mean(correction**2)))
    scale = min(1.0, rms_cap / current_rms) if current_rms > 0 else 1.0
    correction *= scale
    p99 = float(np.quantile(np.abs(correction), 0.99)) if len(correction) else 0.0
    diagnostics = {
        "rows": int(len(correction)),
        "enabled_rows": int(gate.sum()),
        "enabled_fraction": float(gate.mean()) if len(gate) else 0.0,
        "scale_to_rms_cap": float(scale),
        "rms_c": float(np.sqrt(np.mean(correction**2))),
        "p99_absolute_c": p99,
        "maximum_absolute_c": float(np.max(np.abs(correction))) if len(correction) else 0.0,
        "fallback_maximum_absolute_c": float(np.max(np.abs(correction[~gate]))) if (~gate).any() else 0.0,
    }
    if diagnostics["rms_c"] > rms_cap + 1e-12 or p99 > p99_cap + 1e-12:
        raise RuntimeError("correction cap failed")
    if diagnostics["fallback_maximum_absolute_c"] != 0.0:
        raise RuntimeError("no-op fallback changed")
    return correction, diagnostics


def paired_kst_day_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    work = frame.loc[:, ["time", "truth", "reference", "candidate"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work["day"] = work["time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    work["se_reference"] = (work["reference"] - work["truth"]) ** 2
    work["se_candidate"] = (work["candidate"] - work["truth"]) ** 2
    daily = work.groupby("day", sort=True).agg(
        rows=("truth", "size"),
        se_reference=("se_reference", "sum"),
        se_candidate=("se_candidate", "sum"),
    )
    if len(daily) < 10:
        raise ValueError("too few KST days")
    values = daily.to_numpy(np.float64)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    for draw in range(int(replicates)):
        sampled = values[rng.integers(0, len(values), size=len(values))]
        rows = sampled[:, 0].sum()
        draws[draw] = np.sqrt(sampled[:, 2].sum() / rows) - np.sqrt(
            sampled[:, 1].sum() / rows
        )
    return {
        "unit": "KST calendar day",
        "days": int(len(daily)),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_delta_rmse": float(draws.mean()),
        "ci90_low": float(np.quantile(draws, 0.05)),
        "ci90_high": float(np.quantile(draws, 0.95)),
        "probability_improved": float(np.mean(draws < 0.0)),
    }


def evaluate_gate(
    *,
    aggregate_delta: float,
    ci90_high: float,
    fold_deltas: dict[str, float],
    layer_deltas: dict[str, float],
    correction_rms: float,
    correction_p99: float,
    thresholds: dict[str, float | int],
) -> dict[str, object]:
    checks = {
        "aggregate_delta_rmse": aggregate_delta <= float(thresholds["aggregate_delta_rmse_max_c"]),
        "paired_ci90_upper": ci90_high < float(thresholds["paired_kst_day_bootstrap_ci90_upper_max_c"]),
        "improved_folds": sum(value < 0.0 for value in fold_deltas.values()) >= int(thresholds["minimum_improved_folds"]),
        "worst_fold_regression": max(fold_deltas.values()) <= float(thresholds["maximum_worst_fold_regression_c"]),
        "maximum_layer_regression": max(layer_deltas.values()) <= float(thresholds["maximum_layer_regression_c"]),
        "correction_rms": correction_rms <= float(thresholds["maximum_correction_rms_c"]) + 1e-12,
        "correction_p99": correction_p99 <= float(thresholds["maximum_correction_p99_c"]) + 1e-12,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "aggregate_delta_rmse_c": aggregate_delta,
            "bootstrap_ci90_high_c": ci90_high,
            "improved_folds": int(sum(value < 0.0 for value in fold_deltas.values())),
            "worst_fold_regression_c": float(max(fold_deltas.values())),
            "maximum_layer_regression_c": float(max(layer_deltas.values())),
            "correction_rms_c": correction_rms,
            "correction_p99_c": correction_p99,
        },
        "thresholds": thresholds,
    }


__all__ = [
    "FittedResidualLayer",
    "M2_PERIOD_HOURS",
    "QuasiPeriodicFeatureMap",
    "TARGET_LAYERS",
    "bounded_profile_correction",
    "evaluate_gate",
    "paired_kst_day_bootstrap",
    "predict_forward_seasonal_oas",
    "rmse",
]
