"""Public-profile supervised rank-one functional residual utilities for P2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from .depth_registered_cmfpca import cubic_bspline_df5, seasonal_harmonics

PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)
COEFFICIENT_COLUMNS = tuple(
    [f"public_temp_spline_{index}" for index in range(5)]
    + [f"public_psal_spline_{index}" for index in range(5)]
)


def _effective_depth(frame: pd.DataFrame) -> np.ndarray:
    actual = pd.to_numeric(frame["depth"], errors="coerce").to_numpy(dtype=np.float64)
    nominal = pd.to_numeric(frame["nominal_depth"], errors="coerce").to_numpy(dtype=np.float64)
    return np.where(np.isfinite(actual) & (actual > 0.0), actual, nominal)


def _ridge_coefficient(depth: np.ndarray, values: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    keep = np.isfinite(depth) & np.isfinite(values) & (depth >= 4.0) & (depth <= 50.0)
    if int(keep.sum()) < 4 or float(np.ptp(depth[keep])) < 10.0:
        return np.full(5, np.nan), float("inf")
    basis = cubic_bspline_df5(depth[keep])
    gram = basis.T @ basis + ridge * np.eye(5)
    condition = float(np.linalg.cond(gram))
    if not np.isfinite(condition) or condition > 1e8:
        return np.full(5, np.nan), condition
    coefficient = np.linalg.solve(gram, basis.T @ values[keep])
    return coefficient, condition


def build_public_functional_features(
    observations: pd.DataFrame,
    *,
    ridge: float = 1e-3,
    change_hours: tuple[int, ...] = (6, 24, 72),
) -> pd.DataFrame:
    """Build target-invariant public T/S spline coefficients and past changes."""

    required = {"time", "layer", "temp", "psal", "depth", "nominal_depth"}
    if missing := required.difference(observations.columns):
        raise ValueError(f"observations missing: {sorted(missing)}")
    source = observations.loc[observations["layer"].isin(PUBLIC_LAYERS)].copy()
    source["time"] = pd.to_datetime(source["time"], utc=True)
    source["effective_depth"] = _effective_depth(source)
    rows: list[dict[str, float | pd.Timestamp]] = []
    for timestamp, group in source.groupby("time", sort=True, observed=True):
        depth = group["effective_depth"].to_numpy(dtype=np.float64)
        temp = group["temp"].to_numpy(dtype=np.float64)
        psal = group["psal"].to_numpy(dtype=np.float64)
        temp_coefficient, temp_condition = _ridge_coefficient(depth, temp, ridge)
        psal_coefficient, psal_condition = _ridge_coefficient(depth, psal, ridge)
        record: dict[str, float | pd.Timestamp] = {"time": pd.Timestamp(timestamp)}
        record.update(
            {
                name: float(value)
                for name, value in zip(COEFFICIENT_COLUMNS[:5], temp_coefficient, strict=True)
            }
        )
        record.update(
            {
                name: float(value)
                for name, value in zip(COEFFICIENT_COLUMNS[5:], psal_coefficient, strict=True)
            }
        )
        record.update(
            {
                "public_temp_count": float(np.isfinite(temp).sum()),
                "public_psal_count": float(np.isfinite(psal).sum()),
                "public_depth_span": float(np.ptp(depth[np.isfinite(depth)])) if np.isfinite(depth).any() else 0.0,
                "public_temp_condition_log": float(np.log1p(temp_condition)) if np.isfinite(temp_condition) else np.nan,
                "public_psal_condition_log": float(np.log1p(psal_condition)) if np.isfinite(psal_condition) else np.nan,
                "public_profile_valid": float(np.isfinite(temp_coefficient).all() and np.isfinite(psal_coefficient).all()),
            }
        )
        rows.append(record)
    base = pd.DataFrame(rows).set_index("time").sort_index()
    result = base.copy()
    for hours in change_hours:
        lag = base.loc[:, COEFFICIENT_COLUMNS].copy()
        lag.index = lag.index + pd.Timedelta(hours=hours)
        lag = lag.rename(columns={name: f"{name}_lag{hours}h" for name in COEFFICIENT_COLUMNS})
        result = result.join(lag, how="left")
        for name in COEFFICIENT_COLUMNS:
            result[f"{name}_change{hours}h"] = result[name] - result[f"{name}_lag{hours}h"]
        result[f"public_change{hours}h_valid"] = result[
            [f"{name}_change{hours}h" for name in COEFFICIENT_COLUMNS]
        ].notna().all(axis=1).astype(float)
        result = result.drop(columns=[f"{name}_lag{hours}h" for name in COEFFICIENT_COLUMNS])
    return result


@dataclass
class SupervisedRank1Residual:
    feature_columns: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    seasonal_beta: np.ndarray
    leverage_limit: float
    model: PLSRegression
    train_rows: int

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        response: np.ndarray,
        times: pd.DatetimeIndex,
    ) -> SupervisedRank1Residual:
        matrix = features.to_numpy(dtype=np.float64)
        target = np.asarray(response, dtype=np.float64)
        if target.shape != (len(matrix), 3) or len(matrix) < 100:
            raise ValueError("rank-one residual fit has insufficient or invalid rows")
        median = np.nanmedian(matrix, axis=0)
        median[~np.isfinite(median)] = 0.0
        filled = np.where(np.isfinite(matrix), matrix, median)
        scale = np.std(filled, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        standardized = (filled - median) / scale
        harmonics = seasonal_harmonics(times)
        seasonal_beta = np.linalg.lstsq(harmonics, target, rcond=None)[0]
        residual = target - harmonics @ seasonal_beta
        model = PLSRegression(n_components=1, scale=True, max_iter=500, tol=1e-6)
        model.fit(standardized, residual)
        leverage = np.sqrt(np.mean(np.square(standardized), axis=1))
        return cls(
            feature_columns=tuple(features.columns),
            median=median,
            scale=scale,
            seasonal_beta=seasonal_beta,
            leverage_limit=float(np.quantile(leverage, 0.975)),
            model=model,
            train_rows=len(matrix),
        )

    def predict(
        self,
        features: pd.DataFrame,
        times: pd.DatetimeIndex,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if tuple(features.columns) != self.feature_columns:
            raise ValueError("functional feature surface drifted")
        matrix = features.to_numpy(dtype=np.float64)
        filled = np.where(np.isfinite(matrix), matrix, self.median)
        standardized = (filled - self.median) / self.scale
        leverage = np.sqrt(np.mean(np.square(standardized), axis=1))
        prediction = seasonal_harmonics(times) @ self.seasonal_beta + self.model.predict(standardized)
        enabled = (
            features["public_profile_valid"].to_numpy(dtype=bool)
            & np.isfinite(leverage)
            & (leverage <= self.leverage_limit)
        )
        return np.asarray(prediction, dtype=np.float64), enabled, leverage

    def receipt(self) -> dict[str, float | int]:
        return {
            "rank": 1,
            "train_rows": self.train_rows,
            "feature_count": len(self.feature_columns),
            "leverage_limit": self.leverage_limit,
            "n_iter": int(np.max(np.atleast_1d(self.model.n_iter_))),
        }


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 1e-15 else 0.0


def orthogonal_share(vector: np.ndarray, reference: np.ndarray) -> float:
    x = np.asarray(vector, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    denominator = float(np.dot(y, y))
    if float(np.dot(x, x)) <= 1e-15:
        return 0.0
    projection = np.zeros_like(x) if denominator <= 1e-15 else y * float(np.dot(x, y) / denominator)
    return float(np.linalg.norm(x - projection) / np.linalg.norm(x))
