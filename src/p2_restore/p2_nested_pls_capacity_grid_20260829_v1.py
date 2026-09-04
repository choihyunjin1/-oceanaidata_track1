"""Deterministic capacity grid helpers for the sealed P2 nested PLS run."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from .depth_registered_cmfpca import seasonal_harmonics


@dataclass(frozen=True, order=True)
class CapacityPoint:
    """One fully specified point on the preregistered 243-point surface."""

    rank: int
    spline_ridge: float
    leverage_quantile: float
    rms_cap_c: float
    strength: float

    @property
    def point_id(self) -> str:
        return (
            f"r{self.rank}_ridge{self.spline_ridge:.0e}_q{self.leverage_quantile:.3f}_"
            f"cap{self.rms_cap_c:.3f}_s{self.strength:.2f}"
        )


def capacity_grid(grid: dict[str, list[float | int]]) -> tuple[CapacityPoint, ...]:
    """Materialize the exact Cartesian grid in a stable, auditable order."""

    points = tuple(
        CapacityPoint(int(rank), float(ridge), float(quantile), float(cap), float(strength))
        for rank, ridge, quantile, cap, strength in product(
            grid["rank"],
            grid["spline_ridge"],
            grid["leverage_quantile"],
            grid["rms_cap_c"],
            grid["strength"],
        )
    )
    if len(points) != 243 or len({point.point_id for point in points}) != 243:
        raise ValueError("capacity grid must contain exactly 243 unique points")
    return points


@dataclass
class FittedPLSResidual:
    """PLS residual fit whose leverage cutoff remains a post-fit grid axis."""

    feature_columns: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    seasonal_beta: np.ndarray
    train_leverage: np.ndarray
    model: PLSRegression
    rank: int
    train_rows: int

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        response: np.ndarray,
        times: pd.DatetimeIndex,
        *,
        rank: int,
    ) -> FittedPLSResidual:
        matrix = features.to_numpy(dtype=np.float64)
        target = np.asarray(response, dtype=np.float64)
        if target.shape != (len(matrix), 3) or len(matrix) < 100:
            raise ValueError("PLS residual fit has insufficient or invalid rows")
        if rank not in (1, 2, 3):
            raise ValueError("PLS capacity rank must be one of 1, 2, 3")
        median = np.nanmedian(matrix, axis=0)
        median[~np.isfinite(median)] = 0.0
        filled = np.where(np.isfinite(matrix), matrix, median)
        scale = np.std(filled, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        standardized = (filled - median) / scale
        harmonics = seasonal_harmonics(times)
        seasonal_beta = np.linalg.lstsq(harmonics, target, rcond=None)[0]
        residual = target - harmonics @ seasonal_beta
        model = PLSRegression(n_components=rank, scale=True, max_iter=500, tol=1e-6)
        model.fit(standardized, residual)
        leverage = np.sqrt(np.mean(np.square(standardized), axis=1))
        return cls(
            feature_columns=tuple(features.columns),
            median=median,
            scale=scale,
            seasonal_beta=seasonal_beta,
            train_leverage=leverage,
            model=model,
            rank=rank,
            train_rows=len(matrix),
        )

    def predict_raw(
        self,
        features: pd.DataFrame,
        times: pd.DatetimeIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        if tuple(features.columns) != self.feature_columns:
            raise ValueError("functional feature surface drifted")
        matrix = features.to_numpy(dtype=np.float64)
        filled = np.where(np.isfinite(matrix), matrix, self.median)
        standardized = (filled - self.median) / self.scale
        leverage = np.sqrt(np.mean(np.square(standardized), axis=1))
        prediction = seasonal_harmonics(times) @ self.seasonal_beta
        prediction += self.model.predict(standardized)
        return np.asarray(prediction, dtype=np.float64), leverage

    def leverage_limit(self, quantile: float) -> float:
        if quantile not in (0.95, 0.975, 0.99):
            raise ValueError("leverage quantile is outside the sealed grid")
        return float(np.quantile(self.train_leverage, quantile))

    def receipt(self, quantile: float) -> dict[str, float | int]:
        return {
            "rank": self.rank,
            "train_rows": self.train_rows,
            "feature_count": len(self.feature_columns),
            "leverage_quantile": quantile,
            "leverage_limit": self.leverage_limit(quantile),
            "n_iter": int(np.max(np.atleast_1d(self.model.n_iter_))),
        }


def select_inner_point(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select only from truth-bearing *inner* records using a fixed lexicographic key."""

    if len(records) != 243:
        raise ValueError("selection requires all 243 inner grid records")
    ids = [str(record.get("point_id")) for record in records]
    if len(set(ids)) != 243:
        raise ValueError("inner grid records are missing or duplicated")
    forbidden = {"outer_truth", "outer_rmse", "outer_delta_rmse", "outer_score"}
    if any(forbidden.intersection(record) for record in records):
        raise ValueError("outer outcome leaked into inner selection records")
    required = {
        "point_id",
        "candidate_rmse",
        "delta_rmse",
        "worst_group_delta_rmse",
        "worst_layer_delta_rmse",
        "correction_p99_c",
        "point",
    }
    if any(required.difference(record) for record in records):
        raise ValueError("inner selection record schema drifted")

    def selection_key(record: dict[str, Any]) -> tuple[Any, ...]:
        point = record["point"]
        eligible = bool(
            float(record["delta_rmse"]) < 0.0
            and float(record["worst_group_delta_rmse"]) <= 0.003
            and float(record["worst_layer_delta_rmse"]) <= 0.003
            and float(record["correction_p99_c"]) <= 0.20 + 1e-12
        )
        return (
            0 if eligible else 1,
            float(record["candidate_rmse"]),
            int(point["rank"]),
            abs(np.log10(float(point["spline_ridge"])) + 3.0),
            abs(float(point["leverage_quantile"]) - 0.975),
            abs(float(point["rms_cap_c"]) - 0.05),
            abs(float(point["strength"]) - 0.75),
            str(record["point_id"]),
        )

    selected = min(records, key=selection_key).copy()
    selected["inner_selection_eligible"] = selection_key(selected)[0] == 0
    return selected


__all__ = [
    "CapacityPoint",
    "FittedPLSResidual",
    "capacity_grid",
    "select_inner_point",
]
