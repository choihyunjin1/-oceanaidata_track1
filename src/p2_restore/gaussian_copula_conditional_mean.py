"""Train-only Gaussian-copula conditional means for P2 residual profiles.

The implementation deliberately keeps the nonlinear pilot small: empirical
margins, a Kendall-tau latent correlation, fixed diagonal shrinkage, and
Gauss-Hermite integration back to the residual scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import kendalltau, norm, rankdata


class CopulaContractError(RuntimeError):
    """Raised when the finite, monotone, or covariance contract is violated."""


def normal_scores(values: np.ndarray) -> np.ndarray:
    """Return finite average-rank Gaussian scores for a one-dimensional sample."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise CopulaContractError("normal scores require a finite 1D sample")
    probability = (rankdata(array, method="average") - 0.5) / len(array)
    scores = norm.ppf(probability)
    if not np.isfinite(scores).all():
        raise CopulaContractError("normal scores became nonfinite")
    return scores


def empirical_to_normal(sorted_training: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map values through a training-only mid-rank empirical CDF."""
    support = np.asarray(sorted_training, dtype=np.float64)
    query = np.asarray(values, dtype=np.float64)
    if support.ndim != 1 or len(support) < 2 or not np.isfinite(support).all():
        raise CopulaContractError("invalid empirical support")
    if not np.isfinite(query).all():
        raise CopulaContractError("query contains nonfinite values")
    left = np.searchsorted(support, query, side="left")
    right = np.searchsorted(support, query, side="right")
    rank = 0.5 * (left + right)
    probability = np.clip((rank + 0.5) / len(support), 0.5 / len(support), 1.0 - 0.5 / len(support))
    transformed = norm.ppf(probability)
    if not np.isfinite(transformed).all():
        raise CopulaContractError("empirical transform became nonfinite")
    return transformed


def empirical_quantile(sorted_training: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Linearly interpolate a training empirical quantile function."""
    support = np.asarray(sorted_training, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    if support.ndim != 1 or len(support) < 2 or not np.isfinite(support).all():
        raise CopulaContractError("invalid inverse empirical support")
    position = p * (len(support) - 1)
    low = np.floor(position).astype(np.int64)
    high = np.ceil(position).astype(np.int64)
    weight = position - low
    return support[low] * (1.0 - weight) + support[high] * weight


def kendall_latent_correlation(values: np.ndarray) -> np.ndarray:
    """Estimate a latent Gaussian correlation using sin(pi*tau/2)."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 4 or not np.isfinite(matrix).all():
        raise CopulaContractError("Kendall correlation needs a finite 2D matrix")
    dimensions = matrix.shape[1]
    result = np.eye(dimensions, dtype=np.float64)
    for left in range(dimensions):
        if np.unique(matrix[:, left]).size < 2:
            raise CopulaContractError(f"constant copula coordinate {left}")
        for right in range(left + 1, dimensions):
            tau = float(kendalltau(matrix[:, left], matrix[:, right], method="auto").statistic)
            if not np.isfinite(tau):
                raise CopulaContractError("Kendall tau became nonfinite")
            value = float(np.sin(0.5 * np.pi * np.clip(tau, -1.0, 1.0)))
            result[left, right] = value
            result[right, left] = value
    return result


def _nearest_correlation(matrix: np.ndarray, *, eigen_floor: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = (eigenvectors * np.maximum(eigenvalues, eigen_floor)) @ eigenvectors.T
    scale = np.sqrt(np.diag(projected))
    correlation = projected / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    return 0.5 * (correlation + correlation.T)


@dataclass(frozen=True)
class GaussianCopulaConditionalMean:
    """A fitted residual conditional-mean model."""

    x_support: tuple[np.ndarray, ...]
    y_support: tuple[np.ndarray, ...]
    beta: np.ndarray
    conditional_variance: np.ndarray
    shrinkage: float
    quadrature_nodes: np.ndarray
    quadrature_weights: np.ndarray
    minimum_eigenvalue: float
    condition_number: float
    rows: int

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        shrinkage: float,
        quadrature_order: int,
        eigen_floor: float,
        maximum_condition_number: float,
    ) -> GaussianCopulaConditionalMean:
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        if x_array.ndim != 2 or y_array.ndim != 2 or len(x_array) != len(y_array):
            raise CopulaContractError("copula X/Y shape mismatch")
        if len(x_array) < max(100, x_array.shape[1] + y_array.shape[1] + 2):
            raise CopulaContractError("too few profiles for copula fit")
        if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
            raise CopulaContractError("copula training matrix is nonfinite")
        if not 0.0 < float(shrinkage) < 1.0:
            raise CopulaContractError("shrinkage must lie in (0,1)")
        joined = np.column_stack([x_array, y_array])
        raw = kendall_latent_correlation(joined)
        shrunk = (1.0 - float(shrinkage)) * raw + float(shrinkage) * np.eye(raw.shape[0])
        correlation = _nearest_correlation(shrunk, eigen_floor=float(eigen_floor))
        eigenvalues = np.linalg.eigvalsh(correlation)
        minimum = float(np.min(eigenvalues))
        condition = float(np.linalg.cond(correlation))
        if minimum < -1e-10 or not np.isfinite(condition) or condition > maximum_condition_number:
            raise CopulaContractError("latent covariance PSD/condition guard failed")
        x_count = x_array.shape[1]
        sigma_xx = correlation[:x_count, :x_count]
        sigma_yx = correlation[x_count:, :x_count]
        sigma_yy = correlation[x_count:, x_count:]
        beta = np.linalg.solve(sigma_xx, sigma_yx.T).T
        conditional = sigma_yy - beta @ sigma_yx.T
        conditional = 0.5 * (conditional + conditional.T)
        conditional_variance = np.maximum(np.diag(conditional), float(eigen_floor))
        nodes, weights = np.polynomial.hermite.hermgauss(int(quadrature_order))
        return cls(
            x_support=tuple(np.sort(x_array[:, column]) for column in range(x_array.shape[1])),
            y_support=tuple(np.sort(y_array[:, column]) for column in range(y_array.shape[1])),
            beta=beta,
            conditional_variance=conditional_variance,
            shrinkage=float(shrinkage),
            quadrature_nodes=nodes.astype(np.float64),
            quadrature_weights=weights.astype(np.float64),
            minimum_eigenvalue=minimum,
            condition_number=condition,
            rows=int(len(x_array)),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        query = np.asarray(x, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != len(self.x_support):
            raise CopulaContractError("copula query shape changed")
        z_x = np.column_stack(
            [empirical_to_normal(support, query[:, column]) for column, support in enumerate(self.x_support)]
        )
        latent_mean = z_x @ self.beta.T
        prediction = np.empty((len(query), len(self.y_support)), dtype=np.float64)
        scale_weights = self.quadrature_weights / np.sqrt(np.pi)
        for column, support in enumerate(self.y_support):
            latent = latent_mean[:, column, None] + np.sqrt(
                2.0 * self.conditional_variance[column]
            ) * self.quadrature_nodes[None, :]
            original = empirical_quantile(support, norm.cdf(latent))
            prediction[:, column] = original @ scale_weights
        if not np.isfinite(prediction).all():
            raise CopulaContractError("copula conditional mean became nonfinite")
        return prediction

    def receipt(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "x_dimensions": len(self.x_support),
            "y_dimensions": len(self.y_support),
            "shrinkage": self.shrinkage,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "condition_number": self.condition_number,
            "quadrature_order": int(len(self.quadrature_nodes)),
            "finite_monotone_empirical_margins": True,
            "kendall_tau_latent_correlation": True,
        }


@dataclass(frozen=True)
class SeasonalCopulaConditionalMean:
    """Season-specific models with a train-only global fallback."""

    global_model: GaussianCopulaConditionalMean
    seasonal_models: dict[str, GaussianCopulaConditionalMean]
    minimum_season_profiles: int

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        seasons: np.ndarray,
        **kwargs: Any,
    ) -> SeasonalCopulaConditionalMean:
        season_array = np.asarray(seasons).astype(str)
        if len(season_array) != len(x):
            raise CopulaContractError("season labels do not align")
        minimum = int(kwargs.pop("minimum_season_profiles"))
        global_model = GaussianCopulaConditionalMean.fit(x, y, **kwargs)
        models: dict[str, GaussianCopulaConditionalMean] = {}
        for season in sorted(np.unique(season_array)):
            mask = season_array == season
            if int(mask.sum()) >= minimum:
                models[str(season)] = GaussianCopulaConditionalMean.fit(x[mask], y[mask], **kwargs)
        return cls(global_model=global_model, seasonal_models=models, minimum_season_profiles=minimum)

    def predict(self, x: np.ndarray, seasons: np.ndarray) -> np.ndarray:
        query = np.asarray(x, dtype=np.float64)
        season_array = np.asarray(seasons).astype(str)
        if len(query) != len(season_array):
            raise CopulaContractError("query seasons do not align")
        prediction = np.empty((len(query), len(self.global_model.y_support)), dtype=np.float64)
        for season in sorted(np.unique(season_array)):
            mask = season_array == season
            model = self.seasonal_models.get(str(season), self.global_model)
            prediction[mask] = model.predict(query[mask])
        return prediction

    def receipt(self) -> dict[str, Any]:
        return {
            "global": self.global_model.receipt(),
            "seasonal": {key: value.receipt() for key, value in sorted(self.seasonal_models.items())},
            "minimum_season_profiles": self.minimum_season_profiles,
        }


__all__ = [
    "CopulaContractError",
    "GaussianCopulaConditionalMean",
    "SeasonalCopulaConditionalMean",
    "empirical_quantile",
    "empirical_to_normal",
    "kendall_latent_correlation",
    "normal_scores",
]
