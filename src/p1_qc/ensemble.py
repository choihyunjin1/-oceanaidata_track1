"""Leakage-safe non-negative convex probability blending."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConvexBlendResult:
    model_names: tuple[str, ...]
    weights: np.ndarray
    threshold: float
    validation_f1: float

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.model_names, self.weights.tolist(), strict=True))


def _prediction_matrix(
    predictions: Mapping[str, Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(predictions, Mapping):
        if not predictions:
            raise ValueError("predictions must contain at least one model")
        names = tuple(str(name) for name in predictions)
        columns = [np.asarray(predictions[name], dtype=float) for name in predictions]
        if any(column.ndim != 1 for column in columns):
            raise ValueError("each model prediction must be one-dimensional")
        lengths = {len(column) for column in columns}
        if len(lengths) != 1:
            raise ValueError("all prediction columns must have equal length")
        matrix = np.column_stack(columns)
    else:
        matrix = np.asarray(predictions, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("predictions must have shape (rows, models)")
        names = tuple(f"model_{index}" for index in range(matrix.shape[1]))
    if not np.isfinite(matrix).all():
        raise ValueError("predictions must be finite")
    if ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("model probabilities must lie in [0, 1]")
    return matrix, names


def normalize_convex_weights(
    weights: Iterable[float] | np.ndarray,
    *,
    n_models: int | None = None,
) -> np.ndarray:
    """Project non-negative weights onto the probability simplex by scaling."""

    array = np.asarray(weights, dtype=float)
    if array.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if n_models is not None and len(array) != n_models:
        raise ValueError(f"expected {n_models} weights, got {len(array)}")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("weights must be finite and non-negative")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return array / total


def convex_blend(
    predictions: Mapping[str, Sequence[float]] | np.ndarray,
    weights: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return a convex combination that remains a calibrated probability."""

    matrix, _ = _prediction_matrix(predictions)
    normalized = normalize_convex_weights(weights, n_models=matrix.shape[1])
    blended = matrix @ normalized
    return np.clip(blended, 0.0, 1.0)


def binary_f1(
    truth: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    threshold: float,
) -> float:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    target = np.asarray(truth, dtype=bool)
    score = np.asarray(probability, dtype=float)
    if target.ndim != 1 or score.shape != target.shape:
        raise ValueError("truth and probability must be equal-length vectors")
    predicted = score >= threshold
    true_positive = int(np.count_nonzero(predicted & target))
    denominator = int(np.count_nonzero(predicted)) + int(np.count_nonzero(target))
    return 2.0 * true_positive / denominator if denominator else 0.0


def simplex_lattice(n_models: int, resolution: int) -> np.ndarray:
    """Enumerate a deterministic simplex lattice of ``resolution`` units."""

    if n_models < 1:
        raise ValueError("n_models must be positive")
    if resolution < 1:
        raise ValueError("resolution must be positive")
    rows: list[list[int]] = []

    def visit(prefix: list[int], remaining_models: int, remaining: int) -> None:
        if remaining_models == 1:
            rows.append([*prefix, remaining])
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining_models - 1, remaining - value)

    visit([], n_models, resolution)
    return np.asarray(rows, dtype=float) / resolution


def fit_convex_blend(
    validation_predictions: Mapping[str, Sequence[float]] | np.ndarray,
    validation_target: Sequence[int] | np.ndarray,
    *,
    weight_resolution: int = 20,
    thresholds: Iterable[float] | np.ndarray | None = None,
) -> ConvexBlendResult:
    """Select non-negative weights and threshold on OOF/validation rows only.

    This exhaustive lattice search is intentionally transparent and robust for
    the small model families used in P1.  Never pass test predictions here.
    """

    matrix, names = _prediction_matrix(validation_predictions)
    target = np.asarray(validation_target)
    if target.shape != (matrix.shape[0],):
        raise ValueError("validation_target must have one value per prediction row")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("validation_target must be binary 0/1")
    target_bool = target.astype(bool)
    threshold_values = (
        np.linspace(0.1, 0.9, 81)
        if thresholds is None
        else np.asarray(list(thresholds), dtype=float)
    )
    if threshold_values.ndim != 1 or len(threshold_values) == 0:
        raise ValueError("thresholds must be a non-empty vector")
    if (
        not np.isfinite(threshold_values).all()
        or ((threshold_values < 0) | (threshold_values > 1)).any()
    ):
        raise ValueError("thresholds must lie in [0, 1]")

    best_score = -1.0
    best_weights: np.ndarray | None = None
    best_threshold = 0.5
    for weights in simplex_lattice(matrix.shape[1], weight_resolution):
        probability = matrix @ weights
        for threshold in threshold_values:
            score = binary_f1(target_bool, probability, float(threshold))
            # Strict comparison makes ties deterministic: lattice order and
            # threshold order are stable and recorded in the result.
            if score > best_score + 1e-15:
                best_score = score
                best_weights = weights.copy()
                best_threshold = float(threshold)
    if best_weights is None:  # pragma: no cover - guarded by validation above
        raise RuntimeError("convex blend search produced no candidate")
    return ConvexBlendResult(
        model_names=names,
        weights=best_weights,
        threshold=best_threshold,
        validation_f1=best_score,
    )


class ConvexProbabilityBlender:
    """Fit/predict facade that preserves model-name order."""

    def __init__(
        self,
        *,
        weight_resolution: int = 20,
        thresholds: Iterable[float] | np.ndarray | None = None,
    ) -> None:
        self.weight_resolution = weight_resolution
        self.thresholds = None if thresholds is None else tuple(thresholds)
        self.result_: ConvexBlendResult | None = None

    def fit(
        self,
        predictions: Mapping[str, Sequence[float]] | np.ndarray,
        target: Sequence[int] | np.ndarray,
    ) -> ConvexProbabilityBlender:
        self.result_ = fit_convex_blend(
            predictions,
            target,
            weight_resolution=self.weight_resolution,
            thresholds=self.thresholds,
        )
        return self

    def predict_proba(
        self,
        predictions: Mapping[str, Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("fit must be called before predict_proba")
        matrix, names = _prediction_matrix(predictions)
        if names != self.result_.model_names:
            raise ValueError(
                f"prediction model order/names {names} differ from fitted "
                f"{self.result_.model_names}"
            )
        positive = convex_blend(matrix, self.result_.weights)
        return np.column_stack((1.0 - positive, positive))

    def predict(
        self,
        predictions: Mapping[str, Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("fit must be called before predict")
        return (self.predict_proba(predictions)[:, 1] >= self.result_.threshold).astype(np.int8)
