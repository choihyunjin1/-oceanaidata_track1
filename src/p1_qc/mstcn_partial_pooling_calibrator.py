"""L2-shrunk station-layer calibration for frozen P1 MS-TCN scores.

The calibrator is deliberately small: it learns one global logit slope and
station-layer intercept adjustments.  L2 regularisation pulls poorly supported
groups towards the global relation, unlike independent per-cell thresholds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _groups(station: Sequence[Any], layer: Sequence[Any]) -> np.ndarray:
    left = np.asarray(station, dtype=str)
    right = np.asarray(layer)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right):
        raise ValueError("station and layer must be aligned one-dimensional arrays")
    return np.asarray([f"{s}|{int(v)}" for s, v in zip(left, right, strict=True)], dtype=str)


def _logit(score: Sequence[float], *, epsilon: float) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("score must be a finite one-dimensional array")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


@dataclass(frozen=True)
class PartialPoolingState:
    """Portable fitted logistic calibration state."""

    categories: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    regularization_c: float
    epsilon: float
    fitted_rows: int
    positive_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p1.mstcn_partial_pooling.state.v1",
            "categories": list(self.categories),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "regularization_c": self.regularization_c,
            "epsilon": self.epsilon,
            "fitted_rows": self.fitted_rows,
            "positive_rows": self.positive_rows,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PartialPoolingState:
        if value.get("schema_version") != "p1.mstcn_partial_pooling.state.v1":
            raise ValueError("partial-pooling state schema differs")
        state = cls(
            categories=tuple(str(item) for item in value["categories"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            intercept=float(value["intercept"]),
            regularization_c=float(value["regularization_c"]),
            epsilon=float(value["epsilon"]),
            fitted_rows=int(value["fitted_rows"]),
            positive_rows=int(value["positive_rows"]),
        )
        if len(state.coefficients) != 1 + len(state.categories):
            raise ValueError("partial-pooling coefficient width differs")
        if len(set(state.categories)) != len(state.categories):
            raise ValueError("partial-pooling categories are not unique")
        if state.fitted_rows <= 0 or not 0 < state.positive_rows < state.fitted_rows:
            raise ValueError("partial-pooling target support is invalid")
        return state


def _design(
    score: Sequence[float],
    station: Sequence[Any],
    layer: Sequence[Any],
    categories: Sequence[str],
    *,
    epsilon: float,
) -> np.ndarray:
    logits = _logit(score, epsilon=epsilon)
    groups = _groups(station, layer)
    category_index = {name: index for index, name in enumerate(categories)}
    output = np.zeros((len(logits), 1 + len(categories)), dtype=np.float64)
    output[:, 0] = logits
    for row, name in enumerate(groups):
        index = category_index.get(str(name))
        if index is not None:
            output[row, 1 + index] = 1.0
    return output


def fit_partial_pooling(
    score: Sequence[float],
    station: Sequence[Any],
    layer: Sequence[Any],
    target: Sequence[int],
    *,
    eligible: Sequence[bool] | None = None,
    regularization_c: float = 0.1,
    epsilon: float = 1.0e-5,
    maximum_iterations: int = 500,
) -> PartialPoolingState:
    """Fit global logit plus L2-shrunk station-layer intercepts."""

    from sklearn.linear_model import LogisticRegression

    labels = np.asarray(target, dtype=np.int8)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("target must be a binary one-dimensional array")
    mask = np.ones(len(labels), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(labels):
        raise ValueError("eligible mask differs from target")
    categories = tuple(sorted(set(_groups(station, layer)[mask].tolist())))
    design = _design(score, station, layer, categories, epsilon=epsilon)[mask]
    selected = labels[mask]
    if len(np.unique(selected)) != 2:
        raise ValueError("eligible calibration rows lack both target classes")
    model = LogisticRegression(
        C=float(regularization_c),
        l1_ratio=0.0,
        solver="lbfgs",
        fit_intercept=True,
        class_weight=None,
        max_iter=int(maximum_iterations),
        tol=1.0e-8,
        random_state=20260829,
    )
    model.fit(design, selected)
    return PartialPoolingState.from_dict(
        PartialPoolingState(
            categories=categories,
            coefficients=tuple(float(item) for item in model.coef_[0]),
            intercept=float(model.intercept_[0]),
            regularization_c=float(regularization_c),
            epsilon=float(epsilon),
            fitted_rows=int(mask.sum()),
            positive_rows=int(selected.sum()),
        ).as_dict()
    )


def predict_partial_pooling(
    state: PartialPoolingState,
    score: Sequence[float],
    station: Sequence[Any],
    layer: Sequence[Any],
) -> np.ndarray:
    """Return calibrated row probabilities with global fallback for unseen cells."""

    state = PartialPoolingState.from_dict(state.as_dict())
    design = _design(score, station, layer, state.categories, epsilon=state.epsilon)
    logits = design @ np.asarray(state.coefficients, dtype=np.float64) + state.intercept
    probability = np.empty(len(logits), dtype=np.float64)
    nonnegative = logits >= 0.0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logits = np.exp(logits[~nonnegative])
    probability[~nonnegative] = exp_logits / (1.0 + exp_logits)
    if not np.isfinite(probability).all():
        raise RuntimeError("partial-pooling prediction produced nonfinite values")
    return probability.astype(np.float32)


__all__ = ["PartialPoolingState", "fit_partial_pooling", "predict_partial_pooling"]
