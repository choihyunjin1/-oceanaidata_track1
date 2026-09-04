"""Small, leakage-auditable calibration helpers for P3 point forecasts."""

from __future__ import annotations

import numpy as np


def estimate_global_bias_correction(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    max_absolute_correction: float = 0.35,
) -> float:
    """Return the clipped RMSE-optimal additive correction on calibration rows."""

    truth_values = np.asarray(truth, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    if truth_values.shape != prediction_values.shape or truth_values.ndim != 1:
        raise ValueError("truth and prediction must be aligned one-dimensional arrays")
    if len(truth_values) == 0:
        raise ValueError("calibration rows cannot be empty")
    if not np.isfinite(truth_values).all() or not np.isfinite(prediction_values).all():
        raise ValueError("calibration values must be finite")
    if not np.isfinite(max_absolute_correction) or max_absolute_correction < 0.0:
        raise ValueError("max_absolute_correction must be finite and non-negative")
    raw = float(np.mean(truth_values - prediction_values))
    return float(np.clip(raw, -max_absolute_correction, max_absolute_correction))


def apply_global_bias_correction(
    prediction: np.ndarray,
    correction: float,
    *,
    lower: float = 0.0,
    upper: float = 30.0,
) -> np.ndarray:
    """Apply one frozen additive correction and the official physical bounds."""

    values = np.asarray(prediction, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("prediction must be a finite one-dimensional array")
    if not np.isfinite(correction):
        raise ValueError("correction must be finite")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("invalid clipping bounds")
    return np.clip(values + float(correction), lower, upper)
