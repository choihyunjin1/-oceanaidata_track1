"""Training-target sea-state transition labels for P3 research diagnostics."""

from __future__ import annotations

import numpy as np

TRAJECTORY_CLASSES = (
    "continued_growth",
    "decay",
    "near_flat",
    "peak_then_decay",
)


def classify_future_trajectory(
    current_hs: np.ndarray,
    future_hs: np.ndarray,
    *,
    change_threshold_m: float = 0.30,
) -> np.ndarray:
    """Classify six-lead outcomes; labels are training targets, never inference inputs."""

    current = np.asarray(current_hs, dtype=float)
    future = np.asarray(future_hs, dtype=float)
    if current.ndim != 1 or future.ndim != 2 or future.shape != (len(current), 6):
        raise ValueError("expected current shape (n,) and future shape (n, 6)")
    if not np.isfinite(current).all() or not np.isfinite(future).all():
        raise ValueError("trajectory inputs must be finite")
    if not np.isfinite(change_threshold_m) or change_threshold_m <= 0.0:
        raise ValueError("change_threshold_m must be positive and finite")
    peak_gain = np.max(future, axis=1) - current
    final_gain = future[:, -1] - current
    drawdown = np.max(future, axis=1) - future[:, -1]
    result = np.full(len(current), "near_flat", dtype=object)
    result[final_gain <= -change_threshold_m] = "decay"
    result[final_gain >= change_threshold_m] = "continued_growth"
    result[(peak_gain >= change_threshold_m) & (drawdown >= change_threshold_m)] = "peak_then_decay"
    return result.astype(str)
