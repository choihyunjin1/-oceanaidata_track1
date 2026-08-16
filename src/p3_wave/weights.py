"""Fixed training-only weighting schemes for P3 research ablations."""

from __future__ import annotations

import numpy as np


def amplitude_emphasis_weights(
    base_weight: np.ndarray,
    target_residual: np.ndarray,
    *,
    strength: float = 0.5,
    scale_m: float = 2.0,
) -> np.ndarray:
    """Mildly emphasize large training residual amplitudes without inference inputs."""

    base = np.asarray(base_weight, dtype=float)
    residual = np.asarray(target_residual, dtype=float)
    if base.ndim != 1 or residual.ndim != 1 or base.shape != residual.shape:
        raise ValueError("base_weight and target_residual must be aligned vectors")
    if not np.isfinite(base).all() or not np.isfinite(residual).all() or np.any(base <= 0.0):
        raise ValueError("weights and residuals must be finite; base weights must be positive")
    if not 0.0 <= strength <= 1.0 or not np.isfinite(scale_m) or scale_m <= 0.0:
        raise ValueError("strength must be 0..1 and scale_m must be positive")
    multiplier = 1.0 + strength * np.clip(np.abs(residual) / scale_m, 0.0, 1.0)
    return base * multiplier
