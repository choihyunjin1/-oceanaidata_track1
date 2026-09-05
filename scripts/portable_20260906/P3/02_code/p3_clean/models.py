"""Minimal clean recipe functions extracted without algorithm changes."""

from __future__ import annotations

import numpy as np


def compact_feature_columns(columns: list[str] | tuple[str, ...]) -> list[str]:
    """Predeclared low-variance feature surface for the first structural tournament."""

    keep_tokens = (
        "_current",
        "_lag_1h",
        "_lag_3h",
        "_lag_6h",
        "_lag_12h",
        "_lag_24h",
        "_lag_48h",
        "_mean_3h",
        "_std_3h",
        "_delta_3h",
        "_slope_3h",
        "_valid_3h",
        "_mean_6h",
        "_std_6h",
        "_delta_6h",
        "_slope_6h",
        "_valid_6h",
        "_mean_12h",
        "_std_12h",
        "_delta_12h",
        "_slope_12h",
        "_valid_12h",
        "_mean_24h",
        "_std_24h",
        "_delta_24h",
        "_slope_24h",
        "_valid_24h",
        "_mean_48h",
        "_std_48h",
        "_delta_48h",
        "_slope_48h",
        "_valid_48h",
        "hs_change_",
        "wspd_change_",
        "caph_change_",
        "event_",
    )
    return [column for column in columns if any(token in column for token in keep_tokens)]


def threshold_case_weights(current_hs: np.ndarray) -> np.ndarray:
    """Fixed mild weighting for the public case-selection shift toward the 1.5m threshold."""

    current = np.asarray(current_hs, dtype=float)
    weight = np.exp(-0.45 * np.maximum(current - 1.5, 0.0))
    return weight / np.mean(weight)
