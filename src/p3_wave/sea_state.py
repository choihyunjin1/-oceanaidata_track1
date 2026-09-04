"""Compact, label-blind wind-sea coupling features for P3."""

from __future__ import annotations

import numpy as np
import pandas as pd

GRAVITY_M_S2 = 9.80665
WINDOW_SUFFIXES = ("current", "mean_3h", "mean_12h", "mean_24h")
SEA_STATE_FEATURES = tuple(
    name
    for suffix in WINDOW_SUFFIXES
    for name in (
        f"inverse_wave_age_proxy_{suffix}",
        f"wind_sea_growth_potential_{suffix}",
    )
)


def append_sea_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append eight fixed wind-wave coupling features without labels or future rows."""

    forbidden = {"target_hs", "label", "anomaly_type"}.intersection(frame.columns)
    if forbidden:
        raise ValueError(f"target columns are forbidden: {sorted(forbidden)}")
    result = frame.copy()
    for suffix in WINDOW_SUFFIXES:
        tp_column = f"tp_{suffix}"
        wind_column = f"wspd_{suffix}"
        alignment_column = f"wind_wave_alignment_{suffix}"
        missing = [
            column for column in (tp_column, wind_column, alignment_column) if column not in result
        ]
        if missing:
            raise ValueError(f"missing sea-state source columns: {missing}")
        period = pd.to_numeric(result[tp_column], errors="coerce").to_numpy(float)
        wind = pd.to_numeric(result[wind_column], errors="coerce").to_numpy(float)
        alignment = pd.to_numeric(result[alignment_column], errors="coerce").to_numpy(float)
        phase_speed = GRAVITY_M_S2 * period / (2.0 * np.pi)
        projected_wind = wind * np.clip(alignment, 0.0, 1.0)
        ratio = np.divide(
            projected_wind,
            phase_speed,
            out=np.full(len(result), np.nan, dtype=float),
            where=np.isfinite(phase_speed) & (phase_speed > 0.0),
        )
        ratio[~np.isfinite(projected_wind)] = np.nan
        result[f"inverse_wave_age_proxy_{suffix}"] = ratio.astype(np.float32)
        result[f"wind_sea_growth_potential_{suffix}"] = np.maximum(ratio - 0.84, 0.0).astype(
            np.float32
        )
    return result
