"""Past-only wind-wave memory features for the sealed P3 diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd

GRAVITY = 9.80665
RAW_COLUMNS = ("hs", "tp", "hmax", "wvdir", "wspd", "gust", "wdir", "airt", "relh", "caph")
VALUE_FEATURES = (
    "wwm_phase_speed_proxy_last",
    "wwm_phase_speed_proxy_ewma6",
    "wwm_phase_speed_proxy_ewma24",
    "wwm_aligned_wind_power_ewma6",
    "wwm_aligned_wind_power_ewma24",
    "wwm_aligned_wind_memory_contrast",
    "wwm_q_energy_delta_corr_lag0",
    "wwm_q_energy_delta_corr_lag3",
    "wwm_q_energy_delta_corr_lag6",
    "wwm_q_energy_delta_corr_lag12",
)
MASK_FEATURES = tuple(f"{name}_valid" for name in VALUE_FEATURES)
MEMORY_FEATURES = VALUE_FEATURES + MASK_FEATURES


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _hourly_endpoint_median(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (289,):
        raise ValueError("a P3 context must contain exactly 289 ten-minute rows")
    return np.asarray(
        [_finite_median(array[max(0, 6 * endpoint - 5) : 6 * endpoint + 1]) for endpoint in range(49)],
        dtype=np.float64,
    )


def _ewma(values: np.ndarray, tau_hours: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    if not valid.any():
        return float("nan")
    age = np.arange(len(array) - 1, -1, -1, dtype=np.float64)
    weight = np.exp(-age / tau_hours) * valid
    return float(np.nansum(array * weight) / np.sum(weight))


def _lagged_correlation(q: np.ndarray, energy: np.ndarray, lag: int) -> float:
    delta = np.diff(energy)
    start = max(1, lag)
    times = np.arange(start, len(energy), dtype=np.int64)
    left = q[times - lag]
    right = delta[times - 1]
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 12:
        return float("nan")
    x = left[valid]
    y = right[valid]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_wind_wave_memory_context(raw: np.ndarray) -> dict[str, float]:
    """Summarize one immutable 48-hour context without future interpolation."""

    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (289, len(RAW_COLUMNS)):
        raise ValueError(f"unexpected raw context shape: {values.shape}")
    hs, tp, _, wave_direction, wind_speed, _, wind_direction, _, _, _ = values.T
    simultaneous = (
        np.isfinite(tp)
        & (tp > 0.0)
        & np.isfinite(wave_direction)
        & np.isfinite(wind_speed)
        & np.isfinite(wind_direction)
    )
    alignment = np.full(len(values), np.nan, dtype=np.float64)
    alignment[simultaneous] = np.maximum(
        np.cos(np.deg2rad(wind_direction[simultaneous] - wave_direction[simultaneous])), 0.0
    )
    q_native = np.full(len(values), np.nan, dtype=np.float64)
    z_native = np.full(len(values), np.nan, dtype=np.float64)
    q_native[simultaneous] = np.square(wind_speed[simultaneous]) * alignment[simultaneous]
    phase_speed = GRAVITY * tp[simultaneous] / (2.0 * np.pi)
    z_native[simultaneous] = wind_speed[simultaneous] * alignment[simultaneous] / phase_speed
    q = _hourly_endpoint_median(q_native)
    z = _hourly_endpoint_median(z_native)
    hs_hourly = _hourly_endpoint_median(hs)
    energy = np.square(hs_hourly)
    z6, z24 = _ewma(z, 6.0), _ewma(z, 24.0)
    q6, q24 = _ewma(q, 6.0), _ewma(q, 24.0)
    values_out = (
        float(z[-1]),
        z6,
        z24,
        q6,
        q24,
        float((q24 - q6) / (abs(q24) + abs(q6) + 1e-6))
        if np.isfinite(q6) and np.isfinite(q24)
        else float("nan"),
        *(_lagged_correlation(q, energy, lag) for lag in (0, 3, 6, 12)),
    )
    result = {name: float(value) for name, value in zip(VALUE_FEATURES, values_out, strict=True)}
    result.update(
        {
            mask_name: float(np.isfinite(result[value_name]))
            for value_name, mask_name in zip(VALUE_FEATURES, MASK_FEATURES, strict=True)
        }
    )
    return result


def build_wind_wave_memory_features(
    raw_contexts: np.ndarray,
    anchor_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build one target-free feature row per immutable train context."""

    contexts = np.asarray(raw_contexts)
    if contexts.ndim != 3 or contexts.shape[1:] != (289, len(RAW_COLUMNS)):
        raise ValueError(f"unexpected raw context tensor: {contexts.shape}")
    ids = np.arange(len(contexts), dtype=np.int64) if anchor_ids is None else np.asarray(anchor_ids)
    if len(ids) != len(contexts) or len(np.unique(ids)) != len(ids):
        raise ValueError("anchor ids must bind one-to-one")
    rows = [summarize_wind_wave_memory_context(context) for context in contexts]
    frame = pd.DataFrame(rows)
    frame.insert(0, "anchor_id", ids.astype(np.int64))
    return frame
