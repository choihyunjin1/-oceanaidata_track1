"""Small event-phase and multiscale feature bank for high-wave P3 anchors."""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_FEATURE_PREFIX = "event_"


def _slope(values: np.ndarray, hours: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return np.nan
    x = hours[mask]
    y = values[mask]
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / denominator) if denominator > 0 else np.nan


def _run_duration(values: np.ndarray, threshold: float) -> float:
    count = 0
    for value in values[::-1]:
        if not np.isfinite(value) or value < threshold:
            break
        count += 1
    return count / 3.0


def _peak_features(values: np.ndarray, window_hours: int) -> tuple[float, float, float]:
    count = window_hours * 3 + 1
    section = values[-count:]
    finite = np.flatnonzero(np.isfinite(section))
    if not len(finite):
        return np.nan, np.nan, np.nan
    maximum_position = finite[np.argmax(section[finite])]
    peak = float(section[maximum_position])
    current = float(section[finite[-1]])
    hours_since = (len(section) - 1 - maximum_position) / 3.0
    return peak, hours_since, current - peak


def _band_power(values: np.ndarray) -> dict[str, float]:
    finite = np.flatnonzero(np.isfinite(values))
    names = ("1_3h", "3_6h", "6_12h", "12_24h", "24_48h")
    if len(finite) < 72:
        return {name: np.nan for name in names}
    filled = np.interp(np.arange(len(values)), finite, values[finite])
    filled = filled - np.mean(filled)
    power = np.abs(np.fft.rfft(filled)) ** 2
    frequency = np.fft.rfftfreq(len(filled), d=1.0 / 3.0)
    total = float(power[1:].sum())
    if total <= 0:
        return {name: 0.0 for name in names}
    bands = {
        "1_3h": (1 / 3, 1.0),
        "3_6h": (1 / 6, 1 / 3),
        "6_12h": (1 / 12, 1 / 6),
        "12_24h": (1 / 24, 1 / 12),
        "24_48h": (1 / 48, 1 / 24),
    }
    return {
        name: float(power[(frequency >= low) & (frequency < high)].sum() / total)
        for name, (low, high) in bands.items()
    }


def _wind_wave_lag(hs: np.ndarray, wind: np.ndarray) -> tuple[float, float]:
    correlations: list[tuple[float, float]] = []
    for lag_h in (0, 1, 3, 6, 9, 12):
        lag = lag_h * 3
        wave = hs[lag:] if lag else hs
        forcing = wind[:-lag] if lag else wind
        mask = np.isfinite(wave) & np.isfinite(forcing)
        if mask.sum() < 24 or np.std(wave[mask]) < 1e-6 or np.std(forcing[mask]) < 1e-6:
            continue
        correlations.append((float(np.corrcoef(wave[mask], forcing[mask])[0, 1]), float(lag_h)))
    if not correlations:
        return np.nan, np.nan
    return max(correlations, key=lambda value: value[0])


def summarize_event_phase(raw: np.ndarray) -> dict[str, float]:
    """Summarize one 289x10 sequence; raw order follows sequences.RAW_COLUMNS."""

    if raw.shape != (289, 10):
        raise ValueError("raw context must have shape (289, 10)")
    wave_slots = np.arange(0, 289, 2)
    hs = raw[wave_slots, 0].astype(float)
    wind = raw[wave_slots, 4].astype(float)
    gust = raw[wave_slots, 5].astype(float)
    pressure = raw[wave_slots, 9].astype(float)
    hours = np.arange(-len(hs) + 1, 1, dtype=float) / 3.0
    result: dict[str, float] = {}
    for threshold, label in ((1.0, "1p0"), (1.5, "1p5"), (2.0, "2p0")):
        result[f"event_hs_run_above_{label}_hours"] = _run_duration(hs, threshold)
        above = np.isfinite(hs) & (hs >= threshold)
        result[f"event_hs_crossings_{label}_48h"] = float(np.sum((~above[:-1]) & above[1:]))
    for window in (3, 6, 12, 24, 48):
        peak, recency, drop = _peak_features(hs, window)
        result[f"event_hs_peak_{window}h"] = peak
        result[f"event_hs_hours_since_peak_{window}h"] = recency
        result[f"event_hs_drop_from_peak_{window}h"] = drop
        wind_peak, wind_recency, _ = _peak_features(wind, window)
        result[f"event_wind_peak_{window}h"] = wind_peak
        result[f"event_wind_hours_since_peak_{window}h"] = wind_recency
    for window in (3, 6, 12):
        count = window * 3 + 1
        recent_hs = hs[-count:]
        previous_hs = hs[-2 * count : -count]
        recent_hours = hours[-count:]
        previous_hours = hours[-2 * count : -count]
        result[f"event_hs_slope_acceleration_{window}h"] = _slope(recent_hs, recent_hours) - _slope(
            previous_hs, previous_hours
        )
        recent_mean = np.nanmean(recent_hs) if np.isfinite(recent_hs).any() else np.nan
        previous_mean = np.nanmean(previous_hs) if np.isfinite(previous_hs).any() else np.nan
        result[f"event_hs_recent_prior_mean_delta_{window}h"] = recent_mean - previous_mean
    for name, values in (("hs", hs), ("wind", wind), ("gust", gust), ("pressure", pressure)):
        for band, power in _band_power(values).items():
            result[f"event_{name}_fft_power_{band}"] = power
    correlation, lag = _wind_wave_lag(hs, wind)
    result["event_wind_wave_max_correlation"] = correlation
    result["event_wind_lead_lag_hours"] = lag
    return result


def build_event_phase_features(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([summarize_event_phase(raw) for raw in values]).astype(np.float32)
