"""Case-local 48-hour summary features shared by train and anonymous test cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import P3Data, build_anchor_table, build_training_grid

BASE_COLUMNS = ("hs", "tp", "hmax", "wspd", "gust", "airt", "relh", "caph")
DIRECTION_COLUMNS = ("wvdir", "wdir")
WINDOW_HOURS = (1, 3, 6, 12, 24, 48)
LAG_HOURS = (1, 3, 6, 9, 12, 18, 24, 36, 48)
CONTEXT_ROWS = 289


@dataclass(frozen=True)
class FeatureSet:
    features: pd.DataFrame
    anchors: pd.DataFrame
    feature_columns: tuple[str, ...]


def _safe_last(values: np.ndarray) -> float:
    finite = np.flatnonzero(np.isfinite(values))
    return float(values[finite[-1]]) if len(finite) else np.nan


def _summary(values: np.ndarray, x_hours: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(values)
    if not mask.any():
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q10": np.nan,
            "q50": np.nan,
            "q90": np.nan,
            "delta": np.nan,
            "slope": np.nan,
            "valid": 0.0,
        }
    y = values[mask]
    x = x_hours[mask]
    slope = np.nan
    if len(y) >= 3 and float(np.ptp(x)) > 0:
        x_centered = x - x.mean()
        denominator = float(np.dot(x_centered, x_centered))
        if denominator > 0:
            slope = float(np.dot(x_centered, y - y.mean()) / denominator)
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "q10": float(np.quantile(y, 0.10)),
        "q50": float(np.quantile(y, 0.50)),
        "q90": float(np.quantile(y, 0.90)),
        "delta": float(y[-1] - y[0]),
        "slope": slope,
        "valid": float(mask.mean()),
    }


def summarize_context(context: pd.DataFrame) -> dict[str, float]:
    """Summarize one ordered 48-hour case without absolute-time features."""

    if len(context) != CONTEXT_ROWS:
        raise ValueError(f"context must contain {CONTEXT_ROWS} rows")
    frame = context.reset_index(drop=True)
    result: dict[str, float] = {}
    arrays: dict[str, np.ndarray] = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        for column in BASE_COLUMNS + DIRECTION_COLUMNS
    }
    for direction in DIRECTION_COLUMNS:
        radians = np.deg2rad(arrays.pop(direction))
        arrays[f"{direction}_sin"] = np.sin(radians)
        arrays[f"{direction}_cos"] = np.cos(radians)

    hs = arrays["hs"]
    tp = arrays["tp"]
    hmax = arrays["hmax"]
    wspd = arrays["wspd"]
    gust = arrays["gust"]
    wave_sin = arrays["wvdir_sin"]
    wave_cos = arrays["wvdir_cos"]
    wind_sin = arrays["wdir_sin"]
    wind_cos = arrays["wdir_cos"]
    with np.errstate(divide="ignore", invalid="ignore"):
        arrays["wave_energy"] = hs**2
        arrays["hmax_hs_ratio"] = np.where(hs > 0.05, hmax / hs, np.nan)
        arrays["steepness_proxy"] = np.where(tp > 0.1, hs / (tp**2), np.nan)
        arrays["gust_excess"] = gust - wspd
        alignment = wind_cos * wave_cos + wind_sin * wave_sin
        alignment[~(np.isfinite(wind_cos) & np.isfinite(wave_cos))] = np.nan
        arrays["wind_wave_alignment"] = alignment
        arrays["wind_input_proxy"] = wspd**2 * np.maximum(alignment, 0.0)

    for name, values in arrays.items():
        result[f"{name}_current"] = _safe_last(values)
        for lag_h in LAG_HOURS:
            position = -1 - lag_h * 6
            result[f"{name}_lag_{lag_h}h"] = (
                float(values[position]) if np.isfinite(values[position]) else np.nan
            )
        for window_h in WINDOW_HOURS:
            length = window_h * 6 + 1
            section = values[-length:]
            x = np.arange(-len(section) + 1, 1, dtype=float) / 6.0
            for statistic, value in _summary(section, x).items():
                result[f"{name}_{statistic}_{window_h}h"] = value

    for horizon in (1, 3, 6, 12, 24):
        result[f"hs_change_{horizon}h"] = result["hs_current"] - result[f"hs_lag_{horizon}h"]
        result[f"wspd_change_{horizon}h"] = result["wspd_current"] - result[f"wspd_lag_{horizon}h"]
        result[f"caph_change_{horizon}h"] = result["caph_current"] - result[f"caph_lag_{horizon}h"]
    return result


def _extract_contexts(
    iterator: list[tuple[int | str, str, pd.DataFrame]],
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    total = len(iterator)
    for number, (identifier, station, context) in enumerate(iterator, start=1):
        row: dict[str, float | int | str] = {"identifier": identifier, "station": station}
        row.update(summarize_context(context))
        rows.append(row)
        if progress is not None and (number == total or number % 250 == 0):
            progress(number, total)
    return pd.DataFrame(rows)


def build_training_features(
    data: P3Data,
    *,
    dense_spacing_minutes: int = 60,
    anchors: pd.DataFrame | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FeatureSet:
    grid = build_training_grid(data)
    if anchors is None:
        anchors = build_anchor_table(grid, dense_spacing_minutes=dense_spacing_minutes)
    else:
        anchors = anchors.copy().reset_index(drop=True)
    iterators: list[tuple[int, str, pd.DataFrame]] = []
    by_station = {
        station: group.sort_values("time").reset_index(drop=True)
        for station, group in grid.groupby("station", sort=False, observed=True)
    }
    for row in anchors.itertuples(index=False):
        group = by_station[str(row.station)]
        stop = int(row.grid_position) + 1
        start = stop - CONTEXT_ROWS
        if start < 0:
            raise ValueError("training anchor lacks full 48-hour context")
        iterators.append((int(row.anchor_id), str(row.station), group.iloc[start:stop]))
    features = _extract_contexts(iterators, progress)
    features = features.rename(columns={"identifier": "anchor_id"})
    if not features["anchor_id"].equals(anchors["anchor_id"]):
        raise ValueError("training feature alignment mismatch")
    columns = tuple(c for c in features.columns if c not in {"anchor_id", "station"})
    return FeatureSet(features, anchors, columns)


def build_test_features(
    data: P3Data,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> FeatureSet:
    iterator: list[tuple[str, str, pd.DataFrame]] = []
    for case_id, group in data.test_context.groupby("case_id", sort=False, observed=True):
        ordered = group.sort_values("step_minute")
        iterator.append((str(case_id), str(ordered["station"].iloc[0]), ordered))
    features = _extract_contexts(iterator, progress).rename(columns={"identifier": "case_id"})
    anchors = data.test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    if not features[["case_id", "station"]].equals(anchors):
        raise ValueError("test feature alignment mismatch")
    columns = tuple(c for c in features.columns if c not in {"case_id", "station"})
    return FeatureSet(features, anchors, columns)
