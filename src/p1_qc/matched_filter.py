"""Generator-aligned, label-blind offset/drift matched-filter features.

The competition injects finite-duration offsets and linear drifts into otherwise
good observations.  Those transformations leave paired signatures in the
first difference: an offset has opposite entry/exit jumps, while a drift has a
gradual level change followed by an opposite reset.  This module searches the
published duration ranges without reading labels and projects the strongest
compatible interval score back onto its covered rows.

All searches stay inside exact-cadence station/layer segments.  The local
difference scale is a centered seven-day rolling median absolute difference;
its 3.5-day look-ahead remains below the existing fourteen-day offline feature
dependency.  The fixed contract intentionally exposes no tuning arguments.
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from .data import segment_timeseries
from .features import FeatureBundle

MATCHED_FILTER_FEATURES = (
    "offset_pair_score_8_86h",
    "offset_pair_duration_rows",
    "drift_reset_score_9_86h",
    "drift_reset_duration_rows",
)


@dataclass(frozen=True)
class MatchedFilterConfig:
    """Immutable problem-derived geometry for the first experiment."""

    cadence_minutes: int = field(default=10, init=False)
    offset_min_rows: int = field(default=48, init=False)
    drift_min_rows: int = field(default=54, init=False)
    maximum_rows: int = field(default=519, init=False)
    scale_window_rows: int = field(default=1008, init=False)
    scale_min_observations: int = field(default=72, init=False)
    minimum_reset_z: float = field(default=3.0, init=False)
    epsilon: float = field(default=1e-6, init=False)


class _Interval(NamedTuple):
    start: int
    stop: int
    score: float
    duration: int


def _local_difference_scale(values: np.ndarray, config: MatchedFilterConfig) -> np.ndarray:
    difference = np.empty(len(values), dtype=np.float64)
    difference[0] = np.nan
    difference[1:] = np.diff(values)
    absolute = pd.Series(np.abs(difference), copy=False)
    rolling = absolute.rolling(
        config.scale_window_rows,
        min_periods=config.scale_min_observations,
        center=True,
    ).median()
    finite = absolute[np.isfinite(absolute.to_numpy())]
    fallback = float(finite.median()) if len(finite) else config.epsilon
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = config.epsilon
    scale = 1.4826 * rolling.fillna(fallback).to_numpy(dtype=np.float64)
    return np.maximum(scale, config.epsilon)


def _push_maximum(queue: deque[tuple[float, int]], value: float, index: int) -> None:
    while queue and queue[-1][0] <= value:
        queue.pop()
    queue.append((value, index))


def _expire(queue: deque[tuple[float, int]], minimum_index: int) -> None:
    while queue and queue[0][1] < minimum_index:
        queue.popleft()


def _offset_intervals(
    values: np.ndarray,
    scale: np.ndarray,
    config: MatchedFilterConfig,
) -> list[_Interval]:
    difference = np.empty(len(values), dtype=np.float64)
    difference[0] = np.nan
    difference[1:] = np.diff(values)
    standardized = difference / scale
    positive: deque[tuple[float, int]] = deque()
    negative: deque[tuple[float, int]] = deque()
    intervals: list[_Interval] = []

    for stop in range(len(values)):
        added = stop - config.offset_min_rows
        if added >= 1 and np.isfinite(standardized[added]):
            value = float(standardized[added])
            if value > 0:
                _push_maximum(positive, value, added)
            elif value < 0:
                _push_maximum(negative, -value, added)
        minimum = stop - config.maximum_rows
        _expire(positive, minimum)
        _expire(negative, minimum)

        exit_value = standardized[stop]
        if not np.isfinite(exit_value) or exit_value == 0:
            continue
        candidate = positive[0] if exit_value < 0 and positive else None
        if exit_value > 0 and negative:
            candidate = negative[0]
        if candidate is None:
            continue
        entry_magnitude, start = candidate
        duration = stop - start
        if not config.offset_min_rows <= duration <= config.maximum_rows:
            continue
        score = min(entry_magnitude, abs(float(exit_value)))
        if score > 0:
            intervals.append(_Interval(start, stop, score, duration))
    return intervals


def _linear_r_squared(values: np.ndarray, start: int, stop: int) -> float:
    y = values[start:stop]
    if len(y) < 3 or not np.isfinite(y).all():
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    x -= x.mean()
    centered = y - y.mean()
    denominator = float(np.dot(x, x) * np.dot(centered, centered))
    if denominator <= 0:
        return 0.0
    correlation = float(np.dot(x, centered))
    return float(np.clip(correlation * correlation / denominator, 0.0, 1.0))


def _drift_intervals(
    values: np.ndarray,
    scale: np.ndarray,
    config: MatchedFilterConfig,
) -> list[_Interval]:
    intervals: list[_Interval] = []
    index = np.arange(len(values), dtype=np.float64)
    prefix_y = np.r_[0.0, np.cumsum(values)]
    prefix_y2 = np.r_[0.0, np.cumsum(values * values)]
    prefix_iy = np.r_[0.0, np.cumsum(index * values)]

    for stop in range(len(values)):
        if stop < 1 or not np.isfinite(values[stop]) or not np.isfinite(values[stop - 1]):
            continue
        exit_change = float(values[stop] - values[stop - 1])
        exit_z = exit_change / float(scale[stop])
        if abs(exit_z) < config.minimum_reset_z:
            continue
        first_start = max(1, stop - config.maximum_rows)
        last_start = stop - config.drift_min_rows
        if last_start < first_start:
            continue
        starts = np.arange(first_start, last_start + 1, dtype=np.int64)
        durations = stop - starts
        total_change = values[stop - 1] - values[starts]
        opposite = total_change * exit_change < 0
        if not opposite.any():
            continue
        sum_y = prefix_y[stop] - prefix_y[starts]
        sum_y2 = prefix_y2[stop] - prefix_y2[starts]
        sum_iy = prefix_iy[stop] - prefix_iy[starts]
        mean_i = (starts + stop - 1) / 2.0
        covariance = sum_iy - mean_i * sum_y
        variance_i = durations * (durations * durations - 1) / 12.0
        variance_y = sum_y2 - sum_y * sum_y / durations
        denominator = variance_i * np.maximum(variance_y, 0.0)
        linearity = np.divide(
            covariance * covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0,
        )
        pair_scale = np.maximum(scale[starts], scale[stop])
        pair_score = np.minimum(np.abs(total_change) / pair_scale, abs(exit_z))
        scores = np.where(opposite, pair_score * np.sqrt(np.clip(linearity, 0.0, 1.0)), 0.0)
        best_local = int(np.argmax(scores))
        score = float(scores[best_local])
        if score <= 0:
            continue
        start = int(starts[best_local])
        intervals.append(_Interval(start, stop, score, stop - start))
    return intervals


def _project_intervals(intervals: list[_Interval], row_count: int) -> tuple[np.ndarray, np.ndarray]:
    starts: dict[int, list[_Interval]] = defaultdict(list)
    for interval in intervals:
        starts[interval.start].append(interval)
    score = np.zeros(row_count, dtype=np.float64)
    duration = np.zeros(row_count, dtype=np.float64)
    active: list[tuple[float, int, int, int]] = []
    for row in range(row_count):
        for interval in starts.get(row, ()):
            heapq.heappush(
                active,
                (-interval.score, interval.stop, interval.start, interval.duration),
            )
        while active and active[0][1] <= row:
            heapq.heappop(active)
        if active:
            negative_score, _, _, interval_duration = active[0]
            score[row] = -negative_score
            duration[row] = interval_duration
    return score, duration


def build_matched_filter_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build four aligned source-only matched-filter features."""

    required = {"station", "layer", "time", "temp"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing matched-filter source columns: {missing}")
    config = MatchedFilterConfig()
    source = frame.loc[:, ["station", "layer", "time", "temp"]].copy()
    segmented = segment_timeseries(
        source,
        group_columns=("station", "layer"),
        cadence_minutes=config.cadence_minutes,
    )
    segmented["__original_position"] = np.arange(len(segmented), dtype=np.int64)
    ordered = segmented.sort_values(
        ["station", "layer", "parsed_time", "__original_position"], kind="mergesort"
    ).reset_index(drop=True)

    output = np.zeros((len(ordered), len(MATCHED_FILTER_FEATURES)), dtype=np.float64)
    groups = ordered.groupby("segment_id", sort=False, observed=True).indices
    for positions_raw in groups.values():
        positions = np.asarray(positions_raw, dtype=np.int64)
        values = pd.to_numeric(ordered.iloc[positions]["temp"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if len(values) <= config.offset_min_rows or not np.isfinite(values).all():
            continue
        scale = _local_difference_scale(values, config)
        offset_score, offset_duration = _project_intervals(
            _offset_intervals(values, scale, config), len(values)
        )
        drift_score, drift_duration = _project_intervals(
            _drift_intervals(values, scale, config), len(values)
        )
        output[positions, 0] = offset_score
        output[positions, 1] = offset_duration
        output[positions, 2] = drift_score
        output[positions, 3] = drift_duration

    result = pd.DataFrame(output, columns=MATCHED_FILTER_FEATURES)
    result["__original_position"] = ordered["__original_position"].to_numpy()
    result = result.sort_values("__original_position", kind="mergesort").drop(
        columns="__original_position"
    )
    result.index = frame.index.copy()
    for column in MATCHED_FILTER_FEATURES:
        result[column] = result[column].astype(np.float32)
    result.attrs.update(
        {
            "feature_columns": MATCHED_FILTER_FEATURES,
            "feature_mode": "offline",
            "cadence_minutes": config.cadence_minutes,
            "maximum_dependency_rows": config.maximum_rows,
            "scale_window_rows": config.scale_window_rows,
            "label_blind": True,
        }
    )
    return result


def append_matched_filter_features(bundle: FeatureBundle, source: pd.DataFrame) -> FeatureBundle:
    """Append exactly the four fixed matched-filter features."""

    if len(bundle.frame) != len(source) or not bundle.frame.index.equals(source.index):
        raise ValueError("source and feature bundle must have identical rows and index")
    duplicates = sorted(set(bundle.frame.columns).intersection(MATCHED_FILTER_FEATURES))
    if duplicates:
        raise ValueError(f"matched-filter features already present: {duplicates}")
    matched = build_matched_filter_features(source)
    combined = bundle.frame.copy()
    for column in MATCHED_FILTER_FEATURES:
        combined[column] = matched[column].to_numpy(dtype=np.float32, copy=True)
    feature_columns = (*bundle.feature_columns, *MATCHED_FILTER_FEATURES)
    combined.attrs = dict(bundle.frame.attrs)
    combined.attrs["feature_columns"] = feature_columns
    combined.attrs["matched_filter"] = dict(matched.attrs)
    return FeatureBundle(combined, feature_columns, bundle.categorical_columns)


__all__ = [
    "MATCHED_FILTER_FEATURES",
    "MatchedFilterConfig",
    "append_matched_filter_features",
    "build_matched_filter_features",
]
