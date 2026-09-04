"""Temperature-salinity consistency gate for the fixed matched-filter scores.

Synthetic P1 anomalies modify temperature, while natural water-mass changes
often move temperature and salinity together.  This module preserves the
temperature interval proposals from :mod:`p1_qc.matched_filter` and softly
divides their scores by salinity evidence measured on the same boundaries.
Missing salinity never becomes a veto: its evidence is zero and the original
temperature score is retained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import segment_timeseries
from .features import FeatureBundle
from .matched_filter import (
    MatchedFilterConfig,
    _drift_intervals,
    _Interval,
    _local_difference_scale,
    _offset_intervals,
    _project_intervals,
)

TS_MATCHED_FILTER_FEATURES = (
    "ts_offset_pair_score_8_86h",
    "ts_offset_pair_duration_rows",
    "ts_drift_reset_score_9_86h",
    "ts_drift_reset_duration_rows",
)


def _offset_psal_evidence(
    psal: np.ndarray,
    scale: np.ndarray,
    interval: _Interval,
) -> float:
    start, stop = interval.start, interval.stop
    required = psal[[start - 1, start, stop - 1, stop]]
    if start < 1 or not np.isfinite(required).all():
        return 0.0
    entry = (psal[start] - psal[start - 1]) / scale[start]
    exit_change = (psal[stop] - psal[stop - 1]) / scale[stop]
    return float(min(abs(entry), abs(exit_change))) if entry * exit_change < 0 else 0.0


def _drift_psal_evidence(
    psal: np.ndarray,
    scale: np.ndarray,
    interval: _Interval,
) -> float:
    start, stop = interval.start, interval.stop
    required = psal[[start, stop - 1, stop]]
    if not np.isfinite(required).all():
        return 0.0
    pair_scale = max(float(scale[start]), float(scale[stop]), 1e-6)
    total_change = float(psal[stop - 1] - psal[start]) / pair_scale
    exit_change = float(psal[stop] - psal[stop - 1]) / pair_scale
    return (
        float(min(abs(total_change), abs(exit_change))) if total_change * exit_change < 0 else 0.0
    )


def _soft_gate(interval: _Interval, evidence: float) -> _Interval:
    return _Interval(
        interval.start,
        interval.stop,
        interval.score / (1.0 + max(evidence, 0.0)),
        interval.duration,
    )


def build_ts_matched_filter_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build four label-blind, gap-aware temperature-only evidence features."""

    required = {"station", "layer", "time", "temp", "psal"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing temperature-salinity matched-filter columns: {missing}")
    config = MatchedFilterConfig()
    source = frame.loc[:, ["station", "layer", "time", "temp", "psal"]].copy()
    segmented = segment_timeseries(
        source,
        group_columns=("station", "layer"),
        cadence_minutes=config.cadence_minutes,
    )
    segmented["__original_position"] = np.arange(len(segmented), dtype=np.int64)
    ordered = segmented.sort_values(
        ["station", "layer", "parsed_time", "__original_position"], kind="mergesort"
    ).reset_index(drop=True)
    output = np.zeros((len(ordered), 4), dtype=np.float64)

    for raw_positions in ordered.groupby("segment_id", sort=False, observed=True).indices.values():
        positions = np.asarray(raw_positions, dtype=np.int64)
        temp = pd.to_numeric(ordered.iloc[positions]["temp"], errors="coerce").to_numpy(float)
        psal = pd.to_numeric(ordered.iloc[positions]["psal"], errors="coerce").to_numpy(float)
        if len(temp) <= config.offset_min_rows or not np.isfinite(temp).all():
            continue
        temp_scale = _local_difference_scale(temp, config)
        psal_scale = _local_difference_scale(psal, config)
        offsets = [
            _soft_gate(interval, _offset_psal_evidence(psal, psal_scale, interval))
            for interval in _offset_intervals(temp, temp_scale, config)
        ]
        drifts = [
            _soft_gate(interval, _drift_psal_evidence(psal, psal_scale, interval))
            for interval in _drift_intervals(temp, temp_scale, config)
        ]
        offset_score, offset_duration = _project_intervals(offsets, len(temp))
        drift_score, drift_duration = _project_intervals(drifts, len(temp))
        output[positions] = np.column_stack(
            [offset_score, offset_duration, drift_score, drift_duration]
        )

    result = pd.DataFrame(output, columns=TS_MATCHED_FILTER_FEATURES)
    result["__original_position"] = ordered["__original_position"].to_numpy()
    result = result.sort_values("__original_position", kind="mergesort").drop(
        columns="__original_position"
    )
    result.index = frame.index.copy()
    for column in TS_MATCHED_FILTER_FEATURES:
        result[column] = result[column].astype(np.float32)
    result.attrs.update(
        {
            "feature_columns": TS_MATCHED_FILTER_FEATURES,
            "feature_mode": "offline",
            "cadence_minutes": config.cadence_minutes,
            "maximum_dependency_rows": config.maximum_rows,
            "scale_window_rows": config.scale_window_rows,
            "label_blind": True,
            "missing_psal_policy": "retain_temperature_score",
            "salinity_gate": "temperature_score/(1+same_boundary_psal_score)",
        }
    )
    return result


def append_ts_matched_filter_features(bundle: FeatureBundle, source: pd.DataFrame) -> FeatureBundle:
    if len(bundle.frame) != len(source) or not bundle.frame.index.equals(source.index):
        raise ValueError("source and feature bundle must have identical rows and index")
    duplicates = sorted(set(bundle.frame.columns).intersection(TS_MATCHED_FILTER_FEATURES))
    if duplicates:
        raise ValueError(f"temperature-salinity features already present: {duplicates}")
    features = build_ts_matched_filter_features(source)
    combined = bundle.frame.copy()
    for column in TS_MATCHED_FILTER_FEATURES:
        combined[column] = features[column].to_numpy(dtype=np.float32, copy=True)
    feature_columns = (*bundle.feature_columns, *TS_MATCHED_FILTER_FEATURES)
    combined.attrs = dict(bundle.frame.attrs)
    combined.attrs["feature_columns"] = feature_columns
    combined.attrs["ts_matched_filter"] = dict(features.attrs)
    return FeatureBundle(combined, feature_columns, bundle.categorical_columns)


__all__ = [
    "TS_MATCHED_FILTER_FEATURES",
    "append_ts_matched_filter_features",
    "build_ts_matched_filter_features",
]
