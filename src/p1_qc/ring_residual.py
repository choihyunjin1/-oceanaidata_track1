"""Fixed, gap-aware ring baselines for long offset and drift candidates.

The injected offset/drift maximum is 86.5 hours.  This deliberately fixed
offline feature excludes the target's entire +/-96-hour neighbourhood, then
uses only the two distant-candidate flanks ``[-168h, -96h)`` and
``(+96h, +168h]``.  The 96-hour exclusion is therefore wider than the
published maximum for an individual injected type and aims to reduce
self-contamination.  It does not prove that a flank is normal: composite or
adjacent events can form a longer positive run.  Every calculation remains
inside one exact-10-minute station/layer
segment; gaps are never bridged.

This module is label-blind and intentionally exposes no tunable arguments.
Each 72-hour flank contains 432 expected observations and requires the
pre-registered 50% coverage (216 finite observations).  A deficient flank is
``NaN`` rather than being filled from the other flank or another segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import segment_timeseries
from .features import FeatureBundle


@dataclass(frozen=True)
class RingResidualConfig:
    """Immutable pre-registered ring geometry; fields cannot be overridden."""

    cadence_minutes: int = field(default=10, init=False)
    exclusion_hours: int = field(default=96, init=False)
    outer_hours: int = field(default=168, init=False)
    min_flank_observations: int = field(default=216, init=False)

    @property
    def exclusion_rows(self) -> int:
        return self.exclusion_hours * 60 // self.cadence_minutes

    @property
    def outer_rows(self) -> int:
        return self.outer_hours * 60 // self.cadence_minutes

    @property
    def flank_rows(self) -> int:
        return self.outer_rows - self.exclusion_rows


RING_RESIDUAL_FEATURES = (
    "temp_ring_past_resid_96_168h",
    "temp_ring_future_resid_96_168h",
    "temp_ring_consensus_resid_96_168h",
    "temp_ring_flank_disagreement_96_168h",
)

_CONTRACT_ATTRS: dict[str, object] = {
    "feature_columns": RING_RESIDUAL_FEATURES,
    "feature_mode": "offline",
    "cadence_minutes": 10,
    "exclusion_hours": 96,
    "outer_hours": 168,
    "flank_rows": 432,
    "min_flank_observations": 216,
    "future_dependency_hours": 168,
}


def _segment_flank_medians(
    values: pd.Series,
    *,
    config: RingResidualConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact endpoint-aware past and future flank medians."""

    # At target i, shift(577) followed by a 432-row trailing window selects
    # i-1008..i-577: [-168h, -96h).  shift(-1008) selects i+577..i+1008:
    # (+96h, +168h].
    past = (
        values.shift(config.exclusion_rows + 1)
        .rolling(config.flank_rows, min_periods=config.min_flank_observations)
        .median()
    )
    future = (
        values.shift(-config.outer_rows)
        .rolling(config.flank_rows, min_periods=config.min_flank_observations)
        .median()
    )
    return past.to_numpy(dtype=np.float64), future.to_numpy(dtype=np.float64)


def build_ring_residual_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build exactly four aligned fixed-ring features from source columns only.

    The consensus baseline is the median of the two valid flank medians.  With
    two values this is their midpoint; both flanks are required.  ``label`` and
    ``anomaly_type`` are neither selected nor inspected, even if present in
    ``frame``.
    """

    config = RingResidualConfig()
    source_columns = ["station", "layer", "time", "temp"]
    missing = sorted(set(source_columns).difference(frame.columns))
    if missing:
        raise KeyError(f"missing ring-residual source columns: {missing}")

    # Selecting the four permitted source columns up front makes target-column
    # access structurally impossible inside this computation.
    source = frame.loc[:, source_columns].copy()
    segmented = segment_timeseries(
        source,
        group_columns=("station", "layer"),
        cadence_minutes=config.cadence_minutes,
    )
    segmented["__original_position"] = np.arange(len(segmented), dtype=np.int64)
    ordered = segmented.sort_values(
        ["station", "layer", "parsed_time", "__original_position"], kind="mergesort"
    ).reset_index(drop=True)

    temp = pd.to_numeric(ordered["temp"], errors="coerce")
    past_baseline = np.full(len(ordered), np.nan, dtype=np.float64)
    future_baseline = np.full(len(ordered), np.nan, dtype=np.float64)
    grouped_positions = ordered.groupby("segment_id", sort=False, observed=True).indices
    for positions in grouped_positions.values():
        positions = np.asarray(positions, dtype=np.int64)
        local = pd.Series(temp.iloc[positions].to_numpy(dtype=np.float64), copy=False)
        local_past, local_future = _segment_flank_medians(local, config=config)
        past_baseline[positions] = local_past
        future_baseline[positions] = local_future

    current = temp.to_numpy(dtype=np.float64)
    # Arithmetic mean is exactly the median of two flank medians.  Ordinary
    # NaN propagation enforces the both-flanks requirement for consensus.
    consensus_baseline = (past_baseline + future_baseline) / 2.0
    result = pd.DataFrame(
        {
            RING_RESIDUAL_FEATURES[0]: current - past_baseline,
            RING_RESIDUAL_FEATURES[1]: current - future_baseline,
            RING_RESIDUAL_FEATURES[2]: current - consensus_baseline,
            RING_RESIDUAL_FEATURES[3]: np.abs(past_baseline - future_baseline),
            "__original_position": ordered["__original_position"].to_numpy(),
        }
    )
    result = result.sort_values("__original_position", kind="mergesort").drop(
        columns="__original_position"
    )
    result.index = frame.index.copy()
    for column in RING_RESIDUAL_FEATURES:
        result[column] = result[column].astype(np.float32)
    result.attrs.update(_CONTRACT_ATTRS)
    return result


def summarize_ring_residual_coverage(features: pd.DataFrame) -> dict[str, int | float]:
    """Return conservative feature-availability counts under the fixed contract.

    Contract metadata is checked fail-closed so an audit cannot silently report
    coverage for a differently tuned window.  A row is counted as both-flank
    covered only when the consensus residual itself is available.
    """

    missing = sorted(set(RING_RESIDUAL_FEATURES).difference(features.columns))
    if missing:
        raise KeyError(f"missing ring-residual feature columns: {missing}")
    for key, expected in _CONTRACT_ATTRS.items():
        if features.attrs.get(key) != expected:
            raise ValueError(f"ring-residual contract mismatch for {key!r}")

    row_count = len(features)
    past_count = int(features[RING_RESIDUAL_FEATURES[0]].notna().sum())
    future_count = int(features[RING_RESIDUAL_FEATURES[1]].notna().sum())
    both_count = int(features[RING_RESIDUAL_FEATURES[2]].notna().sum())

    def fraction(count: int) -> float:
        return float(count / row_count) if row_count else 0.0

    return {
        "row_count": row_count,
        "past_covered_rows": past_count,
        "future_covered_rows": future_count,
        "both_flanks_covered_rows": both_count,
        "past_covered_fraction": fraction(past_count),
        "future_covered_fraction": fraction(future_count),
        "both_flanks_covered_fraction": fraction(both_count),
    }


def append_ring_residual_features(bundle: FeatureBundle, source: pd.DataFrame) -> FeatureBundle:
    """Return ``bundle`` with only the four fixed ring features appended."""

    if len(bundle.frame) != len(source) or not bundle.frame.index.equals(source.index):
        raise ValueError("source and feature bundle must have identical rows and index")
    ring = build_ring_residual_features(source)
    duplicates = sorted(set(bundle.frame.columns).intersection(RING_RESIDUAL_FEATURES))
    if duplicates:
        raise ValueError(f"ring-residual features already present: {duplicates}")

    combined = bundle.frame.copy()
    for column in RING_RESIDUAL_FEATURES:
        combined[column] = ring[column].to_numpy(dtype=np.float32, copy=True)
    feature_columns = (*bundle.feature_columns, *RING_RESIDUAL_FEATURES)
    combined.attrs = dict(bundle.frame.attrs)
    combined.attrs["feature_columns"] = feature_columns
    combined.attrs["ring_residual"] = dict(ring.attrs)
    return FeatureBundle(combined, feature_columns, bundle.categorical_columns)


__all__ = [
    "RING_RESIDUAL_FEATURES",
    "RingResidualConfig",
    "append_ring_residual_features",
    "build_ring_residual_features",
    "summarize_ring_residual_coverage",
]
