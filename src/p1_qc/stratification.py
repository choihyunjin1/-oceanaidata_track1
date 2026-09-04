"""Small, label-blind stratification gate for simultaneous peer features.

The base feature set exposes cross-layer residuals, but a large residual can be
normal when a thermocline or an internal wave decouples layers.  This module
adds one deliberately small ablation: trust the peer residual only in windows
where the target layer and its simultaneous peers have coherent *changes*.

No label, anomaly type, fitted statistic, calendar bin, or global threshold is
used.  Offline mode has a half-window future dependency; causal mode is prefix
invariant.  Both modes reset at exact-cadence segment boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import segment_timeseries
from .features import FeatureBundle


@dataclass(frozen=True)
class PeerGateConfig:
    """Pre-registered settings for the minimal peer-gate ablation."""

    mode: str = "offline"
    window_hours: int = 24
    min_period_fraction: float = 0.5


PEER_GATE_FEATURES = (
    "peer_change_corr_24h",
    "peer_pair_coverage_24h",
    "peer_trust_gate_24h",
    "temp_abs_peer_residual_gated_24h",
)


def _validate_config(config: PeerGateConfig, cadence_minutes: int) -> tuple[str, int, int]:
    mode = config.mode.lower()
    if mode not in {"offline", "causal"}:
        raise ValueError("peer gate mode must be 'offline' or 'causal'")
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be positive")
    if config.window_hours <= 0:
        raise ValueError("window_hours must be positive")
    if not 0 < config.min_period_fraction <= 1:
        raise ValueError("min_period_fraction must be in (0, 1]")
    window_rows = max(3, int(round(config.window_hours * 60 / cadence_minutes)))
    min_pairs = max(3, int(np.ceil(window_rows * config.min_period_fraction)))
    return mode, window_rows, min_pairs


def _rolling_pair_statistics(
    first: pd.Series,
    second: pd.Series,
    segments: pd.Series,
    *,
    window_rows: int,
    min_pairs: int,
    center: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rolling correlation and observed-pair coverage per segment."""

    correlation = np.full(len(first), np.nan, dtype=np.float64)
    coverage = np.zeros(len(first), dtype=np.float64)
    grouped_positions = segments.groupby(segments, sort=False, observed=True).indices
    for positions in grouped_positions.values():
        positions = np.asarray(positions, dtype=np.int64)
        left = pd.Series(first.iloc[positions].to_numpy(dtype=np.float64), copy=False)
        right = pd.Series(second.iloc[positions].to_numpy(dtype=np.float64), copy=False)
        paired = left.notna() & right.notna()
        rolling = left.rolling(window_rows, min_periods=min_pairs, center=center)
        local_correlation = rolling.corr(right)
        paired_count = paired.astype(float).rolling(window_rows, min_periods=1, center=center).sum()
        possible_count = (
            pd.Series(np.ones(len(positions), dtype=np.float64))
            .rolling(window_rows, min_periods=1, center=center)
            .sum()
        )
        correlation[positions] = local_correlation.to_numpy(dtype=np.float64)
        coverage[positions] = (paired_count / possible_count).to_numpy(dtype=np.float64)
    return correlation, coverage


def build_stratification_peer_gate(
    frame: pd.DataFrame,
    *,
    config: PeerGateConfig | None = None,
    cadence_minutes: int = 10,
    group_columns: Sequence[str] = ("station", "layer"),
) -> pd.DataFrame:
    """Build exactly four aligned, label-blind peer-gate features.

    The gate is ``max(rolling change correlation, 0) * pair coverage``.  It is
    intentionally a continuous feature rather than a hard classification rule.
    A missing or constant peer reference yields zero trust, while its gated
    residual remains missing when no peer exists.
    """

    cfg = PeerGateConfig() if config is None else config
    mode, window_rows, min_pairs = _validate_config(cfg, cadence_minutes)
    required = {"station", "layer", "time", "temp"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing peer-gate source columns: {missing}")

    segmented = segment_timeseries(
        frame,
        group_columns=group_columns,
        cadence_minutes=cadence_minutes,
    )
    segmented["__original_position"] = np.arange(len(segmented), dtype=np.int64)
    ordered = segmented.sort_values(
        [*group_columns, "parsed_time", "__original_position"], kind="mergesort"
    ).reset_index(drop=True)
    segment = ordered["segment_id"]
    temp = pd.to_numeric(ordered["temp"], errors="coerce")

    station_time = ordered.groupby(["station", "parsed_time"], sort=False, observed=True)["temp"]
    valid_count = station_time.transform("count")
    total = station_time.transform("sum")
    target_is_valid = temp.notna().astype(np.int64)
    peer_count = valid_count - target_is_valid
    peer_sum = total - temp.fillna(0.0)
    peer_mean = peer_sum / peer_count.replace(0, np.nan)

    target_change = temp.groupby(segment, sort=False, observed=True).diff()
    peer_change = peer_mean.groupby(segment, sort=False, observed=True).diff()
    previous_peer_count = peer_count.groupby(segment, sort=False, observed=True).shift(1)
    stable_peer_set = peer_count.gt(0) & peer_count.eq(previous_peer_count)
    target_change = target_change.where(stable_peer_set)
    peer_change = peer_change.where(stable_peer_set)

    correlation, coverage = _rolling_pair_statistics(
        target_change,
        peer_change,
        segment,
        window_rows=window_rows,
        min_pairs=min_pairs,
        center=mode == "offline",
    )
    trust = np.clip(np.nan_to_num(correlation, nan=0.0), 0.0, 1.0) * coverage
    absolute_residual = (temp - peer_mean).abs().to_numpy(dtype=np.float64)

    result = pd.DataFrame(
        {
            PEER_GATE_FEATURES[0]: correlation,
            PEER_GATE_FEATURES[1]: coverage,
            PEER_GATE_FEATURES[2]: trust,
            PEER_GATE_FEATURES[3]: absolute_residual * trust,
            "__original_position": ordered["__original_position"].to_numpy(),
        }
    )
    result = result.sort_values("__original_position", kind="mergesort").drop(
        columns="__original_position"
    )
    result.index = frame.index.copy()
    for column in PEER_GATE_FEATURES:
        result[column] = result[column].astype(np.float32)
    result.attrs.update(
        {
            "feature_columns": PEER_GATE_FEATURES,
            "feature_mode": mode,
            "window_hours": cfg.window_hours,
            "future_dependency_hours": cfg.window_hours / 2 if mode == "offline" else 0.0,
        }
    )
    return result


def append_stratification_peer_gate(
    bundle: FeatureBundle,
    source: pd.DataFrame,
    *,
    config: PeerGateConfig | None = None,
    cadence_minutes: int = 10,
    group_columns: Sequence[str] = ("station", "layer"),
) -> FeatureBundle:
    """Return a new feature bundle with the four ablation columns appended."""

    if len(bundle.frame) != len(source) or not bundle.frame.index.equals(source.index):
        raise ValueError("source and feature bundle must have identical rows and index")
    gate = build_stratification_peer_gate(
        source,
        config=config,
        cadence_minutes=cadence_minutes,
        group_columns=group_columns,
    )
    duplicates = sorted(set(bundle.frame.columns).intersection(gate.columns))
    if duplicates:
        raise ValueError(f"peer-gate features already present: {duplicates}")
    combined = pd.concat([bundle.frame.copy(), gate], axis=1)
    feature_columns = (*bundle.feature_columns, *PEER_GATE_FEATURES)
    combined.attrs = dict(bundle.frame.attrs)
    combined.attrs["feature_columns"] = feature_columns
    combined.attrs["peer_gate"] = dict(gate.attrs)
    return FeatureBundle(combined, feature_columns, bundle.categorical_columns)


__all__ = [
    "PEER_GATE_FEATURES",
    "PeerGateConfig",
    "append_stratification_peer_gate",
    "build_stratification_peer_gate",
]
