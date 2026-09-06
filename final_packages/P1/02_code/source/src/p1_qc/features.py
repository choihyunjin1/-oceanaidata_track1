"""Gap-aware temporal and cross-layer features for P1.

Offline mode may use observations on both sides of a target timestamp.  Causal
mode uses only the current and earlier rows in the same exact-cadence segment.
No rolling operation crosses a missing-observation gap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import FeatureConfig, P1QCConfig
from .data import add_depth_regime, segment_timeseries


@dataclass(frozen=True)
class FeatureBundle:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        categorical = set(self.categorical_columns)
        return tuple(column for column in self.feature_columns if column not in categorical)

    def numeric_matrix(self, *, dtype: str | np.dtype = np.float32) -> np.ndarray:
        return self.frame.loc[:, self.numeric_columns].to_numpy(dtype=dtype, copy=True)


def _resolved_feature_config(
    config: FeatureConfig | P1QCConfig | None,
    mode: str | None,
) -> FeatureConfig:
    if config is None:
        resolved = FeatureConfig()
    elif isinstance(config, P1QCConfig):
        resolved = config.features
    elif isinstance(config, FeatureConfig):
        resolved = config
    else:
        raise TypeError("config must be FeatureConfig, P1QCConfig, or None")
    if mode is not None:
        resolved = replace(resolved, mode=mode.lower())
    if resolved.mode not in {"offline", "causal"}:
        raise ValueError("feature mode must be 'offline' or 'causal'")
    return resolved


def _rolling(
    values: pd.Series,
    segments: pd.Series,
    *,
    window: int,
    min_periods: int,
    statistic: str,
    center: bool,
) -> pd.Series:
    rolled = values.groupby(segments, sort=False, observed=True).rolling(
        window=window,
        min_periods=min_periods,
        center=center,
    )
    if statistic == "median":
        result = rolled.median()
    elif statistic == "mean":
        result = rolled.mean()
    elif statistic == "std":
        result = rolled.std(ddof=1)
    elif statistic == "min":
        result = rolled.min()
    elif statistic == "max":
        result = rolled.max()
    else:
        raise ValueError(f"unsupported rolling statistic: {statistic}")
    return result.reset_index(level=0, drop=True).sort_index()


def _window_rows(hours: int, cadence_minutes: int) -> int:
    return max(2, int(round(hours * 60 / cadence_minutes)))


def _depth_metadata(
    ordered: pd.DataFrame,
    *,
    config: FeatureConfig,
    mode: str,
) -> tuple[pd.Series, pd.Series]:
    width = config.depth_regime_width_m
    if mode == "offline":
        enriched = add_depth_regime(ordered, width_m=width)
        return enriched["nominal_depth_m"], enriched["depth_regime"]
    # Causal metadata cannot use a future group median.  The current pressure
    # observation is rounded directly; missing depth has an explicit fallback.
    nominal = (ordered["depth"] / width).round() * width
    station = ordered["station"].astype("string")
    layer = ordered["layer"].astype("string")
    regime = (
        station
        + "|"
        + nominal.map(lambda value: f"d{value:06.1f}" if pd.notna(value) else "").astype("string")
    )
    regime = regime.mask(nominal.isna(), station + "|unknown|l" + layer)
    return nominal.astype(float), regime.astype("string")


def build_features(
    frame: pd.DataFrame,
    *,
    config: FeatureConfig | P1QCConfig | None = None,
    mode: str | None = None,
    cadence_minutes: int = 10,
    group_columns: Sequence[str] = ("station", "layer"),
) -> FeatureBundle:
    """Create an aligned feature bundle without modifying ``frame``.

    Keys, ``label`` and ``anomaly_type`` are never included in
    ``feature_columns``.  The returned frame has exactly the input index and
    contains only model features, with categorical columns explicitly listed.
    """

    cfg = _resolved_feature_config(config, mode)
    required = {"station", "year", "layer", "time", "temp", "psal", "depth"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing feature source columns: {missing}")
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be positive")

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
    segment_group = ordered.groupby("segment_id", sort=False, observed=True)
    center = cfg.mode == "offline"

    out = pd.DataFrame(index=ordered.index)
    categorical = ("station", "layer_category", "depth_regime")
    out["station"] = ordered["station"].astype("string")
    out["layer_category"] = ordered["layer"].astype("string")
    nominal_depth, depth_regime = _depth_metadata(ordered, config=cfg, mode=cfg.mode)
    out["depth_regime"] = depth_regime

    # Raw reference values remain available to tree models; local-normalised
    # features below should carry most of the anomaly signal under year shift.
    out["temp_raw"] = pd.to_numeric(ordered["temp"], errors="coerce")
    out["psal_raw"] = pd.to_numeric(ordered["psal"], errors="coerce")
    out["depth_raw"] = pd.to_numeric(ordered["depth"], errors="coerce")
    out["nominal_depth_m"] = nominal_depth
    out["psal_missing"] = ordered["psal"].isna().astype(np.int8)
    out["depth_missing"] = ordered["depth"].isna().astype(np.int8)
    out["has_gap_before"] = (~ordered["is_contiguous"]).astype(np.int8)

    local_time = ordered["parsed_time"].dt.tz_convert("Asia/Seoul")
    day_phase = 2 * np.pi * (local_time.dt.dayofyear - 1) / 365.2425
    hour_value = local_time.dt.hour + local_time.dt.minute / 60.0
    hour_phase = 2 * np.pi * hour_value / 24.0
    out["day_sin"] = np.sin(day_phase)
    out["day_cos"] = np.cos(day_phase)
    out["hour_sin"] = np.sin(hour_phase)
    out["hour_cos"] = np.cos(hour_phase)

    temp = pd.to_numeric(ordered["temp"], errors="coerce")
    previous = segment_group["temp"].shift(1)
    previous_two = segment_group["temp"].shift(2)
    next_value = segment_group["temp"].shift(-1)
    first_difference = temp - previous
    out["temp_diff_1"] = first_difference
    out["temp_abs_diff_1"] = first_difference.abs()
    out["temp_backward_acceleration"] = (temp - previous) - (previous - previous_two)

    for source_column in ("psal", "depth"):
        numeric = pd.to_numeric(ordered[source_column], errors="coerce")
        prior = numeric.groupby(segment, sort=False, observed=True).shift(1)
        out[f"{source_column}_diff_1"] = numeric - prior
        out[f"{source_column}_abs_diff_1"] = (numeric - prior).abs()

    if cfg.mode == "offline":
        difference_next = next_value - temp
        out["temp_diff_next"] = difference_next
        out["spike_min_abs_diff"] = np.minimum(first_difference.abs(), difference_next.abs())
        out["temp_center_curvature"] = (temp - (previous + next_value) / 2.0).abs()

    plateau_values = (
        temp if cfg.plateau_round_decimals is None else temp.round(cfg.plateau_round_decimals)
    )
    plateau_previous = plateau_values.groupby(segment, sort=False, observed=True).shift(1)
    same_as_previous = plateau_values.eq(plateau_previous)
    plateau_local_id = (~same_as_previous).groupby(segment, sort=False).cumsum()
    plateau_groups = [segment, plateau_local_id]
    plateau_elapsed = same_as_previous.groupby(plateau_groups, sort=False).cumsum() + 1
    out["plateau_elapsed"] = plateau_elapsed.astype(float)
    if cfg.mode == "offline":
        out["plateau_full_length"] = (
            plateau_values.groupby(plateau_groups, sort=False, observed=True)
            .transform("size")
            .astype(float)
        )
        out["plateau_count"] = out["plateau_full_length"]
    else:
        out["plateau_count"] = out["plateau_elapsed"]

    rolling_feature_names: list[str] = []
    for hours in cfg.rolling_hours:
        rows = _window_rows(hours, cadence_minutes)
        minimum = max(3, int(np.ceil(rows * cfg.min_period_fraction)))
        tag = f"{hours}h"
        baseline_source = (
            temp.groupby(segment, sort=False, observed=True).shift(1)
            if cfg.mode == "causal"
            else temp
        )
        median = _rolling(
            baseline_source,
            segment,
            window=rows,
            min_periods=minimum,
            statistic="median",
            center=center,
        )
        residual = temp - median
        out[f"temp_median_resid_{tag}"] = residual
        out[f"temp_abs_median_resid_{tag}"] = residual.abs()
        local_std = _rolling(
            temp,
            segment,
            window=rows,
            min_periods=minimum,
            statistic="std",
            center=center,
        )
        diff_std = _rolling(
            first_difference,
            segment,
            window=rows,
            min_periods=minimum,
            statistic="std",
            center=center,
        )
        out[f"temp_roll_std_{tag}"] = local_std
        out[f"diff_roll_std_{tag}"] = diff_std
        deviation_source = residual.abs()
        if cfg.mode == "causal":
            deviation_source = deviation_source.groupby(segment, sort=False, observed=True).shift(1)
        mad = _rolling(
            deviation_source,
            segment,
            window=rows,
            min_periods=minimum,
            statistic="median",
            center=center,
        )
        out[f"temp_robust_z_{tag}"] = residual / (1.4826 * mad + cfg.robust_epsilon)
        rolling_feature_names.extend(
            [
                f"temp_median_resid_{tag}",
                f"temp_abs_median_resid_{tag}",
                f"temp_roll_std_{tag}",
                f"diff_roll_std_{tag}",
                f"temp_robust_z_{tag}",
            ]
        )

    # Simultaneous same-station layers.  The mean excludes the target row; all
    # downstream peer features remain NA if no peer exists (e.g. G-ORS).
    station_time_group = ordered.groupby(["station", "parsed_time"], sort=False, observed=True)[
        "temp"
    ]
    simultaneous_count = station_time_group.transform("size")
    simultaneous_sum = station_time_group.transform("sum")
    peer_count = simultaneous_count - 1
    peer_mean = (simultaneous_sum - temp) / peer_count.replace(0, np.nan)
    peer_residual = temp - peer_mean
    out["peer_count"] = peer_count.astype(float)
    out["peer_available"] = peer_count.gt(0).astype(np.int8)
    out["peer_temp_mean"] = peer_mean
    out["temp_peer_residual"] = peer_residual
    out["temp_abs_peer_residual"] = peer_residual.abs()
    out["station_layer_temp_std"] = station_time_group.transform("std")

    for days in cfg.long_windows_days:
        hours = days * 24
        rows = _window_rows(hours, cadence_minutes)
        minimum = max(12, int(np.ceil(rows * cfg.min_period_fraction)))
        tag = f"{days}d"
        temporal_source = (
            temp.groupby(segment, sort=False, observed=True).shift(1)
            if cfg.mode == "causal"
            else temp
        )
        temporal_base = _rolling(
            temporal_source,
            segment,
            window=rows,
            min_periods=minimum,
            statistic="median",
            center=center,
        )
        temporal_residual = temp - temporal_base
        peer_source = (
            peer_residual.groupby(segment, sort=False, observed=True).shift(1)
            if cfg.mode == "causal"
            else peer_residual
        )
        peer_base = _rolling(
            peer_source,
            segment,
            window=rows,
            min_periods=minimum,
            statistic="median",
            center=center,
        )
        peer_detrended = peer_residual - peer_base
        out[f"temp_long_resid_{tag}"] = temporal_residual
        out[f"peer_detrended_resid_{tag}"] = peer_detrended
        out[f"peer_abs_detrended_resid_{tag}"] = peer_detrended.abs()
        reference = peer_detrended.where(peer_count.gt(0), temporal_residual)
        out[f"reference_resid_{tag}"] = reference
        out[f"reference_abs_resid_{tag}"] = reference.abs()

        # A robust slope proxy: change in the long-window residual over one
        # hour, kept inside the same exact-cadence segment.
        lag_rows = max(1, int(round(60 / cadence_minutes)))
        lagged_reference = reference.groupby(segment, sort=False, observed=True).shift(lag_rows)
        out[f"reference_slope_1h_{tag}"] = (reference - lagged_reference) / lag_rows

    # Restore the exact input row order/index and compact numeric storage.
    out["__original_position"] = ordered["__original_position"].to_numpy()
    out = out.sort_values("__original_position", kind="mergesort").drop(
        columns="__original_position"
    )
    out.index = frame.index.copy()
    for column in out.columns:
        if column not in categorical:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype(np.float32)
    feature_columns = tuple(str(column) for column in out.columns)
    out.attrs.update(
        {
            "feature_columns": feature_columns,
            "categorical_columns": categorical,
            "feature_mode": cfg.mode,
            "cadence_minutes": cadence_minutes,
        }
    )
    return FeatureBundle(out, feature_columns, categorical)


def build_feature_frame(*args, **kwargs) -> pd.DataFrame:
    """Compatibility helper returning only ``FeatureBundle.frame``."""

    return build_features(*args, **kwargs).frame


__all__ = ["FeatureBundle", "build_feature_frame", "build_features"]
