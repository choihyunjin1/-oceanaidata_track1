"""Causal public-layer leave-one-out residual correction for P2.

This module never reads target-layer temperature or salinity while constructing
the correction signal.  Every public layer is jointly masked in temperature and
salinity, reconstructed from the remaining public temperatures, and summarized
with a gap-reset causal 24-hour median.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from p2_restore.dynamic_sigmoid_profile import effective_depth
from p2_restore.features import PUBLIC_LAYERS, _nearest_public_baseline, _wide
from p2_restore.profile_projection import ProjectionResult, project_profiles_vectorized


@dataclass(frozen=True)
class CausalResidualSpec:
    public_layers: tuple[int, ...] = PUBLIC_LAYERS
    rolling_hours: int = 24
    cadence_minutes: int = 10
    minimum_samples: int = 72
    residual_clip_c: float = 0.5
    minimum_anchors: int = 4
    ridge_slope_lambda: float = 10.0
    correction_scale: float = 0.25
    correction_clip_c: float = 0.125
    maximum_anchor_span_c: float = 0.5
    depth_scale_m: float = 50.0
    nonzero_epsilon: float = 1e-12


@dataclass(frozen=True)
class CorrectionResult:
    correction: np.ndarray
    supported_mask: np.ndarray
    diagnostics: dict[str, float | int]


DEFAULT_SPEC = CausalResidualSpec()


def _causal_gap_reset_median(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    *,
    rolling_hours: int,
    cadence_minutes: int,
    minimum_samples: int,
) -> np.ndarray:
    """Return ``(t-window, t]`` medians, restarting after every cadence gap."""

    index = pd.DatetimeIndex(times)
    source = np.asarray(values, dtype=np.float64)
    if len(index) != len(source) or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("causal residual index must be unique, sorted, and aligned")
    expected = pd.Timedelta(minutes=cadence_minutes)
    segment = np.zeros(len(index), dtype=np.int64)
    if len(index) > 1:
        segment[1:] = np.cumsum((index[1:] - index[:-1]) != expected)
    result = np.full(len(source), np.nan, dtype=np.float64)
    for identifier in np.unique(segment):
        selected = segment == identifier
        series = pd.Series(source[selected], index=index[selected])
        result[selected] = series.rolling(
            f"{rolling_hours}h",
            min_periods=minimum_samples,
            closed="right",
        ).median().to_numpy(dtype=np.float64)
    return result


def build_public_residual_state(
    observations: pd.DataFrame,
    spec: CausalResidualSpec = DEFAULT_SPEC,
) -> pd.DataFrame:
    """Build label-blind causal residual anchors for all observation timestamps."""

    required = {"time", "layer", "temp", "psal", "depth", "nominal_depth"}
    if missing := required.difference(observations.columns):
        raise ValueError(f"observations missing columns: {sorted(missing)}")

    public = observations.loc[observations["layer"].isin(spec.public_layers), list(required)].copy()
    temp = _wide(public, "temp").reindex(columns=spec.public_layers)
    psal = _wide(public, "psal").reindex(index=temp.index, columns=spec.public_layers)
    depth = _wide(public, "depth").reindex(index=temp.index, columns=spec.public_layers)
    nominal = _wide(public, "nominal_depth").reindex(
        index=temp.index, columns=spec.public_layers
    )
    times = pd.DatetimeIndex(pd.to_datetime(temp.index, utc=True))
    order = np.argsort(times.asi8)
    times = times[order]
    temp_values = temp.to_numpy(dtype=np.float64)[order]
    psal_values = psal.to_numpy(dtype=np.float64)[order]
    depth_values = effective_depth(
        depth.to_numpy(dtype=np.float64)[order], nominal.to_numpy(dtype=np.float64)[order]
    )
    state = pd.DataFrame({"time": times})

    for position, layer in enumerate(spec.public_layers):
        masked_temp = temp_values.copy()
        masked_psal = psal_values.copy()
        masked_temp[:, position] = np.nan
        masked_psal[:, position] = np.nan
        if np.isfinite(masked_temp[:, position]).any() or np.isfinite(
            masked_psal[:, position]
        ).any():
            raise AssertionError("joint public-layer mask failed")
        reconstructed = _nearest_public_baseline(
            masked_temp,
            depth_values,
            depth_values[:, position],
        )
        residual = temp_values[:, position] - reconstructed
        residual = np.clip(residual, -spec.residual_clip_c, spec.residual_clip_c)
        state[f"median_residual_{layer}"] = _causal_gap_reset_median(
            times,
            residual,
            rolling_hours=spec.rolling_hours,
            cadence_minutes=spec.cadence_minutes,
            minimum_samples=spec.minimum_samples,
        )
        state[f"depth_{layer}"] = depth_values[:, position]

    if state.duplicated("time").any():
        raise AssertionError("public residual state contains duplicate timestamps")
    return state


def correction_for_rows(
    frame: pd.DataFrame,
    target_depth_m: np.ndarray,
    state: pd.DataFrame,
    spec: CausalResidualSpec = DEFAULT_SPEC,
) -> CorrectionResult:
    """Evaluate the fixed ridge-affine correction at aligned target depths."""

    if "time" not in frame:
        raise ValueError("prediction frame is missing time")
    target_depth = np.asarray(target_depth_m, dtype=np.float64)
    if target_depth.shape != (len(frame),):
        raise ValueError("target depth vector is not aligned")
    keyed = pd.DataFrame(
        {
            "time": pd.to_datetime(frame["time"], utc=True),
            "_row": np.arange(len(frame), dtype=np.int64),
        }
    )
    aligned = keyed.merge(state, on="time", how="left", validate="many_to_one").sort_values(
        "_row"
    )
    residual = aligned[
        [f"median_residual_{layer}" for layer in spec.public_layers]
    ].to_numpy(dtype=np.float64)
    depth = aligned[[f"depth_{layer}" for layer in spec.public_layers]].to_numpy(
        dtype=np.float64
    )
    valid = np.isfinite(residual) & np.isfinite(depth) & (depth > 0.0)
    count = valid.sum(axis=1)
    safe_residual = np.where(valid, residual, 0.0)
    scaled_depth = depth / spec.depth_scale_m
    safe_depth = np.where(valid, scaled_depth, 0.0)
    denominator = np.maximum(count, 1)
    mean_residual = safe_residual.sum(axis=1) / denominator
    mean_depth = safe_depth.sum(axis=1) / denominator
    centered_residual = np.where(valid, residual - mean_residual[:, None], 0.0)
    centered_depth = np.where(valid, scaled_depth - mean_depth[:, None], 0.0)
    slope = (centered_depth * centered_residual).sum(axis=1) / (
        (centered_depth**2).sum(axis=1) + spec.ridge_slope_lambda
    )
    intercept = mean_residual - slope * mean_depth

    finite_for_min = np.where(valid, residual, np.inf)
    finite_for_max = np.where(valid, residual, -np.inf)
    span = finite_for_max.max(axis=1) - finite_for_min.min(axis=1)
    layer_positions = {layer: position for position, layer in enumerate(spec.public_layers)}
    first = residual[:, layer_positions[1]]
    fifth = residual[:, layer_positions[5]]
    public_median = np.full(len(residual), np.nan, dtype=np.float64)
    populated = count > 0
    public_median[populated] = np.nanmedian(residual[populated], axis=1)
    sign_first = np.sign(first)
    sign_fifth = np.sign(fifth)
    sign_median = np.sign(public_median)
    coherent_sign = (
        np.isfinite(first)
        & np.isfinite(fifth)
        & np.isfinite(public_median)
        & (np.abs(first) > spec.nonzero_epsilon)
        & (np.abs(fifth) > spec.nonzero_epsilon)
        & (np.abs(public_median) > spec.nonzero_epsilon)
        & (sign_first == sign_fifth)
        & (sign_first == sign_median)
    )
    supported = (
        (count >= spec.minimum_anchors)
        & valid[:, layer_positions[1]]
        & valid[:, layer_positions[5]]
        & coherent_sign
        & np.isfinite(span)
        & (span <= spec.maximum_anchor_span_c)
        & np.isfinite(target_depth)
        & (target_depth > 0.0)
    )
    fitted = intercept + slope * (target_depth / spec.depth_scale_m)
    correction = np.zeros(len(frame), dtype=np.float64)
    correction[supported] = np.clip(
        spec.correction_scale * fitted[supported],
        -spec.correction_clip_c,
        spec.correction_clip_c,
    )
    diagnostics: dict[str, float | int] = {
        "rows": int(len(frame)),
        "supported_rows": int(supported.sum()),
        "supported_share": float(supported.mean()) if len(supported) else 0.0,
        "nonzero_rows": int((np.abs(correction) > spec.nonzero_epsilon).sum()),
        "maximum_absolute_correction_c": float(np.max(np.abs(correction), initial=0.0)),
        "minimum_anchor_count": int(count.min(initial=len(spec.public_layers))),
        "maximum_anchor_count": int(count.max(initial=0)),
    }
    return CorrectionResult(correction=correction, supported_mask=supported, diagnostics=diagnostics)


def apply_correction_and_projection(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    correction: np.ndarray,
    endpoints: pd.DataFrame,
) -> ProjectionResult:
    """Apply the bounded correction and the frozen physical profile projection."""

    base = np.asarray(prediction, dtype=np.float64)
    adjustment = np.asarray(correction, dtype=np.float64)
    if base.shape != (len(frame),) or adjustment.shape != base.shape:
        raise ValueError("prediction/correction vectors are not aligned")
    if not np.isfinite(base).all() or not np.isfinite(adjustment).all():
        raise ValueError("prediction/correction vectors must be finite")
    return project_profiles_vectorized(frame, base + adjustment, endpoints)
