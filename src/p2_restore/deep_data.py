"""Leakage-safe dense panels and gap-aware chunks for P2 deep models.

Target-layer temperature and salinity never enter ``inputs``.  They are kept
only in the supervised target arrays, so an outer validation blackout is
simultaneous across layers 2, 3, and 4 by construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from p2_restore.features import PUBLIC_LAYERS, TARGET_LAYERS, _nearest_public_baseline

CADENCE_MINUTES = 10


@dataclass(frozen=True)
class P2Panel:
    times: pd.DatetimeIndex
    inputs: np.ndarray
    input_names: tuple[str, ...]
    baseline: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    segment_ids: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.times)
        if self.inputs.shape[0] != rows or self.inputs.shape[1] != len(self.input_names):
            raise ValueError("panel input shape is inconsistent")
        if self.baseline.shape != (rows, 3) or self.target.shape != (rows, 3):
            raise ValueError("panel target shape is inconsistent")
        if self.target_mask.shape != (rows, 3) or len(self.segment_ids) != rows:
            raise ValueError("panel mask/segment shape is inconsistent")
        forbidden = {f"temp_{layer}" for layer in TARGET_LAYERS} | {
            f"psal_{layer}" for layer in TARGET_LAYERS
        }
        if forbidden.intersection(self.input_names):
            raise ValueError("target-layer observation leaked into deep inputs")


@dataclass(frozen=True)
class PanelNormalizer:
    input_center: np.ndarray
    input_scale: np.ndarray
    residual_center: np.ndarray
    residual_scale: np.ndarray

    @classmethod
    def fit(cls, panel: P2Panel, train_times: np.ndarray) -> PanelNormalizer:
        selected = np.asarray(train_times, dtype=bool)
        if selected.shape != (len(panel.times),) or not selected.any():
            raise ValueError("normalizer train-time mask is invalid")
        values = panel.inputs[selected]
        center = np.nanmedian(values, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.nanmedian(np.abs(values - center), axis=0) * 1.4826
        fallback = np.nanstd(values, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)

        residual = panel.target - panel.baseline
        residual_center = np.zeros(3, dtype=np.float64)
        residual_scale = np.ones(3, dtype=np.float64)
        for layer in range(3):
            keep = selected & panel.target_mask[:, layer] & np.isfinite(residual[:, layer])
            if not keep.any():
                raise ValueError(f"target layer {layer + 2} has no fold-local training rows")
            current = residual[keep, layer]
            residual_center[layer] = np.median(current)
            robust = np.median(np.abs(current - residual_center[layer])) * 1.4826
            residual_scale[layer] = (
                robust if np.isfinite(robust) and robust > 1e-4 else np.std(current)
            )
            if not np.isfinite(residual_scale[layer]) or residual_scale[layer] <= 1e-4:
                residual_scale[layer] = 1.0
        return cls(center, scale, residual_center, residual_scale)

    def transform_inputs(self, inputs: np.ndarray) -> np.ndarray:
        transformed = (np.asarray(inputs, dtype=np.float64) - self.input_center) / self.input_scale
        transformed = np.where(np.isfinite(transformed), transformed, 0.0)
        return np.clip(transformed, -12.0, 12.0).astype(np.float32)

    def transform_targets(self, panel: P2Panel) -> tuple[np.ndarray, np.ndarray]:
        residual = panel.target - panel.baseline
        transformed = (residual - self.residual_center) / self.residual_scale
        mask = panel.target_mask & np.isfinite(transformed) & np.isfinite(panel.baseline)
        return np.where(mask, transformed, 0.0).astype(np.float32), mask

    def inverse_predictions(self, panel: P2Panel, prediction: np.ndarray) -> np.ndarray:
        residual = (
            np.asarray(prediction, dtype=np.float64) * self.residual_scale + self.residual_center
        )
        return np.clip(panel.baseline + residual, -5.0, 45.0)


def _wide(observations: pd.DataFrame, value: str, times: pd.DatetimeIndex) -> pd.DataFrame:
    keyed = observations.assign(_time=pd.to_datetime(observations["time"], utc=True))
    return keyed.pivot(index="_time", columns="layer", values=value).reindex(times)


def build_panel(observations: pd.DataFrame) -> P2Panel:
    """Build one dense public-covariate row per observed timestamp."""

    times = pd.DatetimeIndex(
        pd.to_datetime(observations["time"], utc=True).drop_duplicates()
    ).sort_values()
    temp = _wide(observations, "temp", times)
    psal = _wide(observations, "psal", times)
    depth = _wide(observations, "depth", times)
    nominal = _wide(observations, "nominal_depth", times)

    values: list[np.ndarray] = []
    names: list[str] = []
    for layer in PUBLIC_LAYERS:
        for prefix, wide in (("temp", temp), ("psal", psal), ("depth", depth)):
            current = wide[layer].to_numpy(dtype=np.float64)
            values.extend((current, np.isfinite(current).astype(np.float64)))
            names.extend((f"public_{prefix}_{layer}", f"public_{prefix}_{layer}_mask"))

    public_temp = np.column_stack([temp[layer].to_numpy(float) for layer in PUBLIC_LAYERS])
    public_nominal = np.column_stack([nominal[layer].to_numpy(float) for layer in PUBLIC_LAYERS])
    target_depth = np.column_stack([nominal[layer].to_numpy(float) for layer in TARGET_LAYERS])
    public_count = np.isfinite(public_temp).sum(axis=1)
    public_mean = np.divide(
        np.nansum(public_temp, axis=1),
        public_count,
        out=np.full(len(public_temp), np.nan),
        where=public_count > 0,
    )
    public_variance = np.divide(
        np.nansum((public_temp - public_mean[:, None]) ** 2, axis=1),
        public_count,
        out=np.full(len(public_temp), np.nan),
        where=public_count > 0,
    )
    public_range = np.full(len(public_temp), np.nan)
    populated = public_count > 0
    public_range[populated] = np.nanmax(public_temp[populated], axis=1) - np.nanmin(
        public_temp[populated], axis=1
    )
    for column, label in (
        (public_mean, "public_temp_mean"),
        (np.sqrt(public_variance), "public_temp_std"),
        (public_range, "public_temp_range"),
        (public_temp[:, 0] - public_temp[:, 1], "public_temp_1_minus_5"),
    ):
        values.append(column)
        names.append(label)

    baseline = np.column_stack(
        [
            _nearest_public_baseline(public_temp, public_nominal, target_depth[:, offset])
            for offset in range(3)
        ]
    )
    for offset, layer in enumerate(TARGET_LAYERS):
        values.append(baseline[:, offset])
        names.append(f"baseline_{layer}")

    kst = times.tz_convert("Asia/Seoul")
    minute = kst.hour.to_numpy() * 60 + kst.minute.to_numpy()
    doy = kst.dayofyear.to_numpy() + minute / 1440.0
    epoch_seconds = times.as_unit("ns").asi8 / 1e9
    cyclic = {
        "doy_sin": np.sin(2 * np.pi * doy / 365.2425),
        "doy_cos": np.cos(2 * np.pi * doy / 365.2425),
        "hour_sin": np.sin(2 * np.pi * minute / 1440.0),
        "hour_cos": np.cos(2 * np.pi * minute / 1440.0),
        "m2_sin": np.sin(2 * np.pi * epoch_seconds / (12.42 * 3600.0)),
        "m2_cos": np.cos(2 * np.pi * epoch_seconds / (12.42 * 3600.0)),
    }
    for name, column in cyclic.items():
        names.append(name)
        values.append(column)

    target = np.column_stack([temp[layer].to_numpy(float) for layer in TARGET_LAYERS])
    target_mask = np.isfinite(target) & np.isfinite(baseline)
    delta = times.to_series().diff().dt.total_seconds().div(60).to_numpy()
    segment_ids = (
        np.cumsum(np.r_[True, ~np.isclose(delta[1:], CADENCE_MINUTES)]).astype(np.int32) - 1
    )
    inputs = np.column_stack(values).astype(np.float32)
    return P2Panel(times, inputs, tuple(names), baseline, target, target_mask, segment_ids)


def time_block_mask(panel: P2Panel, start: str, stop: str) -> np.ndarray:
    left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
    right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
    return np.asarray((panel.times >= left) & (panel.times < right), dtype=bool)


def iter_segment_bounds(segment_ids: np.ndarray) -> Iterator[tuple[int, int]]:
    values = np.asarray(segment_ids)
    if values.ndim != 1:
        raise ValueError("segment ids must be one dimensional")
    if len(values) == 0:
        return
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    yield from zip(starts.tolist(), stops.tolist(), strict=True)


def make_chunk_bounds(
    segment_ids: np.ndarray,
    *,
    length: int,
    stride: int,
) -> tuple[tuple[int, int], ...]:
    if length < 8 or stride < 1 or stride > length:
        raise ValueError("invalid chunk length/stride")
    result: list[tuple[int, int]] = []
    for start, stop in iter_segment_bounds(segment_ids):
        width = stop - start
        if width <= length:
            result.append((start, stop))
            continue
        positions = list(range(start, stop - length + 1, stride))
        if positions[-1] != stop - length:
            positions.append(stop - length)
        result.extend((position, position + length) for position in positions)
    return tuple(result)
