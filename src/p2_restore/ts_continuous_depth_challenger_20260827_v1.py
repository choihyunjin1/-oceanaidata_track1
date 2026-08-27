"""P2 continuous-depth temperature/salinity challenger components.

This module has no filesystem or submission entry point.  It builds model
inputs exclusively from the five public layers and exposes one matched model
whose only experimental switch is the weight of a density-gradient loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from p2_restore.deep_data import CADENCE_MINUTES, make_chunk_bounds
from p2_restore.features import PUBLIC_LAYERS, TARGET_LAYERS, _nearest_public_baseline


TARGET_DEPTH_METERS = np.asarray((7.04, 9.44, 14.74), dtype=np.float64)
GRAVITY = 9.80665
REFERENCE_DENSITY = 1025.0
THERMAL_EXPANSION = 2.0e-4
HALINE_CONTRACTION = 7.7e-4


@dataclass(frozen=True)
class ContinuousDepthPanel:
    """Dense public-input panel and held-out supervision surfaces."""

    times: pd.DatetimeIndex
    inputs: np.ndarray
    input_names: tuple[str, ...]
    query_depths: np.ndarray
    temperature_baseline: np.ndarray
    salinity_baseline: np.ndarray
    target_temperature: np.ndarray
    target_salinity: np.ndarray
    temperature_mask: np.ndarray
    salinity_mask: np.ndarray
    joint_mask: np.ndarray
    segment_ids: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.times)
        layers = len(TARGET_LAYERS)
        if self.inputs.shape != (rows, len(self.input_names)):
            raise ValueError("continuous-depth input shape changed")
        for value in (
            self.query_depths,
            self.temperature_baseline,
            self.salinity_baseline,
            self.target_temperature,
            self.target_salinity,
            self.temperature_mask,
            self.salinity_mask,
            self.joint_mask,
        ):
            if value.shape != (rows, layers):
                raise ValueError("continuous-depth target shape changed")
        if self.segment_ids.shape != (rows,):
            raise ValueError("continuous-depth segment shape changed")
        forbidden = {
            *(f"temp_{layer}" for layer in TARGET_LAYERS),
            *(f"psal_{layer}" for layer in TARGET_LAYERS),
            *(f"target_temp_{layer}" for layer in TARGET_LAYERS),
            *(f"target_psal_{layer}" for layer in TARGET_LAYERS),
        }
        if forbidden.intersection(self.input_names):
            raise ValueError("target-layer temperature/salinity leaked into model inputs")


def _wide(observations: pd.DataFrame, value: str, times: pd.DatetimeIndex) -> pd.DataFrame:
    keyed = observations.assign(_time=pd.to_datetime(observations["time"], utc=True))
    if keyed.duplicated(["_time", "layer"]).any():
        raise ValueError("duplicate time/layer observations are unsupported")
    return keyed.pivot(index="_time", columns="layer", values=value).reindex(times)


def build_continuous_depth_panel(observations: pd.DataFrame) -> ContinuousDepthPanel:
    """Build inputs from public-layer T/S/depth only.

    Target-layer T/S are copied solely to separate supervision arrays.  Query
    depths are fixed preregistered nominal depths, not target-row covariates.
    """

    required = {"layer", "time", "temp", "psal", "depth", "nominal_depth"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"observations are missing columns: {sorted(missing)}")
    times = pd.DatetimeIndex(
        pd.to_datetime(observations["time"], utc=True).drop_duplicates()
    ).sort_values()
    if not len(times) or times.has_duplicates:
        raise ValueError("timestamp surface is invalid")

    temp = _wide(observations, "temp", times)
    psal = _wide(observations, "psal", times)
    depth = _wide(observations, "depth", times)
    nominal = _wide(observations, "nominal_depth", times)
    missing_layers = set(PUBLIC_LAYERS + TARGET_LAYERS).difference(temp.columns)
    if missing_layers:
        raise ValueError(f"required layers missing: {sorted(missing_layers)}")

    values: list[np.ndarray] = []
    names: list[str] = []
    for layer in PUBLIC_LAYERS:
        for prefix, wide in (("temp", temp), ("psal", psal), ("depth", depth)):
            current = wide[layer].to_numpy(dtype=np.float64)
            values.extend((current, np.isfinite(current).astype(np.float64)))
            names.extend((f"public_{prefix}_{layer}", f"public_{prefix}_{layer}_mask"))

    public_temperature = np.column_stack(
        [temp[layer].to_numpy(dtype=np.float64) for layer in PUBLIC_LAYERS]
    )
    public_salinity = np.column_stack(
        [psal[layer].to_numpy(dtype=np.float64) for layer in PUBLIC_LAYERS]
    )
    public_nominal = np.column_stack(
        [nominal[layer].to_numpy(dtype=np.float64) for layer in PUBLIC_LAYERS]
    )
    query_depths = np.broadcast_to(TARGET_DEPTH_METERS, (len(times), len(TARGET_LAYERS))).copy()
    temperature_baseline = np.column_stack(
        [
            _nearest_public_baseline(public_temperature, public_nominal, query_depths[:, offset])
            for offset in range(len(TARGET_LAYERS))
        ]
    )
    salinity_baseline = np.column_stack(
        [
            _nearest_public_baseline(public_salinity, public_nominal, query_depths[:, offset])
            for offset in range(len(TARGET_LAYERS))
        ]
    )
    for offset, layer in enumerate(TARGET_LAYERS):
        for label, array in (
            ("temperature_baseline", temperature_baseline[:, offset]),
            ("salinity_baseline", salinity_baseline[:, offset]),
        ):
            values.extend((array, np.isfinite(array).astype(np.float64)))
            names.extend((f"public_{label}_{layer}", f"public_{label}_{layer}_mask"))

    public_temp_mean = np.nanmean(public_temperature, axis=1)
    public_psal_mean = np.nanmean(public_salinity, axis=1)
    public_temp_range = np.nanmax(public_temperature, axis=1) - np.nanmin(
        public_temperature, axis=1
    )
    public_psal_range = np.nanmax(public_salinity, axis=1) - np.nanmin(
        public_salinity, axis=1
    )
    for name, array in (
        ("public_temp_mean", public_temp_mean),
        ("public_psal_mean", public_psal_mean),
        ("public_temp_range", public_temp_range),
        ("public_psal_range", public_psal_range),
    ):
        values.extend((array, np.isfinite(array).astype(np.float64)))
        names.extend((name, f"{name}_mask"))

    kst = times.tz_convert("Asia/Seoul")
    minute = kst.hour.to_numpy() * 60 + kst.minute.to_numpy()
    day = kst.dayofyear.to_numpy() + minute / 1440.0
    seconds = times.as_unit("ns").asi8 / 1e9
    cyclic = {
        "doy_sin": np.sin(2 * np.pi * day / 365.2425),
        "doy_cos": np.cos(2 * np.pi * day / 365.2425),
        "hour_sin": np.sin(2 * np.pi * minute / 1440.0),
        "hour_cos": np.cos(2 * np.pi * minute / 1440.0),
        "m2_sin": np.sin(2 * np.pi * seconds / (12.42 * 3600.0)),
        "m2_cos": np.cos(2 * np.pi * seconds / (12.42 * 3600.0)),
    }
    for name, array in cyclic.items():
        values.append(array)
        names.append(name)

    target_temperature = np.column_stack(
        [temp[layer].to_numpy(dtype=np.float64) for layer in TARGET_LAYERS]
    )
    target_salinity = np.column_stack(
        [psal[layer].to_numpy(dtype=np.float64) for layer in TARGET_LAYERS]
    )
    temperature_mask = np.isfinite(target_temperature) & np.isfinite(temperature_baseline)
    salinity_mask = np.isfinite(target_salinity) & np.isfinite(salinity_baseline)
    joint_mask = temperature_mask & salinity_mask

    delta = times.to_series().diff().dt.total_seconds().div(60).to_numpy()
    segment_ids = (
        np.cumsum(np.r_[True, ~np.isclose(delta[1:], CADENCE_MINUTES)]).astype(np.int32) - 1
    )
    return ContinuousDepthPanel(
        times=times,
        inputs=np.column_stack(values).astype(np.float32),
        input_names=tuple(names),
        query_depths=query_depths.astype(np.float32),
        temperature_baseline=temperature_baseline,
        salinity_baseline=salinity_baseline,
        target_temperature=target_temperature,
        target_salinity=target_salinity,
        temperature_mask=temperature_mask,
        salinity_mask=salinity_mask,
        joint_mask=joint_mask,
        segment_ids=segment_ids,
    )


def _robust_center_scale(values: np.ndarray, *, floor: float) -> tuple[float, float]:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    if not len(current):
        raise ValueError("normalizer received no finite values")
    center = float(np.median(current))
    robust = float(np.median(np.abs(current - center)) * 1.4826)
    fallback = float(np.std(current))
    scale = robust if np.isfinite(robust) and robust > floor else fallback
    if not np.isfinite(scale) or scale <= floor:
        scale = 1.0
    return center, scale


def density_n2_proxy(temperature: np.ndarray, salinity: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Linearized seawater N2 proxy for adjacent depth queries.

    It intentionally does not claim full TEOS-10 fidelity.  The approximation
    is differentiable and is used identically for truth and predictions.
    """

    temperature_array = np.asarray(temperature, dtype=np.float64)
    salinity_array = np.asarray(salinity, dtype=np.float64)
    depth_array = np.asarray(depths, dtype=np.float64)
    density_anomaly = REFERENCE_DENSITY * (
        -THERMAL_EXPANSION * temperature_array + HALINE_CONTRACTION * salinity_array
    )
    depth_delta = np.diff(depth_array, axis=-1)
    density_delta = np.diff(density_anomaly, axis=-1)
    return (GRAVITY / REFERENCE_DENSITY) * density_delta / depth_delta


@dataclass(frozen=True)
class ContinuousDepthNormalizer:
    input_center: np.ndarray
    input_scale: np.ndarray
    target_center: np.ndarray
    target_scale: np.ndarray
    n2_scale: np.ndarray

    @classmethod
    def fit(
        cls,
        panel: ContinuousDepthPanel,
        selected_times: np.ndarray,
    ) -> "ContinuousDepthNormalizer":
        selected = np.asarray(selected_times, dtype=bool)
        if selected.shape != (len(panel.times),) or not selected.any():
            raise ValueError("normalizer selection is invalid")
        input_center = np.zeros(panel.inputs.shape[1], dtype=np.float64)
        input_scale = np.ones(panel.inputs.shape[1], dtype=np.float64)
        for column in range(panel.inputs.shape[1]):
            current = panel.inputs[selected, column]
            current = current[np.isfinite(current)]
            if not len(current):
                continue
            center, scale = _robust_center_scale(current, floor=1e-6)
            input_center[column] = center
            input_scale[column] = scale

        temperature_residual = panel.target_temperature - panel.temperature_baseline
        salinity_residual = panel.target_salinity - panel.salinity_baseline
        residuals = (temperature_residual, salinity_residual)
        masks = (panel.temperature_mask, panel.salinity_mask)
        target_center = np.zeros(2, dtype=np.float64)
        target_scale = np.ones(2, dtype=np.float64)
        for variable, (residual, mask) in enumerate(zip(residuals, masks, strict=True)):
            keep = selected[:, None] & mask
            center, scale = _robust_center_scale(residual[keep], floor=1e-6)
            target_center[variable] = center
            target_scale[variable] = scale

        n2 = density_n2_proxy(
            panel.target_temperature,
            panel.target_salinity,
            panel.query_depths,
        )
        pair_mask = (
            panel.joint_mask[:, 1:]
            & panel.joint_mask[:, :-1]
            & selected[:, None]
        )
        n2_scale = np.ones(len(TARGET_LAYERS) - 1, dtype=np.float64)
        for pair in range(len(TARGET_LAYERS) - 1):
            _, scale = _robust_center_scale(n2[:, pair][pair_mask[:, pair]], floor=1e-9)
            n2_scale[pair] = scale
        return cls(input_center, input_scale, target_center, target_scale, n2_scale)

    def transform_inputs(self, inputs: np.ndarray) -> np.ndarray:
        values = (np.asarray(inputs, dtype=np.float64) - self.input_center) / self.input_scale
        values = np.where(np.isfinite(values), values, 0.0)
        return np.clip(values, -12.0, 12.0).astype(np.float32)

    def transform_targets(self, panel: ContinuousDepthPanel) -> tuple[np.ndarray, np.ndarray]:
        residuals = np.stack(
            (
                panel.target_temperature - panel.temperature_baseline,
                panel.target_salinity - panel.salinity_baseline,
            ),
            axis=-1,
        )
        values = (residuals - self.target_center) / self.target_scale
        mask = np.stack((panel.temperature_mask, panel.salinity_mask), axis=-1)
        mask &= np.isfinite(values)
        return np.where(mask, values, 0.0).astype(np.float32), mask

    def inverse_outputs(
        self,
        panel: ContinuousDepthPanel,
        normalized_outputs: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(normalized_outputs, dtype=np.float64)
        expected = (len(panel.times), len(TARGET_LAYERS), 2)
        if values.shape != expected:
            raise ValueError(f"normalized prediction shape changed: {values.shape} != {expected}")
        baselines = np.stack(
            (panel.temperature_baseline, panel.salinity_baseline), axis=-1
        )
        result = baselines + values * self.target_scale + self.target_center
        result[:, :, 0] = np.clip(result[:, :, 0], -5.0, 45.0)
        result[:, :, 1] = np.clip(result[:, :, 1], 0.0, 50.0)
        return result


class GatedTemporalBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, *, dropout: float) -> None:
        super().__init__()
        kernel_size = 5
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden,
        )
        self.norm = nn.LayerNorm(hidden)
        self.expand = nn.Linear(hidden, hidden * 4)
        self.contract = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        state = self.depthwise(inputs.transpose(1, 2)).transpose(1, 2)
        left, gate = self.expand(self.norm(state)).chunk(2, dim=-1)
        update = self.contract(F.gelu(left) * torch.sigmoid(gate))
        return inputs + self.dropout(update)


class ContinuousDepthDualDecoder(nn.Module):
    """Decode arbitrary depth queries instead of a fixed three-row buffer."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(7, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def encode_depth(depths: Tensor) -> Tensor:
        scaled = depths / 50.0
        return torch.stack(
            (
                scaled,
                scaled.square(),
                torch.log1p(depths.clamp_min(0.0)) / math.log(51.0),
                torch.sin(2 * math.pi * scaled),
                torch.cos(2 * math.pi * scaled),
                torch.sin(4 * math.pi * scaled),
                torch.cos(4 * math.pi * scaled),
            ),
            dim=-1,
        )

    def forward(self, state: Tensor, query_depths: Tensor) -> Tensor:
        if query_depths.shape[:2] != state.shape[:2]:
            raise ValueError("query depth/time surface changed")
        query = self.query(self.encode_depth(query_depths))
        expanded = state.unsqueeze(2).expand(-1, -1, query_depths.shape[2], -1)
        return self.output(torch.cat((expanded, query), dim=-1))


class TSContinuousDepthTCN(nn.Module):
    """Shared temporal representation with continuous T/S depth queries."""

    def __init__(
        self,
        channels: int,
        *,
        hidden: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.blocks = nn.ModuleList(
            GatedTemporalBlock(hidden, dilation, dropout=dropout) for dilation in dilations
        )
        self.norm = nn.LayerNorm(hidden)
        self.decoder = ContinuousDepthDualDecoder(hidden)

    def forward(self, inputs: Tensor, query_depths: Tensor) -> Tensor:
        state = self.input(inputs)
        for block in self.blocks:
            state = block(state)
        return self.decoder(self.norm(state), query_depths)

    def loss_components(
        self,
        inputs: Tensor,
        query_depths: Tensor,
        baselines: Tensor,
        targets: Tensor,
        mask: Tensor,
        *,
        target_center: Tensor,
        target_scale: Tensor,
        n2_scale: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prediction = self(inputs, query_depths)
        if prediction.shape != targets.shape or mask.shape != targets.shape:
            raise ValueError("continuous-depth loss shapes changed")
        usable = mask.to(dtype=prediction.dtype)
        direct = ((prediction - targets).square() * usable).sum() / usable.sum().clamp_min(1.0)

        pair_mask = mask[:, :, 1:, :] & mask[:, :, :-1, :]
        predicted_difference = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_difference = targets[:, :, 1:, :] - targets[:, :, :-1, :]
        pair_weight = pair_mask.to(dtype=prediction.dtype)
        vertical = (
            (predicted_difference - target_difference).square() * pair_weight
        ).sum() / pair_weight.sum().clamp_min(1.0)

        physical_prediction = baselines + prediction * target_scale + target_center
        physical_target = baselines + targets * target_scale + target_center
        density_prediction = REFERENCE_DENSITY * (
            -THERMAL_EXPANSION * physical_prediction[..., 0]
            + HALINE_CONTRACTION * physical_prediction[..., 1]
        )
        density_target = REFERENCE_DENSITY * (
            -THERMAL_EXPANSION * physical_target[..., 0]
            + HALINE_CONTRACTION * physical_target[..., 1]
        )
        depth_delta = query_depths[:, :, 1:] - query_depths[:, :, :-1]
        predicted_n2 = (GRAVITY / REFERENCE_DENSITY) * torch.diff(
            density_prediction, dim=2
        ) / depth_delta
        target_n2 = (GRAVITY / REFERENCE_DENSITY) * torch.diff(
            density_target, dim=2
        ) / depth_delta
        joint = mask.all(dim=-1)
        density_mask = joint[:, :, 1:] & joint[:, :, :-1]
        density_weight = density_mask.to(dtype=prediction.dtype)
        scaled_error = (predicted_n2 - target_n2) / n2_scale
        density = (scaled_error.square() * density_weight).sum() / density_weight.sum().clamp_min(
            1.0
        )
        return direct, vertical, density


@dataclass(frozen=True)
class ChunkTensors:
    inputs: Tensor
    query_depths: Tensor
    baselines: Tensor
    targets: Tensor
    mask: Tensor
    bounds: tuple[tuple[int, int], ...]


def materialize_training_chunks(
    panel: ContinuousDepthPanel,
    normalizer: ContinuousDepthNormalizer,
    selected_times: np.ndarray,
    *,
    length: int = 512,
    stride: int = 384,
    minimum_target_values: int = 24,
) -> ChunkTensors:
    selected = np.asarray(selected_times, dtype=bool)
    if selected.shape != (len(panel.times),):
        raise ValueError("chunk selection shape changed")
    inputs = normalizer.transform_inputs(panel.inputs)
    targets, target_mask = normalizer.transform_targets(panel)
    target_mask &= selected[:, None, None]
    usable_time = target_mask.any(axis=(1, 2))
    bounds = tuple(
        bound
        for bound in make_chunk_bounds(panel.segment_ids, length=length, stride=stride)
        if int(target_mask[bound[0] : bound[1]].sum()) >= int(minimum_target_values)
        and usable_time[bound[0] : bound[1]].any()
    )
    if not bounds:
        raise RuntimeError("no continuous-depth training chunks are available")
    shape = (len(bounds), length)
    chunk_inputs = np.zeros((*shape, inputs.shape[1]), dtype=np.float32)
    chunk_depths = np.zeros((*shape, len(TARGET_LAYERS)), dtype=np.float32)
    chunk_baselines = np.zeros((*shape, len(TARGET_LAYERS), 2), dtype=np.float32)
    chunk_targets = np.zeros((*shape, len(TARGET_LAYERS), 2), dtype=np.float32)
    chunk_mask = np.zeros_like(chunk_targets, dtype=bool)
    baselines = np.stack((panel.temperature_baseline, panel.salinity_baseline), axis=-1)
    baselines = np.where(np.isfinite(baselines), baselines, 0.0).astype(np.float32)
    for index, (start, stop) in enumerate(bounds):
        width = stop - start
        chunk_inputs[index, :width] = inputs[start:stop]
        chunk_depths[index, :width] = panel.query_depths[start:stop]
        chunk_baselines[index, :width] = baselines[start:stop]
        chunk_targets[index, :width] = targets[start:stop]
        chunk_mask[index, :width] = target_mask[start:stop]
    return ChunkTensors(
        inputs=torch.from_numpy(chunk_inputs),
        query_depths=torch.from_numpy(chunk_depths),
        baselines=torch.from_numpy(chunk_baselines),
        targets=torch.from_numpy(chunk_targets),
        mask=torch.from_numpy(chunk_mask),
        bounds=bounds,
    )


@torch.inference_mode()
def predict_panel(
    model: TSContinuousDepthTCN,
    panel: ContinuousDepthPanel,
    normalizer: ContinuousDepthNormalizer,
    *,
    device: torch.device,
    length: int = 512,
    stride: int = 384,
    batch_size: int = 16,
) -> np.ndarray:
    model.eval()
    normalized_inputs = normalizer.transform_inputs(panel.inputs)
    bounds = make_chunk_bounds(panel.segment_ids, length=length, stride=stride)
    sums = np.zeros((len(panel.times), len(TARGET_LAYERS), 2), dtype=np.float64)
    counts = np.zeros(len(panel.times), dtype=np.int32)
    for batch_start in range(0, len(bounds), batch_size):
        current_bounds = bounds[batch_start : batch_start + batch_size]
        batch_inputs = np.zeros((len(current_bounds), length, panel.inputs.shape[1]), dtype=np.float32)
        batch_depths = np.zeros((len(current_bounds), length, len(TARGET_LAYERS)), dtype=np.float32)
        for offset, (start, stop) in enumerate(current_bounds):
            width = stop - start
            batch_inputs[offset, :width] = normalized_inputs[start:stop]
            batch_depths[offset, :width] = panel.query_depths[start:stop]
        prediction = model(
            torch.from_numpy(batch_inputs).to(device),
            torch.from_numpy(batch_depths).to(device),
        ).cpu().numpy()
        for offset, (start, stop) in enumerate(current_bounds):
            width = stop - start
            sums[start:stop] += prediction[offset, :width]
            counts[start:stop] += 1
    if not np.all(counts > 0):
        raise RuntimeError("inference did not cover every timestamp")
    normalized = sums / counts[:, None, None]
    return normalizer.inverse_outputs(panel, normalized)


__all__ = [
    "ChunkTensors",
    "ContinuousDepthNormalizer",
    "ContinuousDepthPanel",
    "TARGET_DEPTH_METERS",
    "TSContinuousDepthTCN",
    "build_continuous_depth_panel",
    "density_n2_proxy",
    "materialize_training_chunks",
    "predict_panel",
]
