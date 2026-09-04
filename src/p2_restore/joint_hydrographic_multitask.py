"""Pure P2 joint temperature/salinity representation components.

This module deliberately contains no filesystem, scoring, candidate, test, or
upload entry point.  An execution wrapper must provide a fold-blind observation
frame whose active validation target scalars have not been decoded.
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


@dataclass(frozen=True)
class JointHydrographicPanel:
    """One dense public-input row and six supervised residuals per timestamp."""

    times: pd.DatetimeIndex
    inputs: np.ndarray
    input_names: tuple[str, ...]
    temperature_baseline: np.ndarray
    salinity_baseline: np.ndarray
    target_temperature: np.ndarray
    target_salinity: np.ndarray
    reference_target_mask: np.ndarray
    joint_target_mask: np.ndarray
    segment_ids: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.times)
        if self.inputs.shape != (rows, len(self.input_names)):
            raise ValueError("joint hydrographic input shape changed")
        for value in (
            self.temperature_baseline,
            self.salinity_baseline,
            self.target_temperature,
            self.target_salinity,
            self.reference_target_mask,
            self.joint_target_mask,
        ):
            if value.shape != (rows, len(TARGET_LAYERS)):
                raise ValueError("joint hydrographic target shape changed")
        if self.segment_ids.shape != (rows,):
            raise ValueError("joint hydrographic segment shape changed")
        forbidden = {
            *(f"temp_{layer}" for layer in TARGET_LAYERS),
            *(f"psal_{layer}" for layer in TARGET_LAYERS),
        }
        if forbidden.intersection(self.input_names):
            raise ValueError("target-layer scalar leaked into model inputs")


@dataclass(frozen=True)
class JointHydrographicNormalizer:
    input_center: np.ndarray
    input_scale: np.ndarray
    target_center: np.ndarray
    target_scale: np.ndarray

    @classmethod
    def fit(
        cls,
        panel: JointHydrographicPanel,
        selected_times: np.ndarray,
    ) -> JointHydrographicNormalizer:
        selected = np.asarray(selected_times, dtype=bool)
        if selected.shape != (len(panel.times),) or not selected.any():
            raise ValueError("normalizer selection is invalid")

        values = panel.inputs[selected]
        input_center = np.zeros(values.shape[1], dtype=np.float64)
        input_scale = np.ones(values.shape[1], dtype=np.float64)
        for column in range(values.shape[1]):
            current = values[:, column]
            current = current[np.isfinite(current)]
            if not len(current):
                continue
            center = float(np.median(current))
            robust = float(np.median(np.abs(current - center)) * 1.4826)
            fallback = float(np.std(current))
            scale = robust if np.isfinite(robust) and robust > 1e-6 else fallback
            input_center[column] = center
            input_scale[column] = scale if np.isfinite(scale) and scale > 1e-6 else 1.0

        residuals = _physical_residual_targets(panel)
        mask = np.repeat(panel.joint_target_mask[:, :, None], 2, axis=2)
        target_center = np.zeros((len(TARGET_LAYERS), 2), dtype=np.float64)
        target_scale = np.ones((len(TARGET_LAYERS), 2), dtype=np.float64)
        for layer in range(len(TARGET_LAYERS)):
            for variable in range(2):
                keep = selected & mask[:, layer, variable]
                current = residuals[keep, layer, variable]
                if not len(current):
                    raise ValueError(
                        f"joint target layer={layer + 2} variable={variable} has no training rows"
                    )
                center = float(np.median(current))
                robust = float(np.median(np.abs(current - center)) * 1.4826)
                scale = robust if np.isfinite(robust) and robust > 1e-6 else float(np.std(current))
                target_center[layer, variable] = center
                target_scale[layer, variable] = (
                    scale if np.isfinite(scale) and scale > 1e-6 else 1.0
                )
        return cls(input_center, input_scale, target_center, target_scale)

    def transform_inputs(self, inputs: np.ndarray) -> np.ndarray:
        values = (np.asarray(inputs, dtype=np.float64) - self.input_center) / self.input_scale
        values = np.where(np.isfinite(values), values, 0.0)
        return np.clip(values, -12.0, 12.0).astype(np.float32)

    def transform_targets(self, panel: JointHydrographicPanel) -> tuple[np.ndarray, np.ndarray]:
        residuals = _physical_residual_targets(panel)
        transformed = (residuals - self.target_center) / self.target_scale
        joint = panel.joint_target_mask[:, :, None]
        mask = np.repeat(joint, 2, axis=2) & np.isfinite(transformed)
        values = np.where(mask, transformed, 0.0).astype(np.float32)
        return values, mask

    def inverse_temperature(
        self,
        panel: JointHydrographicPanel,
        normalized_temperature: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(normalized_temperature, dtype=np.float64)
        if values.shape != panel.temperature_baseline.shape:
            raise ValueError("normalized temperature prediction shape changed")
        residual = values * self.target_scale[:, 0] + self.target_center[:, 0]
        return np.clip(panel.temperature_baseline + residual, -5.0, 45.0)


def _wide(
    observations: pd.DataFrame,
    value: str,
    times: pd.DatetimeIndex,
) -> pd.DataFrame:
    keyed = observations.assign(_time=pd.to_datetime(observations["time"], utc=True))
    return keyed.pivot(index="_time", columns="layer", values=value).reindex(times)


def _physical_residual_targets(panel: JointHydrographicPanel) -> np.ndarray:
    return np.stack(
        (
            panel.target_temperature - panel.temperature_baseline,
            panel.target_salinity - panel.salinity_baseline,
        ),
        axis=-1,
    )


def build_joint_hydrographic_panel(
    fold_blind_observations: pd.DataFrame,
) -> JointHydrographicPanel:
    """Build public inputs without ever copying target T/S into the input matrix."""

    required = {
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "nominal_depth",
    }
    missing = required.difference(fold_blind_observations.columns)
    if missing:
        raise ValueError(f"fold-blind observations are missing columns: {sorted(missing)}")
    times = pd.DatetimeIndex(
        pd.to_datetime(fold_blind_observations["time"], utc=True).drop_duplicates()
    ).sort_values()
    if not len(times) or times.has_duplicates:
        raise ValueError("fold-blind timestamp surface is invalid")

    temp = _wide(fold_blind_observations, "temp", times)
    psal = _wide(fold_blind_observations, "psal", times)
    depth = _wide(fold_blind_observations, "depth", times)
    nominal = _wide(fold_blind_observations, "nominal_depth", times)

    values: list[np.ndarray] = []
    names: list[str] = []
    for layer in PUBLIC_LAYERS:
        for prefix, wide in (("temp", temp), ("psal", psal), ("depth", depth)):
            current = wide[layer].to_numpy(dtype=np.float64)
            values.extend((current, np.isfinite(current).astype(np.float64)))
            names.extend((f"public_{prefix}_{layer}", f"public_{prefix}_{layer}_mask"))

    public_temperature = np.column_stack([temp[layer].to_numpy(float) for layer in PUBLIC_LAYERS])
    public_salinity = np.column_stack([psal[layer].to_numpy(float) for layer in PUBLIC_LAYERS])
    public_nominal = np.column_stack([nominal[layer].to_numpy(float) for layer in PUBLIC_LAYERS])
    target_nominal = np.column_stack([nominal[layer].to_numpy(float) for layer in TARGET_LAYERS])
    temperature_baseline = np.column_stack(
        [
            _nearest_public_baseline(public_temperature, public_nominal, target_nominal[:, offset])
            for offset in range(len(TARGET_LAYERS))
        ]
    )
    salinity_baseline = np.column_stack(
        [
            _nearest_public_baseline(public_salinity, public_nominal, target_nominal[:, offset])
            for offset in range(len(TARGET_LAYERS))
        ]
    )
    for offset, layer in enumerate(TARGET_LAYERS):
        for label, array in (
            ("temperature_baseline", temperature_baseline[:, offset]),
            ("salinity_baseline", salinity_baseline[:, offset]),
            ("target_nominal_depth", target_nominal[:, offset]),
        ):
            values.extend((array, np.isfinite(array).astype(np.float64)))
            names.extend((f"{label}_{layer}", f"{label}_{layer}_mask"))

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
        names.append(name)
        values.append(array)

    target_temperature = np.column_stack([temp[layer].to_numpy(float) for layer in TARGET_LAYERS])
    target_salinity = np.column_stack([psal[layer].to_numpy(float) for layer in TARGET_LAYERS])
    reference_target_mask = (
        np.isfinite(target_temperature)
        & np.isfinite(target_salinity)
        & np.isfinite(temperature_baseline)
    )
    joint_mask = (
        reference_target_mask & np.isfinite(salinity_baseline) & np.isfinite(target_nominal)
    )
    delta = times.to_series().diff().dt.total_seconds().div(60).to_numpy()
    segment_ids = (
        np.cumsum(np.r_[True, ~np.isclose(delta[1:], CADENCE_MINUTES)]).astype(np.int32) - 1
    )
    return JointHydrographicPanel(
        times=times,
        inputs=np.column_stack(values).astype(np.float32),
        input_names=tuple(names),
        temperature_baseline=temperature_baseline,
        salinity_baseline=salinity_baseline,
        target_temperature=target_temperature,
        target_salinity=target_salinity,
        reference_target_mask=reference_target_mask,
        joint_target_mask=joint_mask,
        segment_ids=segment_ids,
    )


def stage_a_prefix_times(
    panel: JointHydrographicPanel,
    *,
    outer_start: pd.Timestamp,
    embargo_days: int,
    fraction: float,
) -> pd.DatetimeIndex:
    """Reproduce the sealed Stage-A joint-mask prefix independently of loss masking."""

    start = pd.Timestamp(outer_start)
    if start.tzinfo is None:
        raise ValueError("outer start must be timezone-aware")
    start = start.tz_convert("UTC")
    eligible = panel.times[
        (panel.times < start - pd.Timedelta(days=int(embargo_days)))
        & panel.reference_target_mask.any(axis=1)
    ]
    eligible = pd.DatetimeIndex(eligible.unique()).sort_values()
    count = int(math.ceil(len(eligible) * float(fraction)))
    if count < 1 or count > len(eligible):
        raise ValueError("Stage-A prefix timestamp count is invalid")
    return eligible[:count]


class GatedTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        dilation: int,
        *,
        kernel_size: int = 5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
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


class DualHydrographicDepthDecoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        depths = torch.tensor([7.04, 9.44, 14.74], dtype=torch.float32) / 50.0
        encodings = torch.stack(
            (
                depths,
                torch.sin(2 * math.pi * depths),
                torch.cos(2 * math.pi * depths),
                torch.sin(4 * math.pi * depths),
                torch.cos(4 * math.pi * depths),
            ),
            dim=-1,
        )
        self.register_buffer("depth_encodings", encodings)
        self.query = nn.Sequential(
            nn.Linear(5, hidden),
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

    def forward(self, state: Tensor) -> Tensor:
        query = self.query(self.depth_encodings).view(1, 1, len(TARGET_LAYERS), -1)
        expanded = state.unsqueeze(2).expand(-1, -1, len(TARGET_LAYERS), -1)
        return self.output(torch.cat((expanded, query.expand_as(expanded)), dim=-1))


class JointHydrographicTCN(nn.Module):
    """Shared public-sequence encoder with temperature and salinity heads."""

    def __init__(
        self,
        channels: int,
        *,
        hidden: int = 160,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.blocks = nn.ModuleList(
            GatedTemporalBlock(hidden, dilation, dropout=dropout) for dilation in dilations
        )
        self.norm = nn.LayerNorm(hidden)
        self.decoder = DualHydrographicDepthDecoder(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        state = self.input(inputs)
        for block in self.blocks:
            state = block(state)
        return self.decoder(self.norm(state))

    def training_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
        mask: Tensor,
        *,
        vertical_difference_weight: float = 0.25,
    ) -> Tensor:
        prediction = self(inputs)
        if prediction.shape != targets.shape or mask.shape != targets.shape:
            raise ValueError("joint multitask loss shapes changed")
        usable = mask.to(dtype=prediction.dtype)
        squared = (prediction - targets).square()
        direct = (squared * usable).sum() / usable.sum().clamp_min(1.0)

        pair_mask = mask[:, :, 1:, :] & mask[:, :, :-1, :]
        predicted_difference = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_difference = targets[:, :, 1:, :] - targets[:, :, :-1, :]
        pair_weight = pair_mask.to(dtype=prediction.dtype)
        vertical = (
            (predicted_difference - target_difference).square() * pair_weight
        ).sum() / pair_weight.sum().clamp_min(1.0)
        return direct + float(vertical_difference_weight) * vertical


def materialize_joint_chunks(
    panel: JointHydrographicPanel,
    normalizer: JointHydrographicNormalizer,
    selected_times: np.ndarray,
    *,
    length: int = 512,
    stride: int = 384,
    minimum_joint_values: int = 24,
) -> tuple[Tensor, Tensor, Tensor, tuple[tuple[int, int], ...]]:
    selected = np.asarray(selected_times, dtype=bool)
    if selected.shape != (len(panel.times),):
        raise ValueError("chunk selection shape changed")
    inputs = normalizer.transform_inputs(panel.inputs)
    targets, target_mask = normalizer.transform_targets(panel)
    target_mask &= selected[:, None, None]
    selected_joint_mask = panel.joint_target_mask & selected[:, None]
    bounds = tuple(
        bound
        for bound in make_chunk_bounds(
            panel.segment_ids,
            length=length,
            stride=stride,
        )
        if int(selected_joint_mask[bound[0] : bound[1]].sum()) >= int(minimum_joint_values)
    )
    if not bounds:
        raise RuntimeError("no joint hydrographic chunks are available")
    chunk_inputs = np.zeros((len(bounds), length, inputs.shape[1]), dtype=np.float32)
    chunk_targets = np.zeros((len(bounds), length, len(TARGET_LAYERS), 2), dtype=np.float32)
    chunk_mask = np.zeros_like(chunk_targets, dtype=bool)
    for index, (start, stop) in enumerate(bounds):
        width = stop - start
        chunk_inputs[index, :width] = inputs[start:stop]
        chunk_targets[index, :width] = targets[start:stop]
        chunk_mask[index, :width] = target_mask[start:stop]
    return (
        torch.from_numpy(chunk_inputs),
        torch.from_numpy(chunk_targets),
        torch.from_numpy(chunk_mask),
        bounds,
    )


def layer4_only_ablation(
    sealed_reference: np.ndarray,
    multitask_temperature: np.ndarray,
) -> np.ndarray:
    """Replace only layer 4 while preserving layers 2 and 3 bit-for-bit."""

    reference = np.asarray(sealed_reference)
    challenger = np.asarray(multitask_temperature)
    if reference.shape != challenger.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("layer-4 ablation matrices must both have shape [rows, 3]")
    if not np.issubdtype(reference.dtype, np.floating):
        raise ValueError("sealed reference must use a floating dtype")
    result = reference.copy()
    result[:, 2] = challenger[:, 2]
    if not np.array_equal(result[:, :2], reference[:, :2], equal_nan=True):
        raise AssertionError("layer-2/3 no-op invariant failed")
    return result


__all__ = [
    "JointHydrographicNormalizer",
    "JointHydrographicPanel",
    "JointHydrographicTCN",
    "build_joint_hydrographic_panel",
    "layer4_only_ablation",
    "materialize_joint_chunks",
    "stage_a_prefix_times",
]
