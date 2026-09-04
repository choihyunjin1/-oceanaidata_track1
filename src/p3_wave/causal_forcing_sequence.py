"""Causal 48-hour sequence/forcing encoder for the append-only P3 Gen2 probe.

The module is deliberately file-agnostic.  Callers provide already aligned arrays, and
training APIs index only the explicitly supplied prefix IDs.  Anonymous-test files and
absolute timestamps are outside this module's contract.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .causal_forcing_analog import FORCING_COLUMNS
from .data import LEADS, STATIONS
from .revin_patch import (
    ATMOS_ROWS,
    CONTEXT_ROWS,
    RAW_COLUMNS,
    WAVE_ROWS,
    PatchModelConfig,
    PreparedStreams,
    validate_raw_context,
)

FORCING_SEQUENCE_VALUE_COLUMNS = tuple(FORCING_COLUMNS)
FORCING_SEQUENCE_COLUMNS = (
    *FORCING_SEQUENCE_VALUE_COLUMNS,
    *(f"{x}_finite" for x in FORCING_COLUMNS),
)
FORCING_WINDOW_STEPS = 37  # Six hours on the inclusive ten-minute grid.


def _validate_raw_array(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw)
    if values.ndim != 3 or values.shape[1:] != (CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise ValueError(f"raw context must have shape (cases, {CONTEXT_ROWS}, {len(RAW_COLUMNS)})")
    if len(values) == 0:
        raise ValueError("raw context cannot be empty")
    if not np.isfinite(values[:, -1, 0]).all():
        raise ValueError("current hs must be finite")
    if np.isfinite(values[:, 1::2, :4]).any():
        raise ValueError("wave values on structural ten-minute rows are forbidden")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("raw context must be numeric")
    return values


def _rolling_sums(values: np.ndarray, *, window_steps: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    cumulative_value = np.pad(np.cumsum(clean, axis=1), ((0, 0), (1, 0)))
    cumulative_count = np.pad(np.cumsum(finite, axis=1), ((0, 0), (1, 0)))
    stop = np.arange(1, values.shape[1] + 1)
    start = np.maximum(stop - window_steps, 0)
    return (
        cumulative_value[:, stop] - cumulative_value[:, start],
        cumulative_count[:, stop] - cumulative_count[:, start],
    )


def _causal_rolling_mean(values: np.ndarray, *, window_steps: int) -> np.ndarray:
    total, count = _rolling_sums(values, window_steps=window_steps)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _causal_rolling_slope(
    values: np.ndarray,
    *,
    window_steps: int,
    minimum_count: int = 3,
) -> np.ndarray:
    """Return past-only least-squares slopes in source-units per hour."""

    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    hour = np.arange(values.shape[1], dtype=np.float64) / 6.0
    x = np.broadcast_to(hour, values.shape)
    count = finite.astype(np.float64)

    def window_sum(array: np.ndarray) -> np.ndarray:
        cumulative = np.pad(np.cumsum(array, axis=1), ((0, 0), (1, 0)))
        stop = np.arange(1, values.shape[1] + 1)
        start = np.maximum(stop - window_steps, 0)
        return cumulative[:, stop] - cumulative[:, start]

    n = window_sum(count)
    sum_x = window_sum(np.where(finite, x, 0.0))
    sum_y = window_sum(clean)
    sum_xx = window_sum(np.where(finite, np.square(x), 0.0))
    sum_xy = window_sum(np.where(finite, x * clean, 0.0))
    numerator = sum_xy - np.divide(
        sum_x * sum_y,
        n,
        out=np.zeros_like(sum_xy),
        where=n > 0,
    )
    denominator = sum_xx - np.divide(
        np.square(sum_x),
        n,
        out=np.zeros_like(sum_xx),
        where=n > 0,
    )
    result = np.full(values.shape, np.nan, dtype=np.float64)
    eligible = (n >= minimum_count) & (denominator > 1e-12)
    np.divide(numerator, denominator, out=result, where=eligible)
    return result


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def build_causal_forcing_sequence(
    raw: np.ndarray,
    *,
    window_steps: int = FORCING_WINDOW_STEPS,
) -> np.ndarray:
    """Build six rolling forcing states and six masks without looking right of time ``t``.

    The physical definitions match :data:`p3_wave.causal_forcing_analog.FORCING_COLUMNS`,
    but are evaluated at every ten-minute context position.  Signed ``log1p`` compresses
    unbounded components without fitting statistics on validation cases.  Wave-direction
    and period remain missing on structural ten-minute rows, so rolling statistics use
    only native observations and never interpolate from the future.
    """

    if window_steps != FORCING_WINDOW_STEPS:
        raise ValueError("the Gen2 forcing window is frozen at 37 inclusive ten-minute steps")
    source = _validate_raw_array(raw)
    result = np.empty(
        (len(source), CONTEXT_ROWS, 2 * len(FORCING_COLUMNS)),
        dtype=np.float32,
    )
    # The full train cache is roughly 282 MB.  Work case blocks independently so the
    # float64 cumulative-statistic scratch space stays bounded.
    for batch_start in range(0, len(source), 512):
        batch_stop = min(batch_start + 512, len(source))
        values = np.asarray(source[batch_start:batch_stop], dtype=np.float64)
        result[batch_start:batch_stop] = _build_causal_forcing_batch(
            values,
            window_steps=window_steps,
        )
    if result.shape != (len(source), CONTEXT_ROWS, 2 * len(FORCING_COLUMNS)):
        raise AssertionError("forcing sequence shape changed")
    if not np.isfinite(result).all():
        raise AssertionError("forcing sequence contains a non-finite value")
    return result


def _build_causal_forcing_batch(values: np.ndarray, *, window_steps: int) -> np.ndarray:
    wave_direction = values[:, :, 3]
    wind_speed = values[:, :, 4]
    gust = values[:, :, 5]
    wind_direction = values[:, :, 6]
    pressure = values[:, :, 9]
    period = values[:, :, 1]

    direction_finite = np.isfinite(wave_direction) & np.isfinite(wind_direction)
    alignment = np.full(wave_direction.shape, np.nan, dtype=np.float64)
    alignment[direction_finite] = np.cos(
        np.deg2rad(wind_direction[direction_finite] - wave_direction[direction_finite])
    )
    wind_input = np.full(wind_speed.shape, np.nan, dtype=np.float64)
    wind_input_finite = direction_finite & np.isfinite(wind_speed)
    wind_input[wind_input_finite] = np.square(wind_speed[wind_input_finite]) * np.maximum(
        alignment[wind_input_finite], 0.0
    )
    gust_excess = np.where(np.isfinite(gust) & np.isfinite(wind_speed), gust - wind_speed, np.nan)

    physical = np.stack(
        [
            _causal_rolling_mean(wind_input, window_steps=window_steps),
            _causal_rolling_slope(wind_input, window_steps=window_steps),
            _causal_rolling_mean(alignment, window_steps=window_steps),
            _causal_rolling_mean(gust_excess, window_steps=window_steps),
            _causal_rolling_slope(pressure, window_steps=window_steps),
            _causal_rolling_slope(period, window_steps=window_steps),
        ],
        axis=-1,
    )
    mask = np.isfinite(physical)
    transformed = np.zeros_like(physical, dtype=np.float64)
    transformed[..., 2] = np.where(mask[..., 2], physical[..., 2], 0.0)
    for column in (0, 1, 3, 4, 5):
        transformed[..., column] = np.where(
            mask[..., column], _signed_log1p(physical[..., column]), 0.0
        )
    result = np.concatenate([transformed, mask.astype(np.float64)], axis=-1).astype(np.float32)
    return result


def _deterministic_lower_median(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the finite-value lower median along time without ``torch.nanmedian``.

    ``torch.nanmedian`` selects the lower of the two central values for an even count,
    but its CUDA implementation is unavailable when deterministic algorithms are
    enforced.  Sorting finite values ahead of ``+inf`` and gathering ``(n - 1) // 2``
    has the same case/channel semantics and uses a deterministic CUDA implementation.
    """

    if values.shape != mask.shape or values.ndim != 3:
        raise ValueError("median values and mask must be aligned three-dimensional tensors")
    ordered = torch.sort(
        torch.where(mask, values, torch.full_like(values, float("inf"))),
        dim=1,
    ).values
    count = mask.sum(dim=1, keepdim=True)
    lower_index = torch.clamp((count - 1) // 2, min=0)
    median = torch.gather(ordered, dim=1, index=lower_index)
    return torch.where(count > 0, median, torch.zeros_like(median))


def _deterministic_robust_center_scale(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale_floor: float,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    location = _deterministic_lower_median(values, mask)
    absolute_deviation = torch.abs(values - location)
    scale = 1.4826 * _deterministic_lower_median(absolute_deviation, mask)
    fallback = torch.sqrt(
        torch.sum(
            torch.where(mask, torch.square(values - location), torch.zeros_like(values)),
            dim=1,
            keepdim=True,
        )
        / torch.clamp(mask.sum(dim=1, keepdim=True).to(values.dtype), min=1.0)
    )
    scale = torch.where(scale >= scale_floor, scale, fallback)
    scale = torch.where(
        torch.isfinite(scale) & (scale >= scale_floor),
        scale,
        torch.full_like(scale, scale_floor),
    )
    used_location = location if center else torch.zeros_like(location)
    normalized = torch.where(mask, (values - used_location) / scale, torch.zeros_like(values))
    return normalized, used_location, scale


def prepare_streams_deterministic(
    raw: torch.Tensor,
    config: PatchModelConfig | None = None,
) -> PreparedStreams:
    """Match ``revin_patch.prepare_streams`` without nondeterministic CUDA nanmedian."""

    cfg = config or PatchModelConfig()
    cfg.validate()
    validate_raw_context(raw)
    raw = raw.to(dtype=torch.float32)

    wave_raw = raw[:, ::2, :4]
    atmos_raw = raw[:, :, 4:]
    if wave_raw.shape[1] != WAVE_ROWS:
        raise AssertionError("native wave extraction did not produce 145 rows")

    current_hs = wave_raw[:, -1, 0]
    wave_continuous = wave_raw[:, :, :3]
    wave_mask = torch.isfinite(wave_raw)
    hs_delta = wave_continuous[:, :, :1] - current_hs[:, None, None]
    hs_normalized, _, hs_scale = _deterministic_robust_center_scale(
        hs_delta,
        wave_mask[:, :, :1],
        scale_floor=cfg.robust_scale_floor,
        center=False,
    )
    other_wave, _, _ = _deterministic_robust_center_scale(
        wave_continuous[:, :, 1:],
        wave_mask[:, :, 1:3],
        scale_floor=cfg.robust_scale_floor,
    )
    wave_angle = torch.deg2rad(wave_raw[:, :, 3])
    wave_direction = torch.stack([torch.sin(wave_angle), torch.cos(wave_angle)], dim=-1)
    wave_direction = torch.where(
        wave_mask[:, :, 3:4], wave_direction, torch.zeros_like(wave_direction)
    )
    wave_time = torch.linspace(-1.0, 0.0, WAVE_ROWS, device=raw.device, dtype=raw.dtype)
    wave_time = wave_time.view(1, WAVE_ROWS, 1).expand(len(raw), -1, -1)
    wave = torch.cat(
        [
            hs_normalized,
            other_wave,
            wave_direction,
            wave_mask.to(raw.dtype),
            wave_time,
        ],
        dim=-1,
    )

    atmos_mask = torch.isfinite(atmos_raw)
    atmos_continuous = atmos_raw[:, :, [0, 1, 3, 4, 5]]
    atmos_continuous_mask = atmos_mask[:, :, [0, 1, 3, 4, 5]]
    atmos_normalized, _, _ = _deterministic_robust_center_scale(
        atmos_continuous,
        atmos_continuous_mask,
        scale_floor=cfg.robust_scale_floor,
    )
    wind_angle = torch.deg2rad(atmos_raw[:, :, 2])
    wind_direction = torch.stack([torch.sin(wind_angle), torch.cos(wind_angle)], dim=-1)
    wind_direction = torch.where(
        atmos_mask[:, :, 2:3], wind_direction, torch.zeros_like(wind_direction)
    )
    atmos_time = torch.linspace(-1.0, 0.0, ATMOS_ROWS, device=raw.device, dtype=raw.dtype)
    atmos_time = atmos_time.view(1, ATMOS_ROWS, 1).expand(len(raw), -1, -1)
    atmos = torch.cat(
        [atmos_normalized, wind_direction, atmos_mask.to(raw.dtype), atmos_time],
        dim=-1,
    )

    if wave.shape[-1] != 10 or atmos.shape[-1] != 14:
        raise AssertionError("prepared stream channel contract changed")
    return PreparedStreams(wave, atmos, current_hs, hs_scale[:, 0, 0])


def _validated_ids(values: Sequence[int] | np.ndarray, *, size: int, role: str) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1 or len(source) == 0:
        raise ValueError(f"{role} IDs must be a non-empty vector")
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"{role} IDs must be integers")
    result = source.astype(np.int64, copy=False)
    if np.unique(result).size != len(result):
        raise ValueError(f"{role} IDs must be unique")
    if result.min() < 0 or result.max() >= size:
        raise IndexError(f"{role} IDs are outside the aligned array")
    return result


def _ids_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class CompactRobustScaler:
    """Median/IQR scaler fit on one explicitly supplied training prefix."""

    center: np.ndarray
    scale: np.ndarray
    fit_ids_sha256: str

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        train_ids: Sequence[int] | np.ndarray,
        *,
        forbidden_ids: Sequence[int] | np.ndarray | None = None,
    ) -> CompactRobustScaler:
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("compact features must have shape (cases, features)")
        ids = _validated_ids(train_ids, size=len(matrix), role="training")
        if forbidden_ids is not None:
            forbidden = _validated_ids(forbidden_ids, size=len(matrix), role="forbidden")
            if np.intersect1d(ids, forbidden).size:
                raise PermissionError("training IDs overlap forbidden validation IDs")
        selected = np.asarray(matrix[ids], dtype=np.float64)
        center = np.zeros(selected.shape[1], dtype=np.float64)
        scale = np.ones(selected.shape[1], dtype=np.float64)
        for column in range(selected.shape[1]):
            finite = selected[:, column][np.isfinite(selected[:, column])]
            if len(finite) == 0:
                continue
            center[column] = float(np.median(finite))
            q25, q75 = np.quantile(finite, [0.25, 0.75])
            width = float(q75 - q25)
            scale[column] = width if np.isfinite(width) and width > 1e-6 else 1.0
        return cls(center.astype(np.float32), scale.astype(np.float32), _ids_sha256(ids))

    @property
    def feature_count(self) -> int:
        return int(len(self.center))

    @property
    def state_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.center, dtype="<f4").tobytes(order="C"))
        digest.update(np.asarray(self.scale, dtype="<f4").tobytes(order="C"))
        digest.update(self.fit_ids_sha256.encode("ascii"))
        return digest.hexdigest()

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_count:
            raise ValueError("compact feature count differs from the fitted scaler")
        finite = np.isfinite(matrix)
        normalized = np.where(finite, (matrix - self.center) / self.scale, 0.0)
        result = np.concatenate([normalized, finite.astype(np.float32)], axis=1).astype(
            np.float32,
            copy=False,
        )
        if not np.isfinite(result).all():
            raise AssertionError("scaled compact features contain a non-finite value")
        return result


@dataclass(frozen=True)
class CausalForcingSequenceConfig:
    compact_feature_count: int = 591
    width: int = 64
    compact_hidden: int = 128
    attention_heads: int = 4
    norm_groups: int = 8
    dropout: float = 0.1
    wave_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    forcing_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

    def validate(self) -> None:
        if self.compact_feature_count < 1 or self.width < 4 or self.compact_hidden < 4:
            raise ValueError("model dimensions must be positive and nontrivial")
        if self.width % self.attention_heads or self.width % self.norm_groups:
            raise ValueError("width must be divisible by attention heads and norm groups")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        for name, values in (
            ("wave_dilations", self.wave_dilations),
            ("forcing_dilations", self.forcing_dilations),
        ):
            if not values or any(value < 1 for value in values):
                raise ValueError(f"{name} must contain positive dilations")


@dataclass(frozen=True)
class FixedEpochTrainingConfig:
    epochs: int = 8
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 2e-4
    gradient_clip_norm: float = 1.0
    use_bf16_on_cuda: bool = True

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")


class _CausalDepthwiseBlock(nn.Module):
    def __init__(self, width: int, dilation: int, *, groups: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = 2 * int(dilation)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            dilation=int(dilation),
            groups=width,
        )
        self.norm = nn.GroupNorm(groups, width)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(F.pad(values, (self.left_padding, 0)))
        hidden = self.pointwise(F.gelu(self.norm(hidden)))
        return values + self.dropout(hidden)


class LeadCoupledCausalForcingEncoder(nn.Module):
    """Two-stream causal TCN with ordered six-lead decoding."""

    def __init__(self, config: CausalForcingSequenceConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalForcingSequenceConfig()
        self.config.validate()
        cfg = self.config
        self.wave_stem = nn.Linear(10, cfg.width)
        self.forcing_stem = nn.Linear(14 + len(FORCING_SEQUENCE_COLUMNS), cfg.width)
        self.wave_blocks = nn.ModuleList(
            [
                _CausalDepthwiseBlock(
                    cfg.width,
                    dilation,
                    groups=cfg.norm_groups,
                    dropout=cfg.dropout,
                )
                for dilation in cfg.wave_dilations
            ]
        )
        self.forcing_blocks = nn.ModuleList(
            [
                _CausalDepthwiseBlock(
                    cfg.width,
                    dilation,
                    groups=cfg.norm_groups,
                    dropout=cfg.dropout,
                )
                for dilation in cfg.forcing_dilations
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * cfg.width, cfg.width),
            nn.GELU(),
            nn.LayerNorm(cfg.width),
        )
        self.compact_projection = nn.Sequential(
            nn.Linear(2 * cfg.compact_feature_count, cfg.compact_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.compact_hidden, cfg.width),
            nn.LayerNorm(cfg.width),
        )
        self.station_embedding = nn.Embedding(len(STATIONS), cfg.width)
        self.amplitude_projection = nn.Sequential(
            nn.Linear(2, cfg.width),
            nn.GELU(),
            nn.Linear(cfg.width, cfg.width),
        )
        self.lead_embedding = nn.Parameter(torch.empty(1, len(LEADS), cfg.width))
        self.context_attention = nn.MultiheadAttention(
            cfg.width,
            cfg.attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(cfg.width)
        self.lead_decoder = nn.GRU(cfg.width, cfg.width, num_layers=1, batch_first=True)
        self.output_norm = nn.LayerNorm(cfg.width)
        self.residual_head = nn.Linear(cfg.width, 1)
        nn.init.normal_(self.lead_embedding, std=0.02)

    def forward(
        self,
        raw: torch.Tensor,
        station_code: torch.Tensor,
        compact_scaled: torch.Tensor,
        forcing: torch.Tensor,
    ) -> torch.Tensor:
        if station_code.ndim != 1 or len(station_code) != len(raw):
            raise ValueError("station code must align one-to-one with raw contexts")
        if compact_scaled.shape != (len(raw), 2 * self.config.compact_feature_count):
            raise ValueError("scaled compact features differ from the model contract")
        if forcing.shape != (len(raw), CONTEXT_ROWS, len(FORCING_SEQUENCE_COLUMNS)):
            raise ValueError("forcing sequence differs from the model contract")
        if (
            len(raw) == 0
            or station_code.min().item() < 0
            or station_code.max().item() >= len(STATIONS)
        ):
            raise ValueError("station code lies outside the official station set")
        if not torch.isfinite(compact_scaled).all() or not torch.isfinite(forcing).all():
            raise ValueError("compact and forcing inputs must be finite")

        streams = prepare_streams_deterministic(raw)
        wave = self.wave_stem(streams.wave).transpose(1, 2)
        atmospheric = torch.cat([streams.atmos, forcing.to(streams.atmos.dtype)], dim=-1)
        atmospheric = self.forcing_stem(atmospheric).transpose(1, 2)
        for block in self.wave_blocks:
            wave = block(wave)
        for block in self.forcing_blocks:
            atmospheric = block(atmospheric)
        atmospheric = atmospheric[:, :, ::2]
        if wave.shape[-1] != atmospheric.shape[-1] or wave.shape[-1] != 145:
            raise AssertionError("native stream alignment changed")
        context_tokens = self.fusion(
            torch.cat([wave.transpose(1, 2), atmospheric.transpose(1, 2)], dim=-1)
        )

        amplitude = torch.stack(
            [
                torch.log1p(streams.current_hs.clamp_min(0.0)),
                torch.log(streams.hs_scale.clamp_min(1e-6)),
            ],
            dim=-1,
        )
        case_context = (
            self.compact_projection(compact_scaled)
            + self.station_embedding(station_code)
            + self.amplitude_projection(amplitude)
        )
        queries = self.lead_embedding.expand(len(raw), -1, -1) + case_context[:, None, :]
        attended, _ = self.context_attention(
            queries,
            context_tokens,
            context_tokens,
            need_weights=False,
        )
        coupled, _ = self.lead_decoder(self.query_norm(queries + attended))
        normalized_delta = self.residual_head(self.output_norm(coupled)).squeeze(-1)
        return normalized_delta * streams.hs_scale[:, None]

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def _seed_deterministically(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def model_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass
class FittedSequenceModel:
    model_config: CausalForcingSequenceConfig
    training_config: FixedEpochTrainingConfig
    seed: int
    scaler: CompactRobustScaler
    state_dict: dict[str, torch.Tensor]
    train_ids_sha256: str
    model_state_sha256: str


def _aligned_case_count(*arrays: np.ndarray) -> int:
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("aligned input arrays have different case counts")
    return lengths.pop()


def _resolve_forcing(
    raw: np.ndarray,
    ids: np.ndarray,
    forcing: np.ndarray | None,
) -> np.ndarray:
    if forcing is None:
        return build_causal_forcing_sequence(np.asarray(raw)[ids])
    source = np.asarray(forcing)
    if source.shape != (len(raw), CONTEXT_ROWS, len(FORCING_SEQUENCE_COLUMNS)):
        raise ValueError("precomputed forcing does not align with raw cases")
    selected = np.asarray(source[ids], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("precomputed forcing contains a non-finite selected value")
    return selected


def fit_fixed_epoch_sequence_model(
    raw: np.ndarray,
    station: np.ndarray,
    compact: np.ndarray,
    target_delta: np.ndarray,
    case_weight: np.ndarray,
    train_ids: Sequence[int] | np.ndarray,
    *,
    seed: int,
    device: str | torch.device,
    model_config: CausalForcingSequenceConfig | None = None,
    training_config: FixedEpochTrainingConfig | None = None,
    forcing: np.ndarray | None = None,
    compact_scaler: CompactRobustScaler | None = None,
    forbidden_ids: Sequence[int] | np.ndarray | None = None,
) -> FittedSequenceModel:
    """Fit one fixed-epoch cell while indexing only its explicit training prefix."""

    raw_array = np.asarray(raw)
    station_array = np.asarray(station)
    compact_array = np.asarray(compact)
    target_array = np.asarray(target_delta)
    weight_array = np.asarray(case_weight)
    size = _aligned_case_count(
        raw_array,
        station_array,
        compact_array,
        target_array,
        weight_array,
    )
    config = model_config or CausalForcingSequenceConfig()
    training = training_config or FixedEpochTrainingConfig()
    config.validate()
    training.validate()
    if raw_array.shape[1:] != (CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise ValueError("raw array shape differs from the 48-hour context contract")
    if station_array.shape != (size,) or compact_array.shape != (
        size,
        config.compact_feature_count,
    ):
        raise ValueError("station or compact arrays differ from the model contract")
    if target_array.shape != (size, len(LEADS)) or weight_array.shape != (size,):
        raise ValueError("target or case-weight arrays differ from the six-lead contract")

    ids = _validated_ids(train_ids, size=size, role="training")
    if forbidden_ids is not None:
        forbidden = _validated_ids(forbidden_ids, size=size, role="forbidden")
        if np.intersect1d(ids, forbidden).size:
            raise PermissionError("training IDs overlap forbidden validation IDs")
    if compact_scaler is None:
        scaler = CompactRobustScaler.fit(
            compact_array,
            ids,
            forbidden_ids=forbidden_ids,
        )
    else:
        scaler = compact_scaler
        if scaler.feature_count != config.compact_feature_count:
            raise ValueError("reused compact scaler feature count differs")
        if scaler.fit_ids_sha256 != _ids_sha256(ids):
            raise PermissionError("reused compact scaler was not fit on the exact training IDs")
    train_raw = np.asarray(raw_array[ids], dtype=np.float32)
    train_station = np.asarray(station_array[ids], dtype=np.int64)
    train_compact = scaler.transform(compact_array[ids])
    train_target = np.asarray(target_array[ids], dtype=np.float32)
    train_weight = np.asarray(weight_array[ids], dtype=np.float32)
    train_forcing = _resolve_forcing(raw_array, ids, forcing)
    if not np.isfinite(train_target).all() or not np.isfinite(train_weight).all():
        raise ValueError("selected training targets and weights must be finite")
    if (train_weight <= 0.0).any():
        raise ValueError("selected case weights must be positive")
    train_weight = train_weight / float(train_weight.mean())

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_deterministically(int(seed))
    model = LeadCoupledCausalForcingEncoder(config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    for epoch in range(1, training.epochs + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + epoch)
        order = torch.randperm(len(ids), generator=generator).numpy()
        for start in range(0, len(order), training.batch_size):
            local = order[start : start + training.batch_size]
            raw_batch = torch.from_numpy(train_raw[local]).to(selected_device)
            station_batch = torch.from_numpy(train_station[local]).to(selected_device)
            compact_batch = torch.from_numpy(train_compact[local]).to(selected_device)
            forcing_batch = torch.from_numpy(train_forcing[local]).to(selected_device)
            target_batch = torch.from_numpy(train_target[local]).to(selected_device)
            weight_batch = torch.from_numpy(train_weight[local]).to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.bfloat16,
                enabled=selected_device.type == "cuda" and training.use_bf16_on_cuda,
            ):
                prediction = model(
                    raw_batch,
                    station_batch,
                    compact_batch,
                    forcing_batch,
                )
                loss = torch.mean(weight_batch[:, None] * torch.square(prediction - target_batch))
            if not torch.isfinite(loss):
                raise RuntimeError("fixed-epoch sequence training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
            optimizer.step()

    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    state_sha = model_state_sha256(state)
    del model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return FittedSequenceModel(
        model_config=config,
        training_config=training,
        seed=int(seed),
        scaler=scaler,
        state_dict=state,
        train_ids_sha256=_ids_sha256(ids),
        model_state_sha256=state_sha,
    )


def predict_with_fitted_sequence_model(
    fitted: FittedSequenceModel,
    raw: np.ndarray,
    station: np.ndarray,
    compact: np.ndarray,
    prediction_ids: Sequence[int] | np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int | None = None,
    forcing: np.ndarray | None = None,
) -> np.ndarray:
    """Reload the sealed CPU state and predict six residuals for explicit IDs."""

    raw_array = np.asarray(raw)
    station_array = np.asarray(station)
    compact_array = np.asarray(compact)
    size = _aligned_case_count(raw_array, station_array, compact_array)
    config = fitted.model_config
    if compact_array.shape != (size, config.compact_feature_count):
        raise ValueError("prediction compact features differ from the fitted model contract")
    ids = _validated_ids(prediction_ids, size=size, role="prediction")
    selected_raw = np.asarray(raw_array[ids], dtype=np.float32)
    selected_station = np.asarray(station_array[ids], dtype=np.int64)
    selected_compact = fitted.scaler.transform(compact_array[ids])
    selected_forcing = _resolve_forcing(raw_array, ids, forcing)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if model_state_sha256(fitted.state_dict) != fitted.model_state_sha256:
        raise PermissionError("fitted model state SHA differs before prediction")
    model = LeadCoupledCausalForcingEncoder(config).to(selected_device)
    model.load_state_dict(fitted.state_dict, strict=True)
    model.eval()
    use_batch = int(batch_size or fitted.training_config.batch_size)
    if use_batch < 1:
        raise ValueError("prediction batch size must be positive")
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ids), use_batch):
            stop = min(start + use_batch, len(ids))
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.bfloat16,
                enabled=(
                    selected_device.type == "cuda" and fitted.training_config.use_bf16_on_cuda
                ),
            ):
                prediction = model(
                    torch.from_numpy(selected_raw[start:stop]).to(selected_device),
                    torch.from_numpy(selected_station[start:stop]).to(selected_device),
                    torch.from_numpy(selected_compact[start:stop]).to(selected_device),
                    torch.from_numpy(selected_forcing[start:stop]).to(selected_device),
                )
            outputs.append(prediction.float().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    if result.shape != (len(ids), len(LEADS)) or not np.isfinite(result).all():
        raise RuntimeError("sequence prediction shape or finiteness changed")
    del model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def fit_fixed_epoch_and_predict(
    raw: np.ndarray,
    station: np.ndarray,
    compact: np.ndarray,
    target_delta: np.ndarray,
    case_weight: np.ndarray,
    train_ids: Sequence[int] | np.ndarray,
    prediction_ids: Sequence[int] | np.ndarray,
    *,
    seed: int,
    device: str | torch.device,
    model_config: CausalForcingSequenceConfig | None = None,
    training_config: FixedEpochTrainingConfig | None = None,
    forcing: np.ndarray | None = None,
    compact_scaler: CompactRobustScaler | None = None,
) -> tuple[np.ndarray, FittedSequenceModel]:
    """Fit one independent prefix cell and predict its disjoint validation IDs."""

    fitted = fit_fixed_epoch_sequence_model(
        raw,
        station,
        compact,
        target_delta,
        case_weight,
        train_ids,
        seed=seed,
        device=device,
        model_config=model_config,
        training_config=training_config,
        forcing=forcing,
        compact_scaler=compact_scaler,
        forbidden_ids=prediction_ids,
    )
    prediction = predict_with_fitted_sequence_model(
        fitted,
        raw,
        station,
        compact,
        prediction_ids,
        device=device,
        forcing=forcing,
    )
    return prediction, fitted


def save_fitted_sequence_model(fitted: FittedSequenceModel, path: str | Path) -> None:
    """Write one model bundle with exclusive-create semantics."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "model_config": asdict(fitted.model_config),
        "training_config": asdict(fitted.training_config),
        "seed": int(fitted.seed),
        "scaler_center": torch.from_numpy(np.asarray(fitted.scaler.center, dtype=np.float32)),
        "scaler_scale": torch.from_numpy(np.asarray(fitted.scaler.scale, dtype=np.float32)),
        "scaler_fit_ids_sha256": fitted.scaler.fit_ids_sha256,
        "scaler_sha256": fitted.scaler.state_sha256,
        "state_dict": fitted.state_dict,
        "train_ids_sha256": fitted.train_ids_sha256,
        "model_state_sha256": fitted.model_state_sha256,
    }
    with target.open("xb") as handle:
        torch.save(payload, handle)


def load_fitted_sequence_model(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> FittedSequenceModel:
    """Load and independently hash-check one saved model bundle."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("saved sequence model schema differs")
    model_config = CausalForcingSequenceConfig(**payload["model_config"])
    training_config = FixedEpochTrainingConfig(**payload["training_config"])
    model_config.validate()
    training_config.validate()
    scaler = CompactRobustScaler(
        payload["scaler_center"].detach().cpu().numpy().astype(np.float32),
        payload["scaler_scale"].detach().cpu().numpy().astype(np.float32),
        str(payload["scaler_fit_ids_sha256"]),
    )
    if scaler.state_sha256 != payload["scaler_sha256"]:
        raise PermissionError("saved compact scaler SHA differs")
    state = {
        str(name): tensor.detach().cpu().clone() for name, tensor in payload["state_dict"].items()
    }
    state_sha = model_state_sha256(state)
    if state_sha != payload["model_state_sha256"]:
        raise PermissionError("saved model state SHA differs")
    return FittedSequenceModel(
        model_config=model_config,
        training_config=training_config,
        seed=int(payload["seed"]),
        scaler=scaler,
        state_dict=state,
        train_ids_sha256=str(payload["train_ids_sha256"]),
        model_state_sha256=state_sha,
    )


__all__ = [
    "CausalForcingSequenceConfig",
    "CompactRobustScaler",
    "FORCING_SEQUENCE_COLUMNS",
    "FORCING_SEQUENCE_VALUE_COLUMNS",
    "FORCING_WINDOW_STEPS",
    "FixedEpochTrainingConfig",
    "FittedSequenceModel",
    "LeadCoupledCausalForcingEncoder",
    "build_causal_forcing_sequence",
    "fit_fixed_epoch_and_predict",
    "fit_fixed_epoch_sequence_model",
    "load_fitted_sequence_model",
    "model_state_sha256",
    "prepare_streams_deterministic",
    "predict_with_fitted_sequence_model",
    "save_fitted_sequence_model",
]
