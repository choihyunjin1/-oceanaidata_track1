"""Leak-safe hierarchical residual-basis forecaster for the P3 Gen5 probe.

The helper is deliberately file agnostic.  It accepts only already aligned in-memory
arrays, performs deterministic case-local preparation, and keeps the core fit API on a
physically sliced training target.  The model predicts a 72-step residual path but the
optimizer sees only the six official lead deltas.
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

from .causal_forcing_sequence import prepare_streams_deterministic
from .data import LEADS, STATIONS
from .revin_patch import CONTEXT_ROWS, RAW_COLUMNS, extract_past_context

CONTEXT_20M_STEPS = 144
INPUT_CHANNELS = 24
FORECAST_20M_STEPS = 72
POOLING_FACTORS = (12, 4, 1)
FORECAST_KNOTS = (6, 18, 72)
BLOCKS_PER_STACK = 2
OFFICIAL_FORECAST_STEPS = tuple(int(lead * 3) for lead in LEADS)
OFFICIAL_FORECAST_INDICES = tuple(step - 1 for step in OFFICIAL_FORECAST_STEPS)
MODEL_BUNDLE_SCHEMA = "p3_hierarchical_residual_basis_v1"


def _validated_indices(
    values: Sequence[int] | np.ndarray,
    *,
    size: int,
    role: str,
) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1 or len(source) == 0:
        raise ValueError(f"{role} IDs must be a non-empty vector")
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"{role} IDs must be integers")
    result = source.astype(np.int64, copy=False)
    if np.unique(result).size != len(result):
        raise ValueError(f"{role} IDs must be unique")
    if result.min() < 0 or result.max() >= size:
        raise IndexError(f"{role} IDs are outside the aligned arrays")
    return result


def _validated_case_ids(
    values: Sequence[int] | np.ndarray,
    *,
    expected_size: int | None,
    role: str,
) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1 or len(source) == 0:
        raise ValueError(f"{role} case IDs must be a non-empty vector")
    if expected_size is not None and len(source) != expected_size:
        raise ValueError(f"{role} case IDs do not align with the sliced arrays")
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"{role} case IDs must be integers")
    result = source.astype(np.int64, copy=False)
    if np.unique(result).size != len(result):
        raise ValueError(f"{role} case IDs must be unique")
    if result.min() < 0:
        raise ValueError(f"{role} case IDs must be non-negative")
    return result


def _ids_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _training_context_sha256(
    raw: np.ndarray,
    station: np.ndarray,
    static: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, values, dtype in (
        ("raw", raw, "<f4"),
        ("station", station, "<i8"),
        ("static", static, "<f4"),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} is not a lowercase SHA-256 digest")
    return text


def extract_past_raw_context(values: np.ndarray, anchor_position: int) -> np.ndarray:
    """Copy the inclusive 48-hour history ending at ``anchor_position``.

    Rows after the anchor are never indexed.  This thin wrapper makes the temporal
    boundary explicit at the Gen5 helper surface.
    """

    return extract_past_context(values, anchor_position)


@dataclass(frozen=True)
class PreparedBasisContext:
    values: torch.Tensor
    current_hs: torch.Tensor
    hs_scale: torch.Tensor


def prepare_hierarchical_context(raw: torch.Tensor) -> PreparedBasisContext:
    """Return a finite 144-step, 24-channel, case-local context.

    The existing deterministic preparation supplies ten native wave channels and
    fourteen atmospheric channels.  Both are aligned on the 20-minute grid; the first
    inclusive endpoint is dropped so that 48 hours are represented by 144 intervals and
    the final row remains the forecast anchor.
    """

    streams = prepare_streams_deterministic(raw)
    wave = streams.wave[:, 1:, :]
    atmosphere = streams.atmos[:, ::2, :][:, 1:, :]
    values = torch.cat([wave, atmosphere], dim=-1)
    expected = (len(raw), CONTEXT_20M_STEPS, INPUT_CHANNELS)
    if values.shape != expected:
        raise AssertionError(f"hierarchical context shape changed: {tuple(values.shape)}")
    if not torch.isfinite(values).all():
        raise AssertionError("hierarchical context contains a non-finite value")
    return PreparedBasisContext(values, streams.current_hs, streams.hs_scale)


def average_pool_context(values: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample contiguous history intervals without learned temporal operators."""

    if values.ndim != 3 or values.shape[1:] != (CONTEXT_20M_STEPS, INPUT_CHANNELS):
        raise ValueError(f"context must have shape (batch, {CONTEXT_20M_STEPS}, {INPUT_CHANNELS})")
    if factor not in POOLING_FACTORS:
        raise ValueError(f"pooling factor must be one of {POOLING_FACTORS}")
    if CONTEXT_20M_STEPS % factor:
        raise AssertionError("context length must be divisible by every pooling factor")
    return values.reshape(
        len(values),
        CONTEXT_20M_STEPS // factor,
        factor,
        INPUT_CHANNELS,
    ).mean(dim=2)


def _linear_interpolate_time(values: torch.Tensor, output_steps: int) -> torch.Tensor:
    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("basis values must have shape (batch, at_least_two_knots, channels)")
    if output_steps < 2:
        raise ValueError("output steps must be at least two")
    knots = values.shape[1]
    position = torch.linspace(
        0.0,
        float(knots - 1),
        output_steps,
        dtype=values.dtype,
        device=values.device,
    )
    left = torch.floor(position).to(dtype=torch.long)
    right = torch.clamp(left + 1, max=knots - 1)
    weight = (position - left.to(position.dtype)).view(1, output_steps, 1)
    return values[:, left, :] * (1.0 - weight) + values[:, right, :] * weight


def interpolate_forecast_knots(
    knots: torch.Tensor,
    *,
    output_steps: int = FORECAST_20M_STEPS,
) -> torch.Tensor:
    """Linearly expand forecast knots while preserving both endpoints exactly."""

    if knots.ndim != 2:
        raise ValueError("forecast knots must have shape (batch, knots)")
    return _linear_interpolate_time(knots.unsqueeze(-1), output_steps).squeeze(-1)


@dataclass(frozen=True)
class StaticRobustScaler:
    """Median/IQR transform fit only on a physically sliced training matrix."""

    center: np.ndarray
    scale: np.ndarray
    fit_ids_sha256: str

    @classmethod
    def fit(
        cls,
        train_values: np.ndarray,
        train_case_ids: Sequence[int] | np.ndarray,
        *,
        forbidden_case_ids: Sequence[int] | np.ndarray,
    ) -> StaticRobustScaler:
        matrix = np.asarray(train_values)
        if matrix.ndim != 2 or matrix.shape[1] == 0 or len(matrix) == 0:
            raise ValueError("sliced training static features must be a non-empty matrix")
        train_ids = _validated_case_ids(
            train_case_ids,
            expected_size=len(matrix),
            role="training",
        )
        forbidden_ids = _validated_case_ids(
            forbidden_case_ids,
            expected_size=None,
            role="forbidden",
        )
        if np.intersect1d(train_ids, forbidden_ids).size:
            raise PermissionError("training case IDs overlap forbidden case IDs")

        selected = np.asarray(matrix, dtype=np.float64)
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
        return cls(center.astype(np.float32), scale.astype(np.float32), _ids_sha256(train_ids))

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
            raise ValueError("static feature count differs from the fitted scaler")
        finite = np.isfinite(matrix)
        normalized = np.where(finite, (matrix - self.center) / self.scale, 0.0)
        result = np.concatenate([normalized, finite.astype(np.float32)], axis=1).astype(
            np.float32,
            copy=False,
        )
        if not np.isfinite(result).all():
            raise AssertionError("scaled static features contain a non-finite value")
        return result


@dataclass(frozen=True)
class HierarchicalResidualBasisConfig:
    """Frozen structural contract; width overrides exist only for bounded unit tests."""

    static_feature_count: int = 591
    hidden_width: int = 192
    conditioning_width: int = 128
    dropout: float = 0.1
    context_steps: int = CONTEXT_20M_STEPS
    input_channels: int = INPUT_CHANNELS
    forecast_steps: int = FORECAST_20M_STEPS
    pooling_factors: tuple[int, ...] = POOLING_FACTORS
    forecast_knots: tuple[int, ...] = FORECAST_KNOTS
    blocks_per_stack: int = BLOCKS_PER_STACK

    def validate(self) -> None:
        if self.static_feature_count < 1 or self.hidden_width < 4 or self.conditioning_width < 4:
            raise ValueError("model dimensions must be positive and nontrivial")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.context_steps != CONTEXT_20M_STEPS:
            raise ValueError("context length differs from the frozen 144-step contract")
        if self.input_channels != INPUT_CHANNELS:
            raise ValueError("input channels differ from the frozen 24-channel contract")
        if self.forecast_steps != FORECAST_20M_STEPS:
            raise ValueError("forecast length differs from the frozen 72-step contract")
        if tuple(self.pooling_factors) != POOLING_FACTORS:
            raise ValueError("pooling factors differ from the frozen 12/4/1 contract")
        if tuple(self.forecast_knots) != FORECAST_KNOTS:
            raise ValueError("forecast knots differ from the frozen 6/18/72 contract")
        if self.blocks_per_stack != BLOCKS_PER_STACK:
            raise ValueError("each stack must contain exactly two blocks")


@dataclass(frozen=True)
class FixedBasisTrainingConfig:
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    use_bf16_on_cuda: bool = True

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer settings are invalid")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")


class _ResidualBasisBlock(nn.Module):
    def __init__(
        self,
        *,
        pooled_steps: int,
        forecast_knots: int,
        config: HierarchicalResidualBasisConfig,
    ) -> None:
        super().__init__()
        self.pooled_steps = int(pooled_steps)
        self.forecast_knots = int(forecast_knots)
        pooled_width = self.pooled_steps * INPUT_CHANNELS
        self.hidden = nn.Sequential(
            nn.Linear(pooled_width + config.conditioning_width, config.hidden_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_width, config.hidden_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.basis = nn.Linear(
            config.hidden_width,
            pooled_width + self.forecast_knots,
        )

    def forward(
        self,
        pooled: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flattened = pooled.reshape(len(pooled), -1)
        hidden = self.hidden(torch.cat([flattened, condition], dim=-1))
        coefficients = self.basis(hidden)
        split = self.pooled_steps * INPUT_CHANNELS
        backcast = coefficients[:, :split].reshape(len(pooled), self.pooled_steps, INPUT_CHANNELS)
        forecast = coefficients[:, split:]
        return backcast, forecast


class HierarchicalResidualBasisForecaster(nn.Module):
    """Three-scale residual-basis network with a direct 72-step forecast path."""

    def __init__(self, config: HierarchicalResidualBasisConfig | None = None) -> None:
        super().__init__()
        self.config = config or HierarchicalResidualBasisConfig()
        self.config.validate()
        cfg = self.config
        self.static_projection = nn.Sequential(
            nn.Linear(2 * cfg.static_feature_count, cfg.conditioning_width),
            nn.GELU(),
            nn.LayerNorm(cfg.conditioning_width),
        )
        self.station_embedding = nn.Embedding(len(STATIONS), cfg.conditioning_width)
        self.amplitude_projection = nn.Sequential(
            nn.Linear(2, cfg.conditioning_width),
            nn.GELU(),
            nn.Linear(cfg.conditioning_width, cfg.conditioning_width),
        )
        self.stacks = nn.ModuleList()
        for factor, knot_count in zip(
            cfg.pooling_factors,
            cfg.forecast_knots,
            strict=True,
        ):
            pooled_steps = CONTEXT_20M_STEPS // factor
            self.stacks.append(
                nn.ModuleList(
                    [
                        _ResidualBasisBlock(
                            pooled_steps=pooled_steps,
                            forecast_knots=knot_count,
                            config=cfg,
                        )
                        for _ in range(cfg.blocks_per_stack)
                    ]
                )
            )

    def _validate_model_inputs(
        self,
        raw: torch.Tensor,
        station_code: torch.Tensor,
        static_scaled: torch.Tensor,
    ) -> None:
        if len(raw) == 0:
            raise ValueError("model batch cannot be empty")
        if station_code.shape != (len(raw),):
            raise ValueError("station code must align one-to-one with raw contexts")
        if station_code.min().item() < 0 or station_code.max().item() >= len(STATIONS):
            raise ValueError("station code lies outside the official station set")
        expected_static = (len(raw), 2 * self.config.static_feature_count)
        if static_scaled.shape != expected_static:
            raise ValueError("scaled static features differ from the model contract")
        if not torch.isfinite(static_scaled).all():
            raise ValueError("scaled static features must be finite")

    def forward_dense(
        self,
        raw: torch.Tensor,
        station_code: torch.Tensor,
        static_scaled: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_model_inputs(raw, station_code, static_scaled)
        prepared = prepare_hierarchical_context(raw)
        amplitude = torch.stack(
            [
                torch.log1p(prepared.current_hs.clamp_min(0.0)),
                torch.log(prepared.hs_scale.clamp_min(1e-6)),
            ],
            dim=-1,
        )
        condition = (
            self.static_projection(static_scaled)
            + self.station_embedding(station_code)
            + self.amplitude_projection(amplitude)
        )

        residual = prepared.values
        forecast = torch.zeros(
            len(raw),
            FORECAST_20M_STEPS,
            dtype=residual.dtype,
            device=residual.device,
        )
        for factor, stack in zip(self.config.pooling_factors, self.stacks, strict=True):
            for block in stack:
                pooled = average_pool_context(residual, factor)
                backcast_knots, forecast_knots = block(pooled, condition)
                backcast = _linear_interpolate_time(
                    backcast_knots,
                    CONTEXT_20M_STEPS,
                )
                residual = residual - backcast
                forecast = forecast + interpolate_forecast_knots(forecast_knots)
        return forecast * prepared.hs_scale[:, None]

    def forward(
        self,
        raw: torch.Tensor,
        station_code: torch.Tensor,
        static_scaled: torch.Tensor,
    ) -> torch.Tensor:
        dense = self.forward_dense(raw, station_code, static_scaled)
        return dense[:, OFFICIAL_FORECAST_INDICES]

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
class FittedHierarchicalResidualBasis:
    model_config: HierarchicalResidualBasisConfig
    training_config: FixedBasisTrainingConfig
    seed: int
    scaler: StaticRobustScaler
    state_dict: dict[str, torch.Tensor]
    training_steps: int
    train_ids_sha256: str
    train_context_sha256: str
    scaler_state_sha256: str
    model_state_sha256: str


def _aligned_case_count(*arrays: np.ndarray) -> int:
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("aligned input arrays have different case counts")
    return lengths.pop()


def _validate_station_array(station: np.ndarray, *, size: int) -> np.ndarray:
    source = np.asarray(station)
    if source.shape != (size,) or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("station codes must be an aligned integer vector")
    result = source.astype(np.int64, copy=False)
    if len(result) == 0 or result.min() < 0 or result.max() >= len(STATIONS):
        raise ValueError("station codes lie outside the official station set")
    return result


def fit_fixed_epoch_hierarchical_model(
    train_raw: np.ndarray,
    train_station: np.ndarray,
    train_static: np.ndarray,
    train_target_delta: np.ndarray,
    train_case_weight: np.ndarray,
    train_case_ids: Sequence[int] | np.ndarray,
    *,
    forbidden_case_ids: Sequence[int] | np.ndarray,
    seed: int,
    device: str | torch.device,
    model_config: HierarchicalResidualBasisConfig | None = None,
    training_config: FixedBasisTrainingConfig | None = None,
    static_scaler: StaticRobustScaler | None = None,
) -> FittedHierarchicalResidualBasis:
    """Fit one cell from physically sliced train-only arrays.

    ``train_target_delta`` must contain only the six official labels for these training
    rows.  Validation targets are intentionally absent from this API.
    """

    raw_array = np.asarray(train_raw)
    station_array = np.asarray(train_station)
    static_array = np.asarray(train_static)
    target_array = np.asarray(train_target_delta)
    weight_array = np.asarray(train_case_weight)
    size = _aligned_case_count(
        raw_array,
        station_array,
        static_array,
        target_array,
        weight_array,
    )
    if size == 0:
        raise ValueError("sliced training arrays cannot be empty")
    config = model_config or HierarchicalResidualBasisConfig()
    training = training_config or FixedBasisTrainingConfig()
    config.validate()
    training.validate()
    if raw_array.shape != (size, CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise ValueError("raw training shape differs from the 48-hour contract")
    station_codes = _validate_station_array(station_array, size=size)
    if static_array.shape != (size, config.static_feature_count):
        raise ValueError("static training features differ from the model contract")
    if target_array.shape != (size, len(LEADS)):
        raise ValueError("training target must contain only the six official deltas")
    if weight_array.shape != (size,):
        raise ValueError("training case weights must be an aligned vector")

    train_ids = _validated_case_ids(
        train_case_ids,
        expected_size=size,
        role="training",
    )
    forbidden_ids = _validated_case_ids(
        forbidden_case_ids,
        expected_size=None,
        role="forbidden",
    )
    if np.intersect1d(train_ids, forbidden_ids).size:
        raise PermissionError("training case IDs overlap forbidden case IDs")
    if not np.isfinite(target_array).all():
        raise ValueError("sliced training targets must be finite")
    if not np.isfinite(weight_array).all() or (weight_array <= 0.0).any():
        raise ValueError("sliced training case weights must be finite and positive")

    if static_scaler is None:
        scaler = StaticRobustScaler.fit(
            static_array,
            train_ids,
            forbidden_case_ids=forbidden_ids,
        )
    else:
        scaler = static_scaler
        if scaler.feature_count != config.static_feature_count:
            raise ValueError("reused static scaler feature count differs")
        if scaler.fit_ids_sha256 != _ids_sha256(train_ids):
            raise PermissionError("reused static scaler was not fit on the exact training IDs")
    scaled_static = scaler.transform(static_array)
    normalized_weight = np.asarray(weight_array, dtype=np.float32)
    normalized_weight = normalized_weight / float(normalized_weight.mean())

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_deterministically(int(seed))
    model = HierarchicalResidualBasisForecaster(config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    train_raw_values = np.asarray(raw_array, dtype=np.float32)
    train_targets = np.asarray(target_array, dtype=np.float32)
    for epoch in range(1, training.epochs + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + epoch)
        order = torch.randperm(size, generator=generator).numpy()
        for start in range(0, size, training.batch_size):
            local = order[start : start + training.batch_size]
            raw_batch = torch.from_numpy(train_raw_values[local]).to(selected_device)
            station_batch = torch.from_numpy(station_codes[local]).to(selected_device)
            static_batch = torch.from_numpy(scaled_static[local]).to(selected_device)
            target_batch = torch.from_numpy(train_targets[local]).to(selected_device)
            weight_batch = torch.from_numpy(normalized_weight[local]).to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.bfloat16,
                enabled=selected_device.type == "cuda" and training.use_bf16_on_cuda,
            ):
                prediction = model(raw_batch, station_batch, static_batch)
                loss = torch.mean(weight_batch[:, None] * torch.square(prediction - target_batch))
            if not torch.isfinite(loss):
                raise RuntimeError("fixed-epoch hierarchical training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
            optimizer.step()

    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    state_sha = model_state_sha256(state)
    del model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return FittedHierarchicalResidualBasis(
        model_config=config,
        training_config=training,
        seed=int(seed),
        scaler=scaler,
        state_dict=state,
        training_steps=int(
            training.epochs * ((size + training.batch_size - 1) // training.batch_size)
        ),
        train_ids_sha256=_ids_sha256(train_ids),
        train_context_sha256=_training_context_sha256(
            train_raw_values,
            station_codes,
            static_array,
        ),
        scaler_state_sha256=scaler.state_sha256,
        model_state_sha256=state_sha,
    )


def predict_with_fitted_hierarchical_model(
    fitted: FittedHierarchicalResidualBasis,
    raw: np.ndarray,
    station: np.ndarray,
    static: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int | None = None,
) -> np.ndarray:
    """Reload the sealed CPU state and predict six official residual deltas."""

    raw_array = np.asarray(raw)
    station_array = np.asarray(station)
    static_array = np.asarray(static)
    size = _aligned_case_count(raw_array, station_array, static_array)
    if size == 0 or raw_array.shape != (size, CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise ValueError("prediction raw contexts differ from the 48-hour contract")
    station_codes = _validate_station_array(station_array, size=size)
    config = fitted.model_config
    if static_array.shape != (size, config.static_feature_count):
        raise ValueError("prediction static features differ from the fitted contract")
    if model_state_sha256(fitted.state_dict) != fitted.model_state_sha256:
        raise PermissionError("fitted model state SHA differs before prediction")
    if fitted.scaler.state_sha256 != fitted.scaler_state_sha256:
        raise PermissionError("fitted static scaler SHA differs before prediction")
    scaled_static = fitted.scaler.transform(static_array)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = HierarchicalResidualBasisForecaster(config).to(selected_device)
    model.load_state_dict(fitted.state_dict, strict=True)
    model.eval()
    use_batch = int(batch_size or fitted.training_config.batch_size)
    if use_batch < 1:
        raise ValueError("prediction batch size must be positive")
    raw_values = np.asarray(raw_array, dtype=np.float32)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, size, use_batch):
            stop = min(start + use_batch, size)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.bfloat16,
                enabled=(
                    selected_device.type == "cuda" and fitted.training_config.use_bf16_on_cuda
                ),
            ):
                prediction = model(
                    torch.from_numpy(raw_values[start:stop]).to(selected_device),
                    torch.from_numpy(station_codes[start:stop]).to(selected_device),
                    torch.from_numpy(scaled_static[start:stop]).to(selected_device),
                )
            outputs.append(prediction.float().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    if result.shape != (size, len(LEADS)) or not np.isfinite(result).all():
        raise RuntimeError("hierarchical prediction shape or finiteness changed")
    del model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def fit_fixed_epoch_and_predict(
    raw: np.ndarray,
    station: np.ndarray,
    static: np.ndarray,
    target_delta: np.ndarray,
    case_weight: np.ndarray,
    train_ids: Sequence[int] | np.ndarray,
    prediction_ids: Sequence[int] | np.ndarray,
    *,
    seed: int,
    device: str | torch.device,
    model_config: HierarchicalResidualBasisConfig | None = None,
    training_config: FixedBasisTrainingConfig | None = None,
) -> tuple[np.ndarray, FittedHierarchicalResidualBasis]:
    """Slice a common cache before calling the train-only core fit.

    Validation target values are never selected or passed to the core.  They may be NaN
    poison values without changing the fitted state or predictions.
    """

    raw_array = np.asarray(raw)
    station_array = np.asarray(station)
    static_array = np.asarray(static)
    target_array = np.asarray(target_delta)
    weight_array = np.asarray(case_weight)
    size = _aligned_case_count(
        raw_array,
        station_array,
        static_array,
        target_array,
        weight_array,
    )
    if target_array.shape != (size, len(LEADS)):
        raise ValueError("aligned target must have six columns")
    train = _validated_indices(train_ids, size=size, role="training")
    prediction = _validated_indices(prediction_ids, size=size, role="prediction")
    if np.intersect1d(train, prediction).size:
        raise PermissionError("training IDs overlap forbidden prediction IDs")

    fitted = fit_fixed_epoch_hierarchical_model(
        np.array(raw_array[train], copy=True),
        np.array(station_array[train], copy=True),
        np.array(static_array[train], copy=True),
        np.array(target_array[train], copy=True),
        np.array(weight_array[train], copy=True),
        train,
        forbidden_case_ids=prediction,
        seed=seed,
        device=device,
        model_config=model_config,
        training_config=training_config,
    )
    predicted = predict_with_fitted_hierarchical_model(
        fitted,
        np.array(raw_array[prediction], copy=True),
        np.array(station_array[prediction], copy=True),
        np.array(static_array[prediction], copy=True),
        device=device,
    )
    return predicted, fitted


def save_fitted_hierarchical_model(
    fitted: FittedHierarchicalResidualBasis,
    path: str | Path,
) -> None:
    """Write one sealed model bundle with exclusive-create semantics."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if model_state_sha256(fitted.state_dict) != fitted.model_state_sha256:
        raise PermissionError("fitted model state SHA differs before save")
    if fitted.scaler.state_sha256 != fitted.scaler_state_sha256:
        raise PermissionError("fitted static scaler SHA differs before save")
    payload: dict[str, Any] = {
        "schema_version": MODEL_BUNDLE_SCHEMA,
        "model_config": asdict(fitted.model_config),
        "training_config": asdict(fitted.training_config),
        "seed": int(fitted.seed),
        "scaler_center": torch.from_numpy(np.asarray(fitted.scaler.center, dtype=np.float32)),
        "scaler_scale": torch.from_numpy(np.asarray(fitted.scaler.scale, dtype=np.float32)),
        "scaler_fit_ids_sha256": fitted.scaler.fit_ids_sha256,
        "scaler_sha256": fitted.scaler.state_sha256,
        "state_dict": fitted.state_dict,
        "training_steps": int(fitted.training_steps),
        "train_ids_sha256": fitted.train_ids_sha256,
        "train_context_sha256": fitted.train_context_sha256,
        "scaler_state_sha256": fitted.scaler_state_sha256,
        "model_state_sha256": fitted.model_state_sha256,
    }
    with target.open("xb") as handle:
        torch.save(payload, handle)


def load_fitted_hierarchical_model(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> FittedHierarchicalResidualBasis:
    """Load a bundle and independently verify scaler and model-state hashes."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != MODEL_BUNDLE_SCHEMA:
        raise ValueError("saved hierarchical model schema differs")
    model_config = HierarchicalResidualBasisConfig(**payload["model_config"])
    training_config = FixedBasisTrainingConfig(**payload["training_config"])
    model_config.validate()
    training_config.validate()
    scaler = StaticRobustScaler(
        payload["scaler_center"].detach().cpu().numpy().astype(np.float32),
        payload["scaler_scale"].detach().cpu().numpy().astype(np.float32),
        str(payload["scaler_fit_ids_sha256"]),
    )
    if scaler.state_sha256 != payload["scaler_sha256"]:
        raise PermissionError("saved static scaler SHA differs")
    if scaler.state_sha256 != payload["scaler_state_sha256"]:
        raise PermissionError("saved fitted scaler-state SHA differs")
    state = {
        str(name): tensor.detach().cpu().clone() for name, tensor in payload["state_dict"].items()
    }
    state_sha = model_state_sha256(state)
    if state_sha != payload["model_state_sha256"]:
        raise PermissionError("saved hierarchical model state SHA differs")
    train_ids_sha = _require_sha256(payload["train_ids_sha256"], field="train_ids_sha256")
    if train_ids_sha != scaler.fit_ids_sha256:
        raise PermissionError("saved training-ID hashes disagree")
    context_sha = _require_sha256(
        payload["train_context_sha256"],
        field="train_context_sha256",
    )
    training_steps = int(payload["training_steps"])
    if training_steps < 1:
        raise ValueError("saved training step count must be positive")
    return FittedHierarchicalResidualBasis(
        model_config=model_config,
        training_config=training_config,
        seed=int(payload["seed"]),
        scaler=scaler,
        state_dict=state,
        training_steps=training_steps,
        train_ids_sha256=train_ids_sha,
        train_context_sha256=context_sha,
        scaler_state_sha256=scaler.state_sha256,
        model_state_sha256=state_sha,
    )


__all__ = [
    "BLOCKS_PER_STACK",
    "CONTEXT_20M_STEPS",
    "FORECAST_20M_STEPS",
    "FORECAST_KNOTS",
    "FixedBasisTrainingConfig",
    "FittedHierarchicalResidualBasis",
    "HierarchicalResidualBasisConfig",
    "HierarchicalResidualBasisForecaster",
    "INPUT_CHANNELS",
    "OFFICIAL_FORECAST_INDICES",
    "OFFICIAL_FORECAST_STEPS",
    "POOLING_FACTORS",
    "PreparedBasisContext",
    "StaticRobustScaler",
    "average_pool_context",
    "extract_past_raw_context",
    "fit_fixed_epoch_and_predict",
    "fit_fixed_epoch_hierarchical_model",
    "interpolate_forecast_knots",
    "load_fitted_hierarchical_model",
    "model_state_sha256",
    "predict_with_fitted_hierarchical_model",
    "prepare_hierarchical_context",
    "save_fitted_hierarchical_model",
]
