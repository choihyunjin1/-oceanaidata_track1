"""Lazy PyTorch sequence models for P1.

Importing this module never imports PyTorch.  The large CUDA wheel is required
only when a model/loss factory or :func:`seed_torch` is called.
"""

from __future__ import annotations

import importlib
import os
import random
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


class OptionalDependencyError(ImportError):
    """Raised when a deep-learning API is used without the PyTorch overlay."""


@dataclass(frozen=True)
class SequenceModelOutput:
    """Per-timestep binary logits and optional five-type auxiliary logits."""

    logits: Any
    aux_logits: Any | None = None


@dataclass(frozen=True)
class TCNConfig:
    input_dim: int
    channels: tuple[int, ...] = (32, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.1
    aux_classes: int = 5
    causal: bool = True

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if not self.channels or any(channel < 1 for channel in self.channels):
            raise ValueError("channels must contain positive integers")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.aux_classes < 0:
            raise ValueError("aux_classes must be non-negative")


@dataclass(frozen=True)
class PatchTransformerConfig:
    input_dim: int
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    patch_size: int = 12
    patch_stride: int = 6
    aux_classes: int = 5
    causal: bool = True

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.d_model < 1 or self.nhead < 1 or self.d_model % self.nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if self.num_layers < 1 or self.dim_feedforward < 1:
            raise ValueError("num_layers and dim_feedforward must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.patch_size < 1 or self.patch_stride < 1:
            raise ValueError("patch_size and patch_stride must be positive")
        if self.aux_classes < 0:
            raise ValueError("aux_classes must be non-negative")


@dataclass(frozen=True)
class LossConfig:
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    auxiliary_weight: float = 0.2
    positive_weight: float | None = None
    smooth: float = 1.0

    def __post_init__(self) -> None:
        if self.bce_weight < 0 or self.dice_weight < 0 or self.auxiliary_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.bce_weight + self.dice_weight <= 0:
            raise ValueError("at least one binary loss weight must be positive")
        if self.positive_weight is not None and self.positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        if self.smooth <= 0:
            raise ValueError("smooth must be positive")


def torch_available() -> bool:
    """Return availability without importing the package."""

    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OptionalDependencyError(
            "PyTorch is required for deep models; install requirements-dl.txt "
            "after the CPU/data requirements"
        ) from exc
    return torch, nn, functional


def seed_torch(seed: int = 20260813, *, deterministic: bool = True) -> None:
    """Seed CPU/CUDA RNGs and request deterministic kernels where available."""

    if deterministic:
        # Must be present before CUDA creates a cuBLAS workspace.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch, _, _ = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def build_tcn(config: TCNConfig) -> Any:
    """Build a residual dilated TCN returning one logit per input timestep."""

    torch, nn, functional = _require_torch()

    class _TemporalBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int, dilation: int) -> None:
            super().__init__()
            self.causal = config.causal
            self.left_padding = (config.kernel_size - 1) * dilation
            symmetric_padding = self.left_padding // 2
            self.conv1 = nn.Conv1d(
                input_channels,
                output_channels,
                config.kernel_size,
                dilation=dilation,
                padding=0 if self.causal else symmetric_padding,
            )
            self.conv2 = nn.Conv1d(
                output_channels,
                output_channels,
                config.kernel_size,
                dilation=dilation,
                padding=0 if self.causal else symmetric_padding,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.activation = nn.GELU()
            self.residual = (
                nn.Identity()
                if input_channels == output_channels
                else nn.Conv1d(input_channels, output_channels, kernel_size=1)
            )
            self.norm = nn.GroupNorm(1, output_channels)

        def _convolve(self, inputs: Any, convolution: Any) -> Any:
            if self.causal:
                inputs = functional.pad(inputs, (self.left_padding, 0))
            output = convolution(inputs)
            if not self.causal and output.shape[-1] != inputs.shape[-1]:
                # Even kernels cannot use a single symmetric integer padding.
                difference = output.shape[-1] - inputs.shape[-1]
                if difference > 0:
                    output = output[..., : inputs.shape[-1]]
                elif difference < 0:
                    output = functional.pad(output, (0, -difference))
            return output

        def forward(self, inputs: Any) -> Any:
            output = self.dropout(self.activation(self._convolve(inputs, self.conv1)))
            output = self.dropout(self.activation(self._convolve(output, self.conv2)))
            return self.activation(self.norm(output + self.residual(inputs)))

    class _TemporalConvNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            input_channels = config.input_dim
            for level, output_channels in enumerate(config.channels):
                blocks.append(_TemporalBlock(input_channels, output_channels, 2**level))
                input_channels = output_channels
            self.blocks = nn.ModuleList(blocks)
            self.binary_head = nn.Conv1d(input_channels, 1, kernel_size=1)
            self.auxiliary_head = (
                nn.Conv1d(input_channels, config.aux_classes, kernel_size=1)
                if config.aux_classes
                else None
            )

        def forward(self, inputs: Any, padding_mask: Any | None = None) -> SequenceModelOutput:
            if inputs.ndim != 3 or inputs.shape[-1] != config.input_dim:
                raise ValueError(f"inputs must have shape (batch, time, {config.input_dim})")
            if padding_mask is not None and tuple(padding_mask.shape) != tuple(inputs.shape[:2]):
                raise ValueError("padding_mask must have shape (batch, time)")
            hidden = inputs.transpose(1, 2)
            for block in self.blocks:
                hidden = block(hidden)
            logits = self.binary_head(hidden).squeeze(1)
            auxiliary = (
                self.auxiliary_head(hidden).transpose(1, 2)
                if self.auxiliary_head is not None
                else None
            )
            return SequenceModelOutput(logits=logits, aux_logits=auxiliary)

    return _TemporalConvNet()


def build_patch_transformer(config: PatchTransformerConfig) -> Any:
    """Build a patch-token Transformer with timestep-aligned anomaly logits."""

    torch, nn, functional = _require_torch()

    class _PatchTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_projection = nn.Linear(
                config.input_dim * config.patch_size,
                config.d_model,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                config.num_layers,
                enable_nested_tensor=False,
            )
            self.normalization = nn.LayerNorm(config.d_model)
            self.binary_head = nn.Linear(config.d_model, 1)
            self.auxiliary_head = (
                nn.Linear(config.d_model, config.aux_classes) if config.aux_classes else None
            )

        @staticmethod
        def _position_encoding(length: int, device: Any, dtype: Any) -> Any:
            position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
            even_indices = torch.arange(0, config.d_model, 2, device=device, dtype=dtype)
            divisor = torch.exp(even_indices * (-float(np.log(10_000.0)) / config.d_model))
            encoding = torch.zeros((length, config.d_model), device=device, dtype=dtype)
            encoding[:, 0::2] = torch.sin(position * divisor)
            odd_width = encoding[:, 1::2].shape[1]
            encoding[:, 1::2] = torch.cos(position * divisor[:odd_width])
            return encoding

        def _patches(self, inputs: Any) -> Any:
            channels_first = inputs.transpose(1, 2)
            if config.causal:
                channels_first = functional.pad(
                    channels_first,
                    (config.patch_size - 1, 0),
                )
            else:
                total = config.patch_size - 1
                channels_first = functional.pad(
                    channels_first,
                    (total // 2, total - total // 2),
                )
            patches = channels_first.unfold(
                dimension=2,
                size=config.patch_size,
                step=config.patch_stride,
            )
            return patches.permute(0, 2, 1, 3).flatten(start_dim=2)

        def _patch_padding_mask(self, padding_mask: Any, patch_count: int) -> Any:
            valid = (~padding_mask.bool()).to(dtype=torch.float32).unsqueeze(1)
            if config.causal:
                valid = functional.pad(valid, (config.patch_size - 1, 0))
            else:
                total = config.patch_size - 1
                valid = functional.pad(valid, (total // 2, total - total // 2))
            patched = valid.unfold(2, config.patch_size, config.patch_stride)
            mask = patched.sum(dim=-1).squeeze(1) == 0
            return mask[:, :patch_count]

        def forward(self, inputs: Any, padding_mask: Any | None = None) -> SequenceModelOutput:
            if inputs.ndim != 3 or inputs.shape[-1] != config.input_dim:
                raise ValueError(f"inputs must have shape (batch, time, {config.input_dim})")
            if padding_mask is not None and tuple(padding_mask.shape) != tuple(inputs.shape[:2]):
                raise ValueError("padding_mask must have shape (batch, time)")
            patches = self._patches(inputs)
            tokens = self.patch_projection(patches)
            tokens = tokens + self._position_encoding(
                tokens.shape[1],
                tokens.device,
                tokens.dtype,
            ).unsqueeze(0)
            causal_mask = None
            if config.causal:
                causal_mask = torch.full(
                    (tokens.shape[1], tokens.shape[1]),
                    float("-inf"),
                    device=tokens.device,
                    dtype=tokens.dtype,
                ).triu(1)
            token_padding = (
                self._patch_padding_mask(padding_mask, tokens.shape[1])
                if padding_mask is not None
                else None
            )
            encoded = self.encoder(
                tokens,
                mask=causal_mask,
                src_key_padding_mask=token_padding,
            )
            encoded = self.normalization(encoded)
            time_index = torch.div(
                torch.arange(inputs.shape[1], device=inputs.device),
                config.patch_stride,
                rounding_mode="floor",
            ).clamp(max=encoded.shape[1] - 1)
            timestep_features = encoded.index_select(1, time_index)
            logits = self.binary_head(timestep_features).squeeze(-1)
            auxiliary = (
                self.auxiliary_head(timestep_features) if self.auxiliary_head is not None else None
            )
            return SequenceModelOutput(logits=logits, aux_logits=auxiliary)

    return _PatchTransformer()


def _unpack_output(output: Any) -> tuple[Any, Any | None]:
    if isinstance(output, SequenceModelOutput):
        return output.logits, output.aux_logits
    if isinstance(output, dict):
        return output["logits"], output.get("aux_logits")
    if isinstance(output, (tuple, list)):
        return output[0], output[1] if len(output) > 1 else None
    return output, None


def build_bce_dice_aux_loss(config: LossConfig | None = None) -> Any:
    """Build masked BCE + soft Dice + multi-label type auxiliary loss."""

    config = config or LossConfig()
    torch, nn, functional = _require_torch()

    class _BCEDiceAuxLoss(nn.Module):
        def forward(
            self,
            output: Any,
            target: Any,
            *,
            auxiliary_target: Any | None = None,
            valid_mask: Any | None = None,
            return_components: bool = False,
        ) -> Any:
            logits, auxiliary_logits = _unpack_output(output)
            target_float = target.to(device=logits.device, dtype=logits.dtype)
            if tuple(target_float.shape) != tuple(logits.shape):
                raise ValueError("target and binary logits must have equal shapes")
            if valid_mask is None:
                valid = torch.ones_like(target_float, dtype=torch.bool)
            else:
                valid = valid_mask.to(device=logits.device, dtype=torch.bool)
                if tuple(valid.shape) != tuple(logits.shape):
                    raise ValueError("valid_mask and logits must have equal shapes")
            if not bool(valid.any()):
                raise ValueError("valid_mask must select at least one timestep")

            positive_weight = (
                None
                if config.positive_weight is None
                else torch.as_tensor(
                    config.positive_weight,
                    device=logits.device,
                    dtype=logits.dtype,
                )
            )
            bce_per_row = functional.binary_cross_entropy_with_logits(
                logits,
                target_float,
                reduction="none",
                pos_weight=positive_weight,
            )
            bce = bce_per_row[valid].mean()
            probability = torch.sigmoid(logits[valid])
            selected_target = target_float[valid]
            intersection = (probability * selected_target).sum()
            dice_score = (2.0 * intersection + config.smooth) / (
                probability.sum() + selected_target.sum() + config.smooth
            )
            dice = 1.0 - dice_score

            auxiliary = logits.new_zeros(())
            if auxiliary_logits is not None and auxiliary_target is not None:
                expected_shape = (*logits.shape, auxiliary_logits.shape[-1])
                if tuple(auxiliary_logits.shape) != expected_shape:
                    raise ValueError("auxiliary logits must have shape (*logits, classes)")
                auxiliary_target_float = auxiliary_target.to(
                    device=auxiliary_logits.device,
                    dtype=auxiliary_logits.dtype,
                )
                if tuple(auxiliary_target_float.shape) != tuple(auxiliary_logits.shape):
                    raise ValueError("auxiliary_target and auxiliary logits must match")
                auxiliary_per_class = functional.binary_cross_entropy_with_logits(
                    auxiliary_logits,
                    auxiliary_target_float,
                    reduction="none",
                )
                auxiliary = auxiliary_per_class[
                    valid.unsqueeze(-1).expand_as(auxiliary_per_class)
                ].mean()
            elif auxiliary_logits is None and auxiliary_target is not None:
                raise ValueError(
                    "auxiliary_target was provided but the model has no auxiliary head"
                )

            total = (
                config.bce_weight * bce
                + config.dice_weight * dice
                + config.auxiliary_weight * auxiliary
            )
            if return_components:
                return {
                    "loss": total,
                    "bce": bce,
                    "dice": dice,
                    "auxiliary": auxiliary,
                }
            return total

    return _BCEDiceAuxLoss()


def build_sequence_model(
    architecture: Literal["tcn", "patch_transformer"],
    config: TCNConfig | PatchTransformerConfig,
) -> Any:
    if architecture == "tcn":
        if not isinstance(config, TCNConfig):
            raise TypeError("tcn requires TCNConfig")
        return build_tcn(config)
    if architecture == "patch_transformer":
        if not isinstance(config, PatchTransformerConfig):
            raise TypeError("patch_transformer requires PatchTransformerConfig")
        return build_patch_transformer(config)
    raise ValueError("architecture must be 'tcn' or 'patch_transformer'")


class TCNClassifier:
    """Constructor-compatible proxy returning a real ``torch.nn.Module``."""

    def __new__(cls, config: TCNConfig | None = None, **kwargs: Any) -> Any:
        if config is not None and kwargs:
            raise TypeError("pass either config or keyword config fields, not both")
        return build_tcn(config or TCNConfig(**kwargs))


class PatchTransformerClassifier:
    """Constructor-compatible proxy returning a real ``torch.nn.Module``."""

    def __new__(
        cls,
        config: PatchTransformerConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        if config is not None and kwargs:
            raise TypeError("pass either config or keyword config fields, not both")
        return build_patch_transformer(config or PatchTransformerConfig(**kwargs))


class BCEDiceAuxLoss:
    """Constructor-compatible proxy returning a real ``torch.nn.Module`` loss."""

    def __new__(cls, config: LossConfig | None = None, **kwargs: Any) -> Any:
        if config is not None and kwargs:
            raise TypeError("pass either config or keyword config fields, not both")
        return build_bce_dice_aux_loss(config or LossConfig(**kwargs))
