"""Warm-startable type/boundary-conditioned MS-TCN refinement for P1.

This module is a drop-in research alternative to :mod:`p1_qc.ms_tcn_asrf`.
It deliberately preserves the public config, output, and loss API used by the
sealed runner while changing only the information available to refinement
stages.  Each refinement stage receives the previous row, boundary, and
multi-label type posteriors plus a compact projection of the prediction
generator feature map.

The ``initialize_from_baseline_state_dict`` method embeds a trained baseline
MS-TCN++/ASRF model as an exact initial row predictor.  New input-channel
weights and boundary/type residual heads start at zero, so the cascade cannot
change the baseline merely because its tensors are wider.  Fine-tuning must
learn a non-zero use of the new information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from p1_qc import ms_tcn_asrf as baseline

ANOMALY_TYPE_COUNT = baseline.ANOMALY_TYPE_COUNT
BOUNDARY_COUNT = baseline.BOUNDARY_COUNT
STANDARD_DILATIONS = baseline.STANDARD_DILATIONS
MSTCNASRFConfig = baseline.MSTCNASRFConfig
MSTCNASRFLossConfig = baseline.MSTCNASRFLossConfig
MSTCNASRFLossOutput = baseline.MSTCNASRFLossOutput


@dataclass(frozen=True)
class MSTCNASRFOutput:
    """Row stages and the auxiliary logits produced at every cascade stage."""

    stage_logits: tuple[torch.Tensor, ...]
    boundary_logits: torch.Tensor
    type_logits: torch.Tensor
    stage_boundary_logits: tuple[torch.Tensor, ...]
    stage_type_logits: tuple[torch.Tensor, ...]

    @property
    def final_logits(self) -> torch.Tensor:
        return self.stage_logits[-1]


class _TypeBoundaryContextRefinementStage(nn.Module):
    """Refine rows while carrying boundary/type state and generator context."""

    def __init__(self, config: MSTCNASRFConfig, *, context_channels: int) -> None:
        super().__init__()
        input_channels = 1 + BOUNDARY_COUNT + ANOMALY_TYPE_COUNT + context_channels
        self.stem = nn.Conv1d(input_channels, config.width, kernel_size=1)
        self.layers = nn.ModuleList(
            baseline._DilatedResidualLayer(  # noqa: SLF001 - intentional pinned reuse
                config.width,
                dilation,
                dropout=config.dropout,
            )
            for dilation in config.refinement_dilations
        )
        self.row_head = nn.Conv1d(config.width, 1, kernel_size=1)
        self.boundary_delta_head = nn.Conv1d(
            config.width, BOUNDARY_COUNT, kernel_size=1
        )
        self.type_delta_head = nn.Conv1d(
            config.width, ANOMALY_TYPE_COUNT, kernel_size=1
        )
        nn.init.zeros_(self.boundary_delta_head.weight)
        nn.init.zeros_(self.boundary_delta_head.bias)
        nn.init.zeros_(self.type_delta_head.weight)
        nn.init.zeros_(self.type_delta_head.bias)

    def forward(
        self,
        previous_row_logits: torch.Tensor,
        previous_boundary_logits: torch.Tensor,
        previous_type_logits: torch.Tensor,
        context: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_mask = channel_mask.squeeze(1)
        output_mask = row_mask.unsqueeze(-1)
        safe_row = torch.where(
            row_mask, previous_row_logits, torch.zeros_like(previous_row_logits)
        )
        safe_boundary = torch.where(
            output_mask,
            previous_boundary_logits,
            torch.zeros_like(previous_boundary_logits),
        )
        safe_type = torch.where(
            output_mask,
            previous_type_logits,
            torch.zeros_like(previous_type_logits),
        )
        inputs = torch.cat(
            (
                torch.sigmoid(safe_row).unsqueeze(1),
                torch.sigmoid(safe_boundary).transpose(1, 2),
                torch.sigmoid(safe_type).transpose(1, 2),
                torch.where(channel_mask, context, torch.zeros_like(context)),
            ),
            dim=1,
        )
        hidden = self.stem(inputs)
        hidden = torch.where(channel_mask, hidden, torch.zeros_like(hidden))
        for layer in self.layers:
            hidden = layer(hidden, channel_mask)

        row_logits = self.row_head(hidden).squeeze(1)
        row_logits = torch.where(row_mask, row_logits, torch.zeros_like(row_logits))
        boundary_logits = safe_boundary + self.boundary_delta_head(hidden).transpose(1, 2)
        type_logits = safe_type + self.type_delta_head(hidden).transpose(1, 2)
        boundary_logits = torch.where(
            output_mask, boundary_logits, torch.zeros_like(boundary_logits)
        )
        type_logits = torch.where(output_mask, type_logits, torch.zeros_like(type_logits))
        return row_logits, boundary_logits, type_logits


class MSTCNASRF(nn.Module):
    """MS-TCN++ generator with feature/type/boundary-aware refinement."""

    def __init__(self, config: MSTCNASRFConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.context_channels = max(8, min(64, config.width // 8))
        self.prediction_generator = baseline._PredictionGenerator(  # noqa: SLF001
            config
        )
        self.context_projection = nn.Conv1d(
            config.width, self.context_channels, kernel_size=1
        )
        self.boundary_head = nn.Conv1d(config.width, BOUNDARY_COUNT, kernel_size=1)
        self.type_head = nn.Conv1d(config.width, ANOMALY_TYPE_COUNT, kernel_size=1)
        self.refinement_stages = nn.ModuleList(
            _TypeBoundaryContextRefinementStage(
                config,
                context_channels=self.context_channels,
            )
            for _ in range(config.refinement_stages)
        )

    def forward(
        self,
        values: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> MSTCNASRFOutput:
        if values.ndim != 3:
            raise ValueError("input must have [batch, time, feature] shape")
        if values.shape[2] != self.config.input_feature_count:
            raise ValueError("input feature count differs from the model configuration")
        if values.shape[1] < 1:
            raise ValueError("input time dimension cannot be empty")
        if not torch.is_floating_point(values):
            raise TypeError("input features must use a floating dtype")

        row_shape = (values.shape[0], values.shape[1])
        if valid_mask is None:
            valid_mask = torch.ones(row_shape, dtype=torch.bool, device=values.device)
        elif tuple(valid_mask.shape) != row_shape:
            raise ValueError(f"valid_mask must have shape {row_shape}")
        elif valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must be boolean")
        elif valid_mask.device != values.device:
            raise ValueError("valid_mask and input features must use the same device")

        channel_mask = valid_mask.unsqueeze(1)
        output_mask = valid_mask.unsqueeze(-1)
        first_row, shared = self.prediction_generator(
            values.transpose(1, 2), channel_mask
        )
        context = self.context_projection(shared)
        context = torch.where(channel_mask, context, torch.zeros_like(context))
        first_boundary = self.boundary_head(shared).transpose(1, 2)
        first_type = self.type_head(shared).transpose(1, 2)
        first_boundary = torch.where(
            output_mask, first_boundary, torch.zeros_like(first_boundary)
        )
        first_type = torch.where(output_mask, first_type, torch.zeros_like(first_type))

        row_stages: list[torch.Tensor] = [first_row]
        boundary_stages: list[torch.Tensor] = [first_boundary]
        type_stages: list[torch.Tensor] = [first_type]
        for stage in self.refinement_stages:
            row_logits, boundary_logits, type_logits = stage(
                row_stages[-1],
                boundary_stages[-1],
                type_stages[-1],
                context,
                channel_mask,
            )
            row_stages.append(row_logits)
            boundary_stages.append(boundary_logits)
            type_stages.append(type_logits)

        return MSTCNASRFOutput(
            stage_logits=tuple(row_stages),
            boundary_logits=boundary_stages[-1],
            type_logits=type_stages[-1],
            stage_boundary_logits=tuple(boundary_stages),
            stage_type_logits=tuple(type_stages),
        )

    def initialize_from_baseline_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Embed a baseline checkpoint while zero-opening every new pathway."""

        current = self.state_dict()
        copied: list[str] = []
        custom: list[str] = []
        for name, target in current.items():
            source = state_dict.get(name)
            if source is not None and tuple(source.shape) == tuple(target.shape):
                target.copy_(source)
                copied.append(name)

        for index, stage in enumerate(self.refinement_stages):
            source_name = f"refinement_stages.{index}.stem.weight"
            source = state_dict.get(source_name)
            if source is None or tuple(source.shape[0:1] + source.shape[2:]) != tuple(
                stage.stem.weight.shape[0:1] + stage.stem.weight.shape[2:]
            ):
                raise ValueError(f"baseline checkpoint lacks compatible {source_name}")
            if source.shape[1] != 1:
                raise ValueError("baseline refinement stem must consume one row channel")
            with torch.no_grad():
                stage.stem.weight.zero_()
                stage.stem.weight[:, :1].copy_(source)
                stage.boundary_delta_head.weight.zero_()
                stage.boundary_delta_head.bias.zero_()
                stage.type_delta_head.weight.zero_()
                stage.type_delta_head.bias.zero_()
            custom.append(source_name)

        missing_baseline = sorted(
            name
            for name in state_dict
            if name.startswith(("prediction_generator.", "refinement_stages.", "boundary_head.", "type_head."))
            and name not in copied
            and not name.endswith("stem.weight")
        )
        if missing_baseline:
            raise ValueError(f"baseline checkpoint was not fully embedded: {missing_baseline[:5]}")
        return {
            "copied_tensor_count": len(copied),
            "custom_stem_count": len(custom),
            "new_context_channels": self.context_channels,
            "baseline_tensor_count": len(state_dict),
            "exact_initial_row_predictor": True,
            "new_pathways_zero_opened": True,
        }

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def _auxiliary_masks(
    event_targets: torch.Tensor,
    boundary_logits: torch.Tensor,
    type_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    type_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    boundary_mask = valid_mask.unsqueeze(-1).expand_as(boundary_logits)
    if type_mask is None:
        safe_events = torch.where(valid_mask, event_targets, torch.zeros_like(event_targets))
        selected_types = valid_mask & (safe_events > 0.5)
        expanded_type_mask = selected_types.unsqueeze(-1).expand_as(type_logits)
    elif tuple(type_mask.shape) == tuple(valid_mask.shape):
        if type_mask.dtype is not torch.bool:
            raise TypeError("type_mask must be boolean")
        expanded_type_mask = (valid_mask & type_mask).unsqueeze(-1).expand_as(type_logits)
    elif tuple(type_mask.shape) == tuple(type_logits.shape):
        if type_mask.dtype is not torch.bool:
            raise TypeError("type_mask must be boolean")
        expanded_type_mask = valid_mask.unsqueeze(-1) & type_mask
    else:
        raise ValueError("type_mask must have [B,T] or [B,T,5] shape")
    return boundary_mask, expanded_type_mask


def compute_ms_tcn_asrf_loss(
    output: MSTCNASRFOutput,
    event_targets: torch.Tensor,
    boundary_targets: torch.Tensor,
    type_targets: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    type_mask: torch.Tensor | None = None,
    config: MSTCNASRFLossConfig | None = None,
) -> MSTCNASRFLossOutput:
    """Deep-supervise boundary/type state at every generator/cascade stage."""

    loss_config = config or MSTCNASRFLossConfig()
    base_output = baseline.MSTCNASRFOutput(
        stage_logits=output.stage_logits,
        boundary_logits=output.boundary_logits,
        type_logits=output.type_logits,
    )
    base_loss = baseline.compute_ms_tcn_asrf_loss(
        base_output,
        event_targets,
        boundary_targets,
        type_targets,
        valid_mask=valid_mask,
        type_mask=type_mask,
        config=loss_config,
    )
    if len(output.stage_boundary_logits) != len(output.stage_logits):
        raise ValueError("boundary stages must align with row stages")
    if len(output.stage_type_logits) != len(output.stage_logits):
        raise ValueError("type stages must align with row stages")

    boundary_mask, expanded_type_mask = _auxiliary_masks(
        event_targets,
        output.boundary_logits,
        output.type_logits,
        valid_mask,
        type_mask,
    )
    boundary_positive_weight = baseline._balanced_positive_weight(  # noqa: SLF001
        boundary_targets,
        boundary_mask,
        role="boundary",
        cap=loss_config.auxiliary_positive_weight_cap,
    )
    type_positive_weight = baseline._balanced_positive_weight(  # noqa: SLF001
        type_targets,
        expanded_type_mask,
        role="anomaly-type",
        cap=loss_config.auxiliary_positive_weight_cap,
    )
    boundary_losses = tuple(
        baseline._masked_bce(  # noqa: SLF001
            logits,
            boundary_targets,
            boundary_mask,
            role="boundary",
            positive_weight=boundary_positive_weight,
        )
        for logits in output.stage_boundary_logits
    )
    type_losses = tuple(
        baseline._masked_bce(  # noqa: SLF001
            logits,
            type_targets,
            expanded_type_mask,
            role="anomaly-type",
            positive_weight=type_positive_weight,
        )
        for logits in output.stage_type_logits
    )
    boundary_loss = sum(boundary_losses) / len(boundary_losses)
    type_loss = sum(type_losses) / len(type_losses)
    total = (
        base_loss.event
        + loss_config.smoothing_weight * base_loss.temporal_smoothing
        + loss_config.boundary_weight * boundary_loss
        + loss_config.type_weight * type_loss
    )
    return MSTCNASRFLossOutput(
        total=total,
        event=base_loss.event,
        temporal_smoothing=base_loss.temporal_smoothing,
        boundary=boundary_loss,
        anomaly_type=type_loss,
        stage_event_losses=base_loss.stage_event_losses,
        stage_smoothing_losses=base_loss.stage_smoothing_losses,
    )


__all__ = [
    "ANOMALY_TYPE_COUNT",
    "BOUNDARY_COUNT",
    "MSTCNASRF",
    "MSTCNASRFConfig",
    "MSTCNASRFLossConfig",
    "MSTCNASRFLossOutput",
    "MSTCNASRFOutput",
    "STANDARD_DILATIONS",
    "compute_ms_tcn_asrf_loss",
]
