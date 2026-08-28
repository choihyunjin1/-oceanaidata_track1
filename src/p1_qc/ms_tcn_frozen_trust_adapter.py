"""Frozen e150 MS-TCN with a bounded type/boundary/context residual adapter.

The baseline network is kept in evaluation mode and every baseline parameter
is frozen.  Three small temporal adapters can change row logits by at most
``0.25`` per stage, creating an explicit trust region around the official e150
lineage instead of fine-tuning all 52 million baseline parameters.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from p1_qc import ms_tcn_asrf as baseline
from p1_qc.ms_tcn_type_boundary_cascade import (
    MSTCNASRFOutput,
    compute_ms_tcn_asrf_loss,
)

ANOMALY_TYPE_COUNT = baseline.ANOMALY_TYPE_COUNT
BOUNDARY_COUNT = baseline.BOUNDARY_COUNT
STANDARD_DILATIONS = baseline.STANDARD_DILATIONS
MSTCNASRFConfig = baseline.MSTCNASRFConfig
MSTCNASRFLossConfig = baseline.MSTCNASRFLossConfig
MSTCNASRFLossOutput = baseline.MSTCNASRFLossOutput
ADAPTER_DILATIONS: tuple[int, ...] = tuple(1 << exponent for exponent in range(8))


class _BoundedAdapterStage(nn.Module):
    def __init__(
        self,
        *,
        context_channels: int,
        width: int,
        dropout: float,
        row_logit_cap: float,
        auxiliary_logit_cap: float,
    ) -> None:
        super().__init__()
        self.row_logit_cap = float(row_logit_cap)
        self.auxiliary_logit_cap = float(auxiliary_logit_cap)
        input_channels = 1 + BOUNDARY_COUNT + ANOMALY_TYPE_COUNT + context_channels
        self.stem = nn.Conv1d(input_channels, width, kernel_size=1)
        self.layers = nn.ModuleList(
            baseline._DilatedResidualLayer(width, dilation, dropout=dropout)  # noqa: SLF001
            for dilation in ADAPTER_DILATIONS
        )
        self.row_delta_head = nn.Conv1d(width, 1, kernel_size=1)
        self.boundary_delta_head = nn.Conv1d(width, BOUNDARY_COUNT, kernel_size=1)
        self.type_delta_head = nn.Conv1d(width, ANOMALY_TYPE_COUNT, kernel_size=1)
        for head in (
            self.row_delta_head,
            self.boundary_delta_head,
            self.type_delta_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        row_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        type_logits: torch.Tensor,
        context: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_mask = channel_mask.squeeze(1)
        output_mask = row_mask.unsqueeze(-1)
        safe_row = torch.where(row_mask, row_logits, torch.zeros_like(row_logits))
        safe_boundary = torch.where(
            output_mask, boundary_logits, torch.zeros_like(boundary_logits)
        )
        safe_type = torch.where(output_mask, type_logits, torch.zeros_like(type_logits))
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
        row_delta = self.row_logit_cap * torch.tanh(self.row_delta_head(hidden).squeeze(1))
        boundary_delta = self.auxiliary_logit_cap * torch.tanh(
            self.boundary_delta_head(hidden).transpose(1, 2)
        )
        type_delta = self.auxiliary_logit_cap * torch.tanh(
            self.type_delta_head(hidden).transpose(1, 2)
        )
        next_row = torch.where(
            row_mask, safe_row + row_delta, torch.zeros_like(safe_row)
        )
        next_boundary = torch.where(
            output_mask,
            safe_boundary + boundary_delta,
            torch.zeros_like(safe_boundary),
        )
        next_type = torch.where(
            output_mask, safe_type + type_delta, torch.zeros_like(safe_type)
        )
        return next_row, next_boundary, next_type


class MSTCNASRF(nn.Module):
    """Drop-in model API whose only trainable parameters are bounded adapters."""

    def __init__(self, config: MSTCNASRFConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.baseline = baseline.MSTCNASRF(config)
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.context_channels = max(8, min(32, config.width // 16))
        self.adapter_width = max(32, min(96, config.width // 8))
        self.context_projection = nn.Conv1d(
            config.width, self.context_channels, kernel_size=1
        )
        self.adapter_stages = nn.ModuleList(
            _BoundedAdapterStage(
                context_channels=self.context_channels,
                width=self.adapter_width,
                dropout=0.2,
                row_logit_cap=0.25,
                auxiliary_logit_cap=0.5,
            )
            for _ in range(config.refinement_stages)
        )
        self.baseline.eval()

    def train(self, mode: bool = True) -> MSTCNASRF:
        super().train(mode)
        self.baseline.eval()
        return self

    def initialize_from_baseline_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        self.baseline.load_state_dict(state_dict, strict=True)
        self.baseline.eval()
        return {
            "copied_tensor_count": len(state_dict),
            "frozen_baseline_parameter_count": sum(
                parameter.numel() for parameter in self.baseline.parameters()
            ),
            "trainable_adapter_parameter_count": self.trainable_parameter_count,
            "new_context_channels": self.context_channels,
            "adapter_width": self.adapter_width,
            "row_logit_total_cap": 0.25 * len(self.adapter_stages),
            "exact_initial_row_predictor": True,
            "new_pathways_zero_opened": True,
            "baseline_frozen": True,
        }

    def _frozen_baseline(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        channel_mask = valid_mask.unsqueeze(1)
        output_mask = valid_mask.unsqueeze(-1)
        self.baseline.eval()
        with torch.no_grad():
            first_row, shared = self.baseline.prediction_generator(
                values.transpose(1, 2), channel_mask
            )
            row = first_row
            for stage in self.baseline.refinement_stages:
                row = stage(row, channel_mask)
            boundary = self.baseline.boundary_head(shared).transpose(1, 2)
            anomaly_type = self.baseline.type_head(shared).transpose(1, 2)
            boundary = torch.where(
                output_mask, boundary, torch.zeros_like(boundary)
            )
            anomaly_type = torch.where(
                output_mask, anomaly_type, torch.zeros_like(anomaly_type)
            )
        return row, boundary, anomaly_type, shared.detach()

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
        row, boundary, anomaly_type, shared = self._frozen_baseline(values, valid_mask)
        context = self.context_projection(shared)
        context = torch.where(channel_mask, context, torch.zeros_like(context))
        rows = [row]
        boundaries = [boundary]
        types = [anomaly_type]
        for stage in self.adapter_stages:
            row, boundary, anomaly_type = stage(
                rows[-1], boundaries[-1], types[-1], context, channel_mask
            )
            rows.append(row)
            boundaries.append(boundary)
            types.append(anomaly_type)
        return MSTCNASRFOutput(
            stage_logits=tuple(rows),
            boundary_logits=boundaries[-1],
            type_logits=types[-1],
            stage_boundary_logits=tuple(boundaries),
            stage_type_logits=tuple(types),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


__all__ = [
    "ADAPTER_DILATIONS",
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
