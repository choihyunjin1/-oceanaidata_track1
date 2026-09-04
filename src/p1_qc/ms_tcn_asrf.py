"""Offline MS-TCN++/ASRF model and masked training losses for P1.

The module is deliberately data agnostic.  It consumes dense row features in
``[batch, time, feature]`` order and leaves segmentation, fold construction,
normalization, and checkpoint selection to the caller.  All temporal
convolutions are centered because P1 permits whole-series offline QC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

STANDARD_DILATIONS: tuple[int, ...] = tuple(1 << exponent for exponent in range(10))
ANOMALY_TYPE_COUNT = 5
BOUNDARY_COUNT = 2


@dataclass(frozen=True)
class MSTCNASRFConfig:
    """Frozen high-capacity topology for the P1 MS-TCN++ challenger."""

    input_feature_count: int
    width: int = 256
    generator_dilations: tuple[int, ...] = STANDARD_DILATIONS
    refinement_stages: int = 3
    refinement_dilations: tuple[int, ...] = STANDARD_DILATIONS
    kernel_size: int = 3
    dropout: float = 0.5

    def validate(self) -> None:
        if self.input_feature_count < 1:
            raise ValueError("input_feature_count must be positive")
        if self.width < 4:
            raise ValueError("width must be at least four")
        if tuple(self.generator_dilations) != STANDARD_DILATIONS:
            raise ValueError("prediction-generator dilations must be 1..512 over ten layers")
        if tuple(self.refinement_dilations) != STANDARD_DILATIONS:
            raise ValueError("refinement dilations must be 1..512 over ten layers")
        if self.refinement_stages != 3:
            raise ValueError("the registered topology requires exactly three refinement stages")
        if self.kernel_size != 3:
            raise ValueError("the registered topology requires kernel_size=3")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

    @property
    def generator_dual_dilations(self) -> tuple[tuple[int, int], ...]:
        """Ascending/descending dilation pairs used by MS-TCN++."""

        return tuple(zip(self.generator_dilations, reversed(self.generator_dilations), strict=True))

    @property
    def prediction_generator_receptive_field(self) -> int:
        radius = sum(max(left, right) for left, right in self.generator_dual_dilations)
        return 1 + (self.kernel_size - 1) * radius

    @property
    def refinement_stage_receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.refinement_dilations)

    @property
    def final_receptive_field(self) -> int:
        return self.prediction_generator_receptive_field + self.refinement_stages * (
            self.refinement_stage_receptive_field - 1
        )


class _DualDilatedResidualLayer(nn.Module):
    """MS-TCN++ dual-dilation generator layer with same-length padding."""

    def __init__(
        self,
        width: int,
        left_dilation: int,
        right_dilation: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_dilation = int(left_dilation)
        self.right_dilation = int(right_dilation)
        self.left = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=self.left_dilation,
            dilation=self.left_dilation,
        )
        self.right = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=self.right_dilation,
            dilation=self.right_dilation,
        )
        self.fuse = nn.Conv1d(2 * width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        # Match the published MS-TCN++ prediction generator: the two raw
        # dilated branches are fused first and the non-linearity is applied to
        # the fused representation.  Masking every residual state prevents a
        # padded suffix from becoming signal through convolution or bias terms.
        masked_values = torch.where(channel_mask, values, torch.zeros_like(values))
        left = self.left(masked_values)
        right = self.right(masked_values)
        update = F.relu(self.fuse(torch.cat((left, right), dim=1)))
        combined = masked_values + self.dropout(update)
        return torch.where(channel_mask, combined, torch.zeros_like(combined))


class _DilatedResidualLayer(nn.Module):
    """One centered residual layer used inside a refinement stage."""

    def __init__(self, width: int, dilation: int, *, dropout: float) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.dilated = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=self.dilation,
            dilation=self.dilation,
        )
        self.project = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        masked_values = torch.where(channel_mask, values, torch.zeros_like(values))
        update = self.project(F.relu(self.dilated(masked_values)))
        combined = masked_values + self.dropout(update)
        return torch.where(channel_mask, combined, torch.zeros_like(combined))


class _PredictionGenerator(nn.Module):
    def __init__(self, config: MSTCNASRFConfig) -> None:
        super().__init__()
        self.stem = nn.Conv1d(config.input_feature_count, config.width, kernel_size=1)
        self.layers = nn.ModuleList(
            _DualDilatedResidualLayer(
                config.width,
                left,
                right,
                dropout=config.dropout,
            )
            for left, right in config.generator_dual_dilations
        )
        self.row_head = nn.Conv1d(config.width, 1, kernel_size=1)

    def forward(
        self,
        values: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_values = torch.where(channel_mask, values, torch.zeros_like(values))
        hidden = self.stem(masked_values)
        hidden = torch.where(channel_mask, hidden, torch.zeros_like(hidden))
        for layer in self.layers:
            hidden = layer(hidden, channel_mask)
        row_logits = self.row_head(hidden).squeeze(1)
        row_mask = channel_mask.squeeze(1)
        row_logits = torch.where(row_mask, row_logits, torch.zeros_like(row_logits))
        return row_logits, hidden


class _RefinementStage(nn.Module):
    def __init__(self, config: MSTCNASRFConfig) -> None:
        super().__init__()
        self.stem = nn.Conv1d(1, config.width, kernel_size=1)
        self.layers = nn.ModuleList(
            _DilatedResidualLayer(config.width, dilation, dropout=config.dropout)
            for dilation in config.refinement_dilations
        )
        self.row_head = nn.Conv1d(config.width, 1, kernel_size=1)

    def forward(
        self,
        previous_row_logits: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        row_mask = channel_mask.squeeze(1)
        safe_logits = torch.where(
            row_mask,
            previous_row_logits,
            torch.zeros_like(previous_row_logits),
        )
        probabilities = torch.sigmoid(safe_logits).unsqueeze(1)
        probabilities = torch.where(
            channel_mask,
            probabilities,
            torch.zeros_like(probabilities),
        )
        hidden = self.stem(probabilities)
        hidden = torch.where(channel_mask, hidden, torch.zeros_like(hidden))
        for layer in self.layers:
            hidden = layer(hidden, channel_mask)
        row_logits = self.row_head(hidden).squeeze(1)
        return torch.where(row_mask, row_logits, torch.zeros_like(row_logits))


@dataclass(frozen=True)
class MSTCNASRFOutput:
    """All supervised outputs, kept in input row order."""

    stage_logits: tuple[torch.Tensor, ...]
    boundary_logits: torch.Tensor
    type_logits: torch.Tensor

    @property
    def final_logits(self) -> torch.Tensor:
        return self.stage_logits[-1]


class MSTCNASRF(nn.Module):
    """MS-TCN++ prediction generator with ASRF-style auxiliary heads."""

    def __init__(self, config: MSTCNASRFConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.prediction_generator = _PredictionGenerator(config)
        self.refinement_stages = nn.ModuleList(
            _RefinementStage(config) for _ in range(config.refinement_stages)
        )
        self.boundary_head = nn.Conv1d(config.width, BOUNDARY_COUNT, kernel_size=1)
        self.type_head = nn.Conv1d(config.width, ANOMALY_TYPE_COUNT, kernel_size=1)

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

        first_logits, shared = self.prediction_generator(
            values.transpose(1, 2),
            channel_mask,
        )
        stage_logits: list[torch.Tensor] = [first_logits]
        for stage in self.refinement_stages:
            stage_logits.append(stage(stage_logits[-1], channel_mask))
        output_mask = valid_mask.unsqueeze(-1)
        raw_boundary_logits = self.boundary_head(shared).transpose(1, 2)
        boundary_logits = torch.where(
            output_mask,
            raw_boundary_logits,
            torch.zeros_like(raw_boundary_logits),
        )
        raw_type_logits = self.type_head(shared).transpose(1, 2)
        type_logits = torch.where(
            output_mask,
            raw_type_logits,
            torch.zeros_like(raw_type_logits),
        )
        return MSTCNASRFOutput(tuple(stage_logits), boundary_logits, type_logits)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


@dataclass(frozen=True)
class MSTCNASRFLossConfig:
    """Weights for the preregistered multi-stage objective."""

    stage_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    event_bce_weight: float = 1.0
    event_positive_weight: float | None = None
    event_dice_weight: float = 1.0
    smoothing_weight: float = 0.15
    smoothing_tau: float = 4.0
    boundary_weight: float = 0.2
    type_weight: float = 0.2
    auxiliary_positive_weight_cap: float | None = 20.0
    dice_epsilon: float = 1e-6

    def validate(self, *, stage_count: int | None = None) -> None:
        if not self.stage_weights:
            raise ValueError("stage_weights cannot be empty")
        if stage_count is not None and len(self.stage_weights) != stage_count:
            raise ValueError("stage_weights must align with every returned row-logit stage")
        if any(weight < 0.0 for weight in self.stage_weights) or sum(self.stage_weights) <= 0.0:
            raise ValueError("stage_weights must be nonnegative with a positive sum")
        component_weights = (
            self.event_bce_weight,
            self.event_dice_weight,
            self.smoothing_weight,
            self.boundary_weight,
            self.type_weight,
        )
        if any(weight < 0.0 for weight in component_weights):
            raise ValueError("loss component weights cannot be negative")
        if self.event_bce_weight + self.event_dice_weight <= 0.0:
            raise ValueError("at least one event-classification loss must be active")
        if self.event_positive_weight is not None and (
            not math.isfinite(self.event_positive_weight) or self.event_positive_weight <= 0.0
        ):
            raise ValueError("event_positive_weight must be finite and positive when supplied")
        if self.auxiliary_positive_weight_cap is not None and (
            not math.isfinite(self.auxiliary_positive_weight_cap)
            or self.auxiliary_positive_weight_cap < 1.0
        ):
            raise ValueError(
                "auxiliary_positive_weight_cap must be finite and at least one when supplied"
            )
        if self.smoothing_tau <= 0.0 or self.dice_epsilon <= 0.0:
            raise ValueError("smoothing_tau and dice_epsilon must be positive")


@dataclass(frozen=True)
class MSTCNASRFLossOutput:
    total: torch.Tensor
    event: torch.Tensor
    temporal_smoothing: torch.Tensor
    boundary: torch.Tensor
    anomaly_type: torch.Tensor
    stage_event_losses: tuple[torch.Tensor, ...]
    stage_smoothing_losses: tuple[torch.Tensor, ...]


def _validate_row_shape(name: str, value: torch.Tensor, shape: tuple[int, int]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")


def _selected_binary_targets(
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    role: str,
) -> torch.Tensor:
    selected = targets.masked_select(mask).float()
    if selected.numel() == 0:
        return selected
    if not bool(torch.isfinite(selected).all()):
        raise ValueError(f"valid {role} targets must be finite")
    if bool(((selected < 0.0) | (selected > 1.0)).any()):
        raise ValueError(f"valid {role} targets must lie in [0, 1]")
    return selected


def _masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    role: str,
    positive_weight: float | None = None,
) -> torch.Tensor:
    selected_logits = logits.masked_select(mask).float()
    selected_targets = _selected_binary_targets(targets, mask, role=role)
    if selected_logits.numel() == 0:
        return torch.nan_to_num(logits.float()).sum() * 0.0
    pos_weight = None
    if positive_weight is not None:
        pos_weight = selected_logits.new_tensor(float(positive_weight))
    return F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        pos_weight=pos_weight,
    )


def _balanced_positive_weight(
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    role: str,
    cap: float | None,
) -> float | None:
    """Return a clipped negative/positive ratio over the supervised entries."""

    if cap is None:
        return None
    selected = _selected_binary_targets(targets, mask, role=role)
    if selected.numel() == 0:
        return None
    positive = float(selected.sum().detach().cpu())
    negative = float(selected.numel()) - positive
    if positive <= 0.0 or negative <= 0.0:
        return 1.0
    return min(float(cap), max(1.0, negative / positive))


def _soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    selected_logits = logits.masked_select(valid_mask).float()
    selected_targets = _selected_binary_targets(targets, valid_mask, role="event")
    probabilities = torch.sigmoid(selected_logits)
    intersection = torch.sum(probabilities * selected_targets)
    denominator = torch.sum(probabilities) + torch.sum(selected_targets)
    return 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)


def _truncated_log_probability_smoothing(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    tau: float,
) -> torch.Tensor:
    if logits.shape[1] < 2:
        return torch.nan_to_num(logits.float()).sum() * 0.0
    safe_logits = torch.where(valid_mask, logits, torch.zeros_like(logits)).float()
    two_class_logits = torch.stack((torch.zeros_like(safe_logits), safe_logits), dim=-1)
    log_probabilities = F.log_softmax(two_class_logits, dim=-1)
    differences = log_probabilities[:, 1:] - log_probabilities[:, :-1].detach()
    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    if not bool(pair_mask.any()):
        return torch.nan_to_num(logits.float()).sum() * 0.0
    per_pair = torch.clamp(differences.square(), min=0.0, max=tau * tau).mean(dim=-1)
    return per_pair.masked_select(pair_mask).mean()


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
    """Compute the masked multi-stage loss.

    ``valid_mask`` is mandatory and is applied before every reduction.  A
    ``type_mask`` can be either ``[B,T]`` (masking complete rows) or
    ``[B,T,5]`` (masking individual type annotations).  When it is omitted,
    type supervision is restricted to valid event-positive rows because an
    anomaly type is undefined for normal rows.  Boundary and type BCE use a
    batch-observed negative/positive ratio clipped by
    ``auxiliary_positive_weight_cap``; set the cap to ``None`` to request
    unweighted auxiliary BCE.  Invalid/padded target values are never
    inspected.
    """

    if not output.stage_logits:
        raise ValueError("output must contain at least one row-logit stage")
    row_shape = tuple(output.stage_logits[0].shape)
    if len(row_shape) != 2:
        raise ValueError("stage row logits must have [batch, time] shape")
    for index, logits in enumerate(output.stage_logits):
        _validate_row_shape(f"stage_logits[{index}]", logits, row_shape)
    _validate_row_shape("event_targets", event_targets, row_shape)
    if valid_mask is None:
        raise ValueError("valid_mask is mandatory")
    _validate_row_shape("valid_mask", valid_mask, row_shape)
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must be boolean")
    if not bool(valid_mask.any()):
        raise ValueError("valid_mask must select at least one row")
    expected_boundary_shape = (*row_shape, BOUNDARY_COUNT)
    expected_type_shape = (*row_shape, ANOMALY_TYPE_COUNT)
    if tuple(output.boundary_logits.shape) != expected_boundary_shape:
        raise ValueError(f"boundary_logits must have shape {expected_boundary_shape}")
    if tuple(boundary_targets.shape) != expected_boundary_shape:
        raise ValueError(f"boundary_targets must have shape {expected_boundary_shape}")
    if tuple(output.type_logits.shape) != expected_type_shape:
        raise ValueError(f"type_logits must have shape {expected_type_shape}")
    if tuple(type_targets.shape) != expected_type_shape:
        raise ValueError(f"type_targets must have shape {expected_type_shape}")

    loss_config = config or MSTCNASRFLossConfig()
    loss_config.validate(stage_count=len(output.stage_logits))
    normalized_stage_weights = tuple(
        weight / sum(loss_config.stage_weights) for weight in loss_config.stage_weights
    )

    stage_event_losses: list[torch.Tensor] = []
    stage_smoothing_losses: list[torch.Tensor] = []
    for logits in output.stage_logits:
        bce = _masked_bce(
            logits,
            event_targets,
            valid_mask,
            role="event",
            positive_weight=loss_config.event_positive_weight,
        )
        dice = _soft_dice_loss(
            logits,
            event_targets,
            valid_mask,
            epsilon=loss_config.dice_epsilon,
        )
        stage_event_losses.append(
            loss_config.event_bce_weight * bce + loss_config.event_dice_weight * dice
        )
        stage_smoothing_losses.append(
            _truncated_log_probability_smoothing(
                logits,
                valid_mask,
                tau=loss_config.smoothing_tau,
            )
        )

    event_loss = sum(
        weight * loss
        for weight, loss in zip(normalized_stage_weights, stage_event_losses, strict=True)
    )
    smoothing_loss = sum(
        weight * loss
        for weight, loss in zip(normalized_stage_weights, stage_smoothing_losses, strict=True)
    )
    boundary_mask = valid_mask.unsqueeze(-1).expand_as(output.boundary_logits)
    boundary_positive_weight = _balanced_positive_weight(
        boundary_targets,
        boundary_mask,
        role="boundary",
        cap=loss_config.auxiliary_positive_weight_cap,
    )
    boundary_loss = _masked_bce(
        output.boundary_logits,
        boundary_targets,
        boundary_mask,
        role="boundary",
        positive_weight=boundary_positive_weight,
    )

    if type_mask is None:
        safe_event_targets = torch.where(
            valid_mask,
            event_targets,
            torch.zeros_like(event_targets),
        )
        event_positive_rows = valid_mask & (safe_event_targets > 0.5)
        expanded_type_mask = event_positive_rows.unsqueeze(-1).expand_as(output.type_logits)
    elif tuple(type_mask.shape) == row_shape:
        if type_mask.dtype is not torch.bool:
            raise TypeError("type_mask must be boolean")
        expanded_type_mask = (valid_mask & type_mask).unsqueeze(-1).expand_as(output.type_logits)
    elif tuple(type_mask.shape) == expected_type_shape:
        if type_mask.dtype is not torch.bool:
            raise TypeError("type_mask must be boolean")
        expanded_type_mask = valid_mask.unsqueeze(-1) & type_mask
    else:
        raise ValueError("type_mask must have [B,T] or [B,T,5] shape")
    type_positive_weight = _balanced_positive_weight(
        type_targets,
        expanded_type_mask,
        role="anomaly-type",
        cap=loss_config.auxiliary_positive_weight_cap,
    )
    type_loss = _masked_bce(
        output.type_logits,
        type_targets,
        expanded_type_mask,
        role="anomaly-type",
        positive_weight=type_positive_weight,
    )

    total = (
        event_loss
        + loss_config.smoothing_weight * smoothing_loss
        + loss_config.boundary_weight * boundary_loss
        + loss_config.type_weight * type_loss
    )
    return MSTCNASRFLossOutput(
        total=total,
        event=event_loss,
        temporal_smoothing=smoothing_loss,
        boundary=boundary_loss,
        anomaly_type=type_loss,
        stage_event_losses=tuple(stage_event_losses),
        stage_smoothing_losses=tuple(stage_smoothing_losses),
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
