"""Masked-sequence pretraining followed by binary-event fine-tuning for P1 Gen4."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from p1_qc.binary_event_tcn import (
    _balanced_auxiliary_positive_weight,
    _CenteredDepthwiseBlock,
    _DenseEpochSampler,
    build_prefix_event_boundary_targets,
)
from p1_qc.temporal_event_tcn import (
    PrefixRobustScaler,
    SequenceLayout,
    ids_sha256,
    model_state_sha256,
)


def _ids(values: Sequence[int] | np.ndarray, *, size: int, role: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{role} IDs must be a non-empty vector")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError(f"{role} IDs must be integers")
    result = result.astype(np.int64, copy=False)
    if np.unique(result).size != len(result):
        raise ValueError(f"{role} IDs must be unique")
    if result.min() < 0 or result.max() >= size:
        raise IndexError(f"{role} IDs are outside the aligned arrays")
    return result


@dataclass(frozen=True)
class MaskedPretrainModelConfig:
    input_feature_count: int
    group_count: int
    width: int = 64
    group_embedding_width: int = 24
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 3
    dropout: float = 0.1
    norm_groups: int = 8

    @property
    def receptive_field_rows(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(int(value) for value in self.dilations)

    def validate(self) -> None:
        if self.input_feature_count < 1 or self.group_count < 1 or self.width < 8:
            raise ValueError("model dimensions must be positive and nontrivial")
        if self.kernel_size != 3 or self.width % self.norm_groups:
            raise ValueError("kernel or group-normalization contract differs")
        if tuple(self.dilations) != (1, 2, 4, 8) or self.receptive_field_rows != 31:
            raise ValueError("the Gen4 receptive field is frozen at 31 ten-minute rows")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")


@dataclass(frozen=True)
class MaskedPretrainTrainingConfig:
    optimizer_steps: int = 120
    pretrain_steps: int = 30
    finetune_steps: int = 90
    batch_size: int = 8192
    pretrain_learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    mask_probability: float = 0.3
    auxiliary_loss_weight: float = 0.1
    boundary_band_rows: int = 6

    def validate(self) -> None:
        if self.optimizer_steps < 2 or self.batch_size < 8:
            raise ValueError("optimizer steps and batch size must be positive")
        if self.pretrain_steps < 1 or self.finetune_steps < 1:
            raise ValueError("both training stages require optimizer steps")
        if self.pretrain_steps + self.finetune_steps != self.optimizer_steps:
            raise ValueError("pretrain plus fine-tune steps must equal total optimizer steps")
        if self.pretrain_learning_rate <= 0.0 or self.finetune_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("optimizer regularization differs")
        if not 0.0 < self.mask_probability < 1.0:
            raise ValueError("mask probability must lie strictly inside (0,1)")
        if not 0.0 <= self.auxiliary_loss_weight <= 1.0:
            raise ValueError("auxiliary loss weight must lie in [0,1]")
        if self.boundary_band_rows < 1:
            raise ValueError("boundary band rows must be positive")


class MaskedPretrainBinaryEventModel(nn.Module):
    """One encoder with a pretraining decoder and event/auxiliary fine-tune heads."""

    def __init__(self, config: MaskedPretrainModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.stem = nn.Conv1d(2 * config.input_feature_count, config.width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _CenteredDepthwiseBlock(
                    config.width,
                    dilation,
                    norm_groups=config.norm_groups,
                    dropout=config.dropout,
                )
                for dilation in config.dilations
            ]
        )
        self.group_embedding = nn.Embedding(config.group_count, config.group_embedding_width)
        self.shared_head = nn.Sequential(
            nn.Linear(config.width + config.group_embedding_width, config.width),
            nn.GELU(),
            nn.LayerNorm(config.width),
        )
        self.reconstruction_head = nn.Linear(config.width, config.input_feature_count)
        self.event_head = nn.Linear(config.width, 1)
        self.onset_auxiliary_head = nn.Linear(config.width, 1)
        self.offset_auxiliary_head = nn.Linear(config.width, 1)

    def encode(self, windows: torch.Tensor, group_code: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[1] != 2 * self.config.input_feature_count:
            raise ValueError("window channel contract differs")
        if windows.shape[2] != self.config.receptive_field_rows:
            raise ValueError("window receptive field differs")
        if group_code.shape != (len(windows),):
            raise ValueError("group code must align with windows")
        hidden = self.stem(windows)
        for block in self.blocks:
            hidden = block(hidden)
        center = hidden[:, :, hidden.shape[2] // 2]
        joined = torch.cat([center, self.group_embedding(group_code)], dim=1)
        return self.shared_head(joined)

    def forward(
        self, windows: torch.Tensor, group_code: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.encode(windows, group_code)
        return (
            self.event_head(shared).squeeze(1),
            self.onset_auxiliary_head(shared).squeeze(1),
            self.offset_auxiliary_head(shared).squeeze(1),
            self.reconstruction_head(shared),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def _seed(seed: int) -> None:
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


@dataclass
class FittedMaskedPretrainBinaryModel:
    model_config: MaskedPretrainModelConfig
    training_config: MaskedPretrainTrainingConfig
    seed: int
    scaler: PrefixRobustScaler
    natural_priors: np.ndarray
    sampling_priors: np.ndarray
    state_dict: dict[str, torch.Tensor]
    train_ids_sha256: str
    model_state_sha256: str
    phase_counts: tuple[int, int, int, int]
    mean_training_loss: float
    mean_pretrain_loss: float
    mean_finetune_loss: float
    mean_event_loss: float
    mean_auxiliary_loss: float
    labels_materialized_after_pretraining: bool


def _masked_center_batch(
    windows: np.ndarray,
    *,
    feature_count: int,
    mask_probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = np.asarray(windows, dtype=np.float32).copy()
    center = result.shape[2] // 2
    target = result[:, :feature_count, center].copy()
    finite = result[:, feature_count:, center] > 0.5
    masked = (rng.random(size=target.shape) < mask_probability) & finite
    missing_rows = ~masked.any(axis=1)
    if missing_rows.any():
        for row in np.flatnonzero(missing_rows):
            available = np.flatnonzero(finite[row])
            if len(available):
                masked[row, available[int(rng.integers(len(available)))]] = True
    if not masked.any():
        raise ValueError("masked pretraining batch has no finite feature to reconstruct")
    result[:, :feature_count, center][masked] = 0.0
    result[:, feature_count:, center][masked] = 0.0
    return result, target, masked


def fit_masked_pretrain_binary_event_model(
    feature_values: np.ndarray,
    metadata: pd.DataFrame,
    binary_labels: Sequence[int] | np.ndarray | Callable[[], Sequence[int] | np.ndarray],
    layout: SequenceLayout,
    train_ids: Sequence[int] | np.ndarray,
    *,
    forbidden_ids: Sequence[int] | np.ndarray | None,
    seed: int,
    device: str | torch.device,
    model_config: MaskedPretrainModelConfig,
    training_config: MaskedPretrainTrainingConfig,
    scaler: PrefixRobustScaler | None = None,
) -> FittedMaskedPretrainBinaryModel:
    """Pretrain on prefix features only, then fine-tune on the same prefix labels."""

    values = np.asarray(feature_values)
    if values.ndim != 2 or len(values) != len(metadata):
        raise ValueError("feature values and metadata must align")
    model_config.validate()
    training_config.validate()
    ids = _ids(train_ids, size=len(metadata), role="training")
    if forbidden_ids is not None:
        forbidden = _ids(forbidden_ids, size=len(metadata), role="forbidden")
        if np.intersect1d(ids, forbidden).size:
            raise PermissionError("training IDs overlap forbidden validation IDs")
    fitted_scaler = scaler or PrefixRobustScaler.fit(values, ids, forbidden_ids=forbidden_ids)
    if fitted_scaler.fit_ids_sha256 != ids_sha256(ids):
        raise PermissionError("reused scaler was not fitted on the exact training prefix")
    scaled = fitted_scaler.transform(values)
    allowed_context = np.zeros(len(values), dtype=bool)
    allowed_context[ids] = True
    scaled[~allowed_context] = 0.0
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed(int(seed))
    model = MaskedPretrainBinaryEventModel(model_config).to(selected_device)
    pretrain_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.pretrain_learning_rate,
        weight_decay=training_config.weight_decay,
    )
    pretrain_sampler = _DenseEpochSampler(ids, seed=int(seed) + 101)
    mask_rng = np.random.default_rng(int(seed) + 202)
    pretrain_losses: list[float] = []
    for _ in range(training_config.pretrain_steps):
        batch_ids = pretrain_sampler.next(training_config.batch_size)
        windows, groups = layout.windows(
            scaled,
            batch_ids,
            receptive_field_rows=model_config.receptive_field_rows,
        )
        masked_windows, reconstruction_target, masked = _masked_center_batch(
            windows,
            feature_count=model_config.input_feature_count,
            mask_probability=training_config.mask_probability,
            rng=mask_rng,
        )
        model.train()
        pretrain_optimizer.zero_grad(set_to_none=True)
        _, _, _, reconstruction = model(
            torch.from_numpy(masked_windows).to(selected_device),
            torch.from_numpy(groups).to(selected_device),
        )
        target_tensor = torch.from_numpy(reconstruction_target).to(selected_device)
        mask_tensor = torch.from_numpy(masked).to(selected_device)
        pretrain_loss = F.smooth_l1_loss(
            reconstruction[mask_tensor], target_tensor[mask_tensor], beta=1.0
        )
        if not torch.isfinite(pretrain_loss):
            raise RuntimeError("masked pretraining produced a non-finite loss")
        pretrain_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
        pretrain_optimizer.step()
        pretrain_losses.append(float(pretrain_loss.detach().cpu()))

    labels_were_deferred = callable(binary_labels)
    label_values = binary_labels() if labels_were_deferred else binary_labels
    target_ids, event_target, onset_target, offset_target = (
        build_prefix_event_boundary_targets(
            metadata,
            label_values,
            ids,
            forbidden_ids=forbidden_ids,
            boundary_band_rows=training_config.boundary_band_rows,
        )
    )
    if not np.array_equal(target_ids, ids):
        raise AssertionError("fine-tune target IDs differ from pretraining prefix IDs")
    event_count = int(np.count_nonzero(event_target))
    normal_count = int(len(event_target) - event_count)
    onset_count = int(np.count_nonzero(onset_target))
    offset_count = int(np.count_nonzero(offset_target))
    if min(normal_count, event_count, onset_count, offset_count) <= 0:
        raise ValueError("every registered prefix must contain all target roles")
    rates = np.asarray(
        [normal_count, event_count, onset_count, offset_count], dtype=np.float64
    ) / float(len(ids))
    prevalence = min(max(event_count / float(len(ids)), 1e-6), 1.0 - 1e-6)
    with torch.no_grad():
        model.event_head.bias.fill_(math.log(prevalence / (1.0 - prevalence)))
    for parameter in model.reconstruction_head.parameters():
        parameter.requires_grad_(False)
    finetune_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    finetune_optimizer = torch.optim.AdamW(
        finetune_parameters,
        lr=training_config.finetune_learning_rate,
        weight_decay=training_config.weight_decay,
    )
    finetune_sampler = _DenseEpochSampler(ids, seed=int(seed) + 303)
    target_position = np.full(len(metadata), -1, dtype=np.int64)
    target_position[ids] = np.arange(len(ids), dtype=np.int64)
    onset_weight = torch.tensor(
        _balanced_auxiliary_positive_weight(onset_target),
        dtype=torch.float32,
        device=selected_device,
    )
    offset_weight = torch.tensor(
        _balanced_auxiliary_positive_weight(offset_target),
        dtype=torch.float32,
        device=selected_device,
    )
    finetune_losses: list[float] = []
    event_losses: list[float] = []
    auxiliary_losses: list[float] = []
    for _ in range(training_config.finetune_steps):
        batch_ids = finetune_sampler.next(training_config.batch_size)
        positions = target_position[batch_ids]
        if (positions < 0).any():
            raise AssertionError("fine-tune sampler emitted an ID outside the prefix")
        windows, groups = layout.windows(
            scaled,
            batch_ids,
            receptive_field_rows=model_config.receptive_field_rows,
        )
        event = torch.from_numpy(event_target[positions]).to(selected_device)
        onset = torch.from_numpy(onset_target[positions]).to(selected_device)
        offset = torch.from_numpy(offset_target[positions]).to(selected_device)
        model.train()
        finetune_optimizer.zero_grad(set_to_none=True)
        event_logits, onset_logits, offset_logits, _ = model(
            torch.from_numpy(windows).to(selected_device),
            torch.from_numpy(groups).to(selected_device),
        )
        event_loss = F.binary_cross_entropy_with_logits(event_logits, event)
        onset_loss = F.binary_cross_entropy_with_logits(
            onset_logits, onset, pos_weight=onset_weight
        )
        offset_loss = F.binary_cross_entropy_with_logits(
            offset_logits, offset, pos_weight=offset_weight
        )
        auxiliary_loss = 0.5 * (onset_loss + offset_loss)
        finetune_loss = event_loss + training_config.auxiliary_loss_weight * auxiliary_loss
        if not torch.isfinite(finetune_loss):
            raise RuntimeError("binary-event fine-tuning produced a non-finite loss")
        finetune_loss.backward()
        torch.nn.utils.clip_grad_norm_(finetune_parameters, training_config.gradient_clip_norm)
        finetune_optimizer.step()
        finetune_losses.append(float(finetune_loss.detach().cpu()))
        event_losses.append(float(event_loss.detach().cpu()))
        auxiliary_losses.append(float(auxiliary_loss.detach().cpu()))

    for parameter in model.reconstruction_head.parameters():
        parameter.requires_grad_(True)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    state_sha = model_state_sha256(state)
    combined = pretrain_losses + finetune_losses
    del model, scaled, pretrain_optimizer, finetune_optimizer
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return FittedMaskedPretrainBinaryModel(
        model_config=model_config,
        training_config=training_config,
        seed=int(seed),
        scaler=fitted_scaler,
        natural_priors=rates,
        sampling_priors=rates.copy(),
        state_dict=state,
        train_ids_sha256=ids_sha256(ids),
        model_state_sha256=state_sha,
        phase_counts=(normal_count, event_count, onset_count, offset_count),
        mean_training_loss=float(np.mean(combined)),
        mean_pretrain_loss=float(np.mean(pretrain_losses)),
        mean_finetune_loss=float(np.mean(finetune_losses)),
        mean_event_loss=float(np.mean(event_losses)),
        mean_auxiliary_loss=float(np.mean(auxiliary_losses)),
        labels_materialized_after_pretraining=labels_were_deferred,
    )


def predict_masked_pretrain_binary_probability(
    fitted: FittedMaskedPretrainBinaryModel,
    feature_values: np.ndarray,
    layout: SequenceLayout,
    prediction_ids: Sequence[int] | np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Predict sigmoid(event_head), discarding reconstruction and auxiliary outputs."""

    ids = _ids(prediction_ids, size=len(feature_values), role="prediction")
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    if model_state_sha256(fitted.state_dict) != fitted.model_state_sha256:
        raise PermissionError("fitted model state SHA differs before prediction")
    scaled = fitted.scaler.transform(feature_values)
    selected_device = torch.device(device)
    model = MaskedPretrainBinaryEventModel(fitted.model_config).to(selected_device)
    model.load_state_dict(fitted.state_dict, strict=True)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            current = ids[start : start + batch_size]
            windows, groups = layout.windows(
                scaled,
                current,
                receptive_field_rows=fitted.model_config.receptive_field_rows,
            )
            event_logits, _, _, _ = model(
                torch.from_numpy(windows).to(selected_device),
                torch.from_numpy(groups).to(selected_device),
            )
            outputs.append(torch.sigmoid(event_logits).float().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    if result.shape != (len(ids),) or not np.isfinite(result).all():
        raise RuntimeError("Gen4 prediction shape or finiteness differs")
    if not np.all((result >= 0.0) & (result <= 1.0)):
        raise RuntimeError("Gen4 probabilities lie outside [0,1]")
    del model, scaled
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_fitted_masked_pretrain_model(
    fitted: FittedMaskedPretrainBinaryModel, path: str | Path
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "p1_masked_pretrain_binary_model.v1",
        "model_config": asdict(fitted.model_config),
        "training_config": asdict(fitted.training_config),
        "seed": fitted.seed,
        "scaler_center": torch.from_numpy(fitted.scaler.center),
        "scaler_scale": torch.from_numpy(fitted.scaler.scale),
        "scaler_fit_ids_sha256": fitted.scaler.fit_ids_sha256,
        "scaler_sha256": fitted.scaler.state_sha256,
        "natural_priors": torch.from_numpy(fitted.natural_priors),
        "sampling_priors": torch.from_numpy(fitted.sampling_priors),
        "state_dict": fitted.state_dict,
        "train_ids_sha256": fitted.train_ids_sha256,
        "model_state_sha256": fitted.model_state_sha256,
        "phase_counts": fitted.phase_counts,
        "mean_training_loss": fitted.mean_training_loss,
        "mean_pretrain_loss": fitted.mean_pretrain_loss,
        "mean_finetune_loss": fitted.mean_finetune_loss,
        "mean_event_loss": fitted.mean_event_loss,
        "mean_auxiliary_loss": fitted.mean_auxiliary_loss,
        "labels_materialized_after_pretraining": fitted.labels_materialized_after_pretraining,
    }
    with target.open("xb") as handle:
        torch.save(payload, handle)


def load_fitted_masked_pretrain_model(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> FittedMaskedPretrainBinaryModel:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "p1_masked_pretrain_binary_model.v1"
    ):
        raise ValueError("saved Gen4 model schema differs")
    model_config = MaskedPretrainModelConfig(**payload["model_config"])
    training_config = MaskedPretrainTrainingConfig(**payload["training_config"])
    model_config.validate()
    training_config.validate()
    scaler = PrefixRobustScaler(
        payload["scaler_center"].cpu().numpy().astype(np.float32),
        payload["scaler_scale"].cpu().numpy().astype(np.float32),
        str(payload["scaler_fit_ids_sha256"]),
    )
    if scaler.state_sha256 != payload["scaler_sha256"]:
        raise PermissionError("saved prefix scaler SHA differs")
    state = {
        str(name): tensor.detach().cpu().clone()
        for name, tensor in payload["state_dict"].items()
    }
    state_sha = model_state_sha256(state)
    if state_sha != payload["model_state_sha256"]:
        raise PermissionError("saved Gen4 state SHA differs")
    return FittedMaskedPretrainBinaryModel(
        model_config=model_config,
        training_config=training_config,
        seed=int(payload["seed"]),
        scaler=scaler,
        natural_priors=payload["natural_priors"].cpu().numpy().astype(np.float64),
        sampling_priors=payload["sampling_priors"].cpu().numpy().astype(np.float64),
        state_dict=state,
        train_ids_sha256=str(payload["train_ids_sha256"]),
        model_state_sha256=state_sha,
        phase_counts=tuple(int(value) for value in payload["phase_counts"]),
        mean_training_loss=float(payload["mean_training_loss"]),
        mean_pretrain_loss=float(payload["mean_pretrain_loss"]),
        mean_finetune_loss=float(payload["mean_finetune_loss"]),
        mean_event_loss=float(payload["mean_event_loss"]),
        mean_auxiliary_loss=float(payload["mean_auxiliary_loss"]),
        labels_materialized_after_pretraining=bool(
            payload["labels_materialized_after_pretraining"]
        ),
    )


__all__ = [
    "FittedMaskedPretrainBinaryModel",
    "MaskedPretrainBinaryEventModel",
    "MaskedPretrainModelConfig",
    "MaskedPretrainTrainingConfig",
    "fit_masked_pretrain_binary_event_model",
    "load_fitted_masked_pretrain_model",
    "predict_masked_pretrain_binary_probability",
    "save_fitted_masked_pretrain_model",
]
