"""Dense-natural binary-event TCN for the P1 Gen3 learning curve.

The main event head is the only inference surface.  Onset and offset heads
share the encoder but contribute auxiliary training losses only; their logits
are never unioned with, added to, or otherwise used to form event
probabilities.  Every main-loss batch follows the exact natural distribution
of the supplied prefix IDs through deterministic shuffled dense epochs.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

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


def build_prefix_event_boundary_targets(
    metadata: pd.DataFrame,
    binary_labels: Sequence[int] | np.ndarray,
    train_ids: Sequence[int] | np.ndarray,
    *,
    forbidden_ids: Sequence[int] | np.ndarray | None = None,
    cadence_minutes: int = 10,
    boundary_band_rows: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build binary event plus onset/offset band targets from prefix labels only."""

    labels = np.asarray(binary_labels)
    if labels.ndim != 1 or len(labels) != len(metadata):
        raise ValueError("binary labels must align with metadata")
    ids = _ids(train_ids, size=len(metadata), role="training")
    if forbidden_ids is not None:
        forbidden = _ids(forbidden_ids, size=len(metadata), role="forbidden")
        if np.intersect1d(ids, forbidden).size:
            raise PermissionError("training IDs overlap forbidden validation IDs")
    selected_labels = labels[ids]
    if not np.isin(selected_labels, [0, 1]).all():
        raise ValueError("selected training-prefix labels must be binary")
    if cadence_minutes < 1 or boundary_band_rows < 1:
        raise ValueError("cadence and boundary band must be positive")

    work = metadata.iloc[ids][["station", "layer", "time"]].copy()
    work["row_id"] = ids
    work["target"] = selected_labels.astype(np.int8)
    work["parsed_time"] = pd.to_datetime(
        work["time"], errors="raise", utc=True, format="mixed"
    )
    work.sort_values(
        ["station", "layer", "parsed_time", "row_id"], kind="mergesort", inplace=True
    )
    onset_by_row: dict[int, int] = {int(row_id): 0 for row_id in ids}
    offset_by_row: dict[int, int] = {int(row_id): 0 for row_id in ids}
    cadence_ns = int(pd.Timedelta(minutes=cadence_minutes).value)
    for _, group in work.groupby(["station", "layer"], sort=False, observed=True):
        row_ids = group["row_id"].to_numpy(np.int64)
        target = group["target"].to_numpy(np.int8)
        times = group["parsed_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        position = 0
        while position < len(group):
            if target[position] == 0:
                position += 1
                continue
            stop = position + 1
            while (
                stop < len(group)
                and target[stop] == 1
                and int(times[stop] - times[stop - 1]) == cadence_ns
            ):
                stop += 1
            onset_stop = min(stop, position + boundary_band_rows)
            offset_start = max(position, stop - boundary_band_rows)
            for inner in range(position, onset_stop):
                onset_by_row[int(row_ids[inner])] = 1
            for inner in range(offset_start, stop):
                offset_by_row[int(row_ids[inner])] = 1
            position = stop

    event = selected_labels.astype(np.float32, copy=False)
    onset = np.asarray([onset_by_row[int(row_id)] for row_id in ids], dtype=np.float32)
    offset = np.asarray([offset_by_row[int(row_id)] for row_id in ids], dtype=np.float32)
    if np.any(onset > event) or np.any(offset > event):
        raise AssertionError("boundary auxiliary targets must be subsets of event targets")
    if not onset.any() or not offset.any():
        raise ValueError("every registered prefix must contain onset and offset targets")
    return ids, event, onset, offset


@dataclass(frozen=True)
class BinaryEventModelConfig:
    input_feature_count: int
    group_count: int
    width: int = 32
    group_embedding_width: int = 16
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 3
    dropout: float = 0.1
    norm_groups: int = 8

    @property
    def receptive_field_rows(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(int(value) for value in self.dilations)

    def validate(self) -> None:
        if self.input_feature_count < 1 or self.group_count < 1 or self.width < 4:
            raise ValueError("model dimensions must be positive and nontrivial")
        if self.kernel_size != 3 or self.width % self.norm_groups:
            raise ValueError("kernel or group-normalization contract differs")
        if tuple(self.dilations) != (1, 2, 4, 8) or self.receptive_field_rows != 31:
            raise ValueError("the Gen3 receptive field is frozen at 31 ten-minute rows")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")


@dataclass(frozen=True)
class DenseNaturalTrainingConfig:
    optimizer_steps: int = 120
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    auxiliary_loss_weight: float = 0.1
    boundary_band_rows: int = 6

    def validate(self) -> None:
        if self.optimizer_steps < 1 or self.batch_size < 8:
            raise ValueError("optimizer steps and batch size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")
        if not 0.0 <= self.auxiliary_loss_weight <= 1.0:
            raise ValueError("auxiliary loss weight must lie in [0,1]")
        if self.boundary_band_rows < 1:
            raise ValueError("boundary band rows must be positive")


class _CenteredDepthwiseBlock(nn.Module):
    def __init__(self, width: int, dilation: int, *, norm_groups: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
            groups=width,
        )
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(norm_groups, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.pointwise(F.gelu(self.norm(self.depthwise(values))))
        return values + self.dropout(hidden)


class StationLayerBinaryEventModel(nn.Module):
    """Centered TCN with one event head and two training-only auxiliary heads."""

    def __init__(self, config: BinaryEventModelConfig) -> None:
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
        self.event_head = nn.Linear(config.width, 1)
        self.onset_auxiliary_head = nn.Linear(config.width, 1)
        self.offset_auxiliary_head = nn.Linear(config.width, 1)

    def forward(
        self, windows: torch.Tensor, group_code: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        shared = self.shared_head(joined)
        return (
            self.event_head(shared).squeeze(1),
            self.onset_auxiliary_head(shared).squeeze(1),
            self.offset_auxiliary_head(shared).squeeze(1),
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
class FittedBinaryEventModel:
    model_config: BinaryEventModelConfig
    training_config: DenseNaturalTrainingConfig
    seed: int
    scaler: PrefixRobustScaler
    natural_priors: np.ndarray
    sampling_priors: np.ndarray
    state_dict: dict[str, torch.Tensor]
    train_ids_sha256: str
    model_state_sha256: str
    phase_counts: tuple[int, int, int, int]
    mean_training_loss: float
    mean_event_loss: float
    mean_auxiliary_loss: float


class _DenseEpochSampler:
    """Deterministic natural-distribution batches without within-epoch replacement."""

    def __init__(self, ids: np.ndarray, *, seed: int) -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        self.rng = np.random.default_rng(int(seed))
        self.order = self.rng.permutation(self.ids)
        self.position = 0

    def next(self, batch_size: int) -> np.ndarray:
        pieces: list[np.ndarray] = []
        remaining = int(batch_size)
        while remaining:
            available = len(self.order) - self.position
            take = min(remaining, available)
            pieces.append(self.order[self.position : self.position + take])
            self.position += take
            remaining -= take
            if self.position == len(self.order):
                self.order = self.rng.permutation(self.ids)
                self.position = 0
        return np.concatenate(pieces).astype(np.int64, copy=False)


def _balanced_auxiliary_positive_weight(target: np.ndarray) -> float:
    positives = int(np.count_nonzero(target))
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("auxiliary target must contain both classes")
    return float(negatives / positives)


def fit_fixed_step_binary_event_model(
    feature_values: np.ndarray,
    metadata: pd.DataFrame,
    binary_labels: Sequence[int] | np.ndarray,
    layout: SequenceLayout,
    train_ids: Sequence[int] | np.ndarray,
    *,
    forbidden_ids: Sequence[int] | np.ndarray | None,
    seed: int,
    device: str | torch.device,
    model_config: BinaryEventModelConfig,
    training_config: DenseNaturalTrainingConfig,
    scaler: PrefixRobustScaler | None = None,
) -> FittedBinaryEventModel:
    """Fit one prefix/fold/seed cell without reading validation targets."""

    values = np.asarray(feature_values)
    if values.ndim != 2 or len(values) != len(metadata):
        raise ValueError("feature values and metadata must align")
    model_config.validate()
    training_config.validate()
    ids, event_target, onset_target, offset_target = build_prefix_event_boundary_targets(
        metadata,
        binary_labels,
        train_ids,
        forbidden_ids=forbidden_ids,
        boundary_band_rows=training_config.boundary_band_rows,
    )
    fitted_scaler = scaler or PrefixRobustScaler.fit(values, ids, forbidden_ids=forbidden_ids)
    if fitted_scaler.fit_ids_sha256 != ids_sha256(ids):
        raise PermissionError("reused scaler was not fitted on the exact training prefix")
    scaled = fitted_scaler.transform(values)

    event_count = int(np.count_nonzero(event_target))
    normal_count = int(len(event_target) - event_count)
    onset_count = int(np.count_nonzero(onset_target))
    offset_count = int(np.count_nonzero(offset_target))
    if min(normal_count, event_count, onset_count, offset_count) <= 0:
        raise ValueError("every registered prefix must contain all training target roles")
    rates = np.asarray(
        [normal_count, event_count, onset_count, offset_count], dtype=np.float64
    ) / float(len(ids))

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed(int(seed))
    model = StationLayerBinaryEventModel(model_config).to(selected_device)
    prevalence = min(max(event_count / float(len(ids)), 1e-6), 1.0 - 1e-6)
    with torch.no_grad():
        model.event_head.bias.fill_(math.log(prevalence / (1.0 - prevalence)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    sampler = _DenseEpochSampler(ids, seed=int(seed))
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
    total_losses: list[float] = []
    event_losses: list[float] = []
    auxiliary_losses: list[float] = []
    for _ in range(training_config.optimizer_steps):
        batch_ids = sampler.next(training_config.batch_size)
        positions = target_position[batch_ids]
        if (positions < 0).any():
            raise AssertionError("dense sampler emitted an ID outside the training prefix")
        windows, groups = layout.windows(
            scaled,
            batch_ids,
            receptive_field_rows=model_config.receptive_field_rows,
        )
        event = torch.from_numpy(event_target[positions]).to(selected_device)
        onset = torch.from_numpy(onset_target[positions]).to(selected_device)
        offset = torch.from_numpy(offset_target[positions]).to(selected_device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        event_logits, onset_logits, offset_logits = model(
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
        loss = event_loss + training_config.auxiliary_loss_weight * auxiliary_loss
        if not torch.isfinite(loss):
            raise RuntimeError("binary-event training produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
        optimizer.step()
        total_losses.append(float(loss.detach().cpu()))
        event_losses.append(float(event_loss.detach().cpu()))
        auxiliary_losses.append(float(auxiliary_loss.detach().cpu()))

    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    state_sha = model_state_sha256(state)
    del model, scaled
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return FittedBinaryEventModel(
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
        mean_training_loss=float(np.mean(total_losses)),
        mean_event_loss=float(np.mean(event_losses)),
        mean_auxiliary_loss=float(np.mean(auxiliary_losses)),
    )


def predict_binary_event_probability(
    fitted: FittedBinaryEventModel,
    feature_values: np.ndarray,
    layout: SequenceLayout,
    prediction_ids: Sequence[int] | np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Predict sigmoid(event_head); auxiliary logits are deliberately discarded."""

    ids = _ids(prediction_ids, size=len(feature_values), role="prediction")
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    if model_state_sha256(fitted.state_dict) != fitted.model_state_sha256:
        raise PermissionError("fitted model state SHA differs before prediction")
    scaled = fitted.scaler.transform(feature_values)
    selected_device = torch.device(device)
    model = StationLayerBinaryEventModel(fitted.model_config).to(selected_device)
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
            event_logits, _, _ = model(
                torch.from_numpy(windows).to(selected_device),
                torch.from_numpy(groups).to(selected_device),
            )
            outputs.append(torch.sigmoid(event_logits).float().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    if result.shape != (len(ids),) or not np.isfinite(result).all():
        raise RuntimeError("binary-event prediction shape or finiteness differs")
    if not np.all((result >= 0.0) & (result <= 1.0)):
        raise RuntimeError("binary-event probabilities lie outside [0,1]")
    del model, scaled
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_fitted_binary_event_model(fitted: FittedBinaryEventModel, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "p1_binary_event_model.v1",
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
        "mean_event_loss": fitted.mean_event_loss,
        "mean_auxiliary_loss": fitted.mean_auxiliary_loss,
    }
    with target.open("xb") as handle:
        torch.save(payload, handle)


def load_fitted_binary_event_model(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> FittedBinaryEventModel:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != "p1_binary_event_model.v1":
        raise ValueError("saved binary-event model schema differs")
    model_config = BinaryEventModelConfig(**payload["model_config"])
    training_config = DenseNaturalTrainingConfig(**payload["training_config"])
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
        raise PermissionError("saved binary-event state SHA differs")
    return FittedBinaryEventModel(
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
        mean_event_loss=float(payload["mean_event_loss"]),
        mean_auxiliary_loss=float(payload["mean_auxiliary_loss"]),
    )


__all__ = [
    "BinaryEventModelConfig",
    "DenseNaturalTrainingConfig",
    "FittedBinaryEventModel",
    "StationLayerBinaryEventModel",
    "build_prefix_event_boundary_targets",
    "fit_fixed_step_binary_event_model",
    "load_fitted_binary_event_model",
    "predict_binary_event_probability",
    "save_fitted_binary_event_model",
]
