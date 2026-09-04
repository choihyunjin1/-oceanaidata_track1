"""Identity-preserving incumbent distillation residual for P1 Gen5.

The model never replaces the incumbent probability.  It learns a bounded
logit correction from strictly causal raw-row context and four incumbent
identity features.  Callers decide whether the correction is enabled using a
sealed train-only gate; a failed gate returns the incumbent array exactly.
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

from p1_qc.binary_event_tcn import _DenseEpochSampler
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


def _probability(values: Sequence[float] | np.ndarray, *, size: int, role: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{role} probability must be a finite aligned vector")
    if ((result < 0.0) | (result > 1.0)).any():
        raise ValueError(f"{role} probability lies outside [0,1]")
    return result


def _binary(values: Sequence[int] | np.ndarray, *, size: int, role: str) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (size,) or not np.isin(result, [0, 1]).all():
        raise ValueError(f"{role} must be an aligned binary vector")
    return result.astype(np.float32, copy=False)


@dataclass(frozen=True)
class InnerChronologicalSplit:
    block: int
    teacher_train_ids: np.ndarray
    teacher_prediction_ids: np.ndarray
    train_end_utc: str
    prediction_start_utc: str
    prediction_end_utc: str
    purge_days: int

    @property
    def train_ids_sha256(self) -> str:
        return ids_sha256(self.teacher_train_ids)

    @property
    def prediction_ids_sha256(self) -> str:
        return ids_sha256(self.teacher_prediction_ids)


def build_three_block_inner_splits(
    metadata: pd.DataFrame,
    outer_prefix_ids: Sequence[int] | np.ndarray,
    *,
    purge_days: int = 7,
) -> tuple[InnerChronologicalSplit, ...]:
    """Build three label-free expanding blocks from timestamp order only.

    The first quarter is warm-up.  The remaining three quarters are the three
    prediction blocks.  Each teacher train set ends seven days before its
    prediction block begins.  No target column is accepted or inspected.
    """

    if "time" not in metadata.columns:
        raise KeyError("metadata time column is required")
    if purge_days < 1:
        raise ValueError("purge_days must be positive")
    ids = _ids(outer_prefix_ids, size=len(metadata), role="outer-prefix")
    time_ns = (
        pd.to_datetime(metadata["time"], errors="raise", utc=True, format="mixed")
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    unique_times = np.unique(time_ns[ids])
    if len(unique_times) < 8:
        raise ValueError("outer prefix has too few timestamps for three inner blocks")
    boundaries = [int(math.floor(len(unique_times) * fraction)) for fraction in (0.25, 0.5, 0.75)]
    if not (0 < boundaries[0] < boundaries[1] < boundaries[2] < len(unique_times)):
        raise ValueError("inner chronological boundaries collapsed")
    starts = [unique_times[index] for index in boundaries]
    stops = [unique_times[boundaries[1]], unique_times[boundaries[2]], unique_times[-1] + 1]
    purge_ns = int(pd.Timedelta(days=purge_days).value)
    result: list[InnerChronologicalSplit] = []
    prior_prediction_ids: list[np.ndarray] = []
    for block, (start, stop) in enumerate(zip(starts, stops, strict=True), 1):
        prediction_ids = ids[(time_ns[ids] >= start) & (time_ns[ids] < stop)]
        train_ids = ids[time_ns[ids] < start - purge_ns]
        if len(train_ids) == 0 or len(prediction_ids) == 0:
            raise ValueError(f"inner block {block} is empty after purge")
        if np.intersect1d(train_ids, prediction_ids).size:
            raise AssertionError("teacher train and prediction IDs overlap")
        if int(time_ns[train_ids].max()) >= int(time_ns[prediction_ids].min()) - purge_ns:
            raise AssertionError("inner split purge interval differs")
        if prior_prediction_ids and np.intersect1d(
            np.concatenate(prior_prediction_ids), prediction_ids
        ).size:
            raise AssertionError("inner prediction blocks overlap")
        prior_prediction_ids.append(prediction_ids)
        result.append(
            InnerChronologicalSplit(
                block=block,
                teacher_train_ids=train_ids,
                teacher_prediction_ids=prediction_ids,
                train_end_utc=pd.Timestamp(time_ns[train_ids].max(), tz="UTC").isoformat(),
                prediction_start_utc=pd.Timestamp(
                    time_ns[prediction_ids].min(), tz="UTC"
                ).isoformat(),
                prediction_end_utc=pd.Timestamp(
                    time_ns[prediction_ids].max(), tz="UTC"
                ).isoformat(),
                purge_days=int(purge_days),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class ResidualModelConfig:
    input_feature_count: int
    group_count: int
    width: int = 32
    group_embedding_width: int = 12
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 3
    dropout: float = 0.1
    norm_groups: int = 8
    maximum_absolute_logit_correction: float = 0.5

    @property
    def receptive_field_rows(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(int(value) for value in self.dilations)

    def validate(self) -> None:
        if self.input_feature_count < 1 or self.group_count < 1 or self.width < 8:
            raise ValueError("model dimensions must be positive and nontrivial")
        if self.kernel_size != 3 or self.width % self.norm_groups:
            raise ValueError("kernel or group-normalization contract differs")
        if tuple(self.dilations) != (1, 2, 4, 8) or self.receptive_field_rows != 31:
            raise ValueError("Gen5 receptive field is frozen at 31 ten-minute rows")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.maximum_absolute_logit_correction != 0.5:
            raise ValueError("Gen5 bounded logit correction must equal 0.5")


@dataclass(frozen=True)
class ResidualTrainingConfig:
    optimizer_steps: int = 120
    batch_size: int = 8192
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    main_loss_weight: float = 1.0
    distillation_loss_weight: float = 1.0
    identity_regularizer_weight: float = 0.05

    def validate(self) -> None:
        if self.optimizer_steps < 1 or self.batch_size < 8:
            raise ValueError("optimizer steps and batch size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")
        if (
            self.main_loss_weight != 1.0
            or self.distillation_loss_weight != 1.0
            or self.identity_regularizer_weight != 0.05
        ):
            raise ValueError("Gen5 fixed loss weights differ")


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


class IncumbentResidualTCN(nn.Module):
    """Causal raw encoder plus incumbent identity features and two heads."""

    def __init__(self, config: ResidualModelConfig) -> None:
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
        joined_width = config.width + config.group_embedding_width + 4
        self.shared = nn.Sequential(
            nn.Linear(joined_width, config.width),
            nn.GELU(),
            nn.LayerNorm(config.width),
        )
        self.residual_head = nn.Linear(config.width, 1)
        self.distillation_head = nn.Linear(config.width, 1)

    def forward(
        self,
        causal_windows: torch.Tensor,
        group_code: torch.Tensor,
        identity_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if causal_windows.ndim != 3 or causal_windows.shape[1] != 2 * self.config.input_feature_count:
            raise ValueError("window channel contract differs")
        if causal_windows.shape[2] != self.config.receptive_field_rows:
            raise ValueError("window receptive field differs")
        if group_code.shape != (len(causal_windows),) or identity_features.shape != (
            len(causal_windows),
            4,
        ):
            raise ValueError("group or identity features do not align")
        hidden = self.stem(causal_windows)
        for block in self.blocks:
            hidden = block(hidden)
        center = hidden[:, :, hidden.shape[2] // 2]
        joined = torch.cat([center, self.group_embedding(group_code), identity_features], dim=1)
        shared = self.shared(joined)
        bounded = self.config.maximum_absolute_logit_correction * torch.tanh(
            self.residual_head(shared).squeeze(1)
        )
        distilled_logit = self.distillation_head(shared).squeeze(1)
        return bounded, distilled_logit

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


def _identity_features(
    base_seed: np.ndarray,
    base_mean: np.ndarray,
    base_std: np.ndarray,
    base_decision: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    clipped_seed = np.clip(base_seed[ids], 1e-6, 1.0 - 1e-6)
    clipped_mean = np.clip(base_mean[ids], 1e-6, 1.0 - 1e-6)
    return np.column_stack(
        [
            np.log(clipped_seed / (1.0 - clipped_seed)),
            np.log(clipped_mean / (1.0 - clipped_mean)),
            base_std[ids],
            base_decision[ids],
        ]
    ).astype(np.float32)


def _causal_windows(
    layout: SequenceLayout,
    scaled_values: np.ndarray,
    center_ids: np.ndarray,
    *,
    receptive_field_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows, groups = layout.windows(
        scaled_values,
        center_ids,
        receptive_field_rows=receptive_field_rows,
    )
    center = receptive_field_rows // 2
    windows[:, :, center + 1 :] = 0.0
    return windows, groups


@dataclass
class FittedIncumbentResidualModel:
    model_config: ResidualModelConfig
    training_config: ResidualTrainingConfig
    seed: int
    scaler: PrefixRobustScaler
    state_dict: dict[str, torch.Tensor]
    train_ids_sha256: str
    context_ids_sha256: str
    model_state_sha256: str
    mean_training_loss: float
    mean_main_loss: float
    mean_distillation_loss: float
    mean_identity_regularizer: float


def fit_incumbent_residual_model(
    feature_values: np.ndarray,
    layout: SequenceLayout,
    train_ids: Sequence[int] | np.ndarray,
    train_labels: Sequence[int] | np.ndarray,
    base_seed_probability: Sequence[float] | np.ndarray,
    base_mean_probability: Sequence[float] | np.ndarray,
    base_probability_std: Sequence[float] | np.ndarray,
    base_decision: Sequence[int] | np.ndarray,
    *,
    context_ids: Sequence[int] | np.ndarray,
    forbidden_ids: Sequence[int] | np.ndarray | None,
    seed: int,
    device: str | torch.device,
    model_config: ResidualModelConfig,
    training_config: ResidualTrainingConfig,
) -> FittedIncumbentResidualModel:
    """Fit on explicitly supplied OOF IDs and labels only."""

    values = np.asarray(feature_values)
    if values.ndim != 2 or values.shape[1] != model_config.input_feature_count:
        raise ValueError("feature matrix shape differs")
    model_config.validate()
    training_config.validate()
    ids = _ids(train_ids, size=len(values), role="residual-training")
    context = _ids(context_ids, size=len(values), role="causal-context")
    if not np.isin(ids, context).all():
        raise PermissionError("residual training IDs are outside causal context")
    if forbidden_ids is not None:
        forbidden = _ids(forbidden_ids, size=len(values), role="forbidden")
        if np.intersect1d(ids, forbidden).size:
            raise PermissionError("residual training IDs overlap forbidden IDs")
    labels = _binary(train_labels, size=len(ids), role="residual training labels")
    if len(np.unique(labels)) != 2:
        raise ValueError("residual training labels must contain both classes")
    base_seed = _probability(base_seed_probability, size=len(values), role="base seed")
    base_mean = _probability(base_mean_probability, size=len(values), role="base mean")
    base_std = np.asarray(base_probability_std, dtype=np.float32)
    if base_std.shape != (len(values),) or not np.isfinite(base_std[ids]).all() or (base_std[ids] < 0).any():
        raise ValueError("base probability standard deviation differs")
    decision = _binary(base_decision, size=len(values), role="base decision")
    for role, array in (("base seed", base_seed), ("base mean", base_mean)):
        if not np.isfinite(array[ids]).all():
            raise ValueError(f"{role} is missing on residual training IDs")

    scaler = PrefixRobustScaler.fit(values, ids, forbidden_ids=forbidden_ids)
    scaled = scaler.transform(values)
    allowed = np.zeros(len(values), dtype=bool)
    allowed[context] = True
    scaled[~allowed] = 0.0
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed(int(seed))
    model = IncumbentResidualTCN(model_config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    sampler = _DenseEpochSampler(ids, seed=int(seed))
    position = np.full(len(values), -1, dtype=np.int64)
    position[ids] = np.arange(len(ids), dtype=np.int64)
    identity = _identity_features(base_seed, base_mean, base_std, decision, ids)
    base_logit = identity[:, 0].copy()
    teacher_logit = identity[:, 1].copy()
    losses: list[float] = []
    main_losses: list[float] = []
    distill_losses: list[float] = []
    regularizers: list[float] = []
    for _ in range(training_config.optimizer_steps):
        batch_ids = sampler.next(training_config.batch_size)
        positions = position[batch_ids]
        windows, groups = _causal_windows(
            layout,
            scaled,
            batch_ids,
            receptive_field_rows=model_config.receptive_field_rows,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        bounded, distilled = model(
            torch.from_numpy(windows).to(selected_device),
            torch.from_numpy(groups).to(selected_device),
            torch.from_numpy(identity[positions]).to(selected_device),
        )
        target = torch.from_numpy(labels[positions]).to(selected_device)
        base_offset = torch.from_numpy(base_logit[positions]).to(selected_device)
        teacher = torch.from_numpy(teacher_logit[positions]).to(selected_device)
        main_loss = F.binary_cross_entropy_with_logits(base_offset + bounded, target)
        distill_loss = F.smooth_l1_loss(distilled, teacher, beta=1.0)
        regularizer = bounded.square().mean()
        loss = (
            training_config.main_loss_weight * main_loss
            + training_config.distillation_loss_weight * distill_loss
            + training_config.identity_regularizer_weight * regularizer
        )
        if not torch.isfinite(loss):
            raise RuntimeError("incumbent residual training produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        main_losses.append(float(main_loss.detach().cpu()))
        distill_losses.append(float(distill_loss.detach().cpu()))
        regularizers.append(float(regularizer.detach().cpu()))
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    state_sha = model_state_sha256(state)
    del model, optimizer, scaled
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return FittedIncumbentResidualModel(
        model_config=model_config,
        training_config=training_config,
        seed=int(seed),
        scaler=scaler,
        state_dict=state,
        train_ids_sha256=ids_sha256(ids),
        context_ids_sha256=ids_sha256(context),
        model_state_sha256=state_sha,
        mean_training_loss=float(np.mean(losses)),
        mean_main_loss=float(np.mean(main_losses)),
        mean_distillation_loss=float(np.mean(distill_losses)),
        mean_identity_regularizer=float(np.mean(regularizers)),
    )


def predict_incumbent_residual_probability(
    fitted: FittedIncumbentResidualModel,
    feature_values: np.ndarray,
    layout: SequenceLayout,
    prediction_ids: Sequence[int] | np.ndarray,
    base_seed_probability: Sequence[float] | np.ndarray,
    base_mean_probability: Sequence[float] | np.ndarray,
    base_probability_std: Sequence[float] | np.ndarray,
    base_decision: Sequence[int] | np.ndarray,
    *,
    context_ids: Sequence[int] | np.ndarray,
    device: str | torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return sigmoid(logit(incumbent seed) + bounded residual)."""

    values = np.asarray(feature_values)
    ids = _ids(prediction_ids, size=len(values), role="residual-prediction")
    context = _ids(context_ids, size=len(values), role="causal-context")
    if not np.isin(ids, context).all():
        raise PermissionError("prediction IDs are outside causal context")
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    if model_state_sha256(fitted.state_dict) != fitted.model_state_sha256:
        raise PermissionError("fitted model state SHA differs")
    base_seed = _probability(base_seed_probability, size=len(values), role="base seed")
    base_mean = _probability(base_mean_probability, size=len(values), role="base mean")
    base_std = np.asarray(base_probability_std, dtype=np.float32)
    if base_std.shape != (len(values),) or not np.isfinite(base_std[ids]).all():
        raise ValueError("base probability standard deviation differs")
    decision = _binary(base_decision, size=len(values), role="base decision")
    identity = _identity_features(base_seed, base_mean, base_std, decision, ids)
    scaled = fitted.scaler.transform(values)
    allowed = np.zeros(len(values), dtype=bool)
    allowed[context] = True
    scaled[~allowed] = 0.0
    selected_device = torch.device(device)
    model = IncumbentResidualTCN(fitted.model_config).to(selected_device)
    model.load_state_dict(fitted.state_dict, strict=True)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            current = ids[start : start + batch_size]
            windows, groups = _causal_windows(
                layout,
                scaled,
                current,
                receptive_field_rows=fitted.model_config.receptive_field_rows,
            )
            bounded, _ = model(
                torch.from_numpy(windows).to(selected_device),
                torch.from_numpy(groups).to(selected_device),
                torch.from_numpy(identity[start : start + len(current)]).to(selected_device),
            )
            logits = torch.from_numpy(identity[start : start + len(current), 0]).to(
                selected_device
            ) + bounded
            outputs.append(torch.sigmoid(logits).float().cpu().numpy())
    result = np.clip(
        np.concatenate(outputs).astype(np.float32, copy=False),
        np.float32(1e-6),
        np.float32(0.999999),
    )
    if result.shape != (len(ids),) or not np.isfinite(result).all():
        raise RuntimeError("residual prediction shape or finiteness differs")
    del model, scaled
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def exact_identity_or_residual(
    incumbent_probability: Sequence[float] | np.ndarray,
    residual_probability: Sequence[float] | np.ndarray,
    *,
    gate_passed: bool,
) -> np.ndarray:
    incumbent = np.asarray(incumbent_probability)
    residual = np.asarray(residual_probability)
    if incumbent.shape != residual.shape:
        raise ValueError("incumbent and residual probability shapes differ")
    if gate_passed:
        result = residual.copy()
    else:
        result = incumbent.copy()
        if not np.array_equal(result, incumbent):
            raise AssertionError("failed gate did not preserve exact incumbent bytes")
    return result


def save_fitted_incumbent_residual_model(
    fitted: FittedIncumbentResidualModel, path: str | Path
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "p1_incumbent_residual_tcn.v1",
        "model_config": asdict(fitted.model_config),
        "training_config": asdict(fitted.training_config),
        "seed": fitted.seed,
        "scaler_center": torch.from_numpy(fitted.scaler.center),
        "scaler_scale": torch.from_numpy(fitted.scaler.scale),
        "scaler_fit_ids_sha256": fitted.scaler.fit_ids_sha256,
        "scaler_sha256": fitted.scaler.state_sha256,
        "state_dict": fitted.state_dict,
        "train_ids_sha256": fitted.train_ids_sha256,
        "context_ids_sha256": fitted.context_ids_sha256,
        "model_state_sha256": fitted.model_state_sha256,
        "mean_training_loss": fitted.mean_training_loss,
        "mean_main_loss": fitted.mean_main_loss,
        "mean_distillation_loss": fitted.mean_distillation_loss,
        "mean_identity_regularizer": fitted.mean_identity_regularizer,
    }
    with target.open("xb") as handle:
        torch.save(payload, handle)


def load_fitted_incumbent_residual_model(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> FittedIncumbentResidualModel:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != "p1_incumbent_residual_tcn.v1":
        raise ValueError("saved Gen5 residual model schema differs")
    model_config = ResidualModelConfig(**payload["model_config"])
    training_config = ResidualTrainingConfig(**payload["training_config"])
    model_config.validate()
    training_config.validate()
    scaler = PrefixRobustScaler(
        payload["scaler_center"].cpu().numpy().astype(np.float32),
        payload["scaler_scale"].cpu().numpy().astype(np.float32),
        str(payload["scaler_fit_ids_sha256"]),
    )
    if scaler.state_sha256 != payload["scaler_sha256"]:
        raise PermissionError("saved residual scaler SHA differs")
    state = {
        str(name): tensor.detach().cpu().clone()
        for name, tensor in payload["state_dict"].items()
    }
    state_sha = model_state_sha256(state)
    if state_sha != payload["model_state_sha256"]:
        raise PermissionError("saved residual model state SHA differs")
    return FittedIncumbentResidualModel(
        model_config=model_config,
        training_config=training_config,
        seed=int(payload["seed"]),
        scaler=scaler,
        state_dict=state,
        train_ids_sha256=str(payload["train_ids_sha256"]),
        context_ids_sha256=str(payload["context_ids_sha256"]),
        model_state_sha256=state_sha,
        mean_training_loss=float(payload["mean_training_loss"]),
        mean_main_loss=float(payload["mean_main_loss"]),
        mean_distillation_loss=float(payload["mean_distillation_loss"]),
        mean_identity_regularizer=float(payload["mean_identity_regularizer"]),
    )


__all__ = [
    "FittedIncumbentResidualModel",
    "IncumbentResidualTCN",
    "InnerChronologicalSplit",
    "ResidualModelConfig",
    "ResidualTrainingConfig",
    "build_three_block_inner_splits",
    "exact_identity_or_residual",
    "fit_incumbent_residual_model",
    "load_fitted_incumbent_residual_model",
    "predict_incumbent_residual_probability",
    "save_fitted_incumbent_residual_model",
]
