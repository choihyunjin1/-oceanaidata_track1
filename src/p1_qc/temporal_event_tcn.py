"""Fixed-receptive-field temporal event model for the P1 Gen2 curve.

The model is intentionally file agnostic.  It receives an immutable numeric
feature matrix plus explicit training and prediction row IDs.  Phase targets
are constructed only from the supplied training IDs; validation IDs are
accepted as a forbidden set and are rejected on overlap.
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
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

PHASE_NAMES = ("normal", "onset", "interior", "offset")


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


def ids_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class PrefixRobustScaler:
    """Componentwise median/IQR state fitted on one exact training prefix."""

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
    ) -> PrefixRobustScaler:
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("feature matrix must be two dimensional and non-empty")
        selected_ids = _ids(train_ids, size=len(matrix), role="training")
        if forbidden_ids is not None:
            forbidden = _ids(forbidden_ids, size=len(matrix), role="forbidden")
            if np.intersect1d(selected_ids, forbidden).size:
                raise PermissionError("training IDs overlap forbidden validation IDs")
        selected = np.asarray(matrix[selected_ids], dtype=np.float64)
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
        return cls(center.astype(np.float32), scale.astype(np.float32), ids_sha256(selected_ids))

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
            raise ValueError("feature count differs from the fitted scaler")
        finite = np.isfinite(matrix)
        normalized = np.where(finite, (matrix - self.center) / self.scale, 0.0)
        normalized = np.clip(normalized, -20.0, 20.0)
        result = np.concatenate([normalized, finite.astype(np.float32)], axis=1).astype(
            np.float32, copy=False
        )
        if not np.isfinite(result).all():
            raise AssertionError("scaled feature matrix contains non-finite values")
        return result


@dataclass(frozen=True)
class SequenceLayout:
    """Station-layer chronological index used to gather exact-cadence windows."""

    group_code_by_row: np.ndarray
    local_rank_by_row: np.ndarray
    time_ns_by_row: np.ndarray
    rows_by_group: tuple[np.ndarray, ...]
    group_labels: tuple[str, ...]
    cadence_ns: int

    @classmethod
    def build(cls, metadata: pd.DataFrame, *, cadence_minutes: int = 10) -> SequenceLayout:
        required = {"station", "layer", "time"}
        if missing := sorted(required.difference(metadata.columns)):
            raise KeyError(f"metadata columns missing: {missing}")
        if len(metadata) == 0:
            raise ValueError("metadata cannot be empty")
        parsed = pd.to_datetime(metadata["time"], errors="raise", utc=True, format="mixed")
        time_ns = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        keys = metadata["station"].astype(str) + "|L" + metadata["layer"].astype(str)
        codes, labels = pd.factorize(keys, sort=True)
        group_code = np.asarray(codes, dtype=np.int64)
        rank = np.empty(len(metadata), dtype=np.int64)
        rows_by_group: list[np.ndarray] = []
        for code in range(len(labels)):
            rows = np.flatnonzero(group_code == code)
            order = np.lexsort((rows, time_ns[rows]))
            rows = rows[order].astype(np.int64, copy=False)
            if len(np.unique(time_ns[rows])) != len(rows):
                raise ValueError(f"duplicate station-layer timestamp: {labels[code]}")
            rank[rows] = np.arange(len(rows), dtype=np.int64)
            rows_by_group.append(rows)
        return cls(
            group_code,
            rank,
            time_ns,
            tuple(rows_by_group),
            tuple(str(value) for value in labels),
            int(pd.Timedelta(minutes=cadence_minutes).value),
        )

    @property
    def group_count(self) -> int:
        return len(self.rows_by_group)

    def windows(
        self,
        scaled_values: np.ndarray,
        center_ids: Sequence[int] | np.ndarray,
        *,
        receptive_field_rows: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(scaled_values, dtype=np.float32)
        centers = np.asarray(center_ids)
        if centers.ndim != 1 or len(centers) == 0:
            raise ValueError("window center IDs must be a non-empty vector")
        if not np.issubdtype(centers.dtype, np.integer):
            raise TypeError("window center IDs must be integers")
        centers = centers.astype(np.int64, copy=False)
        if centers.min() < 0 or centers.max() >= len(values):
            raise IndexError("window center IDs are outside the aligned arrays")
        if receptive_field_rows < 3 or receptive_field_rows % 2 != 1:
            raise ValueError("receptive field must be an odd integer of at least three rows")
        radius = receptive_field_rows // 2
        offsets = np.arange(-radius, radius + 1, dtype=np.int64)
        result = np.zeros((len(centers), receptive_field_rows, values.shape[1]), dtype=np.float32)
        center_groups = self.group_code_by_row[centers]
        for code in np.unique(center_groups):
            batch_positions = np.flatnonzero(center_groups == code)
            current_rows = centers[batch_positions]
            local = self.local_rank_by_row[current_rows, None] + offsets[None, :]
            group_rows = self.rows_by_group[int(code)]
            valid = (local >= 0) & (local < len(group_rows))
            safe_local = np.clip(local, 0, len(group_rows) - 1)
            neighbor_rows = group_rows[safe_local]
            expected_time = self.time_ns_by_row[current_rows, None] + offsets[None, :] * self.cadence_ns
            valid &= self.time_ns_by_row[neighbor_rows] == expected_time
            gathered = values[neighbor_rows]
            gathered[~valid] = 0.0
            result[batch_positions] = gathered
        return result.transpose(0, 2, 1), center_groups.astype(np.int64, copy=False)


def build_prefix_phase_targets(
    metadata: pd.DataFrame,
    binary_labels: Sequence[int] | np.ndarray,
    train_ids: Sequence[int] | np.ndarray,
    *,
    forbidden_ids: Sequence[int] | np.ndarray | None = None,
    cadence_minutes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint normal/onset/interior/offset classes for exact prefix IDs.

    A multi-row event's first and last rows are onset and offset.  All middle
    rows are interior.  A singleton event is interior, avoiding an arbitrary
    choice of one boundary head for the same physical row.
    """

    labels = np.asarray(binary_labels)
    if labels.ndim != 1 or len(labels) != len(metadata):
        raise ValueError("binary labels must align with metadata")
    ids = _ids(train_ids, size=len(metadata), role="training")
    if forbidden_ids is not None:
        forbidden = _ids(forbidden_ids, size=len(metadata), role="forbidden")
        if np.intersect1d(ids, forbidden).size:
            raise PermissionError("training IDs overlap forbidden validation IDs")
    if not np.isin(labels[ids], [0, 1]).all():
        raise ValueError("selected training-prefix labels must be binary")
    work = metadata.iloc[ids][["station", "layer", "time"]].copy()
    work["row_id"] = ids
    work["target"] = labels[ids].astype(np.int8)
    work["parsed_time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(["station", "layer", "parsed_time", "row_id"], kind="mergesort", inplace=True)
    phase_by_row: dict[int, int] = {}
    cadence_seconds = cadence_minutes * 60
    for _, group in work.groupby(["station", "layer"], sort=False, observed=True):
        row_ids = group["row_id"].to_numpy(np.int64)
        target = group["target"].to_numpy(np.int8)
        times = group["parsed_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        position = 0
        while position < len(group):
            if target[position] == 0:
                phase_by_row[int(row_ids[position])] = 0
                position += 1
                continue
            stop = position + 1
            while (
                stop < len(group)
                and target[stop] == 1
                and int(times[stop] - times[stop - 1]) == cadence_seconds * 1_000_000_000
            ):
                stop += 1
            length = stop - position
            if length == 1:
                phase_by_row[int(row_ids[position])] = 2
            else:
                phase_by_row[int(row_ids[position])] = 1
                phase_by_row[int(row_ids[stop - 1])] = 3
                for inner in range(position + 1, stop - 1):
                    phase_by_row[int(row_ids[inner])] = 2
            position = stop
    phases = np.asarray([phase_by_row[int(row_id)] for row_id in ids], dtype=np.int64)
    if not np.array_equal(phases > 0, labels[ids].astype(bool)):
        raise AssertionError("phase union differs from the supplied binary labels")
    return ids, phases


@dataclass(frozen=True)
class TemporalEventModelConfig:
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
            raise ValueError("the Gen2 receptive field is frozen at 31 ten-minute rows")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")


@dataclass(frozen=True)
class FixedStepTrainingConfig:
    optimizer_steps: int = 120
    batch_size: int = 1024
    learning_rate: float = 7e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    def validate(self) -> None:
        if self.optimizer_steps < 1 or self.batch_size < 8:
            raise ValueError("optimizer steps and batch size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")


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


class StationLayerTemporalEventModel(nn.Module):
    """Centered TCN with separate onset/interior/offset logits."""

    def __init__(self, config: TemporalEventModelConfig) -> None:
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
        self.head = nn.Sequential(
            nn.Linear(config.width + config.group_embedding_width, config.width),
            nn.GELU(),
            nn.LayerNorm(config.width),
            nn.Linear(config.width, len(PHASE_NAMES)),
        )

    def forward(self, windows: torch.Tensor, group_code: torch.Tensor) -> torch.Tensor:
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
        return self.head(joined)

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
class FittedTemporalEventModel:
    model_config: TemporalEventModelConfig
    training_config: FixedStepTrainingConfig
    seed: int
    scaler: PrefixRobustScaler
    natural_priors: np.ndarray
    sampling_priors: np.ndarray
    state_dict: dict[str, torch.Tensor]
    train_ids_sha256: str
    model_state_sha256: str
    phase_counts: tuple[int, int, int, int]
    mean_training_loss: float


def _sample_phase_balanced(
    rng: np.random.Generator,
    ids: np.ndarray,
    phases: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray([batch_size // 2, batch_size // 6, batch_size // 6, 0], dtype=int)
    counts[3] = batch_size - int(counts[:3].sum())
    selected_ids: list[np.ndarray] = []
    selected_phases: list[np.ndarray] = []
    for phase, count in enumerate(counts):
        pool = ids[phases == phase]
        if len(pool) == 0:
            raise ValueError(f"training prefix has no examples for phase {PHASE_NAMES[phase]}")
        chosen = rng.choice(pool, size=int(count), replace=len(pool) < count)
        selected_ids.append(chosen.astype(np.int64, copy=False))
        selected_phases.append(np.full(int(count), phase, dtype=np.int64))
    batch_ids = np.concatenate(selected_ids)
    batch_phases = np.concatenate(selected_phases)
    permutation = rng.permutation(len(batch_ids))
    return batch_ids[permutation], batch_phases[permutation], counts / float(batch_size)


def fit_fixed_step_temporal_event_model(
    feature_values: np.ndarray,
    metadata: pd.DataFrame,
    binary_labels: Sequence[int] | np.ndarray,
    layout: SequenceLayout,
    train_ids: Sequence[int] | np.ndarray,
    *,
    forbidden_ids: Sequence[int] | np.ndarray,
    seed: int,
    device: str | torch.device,
    model_config: TemporalEventModelConfig,
    training_config: FixedStepTrainingConfig,
    scaler: PrefixRobustScaler | None = None,
) -> FittedTemporalEventModel:
    """Fit one independent prefix/fold/seed cell without validation targets."""

    values = np.asarray(feature_values)
    if values.ndim != 2 or len(values) != len(metadata):
        raise ValueError("feature values and metadata must align")
    model_config.validate()
    training_config.validate()
    ids, phases = build_prefix_phase_targets(
        metadata,
        binary_labels,
        train_ids,
        forbidden_ids=forbidden_ids,
    )
    fitted_scaler = scaler or PrefixRobustScaler.fit(values, ids, forbidden_ids=forbidden_ids)
    if fitted_scaler.fit_ids_sha256 != ids_sha256(ids):
        raise PermissionError("reused scaler was not fitted on the exact training prefix")
    scaled = fitted_scaler.transform(values)
    natural_counts = np.bincount(phases, minlength=len(PHASE_NAMES)).astype(np.int64)
    natural_priors = natural_counts.astype(np.float64) / float(natural_counts.sum())
    if (natural_counts == 0).any():
        raise ValueError("all four phase classes must occur in every registered prefix")

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed(int(seed))
    model = StationLayerTemporalEventModel(model_config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    rng = np.random.default_rng(int(seed))
    losses: list[float] = []
    sampling_priors: np.ndarray | None = None
    for _ in range(training_config.optimizer_steps):
        batch_ids, batch_phases, current_sampling = _sample_phase_balanced(
            rng, ids, phases, training_config.batch_size
        )
        sampling_priors = current_sampling
        windows, groups = layout.windows(
            scaled,
            batch_ids,
            receptive_field_rows=model_config.receptive_field_rows,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            torch.from_numpy(windows).to(selected_device),
            torch.from_numpy(groups).to(selected_device),
        )
        loss = F.cross_entropy(logits, torch.from_numpy(batch_phases).to(selected_device))
        if not torch.isfinite(loss):
            raise RuntimeError("temporal-event training produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if sampling_priors is None:
        raise AssertionError("fixed-step optimizer loop did not execute")
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    state_sha = model_state_sha256(state)
    del model, scaled
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return FittedTemporalEventModel(
        model_config=model_config,
        training_config=training_config,
        seed=int(seed),
        scaler=fitted_scaler,
        natural_priors=natural_priors.astype(np.float64),
        sampling_priors=np.asarray(sampling_priors, dtype=np.float64),
        state_dict=state,
        train_ids_sha256=ids_sha256(ids),
        model_state_sha256=state_sha,
        phase_counts=tuple(int(value) for value in natural_counts),
        mean_training_loss=float(np.mean(losses)),
    )


def predict_temporal_event_probability(
    fitted: FittedTemporalEventModel,
    feature_values: np.ndarray,
    layout: SequenceLayout,
    prediction_ids: Sequence[int] | np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Predict the prior-corrected union of onset/interior/offset heads."""

    ids = _ids(prediction_ids, size=len(feature_values), role="prediction")
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    if model_state_sha256(fitted.state_dict) != fitted.model_state_sha256:
        raise PermissionError("fitted model state SHA differs before prediction")
    scaled = fitted.scaler.transform(feature_values)
    selected_device = torch.device(device)
    model = StationLayerTemporalEventModel(fitted.model_config).to(selected_device)
    model.load_state_dict(fitted.state_dict, strict=True)
    model.eval()
    prior_adjustment = np.log(np.clip(fitted.natural_priors, 1e-12, None)) - np.log(
        np.clip(fitted.sampling_priors, 1e-12, None)
    )
    adjustment = torch.from_numpy(prior_adjustment.astype(np.float32)).to(selected_device)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            current = ids[start : start + batch_size]
            windows, groups = layout.windows(
                scaled,
                current,
                receptive_field_rows=fitted.model_config.receptive_field_rows,
            )
            logits = model(
                torch.from_numpy(windows).to(selected_device),
                torch.from_numpy(groups).to(selected_device),
            )
            probability = torch.softmax(logits + adjustment, dim=1)[:, 1:].sum(dim=1)
            outputs.append(probability.float().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    if result.shape != (len(ids),) or not np.isfinite(result).all():
        raise RuntimeError("temporal-event prediction shape or finiteness differs")
    if not np.all((result >= 0.0) & (result <= 1.0)):
        raise RuntimeError("temporal-event probabilities lie outside [0,1]")
    del model, scaled
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_fitted_temporal_event_model(fitted: FittedTemporalEventModel, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "p1_temporal_event_model.v1",
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
    }
    with target.open("xb") as handle:
        torch.save(payload, handle)


def load_fitted_temporal_event_model(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> FittedTemporalEventModel:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != "p1_temporal_event_model.v1":
        raise ValueError("saved temporal-event model schema differs")
    model_config = TemporalEventModelConfig(**payload["model_config"])
    training_config = FixedStepTrainingConfig(**payload["training_config"])
    model_config.validate()
    training_config.validate()
    scaler = PrefixRobustScaler(
        payload["scaler_center"].cpu().numpy().astype(np.float32),
        payload["scaler_scale"].cpu().numpy().astype(np.float32),
        str(payload["scaler_fit_ids_sha256"]),
    )
    if scaler.state_sha256 != payload["scaler_sha256"]:
        raise PermissionError("saved prefix scaler SHA differs")
    state = {str(name): tensor.detach().cpu().clone() for name, tensor in payload["state_dict"].items()}
    state_sha = model_state_sha256(state)
    if state_sha != payload["model_state_sha256"]:
        raise PermissionError("saved temporal-event state SHA differs")
    return FittedTemporalEventModel(
        model_config,
        training_config,
        int(payload["seed"]),
        scaler,
        payload["natural_priors"].cpu().numpy().astype(np.float64),
        payload["sampling_priors"].cpu().numpy().astype(np.float64),
        state,
        str(payload["train_ids_sha256"]),
        state_sha,
        tuple(int(value) for value in payload["phase_counts"]),
        float(payload["mean_training_loss"]),
    )


__all__ = [
    "FixedStepTrainingConfig",
    "FittedTemporalEventModel",
    "PHASE_NAMES",
    "PrefixRobustScaler",
    "SequenceLayout",
    "StationLayerTemporalEventModel",
    "TemporalEventModelConfig",
    "build_prefix_phase_targets",
    "fit_fixed_step_temporal_event_model",
    "ids_sha256",
    "load_fitted_temporal_event_model",
    "model_state_sha256",
    "predict_temporal_event_probability",
    "save_fitted_temporal_event_model",
]
