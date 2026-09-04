"""Nested, train-only checkpoint selection for the P3 causal sequence model.

This module is append-only.  It deliberately leaves the sealed fixed-eight-epoch
implementation untouched and adds only an inner-training loop that exposes one
hashed checkpoint and one disjoint-inner prediction per epoch.  Outer scoring and
artifact policy belong to the versioned runner.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .causal_forcing_sequence import (
    CausalForcingSequenceConfig,
    CompactRobustScaler,
    FORCING_SEQUENCE_COLUMNS,
    FixedEpochTrainingConfig,
    FittedSequenceModel,
    LeadCoupledCausalForcingEncoder,
    build_causal_forcing_sequence,
    model_state_sha256,
)
from .corrected_repeated_forward import OFFICIAL_LEADS
from .persistence_shrink import (
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)


CheckpointCallback = Callable[[int, FittedSequenceModel], None]


@dataclass(frozen=True)
class InnerSeedCheckpointCurve:
    """Predictions and state hashes from one seed's inner fit."""

    seed: int
    epochs: tuple[int, ...]
    prediction_delta_by_epoch: tuple[np.ndarray, ...]
    model_state_sha256_by_epoch: tuple[str, ...]
    scaler_state_sha256: str
    train_ids_sha256: str
    validation_ids_sha256: str
    optimizer_steps: int


@dataclass(frozen=True)
class EnsembleEpochSelection:
    """Earliest exact minimum of the seed-ensemble inner RMSE curve."""

    selected_epoch: int
    rmse_by_epoch: tuple[float, ...]
    seed_ids: tuple[int, ...]
    selection_prediction_sha256_by_epoch: tuple[str, ...]


def ids_sha256(ids: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(ids, dtype="<i8")
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("ID vector must be non-empty and one-dimensional")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    if not np.isfinite(array).all():
        raise ValueError("only finite arrays may be committed")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _validated_ids(
    values: Sequence[int] | np.ndarray,
    *,
    size: int,
    role: str,
) -> np.ndarray:
    ids = np.asarray(values, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError(f"{role} IDs must be non-empty and one-dimensional")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{role} IDs are duplicated")
    if ids.min() < 0 or ids.max() >= size:
        raise IndexError(f"{role} ID is outside the aligned case range")
    return ids


def _seed_deterministically(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _resolve_selected_forcing(
    raw: np.ndarray,
    ids: np.ndarray,
    forcing: np.ndarray | None,
) -> np.ndarray:
    if forcing is None:
        return build_causal_forcing_sequence(np.asarray(raw)[ids])
    source = np.asarray(forcing)
    expected = (len(raw), 289, len(FORCING_SEQUENCE_COLUMNS))
    if source.shape != expected:
        raise ValueError("precomputed forcing does not align with the case cache")
    selected = np.asarray(source[ids], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("selected forcing contains a non-finite value")
    return selected


def postprocess_sequence_delta(delta: np.ndarray, current_hs: np.ndarray) -> np.ndarray:
    """Apply the unchanged fixed-eight clip and long-lead persistence shrink."""

    values = np.asarray(delta, dtype=np.float64)
    current = np.asarray(current_hs, dtype=np.float64)
    if values.shape != (len(current), len(OFFICIAL_LEADS)):
        raise ValueError("sequence delta must align with six leads")
    absolute = np.clip(current[:, None] + values, 0.0, 30.0)
    flat = apply_long_lead_persistence_shrink(
        absolute.reshape(-1),
        np.repeat(current, len(OFFICIAL_LEADS)),
        np.tile(np.asarray(OFFICIAL_LEADS, dtype=int), len(current)),
        config=LongLeadPersistenceShrink(weight=0.2, active_leads=(12, 18, 24)),
    )
    result = flat.reshape(len(current), len(OFFICIAL_LEADS))
    if not np.isfinite(result).all() or not np.all((result >= 0.0) & (result <= 30.0)):
        raise RuntimeError("postprocessed prediction violates the fixed contract")
    return result


def select_earliest_ensemble_epoch(
    curves: Sequence[InnerSeedCheckpointCurve],
    *,
    current_hs: np.ndarray,
    target_hs: np.ndarray,
) -> EnsembleEpochSelection:
    """Select using only the mean of the registered seed predictions on inner labels."""

    if not curves:
        raise ValueError("at least one seed curve is required")
    seed_ids = tuple(int(curve.seed) for curve in curves)
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError("seed curves are duplicated")
    expected_epochs = curves[0].epochs
    if expected_epochs != tuple(range(1, len(expected_epochs) + 1)):
        raise ValueError("checkpoint epochs must be contiguous and one-based")
    if any(curve.epochs != expected_epochs for curve in curves[1:]):
        raise ValueError("seed checkpoint epoch grids differ")
    target = np.asarray(target_hs, dtype=np.float64)
    current = np.asarray(current_hs, dtype=np.float64)
    if target.shape != (len(current), len(OFFICIAL_LEADS)) or not np.isfinite(target).all():
        raise ValueError("inner target surface is invalid")

    rmse_by_epoch: list[float] = []
    prediction_hashes: list[str] = []
    for index, _epoch in enumerate(expected_epochs):
        deltas = [np.asarray(curve.prediction_delta_by_epoch[index], dtype=np.float64) for curve in curves]
        if any(delta.shape != target.shape or not np.isfinite(delta).all() for delta in deltas):
            raise ValueError("inner seed prediction surface is invalid")
        mean_delta = np.mean(np.stack(deltas, axis=0), axis=0)
        prediction = postprocess_sequence_delta(mean_delta, current)
        score = float(np.sqrt(np.mean(np.square(prediction - target))))
        if not np.isfinite(score):
            raise RuntimeError("inner checkpoint RMSE is non-finite")
        rmse_by_epoch.append(score)
        prediction_hashes.append(array_sha256(prediction))
    selected_index = int(np.argmin(np.asarray(rmse_by_epoch, dtype=np.float64)))
    return EnsembleEpochSelection(
        selected_epoch=int(expected_epochs[selected_index]),
        rmse_by_epoch=tuple(rmse_by_epoch),
        seed_ids=seed_ids,
        selection_prediction_sha256_by_epoch=tuple(prediction_hashes),
    )


def _predict_current_model(
    model: LeadCoupledCausalForcingEncoder,
    *,
    raw: np.ndarray,
    station: np.ndarray,
    compact: np.ndarray,
    forcing: np.ndarray,
    scaler: CompactRobustScaler,
    prediction_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    use_bf16_on_cuda: bool,
) -> np.ndarray:
    selected_raw = np.asarray(raw[prediction_ids], dtype=np.float32)
    selected_station = np.asarray(station[prediction_ids], dtype=np.int64)
    selected_compact = scaler.transform(np.asarray(compact[prediction_ids], dtype=np.float32))
    selected_forcing = np.asarray(forcing[prediction_ids], dtype=np.float32)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prediction_ids), batch_size):
            stop = min(start + batch_size, len(prediction_ids))
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and use_bf16_on_cuda,
            ):
                prediction = model(
                    torch.from_numpy(selected_raw[start:stop]).to(device),
                    torch.from_numpy(selected_station[start:stop]).to(device),
                    torch.from_numpy(selected_compact[start:stop]).to(device),
                    torch.from_numpy(selected_forcing[start:stop]).to(device),
                )
            outputs.append(prediction.float().cpu().numpy())
    model.train()
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    if result.shape != (len(prediction_ids), len(OFFICIAL_LEADS)) or not np.isfinite(result).all():
        raise RuntimeError("inner checkpoint prediction surface is invalid")
    return result


def fit_inner_checkpoint_curve(
    raw: np.ndarray,
    station: np.ndarray,
    compact: np.ndarray,
    target_delta: np.ndarray,
    case_weight: np.ndarray,
    train_ids: Sequence[int] | np.ndarray,
    validation_ids: Sequence[int] | np.ndarray,
    *,
    seed: int,
    device: str | torch.device,
    model_config: CausalForcingSequenceConfig | None = None,
    training_config: FixedEpochTrainingConfig | None = None,
    forcing: np.ndarray | None = None,
    compact_scaler: CompactRobustScaler | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> InnerSeedCheckpointCurve:
    """Fit once through max epoch and commit/predict every inner checkpoint."""

    raw_array = np.asarray(raw)
    station_array = np.asarray(station)
    compact_array = np.asarray(compact)
    target_array = np.asarray(target_delta)
    weight_array = np.asarray(case_weight)
    sizes = {len(raw_array), len(station_array), len(compact_array), len(target_array), len(weight_array)}
    if len(sizes) != 1:
        raise ValueError("aligned training arrays have different case counts")
    size = sizes.pop()
    config = model_config or CausalForcingSequenceConfig()
    training = training_config or FixedEpochTrainingConfig()
    config.validate()
    training.validate()
    train = _validated_ids(train_ids, size=size, role="inner training")
    validation = _validated_ids(validation_ids, size=size, role="inner validation")
    if np.intersect1d(train, validation).size:
        raise PermissionError("inner train and validation IDs overlap")
    if target_array.shape != (size, len(OFFICIAL_LEADS)) or weight_array.shape != (size,):
        raise ValueError("target or weight array differs from the six-lead contract")
    if compact_array.shape != (size, config.compact_feature_count):
        raise ValueError("compact feature array differs from the model contract")
    scaler = compact_scaler or CompactRobustScaler.fit(
        compact_array,
        train,
        forbidden_ids=validation,
    )
    if scaler.fit_ids_sha256 != ids_sha256(train):
        raise PermissionError("inner scaler was not fit on the exact inner-train IDs")

    train_raw = np.asarray(raw_array[train], dtype=np.float32)
    train_station = np.asarray(station_array[train], dtype=np.int64)
    train_compact = scaler.transform(compact_array[train])
    train_target = np.asarray(target_array[train], dtype=np.float32)
    train_weight = np.asarray(weight_array[train], dtype=np.float32)
    train_forcing = _resolve_selected_forcing(raw_array, train, forcing)
    full_forcing = (
        np.asarray(forcing, dtype=np.float32)
        if forcing is not None
        else build_causal_forcing_sequence(raw_array)
    )
    if not np.isfinite(train_target).all() or not np.isfinite(train_weight).all():
        raise ValueError("inner training targets and weights must be finite")
    if (train_weight <= 0.0).any():
        raise ValueError("inner training weights must be positive")
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
    predictions: list[np.ndarray] = []
    state_hashes: list[str] = []
    optimizer_steps = 0
    for epoch in range(1, training.epochs + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + epoch)
        order = torch.randperm(len(train), generator=generator).numpy()
        for start in range(0, len(order), training.batch_size):
            local = order[start : start + training.batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.bfloat16,
                enabled=selected_device.type == "cuda" and training.use_bf16_on_cuda,
            ):
                prediction = model(
                    torch.from_numpy(train_raw[local]).to(selected_device),
                    torch.from_numpy(train_station[local]).to(selected_device),
                    torch.from_numpy(train_compact[local]).to(selected_device),
                    torch.from_numpy(train_forcing[local]).to(selected_device),
                )
                target_batch = torch.from_numpy(train_target[local]).to(selected_device)
                weight_batch = torch.from_numpy(train_weight[local]).to(selected_device)
                loss = torch.mean(weight_batch[:, None] * torch.square(prediction - target_batch))
            if not torch.isfinite(loss):
                raise RuntimeError("inner checkpoint training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
            optimizer.step()
            optimizer_steps += 1

        predictions.append(
            _predict_current_model(
                model,
                raw=raw_array,
                station=station_array,
                compact=compact_array,
                forcing=full_forcing,
                scaler=scaler,
                prediction_ids=validation,
                device=selected_device,
                batch_size=training.batch_size,
                use_bf16_on_cuda=training.use_bf16_on_cuda,
            )
        )
        state: dict[str, torch.Tensor] = {
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        }
        state_sha = model_state_sha256(state)
        state_hashes.append(state_sha)
        fitted = FittedSequenceModel(
            model_config=config,
            training_config=FixedEpochTrainingConfig(
                epochs=epoch,
                batch_size=training.batch_size,
                learning_rate=training.learning_rate,
                weight_decay=training.weight_decay,
                gradient_clip_norm=training.gradient_clip_norm,
                use_bf16_on_cuda=training.use_bf16_on_cuda,
            ),
            seed=int(seed),
            scaler=scaler,
            state_dict=state,
            train_ids_sha256=ids_sha256(train),
            model_state_sha256=state_sha,
        )
        if checkpoint_callback is not None:
            checkpoint_callback(epoch, fitted)

    del model, optimizer
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return InnerSeedCheckpointCurve(
        seed=int(seed),
        epochs=tuple(range(1, training.epochs + 1)),
        prediction_delta_by_epoch=tuple(predictions),
        model_state_sha256_by_epoch=tuple(state_hashes),
        scaler_state_sha256=scaler.state_sha256,
        train_ids_sha256=ids_sha256(train),
        validation_ids_sha256=ids_sha256(validation),
        optimizer_steps=int(optimizer_steps),
    )


__all__ = [
    "EnsembleEpochSelection",
    "InnerSeedCheckpointCurve",
    "array_sha256",
    "fit_inner_checkpoint_curve",
    "ids_sha256",
    "postprocess_sequence_delta",
    "select_earliest_ensemble_epoch",
]
