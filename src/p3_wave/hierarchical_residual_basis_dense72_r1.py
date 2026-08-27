"""Dense72 masked-supervision correction for the P3 Gen5 basis model.

The failed static v1 model module remains immutable.  This append-only module
reuses its frozen 72-step architecture while replacing the six-slice optimizer
surface with a genuine train-only, masked 72-step loss.  Validation targets are
absent from the core fit API and may be poisoned in the convenience wrapper.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dense72_targets_r1 import DENSE_TARGET_STEPS, OFFICIAL_DENSE_INDICES
from .hierarchical_residual_basis import (
    CONTEXT_20M_STEPS,
    FORECAST_20M_STEPS,
    INPUT_CHANNELS,
    FixedBasisTrainingConfig,
    HierarchicalResidualBasisConfig,
    HierarchicalResidualBasisForecaster,
    StaticRobustScaler,
    model_state_sha256,
)
from .revin_patch import CONTEXT_ROWS, RAW_COLUMNS

MODEL_BUNDLE_SCHEMA = "p3_hierarchical_residual_basis_dense72_r1"


def _case_ids(
    values: Sequence[int] | np.ndarray,
    *,
    expected_size: int | None,
    role: str,
) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1 or len(source) == 0:
        raise ValueError(f"{role} case IDs must be a non-empty vector")
    if expected_size is not None and len(source) != expected_size:
        raise ValueError(f"{role} case IDs do not align with sliced arrays")
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"{role} case IDs must be integers")
    result = source.astype(np.int64, copy=False)
    if np.unique(result).size != len(result):
        raise ValueError(f"{role} case IDs must be unique")
    if result.min() < 0:
        raise ValueError(f"{role} case IDs must be non-negative")
    return result


def _aligned_case_count(*arrays: np.ndarray) -> int:
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("aligned input arrays have different case counts")
    return lengths.pop()


def _ids_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _array_sha256(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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


def _training_target_sha256(target: np.ndarray, mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(_array_sha256(target, dtype="<f4").encode("ascii"))
    digest.update(_array_sha256(mask, dtype="|b1").encode("ascii"))
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} is not a lowercase SHA-256 digest")
    return text


def _station_codes(station: np.ndarray, *, size: int) -> np.ndarray:
    source = np.asarray(station)
    if source.shape != (size,) or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("station codes must be an aligned integer vector")
    result = source.astype(np.int64, copy=False)
    if len(result) == 0 or result.min() < 0 or result.max() > 2:
        raise ValueError("station codes lie outside the official three-station set")
    return result


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


@dataclass
class FittedDense72HierarchicalResidualBasis:
    model_config: HierarchicalResidualBasisConfig
    training_config: FixedBasisTrainingConfig
    seed: int
    scaler: StaticRobustScaler
    state_dict: dict[str, torch.Tensor]
    training_steps: int
    valid_target_scalars_per_epoch: int
    train_ids_sha256: str
    train_context_sha256: str
    train_target_sha256: str
    scaler_state_sha256: str
    model_state_sha256: str


def fit_dense72_hierarchical_model(
    train_raw: np.ndarray,
    train_station: np.ndarray,
    train_static: np.ndarray,
    train_target_delta: np.ndarray,
    train_target_mask: np.ndarray,
    train_case_weight: np.ndarray,
    train_case_ids: Sequence[int] | np.ndarray,
    *,
    forbidden_case_ids: Sequence[int] | np.ndarray,
    seed: int,
    device: str | torch.device,
    model_config: HierarchicalResidualBasisConfig | None = None,
    training_config: FixedBasisTrainingConfig | None = None,
    static_scaler: StaticRobustScaler | None = None,
) -> FittedDense72HierarchicalResidualBasis:
    """Fit one cell using only sliced train rows and a masked dense72 loss."""

    raw_array = np.asarray(train_raw)
    station_array = np.asarray(train_station)
    static_array = np.asarray(train_static)
    target_array = np.asarray(train_target_delta)
    mask_array = np.asarray(train_target_mask)
    weight_array = np.asarray(train_case_weight)
    size = _aligned_case_count(
        raw_array,
        station_array,
        static_array,
        target_array,
        mask_array,
        weight_array,
    )
    if size == 0:
        raise ValueError("sliced training arrays cannot be empty")
    config = model_config or HierarchicalResidualBasisConfig()
    training = training_config or FixedBasisTrainingConfig()
    config.validate()
    training.validate()
    if config.forecast_steps != DENSE_TARGET_STEPS or FORECAST_20M_STEPS != DENSE_TARGET_STEPS:
        raise ValueError("model forecast surface is not dense72")
    if raw_array.shape != (size, CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise ValueError("raw training shape differs from the 48-hour contract")
    station_codes = _station_codes(station_array, size=size)
    if static_array.shape != (size, config.static_feature_count):
        raise ValueError("static training features differ from the model contract")
    if target_array.shape != (size, DENSE_TARGET_STEPS):
        raise ValueError("training target must contain 72 future 20-minute deltas")
    if mask_array.shape != target_array.shape or mask_array.dtype != np.bool_:
        raise ValueError("training target mask must be an aligned boolean dense72 matrix")
    if weight_array.shape != (size,):
        raise ValueError("training case weights must be an aligned vector")

    train_ids = _case_ids(train_case_ids, expected_size=size, role="training")
    forbidden_ids = _case_ids(forbidden_case_ids, expected_size=None, role="forbidden")
    if np.intersect1d(train_ids, forbidden_ids).size:
        raise PermissionError("training case IDs overlap forbidden case IDs")
    if not mask_array.any(axis=1).all():
        raise ValueError("every training case must have at least one dense target")
    if not mask_array[:, OFFICIAL_DENSE_INDICES].all():
        raise ValueError("the six official positions must be present for every train case")
    if not np.isfinite(target_array[mask_array]).all():
        raise ValueError("selected dense target values must be finite")
    if np.any(target_array[~mask_array] != 0.0):
        raise ValueError("masked dense target slots must be canonical zero")
    if not np.isfinite(weight_array).all() or (weight_array <= 0.0).any():
        raise ValueError("training case weights must be finite and positive")

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
            raise PermissionError("reused scaler was not fit on the exact train IDs")
    scaled_static = scaler.transform(static_array)
    normalized_weight = np.asarray(weight_array, dtype=np.float32)
    normalized_weight /= float(normalized_weight.mean())

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
    raw_values = np.asarray(raw_array, dtype=np.float32)
    target_values = np.asarray(target_array, dtype=np.float32)
    target_masks = np.asarray(mask_array, dtype=bool)
    valid_per_epoch = int(target_masks.sum())
    for epoch in range(1, training.epochs + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + epoch)
        order = torch.randperm(size, generator=generator).numpy()
        for start in range(0, size, training.batch_size):
            local = order[start : start + training.batch_size]
            raw_batch = torch.from_numpy(raw_values[local]).to(selected_device)
            station_batch = torch.from_numpy(station_codes[local]).to(selected_device)
            static_batch = torch.from_numpy(scaled_static[local]).to(selected_device)
            target_batch = torch.from_numpy(target_values[local]).to(selected_device)
            mask_batch = torch.from_numpy(target_masks[local]).to(selected_device)
            weight_batch = torch.from_numpy(normalized_weight[local]).to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.bfloat16,
                enabled=selected_device.type == "cuda" and training.use_bf16_on_cuda,
            ):
                prediction = model.forward_dense(raw_batch, station_batch, static_batch)
                weighted_mask = weight_batch[:, None] * mask_batch.to(prediction.dtype)
                numerator = torch.sum(
                    weighted_mask * torch.square(prediction - target_batch)
                )
                denominator = torch.sum(weighted_mask)
                loss = numerator / denominator.clamp_min(1.0)
            if not torch.isfinite(loss):
                raise RuntimeError("dense72 masked training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
            optimizer.step()

    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    state_sha = model_state_sha256(state)
    del model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return FittedDense72HierarchicalResidualBasis(
        model_config=config,
        training_config=training,
        seed=int(seed),
        scaler=scaler,
        state_dict=state,
        training_steps=int(
            training.epochs * ((size + training.batch_size - 1) // training.batch_size)
        ),
        valid_target_scalars_per_epoch=valid_per_epoch,
        train_ids_sha256=_ids_sha256(train_ids),
        train_context_sha256=_training_context_sha256(
            raw_values,
            station_codes,
            static_array,
        ),
        train_target_sha256=_training_target_sha256(target_values, target_masks),
        scaler_state_sha256=scaler.state_sha256,
        model_state_sha256=state_sha,
    )


def predict_with_fitted_dense72_model(
    fitted: FittedDense72HierarchicalResidualBasis,
    raw: np.ndarray,
    station: np.ndarray,
    static: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int | None = None,
) -> np.ndarray:
    """Predict only the six official deltas from the sealed dense path."""

    raw_array = np.asarray(raw)
    station_array = np.asarray(station)
    static_array = np.asarray(static)
    size = _aligned_case_count(raw_array, station_array, static_array)
    if size == 0 or raw_array.shape != (size, CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise ValueError("prediction raw contexts differ from the 48-hour contract")
    station_codes = _station_codes(station_array, size=size)
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
                    selected_device.type == "cuda"
                    and fitted.training_config.use_bf16_on_cuda
                ),
            ):
                dense = model.forward_dense(
                    torch.from_numpy(raw_values[start:stop]).to(selected_device),
                    torch.from_numpy(station_codes[start:stop]).to(selected_device),
                    torch.from_numpy(scaled_static[start:stop]).to(selected_device),
                )
                prediction = dense[:, OFFICIAL_DENSE_INDICES]
            outputs.append(prediction.float().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    if result.shape != (size, len(OFFICIAL_DENSE_INDICES)) or not np.isfinite(result).all():
        raise RuntimeError("official dense72 prediction shape or finiteness changed")
    del model
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def fit_dense72_and_predict(
    raw: np.ndarray,
    station: np.ndarray,
    static: np.ndarray,
    target_delta: np.ndarray,
    target_mask: np.ndarray,
    case_weight: np.ndarray,
    train_ids: Sequence[int] | np.ndarray,
    prediction_ids: Sequence[int] | np.ndarray,
    *,
    seed: int,
    device: str | torch.device,
    model_config: HierarchicalResidualBasisConfig | None = None,
    training_config: FixedBasisTrainingConfig | None = None,
) -> tuple[np.ndarray, FittedDense72HierarchicalResidualBasis]:
    """Select train rows before any target validation or conversion."""

    raw_array = np.asarray(raw)
    station_array = np.asarray(station)
    static_array = np.asarray(static)
    target_array = np.asarray(target_delta)
    mask_array = np.asarray(target_mask)
    weight_array = np.asarray(case_weight)
    size = _aligned_case_count(
        raw_array,
        station_array,
        static_array,
        target_array,
        mask_array,
        weight_array,
    )
    train = _case_ids(train_ids, expected_size=None, role="training index")
    prediction = _case_ids(prediction_ids, expected_size=None, role="prediction index")
    if train.max() >= size or prediction.max() >= size:
        raise IndexError("train or prediction index lies outside aligned arrays")
    if np.intersect1d(train, prediction).size:
        raise PermissionError("training IDs overlap forbidden prediction IDs")
    fitted = fit_dense72_hierarchical_model(
        np.array(raw_array[train], copy=True),
        np.array(station_array[train], copy=True),
        np.array(static_array[train], copy=True),
        np.array(target_array[train], copy=True),
        np.array(mask_array[train], dtype=bool, copy=True),
        np.array(weight_array[train], copy=True),
        train,
        forbidden_case_ids=prediction,
        seed=seed,
        device=device,
        model_config=model_config,
        training_config=training_config,
    )
    predicted = predict_with_fitted_dense72_model(
        fitted,
        np.array(raw_array[prediction], copy=True),
        np.array(station_array[prediction], copy=True),
        np.array(static_array[prediction], copy=True),
        device=device,
    )
    return predicted, fitted


def save_fitted_dense72_model(
    fitted: FittedDense72HierarchicalResidualBasis,
    path: str | Path,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if model_state_sha256(fitted.state_dict) != fitted.model_state_sha256:
        raise PermissionError("fitted model state SHA differs before save")
    if fitted.scaler.state_sha256 != fitted.scaler_state_sha256:
        raise PermissionError("fitted scaler SHA differs before save")
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
        "valid_target_scalars_per_epoch": int(fitted.valid_target_scalars_per_epoch),
        "train_ids_sha256": fitted.train_ids_sha256,
        "train_context_sha256": fitted.train_context_sha256,
        "train_target_sha256": fitted.train_target_sha256,
        "scaler_state_sha256": fitted.scaler_state_sha256,
        "model_state_sha256": fitted.model_state_sha256,
    }
    with target.open("xb") as stream:
        torch.save(payload, stream)


def load_fitted_dense72_model(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> FittedDense72HierarchicalResidualBasis:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != MODEL_BUNDLE_SCHEMA:
        raise ValueError("saved dense72 model schema differs")
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
        raise PermissionError("saved dense72 scaler SHA differs")
    state = {
        str(name): tensor.detach().cpu().clone()
        for name, tensor in payload["state_dict"].items()
    }
    state_sha = model_state_sha256(state)
    if state_sha != payload["model_state_sha256"]:
        raise PermissionError("saved dense72 model state SHA differs")
    train_ids_sha = _require_sha256(payload["train_ids_sha256"], field="train_ids_sha256")
    if train_ids_sha != scaler.fit_ids_sha256:
        raise PermissionError("saved training-ID hashes disagree")
    return FittedDense72HierarchicalResidualBasis(
        model_config=model_config,
        training_config=training_config,
        seed=int(payload["seed"]),
        scaler=scaler,
        state_dict=state,
        training_steps=int(payload["training_steps"]),
        valid_target_scalars_per_epoch=int(payload["valid_target_scalars_per_epoch"]),
        train_ids_sha256=train_ids_sha,
        train_context_sha256=_require_sha256(
            payload["train_context_sha256"], field="train_context_sha256"
        ),
        train_target_sha256=_require_sha256(
            payload["train_target_sha256"], field="train_target_sha256"
        ),
        scaler_state_sha256=scaler.state_sha256,
        model_state_sha256=state_sha,
    )


__all__ = [
    "CONTEXT_20M_STEPS",
    "DENSE_TARGET_STEPS",
    "FixedBasisTrainingConfig",
    "FittedDense72HierarchicalResidualBasis",
    "HierarchicalResidualBasisConfig",
    "HierarchicalResidualBasisForecaster",
    "INPUT_CHANNELS",
    "MODEL_BUNDLE_SCHEMA",
    "OFFICIAL_DENSE_INDICES",
    "StaticRobustScaler",
    "fit_dense72_and_predict",
    "fit_dense72_hierarchical_model",
    "load_fitted_dense72_model",
    "predict_with_fitted_dense72_model",
    "save_fitted_dense72_model",
]
