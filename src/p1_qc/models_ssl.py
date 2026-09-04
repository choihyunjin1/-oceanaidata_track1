"""Fold-local masked reconstruction pretraining for robust P1 features.

This module expects *already robust-normalized* numeric features, never raw
absolute temperature.  Windows are made inside contiguous segment IDs, and
row provenance is checked before training so validation rows cannot enter the
self-supervised training corpus by accident.
"""

from __future__ import annotations

import copy
import importlib
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


class OptionalDependencyError(ImportError):
    """Raised when the optional PyTorch overlay is unavailable."""


@dataclass(frozen=True)
class SSLModelConfig:
    input_dim: int
    channels: tuple[int, ...] = (32, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.1
    causal: bool = False

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if not self.channels or any(channel < 1 for channel in self.channels):
            raise ValueError("channels must contain positive integers")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class SSLTrainConfig:
    window_steps: int = 288
    stride_steps: int = 144
    mask_fraction: float = 0.25
    mask_block_steps: int = 12
    batch_size: int = 32
    max_epochs: int = 40
    patience: int = 6
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    use_bfloat16: bool = True
    seed: int = 20260813
    deterministic: bool = True
    normalized_abs_limit: float = 100.0

    def __post_init__(self) -> None:
        if self.window_steps < 2 or self.stride_steps < 1:
            raise ValueError("window_steps must be >=2 and stride_steps positive")
        if not 0 < self.mask_fraction < 1:
            raise ValueError("mask_fraction must be in (0, 1)")
        if not 1 <= self.mask_block_steps <= self.window_steps:
            raise ValueError("mask_block_steps must lie in [1, window_steps]")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if self.normalized_abs_limit <= 0:
            raise ValueError("normalized_abs_limit must be positive")


@dataclass(frozen=True)
class SSLWindow:
    start: int
    stop: int
    segment_id: Any

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class SSLTrainingResult:
    model: Any
    model_config: SSLModelConfig
    train_config: SSLTrainConfig
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float
    train_row_ids: np.ndarray
    train_windows: tuple[SSLWindow, ...]
    validation_windows: tuple[SSLWindow, ...]


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OptionalDependencyError(
            "PyTorch is required for SSL; install requirements-dl.txt"
        ) from exc
    return torch, nn, functional


def _seed(seed: int, deterministic: bool) -> None:
    if deterministic:
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


def _validated_features(
    features: np.ndarray | Sequence[Sequence[float]],
    *,
    input_dim: int | None = None,
    normalized_abs_limit: float = 100.0,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must be a non-empty (rows, features) matrix")
    if input_dim is not None and values.shape[1] != input_dim:
        raise ValueError(f"expected {input_dim} features, got {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError("robust-normalized features must be finite")
    if float(np.max(np.abs(values))) > normalized_abs_limit:
        raise ValueError(
            "features exceed normalized_abs_limit; pass robust-normalized/clipped "
            "features, not raw absolute temperature"
        )
    return values


def _validated_vector(values: Sequence[Any] | np.ndarray, rows: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (rows,):
        raise ValueError(f"{name} must be one-dimensional with {rows} rows")
    return array


def assert_fold_local_rows(
    train_row_ids: Sequence[Any] | np.ndarray,
    validation_row_ids: Sequence[Any] | np.ndarray | None = None,
) -> None:
    """Reject duplicate training provenance and train/validation overlap."""

    train = np.asarray(train_row_ids)
    if train.ndim != 1:
        raise ValueError("train_row_ids must be one-dimensional")
    if len(np.unique(train)) != len(train):
        raise ValueError("train_row_ids must be unique")
    if validation_row_ids is None:
        return
    validation = np.asarray(validation_row_ids)
    if validation.ndim != 1:
        raise ValueError("validation_row_ids must be one-dimensional")
    overlap = np.intersect1d(train, validation)
    if len(overlap):
        preview = overlap[:5].tolist()
        raise ValueError(
            f"validation rows leaked into SSL training ({len(overlap)} overlaps; "
            f"examples={preview})"
        )


def gap_aware_windows(
    segment_ids: Sequence[Any] | np.ndarray,
    *,
    window_steps: int,
    stride_steps: int,
) -> list[SSLWindow]:
    """Create full/partial covering windows that never cross a gap segment."""

    segments = np.asarray(segment_ids)
    if segments.ndim != 1:
        raise ValueError("segment_ids must be one-dimensional")
    if window_steps < 2 or stride_steps < 1:
        raise ValueError("window_steps must be >=2 and stride_steps positive")
    if len(segments) == 0:
        return []
    change = np.flatnonzero(segments[1:] != segments[:-1]) + 1
    boundaries = np.concatenate(([0], change, [len(segments)]))
    windows: list[SSLWindow] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        lower, upper = int(start), int(stop)
        segment_id = segments[lower].item() if hasattr(segments[lower], "item") else segments[lower]
        length = upper - lower
        if length <= window_steps:
            windows.append(SSLWindow(lower, upper, segment_id))
            continue
        starts = list(range(lower, upper - window_steps + 1, stride_steps))
        last = upper - window_steps
        if starts[-1] != last:
            starts.append(last)
        windows.extend(
            SSLWindow(position, position + window_steps, segment_id) for position in starts
        )
    return windows


def block_mask(
    length: int,
    *,
    mask_fraction: float,
    block_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a deterministic block mask with at least one visible timestep."""

    if length < 2:
        raise ValueError("length must be at least 2 for masked reconstruction")
    target = min(length - 1, max(1, int(round(length * mask_fraction))))
    result = np.zeros(length, dtype=bool)
    attempts = 0
    while int(result.sum()) < target and attempts < length * 8:
        width = min(block_steps, target - int(result.sum()), length - 1)
        start = int(rng.integers(0, length - width + 1))
        result[start : start + width] = True
        attempts += 1
    if int(result.sum()) < target:
        available = np.flatnonzero(~result)
        result[rng.choice(available, size=target - int(result.sum()), replace=False)] = True
    return result


def build_masked_tcn_autoencoder(config: SSLModelConfig) -> Any:
    """Build a residual dilated 1-D encoder and pointwise reconstruction head."""

    torch, nn, functional = _require_torch()

    class _Block(nn.Module):
        def __init__(self, input_channels: int, output_channels: int, dilation: int) -> None:
            super().__init__()
            padding = (config.kernel_size - 1) * dilation
            self.left_padding = padding
            self.conv = nn.Conv1d(
                input_channels,
                output_channels,
                config.kernel_size,
                dilation=dilation,
                padding=0 if config.causal else padding // 2,
            )
            self.project = (
                nn.Identity()
                if input_channels == output_channels
                else nn.Conv1d(input_channels, output_channels, 1)
            )
            self.norm = nn.GroupNorm(1, output_channels)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, inputs: Any) -> Any:
            convolved_inputs = (
                functional.pad(inputs, (self.left_padding, 0)) if config.causal else inputs
            )
            output = self.conv(convolved_inputs)
            if not config.causal and output.shape[-1] != inputs.shape[-1]:
                output = output[..., : inputs.shape[-1]]
                if output.shape[-1] < inputs.shape[-1]:
                    output = functional.pad(output, (0, inputs.shape[-1] - output.shape[-1]))
            output = self.dropout(functional.gelu(output))
            return functional.gelu(self.norm(output + self.project(inputs)))

    class _MaskedTCNAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            incoming = config.input_dim
            for level, outgoing in enumerate(config.channels):
                blocks.append(_Block(incoming, outgoing, dilation=2**level))
                incoming = outgoing
            self.encoder = nn.ModuleList(blocks)
            self.mask_token = nn.Parameter(torch.zeros(config.input_dim))
            self.reconstruction_head = nn.Conv1d(incoming, config.input_dim, 1)
            self.embedding_dim = incoming

        def encode(self, inputs: Any) -> Any:
            if inputs.ndim != 3 or inputs.shape[-1] != config.input_dim:
                raise ValueError(f"inputs must have shape (batch, time, {config.input_dim})")
            hidden = inputs.transpose(1, 2)
            for layer in self.encoder:
                hidden = layer(hidden)
            return hidden.transpose(1, 2)

        def forward(self, inputs: Any, mask: Any | None = None) -> tuple[Any, Any]:
            if mask is not None:
                if tuple(mask.shape) != tuple(inputs.shape[:2]):
                    raise ValueError("mask must have shape (batch, time)")
                token = self.mask_token.to(dtype=inputs.dtype).view(1, 1, -1)
                inputs = torch.where(mask.bool().unsqueeze(-1), token, inputs)
            embedding = self.encode(inputs)
            reconstruction = self.reconstruction_head(embedding.transpose(1, 2)).transpose(1, 2)
            return reconstruction, embedding

    return _MaskedTCNAutoencoder()


class _WindowDataset:
    def __init__(
        self,
        features: np.ndarray,
        windows: Sequence[SSLWindow],
        config: SSLTrainConfig,
        *,
        training: bool,
    ) -> None:
        torch, _, _ = _require_torch()
        self.torch = torch
        self.features = features
        self.windows = tuple(windows)
        self.config = config
        self.training = training
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        length = window.length
        if length < 2:
            raise ValueError("each SSL segment must contain at least two rows")
        feature = np.zeros(
            (self.config.window_steps, self.features.shape[1]),
            dtype=np.float32,
        )
        valid = np.zeros(self.config.window_steps, dtype=bool)
        mask = np.zeros(self.config.window_steps, dtype=bool)
        feature[:length] = self.features[window.start : window.stop]
        valid[:length] = True
        seed_offset = self.epoch if self.training else 1_000_003
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, seed_offset, index]))
        mask[:length] = block_mask(
            length,
            mask_fraction=self.config.mask_fraction,
            block_steps=self.config.mask_block_steps,
            rng=rng,
        )
        return {
            "feature": self.torch.from_numpy(feature),
            "valid": self.torch.from_numpy(valid),
            "mask": self.torch.from_numpy(mask),
        }


def _masked_reconstruction_loss(
    reconstruction: Any,
    target: Any,
    mask: Any,
    valid: Any,
) -> Any:
    selected = mask.bool() & valid.bool()
    if not bool(selected.any()):
        raise ValueError("every SSL batch must contain at least one masked valid row")
    error = (reconstruction - target).square().mean(dim=-1)
    return error[selected].mean()


def _run_epoch(
    *,
    model: Any,
    loader: Any,
    optimizer: Any | None,
    device: str,
    use_bfloat16: bool,
) -> float:
    torch, _, _ = _require_torch()
    training = optimizer is not None
    model.train(training)
    total = 0.0
    rows = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            feature = batch["feature"].to(device)
            valid = batch["valid"].to(device)
            mask = batch["mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                reconstruction, _ = model(feature, mask)
                loss = _masked_reconstruction_loss(reconstruction, feature, mask, valid)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += float(loss.detach().cpu()) * len(feature)
            rows += len(feature)
    return total / max(1, rows)


def train_masked_reconstruction(
    train_features: np.ndarray | Sequence[Sequence[float]],
    train_segment_ids: Sequence[Any] | np.ndarray,
    train_row_ids: Sequence[Any] | np.ndarray,
    *,
    validation_features: np.ndarray | Sequence[Sequence[float]] | None = None,
    validation_segment_ids: Sequence[Any] | np.ndarray | None = None,
    validation_row_ids: Sequence[Any] | np.ndarray | None = None,
    model_config: SSLModelConfig | None = None,
    train_config: SSLTrainConfig | None = None,
    device: str | None = None,
) -> SSLTrainingResult:
    """Train only on explicit training rows and early-stop on held-out rows.

    ``train_row_ids`` and ``validation_row_ids`` are immutable provenance keys
    (a MultiIndex-derived integer/string ID is suitable), not array positions.
    Any overlap raises before model construction.
    """

    torch, _, _ = _require_torch()
    from torch.utils.data import DataLoader

    train_config = train_config or SSLTrainConfig()
    train_values = _validated_features(
        train_features,
        normalized_abs_limit=train_config.normalized_abs_limit,
    )
    model_config = model_config or SSLModelConfig(input_dim=train_values.shape[1])
    train_values = _validated_features(
        train_values,
        input_dim=model_config.input_dim,
        normalized_abs_limit=train_config.normalized_abs_limit,
    )
    train_segments = _validated_vector(
        train_segment_ids,
        len(train_values),
        "train_segment_ids",
    )
    train_ids = _validated_vector(train_row_ids, len(train_values), "train_row_ids")
    assert_fold_local_rows(train_ids, validation_row_ids)

    provided_validation = any(
        item is not None
        for item in (validation_features, validation_segment_ids, validation_row_ids)
    )
    if provided_validation and not all(
        item is not None
        for item in (validation_features, validation_segment_ids, validation_row_ids)
    ):
        raise ValueError("all validation arrays must be supplied together")
    if validation_features is None:
        validation_values = train_values
        validation_segments = train_segments
    else:
        validation_values = _validated_features(
            validation_features,
            input_dim=model_config.input_dim,
            normalized_abs_limit=train_config.normalized_abs_limit,
        )
        validation_segments = _validated_vector(
            validation_segment_ids,
            len(validation_values),
            "validation_segment_ids",
        )
        validation_ids = _validated_vector(
            validation_row_ids,
            len(validation_values),
            "validation_row_ids",
        )
        assert_fold_local_rows(train_ids, validation_ids)

    train_windows = gap_aware_windows(
        train_segments,
        window_steps=train_config.window_steps,
        stride_steps=train_config.stride_steps,
    )
    validation_windows = gap_aware_windows(
        validation_segments,
        window_steps=train_config.window_steps,
        stride_steps=train_config.stride_steps,
    )
    if not train_windows or not validation_windows:
        raise ValueError("training and validation each require at least one window")
    if any(window.length < 2 for window in (*train_windows, *validation_windows)):
        raise ValueError("every contiguous segment used by SSL needs at least two rows")

    _seed(train_config.seed, train_config.deterministic)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    model = build_masked_tcn_autoencoder(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    train_dataset = _WindowDataset(train_values, train_windows, train_config, training=True)
    validation_dataset = _WindowDataset(
        validation_values,
        validation_windows,
        train_config,
        training=False,
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    use_bfloat16 = (
        train_config.use_bfloat16 and device.startswith("cuda") and torch.cuda.is_bf16_supported()
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0
    for epoch in range(train_config.max_epochs):
        train_dataset.epoch = epoch
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_bfloat16=use_bfloat16,
        )
        validation_loss = _run_epoch(
            model=model,
            loader=validation_loader,
            optimizer=None,
            device=device,
            use_bfloat16=use_bfloat16,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= train_config.patience:
                break
    if best_state is None:  # pragma: no cover - finite validated inputs guarantee a first state
        raise RuntimeError("SSL training produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return SSLTrainingResult(
        model=model,
        model_config=model_config,
        train_config=train_config,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        train_row_ids=train_ids.copy(),
        train_windows=tuple(train_windows),
        validation_windows=tuple(validation_windows),
    )


def extract_ssl_embeddings(
    result: SSLTrainingResult,
    features: np.ndarray | Sequence[Sequence[float]],
    segment_ids: Sequence[Any] | np.ndarray,
    *,
    window_steps: int | None = None,
    stride_steps: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Return overlap-averaged per-row encoder embeddings without crossing gaps."""

    torch, _, _ = _require_torch()
    from torch.utils.data import DataLoader

    values = _validated_features(
        features,
        input_dim=result.model_config.input_dim,
        normalized_abs_limit=result.train_config.normalized_abs_limit,
    )
    segments = _validated_vector(segment_ids, len(values), "segment_ids")
    effective_window = window_steps or result.train_config.window_steps
    effective_stride = stride_steps or result.train_config.stride_steps
    windows = gap_aware_windows(
        segments,
        window_steps=effective_window,
        stride_steps=effective_stride,
    )
    inference_config = SSLTrainConfig(
        **{
            **asdict(result.train_config),
            "window_steps": effective_window,
            "stride_steps": effective_stride,
            "mask_block_steps": min(result.train_config.mask_block_steps, effective_window),
        }
    )
    dataset = _WindowDataset(values, windows, inference_config, training=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or result.train_config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    result.model.to(device).eval()
    embedding_dim = int(result.model.embedding_dim)
    total = np.zeros((len(values), embedding_dim), dtype=np.float64)
    count = np.zeros(len(values), dtype=np.int32)
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            feature = batch["feature"].to(device)
            embedding = result.model.encode(feature).float().cpu().numpy()
            for row in range(len(feature)):
                window = windows[offset + row]
                total[window.start : window.stop] += embedding[row, : window.length]
                count[window.start : window.stop] += 1
            offset += len(feature)
    if not (count > 0).all():
        raise RuntimeError("embedding windows did not cover every row")
    return (total / count[:, None]).astype(np.float32)


def save_ssl_checkpoint(result: SSLTrainingResult, path: str | Path) -> Path:
    """Save state and reproducibility metadata; source observations are omitted."""

    torch, _, _ = _require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "state_dict": result.model.state_dict(),
            "model_config": asdict(result.model_config),
            "train_config": asdict(result.train_config),
            "history": result.history,
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            # Tensor storage keeps ``weights_only=True`` loading compatible;
            # string provenance falls back to a primitive Python list.
            "train_row_ids": (
                torch.as_tensor(result.train_row_ids)
                if np.issubdtype(result.train_row_ids.dtype, np.number)
                else result.train_row_ids.tolist()
            ),
            "train_windows": [asdict(window) for window in result.train_windows],
            "validation_windows": [asdict(window) for window in result.validation_windows],
        },
        destination,
    )
    return destination


def load_ssl_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> SSLTrainingResult:
    """Restore an SSL encoder with explicit safe state-dict construction."""

    torch, _, _ = _require_torch()
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch releases
        payload = torch.load(Path(path), map_location=device)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported SSL checkpoint format")
    model_config = SSLModelConfig(**payload["model_config"])
    train_config = SSLTrainConfig(**payload["train_config"])
    model = build_masked_tcn_autoencoder(model_config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return SSLTrainingResult(
        model=model,
        model_config=model_config,
        train_config=train_config,
        history=[dict(row) for row in payload["history"]],
        best_epoch=int(payload["best_epoch"]),
        best_validation_loss=float(payload["best_validation_loss"]),
        train_row_ids=(
            payload["train_row_ids"].cpu().numpy()
            if hasattr(payload["train_row_ids"], "cpu")
            else np.asarray(payload["train_row_ids"])
        ),
        train_windows=tuple(SSLWindow(**item) for item in payload["train_windows"]),
        validation_windows=tuple(SSLWindow(**item) for item in payload["validation_windows"]),
    )


# Short aliases for experiment orchestration code.
train_ssl = train_masked_reconstruction
extract_embeddings = extract_ssl_embeddings
