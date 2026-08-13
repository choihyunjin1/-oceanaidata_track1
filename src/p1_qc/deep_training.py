"""Fold-safe training utilities for the P1 sequence models."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from p1_qc.models_deep import (
    LossConfig,
    PatchTransformerConfig,
    TCNConfig,
    build_bce_dice_aux_loss,
    build_sequence_model,
    seed_torch,
)


@dataclass(frozen=True)
class WindowSpec:
    start: int
    stop: int


@dataclass(frozen=True)
class SequenceTrainConfig:
    architecture: Literal["tcn", "patch_transformer"] = "tcn"
    window_steps: int = 2016
    stride_steps: int = 1008
    batch_size: int = 16
    max_epochs: int = 50
    patience: int = 8
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    negative_ratio: float = 1.0
    auxiliary_weight: float = 0.2
    use_bfloat16: bool = True
    seed: int = 20260813

    def __post_init__(self) -> None:
        if self.window_steps < 2 or self.stride_steps < 1:
            raise ValueError("window_steps and stride_steps must be positive")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.negative_ratio < 0:
            raise ValueError("invalid optimizer or sampling value")


@dataclass
class SequenceTrainingResult:
    model: Any
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float
    center: np.ndarray
    scale: np.ndarray
    config: dict[str, Any]


def robust_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    values = np.where(np.isfinite(values), values, np.nan)
    center = np.nanmedian(values, axis=0).astype(np.float32)
    q25 = np.nanpercentile(values, 25, axis=0).astype(np.float32)
    q75 = np.nanpercentile(values, 75, axis=0).astype(np.float32)
    scale = q75 - q25
    fallback = np.nanstd(values, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-6), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-6), scale, 1.0).astype(np.float32)
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    return center, scale


def robust_transform(features: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = (np.asarray(features, dtype=np.float32) - center) / scale
    return np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10.0, 10.0)


def segment_bounds(segment_ids: np.ndarray) -> list[tuple[int, int]]:
    segments = np.asarray(segment_ids)
    if segments.ndim != 1:
        raise ValueError("segment_ids must be one-dimensional")
    if len(segments) == 0:
        return []
    changes = np.flatnonzero(segments[1:] != segments[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(segments)]))
    return [
        (int(start), int(stop)) for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def _centered_window(center: int, lower: int, upper: int, window: int) -> WindowSpec:
    start = center - window // 2
    start = max(lower, min(start, max(lower, upper - window)))
    return WindowSpec(start=start, stop=min(upper, start + window))


def balanced_window_specs(
    labels: np.ndarray,
    segment_ids: np.ndarray,
    *,
    window_steps: int,
    negative_ratio: float = 1.0,
    seed: int = 20260813,
) -> list[WindowSpec]:
    target = np.asarray(labels, dtype=np.int8)
    if target.ndim != 1 or not np.isin(target, [0, 1]).all():
        raise ValueError("labels must be a one-dimensional binary array")
    bounds = segment_bounds(np.asarray(segment_ids))
    positive_specs: list[WindowSpec] = []
    normal_candidates: list[WindowSpec] = []
    for lower, upper in bounds:
        local = target[lower:upper]
        positive = local == 1
        starts = np.flatnonzero(positive & np.r_[True, ~positive[:-1]])
        stops = np.flatnonzero(positive & np.r_[~positive[1:], True]) + 1
        for event_start, event_stop in zip(starts, stops, strict=True):
            center = lower + int((event_start + event_stop - 1) // 2)
            positive_specs.append(_centered_window(center, lower, upper, window_steps))
        if not positive.any():
            stride = max(1, window_steps // 2)
            for start in range(lower, upper, stride):
                normal_candidates.append(WindowSpec(start, min(upper, start + window_steps)))
        else:
            stride = max(1, window_steps // 2)
            for start in range(lower, upper, stride):
                stop = min(upper, start + window_steps)
                if not target[start:stop].any():
                    normal_candidates.append(WindowSpec(start, stop))
    rng = np.random.default_rng(seed)
    normal_count = min(len(normal_candidates), int(np.ceil(len(positive_specs) * negative_ratio)))
    if positive_specs and normal_count:
        selected = rng.choice(len(normal_candidates), size=normal_count, replace=False)
        normal_specs = [normal_candidates[int(index)] for index in selected]
    elif not positive_specs:
        normal_specs = normal_candidates
    else:
        normal_specs = []
    specs = positive_specs + normal_specs
    rng.shuffle(specs)
    return specs


def inference_window_specs(
    segment_ids: np.ndarray,
    *,
    window_steps: int,
    stride_steps: int,
) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    for lower, upper in segment_bounds(np.asarray(segment_ids)):
        length = upper - lower
        if length <= window_steps:
            specs.append(WindowSpec(lower, upper))
            continue
        starts = list(range(lower, upper - window_steps + 1, stride_steps))
        last = upper - window_steps
        if not starts or starts[-1] != last:
            starts.append(last)
        specs.extend(WindowSpec(start, start + window_steps) for start in starts)
    return specs


class SequenceWindowDataset:
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        auxiliary: np.ndarray | None,
        specs: list[WindowSpec],
        window_steps: int,
    ) -> None:
        import torch

        self.torch = torch
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.auxiliary = None if auxiliary is None else np.asarray(auxiliary, dtype=np.float32)
        self.specs = specs
        self.window_steps = window_steps

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        spec = self.specs[index]
        length = spec.stop - spec.start
        feature = np.zeros((self.window_steps, self.features.shape[1]), dtype=np.float32)
        target = np.zeros(self.window_steps, dtype=np.float32)
        valid = np.zeros(self.window_steps, dtype=bool)
        feature[:length] = self.features[spec.start : spec.stop]
        target[:length] = self.labels[spec.start : spec.stop]
        valid[:length] = True
        item: dict[str, Any] = {
            "features": self.torch.from_numpy(feature),
            "target": self.torch.from_numpy(target),
            "valid": self.torch.from_numpy(valid),
            "start": spec.start,
            "stop": spec.stop,
        }
        if self.auxiliary is not None:
            aux = np.zeros((self.window_steps, self.auxiliary.shape[1]), dtype=np.float32)
            aux[:length] = self.auxiliary[spec.start : spec.stop]
            item["auxiliary"] = self.torch.from_numpy(aux)
        return item


def _model_configuration(
    architecture: str,
    input_dim: int,
    parameters: dict[str, Any] | None,
) -> TCNConfig | PatchTransformerConfig:
    values = dict(parameters or {})
    values["input_dim"] = input_dim
    if architecture == "tcn":
        return TCNConfig(**values)
    if architecture == "patch_transformer":
        return PatchTransformerConfig(**values)
    raise ValueError(f"unknown architecture: {architecture}")


def train_sequence_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_auxiliary: np.ndarray | None,
    train_segments: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_auxiliary: np.ndarray | None,
    validation_segments: np.ndarray,
    *,
    config: SequenceTrainConfig,
    model_parameters: dict[str, Any] | None = None,
    device: str | None = None,
) -> SequenceTrainingResult:
    import torch
    from torch.utils.data import DataLoader

    seed_torch(config.seed, deterministic=False)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    center, scale = robust_fit(train_features)
    train_values = robust_transform(train_features, center, scale)
    validation_values = robust_transform(validation_features, center, scale)
    train_specs = balanced_window_specs(
        train_labels,
        train_segments,
        window_steps=config.window_steps,
        negative_ratio=config.negative_ratio,
        seed=config.seed,
    )
    validation_specs = balanced_window_specs(
        validation_labels,
        validation_segments,
        window_steps=config.window_steps,
        negative_ratio=max(2.0, config.negative_ratio),
        seed=config.seed + 1,
    )
    if not train_specs or not validation_specs:
        raise ValueError("training and validation must each provide at least one sequence window")
    train_dataset = SequenceWindowDataset(
        train_values, train_labels, train_auxiliary, train_specs, config.window_steps
    )
    validation_dataset = SequenceWindowDataset(
        validation_values,
        validation_labels,
        validation_auxiliary,
        validation_specs,
        config.window_steps,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model_config = _model_configuration(
        config.architecture, train_values.shape[1], model_parameters
    )
    model = build_sequence_model(config.architecture, model_config).to(device)
    positives = max(1, int(np.asarray(train_labels).sum()))
    negatives = max(1, len(train_labels) - positives)
    positive_weight = float(np.sqrt(negatives / positives))
    loss_function = build_bce_dice_aux_loss(
        LossConfig(
            bce_weight=1.0,
            dice_weight=1.0,
            auxiliary_weight=config.auxiliary_weight,
            positive_weight=positive_weight,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    use_amp = device.startswith("cuda") and config.use_bfloat16
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0

    for epoch in range(config.max_epochs):
        model.train()
        train_total = 0.0
        train_rows = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            valid = batch["valid"].to(device)
            auxiliary = batch.get("auxiliary")
            auxiliary = None if auxiliary is None else auxiliary.to(device)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                output = model(features, padding_mask=~valid)
                loss = loss_function(output, target, auxiliary_target=auxiliary, valid_mask=valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.detach().cpu()) * len(features)
            train_rows += len(features)

        model.eval()
        validation_total = 0.0
        validation_rows = 0
        with torch.inference_mode():
            for batch in validation_loader:
                features = batch["features"].to(device)
                target = batch["target"].to(device)
                valid = batch["valid"].to(device)
                auxiliary = batch.get("auxiliary")
                auxiliary = None if auxiliary is None else auxiliary.to(device)
                with torch.autocast(
                    device_type="cuda" if device.startswith("cuda") else "cpu",
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    output = model(features, padding_mask=~valid)
                    loss = loss_function(
                        output, target, auxiliary_target=auxiliary, valid_mask=valid
                    )
                validation_total += float(loss.detach().cpu()) * len(features)
                validation_rows += len(features)
        row = {
            "epoch": float(epoch),
            "train_loss": train_total / max(1, train_rows),
            "validation_loss": validation_total / max(1, validation_rows),
        }
        history.append(row)
        if row["validation_loss"] < best_loss - 1.0e-5:
            best_loss = row["validation_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("sequence training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return SequenceTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        center=center,
        scale=scale,
        config={
            "training": asdict(config),
            "model": asdict(model_config),
            "device": device,
        },
    )


def predict_sequence(
    result: SequenceTrainingResult,
    features: np.ndarray,
    segment_ids: np.ndarray,
    *,
    window_steps: int,
    stride_steps: int,
    batch_size: int = 16,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    import torch
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    values = robust_transform(features, result.center, result.scale)
    specs = inference_window_specs(
        segment_ids, window_steps=window_steps, stride_steps=stride_steps
    )
    zeros = np.zeros(len(values), dtype=np.float32)
    dataset = SequenceWindowDataset(values, zeros, None, specs, window_steps)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    result.model.to(device).eval()
    probability_sum = np.zeros(len(values), dtype=np.float64)
    probability_count = np.zeros(len(values), dtype=np.int32)
    auxiliary_sum: np.ndarray | None = None
    auxiliary_count = np.zeros(len(values), dtype=np.int32)
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["features"].to(device)
            valid = batch["valid"].to(device)
            output = result.model(inputs, padding_mask=~valid)
            binary = torch.sigmoid(output.logits).cpu().numpy()
            auxiliary = (
                None
                if output.aux_logits is None
                else torch.sigmoid(output.aux_logits).cpu().numpy()
            )
            if auxiliary is not None and auxiliary_sum is None:
                auxiliary_sum = np.zeros((len(values), auxiliary.shape[-1]), dtype=np.float64)
            for row, (start, stop) in enumerate(
                zip(batch["start"].numpy(), batch["stop"].numpy(), strict=True)
            ):
                length = int(stop - start)
                probability_sum[start:stop] += binary[row, :length]
                probability_count[start:stop] += 1
                if auxiliary is not None and auxiliary_sum is not None:
                    auxiliary_sum[start:stop] += auxiliary[row, :length]
                    auxiliary_count[start:stop] += 1
    if not (probability_count > 0).all():
        raise RuntimeError("sequence inference failed to cover every row")
    probabilities = probability_sum / probability_count
    auxiliary_probabilities = None
    if auxiliary_sum is not None:
        auxiliary_probabilities = auxiliary_sum / np.maximum(auxiliary_count[:, None], 1)
    return probabilities.astype(np.float32), (
        None if auxiliary_probabilities is None else auxiliary_probabilities.astype(np.float32)
    )


def save_sequence_checkpoint(result: SequenceTrainingResult, path: str | Path) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "center": result.center,
            "scale": result.scale,
            "config": result.config,
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "history": result.history,
        },
        path,
    )
    return path


def predefined_search_space(
    architecture: Literal["tcn", "patch_transformer"],
    *,
    causal: bool,
) -> list[dict[str, Any]]:
    if architecture == "tcn":
        return [
            {"channels": channels, "kernel_size": kernel, "dropout": dropout, "causal": causal}
            for channels in ((32, 64, 64), (64, 64, 128), (64, 128, 128))
            for kernel in (3, 5)
            for dropout in (0.1, 0.2)
        ]
    if architecture == "patch_transformer":
        return [
            {
                "d_model": width,
                "nhead": 4,
                "num_layers": layers,
                "dim_feedforward": width * 2,
                "dropout": dropout,
                "patch_size": 6,
                "patch_stride": 6,
                "causal": causal,
            }
            for width in (48, 64, 96)
            for layers in (2, 3)
            for dropout in (0.1, 0.2)
        ]
    raise ValueError(f"unknown architecture: {architecture}")
