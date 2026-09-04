"""Deterministic GPU training and blocked evaluation for the P2 model tournament."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from p2_restore.deep_data import (
    P2Panel,
    PanelNormalizer,
    make_chunk_bounds,
    time_block_mask,
)
from p2_restore.deep_models import ConditionalDiffusion, ModelName, build_model, count_parameters
from p2_restore.model import VALIDATION_BLOCKS

DEV_BLOCK = ("2024-07-01", "2024-09-01")


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelName
    learning_rate: float
    weight_decay: float = 1e-3
    max_epochs: int = 300
    patience: int = 30
    chunk_length: int = 512
    chunk_stride: int = 384
    batch_size: int = 12
    seed: int = 20260816
    evaluation_interval: int = 2
    diffusion_samples: int = 4


@dataclass
class FoldTrainingResult:
    block: str
    config: TrainingConfig
    parameter_count: int
    best_epoch: int
    best_rmse: float
    history: list[dict[str, float | int]]
    normalizer: PanelNormalizer
    state_dict: dict[str, Tensor]
    oof: pd.DataFrame

    def summary(self) -> dict[str, object]:
        return {
            "block": self.block,
            "config": asdict(self.config),
            "parameter_count": self.parameter_count,
            "best_epoch": self.best_epoch,
            "best_rmse": self.best_rmse,
            "history": self.history,
            "rows": len(self.oof),
            "by_layer_rmse": {
                str(layer): _rmse(
                    self.oof.loc[self.oof["layer"] == layer, "truth"].to_numpy(float),
                    self.oof.loc[self.oof["layer"] == layer, "prediction"].to_numpy(float),
                )
                for layer in (2, 3, 4)
            },
        }


@dataclass
class FullTrainingResult:
    config: TrainingConfig
    parameter_count: int
    epochs: int
    normalizer: PanelNormalizer
    state_dict: dict[str, Tensor]
    prediction: np.ndarray
    final_train_mse_c: float


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(prediction)) ** 2)))


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("P2 deep tournament requires the validated CUDA environment")
    return torch.device("cuda")


def _materialize_chunks(
    inputs: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    bounds: Sequence[tuple[int, int]],
    length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    x = np.zeros((len(bounds), length, inputs.shape[1]), dtype=np.float32)
    y = np.zeros((len(bounds), length, 3), dtype=np.float32)
    m = np.zeros((len(bounds), length, 3), dtype=np.float32)
    for number, (start, stop) in enumerate(bounds):
        width = stop - start
        x[number, :width] = inputs[start:stop]
        y[number, :width] = target[start:stop]
        m[number, :width] = mask[start:stop]
        if width and width < length:
            x[number, width:] = inputs[stop - 1]
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


def _predict_chunks(
    model: nn.Module,
    inputs: np.ndarray,
    bounds: Sequence[tuple[int, int]],
    *,
    length: int,
    batch_size: int,
    diffusion_samples: int,
    seed: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    total = np.zeros((len(inputs), 3), dtype=np.float64)
    weights = np.zeros(len(inputs), dtype=np.float64)
    taper = np.hanning(length + 2)[1:-1]
    taper = np.maximum(taper, 0.05)
    model.eval()
    set_deterministic_seed(seed)
    for offset in range(0, len(bounds), batch_size):
        selected = bounds[offset : offset + batch_size]
        batch = np.zeros((len(selected), length, inputs.shape[1]), dtype=np.float32)
        widths: list[int] = []
        for number, (start, stop) in enumerate(selected):
            width = stop - start
            widths.append(width)
            batch[number, :width] = inputs[start:stop]
            if width and width < length:
                batch[number, width:] = inputs[stop - 1]
        current = torch.from_numpy(batch).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = (
                model.predict(current, samples=diffusion_samples)
                if isinstance(model, ConditionalDiffusion)
                else model(current)
            )
        values = prediction.float().cpu().numpy()
        for number, ((start, stop), width) in enumerate(zip(selected, widths, strict=True)):
            local_weight = taper[:width]
            total[start:stop] += values[number, :width] * local_weight[:, None]
            weights[start:stop] += local_weight
    if (weights <= 0).any():
        raise RuntimeError("gap-aware inference failed to cover every timestamp")
    return total / weights[:, None]


def _validation_oof(
    panel: P2Panel,
    prediction: np.ndarray,
    validation_times: np.ndarray,
    block: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    kst = panel.times.tz_convert("Asia/Seoul")
    for offset, layer in enumerate((2, 3, 4)):
        keep = validation_times & panel.target_mask[:, offset] & np.isfinite(prediction[:, offset])
        rows.append(
            pd.DataFrame(
                {
                    "time": kst[keep].astype(str),
                    "layer": layer,
                    "truth": panel.target[keep, offset],
                    "prediction": prediction[keep, offset],
                    "baseline": panel.baseline[keep, offset],
                    "block": block,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def train_fold(
    panel: P2Panel,
    *,
    block: str,
    start: str,
    stop: str,
    config: TrainingConfig,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> FoldTrainingResult:
    set_deterministic_seed(config.seed)
    validation_times = time_block_mask(panel, start, stop)
    train_times = ~validation_times
    normalizer = PanelNormalizer.fit(panel, train_times)
    inputs = normalizer.transform_inputs(panel.inputs)
    target, target_mask = normalizer.transform_targets(panel)
    training_mask = target_mask & train_times[:, None]
    all_bounds = make_chunk_bounds(
        panel.segment_ids, length=config.chunk_length, stride=config.chunk_stride
    )
    train_bounds = tuple(
        bound for bound in all_bounds if training_mask[bound[0] : bound[1]].sum() >= 24
    )
    if not train_bounds:
        raise RuntimeError("no fold-local training chunks")
    chunk_x, chunk_y, chunk_mask = _materialize_chunks(
        inputs, target, training_mask, train_bounds, config.chunk_length
    )

    model = build_model(config.model, inputs.shape[1]).to(_device())
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.max_epochs, 1), eta_min=config.learning_rate * 0.05
    )
    layer_weights = torch.tensor(
        normalizer.residual_scale**2, device=_device(), dtype=torch.float32
    )
    best_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    rng = np.random.default_rng(config.seed)

    for epoch in range(1, config.max_epochs + 1):
        order = rng.permutation(len(train_bounds))
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for begin in range(0, len(order), config.batch_size):
            ids = torch.from_numpy(order[begin : begin + config.batch_size]).long()
            x = chunk_x[ids].to(_device(), non_blocking=True)
            y = chunk_y[ids].to(_device(), non_blocking=True)
            mask = chunk_mask[ids].to(_device(), non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if isinstance(model, ConditionalDiffusion):
                    loss = model.training_loss(x, y, mask, layer_weights)
                else:
                    predicted = model(x)
                    squared = (predicted - y).square() * layer_weights.view(1, 1, 3)
                    loss = (squared * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_weight = float(mask.sum().item())
            loss_sum += float(loss.detach().item()) * batch_weight
            weight_sum += batch_weight
        scheduler.step()

        should_score = (
            epoch == 1 or epoch % config.evaluation_interval == 0 or epoch == config.max_epochs
        )
        if not should_score:
            continue
        normalized_prediction = _predict_chunks(
            model,
            inputs,
            all_bounds,
            length=config.chunk_length,
            batch_size=config.batch_size,
            diffusion_samples=config.diffusion_samples,
            seed=config.seed + epoch,
        )
        prediction = normalizer.inverse_predictions(panel, normalized_prediction)
        oof = _validation_oof(panel, prediction, validation_times, block)
        score = _rmse(oof["truth"].to_numpy(float), oof["prediction"].to_numpy(float))
        current = {
            "epoch": epoch,
            "train_mse_c": loss_sum / max(weight_sum, 1.0),
            "validation_rmse": score,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(current)
        if progress is not None:
            progress(
                {
                    "model": config.model,
                    "block": block,
                    "epoch": epoch,
                    "max_epochs": config.max_epochs,
                    "rmse": score,
                    "best_rmse": min(best_rmse, score),
                }
            )
        if score < best_rmse - 1e-6:
            best_rmse = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += config.evaluation_interval
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    normalized_prediction = _predict_chunks(
        model,
        inputs,
        all_bounds,
        length=config.chunk_length,
        batch_size=config.batch_size,
        diffusion_samples=config.diffusion_samples,
        seed=config.seed + best_epoch,
    )
    prediction = normalizer.inverse_predictions(panel, normalized_prediction)
    final_oof = _validation_oof(panel, prediction, validation_times, block)
    final_rmse = _rmse(final_oof["truth"].to_numpy(float), final_oof["prediction"].to_numpy(float))
    return FoldTrainingResult(
        block,
        config,
        parameters,
        best_epoch,
        final_rmse,
        history,
        normalizer,
        best_state,
        final_oof,
    )


def run_blocked_model(
    panel: P2Panel,
    config: TrainingConfig,
    *,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, list[FoldTrainingResult]]:
    folds: list[FoldTrainingResult] = []
    for number, (block, (start, stop)) in enumerate(VALIDATION_BLOCKS.items()):
        folds.append(
            train_fold(
                panel,
                block=block,
                start=start,
                stop=stop,
                config=TrainingConfig(**{**asdict(config), "seed": config.seed + number}),
                progress=progress,
            )
        )
    oof = pd.concat([fold.oof for fold in folds], ignore_index=True)
    summary = {
        "model": config.model,
        "config": asdict(config),
        "rows": len(oof),
        "rmse": _rmse(oof["truth"].to_numpy(float), oof["prediction"].to_numpy(float)),
        "baseline_rmse": _rmse(oof["truth"].to_numpy(float), oof["baseline"].to_numpy(float)),
        "parameter_count": folds[0].parameter_count,
        "folds": {fold.block: fold.summary() for fold in folds},
    }
    return summary, oof, folds


def train_full_model(
    panel: P2Panel,
    config: TrainingConfig,
    *,
    epochs: int,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> FullTrainingResult:
    """Fit a fixed-epoch full-data model after outer validation has frozen the epoch."""

    if epochs < 1:
        raise ValueError("full-model epochs must be positive")
    set_deterministic_seed(config.seed)
    supervised_times = panel.target_mask.any(axis=1)
    normalizer = PanelNormalizer.fit(panel, supervised_times)
    inputs = normalizer.transform_inputs(panel.inputs)
    target, target_mask = normalizer.transform_targets(panel)
    all_bounds = make_chunk_bounds(
        panel.segment_ids, length=config.chunk_length, stride=config.chunk_stride
    )
    train_bounds = tuple(
        bound for bound in all_bounds if target_mask[bound[0] : bound[1]].sum() >= 24
    )
    chunk_x, chunk_y, chunk_mask = _materialize_chunks(
        inputs, target, target_mask, train_bounds, config.chunk_length
    )
    model = build_model(config.model, inputs.shape[1]).to(_device())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config.learning_rate * 0.05
    )
    layer_weights = torch.tensor(
        normalizer.residual_scale**2, device=_device(), dtype=torch.float32
    )
    rng = np.random.default_rng(config.seed)
    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_bounds))
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for begin in range(0, len(order), config.batch_size):
            ids = torch.from_numpy(order[begin : begin + config.batch_size]).long()
            x = chunk_x[ids].to(_device(), non_blocking=True)
            y = chunk_y[ids].to(_device(), non_blocking=True)
            mask = chunk_mask[ids].to(_device(), non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if isinstance(model, ConditionalDiffusion):
                    loss = model.training_loss(x, y, mask, layer_weights)
                else:
                    predicted = model(x)
                    squared = (predicted - y).square() * layer_weights.view(1, 1, 3)
                    loss = (squared * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_weight = float(mask.sum().item())
            loss_sum += float(loss.detach().item()) * batch_weight
            weight_sum += batch_weight
        scheduler.step()
        final_loss = loss_sum / max(weight_sum, 1.0)
        if progress is not None:
            progress(
                {
                    "model": config.model,
                    "epoch": epoch,
                    "max_epochs": epochs,
                    "train_mse_c": final_loss,
                }
            )
    normalized = _predict_chunks(
        model,
        inputs,
        all_bounds,
        length=config.chunk_length,
        batch_size=config.batch_size,
        diffusion_samples=config.diffusion_samples,
        seed=config.seed + epochs,
    )
    prediction = normalizer.inverse_predictions(panel, normalized)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return FullTrainingResult(
        config,
        count_parameters(model),
        epochs,
        normalizer,
        state,
        prediction,
        final_loss,
    )


def save_checkpoint(path: Path, result: FoldTrainingResult) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": result.config.model,
        "config": asdict(result.config),
        "block": result.block,
        "best_epoch": result.best_epoch,
        "input_center": result.normalizer.input_center,
        "input_scale": result.normalizer.input_scale,
        "residual_center": result.normalizer.residual_center,
        "residual_scale": result.normalizer.residual_scale,
        "state_dict": result.state_dict,
    }
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_full_checkpoint(path: Path, panel: P2Panel) -> np.ndarray:
    """Recreate a physical-temperature panel from one trusted local checkpoint."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TrainingConfig(**payload["config"])
    normalizer = PanelNormalizer(
        np.asarray(payload["input_center"], dtype=np.float64),
        np.asarray(payload["input_scale"], dtype=np.float64),
        np.asarray(payload["residual_center"], dtype=np.float64),
        np.asarray(payload["residual_scale"], dtype=np.float64),
    )
    inputs = normalizer.transform_inputs(panel.inputs)
    if inputs.shape[1] != len(normalizer.input_center):
        raise ValueError("saved P2 checkpoint input schema differs from the panel")
    model = build_model(config.model, inputs.shape[1]).to(_device())
    model.load_state_dict(payload["state_dict"])
    bounds = make_chunk_bounds(
        panel.segment_ids, length=config.chunk_length, stride=config.chunk_stride
    )
    normalized = _predict_chunks(
        model,
        inputs,
        bounds,
        length=config.chunk_length,
        batch_size=config.batch_size,
        diffusion_samples=config.diffusion_samples,
        seed=config.seed + int(payload["epochs"]),
    )
    return normalizer.inverse_predictions(panel, normalized)


def canonical_json_sha(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def blend_grid(
    deep_oof: pd.DataFrame,
    incumbent_oof: pd.DataFrame,
    *,
    weights: Sequence[float] = tuple(np.linspace(0.0, 1.0, 21)),
) -> dict[str, object]:
    keys = ["time", "layer", "block"]
    left = deep_oof.copy()
    right = incumbent_oof.loc[:, [*keys, "truth", "router_400"]].copy()
    left["time"] = pd.to_datetime(left["time"], utc=True)
    right["time"] = pd.to_datetime(right["time"], utc=True)
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one", suffixes=("", "_ref"))
    if len(merged) != len(deep_oof) or not np.allclose(merged["truth"], merged["truth_ref"]):
        raise ValueError("deep/incumbent OOF grain or labels differ")
    candidates: list[dict[str, float]] = []
    truth = merged["truth"].to_numpy(float)
    deep = merged["prediction"].to_numpy(float)
    incumbent = merged["router_400"].to_numpy(float)
    for deep_weight in weights:
        prediction = deep_weight * deep + (1.0 - deep_weight) * incumbent
        candidates.append({"deep_weight": float(deep_weight), "rmse": _rmse(truth, prediction)})
    selected = min(candidates, key=lambda row: (row["rmse"], row["deep_weight"]))
    return {"selected": selected, "candidates": candidates, "rows": len(merged)}
