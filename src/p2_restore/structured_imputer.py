"""Long-gap, target-aware structured masking for P2 profile restoration.

The public-only deep stack deliberately excluded target-layer observations.  This
module implements the complementary problem that the competition actually
provides: layers 2/3/4 are observed outside one simultaneous 61-day blackout.
Training examples reproduce that missingness pattern and never expose a held
block's target temperature or salinity to the model.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from p2_restore.deep_data import P2Panel
from p2_restore.deep_models import ModernTCNBlock

TARGET_LAYERS = (2, 3, 4)


@dataclass(frozen=True)
class StructuredMaskConfig:
    window_hours: int = 2160
    context_hours: int = 336
    mask_hours: tuple[int, ...] = (168, 720, 1464)
    hidden: int = 96
    blocks: int = 10
    dropout: float = 0.08
    batch_size: int = 4
    samples_per_epoch: int = 24
    min_mask_coverage: float = 0.70
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    max_epochs: int = 120
    evaluation_interval: int = 5
    patience: int = 25
    seed: int = 20260816

    def __post_init__(self) -> None:
        if self.window_hours < max(self.mask_hours) + 2 * self.context_hours:
            raise ValueError("window must contain the longest mask plus both contexts")
        if not self.mask_hours or min(self.mask_hours) < 24:
            raise ValueError("structured masks must be non-empty and at least one day")
        if not 0 < self.min_mask_coverage <= 1:
            raise ValueError("min_mask_coverage must be in (0, 1]")


@dataclass(frozen=True)
class HourlyPanel:
    times: pd.DatetimeIndex
    public_inputs: np.ndarray
    public_names: tuple[str, ...]
    baseline: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.times)
        if self.public_inputs.shape[0] != rows:
            raise ValueError("hourly public input rows do not match times")
        for values in (self.baseline, self.target, self.target_mask):
            if values.shape != (rows, 3):
                raise ValueError("hourly target arrays must have shape (rows, 3)")


@dataclass(frozen=True)
class StructuredNormalizer:
    public_center: np.ndarray
    public_scale: np.ndarray
    residual_center: np.ndarray
    residual_scale: np.ndarray

    @classmethod
    def fit(cls, panel: HourlyPanel, train_hours: np.ndarray) -> StructuredNormalizer:
        selected = np.asarray(train_hours, dtype=bool)
        if selected.shape != (len(panel.times),) or not selected.any():
            raise ValueError("invalid fold-local training hours")
        public = panel.public_inputs[selected]
        center = np.nanmedian(public, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.nanmedian(np.abs(public - center), axis=0) * 1.4826
        fallback = np.nanstd(public, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)

        residual = panel.target - panel.baseline
        residual_center = np.zeros(3, dtype=np.float64)
        residual_scale = np.ones(3, dtype=np.float64)
        for layer in range(3):
            keep = selected & panel.target_mask[:, layer] & np.isfinite(residual[:, layer])
            if not keep.any():
                raise ValueError(f"target layer {layer + 2} lacks fold-local observations")
            values = residual[keep, layer]
            residual_center[layer] = np.median(values)
            robust = np.median(np.abs(values - residual_center[layer])) * 1.4826
            fallback = np.std(values)
            residual_scale[layer] = robust if np.isfinite(robust) and robust > 1e-4 else fallback
            if not np.isfinite(residual_scale[layer]) or residual_scale[layer] <= 1e-4:
                residual_scale[layer] = 1.0
        return cls(center, scale, residual_center, residual_scale)

    def public(self, values: np.ndarray) -> np.ndarray:
        output = (np.asarray(values, dtype=np.float64) - self.public_center) / self.public_scale
        return np.clip(np.where(np.isfinite(output), output, 0.0), -12, 12).astype(np.float32)

    def residual(self, panel: HourlyPanel) -> np.ndarray:
        values = (panel.target - panel.baseline - self.residual_center) / self.residual_scale
        return np.where(panel.target_mask & np.isfinite(values), values, 0.0).astype(np.float32)

    def inverse(self, panel: HourlyPanel, prediction: np.ndarray) -> np.ndarray:
        residual = np.asarray(prediction, dtype=np.float64) * self.residual_scale
        residual += self.residual_center
        return np.clip(panel.baseline + residual, -5.0, 45.0)


def _hourly_reduce(values: np.ndarray, group: np.ndarray, rows: int) -> np.ndarray:
    frame = pd.DataFrame(values)
    frame["_group"] = group
    reduced = frame.groupby("_group", sort=True).median(numeric_only=True)
    return reduced.reindex(range(rows)).to_numpy(dtype=np.float64)


def build_hourly_panel(panel: P2Panel) -> HourlyPanel:
    """Reduce a 10-minute panel to one robust row per UTC hour."""

    floors = panel.times.floor("h")
    times = pd.date_range(floors.min(), floors.max(), freq="h", tz="UTC")
    positions = times.get_indexer(floors)
    if (positions < 0).any():
        raise RuntimeError("hourly reindexing failed")
    public = _hourly_reduce(panel.inputs, positions, len(times))
    baseline = _hourly_reduce(panel.baseline, positions, len(times))
    target = _hourly_reduce(panel.target, positions, len(times))
    target_mask = np.isfinite(target) & np.isfinite(baseline)
    return HourlyPanel(times, public, panel.input_names, baseline, target, target_mask)


def time_mask(panel: HourlyPanel, start: str, stop: str) -> np.ndarray:
    left = pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC")
    right = pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC")
    return np.asarray((panel.times >= left) & (panel.times < right), dtype=bool)


class StructuredMaskBiTCN(nn.Module):
    """Bidirectional multi-scale TCN for three simultaneous target-layer gaps."""

    def __init__(self, channels: int, hidden: int = 96, blocks: int = 10, dropout: float = 0.08):
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.blocks = nn.ModuleList(
            ModernTCNBlock(hidden, 2 ** (index % 10), kernel=5, dropout=dropout)
            for index in range(blocks)
        )
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        state = self.input(inputs)
        for block in self.blocks:
            state = block(state)
        return self.output(self.norm(state))


@dataclass
class StructuredTrainingResult:
    config: StructuredMaskConfig
    normalizer: StructuredNormalizer
    state_dict: dict[str, Tensor]
    epochs: int
    best_epoch: int
    best_rmse: float | None
    history: list[dict[str, float | int]]
    hourly_prediction: np.ndarray | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def structured_mask_candidates(
    available: np.ndarray,
    train_hours: np.ndarray,
    length: int,
    *,
    min_coverage: float,
) -> np.ndarray:
    """Return starts whose simultaneous target blackout has enough supervision."""

    allowed = np.asarray(available, dtype=bool) & np.asarray(train_hours, dtype=bool)[:, None]
    score = allowed.sum(axis=1).astype(np.int64)
    sums = np.convolve(score, np.ones(length, dtype=np.int64), mode="valid")
    threshold = int(np.ceil(length * allowed.shape[1] * min_coverage))
    return np.flatnonzero(sums >= threshold)


def sample_training_windows(
    panel: HourlyPanel,
    train_hours: np.ndarray,
    config: StructuredMaskConfig,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], ...]:
    """Sample (window_start, window_stop, mask_start, mask_stop) without labels outside train."""

    per_length = max(1, config.samples_per_epoch // len(config.mask_hours))
    result: list[tuple[int, int, int, int]] = []
    for length in config.mask_hours:
        starts = structured_mask_candidates(
            panel.target_mask, train_hours, length, min_coverage=config.min_mask_coverage
        )
        if not len(starts):
            continue
        chosen = rng.choice(starts, size=per_length, replace=len(starts) < per_length)
        for mask_start in chosen.tolist():
            center = mask_start + length // 2
            window_start = max(
                0, min(center - config.window_hours // 2, len(panel.times) - config.window_hours)
            )
            window_stop = min(len(panel.times), window_start + config.window_hours)
            result.append((window_start, window_stop, mask_start, mask_start + length))
    if not result:
        raise RuntimeError("no fold-local structured blackout windows are available")
    rng.shuffle(result)
    return tuple(result)


def materialize_window(
    public: np.ndarray,
    residual: np.ndarray,
    observed: np.ndarray,
    target_mask: np.ndarray,
    window: tuple[int, int, int, int],
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one padded example; observed target values inside the blackout are removed."""

    start, stop, mask_start, mask_stop = window
    width = stop - start
    local_observed = observed[start:stop].copy()
    left = max(mask_start, start) - start
    right = min(mask_stop, stop) - start
    local_observed[left:right] = False
    target_input = np.where(local_observed, residual[start:stop], 0.0)
    features = np.concatenate(
        (public[start:stop], target_input, local_observed.astype(np.float32)), axis=1
    )
    supervision = np.zeros((width, 3), dtype=np.float32)
    supervision[left:right] = target_mask[start + left : start + right]
    x = np.zeros((length, features.shape[1]), dtype=np.float32)
    y = np.zeros((length, 3), dtype=np.float32)
    m = np.zeros((length, 3), dtype=np.float32)
    x[:width], y[:width], m[:width] = features, residual[start:stop], supervision
    if width and width < length:
        x[width:] = features[-1]
    return x, y, m


def inference_window(
    panel: HourlyPanel,
    public: np.ndarray,
    residual: np.ndarray,
    observed: np.ndarray,
    block: np.ndarray,
    config: StructuredMaskConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(block)
    if not len(indices):
        raise ValueError("inference block is empty")
    start = max(0, indices[0] - config.context_hours)
    stop = min(len(panel.times), indices[-1] + 1 + config.context_hours)
    if stop - start > config.window_hours:
        raise ValueError("inference block plus contexts exceeds the fixed window")
    local_observed = observed[start:stop].copy()
    local_observed[block[start:stop]] = False
    target_input = np.where(local_observed, residual[start:stop], 0.0)
    x = np.concatenate(
        (public[start:stop], target_input, local_observed.astype(np.float32)), axis=1
    ).astype(np.float32)
    return x, np.flatnonzero(block[start:stop]), np.arange(start, stop)


def _predict_block(
    model: nn.Module,
    panel: HourlyPanel,
    normalizer: StructuredNormalizer,
    public: np.ndarray,
    residual: np.ndarray,
    observed: np.ndarray,
    block: np.ndarray,
    config: StructuredMaskConfig,
) -> np.ndarray:
    inputs, _, global_rows = inference_window(panel, public, residual, observed, block, config)
    padded = np.zeros((config.window_hours, inputs.shape[1]), dtype=np.float32)
    padded[: len(inputs)] = inputs
    if len(inputs) < config.window_hours:
        padded[len(inputs) :] = inputs[-1]
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        normalized = model(torch.from_numpy(padded[None]).to(device))[0, : len(inputs)]
    normalized = normalized.float().cpu().numpy()
    physical = (
        panel.baseline[global_rows]
        + normalized * normalizer.residual_scale
        + normalizer.residual_center
    )
    result = np.full((len(panel.times), 3), np.nan, dtype=np.float64)
    result[global_rows] = np.clip(physical, -5.0, 45.0)
    return result


def train_structured_model(
    panel: HourlyPanel,
    *,
    train_hours: np.ndarray,
    config: StructuredMaskConfig,
    evaluation_block: np.ndarray | None = None,
    select_best: bool = False,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> StructuredTrainingResult:
    """Train on synthetic simultaneous gaps; optionally select only on one development block."""

    if not torch.cuda.is_available():
        raise RuntimeError("structured imputer requires the validated CUDA environment")
    train_hours = np.asarray(train_hours, dtype=bool)
    if train_hours.shape != (len(panel.times),):
        raise ValueError("train_hours shape mismatch")
    if evaluation_block is not None:
        evaluation_block = np.asarray(evaluation_block, dtype=bool)
        if evaluation_block.shape != train_hours.shape or np.any(evaluation_block & train_hours):
            raise ValueError("evaluation block must be disjoint from training hours")
    if select_best and evaluation_block is None:
        raise ValueError("checkpoint selection requires a development block")

    set_seed(config.seed)
    normalizer = StructuredNormalizer.fit(panel, train_hours)
    public = normalizer.public(panel.public_inputs)
    residual = normalizer.residual(panel)
    observed = panel.target_mask & train_hours[:, None]
    channels = public.shape[1] + 6
    device = torch.device("cuda")
    model = StructuredMaskBiTCN(
        channels, hidden=config.hidden, blocks=config.blocks, dropout=config.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=config.learning_rate * 0.05
    )
    physical_weights = torch.tensor(
        normalizer.residual_scale**2, device=device, dtype=torch.float32
    )
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float | int]] = []
    best_state: dict[str, Tensor] | None = None
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, config.max_epochs + 1):
        windows = sample_training_windows(panel, train_hours, config, rng)
        model.train()
        losses: list[float] = []
        for offset in range(0, len(windows), config.batch_size):
            current = windows[offset : offset + config.batch_size]
            examples = [
                materialize_window(
                    public,
                    residual,
                    observed,
                    panel.target_mask & train_hours[:, None],
                    window,
                    config.window_hours,
                )
                for window in current
            ]
            x = torch.from_numpy(np.stack([item[0] for item in examples])).to(device)
            y = torch.from_numpy(np.stack([item[1] for item in examples])).to(device)
            mask = torch.from_numpy(np.stack([item[2] for item in examples])).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(x)
                squared = (prediction - y).square() * physical_weights.view(1, 1, 3)
                loss = (squared * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()

        should_score = (
            select_best
            and evaluation_block is not None
            and (
                epoch == 1 or epoch % config.evaluation_interval == 0 or epoch == config.max_epochs
            )
        )
        score: float | None = None
        if should_score:
            prediction = _predict_block(
                model,
                panel,
                normalizer,
                public,
                residual,
                observed,
                evaluation_block,
                config,
            )
            valid = evaluation_block[:, None] & panel.target_mask & np.isfinite(prediction)
            score = float(np.sqrt(np.mean((prediction[valid] - panel.target[valid]) ** 2)))
            if score < best_rmse - 1e-6:
                best_rmse = score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += config.evaluation_interval
        history.append(
            {
                "epoch": epoch,
                "train_mse_c": float(np.mean(losses)),
                **({"validation_rmse": score} if score is not None else {}),
            }
        )
        if progress is not None:
            progress(
                {
                    "epoch": epoch,
                    "max_epochs": config.max_epochs,
                    "train_mse_c": history[-1]["train_mse_c"],
                    "validation_rmse": score,
                }
            )
        if select_best and stale >= config.patience:
            break

    if select_best:
        if best_state is None:
            raise RuntimeError("development training produced no finite checkpoint")
        model.load_state_dict(best_state)
        final_epoch = best_epoch
        final_rmse: float | None = best_rmse
    else:
        best_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        final_epoch = config.max_epochs
        final_rmse = None

    hourly_prediction = None
    if evaluation_block is not None:
        hourly_prediction = _predict_block(
            model,
            panel,
            normalizer,
            public,
            residual,
            observed,
            evaluation_block,
            config,
        )
    return StructuredTrainingResult(
        config=config,
        normalizer=normalizer,
        state_dict=deepcopy(best_state),
        epochs=final_epoch,
        best_epoch=best_epoch,
        best_rmse=final_rmse,
        history=history,
        hourly_prediction=hourly_prediction,
    )


def interpolate_hourly_prediction(
    hourly_times: pd.DatetimeIndex,
    hourly_prediction: np.ndarray,
    target_times: Iterable[object],
) -> np.ndarray:
    """Linearly interpolate three hourly target predictions to arbitrary timestamps."""

    source = hourly_times.as_unit("ns").asi8.astype(np.float64)
    target = (
        pd.DatetimeIndex(pd.to_datetime(list(target_times), utc=True))
        .as_unit("ns")
        .asi8.astype(np.float64)
    )
    return np.column_stack(
        [np.interp(target, source, hourly_prediction[:, layer]) for layer in range(3)]
    )
