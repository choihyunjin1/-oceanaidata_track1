"""Exact degradation-mask Transformer utilities for the sealed P1 pilot.

The implementation borrows the data-degradation training idea from AnomalyBERT,
but is deliberately small and task-specific.  It predicts the exact corrupted rows
without point adjustment, score smoothing, truth filling, or anchor deletion.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

CADENCE = pd.Timedelta(minutes=10)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class Window:
    """One segment-local model window with explicit valid rows."""

    rows: np.ndarray
    valid_rows: int


@dataclass(frozen=True)
class CellScale:
    cells: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    global_center: np.ndarray
    global_scale: np.ndarray


def _finite_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    center = np.nanmedian(matrix, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    filled = np.where(np.isfinite(matrix), matrix, center)
    mad = 1.4826 * np.nanmedian(np.abs(filled - center), axis=0)
    fallback = np.nanstd(filled, axis=0)
    scale = np.where(np.isfinite(mad) & (mad >= 1e-6), mad, fallback)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-6), scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def fit_cell_scale(
    raw: np.ndarray,
    cells: np.ndarray,
    fit_normal: np.ndarray,
) -> CellScale:
    """Fit robust raw-variable scales on fit-prefix normal rows only."""

    values = np.asarray(raw, dtype=np.float32)
    cell_values = np.asarray(cells, dtype=str)
    eligible = np.asarray(fit_normal, dtype=bool)
    global_center, global_scale = _finite_center_scale(values[eligible])
    unique = tuple(sorted(set(cell_values.tolist())))
    centers: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for cell in unique:
        mask = eligible & (cell_values == cell)
        if int(mask.sum()) < 32:
            centers.append(global_center)
            scales.append(global_scale)
        else:
            center, scale = _finite_center_scale(values[mask])
            centers.append(center)
            scales.append(scale)
    return CellScale(
        cells=unique,
        center=np.asarray(centers, dtype=np.float32),
        scale=np.asarray(scales, dtype=np.float32),
        global_center=global_center,
        global_scale=global_scale,
    )


def transform_cell_scale(raw: np.ndarray, cells: np.ndarray, state: CellScale) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    cell_values = np.asarray(cells, dtype=str)
    mapping = {cell: index for index, cell in enumerate(state.cells)}
    output = np.empty_like(values, dtype=np.float32)
    for cell in sorted(set(cell_values.tolist())):
        mask = cell_values == cell
        index = mapping.get(cell)
        center = state.global_center if index is None else state.center[index]
        scale = state.global_scale if index is None else state.scale[index]
        output[mask] = (values[mask] - center) / scale
    output[~np.isfinite(output)] = 0.0
    return np.clip(output, -12.0, 12.0).astype(np.float32, copy=False)


def build_features(
    scaled_raw: np.ndarray,
    raw_missing: np.ndarray,
    time_utc: pd.Series | pd.DatetimeIndex,
) -> np.ndarray:
    """Build the sealed eleven row features without centered/future transforms."""

    base = np.asarray(scaled_raw, dtype=np.float32)
    missing = np.asarray(raw_missing, dtype=np.float32)
    times = pd.DatetimeIndex(time_utc).tz_convert("Asia/Seoul")
    radians_day = 2.0 * np.pi * (times.dayofyear.to_numpy() - 1) / 365.2425
    fractional_hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    radians_hour = 2.0 * np.pi * fractional_hour / 24.0
    temp_diff = np.zeros(len(base), dtype=np.float32)
    if len(base) > 1:
        temp_diff[1:] = np.diff(base[:, 0])
    return np.column_stack(
        [
            base[:, 0],
            base[:, 1],
            base[:, 2],
            missing[:, 1],
            missing[:, 2],
            temp_diff,
            np.abs(temp_diff),
            np.sin(radians_day),
            np.cos(radians_day),
            np.sin(radians_hour),
            np.cos(radians_hour),
        ]
    ).astype(np.float32)


def refresh_temperature_differences(features: np.ndarray) -> np.ndarray:
    output = np.asarray(features, dtype=np.float32).copy()
    difference = np.zeros(len(output), dtype=np.float32)
    if len(output) > 1:
        difference[1:] = np.diff(output[:, 0])
    output[:, 5] = difference
    output[:, 6] = np.abs(difference)
    return output


def segment_ids(keys: pd.DataFrame) -> np.ndarray:
    times = pd.to_datetime(keys["time"], utc=True, format="mixed")
    boundary = (
        keys["station"].astype(str).ne(keys["station"].astype(str).shift())
        | keys["year"].ne(keys["year"].shift())
        | keys["layer"].ne(keys["layer"].shift())
        | times.diff().ne(CADENCE)
    )
    if len(boundary):
        boundary.iloc[0] = True
    return boundary.cumsum().to_numpy(dtype=np.int64)


def contiguous_full_windows(
    segments: np.ndarray,
    eligible: np.ndarray,
    window_rows: int,
    stride_rows: int,
) -> np.ndarray:
    """Return full windows that cross neither gaps nor ineligible rows."""

    groups = np.asarray(segments)
    keep = np.asarray(eligible, dtype=bool)
    output: list[tuple[int, int]] = []
    start = 0
    while start < len(keep):
        if not keep[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(keep) and keep[stop] and groups[stop] == groups[start]:
            stop += 1
        if stop - start >= window_rows:
            last = stop - window_rows
            positions = list(range(start, last + 1, stride_rows))
            if positions[-1] != last:
                positions.append(last)
            output.extend((position, position + window_rows) for position in positions)
        start = stop
    return np.asarray(output, dtype=np.int64).reshape(-1, 2)


def coverage_windows(
    segments: np.ndarray,
    eligible: np.ndarray,
    window_rows: int,
    stride_rows: int,
) -> list[Window]:
    """Create segment-local windows with right-edge padding for complete coverage."""

    groups = np.asarray(segments)
    keep = np.asarray(eligible, dtype=bool)
    output: list[Window] = []
    start = 0
    while start < len(keep):
        if not keep[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(keep) and keep[stop] and groups[stop] == groups[start]:
            stop += 1
        run = np.arange(start, stop, dtype=np.int64)
        if len(run) <= window_rows:
            padded = np.pad(run, (0, window_rows - len(run)), mode="edge")
            output.append(Window(padded, len(run)))
        else:
            last = len(run) - window_rows
            positions = list(range(0, last + 1, stride_rows))
            if positions[-1] != last:
                positions.append(last)
            output.extend(Window(run[position : position + window_rows], window_rows) for position in positions)
        start = stop
    return output


def inject_exact_degradation(
    clean_features: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    duration_rows: Mapping[str, Sequence[int]],
    amplitudes: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Mutate temperature only and return the exact corrupted-row mask."""

    features = np.asarray(clean_features, dtype=np.float32).copy()
    if kind not in duration_rows or kind not in amplitudes:
        raise ValueError(f"unregistered synthetic family: {kind}")
    minimum, maximum = (int(value) for value in duration_rows[kind])
    length = int(rng.integers(minimum, min(maximum, len(features) - 32) + 1))
    start = int(rng.integers(16, len(features) - length - 15))
    stop = start + length
    mask = np.zeros(len(features), dtype=np.float32)
    mask[start:stop] = 1.0
    low, high = (float(value) for value in amplitudes[kind])
    amplitude = low if low == high else float(rng.uniform(low, high))
    sign = float(rng.choice(np.asarray([-1.0, 1.0])))
    if kind == "spike":
        features[start, 0] += sign * amplitude
    elif kind == "noise":
        features[start:stop, 0] += rng.normal(0.0, amplitude, length).astype(np.float32)
    elif kind == "flatline":
        features[start:stop, 0] = features[start - 1, 0]
    elif kind == "offset":
        features[start:stop, 0] += sign * amplitude
    elif kind == "drift":
        features[start:stop, 0] += np.linspace(
            0.0, sign * amplitude, length, dtype=np.float32
        )
    else:
        raise ValueError(kind)
    return refresh_temperature_differences(features), mask, (start, stop)


class RelativePositionBias(nn.Module):
    def __init__(self, heads: int, maximum_tokens: int) -> None:
        super().__init__()
        self.maximum_tokens = maximum_tokens
        self.table = nn.Parameter(torch.zeros(2 * maximum_tokens - 1, heads))
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, tokens: int) -> torch.Tensor:
        if tokens > self.maximum_tokens:
            raise ValueError("relative-position table is too short")
        position = torch.arange(tokens, device=self.table.device)
        relative = position[:, None] - position[None, :] + self.maximum_tokens - 1
        return self.table[relative].permute(2, 0, 1)


class RelativeSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, maximum_tokens: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("model width must be divisible by attention heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.bias = RelativePositionBias(heads, maximum_tokens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = values.shape
        qkv = self.qkv(values).reshape(batch, tokens, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_width)
        logits = logits + self.bias(tokens).unsqueeze(0)
        attention = self.dropout(torch.softmax(logits, dim=-1))
        context = torch.matmul(attention, value).transpose(1, 2).reshape(batch, tokens, width)
        return self.output(context)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        feedforward_width: int,
        maximum_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = RelativeSelfAttention(width, heads, maximum_tokens, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.dropout(self.attention(self.norm1(values)))
        return values + self.dropout(self.feedforward(self.norm2(values)))


class ExactDegradationMaskTransformer(nn.Module):
    """Patch-token Transformer with an exact per-row binary decoder."""

    def __init__(
        self,
        input_width: int,
        window_rows: int,
        patch_rows: int,
        d_model: int,
        heads: int,
        layers: int,
        feedforward_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if window_rows % patch_rows:
            raise ValueError("window_rows must be divisible by patch_rows")
        self.window_rows = window_rows
        self.patch_rows = patch_rows
        maximum_tokens = window_rows // patch_rows
        self.patch_projection = nn.Conv1d(
            input_width, d_model, kernel_size=patch_rows, stride=patch_rows
        )
        self.blocks = nn.Sequential(
            *(
                TransformerBlock(
                    d_model,
                    heads,
                    feedforward_width,
                    maximum_tokens,
                    dropout,
                )
                for _ in range(layers)
            )
        )
        self.row_projection = nn.Linear(input_width, d_model)
        self.decoder = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, 1, kernel_size=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.window_rows:
            raise ValueError("input must have shape [batch, sealed_window_rows, features]")
        tokens = self.patch_projection(values.transpose(1, 2)).transpose(1, 2)
        tokens = self.blocks(tokens)
        context = tokens.repeat_interleave(self.patch_rows, dim=1)
        rows = self.row_projection(values)
        return self.decoder((context + rows).transpose(1, 2)).squeeze(1)


def mask_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    weight = torch.as_tensor(positive_weight, device=logits.device)
    binary = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * targets).sum(dim=1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (probability.sum(dim=1) + targets.sum(dim=1) + 1.0))
    return binary + dice_weight * dice.mean()


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(truth, dtype=np.int8)
    estimate = np.asarray(prediction, dtype=np.int8)
    tp = int(((target == 1) & (estimate == 1)).sum())
    fp = int(((target == 0) & (estimate == 1)).sum())
    fn = int(((target == 1) & (estimate == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def best_overlap_boundary_error(truth: np.ndarray, prediction: np.ndarray) -> float:
    target = np.flatnonzero(np.asarray(truth, dtype=bool))
    if not len(target):
        return 0.0
    active = np.asarray(prediction, dtype=bool)
    components: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(active):
        if not active[cursor]:
            cursor += 1
            continue
        stop = cursor + 1
        while stop < len(active) and active[stop]:
            stop += 1
        components.append((cursor, stop))
        cursor = stop
    if not components:
        return float(len(active))
    truth_start, truth_stop = int(target[0]), int(target[-1] + 1)
    best = min(
        components,
        key=lambda item: (
            -max(0, min(item[1], truth_stop) - max(item[0], truth_start)),
            abs(item[0] - truth_start) + abs(item[1] - truth_stop),
        ),
    )
    return 0.5 * (abs(best[0] - truth_start) + abs(best[1] - truth_stop))


def synthetic_family_metrics(
    truth_by_family: Mapping[str, Sequence[np.ndarray]],
    prediction_by_family: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for family in sorted(truth_by_family):
        truth = list(truth_by_family[family])
        prediction = list(prediction_by_family[family])
        metrics = binary_metrics(np.concatenate(truth), np.concatenate(prediction))
        errors = [
            best_overlap_boundary_error(target, estimate)
            for target, estimate in zip(truth, prediction, strict=True)
        ]
        output[family] = {
            **metrics,
            "examples": len(truth),
            "boundary_mae_rows": float(np.mean(errors)),
        }
    return output


def decode_components(
    scores: np.ndarray,
    segments: np.ndarray,
    eligible: np.ndarray,
    *,
    threshold: float,
    minimum_rows: int,
    maximum_rows: int,
) -> np.ndarray:
    """Raw threshold decoder; no smoothing, bridging, or event adjustment."""

    values = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(segments)
    keep = np.asarray(eligible, dtype=bool)
    active = np.isfinite(values) & (values >= threshold) & keep
    output = np.zeros(len(values), dtype=np.int8)
    cursor = 0
    while cursor < len(active):
        if not active[cursor]:
            cursor += 1
            continue
        stop = cursor + 1
        while (
            stop < len(active)
            and active[stop]
            and groups[stop] == groups[cursor]
            and keep[stop]
        ):
            stop += 1
        if minimum_rows <= stop - cursor <= maximum_rows:
            output[cursor:stop] = 1
        cursor = stop
    return output


def anchor_union(anchor: np.ndarray, additions: np.ndarray) -> np.ndarray:
    baseline = np.asarray(anchor, dtype=np.int8)
    extra = np.asarray(additions, dtype=np.int8)
    if baseline.shape != extra.shape or not np.isin(baseline, [0, 1]).all():
        raise ValueError("anchor and additions must be aligned binary arrays")
    return np.maximum(baseline, extra).astype(np.int8)


def union_metrics(
    truth: np.ndarray,
    anchor: np.ndarray,
    additions: np.ndarray,
) -> dict[str, float | int]:
    baseline = np.asarray(anchor, dtype=np.int8)
    candidate = anchor_union(baseline, additions)
    base = binary_metrics(truth, baseline)
    updated = binary_metrics(truth, candidate)
    changed = (baseline == 0) & (candidate == 1)
    target = np.asarray(truth, dtype=np.int8)
    added_tp = int((changed & (target == 1)).sum())
    added_fp = int((changed & (target == 0)).sum())
    return {
        "anchor_f1": float(base["f1"]),
        "candidate_f1": float(updated["f1"]),
        "delta_f1": float(updated["f1"] - base["f1"]),
        "added_rows": int(changed.sum()),
        "added_tp": added_tp,
        "added_fp": added_fp,
        "added_precision": added_tp / (added_tp + added_fp) if added_tp + added_fp else 1.0,
        "anchor_f1_over_2": float(base["f1"]) / 2.0,
        "anchor_positive_removed_rows": int(((baseline == 1) & (candidate == 0)).sum()),
    }


def day_block_bootstrap_probability(
    truth: np.ndarray,
    candidate: np.ndarray,
    anchor: np.ndarray,
    day_ordinal: np.ndarray,
    *,
    block_days: int,
    replicates: int,
    seed: int,
) -> float:
    target = np.asarray(truth, dtype=np.int8)
    challenger = np.asarray(candidate, dtype=np.int8)
    baseline = np.asarray(anchor, dtype=np.int8)
    days = np.asarray(day_ordinal, dtype=np.int64)
    unique = np.unique(days)
    blocks = [
        np.flatnonzero(np.isin(days, unique[start : start + block_days]))
        for start in range(0, len(unique), block_days)
    ]
    if not blocks:
        raise ValueError("bootstrap requires at least one day block")
    rng = np.random.default_rng(seed)
    improved = 0
    for _ in range(replicates):
        rows = np.concatenate([blocks[index] for index in rng.integers(0, len(blocks), len(blocks))])
        delta = binary_metrics(target[rows], challenger[rows])["f1"] - binary_metrics(
            target[rows], baseline[rows]
        )["f1"]
        improved += bool(delta > 0)
    return improved / replicates


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def smoke_step(device: str, seed: int = 20260828) -> dict[str, Any]:
    seed_everything(seed)
    target = torch.device(device)
    model = ExactDegradationMaskTransformer(
        input_width=11,
        window_rows=1024,
        patch_rows=8,
        d_model=32,
        heads=4,
        layers=2,
        feedforward_width=64,
        dropout=0.0,
    ).to(target)
    features = torch.randn(2, 1024, 11, device=target)
    truth = torch.zeros(2, 1024, device=target)
    truth[:, 128:256] = 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(features)
    loss = mask_loss(logits, truth, positive_weight=8.0, dice_weight=0.5)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "device": str(target),
        "loss": float(loss.detach().cpu()),
        "finite": bool(torch.isfinite(logits).all()),
        "output_shape": list(logits.shape),
        "parameters": model_parameter_count(model),
    }


__all__ = [
    "CellScale",
    "ExactDegradationMaskTransformer",
    "RelativePositionBias",
    "Window",
    "anchor_union",
    "best_overlap_boundary_error",
    "binary_metrics",
    "build_features",
    "contiguous_full_windows",
    "coverage_windows",
    "day_block_bootstrap_probability",
    "decode_components",
    "fit_cell_scale",
    "inject_exact_degradation",
    "mask_loss",
    "model_parameter_count",
    "refresh_temperature_differences",
    "seed_everything",
    "segment_ids",
    "sha256_file",
    "smoke_step",
    "synthetic_family_metrics",
    "transform_cell_scale",
    "union_metrics",
]
