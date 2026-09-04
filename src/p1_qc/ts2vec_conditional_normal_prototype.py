"""Frozen primitives for the P1 TS2Vec-style normal-prototype experiment.

The encoder is deliberately described as TS2Vec-style rather than a faithful
reimplementation.  Its training objective is label-free and hierarchical;
labels are accepted only by the downstream audit helpers.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PrototypeState:
    harmonic: np.ndarray
    cells: tuple[str, ...]
    cell_residual: np.ndarray
    cell_weight: np.ndarray
    scale: np.ndarray
    banks: tuple[np.ndarray, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SameLengthConv1d(nn.Conv1d):
    def __init__(self, channels_in: int, channels_out: int, dilation: int) -> None:
        super().__init__(
            channels_in,
            channels_out,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SameLengthConv1d(width, width, dilation)
        self.conv2 = SameLengthConv1d(width, width, dilation)
        self.norm1 = nn.GroupNorm(1, width)
        self.norm2 = nn.GroupNorm(1, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(F.gelu(self.norm1(self.conv1(inputs))))
        hidden = self.dropout(F.gelu(self.norm2(self.conv2(hidden))))
        return inputs + hidden


class HierarchicalContrastiveEncoder(nn.Module):
    """Small dilated encoder returning one normalized embedding per row."""

    def __init__(
        self,
        input_width: int,
        hidden_width: int = 64,
        embedding_width: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_width, hidden_width, kernel_size=1)
        self.blocks = nn.Sequential(
            *(ResidualBlock(hidden_width, dilation, dropout) for dilation in dilations)
        )
        self.output_projection = nn.Conv1d(hidden_width, embedding_width, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("encoder input must have shape [batch, rows, features]")
        hidden = self.input_projection(inputs.transpose(1, 2))
        hidden = self.blocks(hidden)
        embedding = self.output_projection(hidden).transpose(1, 2)
        return F.normalize(embedding, dim=-1)


def masked_views(
    windows: torch.Tensor,
    mask_probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < mask_probability < 1.0:
        raise ValueError("mask_probability must be in (0, 1)")
    masks = []
    for _ in range(2):
        mask = torch.rand(
            windows.shape[:2], generator=generator, device=windows.device
        ).ge(mask_probability)
        masks.append(windows * mask.unsqueeze(-1))
    return masks[0], masks[1]


def contrastive_matrix_bound(batch: int, rows: int, timestamp_cap: int) -> dict[str, int]:
    """Expose the sealed maximum matrix dimensions for memory regression tests."""
    capped = min(rows, timestamp_cap)
    return {
        "instance_matrices": capped,
        "instance_matrix_side": batch,
        "temporal_matrices": batch,
        "temporal_matrix_side": capped,
        "maximum_single_matrix_elements": max(batch * batch, capped * capped),
    }


def _symmetric_ce(logits: torch.Tensor) -> torch.Tensor:
    targets = torch.arange(logits.shape[-1], device=logits.device)
    targets = targets.expand(logits.shape[0], -1).reshape(-1)
    flat = logits.reshape(-1, logits.shape[-1])
    transposed = logits.transpose(-1, -2).reshape(-1, logits.shape[-1])
    return 0.5 * (F.cross_entropy(flat, targets) + F.cross_entropy(transposed, targets))


def _instance_temporal_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float,
    timestamp_cap: int,
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("contrastive views must share [batch, rows, width]")
    batch, rows, _ = first.shape
    if min(batch, rows) < 2:
        return (first - second).square().mean()
    if rows > timestamp_cap:
        positions = torch.linspace(
            0, rows - 1, timestamp_cap, device=first.device
        ).round().long()
        first = first[:, positions]
        second = second[:, positions]
    # Instance discrimination is B x B independently at each sampled time.
    instance_logits = torch.einsum("btd,ctd->tbc", first, second) / temperature
    instance_loss = _symmetric_ce(instance_logits)
    # Temporal discrimination is T x T independently for each series.
    temporal_logits = torch.einsum("btd,bsd->bts", first, second) / temperature
    temporal_loss = _symmetric_ce(temporal_logits)
    return 0.5 * (instance_loss + temporal_loss)


def hierarchical_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float = 0.2,
    timestamp_cap: int = 128,
) -> torch.Tensor:
    """InfoNCE at successively pooled time resolutions."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    losses: list[torch.Tensor] = []
    left, right = first, second
    while left.shape[1] >= 2:
        losses.append(_instance_temporal_loss(left, right, temperature, timestamp_cap))
        left = F.max_pool1d(left.transpose(1, 2), 2).transpose(1, 2)
        right = F.max_pool1d(right.transpose(1, 2), 2).transpose(1, 2)
    return torch.stack(losses).mean()


def robust_fit_transform(
    fit: np.ndarray, values: np.ndarray, floor: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_values = np.asarray(fit, dtype=np.float64)
    target = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(fit_values, axis=0)
    filled_fit = np.where(np.isfinite(fit_values), fit_values, median)
    mad = np.nanmedian(np.abs(filled_fit - median), axis=0) * 1.4826
    mad = np.where(np.isfinite(mad) & (mad >= floor), mad, 1.0)
    filled = np.where(np.isfinite(target), target, median)
    return ((filled - median) / mad).astype(np.float32), median, mad


def contiguous_windows(
    segments: np.ndarray,
    eligible: np.ndarray,
    window_rows: int,
    stride_rows: int,
) -> np.ndarray:
    """Return [start, stop] windows that never cross a segment or eligibility edge."""
    segment_ids = np.asarray(segments)
    keep = np.asarray(eligible, dtype=bool)
    if len(segment_ids) != len(keep):
        raise ValueError("segments and eligible must have equal length")
    output: list[tuple[int, int]] = []
    start = 0
    while start < len(keep):
        if not keep[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(keep) and keep[stop] and segment_ids[stop] == segment_ids[start]:
            stop += 1
        if stop - start >= window_rows:
            last = stop - window_rows
            positions = list(range(start, last + 1, stride_rows))
            if positions[-1] != last:
                positions.append(last)
            output.extend((position, position + window_rows) for position in positions)
        start = stop
    return np.asarray(output, dtype=np.int64).reshape(-1, 2)


def fit_conditional_prototype(
    embeddings: np.ndarray,
    day_sin: np.ndarray,
    day_cos: np.ndarray,
    cells: np.ndarray,
    normal: np.ndarray,
    *,
    shrinkage_rows: int = 512,
    scale_floor: float = 1e-3,
    bank_per_cell: int = 512,
) -> PrototypeState:
    z = np.asarray(embeddings, dtype=np.float64)
    normal_mask = np.asarray(normal, dtype=bool) & np.isfinite(z).all(axis=1)
    if normal_mask.sum() < max(16, z.shape[1]):
        raise ValueError("normal reference has insufficient finite rows")
    design = np.column_stack(
        [np.ones(len(z)), np.asarray(day_sin), np.asarray(day_cos)]
    )
    x = design[normal_mask]
    y = z[normal_mask]
    ridge = np.eye(3) * 1e-3
    ridge[0, 0] = 0.0
    harmonic = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    residual = y - x @ harmonic
    global_scale = np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    global_scale = np.maximum(1.4826 * global_scale, scale_floor)
    normal_cells = np.asarray(cells, dtype=str)[normal_mask]
    unique_cells = tuple(sorted(set(normal_cells.tolist())))
    residuals: list[np.ndarray] = []
    weights: list[float] = []
    scales: list[np.ndarray] = []
    banks: list[np.ndarray] = []
    for cell in unique_cells:
        cell_rows = normal_cells == cell
        cell_residual = residual[cell_rows]
        count = len(cell_residual)
        weight = count / (count + shrinkage_rows)
        center = np.mean(cell_residual, axis=0)
        cell_scale = 1.4826 * np.median(
            np.abs(cell_residual - np.median(cell_residual, axis=0)), axis=0
        )
        cell_scale = weight * np.maximum(cell_scale, scale_floor) + (
            1.0 - weight
        ) * global_scale
        residuals.append(center)
        weights.append(weight)
        scales.append(cell_scale)
        take = np.linspace(0, count - 1, min(count, bank_per_cell), dtype=int)
        banks.append(cell_residual[take].astype(np.float32))
    return PrototypeState(
        harmonic=harmonic.astype(np.float32),
        cells=unique_cells,
        cell_residual=np.asarray(residuals, dtype=np.float32),
        cell_weight=np.asarray(weights, dtype=np.float32),
        scale=np.asarray(scales, dtype=np.float32),
        banks=tuple(banks),
    )


def score_conditional_prototype(
    state: PrototypeState,
    embeddings: np.ndarray,
    day_sin: np.ndarray,
    day_cos: np.ndarray,
    cells: np.ndarray,
    *,
    knn_k: int = 5,
    prototype_weight: float = 0.75,
) -> np.ndarray:
    z = np.asarray(embeddings, dtype=np.float32)
    design = np.column_stack([np.ones(len(z)), day_sin, day_cos]).astype(np.float32)
    seasonal = design @ state.harmonic
    mapping = {cell: index for index, cell in enumerate(state.cells)}
    scores = np.full(len(z), np.nan, dtype=np.float32)
    global_scale = np.median(state.scale, axis=0)
    for row, cell in enumerate(np.asarray(cells, dtype=str)):
        if not np.isfinite(z[row]).all():
            continue
        index = mapping.get(cell)
        if index is None:
            center = seasonal[row]
            scale = global_scale
            bank_score = 0.0
        else:
            center = seasonal[row] + state.cell_weight[index] * state.cell_residual[index]
            scale = state.scale[index]
            residual = z[row] - seasonal[row]
            distances = np.sqrt(np.mean(((state.banks[index] - residual) / scale) ** 2, axis=1))
            bank_score = float(np.mean(np.partition(distances, min(knn_k, len(distances)) - 1)[:knn_k]))
        prototype_score = float(np.sqrt(np.mean(((z[row] - center) / scale) ** 2)))
        scores[row] = prototype_weight * prototype_score + (1.0 - prototype_weight) * bank_score
    return scores


def finite_normal_tail_threshold(
    normal_scores: np.ndarray, alpha: float = 0.001
) -> float:
    values = np.sort(np.asarray(normal_scores, dtype=np.float64))
    values = values[np.isfinite(values)]
    if not len(values) or not 0.0 < alpha < 1.0:
        raise ValueError("finite normal scores and alpha in (0,1) are required")
    rank = math.ceil((1.0 - alpha) * (len(values) + 1)) - 1
    return float(values[min(max(rank, 0), len(values) - 1)])


def decode_components(
    scores: np.ndarray,
    segments: np.ndarray,
    threshold: float,
    *,
    minimum_rows: int = 19,
    bridge_rows: int = 2,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    segment_ids = np.asarray(segments)
    raw = np.isfinite(values) & (values > threshold)
    prediction = np.zeros(len(raw), dtype=np.int8)
    start = 0
    while start < len(raw):
        stop = start + 1
        while stop < len(raw) and segment_ids[stop] == segment_ids[start]:
            stop += 1
        local = raw[start:stop].copy()
        if bridge_rows:
            positive = np.flatnonzero(local)
            for left, right in zip(positive[:-1], positive[1:], strict=False):
                if 1 < right - left <= bridge_rows + 1:
                    local[left : right + 1] = True
        cursor = 0
        while cursor < len(local):
            if not local[cursor]:
                cursor += 1
                continue
            end = cursor + 1
            while end < len(local) and local[end]:
                end += 1
            if end - cursor >= minimum_rows:
                prediction[start + cursor : start + end] = 1
            cursor = end
        start = stop
    return prediction


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def anchor_union(anchor: np.ndarray, additions: np.ndarray) -> np.ndarray:
    base = np.asarray(anchor, dtype=np.int8)
    extra = np.asarray(additions, dtype=np.int8)
    if base.shape != extra.shape or not np.isin(base, [0, 1]).all():
        raise ValueError("anchor and additions must be aligned binary arrays")
    return np.maximum(base, extra).astype(np.int8)


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
    """Probability of positive F1 delta under fixed contiguous day blocks."""
    y = np.asarray(truth, dtype=np.int8)
    challenger = np.asarray(candidate, dtype=np.int8)
    baseline = np.asarray(anchor, dtype=np.int8)
    days = np.asarray(day_ordinal, dtype=np.int64)
    if not (len(y) == len(challenger) == len(baseline) == len(days)):
        raise ValueError("bootstrap arrays must be aligned")
    unique_days = np.unique(days)
    blocks = [
        np.flatnonzero(np.isin(days, unique_days[start : start + block_days]))
        for start in range(0, len(unique_days), block_days)
    ]
    if not blocks:
        raise ValueError("bootstrap needs at least one day block")
    rng = np.random.default_rng(seed)
    improvements = 0
    for _ in range(replicates):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        rows = np.concatenate([blocks[index] for index in selected])
        delta = binary_metrics(y[rows], challenger[rows])["f1"] - binary_metrics(
            y[rows], baseline[rows]
        )["f1"]
        improvements += bool(delta > 0)
    return improvements / replicates


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def smoke_train(device: str, seed: int = 20260828) -> dict[str, Any]:
    seed_everything(seed)
    target = torch.device(device)
    model = HierarchicalContrastiveEncoder(12, hidden_width=16, embedding_width=16).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=target).manual_seed(seed)
    windows = torch.randn(4, 64, 12, device=target)
    losses: list[float] = []
    for _ in range(2):
        first, second = masked_views(windows, 0.2, generator)
        loss = hierarchical_contrastive_loss(model(first), model(second), 0.2, 32)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        embedding = model(windows).detach().cpu().numpy()
    return {
        "device": str(target),
        "steps": 2,
        "losses": losses,
        "finite": bool(np.isfinite(embedding).all()),
        "embedding_variance": float(np.var(embedding)),
        "parameters": model_parameter_count(model),
    }
