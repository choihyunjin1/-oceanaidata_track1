"""Small supervised direct temporal interval-set predictor for P1.

This module is deliberately independent of the sealed tinygrad TE-TAD-lite
experiment.  It predicts an unordered set of half-open intervals and uses a
patch-actionness auxiliary loss only as training supervision; inference is
performed exclusively by the query set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DirectIntervalConfig:
    input_features: int
    window_rows: int = 512
    patch_rows: int = 16
    d_model: int = 48
    heads: int = 4
    encoder_layers: int = 2
    queries: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.window_rows % self.patch_rows:
            raise ValueError("window_rows must be divisible by patch_rows")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")


class DirectIntervalSetPredictor(nn.Module):
    """Patch encoder plus learned-query decoder and three direct heads."""

    def __init__(self, config: DirectIntervalConfig) -> None:
        super().__init__()
        self.config = config
        self.patch = nn.Conv1d(
            config.input_features,
            config.d_model,
            kernel_size=config.patch_rows,
            stride=config.patch_rows,
        )
        tokens = config.window_rows // config.patch_rows
        self.position = nn.Parameter(torch.zeros(1, tokens, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, 1)
        self.query = nn.Parameter(torch.randn(1, config.queries, config.d_model) * 0.02)
        self.objectness = nn.Linear(config.d_model, 1)
        self.endpoints = nn.Linear(config.d_model, 2)
        self.actionness = nn.Linear(config.d_model, 1)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if features.ndim != 3 or features.shape[1:] != (
            self.config.window_rows,
            self.config.input_features,
        ):
            raise ValueError("features must be [batch, window_rows, input_features]")
        memory = self.patch(features.transpose(1, 2)).transpose(1, 2)
        memory = self.encoder(memory + self.position)
        query = self.query.expand(features.shape[0], -1, -1)
        decoded = self.decoder(query, memory)
        logits = self.objectness(decoded).squeeze(-1)
        raw = self.endpoints(decoded).sigmoid()
        left = torch.minimum(raw[..., 0], raw[..., 1])
        right = torch.maximum(raw[..., 0], raw[..., 1])
        intervals = torch.stack((left, right), dim=-1)
        patch_logits = self.actionness(memory).squeeze(-1)
        return logits, intervals, patch_logits


def pairwise_iou_numpy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    left = np.maximum(predicted[:, None, 0], target[None, :, 0])
    right = np.minimum(predicted[:, None, 1], target[None, :, 1])
    intersection = np.maximum(right - left, 0.0)
    pred_length = np.maximum(predicted[:, 1] - predicted[:, 0], 0.0)[:, None]
    target_length = np.maximum(target[:, 1] - target[:, 0], 0.0)[None, :]
    return intersection / np.maximum(pred_length + target_length - intersection, 1e-7)


def interval_set_loss(
    logits: Tensor,
    intervals: Tensor,
    patch_logits: Tensor,
    targets: Sequence[np.ndarray],
    *,
    positive_weight: float = 6.0,
    endpoint_weight: float = 3.0,
    iou_weight: float = 3.0,
    actionness_weight: float = 0.5,
) -> Tensor:
    """Hungarian set loss with an auxiliary patch-occupancy target."""

    batch, queries = logits.shape
    if intervals.shape != (batch, queries, 2) or len(targets) != batch:
        raise ValueError("unaligned interval-set batch")
    class_target = torch.zeros_like(logits)
    patch_target = torch.zeros_like(patch_logits)
    selected_pred: list[Tensor] = []
    selected_truth: list[Tensor] = []
    pred_np = intervals.detach().cpu().numpy()
    prob_np = logits.detach().sigmoid().cpu().numpy()
    for row, value in enumerate(targets):
        truth = np.asarray(value, dtype=np.float32).reshape(-1, 2)
        if len(truth) > queries:
            raise ValueError("target count exceeds query budget")
        for start, end in truth:
            lo = max(0, min(patch_logits.shape[1], int(np.floor(start * patch_logits.shape[1]))))
            hi = max(lo + 1, min(patch_logits.shape[1], int(np.ceil(end * patch_logits.shape[1]))))
            patch_target[row, lo:hi] = 1.0
        if not len(truth):
            continue
        classification = -np.log(np.maximum(prob_np[row], 1e-7))[:, None]
        endpoint = np.abs(pred_np[row, :, None, :] - truth[None, :, :]).sum(axis=2)
        cost = classification + 3.0 * endpoint + 3.0 * (1.0 - pairwise_iou_numpy(pred_np[row], truth))
        pred_idx, truth_idx = linear_sum_assignment(cost)
        class_target[row, pred_idx] = 1.0
        selected_pred.append(intervals[row, pred_idx.tolist()])
        selected_truth.append(torch.as_tensor(truth[truth_idx], device=intervals.device))
    class_loss = F.binary_cross_entropy_with_logits(
        logits,
        class_target,
        pos_weight=torch.as_tensor(positive_weight, device=logits.device),
    )
    patch_loss = F.binary_cross_entropy_with_logits(
        patch_logits,
        patch_target,
        pos_weight=torch.as_tensor(positive_weight, device=logits.device),
    )
    if selected_pred:
        predicted = torch.cat(selected_pred)
        truth_tensor = torch.cat(selected_truth)
        endpoint_loss = F.smooth_l1_loss(predicted, truth_tensor)
        left = torch.maximum(predicted[:, 0], truth_tensor[:, 0])
        right = torch.minimum(predicted[:, 1], truth_tensor[:, 1])
        intersection = (right - left).clamp_min(0.0)
        union = (predicted[:, 1] - predicted[:, 0]).clamp_min(0.0) + (
            truth_tensor[:, 1] - truth_tensor[:, 0]
        ).clamp_min(0.0) - intersection
        iou_loss = 1.0 - (intersection / union.clamp_min(1e-7)).mean()
    else:
        endpoint_loss = intervals.sum() * 0.0
        iou_loss = intervals.sum() * 0.0
    return class_loss + endpoint_weight * endpoint_loss + iou_weight * iou_loss + actionness_weight * patch_loss


@torch.no_grad()
def predict(model: DirectIntervalSetPredictor, features: np.ndarray, *, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    interval_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for offset in range(0, len(features), batch_size):
        logits, intervals, _ = model(torch.as_tensor(features[offset : offset + batch_size], device=device))
        interval_parts.append(intervals.cpu().numpy().astype(np.float32))
        score_parts.append(logits.sigmoid().cpu().numpy().astype(np.float32))
    return np.concatenate(interval_parts), np.concatenate(score_parts)


def interval_metrics(intervals: np.ndarray, scores: np.ndarray, targets: Sequence[np.ndarray], *, threshold: float = 0.5, iou_cutoff: float = 0.7) -> dict[str, float | int]:
    matched: list[float] = []
    total = 0
    negative_fp = 0
    for predicted, confidence, target in zip(intervals, scores, targets, strict=True):
        selected = predicted[confidence >= threshold]
        truth = np.asarray(target, dtype=np.float32).reshape(-1, 2)
        if not len(truth):
            negative_fp += int(bool(len(selected)))
            continue
        total += len(truth)
        if not len(selected):
            continue
        ious = pairwise_iou_numpy(selected, truth)
        pred_idx, truth_idx = linear_sum_assignment(1.0 - ious)
        matched.extend(float(ious[p, t]) for p, t in zip(pred_idx, truth_idx, strict=True) if ious[p, t] >= iou_cutoff)
    return {
        "targets": total,
        "matched_targets": len(matched),
        "target_recall": len(matched) / total if total else 0.0,
        "median_matched_iou": float(np.median(matched)) if matched else 0.0,
        "negative_window_fp": negative_fp,
    }


def synthetic_feasibility_smoke(*, device: str, seed: int = 20260828) -> dict[str, float | int | bool]:
    """Bounded learnability smoke; fixed data, 48 epochs, no external labels."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    rows, channels, count = 256, 8, 48
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 0.15, size=(count, rows, channels)).astype(np.float32)
    targets: list[np.ndarray] = []
    for index in range(count):
        if index % 3 == 0:
            targets.append(np.empty((0, 2), dtype=np.float32))
            continue
        start = 24 + (index * 17) % 128
        length = 40 + (index * 7) % 48
        end = min(rows - 8, start + length)
        features[index, start:end, 0] += 3.0
        features[index, start:end, 1] += np.linspace(0.0, 2.0, end - start, dtype=np.float32)
        targets.append(np.asarray([[start / rows, end / rows]], dtype=np.float32))
    config = DirectIntervalConfig(input_features=channels, window_rows=rows, patch_rows=16, d_model=32, heads=4, encoder_layers=1, queries=2)
    model = DirectIntervalSetPredictor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    losses: list[float] = []
    tensor = torch.as_tensor(features, device=device)
    for _ in range(48):
        optimizer.zero_grad(set_to_none=True)
        logits, intervals, patch_logits = model(tensor)
        loss = interval_set_loss(logits, intervals, patch_logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    predicted, scores = predict(model, features, batch_size=48)
    metrics = interval_metrics(predicted, scores, targets, threshold=0.5, iou_cutoff=0.65)
    metrics.update(
        {
            "finite": bool(np.isfinite(losses).all()),
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "passed": bool(
                np.isfinite(losses).all()
                and metrics["target_recall"] >= 0.9
                and metrics["median_matched_iou"] >= 0.75
                and metrics["negative_window_fp"] <= 2
            ),
        }
    )
    return metrics
