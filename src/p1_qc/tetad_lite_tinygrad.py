"""A small direct temporal interval-set predictor implemented with tinygrad.

The module is deliberately independent from the P1 data pipeline.  Inputs are
dense, already-prepared feature windows with shape ``[batch, time, feature]``;
outputs are five unordered anomaly proposals per window.  Intervals use
normalized half-open coordinates in ``[0, 1]``.

Hungarian matching is discrete (and therefore detached), while every loss
evaluated after matching remains differentiable with respect to the model
outputs.  This is the same separation used by set-prediction objectives: the
assignment selects pairs, and BCE/L1/IoU train the selected predictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from tinygrad import Tensor, nn


@dataclass(frozen=True)
class TETADLiteConfig:
    """Architecture parameters for :class:`TETADLiteTinygrad`."""

    input_features: int
    patch_size: int = 8
    d_model: int = 64
    num_heads: int = 4
    ff_multiplier: int = 4
    num_encoder_layers: int = 2
    num_queries: int = 5
    max_patches: int = 512

    def __post_init__(self) -> None:
        if self.input_features <= 0 or self.patch_size <= 0:
            raise ValueError("input_features and patch_size must be positive")
        if self.d_model <= 0 or self.num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.ff_multiplier <= 0 or self.num_encoder_layers != 2:
            raise ValueError("this fixed architecture requires exactly two encoder layers")
        if self.num_queries != 5:
            raise ValueError("this fixed architecture requires exactly five queries")
        if self.max_patches <= 0:
            raise ValueError("max_patches must be positive")


class _MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int) -> None:
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, query: Tensor, key_value: Tensor) -> Tensor:
        batch, query_length, d_model = query.shape
        key_length = key_value.shape[1]
        q = self.q_proj(query).reshape(
            batch, query_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(
            batch, key_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(
            batch, key_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        attended = q.scaled_dot_product_attention(k, v, is_causal=False)
        merged = attended.permute(0, 2, 1, 3).reshape(batch, query_length, d_model)
        return self.out_proj(merged)


class _EncoderLayer:
    def __init__(self, d_model: int, num_heads: int, ff_multiplier: int) -> None:
        self.attention = _MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * ff_multiplier)
        self.ff2 = nn.Linear(d_model * ff_multiplier, d_model)

    def __call__(self, x: Tensor) -> Tensor:
        # No causal mask: anomaly QC is an explicitly bidirectional offline task.
        x = x + self.attention(self.norm1(x), self.norm1(x))
        normalized = self.norm2(x)
        return x + self.ff2(self.ff1(normalized).gelu())


class TETADLiteTinygrad:
    """Patch-transformer that predicts an unordered set of anomaly intervals.

    The model contains a patch projection, exactly two bidirectional
    self-attention layers, five learned queries, a cross-attention query head,
    and prediction heads for presence and normalized start/end coordinates.
    """

    def __init__(self, config: TETADLiteConfig) -> None:
        self.config = config
        patch_width = config.patch_size * config.input_features
        self.patch_projection = nn.Linear(patch_width, config.d_model)
        self.position_embedding = Tensor.scaled_uniform(config.max_patches, config.d_model)
        self.encoder_layers = [
            _EncoderLayer(config.d_model, config.num_heads, config.ff_multiplier)
            for _ in range(config.num_encoder_layers)
        ]
        self.query_embedding = Tensor.scaled_uniform(config.num_queries, config.d_model)
        self.query_norm = nn.LayerNorm(config.d_model)
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = _MultiHeadAttention(config.d_model, config.num_heads)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.class_head = nn.Linear(config.d_model, 1)
        self.endpoint_head = nn.Linear(config.d_model, 2)

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, feature]")
        batch, time_steps, features = x.shape
        if features != self.config.input_features:
            raise ValueError(
                f"expected {self.config.input_features} input features, got {features}"
            )
        if time_steps <= 0:
            raise ValueError("time dimension must be non-empty")

        patch_count = (time_steps + self.config.patch_size - 1) // self.config.patch_size
        if patch_count > self.config.max_patches:
            raise ValueError(
                f"input needs {patch_count} patches, above max_patches={self.config.max_patches}"
            )
        padded_steps = patch_count * self.config.patch_size
        if padded_steps != time_steps:
            x = x.pad((None, (0, padded_steps - time_steps), None))
        patches = x.reshape(batch, patch_count, -1)
        memory = self.patch_projection(patches) + self.position_embedding[:patch_count]
        for layer in self.encoder_layers:
            memory = layer(memory)

        # Broadcasting [Q,D] against [B,1,D] creates one learned query bank per batch.
        queries = self.query_embedding + memory[:, :1, :] * 0.0
        query_state = queries + self.cross_attention(
            self.query_norm(queries), self.memory_norm(memory)
        )
        query_state = self.output_norm(query_state)
        class_logits = self.class_head(query_state).reshape(batch, self.config.num_queries)

        raw_endpoints = self.endpoint_head(query_state).sigmoid()
        first, second = raw_endpoints[:, :, 0], raw_endpoints[:, :, 1]
        starts = first.minimum(second)
        ends = first.maximum(second)
        intervals = Tensor.stack(starts, ends, dim=-1)
        return class_logits, intervals


Assignment = tuple[np.ndarray, np.ndarray]


def _as_target_arrays(
    targets: Sequence[np.ndarray | Sequence[Sequence[float]]], batch_size: int
) -> list[np.ndarray]:
    if len(targets) != batch_size:
        raise ValueError(f"expected {batch_size} target arrays, got {len(targets)}")
    normalized: list[np.ndarray] = []
    for target in targets:
        array = np.asarray(target, dtype=np.float32)
        if array.size == 0:
            array = np.empty((0, 2), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("each target must have shape [number_of_intervals, 2]")
        if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
            raise ValueError("target endpoints must be finite and lie in [0, 1]")
        if (array[:, 0] > array[:, 1]).any():
            raise ValueError("target starts must not exceed target ends")
        normalized.append(array)
    return normalized


def _pairwise_iou_numpy(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    left = np.maximum(predicted[:, None, 0], target[None, :, 0])
    right = np.minimum(predicted[:, None, 1], target[None, :, 1])
    intersection = np.maximum(right - left, 0.0)
    pred_length = np.maximum(predicted[:, 1] - predicted[:, 0], 0.0)[:, None]
    target_length = np.maximum(target[:, 1] - target[:, 0], 0.0)[None, :]
    union = pred_length + target_length - intersection
    return intersection / np.maximum(union, 1e-7)


def hungarian_assignments(
    class_logits: Tensor,
    intervals: Tensor,
    targets: Sequence[np.ndarray | Sequence[Sequence[float]]],
    *,
    class_cost: float = 1.0,
    endpoint_cost: float = 2.0,
    iou_cost: float = 2.0,
) -> list[Assignment]:
    """Match predicted queries to ground-truth intervals with Hungarian assignment.

    Matching is intentionally evaluated on realized NumPy arrays.  Its indices
    are treated as constants by :func:`interval_set_loss`; gradients flow
    through the losses calculated for those selected query/target pairs.
    """

    if class_logits.ndim != 2 or intervals.shape != (*class_logits.shape, 2):
        raise ValueError("expected logits [B,Q] and intervals [B,Q,2]")
    if min(class_cost, endpoint_cost, iou_cost) < 0:
        raise ValueError("matching costs must be non-negative")
    target_arrays = _as_target_arrays(targets, class_logits.shape[0])
    overflowing = [
        index
        for index, target in enumerate(target_arrays)
        if target.shape[0] > class_logits.shape[1]
    ]
    if overflowing:
        raise ValueError(
            "ground-truth interval count exceeds the fixed query budget in "
            f"batch items {overflowing}"
        )
    logits_np = class_logits.detach().numpy()
    intervals_np = intervals.detach().numpy()
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -30.0, 30.0)))

    assignments: list[Assignment] = []
    for batch_index, target in enumerate(target_arrays):
        if target.shape[0] == 0:
            empty = np.empty(0, dtype=np.int64)
            assignments.append((empty, empty.copy()))
            continue
        predicted = intervals_np[batch_index]
        classification = -np.log(np.maximum(probabilities[batch_index], 1e-7))[:, None]
        endpoint_l1 = np.abs(predicted[:, None, :] - target[None, :, :]).sum(axis=2)
        one_minus_iou = 1.0 - _pairwise_iou_numpy(predicted, target)
        cost = class_cost * classification + endpoint_cost * endpoint_l1 + iou_cost * one_minus_iou
        pred_indices, target_indices = linear_sum_assignment(cost)
        assignments.append(
            (pred_indices.astype(np.int64), target_indices.astype(np.int64))
        )
    return assignments


@dataclass(frozen=True)
class IntervalSetLoss:
    total: Tensor
    classification: Tensor
    endpoint_l1: Tensor
    interval_iou: Tensor
    assignments: list[Assignment]


def interval_set_loss(
    class_logits: Tensor,
    intervals: Tensor,
    targets: Sequence[np.ndarray | Sequence[Sequence[float]]],
    *,
    classification_weight: float = 1.0,
    endpoint_weight: float = 2.0,
    iou_weight: float = 2.0,
    positive_class_weight: float = 1.0,
) -> IntervalSetLoss:
    """Compute differentiable BCE, endpoint-L1, and one-minus-IoU losses."""

    if min(classification_weight, endpoint_weight, iou_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if positive_class_weight <= 0:
        raise ValueError("positive_class_weight must be positive")
    target_arrays = _as_target_arrays(targets, class_logits.shape[0])
    assignments = hungarian_assignments(class_logits, intervals, target_arrays)

    class_targets_np = np.zeros(class_logits.shape, dtype=np.float32)
    selected_predictions: list[Tensor] = []
    selected_targets: list[Tensor] = []
    for batch_index, (pred_indices, target_indices) in enumerate(assignments):
        if pred_indices.size == 0:
            continue
        class_targets_np[batch_index, pred_indices] = 1.0
        selected_predictions.append(intervals[batch_index, pred_indices.tolist()])
        selected_targets.append(
            Tensor(
                target_arrays[batch_index][target_indices],
                device=intervals.device,
                dtype=intervals.dtype,
            )
        )

    class_targets = Tensor(
        class_targets_np, device=class_logits.device, dtype=class_logits.dtype
    )
    classification = class_logits.binary_crossentropy_logits(
        class_targets,
        pos_weight=Tensor(
            [positive_class_weight],
            device=class_logits.device,
            dtype=class_logits.dtype,
        ),
    )

    if selected_predictions:
        predicted = Tensor.cat(*selected_predictions, dim=0)
        target = Tensor.cat(*selected_targets, dim=0)
        endpoint_l1 = (predicted - target).abs().mean()
        left = predicted[:, 0].maximum(target[:, 0])
        right = predicted[:, 1].minimum(target[:, 1])
        intersection = (right - left).relu()
        predicted_length = (predicted[:, 1] - predicted[:, 0]).relu()
        target_length = (target[:, 1] - target[:, 0]).relu()
        union = predicted_length + target_length - intersection
        interval_iou = 1.0 - (intersection / (union + 1e-7)).mean()
    else:
        # Keep zero-valued terms attached to the graph for a uniform backward path.
        endpoint_l1 = intervals.sum() * 0.0
        interval_iou = intervals.sum() * 0.0

    total = (
        classification_weight * classification
        + endpoint_weight * endpoint_l1
        + iou_weight * interval_iou
    )
    return IntervalSetLoss(total, classification, endpoint_l1, interval_iou, assignments)


def rasterize_intervals(
    intervals: np.ndarray | Sequence[Sequence[float]],
    scores: np.ndarray | Sequence[float],
    length: int,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Rasterize scored normalized intervals into a deterministic binary mask.

    Start indices use floor, end indices use ceil, and selected zero-width
    proposals occupy one cell.  Overlapping intervals are combined by union.
    """

    if length <= 0:
        raise ValueError("length must be positive")
    interval_array = np.asarray(intervals, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if interval_array.ndim != 2 or interval_array.shape[1] != 2:
        raise ValueError("intervals must have shape [number_of_queries, 2]")
    if score_array.shape != (interval_array.shape[0],):
        raise ValueError("scores must have one value per interval")
    if not np.isfinite(interval_array).all() or not np.isfinite(score_array).all():
        raise ValueError("intervals and scores must be finite")

    mask = np.zeros(length, dtype=np.int8)
    for (raw_start, raw_end), score in zip(interval_array, score_array, strict=True):
        if score < threshold:
            continue
        start_value, end_value = sorted(
            (float(np.clip(raw_start, 0.0, 1.0)), float(np.clip(raw_end, 0.0, 1.0)))
        )
        start = min(int(np.floor(start_value * length)), length - 1)
        end = min(int(np.ceil(end_value * length)), length)
        end = max(end, start + 1)
        mask[start:end] = 1
    return mask


def aggregate_binary_metrics(
    truth: np.ndarray | Sequence[int], prediction: np.ndarray | Sequence[int]
) -> dict[str, int | float]:
    """Return aggregate binary confusion counts and row-level precision/recall/F1."""

    truth_array = np.asarray(truth).reshape(-1)
    prediction_array = np.asarray(prediction).reshape(-1)
    if truth_array.shape != prediction_array.shape:
        raise ValueError("truth and prediction must have the same flattened shape")
    if not np.isin(truth_array, (0, 1)).all() or not np.isin(prediction_array, (0, 1)).all():
        raise ValueError("truth and prediction must contain only 0 and 1")
    truth_positive = truth_array.astype(bool)
    predicted_positive = prediction_array.astype(bool)
    tp = int(np.sum(truth_positive & predicted_positive))
    fp = int(np.sum(~truth_positive & predicted_positive))
    fn = int(np.sum(truth_positive & ~predicted_positive))
    tn = int(np.sum(~truth_positive & ~predicted_positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": int(truth_array.size),
        "positive_support": int(np.sum(truth_positive)),
        "predicted_positive": int(np.sum(predicted_positive)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
