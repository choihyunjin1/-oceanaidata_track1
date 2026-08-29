"""Leakage-safe helpers for the P1 real-event SupCon and top-k screen.

The module has no filesystem entry point.  It only consumes already encoded
historical training surfaces and aligned blind probabilities supplied by the
one-shot runner.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from p1_qc.ms_tcn_asrf_data import WindowIndex, build_window_index


@dataclass(frozen=True)
class RealEvent:
    event_id: int
    segment_id: int
    start_local: int
    stop_local: int
    center_local: int
    type_index: int
    station: str
    layer: str
    season: str


def meteorological_season(values: Any) -> np.ndarray:
    parsed = pd.to_datetime(values, utc=True, format="mixed")
    month = parsed.dt.month.to_numpy() if isinstance(parsed, pd.Series) else parsed.month.to_numpy()
    labels = np.empty(len(month), dtype=object)
    labels[np.isin(month, [12, 1, 2])] = "DJF"
    labels[np.isin(month, [3, 4, 5])] = "MAM"
    labels[np.isin(month, [6, 7, 8])] = "JJA"
    labels[np.isin(month, [9, 10, 11])] = "SON"
    return labels.astype(str)


def build_real_events(encoded: Any) -> tuple[tuple[RealEvent, ...], dict[str, Any]]:
    """Enumerate contiguous real positive runs without generating anomalies."""

    if encoded.targets is None:
        raise ValueError("real-event extraction requires training targets")
    labels = np.asarray(encoded.targets.row_label, dtype=np.int8)
    kinds = np.asarray(encoded.targets.anomaly_type, dtype=np.float32)
    if labels.shape != (encoded.surface.rows,) or kinds.shape != (encoded.surface.rows, 5):
        raise ValueError("training targets are not aligned")
    seasons = meteorological_season(encoded.surface.keys["time"])
    events: list[RealEvent] = []
    missing_type = 0
    multi_type = 0
    for segment in encoded.layout.segments:
        local = labels[segment.row_ids]
        cursor = 0
        while cursor < segment.size:
            if int(local[cursor]) == 0:
                cursor += 1
                continue
            stop = cursor + 1
            while stop < segment.size and int(local[stop]) == 1:
                stop += 1
            rows = segment.row_ids[cursor:stop]
            support = kinds[rows].sum(axis=0)
            present = np.flatnonzero(support > 0)
            if len(present) == 0:
                missing_type += 1
                cursor = stop
                continue
            if len(present) > 1:
                multi_type += 1
            type_index = int(np.flatnonzero(support == support.max())[0])
            center_local = int((cursor + stop - 1) // 2)
            center_row = int(segment.row_ids[center_local])
            events.append(
                RealEvent(
                    event_id=len(events),
                    segment_id=int(segment.segment_id),
                    start_local=int(cursor),
                    stop_local=int(stop),
                    center_local=center_local,
                    type_index=type_index,
                    station=str(encoded.surface.keys.iloc[center_row]["station"]),
                    layer=str(encoded.surface.keys.iloc[center_row]["layer"]),
                    season=str(seasons[center_row]),
                )
            )
            cursor = stop
    if not events:
        raise ValueError("no typed real events are available")
    type_counts = Counter(event.type_index for event in events)
    cell_counts = Counter((event.station, event.layer, event.season) for event in events)
    return tuple(events), {
        "real_event_count": len(events),
        "untyped_event_count_excluded": missing_type,
        "multi_type_event_count": multi_type,
        "event_count_by_type_index": {str(key): int(value) for key, value in sorted(type_counts.items())},
        "supported_type_count": len(type_counts),
        "supported_station_layer_season_cells": len(cell_counts),
        "maximum_event_cell_share": max(cell_counts.values()) / len(events),
        "synthetic_event_count": 0,
    }


def _centered_window(segment: Any, center: int, window_size: int) -> WindowIndex:
    maximum_start = max(0, int(segment.size) - window_size)
    start = min(maximum_start, max(0, int(center) - window_size // 2))
    rows = segment.row_ids[start : min(start + window_size, segment.size)].copy()
    return WindowIndex(int(segment.segment_id), int(start), int(window_size), rows)


def _window_key(encoded: Any, window: WindowIndex, seasons: np.ndarray) -> tuple[str, str, str]:
    row = int(window.row_ids[len(window.row_ids) // 2])
    keys = encoded.surface.keys.iloc[row]
    return str(keys["station"]), str(keys["layer"]), str(seasons[row])


def _rank(seed: int, event_id: int, window: WindowIndex) -> bytes:
    payload = (
        f"{seed}|{event_id}|{window.segment_id}|{window.start}|{window.valid_length}"
    ).encode("ascii")
    return hashlib.sha256(payload).digest()


def event_balanced_windows(
    encoded: Any,
    *,
    window_size: int,
    stride: int,
    seed: int,
) -> tuple[tuple[WindowIndex, ...], np.ndarray, np.ndarray, dict[str, Any]]:
    """Return one centered real-event window and one matched normal per event."""

    events, support = build_real_events(encoded)
    labels = np.asarray(encoded.targets.row_label, dtype=np.int8)
    seasons = meteorological_season(encoded.surface.keys["time"])
    all_windows = build_window_index(encoded.layout, window_size=window_size, stride=stride)
    normal = tuple(window for window in all_windows if not bool(labels[window.row_ids].any()))
    if not normal:
        raise ValueError("no normal windows are available for event matching")
    by_exact: dict[tuple[str, str, str], list[WindowIndex]] = defaultdict(list)
    by_station_layer: dict[tuple[str, str], list[WindowIndex]] = defaultdict(list)
    by_layer_season: dict[tuple[str, str], list[WindowIndex]] = defaultdict(list)
    for window in normal:
        station, layer, season = _window_key(encoded, window, seasons)
        by_exact[(station, layer, season)].append(window)
        by_station_layer[(station, layer)].append(window)
        by_layer_season[(layer, season)].append(window)

    segments = {int(segment.segment_id): segment for segment in encoded.layout.segments}
    selected: list[WindowIndex] = []
    class_ids: list[int] = []
    is_event: list[bool] = []
    used: Counter[tuple[int, int]] = Counter()
    match_tiers: Counter[str] = Counter()
    for event in events:
        selected.append(_centered_window(segments[event.segment_id], event.center_local, window_size))
        class_ids.append(event.type_index)
        is_event.append(True)
        candidates: list[WindowIndex]
        if by_exact[(event.station, event.layer, event.season)]:
            candidates = by_exact[(event.station, event.layer, event.season)]
            tier = "station_layer_season"
        elif by_station_layer[(event.station, event.layer)]:
            candidates = by_station_layer[(event.station, event.layer)]
            tier = "station_layer"
        elif by_layer_season[(event.layer, event.season)]:
            candidates = by_layer_season[(event.layer, event.season)]
            tier = "layer_season"
        else:
            candidates = list(normal)
            tier = "global"
        choice = min(
            candidates,
            key=lambda window: (
                used[(int(window.segment_id), int(window.start))],
                _rank(seed, event.event_id, window),
            ),
        )
        used[(int(choice.segment_id), int(choice.start))] += 1
        selected.append(choice)
        class_ids.append(5)
        is_event.append(False)
        match_tiers[tier] += 1
    receipt = {
        **support,
        "balanced_window_count": len(selected),
        "event_window_count": len(events),
        "matched_normal_window_count": len(events),
        "normal_class_index": 5,
        "normal_match_tiers": dict(sorted(match_tiers.items())),
        "unique_matched_normal_windows": len(used),
        "event_sampling_unit": "one_centered_window_per_contiguous_real_event",
        "synthetic_event_count": 0,
    }
    return (
        tuple(selected),
        np.asarray(class_ids, dtype=np.int64),
        np.asarray(is_event, dtype=bool),
        receipt,
    )


def supervised_contrastive_loss(embeddings: Any, class_ids: Any, *, temperature: float) -> Any:
    """Standard supervised contrastive loss over same-class batch positives."""

    import torch

    if embeddings.ndim != 2 or class_ids.ndim != 1 or embeddings.shape[0] != class_ids.shape[0]:
        raise ValueError("SupCon inputs must be aligned [batch, feature] and [batch]")
    if embeddings.shape[0] < 2 or temperature <= 0.0:
        raise ValueError("SupCon needs at least two samples and positive temperature")
    normalized = torch.nn.functional.normalize(embeddings.float(), dim=1)
    logits = normalized @ normalized.transpose(0, 1) / float(temperature)
    eye = torch.eye(len(class_ids), dtype=torch.bool, device=embeddings.device)
    same = class_ids[:, None].eq(class_ids[None, :]) & ~eye
    valid_anchor = same.any(dim=1)
    if not bool(valid_anchor.any()):
        return embeddings.float().sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive = (log_prob * same.float()).sum(dim=1) / same.sum(dim=1).clamp_min(1)
    return -mean_positive[valid_anchor].mean()


def soft_f1_loss(logits: Any, labels: Any, valid_mask: Any, *, epsilon: float = 1e-6) -> Any:
    import torch

    if logits.shape != labels.shape or logits.shape != valid_mask.shape:
        raise ValueError("soft-F1 tensors must be aligned")
    probability = torch.sigmoid(logits.float()).masked_select(valid_mask)
    truth = labels.float().masked_select(valid_mask)
    if probability.numel() == 0:
        raise ValueError("soft-F1 valid surface is empty")
    intersection = (probability * truth).sum()
    score = (2.0 * intersection + epsilon) / (probability.sum() + truth.sum() + epsilon)
    return 1.0 - score


def pool_shared_hidden(shared: Any, event: Any, kinds: Any, valid: Any, class_ids: Any) -> Any:
    """Pool the generator state on the selected event type or the normal window."""

    import torch

    if shared.ndim != 3 or event.ndim != 2 or kinds.ndim != 3:
        raise ValueError("hidden pooling tensors have invalid rank")
    masks: list[Any] = []
    for index, class_id in enumerate(class_ids.detach().cpu().tolist()):
        if int(class_id) == 5:
            mask = valid[index]
        else:
            mask = (kinds[index, :, int(class_id)] > 0.0) & valid[index]
            if not bool(mask.any()):
                mask = (event[index] > 0.0) & valid[index]
        masks.append(mask)
    mask_tensor = torch.stack(masks).to(dtype=shared.dtype).unsqueeze(1)
    return (shared * mask_tensor).sum(dim=2) / mask_tensor.sum(dim=2).clamp_min(1.0)


def best_topk_rate(
    probability: Any, labels: Any, *, maximum_rate: float
) -> tuple[float, dict[str, Any]]:
    score = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int8)
    if score.ndim != 1 or score.shape != truth.shape or not np.isfinite(score).all():
        raise ValueError("top-k calibration vectors are invalid")
    if not np.isin(truth, [0, 1]).all() or not 0.0 <= maximum_rate <= 1.0:
        raise ValueError("top-k calibration contract is invalid")
    n_rows = len(score)
    maximum_k = min(n_rows, int(np.floor(maximum_rate * n_rows)))
    order = np.argsort(-score, kind="stable")
    positives = truth[order]
    cumulative = np.cumsum(positives)
    total_positive = int(truth.sum())
    best = (0.0, 0)
    for k in range(1, maximum_k + 1):
        tp = int(cumulative[k - 1])
        fp = k - tp
        fn = total_positive - tp
        f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if f1 > best[0]:
            best = (float(f1), k)
    return (best[1] / n_rows if n_rows else 0.0), {
        "rows": n_rows,
        "positive_rows": total_positive,
        "selected_k": int(best[1]),
        "selected_rate": best[1] / n_rows if n_rows else 0.0,
        "training_f1": best[0],
        "maximum_rate": float(maximum_rate),
    }


def calibrate_cell_topk_rates(
    keys: pd.DataFrame,
    probability: Any,
    labels: Any,
    *,
    minimum_rows: int,
    minimum_positives: int,
    maximum_rate: float,
) -> tuple[dict[tuple[str, str, str], float], float, dict[str, Any]]:
    if minimum_rows < 1 or minimum_positives < 1:
        raise ValueError("cell support minima must be positive")
    seasons = meteorological_season(keys["time"])
    cells = list(
        zip(keys["station"].astype(str), keys["layer"].astype(str), seasons, strict=True)
    )
    score = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int8)
    global_rate, global_receipt = best_topk_rate(score, truth, maximum_rate=maximum_rate)
    by_cell: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        by_cell[cell].append(index)
    rates: dict[tuple[str, str, str], float] = {}
    eligible = 0
    for cell, indices in sorted(by_cell.items()):
        ids = np.asarray(indices, dtype=np.int64)
        if len(ids) < minimum_rows or int(truth[ids].sum()) < minimum_positives:
            continue
        rates[cell], _receipt = best_topk_rate(
            score[ids], truth[ids], maximum_rate=maximum_rate
        )
        eligible += 1
    return rates, global_rate, {
        "definition": "station_x_layer_x_meteorological_season",
        "total_cell_count": len(by_cell),
        "eligible_cell_count": eligible,
        "fallback_cell_count": len(by_cell) - eligible,
        "minimum_rows": int(minimum_rows),
        "minimum_positives": int(minimum_positives),
        "global_fallback": global_receipt,
        "holdout_rows_used": 0,
        "holdout_truth_rows_used": 0,
    }


def apply_cell_topk(
    keys: pd.DataFrame,
    probability: Any,
    anchor: Any,
    *,
    cell_rates: dict[tuple[str, str, str], float],
    fallback_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    score = np.asarray(probability, dtype=np.float64)
    frozen = np.asarray(anchor, dtype=np.int8)
    if score.shape != frozen.shape or not np.isfinite(score).all() or not np.isin(frozen, [0, 1]).all():
        raise ValueError("blind top-k vectors are invalid")
    seasons = meteorological_season(keys["time"])
    cells = list(
        zip(keys["station"].astype(str), keys["layer"].astype(str), seasons, strict=True)
    )
    by_cell: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        by_cell[cell].append(index)
    proposal = np.zeros(len(score), dtype=np.int8)
    selected_by_cell: dict[str, int] = {}
    fallback_count = 0
    for cell, indices in sorted(by_cell.items()):
        ids = np.asarray(indices, dtype=np.int64)
        eligible = ids[frozen[ids] == 0]
        rate = float(cell_rates.get(cell, fallback_rate))
        if cell not in cell_rates:
            fallback_count += 1
        k = min(len(eligible), int(np.rint(rate * len(eligible))))
        if k > 0:
            chosen = eligible[np.argsort(-score[eligible], kind="stable")[:k]]
            proposal[chosen] = 1
        selected_by_cell["|".join(cell)] = int(k)
    return proposal, {
        "proposal_rows": int(proposal.sum()),
        "cell_count": len(by_cell),
        "fallback_cell_count": fallback_count,
        "selected_rows_by_cell": selected_by_cell,
        "anchor_positive_rows_proposed": int(np.sum((proposal == 1) & (frozen == 1))),
        "holdout_truth_rows_used": 0,
    }


__all__ = [
    "RealEvent",
    "apply_cell_topk",
    "best_topk_rate",
    "build_real_events",
    "calibrate_cell_topk_rates",
    "event_balanced_windows",
    "meteorological_season",
    "pool_shared_hidden",
    "soft_f1_loss",
    "supervised_contrastive_loss",
]
