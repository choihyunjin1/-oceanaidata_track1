"""Utilities for the bounded P1 conditional real-event donor pilot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

CADENCE = pd.Timedelta(minutes=10)


@dataclass(frozen=True)
class EventSpan:
    station: str
    layer: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    rows: np.ndarray


def binary_metrics(truth: Sequence[int], prediction: Sequence[int]) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _segments(keys: pd.DataFrame, selected: np.ndarray) -> list[np.ndarray]:
    if not len(selected):
        return []
    frame = keys.iloc[selected].loc[:, ["station", "year", "layer", "time"]].copy()
    frame["position"] = selected
    frame["time"] = pd.to_datetime(frame["time"], utc=True, format="mixed")
    output: list[np.ndarray] = []
    for _, part in frame.groupby(["station", "year", "layer"], sort=False, observed=True):
        part = part.sort_values("time")
        positions = part["position"].to_numpy(dtype=np.int64)
        times = part["time"].to_numpy()
        groups = np.cumsum(np.r_[True, np.diff(times) != CADENCE])
        output.extend(positions[groups == group] for group in np.unique(groups))
    return output


def extract_long_events(
    keys: pd.DataFrame,
    labels: Sequence[int],
    anomaly_types: Sequence[str],
    rows: Sequence[int],
    *,
    eligible_types: Sequence[str],
    minimum_rows: int,
) -> list[EventSpan]:
    label = np.asarray(labels, dtype=np.int8)
    anomaly = pd.Series(anomaly_types, dtype="string").fillna("").str.lower()
    eligible = np.zeros(len(keys), dtype=bool)
    for name in eligible_types:
        eligible |= anomaly.str.contains(str(name).lower(), regex=False).to_numpy()
    membership = np.zeros(len(keys), dtype=bool)
    membership[np.asarray(rows, dtype=np.int64)] = True
    selected = np.flatnonzero(membership & eligible & (label == 1))
    times = pd.to_datetime(keys["time"], utc=True, format="mixed")
    output: list[EventSpan] = []
    for component in _segments(keys, selected):
        if len(component) < minimum_rows:
            continue
        first = int(component[0])
        output.append(
            EventSpan(
                str(keys.iloc[first]["station"]),
                int(keys.iloc[first]["layer"]),
                pd.Timestamp(times.iloc[first]),
                pd.Timestamp(times.iloc[int(component[-1])]),
                component,
            )
        )
    return output


def extract_mask_events(
    keys: pd.DataFrame, mask: Sequence[int], *, minimum_rows: int
) -> list[EventSpan]:
    active = np.flatnonzero(np.asarray(mask, dtype=np.int8) == 1)
    times = pd.to_datetime(keys["time"], utc=True, format="mixed")
    output: list[EventSpan] = []
    for component in _segments(keys, active):
        if len(component) < minimum_rows:
            continue
        first = int(component[0])
        output.append(
            EventSpan(
                str(keys.iloc[first]["station"]),
                int(keys.iloc[first]["layer"]),
                pd.Timestamp(times.iloc[first]),
                pd.Timestamp(times.iloc[int(component[-1])]),
                component,
            )
        )
    return output


def event_support(events: Sequence[EventSpan]) -> dict[str, float | int]:
    counts: dict[tuple[str, int], int] = {}
    for event in events:
        cell = (event.station, event.layer)
        counts[cell] = counts.get(cell, 0) + 1
    maximum = max(counts.values(), default=0)
    return {
        "events": len(events),
        "station_layer_cells": len(counts),
        "maximum_single_cell_share": maximum / len(events) if events else 0.0,
    }


def conditional_transplant(
    donor: np.ndarray,
    recipient: np.ndarray,
    *,
    replace_indices: Sequence[int],
) -> np.ndarray:
    if donor.shape != recipient.shape:
        raise ValueError("donor and recipient shapes differ")
    output = np.asarray(recipient, dtype=np.float32).copy()
    indices = np.asarray(replace_indices, dtype=np.int64)
    output[:, indices] = np.asarray(donor, dtype=np.float32)[:, indices]
    output[~np.isfinite(output)] = np.nan
    return output


def decode_scores(
    keys: pd.DataFrame,
    scores: Sequence[float],
    *,
    threshold: float,
    smoothing_rows: int,
    minimum_rows: int,
    bridge_rows: int,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(keys),):
        raise ValueError("score shape differs from keys")
    output = np.zeros(len(keys), dtype=np.int8)
    for segment in _segments(keys, np.arange(len(keys), dtype=np.int64)):
        smooth = (
            pd.Series(values[segment])
            .rolling(smoothing_rows, center=True, min_periods=max(1, smoothing_rows // 2))
            .mean()
            .to_numpy()
        )
        active = smooth >= threshold
        cursor = 0
        while cursor < len(active):
            if active[cursor]:
                cursor += 1
                continue
            stop = cursor
            while stop < len(active) and not active[stop]:
                stop += 1
            if cursor > 0 and stop < len(active) and stop - cursor <= bridge_rows:
                active[cursor:stop] = True
            cursor = stop
        cursor = 0
        while cursor < len(active):
            if not active[cursor]:
                cursor += 1
                continue
            stop = cursor + 1
            while stop < len(active) and active[stop]:
                stop += 1
            if stop - cursor >= minimum_rows:
                output[segment[cursor:stop]] = 1
            cursor = stop
    return output


def _iou(left: EventSpan, right: EventSpan) -> float:
    if (left.station, left.layer) != (right.station, right.layer):
        return 0.0
    intersection = len(np.intersect1d(left.rows, right.rows, assume_unique=True))
    union = len(left.rows) + len(right.rows) - intersection
    return intersection / union if union else 0.0


def proposal_support_metrics(
    truth_events: Sequence[EventSpan],
    proposals: Sequence[EventSpan],
    row_truth: Sequence[int],
    *,
    iou_threshold: float,
) -> dict[str, float | int]:
    truth = np.asarray(row_truth, dtype=np.int8)
    matched_proposals: list[EventSpan] = []
    for proposal in proposals:
        if any(_iou(proposal, event) >= iou_threshold for event in truth_events):
            matched_proposals.append(proposal)
    matched_truth = sum(
        any(_iou(event, proposal) >= iou_threshold for proposal in proposals)
        for event in truth_events
    )
    cells = {(event.station, event.layer) for event in matched_proposals}
    proposal_rows = (
        np.unique(np.concatenate([event.rows for event in proposals]))
        if proposals
        else np.empty(0, dtype=np.int64)
    )
    return {
        "proposals": len(proposals),
        "matched_proposals": len(matched_proposals),
        "matched_station_layer_cells": len(cells),
        "matched_truth_events": matched_truth,
        "truth_events": len(truth_events),
        "real_event_recall": matched_truth / len(truth_events) if truth_events else 0.0,
        "proposal_row_precision": float(np.mean(truth[proposal_rows] == 1))
        if len(proposal_rows)
        else 0.0,
    }


def evaluate_anchor_union(
    truth: Sequence[int], anchor: Sequence[int], additions: Sequence[int]
) -> dict[str, object]:
    y = np.asarray(truth, dtype=np.int8)
    base = np.asarray(anchor, dtype=np.int8)
    add = np.asarray(additions, dtype=np.int8)
    candidate = np.maximum(base, add).astype(np.int8)
    anchor_metrics = binary_metrics(y, base)
    candidate_metrics = binary_metrics(y, candidate)
    changed = (base == 0) & (candidate == 1)
    added_tp = int(np.sum(changed & (y == 1)))
    added_fp = int(np.sum(changed & (y == 0)))
    normal_days = max(1.0, float(np.sum(y == 0)) / 144.0)
    anchor_fp_day = float(anchor_metrics["fp"]) / normal_days
    candidate_fp_day = float(candidate_metrics["fp"]) / normal_days
    if anchor_fp_day > 0:
        fp_relative: float | None = candidate_fp_day / anchor_fp_day
    else:
        fp_relative = 1.0 if candidate_fp_day == 0 else None
    return {
        "anchor": anchor_metrics,
        "candidate": candidate_metrics,
        "delta_f1": float(candidate_metrics["f1"]) - float(anchor_metrics["f1"]),
        "added_rows": int(changed.sum()),
        "added_tp": added_tp,
        "added_fp": added_fp,
        "added_precision": added_tp / (added_tp + added_fp) if added_tp + added_fp else 1.0,
        "anchor_f1_over_2": float(anchor_metrics["f1"]) / 2.0,
        "anchor_fp_per_day": anchor_fp_day,
        "candidate_fp_per_day": candidate_fp_day,
        "normal_fp_per_day_relative": fp_relative,
        "anchor_positive_removed_rows": int(np.sum((base == 1) & (candidate == 0))),
    }


def newly_recovered_events(
    truth_events: Sequence[EventSpan], anchor: Sequence[int], additions: Sequence[int]
) -> int:
    base = np.asarray(anchor, dtype=np.int8)
    add = np.asarray(additions, dtype=np.int8)
    return sum(
        not np.any(base[event.rows] == 1) and np.any(add[event.rows] == 1)
        for event in truth_events
    )


def block_bootstrap_delta(
    keys: pd.DataFrame,
    truth: Sequence[int],
    anchor: Sequence[int],
    additions: Sequence[int],
    *,
    replicates: int,
    block_days: int,
    seed: int,
) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    base = np.asarray(anchor, dtype=np.int8)
    candidate = np.maximum(base, np.asarray(additions, dtype=np.int8)).astype(np.int8)
    day = pd.to_datetime(keys["time"], utc=True, format="mixed").dt.tz_convert(
        "Asia/Seoul"
    ).dt.floor("D")
    unique_days = pd.DatetimeIndex(day.unique()).sort_values()
    day_rows = [np.flatnonzero((day == value).to_numpy()) for value in unique_days]
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    needed_blocks = int(np.ceil(len(unique_days) / block_days))
    for replicate in range(replicates):
        starts = rng.integers(0, len(unique_days), size=needed_blocks)
        sampled_days = [
            int((start + offset) % len(unique_days))
            for start in starts
            for offset in range(block_days)
        ][: len(unique_days)]
        rows = np.concatenate([day_rows[index] for index in sampled_days])
        deltas[replicate] = float(binary_metrics(y[rows], candidate[rows])["f1"]) - float(
            binary_metrics(y[rows], base[rows])["f1"]
        )
    return {
        "replicates": replicates,
        "block_days": block_days,
        "median": float(np.median(deltas)),
        "ci90_lower": float(np.quantile(deltas, 0.05)),
        "ci90_upper": float(np.quantile(deltas, 0.95)),
        "probability_positive": float(np.mean(deltas > 0.0)),
    }
