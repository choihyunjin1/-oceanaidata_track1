"""Event-level verifier utilities for a frozen row-wise proposal generator."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

CADENCE = pd.Timedelta(minutes=10)


@dataclass(frozen=True)
class EventProposal:
    proposal_id: str
    station: str
    year: int
    layer: int
    start: int
    end: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    row_ids: np.ndarray
    features: np.ndarray


def binary_metrics(truth: Sequence[int], prediction: Sequence[int]) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def chronological_boundaries(times: Sequence[Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique = pd.DatetimeIndex(pd.to_datetime(pd.Series(times), utc=True, format="mixed").sort_values().unique())
    if len(unique) < 4:
        raise ValueError("insufficient unique timestamps")
    return unique[int(np.floor(0.50 * len(unique)))], unique[int(np.floor(0.75 * len(unique)))]


def split_intervals(boundary1: pd.Timestamp, boundary2: pd.Timestamp, purge_days: int) -> dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]:
    purge = pd.Timedelta(days=purge_days)
    return {
        "train": (None, boundary1 - purge),
        "calibration": (boundary1, boundary2 - purge),
        "qualification": (boundary2, None),
    }


def _safe_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(finite)), float(np.std(finite)), float(np.median(finite)), float(np.max(np.abs(finite)))


def _component_features(
    numeric: np.ndarray,
    confidence: np.ndarray,
    anchor: np.ndarray,
    rows: np.ndarray,
    segment_rows: np.ndarray,
    feature_indices: Sequence[int],
    context_rows: int,
) -> np.ndarray:
    local_start = int(np.flatnonzero(segment_rows == rows[0])[0])
    local_end = local_start + len(rows)
    before = segment_rows[max(0, local_start - context_rows) : local_start]
    after = segment_rows[local_end : min(len(segment_rows), local_end + context_rows)]
    context = np.concatenate([before, after])
    result = [
        float(len(rows)),
        float(np.mean(confidence[rows])),
        float(np.max(confidence[rows])),
        float(np.min(confidence[rows])),
        float(np.quantile(confidence[rows], 0.9)),
        float(np.mean(anchor[rows])),
        float(np.sum(anchor[rows])),
    ]
    for index in feature_indices:
        inside = numeric[rows, index].astype(np.float64)
        inside_stats = _safe_stats(inside)
        if len(context):
            context_values = numeric[context, index].astype(np.float64)
            context_median = _safe_stats(context_values)[2]
        else:
            context_median = 0.0
        before_median = _safe_stats(numeric[before, index].astype(np.float64))[2] if len(before) else 0.0
        after_median = _safe_stats(numeric[after, index].astype(np.float64))[2] if len(after) else 0.0
        result.extend([*inside_stats, inside_stats[2] - context_median, after_median - before_median])
    output = np.asarray(result, dtype=np.float64)
    output[~np.isfinite(output)] = 0.0
    return output


def build_event_proposals(
    keys: pd.DataFrame,
    numeric: np.ndarray,
    numeric_names: Sequence[str],
    confidence: np.ndarray,
    proposal_mask: np.ndarray,
    anchor: np.ndarray,
    *,
    selected_numeric: Sequence[str],
    minimum_rows: int,
    context_rows: int,
) -> tuple[list[EventProposal], tuple[str, ...]]:
    rows = len(keys)
    if numeric.shape[0] != rows or any(len(value) != rows for value in (confidence, proposal_mask, anchor)):
        raise ValueError("unaligned proposal inputs")
    index = {name: offset for offset, name in enumerate(numeric_names)}
    if missing := set(selected_numeric).difference(index):
        raise ValueError(f"missing numeric features: {sorted(missing)}")
    feature_indices = [index[name] for name in selected_numeric]
    feature_names = ["duration_rows", "score_mean", "score_max", "score_min", "score_q90", "anchor_fraction", "anchor_rows"]
    for name in selected_numeric:
        feature_names.extend([f"{name}__mean", f"{name}__std", f"{name}__median", f"{name}__max_abs", f"{name}__inside_minus_context", f"{name}__after_minus_before"])

    frame = keys.loc[:, ["station", "year", "layer", "time"]].reset_index(drop=True).copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, format="mixed")
    frame["row_id"] = np.arange(rows, dtype=np.int64)
    proposals: list[EventProposal] = []
    for (station, year, layer), part in frame.groupby(["station", "year", "layer"], sort=False, observed=True):
        row_ids = part["row_id"].to_numpy(dtype=np.int64)
        times = part["time"].to_numpy()
        breaks = np.r_[True, np.diff(times) != CADENCE]
        segment_ids = np.cumsum(breaks)
        for segment_id in np.unique(segment_ids):
            segment_rows = row_ids[segment_ids == segment_id]
            active = proposal_mask[segment_rows].astype(bool)
            cursor = 0
            while cursor < len(active):
                if not active[cursor]:
                    cursor += 1
                    continue
                end = cursor + 1
                while end < len(active) and active[end]:
                    end += 1
                component = segment_rows[cursor:end]
                if len(component) >= minimum_rows:
                    payload = f"{station}|{year}|{layer}|{frame.iloc[component[0]]['time']}|{frame.iloc[component[-1]]['time']}"
                    proposals.append(
                        EventProposal(
                            hashlib.sha256(payload.encode()).hexdigest()[:20],
                            str(station),
                            int(year),
                            int(layer),
                            int(component[0]),
                            int(component[-1]) + 1,
                            pd.Timestamp(frame.iloc[component[0]]["time"]),
                            pd.Timestamp(frame.iloc[component[-1]]["time"]),
                            component,
                            _component_features(numeric, confidence, anchor, component, segment_rows, feature_indices, context_rows),
                        )
                    )
                cursor = end
    proposals.sort(key=lambda item: (item.end_time, item.station, item.layer, item.proposal_id))
    return proposals, tuple(feature_names)


def assign_split(proposal: EventProposal, intervals: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]) -> str | None:
    for name, (start, end) in intervals.items():
        if (start is None or proposal.start_time >= start) and (end is None or proposal.end_time < end):
            return name
    return None


def utility_targets(proposals: Sequence[EventProposal], truth: np.ndarray, anchor: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    base = binary_metrics(truth[rows], anchor[rows])
    f1 = float(base["f1"])
    targets = np.zeros(len(proposals), dtype=np.int8)
    diagnostics: list[dict[str, float | int]] = []
    row_membership = np.zeros(len(truth), dtype=bool)
    row_membership[rows] = True
    for index, proposal in enumerate(proposals):
        eligible = proposal.row_ids[row_membership[proposal.row_ids] & (anchor[proposal.row_ids] == 0)]
        tp = int(np.sum(truth[eligible] == 1))
        fp = int(np.sum(truth[eligible] == 0))
        utility = (2.0 - f1) * tp - f1 * fp
        targets[index] = int(utility > 0.0)
        diagnostics.append({"added_tp": tp, "added_fp": fp, "utility": float(utility)})
    return targets, diagnostics


def decode_additions(rows: int, proposals: Sequence[EventProposal], selected: np.ndarray) -> np.ndarray:
    output = np.zeros(rows, dtype=np.int8)
    for proposal, keep in zip(proposals, selected, strict=True):
        if keep:
            output[proposal.row_ids] = 1
    return output


def evaluate_union(truth: np.ndarray, anchor: np.ndarray, additions: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    candidate = np.maximum(anchor, additions).astype(np.int8)
    base = binary_metrics(truth[rows], anchor[rows])
    new = binary_metrics(truth[rows], candidate[rows])
    changed = (anchor[rows] == 0) & (candidate[rows] == 1)
    local_truth = truth[rows]
    added_tp = int(np.sum(changed & (local_truth == 1)))
    added_fp = int(np.sum(changed & (local_truth == 0)))
    normal_days = max(1.0, float(np.sum(local_truth == 0)) / 144.0)
    anchor_fp_day = float(base["fp"]) / normal_days
    candidate_fp_day = float(new["fp"]) / normal_days
    fp_ratio = candidate_fp_day / anchor_fp_day if anchor_fp_day > 0 else (1.0 if candidate_fp_day == 0 else float("inf"))
    return {
        "anchor": base,
        "candidate": new,
        "delta_f1": float(new["f1"] - base["f1"]),
        "delta_recall": float(new["recall"] - base["recall"]),
        "added_rows": int(changed.sum()),
        "added_tp": added_tp,
        "added_fp": added_fp,
        "added_precision": added_tp / (added_tp + added_fp) if added_tp + added_fp else 1.0,
        "anchor_f1_over_2": float(base["f1"]) / 2.0,
        "anchor_fp_per_day": anchor_fp_day,
        "candidate_fp_per_day": candidate_fp_day,
        "fp_per_day_relative": fp_ratio,
        "anchor_positive_removed_rows": int(np.sum((anchor[rows] == 1) & (candidate[rows] == 0))),
    }
