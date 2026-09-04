"""Event-level utilities for the frozen 83-proposal P1 ranker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from p1_qc.p1_conditional_real_event_donor_20260828_v1 import (
    EventSpan,
    binary_metrics,
)


@dataclass(frozen=True)
class ProposalSplit:
    train: np.ndarray
    calibration: np.ndarray
    qualification: np.ndarray
    purged: np.ndarray
    first_boundary: pd.Timestamp
    second_boundary: pd.Timestamp


def event_iou(left: EventSpan, right: EventSpan) -> float:
    """Return row-index IoU for events from the same station and layer."""
    if (left.station, left.layer) != (right.station, right.layer):
        return 0.0
    intersection = len(np.intersect1d(left.rows, right.rows, assume_unique=True))
    union = len(left.rows) + len(right.rows) - intersection
    return intersection / union if union else 0.0


def proposal_truth_matrix(
    proposals: Sequence[EventSpan],
    truth_events: Sequence[EventSpan],
    *,
    iou_threshold: float,
) -> np.ndarray:
    output = np.zeros((len(proposals), len(truth_events)), dtype=bool)
    for proposal_index, proposal in enumerate(proposals):
        for truth_index, truth in enumerate(truth_events):
            output[proposal_index, truth_index] = (
                event_iou(proposal, truth) >= iou_threshold
            )
    return output


def chronological_split(
    proposals: Sequence[EventSpan],
    *,
    first_fraction: float,
    second_cumulative_fraction: float,
    purge_days: int,
) -> ProposalSplit:
    """Split ordered proposals and embargo the starts of later partitions."""
    if not 0.0 < first_fraction < second_cumulative_fraction < 1.0:
        raise ValueError("invalid split fractions")
    if len(proposals) < 3:
        raise ValueError("at least three proposals are required")
    order = np.asarray(
        sorted(range(len(proposals)), key=lambda index: proposals[index].end_time),
        dtype=np.int64,
    )
    first_index = max(1, int(np.floor(len(order) * first_fraction))) - 1
    second_index = max(first_index + 1, int(np.floor(len(order) * second_cumulative_fraction))) - 1
    second_index = min(second_index, len(order) - 2)
    first_boundary = proposals[int(order[first_index])].end_time
    second_boundary = proposals[int(order[second_index])].end_time
    embargo = pd.Timedelta(days=purge_days)
    train: list[int] = []
    calibration: list[int] = []
    qualification: list[int] = []
    purged: list[int] = []
    for index in order:
        proposal = proposals[int(index)]
        if proposal.end_time <= first_boundary:
            train.append(int(index))
        elif proposal.end_time <= second_boundary:
            if proposal.start_time >= first_boundary + embargo:
                calibration.append(int(index))
            else:
                purged.append(int(index))
        elif proposal.start_time >= second_boundary + embargo:
            qualification.append(int(index))
        else:
            purged.append(int(index))
    return ProposalSplit(
        train=np.asarray(train, dtype=np.int64),
        calibration=np.asarray(calibration, dtype=np.int64),
        qualification=np.asarray(qualification, dtype=np.int64),
        purged=np.asarray(purged, dtype=np.int64),
        first_boundary=first_boundary,
        second_boundary=second_boundary,
    )


def remove_cross_split_truth_matches(
    split: ProposalSplit,
    matches: np.ndarray,
) -> tuple[ProposalSplit, np.ndarray]:
    """Purge later proposals when one truth event would occur in multiple splits."""
    arrays = [split.train.copy(), split.calibration.copy(), split.qualification.copy()]
    removed: set[int] = set()
    for truth_index in range(matches.shape[1]):
        present = [
            position
            for position, indices in enumerate(arrays)
            if np.any(matches[indices, truth_index])
        ]
        if len(present) <= 1:
            continue
        earliest = min(present)
        for position in present:
            if position == earliest:
                continue
            affected = arrays[position][matches[arrays[position], truth_index]]
            removed.update(int(index) for index in affected)
    if removed:
        for position in range(len(arrays)):
            arrays[position] = np.asarray(
                [index for index in arrays[position] if int(index) not in removed],
                dtype=np.int64,
            )
    purged = np.unique(
        np.concatenate(
            [split.purged, np.asarray(sorted(removed), dtype=np.int64)]
        )
    )
    return (
        ProposalSplit(
            train=arrays[0],
            calibration=arrays[1],
            qualification=arrays[2],
            purged=purged,
            first_boundary=split.first_boundary,
            second_boundary=split.second_boundary,
        ),
        np.asarray(sorted(removed), dtype=np.int64),
    )


def _safe_summary(values: np.ndarray, operation: str) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan
    if operation == "median":
        return float(np.median(finite))
    if operation == "absolute_q90":
        return float(np.quantile(np.abs(finite), 0.9))
    raise ValueError(operation)


def _context_rows(
    keys: pd.DataFrame,
    proposal: EventSpan,
    *,
    context_rows_each_side: int,
) -> np.ndarray:
    times = pd.to_datetime(keys["time"], utc=True, format="mixed")
    same_cell = (keys["station"].astype(str) == proposal.station) & (
        keys["layer"].astype(int) == proposal.layer
    )
    cadence = pd.Timedelta(minutes=10)
    width = cadence * context_rows_each_side
    before = same_cell & (times >= proposal.start_time - width) & (
        times < proposal.start_time
    )
    after = same_cell & (times > proposal.end_time) & (times <= proposal.end_time + width)
    return np.flatnonzero((before | after).to_numpy())


def _reference_values(
    keys: pd.DataFrame,
    feature_values: np.ndarray,
    reference_rows: np.ndarray,
    proposal: EventSpan,
    feature_index: int,
    *,
    minimum_group_rows: int,
) -> tuple[np.ndarray, bool]:
    reference = keys.iloc[reference_rows]
    quarter = int(proposal.start_time.quarter)
    ref_time = pd.to_datetime(reference["time"], utc=True, format="mixed")
    quarter_mask = ref_time.dt.quarter.to_numpy() == quarter
    station_mask = reference["station"].astype(str).to_numpy() == proposal.station
    layer_mask = reference["layer"].astype(int).to_numpy() == proposal.layer
    candidates = (
        station_mask & layer_mask & quarter_mask,
        station_mask & quarter_mask,
        np.ones(len(reference_rows), dtype=bool),
    )
    for level, mask in enumerate(candidates):
        values = feature_values[reference_rows[mask], feature_index]
        values = values[np.isfinite(values)]
        if len(values) >= minimum_group_rows or level == len(candidates) - 1:
            return values, level == 0
    raise AssertionError("unreachable")


def proposal_feature_matrix(
    keys: pd.DataFrame,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    proposals: Sequence[EventSpan],
    proposal_scores: np.ndarray,
    reference_rows: np.ndarray,
    *,
    frozen_threshold: float,
    context_rows_each_side: int,
    minimum_normality_group_rows: int = 100,
) -> tuple[np.ndarray, list[str], dict[str, float | int]]:
    """Create fixed event, context, physical, score, and normality aggregates."""
    if proposal_scores.shape != (len(keys),):
        raise ValueError("proposal scores differ from keys")
    if feature_values.shape != (len(keys), len(feature_names)):
        raise ValueError("feature matrix differs from keys or names")
    names = ["duration_rows"]
    names.extend(
        [
            "score_mean",
            "score_minimum",
            "score_maximum",
            "score_q10",
            "score_q50",
            "score_q90",
            "score_fraction_gte_threshold",
            "score_center_edge_contrast",
        ]
    )
    for name in feature_names:
        names.extend(
            [
                f"{name}__median",
                f"{name}__absolute_q90",
                f"{name}__absolute_event_minus_context_median",
                f"{name}__normality_robust_distance",
            ]
        )
    rows: list[list[float]] = []
    primary_support = 0
    support_trials = 0
    for proposal in proposals:
        score = np.asarray(proposal_scores[proposal.rows], dtype=np.float64)
        finite_score = score[np.isfinite(score)]
        if len(finite_score):
            edge_width = max(1, len(finite_score) // 4)
            edge = np.r_[finite_score[:edge_width], finite_score[-edge_width:]]
            center = finite_score[edge_width:-edge_width]
            center_edge = (
                float(np.mean(center) - np.mean(edge)) if len(center) else 0.0
            )
            score_features = [
                float(np.mean(finite_score)),
                float(np.min(finite_score)),
                float(np.max(finite_score)),
                float(np.quantile(finite_score, 0.1)),
                float(np.quantile(finite_score, 0.5)),
                float(np.quantile(finite_score, 0.9)),
                float(np.mean(finite_score >= frozen_threshold)),
                center_edge,
            ]
        else:
            score_features = [np.nan] * 8
        output = [float(len(proposal.rows)), *score_features]
        context = _context_rows(
            keys, proposal, context_rows_each_side=context_rows_each_side
        )
        for feature_index in range(len(feature_names)):
            event_values = feature_values[proposal.rows, feature_index]
            context_values = feature_values[context, feature_index]
            event_median = _safe_summary(event_values, "median")
            context_median = _safe_summary(context_values, "median")
            reference, primary = _reference_values(
                keys,
                feature_values,
                reference_rows,
                proposal,
                feature_index,
                minimum_group_rows=minimum_normality_group_rows,
            )
            support_trials += 1
            primary_support += int(primary)
            reference_median = float(np.median(reference)) if len(reference) else np.nan
            reference_mad = (
                float(np.median(np.abs(reference - reference_median)))
                if len(reference)
                else np.nan
            )
            scale = max(1e-6, 1.4826 * reference_mad) if np.isfinite(reference_mad) else np.nan
            output.extend(
                [
                    event_median,
                    _safe_summary(event_values, "absolute_q90"),
                    abs(event_median - context_median)
                    if np.isfinite(event_median) and np.isfinite(context_median)
                    else np.nan,
                    abs(event_median - reference_median) / scale
                    if np.isfinite(event_median)
                    and np.isfinite(reference_median)
                    and np.isfinite(scale)
                    else np.nan,
                ]
            )
        rows.append(output)
    matrix = np.asarray(rows, dtype=np.float64)
    return matrix, names, {
        "primary_reference_trials": support_trials,
        "primary_reference_supported": primary_support,
        "primary_reference_coverage": primary_support / support_trials
        if support_trials
        else 0.0,
    }


def select_recall_threshold(
    scores: Sequence[float],
    matches: np.ndarray,
    *,
    minimum_retention: float,
) -> tuple[float, dict[str, float | int]]:
    """Choose the largest observed score satisfying matched-event retention."""
    values = np.asarray(scores, dtype=np.float64)
    if matches.shape[0] != len(values):
        raise ValueError("match rows differ from scores")
    raw_matched = int(np.sum(np.any(matches, axis=0)))
    if raw_matched == 0:
        raise ValueError("raw proposal bank matches no truth events")
    required = int(np.ceil(minimum_retention * raw_matched))
    for threshold in np.unique(values)[::-1]:
        selected = values >= threshold
        retained = int(np.sum(np.any(matches[selected], axis=0)))
        if retained >= required:
            return float(threshold), {
                "raw_matched_truth_events": raw_matched,
                "required_matched_truth_events": required,
                "retained_matched_truth_events": retained,
                "selected_proposals": int(selected.sum()),
                "retention": retained / raw_matched,
            }
    raise AssertionError("lowest observed threshold must retain every proposal")


def mask_from_proposals(
    row_count: int, proposals: Sequence[EventSpan], selected: Sequence[bool] | None = None
) -> np.ndarray:
    mask = np.zeros(row_count, dtype=np.int8)
    chosen = (
        np.ones(len(proposals), dtype=bool)
        if selected is None
        else np.asarray(selected, dtype=bool)
    )
    for keep, proposal in zip(chosen, proposals, strict=True):
        if keep:
            mask[proposal.rows] = 1
    return mask


def proposal_partition_metrics(
    truth: Sequence[int],
    truth_events: Sequence[EventSpan],
    proposals: Sequence[EventSpan],
    selected: Sequence[bool],
    *,
    iou_threshold: float,
    evaluation_rows: Sequence[int] | None = None,
) -> dict[str, object]:
    chosen = np.asarray(selected, dtype=bool)
    matches = proposal_truth_matrix(
        proposals, truth_events, iou_threshold=iou_threshold
    )
    raw_mask = mask_from_proposals(len(truth), proposals)
    selected_mask = mask_from_proposals(len(truth), proposals, chosen)
    evaluated = (
        np.arange(len(truth), dtype=np.int64)
        if evaluation_rows is None
        else np.asarray(evaluation_rows, dtype=np.int64)
    )
    raw_matched = int(np.sum(np.any(matches, axis=0)))
    selected_matched = int(np.sum(np.any(matches[chosen], axis=0)))
    raw_false = int(np.sum(~np.any(matches, axis=1)))
    selected_false = int(np.sum(chosen & ~np.any(matches, axis=1)))
    raw_cells = {
        (proposal.station, proposal.layer)
        for proposal, matched in zip(proposals, np.any(matches, axis=1), strict=True)
        if matched
    }
    selected_cells = {
        (proposal.station, proposal.layer)
        for proposal, keep, matched in zip(
            proposals, chosen, np.any(matches, axis=1), strict=True
        )
        if keep and matched
    }
    eligible_rows = np.zeros(len(truth), dtype=bool)
    for event in truth_events:
        eligible_rows[event.rows] = True
    raw_eligible_recall = (
        float(np.mean(raw_mask[eligible_rows] == 1)) if np.any(eligible_rows) else 0.0
    )
    selected_eligible_recall = (
        float(np.mean(selected_mask[eligible_rows] == 1))
        if np.any(eligible_rows)
        else 0.0
    )
    return {
        "raw": {
            "proposals": len(proposals),
            "false_proposals": raw_false,
            "matched_truth_events": raw_matched,
            "matched_cells": len(raw_cells),
            "row_metrics": binary_metrics(
                np.asarray(truth)[evaluated], raw_mask[evaluated]
            ),
            "eligible_truth_row_recall": raw_eligible_recall,
        },
        "selected": {
            "proposals": int(chosen.sum()),
            "false_proposals": selected_false,
            "matched_truth_events": selected_matched,
            "matched_cells": len(selected_cells),
            "row_metrics": binary_metrics(
                np.asarray(truth)[evaluated], selected_mask[evaluated]
            ),
            "eligible_truth_row_recall": selected_eligible_recall,
        },
        "matched_event_retention": selected_matched / raw_matched
        if raw_matched
        else 0.0,
        "false_proposal_reduction": (raw_false - selected_false) / raw_false
        if raw_false
        else 0.0,
        "eligible_truth_row_recall_relative": selected_eligible_recall
        / raw_eligible_recall
        if raw_eligible_recall
        else 0.0,
        "raw_mask": raw_mask,
        "selected_mask": selected_mask,
    }


def moving_block_bootstrap_delta_f1(
    keys: pd.DataFrame,
    truth: Sequence[int],
    baseline: Sequence[int],
    candidate: Sequence[int],
    *,
    replicates: int,
    block_days: int,
    seed: int,
) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    base = np.asarray(baseline, dtype=np.int8)
    chosen = np.asarray(candidate, dtype=np.int8)
    days = (
        pd.to_datetime(keys["time"], utc=True, format="mixed")
        .dt.tz_convert("Asia/Seoul")
        .dt.floor("D")
    )
    unique_days = pd.DatetimeIndex(days.unique()).sort_values()
    day_rows = [np.flatnonzero((days == day).to_numpy()) for day in unique_days]
    rng = np.random.default_rng(seed)
    needed = int(np.ceil(len(unique_days) / block_days))
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        starts = rng.integers(0, len(unique_days), size=needed)
        sampled = [
            int((start + offset) % len(unique_days))
            for start in starts
            for offset in range(block_days)
        ][: len(unique_days)]
        rows = np.concatenate([day_rows[index] for index in sampled])
        deltas[replicate] = float(binary_metrics(y[rows], chosen[rows])["f1"]) - float(
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
