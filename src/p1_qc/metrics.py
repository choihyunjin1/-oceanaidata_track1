"""Official row-level F1 plus weighted, group, and event diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import ANOMALY_TYPES, parse_anomaly_types


@dataclass(frozen=True)
class BinaryCounts:
    tp: float
    fp: float
    fn: float
    tn: float

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0

    @property
    def support(self) -> float:
        return self.tp + self.fn

    @property
    def predicted_positive(self) -> float:
        return self.tp + self.fp

    def to_dict(self) -> dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "predicted_positive": self.predicted_positive,
        }


@dataclass(frozen=True)
class EventReport:
    true_events: int
    detected_true_events: int
    predicted_events: int
    matched_predicted_events: int
    precision: float
    recall: float
    f1: float
    mean_best_iou: float
    median_detection_delay_rows: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    micro: BinaryCounts
    weighted: BinaryCounts
    groups: pd.DataFrame
    events: EventReport
    type_recall: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro": self.micro.to_dict(),
            "weighted": self.weighted.to_dict(),
            "groups": self.groups.to_dict(orient="records"),
            "events": self.events.to_dict(),
            "type_recall": dict(self.type_recall),
        }


def _binary_arrays(y_true: Sequence, y_pred: Sequence) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if len(truth) != len(prediction):
        raise ValueError("y_true and y_pred lengths differ")
    try:
        truth_float = truth.astype(float)
        prediction_float = prediction.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be numeric 0/1") from exc
    if not np.isfinite(truth_float).all() or not np.isfinite(prediction_float).all():
        raise ValueError("labels must be finite")
    if not np.isin(truth_float, [0, 1]).all() or not np.isin(prediction_float, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    return truth_float.astype(np.int8), prediction_float.astype(np.int8)


def binary_counts(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    sample_weight: Sequence[float] | None = None,
) -> BinaryCounts:
    truth, prediction = _binary_arrays(y_true, y_pred)
    if sample_weight is None:
        weight = np.ones(len(truth), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float)
        if weight.shape != truth.shape:
            raise ValueError("sample_weight shape differs from labels")
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("sample_weight must be finite and non-negative")
    tp = float(weight[(truth == 1) & (prediction == 1)].sum())
    fp = float(weight[(truth == 0) & (prediction == 1)].sum())
    fn = float(weight[(truth == 1) & (prediction == 0)].sum())
    tn = float(weight[(truth == 0) & (prediction == 0)].sum())
    return BinaryCounts(tp, fp, fn, tn)


def micro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """The exact row-level binary F1 used by the organiser scorer."""

    return binary_counts(y_true, y_pred).f1


def binary_report(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    return binary_counts(y_true, y_pred).to_dict()


def group_report(
    y_true: Sequence,
    y_pred: Sequence,
    metadata: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("station", "layer"),
) -> pd.DataFrame:
    truth, prediction = _binary_arrays(y_true, y_pred)
    if len(metadata) != len(truth):
        raise ValueError("metadata length differs from labels")
    missing = sorted(set(group_columns).difference(metadata.columns))
    if missing:
        raise KeyError(f"missing group columns: {missing}")
    working = metadata.loc[:, list(group_columns)].reset_index(drop=True).copy()
    working["y_true"] = truth
    working["y_pred"] = prediction
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = list(group_columns)
    if len(group_columns) == 1:
        grouper = group_columns[0]
    for key, part in working.groupby(grouper, sort=True, observed=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        counts = binary_counts(part["y_true"], part["y_pred"])
        row = {column: value for column, value in zip(group_columns, key_tuple, strict=True)}
        row.update(
            {
                "rows": len(part),
                "positive_rows": int(part["y_true"].sum()),
                "predicted_positive_rows": int(part["y_pred"].sum()),
                **counts.to_dict(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def group_row_shares(
    reference: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("station", "layer"),
) -> dict[tuple, float]:
    """Return target row-share weights, normally calculated from test keys."""

    if not len(reference):
        raise ValueError("reference frame is empty")
    counts = reference.groupby(list(group_columns), observed=True, dropna=False).size()
    return {
        tuple(key if isinstance(key, tuple) else (key,)): value / len(reference)
        for key, value in counts.items()
    }


def weighted_group_counts(
    y_true: Sequence,
    y_pred: Sequence,
    metadata: pd.DataFrame,
    group_weights: Mapping[Any, float],
    *,
    group_columns: Sequence[str] = ("station", "layer"),
    strict_groups: bool = False,
) -> BinaryCounts:
    """Reweight within-group confusion rates to target group row shares."""

    truth, prediction = _binary_arrays(y_true, y_pred)
    if len(metadata) != len(truth):
        raise ValueError("metadata length differs from labels")
    keys = [
        tuple(values)
        for values in metadata.loc[:, list(group_columns)].itertuples(index=False, name=None)
    ]
    source_counts: dict[tuple, int] = {}
    for key in keys:
        source_counts[key] = source_counts.get(key, 0) + 1
    normalised_weights: dict[tuple, float] = {}
    for raw_key, value in group_weights.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        normalised_weights[tuple(key)] = float(value)
    missing = sorted(set(source_counts).difference(normalised_weights), key=str)
    if missing and strict_groups:
        raise KeyError(f"group weights are missing observed groups: {missing}")
    for key in missing:
        # A group absent from the target/test support has target row share 0.
        normalised_weights[key] = 0.0
    total_target_weight = sum(normalised_weights[key] for key in source_counts)
    if total_target_weight <= 0:
        raise ValueError("group weights must have positive mass")
    row_weight = np.asarray(
        [normalised_weights[key] / total_target_weight / source_counts[key] for key in keys],
        dtype=float,
    )
    return binary_counts(truth, prediction, sample_weight=row_weight)


def weighted_group_f1(
    y_true: Sequence,
    y_pred: Sequence,
    metadata: pd.DataFrame,
    group_weights: Mapping[Any, float],
    *,
    group_columns: Sequence[str] = ("station", "layer"),
    strict_groups: bool = False,
) -> float:
    return weighted_group_counts(
        y_true,
        y_pred,
        metadata,
        group_weights,
        group_columns=group_columns,
        strict_groups=strict_groups,
    ).f1


def _run_ids(
    labels: np.ndarray,
    metadata: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    time_column: str,
    cadence_minutes: int,
) -> np.ndarray:
    n = len(labels)
    working = metadata.loc[:, [*group_columns, time_column]].reset_index(drop=True).copy()
    working["__position"] = np.arange(n, dtype=np.int64)
    working["__label"] = labels.astype(bool)
    working["__time"] = pd.to_datetime(working[time_column], errors="coerce", utc=True)
    if working["__time"].isna().any():
        raise ValueError("event timestamps could not be parsed")
    ordered = working.sort_values([*group_columns, "__time", "__position"], kind="mergesort")
    grouped = ordered.groupby(list(group_columns), sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(cadence_minutes * 60)
    prior = grouped["__label"].shift(1).fillna(False).astype(bool)
    starts = ordered["__label"] & (~contiguous | ~prior)
    ordered["__run_id"] = starts.cumsum().where(ordered["__label"], -1).astype(np.int64)
    return ordered.sort_values("__position", kind="mergesort")["__run_id"].to_numpy()


def event_report(
    y_true: Sequence,
    y_pred: Sequence,
    metadata: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("station", "layer"),
    time_column: str = "time",
    cadence_minutes: int = 10,
    min_iou: float = 0.0,
) -> EventReport:
    """Measure event overlap while respecting groups and time gaps."""

    truth, prediction = _binary_arrays(y_true, y_pred)
    if len(metadata) != len(truth):
        raise ValueError("metadata length differs from labels")
    if not 0 <= min_iou <= 1:
        raise ValueError("min_iou must be in [0, 1]")
    required = set(group_columns) | {time_column}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise KeyError(f"missing event metadata columns: {missing}")
    true_run = _run_ids(
        truth,
        metadata,
        group_columns=group_columns,
        time_column=time_column,
        cadence_minutes=cadence_minutes,
    )
    pred_run = _run_ids(
        prediction,
        metadata,
        group_columns=group_columns,
        time_column=time_column,
        cadence_minutes=cadence_minutes,
    )
    true_ids = np.unique(true_run[true_run >= 0])
    pred_ids = np.unique(pred_run[pred_run >= 0])
    true_sizes = {run: int((true_run == run).sum()) for run in true_ids}
    pred_sizes = {run: int((pred_run == run).sum()) for run in pred_ids}

    detected = 0
    best_ious: list[float] = []
    delays: list[int] = []
    for true_id in true_ids:
        true_positions = np.flatnonzero(true_run == true_id)
        candidates = np.unique(pred_run[true_positions])
        candidates = candidates[candidates >= 0]
        best_iou = 0.0
        for pred_id in candidates:
            intersection = int(((true_run == true_id) & (pred_run == pred_id)).sum())
            union = true_sizes[true_id] + pred_sizes[pred_id] - intersection
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou = iou
        qualifies = best_iou > 0 if min_iou == 0 else best_iou >= min_iou
        if qualifies:
            detected += 1
            best_ious.append(best_iou)
            overlap_positions = true_positions[prediction[true_positions] == 1]
            if len(overlap_positions):
                delays.append(int(overlap_positions.min() - true_positions.min()))

    matched_predicted = 0
    for pred_id in pred_ids:
        pred_positions = np.flatnonzero(pred_run == pred_id)
        candidates = np.unique(true_run[pred_positions])
        candidates = candidates[candidates >= 0]
        best_iou = 0.0
        for true_id in candidates:
            intersection = int(((true_run == true_id) & (pred_run == pred_id)).sum())
            union = true_sizes[true_id] + pred_sizes[pred_id] - intersection
            best_iou = max(best_iou, intersection / union if union else 0.0)
        if best_iou > 0 if min_iou == 0 else best_iou >= min_iou:
            matched_predicted += 1

    event_precision = matched_predicted / len(pred_ids) if len(pred_ids) else 0.0
    event_recall = detected / len(true_ids) if len(true_ids) else 0.0
    denominator = event_precision + event_recall
    event_f1 = 2 * event_precision * event_recall / denominator if denominator else 0.0
    return EventReport(
        true_events=len(true_ids),
        detected_true_events=detected,
        predicted_events=len(pred_ids),
        matched_predicted_events=matched_predicted,
        precision=event_precision,
        recall=event_recall,
        f1=event_f1,
        mean_best_iou=float(np.mean(best_ious)) if best_ious else 0.0,
        median_detection_delay_rows=float(np.median(delays)) if delays else float("nan"),
    )


def anomaly_type_recall(
    y_pred: Sequence,
    anomaly_type: pd.Series,
    *,
    known_types: Sequence[str] = ANOMALY_TYPES,
) -> dict[str, float]:
    prediction = np.asarray(y_pred)
    if len(prediction) != len(anomaly_type):
        raise ValueError("anomaly_type length differs from predictions")
    membership = parse_anomaly_types(anomaly_type, known_types=known_types)
    result: dict[str, float] = {}
    for anomaly in known_types:
        mask = membership[anomaly].to_numpy()
        result[anomaly] = float(np.mean(prediction[mask] == 1)) if mask.any() else float("nan")
    return result


def evaluate_predictions(
    y_true: Sequence,
    y_pred: Sequence,
    metadata: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("station", "layer"),
    group_weights: Mapping[Any, float] | None = None,
    anomaly_type: pd.Series | None = None,
    cadence_minutes: int = 10,
    event_min_iou: float = 0.0,
) -> EvaluationReport:
    micro = binary_counts(y_true, y_pred)
    if group_weights is None:
        weighted = micro
    else:
        weighted = weighted_group_counts(
            y_true,
            y_pred,
            metadata,
            group_weights,
            group_columns=group_columns,
        )
    groups = group_report(y_true, y_pred, metadata, group_columns=group_columns)
    events = event_report(
        y_true,
        y_pred,
        metadata,
        group_columns=group_columns,
        cadence_minutes=cadence_minutes,
        min_iou=event_min_iou,
    )
    type_recall = anomaly_type_recall(y_pred, anomaly_type) if anomaly_type is not None else {}
    return EvaluationReport(micro, weighted, groups, events, type_recall)


__all__ = [
    "BinaryCounts",
    "EvaluationReport",
    "EventReport",
    "anomaly_type_recall",
    "binary_counts",
    "binary_report",
    "evaluate_predictions",
    "event_report",
    "group_report",
    "group_row_shares",
    "micro_f1",
    "weighted_group_counts",
    "weighted_group_f1",
]
