"""Uncertainty and promotion diagnostics for blocked P1 predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from p1_qc.metrics import binary_counts


def _positive_event_ids(
    truth: np.ndarray,
    metadata: pd.DataFrame,
    *,
    cadence_minutes: int = 10,
) -> np.ndarray:
    working = metadata.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    working["position"] = np.arange(len(working), dtype=np.int64)
    working["truth"] = truth.astype(bool)
    working["parsed_time"] = pd.to_datetime(working["time"], errors="raise", utc=True)
    ordered = working.sort_values(["station", "layer", "parsed_time", "position"], kind="mergesort")
    grouped = ordered.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["parsed_time"].diff().dt.total_seconds().eq(cadence_minutes * 60)
    previous = grouped["truth"].shift(1).fillna(False).astype(bool)
    start = ordered["truth"] & (~contiguous | ~previous)
    ordered["event"] = start.cumsum().where(ordered["truth"], -1).astype(np.int64)
    return ordered.sort_values("position", kind="mergesort")["event"].to_numpy()


def paired_block_bootstrap(
    truth: Sequence[int],
    candidate: Sequence[int],
    baseline: Sequence[int],
    metadata: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 20260813,
    cadence_minutes: int = 10,
    normal_day_timezone: str = "UTC",
) -> dict[str, Any]:
    """Paired bootstrap over positive events and normal station-layer days."""

    y = np.asarray(truth, dtype=np.int8)
    candidate_array = np.asarray(candidate, dtype=np.int8)
    baseline_array = np.asarray(baseline, dtype=np.int8)
    if y.shape != candidate_array.shape or y.shape != baseline_array.shape or y.ndim != 1:
        raise ValueError("truth and predictions must be equal-length vectors")
    if len(metadata) != len(y):
        raise ValueError("metadata length differs from predictions")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    event = _positive_event_ids(y, metadata, cadence_minutes=cadence_minutes)
    parsed = pd.to_datetime(metadata["time"], errors="raise", utc=True)
    normal_day = parsed.dt.tz_convert(normal_day_timezone).dt.strftime("%Y-%m-%d")
    normal_key = pd.MultiIndex.from_arrays(
        [
            metadata["station"].astype(str),
            metadata["layer"].astype(str),
            normal_day,
        ]
    )
    normal_code = pd.factorize(normal_key, sort=True)[0]
    block_key = np.empty(len(y), dtype=object)
    positive = event >= 0
    block_key[positive] = [f"event:{value}" for value in event[positive]]
    block_key[~positive] = [f"normal:{value}" for value in normal_code[~positive]]
    codes, uniques = pd.factorize(block_key, sort=True)
    block_count = len(uniques)

    def confusion_by_block(prediction: np.ndarray) -> np.ndarray:
        values = np.empty((block_count, 3), dtype=np.float64)
        for block in range(block_count):
            mask = codes == block
            counts = binary_counts(y[mask], prediction[mask])
            values[block] = (counts.tp, counts.fp, counts.fn)
        return values

    candidate_confusion = confusion_by_block(candidate_array)
    baseline_confusion = confusion_by_block(baseline_array)
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, block_count, size=block_count)
        candidate_total = candidate_confusion[sampled].sum(axis=0)
        baseline_total = baseline_confusion[sampled].sum(axis=0)

        def f1(values: np.ndarray) -> float:
            tp, fp, fn = values
            denominator = 2 * tp + fp + fn
            return float(2 * tp / denominator) if denominator else 0.0

        differences[replicate] = f1(candidate_total) - f1(baseline_total)
    quantiles = np.quantile(differences, [0.05, 0.5, 0.95])
    return {
        "replicates": replicates,
        "blocks": block_count,
        "positive_event_blocks": int(len(np.unique(event[event >= 0]))),
        "normal_day_blocks": int(block_count - len(np.unique(event[event >= 0]))),
        "normal_day_timezone": normal_day_timezone,
        "difference_mean": float(differences.mean()),
        "difference_ci90": [float(quantiles[0]), float(quantiles[2])],
        "difference_median": float(quantiles[1]),
        "probability_improved": float(np.mean(differences > 0)),
    }


def normal_station_layer_day_fp(
    truth: Sequence[int],
    candidate: Sequence[int],
    baseline: Sequence[int],
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Compare false positives per normal station-layer KST day.

    This is the shared implementation used by one-shot experiment gates.  A
    row is normal only when its supplied truth is zero; metadata are used only
    to construct station-layer calendar-day blocks.
    """

    y = np.asarray(truth, dtype=np.int8)
    candidate_array = np.asarray(candidate, dtype=np.int8)
    baseline_array = np.asarray(baseline, dtype=np.int8)
    if y.ndim != 1 or y.shape != candidate_array.shape or y.shape != baseline_array.shape:
        raise ValueError("truth and predictions must be equal-length vectors")
    if len(metadata) != len(y):
        raise ValueError("metadata length differs from predictions")
    required = {"station", "layer", "time"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise KeyError(f"metadata is missing required columns: {missing}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("truth must be binary")

    parsed = pd.to_datetime(metadata["time"], errors="raise", utc=True, format="mixed")
    normal = y == 0
    work = metadata.loc[normal, ["station", "layer"]].reset_index(drop=True).copy()
    work["kst_day"] = parsed[normal].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d").to_numpy()
    block_count = len(work.drop_duplicates(["station", "layer", "kst_day"]))

    def summarize(prediction: np.ndarray) -> dict[str, Any]:
        false_positive = prediction[normal] == 1
        rows = int(false_positive.sum())
        days = int(
            work.loc[false_positive, ["station", "layer", "kst_day"]].drop_duplicates().shape[0]
        )
        return {
            "false_positive_rows": rows,
            "false_positive_rows_per_normal_station_layer_day": (
                rows / block_count if block_count else None
            ),
            "normal_station_layer_days_with_fp": days,
            "normal_station_layer_day_fp_incidence": (days / block_count if block_count else None),
        }

    candidate_summary = summarize(candidate_array)
    baseline_summary = summarize(baseline_array)
    return {
        "timezone": "Asia/Seoul",
        "normal_rows": int(normal.sum()),
        "normal_station_layer_days": block_count,
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "delta_false_positive_rows_per_normal_station_layer_day": (
            candidate_summary["false_positive_rows_per_normal_station_layer_day"]
            - baseline_summary["false_positive_rows_per_normal_station_layer_day"]
            if block_count
            else None
        ),
    }
