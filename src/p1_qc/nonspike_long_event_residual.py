"""Pure helpers for the preregistered P1 Round-B residual screen.

The residual is deliberately conservative: it is trained only to distinguish
normal rows from non-spike positive events lasting at least ``min_event_rows``
and may only extend an existing Round-B positive run through a contiguous chain
of eligible rows.  Consequently it cannot create a disconnected event.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResidualTrainingView:
    """Indices, targets, weights, and audit counts for one training prefix."""

    indices: np.ndarray
    target: np.ndarray
    sample_weight: np.ndarray
    positive_event_count: int
    positive_row_count: int
    excluded_positive_rows: int
    right_censored_event_count: int


def _checked_binary(values: Sequence[int], *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int8)
    if result.ndim != 1 or not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must be a one-dimensional binary vector")
    return result


def _ordered_frame(metadata: pd.DataFrame, truth: np.ndarray) -> pd.DataFrame:
    required = {"station", "layer", "time"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"metadata columns missing: {missing}")
    if len(metadata) != len(truth):
        raise ValueError("metadata and truth lengths differ")
    work = metadata.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["__truth"] = truth
    work["__time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(
        ["station", "layer", "__time", "__position"],
        kind="mergesort",
        inplace=True,
    )
    return work


def long_nonspike_event_target(
    truth: Sequence[int],
    anomaly_type: Sequence[object],
    metadata: pd.DataFrame,
    *,
    min_event_rows: int = 19,
    cadence_minutes: int = 10,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return a target for complete, non-spike, sufficiently long events.

    Event length and spike membership are derived only inside the supplied
    prefix.  A positive event touching the right edge of a station-layer
    prefix is excluded because its final duration is not yet observable.
    """

    y = _checked_binary(truth, name="truth")
    types = pd.Series(anomaly_type, dtype="string").fillna("")
    if len(types) != len(y):
        raise ValueError("anomaly_type and truth lengths differ")
    if min_event_rows < 2 or cadence_minutes < 1:
        raise ValueError("invalid event contract")

    work = _ordered_frame(metadata, y)
    work["__type"] = types.iloc[work["__position"].to_numpy()].to_numpy(dtype=str)
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(cadence_minutes * 60)
    prior_positive = grouped["__truth"].shift(1).fillna(0).eq(1)
    event_start = work["__truth"].eq(1) & (~contiguous | ~prior_positive)
    work["__event"] = event_start.cumsum().where(work["__truth"].eq(1), -1).astype(np.int64)

    positive = work["__truth"].eq(1)
    work["__event_length"] = 0
    if positive.any():
        work.loc[positive, "__event_length"] = (
            work.loc[positive]
            .groupby("__event", sort=False)["__event"]
            .transform("size")
            .to_numpy(dtype=np.int64)
        )
    token_sets = work["__type"].map(
        lambda value: {token.strip() for token in str(value).split("+") if token.strip()}
    )
    work["__spike"] = token_sets.map(lambda tokens: "spike" in tokens)
    work["__event_has_spike"] = False
    if positive.any():
        work.loc[positive, "__event_has_spike"] = (
            work.loc[positive]
            .groupby("__event", sort=False)["__spike"]
            .transform("max")
            .to_numpy(dtype=bool)
        )

    last_in_group = grouped.cumcount(ascending=False).eq(0)
    censored_event_ids = set(
        work.loc[last_in_group & positive, "__event"].to_numpy(dtype=np.int64).tolist()
    )
    work["__right_censored"] = work["__event"].isin(censored_event_ids) & positive
    selected = (
        positive
        & work["__event_length"].ge(min_event_rows)
        & ~work["__event_has_spike"]
        & ~work["__right_censored"]
    )
    selected_event_count = int(work.loc[selected, "__event"].nunique())
    work["__selected"] = selected
    restored = work.sort_values("__position", kind="mergesort")
    target = restored["__selected"].to_numpy(dtype=np.int8)
    audit = {
        "positive_event_count": selected_event_count,
        "positive_row_count": int(target.sum()),
        "excluded_positive_rows": int(y.sum() - target.sum()),
        "right_censored_event_count": len(censored_event_ids),
    }
    return target, audit


def _event_day_weights(
    metadata: pd.DataFrame,
    target: np.ndarray,
    *,
    cadence_minutes: int,
) -> np.ndarray:
    work = _ordered_frame(metadata, target)
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(cadence_minutes * 60)
    prior_positive = grouped["__truth"].shift(1).fillna(0).eq(1)
    event_start = work["__truth"].eq(1) & (~contiguous | ~prior_positive)
    work["__event"] = event_start.cumsum().where(work["__truth"].eq(1), -1).astype(np.int64)
    positive = work["__truth"].eq(1)
    normal = ~positive
    if not positive.any() or not normal.any():
        raise ValueError("residual training view must contain both classes")

    event_length = (
        work.loc[positive]
        .groupby("__event", sort=False)["__event"]
        .transform("size")
        .to_numpy(dtype=float)
    )
    positive_weight = 1.0 / np.sqrt(event_length)
    positive_weight /= positive_weight.mean()
    day = work["__time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    normal_length = (
        work.loc[normal]
        .assign(__day=day.loc[normal])
        .groupby(["station", "layer", "__day"], sort=False, observed=True)["__day"]
        .transform("size")
        .to_numpy(dtype=float)
    )
    normal_weight = 1.0 / np.sqrt(normal_length)
    normal_weight /= normal_weight.mean()
    ordered = np.empty(len(work), dtype=np.float64)
    ordered[positive.to_numpy()] = positive_weight * np.sqrt(normal.sum() / positive.sum())
    ordered[normal.to_numpy()] = normal_weight
    work["__weight"] = ordered
    result = work.sort_values("__position", kind="mergesort")["__weight"].to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(result).all() or (result <= 0).any():
        raise RuntimeError("invalid residual sample weight")
    return result


def build_residual_training_view(
    truth: Sequence[int],
    anomaly_type: Sequence[object],
    metadata: pd.DataFrame,
    *,
    min_event_rows: int = 19,
    cadence_minutes: int = 10,
) -> ResidualTrainingView:
    """Build the fixed long-event-vs-normal residual training surface."""

    y = _checked_binary(truth, name="truth")
    target, audit = long_nonspike_event_target(
        y,
        anomaly_type,
        metadata,
        min_event_rows=min_event_rows,
        cadence_minutes=cadence_minutes,
    )
    eligible = (y == 0) | (target == 1)
    indices = np.flatnonzero(eligible).astype(np.int64, copy=False)
    residual_target = target[indices]
    if len(np.unique(residual_target)) != 2:
        raise ValueError("residual training surface lacks both classes")
    weights = _event_day_weights(
        metadata.iloc[indices].reset_index(drop=True),
        residual_target,
        cadence_minutes=cadence_minutes,
    )
    return ResidualTrainingView(
        indices=indices,
        target=residual_target,
        sample_weight=weights,
        positive_event_count=audit["positive_event_count"],
        positive_row_count=audit["positive_row_count"],
        excluded_positive_rows=audit["excluded_positive_rows"],
        right_censored_event_count=audit["right_censored_event_count"],
    )


def connected_rescue(
    base_prediction: Sequence[int],
    residual_probability: Sequence[float],
    metadata: pd.DataFrame,
    spike_candidate: Sequence[bool],
    *,
    threshold: float = 0.8,
    max_distance_rows: int = 18,
    cadence_minutes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend Round-B runs through high-confidence contiguous residual rows.

    Growth starts only from a base-positive row and advances by one adjacent
    10-minute row per iteration.  This makes every rescue connected to an
    existing base run and prevents a new event or singleton from being born.
    """

    base = _checked_binary(base_prediction, name="base_prediction")
    probability = np.asarray(residual_probability, dtype=float)
    spike = np.asarray(spike_candidate, dtype=bool)
    if probability.shape != base.shape or spike.shape != base.shape:
        raise ValueError("prediction vectors must have equal shapes")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("residual_probability must be finite and within [0, 1]")
    if not 0 < threshold < 1 or max_distance_rows < 1 or cadence_minutes < 1:
        raise ValueError("invalid rescue contract")

    work = _ordered_frame(metadata, base)
    ordered_positions = work["__position"].to_numpy(dtype=np.int64)
    ordered_base = base[ordered_positions].astype(bool)
    ordered_probability = probability[ordered_positions]
    ordered_spike = spike[ordered_positions]
    eligible = (~ordered_base) & (ordered_probability >= threshold) & (~ordered_spike)
    active = ordered_base.copy()
    rescue = np.zeros(len(work), dtype=bool)
    group_codes = pd.factorize(
        pd.MultiIndex.from_frame(work.loc[:, ["station", "layer"]]), sort=False
    )[0]
    times = work["__time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    cadence_ns = int(pd.Timedelta(minutes=cadence_minutes).value)

    for _ in range(max_distance_rows):
        left = np.zeros(len(work), dtype=bool)
        right = np.zeros(len(work), dtype=bool)
        left[1:] = (
            active[:-1]
            & (group_codes[1:] == group_codes[:-1])
            & ((times[1:] - times[:-1]) == cadence_ns)
        )
        right[:-1] = (
            active[1:]
            & (group_codes[:-1] == group_codes[1:])
            & ((times[1:] - times[:-1]) == cadence_ns)
        )
        added = eligible & ~active & (left | right)
        if not added.any():
            break
        active |= added
        rescue |= added

    ordered_final = active.astype(np.int8)
    final_prediction = np.empty(len(work), dtype=np.int8)
    rescue_mask = np.empty(len(work), dtype=bool)
    final_prediction[ordered_positions] = ordered_final
    rescue_mask[ordered_positions] = rescue
    if np.any(final_prediction < base):
        raise AssertionError("residual may not remove a Round-B positive")
    return final_prediction, rescue_mask


def binary_metrics(truth: Sequence[int], prediction: Sequence[int]) -> dict[str, float | int]:
    """Return exact confusion counts and binary precision/recall/F1."""

    y = _checked_binary(truth, name="truth")
    p = _checked_binary(prediction, name="prediction")
    if y.shape != p.shape:
        raise ValueError("truth and prediction shapes differ")
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    f1 = float(2 * tp / denominator) if denominator else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
