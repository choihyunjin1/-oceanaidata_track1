"""Gap-aware probability post-processing for the P1 time series."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PostprocessConfig:
    """Hysteresis and morphology parameters selected on validation folds only."""

    high_threshold: float = 0.65
    low_threshold: float = 0.35
    close_gap_rows: int = 1
    minimum_positive_run: int = 2
    expected_interval: str | pd.Timedelta | None = "10min"
    time_column: str = "time"
    group_columns: tuple[str, ...] = ("station", "layer")

    def __post_init__(self) -> None:
        if not 0 <= self.low_threshold <= self.high_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= low_threshold <= high_threshold <= 1")
        if self.close_gap_rows < 0:
            raise ValueError("close_gap_rows must be non-negative")
        if self.minimum_positive_run < 1:
            raise ValueError("minimum_positive_run must be at least 1")
        if not self.group_columns:
            raise ValueError("group_columns must not be empty")


@dataclass(frozen=True)
class Segment:
    """Half-open positive segment in chronological working order."""

    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class PostprocessResult:
    """Auditable masks, all aligned to the original frame index."""

    label: pd.Series
    hysteresis: pd.Series
    hard_rule: pd.Series
    preserved_singleton_spike: pd.Series


def _one_dimensional_bool(
    values: Iterable[bool | int] | np.ndarray,
    *,
    length: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.shape != (length,):
        raise ValueError(f"{name} must be one-dimensional with length {length}")
    return array


def _break_array(
    breaks: Iterable[bool] | np.ndarray | None,
    length: int,
) -> np.ndarray:
    if breaks is None:
        result = np.zeros(length, dtype=bool)
        if length:
            result[0] = True
        return result
    result = _one_dimensional_bool(breaks, length=length, name="breaks").copy()
    if length:
        result[0] = True
    return result


def hysteresis_threshold(
    probabilities: Sequence[float] | np.ndarray,
    *,
    high_threshold: float,
    low_threshold: float,
    breaks: Iterable[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Grow high-confidence seeds through adjacent low-confidence candidates.

    Sequence breaks prevent growth across missing timestamps or sensor groups.
    """

    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if not np.isfinite(probability).all():
        raise ValueError("probabilities must be finite")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not 0 <= low_threshold <= high_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= low <= high <= 1")

    n_rows = len(probability)
    sequence_break = _break_array(breaks, n_rows)
    candidate = probability >= low_threshold
    seed = probability >= high_threshold
    result = np.zeros(n_rows, dtype=bool)

    run_start = 0
    for position in range(1, n_rows + 1):
        at_end = position == n_rows
        continues = (
            not at_end
            and not sequence_break[position]
            and candidate[position]
            and candidate[position - 1]
        )
        if continues:
            continue
        if candidate[run_start:position].any() and seed[run_start:position].any():
            # A run can start on a non-candidate after a physical sequence break.
            active = candidate[run_start:position]
            result[run_start:position] = active
        run_start = position
    return result


def close_short_gaps(
    mask: Iterable[bool] | np.ndarray,
    *,
    max_gap_rows: int,
    breaks: Iterable[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Close bounded zero gaps without crossing a physical sequence break."""

    result = np.asarray(mask, dtype=bool).copy()
    if result.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if max_gap_rows < 0:
        raise ValueError("max_gap_rows must be non-negative")
    if max_gap_rows == 0 or len(result) < 3:
        return result
    sequence_break = _break_array(breaks, len(result))

    position = 1
    while position < len(result) - 1:
        if result[position] or sequence_break[position]:
            position += 1
            continue
        start = position
        while position < len(result) and not result[position] and not sequence_break[position]:
            position += 1
        stop = position
        bounded = (
            start > 0
            and stop < len(result)
            and result[start - 1]
            and result[stop]
            and not sequence_break[stop]
        )
        if bounded and stop - start <= max_gap_rows:
            result[start:stop] = True
    return result


def remove_short_runs(
    mask: Iterable[bool] | np.ndarray,
    *,
    minimum_run: int,
    preserve: Iterable[bool] | np.ndarray | None = None,
    breaks: Iterable[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Remove short positive runs except explicitly preserved singleton spikes."""

    result = np.asarray(mask, dtype=bool).copy()
    if result.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if minimum_run < 1:
        raise ValueError("minimum_run must be at least 1")
    if preserve is None:
        preserve_array = np.zeros(len(result), dtype=bool)
    else:
        preserve_array = _one_dimensional_bool(
            preserve,
            length=len(result),
            name="preserve",
        )
    sequence_break = _break_array(breaks, len(result))

    start: int | None = None
    for position in range(len(result) + 1):
        active = position < len(result) and result[position]
        if position < len(result) and sequence_break[position] and start is not None:
            if position - start < minimum_run and not preserve_array[start:position].any():
                result[start:position] = False
            start = None
        if active and start is None:
            start = position
        if start is not None and (position == len(result) or not active):
            if position - start < minimum_run and not preserve_array[start:position].any():
                result[start:position] = False
            start = None
    result[preserve_array] = True
    return result


def segments_from_mask(
    mask: Iterable[bool] | np.ndarray,
    *,
    breaks: Iterable[bool] | np.ndarray | None = None,
) -> list[Segment]:
    """Return positive segments while respecting physical sequence breaks."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    sequence_break = _break_array(breaks, len(array))
    segments: list[Segment] = []
    start: int | None = None
    for position in range(len(array) + 1):
        if position < len(array) and sequence_break[position] and start is not None:
            segments.append(Segment(start, position))
            start = None
        active = position < len(array) and array[position]
        if active and start is None:
            start = position
        if start is not None and (position == len(array) or not active):
            segments.append(Segment(start, position))
            start = None
    return segments


def _chronological_order(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    time_column: str,
    expected_interval: str | pd.Timedelta | None,
) -> tuple[np.ndarray, np.ndarray]:
    required = [*group_columns, time_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")
    if frame.empty:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)

    work = frame[required].copy()
    work["__p1_position"] = np.arange(len(work), dtype=np.int64)
    work["__p1_time"] = pd.to_datetime(work[time_column], errors="coerce", utc=True)
    if work["__p1_time"].isna().any():
        raise ValueError("time values must all be parseable")
    work.sort_values([*group_columns, "__p1_time", "__p1_position"], inplace=True)
    positions = work["__p1_position"].to_numpy(dtype=np.int64)
    group_codes = pd.factorize(
        pd.MultiIndex.from_frame(work[list(group_columns)]),
        sort=False,
    )[0]
    times = work["__p1_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    breaks = np.ones(len(work), dtype=bool)
    if len(work) > 1:
        same_group = group_codes[1:] == group_codes[:-1]
        if expected_interval is None:
            contiguous = times[1:] > times[:-1]
        else:
            interval_ns = pd.Timedelta(expected_interval).value
            if interval_ns <= 0:
                raise ValueError("expected_interval must be positive")
            contiguous = times[1:] - times[:-1] == interval_ns
        breaks[1:] = ~(same_group & contiguous)
    return positions, breaks


def postprocess_probabilities(
    frame: pd.DataFrame,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    *,
    config: PostprocessConfig | None = None,
    hard_rule_mask: Iterable[bool] | pd.Series | np.ndarray | None = None,
    singleton_spike_mask: Iterable[bool] | pd.Series | np.ndarray | None = None,
) -> PostprocessResult:
    """Apply gap-aware hysteresis, hard overrides, closing, and run filtering.

    The function does not tune any parameter.  Callers must select the config
    solely from out-of-fold validation predictions.
    """

    config = config or PostprocessConfig()
    probability = np.asarray(probabilities, dtype=float)
    if probability.shape != (len(frame),):
        raise ValueError("probabilities must have one value per frame row")
    hard = (
        np.zeros(len(frame), dtype=bool)
        if hard_rule_mask is None
        else _one_dimensional_bool(hard_rule_mask, length=len(frame), name="hard_rule_mask")
    )
    spikes = (
        np.zeros(len(frame), dtype=bool)
        if singleton_spike_mask is None
        else _one_dimensional_bool(
            singleton_spike_mask,
            length=len(frame),
            name="singleton_spike_mask",
        )
    )

    positions, breaks = _chronological_order(
        frame,
        group_columns=config.group_columns,
        time_column=config.time_column,
        expected_interval=config.expected_interval,
    )
    ordered_probability = probability[positions]
    ordered_hard = hard[positions]
    ordered_spikes = spikes[positions]
    ordered_hysteresis = hysteresis_threshold(
        ordered_probability,
        high_threshold=config.high_threshold,
        low_threshold=config.low_threshold,
        breaks=breaks,
    )
    ordered_label = ordered_hysteresis | ordered_hard | ordered_spikes
    ordered_label = close_short_gaps(
        ordered_label,
        max_gap_rows=config.close_gap_rows,
        breaks=breaks,
    )
    ordered_label = remove_short_runs(
        ordered_label,
        minimum_run=config.minimum_positive_run,
        preserve=ordered_spikes,
        breaks=breaks,
    )

    label = np.zeros(len(frame), dtype=bool)
    hysteresis = np.zeros(len(frame), dtype=bool)
    label[positions] = ordered_label
    hysteresis[positions] = ordered_hysteresis
    return PostprocessResult(
        label=pd.Series(label.astype(np.int8), index=frame.index, name="label"),
        hysteresis=pd.Series(hysteresis, index=frame.index, name="hysteresis"),
        hard_rule=pd.Series(hard, index=frame.index, name="hard_rule"),
        preserved_singleton_spike=pd.Series(
            spikes,
            index=frame.index,
            name="preserved_singleton_spike",
        ),
    )
