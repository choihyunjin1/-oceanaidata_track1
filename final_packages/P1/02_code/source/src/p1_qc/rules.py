"""Deterministic, auditable hard rules for the P1 temperature signal.

The rules in this module intentionally return their component masks.  This is
important for P1: a hard rule must be scored on each chronological validation
fold before it is allowed to override a learned model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_GROUP_COLUMNS = ("station", "layer")


@dataclass(frozen=True)
class PlateauRuleConfig:
    """Configuration for the exact/near-exact plateau hard rule."""

    min_run: int = 6
    atol: float = 0.0
    expected_interval: str | pd.Timedelta | None = "10min"
    value_column: str = "temp"
    time_column: str = "time"
    group_columns: tuple[str, ...] = DEFAULT_GROUP_COLUMNS

    def __post_init__(self) -> None:
        if self.min_run < 2:
            raise ValueError("min_run must be at least 2")
        if self.atol < 0:
            raise ValueError("atol must be non-negative")
        if not self.group_columns:
            raise ValueError("group_columns must not be empty")


@dataclass(frozen=True)
class SpikeRuleConfig:
    """Robust isolated-spike rule.

    ``z_threshold`` is relative to the median absolute first difference in the
    station/layer series.  ``min_abs_jump`` prevents very quiet periods from
    turning floating-point noise into a hard anomaly.
    """

    z_threshold: float = 8.0
    min_abs_jump: float = 0.5
    neighbor_ratio: float = 0.35
    expected_interval: str | pd.Timedelta | None = "10min"
    value_column: str = "temp"
    time_column: str = "time"
    group_columns: tuple[str, ...] = DEFAULT_GROUP_COLUMNS

    def __post_init__(self) -> None:
        if self.z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if self.min_abs_jump < 0:
            raise ValueError("min_abs_jump must be non-negative")
        if not 0 <= self.neighbor_ratio <= 1:
            raise ValueError("neighbor_ratio must be in [0, 1]")
        if not self.group_columns:
            raise ValueError("group_columns must not be empty")


@dataclass(frozen=True)
class RuleMetrics:
    precision: float
    recall: float
    f1: float
    predicted_positive: int
    true_positive: int
    support: int


@dataclass(frozen=True)
class HardRuleResult:
    """Component masks and their union, aligned to the input frame index."""

    label: pd.Series
    plateau: pd.Series
    singleton_spike: pd.Series


def _required_columns(
    frame: pd.DataFrame,
    value_column: str,
    time_column: str,
    group_columns: Sequence[str],
) -> None:
    required = [*group_columns, time_column, value_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")


def _ordered_arrays(
    frame: pd.DataFrame,
    *,
    value_column: str,
    time_column: str,
    group_columns: Sequence[str],
    expected_interval: str | pd.Timedelta | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return values/group codes/breaks/original positions in chronological order."""

    _required_columns(frame, value_column, time_column, group_columns)
    if frame.empty:
        empty = np.empty(0, dtype=np.int64)
        return (
            np.empty(0, dtype=float),
            empty,
            np.empty(0, dtype=bool),
            empty,
        )

    work = frame[[*group_columns, time_column, value_column]].copy()
    work["__p1_position"] = np.arange(len(work), dtype=np.int64)
    work["__p1_time"] = pd.to_datetime(work[time_column], errors="coerce", utc=True)
    if work["__p1_time"].isna().any():
        count = int(work["__p1_time"].isna().sum())
        raise ValueError(f"{count} time values could not be parsed")
    work.sort_values([*group_columns, "__p1_time", "__p1_position"], inplace=True)

    group_index = pd.MultiIndex.from_frame(work[list(group_columns)])
    group_codes = pd.factorize(group_index, sort=False)[0].astype(np.int64, copy=False)
    # pandas 3 may store parsed timestamps at microsecond resolution.  Convert
    # explicitly so comparison with ``Timedelta.value`` (nanoseconds) is exact.
    times = work["__p1_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    breaks = np.ones(len(work), dtype=bool)
    if len(work) > 1:
        same_group = group_codes[1:] == group_codes[:-1]
        if expected_interval is None:
            contiguous = times[1:] > times[:-1]
        else:
            expected_ns = pd.Timedelta(expected_interval).value
            if expected_ns <= 0:
                raise ValueError("expected_interval must be positive")
            contiguous = times[1:] - times[:-1] == expected_ns
        breaks[1:] = ~(same_group & contiguous)

    values = pd.to_numeric(work[value_column], errors="coerce").to_numpy(dtype=float)
    positions = work["__p1_position"].to_numpy(dtype=np.int64)
    return values, group_codes, breaks, positions


def plateau_runs(
    values: Iterable[float],
    *,
    min_run: int = 6,
    atol: float = 0.0,
    breaks: Iterable[bool] | None = None,
) -> np.ndarray:
    """Flag all rows in repeated-value runs of at least ``min_run`` rows.

    ``breaks[i]`` marks row ``i`` as the start of a new physical sequence.  It
    lets callers prevent a rule from bridging a station/layer boundary or an
    observation gap.
    """

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    array = array.astype(float, copy=False)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if min_run < 2:
        raise ValueError("min_run must be at least 2")
    if atol < 0:
        raise ValueError("atol must be non-negative")
    n_rows = len(array)
    result = np.zeros(n_rows, dtype=bool)
    if n_rows == 0:
        return result

    if breaks is None:
        sequence_break = np.zeros(n_rows, dtype=bool)
        sequence_break[0] = True
    else:
        sequence_break = np.asarray(
            list(breaks) if not isinstance(breaks, np.ndarray) else breaks,
            dtype=bool,
        )
        if sequence_break.shape != (n_rows,):
            raise ValueError("breaks must have the same length as values")
        sequence_break = sequence_break.copy()
        sequence_break[0] = True

    run_start = 0
    for position in range(1, n_rows + 1):
        at_end = position == n_rows
        if not at_end:
            same_value = (
                np.isfinite(array[position])
                and np.isfinite(array[position - 1])
                and abs(array[position] - array[position - 1]) <= atol
            )
            continues = not sequence_break[position] and same_value
        else:
            continues = False
        if continues:
            continue
        if position - run_start >= min_run and np.isfinite(array[run_start]):
            result[run_start:position] = True
        run_start = position
    return result


def detect_plateaus(
    frame: pd.DataFrame,
    config: PlateauRuleConfig | None = None,
) -> pd.Series:
    """Return the plateau hard-rule mask in the original row order."""

    config = config or PlateauRuleConfig()
    values, _, breaks, positions = _ordered_arrays(
        frame,
        value_column=config.value_column,
        time_column=config.time_column,
        group_columns=config.group_columns,
        expected_interval=config.expected_interval,
    )
    ordered_mask = plateau_runs(
        values,
        min_run=config.min_run,
        atol=config.atol,
        breaks=breaks,
    )
    mask = np.zeros(len(frame), dtype=bool)
    mask[positions] = ordered_mask
    return pd.Series(mask, index=frame.index, name="plateau_rule")


def _group_step_scales(
    values: np.ndarray,
    group_codes: np.ndarray,
    breaks: np.ndarray,
) -> dict[int, float]:
    differences: dict[int, list[float]] = {}
    for position in range(1, len(values)):
        if breaks[position] or not np.isfinite(values[position : position + 1]).all():
            continue
        if not np.isfinite(values[position - 1]):
            continue
        differences.setdefault(int(group_codes[position]), []).append(
            abs(float(values[position] - values[position - 1]))
        )
    scales: dict[int, float] = {}
    for code, candidates in differences.items():
        finite = np.asarray(candidates, dtype=float)
        finite = finite[np.isfinite(finite)]
        # Preserve zero steps: in a quiet/quantized series they are genuine
        # scale evidence.  ``min_abs_jump`` still supplies a physical floor.
        scales[code] = float(np.median(finite)) if len(finite) else 0.0
    return scales


def detect_singleton_spikes(
    frame: pd.DataFrame,
    config: SpikeRuleConfig | None = None,
) -> pd.Series:
    """Detect one-row excursions whose immediate neighbours return to baseline."""

    config = config or SpikeRuleConfig()
    values, group_codes, breaks, positions = _ordered_arrays(
        frame,
        value_column=config.value_column,
        time_column=config.time_column,
        group_columns=config.group_columns,
        expected_interval=config.expected_interval,
    )
    ordered_mask = np.zeros(len(values), dtype=bool)
    scales = _group_step_scales(values, group_codes, breaks)
    for position in range(1, len(values) - 1):
        if breaks[position] or breaks[position + 1]:
            continue
        left, center, right = values[position - 1 : position + 2]
        if not np.isfinite((left, center, right)).all():
            continue
        jump_left = abs(float(center - left))
        jump_right = abs(float(center - right))
        scale = scales.get(int(group_codes[position]), 0.0)
        threshold = max(config.min_abs_jump, config.z_threshold * scale)
        excursion = min(jump_left, jump_right)
        neighbour_distance = abs(float(right - left))
        if excursion >= threshold and neighbour_distance <= config.neighbor_ratio * excursion:
            ordered_mask[position] = True

    mask = np.zeros(len(frame), dtype=bool)
    mask[positions] = ordered_mask
    return pd.Series(mask, index=frame.index, name="singleton_spike_rule")


def apply_hard_rules(
    frame: pd.DataFrame,
    *,
    plateau: PlateauRuleConfig | None = None,
    spike: SpikeRuleConfig | None = None,
    include_singleton_spikes: bool = True,
) -> HardRuleResult:
    """Evaluate P1 hard rules and expose each mask for fold-level validation."""

    plateau_mask = detect_plateaus(frame, plateau)
    if include_singleton_spikes:
        spike_mask = detect_singleton_spikes(frame, spike)
    else:
        spike_mask = pd.Series(False, index=frame.index, name="singleton_spike_rule")
    label = (plateau_mask | spike_mask).rename("hard_rule_label")
    return HardRuleResult(label=label, plateau=plateau_mask, singleton_spike=spike_mask)


def evaluate_binary_rule(
    predicted: Sequence[bool | int] | pd.Series,
    truth: Sequence[bool | int] | pd.Series,
) -> RuleMetrics:
    """Compute auditable binary metrics without requiring scikit-learn."""

    predicted_array = np.asarray(predicted, dtype=bool)
    truth_array = np.asarray(truth, dtype=bool)
    if predicted_array.ndim != 1 or truth_array.ndim != 1:
        raise ValueError("predicted and truth must be one-dimensional")
    if predicted_array.shape != truth_array.shape:
        raise ValueError("predicted and truth must have equal length")
    true_positive = int(np.count_nonzero(predicted_array & truth_array))
    predicted_positive = int(np.count_nonzero(predicted_array))
    support = int(np.count_nonzero(truth_array))
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / support if support else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RuleMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_positive=predicted_positive,
        true_positive=true_positive,
        support=support,
    )
