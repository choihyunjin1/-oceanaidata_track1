"""Fold-local synthetic anomaly injection for P1 model training.

Only call :func:`augment_training_fold` with the training portion of a split.
The API deliberately accepts no validation frame, making accidental validation
mutation or statistic leakage harder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ANOMALY_TYPES = ("spike", "noise", "flatline", "offset", "drift")
# Official durations converted to 10-minute rows.  Decimal upper endpoints are
# rounded to the nearest observable row count used in the released labels.
DURATION_ROWS: dict[str, tuple[int, int]] = {
    "spike": (1, 1),
    "noise": (18, 353),
    "flatline": (12, 283),
    "offset": (48, 519),
    "drift": (54, 519),
}


@dataclass(frozen=True)
class AugmentConfig:
    """Synthetic injection controls; all randomness comes from ``seed``."""

    target_fraction: float = 0.04
    overlap_fraction: float = 0.15
    seed: int = 20260813
    expected_interval: str | pd.Timedelta = "10min"
    group_columns: tuple[str, ...] = ("station", "layer")
    time_column: str = "time"
    value_column: str = "temp"
    label_column: str = "label"
    anomaly_type_column: str = "anomaly_type"
    max_selection_attempts: int = 256

    def __post_init__(self) -> None:
        if not 0 <= self.target_fraction <= 1:
            raise ValueError("target_fraction must be in [0, 1]")
        if not 0 <= self.overlap_fraction <= 1:
            raise ValueError("overlap_fraction must be in [0, 1]")
        if self.max_selection_attempts < 1:
            raise ValueError("max_selection_attempts must be positive")
        if not self.group_columns:
            raise ValueError("group_columns must not be empty")
        if pd.Timedelta(self.expected_interval) <= pd.Timedelta(0):
            raise ValueError("expected_interval must be positive")


@dataclass(frozen=True)
class AugmentationResult:
    """Augmented copy plus row- and event-level audit information."""

    frame: pd.DataFrame
    injected_mask: pd.Series
    injected_types: pd.Series
    events: pd.DataFrame


@dataclass(frozen=True)
class _SelectedEvent:
    anomaly_type: str
    positions: np.ndarray
    group_key: tuple[Any, ...]
    is_overlap: bool
    parent_event: int | None


def _validate_input(frame: pd.DataFrame, config: AugmentConfig) -> None:
    required = [
        *config.group_columns,
        config.time_column,
        config.value_column,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")
    numeric = pd.to_numeric(frame[config.value_column], errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{config.value_column} must be finite for synthetic injection")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{config.value_column} must be finite for synthetic injection")
    parsed_time = pd.to_datetime(frame[config.time_column], errors="coerce", utc=True)
    if parsed_time.isna().any():
        raise ValueError(f"{config.time_column} contains unparseable values")
    if config.label_column in frame.columns:
        labels = pd.to_numeric(frame[config.label_column], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{config.label_column} must be binary 0/1")


def _normal_runs(
    frame: pd.DataFrame,
    eligible: np.ndarray,
    config: AugmentConfig,
) -> tuple[list[np.ndarray], dict[tuple[Any, ...], np.ndarray]]:
    work = frame[[*config.group_columns, config.time_column]].copy()
    work["__p1_position"] = np.arange(len(frame), dtype=np.int64)
    work["__p1_time"] = pd.to_datetime(work[config.time_column], utc=True)
    work.sort_values(
        [*config.group_columns, "__p1_time", "__p1_position"],
        inplace=True,
    )
    positions = work["__p1_position"].to_numpy(dtype=np.int64)
    times = work["__p1_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    group_values = list(work[list(config.group_columns)].itertuples(index=False, name=None))
    interval_ns = pd.Timedelta(config.expected_interval).value

    runs: list[np.ndarray] = []
    group_positions: dict[tuple[Any, ...], list[int]] = {}
    start: int | None = None
    for ordered_position, original_position in enumerate(positions):
        key = tuple(group_values[ordered_position])
        if eligible[original_position]:
            group_positions.setdefault(key, []).append(int(original_position))
        contiguous = False
        if ordered_position > 0:
            contiguous = (
                key == tuple(group_values[ordered_position - 1])
                and times[ordered_position] - times[ordered_position - 1] == interval_ns
                and eligible[positions[ordered_position - 1]]
                and eligible[original_position]
            )
        if eligible[original_position] and (ordered_position == 0 or not contiguous):
            if start is not None:
                runs.append(positions[start:ordered_position].copy())
            start = ordered_position
        elif not eligible[original_position] and start is not None:
            runs.append(positions[start:ordered_position].copy())
            start = None
    if start is not None:
        runs.append(positions[start:].copy())

    # Empty runs are impossible, but filtering here keeps the downstream
    # selection code defensive and explicit.
    runs = [run for run in runs if len(run)]
    compact_groups = {
        key: np.asarray(value, dtype=np.int64) for key, value in group_positions.items()
    }
    return runs, compact_groups


def _group_key_for_run(
    frame: pd.DataFrame, run: np.ndarray, config: AugmentConfig
) -> tuple[Any, ...]:
    first = int(run[0])
    return tuple(frame.iloc[first][column] for column in config.group_columns)


def _robust_step_scales(
    frame: pd.DataFrame,
    runs: Sequence[np.ndarray],
    config: AugmentConfig,
) -> dict[tuple[Any, ...], float]:
    values = pd.to_numeric(frame[config.value_column]).to_numpy(dtype=float)
    differences: dict[tuple[Any, ...], list[float]] = {}
    for run in runs:
        if len(run) < 2:
            continue
        key = _group_key_for_run(frame, run, config)
        step = np.abs(np.diff(values[run]))
        finite_positive = step[np.isfinite(step) & (step > np.finfo(float).eps)]
        if len(finite_positive):
            differences.setdefault(key, []).extend(finite_positive.tolist())
    return {key: max(float(np.median(candidate)), 0.02) for key, candidate in differences.items()}


def _choose_free_event(
    *,
    anomaly_type: str,
    runs: Sequence[np.ndarray],
    occupied: np.ndarray,
    rng: np.random.Generator,
    remaining: int,
    config: AugmentConfig,
    frame: pd.DataFrame,
) -> _SelectedEvent | None:
    minimum, maximum = DURATION_ROWS[anomaly_type]
    if remaining < minimum:
        return None
    maximum = min(maximum, remaining)
    candidates = [run for run in runs if len(run) >= minimum]
    if not candidates:
        return None

    for _ in range(config.max_selection_attempts):
        run = candidates[int(rng.integers(0, len(candidates)))]
        upper = min(maximum, len(run))
        if upper < minimum:
            continue
        duration = int(rng.integers(minimum, upper + 1))
        start = int(rng.integers(0, len(run) - duration + 1))
        positions = run[start : start + duration]
        if occupied[positions].any():
            continue
        return _SelectedEvent(
            anomaly_type=anomaly_type,
            positions=positions.copy(),
            group_key=_group_key_for_run(frame, run, config),
            is_overlap=False,
            parent_event=None,
        )
    return None


def _apply_event(
    values: np.ndarray,
    event: _SelectedEvent,
    *,
    step_scale: float,
    rng: np.random.Generator,
) -> None:
    positions = event.positions
    kind = event.anomaly_type
    sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
    if kind == "spike":
        amplitude = max(1.0, float(rng.uniform(8.0, 16.0)) * step_scale)
        values[positions[0]] += sign * amplitude
    elif kind == "noise":
        noise_scale = max(0.25, float(rng.uniform(4.0, 9.0)) * step_scale)
        values[positions] += rng.normal(0.0, noise_scale, size=len(positions))
    elif kind == "flatline":
        # Exact equality is deliberate: this corresponds to a stuck sensor and
        # makes the >=6 plateau hard rule independently testable.
        values[positions] = values[positions[0]]
    elif kind == "offset":
        amplitude = max(0.5, float(rng.uniform(5.0, 12.0)) * step_scale)
        values[positions] += sign * amplitude
    elif kind == "drift":
        amplitude = max(1.0, float(rng.uniform(10.0, 24.0)) * step_scale)
        values[positions] += np.linspace(0.0, sign * amplitude, len(positions))
    else:  # pragma: no cover - protected by the internal type queue
        raise ValueError(f"unknown anomaly type: {kind}")


def _select_overlap_events(
    primary_events: Sequence[_SelectedEvent],
    *,
    fraction: float,
    rng: np.random.Generator,
) -> list[_SelectedEvent]:
    eligible_indices = [
        index
        for index, event in enumerate(primary_events)
        if any(
            kind != event.anomaly_type and DURATION_ROWS[kind][0] <= len(event.positions)
            for kind in ANOMALY_TYPES
        )
    ]
    if not eligible_indices or fraction <= 0:
        return []
    count = min(len(eligible_indices), int(round(len(primary_events) * fraction)))
    if count == 0 and primary_events:
        count = 1
    selected = rng.choice(np.asarray(eligible_indices), size=count, replace=False)
    overlaps: list[_SelectedEvent] = []
    for parent_index_raw in np.atleast_1d(selected):
        parent_index = int(parent_index_raw)
        parent = primary_events[parent_index]
        candidates = [
            kind
            for kind in ANOMALY_TYPES
            if kind != parent.anomaly_type and DURATION_ROWS[kind][0] <= len(parent.positions)
        ]
        kind = candidates[int(rng.integers(0, len(candidates)))]
        minimum, maximum = DURATION_ROWS[kind]
        duration = int(rng.integers(minimum, min(maximum, len(parent.positions)) + 1))
        start = int(rng.integers(0, len(parent.positions) - duration + 1))
        overlaps.append(
            _SelectedEvent(
                anomaly_type=kind,
                positions=parent.positions[start : start + duration].copy(),
                group_key=parent.group_key,
                is_overlap=True,
                parent_event=parent_index,
            )
        )
    return overlaps


def _type_string(types: set[str]) -> str:
    return "+".join(kind for kind in ANOMALY_TYPES if kind in types)


def augment_training_fold(
    train_fold: pd.DataFrame,
    config: AugmentConfig | None = None,
) -> AugmentationResult:
    """Return an augmented copy of one chronological/grouped training fold.

    Existing positives are retained byte-for-byte in the signal and are never
    candidates for synthetic injection.  New event statistics are calculated
    from this function's input only; therefore the caller must pass only the
    fold's training rows.
    """

    config = config or AugmentConfig()
    _validate_input(train_fold, config)
    output = train_fold.copy(deep=True)
    if config.label_column not in output.columns:
        output[config.label_column] = np.zeros(len(output), dtype=np.int8)
    else:
        output[config.label_column] = pd.to_numeric(
            output[config.label_column],
        ).astype(np.int8)
    if config.anomaly_type_column not in output.columns:
        output[config.anomaly_type_column] = pd.Series(
            [""] * len(output),
            index=output.index,
            dtype=object,
        )
    else:
        output[config.anomaly_type_column] = (
            output[config.anomaly_type_column].fillna("").astype(object)
        )

    original_label = output[config.label_column].to_numpy(dtype=np.int8)
    values = pd.to_numeric(output[config.value_column]).to_numpy(dtype=float, copy=True)
    eligible = (original_label == 0) & np.isfinite(values)
    runs, _ = _normal_runs(output, eligible, config)
    step_scales = _robust_step_scales(output, runs, config)
    target_rows = int(round(int(eligible.sum()) * config.target_fraction))
    occupied = np.zeros(len(output), dtype=bool)
    rng = np.random.default_rng(config.seed)

    primary_events: list[_SelectedEvent] = []
    minimum_all_types = sum(DURATION_ROWS[kind][0] for kind in ANOMALY_TYPES)
    required_queue = list(ANOMALY_TYPES) if target_rows >= minimum_all_types else []
    rng.shuffle(required_queue)
    type_queue: list[str] = []
    failed_cycles = 0
    while int(occupied.sum()) < target_rows and failed_cycles < len(ANOMALY_TYPES) * 3:
        if required_queue:
            kind = required_queue.pop()
            # Keep enough row budget to inject every still-missing official
            # type at least once whenever the requested target permits it.
            reserved = sum(DURATION_ROWS[item][0] for item in required_queue)
            remaining = target_rows - int(occupied.sum())
            event_budget = max(DURATION_ROWS[kind][0], remaining - reserved)
        else:
            if not type_queue:
                type_queue = list(ANOMALY_TYPES)
                rng.shuffle(type_queue)
            kind = type_queue.pop()
            remaining = target_rows - int(occupied.sum())
            event_budget = remaining
        if event_budget < DURATION_ROWS[kind][0]:
            kind = "spike"
        event = _choose_free_event(
            anomaly_type=kind,
            runs=runs,
            occupied=occupied,
            rng=rng,
            remaining=event_budget,
            config=config,
            frame=output,
        )
        if event is None:
            failed_cycles += 1
            continue
        failed_cycles = 0
        occupied[event.positions] = True
        primary_events.append(event)

    overlap_events = _select_overlap_events(
        primary_events,
        fraction=config.overlap_fraction,
        rng=rng,
    )
    all_events = [*primary_events, *overlap_events]
    row_types: list[set[str]] = [set() for _ in range(len(output))]
    event_records: list[dict[str, Any]] = []
    for event_id, event in enumerate(all_events):
        scale = step_scales.get(event.group_key, 0.02)
        _apply_event(values, event, step_scale=scale, rng=rng)
        for position in event.positions:
            row_types[int(position)].add(event.anomaly_type)
        event_records.append(
            {
                "event_id": event_id,
                "anomaly_type": event.anomaly_type,
                "start_position": int(event.positions[0]),
                "stop_position": int(event.positions[-1]) + 1,
                "rows": int(len(event.positions)),
                "start_time": output.iloc[int(event.positions[0])][config.time_column],
                "end_time": output.iloc[int(event.positions[-1])][config.time_column],
                "group_key": event.group_key,
                "is_overlap": event.is_overlap,
                "parent_event": event.parent_event,
            }
        )

    output[config.value_column] = values
    synthetic_positions = np.flatnonzero(occupied)
    if len(synthetic_positions):
        label_column_index = output.columns.get_loc(config.label_column)
        type_column_index = output.columns.get_loc(config.anomaly_type_column)
        output.iloc[synthetic_positions, label_column_index] = 1
        for position in synthetic_positions:
            output.iat[int(position), type_column_index] = _type_string(row_types[int(position)])

    injected_types = pd.Series(
        [_type_string(types) for types in row_types],
        index=output.index,
        name="synthetic_anomaly_type",
        dtype=object,
    )
    event_columns = [
        "event_id",
        "anomaly_type",
        "start_position",
        "stop_position",
        "rows",
        "start_time",
        "end_time",
        "group_key",
        "is_overlap",
        "parent_event",
    ]
    return AugmentationResult(
        frame=output,
        injected_mask=pd.Series(occupied, index=output.index, name="synthetic_injected"),
        injected_types=injected_types,
        events=pd.DataFrame.from_records(event_records, columns=event_columns),
    )


# Concise alias for callers that already operate inside a fold loop.
inject_synthetic_anomalies = augment_training_fold
