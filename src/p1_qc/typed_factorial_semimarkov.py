"""Typed anomaly grammar and deterministic factorial semi-Markov decoding.

This module is deliberately independent of P1 outer folds and submissions.  It
contains only the structural machinery used by the historical-inner experiment
``p1_typed_factorial_semimarkov_v1``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import ceil, floor
from typing import Any

import numpy as np
import pandas as pd

ANOMALY_TYPES = ("spike", "noise", "flatline", "offset", "drift")
TYPE_TO_INDEX = {name: index for index, name in enumerate(ANOMALY_TYPES)}
OFFICIAL_DURATION_ROWS: dict[str, tuple[int, int]] = {
    "spike": (1, 1),
    "noise": (18, 353),
    "flatline": (12, 283),
    "offset": (48, 519),
    "drift": (54, 519),
}


@dataclass(frozen=True)
class RawTypeSequence:
    """One raw label with order and multiplicity retained exactly."""

    raw: str
    tokens: tuple[str, ...]
    counts: tuple[int, ...]
    membership: tuple[bool, ...]

    @property
    def is_composite(self) -> bool:
        return len(self.tokens) >= 2


def parse_raw_anomaly_type(
    value: Any,
    *,
    known_types: Sequence[str] = ANOMALY_TYPES,
    strict: bool = True,
) -> RawTypeSequence:
    """Parse ``+`` tokens without canonicalising, sorting, or deduplicating."""

    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        raw = ""
    else:
        raw = str(value).strip()
    tokens = tuple(part.strip() for part in raw.split("+") if part.strip()) if raw else ()
    known = tuple(str(item) for item in known_types)
    unknown = tuple(token for token in tokens if token not in known)
    if strict and unknown:
        raise ValueError(f"unknown anomaly type tokens: {sorted(set(unknown))}")
    counter = Counter(tokens)
    counts = tuple(int(counter[item]) for item in known)
    return RawTypeSequence(raw, tokens, counts, tuple(count > 0 for count in counts))


def parse_raw_anomaly_types(
    values: pd.Series,
    *,
    known_types: Sequence[str] = ANOMALY_TYPES,
    strict: bool = True,
) -> tuple[list[RawTypeSequence], np.ndarray]:
    """Return parsed sequences and an integer ``rows x types`` count matrix."""

    sequences = [
        parse_raw_anomaly_type(value, known_types=known_types, strict=strict)
        for value in values.tolist()
    ]
    matrix = np.asarray([item.counts for item in sequences], dtype=np.int8)
    if not sequences:
        matrix = np.empty((0, len(tuple(known_types))), dtype=np.int8)
    return sequences, matrix


def exact_segment_ids(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("station", "layer"),
    time_column: str = "time",
    cadence_minutes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic segment IDs and UTC nanosecond timestamps in row order."""

    missing = sorted((set(group_columns) | {time_column}).difference(frame.columns))
    if missing:
        raise KeyError(f"missing segmentation columns: {missing}")
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be positive")
    times = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError("timestamps must all be parseable")
    work = pd.DataFrame({"__pos": np.arange(len(frame), dtype=np.int64), "__time": times})
    for column in group_columns:
        work[column] = frame[column].to_numpy()
    ordered = work.sort_values([*group_columns, "__time", "__pos"], kind="mergesort")
    group_values = [ordered[column] for column in group_columns]
    delta = ordered.groupby(list(group_columns), sort=False, observed=True)["__time"].diff()
    new_segment = delta.ne(pd.Timedelta(minutes=cadence_minutes)) | delta.isna()
    local = new_segment.groupby(group_values, sort=False).cumsum()
    ordered["__segment"] = ordered.groupby(
        [*group_columns, local], sort=False, observed=True
    ).ngroup()
    restored = ordered.sort_values("__pos", kind="mergesort")
    return (
        restored["__segment"].to_numpy(dtype=np.int64),
        restored["__time"].astype("int64").to_numpy(dtype=np.int64),
    )


@dataclass(frozen=True)
class AtomicPresenceRun:
    segment_id: int
    anomaly_type: str
    occurrence_rank: int
    start_position: int
    stop_position: int

    @property
    def length(self) -> int:
        return self.stop_position - self.start_position + 1


@dataclass(frozen=True)
class SuperEvent:
    super_event_id: int
    segment_id: int
    start_position: int
    stop_position: int
    atomic_run_indices: tuple[int, ...]

    @property
    def length(self) -> int:
        return self.stop_position - self.start_position + 1


@dataclass(frozen=True)
class GrammarAudit:
    sequences: tuple[RawTypeSequence, ...]
    token_counts: np.ndarray
    segment_ids: np.ndarray
    atomic_runs: tuple[AtomicPresenceRun, ...]
    super_events: tuple[SuperEvent, ...]
    row_super_event_ids: np.ndarray
    recognized_positive_rows: np.ndarray
    max_two_positive_rows: np.ndarray
    duration_decomposable_positive_rows: np.ndarray
    signature_counts: Mapping[str, int]


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    if not values.any():
        return []
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def duration_is_decomposable(length: int, minimum: int, maximum: int) -> bool:
    """Whether a run can be tiled by one or more legal adjacent events."""

    if length <= 0 or minimum <= 0 or maximum < minimum:
        return False
    return ceil(length / maximum) <= floor(length / minimum)


def build_grammar_audit(
    frame: pd.DataFrame,
    *,
    anomaly_column: str = "anomaly_type",
    label_column: str = "label",
    cadence_minutes: int = 10,
    duration_rows: Mapping[str, tuple[int, int]] = OFFICIAL_DURATION_ROWS,
) -> GrammarAudit:
    """Reconstruct presence runs and overlap-connected super-events."""

    if anomaly_column not in frame or label_column not in frame:
        raise KeyError("grammar audit needs anomaly_type and label")
    sequences, counts = parse_raw_anomaly_types(frame[anomaly_column], strict=True)
    segment_ids, _ = exact_segment_ids(frame, cadence_minutes=cadence_minutes)
    labels = pd.to_numeric(frame[label_column], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("label must be binary")
    positive = labels == 1
    recognized = np.asarray([bool(item.tokens) for item in sequences], dtype=bool) & positive
    if not np.array_equal(positive, np.asarray([bool(item.tokens) for item in sequences])):
        raise ValueError("label and raw anomaly_type emptiness disagree")

    atomic: list[AtomicPresenceRun] = []
    decomposable = np.ones(len(frame), dtype=bool)
    for segment in np.unique(segment_ids):
        positions = np.flatnonzero(segment_ids == segment)
        for type_index, name in enumerate(ANOMALY_TYPES):
            maximum_rank = int(counts[positions, type_index].max(initial=0))
            minimum, maximum = duration_rows[name]
            for rank in range(1, maximum_rank + 1):
                active = counts[positions, type_index] >= rank
                for local_start, local_stop in _true_runs(active):
                    start = int(positions[local_start])
                    stop = int(positions[local_stop])
                    run = AtomicPresenceRun(int(segment), name, rank, start, stop)
                    atomic.append(run)
                    if not duration_is_decomposable(run.length, minimum, maximum):
                        decomposable[start : stop + 1] = False

    super_events: list[SuperEvent] = []
    row_event = np.full(len(frame), -1, dtype=np.int64)
    by_segment: dict[int, list[int]] = {}
    for index, run in enumerate(atomic):
        by_segment.setdefault(run.segment_id, []).append(index)
    for segment, indices in sorted(by_segment.items()):
        ordered_indices = sorted(
            indices,
            key=lambda index: (
                atomic[index].start_position,
                atomic[index].stop_position,
                TYPE_TO_INDEX[atomic[index].anomaly_type],
                atomic[index].occurrence_rank,
            ),
        )
        component: list[int] = []
        component_start = -1
        component_stop = -1
        for index in ordered_indices:
            run = atomic[index]
            if component and run.start_position > component_stop + 1:
                event_id = len(super_events)
                event = SuperEvent(
                    event_id,
                    segment,
                    component_start,
                    component_stop,
                    tuple(component),
                )
                super_events.append(event)
                row_event[component_start : component_stop + 1] = event_id
                component = []
            if not component:
                component_start = run.start_position
                component_stop = run.stop_position
            else:
                component_stop = max(component_stop, run.stop_position)
            component.append(index)
        if component:
            event_id = len(super_events)
            event = SuperEvent(
                event_id,
                segment,
                component_start,
                component_stop,
                tuple(component),
            )
            super_events.append(event)
            row_event[component_start : component_stop + 1] = event_id

    signature_counts = Counter(item.raw for item in sequences if item.raw)
    max_two = positive & (counts.sum(axis=1) <= 2)
    duration_supported = positive & decomposable
    return GrammarAudit(
        tuple(sequences),
        counts,
        segment_ids,
        tuple(atomic),
        tuple(super_events),
        row_event,
        recognized,
        max_two,
        duration_supported,
        dict(sorted(signature_counts.items())),
    )


def chronological_split_masks(
    frame: pd.DataFrame,
    grammar: GrammarAudit,
    *,
    fit_end_inclusive: str,
    validation_start_inclusive: str,
    validation_end_inclusive: str,
    embargo_days: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an embargoed historical split and reject super-event leakage."""

    times = pd.to_datetime(frame["time"], errors="raise", utc=True)
    fit_end = pd.Timestamp(fit_end_inclusive).tz_convert("UTC")
    val_start = pd.Timestamp(validation_start_inclusive).tz_convert("UTC")
    val_end = pd.Timestamp(validation_end_inclusive).tz_convert("UTC")
    if val_end < val_start:
        raise ValueError("validation_end must not precede validation_start")
    if val_start - fit_end <= pd.Timedelta(days=embargo_days):
        raise ValueError("split does not provide the required full embargo")
    fit = (times <= fit_end).to_numpy(dtype=bool)
    validation = ((times >= val_start) & (times <= val_end)).to_numpy(dtype=bool)
    fit_events = set(grammar.row_super_event_ids[fit & (grammar.row_super_event_ids >= 0)])
    validation_events = set(
        grammar.row_super_event_ids[validation & (grammar.row_super_event_ids >= 0)]
    )
    overlap = fit_events.intersection(validation_events)
    if overlap:
        raise ValueError(f"super-event leakage across split: {len(overlap)} events")
    return fit, validation


@dataclass(frozen=True)
class DecoderConfig:
    duration_rows: Mapping[str, tuple[int, int]]
    beam_width: int = 24
    maximum_concurrent_events: int = 2
    maximum_start_candidates_per_row: int = 2
    start_penalty: float = 3.1780538303479458
    overlap_penalty: float = 1.0
    duplicate_type_start_penalty: float = 3.1780538303479458
    probability_clip: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.maximum_concurrent_events != 2:
            raise ValueError("this preregistered decoder requires maximum concurrency two")
        if not 1 <= self.maximum_start_candidates_per_row <= len(ANOMALY_TYPES):
            raise ValueError("invalid maximum_start_candidates_per_row")
        if not 0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must lie in (0, .5)")
        if set(self.duration_rows) != set(ANOMALY_TYPES):
            raise ValueError("duration_rows must define exactly the five anomaly types")


ActiveEvent = tuple[int, int]
BeamState = tuple[ActiveEvent, ...]


def rowwise_union(probabilities: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(ANOMALY_TYPES):
        raise ValueError("probabilities must have shape (rows, five)")
    finite = np.where(np.isfinite(values), values, -np.inf)
    return (finite >= threshold).any(axis=1).astype(np.int8)


def _survivor_states(state: BeamState, config: DecoderConfig) -> tuple[BeamState, ...]:
    options: list[list[ActiveEvent | None]] = []
    for type_index, age in state:
        minimum, maximum = config.duration_rows[ANOMALY_TYPES[type_index]]
        choices: list[ActiveEvent | None] = []
        if age < maximum:
            choices.append((type_index, age + 1))
        if age >= minimum:
            choices.append(None)
        options.append(choices)
    if not options:
        return ((),)
    result: list[BeamState] = [()]
    for choices in options:
        result = [
            prefix + (() if choice is None else (choice,))
            for prefix in result
            for choice in choices
        ]
    return tuple(result)


def _start_sets(
    survivor: BeamState,
    ranked_types: Sequence[int],
    remaining_rows: int,
    config: DecoderConfig,
) -> tuple[tuple[int, ...], ...]:
    slots = config.maximum_concurrent_events - len(survivor)
    if slots <= 0:
        return ((),)
    feasible = [
        type_index
        for type_index in ranked_types[: config.maximum_start_candidates_per_row]
        if config.duration_rows[ANOMALY_TYPES[type_index]][0] <= remaining_rows
    ]
    starts: list[tuple[int, ...]] = [()]
    starts.extend((item,) for item in feasible)
    if slots >= 2 and len(feasible) >= 2:
        starts.extend(tuple(pair) for pair in combinations(feasible, 2))
    if slots >= 2 and feasible:
        starts.append((feasible[0], feasible[0]))
    return tuple(starts)


def _transition_score(
    survivor: BeamState,
    starts: tuple[int, ...],
    logits: np.ndarray,
    config: DecoderConfig,
) -> tuple[BeamState, float]:
    state = survivor + tuple((type_index, 1) for type_index in starts)
    score = float(sum(logits[type_index] for type_index, _ in state))
    score -= config.start_penalty * len(starts)
    if len(state) == 2:
        score -= config.overlap_penalty
    types = [item[0] for item in state]
    if len(types) != len(set(types)):
        score -= config.duplicate_type_start_penalty
    return state, score


def decode_segment(
    probabilities: np.ndarray,
    config: DecoderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Beam-MAP decode one exact-cadence segment.

    Returns a binary union and a five-column type-activity matrix.  The
    implementation permits two active latent events, including two instances
    of the same type, while enforcing each event's inclusive duration support.
    """

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(ANOMALY_TYPES):
        raise ValueError("probabilities must have shape (rows, five)")
    if len(values) == 0:
        return np.empty(0, dtype=np.int8), np.empty((0, len(ANOMALY_TYPES)), dtype=np.int8)
    if not np.isfinite(values).all():
        raise ValueError("decode_segment requires finite probabilities")
    clipped = np.clip(values, config.probability_clip, 1.0 - config.probability_clip)
    logits = np.log(clipped) - np.log1p(-clipped)

    beam_states: list[BeamState] = [()]
    beam_scores = np.asarray([0.0], dtype=float)
    state_history: list[list[BeamState]] = []
    back_history: list[np.ndarray] = []
    for row_index in range(len(values)):
        ranked = np.argsort(-logits[row_index], kind="stable").tolist()
        remaining = len(values) - row_index
        candidates: dict[BeamState, tuple[float, int]] = {}
        for previous_index, (previous, previous_score) in enumerate(
            zip(beam_states, beam_scores, strict=True)
        ):
            for survivor in _survivor_states(previous, config):
                for starts in _start_sets(survivor, ranked, remaining, config):
                    state, local_score = _transition_score(
                        survivor, starts, logits[row_index], config
                    )
                    total = float(previous_score + local_score)
                    old = candidates.get(state)
                    if (
                        old is None
                        or total > old[0]
                        or (total == old[0] and previous_index < old[1])
                    ):
                        candidates[state] = (total, previous_index)
        ranked_candidates = sorted(
            candidates.items(), key=lambda item: (-item[1][0], item[0], item[1][1])
        )[: config.beam_width]
        beam_states = [item[0] for item in ranked_candidates]
        beam_scores = np.asarray([item[1][0] for item in ranked_candidates], dtype=float)
        state_history.append(beam_states.copy())
        back_history.append(np.asarray([item[1][1] for item in ranked_candidates], dtype=np.int32))

    valid_final = [
        index
        for index, state in enumerate(beam_states)
        if all(
            age >= config.duration_rows[ANOMALY_TYPES[type_index]][0] for type_index, age in state
        )
    ]
    if not valid_final:
        raise RuntimeError("decoder beam contains no duration-valid final state")
    best_index = max(valid_final, key=lambda index: (beam_scores[index], -index))
    selected: list[BeamState] = [()] * len(values)
    for row_index in range(len(values) - 1, -1, -1):
        selected[row_index] = state_history[row_index][best_index]
        best_index = int(back_history[row_index][best_index])

    typed = np.zeros((len(values), len(ANOMALY_TYPES)), dtype=np.int8)
    for row_index, state in enumerate(selected):
        for type_index, _ in state:
            typed[row_index, type_index] = 1
    union = typed.any(axis=1).astype(np.int8)
    return union, typed


def decode_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    config: DecoderConfig,
    *,
    threshold: float = 0.5,
    cadence_minutes: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode every gap-safe segment, failing over only invalid unary rows."""

    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(frame), len(ANOMALY_TYPES)):
        raise ValueError("probability shape does not match frame")
    control = rowwise_union(values, threshold=threshold)
    union = np.zeros(len(frame), dtype=np.int8)
    typed = np.zeros((len(frame), len(ANOMALY_TYPES)), dtype=np.int8)
    no_op = np.zeros(len(frame), dtype=bool)
    segments, timestamps = exact_segment_ids(frame, cadence_minutes=cadence_minutes)
    for segment in np.unique(segments):
        positions = np.flatnonzero(segments == segment)
        order = np.argsort(timestamps[positions], kind="stable")
        positions = positions[order]
        finite = np.isfinite(values[positions]).all(axis=1)
        for local_start, local_stop in _true_runs(finite):
            run_positions = positions[local_start : local_stop + 1]
            decoded, decoded_types = decode_segment(values[run_positions], config)
            union[run_positions] = decoded
            typed[run_positions] = decoded_types
        invalid_positions = positions[~finite]
        if len(invalid_positions):
            union[invalid_positions] = control[invalid_positions]
            typed[invalid_positions] = (
                np.where(np.isfinite(values[invalid_positions]), values[invalid_positions], -np.inf)
                >= threshold
            ).astype(np.int8)
            no_op[invalid_positions] = True
    return union, typed, no_op


def binary_counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    if (
        truth.shape != prediction.shape
        or not np.isin(truth, (0, 1)).all()
        or not np.isin(prediction, (0, 1)).all()
    ):
        raise ValueError("truth and prediction must be aligned binary vectors")
    tp = int(((truth == 1) & (prediction == 1)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(((truth == 0) & (prediction == 0)).sum()),
        "f1": float(2 * tp / denominator) if denominator else 0.0,
    }


def recall_by_type(
    membership_counts: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | None]:
    counts = np.asarray(membership_counts)
    prediction = np.asarray(prediction, dtype=np.int8)
    if counts.shape != (len(prediction), len(ANOMALY_TYPES)):
        raise ValueError("membership_counts shape mismatch")
    result: dict[str, float | None] = {}
    for type_index, name in enumerate(ANOMALY_TYPES):
        mask = counts[:, type_index] > 0
        result[name] = float(prediction[mask].mean()) if mask.any() else None
    return result


def station_layer_f1(
    frame: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for (station, layer), indices in frame.groupby(
        ["station", "layer"], sort=True, observed=True
    ).indices.items():
        positions = np.asarray(indices, dtype=np.int64)
        result[f"{station}|{layer}"] = float(
            binary_counts(truth[positions], prediction[positions])["f1"]
        )
    return result


__all__ = [
    "ANOMALY_TYPES",
    "DecoderConfig",
    "GrammarAudit",
    "OFFICIAL_DURATION_ROWS",
    "RawTypeSequence",
    "build_grammar_audit",
    "binary_counts",
    "chronological_split_masks",
    "decode_frame",
    "decode_segment",
    "duration_is_decomposable",
    "exact_segment_ids",
    "parse_raw_anomaly_type",
    "parse_raw_anomaly_types",
    "recall_by_type",
    "rowwise_union",
    "station_layer_f1",
]
