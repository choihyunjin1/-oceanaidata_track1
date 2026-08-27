"""Exact independent duration decoding for P1 typed unary probabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .typed_factorial_semimarkov import ANOMALY_TYPES, exact_segment_ids, rowwise_union


@dataclass(frozen=True)
class DurationDecoderConfig:
    duration_rows: Mapping[str, tuple[int, int]]
    start_penalty: float = 3.1780538303479458
    stop_penalty: float = 0.0
    probability_clip: float = 1.0e-6

    def __post_init__(self) -> None:
        if set(self.duration_rows) != set(ANOMALY_TYPES):
            raise ValueError("duration_rows must define exactly the five anomaly types")
        if not np.isfinite(self.start_penalty) or self.start_penalty < 0:
            raise ValueError("start_penalty must be finite and non-negative")
        if not np.isfinite(self.stop_penalty) or self.stop_penalty < 0:
            raise ValueError("stop_penalty must be finite and non-negative")
        if not 0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must lie in (0, .5)")
        for minimum, maximum in self.duration_rows.values():
            if minimum <= 0 or maximum < minimum:
                raise ValueError("each duration interval must be positive and ordered")


def decode_binary_segment(
    probabilities: np.ndarray,
    *,
    minimum_duration: int,
    maximum_duration: int,
    start_penalty: float,
    stop_penalty: float = 0.0,
    probability_clip: float = 1.0e-6,
) -> np.ndarray:
    """Exact two-state semi-Markov Viterbi decoding for one finite segment.

    State zero is normal. Positive states encode the current anomaly age from
    one through ``maximum_duration``. A positive event may stop only after its
    minimum duration and is forced to stop at its maximum. Two same-type events
    require at least one intervening normal row; overlap multiplicity is outside
    this duration-only generation's declared scope.
    """

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if len(values) == 0:
        return np.empty(0, dtype=np.int8)
    if not np.isfinite(values).all():
        raise ValueError("decode_binary_segment requires finite probabilities")
    if minimum_duration <= 0 or maximum_duration < minimum_duration:
        raise ValueError("invalid duration interval")
    if not 0 < probability_clip < 0.5:
        raise ValueError("invalid probability clip")

    clipped = np.clip(values, probability_clip, 1.0 - probability_clip)
    logits = np.log(clipped) - np.log1p(-clipped)
    negative_infinity = -np.inf
    previous_normal = 0.0
    previous_active = np.full(maximum_duration + 1, negative_infinity, dtype=float)
    normal_previous_age = np.full(len(values), -1, dtype=np.int32)
    start_previous_age = np.full(len(values), -1, dtype=np.int32)

    for row_index, logit in enumerate(logits):
        eligible = previous_active[minimum_duration : maximum_duration + 1]
        if np.isfinite(eligible).any():
            local_index = int(np.argmax(eligible))
            stopped_age = minimum_duration + local_index
            stopped_score = float(eligible[local_index] - stop_penalty)
        else:
            stopped_age = -1
            stopped_score = negative_infinity

        start_base = previous_normal
        if previous_normal >= stopped_score:
            current_normal = previous_normal
            normal_previous_age[row_index] = -1
        else:
            current_normal = stopped_score
            normal_previous_age[row_index] = stopped_age
        start_previous_age[row_index] = -1

        current_active = np.full_like(previous_active, negative_infinity)
        remaining_rows = len(values) - row_index
        if remaining_rows >= minimum_duration:
            current_active[1] = start_base - start_penalty + float(logit)
        if maximum_duration >= 2:
            current_active[2:] = previous_active[1:maximum_duration] + float(logit)
        previous_normal = current_normal
        previous_active = current_active

    eligible_final = previous_active[minimum_duration : maximum_duration + 1]
    if np.isfinite(eligible_final).any():
        final_local = int(np.argmax(eligible_final))
        final_age = minimum_duration + final_local
        final_active_score = float(eligible_final[final_local])
    else:
        final_age = -1
        final_active_score = negative_infinity
    state_age = final_age if final_active_score > previous_normal else 0
    decoded = np.zeros(len(values), dtype=np.int8)
    for row_index in range(len(values) - 1, -1, -1):
        if state_age > 0:
            decoded[row_index] = 1
            if state_age == 1:
                previous = int(start_previous_age[row_index])
                state_age = 0 if previous == -1 else previous
            else:
                state_age -= 1
        else:
            previous = int(normal_previous_age[row_index])
            state_age = 0 if previous == -1 else previous
    return decoded


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    if not values.any():
        return []
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def decode_independent_types(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    config: DurationDecoderConfig,
    *,
    threshold: float = 0.5,
    cadence_minutes: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode five independent chains gap-safely, then return their union."""

    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(frame), len(ANOMALY_TYPES)):
        raise ValueError("probability shape does not match frame")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    control_types = (np.where(np.isfinite(values), values, -np.inf) >= threshold).astype(np.int8)
    decoded_types = np.zeros_like(control_types, dtype=np.int8)
    no_op = np.zeros_like(control_types, dtype=bool)
    segments, timestamps = exact_segment_ids(frame, cadence_minutes=cadence_minutes)
    for segment in np.unique(segments):
        positions = np.flatnonzero(segments == segment)
        positions = positions[np.argsort(timestamps[positions], kind="stable")]
        for type_index, anomaly_type in enumerate(ANOMALY_TYPES):
            finite = np.isfinite(values[positions, type_index])
            minimum, maximum = config.duration_rows[anomaly_type]
            for local_start, local_stop in _true_runs(finite):
                run_positions = positions[local_start : local_stop + 1]
                decoded_types[run_positions, type_index] = decode_binary_segment(
                    values[run_positions, type_index],
                    minimum_duration=minimum,
                    maximum_duration=maximum,
                    start_penalty=config.start_penalty,
                    stop_penalty=config.stop_penalty,
                    probability_clip=config.probability_clip,
                )
            invalid = positions[~finite]
            if len(invalid):
                decoded_types[invalid, type_index] = control_types[invalid, type_index]
                no_op[invalid, type_index] = True
    union = decoded_types.any(axis=1).astype(np.int8)
    return union, decoded_types, no_op


def same_unary_control(probabilities: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """Named wrapper documenting that the control consumes identical unaries."""

    return rowwise_union(probabilities, threshold=threshold)


__all__ = [
    "DurationDecoderConfig",
    "decode_binary_segment",
    "decode_independent_types",
    "same_unary_control",
]
