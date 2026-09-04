"""Leakage-resistant selection utilities for R1 interval proposals.

The interval algorithm itself deliberately lives outside this module.  A caller
provides either explicit half-open row intervals or a boolean proposal mask via
``candidate_factory(parameters)``.  This module validates topology, composes
protected rules, and selects a finite grid using labels from an inner validation
block only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .metrics import BinaryCounts, binary_counts, weighted_group_counts


class LeakageError(RuntimeError):
    """Raised when provenance cannot prove that model selection is inner-only."""


@dataclass(frozen=True)
class IntervalCandidate:
    """One half-open interval in positional ``frame.iloc`` coordinates."""

    start: int
    stop: int
    score: float = 1.0
    source: str = "candidate"

    def __post_init__(self) -> None:
        if isinstance(self.start, (bool, np.bool_)) or not isinstance(
            self.start, (int, np.integer)
        ):
            raise TypeError("interval start must be an integer")
        if isinstance(self.stop, (bool, np.bool_)) or not isinstance(self.stop, (int, np.integer)):
            raise TypeError("interval stop must be an integer")
        if int(self.stop) <= int(self.start):
            raise ValueError("interval stop must be greater than start")
        if not math.isfinite(float(self.score)):
            raise ValueError("interval score must be finite")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("interval source must be a non-empty string")
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "stop", int(self.stop))
        object.__setattr__(self, "score", float(self.score))


type IntervalLike = IntervalCandidate | tuple[int, int]
type CandidateOutput = np.ndarray | Sequence[IntervalLike]
type CandidateFactory = Callable[[Mapping[str, Any]], CandidateOutput]


def _normalise_rows(values: Sequence[int] | np.ndarray, *, name: str) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind == "b":
        raise TypeError(f"{name} must contain row positions, not booleans")
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain integer row positions") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer row positions")
    integer = numeric.astype(np.int64)
    if (integer < 0).any():
        raise ValueError(f"{name} cannot contain negative row positions")
    if len(np.unique(integer)) != len(integer):
        raise ValueError(f"{name} contains duplicate row positions")
    return tuple(sorted(int(value) for value in integer))


@dataclass(frozen=True)
class SelectionProvenance:
    """Global row provenance required before an inner-label grid search.

    ``inner_validation_rows`` must correspond one-for-one to the local frame
    passed to :func:`select_interval_grid`.  Row identities are global dataset
    positions, so disjointness remains auditable even though the selection frame
    is a local ``iloc`` subset.
    """

    fit_rows: Sequence[int] | np.ndarray
    inner_validation_rows: Sequence[int] | np.ndarray
    outer_validation_rows: Sequence[int] | np.ndarray = ()
    label_scope: Literal["inner_validation", "outer_validation", "test", "unknown"] = (
        "inner_validation"
    )
    generator_id: str = "unspecified"

    def __post_init__(self) -> None:
        fit = _normalise_rows(self.fit_rows, name="fit_rows")
        inner = _normalise_rows(self.inner_validation_rows, name="inner_validation_rows")
        outer = _normalise_rows(self.outer_validation_rows, name="outer_validation_rows")
        if not fit:
            raise ValueError("fit_rows cannot be empty")
        if not inner:
            raise ValueError("inner_validation_rows cannot be empty")
        allowed_scopes = {"inner_validation", "outer_validation", "test", "unknown"}
        if self.label_scope not in allowed_scopes:
            raise ValueError(f"unsupported label_scope: {self.label_scope!r}")
        if not isinstance(self.generator_id, str) or not self.generator_id.strip():
            raise ValueError("generator_id must be a non-empty string")
        object.__setattr__(self, "fit_rows", fit)
        object.__setattr__(self, "inner_validation_rows", inner)
        object.__setattr__(self, "outer_validation_rows", outer)

    def assert_inner_selection(self, frame_rows: int) -> None:
        """Fail closed unless all selection labels are proven inner-only."""

        if self.label_scope != "inner_validation":
            raise LeakageError(
                "grid selection requires label_scope='inner_validation'; "
                f"received {self.label_scope!r}"
            )
        fit = set(self.fit_rows)
        inner = set(self.inner_validation_rows)
        outer = set(self.outer_validation_rows)
        overlaps = {
            "fit_inner": len(fit & inner),
            "fit_outer": len(fit & outer),
            "inner_outer": len(inner & outer),
        }
        if any(overlaps.values()):
            raise LeakageError(f"provenance row sets overlap: {overlaps}")
        if len(self.inner_validation_rows) != frame_rows:
            raise LeakageError(
                "inner_validation_rows must correspond one-for-one to the selection frame: "
                f"provenance={len(self.inner_validation_rows)}, frame={frame_rows}"
            )


@dataclass(frozen=True)
class GridSelectionResult:
    """Deterministic output of an inner-only finite-grid search."""

    parameters: Mapping[str, Any]
    prediction: np.ndarray
    proposal_mask: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        prediction = np.asarray(self.prediction, dtype=np.int8).copy()
        proposal = np.asarray(self.proposal_mask, dtype=bool).copy()
        if prediction.ndim != 1 or proposal.ndim != 1:
            raise ValueError("result masks must be one-dimensional")
        if prediction.shape != proposal.shape:
            raise ValueError("prediction and proposal masks differ in shape")
        prediction.setflags(write=False)
        proposal.setflags(write=False)
        object.__setattr__(self, "parameters", copy.deepcopy(dict(self.parameters)))
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "proposal_mask", proposal)
        object.__setattr__(self, "diagnostics", copy.deepcopy(dict(self.diagnostics)))


def _binary_mask(values: Sequence[Any] | np.ndarray, length: int, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must be a one-dimensional array of length {length}")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only boolean/0/1 values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain only finite boolean/0/1 values")
    return numeric.astype(bool)


def _segment_ids(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    time_column: str,
    cadence_minutes: int,
) -> np.ndarray:
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be positive")
    required = [*group_columns, time_column]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"missing interval metadata columns: {missing}")
    if len(frame) == 0:
        return np.empty(0, dtype=np.int64)
    time = pd.to_datetime(frame[time_column], errors="coerce", utc=True, format="mixed")
    if time.isna().any():
        raise ValueError("interval timestamps could not be parsed")
    starts = np.zeros(len(frame), dtype=bool)
    starts[0] = True
    for column in group_columns:
        current = frame[column].reset_index(drop=True)
        prior = current.shift(1)
        same = current.eq(prior) | (current.isna() & prior.isna())
        starts[1:] |= ~same.to_numpy(dtype=bool)[1:]
    contiguous = time.reset_index(drop=True).diff().dt.total_seconds().eq(cadence_minutes * 60)
    starts[1:] |= ~contiguous.to_numpy(dtype=bool)[1:]
    return np.cumsum(starts, dtype=np.int64) - 1


def _coerce_interval(value: IntervalLike) -> IntervalCandidate:
    if isinstance(value, IntervalCandidate):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return IntervalCandidate(value[0], value[1])
    raise TypeError(
        "interval output must contain IntervalCandidate objects or (start, stop) tuples"
    )


def intervals_to_mask(
    frame: pd.DataFrame,
    intervals: Sequence[IntervalLike],
    *,
    group_columns: Sequence[str] = ("station", "layer"),
    time_column: str = "time",
    cadence_minutes: int = 10,
) -> np.ndarray:
    """Project intervals to rows and reject every group or observation-gap crossing."""

    segments = _segment_ids(
        frame,
        group_columns=group_columns,
        time_column=time_column,
        cadence_minutes=cadence_minutes,
    )
    mask = np.zeros(len(frame), dtype=bool)
    for raw in intervals:
        interval = _coerce_interval(raw)
        if interval.start < 0 or interval.stop > len(frame):
            raise IndexError(
                f"interval [{interval.start}, {interval.stop}) is outside {len(frame)} rows"
            )
        if segments[interval.start] != segments[interval.stop - 1]:
            raise ValueError(
                f"interval [{interval.start}, {interval.stop}) crosses a group or time gap"
            )
        mask[interval.start : interval.stop] = True
    return mask


def candidate_output_to_mask(
    frame: pd.DataFrame,
    output: CandidateOutput,
    *,
    group_columns: Sequence[str] = ("station", "layer"),
    time_column: str = "time",
    cadence_minutes: int = 10,
) -> np.ndarray:
    """Normalise the injectable interval-or-array contract to a row mask."""

    if isinstance(output, np.ndarray):
        if output.ndim == 1:
            # A row mask cannot fill an unobserved gap.  Its runs are interpreted
            # independently on each segment by downstream diagnostics.
            return _binary_mask(output, len(frame), name="candidate mask")
        if output.ndim == 2 and output.shape[1] == 2:
            if output.dtype.kind == "b":
                raise TypeError("interval array must contain integer row positions")
            try:
                numeric = output.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError("interval array must contain integer row positions") from exc
            if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
                raise ValueError("interval array must contain finite integer row positions")
            intervals = [
                (int(start), int(stop)) for start, stop in numeric.astype(np.int64).tolist()
            ]
            return intervals_to_mask(
                frame,
                intervals,
                group_columns=group_columns,
                time_column=time_column,
                cadence_minutes=cadence_minutes,
            )
        raise ValueError("candidate array must be a row mask or an (n, 2) interval array")
    if isinstance(output, (str, bytes)) or not isinstance(output, Sequence):
        raise TypeError("candidate output must be a finite sequence or numpy array")
    return intervals_to_mask(
        frame,
        output,
        group_columns=group_columns,
        time_column=time_column,
        cadence_minutes=cadence_minutes,
    )


def apply_interval_candidate(
    frame: pd.DataFrame,
    base_prediction: Sequence[Any] | np.ndarray,
    intervals_or_mask: CandidateOutput,
    *,
    spike_protected: Sequence[Any] | np.ndarray | None = None,
    plateau_protected: Sequence[Any] | np.ndarray | None = None,
    group_columns: Sequence[str] = ("station", "layer"),
    time_column: str = "time",
    cadence_minutes: int = 10,
) -> np.ndarray:
    """Union interval proposals with base predictions and protected rule rows."""

    base = _binary_mask(base_prediction, len(frame), name="base_prediction")
    spike = (
        np.zeros(len(frame), dtype=bool)
        if spike_protected is None
        else _binary_mask(spike_protected, len(frame), name="spike_protected")
    )
    plateau = (
        np.zeros(len(frame), dtype=bool)
        if plateau_protected is None
        else _binary_mask(plateau_protected, len(frame), name="plateau_protected")
    )
    proposal = candidate_output_to_mask(
        frame,
        intervals_or_mask,
        group_columns=group_columns,
        time_column=time_column,
        cadence_minutes=cadence_minutes,
    )
    prediction = base | proposal | spike | plateau
    if not prediction[spike | plateau].all():  # defensive invariant
        raise RuntimeError("protected spike/plateau rows were not preserved")
    return prediction.astype(np.int8)


def _normalise_parameter_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("grid parameters must be finite")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_parameter_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_parameter_value(item) for item in value)
    raise TypeError(f"unsupported grid parameter type: {type(value).__name__}")


def _normalise_grid(
    parameter_grid: Sequence[Mapping[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(parameter_grid, (str, bytes)) or not isinstance(parameter_grid, Sequence):
        raise TypeError("parameter_grid must be a finite sequence of mappings")
    if not parameter_grid:
        raise ValueError("parameter_grid cannot be empty")
    normalised: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for raw in parameter_grid:
        if not isinstance(raw, Mapping):
            raise TypeError("each grid candidate must be a mapping")
        parameters = {
            str(key): _normalise_parameter_value(value)
            for key, value in sorted(raw.items(), key=lambda pair: str(pair[0]))
        }
        canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        if canonical in seen:
            raise ValueError(f"parameter_grid contains a duplicate candidate: {canonical}")
        seen.add(canonical)
        normalised.append((canonical, parameters))
    return sorted(normalised, key=lambda item: item[0])


def _row_hash(rows: Sequence[int]) -> str:
    values = np.asarray(rows, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _counts_dict(counts: BinaryCounts) -> dict[str, float]:
    return {key: float(value) for key, value in counts.to_dict().items()}


def _proposal_run_count(mask: np.ndarray, segments: np.ndarray) -> int:
    if not len(mask):
        return 0
    previous_positive = np.r_[False, mask[:-1]]
    same_segment = np.r_[False, segments[1:] == segments[:-1]]
    return int(np.sum(mask & ~(previous_positive & same_segment)))


def select_interval_grid(
    frame: pd.DataFrame,
    truth: Sequence[Any] | np.ndarray,
    base_prediction: Sequence[Any] | np.ndarray,
    parameter_grid: Sequence[Mapping[str, Any]],
    candidate_factory: CandidateFactory,
    *,
    provenance: SelectionProvenance,
    group_weights: Mapping[Any, float] | None = None,
    spike_protected: Sequence[Any] | np.ndarray | None = None,
    plateau_protected: Sequence[Any] | np.ndarray | None = None,
    primary_metric: Literal["micro_f1", "weighted_f1"] = "micro_f1",
    group_columns: Sequence[str] = ("station", "layer"),
    time_column: str = "time",
    cadence_minutes: int = 10,
) -> GridSelectionResult:
    """Select an interval configuration using inner labels and a finite grid only.

    The callback receives only a defensive copy of one parameter mapping; truth
    is never an argument.  On an exact F1 tie, the other F1, micro precision,
    weighted precision, lower micro FP, lower weighted FP, and finally the
    canonical parameter JSON determine the winner in that order.
    """

    if not isinstance(provenance, SelectionProvenance):
        raise TypeError("provenance must be SelectionProvenance")
    # This check intentionally precedes label parsing and callback execution.
    provenance.assert_inner_selection(len(frame))
    if primary_metric not in {"micro_f1", "weighted_f1"}:
        raise ValueError("primary_metric must be 'micro_f1' or 'weighted_f1'")
    if not callable(candidate_factory):
        raise TypeError("candidate_factory must be callable")
    grid = _normalise_grid(parameter_grid)
    target = _binary_mask(truth, len(frame), name="truth").astype(np.int8)
    base = _binary_mask(base_prediction, len(frame), name="base_prediction")
    spike = (
        np.zeros(len(frame), dtype=bool)
        if spike_protected is None
        else _binary_mask(spike_protected, len(frame), name="spike_protected")
    )
    plateau = (
        np.zeros(len(frame), dtype=bool)
        if plateau_protected is None
        else _binary_mask(plateau_protected, len(frame), name="plateau_protected")
    )
    segments = _segment_ids(
        frame,
        group_columns=group_columns,
        time_column=time_column,
        cadence_minutes=cadence_minutes,
    )

    best_rank: tuple[float, ...] | None = None
    best_parameters: dict[str, Any] | None = None
    best_prediction: np.ndarray | None = None
    best_proposal: np.ndarray | None = None
    best_canonical: str | None = None
    rows: list[dict[str, Any]] = []
    for canonical, parameters in grid:
        output = candidate_factory(copy.deepcopy(parameters))
        proposal = candidate_output_to_mask(
            frame,
            output,
            group_columns=group_columns,
            time_column=time_column,
            cadence_minutes=cadence_minutes,
        )
        prediction = (base | proposal | spike | plateau).astype(np.int8)
        micro = binary_counts(target, prediction)
        weighted = (
            micro
            if group_weights is None
            else weighted_group_counts(
                target,
                prediction,
                frame,
                group_weights,
                group_columns=group_columns,
            )
        )
        primary = micro.f1 if primary_metric == "micro_f1" else weighted.f1
        secondary = weighted.f1 if primary_metric == "micro_f1" else micro.f1
        rank = (
            primary,
            secondary,
            micro.precision,
            weighted.precision,
            -micro.fp,
            -weighted.fp,
        )
        candidate_row = {
            "canonical_parameters": canonical,
            "parameters": copy.deepcopy(parameters),
            "micro": _counts_dict(micro),
            "weighted": _counts_dict(weighted),
            "proposal_rows": int(proposal.sum()),
            "proposal_runs": _proposal_run_count(proposal, segments),
            "new_rows_over_base": int((proposal & ~base).sum()),
        }
        rows.append(candidate_row)
        # The grid is canonical-sorted, so retaining the first exact rank tie
        # makes the final parameter tie-break independent of caller grid order.
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_parameters = copy.deepcopy(parameters)
            best_prediction = prediction.copy()
            best_proposal = proposal.copy()
            best_canonical = canonical

    if (
        best_parameters is None
        or best_prediction is None
        or best_proposal is None
        or best_rank is None
        or best_canonical is None
    ):
        raise RuntimeError("interval grid search produced no result")
    selected_row = next(row for row in rows if row["canonical_parameters"] == best_canonical)
    diagnostics = {
        "selection_scope": provenance.label_scope,
        "generator_id": provenance.generator_id,
        "primary_metric": primary_metric,
        "grid_size": len(grid),
        "group_weights_supplied": group_weights is not None,
        "provenance": {
            "fit_rows": len(provenance.fit_rows),
            "inner_validation_rows": len(provenance.inner_validation_rows),
            "outer_validation_rows": len(provenance.outer_validation_rows),
            "fit_rows_sha256": _row_hash(provenance.fit_rows),
            "inner_validation_rows_sha256": _row_hash(provenance.inner_validation_rows),
            "outer_validation_rows_sha256": _row_hash(provenance.outer_validation_rows),
            "all_disjoint": True,
        },
        "protected_rows": {
            "spike": int(spike.sum()),
            "plateau": int(plateau.sum()),
            "union": int((spike | plateau).sum()),
        },
        "selected_canonical_parameters": best_canonical,
        "selected": copy.deepcopy(selected_row),
        "candidates": rows,
        "tie_break_order": [
            primary_metric,
            "weighted_f1" if primary_metric == "micro_f1" else "micro_f1",
            "micro_precision",
            "weighted_precision",
            "micro_fp_ascending",
            "weighted_fp_ascending",
            "canonical_parameters_ascending",
        ],
    }
    return GridSelectionResult(
        parameters=best_parameters,
        prediction=best_prediction,
        proposal_mask=best_proposal,
        diagnostics=diagnostics,
    )


def apply_selected_interval_grid(
    frame: pd.DataFrame,
    base_prediction: Sequence[Any] | np.ndarray,
    selection: GridSelectionResult | Mapping[str, Any],
    candidate_factory: CandidateFactory,
    *,
    spike_protected: Sequence[Any] | np.ndarray | None = None,
    plateau_protected: Sequence[Any] | np.ndarray | None = None,
    group_columns: Sequence[str] = ("station", "layer"),
    time_column: str = "time",
    cadence_minutes: int = 10,
) -> np.ndarray:
    """Apply selected parameters to outer/test rows without accepting labels."""

    if not callable(candidate_factory):
        raise TypeError("candidate_factory must be callable")
    raw_parameters = (
        selection.parameters if isinstance(selection, GridSelectionResult) else selection
    )
    if not isinstance(raw_parameters, Mapping):
        raise TypeError("selection must provide a parameter mapping")
    normalised = _normalise_grid([raw_parameters])[0][1]
    output = candidate_factory(copy.deepcopy(normalised))
    return apply_interval_candidate(
        frame,
        base_prediction,
        output,
        spike_protected=spike_protected,
        plateau_protected=plateau_protected,
        group_columns=group_columns,
        time_column=time_column,
        cadence_minutes=cadence_minutes,
    )


__all__ = [
    "CandidateFactory",
    "CandidateOutput",
    "GridSelectionResult",
    "IntervalCandidate",
    "LeakageError",
    "SelectionProvenance",
    "apply_interval_candidate",
    "apply_selected_interval_grid",
    "candidate_output_to_mask",
    "intervals_to_mask",
    "select_interval_grid",
]
