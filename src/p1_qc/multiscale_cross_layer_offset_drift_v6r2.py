"""Corrected, capability-gated science for P1 multiscale Gen6r2.

The canonical bootstrap authenticates this exact source buffer before it is
compiled.  Fit, prediction, scoring, and candidate-union functions additionally
require a live post-lock capability supplied by the authenticated contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

try:
    _CONTEXT = _P1_V6R2_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r2 science requires the authenticated bootstrap") from exc

if not isinstance(_CONTEXT, dict) or _CONTEXT.get("all_owner_roles_authenticated") is not True:
    raise RuntimeError("P1 Gen6r2 science loaded before source authentication")

EXPERIMENT_ID = "p1_multiscale_cross_layer_offset_drift_unary_v6r2"
HYPOTHESIS_ID = "robust_multiscale_cross_layer_offset_drift_unary"
KEY_COLUMNS = ("station", "year", "layer", "time")
INPUT_ONLY_COLUMNS = KEY_COLUMNS + ("temp", "psal", "depth")
REQUIRED_STATION_LAYERS = (
    "G-ORS|1",
    "I-ORS|1",
    "I-ORS|2",
    "I-ORS|3",
    "I-ORS|4",
    "I-ORS|5",
    "I-ORS|6",
    "I-ORS|7",
    "S-ORS|1",
    "S-ORS|2",
    "S-ORS|3",
    "S-ORS|4",
    "S-ORS|5",
    "S-ORS|6",
    "S-ORS|7",
    "S-ORS|8",
)
FOLDS = ("2025_q2", "2025_q3", "2025_q4")
FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
MULTISCALE_ROWS = (48, 96, 192, 384, 576)
MAXIMUM_DEPENDENCY_ROWS = 576
PURGE_ROWS = 1008
PURGE_DAYS = 7
GAP_BREAK_MINUTES = 30
MIN_SLOW_RUN_ROWS = 48
MAX_SLOW_RUN_ROWS = 519
SPIKE_SINGLETON_MAX_ROWS = 1
SPIKE_PROTECTION_RADIUS_ROWS = 6
UNARY_THRESHOLD = 0.5
UNARY_C = 0.25
UNARY_MAX_ITER = 64
UNARY_TOL = 1.0e-6
DETERMINISTIC_SEED = 20260823
BOOTSTRAP_REPLICATES = 5000
ANNUAL_HARMONICS = (1, 2, 3, 4)
DIURNAL_HARMONICS = (1, 2)
SEASONAL_IRLS_ITERATIONS = 8
SEASONAL_HUBER_DELTA = 1.5
SEASONAL_RIDGE = 1.0e-6
MIN_GROUP_FIT_ROWS = 32
FIXED_POSTPROCESS_GOLDEN = {
    "fixture_sha256": "c79c62b2a11d4d8942ddd824815fb7588dd64597c27a0b97f56fc15e14e3c882",
    "2025_q2": "e0b888ec76897d91beef59ffca652fb910c28c57be391ac485ca71307b729e66",
    "2025_q3": "8366e4de39a6897417bfe2f60fcabf45f428c50c03fd2c62702600e0854be723",
    "2025_q4": "9e38cc5f1c0259d10db9a1d7565147fd37519c31a66cd556bdf63b8c24b011e1",
}

BASE_GEOMETRY_FEATURES = (
    "abs_seasonal_residual_z",
    "abs_graph_residual_z",
    "graph_available",
    "peer_count",
)
SCALE_GEOMETRY_KINDS = (
    "level_abs_z",
    "haar_abs_z",
    "slope_abs_z",
    "curvature_abs_z",
    "coherence_deficit_z",
)
GEOMETRY_FEATURES = BASE_GEOMETRY_FEATURES + tuple(
    f"{kind}_{rows}" for rows in MULTISCALE_ROWS for kind in SCALE_GEOMETRY_KINDS
)

INNER_GATE_THRESHOLDS: dict[str, float | int] = {
    "minimum_micro_f1_delta": 0.002,
    "minimum_offset_recall_delta": 0.03,
    "minimum_drift_recall_delta": 0.03,
    "minimum_mean_offset_drift_recall_delta": 0.04,
    "minimum_spike_f1_delta": 0.0,
    "minimum_worst_station_layer_f1_delta": 0.0,
    "maximum_normal_fp_relative_increase": 0.05,
    "minimum_nondegrading_inner_blocks": 3,
    "required_inner_blocks": 3,
}


class ScienceContractError(ValueError):
    """A typed or scientific invariant failed closed."""


def _np_pd() -> tuple[Any, Any]:
    if _CONTEXT.get("all_owner_roles_authenticated") is not True:
        raise PermissionError("numerical import requested before source authentication")
    import numpy as np
    import pandas as pd

    verifier = _CONTEXT.get("verify_numerical_runtime")
    if not callable(verifier):
        raise PermissionError("numerical runtime origin verifier is unavailable")
    verifier()
    return np, pd


def _require_capability(capability: object, entry_name: str) -> None:
    guard = _CONTEXT.get("require_engine_capability")
    if not callable(guard):
        raise PermissionError("P1 Gen6r2 post-lock capability guard is unavailable")
    guard(capability, entry_name)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def deep_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_real_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _is_exact_int(value: object, expected: int | None = None) -> bool:
    return type(value) is int and (expected is None or value == expected)


def _failed_gate(reason: str, checks: dict[str, bool] | None = None) -> dict[str, Any]:
    return {
        "passed": False,
        "checks": checks or {},
        "reason": reason,
        "fallback": "EXACT_INCUMBENT_BYTES",
    }


def strict_inner_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the corrected gate without accepting coercible count values."""

    required = {
        "micro_f1_delta",
        "offset_recall_delta",
        "drift_recall_delta",
        "spike_f1_delta",
        "worst_station_layer_f1_delta",
        "normal_fp_relative_increase",
        "nondegrading_inner_blocks",
        "inner_block_count",
        "both_slow_types_observed",
        "spike_observed",
        "all_required_station_layers_observed",
        "blind_predictions_sealed_before_gate_labels",
    }
    if type(metrics) is not dict or set(metrics) != required:
        return _failed_gate("metric_field_set_differs")
    continuous = (
        "micro_f1_delta",
        "offset_recall_delta",
        "drift_recall_delta",
        "spike_f1_delta",
        "worst_station_layer_f1_delta",
        "normal_fp_relative_increase",
    )
    if not all(_is_real_number(metrics[name]) for name in continuous):
        return _failed_gate("continuous_metric_domain_differs")
    if not _is_exact_int(metrics["nondegrading_inner_blocks"], 3):
        return _failed_gate("nondegrading_inner_blocks_must_be_exact_integer_3")
    if not _is_exact_int(metrics["inner_block_count"], 3):
        return _failed_gate("inner_block_count_must_be_exact_integer_3")
    if any(
        metrics[name] is not True
        for name in (
            "both_slow_types_observed",
            "spike_observed",
            "all_required_station_layers_observed",
            "blind_predictions_sealed_before_gate_labels",
        )
    ):
        return _failed_gate("support_or_commitment_guard_failed")
    offset = float(metrics["offset_recall_delta"])
    drift = float(metrics["drift_recall_delta"])
    checks = {
        "micro_f1_gain": float(metrics["micro_f1_delta"]) >= 0.002,
        "offset_recall_gain": offset >= 0.03,
        "drift_recall_gain": drift >= 0.03,
        "mean_offset_drift_recall_gain": 0.5 * (offset + drift) >= 0.04,
        "spike_f1_nonregression": float(metrics["spike_f1_delta"]) >= 0.0,
        "worst_station_layer_nonregression": (
            float(metrics["worst_station_layer_f1_delta"]) >= 0.0
        ),
        "normal_fp_guard": float(metrics["normal_fp_relative_increase"]) <= 0.05,
        "nondegrading_inner_blocks_exact": metrics["nondegrading_inner_blocks"] == 3,
        "inner_block_count_exact": metrics["inner_block_count"] == 3,
        "support_complete": True,
        "blind_commitment_complete": True,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "reason": "all_checks_passed" if passed else "one_or_more_checks_failed",
        "fallback": "APPLY_FIXED_SLOW_UNARY" if passed else "EXACT_INCUMBENT_BYTES",
    }


def strict_final_curve_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Machine-enforce the five-point final learning-curve gate."""

    required = {
        "fraction_metrics",
        "fold_full_micro_f1_deltas",
        "all_leakage_checks",
        "all_reproducibility_checks",
        "all_commitments_verified",
    }
    if type(summary) is not dict or set(summary) != required:
        return _failed_gate("final_summary_field_set_differs")
    if (
        summary["all_leakage_checks"] is not True
        or summary["all_reproducibility_checks"] is not True
        or summary["all_commitments_verified"] is not True
    ):
        return _failed_gate("final_boolean_guard_failed")
    points = summary["fraction_metrics"]
    if type(points) is not list or len(points) != len(FRACTIONS):
        return _failed_gate("final_fraction_count_differs")
    point_fields = {
        "fraction",
        "micro_f1_delta",
        "ci90",
        "offset_recall_delta",
        "drift_recall_delta",
        "spike_f1_delta",
        "worst_station_layer_f1_delta",
        "bootstrap_replicates",
        "offset_observed",
        "drift_observed",
        "spike_observed",
        "all_required_station_layers_observed",
    }
    typed_points: list[dict[str, Any]] = []
    for expected_fraction, point in zip(FRACTIONS, points, strict=True):
        if type(point) is not dict or set(point) != point_fields:
            return _failed_gate("final_point_field_set_differs")
        if not _is_real_number(point["fraction"]) or float(point["fraction"]) != expected_fraction:
            return _failed_gate("final_fraction_order_differs")
        numeric = (
            "micro_f1_delta",
            "offset_recall_delta",
            "drift_recall_delta",
            "spike_f1_delta",
            "worst_station_layer_f1_delta",
        )
        if not all(_is_real_number(point[name]) for name in numeric):
            return _failed_gate("final_metric_domain_differs")
        ci = point["ci90"]
        if (
            type(ci) is not list
            or len(ci) != 2
            or not all(_is_real_number(value) for value in ci)
            or float(ci[0]) > float(ci[1])
        ):
            return _failed_gate("final_ci_domain_differs")
        if not _is_exact_int(point["bootstrap_replicates"], BOOTSTRAP_REPLICATES):
            return _failed_gate("bootstrap_replicates_must_be_exact_integer_5000")
        if any(
            point[name] is not True
            for name in (
                "offset_observed",
                "drift_observed",
                "spike_observed",
                "all_required_station_layers_observed",
            )
        ):
            return _failed_gate("final_support_incomplete")
        typed_points.append(point)
    fold_deltas = summary["fold_full_micro_f1_deltas"]
    if type(fold_deltas) is not dict or tuple(fold_deltas) != FOLDS:
        return _failed_gate("full_fold_delta_order_differs")
    if not all(_is_real_number(fold_deltas[fold]) for fold in FOLDS):
        return _failed_gate("full_fold_delta_domain_differs")
    late = typed_points[2:]
    full = typed_points[-1]
    offset = float(full["offset_recall_delta"])
    drift = float(full["drift_recall_delta"])
    checks = {
        "all_late_fractions_improve": all(float(point["micro_f1_delta"]) > 0.0 for point in late),
        "full_micro_gain": float(full["micro_f1_delta"]) >= 0.02,
        "full_ci_excludes_zero": float(full["ci90"][0]) > 0.0,
        "another_late_ci_excludes_zero": any(
            float(point["ci90"][0]) > 0.0 for point in typed_points[2:4]
        ),
        "two_outer_folds_improve": sum(float(fold_deltas[fold]) > 0.0 for fold in FOLDS) >= 2,
        "full_offset_gain": offset >= 0.03,
        "full_drift_gain": drift >= 0.03,
        "full_mean_slow_gain": 0.5 * (offset + drift) >= 0.04,
        "late_spike_nonregression": all(float(point["spike_f1_delta"]) >= 0.0 for point in late),
        "late_worst_group_nonregression": all(
            float(point["worst_station_layer_f1_delta"]) >= 0.0 for point in late
        ),
        "bootstrap_exact": all(point["bootstrap_replicates"] == 5000 for point in typed_points),
        "leakage_repro_commitments": True,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "reason": "all_checks_passed" if passed else "one_or_more_checks_failed",
        "fallback": "CURVE_RESEARCH_PASS_NO_CANDIDATE" if passed else "EXACT_INCUMBENT_BYTES",
        "candidate_creation_allowed": False,
        "test_prediction_allowed": False,
        "upload_allowed": False,
    }


def _strict_int64_ids(values: Any, *, label: str, allow_empty: bool = False) -> Any:
    np, _pd = _np_pd()
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("int64") or values.ndim != 1:
        raise ScienceContractError(f"{label} must be an exact one-dimensional int64 ndarray")
    if not allow_empty and len(values) == 0:
        raise ScienceContractError(f"{label} may not be empty")
    if len(values) and (int(values.min()) < 0 or len(np.unique(values)) != len(values)):
        raise ScienceContractError(f"{label} must contain unique nonnegative IDs")
    return values


def _strict_segment_ids(values: Any, *, label: str, allow_empty: bool = False) -> Any:
    np, _pd = _np_pd()
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("int64") or values.ndim != 1:
        raise ScienceContractError(f"{label} must be an exact one-dimensional int64 ndarray")
    if not allow_empty and len(values) == 0:
        raise ScienceContractError(f"{label} may not be empty")
    if len(values) and int(values.min()) < 0:
        raise ScienceContractError(f"{label} must contain nonnegative IDs")
    return values


def int64_ids_sha256(values: Any) -> str:
    np, _pd = _np_pd()
    ids = _strict_int64_ids(values, label="row IDs", allow_empty=True)
    return hashlib.sha256(ids.astype("<i8", copy=False).tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class InnerChronologicalSplitV6R2:
    block: int
    train_ids: Any
    prediction_ids: Any
    train_end_utc: str
    prediction_start_utc: str
    prediction_end_utc: str
    purge_days: int

    def as_audit(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "train_rows": len(self.train_ids),
            "prediction_rows": len(self.prediction_ids),
            "train_ids_sha256": int64_ids_sha256(self.train_ids),
            "prediction_ids_sha256": int64_ids_sha256(self.prediction_ids),
            "train_end_utc": self.train_end_utc,
            "prediction_start_utc": self.prediction_start_utc,
            "prediction_end_utc": self.prediction_end_utc,
            "purge_days": self.purge_days,
        }


def build_three_block_inner_splits(
    *, capability: object, metadata: Any, outer_prefix_ids: Any
) -> tuple[InnerChronologicalSplitV6R2, ...]:
    """Rebuild the pinned Gen5r6 split using input-only timestamps."""

    _require_capability(capability, "build_three_block_inner_splits")
    np, pd = _np_pd()
    if "time" not in metadata.columns:
        raise ScienceContractError("metadata time column is required")
    ids = _strict_int64_ids(outer_prefix_ids, label="outer prefix IDs")
    if int(ids.max()) >= len(metadata):
        raise ScienceContractError("outer prefix ID escaped metadata")
    time_ns = (
        pd.to_datetime(metadata["time"], errors="raise", utc=True, format="mixed")
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    unique_times = np.unique(time_ns[ids])
    if len(unique_times) < 8:
        raise ScienceContractError("outer prefix has too few timestamps")
    boundaries = [int(math.floor(len(unique_times) * value)) for value in (0.25, 0.5, 0.75)]
    if not (0 < boundaries[0] < boundaries[1] < boundaries[2] < len(unique_times)):
        raise ScienceContractError("inner chronological boundaries collapsed")
    starts = [unique_times[index] for index in boundaries]
    stops = [unique_times[boundaries[1]], unique_times[boundaries[2]], unique_times[-1] + 1]
    purge_ns = int(pd.Timedelta(days=PURGE_DAYS).value)
    result: list[InnerChronologicalSplitV6R2] = []
    prior_prediction_ids: list[Any] = []
    for block, (start, stop) in enumerate(zip(starts, stops, strict=True), 1):
        prediction_ids = np.ascontiguousarray(
            ids[(time_ns[ids] >= start) & (time_ns[ids] < stop)], dtype=np.int64
        )
        train_ids = np.ascontiguousarray(ids[time_ns[ids] < start - purge_ns], dtype=np.int64)
        if not len(train_ids) or not len(prediction_ids):
            raise ScienceContractError(f"inner block {block} is empty after purge")
        if np.intersect1d(train_ids, prediction_ids).size:
            raise ScienceContractError("inner train and prediction IDs overlap")
        if int(time_ns[train_ids].max()) >= int(time_ns[prediction_ids].min()) - purge_ns:
            raise ScienceContractError("inner split seven-day purge differs")
        if (
            prior_prediction_ids
            and np.intersect1d(np.concatenate(prior_prediction_ids), prediction_ids).size
        ):
            raise ScienceContractError("inner prediction blocks overlap")
        prior_prediction_ids.append(prediction_ids)
        result.append(
            InnerChronologicalSplitV6R2(
                block=block,
                train_ids=train_ids,
                prediction_ids=prediction_ids,
                train_end_utc=pd.Timestamp(time_ns[train_ids].max(), tz="UTC").isoformat(),
                prediction_start_utc=pd.Timestamp(
                    time_ns[prediction_ids].min(), tz="UTC"
                ).isoformat(),
                prediction_end_utc=pd.Timestamp(
                    time_ns[prediction_ids].max(), tz="UTC"
                ).isoformat(),
                purge_days=PURGE_DAYS,
            )
        )
    return tuple(result)


_FIXED_FOLD_POSTPROCESS = {
    "2025_q2": (0.15, 0.075, 0, 12),
    "2025_q3": (0.2, 0.1, 0, 12),
    "2025_q4": (0.15, 0.075, 6, 6),
}


def mean_seed_incumbent_probability(
    *, capability: object, seed_probabilities: tuple[Any, Any, Any]
) -> Any:
    _require_capability(capability, "mean_seed_incumbent_probability")
    np, _pd = _np_pd()
    if type(seed_probabilities) is not tuple or len(seed_probabilities) != 3:
        raise ScienceContractError("exactly three seed probability arrays are required")
    length: int | None = None
    verified: list[Any] = []
    for values in seed_probabilities:
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.dtype("float32")
            or values.ndim != 1
            or not values.flags.c_contiguous
            or not np.isfinite(values).all()
            or ((values < 0.0) | (values > 1.0)).any()
        ):
            raise ScienceContractError("teacher probability dtype/shape/domain differs")
        if length is None:
            length = len(values)
        elif len(values) != length:
            raise ScienceContractError("teacher probability lengths differ")
        verified.append(values)
    return np.ascontiguousarray(
        np.mean(np.column_stack(verified), axis=1).astype(np.float32, copy=False)
    )


def _ordered_input_arrays(frame: Any) -> tuple[Any, Any, Any, Any]:
    np, pd = _np_pd()
    required = {"station", "layer", "time", "temp"}
    if not required.issubset(frame.columns):
        raise ScienceContractError("fixed incumbent input columns differ")
    work = frame.loc[:, ["station", "layer", "time", "temp"]].copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["__time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(["station", "layer", "__time", "__position"], inplace=True)
    positions = work["__position"].to_numpy(dtype=np.int64)
    group_codes = pd.factorize(
        pd.MultiIndex.from_frame(work.loc[:, ["station", "layer"]]), sort=False
    )[0].astype(np.int64, copy=False)
    time_ns = work["__time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    breaks = np.ones(len(work), dtype=bool)
    if len(work) > 1:
        breaks[1:] = ~(
            (group_codes[1:] == group_codes[:-1]) & (time_ns[1:] - time_ns[:-1] == 600_000_000_000)
        )
    values = pd.to_numeric(work["temp"], errors="coerce").to_numpy(dtype=np.float64)
    return values, group_codes, breaks, positions


def _fixed_hard_rule_masks(frame: Any) -> tuple[Any, Any]:
    np, _pd = _np_pd()
    values, groups, breaks, positions = _ordered_input_arrays(frame)
    plateau_ordered = np.zeros(len(values), dtype=bool)
    run_start = 0
    for position in range(1, len(values) + 1):
        continues = False
        if position < len(values):
            continues = (
                not breaks[position]
                and np.isfinite(values[position])
                and np.isfinite(values[position - 1])
                and values[position] == values[position - 1]
            )
        if continues:
            continue
        if position - run_start >= 6 and run_start < len(values) and np.isfinite(values[run_start]):
            plateau_ordered[run_start:position] = True
        run_start = position
    steps: dict[int, list[float]] = {}
    for position in range(1, len(values)):
        if breaks[position] or not np.isfinite(values[position - 1 : position + 1]).all():
            continue
        steps.setdefault(int(groups[position]), []).append(
            abs(float(values[position] - values[position - 1]))
        )
    scales = {
        group: float(np.median(np.asarray(items, dtype=np.float64))) if items else 0.0
        for group, items in steps.items()
    }
    spike_ordered = np.zeros(len(values), dtype=bool)
    for position in range(1, len(values) - 1):
        if breaks[position] or breaks[position + 1]:
            continue
        left, center, right = values[position - 1 : position + 2]
        if not np.isfinite((left, center, right)).all():
            continue
        jump_left = abs(float(center - left))
        jump_right = abs(float(center - right))
        excursion = min(jump_left, jump_right)
        threshold = max(0.5, 8.0 * scales.get(int(groups[position]), 0.0))
        if excursion >= threshold and abs(float(right - left)) <= 0.35 * excursion:
            spike_ordered[position] = True
    plateau = np.zeros(len(frame), dtype=bool)
    spike = np.zeros(len(frame), dtype=bool)
    plateau[positions] = plateau_ordered
    spike[positions] = spike_ordered
    return plateau, spike


def _hysteresis(probability: Any, high: float, low: float, breaks: Any) -> Any:
    np, _pd = _np_pd()
    candidate = probability >= low
    seed = probability >= high
    result = np.zeros(len(probability), dtype=bool)
    run_start = 0
    for position in range(1, len(probability) + 1):
        continues = (
            position < len(probability)
            and not breaks[position]
            and candidate[position]
            and candidate[position - 1]
        )
        if continues:
            continue
        if candidate[run_start:position].any() and seed[run_start:position].any():
            result[run_start:position] = candidate[run_start:position]
        run_start = position
    return result


def _close_gaps(mask: Any, maximum: int, breaks: Any) -> Any:
    result = mask.copy()
    position = 1
    while maximum and position < len(result) - 1:
        if result[position] or breaks[position]:
            position += 1
            continue
        start = position
        while position < len(result) and not result[position] and not breaks[position]:
            position += 1
        if (
            position < len(result)
            and result[start - 1]
            and result[position]
            and not breaks[position]
            and position - start <= maximum
        ):
            result[start:position] = True
    return result


def _remove_short_runs(mask: Any, minimum: int, preserve: Any, breaks: Any) -> Any:
    result = mask.copy()
    start: int | None = None
    for position in range(len(result) + 1):
        active = position < len(result) and bool(result[position])
        if position < len(result) and breaks[position] and start is not None:
            if position - start < minimum and not preserve[start:position].any():
                result[start:position] = False
            start = None
        if active and start is None:
            start = position
        if start is not None and (position == len(result) or not active):
            if position - start < minimum and not preserve[start:position].any():
                result[start:position] = False
            start = None
    result[preserve] = True
    return result


def fixed_incumbent_postprocess(
    *, capability: object, frame: Any, probabilities: Any, fold: str
) -> Any:
    """Reproduce the pinned incumbent's fixed fold decision without target access."""

    _require_capability(capability, "fixed_incumbent_postprocess")
    np, _pd = _np_pd()
    if tuple(frame.columns) != INPUT_ONLY_COLUMNS:
        raise ScienceContractError("fixed incumbent frame is not the exact input-only projection")
    if fold not in _FIXED_FOLD_POSTPROCESS:
        raise ScienceContractError("unknown fixed fold postprocess")
    if (
        not isinstance(probabilities, np.ndarray)
        or probabilities.dtype != np.dtype("float32")
        or probabilities.shape != (len(frame),)
        or not probabilities.flags.c_contiguous
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0.0) | (probabilities > 1.0)).any()
    ):
        raise ScienceContractError("fixed incumbent probability identity differs")
    plateau, spike = _fixed_hard_rule_masks(frame)
    _values, _groups, breaks, positions = _ordered_input_arrays(frame)
    high, low, maximum_gap, minimum_run = _FIXED_FOLD_POSTPROCESS[fold]
    ordered_probability = probabilities[positions].astype(np.float64, copy=False)
    ordered_plateau = plateau[positions]
    ordered_spike = spike[positions]
    preserve = ordered_spike & (ordered_probability >= high)
    label = _hysteresis(ordered_probability, high, low, breaks)
    label |= ordered_plateau | preserve
    label = _close_gaps(label, maximum_gap, breaks)
    label = _remove_short_runs(label, minimum_run, preserve, breaks)
    result = np.zeros(len(frame), dtype=np.int8)
    result[positions] = label.astype(np.int8)
    return np.ascontiguousarray(result)


def exact_gap_safe_segment_ids(frame: Any) -> Any:
    np, pd = _np_pd()
    missing = sorted(set(("station", "layer", "time")).difference(frame.columns))
    if missing:
        raise ScienceContractError(f"missing segment columns: {missing}")
    if frame.empty:
        return np.empty(0, dtype=np.int64)
    parsed = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    station = frame["station"].astype(str).to_numpy()
    layer = frame["layer"].to_numpy(dtype=np.int64)
    nanos = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    boundary = np.ones(len(frame), dtype=bool)
    if len(frame) > 1:
        minutes = (nanos[1:] - nanos[:-1]) / (60.0 * 1.0e9)
        boundary[1:] = (
            (station[1:] != station[:-1])
            | (layer[1:] != layer[:-1])
            | (minutes <= 0.0)
            | (minutes > GAP_BREAK_MINUTES)
        )
    return np.cumsum(boundary, dtype=np.int64) - 1


def verify_dependency_closed_split(
    *,
    capability: object,
    frame: Any,
    train_ids: Any,
    holdout_ids: Any,
    segment_ids: Any,
) -> dict[str, Any]:
    """Prove row and time embargoes exceed the centered dependency."""

    _require_capability(capability, "verify_dependency_closed_split")
    np, pd = _np_pd()
    if tuple(frame.columns) != INPUT_ONLY_COLUMNS:
        raise ScienceContractError("dependency proof frame is not the exact input-only projection")
    train = _strict_int64_ids(train_ids, label="train IDs")
    holdout = _strict_int64_ids(holdout_ids, label="holdout IDs")
    segments = _strict_segment_ids(segment_ids, label="segment IDs")
    if len(segments) != len(frame):
        raise ScienceContractError("segment vector length differs")
    expected_segments = exact_gap_safe_segment_ids(frame)
    if not np.array_equal(segments, expected_segments):
        raise ScienceContractError("segment IDs differ from the exact input-only gap segmentation")
    if int(max(train.max(), holdout.max())) >= len(frame):
        raise ScienceContractError("split ID escaped the frame")
    if np.intersect1d(train, holdout).size:
        raise ScienceContractError("train and holdout IDs overlap")
    train_by_segment: dict[int, Any] = {}
    holdout_by_segment: dict[int, Any] = {}
    for segment in np.unique(segments[np.concatenate((train, holdout))]):
        train_by_segment[int(segment)] = np.sort(train[segments[train] == segment])
        holdout_by_segment[int(segment)] = np.sort(holdout[segments[holdout] == segment])
    minimum_row_distance: int | None = None
    for segment, left in train_by_segment.items():
        right = holdout_by_segment.get(segment)
        if right is None or not len(left) or not len(right):
            continue
        positions = np.searchsorted(right, left)
        distances: list[int] = []
        valid_right = positions < len(right)
        if valid_right.any():
            distances.extend(np.abs(right[positions[valid_right]] - left[valid_right]).tolist())
        valid_left = positions > 0
        if valid_left.any():
            distances.extend(np.abs(right[positions[valid_left] - 1] - left[valid_left]).tolist())
        if distances:
            local = int(min(distances))
            minimum_row_distance = (
                local if minimum_row_distance is None else min(minimum_row_distance, local)
            )
    if minimum_row_distance is not None and minimum_row_distance <= PURGE_ROWS:
        raise ScienceContractError("row purge does not exceed 1008 rows")
    parsed = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    train_ns = parsed.iloc[train].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    holdout_ns = parsed.iloc[holdout].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    minimum_time_ns: int | None = None
    for segment in set(train_by_segment).intersection(holdout_by_segment):
        left = np.sort(train_ns[segments[train] == segment])
        right = np.sort(holdout_ns[segments[holdout] == segment])
        positions = np.searchsorted(right, left)
        distances = []
        valid_right = positions < len(right)
        if valid_right.any():
            distances.extend(np.abs(right[positions[valid_right]] - left[valid_right]).tolist())
        valid_left = positions > 0
        if valid_left.any():
            distances.extend(np.abs(right[positions[valid_left] - 1] - left[valid_left]).tolist())
        if distances:
            local = int(min(distances))
            minimum_time_ns = local if minimum_time_ns is None else min(minimum_time_ns, local)
    seven_days_ns = PURGE_DAYS * 24 * 60 * 60 * 1_000_000_000
    if minimum_time_ns is not None and minimum_time_ns < seven_days_ns:
        raise ScienceContractError("time purge is shorter than seven days")
    return {
        "passed": True,
        "train_ids_sha256": int64_ids_sha256(train),
        "holdout_ids_sha256": int64_ids_sha256(holdout),
        "segment_ids_sha256": hashlib.sha256(
            segments.astype("<i8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "minimum_row_distance": minimum_row_distance,
        "minimum_time_distance_ns": minimum_time_ns,
        "maximum_dependency_rows": MAXIMUM_DEPENDENCY_ROWS,
        "purge_rows": PURGE_ROWS,
        "purge_days": PURGE_DAYS,
    }


def seasonal_design(time_values: Any) -> Any:
    np, pd = _np_pd()
    parsed = pd.to_datetime(time_values, errors="raise", utc=True, format="mixed")
    seconds = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64).astype(np.float64) / 1e9
    day = seconds / 86_400.0
    columns = [np.ones(len(day), dtype=np.float64)]
    for harmonic in ANNUAL_HARMONICS:
        angle = 2.0 * np.pi * harmonic * day / 365.2425
        columns.extend((np.sin(angle), np.cos(angle)))
    for harmonic in DIURNAL_HARMONICS:
        angle = 2.0 * np.pi * harmonic * day
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def _robust_scale(values: Any) -> float:
    np, _pd = _np_pd()
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 1.0e-6
    center = float(np.median(finite))
    return max(1.4826 * float(np.median(np.abs(finite - center))), 1.0e-6)


def _fixed_huber_irls(design: Any, target: Any) -> tuple[Any, float]:
    np, _pd = _np_pd()
    finite = np.isfinite(target) & np.isfinite(design).all(axis=1)
    x = np.asarray(design[finite], dtype=np.float64)
    y = np.asarray(target[finite], dtype=np.float64)
    if len(y) < MIN_GROUP_FIT_ROWS:
        raise ScienceContractError("seasonal group has too few finite prefix rows")
    ridge = np.eye(x.shape[1], dtype=np.float64) * SEASONAL_RIDGE
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    scale = _robust_scale(y - x @ beta)
    for _ in range(SEASONAL_IRLS_ITERATIONS):
        residual = y - x @ beta
        scale = _robust_scale(residual)
        ratio = np.abs(residual) / (SEASONAL_HUBER_DELTA * scale)
        weights = np.ones_like(ratio)
        large = ratio > 1.0
        weights[large] = 1.0 / ratio[large]
        root = np.sqrt(weights)
        weighted_x = x * root[:, None]
        beta = np.linalg.solve(weighted_x.T @ weighted_x + ridge, weighted_x.T @ (y * root))
    scale = _robust_scale(y - x @ beta)
    return beta, scale


@dataclass(frozen=True)
class RobustSeasonalGraphState:
    train_ids_sha256: str
    split_audit_sha256: str
    seasonal_coefficients: dict[str, tuple[float, ...]]
    seasonal_scales: dict[str, float]
    edge_residual_deltas: dict[str, float]
    edge_residual_scales: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p1_v6r2_robust_seasonal_graph_state.v1",
            "train_ids_sha256": self.train_ids_sha256,
            "split_audit_sha256": self.split_audit_sha256,
            "seasonal_coefficients": {
                key: list(value) for key, value in sorted(self.seasonal_coefficients.items())
            },
            "seasonal_scales": dict(sorted(self.seasonal_scales.items())),
            "edge_residual_deltas": dict(sorted(self.edge_residual_deltas.items())),
            "edge_residual_scales": dict(sorted(self.edge_residual_scales.items())),
        }

    @property
    def state_sha256(self) -> str:
        return deep_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RobustSeasonalGraphState:
        expected = {
            "schema_version",
            "train_ids_sha256",
            "split_audit_sha256",
            "seasonal_coefficients",
            "seasonal_scales",
            "edge_residual_deltas",
            "edge_residual_scales",
        }
        if type(value) is not dict or set(value) != expected:
            raise ScienceContractError("seasonal graph state field set differs")
        if value["schema_version"] != "p1_v6r2_robust_seasonal_graph_state.v1":
            raise ScienceContractError("seasonal graph state schema differs")
        if not all(
            type(value[name]) is str and len(value[name]) == 64
            for name in ("train_ids_sha256", "split_audit_sha256")
        ):
            raise ScienceContractError("seasonal graph state hash domain differs")
        coefficient_width = 1 + 2 * (len(ANNUAL_HARMONICS) + len(DIURNAL_HARMONICS))
        coefficients = value["seasonal_coefficients"]
        scales = value["seasonal_scales"]
        edge_deltas = value["edge_residual_deltas"]
        edge_scales = value["edge_residual_scales"]
        if not all(type(item) is dict for item in (coefficients, scales, edge_deltas, edge_scales)):
            raise ScienceContractError("seasonal graph map domain differs")
        if set(coefficients) != set(scales) or set(edge_deltas) != set(edge_scales):
            raise ScienceContractError("seasonal graph paired key sets differ")
        if any(
            type(vector) is not list
            or len(vector) != coefficient_width
            or not all(_is_real_number(item) for item in vector)
            for vector in coefficients.values()
        ):
            raise ScienceContractError("seasonal coefficient vector domain differs")
        if any(not _is_real_number(item) or float(item) <= 0.0 for item in scales.values()):
            raise ScienceContractError("seasonal scale domain differs")
        if any(not _is_real_number(item) for item in edge_deltas.values()) or any(
            not _is_real_number(item) or float(item) <= 0.0 for item in edge_scales.values()
        ):
            raise ScienceContractError("graph residual state domain differs")
        state = cls(
            train_ids_sha256=value["train_ids_sha256"],
            split_audit_sha256=value["split_audit_sha256"],
            seasonal_coefficients={
                str(key): tuple(float(item) for item in vector)
                for key, vector in coefficients.items()
            },
            seasonal_scales={str(key): float(item) for key, item in scales.items()},
            edge_residual_deltas={str(key): float(item) for key, item in edge_deltas.items()},
            edge_residual_scales={str(key): float(item) for key, item in edge_scales.items()},
        )
        if state.as_dict() != value:
            raise ScienceContractError("seasonal graph state is not canonically reloadable")
        return state


@dataclass(frozen=True)
class FixedSlowUnaryState:
    train_ids_sha256: str
    feature_names: tuple[str, ...]
    robust_center: tuple[float, ...]
    robust_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    negative_rows: int
    positive_rows: int
    optimizer_iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p1_v6r2_fixed_slow_unary_state.v1",
            "train_ids_sha256": self.train_ids_sha256,
            "feature_names": list(self.feature_names),
            "robust_center": list(self.robust_center),
            "robust_scale": list(self.robust_scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "negative_rows": self.negative_rows,
            "positive_rows": self.positive_rows,
            "optimizer_iterations": self.optimizer_iterations,
            "fixed_hyperparameters": {
                "C": 0.25,
                "class_weight": "balanced",
                "max_iter": 64,
                "penalty": "l2",
                "random_state": 20260823,
                "robust_quantile_range": [25.0, 75.0],
                "solver": "lbfgs",
                "tol": 1.0e-6,
            },
        }

    @property
    def state_sha256(self) -> str:
        return deep_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FixedSlowUnaryState:
        expected = {
            "schema_version",
            "train_ids_sha256",
            "feature_names",
            "robust_center",
            "robust_scale",
            "coefficients",
            "intercept",
            "negative_rows",
            "positive_rows",
            "optimizer_iterations",
            "fixed_hyperparameters",
        }
        if type(value) is not dict or set(value) != expected:
            raise ScienceContractError("unary state field set differs")
        if value["schema_version"] != "p1_v6r2_fixed_slow_unary_state.v1":
            raise ScienceContractError("unary state schema differs")
        if type(value["train_ids_sha256"]) is not str or len(value["train_ids_sha256"]) != 64:
            raise ScienceContractError("unary state train ID hash differs")
        if value["feature_names"] != list(GEOMETRY_FEATURES):
            raise ScienceContractError("unary state feature order differs")
        width = len(GEOMETRY_FEATURES)
        vectors = ("robust_center", "robust_scale", "coefficients")
        if any(
            type(value[name]) is not list
            or len(value[name]) != width
            or not all(_is_real_number(item) for item in value[name])
            for name in vectors
        ):
            raise ScienceContractError("unary state vector domain differs")
        if not _is_real_number(value["intercept"]):
            raise ScienceContractError("unary intercept domain differs")
        if not _is_exact_int(value["negative_rows"]) or value["negative_rows"] <= 0:
            raise ScienceContractError("negative row count must be a positive exact integer")
        if not _is_exact_int(value["positive_rows"]) or value["positive_rows"] <= 0:
            raise ScienceContractError("positive row count must be a positive exact integer")
        if (
            not _is_exact_int(value["optimizer_iterations"])
            or not 0 <= value["optimizer_iterations"] <= 64
        ):
            raise ScienceContractError("optimizer iteration count domain differs")
        expected_hyperparameters = {
            "C": 0.25,
            "class_weight": "balanced",
            "max_iter": 64,
            "penalty": "l2",
            "random_state": 20260823,
            "robust_quantile_range": [25.0, 75.0],
            "solver": "lbfgs",
            "tol": 1.0e-6,
        }
        if value["fixed_hyperparameters"] != expected_hyperparameters:
            raise ScienceContractError("fixed unary hyperparameters differ")
        state = cls(
            train_ids_sha256=value["train_ids_sha256"],
            feature_names=tuple(value["feature_names"]),
            robust_center=tuple(float(item) for item in value["robust_center"]),
            robust_scale=tuple(float(item) for item in value["robust_scale"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            intercept=float(value["intercept"]),
            negative_rows=value["negative_rows"],
            positive_rows=value["positive_rows"],
            optimizer_iterations=value["optimizer_iterations"],
        )
        if min(state.robust_scale) <= 0.0:
            raise ScienceContractError("unary robust scale is not positive")
        return state


def _group_key(station: object, layer: object) -> str:
    return f"{station}|{int(layer)}"


def _edge_key(station: object, low: int, high: int) -> str:
    return f"{station}|{low}|{high}"


def fit_robust_seasonal_graph_state(
    *,
    capability: object,
    input_only_frame: Any,
    train_ids: Any,
    split_audit: dict[str, Any],
) -> RobustSeasonalGraphState:
    _require_capability(capability, "fit_robust_seasonal_graph_state")
    np, _pd = _np_pd()
    if tuple(input_only_frame.columns) != INPUT_ONLY_COLUMNS:
        raise ScienceContractError("baseline frame is not the exact input-only projection")
    ids = _strict_int64_ids(train_ids, label="baseline train IDs")
    if int(ids.max()) >= len(input_only_frame):
        raise ScienceContractError("baseline train ID escaped input frame")
    ids_hash = int64_ids_sha256(ids)
    if (
        type(split_audit) is not dict
        or split_audit.get("passed") is not True
        or split_audit.get("train_ids_sha256") != ids_hash
        or split_audit.get("purge_rows") != PURGE_ROWS
    ):
        raise ScienceContractError("baseline split/dependency proof differs")
    prefix = input_only_frame.iloc[ids].copy()
    design = seasonal_design(prefix["time"])
    coefficients: dict[str, tuple[float, ...]] = {}
    scales: dict[str, float] = {}
    residual = np.full(len(prefix), np.nan, dtype=np.float64)
    grouped = prefix.groupby(["station", "layer"], sort=True, observed=True).indices
    for (station, layer), positions_raw in grouped.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        beta, scale = _fixed_huber_irls(
            design[positions], prefix.iloc[positions]["temp"].to_numpy(dtype=np.float64)
        )
        key = _group_key(station, layer)
        coefficients[key] = tuple(float(item) for item in beta)
        scales[key] = float(scale)
        residual[positions] = (
            prefix.iloc[positions]["temp"].to_numpy(dtype=np.float64) - design[positions] @ beta
        )
    prefix["_seasonal_residual"] = residual
    edge_deltas: dict[str, float] = {}
    edge_scales: dict[str, float] = {}
    for station, station_rows in prefix.groupby("station", sort=True, observed=True):
        pivot = station_rows.pivot_table(
            index="time",
            columns="layer",
            values="_seasonal_residual",
            aggfunc="first",
            observed=True,
        )
        layers = sorted(int(item) for item in pivot.columns)
        for low, high in zip(layers[:-1], layers[1:], strict=True):
            delta = pivot[low].to_numpy(dtype=np.float64) - pivot[high].to_numpy(dtype=np.float64)
            finite = delta[np.isfinite(delta)]
            if len(finite) < MIN_GROUP_FIT_ROWS:
                continue
            key = _edge_key(station, low, high)
            edge_deltas[key] = float(np.median(finite))
            edge_scales[key] = _robust_scale(finite)
    return RobustSeasonalGraphState(
        train_ids_sha256=ids_hash,
        split_audit_sha256=deep_sha256(split_audit),
        seasonal_coefficients=coefficients,
        seasonal_scales=scales,
        edge_residual_deltas=edge_deltas,
        edge_residual_scales=edge_scales,
    )


def apply_robust_seasonal_graph_state(
    *, capability: object, input_only_frame: Any, state: RobustSeasonalGraphState
) -> Any:
    _require_capability(capability, "apply_robust_seasonal_graph_state")
    np, pd = _np_pd()
    state = RobustSeasonalGraphState.from_dict(state.as_dict())
    if tuple(input_only_frame.columns) != INPUT_ONLY_COLUMNS:
        raise ScienceContractError("baseline apply frame is not the exact input-only projection")
    design = seasonal_design(input_only_frame["time"])
    seasonal = np.full(len(input_only_frame), np.nan, dtype=np.float64)
    seasonal_scale = np.full(len(input_only_frame), np.nan, dtype=np.float64)
    for (station, layer), positions_raw in input_only_frame.groupby(
        ["station", "layer"], sort=True, observed=True
    ).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        key = _group_key(station, layer)
        if key not in state.seasonal_coefficients:
            continue
        beta = np.asarray(state.seasonal_coefficients[key], dtype=np.float64)
        seasonal[positions] = (
            input_only_frame.iloc[positions]["temp"].to_numpy(dtype=np.float64)
            - design[positions] @ beta
        )
        seasonal_scale[positions] = state.seasonal_scales[key]
    peers: list[list[float]] = [[] for _ in range(len(input_only_frame))]
    peer_scales: list[list[float]] = [[] for _ in range(len(input_only_frame))]
    working = input_only_frame.loc[:, ["station", "layer", "time"]].copy()
    working["_residual"] = seasonal
    working["_row"] = np.arange(len(working), dtype=np.int64)
    for (station, _time), rows in working.groupby(["station", "time"], sort=False, observed=True):
        by_layer = {
            int(layer): (int(row_id), float(residual))
            for row_id, residual, layer in rows.loc[:, ["_row", "_residual", "layer"]].itertuples(
                index=False, name=None
            )
            if math.isfinite(float(residual))
        }
        for edge, delta in state.edge_residual_deltas.items():
            edge_station, low_raw, high_raw = edge.rsplit("|", 2)
            if edge_station != str(station):
                continue
            low, high = int(low_raw), int(high_raw)
            if low not in by_layer or high not in by_layer:
                continue
            low_row, low_value = by_layer[low]
            high_row, high_value = by_layer[high]
            peers[low_row].append(high_value + delta)
            peers[high_row].append(low_value - delta)
            scale = state.edge_residual_scales[edge]
            peer_scales[low_row].append(scale)
            peer_scales[high_row].append(scale)
    peer_count = np.asarray([len(items) for items in peers], dtype=np.float64)
    consensus = np.asarray(
        [float(np.median(items)) if items else np.nan for items in peers], dtype=np.float64
    )
    consensus_scale = np.asarray(
        [float(np.median(items)) if items else np.nan for items in peer_scales],
        dtype=np.float64,
    )
    has_peer = np.isfinite(consensus)
    safe_seasonal_scale = np.where(
        np.isfinite(seasonal_scale), np.maximum(seasonal_scale, 1.0e-6), np.nan
    )
    graph_residual = np.where(has_peer, seasonal - consensus, seasonal)
    graph_scale = np.where(has_peer, np.maximum(consensus_scale, 1.0e-6), safe_seasonal_scale)
    return pd.DataFrame(
        {
            "seasonal_residual_z": seasonal / safe_seasonal_scale,
            "graph_residual_z": graph_residual / graph_scale,
            "graph_available": (has_peer & np.isfinite(graph_residual)).astype(np.float64),
            "peer_count": peer_count,
        },
        index=input_only_frame.index,
    )


def _bounds(segment_ids: Any) -> list[tuple[int, int]]:
    np, _pd = _np_pd()
    changes = np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1
    edges = np.concatenate(([0], changes, [len(segment_ids)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def _rolling_median(values: Any, window: int, *, center: bool) -> Any:
    np, pd = _np_pd()
    return (
        pd.Series(values, dtype=np.float64)
        .rolling(window=window, min_periods=max(3, window // 2), center=center)
        .median()
        .to_numpy(dtype=np.float64)
    )


def build_multiscale_geometry(
    *, capability: object, baseline_projection: Any, segment_ids: Any, row_ids: Any
) -> Any:
    _require_capability(capability, "build_multiscale_geometry")
    np, pd = _np_pd()
    ids = _strict_int64_ids(row_ids, label="geometry row IDs", allow_empty=True)
    segments = _strict_segment_ids(segment_ids, label="geometry segment IDs", allow_empty=True)
    if len(ids) != len(baseline_projection) or len(segments) != len(ids):
        raise ScienceContractError("geometry row/segment length differs")
    required = {"seasonal_residual_z", "graph_residual_z", "graph_available", "peer_count"}
    if not required.issubset(baseline_projection.columns):
        raise ScienceContractError("baseline projection columns differ")
    seasonal = baseline_projection["seasonal_residual_z"].to_numpy(dtype=np.float64)
    graph = baseline_projection["graph_residual_z"].to_numpy(dtype=np.float64)
    output: dict[str, Any] = {
        "abs_seasonal_residual_z": np.nan_to_num(np.abs(seasonal)),
        "abs_graph_residual_z": np.nan_to_num(np.abs(graph)),
        "graph_available": np.nan_to_num(
            baseline_projection["graph_available"].to_numpy(dtype=np.float64)
        ),
        "peer_count": np.nan_to_num(baseline_projection["peer_count"].to_numpy(dtype=np.float64)),
    }
    for rows in MULTISCALE_ROWS:
        values = {name: np.zeros(len(ids), dtype=np.float64) for name in SCALE_GEOMETRY_KINDS}
        for start, end in _bounds(segments):
            local = seasonal[start:end]
            local_graph = graph[start:end]
            center = _rolling_median(local, max(3, rows // 4), center=True)
            left = np.roll(_rolling_median(local, rows, center=False), 1)
            right = np.roll(_rolling_median(local[::-1], rows, center=False)[::-1], -1)
            if len(local):
                left[0] = np.nan
                right[-1] = np.nan
            values["level_abs_z"][start:end] = _rolling_median(np.abs(local), rows, center=True)
            values["haar_abs_z"][start:end] = np.abs(center - 0.5 * (left + right))
            values["slope_abs_z"][start:end] = np.abs(right - left) / float(2 * rows)
            values["curvature_abs_z"][start:end] = np.abs(left - 2.0 * center + right)
            values["coherence_deficit_z"][start:end] = _rolling_median(
                np.abs(local_graph), rows, center=True
            )
        for name, array in values.items():
            output[f"{name}_{rows}"] = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    result = pd.DataFrame(output, index=ids).loc[:, list(GEOMETRY_FEATURES)].astype(np.float32)
    if not np.isfinite(result.to_numpy(dtype=np.float32)).all():
        raise ScienceContractError("geometry contains nonfinite values")
    return result


def fit_fixed_slow_unary_head(
    *,
    capability: object,
    train_geometry: Any,
    decoded_train_target: Any,
    explicit_train_ids: Any,
    baseline_state: RobustSeasonalGraphState,
) -> FixedSlowUnaryState:
    _require_capability(capability, "fit_fixed_slow_unary_head")
    np, _pd = _np_pd()
    ids = _strict_int64_ids(explicit_train_ids, label="unary train IDs")
    if tuple(train_geometry.columns) != GEOMETRY_FEATURES:
        raise ScienceContractError("unary feature order differs")
    index = train_geometry.index.to_numpy()
    if index.dtype != np.dtype("int64") or not np.array_equal(index, ids):
        raise ScienceContractError("geometry rows are not the exact explicit train IDs")
    ids_hash = int64_ids_sha256(ids)
    if ids_hash != baseline_state.train_ids_sha256:
        raise ScienceContractError("baseline/scaler/unary train ID hash differs")
    target = decoded_train_target
    if (
        not isinstance(target, np.ndarray)
        or target.dtype != np.dtype("int8")
        or target.ndim != 1
        or len(target) != len(ids)
        or not np.isin(target, [0, 1]).all()
        or len(np.unique(target)) != 2
    ):
        raise ScienceContractError("decoded unary target domain differs")
    values = train_geometry.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ScienceContractError("unary geometry contains nonfinite values")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler

    verifier = _CONTEXT.get("verify_numerical_runtime")
    if not callable(verifier):
        raise PermissionError("sklearn runtime origin verifier is unavailable")
    verifier()

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        unit_variance=False,
        copy=True,
    )
    scaled = scaler.fit_transform(values)
    model = LogisticRegression(
        penalty="l2",
        C=UNARY_C,
        solver="lbfgs",
        tol=UNARY_TOL,
        max_iter=UNARY_MAX_ITER,
        class_weight="balanced",
        fit_intercept=True,
        random_state=DETERMINISTIC_SEED,
    )
    model.fit(scaled, target)
    state = FixedSlowUnaryState(
        train_ids_sha256=ids_hash,
        feature_names=GEOMETRY_FEATURES,
        robust_center=tuple(float(item) for item in scaler.center_),
        robust_scale=tuple(float(item) for item in scaler.scale_),
        coefficients=tuple(float(item) for item in model.coef_[0]),
        intercept=float(model.intercept_[0]),
        negative_rows=int(np.count_nonzero(target == 0)),
        positive_rows=int(np.count_nonzero(target == 1)),
        optimizer_iterations=int(model.n_iter_[0]),
    )
    return FixedSlowUnaryState.from_dict(state.as_dict())


def predict_fixed_slow_unary_probability(
    *, capability: object, geometry: Any, state: FixedSlowUnaryState
) -> Any:
    _require_capability(capability, "predict_fixed_slow_unary_probability")
    np, _pd = _np_pd()
    validated = FixedSlowUnaryState.from_dict(state.as_dict())
    if tuple(geometry.columns) != validated.feature_names:
        raise ScienceContractError("inference feature order differs")
    values = geometry.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ScienceContractError("inference geometry contains nonfinite values")
    center = np.asarray(validated.robust_center, dtype=np.float64)
    scale = np.asarray(validated.robust_scale, dtype=np.float64)
    coefficients = np.asarray(validated.coefficients, dtype=np.float64)
    logits = ((values - center) / scale) @ coefficients + validated.intercept
    probability = np.empty(len(logits), dtype=np.float64)
    nonnegative = logits >= 0.0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logits = np.exp(logits[~nonnegative])
    probability[~nonnegative] = exp_logits / (1.0 + exp_logits)
    if not np.isfinite(probability).all():
        raise ScienceContractError("inference produced nonfinite probability")
    return probability


def _positive_runs(mask: Any, segments: Any) -> list[tuple[int, int]]:
    np, _pd = _np_pd()
    runs: list[tuple[int, int]] = []
    for start, end in _bounds(segments):
        local = np.asarray(mask[start:end], dtype=bool)
        transitions = np.diff(np.concatenate(([False], local, [False])).astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        runs.extend((start + int(a), start + int(b)) for a, b in zip(starts, ends, strict=True))
    return runs


def protected_incumbent_union(
    *,
    capability: object,
    incumbent_probability: Any,
    incumbent_prediction: Any,
    gate_passed: bool,
    slow_probability: Any = None,
    segment_ids: Any = None,
) -> tuple[Any, Any, Any]:
    """Return exact incumbent first on every failed/unavailable path."""

    _require_capability(capability, "protected_incumbent_union")
    np, _pd = _np_pd()
    if (
        not isinstance(incumbent_probability, np.ndarray)
        or incumbent_probability.dtype != np.dtype("float32")
        or incumbent_probability.ndim != 1
        or not incumbent_probability.flags.c_contiguous
        or not np.isfinite(incumbent_probability).all()
        or ((incumbent_probability < 0.0) | (incumbent_probability > 1.0)).any()
    ):
        raise ScienceContractError("incumbent probability identity differs")
    if (
        not isinstance(incumbent_prediction, np.ndarray)
        or incumbent_prediction.dtype != np.dtype("int8")
        or incumbent_prediction.ndim != 1
        or not incumbent_prediction.flags.c_contiguous
        or len(incumbent_prediction) != len(incumbent_probability)
        or not np.isin(incumbent_prediction, [0, 1]).all()
    ):
        raise ScienceContractError("incumbent prediction identity differs")
    fallback_probability = incumbent_probability.copy(order="C")
    fallback_prediction = incumbent_prediction.copy(order="C")
    fallback_additions = np.zeros(len(incumbent_prediction), dtype=bool)
    if gate_passed is not True:
        if (
            fallback_probability.dtype != incumbent_probability.dtype
            or fallback_probability.shape != incumbent_probability.shape
            or fallback_probability.tobytes(order="C") != incumbent_probability.tobytes(order="C")
            or fallback_prediction.dtype != incumbent_prediction.dtype
            or fallback_prediction.shape != incumbent_prediction.shape
            or fallback_prediction.tobytes(order="C") != incumbent_prediction.tobytes(order="C")
        ):
            raise AssertionError("incumbent fallback identity changed")
        return fallback_probability, fallback_prediction, fallback_additions
    if slow_probability is None or segment_ids is None:
        return fallback_probability, fallback_prediction, fallback_additions
    if (
        not isinstance(slow_probability, np.ndarray)
        or slow_probability.ndim != 1
        or len(slow_probability) != len(incumbent_prediction)
        or slow_probability.dtype not in {np.dtype("float32"), np.dtype("float64")}
        or not np.isfinite(slow_probability).all()
    ):
        return fallback_probability, fallback_prediction, fallback_additions
    if (
        not isinstance(segment_ids, np.ndarray)
        or segment_ids.dtype != np.dtype("int64")
        or segment_ids.ndim != 1
        or len(segment_ids) != len(incumbent_prediction)
    ):
        return fallback_probability, fallback_prediction, fallback_additions
    singleton_block = np.zeros(len(segment_ids), dtype=bool)
    for start, end in _positive_runs(incumbent_prediction == 1, segment_ids):
        if end - start <= SPIKE_SINGLETON_MAX_ROWS:
            segment_start = start
            while segment_start > 0 and segment_ids[segment_start - 1] == segment_ids[start]:
                segment_start -= 1
            segment_end = end
            while segment_end < len(segment_ids) and segment_ids[segment_end] == segment_ids[start]:
                segment_end += 1
            singleton_block[
                max(segment_start, start - SPIKE_PROTECTION_RADIUS_ROWS) : min(
                    segment_end, end + SPIKE_PROTECTION_RADIUS_ROWS
                )
            ] = True
    proposal = (
        (slow_probability >= UNARY_THRESHOLD) & (incumbent_prediction == 0) & ~singleton_block
    )
    additions = np.zeros(len(segment_ids), dtype=bool)
    for start, end in _positive_runs(proposal, segment_ids):
        if MIN_SLOW_RUN_ROWS <= end - start <= MAX_SLOW_RUN_ROWS:
            additions[start:end] = True
    candidate_probability = fallback_probability
    candidate_prediction = fallback_prediction
    candidate_prediction[additions] = 1
    floor = np.nextafter(np.float32(UNARY_THRESHOLD), np.float32(np.inf), dtype=np.float32)
    candidate_probability[additions] = np.maximum(
        slow_probability[additions].astype(np.float32, copy=False), floor
    )
    incumbent_positive = incumbent_prediction == 1
    if candidate_probability[incumbent_positive].tobytes(order="C") != incumbent_probability[
        incumbent_positive
    ].tobytes(order="C") or candidate_prediction[incumbent_positive].tobytes(
        order="C"
    ) != incumbent_prediction[incumbent_positive].tobytes(order="C"):
        raise AssertionError("incumbent positive bytes changed")
    return candidate_probability, candidate_prediction, additions


def _binary_vector(values: Any, *, label: str) -> Any:
    np, _pd = _np_pd()
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 1
        or values.dtype not in {np.dtype("int8"), np.dtype("bool")}
        or not np.isin(values, [0, 1]).all()
    ):
        raise ScienceContractError(f"{label} must be a binary int8/bool ndarray")
    return values.astype(np.int8, copy=False)


def _binary_f1_unchecked(truth: Any, prediction: Any) -> float:
    np, _pd = _np_pd()
    y = _binary_vector(truth, label="truth")
    p = _binary_vector(prediction, label="prediction")
    if len(y) != len(p):
        raise ScienceContractError("F1 vector lengths differ")
    tp = int(np.count_nonzero((y == 1) & (p == 1)))
    fp = int(np.count_nonzero((y == 0) & (p == 1)))
    fn = int(np.count_nonzero((y == 1) & (p == 0)))
    denominator = 2 * tp + fp + fn
    return 2.0 * tp / denominator if denominator else 0.0


def score_candidate_delta(
    *,
    capability: object,
    truth: Any,
    anomaly_type: Any,
    station_layer: Any,
    segment_ids: Any,
    incumbent_prediction: Any,
    candidate_prediction: Any,
) -> dict[str, Any]:
    _require_capability(capability, "score_candidate_delta")
    np, _pd = _np_pd()
    y = _binary_vector(truth, label="truth")
    incumbent = _binary_vector(incumbent_prediction, label="incumbent prediction")
    candidate = _binary_vector(candidate_prediction, label="candidate prediction")
    if not (len(y) == len(incumbent) == len(candidate) == len(anomaly_type) == len(station_layer)):
        raise ScienceContractError("score vector lengths differ")
    anomaly = np.asarray(anomaly_type, dtype=object)
    groups = np.asarray(station_layer, dtype=object)
    segments = _strict_segment_ids(segment_ids, label="score segment IDs", allow_empty=True)
    if len(segments) != len(y):
        raise ScienceContractError("score segment length differs")
    metrics: dict[str, Any] = {
        "micro_f1_delta": _binary_f1_unchecked(y, candidate) - _binary_f1_unchecked(y, incumbent)
    }
    observed: dict[str, bool] = {}
    for kind in ("offset", "drift"):
        mask = (y == 1) & np.asarray([kind in str(item) for item in anomaly], dtype=bool)
        observed[kind] = bool(mask.any())
        if not observed[kind]:
            metrics[f"{kind}_recall_delta"] = None
        else:
            metrics[f"{kind}_recall_delta"] = float(candidate[mask].mean() - incumbent[mask].mean())
    spike_core = (y == 1) & np.asarray(["spike" in str(item) for item in anomaly], dtype=bool)
    observed["spike"] = bool(spike_core.any())
    spike_union = spike_core.copy()
    for row in np.flatnonzero(spike_core):
        segment = segments[row]
        lo = row
        while lo > 0 and segments[lo - 1] == segment and row - lo < 6:
            lo -= 1
        hi = row + 1
        while hi < len(y) and segments[hi] == segment and hi - row <= 6:
            hi += 1
        spike_union[lo:hi] = True
    metrics["spike_f1_delta"] = (
        _binary_f1_unchecked(y[spike_union], candidate[spike_union])
        - _binary_f1_unchecked(y[spike_union], incumbent[spike_union])
        if observed["spike"]
        else None
    )
    group_deltas: dict[str, float] = {}
    for required in REQUIRED_STATION_LAYERS:
        mask = groups == required
        if not mask.any() or len(np.unique(y[mask])) != 2:
            continue
        group_deltas[required] = _binary_f1_unchecked(
            y[mask], candidate[mask]
        ) - _binary_f1_unchecked(y[mask], incumbent[mask])
    support_complete = set(group_deltas) == set(REQUIRED_STATION_LAYERS)
    metrics["worst_station_layer_f1_delta"] = (
        min(group_deltas.values()) if support_complete else None
    )
    normal = y == 0
    incumbent_fp = int(np.count_nonzero(incumbent[normal] == 1))
    candidate_fp = int(np.count_nonzero(candidate[normal] == 1))
    metrics["normal_fp_relative_increase"] = (candidate_fp - incumbent_fp) / max(1, incumbent_fp)
    return {
        "metrics": metrics,
        "offset_observed": observed["offset"],
        "drift_observed": observed["drift"],
        "spike_observed": observed["spike"],
        "all_required_station_layers_observed": support_complete,
        "station_layer_deltas": group_deltas,
    }


def paired_bootstrap_f1_delta_ci90(
    *,
    capability: object,
    truth: Any,
    incumbent_prediction: Any,
    candidate_prediction: Any,
    bootstrap_unit_ids: Any,
    replicates: int,
    seed: int,
) -> list[float]:
    _require_capability(capability, "paired_bootstrap_f1_delta_ci90")
    np, _pd = _np_pd()
    if not _is_exact_int(replicates, BOOTSTRAP_REPLICATES):
        raise ScienceContractError("bootstrap replicate count must be exact integer 5000")
    if not _is_exact_int(seed):
        raise ScienceContractError("bootstrap seed must be an exact integer")
    y = _binary_vector(truth, label="bootstrap truth")
    incumbent = _binary_vector(incumbent_prediction, label="bootstrap incumbent")
    candidate = _binary_vector(candidate_prediction, label="bootstrap candidate")
    units = np.asarray(bootstrap_unit_ids)
    if units.ndim != 1 or not (len(units) == len(y) == len(incumbent) == len(candidate)):
        raise ScienceContractError("bootstrap unit/vector lengths differ")
    unique = np.unique(units)
    if len(unique) < 2:
        raise ScienceContractError("bootstrap requires at least two units")
    rows = {unit: np.flatnonzero(units == unit) for unit in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([rows[unit] for unit in sampled])
        deltas[index] = _binary_f1_unchecked(y[take], candidate[take]) - _binary_f1_unchecked(
            y[take], incumbent[take]
        )
    low, high = np.quantile(deltas, [0.05, 0.95])
    return [float(low), float(high)]


def static_contract_audit() -> dict[str, Any]:
    passing = {
        "micro_f1_delta": 0.01,
        "offset_recall_delta": 0.05,
        "drift_recall_delta": 0.05,
        "spike_f1_delta": 0.0,
        "worst_station_layer_f1_delta": 0.0,
        "normal_fp_relative_increase": 0.0,
        "nondegrading_inner_blocks": 3,
        "inner_block_count": 3,
        "both_slow_types_observed": True,
        "spike_observed": True,
        "all_required_station_layers_observed": True,
        "blind_predictions_sealed_before_gate_labels": True,
    }
    if strict_inner_gate(passing)["passed"] is not True:
        raise AssertionError("corrected exact inner gate rejected its boundary")
    rejected: dict[str, bool] = {}
    for label, value in (
        ("fractional", 3.9),
        ("float", 3.0),
        ("string", "3"),
        ("boolean", True),
    ):
        probe = dict(passing)
        probe["inner_block_count"] = value
        rejected[label] = strict_inner_gate(probe)["passed"] is False
    regression = dict(passing)
    regression["worst_station_layer_f1_delta"] = -0.000001
    rejected["worst_station_layer_regression"] = strict_inner_gate(regression)["passed"] is False
    if not all(rejected.values()):
        raise AssertionError("corrected fail-closed gate audit failed")
    return {
        "experiment_id": EXPERIMENT_ID,
        "feature_count": len(GEOMETRY_FEATURES),
        "minimum_worst_station_layer_f1_delta": 0.0,
        "strict_count_rejections": rejected,
        "final_gate_implemented": True,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "fixed_postprocess_golden": dict(FIXED_POSTPROCESS_GOLDEN),
        "fits": 0,
        "predictions": 0,
        "scores": 0,
        "target_decodes": 0,
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "EXPERIMENT_ID",
    "FOLDS",
    "FRACTIONS",
    "GEOMETRY_FEATURES",
    "HYPOTHESIS_ID",
    "INNER_GATE_THRESHOLDS",
    "MAXIMUM_DEPENDENCY_ROWS",
    "PURGE_ROWS",
    "REQUIRED_STATION_LAYERS",
    "FixedSlowUnaryState",
    "InnerChronologicalSplitV6R2",
    "RobustSeasonalGraphState",
    "apply_robust_seasonal_graph_state",
    "build_multiscale_geometry",
    "build_three_block_inner_splits",
    "exact_gap_safe_segment_ids",
    "fixed_incumbent_postprocess",
    "fit_fixed_slow_unary_head",
    "fit_robust_seasonal_graph_state",
    "int64_ids_sha256",
    "mean_seed_incumbent_probability",
    "paired_bootstrap_f1_delta_ci90",
    "predict_fixed_slow_unary_probability",
    "protected_incumbent_union",
    "score_candidate_delta",
    "static_contract_audit",
    "strict_final_curve_gate",
    "strict_inner_gate",
    "verify_dependency_closed_split",
]
