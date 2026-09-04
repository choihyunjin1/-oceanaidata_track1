"""Core math for the preregistered P1 add-only event-precision experiment.

The module operates only on an already-authenticated OOF frame.  Event geometry
is constructed without reading labels; labels enter only for past-prefix fitting
and, after candidate acceptance is locked, next-fold evaluation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit


class HeadContractError(RuntimeError):
    """Raised when the fixed statistical or add-only contract is violated."""


@dataclass(frozen=True)
class FittedPrecisionHead:
    """A fixed ridge-binomial head with fit-prefix-only feature encoding."""

    categories: Mapping[str, tuple[str, ...]]
    numeric_center: np.ndarray
    numeric_scale: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray
    ridge_strength: float
    iterations: int
    converged: bool
    feature_names: tuple[str, ...]
    fit_event_count: int
    fit_row_count: int
    fit_added_tp: int
    fit_added_fp: int


def _binary(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.isin(array, [0, 1]).all():
        raise HeadContractError(f"{name} must be a one-dimensional binary vector")
    return array.astype(np.int8, copy=False)


def build_event_bank(
    frame: pd.DataFrame,
    source_names: Sequence[str],
    *,
    event_group: Sequence[str],
    cadence_minutes: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build frozen proposal-event geometry without accessing the label column."""
    required = {
        "station",
        "year",
        "layer",
        "time",
        "fold",
        "anchor",
        *source_names,
        *event_group,
    }
    missing = required - set(frame.columns)
    if missing:
        raise HeadContractError(f"event frame is missing columns: {sorted(missing)}")
    if not source_names or len(source_names) != len(set(source_names)):
        raise HeadContractError("proposal source names must be nonempty and unique")

    anchor = _binary(frame["anchor"].to_numpy(), "anchor")
    source_matrix = np.column_stack(
        [_binary(frame[name].to_numpy(), name) for name in source_names]
    )
    proposal_mask = (anchor == 0) & np.any(source_matrix == 1, axis=1)
    proposed = frame.loc[
        proposal_mask,
        [*dict.fromkeys([*event_group, "station", "year", "layer", "time", "fold"]), *source_names],
    ].copy()
    proposed["__row_index"] = np.flatnonzero(proposal_mask)
    proposed["__time"] = pd.to_datetime(proposed["time"], errors="raise", utc=True)
    proposed = proposed.sort_values([*event_group, "__time"], kind="stable").reset_index(
        drop=True
    )
    cadence = pd.Timedelta(minutes=int(cadence_minutes))
    events: list[dict[str, Any]] = []
    if proposed.empty:
        return events, proposal_mask

    group_changed = proposed.loc[:, list(event_group)].ne(
        proposed.loc[:, list(event_group)].shift()
    ).any(axis=1)
    time_gap = proposed["__time"].diff().ne(cadence)
    event_ids = (group_changed | time_gap).cumsum()
    for ordinal, (_, event) in enumerate(
        proposed.groupby(event_ids, sort=False, observed=True)
    ):
        active = event.loc[:, source_names].to_numpy(dtype=np.int8).any(axis=0)
        signature = "+".join(
            name for name, is_active in zip(source_names, active, strict=True) if is_active
        )
        if not signature:
            raise HeadContractError("proposal event has no active frozen source")
        start = event["__time"].iloc[0]
        end = event["__time"].iloc[-1]
        start_kst = start.tz_convert("Asia/Seoul")
        end_kst = end.tz_convert("Asia/Seoul")
        events.append(
            {
                "event_ordinal": ordinal,
                "fold": str(event["fold"].iloc[0]),
                "station": str(event["station"].iloc[0]),
                "year": int(event["year"].iloc[0]),
                "layer": int(event["layer"].iloc[0]),
                "rows": int(len(event)),
                "active_source_count": int(active.sum()),
                "proposal_type": signature,
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
                "kst_month": start_kst.strftime("%Y-%m"),
                "kst_start_day": start_kst.strftime("%Y-%m-%d"),
                "kst_span_days": int((end_kst.date() - start_kst.date()).days + 1),
                "row_indices": tuple(int(value) for value in event["__row_index"]),
            }
        )
    covered = np.zeros(len(frame), dtype=bool)
    for event in events:
        covered[list(event["row_indices"])] = True
    if not np.array_equal(covered, proposal_mask):
        raise HeadContractError("proposal events do not partition the frozen bank")
    return events, proposal_mask


def _feature_schema(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, ...]], np.ndarray, np.ndarray]:
    if not events:
        raise HeadContractError("a precision head cannot fit zero events")
    categories = {
        name: tuple(sorted({str(event[name]) for event in events}))
        for name in ("station", "layer", "proposal_type")
    }
    numeric = np.asarray(
        [
            [np.log1p(int(event["rows"])), float(event["active_source_count"])]
            for event in events
        ],
        dtype=np.float64,
    )
    center = numeric.mean(axis=0)
    scale = numeric.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return categories, center, scale


def _design_matrix(
    events: Sequence[Mapping[str, Any]],
    categories: Mapping[str, tuple[str, ...]],
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    numeric = np.asarray(
        [
            [np.log1p(int(event["rows"])), float(event["active_source_count"])]
            for event in events
        ],
        dtype=np.float64,
    )
    numeric = (numeric - center) / scale
    columns = [np.ones(len(events), dtype=np.float64), numeric[:, 0], numeric[:, 1]]
    names = ["intercept", "z_log1p_event_rows", "z_active_source_count"]
    for field in ("station", "layer", "proposal_type"):
        observed = np.asarray([str(event[field]) for event in events], dtype=object)
        for level in categories[field]:
            columns.append((observed == level).astype(np.float64))
            names.append(f"{field}={level}")
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise HeadContractError("event design matrix contains nonfinite values")
    return matrix, tuple(names)


def _event_outcomes(
    events: Sequence[Mapping[str, Any]], labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    successes = np.empty(len(events), dtype=np.float64)
    trials = np.empty(len(events), dtype=np.float64)
    for position, event in enumerate(events):
        indices = np.asarray(event["row_indices"], dtype=np.int64)
        values = labels[indices]
        successes[position] = float(values.sum())
        trials[position] = float(len(values))
    if np.any(trials <= 0) or np.any(successes < 0) or np.any(successes > trials):
        raise HeadContractError("invalid aggregated binomial outcomes")
    return successes, trials


def fit_partial_pooling_head(
    events: Sequence[Mapping[str, Any]],
    labels: Any,
    *,
    ridge_strength: float,
    maximum_iterations: int,
    tolerance: float,
) -> FittedPrecisionHead:
    """Fit one fixed aggregated-binomial ridge head (one authorized fit)."""
    labels_array = _binary(labels, "fit-prefix truth")
    categories, center, scale = _feature_schema(events)
    design, names = _design_matrix(events, categories, center, scale)
    successes, trials = _event_outcomes(events, labels_array)
    total_successes = float(successes.sum())
    total_trials = float(trials.sum())
    if total_successes <= 0 or total_successes >= total_trials:
        raise HeadContractError("binomial head requires both positive and negative fit rows")
    ridge = float(ridge_strength)
    if not np.isfinite(ridge) or ridge <= 0:
        raise HeadContractError("ridge strength must be finite and positive")

    beta = np.zeros(design.shape[1], dtype=np.float64)
    smoothed = (total_successes + 0.5) / (total_trials + 1.0)
    beta[0] = np.log(smoothed / (1.0 - smoothed))
    penalty = np.ones(design.shape[1], dtype=np.float64)
    penalty[0] = 0.0

    def objective(candidate: np.ndarray) -> float:
        eta = design @ candidate
        log_likelihood = np.sum(successes * eta - trials * np.logaddexp(0.0, eta))
        return float(log_likelihood - 0.5 * ridge * np.sum(penalty * candidate**2))

    converged = False
    iterations = 0
    for iteration in range(1, int(maximum_iterations) + 1):
        iterations = iteration
        probability = np.clip(expit(design @ beta), 1e-10, 1.0 - 1e-10)
        gradient = design.T @ (successes - trials * probability) - ridge * penalty * beta
        weights = trials * probability * (1.0 - probability)
        hessian = design.T @ (weights[:, None] * design) + np.diag(ridge * penalty)
        hessian += np.eye(hessian.shape[0], dtype=np.float64) * 1e-12
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise HeadContractError("precision-head Hessian is singular") from error
        previous = objective(beta)
        multiplier = 1.0
        candidate = beta + step
        while objective(candidate) < previous and multiplier > 2.0**-20:
            multiplier *= 0.5
            candidate = beta + multiplier * step
        actual_step = candidate - beta
        beta = candidate
        if float(np.max(np.abs(actual_step))) <= float(tolerance):
            converged = True
            break
    if not converged:
        raise HeadContractError("precision head did not converge under the preregistered fit")

    probability = np.clip(expit(design @ beta), 1e-10, 1.0 - 1e-10)
    weights = trials * probability * (1.0 - probability)
    hessian = design.T @ (weights[:, None] * design) + np.diag(ridge * penalty)
    hessian += np.eye(hessian.shape[0], dtype=np.float64) * 1e-12
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError as error:
        raise HeadContractError("precision-head covariance is singular") from error
    if not np.isfinite(beta).all() or not np.isfinite(covariance).all():
        raise HeadContractError("precision head contains nonfinite fitted values")
    return FittedPrecisionHead(
        categories=categories,
        numeric_center=center,
        numeric_scale=scale,
        beta=beta,
        covariance=covariance,
        ridge_strength=ridge,
        iterations=iterations,
        converged=converged,
        feature_names=names,
        fit_event_count=len(events),
        fit_row_count=int(total_trials),
        fit_added_tp=int(total_successes),
        fit_added_fp=int(total_trials - total_successes),
    )


def predict_precision_lcb(
    head: FittedPrecisionHead,
    events: Sequence[Mapping[str, Any]],
    *,
    z_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict from geometry only; event truth is neither accepted nor inspected."""
    design, names = _design_matrix(
        events, head.categories, head.numeric_center, head.numeric_scale
    )
    if names != head.feature_names:
        raise HeadContractError("prediction feature lineage differs from fit")
    eta = design @ head.beta
    variance = np.einsum("ij,jk,ik->i", design, head.covariance, design)
    if np.any(variance < -1e-10):
        raise HeadContractError("precision prediction variance is negative")
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    mean = expit(eta)
    lower = expit(eta - float(z_value) * standard_error)
    if not np.isfinite(mean).all() or not np.isfinite(lower).all():
        raise HeadContractError("precision predictions are nonfinite")
    return mean, lower


def binary_counts(truth: Any, prediction: Any) -> dict[str, int]:
    truth_array = _binary(truth, "metric truth")
    prediction_array = _binary(prediction, "metric prediction")
    if truth_array.shape != prediction_array.shape:
        raise HeadContractError("metric vectors have different shapes")
    return {
        "tp": int(np.sum((truth_array == 1) & (prediction_array == 1))),
        "fp": int(np.sum((truth_array == 0) & (prediction_array == 1))),
        "fn": int(np.sum((truth_array == 1) & (prediction_array == 0))),
        "tn": int(np.sum((truth_array == 0) & (prediction_array == 0))),
        "rows": int(len(truth_array)),
    }


def f1_from_counts(counts: Mapping[str, int]) -> tuple[float, int, int]:
    numerator = 2 * int(counts["tp"])
    denominator = numerator + int(counts["fp"]) + int(counts["fn"])
    return (float(numerator / denominator) if denominator else 0.0, numerator, denominator)


def addonly_algebra_sanity(
    anchor_counts: Mapping[str, int],
    candidate_counts: Mapping[str, int],
    *,
    added_tp: int,
    added_fp: int,
    anchor_positive_removed: int,
) -> dict[str, Any]:
    """Check the exact F1/2 identity with integer arithmetic."""
    anchor_f1, numerator, denominator = f1_from_counts(anchor_counts)
    candidate_f1, _, _ = f1_from_counts(candidate_counts)
    added_rows = int(added_tp) + int(added_fp)
    expected = {
        "tp": int(anchor_counts["tp"]) + int(added_tp),
        "fp": int(anchor_counts["fp"]) + int(added_fp),
        "fn": int(anchor_counts["fn"]) - int(added_tp),
        "tn": int(anchor_counts["tn"]) - int(added_fp),
    }
    count_identity = all(int(candidate_counts[key]) == value for key, value in expected.items())
    utility_numerator = (2 * denominator - numerator) * int(added_tp) - numerator * int(
        added_fp
    )
    delta_sign = (candidate_f1 > anchor_f1) - (candidate_f1 < anchor_f1)
    utility_sign = (utility_numerator > 0) - (utility_numerator < 0)
    if added_rows:
        precision_relation = 2 * denominator * int(added_tp) > numerator * added_rows
        precision_equal = 2 * denominator * int(added_tp) == numerator * added_rows
        relation_sign = 1 if precision_relation else (-1 if not precision_equal else 0)
        proposal_precision = float(int(added_tp) / added_rows)
    else:
        relation_sign = 0
        proposal_precision = None
    passed = (
        int(anchor_positive_removed) == 0
        and count_identity
        and utility_sign == delta_sign
        and relation_sign == delta_sign
    )
    return {
        "pass": passed,
        "anchor_positive_removed_rows": int(anchor_positive_removed),
        "count_identity": count_identity,
        "added_rows": added_rows,
        "added_tp": int(added_tp),
        "added_fp": int(added_fp),
        "proposal_precision": proposal_precision,
        "anchor_f1_over_2": anchor_f1 / 2.0,
        "utility_numerator_exact": int(utility_numerator),
        "utility_sign": utility_sign,
        "f1_delta_sign": delta_sign,
        "precision_relation_sign": relation_sign,
    }


def _diagnostic_breakdown(
    events: Sequence[Mapping[str, Any]], accepted: np.ndarray
) -> dict[str, Any]:
    chosen = [event for event, keep in zip(events, accepted, strict=True) if bool(keep)]

    def counts(field: str) -> dict[str, dict[str, int]]:
        event_counts = Counter(str(event[field]) for event in chosen)
        row_counts = Counter()
        for event in chosen:
            row_counts[str(event[field])] += int(event["rows"])
        return {
            key: {"events": int(event_counts[key]), "rows": int(row_counts[key])}
            for key in sorted(event_counts)
        }

    return {
        "role": "DIAGNOSTIC_ONLY_NO_VETO",
        "station": counts("station"),
        "layer": counts("layer"),
        "kst_month": counts("kst_month"),
        "proposal_type": counts("proposal_type"),
    }


def evaluate_prefix(
    frame: pd.DataFrame,
    events: Sequence[Mapping[str, Any]],
    prefix: Mapping[str, Any],
    head_config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Fit past only, lock acceptances, then open the next-fold outcomes once."""
    labels = _binary(frame["label"].to_numpy(), "OOF truth")
    anchor = _binary(frame["anchor"].to_numpy(), "OOF anchor")
    fit_folds = {str(value) for value in prefix["fit_folds"]}
    next_fold = str(prefix["blind_next_fold"])
    fit_events = [event for event in events if str(event["fold"]) in fit_folds]
    next_events = [event for event in events if str(event["fold"]) == next_fold]
    fit_row_mask = frame["fold"].astype(str).isin(fit_folds).to_numpy()
    next_row_mask = frame["fold"].astype(str).eq(next_fold).to_numpy()
    if not fit_row_mask.any() or not next_row_mask.any() or not fit_events or not next_events:
        raise HeadContractError("chronological prefix has an empty fit or next-fold population")

    fit_anchor_counts = binary_counts(labels[fit_row_mask], anchor[fit_row_mask])
    fit_anchor_f1, _, _ = f1_from_counts(fit_anchor_counts)
    threshold = fit_anchor_f1 / 2.0
    head = fit_partial_pooling_head(
        fit_events,
        labels,
        ridge_strength=float(head_config["ridge_strength"]),
        maximum_iterations=int(head_config["maximum_newton_iterations"]),
        tolerance=float(head_config["newton_tolerance"]),
    )
    predicted_mean, predicted_lcb = predict_precision_lcb(
        head, next_events, z_value=float(head_config["precision_lcb_z"])
    )
    accepted = predicted_lcb > threshold

    # Candidate acceptance is now locked. Only the following evaluation block reads
    # next-fold labels; no outcome can flow back into encoding, fitting, or selection.
    candidate = anchor.copy()
    accepted_indices: list[int] = []
    for event, keep in zip(next_events, accepted, strict=True):
        if keep:
            accepted_indices.extend(int(value) for value in event["row_indices"])
    accepted_index_array = np.asarray(accepted_indices, dtype=np.int64)
    if accepted_index_array.size:
        if np.any(~next_row_mask[accepted_index_array]) or np.any(anchor[accepted_index_array] != 0):
            raise HeadContractError("accepted event escaped its anchor-negative blind fold")
        candidate[accepted_index_array] = 1

    anchor_positive_removed = int(np.sum((anchor == 1) & (candidate == 0)))
    next_truth = labels[next_row_mask]
    next_anchor = anchor[next_row_mask]
    next_candidate = candidate[next_row_mask]
    anchor_counts = binary_counts(next_truth, next_anchor)
    candidate_counts = binary_counts(next_truth, next_candidate)
    added_labels = labels[accepted_index_array] if accepted_index_array.size else np.asarray([], dtype=np.int8)
    added_tp = int(added_labels.sum())
    added_fp = int(len(added_labels) - added_tp)
    algebra = addonly_algebra_sanity(
        anchor_counts,
        candidate_counts,
        added_tp=added_tp,
        added_fp=added_fp,
        anchor_positive_removed=anchor_positive_removed,
    )
    if not algebra["pass"]:
        raise HeadContractError("exact add-only F1/2 algebra sanity failed")
    anchor_f1, anchor_num, anchor_den = f1_from_counts(anchor_counts)
    candidate_f1, candidate_num, candidate_den = f1_from_counts(candidate_counts)
    receipt = {
        "name": str(prefix["name"]),
        "fit_folds": sorted(fit_folds),
        "blind_next_fold": next_fold,
        "fit": {
            "model_fit_count": 1,
            "events": head.fit_event_count,
            "proposal_rows": head.fit_row_count,
            "proposal_tp": head.fit_added_tp,
            "proposal_fp": head.fit_added_fp,
            "feature_dimension": len(head.feature_names),
            "newton_iterations": head.iterations,
            "converged": head.converged,
            "ridge_strength": head.ridge_strength,
            "fit_anchor_f1": fit_anchor_f1,
            "acceptance_threshold_fit_anchor_f1_over_2": threshold,
        },
        "blind_acceptance": {
            "events_scored": len(next_events),
            "events_accepted": int(accepted.sum()),
            "rows_accepted": int(len(accepted_index_array)),
            "predicted_precision_mean_min": float(predicted_mean.min()),
            "predicted_precision_mean_max": float(predicted_mean.max()),
            "predicted_precision_lcb_min": float(predicted_lcb.min()),
            "predicted_precision_lcb_max": float(predicted_lcb.max()),
            "holdout_truth_fields_used_before_acceptance": 0,
            "threshold_search_count": 0,
        },
        "next_fold_metrics": {
            "rows": int(next_row_mask.sum()),
            "anchor_counts": anchor_counts,
            "candidate_counts": candidate_counts,
            "anchor_f1": anchor_f1,
            "anchor_f1_numerator": anchor_num,
            "anchor_f1_denominator": anchor_den,
            "candidate_f1": candidate_f1,
            "candidate_f1_numerator": candidate_num,
            "candidate_f1_denominator": candidate_den,
            "candidate_minus_anchor_f1": candidate_f1 - anchor_f1,
            "accepted_proposal_tp": added_tp,
            "accepted_proposal_fp": added_fp,
            "accepted_proposal_precision": algebra["proposal_precision"],
        },
        "f1_over_2_hard_sanity": algebra,
        "support_diagnostics": _diagnostic_breakdown(next_events, accepted),
    }
    return receipt, candidate, accepted_index_array


def paired_event_preserving_day_bootstrap(
    frame: pd.DataFrame,
    candidate: Any,
    events: Sequence[Mapping[str, Any]],
    evaluation_folds: Sequence[str],
    *,
    replicates: int,
    seed: int,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, Any]:
    """Paired circular day-block bootstrap with complete proposal events."""
    labels = _binary(frame["label"].to_numpy(), "bootstrap truth")
    anchor = _binary(frame["anchor"].to_numpy(), "bootstrap anchor")
    candidate_array = _binary(candidate, "bootstrap candidate")
    fold_set = {str(value) for value in evaluation_folds}
    selected = frame["fold"].astype(str).isin(fold_set).to_numpy()
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise HeadContractError("bootstrap evaluation population is empty")
    timestamps = pd.to_datetime(frame.loc[selected, "time"], errors="raise", utc=True)
    days = np.asarray(timestamps.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d"), dtype=object)
    global_to_local = {int(global_index): local for local, global_index in enumerate(indices)}
    evaluation_events = [event for event in events if str(event["fold"]) in fold_set]
    for event in evaluation_events:
        event_day = str(event["kst_start_day"])
        for global_index in event["row_indices"]:
            local = global_to_local.get(int(global_index))
            if local is None:
                raise HeadContractError("evaluation event escaped the bootstrap population")
            days[local] = event_day

    table = pd.DataFrame(
        {
            "day": days,
            "truth": labels[indices],
            "anchor": anchor[indices],
            "candidate": candidate_array[indices],
        }
    )
    day_vectors: list[list[int]] = []
    ordered_days = sorted(str(value) for value in table["day"].unique())
    for day in ordered_days:
        day_frame = table.loc[table["day"].eq(day)]
        anchor_counts = binary_counts(day_frame["truth"], day_frame["anchor"])
        candidate_counts = binary_counts(day_frame["truth"], day_frame["candidate"])
        day_vectors.append(
            [
                anchor_counts["tp"],
                anchor_counts["fp"],
                anchor_counts["fn"],
                candidate_counts["tp"],
                candidate_counts["fp"],
                candidate_counts["fn"],
            ]
        )
    counts = np.asarray(day_vectors, dtype=np.int64)
    day_count = len(counts)
    block_length = max(
        1,
        max((int(event["kst_span_days"]) for event in evaluation_events), default=1),
    )
    block_length = min(block_length, day_count)
    bootstrap_replicates = int(replicates)
    if bootstrap_replicates <= 0:
        raise HeadContractError("bootstrap replicate count must be positive")
    rng = np.random.default_rng(int(seed))
    number_of_blocks = int(ceil(day_count / block_length))
    starts = rng.integers(0, day_count, size=(bootstrap_replicates, number_of_blocks))
    offsets = np.arange(block_length, dtype=np.int64)
    sampled = (starts[:, :, None] + offsets[None, None, :]) % day_count
    sampled = sampled.reshape(bootstrap_replicates, -1)[:, :day_count]
    totals = counts[sampled].sum(axis=1)

    anchor_denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
    candidate_denominator = 2 * totals[:, 3] + totals[:, 4] + totals[:, 5]
    anchor_f1 = np.divide(
        2 * totals[:, 0],
        anchor_denominator,
        out=np.zeros(bootstrap_replicates, dtype=np.float64),
        where=anchor_denominator != 0,
    )
    candidate_f1 = np.divide(
        2 * totals[:, 3],
        candidate_denominator,
        out=np.zeros(bootstrap_replicates, dtype=np.float64),
        where=candidate_denominator != 0,
    )
    deltas = candidate_f1 - anchor_f1
    point_anchor = binary_counts(labels[selected], anchor[selected])
    point_candidate = binary_counts(labels[selected], candidate_array[selected])
    point = f1_from_counts(point_candidate)[0] - f1_from_counts(point_anchor)[0]
    return {
        "method": "paired_circular_moving_block_bootstrap",
        "paired_unit": "event-preserving_joint_KST_day",
        "evaluation_folds": sorted(fold_set),
        "joint_kst_day_units": day_count,
        "block_length_days": block_length,
        "block_length_rule": "max_inclusive_KST_day_span_of_frozen_evaluation_events_min_1",
        "replicates": bootstrap_replicates,
        "seed": int(seed),
        "point_candidate_minus_anchor_f1": point,
        "lower_one_sided_95": float(np.quantile(deltas, float(lower_quantile))),
        "upper_one_sided_95": float(np.quantile(deltas, float(upper_quantile))),
        "bootstrap_probability_delta_gt_zero": float(np.mean(deltas > 0.0)),
        "complete_event_assignment_verified": True,
    }


def evidence_state(point: float, lower: float, upper: float) -> str:
    if point > 0.0 and lower > 0.0:
        return "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
    if point > 0.0:
        return "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
    if point < 0.0 and upper < 0.0:
        return "PRIMARY_HARM_RESEARCH_ONLY"
    return "INCONCLUSIVE_RESEARCH_ONLY"
