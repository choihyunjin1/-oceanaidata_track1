"""Label-free prefix reliability and deterministic daily cap for P1 v30."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p1_qc.causal_scar_pu import ContractError


@dataclass(frozen=True)
class GroupReliability:
    """Frozen score-only reliability receipt for one station-layer group."""

    support: int
    mean_discrepancy: float
    standard_error: float
    upper_bound: float
    global_limit: float
    eligible: bool


def _group_keys(station: np.ndarray, layer: np.ndarray) -> np.ndarray:
    station_value = np.asarray(station)
    layer_value = np.asarray(layer)
    if station_value.ndim != 1 or layer_value.shape != station_value.shape:
        raise ContractError("station and layer keys are not aligned vectors")
    return np.asarray(
        [
            f"{station_name}|{layer_name}"
            for station_name, layer_name in zip(
                station_value,
                layer_value,
                strict=True,
            )
        ],
        dtype=object,
    )


def _source_matrix(source_probabilities: np.ndarray) -> np.ndarray:
    source = np.asarray(source_probabilities, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ContractError("source probability matrix must have exactly three columns")
    if not np.isfinite(source).all() or ((source < 0.0) | (source > 1.0)).any():
        raise ContractError("source probabilities must be finite in [0,1]")
    return source


def fit_label_free_group_reliability(
    calibrated_probability: np.ndarray,
    source_probabilities: np.ndarray,
    station: np.ndarray,
    layer: np.ndarray,
    *,
    minimum_group_rows: int = 256,
    one_sided_z: float = 1.2815515655446004,
    global_absolute_discrepancy_quantile: float = 0.9,
) -> dict[str, GroupReliability]:
    """Estimate group reliability from frozen scores only, without group labels."""
    calibrated = np.asarray(calibrated_probability, dtype=np.float64)
    source = _source_matrix(source_probabilities)
    keys = _group_keys(station, layer)
    if calibrated.ndim != 1 or calibrated.shape[0] != source.shape[0]:
        raise ContractError("calibrated and frozen source probabilities are not aligned")
    if keys.shape != calibrated.shape or not np.isfinite(calibrated).all():
        raise ContractError("label-free reliability inputs are invalid")
    if ((calibrated < 0.0) | (calibrated > 1.0)).any():
        raise ContractError("calibrated probabilities must lie in [0,1]")
    if minimum_group_rows <= 1 or one_sided_z <= 0:
        raise ContractError("invalid group support or one-sided bound")
    if not 0 < global_absolute_discrepancy_quantile < 1:
        raise ContractError("global discrepancy quantile must lie in (0,1)")
    consensus = source.mean(axis=1)
    discrepancy = calibrated - consensus
    global_limit = float(
        np.quantile(
            np.abs(discrepancy),
            global_absolute_discrepancy_quantile,
        )
    )
    receipts: dict[str, GroupReliability] = {}
    for key in np.unique(keys):
        mask = keys == key
        values = discrepancy[mask]
        support = int(mask.sum())
        standard_error = (
            float(np.std(values, ddof=1) / np.sqrt(support))
            if support > 1
            else float("inf")
        )
        mean_discrepancy = float(np.mean(values))
        upper_bound = abs(mean_discrepancy) + one_sided_z * standard_error
        eligible = bool(
            support >= minimum_group_rows
            and np.isfinite(upper_bound)
            and upper_bound <= global_limit
        )
        receipts[str(key)] = GroupReliability(
            support=support,
            mean_discrepancy=mean_discrepancy,
            standard_error=standard_error,
            upper_bound=float(upper_bound),
            global_limit=global_limit,
            eligible=eligible,
        )
    return receipts


def reliability_margin_lower_bound(
    corrected_probability: np.ndarray,
    threshold: float,
    source_probabilities: np.ndarray,
    station: np.ndarray,
    layer: np.ndarray,
    receipts: dict[str, GroupReliability],
    *,
    one_sided_z: float = 1.2815515655446004,
) -> np.ndarray:
    """Return score margin bounds; unknown or unreliable groups map to -inf."""
    corrected = np.asarray(corrected_probability, dtype=np.float64)
    source = _source_matrix(source_probabilities)
    keys = _group_keys(station, layer)
    if corrected.ndim != 1 or corrected.shape[0] != source.shape[0]:
        raise ContractError("outer probabilities are not aligned")
    if keys.shape != corrected.shape or not np.isfinite(corrected).all():
        raise ContractError("outer label-free reliability inputs are invalid")
    if not np.isfinite(threshold) and not np.isinf(threshold):
        raise ContractError("threshold is invalid")
    if one_sided_z <= 0:
        raise ContractError("one-sided bound must be positive")
    source_standard_error = np.std(source, axis=1, ddof=1) / np.sqrt(3.0)
    output = np.full(len(corrected), -np.inf, dtype=np.float64)
    for key in np.unique(keys):
        receipt = receipts.get(str(key))
        if receipt is None or not receipt.eligible:
            continue
        mask = keys == key
        output[mask] = (
            corrected[mask]
            - threshold
            - one_sided_z * source_standard_error[mask]
            - receipt.upper_bound
        )
    return output


def apply_label_free_day_cap(
    proposed: np.ndarray,
    margin_lower_bound: np.ndarray,
    day: np.ndarray,
    *,
    maximum_fraction: float = 0.005,
) -> np.ndarray:
    """Keep the largest label-free margins under an exact per-day row cap."""
    add = np.asarray(proposed, dtype=bool)
    margin = np.asarray(margin_lower_bound, dtype=np.float64)
    day_value = np.asarray(day)
    if add.ndim != 1 or margin.shape != add.shape or day_value.shape != add.shape:
        raise ContractError("day cap arrays are not aligned")
    if np.isnan(margin).any() or not 0 < maximum_fraction <= 0.005:
        raise ContractError("day cap margin or fraction is invalid")
    kept = np.zeros(len(add), dtype=bool)
    positions = np.arange(len(add), dtype=np.int64)
    for current_day in np.unique(day_value):
        rows = positions[day_value == current_day]
        candidates = rows[add[rows]]
        slots = int(np.floor(maximum_fraction * len(rows)))
        if slots <= 0 or not len(candidates):
            continue
        order = np.lexsort((candidates, -margin[candidates]))
        kept[candidates[order[:slots]]] = True
    return kept


__all__ = [
    "GroupReliability",
    "apply_label_free_day_cap",
    "fit_label_free_group_reliability",
    "reliability_margin_lower_bound",
]
