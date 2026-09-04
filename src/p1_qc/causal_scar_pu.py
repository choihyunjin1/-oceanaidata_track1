"""Leakage-safe helpers for the P1 SCAR positive-unlabeled proposal head.

The module deliberately keeps the SCAR assumption explicit.  It estimates the
positive-label propensity only from observed positives in a chronological
inner-calibration tail and never claims that the latent class prior is
identified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


class ContractError(ValueError):
    """Raised when a frozen v21 contract cannot be evaluated safely."""


@dataclass(frozen=True)
class ChronologicalSplit:
    fit_mask: np.ndarray
    calibration_mask: np.ndarray
    cutoff_ns: int


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    additions: int
    true_positives: int
    false_positives: int
    precision: float
    precision_lcb: float
    incumbent_f1: float
    candidate_f1: float


def chronological_inner_split(
    times_ns: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    fit_fraction: float = 0.75,
) -> ChronologicalSplit:
    """Split whole timestamps so equal-time rows cannot straddle the boundary."""
    times = np.asarray(times_ns, dtype=np.int64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if times.ndim != 1 or eligible.shape != times.shape:
        raise ContractError("times and eligible mask must be aligned vectors")
    if not 0.5 <= fit_fraction < 1.0:
        raise ContractError("fit_fraction must be in [0.5, 1)")
    unique_times = np.unique(times[eligible])
    if len(unique_times) < 4:
        raise ContractError("inner split requires at least four unique timestamps")
    cutoff_index = min(len(unique_times) - 1, max(1, int(np.floor(len(unique_times) * fit_fraction))))
    cutoff = int(unique_times[cutoff_index])
    fit_mask = eligible & (times < cutoff)
    calibration_mask = eligible & (times >= cutoff)
    if not fit_mask.any() or not calibration_mask.any():
        raise ContractError("chronological inner split produced an empty side")
    if int(times[fit_mask].max()) >= int(times[calibration_mask].min()):
        raise ContractError("chronological inner split overlaps")
    return ChronologicalSplit(fit_mask=fit_mask, calibration_mask=calibration_mask, cutoff_ns=cutoff)


def estimate_scar_propensity(
    selection_probability: np.ndarray,
    observed_labels: np.ndarray,
    *,
    minimum_positive_support: int = 8,
    lower_clip: float = 0.05,
) -> float:
    """Estimate c=P(observed positive | latent positive) on inner positives."""
    probability = np.asarray(selection_probability, dtype=np.float64)
    labels = np.asarray(observed_labels, dtype=np.int8)
    if probability.ndim != 1 or labels.shape != probability.shape:
        raise ContractError("propensity arrays must be aligned vectors")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ContractError("selection probabilities must be finite in [0,1]")
    positive = labels == 1
    if int(positive.sum()) < minimum_positive_support:
        raise ContractError("insufficient inner positive support for SCAR propensity")
    value = float(probability[positive].mean())
    return float(np.clip(value, lower_clip, 1.0))


def correct_selection_probability(selection_probability: np.ndarray, propensity: float) -> np.ndarray:
    """Apply the frozen Elkan-Noto style p(y=1|x)=p(s=1|x)/c correction."""
    probability = np.asarray(selection_probability, dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ContractError("selection probabilities must be finite in [0,1]")
    if not np.isfinite(propensity) or not 0 < propensity <= 1:
        raise ContractError("propensity must be finite in (0,1]")
    return np.clip(probability / propensity, 0.0, 1.0)


def binary_f1(labels: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.int8)
    candidate = np.asarray(prediction, dtype=np.int8)
    tp = int(((truth == 1) & (candidate == 1)).sum())
    fp = int(((truth == 0) & (candidate == 1)).sum())
    fn = int(((truth == 1) & (candidate == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def wilson_lower(successes: int, total: int, *, confidence: float = 0.90) -> float:
    if total <= 0:
        return 0.0
    if not 0 < confidence < 1:
        raise ContractError("confidence must lie in (0,1)")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float((center - radius) / denominator)


def select_add_only_threshold(
    corrected_probability: np.ndarray,
    observed_labels: np.ndarray,
    incumbent: np.ndarray,
    *,
    maximum_changed_fraction: float = 0.005,
    minimum_additions: int = 1,
) -> ThresholdSelection:
    """Choose an inner-only F1 threshold under a frozen precision-LCB constraint."""
    score = np.asarray(corrected_probability, dtype=np.float64)
    labels = np.asarray(observed_labels, dtype=np.int8)
    anchor = np.asarray(incumbent, dtype=np.int8)
    if score.ndim != 1 or labels.shape != score.shape or anchor.shape != score.shape:
        raise ContractError("threshold arrays must be aligned vectors")
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ContractError("corrected probabilities must be finite in [0,1]")
    if not 0 < maximum_changed_fraction <= 1:
        raise ContractError("maximum_changed_fraction must lie in (0,1]")
    incumbent_f1 = binary_f1(labels, anchor)
    thresholds = np.r_[np.inf, np.unique(score[anchor == 0])[::-1]]
    best: ThresholdSelection | None = None
    for threshold in thresholds:
        added = (anchor == 0) & (score >= threshold)
        additions = int(added.sum())
        if additions < minimum_additions or additions / len(score) > maximum_changed_fraction:
            continue
        tp = int((added & (labels == 1)).sum())
        fp = additions - tp
        precision = tp / additions
        lcb = wilson_lower(tp, additions)
        if not lcb > incumbent_f1 / 2.0:
            continue
        candidate = np.maximum(anchor, added.astype(np.int8))
        candidate_f1 = binary_f1(labels, candidate)
        current = ThresholdSelection(
            threshold=float(threshold),
            additions=additions,
            true_positives=tp,
            false_positives=fp,
            precision=float(precision),
            precision_lcb=lcb,
            incumbent_f1=incumbent_f1,
            candidate_f1=candidate_f1,
        )
        if best is None or (current.candidate_f1, current.threshold, -current.additions) > (
            best.candidate_f1,
            best.threshold,
            -best.additions,
        ):
            best = current
    if best is None:
        return ThresholdSelection(
            threshold=float("inf"),
            additions=0,
            true_positives=0,
            false_positives=0,
            precision=0.0,
            precision_lcb=0.0,
            incumbent_f1=incumbent_f1,
            candidate_f1=incumbent_f1,
        )
    return best


__all__ = [
    "ChronologicalSplit",
    "ContractError",
    "ThresholdSelection",
    "binary_f1",
    "chronological_inner_split",
    "correct_selection_probability",
    "estimate_scar_propensity",
    "select_add_only_threshold",
    "wilson_lower",
]
