"""Low-dimensional prequential label-shift correction for P1 v28."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p1_qc.causal_scar_pu import ContractError, binary_f1


@dataclass(frozen=True)
class EmReceipt:
    source_prevalence: float
    target_prevalence: float
    iterations: int
    converged: bool
    maximum_update: float


@dataclass(frozen=True)
class ThresholdReceipt:
    threshold: float
    additions: int
    true_positives: int
    false_positives: int
    marginal_precision: float
    incumbent_f1: float
    candidate_f1: float


def frozen_logit_matrix(
    base_probability: np.ndarray,
    peer_probability: np.ndarray,
    e150_probability: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Return exactly three clipped frozen-source logits and no other feature."""
    columns = []
    for raw in (base_probability, peer_probability, e150_probability):
        value = np.asarray(raw, dtype=np.float64)
        if value.ndim != 1:
            raise ContractError("frozen source probabilities must be vectors")
        value = np.where(np.isfinite(value), value, 0.5)
        value = np.clip(value, epsilon, 1.0 - epsilon)
        columns.append(np.log(value) - np.log1p(-value))
    if len({len(column) for column in columns}) != 1:
        raise ContractError("frozen probability vectors are not aligned")
    output = np.column_stack(columns)
    if output.shape[1] != 3 or not np.isfinite(output).all():
        raise ContractError("frozen logit matrix is not finite width three")
    return output


def label_shift_em(
    source_probability: np.ndarray,
    source_prevalence: float,
    *,
    maximum_iterations: int = 200,
    tolerance: float = 1e-10,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, EmReceipt]:
    """Apply Saerens-style prior correction using target covariates only."""
    probability = np.asarray(source_probability, dtype=np.float64)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ContractError("source posterior must be a finite vector")
    if not 0 < source_prevalence < 1:
        raise ContractError("source prevalence must lie in (0,1)")
    if maximum_iterations <= 0 or tolerance <= 0:
        raise ContractError("invalid EM stopping contract")
    source = float(np.clip(source_prevalence, epsilon, 1.0 - epsilon))
    posterior_source = np.clip(probability, epsilon, 1.0 - epsilon)
    target = source
    corrected = posterior_source.copy()
    maximum_update = float("inf")
    converged = False
    iteration = 0
    for _iteration in range(1, maximum_iterations + 1):
        iteration = _iteration
        positive = posterior_source * (target / source)
        negative = (1.0 - posterior_source) * ((1.0 - target) / (1.0 - source))
        corrected = positive / np.maximum(positive + negative, epsilon)
        updated = float(np.clip(corrected.mean(), epsilon, 1.0 - epsilon))
        maximum_update = abs(updated - target)
        target = updated
        if maximum_update <= tolerance:
            converged = True
            break
    receipt = EmReceipt(
        source_prevalence=source,
        target_prevalence=target,
        iterations=iteration,
        converged=converged,
        maximum_update=maximum_update,
    )
    return np.clip(corrected, 0.0, 1.0), receipt


def select_inner_threshold(
    probability: np.ndarray,
    labels: np.ndarray,
    incumbent: np.ndarray,
    *,
    maximum_changed_fraction: float = 0.005,
) -> ThresholdReceipt:
    """Select a central-precision inner F1 threshold; no rank/top-k rule."""
    score = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int8)
    anchor = np.asarray(incumbent, dtype=np.int8)
    if score.ndim != 1 or truth.shape != score.shape or anchor.shape != score.shape:
        raise ContractError("inner threshold arrays are not aligned")
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ContractError("inner probabilities must be finite in [0,1]")
    reference_f1 = binary_f1(truth, anchor)
    best: ThresholdReceipt | None = None
    for threshold in np.r_[np.inf, np.unique(score[anchor == 0])[::-1]]:
        added = (anchor == 0) & (score >= threshold)
        additions = int(added.sum())
        if additions == 0 or additions / len(score) > maximum_changed_fraction:
            continue
        tp = int((added & (truth == 1)).sum())
        fp = additions - tp
        precision = tp / additions
        if not precision > reference_f1 / 2.0:
            continue
        candidate = np.maximum(anchor, added.astype(np.int8))
        candidate_f1 = binary_f1(truth, candidate)
        if not candidate_f1 > reference_f1:
            continue
        current = ThresholdReceipt(
            threshold=float(threshold),
            additions=additions,
            true_positives=tp,
            false_positives=fp,
            marginal_precision=float(precision),
            incumbent_f1=reference_f1,
            candidate_f1=candidate_f1,
        )
        if best is None or (current.candidate_f1, current.threshold, -current.additions) > (
            best.candidate_f1,
            best.threshold,
            -best.additions,
        ):
            best = current
    if best is None:
        return ThresholdReceipt(
            threshold=float("inf"),
            additions=0,
            true_positives=0,
            false_positives=0,
            marginal_precision=0.0,
            incumbent_f1=reference_f1,
            candidate_f1=reference_f1,
        )
    return best


__all__ = [
    "EmReceipt",
    "ThresholdReceipt",
    "frozen_logit_matrix",
    "label_shift_em",
    "select_inner_threshold",
]
