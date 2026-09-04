"""Train-prefix-only continuous shrinkage of label-shift prior correction."""

from __future__ import annotations

import numpy as np

from p1_qc.causal_scar_pu import ContractError


def _logit(value: float, epsilon: float) -> float:
    clipped = float(np.clip(value, epsilon, 1.0 - epsilon))
    return float(np.log(clipped) - np.log1p(-clipped))


def shrink_lambda(source: float, em_target: float, observed_target: float, *, epsilon: float = 1e-6) -> float:
    denominator = _logit(em_target, epsilon) - _logit(source, epsilon)
    if abs(denominator) <= 1e-12:
        return 0.0
    value = (_logit(observed_target, epsilon) - _logit(source, epsilon)) / denominator
    return float(np.clip(value, 0.0, 1.0))


def shrunk_target_prevalence(source: float, em_target: float, shrink: float, *, epsilon: float = 1e-6) -> float:
    if not 0.0 <= shrink <= 1.0:
        raise ContractError("shrink must be in [0,1]")
    odds = _logit(source, epsilon) + shrink * (_logit(em_target, epsilon) - _logit(source, epsilon))
    return float(np.clip(1.0 / (1.0 + np.exp(-odds)), epsilon, 1.0 - epsilon))


def correct_to_prior(probability: np.ndarray, source: float, target: float, *, epsilon: float = 1e-6) -> np.ndarray:
    posterior = np.clip(np.asarray(probability, dtype=np.float64), epsilon, 1.0 - epsilon)
    if posterior.ndim != 1 or not np.isfinite(posterior).all():
        raise ContractError("posterior must be a finite vector")
    source = float(np.clip(source, epsilon, 1.0 - epsilon))
    target = float(np.clip(target, epsilon, 1.0 - epsilon))
    positive = posterior * (target / source)
    negative = (1.0 - posterior) * ((1.0 - target) / (1.0 - source))
    return positive / np.maximum(positive + negative, epsilon)


__all__ = ["correct_to_prior", "shrink_lambda", "shrunk_target_prevalence"]
