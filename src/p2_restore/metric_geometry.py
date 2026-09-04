"""Prediction-space geometry from aggregate RMSE observations.

The functions in this module never need target labels.  They use candidate
prediction vectors and already observed aggregate RMSE values to bound the
RMSE of a new candidate.  The bound follows from orthogonally projecting the
unknown reference error onto the span of previously scored prediction
directions and applying Cauchy--Schwarz only to the remaining component.
"""

from __future__ import annotations

from itertools import product

import numpy as np


def _mean_inner(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.asarray(left, dtype=float) * np.asarray(right, dtype=float)))


def _single_bound(
    reference: np.ndarray,
    scored_predictions: np.ndarray,
    scored_rmse: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=float)
    scored_predictions = np.asarray(scored_predictions, dtype=float)
    scored_rmse = np.asarray(scored_rmse, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if scored_predictions.ndim != 2:
        raise ValueError("scored_predictions must have shape (n_candidates, n_rows)")
    if scored_predictions.shape[1] != len(reference):
        raise ValueError("scored prediction row count differs from reference")
    if len(scored_predictions) != len(scored_rmse):
        raise ValueError("one RMSE is required for every scored prediction")
    if len(candidate) != len(reference):
        raise ValueError("candidate row count differs from reference")
    if not all(
        np.isfinite(values).all()
        for values in (reference, scored_predictions, scored_rmse, candidate)
    ):
        raise ValueError("geometry inputs must be finite")

    directions = (scored_predictions - reference).T
    candidate_direction = candidate - reference
    gram = directions.T @ directions / len(reference)
    gram_inverse = np.linalg.pinv(gram, rcond=1e-12)
    reference_loss = float(scored_rmse[0] ** 2)
    scored_loss = scored_rmse**2
    direction_norm = np.diag(gram)
    error_inner = 0.5 * (scored_loss - reference_loss - direction_norm)

    candidate_inner = directions.T @ candidate_direction / len(reference)
    coefficients = gram_inverse @ candidate_inner
    projected = directions @ coefficients
    residual = candidate_direction - projected
    residual_mse = _mean_inner(residual, residual)
    projected_error_mse = float(error_inner @ gram_inverse @ error_inner)
    perpendicular_error_mse = max(0.0, reference_loss - projected_error_mse)
    center_mse = (
        reference_loss
        + 2.0 * float(error_inner @ coefficients)
        + _mean_inner(candidate_direction, candidate_direction)
    )
    radius_mse = 2.0 * float(np.sqrt(perpendicular_error_mse * max(0.0, residual_mse)))
    lower_mse = max(0.0, center_mse - radius_mse)
    upper_mse = max(0.0, center_mse + radius_mse)
    singular_values = np.linalg.svd(gram, compute_uv=False)
    positive = singular_values[singular_values > singular_values.max() * 1e-12]
    condition = float(positive.max() / positive.min()) if len(positive) else float("inf")
    return {
        "basis_rank": int(np.linalg.matrix_rank(gram, tol=singular_values.max() * 1e-12)),
        "gram_condition_number": condition,
        "candidate_direction_rms": float(np.sqrt(_mean_inner(candidate_direction, candidate_direction))),
        "orthogonal_residual_rms": float(np.sqrt(max(0.0, residual_mse))),
        "orthogonal_residual_share": float(
            np.sqrt(max(0.0, residual_mse))
            / max(np.sqrt(_mean_inner(candidate_direction, candidate_direction)), np.finfo(float).eps)
        ),
        "reference_error_projected_rms": float(np.sqrt(max(0.0, projected_error_mse))),
        "reference_error_unresolved_rms": float(np.sqrt(perpendicular_error_mse)),
        "rmse_lower": float(np.sqrt(lower_mse)),
        "rmse_center": float(np.sqrt(max(0.0, center_mse))),
        "rmse_upper": float(np.sqrt(upper_mse)),
    }


def rounded_rmse_geometry_bound(
    reference: np.ndarray,
    scored_predictions: np.ndarray,
    displayed_rmse: np.ndarray,
    candidate: np.ndarray,
    *,
    decimals: int = 6,
) -> dict[str, object]:
    """Return an RMSE interval robust to displayed-score rounding.

    ``scored_predictions[0]`` must equal ``reference`` and the first displayed
    RMSE is its score.  All lower/upper score-rounding corners are evaluated;
    the returned robust interval encloses every numerically feasible corner.
    """

    reference = np.asarray(reference, dtype=float)
    scored_predictions = np.asarray(scored_predictions, dtype=float)
    displayed_rmse = np.asarray(displayed_rmse, dtype=float)
    if not np.allclose(scored_predictions[0], reference, rtol=0.0, atol=0.0):
        raise ValueError("the first scored prediction must be the reference")
    half_unit = 0.5 * 10.0 ** (-decimals)
    bounds: list[dict[str, float | int]] = []
    for signs in product((-1.0, 1.0), repeat=len(displayed_rmse)):
        rmse = displayed_rmse + half_unit * np.asarray(signs)
        candidate_bound = _single_bound(reference, scored_predictions, rmse, candidate)
        # Rounding corners can be slightly infeasible when nearly collinear
        # directions amplify the final decimal.  Exclude only corners whose
        # projected error norm is materially larger than total error norm.
        total = float(rmse[0] ** 2)
        projected = float(candidate_bound["reference_error_projected_rms"]) ** 2
        if projected <= total + 1e-10:
            bounds.append(candidate_bound)
    if not bounds:
        raise ValueError("no score-rounding corner is geometrically feasible")
    center = _single_bound(reference, scored_predictions, displayed_rmse, candidate)
    return {
        "displayed_score_bound": center,
        "rounding_robust_rmse_lower": min(float(item["rmse_lower"]) for item in bounds),
        "rounding_robust_rmse_upper": max(float(item["rmse_upper"]) for item in bounds),
        "feasible_rounding_corners": len(bounds),
        "total_rounding_corners": 2 ** len(displayed_rmse),
        "display_decimals": decimals,
    }
