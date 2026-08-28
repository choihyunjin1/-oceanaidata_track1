"""Temporally fused robust-PCA utilities for the sealed P1 diagnostic.

The module is intentionally independent of the incumbent.  It separates a
shared low-rank water-column state from persistent, sensor-local deviations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Elementwise soft thresholding."""

    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.maximum(np.abs(array) - threshold, 0.0)


def _difference(values: np.ndarray) -> np.ndarray:
    return np.diff(values, axis=0)


def _difference_transpose(values: np.ndarray, rows: int) -> np.ndarray:
    output = np.zeros((rows, values.shape[1]), dtype=np.float64)
    if rows > 1:
        output[0] = -values[0]
        output[-1] = values[-1]
    if rows > 2:
        output[1:-1] = values[:-1] - values[1:]
    return output


def fused_sparse_prox(
    residual: np.ndarray,
    l1_weight: float,
    tv_weight: float,
    *,
    iterations: int = 40,
    rho: float = 1.0,
) -> np.ndarray:
    """Proximal operator for L1 plus temporal total variation.

    A deterministic split-ADMM solve is used.  Columns are independent, while
    the tridiagonal system is solved for all columns at once.
    """

    target = np.asarray(residual, dtype=np.float64)
    rows, columns = target.shape
    if rows < 2:
        return soft_threshold(target, l1_weight)
    sparse = target.copy()
    point = sparse.copy()
    temporal = _difference(sparse)
    point_dual = np.zeros_like(sparse)
    temporal_dual = np.zeros_like(temporal)

    diagonal = np.full(rows, 1.0 + 3.0 * rho, dtype=np.float64)
    diagonal[0] = diagonal[-1] = 1.0 + 2.0 * rho
    banded = np.zeros((3, rows), dtype=np.float64)
    banded[0, 1:] = -rho
    banded[1] = diagonal
    banded[2, :-1] = -rho

    for _ in range(iterations):
        right = (
            target
            + rho * (point - point_dual)
            + rho
            * _difference_transpose(temporal - temporal_dual, rows)
        )
        sparse = solve_banded((1, 1), banded, right, check_finite=False)
        point = soft_threshold(sparse + point_dual, l1_weight / rho)
        difference = _difference(sparse)
        temporal = soft_threshold(difference + temporal_dual, tv_weight / rho)
        point_dual += sparse - point
        temporal_dual += difference - temporal
    return sparse


@dataclass(frozen=True)
class Decomposition:
    low_rank: np.ndarray
    sparse: np.ndarray
    converged: bool
    iterations: int
    relative_error: float


def temporally_fused_rpca(
    values: np.ndarray,
    *,
    lam: float | None = None,
    maximum_iterations: int = 200,
    tolerance: float = 1e-6,
    proximal_iterations: int = 40,
) -> Decomposition:
    """Solve nuclear norm + fused sparse decomposition by inexact ALM."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("values must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("values must be finite")
    rows, columns = matrix.shape
    weight = float(lam if lam is not None else 1.0 / np.sqrt(max(rows, columns)))
    norm = max(float(np.linalg.norm(matrix, ord="fro")), 1e-12)
    spectral = max(float(np.linalg.norm(matrix, ord=2)), 1e-12)
    maximum = max(float(np.max(np.abs(matrix))) / max(weight, 1e-12), 1e-12)
    dual = matrix / max(spectral, maximum)
    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    mu = 1.25 / spectral
    maximum_mu = mu * 1e7
    relative_error = float("inf")

    for iteration in range(1, maximum_iterations + 1):
        shifted = matrix - sparse + dual / mu
        left, singular, right = np.linalg.svd(shifted, full_matrices=False)
        singular = np.maximum(singular - 1.0 / mu, 0.0)
        low_rank = (left * singular) @ right
        residual = matrix - low_rank + dual / mu
        sparse = fused_sparse_prox(
            residual,
            weight / mu,
            weight / mu,
            iterations=proximal_iterations,
        )
        error = matrix - low_rank - sparse
        relative_error = float(np.linalg.norm(error, ord="fro") / norm)
        dual += mu * error
        mu = min(mu * 1.5, maximum_mu)
        if relative_error <= tolerance:
            return Decomposition(low_rank, sparse, True, iteration, relative_error)
    return Decomposition(low_rank, sparse, False, maximum_iterations, relative_error)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open true runs."""

    values = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def duration_mask(mask: np.ndarray, minimum: int, maximum: int) -> np.ndarray:
    """Keep only runs inside a pre-registered duration interval."""

    output = np.zeros(len(mask), dtype=bool)
    for start, stop in contiguous_runs(mask):
        if minimum <= stop - start <= maximum:
            output[start:stop] = True
    return output
