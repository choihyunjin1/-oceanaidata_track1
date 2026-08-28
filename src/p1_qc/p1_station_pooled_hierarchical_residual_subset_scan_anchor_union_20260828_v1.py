"""Frozen station-pooled robust shrinkage and block subset-scan helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from p1_qc.p1_async_latent_state_gp_subset_scan_anchor_union_20260828_v1 import (
    binary_metrics,
    block_bootstrap_delta,
    block_proposals,
    conformal_threshold,
    sha256_file,
)

__all__ = [
    "binary_metrics",
    "block_bootstrap_delta",
    "block_proposals",
    "conformal_threshold",
    "hierarchical_center_scale",
    "sha256_file",
    "tail_layer_share",
]


def _median_mad(values: Sequence[float], minimum_scale: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return 0.0, 1.0
    center = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - center)))
    if not np.isfinite(scale) or scale < minimum_scale:
        scale = max(float(np.std(array)), minimum_scale)
    return center, scale


def hierarchical_center_scale(
    cell_values: Sequence[float],
    station_values: Sequence[float],
    *,
    prior_strength: int,
    minimum_scale: float,
) -> tuple[float, float, float]:
    """Shrink cell median/log-MAD toward its station with a sealed effective sample weight."""

    cell = np.asarray(cell_values, dtype=np.float64)
    cell = cell[np.isfinite(cell)]
    station_center, station_scale = _median_mad(station_values, minimum_scale)
    cell_center, cell_scale = _median_mad(cell, minimum_scale)
    weight = len(cell) / (len(cell) + int(prior_strength))
    center = weight * cell_center + (1.0 - weight) * station_center
    log_scale = weight * np.log(cell_scale) + (1.0 - weight) * np.log(station_scale)
    return float(center), float(np.exp(log_scale)), float(weight)


def tail_layer_share(
    scores: Sequence[float], layers: Sequence[int], *, tail_fraction: float
) -> float:
    values = np.asarray(scores, dtype=np.float64)
    layer = np.asarray(layers, dtype=np.int64)
    if not len(values):
        return 1.0
    count = max(1, int(np.ceil(len(values) * float(tail_fraction))))
    selected = layer[np.argsort(values)[-count:]]
    _, frequencies = np.unique(selected, return_counts=True)
    return float(frequencies.max() / count)
