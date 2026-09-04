from __future__ import annotations

import numpy as np

from p1_qc.p1_station_pooled_hierarchical_residual_subset_scan_anchor_union_20260828_v1 import (
    hierarchical_center_scale,
    tail_layer_share,
)


def test_hierarchical_scale_shrinks_sparse_cell_to_station() -> None:
    cell = np.asarray([10.0, 10.2])
    station = np.linspace(-1.0, 1.0, 1000)
    center, scale, weight = hierarchical_center_scale(
        cell, station, prior_strength=100, minimum_scale=1e-6
    )
    assert 0.0 < weight < 0.03
    assert abs(center) < 0.3
    assert scale > 0.0


def test_tail_share_detects_one_layer_concentration() -> None:
    scores = np.arange(100, dtype=np.float64)
    layers = np.r_[np.ones(90, dtype=int), np.full(10, 2, dtype=int)]
    assert tail_layer_share(scores, layers, tail_fraction=0.1) == 1.0
