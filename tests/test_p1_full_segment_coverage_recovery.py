from __future__ import annotations

import numpy as np
import torch

from p1_qc.full_segment_coverage_recovery import (
    contiguous_segment_bounds,
    infer_complete_segments,
)
from p1_qc.ts2vec_conditional_normal_prototype import HierarchicalContrastiveEncoder


def test_contiguous_segment_bounds_keeps_short_segments() -> None:
    segments = np.asarray([1, 1, 2, 2, 2, 3], dtype=np.int64)
    eligible = np.asarray([True, True, True, True, False, True])
    bounds = contiguous_segment_bounds(segments, eligible)
    assert bounds.tolist() == [[0, 2], [2, 4], [5, 6]]


def test_complete_inference_covers_every_eligible_row() -> None:
    torch.manual_seed(7)
    model = HierarchicalContrastiveEncoder(3, 8, 5, (1, 2), 0.0)
    values = np.random.default_rng(7).normal(size=(8, 3)).astype(np.float32)
    segments = np.asarray([1, 1, 2, 2, 2, 3, 3, 3])
    eligible = np.ones(8, dtype=bool)
    embeddings, covered, audit = infer_complete_segments(
        model, values, segments, eligible, torch.device("cpu")
    )
    assert covered.all()
    assert np.isfinite(embeddings).all()
    assert audit["coverage"] == 1.0
    assert audit["minimum_segment_rows"] == 2
