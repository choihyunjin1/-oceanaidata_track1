import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qa_p1_trainfit_postpolicy_20260906_v1 import counts, f1  # noqa: E402


def test_counts_and_zero_denominator():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([1, 1, 0, 0, 1])
    assert counts(y, p).tolist() == [2, 1, 1]
    assert f1(counts(y, p)) == 2 / 3
    assert f1(np.zeros(3)) == 0


def test_unequal_cluster_weighting_is_pooled_not_mean_f1():
    y = np.array([1, 0, 0, 1, 1, 0, 1])
    p = np.array([1, 1, 1, 1, 0, 0, 0])
    blocks = [np.array([0]), np.arange(1, 7)]
    stats = np.array([counts(y[ix], p[ix]) for ix in blocks])
    draw = np.array([1, 1, 0])
    selected = np.concatenate([blocks[i] for i in draw])
    pooled = f1(np.bincount(draw, minlength=2) @ stats)
    assert pooled == f1(counts(y[selected], p[selected]))
    assert pooled != np.mean([f1(stats[i]) for i in draw])
