from __future__ import annotations

import numpy as np

from scripts.run_p3_target_shift_retroaudit_20260828_v1 import (
    effective_sample_size,
    weighted_rmse,
)


def test_effective_sample_size_uniform_weights_equals_count() -> None:
    weights = np.ones(7)
    assert effective_sample_size(weights) == 7.0


def test_effective_sample_size_concentrated_weights_is_smaller() -> None:
    weights = np.array([9.0, 1.0, 0.0, 0.0])
    assert 1.0 < effective_sample_size(weights) < 2.0


def test_weighted_rmse_respects_zero_weight_rows() -> None:
    target = np.array([0.0, 100.0])
    prediction = np.array([1.0, 0.0])
    weights = np.array([1.0, 0.0])
    assert weighted_rmse(target, prediction, weights) == 1.0


def test_weighted_rmse_matches_unweighted_when_uniform() -> None:
    target = np.array([0.0, 2.0, 4.0])
    prediction = np.array([1.0, 2.0, 2.0])
    expected = float(np.sqrt(np.mean(np.square(prediction - target))))
    assert np.isclose(weighted_rmse(target, prediction, np.ones(3)), expected)
