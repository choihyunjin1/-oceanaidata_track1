from __future__ import annotations

import numpy as np
import pytest

from p3_wave.regimes import classify_future_trajectory


def test_future_trajectory_classes_cover_growth_decay_flat_and_peak() -> None:
    current = np.full(4, 2.0)
    future = np.array(
        [
            [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
            [1.9, 1.8, 1.7, 1.6, 1.5, 1.4],
            [2.0, 2.1, 2.0, 1.9, 2.0, 2.1],
            [2.2, 2.5, 2.7, 2.6, 2.3, 2.1],
        ]
    )
    assert classify_future_trajectory(current, future).tolist() == [
        "continued_growth",
        "decay",
        "near_flat",
        "peak_then_decay",
    ]


def test_future_trajectory_classification_fails_closed() -> None:
    with pytest.raises(ValueError):
        classify_future_trajectory(np.array([2.0]), np.ones((1, 5)))
