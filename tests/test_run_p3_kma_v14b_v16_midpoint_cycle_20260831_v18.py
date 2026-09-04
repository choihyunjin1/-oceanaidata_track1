from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_p3_kma_v14b_v16_midpoint_cycle_20260831_v18 import (
    FIXED_WEIGHT,
    midpoint_prediction,
)


def test_midpoint_is_exact_average_of_v14b_and_v16() -> None:
    frame = pd.DataFrame(
        {
            "anchor_time": pd.to_datetime(["2025-01-01", "2025-07-01"], utc=True),
            "reference": [1.0, 2.0],
        }
    )
    correction = np.asarray([0.2, -0.3])
    theta = np.asarray([0.0, 0.0])
    prediction, multiplier = midpoint_prediction(frame, correction, theta)
    np.testing.assert_allclose(multiplier, [1.0, 1.0])
    np.testing.assert_allclose(prediction, [1.2, 1.7])


def test_fixed_weight_is_half() -> None:
    assert FIXED_WEIGHT == 0.5


def test_midpoint_multiplier_is_bounded_half_to_one() -> None:
    frame = pd.DataFrame(
        {
            "anchor_time": pd.to_datetime(["2025-01-01", "2025-07-01"], utc=True),
            "reference": [1.0, 2.0],
        }
    )
    _, multiplier = midpoint_prediction(frame, np.asarray([0.2, -0.3]), np.asarray([100.0, 100.0]))
    assert np.all((multiplier >= 0.5) & (multiplier <= 1.0))
