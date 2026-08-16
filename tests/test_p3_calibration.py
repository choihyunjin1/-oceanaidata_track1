from __future__ import annotations

import numpy as np
import pytest

from p3_wave.calibration import apply_global_bias_correction, estimate_global_bias_correction


def test_global_bias_is_rmse_optimal_intercept_and_clipped() -> None:
    truth = np.array([2.0, 3.0, 4.0])
    prediction = np.array([1.5, 2.5, 3.5])
    assert estimate_global_bias_correction(truth, prediction) == pytest.approx(0.35)
    assert estimate_global_bias_correction(
        truth, prediction, max_absolute_correction=1.0
    ) == pytest.approx(0.5)


def test_apply_global_bias_preserves_shape_and_physical_bounds() -> None:
    result = apply_global_bias_correction(np.array([0.1, 29.9]), 0.3)
    np.testing.assert_allclose(result, np.array([0.4, 30.0]))


@pytest.mark.parametrize(
    ("truth", "prediction"),
    [
        (np.array([]), np.array([])),
        (np.array([1.0]), np.array([1.0, 2.0])),
        (np.array([np.nan]), np.array([1.0])),
    ],
)
def test_bias_estimation_fails_closed_on_invalid_calibration(
    truth: np.ndarray, prediction: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        estimate_global_bias_correction(truth, prediction)
