from __future__ import annotations

import numpy as np
import pytest

from p3_wave.weights import amplitude_emphasis_weights


def test_amplitude_weights_are_symmetric_bounded_and_monotone() -> None:
    result = amplitude_emphasis_weights(np.ones(5), np.array([-3.0, -1.0, 0.0, 1.0, 3.0]))
    np.testing.assert_allclose(result, np.array([1.5, 1.25, 1.0, 1.25, 1.5]))


def test_amplitude_weights_fail_closed_on_bad_inputs() -> None:
    with pytest.raises(ValueError):
        amplitude_emphasis_weights(np.ones(2), np.ones(3))
    with pytest.raises(ValueError):
        amplitude_emphasis_weights(np.ones(2), np.ones(2), scale_m=0.0)
