from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.profile_projection import project_profiles_vectorized


def test_layer_specific_extrapolation_is_reprojected() -> None:
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * 3,
            "time": ["2025-09-01T00:00:00+09:00"] * 3,
            "layer": [2, 3, 4],
        }
    )
    endpoints = pd.DataFrame(
        {"time": ["2025-09-01T00:00:00+09:00"], "temp_1": [24.0], "temp_5": [18.0]}
    )
    base = np.array([23.0, 21.0, 19.0])
    routed = np.array([22.9, 20.5, 19.2])
    factor = np.array([10.0, 0.0, 2.0])
    prediction = project_profiles_vectorized(
        frame, base + factor * (routed - base), endpoints
    ).prediction
    assert np.all(np.diff(prediction) <= 0)
    assert prediction.min() >= 18.0 and prediction.max() <= 24.0
