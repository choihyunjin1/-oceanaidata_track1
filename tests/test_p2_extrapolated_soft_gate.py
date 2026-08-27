from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.profile_projection import project_profiles_vectorized


def test_extrapolation_is_reprojected_and_uses_raw_only_for_layers_two_and_four() -> None:
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
    raw = np.array([22.0, 100.0, 20.0])
    routed_input = base.copy()
    routed_input[frame["layer"].isin((2, 4))] = raw[frame["layer"].isin((2, 4))]
    routed = project_profiles_vectorized(frame, routed_input, endpoints).prediction
    final = project_profiles_vectorized(frame, base + 2.0 * (routed - base), endpoints).prediction
    assert final[1] != 100.0
    assert np.all(np.diff(final) <= 0)
    assert final.min() >= 18.0 and final.max() <= 24.0
