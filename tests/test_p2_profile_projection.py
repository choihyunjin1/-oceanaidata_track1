from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2_restore.profile_projection import (
    _isotonic_three,
    project_profiles,
    project_profiles_vectorized,
    public_endpoint_frame,
)


def test_isotonic_three_enforces_both_directions() -> None:
    increasing = _isotonic_three(np.array([2.0, 0.0, 1.0]), increasing=True)
    decreasing = _isotonic_three(np.array([0.0, 2.0, 1.0]), increasing=False)
    assert np.all(np.diff(increasing) >= 0)
    assert np.all(np.diff(decreasing) <= 0)
    assert increasing.mean() == pytest.approx(1.0)
    assert decreasing.mean() == pytest.approx(1.0)


def test_projection_clips_and_orders_complete_profile() -> None:
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * 3,
            "layer": [2, 3, 4],
            "time": ["2025-09-01T00:00:00+09:00"] * 3,
        }
    )
    endpoints = pd.DataFrame(
        {
            "time": ["2025-09-01T00:00:00+09:00"],
            "temp_1": [24.0],
            "temp_5": [18.0],
        }
    )
    result = project_profiles(frame, np.array([25.0, 20.0, 21.0]), endpoints)
    assert result.prediction.tolist() == pytest.approx([24.0, 20.5, 20.5])
    assert result.eligible_mask.all()
    assert result.active_mask.all()


def test_missing_endpoint_and_incomplete_profile_are_exact_noop() -> None:
    frame = pd.DataFrame(
        {
            "layer": [2, 3, 4, 2, 3],
            "time": ["2025-09-01T00:00:00+09:00"] * 3 + ["2025-09-01T00:10:00+09:00"] * 2,
        }
    )
    endpoints = pd.DataFrame(
        {
            "time": ["2025-09-01T00:00:00+09:00", "2025-09-01T00:10:00+09:00"],
            "temp_1": [np.nan, 24.0],
            "temp_5": [18.0, 18.0],
        }
    )
    prediction = np.array([23.0, 22.0, 21.0, 23.0, 22.0])
    result = project_profiles(frame, prediction, endpoints)
    assert np.array_equal(result.prediction, prediction)
    assert not result.eligible_mask.any()


def test_public_endpoint_frame_uses_only_public_layers() -> None:
    observations = pd.DataFrame(
        {
            "time": ["2025-01-01T00:00:00+09:00"] * 3,
            "layer": [1, 2, 5],
            "temp": [10.0, 999.0, 8.0],
        }
    )
    result = public_endpoint_frame(observations)
    assert result.loc[0, "temp_1"] == 10.0
    assert result.loc[0, "temp_5"] == 8.0
    assert 999.0 not in result.to_numpy()


def test_vectorized_projection_exactly_matches_reference() -> None:
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * 8,
            "time": ["2025-09-01T00:00:00+09:00"] * 3
            + ["2025-09-01T00:10:00+09:00"] * 3
            + ["2025-09-01T00:20:00+09:00"] * 2,
            "layer": [2, 3, 4, 2, 3, 4, 2, 4],
        }
    )
    prediction = np.array([23.0, 25.0, 22.0, 20.0, 18.0, 19.0, 17.0, 16.0])
    endpoints = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2025-09-01T00:00:00+09:00",
                    "2025-09-01T00:10:00+09:00",
                    "2025-09-01T00:20:00+09:00",
                ],
                utc=True,
            ),
            "temp_1": [26.0, 17.0, 18.0],
            "temp_5": [20.0, 21.0, 15.0],
        }
    )
    reference = project_profiles(frame, prediction, endpoints)
    vectorized = project_profiles_vectorized(frame, prediction, endpoints)
    np.testing.assert_allclose(vectorized.prediction, reference.prediction, atol=0, rtol=0)
    assert np.array_equal(vectorized.eligible_mask, reference.eligible_mask)
    assert np.array_equal(vectorized.active_mask, reference.active_mask)
