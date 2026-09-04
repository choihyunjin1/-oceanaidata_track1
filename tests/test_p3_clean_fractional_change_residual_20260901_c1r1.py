from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p3_wave.clean_fractional_change_residual_20260901_c1r1 import (
    CleanCycleError,
    assert_validation_surface_matches,
    blend_with_clean_fallback,
    fractional_target,
    restore_delta,
)


def test_fractional_target_round_trip_is_unchanged() -> None:
    delta = np.asarray([-0.5, 0.0, 1.25])
    current = np.asarray([1.5, 2.0, 3.5])
    fraction = fractional_target(delta, current, offset_m=0.5)
    np.testing.assert_allclose(
        restore_delta(fraction, current, offset_m=0.5), delta, rtol=0.0, atol=1e-15
    )


def test_fixed_blend_is_unchanged() -> None:
    result = blend_with_clean_fallback(
        np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0]), challenger_weight=0.25
    )
    np.testing.assert_allclose(result, [1.5, 2.5])


def _keys() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": ["b", "a"],
            "anchor_id": [2, 1],
            "station": ["S-ORS", "G-ORS"],
            "episode_id": [20, 10],
        }
    )


def test_validation_surface_comparison_is_order_insensitive() -> None:
    assert_validation_surface_matches(_keys(), _keys().iloc[::-1].reset_index(drop=True))


def test_validation_surface_comparison_rejects_changed_key() -> None:
    changed = _keys()
    changed.loc[0, "anchor_id"] = 3
    with pytest.raises(CleanCycleError, match="differ"):
        assert_validation_surface_matches(_keys(), changed)


def test_validation_surface_comparison_requires_schema() -> None:
    with pytest.raises(CleanCycleError, match="missing"):
        assert_validation_surface_matches(_keys().drop(columns="episode_id"), _keys())
