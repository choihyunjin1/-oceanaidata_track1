from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.public_layer_causal_residual import (
    CausalResidualSpec,
    _causal_gap_reset_median,
    apply_correction_and_projection,
    correction_for_rows,
)


def test_causal_median_resets_after_gap() -> None:
    first = pd.date_range("2025-01-01", periods=4, freq="10min", tz="UTC")
    second = pd.date_range("2025-01-02", periods=3, freq="10min", tz="UTC")
    times = first.append(second)
    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0])
    result = _causal_gap_reset_median(
        times,
        values,
        rolling_hours=24,
        cadence_minutes=10,
        minimum_samples=2,
    )
    np.testing.assert_allclose(result[:4], [np.nan, 1.5, 2.0, 2.5], equal_nan=True)
    np.testing.assert_allclose(result[4:], [np.nan, 15.0, 20.0], equal_nan=True)


def _state(residuals: list[float]) -> pd.DataFrame:
    row: dict[str, object] = {"time": pd.Timestamp("2025-01-01", tz="UTC")}
    for layer, residual, depth in zip(
        (1, 5, 6, 7, 8), residuals, (4, 20, 31, 39, 49), strict=True
    ):
        row[f"median_residual_{layer}"] = residual
        row[f"depth_{layer}"] = depth
    return pd.DataFrame([row])


def test_ridge_affine_correction_obeys_support_and_clip() -> None:
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * 3,
            "layer": [2, 3, 4],
            "time": ["2025-01-01T09:00:00+09:00"] * 3,
        }
    )
    spec = CausalResidualSpec()
    result = correction_for_rows(frame, np.array([7.04, 9.44, 14.74]), _state([0.5] * 5), spec)
    assert result.supported_mask.all()
    np.testing.assert_allclose(result.correction, 0.125)


def test_mixed_endpoint_sign_is_exact_noop() -> None:
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * 3,
            "layer": [2, 3, 4],
            "time": ["2025-01-01T09:00:00+09:00"] * 3,
        }
    )
    result = correction_for_rows(
        frame,
        np.array([7.04, 9.44, 14.74]),
        _state([0.1, -0.1, 0.1, 0.1, 0.1]),
    )
    assert not result.supported_mask.any()
    np.testing.assert_array_equal(result.correction, np.zeros(3))


def test_projection_is_applied_after_correction() -> None:
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * 3,
            "layer": [2, 3, 4],
            "time": ["2025-01-01T09:00:00+09:00"] * 3,
        }
    )
    endpoints = pd.DataFrame(
        {
            "time": [pd.Timestamp("2025-01-01", tz="UTC")],
            "temp_1": [20.0],
            "temp_5": [10.0],
        }
    )
    result = apply_correction_and_projection(
        frame,
        np.array([12.0, 18.0, 14.0]),
        np.array([0.1, 0.1, 0.1]),
        endpoints,
    )
    assert result.eligible_mask.all()
    assert np.all(np.diff(result.prediction) <= 0.0)
    assert result.prediction.min() >= 10.0
    assert result.prediction.max() <= 20.0
