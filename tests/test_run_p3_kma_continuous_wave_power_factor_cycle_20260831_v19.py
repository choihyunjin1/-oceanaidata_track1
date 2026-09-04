from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_p3_kma_continuous_wave_power_factor_cycle_20260831_v19 import (
    NEUTRAL_ECDF,
    compute_wave_power,
    key_values_and_order_equal,
    predict_policy,
    ranks_from_prefix,
)


def test_wave_power_formula() -> None:
    value, valid = compute_wave_power(np.asarray([2.0]), np.asarray([4.0]))
    assert valid[0]
    np.testing.assert_allclose(value, [16.0])


def test_invalid_proxy_is_exact_noop() -> None:
    frame = pd.DataFrame(
        {
            "lead_h": [18, 24],
            "reference": [1.425, 1.425],
            "base": [1.0, 1.0],
            "delta": [1.0, 1.0],
            "wave_power_valid": [False, False],
        }
    )
    prediction, alpha = predict_policy(frame, np.asarray([NEUTRAL_ECDF, NEUTRAL_ECDF]))
    np.testing.assert_array_equal(prediction, frame["reference"].to_numpy())
    np.testing.assert_allclose(alpha, [0.425, 0.425])


def test_short_exact_and_active_formula() -> None:
    frame = pd.DataFrame(
        {
            "lead_h": [3, 18, 24],
            "reference": [1.1, 1.425, 1.425],
            "base": [1.0, 1.0, 1.0],
            "delta": [1.0, 1.0, 1.0],
            "wave_power_valid": [True, True, True],
        }
    )
    prediction, alpha = predict_policy(frame, np.asarray([0.0, 0.0, 0.5]))
    assert prediction[0] == 1.1
    np.testing.assert_allclose(alpha, [0.0, 0.2, 0.4])
    np.testing.assert_allclose(prediction, [1.1, 1.2, 1.4])


def test_dtype_independent_key_order_contract() -> None:
    left = pd.DataFrame({"case_id": pd.Series(["a"], dtype="string"), "station": ["S"], "lead_h": [3]})
    right = pd.DataFrame({"case_id": ["a"], "station": pd.Series(["S"], dtype="string"), "lead_h": [3]})
    assert key_values_and_order_equal(left, right)
    right.loc[0, "lead_h"] = 6
    assert not key_values_and_order_equal(left, right)


def test_invalid_rank_is_neutral() -> None:
    rank = ranks_from_prefix(np.asarray([1.0, 2.0]), np.asarray([np.nan]), np.asarray([False]))
    assert rank[0] == NEUTRAL_ECDF
