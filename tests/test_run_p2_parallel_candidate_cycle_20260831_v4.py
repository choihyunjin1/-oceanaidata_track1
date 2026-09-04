from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_p2_parallel_candidate_cycle_20260831_v4 import (
    cap_profile_correction,
    strict_gate,
    usable_feature_mask,
)


def test_strict_gate_requires_pooled_two_folds_and_official_like_fold() -> None:
    folds = {
        "2024_sep_oct": {"delta_rmse": -0.01},
        "2025_jul_aug": {"delta_rmse": -0.02},
        "2025_nov_dec": {"delta_rmse": 0.01},
    }
    assert all(strict_gate(folds, -0.005).values())
    folds["2024_sep_oct"]["delta_rmse"] = 0.001
    assert not strict_gate(folds, -0.005)["official_like_sep_oct_improved"]


def test_profile_cap_is_train_independent_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "station": ["A"] * 3 + ["A"] * 3,
            "time": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3, utc=True),
            "layer": [2, 3, 4, 2, 3, 4],
        }
    )
    raw = np.asarray([0.2, 0.2, 0.2, 0.01, -0.02, 0.03])
    bounded = cap_profile_correction(frame, raw, absolute_cap=0.15, rms_cap=0.05)
    assert np.max(np.abs(bounded)) <= 0.15
    assert np.sqrt(np.mean(np.square(bounded[:3]))) <= 0.05 + 1e-12
    assert np.allclose(bounded[3:], raw[3:])


def test_usable_feature_mask_drops_all_nan_and_constant_columns() -> None:
    values = np.asarray([[np.nan, 1.0, 1.0], [np.nan, 1.0, 2.0], [np.nan, 1.0, 3.0]])
    assert usable_feature_mask(values).tolist() == [False, False, True]
