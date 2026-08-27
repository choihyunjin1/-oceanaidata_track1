from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.meaningful_learning_curve import (
    chronological_prefix_masks,
    fold_equal_layer_rmse,
    numeric_curve_gate,
)


def test_chronological_prefix_masks_are_nested_and_keep_timestamp_groups() -> None:
    times = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "time": np.repeat(times.astype(str), 3),
            "layer": np.tile([2, 3, 4], len(times)),
        }
    )
    eligible = np.ones(len(frame), dtype=bool)
    fractions = [0.4, 0.55, 0.7, 0.85, 1.0]
    masks, boundaries = chronological_prefix_masks(frame, eligible, fractions)
    assert [int(masks[value].sum()) for value in fractions] == [12, 18, 21, 27, 30]
    assert len(boundaries) == 5
    for earlier, later in zip(fractions, fractions[1:], strict=False):
        assert np.all(~masks[earlier] | masks[later])
        selected_times = frame.loc[masks[earlier], "time"]
        assert selected_times.value_counts().eq(3).all()


def test_fold_equal_layer_rmse_uses_equal_fold_mse() -> None:
    report = {
        "by_fold": {
            "a": {"by_layer": {"2": {"rmse_c": 1.0}}},
            "b": {"by_layer": {"2": {"rmse_c": 2.0}}},
            "c": {"by_layer": {"2": {"rmse_c": 2.0}}},
        }
    }
    assert fold_equal_layer_rmse(report, 2) == np.sqrt(3.0)


def test_numeric_curve_gate_requires_effect_uncertainty_folds_and_slices() -> None:
    points = [
        {
            "fraction": fraction,
            "incumbent": 1.0,
            "challenger": 0.96 if fraction < 1.0 else 0.95,
            "delta_ci90": [-0.06, -0.01],
        }
        for fraction in [0.4, 0.55, 0.7, 0.85, 1.0]
    ]
    gates = numeric_curve_gate(
        points,
        fold_deltas=[-0.04, 0.01, -0.08],
        slice_deltas={
            "layer_2": -0.02,
            "layer_3": 0.0,
            "layer_4": 0.0075,
            "2024_sep_oct": -0.05,
        },
    )
    assert all(gates.values())

    failed = numeric_curve_gate(
        points,
        fold_deltas=[-0.04, 0.01, -0.08],
        slice_deltas={
            "layer_2": -0.02,
            "layer_3": 0.0,
            "layer_4": 0.0075001,
            "2024_sep_oct": -0.05,
        },
    )
    assert failed["critical_slice_regression_within_limit"] is False
