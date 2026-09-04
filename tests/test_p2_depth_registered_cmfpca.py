from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.depth_registered_cmfpca import (
    ConditionalMFPCA,
    PreparedProfiles,
    cubic_bspline_df5,
    evaluate_promotion_gate,
    select_rank,
)


def test_fixed_cubic_basis_has_five_functions_and_partition_of_unity() -> None:
    basis = cubic_bspline_df5([4.18, 7.04, 20.0, 39.45, 49.35])
    assert basis.shape == (5, 5)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12, rtol=0)


def test_physical_depth_not_layer_identity_controls_design() -> None:
    near_49_from_2024_l7 = cubic_bspline_df5([49.05])
    near_49_from_2025_l8 = cubic_bspline_df5([49.35])
    distinct_2025_l7 = cubic_bspline_df5([39.45])
    assert np.linalg.norm(near_49_from_2024_l7 - near_49_from_2025_l8) < 0.05
    assert np.linalg.norm(near_49_from_2024_l7 - distinct_2025_l7) > 0.2


def test_rank_selection_is_train_thresholded_and_capped() -> None:
    assert select_rank(np.array([60.0, 30.0, 5.0, 3.0, 1.0, 1.0])) == 3
    assert select_rank(np.ones(10)) == 4


def test_conditional_prediction_ignores_target_observation_values() -> None:
    times = pd.date_range("2024-01-01", periods=160, freq="D", tz="UTC")
    angle = np.linspace(0, 8 * np.pi, len(times))
    coefficient = np.column_stack(
        [np.sin(angle + shift) for shift in np.linspace(0, 1.0, 10)]
    )
    profiles = PreparedProfiles(
        times=times,
        coefficients=coefficient,
        temp_fit_mse=np.full(len(times), 0.01),
        psal_fit_mse=np.full(len(times), 0.02),
    )
    model = ConditionalMFPCA.fit(profiles, np.ones(len(times), dtype=bool))
    timestamp = "2025-09-01T00:00:00+09:00"
    observations = pd.DataFrame(
        {
            "time": [timestamp] * 5,
            "layer": [1, 5, 7, 2, 3],
            "nominal_depth": [4.19, 19.59, 39.45, 7.04, 9.44],
            "temp": [25.0, 18.0, 14.0, 999.0, -999.0],
            "psal": [31.0, 32.0, 33.0, 999.0, -999.0],
        }
    )
    query = pd.DataFrame(
        {
            "time": [timestamp] * 3,
            "layer": [2, 3, 4],
            "nominal_depth": [7.04, 9.44, 14.74],
        }
    )
    first = model.predict(observations, query)
    observations.loc[observations["layer"].isin([2, 3]), ["temp", "psal"]] = 12345.0
    second = model.predict(observations, query)
    np.testing.assert_allclose(first, second, atol=0, rtol=0)
    assert np.isfinite(first).all()


def test_promotion_gate_is_conjunctive() -> None:
    thresholds = {
        "aggregate_delta_rmse_max_c": -0.003,
        "paired_kst_day_bootstrap_ci90_upper_max_c": 0.0,
        "minimum_improved_folds": 3,
        "maximum_worst_fold_regression_c": 0.010,
        "maximum_layer_regression_c": 0.005,
    }
    passed = evaluate_promotion_gate(
        aggregate_delta=-0.004,
        bootstrap_ci90_high=-0.001,
        fold_deltas={"a": -0.01, "b": -0.01, "c": -0.01, "d": 0.005},
        layer_deltas={"2": -0.01, "3": 0.001, "4": -0.002},
        thresholds=thresholds,
    )
    assert passed["passed"] is True
    failed = evaluate_promotion_gate(
        aggregate_delta=-0.004,
        bootstrap_ci90_high=0.001,
        fold_deltas={"a": -0.01, "b": -0.01, "c": -0.01, "d": 0.005},
        layer_deltas={"2": -0.01, "3": 0.001, "4": -0.002},
        thresholds=thresholds,
    )
    assert failed["passed"] is False
