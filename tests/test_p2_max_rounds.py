from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2_restore.features import FeatureTable
from p2_restore.max_rounds import (
    MAX_ROUNDS,
    _make_estimator,
    fit_max_model,
    predict_model_at,
    select_best_round,
)


def _table() -> FeatureTable:
    x = np.linspace(-2, 2, 1_200)
    frame = pd.DataFrame(
        {
            "station": "S-ORS",
            "layer": np.resize([2, 3, 4], len(x)),
            "time": pd.date_range("2024-01-01", periods=len(x), freq="10min", tz="UTC"),
            "baseline": 20.0 + 0.2 * x,
            "x": x,
            "residual": np.sin(x) * 0.3,
            "target": 20.0 + 0.2 * x + np.sin(x) * 0.3,
        }
    )
    return FeatureTable(frame, ("baseline", "x"))


def test_max_estimator_changes_only_boosting_horizon() -> None:
    estimator = _make_estimator(seed=7)
    parameters = estimator.get_params()
    assert parameters["n_estimators"] == MAX_ROUNDS
    assert parameters["learning_rate"] == 0.04
    assert parameters["num_leaves"] == 31
    assert parameters["max_depth"] == 7
    assert parameters["min_child_samples"] == 200
    assert parameters["deterministic"] is True
    assert parameters["force_row_wise"] is True


def test_checkpoint_prediction_uses_requested_iteration() -> None:
    table = _table()
    model = fit_max_model(table, seed=11, rounds=20)
    at_five = predict_model_at(model, table, 5)
    at_twenty = predict_model_at(model, table, 20)
    assert np.isfinite(at_five).all()
    assert np.isfinite(at_twenty).all()
    assert not np.array_equal(at_five, at_twenty)
    with pytest.raises(ValueError, match="outside"):
        predict_model_at(model, table, 21)


def test_select_best_round_uses_rmse_then_lower_round() -> None:
    curve = [
        {"round": 400, "router_rmse": 0.8},
        {"round": 800, "router_rmse": 0.7},
        {"round": 1200, "router_rmse": 0.7},
    ]
    assert select_best_round(curve) == 800


def test_select_best_round_rejects_non_finite_metric() -> None:
    with pytest.raises(ValueError, match="invalid checkpoint"):
        select_best_round([{"round": 400, "router_rmse": np.nan}])
