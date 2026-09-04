from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2_restore.p2_nested_pls_capacity_grid_20260829_v1 import (
    FittedPLSResidual,
    capacity_grid,
    select_inner_point,
)

GRID = {
    "rank": [1, 2, 3],
    "spline_ridge": [0.0001, 0.001, 0.01],
    "leverage_quantile": [0.95, 0.975, 0.99],
    "rms_cap_c": [0.025, 0.05, 0.075],
    "strength": [0.5, 0.75, 1.0],
}


def test_capacity_grid_is_exactly_243_unique_stable_points() -> None:
    first = capacity_grid(GRID)
    second = capacity_grid(GRID)
    assert len(first) == 243
    assert len({point.point_id for point in first}) == 243
    assert [point.point_id for point in first] == [point.point_id for point in second]
    assert {point.rank for point in first} == {1, 2, 3}


def test_inner_selector_rejects_outer_outcome_field() -> None:
    records = []
    for index, point in enumerate(capacity_grid(GRID)):
        records.append(
            {
                "point_id": point.point_id,
                "point": {
                    "rank": point.rank,
                    "spline_ridge": point.spline_ridge,
                    "leverage_quantile": point.leverage_quantile,
                    "rms_cap_c": point.rms_cap_c,
                    "strength": point.strength,
                },
                "candidate_rmse": 1.0 + index * 1e-6,
                "delta_rmse": -0.001,
                "worst_group_delta_rmse": 0.0,
                "worst_layer_delta_rmse": 0.0,
                "correction_p99_c": 0.1,
            }
        )
    records[0]["outer_truth"] = [1.0]
    with pytest.raises(ValueError, match="outer outcome leaked"):
        select_inner_point(records)


def test_inner_selector_uses_eligible_point_before_lower_ineligible_rmse() -> None:
    records = []
    for index, point in enumerate(capacity_grid(GRID)):
        records.append(
            {
                "point_id": point.point_id,
                "point": {
                    "rank": point.rank,
                    "spline_ridge": point.spline_ridge,
                    "leverage_quantile": point.leverage_quantile,
                    "rms_cap_c": point.rms_cap_c,
                    "strength": point.strength,
                },
                "candidate_rmse": 2.0 + index * 1e-6,
                "delta_rmse": 0.001,
                "worst_group_delta_rmse": 0.0,
                "worst_layer_delta_rmse": 0.0,
                "correction_p99_c": 0.1,
            }
        )
    records[0]["candidate_rmse"] = 0.1
    records[1]["candidate_rmse"] = 1.0
    records[1]["delta_rmse"] = -0.001
    selected = select_inner_point(records)
    assert selected["point_id"] == records[1]["point_id"]
    assert selected["inner_selection_eligible"] is True


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_fitted_pls_capacity_supports_all_sealed_ranks(rank: int) -> None:
    rng = np.random.default_rng(20260829)
    rows = 300
    latent = rng.normal(size=(rows, 3))
    features = pd.DataFrame(
        {
            "public_profile_valid": np.ones(rows),
            "x1": latent[:, 0] + rng.normal(0.0, 0.02, rows),
            "x2": latent[:, 1] + rng.normal(0.0, 0.02, rows),
            "x3": latent[:, 2] + rng.normal(0.0, 0.02, rows),
        }
    )
    response = latent @ np.asarray(
        [[1.0, 0.2, -0.1], [-0.3, 0.8, 0.1], [0.2, -0.1, 0.7]]
    )
    times = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    fitted = FittedPLSResidual.fit(features, response, times, rank=rank)
    prediction, leverage = fitted.predict_raw(features, times)
    assert prediction.shape == response.shape
    assert leverage.shape == (rows,)
    assert fitted.leverage_limit(0.95) <= fitted.leverage_limit(0.99)
    assert fitted.receipt(0.975)["rank"] == rank
