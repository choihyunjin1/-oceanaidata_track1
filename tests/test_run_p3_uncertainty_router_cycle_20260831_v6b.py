from __future__ import annotations

import numpy as np

from scripts.run_p3_uncertainty_router_cycle_20260831_v6b import (
    POLICIES,
    conditional_score_translation,
    policy_sha256,
    select_base_cases,
)


def test_confidence_policies_are_frozen_and_ordered() -> None:
    assert [policy.upper_residual_quantile for policy in POLICIES] == [0.50, 0.65, 0.80]
    assert len({policy.name for policy in POLICIES}) == 3
    assert len(policy_sha256()) == 64


def test_more_conservative_policy_never_selects_more_base_cases() -> None:
    median = np.asarray([-2.0, -0.5, -0.1, 0.2])
    mad = np.asarray([0.1, 0.1, 0.1, 0.1])
    residual = np.asarray([-0.2, 0.0, 0.2, 0.8])
    counts = []
    for policy in POLICIES:
        selected, upper, quantile = select_base_cases(median, mad, residual, policy)
        assert np.isfinite(upper).all()
        assert np.isfinite(quantile)
        counts.append(int(selected.sum()))
    assert counts[0] >= counts[1] >= counts[2]


def test_conditional_score_translation_uses_requested_anchor() -> None:
    translated = conditional_score_translation(-0.01, -0.02, -0.001)
    anchor = translated["official_anchor"]
    assert anchor == {"rmse_m": 0.575233, "points": 24.203599}
    assert translated["linear_score_slope_points_per_rmse_m"] == -15.870739046986959
    assert translated["scenarios"]["central"]["projected_points"] > anchor["points"]
