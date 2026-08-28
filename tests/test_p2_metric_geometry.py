from __future__ import annotations

import numpy as np

from p2_restore.metric_geometry import rounded_rmse_geometry_bound


def test_metric_geometry_contains_actual_hidden_rmse() -> None:
    rng = np.random.default_rng(20260828)
    truth = rng.normal(size=500)
    reference = rng.normal(size=500)
    directions = rng.normal(scale=0.2, size=(3, 500))
    scored = np.vstack([reference, reference + directions[0], reference + directions[1], reference + directions[2]])
    displayed = np.round(np.sqrt(np.mean((scored - truth) ** 2, axis=1)), 6)
    candidate = reference + 0.4 * directions[0] - 0.2 * directions[1] + rng.normal(scale=0.005, size=500)
    actual = float(np.sqrt(np.mean((candidate - truth) ** 2)))

    bound = rounded_rmse_geometry_bound(reference, scored, displayed, candidate)

    assert float(bound["rounding_robust_rmse_lower"]) <= actual
    assert actual <= float(bound["rounding_robust_rmse_upper"])


def test_scored_candidate_collapses_to_display_rounding_interval() -> None:
    rng = np.random.default_rng(17)
    truth = rng.normal(size=300)
    reference = rng.normal(size=300)
    scored = np.vstack(
        [reference, reference + rng.normal(scale=0.1, size=300), reference + rng.normal(scale=0.1, size=300)]
    )
    displayed = np.round(np.sqrt(np.mean((scored - truth) ** 2, axis=1)), 6)

    bound = rounded_rmse_geometry_bound(reference, scored, displayed, scored[1])

    assert float(bound["rounding_robust_rmse_upper"]) - float(bound["rounding_robust_rmse_lower"]) <= 1.1e-6
