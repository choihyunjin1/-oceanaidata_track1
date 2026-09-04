from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.supervised_rank1_functional_residual import (
    SupervisedRank1Residual,
    orthogonal_share,
)


def test_rank_one_residual_recovers_shared_response_direction() -> None:
    rng = np.random.default_rng(20260828)
    rows = 300
    latent = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "public_profile_valid": np.ones(rows),
            "x1": latent + rng.normal(0.0, 0.05, rows),
            "x2": rng.normal(size=rows),
        }
    )
    response = latent[:, None] * np.asarray([[1.0, -0.5, 0.25]])
    times = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    model = SupervisedRank1Residual.fit(frame, response, times)
    prediction, enabled, _ = model.predict(frame, times)
    assert enabled.mean() >= 0.95
    assert np.sqrt(np.mean(np.square(prediction - response))) < 0.2


def test_orthogonal_share_is_one_for_perpendicular_vectors() -> None:
    assert np.isclose(orthogonal_share(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 1.0)
