from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_p3_kma_continuous_energy_affine_cycle_20260831_v15b import (
    THETA_PRIOR,
    fit_theta,
    predict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_empty_prefix_uses_exact_prior() -> None:
    prior = {
        "feature_rms": [2.0, 1.0],
        "prior_beta": [0.4, 0.4],
        "ridge_diag": [1.0, 1.0],
    }
    theta, receipt = fit_theta(pd.DataFrame(), np.asarray([]), prior)
    np.testing.assert_allclose(theta, THETA_PRIOR)
    assert receipt["action"] == "prior_only_empty_prefix"


def test_affine_policy_keeps_short_lead_exact() -> None:
    frame = pd.DataFrame(
        {
            "lead_h": [3, 18, 24],
            "reference": [1.1, 1.425, 1.425],
            "base": [1.0, 1.0, 1.0],
            "delta": [1.0, 1.0, 1.0],
        }
    )
    prediction, alpha = predict(frame, np.asarray([0.0, 0.0, 0.75]), np.asarray([0.2, 0.4]))
    assert prediction[0] == 1.1
    np.testing.assert_allclose(alpha, [0.0, 0.2, 0.3])
    np.testing.assert_allclose(prediction[1:], [1.2, 1.3])


def test_terminal_zero_pass_has_zero_official_access() -> None:
    result = json.loads(
        (ROOT / "artifacts/p3_kma_continuous_energy_affine_cycle_20260831_v15b/result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "COMPLETE"
    assert result["passing_candidate_count"] == 0
    assert result["outputs"] == []
    assert all(value == 0 for value in result["data_access"].values())
