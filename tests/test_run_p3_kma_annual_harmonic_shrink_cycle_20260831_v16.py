from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_p3_kma_annual_harmonic_shrink_cycle_20260831_v16 import (
    fit_theta,
    predict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_empty_prefix_abstains_to_v14b() -> None:
    theta, receipt = fit_theta(pd.DataFrame(), np.asarray([]))
    np.testing.assert_array_equal(theta, [0.0, 0.0])
    assert receipt["action"] == "abstain_empty_prefix"


def test_short_phase_coverage_abstains() -> None:
    train = pd.DataFrame(
        {
            "anchor_time": pd.to_datetime(["2025-01-01", "2025-02-01"], utc=True),
            "target_hs": [1.0, 1.0],
            "reference": [1.0, 1.0],
        }
    )
    theta, receipt = fit_theta(train, np.asarray([0.1, -0.1]))
    np.testing.assert_array_equal(theta, [0.0, 0.0])
    assert receipt["action"] == "abstain_phase_or_condition"


def test_predict_is_continuous_shrink_and_noop_when_theta_zero() -> None:
    frame = pd.DataFrame(
        {
            "anchor_time": pd.to_datetime(["2025-01-01", "2025-07-01"], utc=True),
            "reference": [1.0, 2.0],
        }
    )
    correction = np.asarray([0.2, -0.3])
    prediction, multiplier = predict(frame, correction, np.zeros(2))
    np.testing.assert_allclose(multiplier, [1.0, 1.0])
    np.testing.assert_allclose(prediction, [1.2, 1.7])


def test_terminal_failed_only_transport_lcb_and_read_no_official_data() -> None:
    result = json.loads(
        (ROOT / "artifacts/p3_kma_annual_harmonic_shrink_cycle_20260831_v16/result.json").read_text(encoding="utf-8")
    )
    failed = {key for key, value in result["candidate"]["gate_checks"].items() if not value}
    assert failed == {
        "raw_lcb_points_meets_family_threshold",
        "calibrated_lcb_at_least_0p01",
    }
    assert result["outputs"] == []
    assert all(value == 0 for value in result["data_access"].values())
