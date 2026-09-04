from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_p3_kma_continuous_energy_factor_cycle_20260831_v14b import (
    ecdf,
    predict_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ecdf_uses_right_continuous_prefix_rank() -> None:
    prefix = np.asarray([1.0, 2.0, 2.0, 4.0])
    actual = ecdf(prefix, np.asarray([0.0, 2.0, 3.0, 5.0]))
    np.testing.assert_allclose(actual, [0.0, 0.75, 0.75, 1.0])


def test_policy_keeps_short_leads_exact_and_uses_fixed_formula() -> None:
    frame = pd.DataFrame(
        {
            "lead_h": [3, 18, 24],
            "reference": [1.1, 1.425, 1.425],
            "base": [1.0, 1.0, 1.0],
            "delta": [1.0, 1.0, 1.0],
        }
    )
    prediction, alpha = predict_policy(frame, np.asarray([0.5, 0.5, 0.5]))
    assert prediction[0] == frame.loc[0, "reference"]
    np.testing.assert_allclose(prediction[1:], [1.2, 1.4])
    np.testing.assert_allclose(alpha, [0.0, 0.2, 0.4])


def test_terminal_result_has_no_official_access_on_fail() -> None:
    result = json.loads(
        (ROOT / "artifacts/p3_kma_continuous_energy_factor_cycle_20260831_v14b/result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "COMPLETE"
    assert result["passing_candidate_count"] == 0
    assert result["outputs"] == []
    assert all(value == 0 for value in result["data_access"].values())
    assert result["fit_budget"]["ecdf_calibration_fits"] == 6
    assert result["fit_budget"]["model_fits"] == 0
