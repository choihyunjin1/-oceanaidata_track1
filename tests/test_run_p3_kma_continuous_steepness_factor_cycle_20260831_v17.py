from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_p3_kma_continuous_steepness_factor_cycle_20260831_v17 import (
    NEUTRAL_ECDF,
    compute_steepness,
    predict_policy,
    ranks_from_prefix,
)

ROOT = Path(__file__).resolve().parents[1]


def test_invalid_tp_uses_neutral_comparator_rank() -> None:
    z, valid = compute_steepness(np.asarray([1.0, 1.0]), np.asarray([0.0, 5.0]))
    rank = ranks_from_prefix(np.asarray([0.01, 0.02, 0.03]), z, valid)
    assert not valid[0]
    assert rank[0] == NEUTRAL_ECDF
    assert np.isfinite(rank[1])


def test_steepness_formula_is_log1p_hs_over_tp_squared() -> None:
    z, valid = compute_steepness(np.asarray([2.0]), np.asarray([4.0]))
    assert valid[0]
    np.testing.assert_allclose(z, np.log1p([2.0 / 16.0]))


def test_short_leads_are_exact_and_neutral_24h_is_comparator() -> None:
    frame = pd.DataFrame(
        {
            "lead_h": [3, 18, 24],
            "reference": [1.1, 1.425, 1.425],
            "base": [1.0, 1.0, 1.0],
            "delta": [1.0, 1.0, 1.0],
        }
    )
    prediction, alpha = predict_policy(frame, np.asarray([0.0, 0.0, NEUTRAL_ECDF]))
    assert prediction[0] == 1.1
    assert prediction[2] == frame.loc[2, "reference"]
    np.testing.assert_allclose(alpha, [0.0, 0.2, 0.425])


def test_terminal_fail_kept_official_access_zero() -> None:
    result = json.loads(
        (ROOT / "artifacts/p3_kma_continuous_steepness_factor_cycle_20260831_v17/result.json").read_text(encoding="utf-8")
    )
    assert result["passing_candidate_count"] == 0
    assert result["outputs"] == []
    assert all(value == 0 for value in result["data_access"].values())
