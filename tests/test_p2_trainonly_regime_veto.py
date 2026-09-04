from __future__ import annotations

import numpy as np
import pandas as pd

from p2_restore.trainonly_regime_veto import (
    circular_day_distance,
    paired_day_bootstrap_delta_rmse,
    season_bin,
    trainonly_regime_decisions,
)


def test_circular_distance_wraps_year_boundary() -> None:
    values = circular_day_distance(np.array([365, 2, 180]), 1.0)
    assert np.allclose(values, [2.0, 1.0, 179.0])


def test_bootstrap_detects_uniform_improvement() -> None:
    times = pd.date_range("2025-01-01", periods=30, freq="D", tz="Asia/Seoul")
    frame = pd.DataFrame(
        {
            "truth": np.zeros(30),
            "reference": np.ones(30),
            "candidate": np.full(30, 0.5),
            "kst_day": times.normalize().asi8,
        }
    )
    result = paired_day_bootstrap_delta_rmse(frame, replicates=200, seed=7)
    assert result["ci90_high"] < 0.0
    assert result["probability_improved"] == 1.0


def test_regime_veto_requires_support_and_negative_ci() -> None:
    times = pd.date_range("2024-06-01", periods=120, freq="D", tz="UTC")
    rows = []
    for index, timestamp in enumerate(times):
        for layer in (2, 3, 4):
            rows.append(
                {
                    "time": timestamp,
                    "source_block": "a" if index < 60 else "b",
                    "truth": 0.0,
                    "reference": 1.0,
                    "candidate": 0.5,
                    "layer": layer,
                }
            )
    decisions, receipts = trainonly_regime_decisions(
        pd.DataFrame(rows),
        pd.DatetimeIndex([pd.Timestamp("2024-07-15", tz="UTC")]),
        bin_days=14,
        window_days=60.0,
        minimum_source_blocks=2,
        minimum_profiles=100,
        minimum_kst_days=10,
        bootstrap_replicates=200,
        bootstrap_seed=11,
        ci90_upper_below=0.0,
    )
    assert any(decisions.values())
    assert any(record["support_ok"] for record in receipts.values())


def test_season_bin_is_stable_in_kst() -> None:
    times = pd.DatetimeIndex(["2024-01-01T14:30:00Z", "2024-01-01T15:30:00Z"])
    bins = season_bin(times, 14)
    assert bins.tolist() == [0, 0]
