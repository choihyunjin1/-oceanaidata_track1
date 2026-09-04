from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v13 as cycle  # noqa: E402


def _frame(rows: list[tuple[str, int, int, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["station", "year", "layer", "time", "current_router_prediction"])


def test_async_quorum_adds_with_two_other_recent_positive_layers() -> None:
    frame = _frame([("S", 2025, 1, "2025-01-01T00:00:00Z", 1), ("S", 2025, 2, "2025-01-01T00:00:00Z", 1), ("S", 2025, 3, "2025-01-01T00:10:00Z", 0)])
    assert cycle.async_peer_additions(frame, quorum=2, lookback_minutes=10).tolist() == [False, False, True]


def test_same_layer_is_excluded_and_gap_resets() -> None:
    frame = _frame([("S", 2025, 1, "2025-01-01T00:00:00Z", 1), ("S", 2025, 2, "2025-01-01T00:00:00Z", 1), ("S", 2025, 1, "2025-01-01T00:20:00Z", 0)])
    assert not cycle.async_peer_additions(frame, quorum=2, lookback_minutes=10).any()


def test_contract_is_fixed_zero_fit() -> None:
    config = cycle.load_contract()
    assert config["candidate"]["distinct_other_layer_quorum"] == 2
    assert config["candidate"]["lookback_minutes_inclusive"] == 10
    assert config["fit_budget"]["maximum"] == 0
    assert np.isclose(config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"], 0.015383691373120248)
