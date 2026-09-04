from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v8 as cycle  # noqa: E402


def test_deployable_is_only_l2_drift_anchor_negative() -> None:
    frame = pd.DataFrame(
        {
            "layer": [2, 1, 2, 2],
            "e150_prediction": [0, 0, 1, 0],
            "pmax": [0.02, 0.02, 0.02, 0.001],
        }
    )
    assert cycle.deployable(frame).tolist() == [True, False, False, False]


def test_station_shrunk_rank_stays_in_unit_interval() -> None:
    frame = pd.DataFrame({"station": ["G-ORS", "G-ORS", "S-ORS", "S-ORS"]})
    mask = np.ones(4, dtype=bool)
    rank = cycle.rank_score(
        frame, mask, np.array([0.1, 0.9, 0.2, 0.8]), "shrunk", 0.5
    )
    assert np.all((rank >= 0.0) & (rank <= 1.0))
    assert rank[1] > rank[0]


def test_daily_budget_never_exceeds_one_per_station_day() -> None:
    frame = pd.DataFrame(
        {
            "station": ["G-ORS"] * 4,
            "time": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T01:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-02T01:00:00Z",
            ],
        }
    )
    additions = cycle.budgeted_additions(
        frame,
        np.ones(4, dtype=bool),
        np.array([0.9, 0.8, 0.7, 0.6]),
        fraction=1.0,
        denominator_rows=4,
        daily_cap=1,
    )
    assert additions.tolist() == [True, False, True, False]


def test_root_lcb_and_raw_point_gates_are_exact() -> None:
    policy = cycle.load_contract()["decision_policy"]
    assert np.isclose(policy["bootstrap_ci90_low_minimum"], 0.0005788103467134221)
    assert np.isclose(
        policy["minimum_raw_expected_point_delta_inclusive"],
        0.015383691373120248,
    )


def test_native_prevents_numpy_bool_terminal_failure() -> None:
    assert cycle.native({"ok": np.bool_(True)}) == {"ok": True}
