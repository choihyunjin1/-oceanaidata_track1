from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v19 as cycle  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["G-ORS", "G-ORS", "G-ORS", "G-ORS", "I-ORS", "I-ORS"],
            "year": [2025] * 6,
            "layer": [1] * 6,
            "time": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:10:00Z",
                    "2025-01-01T00:20:00Z",
                    "2025-01-01T01:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:10:00Z",
                ],
                utc=True,
            ),
        }
    )


def test_contract_is_exact_zero_fit_family() -> None:
    config = cycle.load_contract()
    assert config["candidate"]["station"] == "G-ORS"
    assert config["candidate"]["cadence_minutes"] == 10
    assert config["candidate"]["span_rows"] == 1
    assert config["candidate"]["recursive_extension"] is False
    assert config["fit_budget"]["maximum_model_fits"] == 0
    assert np.isclose(
        config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
        0.015383691373120248,
    )


def test_gors_rule_is_one_step_nonrecursive_and_gap_safe() -> None:
    frame = _frame()
    reference = np.array([1, 0, 0, 1, 1, 0], dtype=np.int8)
    additions = cycle.gors_causal_lag1_mask(frame, reference)
    assert additions.tolist() == [False, True, False, False, False, False]


def test_other_stations_remain_bit_exact() -> None:
    frame = _frame()
    reference = np.array([1, 0, 0, 1, 1, 0], dtype=np.int8)
    candidate = cycle.build_candidate(frame, reference)
    other = frame["station"].ne("G-ORS").to_numpy()
    assert np.array_equal(candidate[other], reference[other])
    assert np.all(candidate >= reference)


def test_group_boundary_never_transports_a_positive() -> None:
    frame = pd.DataFrame(
        {
            "station": ["G-ORS", "G-ORS"],
            "year": [2024, 2025],
            "layer": [1, 1],
            "time": pd.to_datetime(
                ["2024-12-31T15:50:00Z", "2024-12-31T16:00:00Z"], utc=True
            ),
        }
    )
    additions = cycle.gors_causal_lag1_mask(
        frame, np.array([1, 0], dtype=np.int8)
    )
    assert additions.tolist() == [False, False]
