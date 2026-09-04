from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v14 as cycle  # noqa: E402


def _frame(layers: list[int], anchors: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["S"] * len(layers),
            "year": [2025] * len(layers),
            "layer": layers,
            "time": ["2025-01-01T00:00:00Z"] * len(layers),
            "current_router_prediction": anchors,
        }
    )


def test_exact_vertical_bracket_adds_only_interior_zero() -> None:
    frame = _frame([1, 2, 3, 4], [1, 0, 1, 0])
    additions = cycle.vertical_bracket_additions(frame, known_ranks=tuple(range(1, 9)))
    assert additions.tolist() == [False, True, False, False]


def test_missing_neighbor_edge_and_duplicate_fail_closed() -> None:
    missing = _frame([1, 3, 4], [1, 0, 1])
    duplicate = _frame([1, 2, 2, 3], [1, 0, 0, 1])
    assert not cycle.vertical_bracket_additions(missing, known_ranks=tuple(range(1, 9))).any()
    assert not cycle.vertical_bracket_additions(duplicate, known_ranks=tuple(range(1, 9))).any()


def test_contract_is_fixed_and_untrimmed() -> None:
    config = cycle.load_contract()
    assert config["candidate"]["exact_layer_rank_step"] == 1
    assert config["fit_budget"]["maximum"] == 0
    assert config["safety"]["proposal_caps_are_gates_not_trimming"] is True
    assert np.isclose(
        config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
        0.015383691373120248,
    )
