from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v12 as cycle  # noqa: E402


def test_exact_family_gate() -> None:
    config = cycle.load_contract()
    assert config["transport_family"]["family_id"] == "P1_FIXED_ADD_ONLY_UNION"
    assert np.isclose(config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"], 0.015383691373120248)


def test_causal_dilation_never_crosses_group_or_gap() -> None:
    frame = pd.DataFrame({"station": ["A", "A", "A", "B"], "year": [2025] * 4, "layer": [1, 1, 1, 1], "time": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:10:00Z", "2025-01-01T01:00:00Z", "2025-01-01T01:10:00Z"], utc=True)})
    additions = cycle.causal_lag1_mask(frame, np.array([1, 0, 1, 0], dtype=np.int8))
    assert additions.tolist() == [False, True, False, False]


def test_candidate_masks_are_add_only() -> None:
    frame = pd.DataFrame({"station": ["A", "A"], "year": [2025, 2025], "layer": [1, 1], "time": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:10:00Z"], utc=True), "current_router_prediction": [1, 0]})
    masks = cycle.candidate_masks(frame)
    assert all(np.all(mask >= frame.current_router_prediction.to_numpy()) for mask in masks.values())
