from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_p1_causal_endpoint_peer_inpaint_20260831_v32c.py"
SPEC = importlib.util.spec_from_file_location("v32c", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def frame(anchors: list[int], peer: list[int]) -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=len(anchors), freq="10min", tz="UTC")
    rows = []
    for index, time in enumerate(times):
        rows.extend(
            [
                {"station": "S", "year": 2025, "layer": 1, "time": time, "current_router_prediction": anchors[index]},
                {"station": "S", "year": 2025, "layer": 2, "time": time, "current_router_prediction": peer[index]},
            ]
        )
    return pd.DataFrame(rows)


def test_requires_trailing_endpoint_and_peer() -> None:
    values = frame([1, 0, 0], [1, 1, 0])
    additions = MODULE.causal_endpoint_peer_additions(values)
    assert np.flatnonzero(additions).tolist() == [2]


def test_future_anchor_cannot_change_past_proposal() -> None:
    first = frame([1, 0, 0], [1, 0, 0])
    second = first.copy()
    second.loc[second.index[-1], "current_router_prediction"] = 1
    before = MODULE.causal_endpoint_peer_additions(first)
    after = MODULE.causal_endpoint_peer_additions(second)
    assert np.array_equal(before[:-2], after[:-2])


def test_contract_is_zero_fit() -> None:
    contract = MODULE.load_contract()
    assert contract["fit_budget"]["maximum"] == 0
    assert contract["candidate"]["threshold_grid"] == 0
