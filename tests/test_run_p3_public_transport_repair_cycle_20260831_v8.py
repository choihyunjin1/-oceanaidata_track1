from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_public_transport_repair_cycle_20260831_v8.py"
SPEC = importlib.util.spec_from_file_location("p3_v8_runner", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    rows = 6
    frame = pd.DataFrame(
        {
            column: np.full(rows, 1.0, dtype=float) for column in MODULE.CASE_FEATURES
        }
    )
    frame["hs_current"] = 1.8
    frame["hs_delta_12h"] = 0.3
    frame["lead_h"] = list(MODULE.LEADS)
    frame["station"] = "G-ORS"
    frame["incumbent_prediction"] = 1.9
    return frame


def test_candidate_contract_is_three_unique_structural_families() -> None:
    assert len(MODULE.SPECS) == 3
    assert len({item.name for item in MODULE.SPECS}) == 3
    assert {item.family for item in MODULE.SPECS} == {"huber", "ridge", "extra_trees"}


def test_design_is_finite_and_has_station_lead_terms() -> None:
    matrix = MODULE.design(_frame())
    assert matrix.shape == (6, len(MODULE.CASE_FEATURES) + 11)
    assert np.isfinite(matrix).all()


def test_support_requires_selection_matched_regime() -> None:
    frame = _frame()
    stats = MODULE.support_stats(frame)
    assert MODULE.supported(frame, stats).all()
    frame.loc[0, "hs_delta_12h"] = 0.1
    assert not MODULE.supported(frame, stats)[0]


def test_transport_gate_constants_match_authoritative_calibration() -> None:
    penalty, evidence = MODULE.load_transport_penalty()
    assert penalty == 0.3219056897594759
    assert MODULE.MIN_CALIBRATED_POINTS == 0.01
    assert evidence["official_p3"]["official_metric"] == 0.590956


def test_rmse_rejects_nonfinite() -> None:
    with np.testing.assert_raises(MODULE.ContractError):
        MODULE.rmse(np.array([1.0]), np.array([np.nan]))
