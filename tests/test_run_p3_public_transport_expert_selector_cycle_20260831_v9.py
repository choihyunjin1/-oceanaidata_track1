from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "scripts/run_p3_public_transport_expert_selector_cycle_20260831_v9.py"
SPEC = importlib.util.spec_from_file_location("p3_v9_runner", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    return MODULE.attach_regime(
        pd.DataFrame(
            {
                "station": ["G-ORS"] * 6,
                "lead_h": [18] * 6,
                "hs_current": [1.8] * 6,
                "hs_delta_12h": [0.3] * 6,
                "reference": [2.0] * 6,
                "base": [1.8] * 6,
                "delta": [0.4] * 6,
                "target_hs": [1.8] * 6,
            }
        )
    )


def test_three_unique_selector_contracts() -> None:
    assert len(MODULE.SPECS) == 3
    assert len({item.name for item in MODULE.SPECS}) == 3
    assert min(MODULE.SPECS[2].alpha_bank) == MODULE.REFERENCE_ALPHA


def test_selection_mask_enforces_high_rising_regime() -> None:
    frame = _frame()
    assert MODULE.selection_mask(frame).all()
    frame.loc[0, "hs_delta_12h"] = 0.1
    assert not MODULE.selection_mask(frame)[0]


def test_selector_falls_back_to_exact_reference() -> None:
    frame = _frame()
    prediction, alpha = MODULE.apply_selector(frame, MODULE.SPECS[0], {})
    assert np.array_equal(prediction, frame["reference"].to_numpy())
    assert np.all(alpha == MODULE.REFERENCE_ALPHA)


def test_prior_fit_selects_supported_better_expert() -> None:
    frame = _frame()
    table, receipt = MODULE.fit_selector(frame, MODULE.SPECS[1])
    assert receipt["fit_count"] == 1
    assert table
    prediction, _ = MODULE.apply_selector(frame, MODULE.SPECS[1], table)
    assert np.allclose(prediction, 1.8)


def test_transport_penalty_is_authoritative() -> None:
    assert MODULE.load_penalty() == 0.3219056897594759
