from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_p3_parallel_candidate_cycle_20260831_v4.py"
)
SPEC = importlib.util.spec_from_file_location("p3_parallel_v4", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for case in range(90):
        station = ("G-ORS", "I-ORS", "S-ORS")[case % 3]
        current = 1.5 + 0.02 * case
        for lead in MODULE.ALL_LEADS:
            base = current + 0.01 * lead
            delta = 0.0 if lead < 18 else 0.2 + 0.01 * (case % 5)
            target = base + (0.15 if current < 2.4 else 0.50) * delta
            rows.append(
                {
                    "anchor_id": case,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                    "base": base,
                    "delta": delta,
                    "reference": base + (MODULE.REFERENCE_ALPHA * delta if lead >= 18 else 0.0),
                    "target_hs": target,
                }
            )
    return pd.DataFrame(rows)


def test_state_basis_is_finite_and_deployable() -> None:
    frame = synthetic_frame()
    basis = MODULE.state_basis(frame)
    assert basis.shape == (len(frame), 19)
    assert np.isfinite(basis).all()


def test_all_frozen_models_preserve_short_leads_and_bounds() -> None:
    frame = synthetic_frame()
    inactive = ~frame["lead_h"].isin(MODULE.ACTIVE_LEADS).to_numpy()
    for index, spec in enumerate(MODULE.SPECS):
        model = MODULE.PhysicalAxisModel(spec, 100 + index).fit(frame)
        prediction, alpha = model.predict(frame)
        assert np.array_equal(prediction[inactive], frame.loc[inactive, "reference"].to_numpy())
        assert np.isfinite(prediction).all()
        assert np.all(alpha >= MODULE.ALPHA_MIN)
        assert np.all(alpha <= MODULE.ALPHA_MAX)


def test_rmse_rejects_nonfinite() -> None:
    try:
        MODULE.rmse(np.array([1.0]), np.array([np.nan]))
    except MODULE.ContractError:
        pass
    else:
        raise AssertionError("non-finite RMSE input was accepted")
