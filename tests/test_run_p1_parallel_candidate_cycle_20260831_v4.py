from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_p1_parallel_candidate_cycle_20260831_v4.py"
)
SPEC = importlib.util.spec_from_file_location("p1_parallel_v4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    rows = 8
    return pd.DataFrame(
        {
            "station": ["G-ORS"] * rows,
            "layer": [1] * rows,
            "time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"),
            "probability_base": [0.0, 0.1, 0.7, 0.8, 0.2, 0.1, 0.2, 0.3],
            "probability_peer": [0.0, 0.1, 0.6, 0.7, 0.2, 0.1, 0.2, 0.3],
            "deployment_prediction_base": [0, 0, 1, 1, 0, 0, 0, 0],
            "deployment_prediction_peer": [0, 0, 1, 1, 0, 0, 0, 0],
            "e150_probability": [0.0, 0.1, 0.9, 0.9, 0.4, 0.1, 0.2, 0.3],
            "e150_boundary_start": [0.0] * rows,
            "e150_boundary_end": [0.0] * rows,
            "e150_prediction": [0, 0, 1, 1, 0, 0, 0, 0],
            "station_code": [0.0] * rows,
            "layer_code": [1.0] * rows,
            "month_sin": [0.5] * rows,
            "month_cos": [0.5] * rows,
            "hour_sin": [0.0] * rows,
            "hour_cos": [1.0] * rows,
        }
    )


def test_causal_features_do_not_change_before_future_mutation() -> None:
    original, columns = MODULE.add_causal_features(_frame())
    changed = _frame()
    changed.loc[7, ["probability_base", "probability_peer", "e150_probability"]] = 1.0
    mutated, mutated_columns = MODULE.add_causal_features(changed)
    assert columns == mutated_columns
    np.testing.assert_allclose(
        original.loc[:6, columns].to_numpy(float),
        mutated.loc[:6, columns].to_numpy(float),
    )


def test_boundary_mask_only_selects_post_anchor_rows() -> None:
    enriched, _ = MODULE.add_causal_features(_frame())
    mask = MODULE.SPECS[0].mask(enriched)
    assert np.flatnonzero(mask).tolist() == [4, 5, 6]
    assert not np.any(mask & enriched["e150_prediction"].eq(1).to_numpy())


def test_run_state_mask_is_add_only() -> None:
    enriched, _ = MODULE.add_causal_features(_frame())
    mask = MODULE.SPECS[1].mask(enriched)
    assert not np.any(mask & enriched["e150_prediction"].eq(1).to_numpy())
    assert np.all(enriched.loc[mask, "signal_run_length"].to_numpy() >= 2)
