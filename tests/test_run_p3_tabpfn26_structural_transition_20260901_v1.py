from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_p3_tabpfn26_structural_transition_20260901_v1.py"


def _runner():
    name = "test_p3_tabpfn26_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_lead_specific_matrix_has_station_plus_numeric_features() -> None:
    runner = _runner()
    frame = pd.DataFrame(
        {
            "station": ["G-ORS", "S-ORS", "G-ORS"],
            "lead_h": [1, 1, 2],
            "current_hs_for_residual": [1.0, 2.0, 3.0],
            "feature_a": [4.0, np.inf, 6.0],
        }
    )
    matrix = runner.lead_specific_matrix(frame, lead=1)
    assert matrix.shape == (2, 3)
    np.testing.assert_array_equal(matrix[:, 0], [0.0, 1.0])
    assert np.isnan(matrix[1, 2])


def test_frozen_p3_transition_config_and_pins_are_valid() -> None:
    runner = _runner()
    config = runner._config()
    assert config["features"]["base_feature_count"] == 591
    assert config["model"]["maximum_fits"] == 18
    assert config["validation"]["fold_or_station_tolerance"] is None
