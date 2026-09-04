from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p1_robust_subspace_block_conformal_feasibility_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_robust_subspace_block_conformal_feasibility_20260901_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location("subspace_conformal", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_method_and_all_ceilings_are_frozen() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    method = payload["frozen_method"]
    assert method["subspace_rank"] == 1
    assert method["block_lengths_rows"] == [72, 144, 288]
    assert method["e_value"] == "0.5/sqrt(p)"
    assert method["e_bh_q"] == 0.01
    assert method["result_based_parameter_choices"] == 0
    assert payload["operations"]["supervised_fits"] == payload["operations"]["target_reads"] == 0


def test_common_factor_removes_shared_cross_layer_shift() -> None:
    module = _load()
    frame = pd.DataFrame({"station": ["G"] * 6, "layer": [1, 2, 3] * 2, "time": ["a"] * 3 + ["b"] * 3, "temp": [5, 5, 5, 8, 8, 8]})
    params = {("G", layer): (0.0, 1.0) for layer in (1, 2, 3)}
    assert np.allclose(module._innovation(frame, params), 0.0)


def test_e_bh_with_valid_bounded_evalues_rejects_none() -> None:
    module = _load()
    times = list(range(1440))
    calibration = pd.DataFrame({"station": ["G"] * 1440, "layer": [1] * 1440, "time": times})
    heldout = calibration.copy()
    result = module._evaluate(calibration, np.zeros(1440), heldout, np.zeros(1440), 72, 0.01, 1, 1)
    assert result["rejected_blocks"] == 0
    assert result["proposal_rows"] == 0
