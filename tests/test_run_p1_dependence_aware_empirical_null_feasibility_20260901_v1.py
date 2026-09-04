from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p1_dependence_aware_empirical_null_feasibility_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_dependence_aware_empirical_null_feasibility_20260901_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location("dependence_null", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thresholds_and_all_stop_ceilings_are_sealed() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["sealed_null_calibration"]["alpha_sweep"] is False
    assert payload["sealed_null_calibration"]["sensitivity_block_lengths_rows"] == [72, 144, 288]
    assert payload["sealed_feasibility_ceilings"] == {
        "realized_false_alarm_rate_lte": 0.02,
        "proposal_row_share_lte": 0.02,
        "maximum_single_station_layer_proposal_concentration_lte": 0.70,
        "all_block_length_decisions_equal": True,
    }
    assert payload["operation_contract"]["supervised_model_fits"] == 0
    assert payload["decision"]["performance_claim"] is False


def test_moving_block_maxima_preserve_contiguous_blocks() -> None:
    module = _load()
    frame = pd.DataFrame({"station": ["G"] * 80, "layer": [1] * 80, "time": range(80)})
    observed = module._block_maxima(frame, np.arange(80, dtype=float), 4)
    assert observed.tolist()[:2] == [3.0, 7.0]
    assert observed[-1] == 79.0


def test_cross_layer_coherence_rejects_single_layer_excursion() -> None:
    module = _load()
    frame = pd.DataFrame({"station": ["G"] * 80, "layer": [1] * 40 + [2] * 40, "time": list(range(40)) * 2})
    signal = np.zeros(80, dtype=float)
    signal[20] = 9.0
    result = module._evaluate(frame, signal, threshold=5.0, block=2, min_layers=2)
    assert result["proposal_rows"] == 0
