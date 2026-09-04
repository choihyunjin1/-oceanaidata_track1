from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p1_dependence_calibrated_sparse_add_only_falsification_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_dependence_calibrated_sparse_add_only_falsification_20260901_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location("dependence_sparse", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rule_is_null_bound_sparse_add_only_without_label_tuning() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    rule = payload["sealed_proposal_rule"]
    assert payload["calibration_authority"]["required_decision"] == "CALIBRATION_FEASIBLE_RESEARCH_ONLY"
    assert rule["label_score_based_tuning_count"] == 0
    assert rule["operation"] == "bitwise_or_with_champion_anchor_no_removals"
    assert rule["maximum_total_proposal_row_share"] == 0.02
    assert payload["operation_contract"]["supervised_model_fits"] == 0


def test_sparse_segments_require_cross_layer_and_minimum_run() -> None:
    module = _load()
    times = pd.date_range("2025-01-01", periods=20, freq="10min", tz="UTC").astype(str).tolist()
    frame = pd.DataFrame({"station": ["G"] * 40, "layer": [1] * 20 + [2] * 20, "time": times * 2})
    signal = np.zeros(40)
    signal[2:14] = 10
    signal[22:34] = 10
    additions, audit = module._sparse_segments(frame, signal, 5, 2, 12)
    assert additions.sum() == 24
    assert audit["accepted_segments"] == 2


def test_single_layer_run_is_rejected() -> None:
    module = _load()
    times = pd.date_range("2025-01-01", periods=20, freq="10min", tz="UTC").astype(str).tolist()
    frame = pd.DataFrame({"station": ["G"] * 40, "layer": [1] * 20 + [2] * 20, "time": times * 2})
    signal = np.zeros(40)
    signal[:15] = 10
    additions, _audit = module._sparse_segments(frame, signal, 5, 2, 12)
    assert not additions.any()
