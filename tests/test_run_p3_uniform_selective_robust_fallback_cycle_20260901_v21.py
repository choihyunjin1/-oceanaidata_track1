from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_uniform_selective_robust_fallback_cycle_20260901_v21.py"
CONFIG = ROOT / "configs/experiments/p3_uniform_selective_robust_fallback_cycle_20260901_v21.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("p3_v21", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_seals_two_target_free_candidates_and_zero_official_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "SEALED_BEFORE_OUTER_SCORING"
    assert len(config["candidates"]) == 2
    assert config["gate"]["threshold_search"] is False
    assert config["gate"]["target_rows_used_to_fit_gate"] == 0
    assert config["decision"]["no_result_based_retry"] is True
    assert all(value == 0 for value in config["official_policy"].values())


def test_component_spread_uses_three_components() -> None:
    runner = load_runner()
    frame = pd.DataFrame(
        {
            "base": [1.0, 2.0],
            "reference": [1.5, 2.1],
            "current_hs": [2.0, 1.9],
        }
    )
    assert np.allclose(runner.component_spread(frame), [1.0, 0.2])


def test_sparse_gate_is_long_lead_only_and_training_input_only() -> None:
    runner = load_runner()
    rows = 20
    train = pd.DataFrame(
        {
            "lead_h": np.tile([18, 24], rows // 2),
            "base": np.linspace(1.0, 2.0, rows),
            "reference": np.linspace(1.02, 2.2, rows),
            "current_hs": np.linspace(1.0, 3.0, rows),
            "hs_std_24h": np.linspace(0.1, 1.0, rows),
            "hmax_hs_ratio_current": np.linspace(1.1, 2.0, rows),
        }
    )
    valid = pd.DataFrame(
        {
            "lead_h": [3, 18, 24],
            "base": [1.0, 1.0, 1.0],
            "reference": [1.0, 1.1, 1.1],
            "current_hs": [10.0, 10.0, 10.0],
            "hs_std_24h": [10.0, 10.0, 10.0],
            "hmax_hs_ratio_current": [10.0, 10.0, 10.0],
        }
    )
    gate, receipt = runner.fixed_sparse_gate(train, valid)
    assert gate.tolist() == [False, True, True]
    assert receipt["target_rows_read_before_gate_fixed"] == 0
    assert receipt["validation_rows"] == 3


def test_winsor_limits_are_training_only_and_do_not_delete() -> None:
    runner = load_runner()
    values = np.array([0.0, 1.0, 2.0, 1000.0])
    low, high = runner.winsor_limits(values)
    clipped = np.clip(values, low, high)
    assert len(clipped) == len(values)
    assert clipped[-1] < values[-1]


def test_runner_has_no_official_csv_hidden_or_future_alignment_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    forbidden = (
        "official_frame(",
        "read_csv(",
        "to_csv(",
        "submission.csv",
        "load_hidden",
        "future_kma",
        "future_era5",
        "anchor_time_recovery",
    )
    assert all(token not in source for token in forbidden)
