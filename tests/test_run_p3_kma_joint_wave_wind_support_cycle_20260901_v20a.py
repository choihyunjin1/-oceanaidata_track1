from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_kma_joint_wave_wind_support_cycle_20260901_v20a.py"
CONFIG = ROOT / "configs/experiments/p3_kma_joint_wave_wind_support_cycle_20260901_v20a.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("p3_v20a", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_is_target_free_final_adjustment() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "SEALED_BEFORE_OUTER_SCORING"
    assert config["candidate"]["target_fits"] == 0
    assert config["candidate"]["grid_or_search"] is False
    assert config["duplication_audit"]["verdict"] == "NON_DUPLICATE_FINAL_ADJUSTMENT"
    assert all(value == 0 for value in config["official_policy"].values())


def test_wind_support_is_past_only_and_invalid_safe() -> None:
    runner = load_runner()
    frame = pd.DataFrame(
        {
            "wind_input_proxy_current": [3.0, 3.0, np.nan],
            "wind_wave_alignment_current": [0.5, -0.5, 0.5],
        }
    )
    score, valid = runner.wind_support(frame)
    assert score[0] == np.log1p(3.0) * 0.5
    assert score[1] == 0.0
    assert np.isnan(score[2])
    assert valid.tolist() == [True, True, False]


def test_prefix_rank_has_fixed_neutral_missing_fallback() -> None:
    runner = load_runner()
    rank = runner.ranks_from_prefix(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.5, np.nan, 3.0]),
        np.array([True, False, True]),
    )
    assert np.allclose(rank, [1.0 / 3.0, 0.5, 1.0])


def test_runner_has_no_official_csv_or_future_alignment_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    forbidden = (
        "official_frame(",
        "read_csv(",
        "to_csv(",
        "submission.csv",
        "future_kma",
        "future_era5",
        "anchor_time_recovery",
    )
    assert all(token not in source for token in forbidden)
