from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_matched_budget_local_compare_20260825_v1.py"
CONFIG_PATH = ROOT / "configs/experiments/p1_matched_budget_local_compare_20260825_v1.json"


def _runner():
    spec = importlib.util.spec_from_file_location("p1_matched_budget_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sealed_contract_has_equal_three_by_three_budget_and_no_forbidden_inputs() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["matched_tuning"]["settings_per_family"] == 3
    assert config["matched_tuning"]["seed_count_per_family"] == 3
    assert config["surface"]["seeds"] == [20260813, 20260829, 20260847]
    assert len(config["families"]["incumbent_offline_xgboost"]["settings"]) == 3
    assert len(config["families"]["causal_event_rescue_ensemble"]["settings"]) == 3
    inputs = [config["common_protocol"], *config["inputs"]["full_prefix_parts"]]
    inputs += [
        config["inputs"]["incumbent_oof"],
        config["inputs"]["causal_oof"],
        config["inputs"]["historical_round_a_oof"],
    ]
    forbidden = tuple(config["prohibitions"]["read_paths_containing"])
    for value in inputs:
        path = value["path"].lower()
        assert not any(token in path for token in forbidden)


def test_setting_selection_uses_registered_order_for_exact_tie() -> None:
    runner = _runner()
    truth = np.array([0, 1, 0, 1], dtype=np.int8)
    predictions = {
        "first": np.array([0, 1, 0, 1], dtype=np.int8),
        "second": np.array([0, 1, 0, 1], dtype=np.int8),
        "third": np.array([1, 1, 0, 0], dtype=np.int8),
    }
    selected, scores = runner._select_best_setting(
        predictions,
        ["first", "second", "third"],
        truth,
        np.ones(len(truth), dtype=bool),
    )
    assert selected == "first"
    assert len(scores) == 3


def test_kst_day_bootstrap_is_deterministic_and_paired() -> None:
    runner = _runner()
    frame = pd.DataFrame(
        {
            "station": ["A"] * 6,
            "layer": [1] * 6,
            "time": [
                "2025-01-01T00:00:00+09:00",
                "2025-01-01T00:10:00+09:00",
                "2025-01-02T00:00:00+09:00",
                "2025-01-02T00:10:00+09:00",
                "2025-01-03T00:00:00+09:00",
                "2025-01-03T00:10:00+09:00",
            ],
        }
    )
    truth = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
    baseline = np.array([0, 0, 0, 0, 0, 0], dtype=np.int8)
    candidate = truth.copy()
    first = runner.paired_kst_day_bootstrap(
        truth, candidate, baseline, frame, replicates=200, seed=17
    )
    second = runner.paired_kst_day_bootstrap(
        truth, candidate, baseline, frame, replicates=200, seed=17
    )
    assert first == second
    assert first["days"] == 3
    assert first["delta_ci90"][0] > 0


def test_normal_fp_day_counts_station_layer_day_blocks() -> None:
    runner = _runner()
    frame = pd.DataFrame(
        {
            "station": ["A", "A", "A", "B"],
            "layer": [1, 1, 2, 1],
            "time": [
                "2025-01-01T00:00:00+09:00",
                "2025-01-01T00:10:00+09:00",
                "2025-01-01T00:00:00+09:00",
                "2025-01-01T00:00:00+09:00",
            ],
        }
    )
    truth = np.zeros(4, dtype=np.int8)
    prediction = np.array([1, 1, 0, 1], dtype=np.int8)
    values = runner.normal_fp_day_metrics(truth, prediction, frame)
    assert values["normal_station_layer_kst_days"] == 3
    assert values["false_positive_rows"] == 3
    assert values["normal_station_layer_kst_days_with_fp"] == 2


def test_binary_metrics_matches_hand_calculation() -> None:
    runner = _runner()
    values = runner.binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert values["tp"] == 1
    assert values["fp"] == 1
    assert values["fn"] == 1
    assert values["f1"] == 0.5
