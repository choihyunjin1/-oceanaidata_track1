from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_p1_tabpfn3_structural_transition_20260901_v1r1.py"


def _runner():
    name = "test_p1_tabpfn3_runner_v1r1"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_binary_sampler_preserves_every_positive_and_caps_rows() -> None:
    runner = _runner()
    labels = np.asarray([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    keys = pd.DataFrame(
        {
            "station": ["S"] * len(labels),
            "year": [2025] * len(labels),
            "layer": [1] * len(labels),
            "time": np.arange(len(labels)),
        }
    )
    selected = runner.deterministic_binary_sample(labels, keys, cap=5, seed=7)
    assert len(selected) == 5
    assert {0, 3}.issubset(set(selected.tolist()))
    assert int(labels[selected].sum()) == 2
    np.testing.assert_array_equal(
        selected,
        runner.deterministic_binary_sample(labels, keys, cap=5, seed=7),
    )


def test_frozen_p1_transition_config_and_pins_are_valid() -> None:
    runner = _runner()
    config = runner._config()
    assert config["features"]["expected_count"] == 165
    assert config["model"]["threshold"] == 0.5
    assert config["protocol"]["threshold_tuning"] is False
