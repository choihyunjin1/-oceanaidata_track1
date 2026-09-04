from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v18_causal_state_dependent_diffusion_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v18_causal_state_dependent_diffusion_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v18_diffusion_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _representation():
    return json.loads(CONFIG.read_text(encoding="utf-8"))["representation"]


def _frame(station: str = "I-ORS", rows: int = 1100) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    values = np.cumsum(rng.normal(scale=0.2, size=rows))
    return pd.DataFrame({"station": [station] * rows, "layer": [2] * rows, "_time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"), "temp": values})


def test_state_dependent_diffusion_standardizes_and_is_finite() -> None:
    module = _module()
    rng = np.random.default_rng(3)
    values = np.zeros(1200)
    for index in range(1, len(values)):
        scale = 0.1 if values[index - 1] < 0 else 0.8
        values[index] = 0.95 * values[index - 1] + rng.normal(scale=scale)
    features = module.conditional_moment_segment(values, 900, _representation())
    assert features.shape == (1200, 8) and np.isfinite(features).all()
    assert np.ptp(features[901:, 1]) > 0


def test_ns_cutoffs_and_future_invariance() -> None:
    module = _module()
    times = module.core._time_ns(pd.Series(pd.date_range("2024-01-01", periods=3, freq="10min", tz="UTC")))
    assert times[0] > 10**18 and times[1] - times[0] == 600_000_000_000
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cutoffs = [pd.Timestamp(json.loads((ROOT / item["audit"]).read_text(encoding="utf-8"))["adjusted_cutoff_utc"]).value for item in config["parts"].values()]
    assert len(set(cutoffs)) == 3
    original = _frame()
    changed = original.copy()
    changed.loc[950:, "temp"] = 1000.0
    boundary = original.loc[900, "_time"].value
    first = module.diffusion_features(original, boundary, _representation())
    second = module.diffusion_features(changed, boundary, _representation())
    np.testing.assert_allclose(first[:950], second[:950], atol=0, rtol=0)


def test_group_and_gap_reset() -> None:
    module = _module()
    first, second = _frame("A"), _frame("B")
    second.index = np.arange(len(first), len(first) + len(second))
    boundary = first.loc[900, "_time"].value
    together = module.diffusion_features(pd.concat([first, second]), boundary, _representation())
    second.index = np.arange(len(second))
    alone = module.diffusion_features(second, boundary, _representation())
    np.testing.assert_allclose(together[len(first) :], alone, atol=0, rtol=0)
    gapped = first.copy()
    gapped.loc[950:, "_time"] += pd.Timedelta(minutes=10)
    reset = module.diffusion_features(gapped, boundary, _representation())
    assert np.all(reset[950] == 0)


def test_add_only_transport_budget_and_access_contract() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(config["model"]["seeds"]) * len(config["parts"]) == config["model"]["fits"] == 9
    assert config["selection"]["transport_stability"]["require_both_chronological_halves"]
    assert config["anchor"]["removals"] == config["selection"]["outer_tuning"] == 0
    assert config["source"]["official_test_sample_submission_hidden_reads"] == 0
