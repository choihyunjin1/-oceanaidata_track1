from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v14_causal_prefix_dictionary_discord_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v14_causal_prefix_dictionary_discord_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v14_discord_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _frame(station: str = "I-ORS", rows: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({"station": [station] * rows, "layer": [2] * rows, "_time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"), "temp": np.sin(np.arange(rows) / 8.0)})


def _representation():
    return json.loads(CONFIG.read_text(encoding="utf-8"))["representation"]


def test_ns_cutoffs_and_future_invariance() -> None:
    module = _module()
    times = module.core._time_ns(pd.Series(pd.date_range("2024-01-01", periods=3, freq="10min", tz="UTC")))
    assert times[0] > 10**18 and times[1] - times[0] == 600_000_000_000
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cutoffs = [pd.Timestamp(json.loads((ROOT / item["audit"]).read_text(encoding="utf-8"))["adjusted_cutoff_utc"]).value for item in config["parts"].values()]
    assert len(set(cutoffs)) == 3
    original = _frame()
    changed = original.copy()
    changed.loc[900:, "temp"] = 1000.0
    boundary = original.loc[850, "_time"].value
    first = module.discord_features(original, boundary, _representation())
    second = module.discord_features(changed, boundary, _representation())
    assert first.shape == (1000, 10) and np.isfinite(first).all()
    np.testing.assert_allclose(first[:900], second[:900], atol=0, rtol=0)


def test_group_and_gap_reset() -> None:
    module = _module()
    first, second = _frame("A"), _frame("B")
    second.index = np.arange(len(first), len(first) + len(second))
    boundary = first.loc[850, "_time"].value
    together = module.discord_features(pd.concat([first, second]), boundary, _representation())
    second.index = np.arange(len(second))
    alone = module.discord_features(second, boundary, _representation())
    np.testing.assert_allclose(together[len(first) :], alone, atol=0, rtol=0)
    gapped = first.copy()
    gapped.loc[900:, "_time"] += pd.Timedelta(minutes=10)
    reset = module.discord_features(gapped, boundary, _representation())
    assert np.all(reset[900] == 0)


def test_no_subsequence_normalization_preserves_level_offset_discord() -> None:
    module = _module()
    values = np.sin(np.arange(1200) / 8.0)
    baseline = module._profile_segment(values, 48, tuple(range(2, 18)))
    shifted = values.copy()
    shifted[1050:] += 4.0
    discord = module._profile_segment(shifted, 48, tuple(range(2, 18)))
    assert discord[1100:, 0].mean() > baseline[1100:, 0].mean()


def test_boundary_metric_and_add_only_transport_contract() -> None:
    module = _module()
    rows = 30
    truth = np.r_[np.zeros(3), np.ones(24), np.zeros(3)].astype(np.int8)
    anchor = np.zeros(rows, dtype=np.int8)
    candidate = anchor.copy()
    candidate[3:9] = 1
    candidate[21:27] = 1
    metadata = pd.DataFrame({"station": ["A"] * rows, "layer": [1] * rows, "time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC")})
    receipt = module._boundary_recall(truth, anchor, candidate, metadata)
    assert receipt["runs"] == 1 and receipt["candidate_recall"] == 1.0
    assert module.core._time_ns(metadata["time"])[1] - module.core._time_ns(metadata["time"])[0] == 600_000_000_000
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(config["model"]["seeds"]) * len(config["parts"]) == config["model"]["fits"] == 9
    assert config["selection"]["transport_stability"]["require_both_chronological_halves"]
    assert config["anchor"]["removals"] == config["selection"]["outer_tuning"] == 0
