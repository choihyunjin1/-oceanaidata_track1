from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v13_causal_endpoint_visibility_topology_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v13_causal_endpoint_visibility_topology_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v13_visibility_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _frame(station: str = "I-ORS", rows: int = 220) -> pd.DataFrame:
    return pd.DataFrame({"station": [station] * rows, "layer": [2] * rows, "_time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"), "temp": np.sin(np.arange(rows) / 8.0)})


def _representation():
    return json.loads(CONFIG.read_text(encoding="utf-8"))["representation"]


def test_ns_cutoffs_and_future_invariance() -> None:
    module = _module()
    values = module.core._time_ns(pd.Series(pd.date_range("2024-01-01", periods=3, freq="10min", tz="UTC")))
    assert values[0] > 10**18 and values[1] - values[0] == 600_000_000_000
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cutoffs = [pd.Timestamp(json.loads((ROOT / item["audit"]).read_text(encoding="utf-8"))["adjusted_cutoff_utc"]).value for item in config["parts"].values()]
    assert len(set(cutoffs)) == 3
    original = _frame()
    changed = original.copy()
    changed.loc[170:, "temp"] = 1000.0
    boundary = original.loc[120, "_time"].value
    first = module.visibility_topology_features(original, boundary, _representation())
    second = module.visibility_topology_features(changed, boundary, _representation())
    assert first.shape == (220, 10) and np.isfinite(first).all()
    np.testing.assert_allclose(first[:170], second[:170], atol=0, rtol=0)


def test_group_and_gap_reset() -> None:
    module = _module()
    first, second = _frame("A"), _frame("B")
    second.index = np.arange(len(first), len(first) + len(second))
    boundary = first.loc[120, "_time"].value
    together = module.visibility_topology_features(pd.concat([first, second]), boundary, _representation())
    second.index = np.arange(len(second))
    alone = module.visibility_topology_features(second, boundary, _representation())
    np.testing.assert_allclose(together[len(first) :], alone, atol=0, rtol=0)
    gapped = first.copy()
    gapped.loc[170:, "_time"] += pd.Timedelta(minutes=10)
    reset = module.visibility_topology_features(gapped, boundary, _representation())
    assert np.all(reset[170] == 0)


def test_visibility_is_positive_affine_invariant_and_peak_sensitive() -> None:
    module = _module()
    values = np.sin(np.arange(80) / 7.0)
    original = module.endpoint_visibility(values, 32)
    transformed = module.endpoint_visibility(4.0 * values + 11.0, 32)
    np.testing.assert_allclose(original, transformed, atol=0, rtol=0)
    peak = values.copy()
    peak[-1] = 8.0
    assert module.endpoint_visibility(peak, 32)[-1, 0] > original[-1, 0]


def test_add_only_fixed_gate_and_budget() -> None:
    module = _module()
    incumbent = np.asarray([1, 0, 0, 0, 0], dtype=np.int8)
    additions = module.base._additions(np.asarray([.99, .98, .97, .96, .95]), incumbent, {"threshold": .95}, .4)
    candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
    assert np.all(candidate[incumbent == 1] == 1)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(config["model"]["seeds"]) * len(config["parts"]) == config["model"]["fits"] == 9
    assert config["selection"]["wilson90_lcb_minimum"] == 0.55
    assert config["selection"]["outer_tuning"] == config["operations"]["retry_retune"] == 0


def test_transport_stability_requires_two_halves_and_no_zero_tp_environment() -> None:
    module = _module()
    scores = np.r_[np.full(12, .99), np.full(12, .99)]
    labels = np.r_[np.ones(12, dtype=np.int8), np.ones(12, dtype=np.int8)]
    environments = {
        "station": np.asarray(["A"] * 12 + ["B"] * 12),
        "layer": np.asarray(["1"] * 24),
        "half": np.asarray([0] * 12 + [1] * 12),
    }
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))["selection"]["transport_stability"]
    passed = module._transport_stability(scores, labels, .9, environments, contract)
    assert passed["passed"]
    labels[12:] = 0
    failed = module._transport_stability(scores, labels, .9, environments, contract)
    assert not failed["passed"]
