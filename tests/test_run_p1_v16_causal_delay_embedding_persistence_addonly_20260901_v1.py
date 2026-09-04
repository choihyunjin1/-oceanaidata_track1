from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v16_causal_delay_embedding_persistence_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v16_causal_delay_embedding_persistence_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v16_persistence_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _frame(station: str = "I-ORS", rows: int = 1100) -> pd.DataFrame:
    return pd.DataFrame({"station": [station] * rows, "layer": [2] * rows, "_time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"), "temp": np.sin(np.arange(rows) / 8.0) + 0.2 * np.sin(np.arange(rows) / 31.0)})


def _representation():
    return json.loads(CONFIG.read_text(encoding="utf-8"))["representation"]


def test_h0_persistence_matches_known_line_mst_and_is_finite() -> None:
    module = _module()
    points = np.array([[0.0], [1.0], [3.0]])
    np.testing.assert_allclose(module.zero_dimensional_persistence(points), [1.0, 2.0])
    summary = module.persistence_summary(points)
    assert summary.shape == (8,) and np.isfinite(summary).all()
    assert summary[0] == 3.0 and summary[6] == 3.0


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
    first = module.persistence_features(original, boundary, _representation())
    second = module.persistence_features(changed, boundary, _representation())
    assert first.shape == (1100, 8) and np.isfinite(first).all()
    np.testing.assert_allclose(first[:950], second[:950], atol=0, rtol=0)


def test_station_layer_and_gap_reset() -> None:
    module = _module()
    first, second = _frame("A"), _frame("B")
    second.index = np.arange(len(first), len(first) + len(second))
    boundary = first.loc[900, "_time"].value
    together = module.persistence_features(pd.concat([first, second]), boundary, _representation())
    second.index = np.arange(len(second))
    alone = module.persistence_features(second, boundary, _representation())
    np.testing.assert_allclose(together[len(first) :], alone, atol=0, rtol=0)
    gapped = first.copy()
    gapped.loc[950:, "_time"] += pd.Timedelta(minutes=10)
    reset = module.persistence_features(gapped, boundary, _representation())
    assert np.all(reset[950] == 0)


def test_topology_responds_to_delay_cloud_shape_not_positive_affine_scale() -> None:
    module = _module()
    representation = _representation()
    periodic = np.sin(np.arange(600) / 7.0)
    constant = np.zeros(600)
    periodic_features = module._segment_features(periodic, representation)
    constant_features = module._segment_features(constant, representation)
    assert periodic_features[:, 0].max() > constant_features[:, 0].max()
    transformed = module._segment_features(4.0 * periodic + 10.0, representation)
    np.testing.assert_allclose(transformed[:, 0], 4.0 * periodic_features[:, 0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(transformed[:, 5], periodic_features[:, 5], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(transformed[:, 6], 4.0 * periodic_features[:, 6], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(transformed[:, 7], periodic_features[:, 7], rtol=1e-10, atol=1e-10)


def test_boundary_metric_add_only_transport_and_budget_contract() -> None:
    module = _module()
    truth = np.r_[np.zeros(3), np.ones(24), np.zeros(3)].astype(np.int8)
    anchor = np.zeros(30, dtype=np.int8)
    candidate = anchor.copy()
    candidate[3:9] = candidate[21:27] = 1
    metadata = pd.DataFrame({"station": ["A"] * 30, "layer": [1] * 30, "time": pd.date_range("2025-01-01", periods=30, freq="10min", tz="UTC")})
    receipt = module._boundary_recall(truth, anchor, candidate, metadata)
    assert receipt["runs"] == 1 and receipt["candidate_recall"] == 1.0
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(config["model"]["seeds"]) * len(config["parts"]) == config["model"]["fits"] == 9
    assert config["selection"]["transport_stability"]["require_both_chronological_halves"]
    assert config["anchor"]["removals"] == config["selection"]["outer_tuning"] == 0
    assert config["source"]["official_test_sample_submission_hidden_reads"] == 0
