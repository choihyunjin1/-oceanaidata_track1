from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v20_causal_fixed_echo_state_reservoir_addonly_20260901_v1.py"
CONFIG = (
    ROOT / "configs/experiments/p1_v20_causal_fixed_echo_state_reservoir_addonly_20260901_v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("p1_v20_test", RUNNER)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _rep():
    return json.loads(CONFIG.read_text(encoding="utf-8"))["representation"]


def _frame(station="A", rows=1100):
    return pd.DataFrame(
        {
            "station": [station] * rows,
            "layer": [2] * rows,
            "_time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"),
            "temp": np.sin(np.arange(rows) / 11.0),
        }
    )


def test_fixed_reservoir_is_deterministic_bounded_and_nontrivial():
    m = _module()
    values = np.sin(np.arange(300) / 9)
    first = m.reservoir_segment(values, _rep())
    second = m.reservoir_segment(values, _rep())
    np.testing.assert_allclose(first, second, atol=0, rtol=0)
    assert first.shape == (300, 16) and np.isfinite(first).all() and np.max(np.abs(first)) <= 1


def test_ns_future_invariance():
    m = _module()
    times = m.core._time_ns(
        pd.Series(pd.date_range("2024-01-01", periods=3, freq="10min", tz="UTC"))
    )
    assert times[0] > 10**18 and times[1] - times[0] == 600_000_000_000
    original = _frame()
    changed = original.copy()
    changed.loc[950:, "temp"] = 1000
    boundary = original.loc[900, "_time"].value
    a = m.reservoir_features(original, boundary, _rep())
    b = m.reservoir_features(changed, boundary, _rep())
    np.testing.assert_allclose(a[:950], b[:950], atol=0, rtol=0)


def test_group_and_gap_reset():
    m = _module()
    a, b = _frame("A"), _frame("B")
    b.index = np.arange(len(a), len(a) + len(b))
    boundary = a.loc[900, "_time"].value
    together = m.reservoir_features(pd.concat([a, b]), boundary, _rep())
    b.index = np.arange(len(b))
    alone = m.reservoir_features(b, boundary, _rep())
    np.testing.assert_allclose(together[len(a) :], alone, atol=0, rtol=0)
    g = a.copy()
    g.loc[950:, "_time"] += pd.Timedelta(minutes=10)
    reset = m.reservoir_features(g, boundary, _rep())
    prefix = g.loc[:900, "temp"].to_numpy()
    center = float(np.median(prefix))
    scale = float(1.4826 * np.median(np.abs(prefix - center)))
    normalized = np.clip((g.loc[950, "temp"] - center) / scale, -12.0, 12.0)
    expected = m.reservoir_segment(np.array([normalized]), _rep())[0]
    np.testing.assert_allclose(reset[950], expected)


def test_contract():
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(c["model"]["seeds"]) * len(c["parts"]) == c["model"]["fits"] == 9
    assert c["representation"]["trained_encoder_parameters"] == 0
    assert c["anchor"]["removals"] == c["selection"]["outer_tuning"] == 0
    assert c["source"]["official_test_sample_submission_hidden_reads"] == 0
