import numpy as np

from scripts import run_p1_causal_cif_lite32_20260831_v32b as runner


def test_exact_32_intervals_and_bounds():
    got = runner.intervals()
    assert len(got) == 32
    assert all(0 <= left < right <= 36 and right - left >= 3 for left, right in got)


def test_feature_contract_width_and_determinism():
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(8, 36, 5)).astype(np.float32)
    a = runner.summarize_windows(windows)
    b = runner.summarize_windows(windows)
    assert a.shape == (8, 1280)
    assert np.array_equal(a, b)


def test_causal_window_uses_only_current_and_past():
    import pandas as pd

    rows = 40
    frame = pd.DataFrame({"station": ["A"] * rows, "year": [2025] * rows, "layer": [1] * rows, "time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"), "temp_raw": np.arange(rows), "temp_diff_1": 1.0, "temp_peer_residual": 0.0, "depth_diff_1": 0.0, "psal_missing": 0, "depth_missing": 0, "has_gap_before": 0})
    before = runner.causal_windows(frame)
    frame.loc[39, "temp_raw"] = 9999
    after = runner.causal_windows(frame)
    assert np.array_equal(before[:39], after[:39])


def test_small_resource_preflight_contract():
    config = __import__("json").loads(runner.CONFIG.read_text())
    result = runner.benchmark(config)
    assert result["checks"]["feature_width"]
    assert result["checks"]["finite"]
