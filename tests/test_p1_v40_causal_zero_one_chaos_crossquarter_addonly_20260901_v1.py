from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v40_causal_zero_one_chaos_crossquarter_addonly_20260901_v1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v40_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()
config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))


def test_chaotic_translation_growth_exceeds_periodic() -> None:
    rows = 320
    periodic = np.sin(2.0 * np.pi * np.arange(rows) / 17.0)
    chaotic = np.empty(rows)
    chaotic[0] = 0.211
    for index in range(1, rows):
        chaotic[index] = 4.0 * chaotic[index - 1] * (1.0 - chaotic[index - 1])
    periodic_result = mod.zero_one_coordinates(periodic, config["representation"])
    chaotic_result = mod.zero_one_coordinates(chaotic, config["representation"])
    assert np.median(chaotic_result[160:, [0, 2, 4]]) > np.median(periodic_result[160:, [0, 2, 4]]) + 0.3
    assert np.isfinite(periodic_result).all() and np.isfinite(chaotic_result).all()


def test_short_segment_has_no_supported_window() -> None:
    result = mod.zero_one_coordinates(np.arange(95, dtype=np.float64), config["representation"])
    assert result.shape == (95, 7)
    assert np.all(result == 0.0)


def test_group_and_cadence_gap_reset() -> None:
    rows = 210
    times = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    second_times = times.copy()
    second_times = second_times.where(
        np.arange(rows) < 110,
        second_times + pd.Timedelta(minutes=10),
    )
    frame = pd.DataFrame(
        {
            "station": np.repeat(["S-A", "S-B"], rows),
            "layer": np.repeat(["L1", "L2"], rows),
            "_time": np.concatenate([times.to_numpy(), second_times.to_numpy()]),
            "temp": np.tile(np.sin(np.arange(rows) / 7.0), 2),
        }
    )
    frame["_time"] = pd.to_datetime(frame["_time"], utc=True)
    features = mod.causal_zero_one_features(frame, int(times[104].value), config["representation"])
    assert np.all(features[:95, -1] == 0.0)
    assert features[95, -1] == 1.0
    assert np.all(features[rows : rows + 95, -1] == 0.0)
    assert features[rows + 95, -1] == 1.0
    assert np.all(features[rows + 110 : rows + 205, -1] == 0.0)
    assert features[rows + 205, -1] == 1.0


def test_ns_cutoff_future_invariance() -> None:
    rows = 180
    times = pd.date_range("2024-03-01", periods=rows, freq="10min", tz="UTC")
    frame = pd.DataFrame({"station": "S-A", "layer": "L1", "_time": times, "temp": np.sin(np.arange(rows) / 9.0)})
    boundary = int(times[119].value)
    first = mod.causal_zero_one_features(frame, boundary, config["representation"])
    changed = frame.copy()
    future = mod.base._time_ns(changed["_time"]) > boundary
    changed.loc[future, "temp"] += 1000.0
    second = mod.causal_zero_one_features(changed, boundary, config["representation"])
    assert mod.base._time_ns(times).dtype == np.dtype("int64")
    assert int(times[118].value) < boundary < int(times[120].value)
    assert np.array_equal(first[~future], second[~future])


def test_add_only_crossquarter_contract() -> None:
    assert config["model"]["maximum_fits"] == 9
    assert config["selection"]["q2_q3_refits"] == config["selection"]["q2_q3_threshold_selection"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"]
    assert config["anchor"]["removals"] == 0
    assert config["operations"]["official"] == config["operations"]["csv"] == config["operations"]["uploads"] == 0


def test_real_preflight_is_zero_operation_and_wrapper_identified() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    assert ready["runner_sha256"] == mod.base._sha(RUNNER)
    assert all(ready["synthetic_guards"].values())
    assert all(value == 0 for value in ready["counters"].values())
