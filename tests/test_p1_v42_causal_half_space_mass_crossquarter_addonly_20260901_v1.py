from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v42_causal_half_space_mass_crossquarter_addonly_20260901_v1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v42_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()
config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))


def test_fixed_tree_is_data_independent_and_deterministic() -> None:
    first = mod.fixed_half_space_tree(config["representation"], 0, 8)
    second = mod.fixed_half_space_tree(config["representation"], 0, 8)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(first[0]) == 2 ** config["representation"]["maximum_depth"] - 1


def test_sparse_tail_states_have_higher_prefix_mass_rarity() -> None:
    rng = np.random.default_rng(17)
    reference = rng.normal(0.0, 0.25, size=(256, 8))
    reference[:, -1] = 0.0
    outliers = np.full((32, 8), 9.0)
    outliers[:, -1] = 0.0
    values = np.vstack([reference, outliers])
    result = mod.half_space_rarity(
        values,
        np.arange(len(values)) < len(reference),
        config["representation"],
    )
    assert result.shape == (288, 3)
    assert np.isfinite(result).all()
    assert result[256:, 0].mean() > result[:256, 0].mean() + 0.1


def test_station_layer_and_cadence_gap_reset() -> None:
    rows = 210
    times = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    second_times = times.where(
        np.arange(rows) < 110,
        times + pd.Timedelta(minutes=10),
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
    states = mod.causal_state_features(frame, int(times[104].value), config["representation"])
    assert states[0, 4] == states[rows, 4] == 0.0
    assert states[0, -1] == states[rows, -1] == 1.0
    assert states[rows + 110, 4] == 0.0
    assert states[rows + 110, -1] == 1.0


def test_ns_cutoff_and_future_invariance() -> None:
    rows = 180
    times = pd.date_range("2024-03-01", periods=rows, freq="10min", tz="UTC")
    frame = pd.DataFrame(
        {"station": "S-A", "layer": "L1", "_time": times, "temp": np.sin(np.arange(rows) / 9.0)}
    )
    boundary = int(times[119].value)
    first = mod.causal_half_space_features(frame, boundary, config["representation"])
    changed = frame.copy()
    future = mod.base._time_ns(changed["_time"]) > boundary
    changed.loc[future, "temp"] += 1000.0
    second = mod.causal_half_space_features(changed, boundary, config["representation"])
    assert mod.base._time_ns(times).dtype == np.dtype("int64")
    assert int(times[118].value) < boundary < int(times[120].value)
    assert np.array_equal(first[~future], second[~future])


def test_add_only_crossquarter_contract() -> None:
    assert config["model"]["maximum_fits"] == 9
    assert config["model"]["fits"] == 3
    assert config["selection"]["q2_q3_refits"] == config["selection"]["q2_q3_threshold_selection"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"]
    assert config["anchor"]["removals"] == 0
    assert config["operations"]["official"] == config["operations"]["csv"] == config["operations"]["uploads"] == 0


def test_real_preflight_is_target_free_and_wrapper_identified() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    assert ready["runner_sha256"] == mod.base._sha(RUNNER)
    assert all(ready["synthetic_guards"].values())
    support = ready["representation_support"]
    rarity_variances = support.get("rarity_variances", support["feature_variances"][-3:])
    assert max(rarity_variances) > config["representation_support_gate"]["minimum_rarity_variance"]
    assert all(value == 0 for value in ready["counters"].values())
