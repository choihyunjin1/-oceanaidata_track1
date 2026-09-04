from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v34_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()


def test_linear_integrated_profile_has_zero_detrended_fluctuation() -> None:
    values = np.ones(120, dtype=np.float64)
    fluctuation = mod._rolling_linear_detrended_rms(values, 24)
    assert np.max(np.abs(fluctuation[23:])) < 1e-5


def test_injected_fluctuation_has_positive_dfa_energy() -> None:
    values = np.tile([1.0, -1.0], 60)
    fluctuation = mod._rolling_linear_detrended_rms(values, 24)
    assert np.max(fluctuation[23:]) > 0.01


def test_gap_resets_rolling_support() -> None:
    times = mod.base._time_ns(pd.date_range("2024-01-01", periods=120, freq="10min", tz="UTC"))
    times[60:] += mod.CADENCE_NS
    feature = mod.rolling_dfa(np.sin(np.arange(120, dtype=np.float64)), times, 24)
    assert np.all(feature[:23] == 0.0)
    assert np.all(feature[60:83] == 0.0)
    assert feature[83] > 0.0


def test_features_reset_groups_and_are_future_invariant() -> None:
    group_rows = 120
    times = pd.date_range("2024-02-01", periods=group_rows, freq="10min", tz="UTC")
    frame = pd.DataFrame(
        {
            "station": np.repeat(["A", "B"], group_rows),
            "layer": np.repeat([1, 2], group_rows),
            "_time": np.tile(times, 2),
            "temp": np.tile(np.sin(np.arange(group_rows) / 7.0), 2),
        }
    )
    boundary = int(times[59].value)
    representation = {"rolling_rows": [24, 48, 96]}
    original = mod.dfa_features(frame, boundary, representation)
    changed_frame = frame.copy()
    future = mod.base._time_ns(changed_frame["_time"]) > boundary
    changed_frame.loc[future, "temp"] += 10000.0
    changed = mod.dfa_features(changed_frame, boundary, representation)
    assert original.shape == (240, 8)
    assert np.isfinite(original).all()
    assert np.array_equal(original[~future], changed[~future])
    assert np.all(original[:95, 7] == 0.0)
    assert np.all(original[group_rows : group_rows + 95, 7] == 0.0)
    assert original[95, 7] == original[group_rows + 95, 7] == 1.0


def test_preregistered_contract_is_fixed_add_only_and_auditable() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    assert config["representation"]["rolling_rows"] == [24, 48, 96]
    assert len(config["representation"]["features"]) == 8
    assert config["model"]["fits"] == 3 <= config["model"]["maximum_fits"] <= 9
    assert config["selection"]["candidate_threshold_fixed_before_q2"]
    assert config["selection"]["q2_q3_threshold_selection"] == config["selection"]["q2_q3_refits"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"]
    assert config["anchor"]["removals"] == 0
    assert config["auditability_amendment"]["preserve_all_pre_q2_threshold_q2_label_blind_actions"]


def test_real_preflight_is_zero_operation_with_synthetic_guards() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    assert ready["status"] == "PASS_ZERO_OPERATION"
    assert ready["representation_support"]["gate"] == "PASS"
    assert all(ready["synthetic_guards"].values())
    assert ready["auditability"]["q2_target_reads"] == 0
    assert all(value == 0 for value in ready["counters"].values())
