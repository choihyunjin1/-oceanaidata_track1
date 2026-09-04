from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v30_causal_backward_teager_energy_crossquarter_addonly_20260901_v1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v30_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()


def test_backward_teager_formula_is_causal_and_gap_aware() -> None:
    cadence = mod.CADENCE_NS
    values = np.array([1.0, 2.0, 3.0, 4.0])
    times = np.array([0, cadence, 2 * cadence, 4 * cadence], dtype=np.int64)
    energy, support = mod.backward_teager(values, times)
    assert energy[2] == 1.0
    assert support.tolist() == [False, False, True, False]


def test_teager_features_reset_groups_and_ignore_future_for_prefix() -> None:
    times = list(pd.date_range("2024-01-01", periods=10, freq="10min", tz="UTC")) * 2
    frame = pd.DataFrame(
        {
            "station": ["A"] * 10 + ["B"] * 10,
            "layer": [1] * 20,
            "time": [value.isoformat() for value in times],
            "_time": times,
            "temp": np.r_[np.arange(10.0), 100.0 + np.arange(10.0)],
        }
    )
    boundary = pd.Timestamp("2024-01-01T01:00:00Z").value
    representation = {"rolling_rows": [6, 24, 96]}
    first = mod.teager_features(frame, boundary, representation)
    changed_frame = frame.copy()
    after = np.array([pd.Timestamp(value).value > boundary for value in frame["_time"]])
    changed_frame.loc[after, "temp"] += 10000.0
    second = mod.teager_features(changed_frame, boundary, representation)
    assert np.array_equal(first[~after], second[~after])
    assert first[0, 9] == first[1, 9] == first[10, 9] == first[11, 9] == 0.0


def test_fixed_linear_probe_scores_are_finite() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))["model"]
    rng = np.random.default_rng(7)
    features = rng.normal(size=(200, 10)).astype(np.float32)
    labels = np.r_[np.zeros(150, dtype=np.int8), np.ones(50, dtype=np.int8)]
    model = mod.LinearProbeClassifier(10, config, 20260901).fit(features, labels)
    scores = model.predict_score(features)
    assert scores.shape == (200,)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_crossquarter_contract_is_unchanged_and_vib_is_not_reused() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    assert config["cross_quarter_guard"]["sha256"] == "a051d4d0837b395f9c1d42d71d65572efb99d6c1aeae6aabdb52c8427539c8d6"
    assert config["model"]["kind"] == "fixed_linear_logistic_probe"
    assert config["model"]["fits"] == 3 <= config["model"]["maximum_fits"] <= 9
    assert config["selection"]["candidate_threshold_fixed_before_q2"]
    assert config["selection"]["q2_q3_threshold_selection"] == config["selection"]["q2_q3_refits"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"]
    assert config["anchor"]["removals"] == 0


def test_real_preflight_is_zero_operation_and_lifecycle_aware() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    assert ready["status"] == "PASS_ZERO_OPERATION"
    assert ready["representation_support"]["gate"] == "PASS"
    assert all(value == 0 for value in ready["counters"].values())
