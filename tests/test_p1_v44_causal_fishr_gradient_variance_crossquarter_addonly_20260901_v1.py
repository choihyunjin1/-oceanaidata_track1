from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v44_causal_fishr_gradient_variance_crossquarter_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v44_causal_fishr_gradient_variance_crossquarter_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("test_p1_v44_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(rows: int = 96) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "station": np.repeat(["G-ORS", "I-ORS"], rows),
            "layer": np.repeat([1, 2], rows),
            "_time": np.tile(time, 2),
            "temp": np.tile(np.sin(np.arange(rows) / 8.0), 2),
        }
    )


def test_preregistered_budget_and_transport_contract() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["model"]["fits"] == 3
    assert config["model"]["maximum_fits"] == 9
    assert config["model"]["fishr_coefficient"] == 1.0
    assert config["selection"]["q2_q3_refits"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"] is True
    assert config["anchor"]["removals"] == 0
    assert config["operations"] == {
        "exactly_once": True,
        "fits_maximum": 9,
        "official": 0,
        "csv": 0,
        "uploads": 0,
        "retry_retune": 0,
    }


def test_gradient_variance_penalty_positive_and_zero_for_equal_domains() -> None:
    module = _module()
    logits = torch.tensor([-2.0, -0.5, 0.5, 2.0, -1.5, -0.25, 1.0, 2.5])
    hidden = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10.0
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    environment = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    penalty, count = module.fishr_gradient_variance_penalty(
        logits, hidden, targets, environment, 3.0, 4
    )
    equal, equal_count = module.fishr_gradient_variance_penalty(
        torch.cat([logits[:4], logits[:4]]),
        torch.cat([hidden[:4], hidden[:4]]),
        torch.cat([targets[:4], targets[:4]]),
        environment,
        3.0,
        4,
    )
    assert count == equal_count == 2
    assert torch.isfinite(penalty) and penalty > 0
    assert torch.abs(equal) < 1e-8


def test_environment_encoding_survives_standardization() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = _frame()
    boundary = int(frame["_time"].max().value)
    features = module.causal_environment_features(frame, boundary, config["representation"])
    scaled = StandardScaler().fit_transform(features).astype(np.float32)
    decoded = module.decode_environment_ids(scaled)
    assert features.shape == (192, 23)
    assert len(np.unique(decoded)) == 2
    assert np.all(decoded[:96] == decoded[0])
    assert np.all(decoded[96:] == decoded[-1])


def test_group_reset_and_future_invariance() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = _frame()
    boundary = int(pd.DatetimeIndex([frame["_time"].iloc[47]]).as_unit("ns").asi8[0])
    first = module.causal_environment_features(frame, boundary, config["representation"])
    changed = frame.copy()
    time_ns = pd.DatetimeIndex(changed["_time"]).as_unit("ns").asi8
    future = time_ns > boundary
    assert time_ns.dtype == np.dtype("int64")
    assert int(time_ns[46]) < boundary < int(time_ns[48])
    changed.loc[future, "temp"] += 1000.0
    second = module.causal_environment_features(changed, boundary, config["representation"])
    assert np.array_equal(first[~future], second[~future])
    assert np.array_equal(first[:96, :8], first[96:, :8])
    assert not np.array_equal(first[:96, 8:], first[96:, 8:])


def test_inference_is_environment_blind() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    classifier = module.FishrClassifier(23, config["model"], 17)
    values = np.zeros((4, 23), dtype=np.float32)
    values[:, :8] = np.arange(32, dtype=np.float32).reshape(4, 8) / 10.0
    altered = values.copy()
    altered[:, 8:] = np.arange(60, dtype=np.float32).reshape(4, 15)
    assert np.array_equal(classifier.predict_score(values), classifier.predict_score(altered))


def test_all_synthetic_guards_pass() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert all(module._synthetic_guards(config["representation"]).values())
