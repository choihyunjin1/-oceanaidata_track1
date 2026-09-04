from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v46_causal_spectral_decoupling_crossquarter_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v46_causal_spectral_decoupling_crossquarter_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("test_p1_v46_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(rows: int = 96) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    return pd.DataFrame({"station": np.repeat(["G-ORS", "I-ORS"], rows), "layer": np.repeat([1, 2], rows), "_time": np.tile(time, 2), "temp": np.tile(np.sin(np.arange(rows) / 8.0), 2)})


def test_preregistered_objective_budget_and_transport_contract() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["model"]["spectral_decoupling_coefficient"] == 0.01
    assert config["model"]["weight_decay"] == 0.0
    assert config["model"]["fits"] == 3
    assert config["model"]["maximum_fits"] == 9
    assert config["model"]["sweep"] == 0
    assert config["selection"]["threshold_quantiles"] == [0.995, 0.9975, 0.999]
    assert config["selection"]["q2_q3_refits"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"] is True
    assert config["anchor"]["removals"] == 0


def test_spectral_decoupling_formula_and_gradient() -> None:
    module = _module()
    logits = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    penalty = module.spectral_decoupling_penalty(logits, 0.01)
    expected = 0.5 * 0.01 * torch.square(logits).mean()
    assert torch.equal(penalty, expected)
    penalty.backward()
    assert torch.allclose(logits.grad, 0.01 * logits.detach() / len(logits), atol=1e-9)


def test_penalty_is_target_independent_sign_symmetric_and_zero_at_origin() -> None:
    module = _module()
    logits = torch.tensor([-3.0, -1.0, 0.0, 2.0])
    assert torch.equal(module.spectral_decoupling_penalty(logits, 0.01), module.spectral_decoupling_penalty(-logits, 0.01))
    assert module.spectral_decoupling_penalty(torch.zeros(4), 0.01) == 0.0
    assert module.spectral_decoupling_penalty(logits, 0.01) > 0.0


def test_ns_group_reset_and_future_invariance() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = _frame()
    time_ns = pd.DatetimeIndex(frame["_time"]).as_unit("ns").asi8
    boundary = int(time_ns[47])
    first = module.CAUSAL_FEATURES(frame, boundary, config["representation"])
    changed = frame.copy()
    future = time_ns > boundary
    changed.loc[future, "temp"] += 1000.0
    second = module.CAUSAL_FEATURES(changed, boundary, config["representation"])
    assert int(time_ns[46]) < boundary < int(time_ns[48])
    assert np.array_equal(first[~future], second[~future])
    assert np.array_equal(first[:96], first[96:])


def test_deterministic_finite_inference_shape() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    classifier = module.SpectralDecouplingClassifier(8, config["model"], 17)
    values = np.arange(64, dtype=np.float32).reshape(8, 8) / 10.0
    first = classifier.predict_score(values)
    second = classifier.predict_score(values)
    assert first.shape == (8,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_all_synthetic_guards_pass() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert all(module._synthetic_guards(config["representation"]).values())
