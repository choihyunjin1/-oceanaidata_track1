from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from p3_wave.timexer_direct_multilead import (
    DirectTimeXerConfig,
    PastExogenousDirectTimeXer,
    fit_hourly_statistics,
    persistence_additive_prediction,
    promotion_gates,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p3_timexer_past_exogenous_direct_multilead_20260828_v1.json"
RUNNER = ROOT / "scripts/run_p3_timexer_past_exogenous_direct_multilead_20260828_v1.py"


def _raw(batch: int = 4) -> np.ndarray:
    rng = np.random.default_rng(20260828)
    raw = rng.normal(size=(batch, 289, 10)).astype(np.float32)
    raw[..., 3] = rng.uniform(0.0, 360.0, size=(batch, 289))
    raw[..., 6] = rng.uniform(0.0, 360.0, size=(batch, 289))
    raw[:, 24, 1] = np.nan
    raw[:, 48, 4] = np.nan
    return raw


def _load_runner():
    spec = importlib.util.spec_from_file_location("_timexer_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asymmetric_tokens_keep_wave_and_weather_separate() -> None:
    raw = _raw()
    center, scale = fit_hourly_statistics(raw)
    model = PastExogenousDirectTimeXer(center, scale)
    original_endogenous, original_exogenous = model.tokenizer(torch.from_numpy(raw))
    changed = raw.copy()
    changed[..., 4] += 7.0
    changed_endogenous, changed_exogenous = model.tokenizer(torch.from_numpy(changed))
    assert original_endogenous.shape == (4, 7, 64)
    assert original_exogenous.shape == (4, 11, 64)
    assert torch.equal(original_endogenous, changed_endogenous)
    assert not torch.equal(original_exogenous, changed_exogenous)


def test_direct_six_lead_forward_backward_is_finite() -> None:
    raw = _raw()
    center, scale = fit_hourly_statistics(raw)
    model = PastExogenousDirectTimeXer(center, scale, DirectTimeXerConfig())
    station = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    prediction = model(torch.from_numpy(raw), station)
    assert prediction.shape == (4, 6)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_persistence_skip_is_direct_for_all_six_leads() -> None:
    current = np.array([1.0, 2.0])
    residual = np.arange(12, dtype=float).reshape(2, 6) / 10.0
    predicted = persistence_additive_prediction(current, residual)
    np.testing.assert_allclose(predicted, current[:, None] + residual)


def test_promotion_gate_is_exact_and_deterministic() -> None:
    passed = promotion_gates(
        pooled_delta_m=-0.006,
        fold_deltas_m={"a": -0.01, "b": -0.002, "c": 0.001},
        station_deltas_m={"G": -0.002, "I": 0.004, "S": -0.001},
        lead_deltas_m={"3": -0.002, "6": -0.001, "9": 0.0, "12": -0.003, "18": 0.0, "24": -0.001},
        bootstrap_ci90_upper_m=-0.0001,
    )
    assert passed["local_promotion_go"] is True
    failed = promotion_gates(
        pooled_delta_m=-0.006,
        fold_deltas_m={"a": -0.01, "b": -0.002, "c": 0.001},
        station_deltas_m={"G": -0.002, "I": 0.004, "S": -0.001},
        lead_deltas_m={"3": -0.002, "6": -0.001, "9": 0.0, "12": -0.003, "18": 0.001, "24": -0.001},
        bootstrap_ci90_upper_m=-0.0001,
    )
    assert failed["local_promotion_go"] is False
    assert failed["long_lead_safety_gate"] is False


def test_runner_accepts_only_the_frozen_authorized_config() -> None:
    runner = _load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    runner._validate_config(config)
    changed = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed["model"]["d_model"] = 128
    try:
        runner._validate_config(changed)
    except ValueError as error:
        assert "model contract changed" in str(error)
    else:
        raise AssertionError("changed model contract was accepted")
