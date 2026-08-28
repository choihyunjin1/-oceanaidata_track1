from __future__ import annotations

import numpy as np
import torch

from p3_wave.tsmixer_residual import (
    ObservedResidualTSMixer,
    decision_gates,
    fit_hourly_statistics,
    hourly_derived_numpy,
    incumbent_preserving_blend,
)


def _raw(batch: int = 3) -> np.ndarray:
    rng = np.random.default_rng(20260828)
    raw = rng.normal(size=(batch, 289, 10)).astype(np.float32)
    raw[..., 3] = rng.uniform(0, 360, size=(batch, 289))
    raw[..., 6] = rng.uniform(0, 360, size=(batch, 289))
    raw[:, 12, 4] = np.nan
    return raw


def test_hourly_observed_contract() -> None:
    raw = _raw()
    derived = hourly_derived_numpy(raw)
    assert derived.shape == (3, 49, 12)
    center, scale = fit_hourly_statistics(raw)
    assert center.shape == scale.shape == (12,)
    assert np.isfinite(center).all()
    assert np.isfinite(scale).all()
    assert (scale > 0).all()


def test_tsmixer_forward_backward_is_finite() -> None:
    raw = _raw()
    center, scale = fit_hourly_statistics(raw)
    model = ObservedResidualTSMixer(center, scale)
    input_tensor = torch.from_numpy(raw)
    station = torch.tensor([0, 1, 2], dtype=torch.long)
    output = model(input_tensor, station)
    assert output.shape == (3, 6)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_incumbent_blend_protects_early_leads_bit_exact() -> None:
    incumbent = np.arange(18, dtype=np.float64).reshape(3, 6) / 10.0
    model = incumbent + 1.0
    candidate = incumbent_preserving_blend(incumbent, model)
    assert np.array_equal(candidate[:, :3], incumbent[:, :3])
    np.testing.assert_allclose(candidate[:, 3:], incumbent[:, 3:] + 0.2)


def test_information_gate_and_stop_are_deterministic() -> None:
    gate = decision_gates(
        pooled_delta_m=-0.005,
        fold_deltas_m={"a": -0.01, "b": -0.002, "c": 0.001},
        station_deltas_m={"G": -0.002, "I": 0.005, "S": -0.001},
        lead_deltas_m={"3": 0.0, "6": 0.0, "9": 0.0, "12": -0.005, "18": 0.0, "24": 0.005},
        bootstrap_ci90_upper_m=0.001,
        probability_improved=0.85,
        novelty_rms_m=0.04,
        seed_rmse_spread_m=0.01,
        runtime_seconds=1000.0,
        maximum_seed_seconds=200.0,
    )
    assert gate["performance_go"] is False
    assert gate["official_info_go"] is True
    assert gate["stop"] is False

    stopped = decision_gates(
        pooled_delta_m=0.001,
        fold_deltas_m={"a": 0.0, "b": 0.001, "c": -0.001},
        station_deltas_m={"G": 0.0, "I": 0.0, "S": 0.0},
        lead_deltas_m={"3": 0.0, "6": 0.0, "9": 0.0, "12": 0.0, "18": 0.02, "24": 0.0},
        bootstrap_ci90_upper_m=0.01,
        probability_improved=0.4,
        novelty_rms_m=0.01,
        seed_rmse_spread_m=0.03,
        runtime_seconds=1000.0,
        maximum_seed_seconds=200.0,
    )
    assert stopped["official_info_go"] is False
    assert stopped["stop"] is True
    assert "pooled_not_better_than_incumbent" in stopped["stop_reasons"]
