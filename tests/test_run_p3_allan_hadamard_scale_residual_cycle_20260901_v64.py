from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_allan_hadamard_scale_residual_cycle_20260901_v64.py"
SPEC = importlib.util.spec_from_file_location("p3_v64", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_or_consumed() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] in {"READY_EXACTLY_ONCE", "STOP_SUPPORT_GATE"}
        assert value["prior_outputs_used"] is False
        assert value["official_v42_used_for_features_gates_selection"] is False


def test_white_noise_random_walk_and_linear_drift_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["white_allan_scale8_over_scale1"] < 0.40
    assert receipt["random_walk_allan_scale8_over_scale1"] > 2.0
    assert receipt["linear_hadamard_scale8"] <= max(receipt["linear_allan_scale8"] * 1e-12, 1e-24)


def test_sign_scale_invariance_and_exact_formulas() -> None:
    rng = np.random.default_rng(9)
    values = np.cumsum(rng.normal(size=128))
    first = MODULE.allan_hadamard_statistics(values)
    second = MODULE.allan_hadamard_statistics(-7.0 * values + 3.0)
    assert np.allclose(first, second, rtol=1e-10, atol=1e-10)
    linear = np.arange(128, dtype=np.float64)
    stats = MODULE.allan_hadamard_statistics(linear)
    assert np.all(stats[1::2] <= 1e-24)


def test_feature_shape_finite_deterministic_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    first = MODULE.allan_hadamard_features(sequence)
    assert first.shape == (64,) and np.isfinite(first).all()
    assert np.array_equal(first, MODULE.allan_hadamard_features(sequence.copy()))
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    assert np.array_equal(first, MODULE.allan_hadamard_features(extended[:289]))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["averaging_scales_rows"] == [1, 2, 4, 8]
    assert config["encoder"]["feature_count"] == 64
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
