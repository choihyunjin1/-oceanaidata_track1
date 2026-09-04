from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_gramian_angular_field_residual_cycle_20260901_v62.py"
SPEC = importlib.util.spec_from_file_location("p3_v62", RUNNER)
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


def test_gadf_direction_and_gasf_reversal_symmetry() -> None:
    rising = np.linspace(-3.0, 5.0, 96)
    up = MODULE.angular_diagonal_statistics(rising)
    down = MODULE.angular_diagonal_statistics(rising[::-1])
    assert up[1] > 0.0 and down[1] < 0.0
    assert np.allclose(up[::2], down[::2], atol=1e-12)


def test_positive_affine_invariance_and_constant_finite() -> None:
    values = np.linspace(-3.0, 5.0, 96)
    assert np.allclose(MODULE.angular_diagonal_statistics(values), MODULE.angular_diagonal_statistics(7 * values + 3), rtol=1e-10, atol=1e-10)
    assert np.isfinite(MODULE.angular_diagonal_statistics(np.ones(96))).all()


def test_feature_shape_finite_deterministic_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    first = MODULE.gramian_features(sequence)
    assert first.shape == (64,) and np.isfinite(first).all()
    assert np.array_equal(first, MODULE.gramian_features(sequence.copy()))
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    assert np.array_equal(first, MODULE.gramian_features(extended[:289]))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["image_points"] == 32
    assert config["encoder"]["lag_diagonals"] == [1, 2, 4, 8]
    assert config["encoder"]["feature_count"] == 64
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
