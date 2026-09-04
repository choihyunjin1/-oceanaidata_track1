from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_natural_time_increment_energy_residual_cycle_20260901_v77.py"
SPEC = importlib.util.spec_from_file_location("p3_v77", RUNNER)
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
        assert value["official_used_for_features_gates_selection"] is False


def test_time_reversal_entropy_contract() -> None:
    path = np.zeros(73)
    path[12:] += 1.0
    path[50:] += 2.0
    forward = MODULE.natural_time_statistics(path)
    reverse = MODULE.natural_time_statistics(path[::-1])
    assert np.isclose(forward[0], reverse[0], atol=1e-12, rtol=0.0)
    assert np.isclose(forward[1], reverse[2], atol=1e-12, rtol=0.0)
    assert np.isclose(forward[3], -reverse[3], atol=1e-12, rtol=0.0)


def test_positive_affine_and_constant_guards() -> None:
    path = np.sin(np.linspace(0.0, 5.0, 73))
    assert np.allclose(
        MODULE.natural_time_statistics(path),
        MODULE.natural_time_statistics(4.0 + 3.0 * path),
        atol=1e-12,
        rtol=0.0,
    )
    constant = MODULE.natural_time_statistics(np.ones(73))
    assert np.isfinite(constant).all()
    assert 0.0 <= constant[0] <= 1.0 / 12.0


def test_feature_shape_determinism_and_future_isolation() -> None:
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack(
        [np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)]
    )
    sequence[:, 5] = 6.0 + np.sin(3.0 * axis)
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = MODULE.natural_time_features(sequence)
    assert direct.shape == (32,) and np.isfinite(direct).all()
    assert np.array_equal(direct, MODULE.natural_time_features(sequence.copy()))
    assert np.array_equal(
        direct, MODULE.natural_time_features(np.vstack([sequence, np.full((12, 10), 1e9)]))
    )


def test_pulse_order_and_sealed_contract() -> None:
    assert MODULE.synthetic_receipt()["pulse_order_distinct"] is True
    config = MODULE.load_config()
    assert config["encoder"]["feature_count"] == 32
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
