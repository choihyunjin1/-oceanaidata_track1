from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_moist_air_momentum_residual_cycle_20260901_v76.py"
SPEC = importlib.util.spec_from_file_location("p3_v76", RUNNER)
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


def test_moist_air_and_quadratic_momentum_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert 1.15 < receipt["standard_density_kg_m3"] < 1.30
    assert receipt["warm_density_kg_m3"] < receipt["standard_density_kg_m3"]
    assert receipt["humid_density_kg_m3"] < receipt["standard_density_kg_m3"]
    assert receipt["low_pressure_density_kg_m3"] < receipt["standard_density_kg_m3"]
    assert receipt["quadratic_wind_ratio"] == 4.0


def test_feature_shape_determinism_and_future_isolation() -> None:
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack(
        [np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)]
    )
    sequence[:, 4] = 8 + np.sin(axis)
    sequence[:, 5] = 10 + np.sin(axis)
    sequence[:, 7] = 15 + 5 * axis
    sequence[:, 8] = 65 + 10 * axis
    sequence[:, 9] = 1013 - 4 * axis
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = MODULE.momentum_features(sequence)
    assert direct.shape == (48,) and np.isfinite(direct).all()
    assert np.array_equal(direct, MODULE.momentum_features(sequence.copy()))
    assert np.array_equal(
        direct, MODULE.momentum_features(np.vstack([sequence, np.full((12, 10), 1e9)]))
    )


def test_gust_excess_is_nonnegative() -> None:
    sequence = np.zeros((289, 10))
    sequence[:, 4] = 12
    sequence[:, 5] = 10
    sequence[:, 7] = 15
    sequence[:, 8] = 60
    sequence[:, 9] = 1013
    assert np.array_equal(MODULE.physical_paths(sequence)[:, 3], np.zeros(145))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["feature_count"] == 48
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
