from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_equilibrium_error_residual_cycle_20260901_v71.py"
SPEC = importlib.util.spec_from_file_location("p3_v71", RUNNER)
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


def test_cointegrated_independent_and_direction_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["cointegrated_residual_rms"] < 0.5 * receipt["independent_residual_rms"]
    assert receipt["cointegrated_lag1"] < receipt["independent_lag1"] - 0.30
    assert receipt["cointegrated_error_correction"] < receipt["independent_error_correction"] - 0.30


def test_affine_pair_order_and_constant_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["positive_affine_invariant"] is True
    assert receipt["ordered_pair"] is True
    assert receipt["constant_zero"] is True


def test_feature_shape_determinism_and_future_isolation() -> None:
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * axis) + 0.1 * index * axis for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = MODULE.equilibrium_features(sequence)
    assert direct.shape == (32,) and np.isfinite(direct).all()
    assert np.array_equal(direct, MODULE.equilibrium_features(sequence.copy()))
    assert np.array_equal(direct, MODULE.equilibrium_features(np.vstack([sequence, np.full((12, 10), 1e9)])))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["feature_count"] == 32
    assert config["encoder"]["ordered_pairs"] == ["hs~hmax", "hs~tp", "hs~wspd", "tp~wspd"]
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
