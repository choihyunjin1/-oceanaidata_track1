from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_pairwise_predictive_causality_residual_cycle_20260901_v73.py"
SPEC = importlib.util.spec_from_file_location("p3_v73", RUNNER)
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


def test_directed_predictive_coupling_guard() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["forward_variance_ratio"] > 0.50
    assert receipt["directed_margin"] > 0.50


def test_affine_sign_constant_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["positive_affine_invariant"] is True
    assert receipt["sign_invariant"] is True
    assert receipt["constant_target_zero"] is True


def test_feature_shape_determinism_and_future_isolation() -> None:
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = MODULE.predictive_causality_features(sequence)
    assert direct.shape == (24,) and np.isfinite(direct).all()
    assert np.array_equal(direct, MODULE.predictive_causality_features(sequence.copy()))
    assert np.array_equal(direct, MODULE.predictive_causality_features(np.vstack([sequence, np.full((12, 10), 1e9)])))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["feature_count"] == 24
    assert config["encoder"]["lag_order"] == 2
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
