from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_lmoment_shape_residual_cycle_20260901_v65.py"
SPEC = importlib.util.spec_from_file_location("p3_v65", RUNNER)
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
        assert value["withdrawn_mmd_used"] is False
        assert value["prior_outputs_used"] is False
        assert value["official_v42_used_for_features_gates_selection"] is False


def test_gaussian_exponential_and_contamination_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert abs(receipt["gaussian_l_skewness"]) < 0.03
    assert receipt["exponential_l_skewness"] > 0.25
    assert receipt["contamination_l_shape_shift"] < receipt["contamination_conventional_shape_shift"] * 0.05


def test_unbiased_pwm_affine_contract() -> None:
    base = np.asarray([-3.0, -1.0, -0.4, 0.2, 1.1, 2.7, 5.0])
    direct = MODULE.sample_lmoments(base)
    positive = MODULE.sample_lmoments(4.0 * base + 9.0)
    negative = MODULE.sample_lmoments(-4.0 * base + 9.0)
    assert np.allclose(positive, [4 * direct[0] + 9, 4 * direct[1], direct[2], direct[3]])
    assert np.allclose(negative, [-4 * direct[0] + 9, 4 * direct[1], -direct[2], direct[3]])


def test_feature_shape_finite_deterministic_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    first = MODULE.lmoment_features(sequence)
    assert first.shape == (64,) and np.isfinite(first).all()
    assert np.array_equal(first, MODULE.lmoment_features(sequence.copy()))
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    assert np.array_equal(first, MODULE.lmoment_features(extended[:289]))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["feature_count"] == 64
    assert config["validation"]["maximum_total_fits"] == 12
    assert "withdrawn" in config["duplication_audit"]["withdrawn_axis"]
    assert all(value == 0 for value in config["official_policy"].values())
