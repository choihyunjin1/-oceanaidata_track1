from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_causal_multichannel_rocket_residual_cycle_20260901_v26.py"
SPEC = importlib.util.spec_from_file_location("p3_v26", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_sequence() -> np.ndarray:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[2::13, (0, 3, 6)] = np.nan
    return sequence


def test_config_is_sealed_and_access_zero() -> None:
    config = MODULE.load_config()
    assert config["status"] == "SEALED_BEFORE_OUTER_SCORING"
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_kernel_bank_is_fixed_and_normalized() -> None:
    first = MODULE.kernel_bank()
    second = MODULE.kernel_bank()
    assert len(first) == len(second) == 256
    assert MODULE.kernel_receipt(first) == MODULE.kernel_receipt(second)
    for left, right in zip(first, second, strict=True):
        assert left.channels == right.channels
        assert left.dilation == right.dilation
        assert np.array_equal(left.weights, right.weights)
        assert abs(float(left.weights.mean())) < 1e-12
        assert np.isclose(np.linalg.norm(left.weights), 1.0)


def test_rocket_features_are_fixed_finite_ppv_max_pairs() -> None:
    feature = MODULE.rocket_features(synthetic_sequence(), MODULE.kernel_bank())
    assert feature.shape == (512,)
    assert np.isfinite(feature).all()
    assert np.all((feature[0::2] >= 0.0) & (feature[0::2] <= 1.0))
    assert np.array_equal(feature, MODULE.rocket_features(synthetic_sequence(), MODULE.kernel_bank()))


def test_representation_is_explicitly_nonduplicate() -> None:
    audit = MODULE.load_config()["duplication_audit"]
    assert audit["repository_rocket_hits"] == 0
    assert audit["semantic_verdict"] == "NON_DUPLICATE_REPRESENTATION_AXIS"


def test_preflight_or_consumed_namespace_is_deterministic() -> None:
    if MODULE.ARTIFACT_DIR.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="already consumed"):
            MODULE.preflight_payload()
    else:
        first = MODULE.preflight_payload()
        second = MODULE.preflight_payload()
        assert first == second
        assert first["maximum_model_fits"] == 12
        assert first["official_access"] == first["csv_materializations"] == first["uploads"] == 0
