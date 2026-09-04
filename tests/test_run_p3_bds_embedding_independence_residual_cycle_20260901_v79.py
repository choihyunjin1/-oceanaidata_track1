from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_bds_embedding_independence_residual_cycle_20260901_v79.py"
SPEC = importlib.util.spec_from_file_location("p3_v79", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_or_consumed() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] in {"READY_EXACTLY_ONCE", "STOP_SUPPORT_GATE_ZERO_FIT"}
        assert value["strict_reconstructibility_verdict"].startswith("PROCEED")
        assert value["prior_outputs_used"] is False
        assert value["official_used_for_features_gates_selection"] is False


def test_serial_dependence_and_shuffle_guard() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["serial_dependence_recovered"] is True
    assert receipt["dependent_contrast"][0] > receipt["shuffled_contrast"][0] + 0.01
    assert receipt["dependent_contrast"][1] > receipt["shuffled_contrast"][1] + 0.25


def test_factorization_affine_and_constant_guards() -> None:
    rng = np.random.default_rng(1179)
    innovations = rng.normal(size=145)
    path = np.zeros(145)
    for index in range(1, len(path)):
        path[index] = 0.85 * path[index - 1] + innovations[index]
    direct = MODULE.embedding_independence_statistics(path)
    affine = MODULE.embedding_independence_statistics(7.0 + 3.0 * path)
    assert np.allclose(direct, affine, atol=1e-12, rtol=0.0)
    assert np.array_equal(
        MODULE.embedding_independence_statistics(np.ones(145)), np.zeros(2)
    )


def test_feature_shape_determinism_and_future_isolation() -> None:
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack(
        [np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)]
    )
    sequence[:, 5] = 6.0 + np.sin(3.0 * axis)
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = MODULE.bds_features(sequence)
    assert direct.shape == (16,) and np.isfinite(direct).all()
    assert np.array_equal(direct, MODULE.bds_features(sequence.copy()))
    assert np.array_equal(
        direct, MODULE.bds_features(np.vstack([sequence, np.full((12, 10), 1e9)]))
    )


def test_sealed_contract_and_v29_boundary() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["embedding_dimension"] == 3
    assert config["encoder"]["epsilon_mad_units"] == 0.5
    assert config["encoder"]["feature_count"] == 16
    assert config["validation"]["maximum_total_fits"] == 12
    assert "not reconstructible" in config["duplication_audit"]["v29_adjacency"]
    assert all(value == 0 for value in config["official_policy"].values())
