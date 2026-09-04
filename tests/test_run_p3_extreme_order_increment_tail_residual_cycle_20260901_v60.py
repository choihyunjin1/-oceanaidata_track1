from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_extreme_order_increment_tail_residual_cycle_20260901_v60.py"
SPEC = importlib.util.spec_from_file_location("p3_v60", RUNNER)
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
        assert value["official_access"] == 0
        assert value["official_v42_used_for_features_gates_selection"] is False


def test_heavy_tail_has_higher_extreme_mass() -> None:
    rng = np.random.default_rng(20260901)
    gaussian = np.abs(rng.normal(size=512))
    heavy = rng.pareto(1.6, size=512) + 1.0
    assert MODULE.extreme_order_statistics(heavy)[2] > MODULE.extreme_order_statistics(gaussian)[2] + 0.05


def test_affine_scale_invariance_and_support() -> None:
    rng = np.random.default_rng(7)
    values = np.abs(rng.normal(size=128))
    assert np.allclose(MODULE.extreme_order_statistics(values), MODULE.extreme_order_statistics(7 * values), rtol=1e-10, atol=1e-10)
    with pytest.raises(MODULE.ContractError, match="4k"):
        MODULE.extreme_order_statistics(values[:15])


def test_feature_shape_finite_deterministic_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    first = MODULE.extreme_order_features(sequence)
    second = MODULE.extreme_order_features(sequence.copy())
    assert first.shape == (72,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    assert np.array_equal(first, MODULE.extreme_order_features(extended[:289]))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["increment_lags_rows"] == [1, 3, 6]
    assert config["encoder"]["extreme_order_k"] == 4
    assert config["encoder"]["feature_count"] == 72
    assert config["validation"]["maximum_total_fits"] == 12
    assert "excluded" in config["duplication_audit"]["official_v42_exclusion"]
    assert all(value == 0 for value in config["official_policy"].values())
