from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_volatility_clustering_residual_cycle_20260901_v63.py"
SPEC = importlib.util.spec_from_file_location("p3_v63", RUNNER)
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


def test_clustered_volatility_exceeds_permuted() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["clustered_abs_acf_lag1"] > receipt["permuted_abs_acf_lag1"] + 0.20


def test_sign_scale_invariance_and_zero_variance_fallback() -> None:
    rng = np.random.default_rng(9)
    values = np.cumsum(rng.normal(size=96))
    assert np.allclose(MODULE.volatility_statistics(values), MODULE.volatility_statistics(-7 * values + 3), rtol=1e-10, atol=1e-10)
    assert MODULE.correlation(np.ones(80), 3) == 0.0


def test_feature_shape_finite_deterministic_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    first = MODULE.volatility_features(sequence)
    assert first.shape == (64,) and np.isfinite(first).all()
    assert np.array_equal(first, MODULE.volatility_features(sequence.copy()))
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    assert np.array_equal(first, MODULE.volatility_features(extended[:289]))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["autocorrelation_lags_rows"] == [1, 3, 6, 12]
    assert config["encoder"]["feature_count"] == 64
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
