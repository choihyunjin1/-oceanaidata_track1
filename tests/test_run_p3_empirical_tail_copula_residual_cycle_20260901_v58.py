from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_empirical_tail_copula_residual_cycle_20260901_v58.py"
SPEC = importlib.util.spec_from_file_location("p3_v58", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_or_consumed() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "STOP_SUPPORT_GATE"
        assert value["maximum_model_fits"] == 0
        assert not value["historical_support"]["passed"]
        assert value["official_access"] == 0


def test_comonotonic_tail_exceeds_independent() -> None:
    rng = np.random.default_rng(20260901)
    left = rng.normal(size=5000)
    same = MODULE.pair_tail_statistics(left, left)
    independent = MODULE.pair_tail_statistics(left, rng.normal(size=len(left)))
    assert same[0] > independent[0] + 0.5
    assert same[1] > independent[1] + 0.5


def test_marginal_monotone_invariance() -> None:
    path = np.linspace(-2.0, 2.0, 145) + 0.2 * np.sin(np.arange(145))
    assert np.array_equal(MODULE.empirical_mid_ranks(path), MODULE.empirical_mid_ranks(np.exp(path)))


def test_feature_shape_finite_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = MODULE.tail_copula_features(sequence)
    assert feature.shape == (60,)
    assert np.isfinite(feature).all()
    extended = np.vstack([sequence, np.full((10, 10), 1e9)])
    assert np.array_equal(feature, MODULE.tail_copula_features(extended[:289]))


def test_sealed_model_and_validation_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["tail_quantiles"] == {"lower": 0.10, "upper": 0.90}
    assert [item["ridge_alpha"] for item in config["model"]["candidates"]] == [512.0, 2048.0]
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
