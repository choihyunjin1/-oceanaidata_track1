from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_lz_rank_complexity_residual_cycle_20260901_v57.py"
SPEC = importlib.util.spec_from_file_location("p3_v57", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_or_consumed() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "READY_EXACTLY_ONCE"
        assert value["maximum_model_fits"] == 12
        assert value["official_access"] == 0


def test_random_more_complex_than_periodic() -> None:
    periodic = np.tile(np.asarray([0, 1, 2, 1], dtype=np.uint8), 36)
    random = np.random.default_rng(20260901).integers(0, 3, size=len(periodic), dtype=np.uint8)
    assert MODULE.normalized_complexity(random) > MODULE.normalized_complexity(periodic)


def test_rank_symbolization_affine_invariant() -> None:
    path = np.sin(np.linspace(0.0, 9.0, 145)) + np.linspace(0.0, 1.0, 145)
    assert np.array_equal(MODULE.symbolize(path), MODULE.symbolize(7.0 * path + 3.0))


def test_feature_shape_finite_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = MODULE.lz_features(sequence)
    assert feature.shape == (32,)
    assert np.isfinite(feature).all()
    extended = np.vstack([sequence, np.full((10, 10), 1e9)])
    assert np.array_equal(feature, MODULE.lz_features(extended[:289]))


def test_sealed_model_and_validation_contract() -> None:
    config = MODULE.load_config()
    assert [item["ridge_alpha"] for item in config["model"]["candidates"]] == [512.0, 2048.0]
    assert config["validation"]["maximum_total_fits"] == 12
    assert config["model"]["row_deletion"] == 0
    assert all(value == 0 for value in config["official_policy"].values())
