from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_intrinsic_dimension_residual_cycle_20260901_v59.py"
SPEC = importlib.util.spec_from_file_location("p3_v59", RUNNER)
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


def test_line_dimension_below_volume() -> None:
    rng = np.random.default_rng(20260901)
    coordinate = np.linspace(-3.0, 3.0, 240)
    line = np.column_stack([coordinate, 2 * coordinate, -coordinate, 0.5 * coordinate])
    volume = rng.normal(size=(240, 4))
    assert MODULE.intrinsic_statistics_from_cloud(volume)[0] > MODULE.intrinsic_statistics_from_cloud(line)[0] + 0.5


def test_affine_scale_invariance() -> None:
    coordinate = np.linspace(-3.0, 3.0, 240)
    line = np.column_stack([coordinate, 2 * coordinate, -coordinate, 0.5 * coordinate])
    assert np.allclose(MODULE.intrinsic_statistics_from_cloud(line), MODULE.intrinsic_statistics_from_cloud(7 * line + 3), rtol=1e-10, atol=1e-10)


def test_feature_shape_finite_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = MODULE.intrinsic_dimension_features(sequence)
    assert feature.shape == (32,)
    assert np.isfinite(feature).all()
    extended = np.vstack([sequence, np.full((10, 10), 1e9)])
    assert np.array_equal(feature, MODULE.intrinsic_dimension_features(extended[:289]))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["delay_rows"] == 2
    assert config["encoder"]["embedding_dimension"] == 4
    assert config["encoder"]["neighbor_k"] == [5, 10]
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
