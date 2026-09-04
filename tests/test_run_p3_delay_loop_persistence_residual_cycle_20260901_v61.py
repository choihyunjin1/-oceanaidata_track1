from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_delay_loop_persistence_residual_cycle_20260901_v61.py"
SPEC = importlib.util.spec_from_file_location("p3_v61", RUNNER)
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
        assert value["v36_features_predictions_used"] is False
        assert value["official_v42_used_for_features_gates_selection"] is False


def test_circle_has_larger_h1_loop_than_line() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, MODULE.CLOUD_POINTS, endpoint=False)
    circle = np.column_stack([np.cos(angle), np.sin(angle)])
    line = np.column_stack([np.linspace(-1.0, 1.0, MODULE.CLOUD_POINTS), np.zeros(MODULE.CLOUD_POINTS)])
    assert MODULE.loop_statistics_from_cloud(circle)[0] > MODULE.loop_statistics_from_cloud(line)[0] + 0.10


def test_scale_translation_invariance() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, MODULE.CLOUD_POINTS, endpoint=False)
    circle = np.column_stack([np.cos(angle), np.sin(angle)])
    assert np.allclose(MODULE.loop_statistics_from_cloud(circle), MODULE.loop_statistics_from_cloud(7 * circle + 3), rtol=1e-10, atol=1e-10)


def test_feature_shape_finite_deterministic_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    first = MODULE.loop_persistence_features(sequence)
    second = MODULE.loop_persistence_features(sequence.copy())
    assert first.shape == (32,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    assert np.array_equal(first, MODULE.loop_persistence_features(extended[:289]))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["delay_rows"] == 2
    assert config["encoder"]["embedding_dimension"] == 2
    assert config["encoder"]["cloud_points"] == 16
    assert config["encoder"]["feature_count"] == 32
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
