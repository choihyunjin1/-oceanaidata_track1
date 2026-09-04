from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_directional_change_intrinsic_time_residual_cycle_20260901_v78.py"
SPEC = importlib.util.spec_from_file_location("p3_v78", RUNNER)
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


def test_trend_threshold_and_alternation_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["trend_event_counts"] == [0, 0, 0]
    counts = receipt["triangle_event_counts"]
    assert counts[0] >= counts[1] >= counts[2] > 0
    assert receipt["alternating_events"] is True


def test_direction_reversal_and_positive_affine_guards() -> None:
    triangle = np.tile(np.r_[np.linspace(-2.0, 2.0, 9), np.linspace(1.5, -2.0, 8)], 5)[:73]
    direct = MODULE.directional_change_events(triangle, 1.0)
    inverted = MODULE.directional_change_events(-triangle, 1.0)
    assert np.array_equal(direct[0], -inverted[0])
    assert np.allclose(direct[1], inverted[1], atol=1e-12, rtol=0.0)
    assert np.array_equal(direct[2], inverted[2])
    assert np.allclose(
        MODULE.robust_normalize(triangle),
        MODULE.robust_normalize(7.0 + 3.0 * triangle),
        atol=1e-12,
        rtol=0.0,
    )


def test_feature_shape_determinism_and_future_isolation() -> None:
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack(
        [np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)]
    )
    sequence[:, 5] = 6.0 + np.sin(3.0 * axis)
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = MODULE.directional_change_features(sequence)
    assert direct.shape == (144,) and np.isfinite(direct).all()
    assert np.array_equal(direct, MODULE.directional_change_features(sequence.copy()))
    assert np.array_equal(
        direct, MODULE.directional_change_features(np.vstack([sequence, np.full((12, 10), 1e9)]))
    )


def test_constant_and_sealed_contract() -> None:
    assert all(
        len(MODULE.directional_change_events(np.ones(73), value)[0]) == 0
        for value in MODULE.THRESHOLDS
    )
    config = MODULE.load_config()
    assert config["encoder"]["feature_count"] == 144
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
