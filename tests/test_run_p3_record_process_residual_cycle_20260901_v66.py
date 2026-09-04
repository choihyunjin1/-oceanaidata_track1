from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_record_process_residual_cycle_20260901_v66.py"
SPEC = importlib.util.spec_from_file_location("p3_v66", RUNNER)
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


def test_monotone_direction_and_permutation_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["rising_upper_count_fraction"] == 1.0
    assert receipt["rising_lower_count_fraction"] == 1.0 / 96.0
    assert receipt["permutation_sensitive"] is True


def test_affine_invariance_and_direction_exchange() -> None:
    rng = np.random.default_rng(11)
    path = np.cumsum(rng.normal(size=96))
    direct = MODULE.robust_record_statistics(path)
    assert np.array_equal(direct, MODULE.robust_record_statistics(7 * path + 3))
    negative = MODULE.robust_record_statistics(-7 * path + 3)
    assert np.array_equal(direct[:4], negative[4:])
    assert np.array_equal(direct[4:], negative[:4])


def test_feature_shape_finite_deterministic_and_future_isolation() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    first = MODULE.record_features(sequence)
    assert first.shape == (64,) and np.isfinite(first).all()
    assert np.array_equal(first, MODULE.record_features(sequence.copy()))
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    assert np.array_equal(first, MODULE.record_features(extended[:289]))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["feature_count"] == 64
    assert config["validation"]["maximum_total_fits"] == 12
    assert config["encoder"]["duplicate_rule"] == "ties do not create a new record"
    assert all(value == 0 for value in config["official_policy"].values())
