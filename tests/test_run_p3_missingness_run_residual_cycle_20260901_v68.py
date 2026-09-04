from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_missingness_run_residual_cycle_20260901_v68.py"
SPEC = importlib.util.spec_from_file_location("p3_v68", RUNNER)
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
        assert value["known_missingness_shift_used_for_selection"] is False
        assert value["prior_outputs_used"] is False
        assert value["official_used_for_features_gates_selection"] is False


def test_equal_fraction_isolated_vs_burst_and_bounds() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["isolated_run_count"] > receipt["burst_run_count"]
    assert receipt["burst_longest_run"] > receipt["isolated_longest_run"]
    assert receipt["all_observed"] == [1.0, 0.0, 0.0, 0.0]
    assert receipt["all_missing"] == [0.0, 1.0 / 96.0, 1.0, 1.0]


def test_missing_run_exactness() -> None:
    finite = np.asarray([True, False, False, True, False, True, True, False])
    assert np.array_equal(MODULE.missing_runs(~finite), [2, 1, 1])
    stats = MODULE.mask_statistics(finite)
    assert np.allclose(stats, [0.5, 3 / 8, 2 / 8, 1 / 8])


def test_feature_shape_determinism_channel_permutation_future_isolation() -> None:
    sequence = np.arange(2890, dtype=np.float64).reshape(289, 10)
    for channel in range(10):
        sequence[(np.arange(289) + channel) % (7 + channel) == 0, channel] = np.nan
    direct = MODULE.mask_features(sequence).reshape(10, 2, 4)
    assert direct.shape == (10, 2, 4) and np.isfinite(direct).all()
    assert np.array_equal(direct.ravel(), MODULE.mask_features(sequence.copy()))
    permutation = np.asarray([9, 7, 5, 3, 1, 8, 6, 4, 2, 0])
    assert np.array_equal(MODULE.mask_features(sequence[:, permutation]).reshape(10, 2, 4), direct[permutation])
    assert np.array_equal(direct.ravel(), MODULE.mask_features(np.vstack([sequence, np.full((12, 10), np.nan)])))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["feature_count"] == 80
    assert config["encoder"]["observed_values_used"] is False
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
