from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_sax_word_histogram_residual_cycle_20260901_v69.py"
SPEC = importlib.util.spec_from_file_location("p3_v69", RUNNER)
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


def test_affine_constant_and_histogram_mass_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["positive_affine_invariant"] is True
    assert receipt["constant_middle_word_mass"] == 1.0
    assert receipt["histogram_mass"] == 1.0


def test_equal_marginal_motif_order_discrimination() -> None:
    first = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1])
    second = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4])
    assert np.array_equal(np.bincount(first, minlength=5), np.bincount(second, minlength=5))
    assert not np.array_equal(MODULE.word_histogram(first), MODULE.word_histogram(second))
    assert np.sum(MODULE.word_histogram(first)) == 1.0


def test_feature_shape_determinism_and_future_isolation() -> None:
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * axis) + 0.1 * index * axis for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = MODULE.sax_features(sequence)
    assert direct.shape == (1000,) and np.isfinite(direct).all()
    assert np.array_equal(direct, MODULE.sax_features(sequence.copy()))
    assert np.array_equal(direct, MODULE.sax_features(np.vstack([sequence, np.full((12, 10), 1e9)])))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["encoder"]["feature_count"] == 1000
    assert config["encoder"]["paa_blocks"] == 12
    assert config["encoder"]["word_length"] == 3
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
