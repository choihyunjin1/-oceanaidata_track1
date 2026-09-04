from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_path_signature_residual_cycle_20260901_v23.py"
SPEC = importlib.util.spec_from_file_location("p3_pathsig_v23", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sealed_candidate_budget_and_feature_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_REPRESENTATION_AXIS"
    assert config["path"]["signature_level"] == 2
    assert config["path"]["feature_count"] == 140
    assert len(MODULE.SPECS) * len(MODULE.BLOCKS) == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_level2_signature_of_one_segment_is_exact() -> None:
    path = np.asarray([[0.0, 1.0], [2.0, 4.0]], dtype=np.float64)
    level1, level2 = MODULE.level2_signature(path)
    expected = np.asarray([2.0, 3.0])
    assert np.array_equal(level1, expected)
    assert np.array_equal(level2, 0.5 * np.outer(expected, expected))


def test_ordered_cross_integrals_change_when_path_order_changes() -> None:
    first = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    second = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    one_l1, one_l2 = MODULE.level2_signature(first)
    two_l1, two_l2 = MODULE.level2_signature(second)
    assert np.array_equal(one_l1, two_l1)
    assert not np.array_equal(one_l2, two_l2)
    assert one_l2[0, 1] > one_l2[1, 0]
    assert two_l2[0, 1] < two_l2[1, 0]


def test_path_features_are_deterministic_finite_and_fixed_width() -> None:
    sequence = np.tile(np.arange(10, dtype=np.float64), (289, 1))
    sequence[:, 0] += np.linspace(0.0, 2.0, num=289)
    sequence[1::2, :4] = np.nan
    first = MODULE.path_signature_features(sequence, 1)
    second = MODULE.path_signature_features(sequence, 1)
    assert first.shape == (140,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_preflight_is_deterministic_zero_op() -> None:
    if MODULE.ARTIFACT_DIR.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="namespace is already consumed"):
            MODULE.preflight_payload()
        with pytest.raises(MODULE.ContractError, match="namespace is already consumed"):
            MODULE.preflight_payload()
        return
    first = MODULE.canonical(MODULE.preflight_payload())
    second = MODULE.canonical(MODULE.preflight_payload())
    assert first == second
    assert not MODULE.ARTIFACT_DIR.exists()
    assert not MODULE.LOCK.exists()
