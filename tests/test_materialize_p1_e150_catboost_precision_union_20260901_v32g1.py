from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_p1_e150_catboost_precision_union_20260901_v32g1.py"
SPEC = importlib.util.spec_from_file_location("p1_v32g1_materializer", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_contract_is_exact_and_policy_clean() -> None:
    deployment, base, result = MODULE.load_contract()
    assert base["model"]["probability_threshold"] == 0.8
    assert result["candidate"]["additions"] == 15
    assert all(value == 0 for value in result["official_access"].values())
    assert deployment["data_policy"]["organizer_distributed_data_only"] is True


def test_guard_accepts_small_nonduplicate_union() -> None:
    champion = np.array([1, 0, 0, 0], dtype=np.int8)
    additions = np.array([0, 1, 0, 0], dtype=bool)
    label = np.maximum(champion, additions).astype(np.int8)
    checks = MODULE.deployability_checks(label, champion, additions, 0.30, 0.30)
    assert all(checks.values())


def test_guard_blocks_duplicate() -> None:
    champion = np.array([1, 0, 0, 0], dtype=np.int8)
    additions = np.zeros(4, dtype=bool)
    checks = MODULE.deployability_checks(champion, champion, additions, 0.30, 0.30)
    assert checks["positive_additions"] is False


def test_guard_blocks_all_positive_collapse() -> None:
    champion = np.array([1, 0, 0, 0], dtype=np.int8)
    additions = np.array([0, 1, 1, 1], dtype=bool)
    label = np.ones(4, dtype=np.int8)
    checks = MODULE.deployability_checks(label, champion, additions, 0.30, 0.30)
    assert checks["binary_nonconstant"] is False
    assert checks["positive_fraction_within_historical_multiplier"] is False
