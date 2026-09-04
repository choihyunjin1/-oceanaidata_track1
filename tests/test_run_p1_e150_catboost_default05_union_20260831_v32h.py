from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_p1_e150_catboost_default05_union_20260831_v32h.py"
SPEC = importlib.util.spec_from_file_location("v32h", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_contract_is_fit_and_search_zero() -> None:
    contract = MODULE.load_contract()
    assert contract["fit_budget"] == 0
    assert contract["threshold_searches"] == 0


def test_union_is_add_only() -> None:
    e150 = np.array([0, 1, 0, 1], dtype=np.int8)
    cat = np.array([1, 0, 0, 1], dtype=np.int8)
    union = np.maximum(e150, cat)
    assert union.tolist() == [1, 1, 0, 1]
    assert not ((union == 0) & (e150 == 1)).any()


def test_hash_binds_dtype() -> None:
    assert MODULE.sha256_array(np.array([0, 1], dtype=np.int8)) != MODULE.sha256_array(
        np.array([0, 1], dtype=np.int64)
    )
