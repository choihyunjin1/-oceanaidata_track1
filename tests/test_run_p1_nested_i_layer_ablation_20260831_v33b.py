from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_p1_nested_i_layer_ablation_20260831_v33b.py"
SPEC = importlib.util.spec_from_file_location("v33b", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_contract_is_fixed() -> None:
    contract = MODULE.load_contract()
    assert contract["selection"]["minimum_layer_addition_support_inclusive"] == 10
    assert contract["selection"]["outer_result_reselection"] is False
    assert contract["fit_budget"] == 0


def test_hash_binds_dtype() -> None:
    assert MODULE.sha256_array(np.array([0, 1], dtype=np.int8)) != MODULE.sha256_array(
        np.array([0, 1], dtype=np.int64)
    )
