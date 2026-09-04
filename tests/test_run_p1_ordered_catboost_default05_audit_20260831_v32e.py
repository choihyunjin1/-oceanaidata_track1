from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_p1_ordered_catboost_default05_audit_20260831_v32e.py"
SPEC = importlib.util.spec_from_file_location("v32e", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_contract_is_fixed_default_threshold_and_fit_zero() -> None:
    contract = MODULE.load_contract()
    assert contract["decision_threshold"] == 0.5
    assert contract["fit_budget"] == 0


def test_action_hash_is_dtype_and_shape_bound() -> None:
    assert MODULE.sha256_array(np.array([0, 1], dtype=np.int8)) != MODULE.sha256_array(
        np.array([0, 1], dtype=np.int64)
    )


def test_concentration() -> None:
    frame = pd.DataFrame(
        {"station": ["S", "S", "I"], "layer": [1, 1, 2], "fold": ["q", "q", "q"]}
    )
    value = MODULE.concentration(frame, np.array([True, True, True]))
    assert value["maximum_station_layer_fold_rows"] == 2
    assert np.isclose(value["maximum_station_layer_fold_share"], 2 / 3)
