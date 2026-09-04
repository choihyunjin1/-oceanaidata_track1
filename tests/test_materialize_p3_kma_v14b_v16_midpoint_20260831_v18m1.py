from __future__ import annotations

import pandas as pd

from scripts.materialize_p3_kma_v14b_v16_midpoint_20260831_v18m1 import (
    key_values_and_order_equal,
)


def test_dtype_only_key_mismatch_passes() -> None:
    left = pd.DataFrame({"case_id": pd.Series(["a", "b"], dtype="string"), "station": pd.Series(["G", "I"], dtype="string"), "lead_h": [3, 6]})
    right = pd.DataFrame({"case_id": ["a", "b"], "station": ["G", "I"], "lead_h": [3, 6]})
    assert key_values_and_order_equal(left, right)


def test_key_perturbation_fails() -> None:
    left = pd.DataFrame({"case_id": ["a", "b"], "station": ["G", "I"], "lead_h": [3, 6]})
    right = pd.DataFrame({"case_id": ["a", "x"], "station": ["G", "I"], "lead_h": [3, 6]})
    assert not key_values_and_order_equal(left, right)


def test_order_perturbation_fails() -> None:
    left = pd.DataFrame({"case_id": ["a", "b"], "station": ["G", "I"], "lead_h": [3, 6]})
    right = left.iloc[::-1].reset_index(drop=True)
    assert not key_values_and_order_equal(left, right)
