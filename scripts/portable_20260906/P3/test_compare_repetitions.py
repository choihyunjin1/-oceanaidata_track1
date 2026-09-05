"""Synthetic population and arithmetic checks; no real data or model access."""

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "compare_repetitions", Path(__file__).with_name("compare_repetitions.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_and_signed_rmse():
    a = pd.DataFrame({"key": [1, 2], "y": [1.0, 2.0], "p": [1.0, 3.0]})
    b = a.assign(p=[2.0, 2.0])
    result = module.paired_difference(a, b, ["key"], "p", target="y")
    assert result["difference_rmse_m"] == 1.0
    assert result["mean_signed_difference_b_minus_a_m"] == 0.0
    assert result["rmse_a_m"] == math.sqrt(0.5)
    assert result["rmse_delta_b_minus_a_m"] == 0.0
    assert module.paired_difference(a, a.copy(), ["key"], "p")["exact_prediction_array"]


@pytest.mark.parametrize("change", ["order", "duplicate", "target", "nonfinite"])
def test_invalid_pair_is_rejected(change):
    a = pd.DataFrame({"key": [1, 2], "y": [1.0, 2.0], "p": [1.0, 3.0]})
    b = a.copy()
    if change == "order":
        b = b.iloc[::-1].reset_index(drop=True)
    elif change == "duplicate":
        b["key"] = [1, 1]
    elif change == "target":
        b["y"] = [0.0, 2.0]
    else:
        b["p"] = [float("nan"), 2.0]
    with pytest.raises(ValueError):
        module.paired_difference(a, b, ["key"], "p", target="y")
