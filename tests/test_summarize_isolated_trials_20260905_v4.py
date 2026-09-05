import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "summary",
    Path(__file__).resolve().parents[1] / "scripts/summarize_isolated_trials_20260905_v4.py",
)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


@pytest.mark.parametrize("values", [[0, 0.5], [0, np.nan]])
def test_reject_invalid_binary(values):
    with pytest.raises(ValueError):
        summary.contributions([0, 1], values, "f1")


def test_pooled_f1_not_average_of_blocks():
    sums = summary.contributions([1, 1, 0, 0], [1, 0, 1, 0], "f1").sum(0)
    assert summary.value(sums, "f1") == 0.5


def test_boolean_f1_uses_count_addition():
    sums = summary.contributions([True, False], [True, False], "f1").sum(0)
    assert summary.value(sums, "f1") == 1.0


def test_exact_rmse_improvement_and_pairing():
    data = pd.DataFrame(
        {
            "y": [0.0, 0.0, 0.0, 0.0],
            "control": [1.0, 1.0, 1.0, 1.0],
            "candidate": [0.0, 0.0, 0.0, 0.0],
            "fold": ["a", "a", "b", "b"],
            "cluster": ["1", "2", "3", "4"],
            "layer": [1, 2, 1, 2],
        }
    )
    result = summary.comparison(data, "rmse", ["layer"])
    assert result["candidate_minus_control"] == -1
    assert result["ci90_descriptive"] == [-1, -1]
    assert result["bootstrap_improvement_fraction_not_posterior"] == 1
    assert result["worst_fold_harm"] == -1


def test_noop_fraction_zero_not_improvement():
    data = pd.DataFrame(
        {
            "y": [0, 1],
            "control": [0, 1],
            "candidate": [0, 1],
            "fold": ["a", "a"],
            "cluster": ["1", "1"],
            "layer": [1, 1],
        }
    )
    result = summary.comparison(data, "f1", ["layer"])
    assert result["ci90_descriptive"] == [0, 0]
    assert result["bootstrap_improvement_fraction_not_posterior"] == 0
