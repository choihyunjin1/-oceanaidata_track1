"""Synthetic, no-source-data boundary tests for postpolicy comparison."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("p1_postpolicy", ROOT / "scripts/run_p1_trainfit_postpolicy_20260906_v1.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_range_fits_only_finite_normal_training_values():
    data = pd.DataFrame({"temp": [2., 10., -100., 100., np.nan], "label": [0, 0, 1, 1, 0]})
    bounds = m.fit_bounds(data)
    assert bounds == {"minimum": 2., "maximum": 10., "normal_training_rows": 2}
    query = pd.DataFrame({"temp": [1., 2., 10., 11., np.nan]})
    np.testing.assert_array_equal(m.range_bits(query, bounds), [1, 0, 0, 1, 0])
    assert m.fit_bounds(data) == bounds


def test_missing_normal_support_fails():
    with pytest.raises(ValueError):
        m.fit_bounds(pd.DataFrame({"temp": [20.], "label": [1]}))


def test_cells_inner_choice_unseen_fallback_and_target_free_apply():
    inner = pd.DataFrame({"station": ["A"] * 4 + ["B"] * 2,
                          "layer": [1] * 6, "label": [0, 1, 0, 1, 0, 0]})
    options = {"original": np.array([1, 1, 1, 1, 0, 0]),
               "balanced": np.array([0, 1, 0, 1, 1, 1])}
    options["balanced_union"] = np.maximum(options["original"], options["balanced"])
    policy = m.fit_cells(inner, options, "balanced_union")
    assert policy["cells"]["A/1"]["choice"] == "B"
    assert policy["cells"]["B/1"]["choice"] == "GLOBAL"
    query = pd.DataFrame({"station": ["A", "B", "C"], "layer": [1, 1, 2]})
    pred = {"original": np.array([1, 0, 1]), "balanced": np.array([0, 1, 0]),
            "balanced_union": np.array([1, 1, 1])}
    np.testing.assert_array_equal(m.apply_cells(query, pred, policy), [0, 1, 1])
    query["label"] = [1, 0, 0]
    np.testing.assert_array_equal(m.apply_cells(query, pred, policy), [0, 1, 1])


def test_equal_metric_tie_order_is_fixed():
    inner = pd.DataFrame({"station": ["A", "A"], "layer": [1, 1], "label": [0, 1]})
    options = {"original": np.array([0, 1]), "balanced": np.array([0, 1])}
    assert m.fit_cells(inner, options, "balanced")["cells"]["A/1"]["choice"] == "B"


def test_and_or_bits_and_inputs_not_mutated():
    options = {"original": np.array([0, 1, 0, 1]), "balanced": np.array([0, 0, 1, 1])}
    before = {k: v.copy() for k, v in options.items()}
    choices = m.arm_bits(options)
    np.testing.assert_array_equal(choices["AND"], [0, 0, 0, 1])
    np.testing.assert_array_equal(choices["OR"], [0, 1, 1, 1])
    for k in before:
        np.testing.assert_array_equal(options[k], before[k])


def test_contract_excludes_official_and_model_fits():
    c = m.load_contract()
    assert c["new_backbone_fits"] == 0
    assert not c["official_input_access"] and not c["gpu"]
