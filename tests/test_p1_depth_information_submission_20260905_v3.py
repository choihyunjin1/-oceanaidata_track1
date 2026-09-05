"""Synthetic-only tests for frozen policy and official output adapter."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p1_info", ROOT / "scripts/run_p1_depth_information_submission_20260905_v3.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def keys():
    return pd.DataFrame({"station": ["S-ORS"] * 3, "year": [2026] * 3,
                         "layer": [1] * 3, "time": ["a", "b", "c"]})


def test_order_is_preserved():
    template = keys()
    result = m.align_answer(template.iloc[[2, 0, 1]], [1, 0, 1], template)
    assert result[m.OLD.KEYS].equals(template)
    assert result.label.tolist() == [0, 1, 1]


def test_duplicate_or_mismatched_keys_fail():
    template = keys()
    with pytest.raises(ValueError, match="duplicate"):
        m.align_answer(template, [0, 1, 0], template.iloc[[0, 0, 1]])
    bad = template.copy()
    bad.loc[2, "time"] = "z"
    with pytest.raises(ValueError, match="sets differ"):
        m.align_answer(template, [0, 1, 0], bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.5, -1, 2])
def test_invalid_labels_fail(bad):
    with pytest.raises(ValueError):
        m.align_answer(keys(), [0, bad, 1], keys())


@pytest.mark.parametrize("change", [{"selection": "balanced_union"}, {"decoder_on": True},
                                  {"depth_policy": "year_lookup"},
                                  {"calibrations": {"balanced": {"threshold": 0.3}}}])
def test_policy_cannot_drift(change):
    recipe = {"selection": "balanced", "decoder_on": False,
              "depth_policy": "current_observation_round_2m_explicit_missing",
              "calibrations": {"balanced": {"threshold": 0.2}}}
    m.check_recipe(recipe)
    with pytest.raises(ValueError, match="frozen"):
        m.check_recipe({**recipe, **change})


def test_no_fitting_and_explicit_sample_key_projection():
    tree = ast.parse((ROOT / "scripts/run_p1_depth_information_submission_20260905_v3.py")
                     .read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in
                   {"fit", "calibrate", "stats_fit", "execute"} for node in calls)
    reads = [node for node in calls if isinstance(node.func, ast.Attribute)
             and node.func.attr == "read_csv"]
    assert len(reads) == 2
    assert {ast.unparse(next(k.value for k in node.keywords if k.arg == "usecols"))
            for node in reads} == {"OLD.RAW", "OLD.KEYS"}
