"""Fulltrain and official-inference boundary checks without official data."""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p1_fulltrain", ROOT / "scripts/run_p1_clean_control_fulltrain_20260905_v1.py"
)
full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)


def sample_keys():
    return pd.DataFrame({"station": ["S-ORS"] * 3, "year": [2026] * 3,
                         "layer": [1] * 3, "time": ["a", "b", "c"]})


def test_answer_realigns_to_template_order():
    keys = sample_keys()
    evaluation = keys.iloc[[2, 0, 1]].copy()
    answer = full.align_answer(evaluation, np.array([1, 1, 0]), keys)
    assert answer[full.screen.KEYS].equals(keys)
    assert answer.label.to_list() == [1, 0, 1]


def test_duplicate_keys_fail():
    keys = sample_keys()
    with pytest.raises(ValueError, match="duplicate"):
        full.align_answer(keys, np.zeros(3), keys.iloc[[0, 0, 1]])


def test_mismatched_key_sets_fail():
    keys = sample_keys()
    bad = keys.copy()
    bad.loc[2, "time"] = "different"
    with pytest.raises(ValueError, match="sets differ"):
        full.align_answer(keys, np.zeros(3), bad)


@pytest.mark.parametrize("bad", [np.nan, 0.5, 2])
def test_nonbinary_answer_fails(bad):
    keys = sample_keys()
    with pytest.raises(ValueError):
        full.align_answer(keys, np.array([0, 1, bad]), keys)


def test_training_function_reads_only_training_csv():
    tree = ast.parse((ROOT / "scripts/run_p1_clean_control_fulltrain_20260905_v1.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    strings = [n.value for n in ast.walk(function) if isinstance(n, ast.Constant)
               and isinstance(n.value, str)]
    assert "train.csv" in strings
    assert "test.csv" not in strings and "sample_submission.csv" not in strings


def test_frozen_contract_matches_prior_q4_inner_selection():
    path = ROOT / "configs/experiments/p1_clean_control_fulltrain_20260905_v1.json"
    cfg = full.checked_contract(path)
    prior = json.loads((ROOT / "reports/p1_score_repair_20260905_v1/result.json").read_text())
    q4 = prior["folds"][-1]
    assert cfg["selection"] == q4["selected_control"]
    for key in ["original", "balanced_union"]:
        assert cfg["calibrations"][key]["threshold"] == q4["calibrations"][key]["threshold"]
    assert cfg["new_calibration_searches"] == 0


def test_missing_seal_blocks_official_read(tmp_path):
    (tmp_path / "train_result.json").write_text(json.dumps({"status": "RUNNING"}))
    with pytest.raises(ValueError, match="seal"):
        full.predict(tmp_path)


def test_no_unseen_year_depth_fallback_added():
    train = pd.DataFrame({"station": ["S-ORS"] * 2, "year": [2025] * 2,
                          "layer": [1] * 2, "time": ["2025-01-01T00:00+09:00",
                                                       "2025-01-01T00:10+09:00"],
                          "temp": [10., 10.1], "depth": [5., 5.]})
    stats = full.screen.stats_fit(train)
    assert ("S-ORS", 2026, 1) not in stats["depth"]
