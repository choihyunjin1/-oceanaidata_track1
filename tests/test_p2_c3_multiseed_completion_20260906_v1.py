"""Synthetic plan/input/selection boundary tests, historical model IO0."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/p2_c3_multiseed_completion_20260906_v1/run.py"
SPEC = importlib.util.spec_from_file_location("p2_completion_test", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_missing_combinations_and_budget():
    c = m.cfg()
    combinations = {(a, f, s) for a in c["arms"] for f in c["new_folds"] for s in c["new_seeds"]}
    assert len(combinations) == 28
    assert all(f != "B3" and s != 20260901 for _, f, s in combinations)
    assert c["reused_models"] + len(combinations) == c["combined_models"] == 48
    assert c["primary"].startswith("B3") and "secondary" in c


def test_recipe_unchanged():
    c = m.r.cfg()
    for arm, epochs in (("C60", 60), ("L120", 120)):
        recipe = next(rec for rec in c["recipes"] if rec["id"] == arm)
        resolved = m.r.recipe_config(c, recipe)
        assert resolved["weight_decay"] == .0001 and resolved["epochs"] == epochs
        assert resolved["gradient_coefficient"] == .01 and resolved["cpu_threads"] == 2
        assert resolved["blockmask"] == {"coverage": .3, "length_days": [3, 7, 14], "seed": 20260905}


def test_mean3_is_not_mean_rmse():
    y = np.array([1., 2.])
    predictions = np.array([[0., 0.], [1., 2.], [2., 4.]])
    assert m.r.metric(y, predictions.mean(axis=0))["rmse_C"] == 0
    assert np.mean([m.r.metric(y, p)["rmse_C"] for p in predictions]) > 0


def test_purge_same_exact8fold_contract():
    assert len(m.r.contract()["folds"]) == 8
    frame = pd.DataFrame({"time": pd.to_datetime(["2024-08-24T23:59Z", "2024-08-25T00:00Z", "2024-09-01T00:00Z", "2024-11-08T00:00Z"], utc=True)})
    train, valid = m.r.split_masks(frame, {"start": "2024-09-01T00:00Z", "end": "2024-11-01T00:00Z"})
    np.testing.assert_array_equal(train, [True, False, False, True])
    np.testing.assert_array_equal(valid, [False, False, True, False])


def test_no_warmstart_or_candidate_reselection():
    code = PATH.read_text(encoding="utf-8")
    worker = code.split("def worker():", 1)[1].split("\ndef replay():", 1)[0]
    assert "torch.load" not in worker and "guard(True)" in worker
    assert "choose_challenger" not in worker
    assert 'primary_B3_unchanged=True' in worker and 'selection_changed=False' in worker
    assert 'if fold["id"] not in c["new_folds"]' in worker


def test_old_artifacts_readonly_guard_declared():
    code = PATH.read_text(encoding="utf-8")
    assert "reused/source artifacts immutable" in code
    assert "no warm start torch.load during new fits" in code
    assert 'r.path_allowed(path, writing, r.source_path(), OUT)' in code
    assert "test_index.csv" not in code and "sample_submission.csv" not in code
