"""Synthetic only: no source data, model, OOF or official I/O."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p1_union_eval_test", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/evaluate_union.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def fixture():
    cfg = E.read_json(E.CONTRACT)
    times = ["2025-07-01T00:00:00+09:00", "2025-07-09T00:00:00+09:00", "2025-07-16T00:00:00+09:00",
             "2025-10-01T00:00:00+09:00", "2025-10-09T00:00:00+09:00", "2025-10-16T00:00:00+09:00"]
    tree = pd.DataFrame({"station": ["S"] * 6, "year": [2025] * 6, "layer": [1] * 6,
                         "time": times, "fold": ["2025_q3"] * 3 + ["2025_q4"] * 3,
                         "label": [1, 0, 1, 0, 1, 0]})
    for arm in E.ARMS:
        tree[arm] = [1, 0, 0, 0, 1, 0]
    prop = tree[E.KEYS].copy()
    prop["proposal"] = [0, 0, 1, 1, 0, 0]
    return tree, prop, cfg


def test_permutation_exact_and_or():
    tree, prop, cfg = fixture()
    prop = prop.iloc[::-1].reset_index(drop=True)
    coverage, permutation = E.key_coverage(tree, prop, cfg)
    frame, result = E.evaluate_frames(tree, prop, permutation, cfg)
    assert coverage["same_keys_and_folds"]
    assert frame.O_OR_e150.tolist() == [1, 0, 1, 1, 1, 0]
    comp = result["scopes"]["primary_Q3_Q4"]["O"]
    assert comp["control"]["f1"] == pytest.approx(0.8)
    assert comp["candidate"]["f1"] == pytest.approx(6 / 7)
    assert comp["removed_rows"] == 0
    assert E.independent_arithmetic(frame, result)["status"] == "PASS"


@pytest.mark.parametrize("side", ["tree", "proposal"])
def test_missing_extra_blocks_without_intersection(side):
    tree, prop, cfg = fixture()
    if side == "tree":
        tree = tree.iloc[1:]
    else:
        prop = prop.iloc[1:]
    coverage, permutation = E.key_coverage(tree, prop, cfg)
    assert not coverage["same_keys_and_folds"] and permutation is None
    assert coverage["implicit_intersection_rows_scored"] == 0
    assert coverage["tree_without_proposal"] + coverage["proposal_without_tree"] == 1


def test_wrong_fold_not_equal_key():
    tree, prop, cfg = fixture()
    tree.loc[0, "fold"] = "2025_q4"
    report, permutation = E.key_coverage(tree, prop, cfg)
    assert not report["same_keys_and_folds"] and permutation is None
    assert report["tree_without_proposal"] == report["proposal_without_tree"] == 1


def test_duplicate_and_invalid_keys_rejected():
    tree, prop, cfg = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        E.key_coverage(tree, pd.concat([prop, prop.iloc[:1]]), cfg)
    tree.loc[0, "layer"] = np.nan
    with pytest.raises(ValueError, match="nonmissing"):
        E.key_coverage(tree, prop, cfg)


def test_timezone_equivalence():
    tree, prop, cfg = fixture()
    prop["time"] = pd.to_datetime(prop.time, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert E.key_coverage(tree, prop, cfg)[0]["same_keys_and_folds"]


def test_actual_historical_cutoff_and_337h_support():
    _, prop, cfg = fixture()
    qa = {"folds": {f["phase"]: {"training_max_time_utc": f["training_max_utc"], "rows": 3} for f in cfg["folds"]}}
    checks = E.chronology(prop, qa, cfg)
    assert all(c["minimum_gap_hours"] == pytest.approx(504 + 1 / 6) for c in checks)
    qa["folds"]["q3"]["training_max_time_utc"] = "2025-06-30T14:50:00+00:00"
    with pytest.raises(ValueError, match="cutoff"):
        E.chronology(prop, qa, cfg)


def test_unsupported_purge_rejected():
    _, prop, cfg = fixture()
    cfg["required_feature_dependency_hours"] = 505
    qa = {"folds": {f["phase"]: {"training_max_time_utc": f["training_max_utc"], "rows": 3} for f in cfg["folds"]}}
    with pytest.raises(ValueError, match="gap insufficient"):
        E.chronology(prop, qa, cfg)


def test_pooled_not_mean_fold_metric():
    assert E.f1_from_counts(np.array([4, 2, 3, 8])) == pytest.approx(8 / 13)
    assert E.f1_from_counts(np.array([0, 0, 0, 8])) == 0
    with pytest.raises(ValueError, match="binary"):
        E.counts([0, 1], [0, 0.2])


def test_bootstrap_seed_repeat_and_noop():
    tree, prop, cfg = fixture()
    prop["proposal"] = 0
    _, permutation = E.key_coverage(tree, prop, cfg)
    frame, _ = E.evaluate_frames(tree, prop, permutation, cfg)
    one, two = E.bootstrap(frame, "B", cfg), E.bootstrap(frame, "B", cfg)
    assert one == two
    assert one["ci90"] == [0, 0] and one["fraction_delta_positive"] == 0
    assert one["fraction_delta_zero"] == 1
    assert one["clusters"] == {"2025_q3": 3, "2025_q4": 3}


def test_too_few_blocks_not_false_certainty():
    tree, prop, cfg = fixture()
    _, permutation = E.key_coverage(tree, prop, cfg)
    frame, _ = E.evaluate_frames(tree, prop, permutation, cfg)
    assert E.bootstrap(frame.iloc[:1], "B", cfg)["status"] == "NOT_ESTIMABLE_TOO_FEW_BLOCKS"


def test_unexpected_proposal_period_rejected():
    _, prop, cfg = fixture()
    prop.loc[0, "time"] = "2025-04-01T00:00:00+09:00"
    with pytest.raises(ValueError, match="outside"):
        E.attach_proposal_folds(prop, cfg)


def test_sources_exclude_fit_original_official_reads():
    import ast
    source = Path(E.__file__).read_text(encoding="utf-8")
    parsed = ast.parse(source)
    calls = [node.func.attr for node in ast.walk(parsed) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not {"read_csv", "to_csv", "fit", "train", "predict", "predict_proba"}.intersection(calls)
    assert "old router/candidate arrays" in source
