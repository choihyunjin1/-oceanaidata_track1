"""Synthetic original-fold ownership regression, including 119 October Q3 rows."""
import copy
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("union_v2_regression", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/evaluate_union_v2.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def synthetic():
    cfg = A.engine.read_json(A.CONTRACT)
    time = list(pd.date_range("2025-07-01T00:00+09:00", periods=3, freq="8D"))
    time += list(pd.date_range("2025-10-01T00:00+09:00", periods=119, freq="10min"))
    time += list(pd.date_range("2025-10-03T00:00+09:00", periods=3, freq="8D"))
    p = pd.DataFrame({"station": ["S"] * len(time), "year": [2025] * len(time), "layer": [1] * len(time),
                      "time": [v.isoformat() for v in time]})
    for fold, part in zip(cfg["folds"], (p.iloc[:122], p.iloc[122:]), strict=True):
        fold["source_rows"] = len(part)
        fold["ordered_source_key_sha256"] = A.original_ordered_key_sha(part)
    return p, cfg


def test_119_cross_boundary_run_rows_preserved_and_exact_keys():
    p, cfg = synthetic()
    owned = A.original_fold_ownership(p, cfg)
    assert owned.fold.value_counts().to_dict() == {"2025_q3": 122, "2025_q4": 3}
    assert owned.iloc[3:122].fold.eq("2025_q3").all()
    tree = p.copy()
    tree["fold"] = owned.fold
    engine = A.configure_engine()
    coverage, permutation = engine.key_coverage(tree, p, cfg)
    assert coverage["same_keys_and_folds"] and (permutation == range(len(p))).all()


def test_original_ordered_key_sha_mismatch_fails():
    p, cfg = synthetic()
    with pytest.raises(ValueError, match="SHA"):
        A.original_fold_ownership(p.iloc[::-1], cfg)


def test_removed_rows_never_implicitly_intersect():
    p, cfg = synthetic()
    with pytest.raises(ValueError, match="population"):
        A.original_fold_ownership(p.iloc[:-1], cfg)


def test_numerical_contract_and_prediction_unchanged():
    original = A.engine.read_json(A.REPORT / "union-contract.json")
    fixed = A.engine.read_json(A.CONTRACT)
    assert original["proposal"] == fixed["proposal"]
    for field in ("resamples", "seed", "block_days", "anchor_kst", "stratify", "minimum_clusters_per_fold", "ci_quantiles"):
        assert original["bootstrap"][field] == fixed["bootstrap"][field]
    assert original["inputs"] == fixed["inputs"]
    assert original["tree_arms"] == fixed["tree_arms"]
    assert original["fit_budget"] == fixed["fit_budget"] == 0


def test_wrong_original_boundary_cannot_be_silently_reassigned():
    p, cfg = synthetic()
    wrong = copy.deepcopy(cfg)
    wrong["folds"][0]["source_rows"] -= 119
    wrong["folds"][1]["source_rows"] += 119
    with pytest.raises(ValueError, match="SHA"):
        A.original_fold_ownership(p, wrong)
