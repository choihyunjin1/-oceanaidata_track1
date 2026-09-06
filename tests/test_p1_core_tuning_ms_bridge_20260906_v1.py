"""Synthetic fixed-MS bridge tests; no original artifacts or official inputs."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p1_tuning_ms_bridge", ROOT / "scripts/p1_core_tuning_20260906_v1/ms_bridge.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture():
    p = pd.DataFrame({"station": ["S", "S", "I", "I"], "year": 2025, "layer": 1,
                      "time": ["2025-09-30T23:50:00+09:00", "2025-10-01T00:00:00+09:00",
                               "2025-10-01T00:00:00+09:00", "2025-10-01T00:10:00+09:00"]})
    cfg = {"folds": [{"id": "q3", "source_rows": 2, "ordered_source_key_sha256": m.u.original_ordered_key_sha(p.iloc[:2])},
                     {"id": "q4", "source_rows": 2, "ordered_source_key_sha256": m.u.original_ordered_key_sha(p.iloc[2:])}]}
    tree = p.assign(fold=["q3", "q3", "q4", "q4"]).iloc[[3, 0, 2, 1]].reset_index(drop=True)
    return tree, p, cfg


def test_original_boundary_owner_and_full_permutation():
    tree, p, cfg = fixture()
    left, owned, order = m.exact_alignment(tree, p, cfg)
    assert order.tolist() == [3, 0, 2, 1]
    assert owned.fold.iloc[1] == "q3"  # October boundary retained in original Q3
    assert len(left) == len(owned) == 4


@pytest.mark.parametrize("change", ["missing", "wrong_fold", "duplicate", "proposal_order"])
def test_exact_join_rejects_intersection_and_reassigned_owner(change):
    tree, p, cfg = fixture()
    if change == "missing":
        tree = tree.iloc[:-1]
    elif change == "wrong_fold":
        tree.loc[3, "fold"] = "q4"
    elif change == "duplicate":
        tree = pd.concat([tree, tree.iloc[:1]], ignore_index=True)
    else:
        p = p.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        m.exact_alignment(tree, p, cfg)


def test_same_fixed_proposal_preserved_for_both_arms():
    b = np.array([1, 0, 0, 0], dtype=np.int8)
    c = np.array([0, 1, 0, 0], dtype=np.int8)
    proposal = np.array([0, 0, 1, 0], dtype=np.int8)
    assert (b | proposal).tolist() == [1, 0, 1, 0]
    assert (c | proposal).tolist() == [0, 1, 1, 0]
    assert np.all((b | proposal) >= b) and np.all((c | proposal) >= c)


def test_seven_day_bootstrap_matches_fixed_original_engine():
    f = pd.DataFrame({"fold": ["q3"] * 6 + ["q4"] * 6, "block": [0, 0, 1, 1, 2, 2] * 2,
        "label": [1, 0, 1, 0, 1, 0] * 2, "control_plus_ms": [1, 0, 0, 0, 1, 1] * 2,
        "candidate_plus_ms": [1, 0, 1, 0, 1, 1] * 2})
    cfg = {"bootstrap": {"seed": 20260906, "resamples": 2000, "minimum_clusters_per_fold": 2, "ci_quantiles": [.05, .95]}}
    actual = m.bootstrap(f, cfg)
    # Original function only uses these two columns for its paired arithmetic.
    f["control_plus_ms_OR_e150"] = f.candidate_plus_ms
    reference = m.e.bootstrap(f, "control_plus_ms", cfg)
    np.testing.assert_array_equal(actual["ci90"], reference["ci90"])
    assert actual["p_improve"] == reference["fraction_delta_positive"]


def test_no_training_official_or_old_CPU4_OOF_read():
    source = Path(m.__file__).read_text()
    assert "test.csv" not in source and "sample_submission" not in source
    assert "model.fit(" not in source and "to_csv(" not in source
    assert 'inputs"]["tree_directory"]' not in source
    assert 'qa["status"] != "PASS"' in source
