"""Independent tree QA helpers tested on synthetic metadata only."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tree_qa_test", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/qa_tree.py")
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def test_counts_sum_not_mean_fold_f1():
    y = np.r_[1, np.zeros(9), 1, 1]
    p = np.r_[1, np.zeros(9), 0, 0]
    assert q.counts(y, p)["f1"] == .5
    assert q.counts(y[:10], p[:10])["f1"] == 1
    assert q.counts(y[10:], p[10:])["f1"] == 0
    assert q.counts([0, 0], [0, 0])["f1"] == 0


@pytest.mark.parametrize("y,p", [([], []), ([1], [np.nan]), ([0, 1], [1]), ([0], [2])])
def test_invalid_counts(y, p):
    with pytest.raises(ValueError):
        q.counts(y, p)


def test_independent_bootstrap_matches_pooled_cluster_implementation():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    b = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    c = np.array([0, 1, 1, 1, 1, 0, 1, 0])
    groups = ["z", "z", "a", "a", "c", "c", "d", "d"]
    expected = q.independent_bootstrap(y, b, c, groups)
    cfg = {"common": {"bootstrap": {"minimum_clusters": 2, "seed": 20260906,
           "resamples": 2000, "ci_quantiles": [.05, .95]}}}
    actual = q.t.evaluation.paired_bootstrap(y, b, c, groups, "f1", cfg)
    assert expected["ci90"] == actual["ci90"]
    assert expected["p_improve"] == actual["p_improve"]
    assert expected["delta"] == actual["delta_candidate_minus_control"]


def test_independent_run_split_matches_start_ownership_and_gap():
    n = 150
    frame = pd.DataFrame({"station": "S", "year": 2025, "layer": 1,
        "time": pd.date_range("2025-01-01", periods=n, freq="10min", tz="Asia/Seoul").astype(str),
        "label": np.zeros(n, dtype=int)})
    frame.loc[1:5, "label"] = 1
    frame.loc[65:70, "label"] = 1
    frame.loc[86:95, "label"] = 1
    cfg = {**q.t.config(), "purge_days": 1 / 144, "inner_days": 20 / 144,
           "folds": [{"id": "toy", "start": frame.time.iloc[68], "end": frame.time.iloc[90]}]}
    independent = q.independent_splits(frame, cfg)
    actual, _ = q.t.split_masks(frame, cfg)
    for key in independent:
        for left, right in zip(independent[key], actual[key], strict=True):
            np.testing.assert_array_equal(left, right)
    assert not independent[("toy", "outer")][1][65:71].any()
    assert independent[("toy", "outer")][1][86:96].all()


def test_schema_binary_missing_and_duplicate_rejected():
    f = pd.DataFrame({"station": ["S", "S"], "year": [2025, 2025], "layer": [1, 1],
        "time": ["a", "b"], "row_id": [0, 1], "label": [0, 1], "fold": ["q", "q"],
        "probability_O": [.1, .9], "probability_B": [.2, .8],
        "O": [0, 1], "B": [0, 1], "union": [0, 1], "router": [0, 1]})
    q.schema(f)
    with pytest.raises(ValueError):
        q.schema(f.drop(columns="B"))
    with pytest.raises(ValueError):
        q.schema(pd.concat([f, f.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError):
        q.schema(f.assign(probability_O=np.nan))


def test_terminal_gate_before_source_access(tmp_path, monkeypatch):
    q.t.core.write_json(tmp_path / "terminal_result.json", {"status": "RUNNING"})
    monkeypatch.delenv("P1_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="complete historical"):
        q.verify(tmp_path, tmp_path / "absent-seal.json")


def test_windows_fit_receipt_path_normalization():
    assert q.normalized_fit_path(r"2025_q2\inner\O_20260813.joblib") == "2025_q2/inner/O_20260813.joblib"
    assert q.normalized_fit_path("2025_q2/inner/O_20260813.joblib") == "2025_q2/inner/O_20260813.joblib"
    for path in ["../model.joblib", r"..\model.joblib", "C:/model.joblib", "/model.joblib"]:
        with pytest.raises(ValueError):
            q.normalized_fit_path(path)


def test_no_fit_or_official_entrypoint():
    source = Path(q.__file__).read_text()
    assert ".fit(" not in source and "test.csv" not in source and "sample_submission.csv" not in source
    assert "to_csv(" not in source
