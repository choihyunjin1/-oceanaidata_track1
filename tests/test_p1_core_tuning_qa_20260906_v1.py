"""Independent tuning QA toy checks; no fits and no data I/O."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p1_tuning_qa", ROOT / "scripts/p1_core_tuning_20260906_v1/qa.py")
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


def rows():
    f = pd.DataFrame({"station": "S-ORS", "year": 2025, "layer": 1,
        "time": pd.date_range("2025-01-01", periods=24, freq="10min", tz="Asia/Seoul").astype(str),
        "row_id": np.arange(24), "label": [1] * 12 + [0] * 12,
        "fold": "toy", "supported": True})
    for c in qa.r.config()["components"]:
        f["probability_" + c] = f.label * .9
    f["control"], f["candidate"] = f.label.copy(), f.label.copy()
    return f


def test_independent_selection_agrees_and_ties_are_frozen(monkeypatch):
    f = rows()
    monkeypatch.setattr(qa.r.core, "decode", lambda frame, p, rules, cfg, threshold: (p >= threshold).astype(np.int8))
    c, b = qa.r.config(), qa.r.base_config()
    actual = qa.independent_selector(f, None, b, c)
    expected = qa.r.select_inner(f, {x: f["probability_" + x].to_numpy() for x in c["components"]}, None, b, c)
    assert actual == expected and actual["selected"] == "control"
    assert set(actual["thresholds"].values()) == {.8}


def test_schema_all_component_probabilities():
    qa.schema(rows(), list(qa.r.config()["components"]))


@pytest.mark.parametrize("change", ["nan", "duplicate", "nonbinary", "missing", "bad_probability"])
def test_schema_rejects(change):
    f = rows()
    if change == "nan":
        f.loc[0, "label"] = np.nan
    elif change == "duplicate":
        f = pd.concat([f, f.iloc[:1]], ignore_index=True)
    elif change == "nonbinary":
        f.loc[0, "candidate"] = 2
    elif change == "missing":
        f = f.drop(columns="probability_O_slow")
    else:
        f.loc[0, "probability_B_control"] = np.inf
    with pytest.raises(ValueError):
        qa.schema(f, list(qa.r.config()["components"]))


def test_independent_confusion_counts():
    c = qa.q.counts([1, 1, 0, 0], [1, 0, 1, 0])
    assert c["tp"] == c["fp"] == c["fn"] == 1 and c["f1"] == .5


def test_paired_day_bootstrap_matches_production():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    b = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    c = np.array([1, 1, 0, 0, 1, 1, 1, 0])
    groups = ["z", "z", "b", "b", "c", "c", "a", "a"]
    actual = qa.q.independent_bootstrap(y, b, c, groups)
    expected = qa.r.t.evaluation.paired_bootstrap(y, b, c, groups, "f1", {"common": {"bootstrap": qa.r.config()["bootstrap"]}})
    np.testing.assert_allclose(actual["ci90"], expected["ci90"], rtol=0, atol=1e-15)
    assert actual["p_improve"] == expected["p_improve"]
    assert actual["delta"] == expected["delta_candidate_minus_control"]
