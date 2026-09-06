"""Independent verifier synthetic regression tests; no real artifact reads."""
import copy
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/p1_champion_reconstruction_20260906_v1" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


E = module("evaluate_union.py", "test_union_evaluate_for_qa")
Q = module("qa_union.py", "test_union_independent_qa")


def synthetic():
    cfg = E.read_json(E.CONTRACT)
    cfg["bootstrap"]["resamples"] = 50
    time = ["2025-07-01T00:00+09:00", "2025-07-09T00:00+09:00", "2025-07-16T00:00+09:00",
            "2025-10-01T00:00+09:00", "2025-10-09T00:00+09:00", "2025-10-16T00:00+09:00"]
    tree = pd.DataFrame({"station": ["S"] * 6, "year": [2025] * 6, "layer": [1] * 6, "time": time,
                         "fold": ["2025_q3"] * 3 + ["2025_q4"] * 3, "label": [1, 0, 1, 0, 1, 0]})
    for arm in E.ARMS:
        tree[arm] = [1, 0, 0, 0, 1, 0]
    proposal = tree[E.KEYS].assign(proposal=[0, 0, 1, 1, 0, 0])
    _, permutation = E.key_coverage(tree, proposal, cfg)
    frame, result = E.evaluate_frames(tree, proposal, permutation, cfg)
    return frame, result, cfg


def test_independent_all_statistics():
    frame, result, cfg = synthetic()
    assert len(Q.verify_statistics(frame, result, cfg)) > 100


@pytest.mark.parametrize("field", ["ci90", "mean_delta_f1", "fraction_delta_positive"])
def test_bootstrap_corruption_rejected(field):
    frame, result, cfg = synthetic()
    broken = copy.deepcopy(result)
    value = broken["scopes"]["primary_Q3_Q4"]["B"]["bootstrap"]
    value[field] = [99, 100] if field == "ci90" else 99
    with pytest.raises(AssertionError, match="bootstrap"):
        Q.verify_statistics(frame, broken, cfg)


def test_risk_corruption_rejected():
    frame, result, cfg = synthetic()
    result["risk"]["router"]["station/layer"]["negative_slice_count"] = 99
    with pytest.raises(AssertionError, match="risk"):
        Q.verify_statistics(frame, result, cfg)


def test_rederived_block_corruption_rejected():
    frame, result, cfg = synthetic()
    frame.loc[0, "block"] += 1
    with pytest.raises(AssertionError, match="7day"):
        Q.verify_statistics(frame, result, cfg)
