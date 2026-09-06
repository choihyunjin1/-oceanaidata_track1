import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/p1_champion_reconstruction_20260906_v1/composition.py"
spec = importlib.util.spec_from_file_location("champion_composition", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def frames():
    tree = pd.DataFrame({"station": ["S-ORS"] * 3, "year": [2025] * 3,
                         "layer": [1] * 3, "time": pd.date_range("2025-04-01", periods=3,
                         freq="10min", tz="Asia/Seoul"), "tree": [1, 0, 0]})
    proposal = tree.drop(columns="tree").assign(proposal=[0, 1, 0])
    return tree, proposal


def test_union_preserves_order_and_positive_rows():
    tree, proposal = frames()
    result, receipt = module.combine(tree, proposal.iloc[::-1])
    assert result.label.tolist() == [1, 1, 0]
    assert receipt["tree_positive_removed_rows"] == 0
    assert receipt["added_rows"] == 1
    assert receipt["proposal_reordered"]
    pd.testing.assert_frame_equal(tree[list(module.KEYS)], result[list(module.KEYS)])


@pytest.mark.parametrize("failure", ["duplicate", "missing", "extra", "other_key", "nonbinary", "nan"])
def test_invalid_proposal_rejected(failure):
    tree, proposal = frames()
    if failure == "duplicate":
        proposal = pd.concat([proposal.iloc[:1], proposal.iloc[:1], proposal.iloc[2:]])
    elif failure == "missing":
        proposal = proposal.iloc[:2]
    elif failure == "extra":
        proposal = pd.concat([proposal, proposal.iloc[:1].assign(layer=2)])
    elif failure == "other_key":
        proposal.loc[0, "layer"] = 2
    else:
        proposal.loc[0, "proposal"] = 2 if failure == "nonbinary" else np.nan
    with pytest.raises(ValueError):
        module.combine(tree, proposal)


def test_timezone_representation_and_naive_kst_equivalence():
    tree, proposal = frames()
    proposal["time"] = proposal.time.dt.tz_convert("UTC")
    assert module.ordered_key_sha256(tree) == module.ordered_key_sha256(proposal)
    proposal["time"] = proposal.time.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    assert module.ordered_key_sha256(tree) == module.ordered_key_sha256(proposal)


def test_truth_and_historical_score_columns_ignored():
    tree, proposal = frames()
    expected, _ = module.combine(tree, proposal)
    tree["label"] = [0, 1, 1]
    tree["official_score"] = [28.9] * 3
    actual, _ = module.combine(tree, proposal)
    pd.testing.assert_frame_equal(actual, expected)


def test_fractional_key_rejected():
    tree, proposal = frames()
    proposal["layer"] = 1.5
    with pytest.raises(ValueError):
        module.combine(tree, proposal)
