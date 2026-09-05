"""Synthetic masks only: no source dataset, models, fits, or official I/O."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_p1_twosided_support_20260906_v1 as audit  # noqa: E402


def test_two_sided_purge_keeps_validation_and_entire_runs():
    contract = audit.evaluation.load_contract()
    fold = next(x for x in contract["P1"]["folds"] if x["id"] == "H1_2025")
    # Two crossing runs at exact left and right exclusion boundaries.
    times = ["2024-11-01T00:00:00+09:00", "2024-12-10T23:50:00+09:00",
             "2024-12-11T00:00:00+09:00", "2025-01-01T00:00:00+09:00",
             "2025-07-21T23:50:00+09:00", "2025-07-22T00:00:00+09:00",
             "2025-08-01T00:00:00+09:00"]
    frame = pd.DataFrame({"station": "A", "layer": 1, "time": times,
                          "label": [0, 1, 1, 0, 1, 1, 0]})
    train, val, _, _, _ = audit.split_masks(frame, fold, contract)
    assert np.flatnonzero(train).tolist() == [0, 6]
    assert np.flatnonzero(val).tolist() == [3]


def test_whole_validation_run_owned_by_start_and_unseen_not_removed():
    contract = audit.evaluation.load_contract()
    fold = next(x for x in contract["P1"]["folds"] if x["id"] == "H1_2025")
    frame = pd.DataFrame({"station": ["A", "B", "B", "A"], "layer": 1,
                          "time": ["2024-11-01T00:00:00+09:00", "2025-06-30T23:50:00+09:00",
                                   "2025-07-01T00:00:00+09:00", "2025-08-01T00:00:00+09:00"],
                          "label": [0, 1, 1, 0]})
    train, val, _, _, _ = audit.split_masks(frame, fold, contract)
    assert np.flatnonzero(train).tolist() == [0, 3]
    assert np.flatnonzero(val).tolist() == [1, 2]
    assert frame.loc[val, "station"].eq("B").all()
