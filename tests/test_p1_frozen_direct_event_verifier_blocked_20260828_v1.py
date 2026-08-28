from __future__ import annotations

import numpy as np
import pandas as pd

from p1_qc.frozen_direct_event_verifier import (
    EventProposal,
    assign_split,
    build_event_proposals,
    chronological_boundaries,
    decode_additions,
    evaluate_union,
    split_intervals,
    utility_targets,
)


def _proposal(rows: np.ndarray, start: str, end: str) -> EventProposal:
    return EventProposal("p", "A", 2025, 1, int(rows[0]), int(rows[-1]) + 1, pd.Timestamp(start), pd.Timestamp(end), rows, np.ones(3))


def test_time_split_has_declared_purge_and_full_containment() -> None:
    times = pd.date_range("2025-01-01", periods=100, freq="1D", tz="UTC")
    b1, b2 = chronological_boundaries(times)
    intervals = split_intervals(b1, b2, 15)
    assert intervals["train"][1] == b1 - pd.Timedelta(days=15)
    assert intervals["calibration"][1] == b2 - pd.Timedelta(days=15)
    assert (
        assign_split(
            _proposal(
                np.arange(19),
                "2025-01-02T00:00:00+00:00",
                "2025-01-03T00:00:00+00:00",
            ),
            intervals,
        )
        == "train"
    )


def test_utility_target_matches_anchor_f1_half_condition() -> None:
    truth = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int8)
    anchor = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.int8)
    proposals = [_proposal(np.asarray([1]), "2025-01-01", "2025-01-01"), _proposal(np.asarray([2, 3]), "2025-01-02", "2025-01-02")]
    targets, diagnostics = utility_targets(proposals, truth, anchor, np.arange(len(truth)))
    assert targets.tolist() == [1, 0]
    assert diagnostics[0]["utility"] > 0


def test_decode_union_never_removes_anchor_positive() -> None:
    truth = np.asarray([1, 1, 0, 0], dtype=np.int8)
    anchor = np.asarray([1, 0, 0, 1], dtype=np.int8)
    proposals = [_proposal(np.asarray([1]), "2025-01-01", "2025-01-01")]
    additions = decode_additions(4, proposals, np.asarray([True]))
    metrics = evaluate_union(truth, anchor, additions, np.arange(4))
    assert metrics["anchor_positive_removed_rows"] == 0
    assert metrics["added_tp"] == 1


def test_component_minimum_is_exact_and_ids_are_not_features() -> None:
    rows = 19
    keys = pd.DataFrame(
        {
            "station": ["A"] * rows,
            "year": [2025] * rows,
            "layer": [1] * rows,
            "time": pd.date_range(
                "2025-01-01T00:00:00+00:00", periods=rows, freq="10min"
            ).astype(str),
        }
    )
    numeric = np.arange(rows, dtype=np.float32).reshape(-1, 1)
    proposals, feature_names = build_event_proposals(
        keys,
        numeric,
        ("signal",),
        np.ones(rows),
        np.ones(rows, dtype=np.int8),
        np.zeros(rows, dtype=np.int8),
        selected_numeric=("signal",),
        minimum_rows=19,
        context_rows=2,
    )
    assert len(proposals) == 1
    assert proposals[0].row_ids.tolist() == list(range(rows))
    assert not any("station" in name or "layer" in name for name in feature_names)
