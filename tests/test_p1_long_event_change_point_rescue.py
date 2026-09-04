from __future__ import annotations

import numpy as np
import pandas as pd

from p1_qc.long_event_change_point_rescue import (
    TARGET_CELLS,
    additions_from_scores,
    anchor_preserving_union,
    build_past_only_row_features,
    generate_proposals,
    select_threshold_arm,
)


def _frame(rows: int = 1200) -> pd.DataFrame:
    time = pd.date_range("2025-01-01", periods=rows, freq="10min", tz="Asia/Seoul")
    records = []
    for station, layers in (("G-ORS", [1]), ("I-ORS", [1, 2]), ("S-ORS", [1, 2, 3])):
        for layer in layers:
            temp = 10 + 0.2 * np.sin(np.arange(rows) / 100) + layer * 0.05
            if (station, layer) in TARGET_CELLS:
                temp = temp.copy()
                temp[800:1100] += np.linspace(0.0, 3.0, 300)
            records.append(
                pd.DataFrame(
                    {
                        "station": station,
                        "year": 2025,
                        "layer": layer,
                        "time": time.astype(str),
                        "temp_raw": temp,
                        "psal_raw": 33.0,
                        "depth_raw": float(layer),
                        "anchor_probability": 0.01,
                        "anchor": 0,
                    }
                )
            )
    return pd.concat(records, ignore_index=True)


def test_past_only_prefix_is_invariant_to_appended_future() -> None:
    frame = _frame()
    prefix_cut = pd.Timestamp("2025-01-07", tz="Asia/Seoul")
    prefix_input = frame.loc[pd.to_datetime(frame["time"], utc=True) < prefix_cut.tz_convert("UTC")]
    prefix = build_past_only_row_features(prefix_input)
    full = build_past_only_row_features(frame)
    full_prefix = full.loc[full["_time"] < prefix_cut.tz_convert("UTC")]
    key_columns = ["station", "layer", "time"]
    value_columns = ["temp_z_72h", "temp_z_168h", "peer_z_72h", "slope_6h", "slope_24h", "physical_score"]
    left = prefix[key_columns + value_columns].sort_values(key_columns).reset_index(drop=True)
    right = full_prefix[key_columns + value_columns].sort_values(key_columns).reset_index(drop=True)
    assert left[key_columns].equals(right[key_columns])
    np.testing.assert_allclose(left[value_columns].to_numpy(), right[value_columns].to_numpy(), equal_nan=True)


def test_anchor_union_never_removes_positive() -> None:
    anchor = np.array([0, 1, 0, 1], dtype=np.int8)
    additions = np.array([1, 0, 0, 1], dtype=np.int8)
    candidate = anchor_preserving_union(anchor, additions)
    assert np.array_equal(candidate, np.array([1, 1, 0, 1], dtype=np.int8))
    assert not np.any((anchor == 1) & (candidate == 0))


def test_proposals_are_scoped_and_zero_add_arm_is_available() -> None:
    frame = _frame()
    features = build_past_only_row_features(frame)
    proposals, _names = generate_proposals(
        features,
        score_thresholds=[1.0],
        minimum_support_rows=[6],
        maximum_gap_rows=12,
        padding_rows=6,
        minimum_interval_rows=19,
        maximum_interval_rows=520,
    )
    assert proposals
    assert {(p.station, p.layer) for p in proposals}.issubset(set(TARGET_CELLS))
    anchor = np.zeros(len(frame), dtype=np.int8)
    scores = np.zeros(len(proposals), dtype=float)
    additions = additions_from_scores(len(frame), proposals, scores, 1.0, anchor)
    assert additions.sum() == 0

    y = np.zeros(len(frame), dtype=np.int8)
    selection, selected_additions = select_threshold_arm(
        frame,
        y,
        anchor,
        proposals,
        scores,
        threshold_candidates=[0.5],
        maximum_added_fp_per_day=0.1,
        minimum_added_precision=0.6,
    )
    assert selection["arm"] == "ZERO_ADD_NO_OP"
    assert selected_additions.sum() == 0
