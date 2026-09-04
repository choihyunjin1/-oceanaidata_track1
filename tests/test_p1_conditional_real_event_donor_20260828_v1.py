from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from p1_qc.p1_conditional_real_event_donor_20260828_v1 import (
    conditional_transplant,
    decode_scores,
    evaluate_anchor_union,
    event_support,
    extract_long_events,
    extract_mask_events,
    proposal_support_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def keys(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["S-ORS"] * rows,
            "year": [2024] * rows,
            "layer": [2] * rows,
            "time": pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC"),
        }
    )


def test_real_event_support_counts_long_eligible_runs() -> None:
    frame = keys()
    labels = np.zeros(len(frame), dtype=np.int8)
    labels[10:35] = 1
    anomaly = np.full(len(frame), "", dtype=object)
    anomaly[10:35] = "offset"
    events = extract_long_events(
        frame,
        labels,
        anomaly,
        np.arange(len(frame)),
        eligible_types=("offset", "drift", "noise"),
        minimum_rows=19,
    )
    assert len(events) == 1
    assert len(events[0].rows) == 25
    assert event_support(events)["station_layer_cells"] == 1


def test_conditional_transplant_preserves_registered_recipient_features() -> None:
    donor = np.full((4, 5), 7.0, dtype=np.float32)
    recipient = np.full((4, 5), 2.0, dtype=np.float32)
    output = conditional_transplant(donor, recipient, replace_indices=(0, 2, 4))
    assert np.all(output[:, [0, 2, 4]] == 7.0)
    assert np.all(output[:, [1, 3]] == 2.0)


def test_decoder_and_proposal_matching_are_event_level() -> None:
    frame = keys()
    scores = np.zeros(len(frame), dtype=np.float64)
    scores[10:20] = 0.9
    scores[22:35] = 0.9
    decoded = decode_scores(
        frame,
        scores,
        threshold=0.5,
        smoothing_rows=1,
        minimum_rows=19,
        bridge_rows=2,
    )
    labels = np.zeros(len(frame), dtype=np.int8)
    labels[10:35] = 1
    anomaly = np.full(len(frame), "", dtype=object)
    anomaly[10:35] = "noise"
    truth = extract_long_events(
        frame,
        labels,
        anomaly,
        np.arange(len(frame)),
        eligible_types=("noise",),
        minimum_rows=19,
    )
    proposals = extract_mask_events(frame, decoded, minimum_rows=19)
    metrics = proposal_support_metrics(truth, proposals, labels, iou_threshold=0.3)
    assert metrics["matched_proposals"] == 1
    assert metrics["real_event_recall"] == 1.0


def test_anchor_union_never_deletes_anchor_rows() -> None:
    metrics = evaluate_anchor_union([1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0])
    assert metrics["delta_f1"] > 0
    assert metrics["anchor_positive_removed_rows"] == 0


def test_config_seals_single_fit_without_ids_or_tuning() -> None:
    path = (
        ROOT
        / "configs"
        / "experiments"
        / "p1_conditional_real_event_donor_20260828_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["model"]["fit_count"] == 1
    assert config["model"]["threshold_search"] is False
    assert config["model"]["grid_search"] is False
    assert config["features"]["station_and_layer_ids_as_model_features"] is False
    assert config["split"]["boundary_purge_days"] == 15
