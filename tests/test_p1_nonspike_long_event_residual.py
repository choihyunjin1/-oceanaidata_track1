from __future__ import annotations

import numpy as np
import pandas as pd

from p1_qc.nonspike_long_event_residual import (
    binary_metrics,
    build_residual_training_view,
    connected_rescue,
    long_nonspike_event_target,
)


def _metadata(rows: int, *, station: str = "S-ORS", layer: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": [station] * rows,
            "layer": [layer] * rows,
            "time": pd.date_range(
                "2024-01-01T00:00:00+09:00", periods=rows, freq="10min"
            ).astype(str),
        }
    )


def test_long_nonspike_target_excludes_spike_and_right_censored_events() -> None:
    first = _metadata(21, station="S-ORS")
    second = _metadata(21, station="I-ORS")
    third = _metadata(19, station="G-ORS")
    metadata = pd.concat([first, second, third], ignore_index=True)
    truth = np.concatenate(
        [
            np.r_[np.ones(19, dtype=np.int8), np.zeros(2, dtype=np.int8)],
            np.r_[np.ones(19, dtype=np.int8), np.zeros(2, dtype=np.int8)],
            np.ones(19, dtype=np.int8),
        ]
    )
    anomaly_type = np.full(len(metadata), "", dtype=object)
    anomaly_type[:19] = "offset"
    anomaly_type[21:40] = "offset+spike"
    anomaly_type[42:] = "drift"

    target, audit = long_nonspike_event_target(truth, anomaly_type, metadata)

    assert target.sum() == 19
    assert target[:19].all()
    assert not target[19:].any()
    assert audit == {
        "positive_event_count": 1,
        "positive_row_count": 19,
        "excluded_positive_rows": 38,
        "right_censored_event_count": 1,
    }


def test_training_view_uses_only_normal_and_selected_positive_rows() -> None:
    metadata = _metadata(24)
    truth = np.r_[np.ones(19, dtype=np.int8), np.zeros(5, dtype=np.int8)]
    anomaly_type = np.r_[np.full(19, "offset", dtype=object), np.full(5, "", dtype=object)]

    view = build_residual_training_view(truth, anomaly_type, metadata)

    assert len(view.indices) == 24
    assert int(view.target.sum()) == 19
    assert len(view.sample_weight) == 24
    assert np.isfinite(view.sample_weight).all()
    assert (view.sample_weight > 0).all()


def test_connected_rescue_requires_an_unbroken_path_to_base_anchor() -> None:
    metadata = _metadata(8)
    base = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    probability = np.array([0.99, 0.0, 0.95, 0.9, 0.1, 0.99, 0.99, 0.0])
    spike = np.array([False, False, False, False, False, False, True, False])

    candidate, rescue = connected_rescue(
        base,
        probability,
        metadata,
        spike,
        threshold=0.8,
        max_distance_rows=6,
    )

    assert candidate.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert rescue.tolist() == [True, False, True, True, False, False, False, False]
    assert np.all(candidate >= base)


def test_connected_rescue_does_not_cross_a_physical_gap() -> None:
    metadata = _metadata(4)
    metadata.loc[2:, "time"] = pd.date_range(
        "2024-01-01T02:00:00+09:00", periods=2, freq="10min"
    ).astype(str)
    base = np.array([0, 1, 0, 0], dtype=np.int8)
    probability = np.array([0.9, 0.0, 0.99, 0.99])

    candidate, rescue = connected_rescue(
        base,
        probability,
        metadata,
        np.zeros(4, dtype=bool),
        threshold=0.8,
        max_distance_rows=18,
    )

    assert candidate.tolist() == [1, 1, 0, 0]
    assert rescue.tolist() == [True, False, False, False]


def test_binary_metrics_reconstruct_confusion_counts() -> None:
    metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
