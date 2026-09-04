from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p1_qc.ms_tcn_asrf_data import (
    ANOMALY_TYPES,
    RobustRowEncoder,
    SegmentLayout,
    assert_disjoint_and_time_purged,
    build_asrf_targets,
    build_window_index,
    materialize_windows,
    select_training_windows,
    stitch_center_weighted,
)


def _times(count: int, *, start: str = "2024-01-01 00:00+09:00") -> np.ndarray:
    return np.asarray(pd.date_range(start, periods=count, freq="10min").astype(str), dtype=object)


def test_exact_segments_split_on_gap_and_keep_original_row_ids() -> None:
    station = np.asarray(["A", "A", "A", "A", "A", "B", "B"])
    year = np.asarray([2024] * 7)
    layer = np.asarray([1] * 7)
    time = np.asarray(
        [
            "2024-01-01 00:20+09:00",
            "2024-01-01 00:00+09:00",
            "2024-01-01 00:10+09:00",
            "2024-01-01 00:40+09:00",
            "2024-01-01 00:50+09:00",
            "2024-01-01 00:10+09:00",
            "2024-01-01 00:00+09:00",
        ],
        dtype=object,
    )

    layout = SegmentLayout.from_aligned(station, year, layer, time)

    assert len(layout.segments) == 3
    assert layout.segments[0].row_ids.tolist() == [1, 2, 0]
    assert layout.segments[1].row_ids.tolist() == [3, 4]
    assert layout.segments[2].row_ids.tolist() == [6, 5]
    assert layout.gap_by_row.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert np.all(np.diff(layout.segments[0].time_ns) == pd.Timedelta(minutes=10).value)


def test_default_windows_are_2048_right_padded_and_selection_keeps_all_events() -> None:
    count = 3000
    layout = SegmentLayout.from_aligned(
        np.asarray(["A"] * count),
        np.asarray([2024] * count),
        np.asarray([1] * count),
        _times(count),
    )
    windows = build_window_index(layout)
    assert [window.start for window in windows] == [0, 512, 1024, 1536, 2048, 2560]
    assert all(window.window_size == 2048 for window in windows)
    assert windows[-1].valid_length == 440

    labels = np.zeros(count, dtype=np.int8)
    labels[[10, 2500]] = 1
    selected_a = select_training_windows(
        windows, labels, negative_ratio=1.0, min_negative_windows=1, seed=17
    )
    selected_b = select_training_windows(
        windows, labels, negative_ratio=1.0, min_negative_windows=1, seed=17
    )
    positive_windows = [window for window in windows if labels[window.row_ids].any()]
    assert [(item.segment_id, item.start) for item in selected_a] == [
        (item.segment_id, item.start) for item in selected_b
    ]
    assert all(item in selected_a for item in positive_windows)

    gathered, valid = materialize_windows(
        np.arange(count, dtype=np.float32)[:, None], (windows[-1],)
    )
    assert gathered.shape == (1, 2048, 1)
    assert valid[0, :440].sum() == 440
    assert valid[0, 440:].sum() == 0
    assert np.all(gathered[0, 440:] == 0.0)


def test_encoder_fits_scaler_and_vocabulary_on_training_ids_only() -> None:
    numeric = np.asarray([[0.0], [1.0], [2.0], [3.0], [1_000_000.0], [np.nan]])
    station = np.asarray(["A", "A", "A", "A", "LEAK", "LEAK"])
    layer = np.asarray([1, 1, 1, 1, 99, 99])
    depth_regime = np.asarray(["known", "known", "known", "known", "LEAK", "LEAK"])
    train_ids = np.asarray([0, 1, 2, 3])
    validation_ids = np.asarray([4, 5])

    encoder = RobustRowEncoder.fit(
        numeric,
        station,
        layer,
        train_ids,
        depth_regime=depth_regime,
        forbidden_row_ids=validation_ids,
        numeric_names=("temp",),
    )
    encoded = encoder.transform(
        numeric,
        station,
        layer,
        depth_regime=depth_regime,
        gap=np.asarray([0, 0, 0, 0, 1, 0]),
        one_hot_categories=True,
    )

    assert encoder.center.tolist() == [1.5]
    assert encoder.scale.tolist() == [1.5]
    assert encoder.station_vocab == ("A",)
    assert encoder.layer_vocab == ("1",)
    assert encoder.fit_ids_sha256
    assert encoded.station_code[4:].tolist() == [0, 0]
    assert encoded.layer_code[4:].tolist() == [0, 0]
    assert encoded.depth_regime_code[4:].tolist() == [0, 0]
    assert encoded.dense[5, 0] == 0.0
    assert encoded.dense[5, 1] == 1.0
    assert encoded.dense[4, 3] == 1.0  # explicit gap channel
    assert encoded.dense.shape[1] == 7
    assert np.all(encoded.dense[4:, 4:] == 0.0)  # unseen categories are all-zero one-hot
    assert np.all(encoded.dense[:4, 4:] == 1.0)
    assert np.isfinite(encoded.dense).all()


def test_targets_have_sigma_three_boundaries_and_plus_delimited_types() -> None:
    count = 7
    layout = SegmentLayout.from_aligned(
        np.asarray(["A"] * count),
        np.asarray([2024] * count),
        np.asarray([1] * count),
        _times(count),
    )
    labels = np.asarray([0, 1, 1, 1, 0, 1, 0], dtype=np.int8)
    anomaly_type = np.asarray(
        ["", "noise", "noise+offset", "offset", "", "spike", ""], dtype=object
    )

    targets = build_asrf_targets(labels, anomaly_type, layout)

    assert targets.anomaly_type_names == ANOMALY_TYPES
    assert targets.start_boundary[1] == pytest.approx(1.0)
    assert targets.end_boundary[3] == pytest.approx(1.0)
    assert targets.start_boundary[5] == pytest.approx(1.0)
    assert targets.end_boundary[5] == pytest.approx(1.0)
    assert targets.start_boundary[0] == pytest.approx(np.exp(-0.5 / 9.0))
    noise = ANOMALY_TYPES.index("noise")
    offset = ANOMALY_TYPES.index("offset")
    assert targets.anomaly_type[2, noise] == 1.0
    assert targets.anomaly_type[2, offset] == 1.0
    assert targets.anomaly_type[2].sum() == 2.0
    assert targets.anomaly_type[0].sum() == 0.0


def test_center_weighted_overlap_add_restores_original_row_order() -> None:
    count = 6
    layout = SegmentLayout.from_aligned(
        np.asarray(["A"] * count),
        np.asarray([2024] * count),
        np.asarray([1] * count),
        _times(count),
    )
    windows = build_window_index(layout, window_size=4, stride=2)
    predictions = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0],
            [20.0, 20.0, 999.0, 999.0],
        ],
        dtype=np.float32,
    )

    stitched = stitch_center_weighted(
        predictions, windows, n_rows=count, require_row_ids=np.arange(count)
    )

    assert stitched.tolist() == pytest.approx(
        [0.0, 0.0, 10.0 / 3.0, 20.0 / 3.0, 40.0 / 3.0, 50.0 / 3.0]
    )


def test_disjoint_and_time_purge_assertions_fail_closed() -> None:
    station = np.asarray(["A", "A", "A", "B"])
    year = np.asarray([2024] * 4)
    layer = np.asarray([1] * 4)
    time = np.asarray(
        [
            "2024-01-01 00:00+09:00",
            "2024-01-01 00:50+09:00",
            "2024-01-01 03:00+09:00",
            "2024-01-01 00:10+09:00",
        ]
    )

    with pytest.raises(AssertionError, match="overlap"):
        assert_disjoint_and_time_purged([0, 1], [1, 2], station, year, layer, time, purge="30min")
    with pytest.raises(AssertionError, match="time purge failed"):
        assert_disjoint_and_time_purged([0], [1], station, year, layer, time, purge="1h")
    assert_disjoint_and_time_purged([0, 3], [1, 2], station, year, layer, time, purge="30min")
