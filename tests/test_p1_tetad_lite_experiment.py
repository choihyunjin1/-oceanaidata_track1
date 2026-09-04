from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from p1_qc.tetad_lite_experiment import (
    CANONICAL_SUFFIXES,
    ROUND_B_PARTS,
    CadenceSegment,
    FrozenParquet,
    FrozenTrainingInputs,
    TargetEvent,
    WindowSpec,
    anchor_preserving_union,
    assert_split_dependency_safe,
    build_windows,
    deterministic_training_sample,
    eligible_target_events,
    load_frozen_anchor_surface,
    load_training_prefix_bundle,
    load_validation_feature_bundle,
    load_validation_membership,
    station_layer_cluster_bootstrap_ci90,
    stitch_proposals,
    train_tetad,
)
from p1_qc.tetad_lite_tinygrad import TETADLiteConfig, TETADLiteTinygrad


def _segment(length: int = 40) -> CadenceSegment:
    start = datetime(2024, 1, 1)
    return CadenceSegment(
        segment_id=7,
        station="S",
        year=2024,
        layer=1,
        row_indices=np.arange(length, dtype=np.int64),
        times=tuple(start + index * timedelta(minutes=10) for index in range(length)),
    )


def test_query_overflow_fails_closed() -> None:
    events = [TargetEvent(7, "offset", start, start + 2) for start in range(0, 12, 2)]
    with pytest.raises(RuntimeError, match="exceeds query budget"):
        build_windows([_segment()], events, window_length=40, stride=20, max_queries=5)


def test_split_purge_must_cover_window_and_feature_dependency() -> None:
    validation = datetime(2025, 4, 1)
    with pytest.raises(RuntimeError, match="below dependency span"):
        assert_split_dependency_safe(
            validation - timedelta(days=7),
            validation,
            purge=timedelta(days=7),
            window_length=1024,
            maximum_feature_lookahead=timedelta(hours=84),
        )
    assert_split_dependency_safe(
        validation - timedelta(days=14),
        validation,
        purge=timedelta(days=14),
        window_length=1024,
        maximum_feature_lookahead=timedelta(hours=84),
    )


def test_window_target_is_clipped_half_open_fragment() -> None:
    event = TargetEvent(7, "drift", 6, 28)
    windows = build_windows([_segment(40)], [event], window_length=16, stride=16)
    np.testing.assert_allclose(windows[0].targets, [[6 / 16, 1.0]])
    np.testing.assert_allclose(windows[1].targets, [[0.0, 12 / 16]])
    # The exact deduplicated final start is 40-16=24 and retains [24,28).
    assert windows[2].start == 24
    np.testing.assert_allclose(windows[2].targets, [[0.0, 4 / 16]])


def test_union_tokens_form_one_event_and_right_censor_is_excluded() -> None:
    segment = _segment(25)
    labels = pl.DataFrame(
        {
            "label": [0] * 5 + [1] * 20,
            "anomaly_type": [""] * 5
            + ["offset+spike"] * 5
            + ["drift"] * 5
            + ["noise+flatline"] * 10,
        }
    )
    events = eligible_target_events([segment], labels)
    assert [(event.start, event.end, event.anomaly_type) for event in events] == [
        (5, 25, "binary_union")
    ]
    censored = eligible_target_events(
        [segment], labels, right_censor_cutoff=segment.times[-1] + timedelta(minutes=10)
    )
    assert censored == []


def test_window_starts_and_short_coordinates_are_fixed_1024() -> None:
    long_windows = build_windows([_segment(2500)], [], window_length=1024, stride=512)
    assert [window.start for window in long_windows] == [0, 512, 1024, 1476]
    short = _segment(100)
    short_windows = build_windows(
        [short], [TargetEvent(7, "binary_union", 0, 20)], window_length=1024, stride=512
    )
    assert len(short_windows) == 1
    np.testing.assert_allclose(short_windows[0].targets, [[0.0, 20 / 1024]])


def test_sampler_is_deterministic_and_keeps_all_positive_windows() -> None:
    windows = [
        WindowSpec(0, index, 8, np.arange(8), np.asarray([[0.1, 0.5]], dtype=np.float32) if index in {1, 8} else np.empty((0, 2), dtype=np.float32))
        for index in range(12)
    ]
    first = deterministic_training_sample(windows, seed=19)
    second = deterministic_training_sample(list(reversed(windows)), seed=19)
    assert [window.identity for window in first] == [window.identity for window in second]
    assert sum(bool(len(window.targets)) for window in first) == 2
    assert len(first) == 6


def test_stitching_uses_valid_length_full_length_gate_and_row_max() -> None:
    windows = [
        WindowSpec(0, 0, 30, np.arange(30), np.empty((0, 2), dtype=np.float32)),
        WindowSpec(0, 10, 30, np.arange(10, 40), np.empty((0, 2), dtype=np.float32)),
    ]
    intervals = np.asarray(
        [
            [[0.0, 0.65], [0.8, 1.0]],
            [[0.0, 0.65], [0.0, 0.1]],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([[0.6, 0.99], [0.8, 0.95]], dtype=np.float32)
    confidence, prediction = stitch_proposals(
        windows,
        intervals,
        scores,
        total_rows=40,
        threshold=0.7,
        minimum_decoded_rows=19,
        coordinate_length=30,
    )
    # Six-row and three-row high-score proposals are rejected before clipping.
    assert np.all(confidence[:10] == pytest.approx(0.6))
    assert np.all(confidence[10:20] == pytest.approx(0.8))
    assert np.all(confidence[20:30] == pytest.approx(0.8))
    assert np.all(confidence[30:] == 0)
    np.testing.assert_array_equal(prediction, np.r_[np.zeros(10), np.ones(20), np.zeros(10)])


def test_short_window_decodes_in_fixed_1024_coordinates_before_clipping() -> None:
    window = WindowSpec(
        0, 0, 100, np.arange(100), np.empty((0, 2), dtype=np.float32)
    )
    intervals = np.asarray([[[90 / 1024, 120 / 1024]]], dtype=np.float32)
    scores = np.asarray([[0.9]], dtype=np.float32)
    confidence, prediction = stitch_proposals(
        [window], intervals, scores, total_rows=100, threshold=0.5
    )
    assert np.all(confidence[:90] == 0)
    assert np.all(confidence[90:] == pytest.approx(0.9))
    np.testing.assert_array_equal(prediction, np.r_[np.zeros(90), np.ones(10)])


def test_anchor_union_never_removes_anchor_positive() -> None:
    anchor = np.asarray([1, 0, 1, 0], dtype=np.int8)
    proposal = np.asarray([0, 1, 1, 0], dtype=np.int8)
    candidate = anchor_preserving_union(anchor, proposal)
    np.testing.assert_array_equal(candidate, [1, 1, 1, 0])
    assert not np.any((anchor == 1) & (candidate == 0))


def test_station_layer_bootstrap_is_deterministic() -> None:
    truth = np.asarray([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8)
    anchor = np.asarray([1, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)
    candidate = anchor_preserving_union(anchor, [0, 0, 1, 0, 1, 1, 0, 0])
    stations = np.asarray(["A"] * 4 + ["B"] * 4)
    layers = np.asarray([1, 1, 2, 2, 1, 1, 2, 2])
    first = station_layer_cluster_bootstrap_ci90(
        truth, anchor, candidate, stations, layers, replicates=100, seed=20260826
    )
    second = station_layer_cluster_bootstrap_ci90(
        truth, anchor, candidate, stations, layers, replicates=100, seed=20260826
    )
    assert first == second
    assert first["observed_delta"] > 0


def test_real_one_step_tinygrad_training() -> None:
    model = TETADLiteTinygrad(
        TETADLiteConfig(
            input_features=2,
            patch_size=4,
            d_model=8,
            num_heads=2,
            ff_multiplier=2,
            num_queries=5,
            max_patches=4,
        )
    )
    result = train_tetad(
        model,
        np.zeros((1, 8, 2), dtype=np.float32),
        [np.asarray([[0.25, 0.75]], dtype=np.float32)],
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        positive_class_weight=1.0,
        seed=11,
    )
    assert len(result.epoch_losses) == 1
    assert np.isfinite(result.epoch_losses[0])


def _frozen(path, suffix: str, rows: int, *, fold: str | None = None) -> FrozenParquet:
    import hashlib

    return FrozenParquet(
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows=rows,
        canonical_suffix=suffix,
        fold=fold,
    )


def test_label_firewall_projection_interfaces(tmp_path) -> None:
    times = [f"2024-01-01T00:{minute:02d}:00+09:00" for minute in (0, 10, 20, 30)]
    keys = pl.DataFrame(
        {
            "ordinal": range(4),
            "station": ["S"] * 4,
            "year": [2024] * 4,
            "layer": [1] * 4,
            "time": times,
        }
    )
    features = pl.DataFrame(
        {"station": ["S"] * 4, "layer_category": ["1"] * 4, "f": [1.0, 2.0, 3.0, 4.0]}
    )
    labels = keys.drop("ordinal").with_columns(
        pl.Series("label", [0, 1, 1, 0]),
        pl.Series("anomaly_type", ["", "offset", "drift", ""]),
    )
    truth = labels.slice(2).with_columns(pl.Series("fold", ["2025_q2", "2025_q2"]))
    part = truth.select(["station", "year", "layer", "time", "fold"]).with_columns(
        pl.Series("row_position", [2, 3]),
        pl.Series("fraction", [1.0, 1.0]),
        pl.Series("event_day_balanced_binary_lgbm__prediction", [1, 0]),
    )

    paths = {}
    for name, suffix, frame in (
        ("feature", CANONICAL_SUFFIXES["feature_cache"], features),
        ("keys", CANONICAL_SUFFIXES["key_sidecar"], keys),
        ("labels", CANONICAL_SUFFIXES["label_cache"], labels),
        ("truth", CANONICAL_SUFFIXES["truth_oof"], truth),
        ("part", ROUND_B_PARTS["2025_q2"][0], part),
    ):
        path = tmp_path / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        paths[name] = path

    inputs = FrozenTrainingInputs(
        _frozen(paths["feature"], CANONICAL_SUFFIXES["feature_cache"], 4),
        _frozen(paths["keys"], CANONICAL_SUFFIXES["key_sidecar"], 4),
        _frozen(paths["labels"], CANONICAL_SUFFIXES["label_cache"], 4),
    )
    truth_spec = _frozen(paths["truth"], CANONICAL_SUFFIXES["truth_oof"], 2)
    part_spec = _frozen(
        paths["part"], ROUND_B_PARTS["2025_q2"][0], 2, fold="2025_q2"
    )
    prefix = load_training_prefix_bundle(
        inputs, ["f"], cutoff="2024-01-01T00:20:00+09:00"
    )
    assert prefix.rows == 2
    assert prefix.labels.height == 2
    membership = load_validation_membership(truth_spec)
    assert "label" not in membership.columns and "anomaly_type" not in membership.columns
    anchor = load_frozen_anchor_surface(truth_spec, [part_spec])
    assert "label" not in anchor.columns and "anomaly_type" not in anchor.columns
    validation = load_validation_feature_bundle(
        inputs, truth_spec, ["f"], fold="2025_q2"
    )
    assert validation.rows == 2
    assert not hasattr(validation, "labels")
    np.testing.assert_array_equal(validation.features.get_column("f"), [3.0, 4.0])
