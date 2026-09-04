from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p1_qc.temporal_event_tcn import (
    FixedStepTrainingConfig,
    PrefixRobustScaler,
    SequenceLayout,
    TemporalEventModelConfig,
    build_prefix_phase_targets,
    fit_fixed_step_temporal_event_model,
    load_fitted_temporal_event_model,
    predict_temporal_event_probability,
    save_fitted_temporal_event_model,
)


def _metadata(rows: int = 48) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["G-ORS"] * rows,
            "layer": [1] * rows,
            "time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="Asia/Seoul").astype(str),
        }
    )


def test_prefix_scaler_ignores_forbidden_validation_values() -> None:
    values = np.asarray(
        [[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [1000.0, -2000.0]], dtype=np.float32
    )
    train_ids = np.asarray([0, 1, 2])
    validation_ids = np.asarray([3])
    first = PrefixRobustScaler.fit(values, train_ids, forbidden_ids=validation_ids)
    altered = values.copy()
    altered[validation_ids] *= 1_000_000.0
    second = PrefixRobustScaler.fit(altered, train_ids, forbidden_ids=validation_ids)
    np.testing.assert_array_equal(first.center, second.center)
    np.testing.assert_array_equal(first.scale, second.scale)
    assert first.fit_ids_sha256 == second.fit_ids_sha256
    assert first.transform(values).shape == (4, 4)
    with pytest.raises(PermissionError, match="overlap"):
        PrefixRobustScaler.fit(values, train_ids, forbidden_ids=np.asarray([2, 3]))


def test_sequence_layout_exact_centered_window_and_gap_mask() -> None:
    metadata = _metadata(35)
    metadata.loc[20, "time"] = pd.Timestamp("2025-01-01 03:25", tz="Asia/Seoul").isoformat()
    layout = SequenceLayout.build(metadata)
    values = np.arange(35, dtype=np.float32)[:, None]
    windows, groups = layout.windows(values, np.asarray([15]), receptive_field_rows=31)
    assert windows.shape == (1, 1, 31)
    assert groups.tolist() == [0]
    assert windows[0, 0, 15] == 15.0
    assert windows[0, 0, 20] == 0.0


def test_phase_targets_use_only_prefix_labels_and_define_boundaries() -> None:
    metadata = _metadata(20)
    labels = np.full(20, -99, dtype=np.int64)
    train_ids = np.arange(15)
    validation_ids = np.arange(15, 20)
    labels[train_ids] = 0
    labels[5:9] = 1
    ids, phases = build_prefix_phase_targets(
        metadata, labels, train_ids, forbidden_ids=validation_ids
    )
    assert np.array_equal(ids, train_ids)
    assert phases[5:9].tolist() == [1, 2, 2, 3]
    assert np.array_equal(phases > 0, labels[train_ids] > 0)
    with pytest.raises(PermissionError, match="overlap"):
        build_prefix_phase_targets(metadata, labels, train_ids, forbidden_ids=np.asarray([14, 15]))


def test_fixed_step_fit_is_deterministic_and_saved_model_reproduces(tmp_path: Path) -> None:
    metadata = _metadata(48)
    rng = np.random.default_rng(41)
    features = rng.normal(size=(48, 3)).astype(np.float32)
    labels = np.full(48, -9, dtype=np.int64)
    train_ids = np.arange(40)
    validation_ids = np.arange(40, 48)
    labels[train_ids] = 0
    labels[5:10] = 1
    labels[20:24] = 1
    layout = SequenceLayout.build(metadata)
    model_config = TemporalEventModelConfig(
        input_feature_count=3,
        group_count=1,
        width=8,
        group_embedding_width=4,
        norm_groups=2,
        dropout=0.0,
    )
    training_config = FixedStepTrainingConfig(
        optimizer_steps=2,
        batch_size=24,
        learning_rate=7e-4,
        weight_decay=1e-4,
    )
    scaler = PrefixRobustScaler.fit(features, train_ids, forbidden_ids=validation_ids)
    first = fit_fixed_step_temporal_event_model(
        features,
        metadata,
        labels,
        layout,
        train_ids,
        forbidden_ids=validation_ids,
        seed=20260813,
        device="cpu",
        model_config=model_config,
        training_config=training_config,
        scaler=scaler,
    )
    altered_labels = labels.copy()
    altered_labels[validation_ids] = 123456
    second = fit_fixed_step_temporal_event_model(
        features,
        metadata,
        altered_labels,
        layout,
        train_ids,
        forbidden_ids=validation_ids,
        seed=20260813,
        device="cpu",
        model_config=model_config,
        training_config=training_config,
        scaler=scaler,
    )
    assert first.model_state_sha256 == second.model_state_sha256
    prediction = predict_temporal_event_probability(
        first, features, layout, validation_ids, device="cpu", batch_size=4
    )
    model_path = tmp_path / "model.pt"
    save_fitted_temporal_event_model(first, model_path)
    with pytest.raises(FileExistsError):
        save_fitted_temporal_event_model(first, model_path)
    loaded = load_fitted_temporal_event_model(model_path)
    reproduced = predict_temporal_event_probability(
        loaded, features, layout, validation_ids, device="cpu", batch_size=4
    )
    np.testing.assert_array_equal(prediction, reproduced)
    assert np.isfinite(prediction).all()
    assert np.all((prediction >= 0.0) & (prediction <= 1.0))
