from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from p1_qc.incumbent_residual_tcn import (
    ResidualModelConfig,
    ResidualTrainingConfig,
    _causal_windows,
    build_three_block_inner_splits,
    exact_identity_or_residual,
    fit_incumbent_residual_model,
    load_fitted_incumbent_residual_model,
    predict_incumbent_residual_probability,
    save_fitted_incumbent_residual_model,
)
from p1_qc.temporal_event_tcn import PrefixRobustScaler, SequenceLayout


def _metadata(rows: int = 96) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["G-ORS"] * rows,
            "layer": [1] * rows,
            "time": pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC"),
        }
    )


def _probabilities(rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = np.linspace(0.08, 0.92, rows, dtype=np.float32)
    mean = np.clip(seed * 0.9 + 0.04, 0.01, 0.99).astype(np.float32)
    std = np.full(rows, 0.03, dtype=np.float32)
    decision = (seed >= 0.5).astype(np.int8)
    return seed, mean, std, decision


def test_three_block_split_is_timestamp_only_disjoint_and_purged() -> None:
    rows = 120
    metadata = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "label": np.arange(rows) % 2,
        }
    )
    ids = np.arange(rows, dtype=np.int64)
    first = build_three_block_inner_splits(metadata, ids, purge_days=7)
    mutated = metadata.copy()
    mutated["label"] = 1 - mutated["label"]
    second = build_three_block_inner_splits(mutated, ids, purge_days=7)
    assert len(first) == len(second) == 3
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.teacher_train_ids, right.teacher_train_ids)
        assert np.array_equal(left.teacher_prediction_ids, right.teacher_prediction_ids)
        assert np.intersect1d(left.teacher_train_ids, left.teacher_prediction_ids).size == 0
        train_end = pd.Timestamp(left.train_end_utc)
        prediction_start = pd.Timestamp(left.prediction_start_utc)
        assert prediction_start - train_end > pd.Timedelta(days=7)
    all_prediction = np.concatenate([split.teacher_prediction_ids for split in first])
    assert len(np.unique(all_prediction)) == len(all_prediction)


def test_causal_windows_ignore_every_future_raw_value() -> None:
    metadata = _metadata(96)
    layout = SequenceLayout.build(metadata)
    rng = np.random.default_rng(7)
    values = rng.normal(size=(96, 3)).astype(np.float32)
    scaler = PrefixRobustScaler.fit(values, np.arange(48, dtype=np.int64))
    scaled = scaler.transform(values)
    centers = np.asarray([20, 31, 47], dtype=np.int64)
    original, groups = _causal_windows(
        layout, scaled, centers, receptive_field_rows=31
    )
    changed = values.copy()
    changed[48:] = 1_000_000.0 + np.arange(len(changed[48:]))[:, None]
    changed_scaled = scaler.transform(changed)
    perturbed, changed_groups = _causal_windows(
        layout, changed_scaled, centers, receptive_field_rows=31
    )
    assert np.array_equal(original, perturbed)
    assert np.array_equal(groups, changed_groups)
    assert np.count_nonzero(original[:, :, 16:]) == 0


def test_fit_predict_save_load_and_logit_bound(tmp_path) -> None:
    metadata = _metadata(96)
    layout = SequenceLayout.build(metadata)
    rng = np.random.default_rng(11)
    values = rng.normal(size=(96, 3)).astype(np.float32)
    train_ids = np.arange(64, dtype=np.int64)
    prediction_ids = np.arange(64, 96, dtype=np.int64)
    labels = ((np.arange(len(train_ids)) // 4) % 2).astype(np.int8)
    base_seed, base_mean, base_std, base_decision = _probabilities(96)
    model_config = ResidualModelConfig(input_feature_count=3, group_count=1)
    training_config = ResidualTrainingConfig(optimizer_steps=2, batch_size=32)
    fitted = fit_incumbent_residual_model(
        values,
        layout,
        train_ids,
        labels,
        base_seed,
        base_mean,
        base_std,
        base_decision,
        context_ids=np.arange(96, dtype=np.int64),
        forbidden_ids=prediction_ids,
        seed=20260813,
        device="cpu",
        model_config=model_config,
        training_config=training_config,
    )
    probability = predict_incumbent_residual_probability(
        fitted,
        values,
        layout,
        prediction_ids,
        base_seed,
        base_mean,
        base_std,
        base_decision,
        context_ids=np.arange(96, dtype=np.int64),
        device="cpu",
        batch_size=17,
    )
    path = tmp_path / "residual.pt"
    save_fitted_incumbent_residual_model(fitted, path)
    loaded = load_fitted_incumbent_residual_model(path)
    reproduced = predict_incumbent_residual_probability(
        loaded,
        values,
        layout,
        prediction_ids,
        base_seed,
        base_mean,
        base_std,
        base_decision,
        context_ids=np.arange(96, dtype=np.int64),
        device="cpu",
        batch_size=17,
    )
    assert np.array_equal(probability, reproduced)
    clipped = np.clip(probability.astype(np.float64), 1e-6, 1.0 - 1e-6)
    base = np.clip(base_seed[prediction_ids].astype(np.float64), 1e-6, 1.0 - 1e-6)
    correction = np.log(clipped / (1.0 - clipped)) - np.log(base / (1.0 - base))
    assert np.max(np.abs(correction)) <= 0.50001
    assert loaded.model_state_sha256 == fitted.model_state_sha256


def test_failed_gate_is_exact_dtype_and_byte_identity() -> None:
    incumbent = np.asarray([0.1, 0.2, 0.9], dtype=np.float32)
    residual = np.asarray([0.3, 0.4, 0.5], dtype=np.float32)
    result = exact_identity_or_residual(incumbent, residual, gate_passed=False)
    assert result.dtype == incumbent.dtype
    assert np.array_equal(result, incumbent)
    assert result.tobytes() == incumbent.tobytes()
    assert result is not incumbent
    assert np.array_equal(
        exact_identity_or_residual(incumbent, residual, gate_passed=True), residual
    )


def test_fit_rejects_forbidden_overlap_and_full_length_label_surface() -> None:
    metadata = _metadata(48)
    layout = SequenceLayout.build(metadata)
    values = np.zeros((48, 2), dtype=np.float32)
    train_ids = np.arange(32, dtype=np.int64)
    base_seed, base_mean, base_std, base_decision = _probabilities(48)
    kwargs = {
        "feature_values": values,
        "layout": layout,
        "train_ids": train_ids,
        "base_seed_probability": base_seed,
        "base_mean_probability": base_mean,
        "base_probability_std": base_std,
        "base_decision": base_decision,
        "context_ids": np.arange(48, dtype=np.int64),
        "seed": 1,
        "device": "cpu",
        "model_config": ResidualModelConfig(input_feature_count=2, group_count=1),
        "training_config": ResidualTrainingConfig(optimizer_steps=1, batch_size=16),
    }
    labels = (np.arange(32) % 2).astype(np.int8)
    with pytest.raises(PermissionError, match="overlap forbidden"):
        fit_incumbent_residual_model(
            train_labels=labels,
            forbidden_ids=np.asarray([31, 47]),
            **kwargs,
        )
    with pytest.raises(ValueError, match="aligned binary vector"):
        fit_incumbent_residual_model(
            train_labels=np.resize(labels, 48),
            forbidden_ids=np.arange(32, 48, dtype=np.int64),
            **kwargs,
        )


def test_frozen_configs_reject_reactive_loss_or_bound_changes() -> None:
    model = ResidualModelConfig(input_feature_count=2, group_count=1)
    training = ResidualTrainingConfig()
    with pytest.raises(ValueError, match="must equal 0.5"):
        replace(model, maximum_absolute_logit_correction=0.6).validate()
    with pytest.raises(ValueError, match="loss weights differ"):
        replace(training, identity_regularizer_weight=0.1).validate()
