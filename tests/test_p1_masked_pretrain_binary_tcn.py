from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import p1_qc.masked_pretrain_binary_tcn as masked_module
from p1_qc.masked_pretrain_binary_tcn import (
    MaskedPretrainBinaryEventModel,
    MaskedPretrainModelConfig,
    MaskedPretrainTrainingConfig,
    _masked_center_batch,
    fit_masked_pretrain_binary_event_model,
    load_fitted_masked_pretrain_model,
    predict_masked_pretrain_binary_probability,
    save_fitted_masked_pretrain_model,
)
from p1_qc.temporal_event_tcn import SequenceLayout, model_state_sha256


def _metadata(rows: int = 52) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["G-ORS"] * rows,
            "layer": [1] * rows,
            "time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"),
        }
    )


def test_masked_center_removes_values_and_finite_indicators_only_at_mask() -> None:
    windows = np.ones((5, 6, 31), dtype=np.float32)
    windows[:, :3, 15] = np.arange(15, dtype=np.float32).reshape(5, 3)
    rng = np.random.default_rng(4)
    masked_windows, target, mask = _masked_center_batch(
        windows, feature_count=3, mask_probability=0.5, rng=rng
    )
    assert mask.shape == target.shape == (5, 3)
    assert mask.any(axis=1).all()
    assert np.all(masked_windows[:, :3, 15][mask] == 0.0)
    assert np.all(masked_windows[:, 3:, 15][mask] == 0.0)
    np.testing.assert_array_equal(target, windows[:, :3, 15])
    np.testing.assert_array_equal(masked_windows[:, :, :15], windows[:, :, :15])
    np.testing.assert_array_equal(masked_windows[:, :, 16:], windows[:, :, 16:])


def test_model_has_reconstruction_event_and_auxiliary_heads() -> None:
    config = MaskedPretrainModelConfig(
        input_feature_count=3,
        group_count=2,
        width=8,
        group_embedding_width=4,
        norm_groups=4,
        dropout=0.0,
    )
    model = MaskedPretrainBinaryEventModel(config)
    event, onset, offset, reconstruction = model(
        torch.zeros(4, 6, 31), torch.zeros(4, dtype=torch.long)
    )
    assert event.shape == onset.shape == offset.shape == (4,)
    assert reconstruction.shape == (4, 3)
    assert config.receptive_field_rows == 31


def test_two_stage_fit_is_deterministic_load_exact_and_non_event_heads_are_inference_dead(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    labels = np.zeros(len(metadata), dtype=np.int8)
    labels[3:12] = 1
    labels[20:33] = 1
    labels[40:48] = 1
    rng = np.random.default_rng(11)
    features = rng.normal(size=(len(metadata), 3)).astype(np.float32)
    features[:, 0] += 2.5 * labels
    train_ids = np.arange(48, dtype=np.int64)
    forbidden = np.arange(48, 52, dtype=np.int64)
    layout = SequenceLayout.build(metadata)
    model_config = MaskedPretrainModelConfig(
        input_feature_count=3,
        group_count=layout.group_count,
        width=8,
        group_embedding_width=4,
        norm_groups=4,
        dropout=0.0,
    )
    training_config = MaskedPretrainTrainingConfig(
        optimizer_steps=3,
        pretrain_steps=1,
        finetune_steps=2,
        batch_size=16,
        boundary_band_rows=2,
    )
    kwargs = {
        "forbidden_ids": forbidden,
        "seed": 20260813,
        "device": "cpu",
        "model_config": model_config,
        "training_config": training_config,
    }
    first = fit_masked_pretrain_binary_event_model(
        features, metadata, labels, layout, train_ids, **kwargs
    )
    second = fit_masked_pretrain_binary_event_model(
        features, metadata, labels, layout, train_ids, **kwargs
    )
    assert first.model_state_sha256 == second.model_state_sha256
    assert first.labels_materialized_after_pretraining is False
    assert np.isfinite(first.mean_pretrain_loss)
    assert np.isfinite(first.mean_finetune_loss)
    prediction_ids = np.arange(len(metadata), dtype=np.int64)
    original = predict_masked_pretrain_binary_probability(
        first, features, layout, prediction_ids, device="cpu", batch_size=13
    )
    path = tmp_path / "model.pt"
    save_fitted_masked_pretrain_model(first, path)
    loaded = load_fitted_masked_pretrain_model(path)
    reproduced = predict_masked_pretrain_binary_probability(
        loaded, features, layout, prediction_ids, device="cpu", batch_size=13
    )
    np.testing.assert_array_equal(original, reproduced)

    changed_state = {name: value.clone() for name, value in loaded.state_dict.items()}
    for name in changed_state:
        if "auxiliary_head" in name or "reconstruction_head" in name:
            changed_state[name].fill_(1000.0)
    dead_heads_changed = replace(
        loaded,
        state_dict=changed_state,
        model_state_sha256=model_state_sha256(changed_state),
    )
    after_change = predict_masked_pretrain_binary_probability(
        dead_heads_changed, features, layout, prediction_ids, device="cpu", batch_size=13
    )
    np.testing.assert_array_equal(original, after_change)


def test_overlap_is_rejected_before_training() -> None:
    metadata = _metadata(20)
    features = np.zeros((20, 2), dtype=np.float32)
    labels = np.zeros(20, dtype=np.int8)
    labels[3:9] = 1
    layout = SequenceLayout.build(metadata)
    with pytest.raises(PermissionError, match="overlap"):
        fit_masked_pretrain_binary_event_model(
            features,
            metadata,
            labels,
            layout,
            np.arange(18),
            forbidden_ids=[17],
            seed=1,
            device="cpu",
            model_config=MaskedPretrainModelConfig(
                input_feature_count=2,
                group_count=1,
                width=8,
                group_embedding_width=4,
                norm_groups=4,
                dropout=0.0,
            ),
            training_config=MaskedPretrainTrainingConfig(
                optimizer_steps=2,
                pretrain_steps=1,
                finetune_steps=1,
                batch_size=8,
            ),
        )


def test_out_of_prefix_features_cannot_change_fitted_model() -> None:
    metadata = _metadata(28)
    labels = np.zeros(len(metadata), dtype=np.int8)
    labels[3:10] = 1
    labels[15:22] = 1
    train_ids = np.arange(24, dtype=np.int64)
    forbidden = np.arange(24, 28, dtype=np.int64)
    features = np.random.default_rng(22).normal(size=(len(metadata), 2)).astype(np.float32)
    changed = features.copy()
    changed[forbidden] = 1_000_000.0
    layout = SequenceLayout.build(metadata)
    kwargs = {
        "forbidden_ids": forbidden,
        "seed": 19,
        "device": "cpu",
        "model_config": MaskedPretrainModelConfig(
            input_feature_count=2,
            group_count=1,
            width=8,
            group_embedding_width=4,
            norm_groups=4,
            dropout=0.0,
        ),
        "training_config": MaskedPretrainTrainingConfig(
            optimizer_steps=2,
            pretrain_steps=1,
            finetune_steps=1,
            batch_size=8,
            boundary_band_rows=2,
        ),
    }
    first = fit_masked_pretrain_binary_event_model(
        features, metadata, labels, layout, train_ids, **kwargs
    )
    second = fit_masked_pretrain_binary_event_model(
        changed, metadata, labels, layout, train_ids, **kwargs
    )
    assert first.model_state_sha256 == second.model_state_sha256


def test_source_reads_prefix_labels_only_after_masked_pretraining_loop() -> None:
    source = inspect.getsource(fit_masked_pretrain_binary_event_model)
    pretrain_loop = source.index("for _ in range(training_config.pretrain_steps):")
    pretrain_completion = source.index("pretrain_losses.append")
    label_materialization = source.index("label_values = binary_labels()")
    target_read = source.index("build_prefix_event_boundary_targets(")
    assert pretrain_loop < pretrain_completion < label_materialization < target_read


def test_deferred_label_loader_runs_only_after_masked_pretraining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(24)
    labels = np.zeros(len(metadata), dtype=np.int8)
    labels[3:9] = 1
    labels[13:20] = 1
    features = np.random.default_rng(31).normal(size=(len(metadata), 2)).astype(np.float32)
    layout = SequenceLayout.build(metadata)
    state = {"masked_batches": 0, "label_loads": 0}
    original = masked_module._masked_center_batch

    def recording_masked_batch(*args: object, **kwargs: object) -> tuple[np.ndarray, ...]:
        state["masked_batches"] += 1
        return original(*args, **kwargs)

    def deferred_labels() -> np.ndarray:
        assert state["masked_batches"] == 1
        state["label_loads"] += 1
        return labels

    monkeypatch.setattr(masked_module, "_masked_center_batch", recording_masked_batch)
    fitted = fit_masked_pretrain_binary_event_model(
        features,
        metadata,
        deferred_labels,
        layout,
        np.arange(22),
        forbidden_ids=np.arange(22, 24),
        seed=7,
        device="cpu",
        model_config=MaskedPretrainModelConfig(
            input_feature_count=2,
            group_count=1,
            width=8,
            group_embedding_width=4,
            norm_groups=4,
            dropout=0.0,
        ),
        training_config=MaskedPretrainTrainingConfig(
            optimizer_steps=2,
            pretrain_steps=1,
            finetune_steps=1,
            batch_size=8,
            boundary_band_rows=2,
        ),
    )
    assert state == {"masked_batches": 1, "label_loads": 1}
    assert fitted.labels_materialized_after_pretraining is True
