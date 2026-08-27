from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from p1_qc.binary_event_tcn import (
    BinaryEventModelConfig,
    DenseNaturalTrainingConfig,
    StationLayerBinaryEventModel,
    _DenseEpochSampler,
    build_prefix_event_boundary_targets,
    fit_fixed_step_binary_event_model,
    load_fitted_binary_event_model,
    predict_binary_event_probability,
    save_fitted_binary_event_model,
)
from p1_qc.temporal_event_tcn import SequenceLayout, model_state_sha256


def _metadata(rows: int = 48) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["G-ORS"] * rows,
            "layer": [1] * rows,
            "time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"),
        }
    )


def test_boundary_targets_are_prefix_only_bands_and_reject_overlap() -> None:
    metadata = _metadata(30)
    labels = np.zeros(30, dtype=np.int8)
    labels[5:15] = 1
    labels[25:] = 1
    ids = np.arange(20, dtype=np.int64)
    selected, event, onset, offset = build_prefix_event_boundary_targets(
        metadata, labels, ids, forbidden_ids=np.arange(20, 30), boundary_band_rows=3
    )
    np.testing.assert_array_equal(selected, ids)
    np.testing.assert_array_equal(event, labels[:20])
    np.testing.assert_array_equal(np.flatnonzero(onset), [5, 6, 7])
    np.testing.assert_array_equal(np.flatnonzero(offset), [12, 13, 14])
    assert np.all(onset <= event)
    assert np.all(offset <= event)
    with pytest.raises(PermissionError, match="overlap"):
        build_prefix_event_boundary_targets(metadata, labels, ids, forbidden_ids=[19])


def test_dense_sampler_covers_each_natural_epoch_without_replacement() -> None:
    ids = np.arange(17, dtype=np.int64)
    sampler = _DenseEpochSampler(ids, seed=20260813)
    first = np.concatenate([sampler.next(5), sampler.next(12)])
    np.testing.assert_array_equal(np.sort(first), ids)
    second = sampler.next(17)
    np.testing.assert_array_equal(np.sort(second), ids)


def test_binary_model_has_three_heads_but_one_event_surface() -> None:
    config = BinaryEventModelConfig(
        input_feature_count=3,
        group_count=2,
        width=8,
        group_embedding_width=4,
        norm_groups=4,
        dropout=0.0,
    )
    model = StationLayerBinaryEventModel(config)
    outputs = model(torch.zeros(5, 6, 31), torch.zeros(5, dtype=torch.long))
    assert len(outputs) == 3
    assert all(output.shape == (5,) for output in outputs)
    assert config.receptive_field_rows == 31
    assert model.trainable_parameter_count > 0


def test_fixed_fit_is_deterministic_load_exact_and_aux_logits_do_not_affect_prediction(
    tmp_path: Path,
) -> None:
    metadata = _metadata(48)
    labels = np.zeros(48, dtype=np.int8)
    labels[4:12] = 1
    labels[20:31] = 1
    labels[38:44] = 1
    rng = np.random.default_rng(7)
    features = rng.normal(size=(48, 3)).astype(np.float32)
    features[:, 0] += labels * 3.0
    train_ids = np.arange(44, dtype=np.int64)
    forbidden = np.arange(44, 48, dtype=np.int64)
    layout = SequenceLayout.build(metadata)
    model_config = BinaryEventModelConfig(
        input_feature_count=3,
        group_count=layout.group_count,
        width=8,
        group_embedding_width=4,
        dropout=0.0,
        norm_groups=4,
    )
    training_config = DenseNaturalTrainingConfig(
        optimizer_steps=4,
        batch_size=16,
        learning_rate=1e-3,
        auxiliary_loss_weight=0.1,
        boundary_band_rows=2,
    )
    first = fit_fixed_step_binary_event_model(
        features,
        metadata,
        labels,
        layout,
        train_ids,
        forbidden_ids=forbidden,
        seed=20260813,
        device="cpu",
        model_config=model_config,
        training_config=training_config,
    )
    second = fit_fixed_step_binary_event_model(
        features,
        metadata,
        labels,
        layout,
        train_ids,
        forbidden_ids=forbidden,
        seed=20260813,
        device="cpu",
        model_config=model_config,
        training_config=training_config,
    )
    assert first.model_state_sha256 == second.model_state_sha256
    prediction_ids = np.arange(48, dtype=np.int64)
    original = predict_binary_event_probability(
        first, features, layout, prediction_ids, device="cpu", batch_size=13
    )
    path = tmp_path / "model.pt"
    save_fitted_binary_event_model(first, path)
    loaded = load_fitted_binary_event_model(path)
    reproduced = predict_binary_event_probability(
        loaded, features, layout, prediction_ids, device="cpu", batch_size=13
    )
    np.testing.assert_array_equal(original, reproduced)

    changed_state = {name: value.clone() for name, value in loaded.state_dict.items()}
    for name in changed_state:
        if "auxiliary_head" in name:
            changed_state[name].fill_(1000.0)
    aux_changed = replace(
        loaded,
        state_dict=changed_state,
        model_state_sha256=model_state_sha256(changed_state),
    )
    after_aux_change = predict_binary_event_probability(
        aux_changed, features, layout, prediction_ids, device="cpu", batch_size=13
    )
    np.testing.assert_array_equal(original, after_aux_change)

