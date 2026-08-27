from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import p3_wave.hierarchical_residual_basis as basis_module
from p3_wave.hierarchical_residual_basis import (
    FORECAST_20M_STEPS,
    FORECAST_KNOTS,
    INPUT_CHANNELS,
    OFFICIAL_FORECAST_INDICES,
    POOLING_FACTORS,
    FixedBasisTrainingConfig,
    HierarchicalResidualBasisConfig,
    HierarchicalResidualBasisForecaster,
    StaticRobustScaler,
    average_pool_context,
    extract_past_raw_context,
    fit_fixed_epoch_and_predict,
    fit_fixed_epoch_hierarchical_model,
    interpolate_forecast_knots,
    load_fitted_hierarchical_model,
    predict_with_fitted_hierarchical_model,
    prepare_hierarchical_context,
    save_fitted_hierarchical_model,
)
from p3_wave.revin_patch import build_synthetic_context


def _small_model_config(feature_count: int) -> HierarchicalResidualBasisConfig:
    return HierarchicalResidualBasisConfig(
        static_feature_count=feature_count,
        hidden_width=16,
        conditioning_width=8,
        dropout=0.0,
    )


def _one_step_training_config() -> FixedBasisTrainingConfig:
    return FixedBasisTrainingConfig(
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        use_bf16_on_cuda=False,
    )


def test_future_and_cross_case_invariance() -> None:
    raw = build_synthetic_context(batch=2, seed=101).numpy()
    future = np.full((13, raw.shape[2]), 999.0, dtype=np.float32)
    full = np.concatenate([raw[0], future], axis=0)
    reference = extract_past_raw_context(full, anchor_position=288)
    altered = full.copy()
    altered[289:] = -1_000_000.0
    np.testing.assert_array_equal(
        reference,
        extract_past_raw_context(altered, anchor_position=288),
    )

    prepared = prepare_hierarchical_context(torch.from_numpy(raw)).values
    changed_raw = raw.copy()
    changed_raw[1, ::2, 0] += 3.0
    changed_raw[1, :, 4] -= 12.0
    changed = prepare_hierarchical_context(torch.from_numpy(changed_raw)).values
    torch.testing.assert_close(prepared[0], changed[0], rtol=0.0, atol=0.0)
    assert not torch.equal(prepared[1], changed[1])


def test_wave_structural_finite_guard() -> None:
    raw = build_synthetic_context(batch=1, seed=103)
    raw[0, 1, 0] = 2.0
    with pytest.raises(ValueError, match="structural"):
        prepare_hierarchical_context(raw)

    raw = build_synthetic_context(batch=1, seed=104)
    raw[0, -1, 0] = float("nan")
    with pytest.raises(ValueError, match="current hs"):
        prepare_hierarchical_context(raw)


def test_context_pooling_shapes_and_interpolation_endpoints() -> None:
    raw = build_synthetic_context(batch=2, seed=107)
    prepared = prepare_hierarchical_context(raw)
    assert prepared.values.shape == (2, 144, INPUT_CHANNELS)
    assert prepared.current_hs.shape == (2,)
    assert prepared.hs_scale.shape == (2,)
    assert torch.isfinite(prepared.values).all()

    expected_steps = {12: 12, 4: 36, 1: 144}
    for factor in POOLING_FACTORS:
        pooled = average_pool_context(prepared.values, factor)
        assert pooled.shape == (2, expected_steps[factor], INPUT_CHANNELS)

    for count in FORECAST_KNOTS:
        knots = torch.linspace(-2.0, 5.0, count).repeat(2, 1)
        dense = interpolate_forecast_knots(knots)
        assert dense.shape == (2, FORECAST_20M_STEPS)
        torch.testing.assert_close(dense[:, 0], knots[:, 0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(dense[:, -1], knots[:, -1], rtol=0.0, atol=0.0)
    assert OFFICIAL_FORECAST_INDICES == (8, 17, 26, 35, 53, 71)


def test_static_scaler_and_core_fit_reject_forbidden_overlap() -> None:
    train_static = np.asarray([[1.0, np.nan], [2.0, 4.0], [3.0, 7.0]], dtype=np.float32)
    with pytest.raises(PermissionError, match="overlap"):
        StaticRobustScaler.fit(
            train_static,
            np.asarray([10, 11, 12]),
            forbidden_case_ids=np.asarray([12, 13]),
        )

    raw = build_synthetic_context(batch=3, seed=109).numpy()
    station = np.asarray([0, 1, 2], dtype=np.int64)
    target = np.zeros((3, 6), dtype=np.float32)
    weight = np.ones(3, dtype=np.float32)
    with pytest.raises(PermissionError, match="overlap"):
        fit_fixed_epoch_hierarchical_model(
            raw,
            station,
            train_static,
            target,
            weight,
            np.asarray([10, 11, 12]),
            forbidden_case_ids=np.asarray([12, 13]),
            seed=7,
            device="cpu",
            model_config=_small_model_config(train_static.shape[1]),
            training_config=_one_step_training_config(),
        )


def test_forward_backward_uses_six_official_targets_only() -> None:
    raw = build_synthetic_context(batch=3, seed=113)
    station = torch.as_tensor([0, 1, 2], dtype=torch.long)
    static = np.asarray(
        [[1.0, np.nan, 3.0], [2.0, 4.0, 5.0], [4.0, 8.0, 6.0]],
        dtype=np.float32,
    )
    scaler = StaticRobustScaler.fit(
        static,
        np.asarray([0, 1, 2]),
        forbidden_case_ids=np.asarray([100]),
    )
    model = HierarchicalResidualBasisForecaster(_small_model_config(static.shape[1]))
    model.train()
    scaled = torch.from_numpy(scaler.transform(static))
    dense = model.forward_dense(raw, station, scaled)
    official = model(raw, station, scaled)
    assert dense.shape == (3, 72)
    assert official.shape == (3, 6)
    torch.testing.assert_close(
        official,
        dense[:, OFFICIAL_FORECAST_INDICES],
        rtol=0.0,
        atol=0.0,
    )
    official.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_train_only_nan_poison_same_seed_and_reload_exact(tmp_path: Path) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cases = 6
        feature_count = 3
        raw = build_synthetic_context(batch=cases, seed=127).numpy()
        station = np.arange(cases, dtype=np.int64) % 3
        rng = np.random.default_rng(127)
        static = rng.normal(size=(cases, feature_count)).astype(np.float32)
        static[1, 1] = np.nan
        target = rng.normal(scale=0.2, size=(cases, 6)).astype(np.float32)
        weight = np.linspace(0.8, 1.2, cases, dtype=np.float32)
        train_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)
        validation_ids = np.asarray([4, 5], dtype=np.int64)
        poisoned = target.copy()
        poisoned[validation_ids] = np.nan
        extreme = target.copy()
        extreme[validation_ids] = 1_000_000.0
        model_config = _small_model_config(feature_count)
        training_config = _one_step_training_config()

        first_prediction, first_fit = fit_fixed_epoch_and_predict(
            raw,
            station,
            static,
            poisoned,
            weight,
            train_ids,
            validation_ids,
            seed=20260823,
            device="cpu",
            model_config=model_config,
            training_config=training_config,
        )
        second_prediction, second_fit = fit_fixed_epoch_and_predict(
            raw,
            station,
            static,
            extreme,
            weight,
            train_ids,
            validation_ids,
            seed=20260823,
            device="cpu",
            model_config=model_config,
            training_config=training_config,
        )
        np.testing.assert_array_equal(first_prediction, second_prediction)
        assert first_fit.model_state_sha256 == second_fit.model_state_sha256
        assert first_fit.scaler.state_sha256 == second_fit.scaler.state_sha256
        assert first_fit.training_steps == 1
        assert first_fit.train_context_sha256 == second_fit.train_context_sha256
        assert first_fit.scaler_state_sha256 == first_fit.scaler.state_sha256

        model_path = tmp_path / "hierarchical_basis.pt"
        save_fitted_hierarchical_model(first_fit, model_path)
        with pytest.raises(FileExistsError):
            save_fitted_hierarchical_model(first_fit, model_path)
        reloaded = load_fitted_hierarchical_model(model_path)
        reloaded_prediction = predict_with_fitted_hierarchical_model(
            reloaded,
            raw[validation_ids],
            station[validation_ids],
            static[validation_ids],
            device="cpu",
        )
        np.testing.assert_array_equal(first_prediction, reloaded_prediction)
        assert reloaded.model_state_sha256 == first_fit.model_state_sha256
        assert reloaded.scaler.state_sha256 == first_fit.scaler.state_sha256
        assert reloaded.train_ids_sha256 == first_fit.train_ids_sha256
        assert reloaded.train_context_sha256 == first_fit.train_context_sha256
        assert reloaded.training_steps == first_fit.training_steps
    finally:
        torch.set_num_threads(previous_threads)


def test_architecture_contains_only_dense_residual_basis_components() -> None:
    config = _small_model_config(feature_count=4)
    model = HierarchicalResidualBasisForecaster(config)
    forbidden_types = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.MultiheadAttention,
        nn.RNN,
        nn.GRU,
        nn.LSTM,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
    )
    assert not any(isinstance(layer, forbidden_types) for layer in model.modules())
    assert len(model.stacks) == 3
    assert all(len(stack) == 2 for stack in model.stacks)
    assert config.pooling_factors == (12, 4, 1)
    assert config.forecast_knots == (6, 18, 72)

    source = inspect.getsource(basis_module).lower()
    for forbidden_term in ("torch.fft", "np.fft", "decisiontree", "randomforest"):
        assert forbidden_term not in source


def test_production_defaults_are_frozen() -> None:
    config = HierarchicalResidualBasisConfig()
    training = FixedBasisTrainingConfig()
    assert config.hidden_width == 192
    assert config.dropout == 0.1
    assert config.pooling_factors == (12, 4, 1)
    assert config.forecast_knots == (6, 18, 72)
    assert config.blocks_per_stack == 2
    assert training == FixedBasisTrainingConfig(
        epochs=12,
        batch_size=512,
        learning_rate=1e-3,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        use_bf16_on_cuda=True,
    )
