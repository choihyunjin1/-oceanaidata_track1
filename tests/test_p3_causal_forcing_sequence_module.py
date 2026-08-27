from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from p3_wave.causal_forcing_sequence import (
    CausalForcingSequenceConfig,
    CompactRobustScaler,
    FixedEpochTrainingConfig,
    LeadCoupledCausalForcingEncoder,
    build_causal_forcing_sequence,
    fit_fixed_epoch_and_predict,
    load_fitted_sequence_model,
    predict_with_fitted_sequence_model,
    prepare_streams_deterministic,
    save_fitted_sequence_model,
)
from p3_wave.revin_patch import build_synthetic_context, prepare_streams


def _small_model_config(feature_count: int) -> CausalForcingSequenceConfig:
    return CausalForcingSequenceConfig(
        compact_feature_count=feature_count,
        width=8,
        compact_hidden=12,
        attention_heads=2,
        norm_groups=2,
        dropout=0.0,
        wave_dilations=(1, 2),
        forcing_dilations=(1, 2),
    )


def test_causal_forcing_shape_and_future_perturbation() -> None:
    raw = build_synthetic_context(batch=2, seed=17).numpy()
    reference = build_causal_forcing_sequence(raw)
    assert reference.shape == (2, 289, 12)
    assert reference.dtype == np.float32
    assert np.isfinite(reference).all()

    cutoff = 150
    perturbed = raw.copy()
    perturbed[:, cutoff + 1 :, 4] += 30.0
    perturbed[:, cutoff + 1 :, 5] += 20.0
    perturbed[:, cutoff + 1 :, 9] -= 40.0
    future_wave = np.arange(cutoff + 1, 289)
    future_wave = future_wave[future_wave % 2 == 0]
    perturbed[:, future_wave, 1] += 7.0
    perturbed[:, future_wave, 3] = np.mod(perturbed[:, future_wave, 3] + 120.0, 360.0)
    changed = build_causal_forcing_sequence(perturbed)

    np.testing.assert_array_equal(reference[:, : cutoff + 1], changed[:, : cutoff + 1])
    assert not np.array_equal(reference[:, cutoff + 1 :], changed[:, cutoff + 1 :])


def test_compact_scaler_is_prefix_only_and_rejects_validation_overlap() -> None:
    values = np.asarray(
        [
            [1.0, np.nan, 10.0],
            [2.0, 4.0, 10.0],
            [3.0, 6.0, 10.0],
            [4.0, 8.0, 10.0],
            [100.0, 200.0, -500.0],
            [300.0, 400.0, 900.0],
        ],
        dtype=np.float32,
    )
    train_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)
    validation_ids = np.asarray([4, 5], dtype=np.int64)
    first = CompactRobustScaler.fit(values, train_ids, forbidden_ids=validation_ids)
    altered = values.copy()
    altered[validation_ids] *= 1_000_000.0
    second = CompactRobustScaler.fit(altered, train_ids, forbidden_ids=validation_ids)

    np.testing.assert_array_equal(first.center, second.center)
    np.testing.assert_array_equal(first.scale, second.scale)
    assert first.fit_ids_sha256 == second.fit_ids_sha256
    transformed = first.transform(values)
    assert transformed.shape == (6, 6)
    assert np.isfinite(transformed).all()
    assert transformed[0, 4] == 0.0
    with pytest.raises(PermissionError, match="overlap"):
        CompactRobustScaler.fit(values, train_ids, forbidden_ids=np.asarray([3, 4]))


def test_deterministic_stream_preparation_matches_existing_cpu() -> None:
    raw = build_synthetic_context(batch=4, seed=19)
    # Exercise even finite counts and a fully missing channel in addition to the
    # structural wave gaps emitted by the synthetic context helper.
    raw[0, 0, 0] = float("nan")
    raw[1, 0, 4] = float("nan")
    raw[2, :, 7] = float("nan")

    expected = prepare_streams(raw)
    actual = prepare_streams_deterministic(raw)

    torch.testing.assert_close(actual.wave, expected.wave, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.atmos, expected.atmos, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.current_hs, expected.current_hs, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.hs_scale, expected.hs_scale, rtol=0.0, atol=0.0)


def test_lead_coupled_encoder_forward_contract() -> None:
    raw = build_synthetic_context(batch=3, seed=23).numpy()
    forcing = build_causal_forcing_sequence(raw)
    compact = np.asarray(
        [[1.0, np.nan, 3.0, 4.0], [2.0, 5.0, 4.0, 3.0], [3.0, 7.0, 5.0, 2.0]],
        dtype=np.float32,
    )
    scaler = CompactRobustScaler.fit(compact, np.arange(3, dtype=np.int64))
    model = LeadCoupledCausalForcingEncoder(_small_model_config(compact.shape[1]))
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(raw),
            torch.as_tensor([0, 1, 2], dtype=torch.long),
            torch.from_numpy(scaler.transform(compact)),
            torch.from_numpy(forcing),
        )
    assert prediction.shape == (3, 6)
    assert torch.isfinite(prediction).all()
    assert model.trainable_parameter_count > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_architecture_cuda_deterministic_forward_backward() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(991)
        torch.cuda.manual_seed_all(991)

        raw = build_synthetic_context(batch=2, seed=991).numpy()
        forcing = build_causal_forcing_sequence(raw)
        compact = np.asarray([[1.0, 2.0, np.nan, 4.0], [2.0, 3.0, 5.0, 7.0]], dtype=np.float32)
        scaler = CompactRobustScaler.fit(compact, np.arange(2, dtype=np.int64))
        model = LeadCoupledCausalForcingEncoder(
            CausalForcingSequenceConfig(compact_feature_count=compact.shape[1])
        ).cuda()
        model.train()
        prediction = model(
            torch.from_numpy(raw).cuda(),
            torch.as_tensor([0, 1], dtype=torch.long, device="cuda"),
            torch.from_numpy(scaler.transform(compact)).cuda(),
            torch.from_numpy(forcing).cuda(),
        )
        prediction.square().mean().backward()

        assert prediction.shape == (2, 6)
        assert torch.isfinite(prediction).all()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.cuda.empty_cache()


def test_fixed_epoch_fit_is_deterministic_excludes_validation_and_reloads(
    tmp_path: Path,
) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cases = 7
        feature_count = 4
        raw = build_synthetic_context(batch=cases, seed=31).numpy()
        station = np.arange(cases, dtype=np.int64) % 3
        rng = np.random.default_rng(31)
        compact = rng.normal(size=(cases, feature_count)).astype(np.float32)
        compact[1, 2] = np.nan
        target = rng.normal(scale=0.2, size=(cases, 6)).astype(np.float32)
        weight = np.linspace(0.8, 1.2, cases, dtype=np.float32)
        train_ids = np.arange(5, dtype=np.int64)
        validation_ids = np.asarray([5, 6], dtype=np.int64)
        forcing = build_causal_forcing_sequence(raw)
        scaler = CompactRobustScaler.fit(
            compact,
            train_ids,
            forbidden_ids=validation_ids,
        )
        model_config = _small_model_config(feature_count)
        training_config = FixedEpochTrainingConfig(
            epochs=1,
            batch_size=5,
            learning_rate=3e-4,
            weight_decay=2e-4,
            gradient_clip_norm=1.0,
            use_bf16_on_cuda=False,
        )

        first_prediction, first_fit = fit_fixed_epoch_and_predict(
            raw,
            station,
            compact,
            target,
            weight,
            train_ids,
            validation_ids,
            seed=20260816,
            device="cpu",
            model_config=model_config,
            training_config=training_config,
            forcing=forcing,
            compact_scaler=scaler,
        )
        altered_target = target.copy()
        altered_target[validation_ids] += 1_000_000.0
        second_prediction, second_fit = fit_fixed_epoch_and_predict(
            raw,
            station,
            compact,
            altered_target,
            weight,
            train_ids,
            validation_ids,
            seed=20260816,
            device="cpu",
            model_config=model_config,
            training_config=training_config,
            forcing=forcing,
            compact_scaler=scaler,
        )

        np.testing.assert_array_equal(first_prediction, second_prediction)
        assert first_fit.model_state_sha256 == second_fit.model_state_sha256
        assert first_fit.scaler.state_sha256 == second_fit.scaler.state_sha256

        model_path = tmp_path / "sequence_model.pt"
        save_fitted_sequence_model(first_fit, model_path)
        with pytest.raises(FileExistsError):
            save_fitted_sequence_model(first_fit, model_path)
        reloaded = load_fitted_sequence_model(model_path)
        reloaded_prediction = predict_with_fitted_sequence_model(
            reloaded,
            raw,
            station,
            compact,
            validation_ids,
            device="cpu",
            forcing=forcing,
        )
        np.testing.assert_array_equal(first_prediction, reloaded_prediction)
        assert reloaded.model_state_sha256 == first_fit.model_state_sha256
        assert reloaded.scaler.fit_ids_sha256 == scaler.fit_ids_sha256
    finally:
        torch.set_num_threads(previous_threads)
