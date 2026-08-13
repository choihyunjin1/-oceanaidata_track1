from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from p1_qc.models_ssl import (
    SSLModelConfig,
    SSLTrainConfig,
    assert_fold_local_rows,
    block_mask,
    extract_ssl_embeddings,
    gap_aware_windows,
    load_ssl_checkpoint,
    save_ssl_checkpoint,
    train_masked_reconstruction,
)


def normalized_signal(rows: int, features: int = 3) -> np.ndarray:
    phase = np.arange(rows, dtype=np.float32)
    columns = [
        np.sin(phase / 5.0),
        np.cos(phase / 9.0),
        0.5 * np.sin(phase / 3.0) + 0.2 * np.cos(phase / 7.0),
    ]
    return np.column_stack(columns[:features]).astype(np.float32)


def tiny_configs(input_dim: int = 3) -> tuple[SSLModelConfig, SSLTrainConfig]:
    return (
        SSLModelConfig(
            input_dim=input_dim,
            channels=(4, 6),
            kernel_size=3,
            dropout=0.0,
        ),
        SSLTrainConfig(
            window_steps=12,
            stride_steps=6,
            mask_fraction=0.25,
            mask_block_steps=3,
            batch_size=4,
            max_epochs=3,
            patience=2,
            learning_rate=5.0e-3,
            use_bfloat16=False,
            seed=19,
        ),
    )


def test_gap_aware_windows_and_block_mask_are_deterministic() -> None:
    segments = np.array([0] * 17 + [1] * 9)
    windows = gap_aware_windows(segments, window_steps=8, stride_steps=5)
    assert windows
    assert all(len(set(segments[item.start : item.stop])) == 1 for item in windows)
    assert np.all(
        block_mask(20, mask_fraction=0.3, block_steps=4, rng=np.random.default_rng(7))
        == block_mask(20, mask_fraction=0.3, block_steps=4, rng=np.random.default_rng(7))
    )


def test_validation_row_overlap_is_rejected_before_training() -> None:
    features = normalized_signal(24)
    model_config, train_config = tiny_configs()
    with pytest.raises(ValueError, match="validation rows leaked"):
        train_masked_reconstruction(
            features,
            np.zeros(24, dtype=np.int16),
            np.arange(100, 124),
            validation_features=features[:8],
            validation_segment_ids=np.zeros(8, dtype=np.int16),
            validation_row_ids=np.arange(120, 128),
            model_config=model_config,
            train_config=train_config,
            device="cpu",
        )
    with pytest.raises(ValueError, match="train_row_ids must be unique"):
        assert_fold_local_rows([1, 1, 2], [3, 4])


def test_tiny_cpu_training_embeddings_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    train_features = normalized_signal(48)
    validation_features = normalized_signal(24) * 0.9
    train_segments = np.array([0] * 24 + [1] * 24)
    validation_segments = np.array([5] * 12 + [6] * 12)
    model_config, train_config = tiny_configs()

    result = train_masked_reconstruction(
        train_features,
        train_segments,
        np.arange(1000, 1048),
        validation_features=validation_features,
        validation_segment_ids=validation_segments,
        validation_row_ids=np.arange(2000, 2024),
        model_config=model_config,
        train_config=train_config,
        device="cpu",
    )
    assert 1 <= len(result.history) <= train_config.max_epochs
    assert np.isfinite(result.best_validation_loss)
    assert all(
        len(set(train_segments[item.start : item.stop])) == 1 for item in result.train_windows
    )
    embeddings = extract_ssl_embeddings(
        result,
        validation_features,
        validation_segments,
        device="cpu",
    )
    assert embeddings.shape == (24, model_config.channels[-1])
    assert np.isfinite(embeddings).all()

    short_embeddings = extract_ssl_embeddings(
        result,
        validation_features[:2],
        np.zeros(2, dtype=np.int16),
        window_steps=2,
        stride_steps=1,
        device="cpu",
    )
    assert short_embeddings.shape == (2, model_config.channels[-1])
    assert np.isfinite(short_embeddings).all()

    checkpoint = save_ssl_checkpoint(result, tmp_path / "ssl.pt")
    restored = load_ssl_checkpoint(checkpoint, device="cpu")
    restored_embeddings = extract_ssl_embeddings(
        restored,
        validation_features,
        validation_segments,
        device="cpu",
    )
    np.testing.assert_allclose(embeddings, restored_embeddings, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_array_equal(restored.train_row_ids, np.arange(1000, 1048))


def test_raw_scale_input_is_rejected() -> None:
    model_config, train_config = tiny_configs(input_dim=1)
    raw_absolute_temperature = np.full((24, 1), 250.0, dtype=np.float32)
    with pytest.raises(ValueError, match="robust-normalized"):
        train_masked_reconstruction(
            raw_absolute_temperature,
            np.zeros(24),
            np.arange(24),
            model_config=model_config,
            train_config=train_config,
            device="cpu",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_forward_smoke() -> None:
    features = normalized_signal(24)
    model_config, train_config = tiny_configs()
    cuda_config = SSLTrainConfig(
        **{
            **train_config.__dict__,
            "max_epochs": 1,
            "patience": 1,
            "use_bfloat16": True,
        }
    )
    result = train_masked_reconstruction(
        features,
        np.zeros(24),
        np.arange(24),
        model_config=model_config,
        train_config=cuda_config,
        device="cuda",
    )
    embeddings = extract_ssl_embeddings(result, features, np.zeros(24), device="cuda")
    assert embeddings.shape == (24, model_config.channels[-1])
    assert np.isfinite(embeddings).all()
