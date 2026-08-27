from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from p3_wave.dense72_targets_r1 import (
    OFFICIAL_DENSE_INDICES,
    Dense72TargetAccessor,
    sha256_file,
)
from p3_wave.hierarchical_residual_basis import (
    FixedBasisTrainingConfig,
    HierarchicalResidualBasisConfig,
)
from p3_wave.hierarchical_residual_basis_dense72_r1 import (
    fit_dense72_and_predict,
    fit_dense72_hierarchical_model,
    load_fitted_dense72_model,
    predict_with_fitted_dense72_model,
    save_fitted_dense72_model,
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


def _write_synthetic_wave(
    directory: Path,
    anchors: pd.DataFrame,
    *,
    validation_offset: float,
) -> Path:
    directory.mkdir(parents=True)
    path = directory / "train_wave.csv"
    rows: list[dict[str, object]] = []
    validation_ids = {2, 3}
    for anchor in anchors.itertuples(index=False):
        base = pd.Timestamp(anchor.anchor_time)
        for step in range(1, 73):
            value = 1.0 + 0.01 * step + 0.1 * int(anchor.anchor_id)
            if int(anchor.anchor_id) in validation_ids:
                value += validation_offset
            rows.append(
                {
                    "station": anchor.station,
                    "time": (base + pd.Timedelta(minutes=20 * step))
                    .tz_convert("Asia/Seoul")
                    .isoformat(),
                    "hs": value,
                    "tp": 7.0,
                    "hmax": 2.0,
                    "wvdir": 180.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    return path


def _synthetic_anchors() -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    return pd.DataFrame(
        {
            "anchor_id": np.arange(4, dtype=np.int64),
            "station": ["G-ORS", "I-ORS", "S-ORS", "G-ORS"],
            "anchor_time": [start + pd.Timedelta(hours=100 * index) for index in range(4)],
            "current_hs": [1.5, 1.6, 1.7, 1.8],
        }
    )


def _accessor(path: Path, anchors: pd.DataFrame) -> Dense72TargetAccessor:
    return Dense72TargetAccessor(
        path,
        anchors,
        validation_groups={"fold_a": np.asarray([2]), "fold_b": np.asarray([3])},
        expected_source_sha256=sha256_file(path),
        expected_source_bytes=path.stat().st_size,
        expected_source_rows=4 * 72,
        enforce_canonical_aggregate=False,
    )


def test_selective_accessor_validation_poison_isolation_and_release(tmp_path: Path) -> None:
    anchors = _synthetic_anchors()
    first = _accessor(
        _write_synthetic_wave(tmp_path / "first", anchors, validation_offset=0.0), anchors
    )
    second = _accessor(
        _write_synthetic_wave(tmp_path / "second", anchors, validation_offset=1_000_000.0),
        anchors,
    )
    train_ids = np.asarray([0, 1], dtype=np.int64)
    active = np.asarray([2], dtype=np.int64)
    first_payload = first.load_training_targets(train_ids, active_validation_case_ids=active)
    second_payload = second.load_training_targets(train_ids, active_validation_case_ids=active)
    np.testing.assert_array_equal(first_payload.target_delta, second_payload.target_delta)
    np.testing.assert_array_equal(first_payload.target_mask, second_payload.target_mask)
    assert first_payload.target_delta_sha256 == second_payload.target_delta_sha256
    assert first.forbidden_scalar_decodes == second.forbidden_scalar_decodes == 0
    assert first_payload.target_mask[:, OFFICIAL_DENSE_INDICES].all()

    with pytest.raises(PermissionError, match="unreleased validation"):
        first.load_training_targets(np.asarray([2]), active_validation_case_ids=np.asarray([3]))
    fold_sha = hashlib.sha256(b"fold-a-commitment").hexdigest()
    first.release_validation_group("fold_a", fold_commitment_sha256=fold_sha)
    released = first.load_training_targets(
        np.asarray([2]), active_validation_case_ids=np.asarray([3])
    )
    assert released.target_mask.shape == (1, 72)
    assert first.forbidden_scalar_decodes == 0


def test_dense72_core_rejects_overlap_and_noncanonical_masked_values() -> None:
    raw = build_synthetic_context(batch=3, seed=301).numpy()
    station = np.asarray([0, 1, 2], dtype=np.int64)
    static = np.ones((3, 2), dtype=np.float32)
    target = np.zeros((3, 72), dtype=np.float32)
    mask = np.ones((3, 72), dtype=bool)
    weight = np.ones(3, dtype=np.float32)
    with pytest.raises(PermissionError, match="overlap"):
        fit_dense72_hierarchical_model(
            raw,
            station,
            static,
            target,
            mask,
            weight,
            np.asarray([10, 11, 12]),
            forbidden_case_ids=np.asarray([12, 13]),
            seed=7,
            device="cpu",
            model_config=_small_model_config(2),
            training_config=_one_step_training_config(),
        )
    mask[0, 0] = False
    target[0, 0] = 99.0
    with pytest.raises(ValueError, match="canonical zero"):
        fit_dense72_hierarchical_model(
            raw,
            station,
            static,
            target,
            mask,
            weight,
            np.asarray([10, 11, 12]),
            forbidden_case_ids=np.asarray([20]),
            seed=7,
            device="cpu",
            model_config=_small_model_config(2),
            training_config=_one_step_training_config(),
        )


def test_dense72_validation_poison_fit_and_reload_are_exact(tmp_path: Path) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cases = 6
        feature_count = 3
        raw = build_synthetic_context(batch=cases, seed=307).numpy()
        station = np.arange(cases, dtype=np.int64) % 3
        rng = np.random.default_rng(307)
        static = rng.normal(size=(cases, feature_count)).astype(np.float32)
        static[1, 1] = np.nan
        target = rng.normal(scale=0.2, size=(cases, 72)).astype(np.float32)
        mask = np.ones((cases, 72), dtype=bool)
        mask[0, 1] = False
        target[0, 1] = 0.0
        weight = np.linspace(0.8, 1.2, cases, dtype=np.float32)
        train_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)
        validation_ids = np.asarray([4, 5], dtype=np.int64)
        poisoned = target.copy()
        poisoned[validation_ids] = np.nan
        extreme = target.copy()
        extreme[validation_ids] = 1_000_000.0
        first_prediction, first_fit = fit_dense72_and_predict(
            raw,
            station,
            static,
            poisoned,
            mask,
            weight,
            train_ids,
            validation_ids,
            seed=20260823,
            device="cpu",
            model_config=_small_model_config(feature_count),
            training_config=_one_step_training_config(),
        )
        second_prediction, second_fit = fit_dense72_and_predict(
            raw,
            station,
            static,
            extreme,
            mask,
            weight,
            train_ids,
            validation_ids,
            seed=20260823,
            device="cpu",
            model_config=_small_model_config(feature_count),
            training_config=_one_step_training_config(),
        )
        np.testing.assert_array_equal(first_prediction, second_prediction)
        assert first_fit.model_state_sha256 == second_fit.model_state_sha256
        assert first_fit.train_target_sha256 == second_fit.train_target_sha256
        assert first_fit.valid_target_scalars_per_epoch == 4 * 72 - 1
        assert first_fit.training_steps == 1

        model_path = tmp_path / "dense72.pt"
        save_fitted_dense72_model(first_fit, model_path)
        with pytest.raises(FileExistsError):
            save_fitted_dense72_model(first_fit, model_path)
        loaded = load_fitted_dense72_model(model_path)
        reproduced = predict_with_fitted_dense72_model(
            loaded,
            raw[validation_ids],
            station[validation_ids],
            static[validation_ids],
            device="cpu",
        )
        np.testing.assert_array_equal(first_prediction, reproduced)
        assert loaded.train_target_sha256 == first_fit.train_target_sha256
    finally:
        torch.set_num_threads(previous_threads)
