from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from p3_wave.causal_spectral_kernel import (
    CausalSpectralKernelConfig,
    TrainOnlyRobustScaler,
    build_causal_spectral_features,
    fit_and_predict_causal_spectral_kernel,
    load_fitted_causal_spectral_kernel,
    predict_causal_spectral_kernel,
    save_fitted_causal_spectral_kernel,
)


def _raw(batch: int = 30) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(batch, 289, 10)).astype(np.float32)
    raw[:, 1::2, :4] = np.nan
    raw[:, ::2, 0] = 2.0 + 0.2 * rng.normal(size=(batch, 145))
    raw[:, ::2, 1] = 7.0 + 0.3 * rng.normal(size=(batch, 145))
    raw[:, ::2, 2] = 3.0 + 0.3 * rng.normal(size=(batch, 145))
    raw[:, ::2, 3] = rng.uniform(0.0, 360.0, size=(batch, 145))
    raw[:, :, 6] = rng.uniform(0.0, 360.0, size=(batch, 289))
    raw[::4, :13, 5] = np.nan
    station = np.arange(batch, dtype=np.int64) % 3
    return raw, station


def test_feature_contract_is_finite_and_station_explicit() -> None:
    raw, station = _raw()
    features, names = build_causal_spectral_features(raw, station)
    assert features.shape == (30, 435)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert len(names) == 435
    assert len(set(names)) == 435
    np.testing.assert_array_equal(features[:, -3:], np.eye(3)[station])


def test_structural_wave_rows_fail_closed() -> None:
    raw, station = _raw()
    raw[0, 1, 0] = 2.0
    with pytest.raises(ValueError, match="structural ten-minute"):
        build_causal_spectral_features(raw, station)


def test_train_only_scaler_rejects_forbidden_and_wrong_reuse() -> None:
    raw, station = _raw()
    features, names = build_causal_spectral_features(raw, station)
    train_ids = np.arange(20, dtype=np.int64)
    validation_ids = np.arange(20, 30, dtype=np.int64)
    scaler = TrainOnlyRobustScaler.fit(features, train_ids, forbidden_ids=validation_ids)
    target = np.random.default_rng(31).normal(0.0, 0.2, size=(30, 6))
    weight = np.ones(30, dtype=np.float64)
    with pytest.raises(PermissionError, match="exact training IDs"):
        fit_and_predict_causal_spectral_kernel(
            features,
            target[np.arange(19, dtype=np.int64)],
            weight[np.arange(19, dtype=np.int64)],
            np.arange(19, dtype=np.int64),
            validation_ids,
            seed=3,
            scaler=scaler,
            forbidden_ids=validation_ids,
            names=names,
        )
    with pytest.raises(PermissionError, match="overlap"):
        TrainOnlyRobustScaler.fit(features, train_ids, forbidden_ids=train_ids[-1:])
    with pytest.raises(PermissionError, match="overlap"):
        fit_and_predict_causal_spectral_kernel(
            features,
            target[train_ids],
            weight[train_ids],
            train_ids,
            train_ids[-2:],
            seed=3,
            forbidden_ids=np.array([], dtype=np.int64),
            names=names,
        )


def test_fit_reload_reproduces_exact_prediction(tmp_path: Path) -> None:
    raw, station = _raw()
    features, names = build_causal_spectral_features(raw, station)
    rng = np.random.default_rng(29)
    target = rng.normal(0.0, 0.2, size=(30, 6))
    weight = np.linspace(0.7, 1.3, 30)
    train_ids = np.arange(24, dtype=np.int64)
    prediction_ids = np.arange(24, 30, dtype=np.int64)
    prediction, fitted = fit_and_predict_causal_spectral_kernel(
        features,
        target[train_ids],
        weight[train_ids],
        train_ids,
        prediction_ids,
        seed=20260816,
        config=CausalSpectralKernelConfig(),
        forbidden_ids=prediction_ids,
        names=names,
    )
    path = tmp_path / "model.npz"
    save_fitted_causal_spectral_kernel(fitted, path)
    reloaded = load_fitted_causal_spectral_kernel(path)
    reproduced = predict_causal_spectral_kernel(reloaded, features, prediction_ids, names=names)
    assert np.array_equal(prediction, reproduced)
    assert reloaded.train_ids_sha256 == fitted.train_ids_sha256
    assert reloaded.feature_names_sha256 == fitted.feature_names_sha256
    with pytest.raises(PermissionError, match="feature-name identity"):
        predict_causal_spectral_kernel(
            reloaded,
            features,
            prediction_ids,
            names=tuple(reversed(names)),
        )


def test_model_is_seeded_and_fixed_width() -> None:
    raw, station = _raw()
    features, names = build_causal_spectral_features(raw, station)
    target = np.random.default_rng(31).normal(0.0, 0.2, size=(30, 6))
    weight = np.ones(30, dtype=np.float64)
    train_ids = np.arange(24, dtype=np.int64)
    prediction_ids = np.arange(24, 30, dtype=np.int64)
    first, _ = fit_and_predict_causal_spectral_kernel(
        features,
        target[train_ids],
        weight[train_ids],
        train_ids,
        prediction_ids,
        seed=3,
        forbidden_ids=prediction_ids,
        names=names,
    )
    second, _ = fit_and_predict_causal_spectral_kernel(
        features,
        target[train_ids],
        weight[train_ids],
        train_ids,
        prediction_ids,
        seed=5,
        forbidden_ids=prediction_ids,
        names=names,
    )
    assert first.shape == (6, 6)
    assert not np.array_equal(first, second)
    with pytest.raises(ValueError, match="frozen at 128"):
        CausalSpectralKernelConfig(random_feature_count=64)


def test_validation_targets_cannot_affect_fit_and_same_seed_is_exact() -> None:
    raw, station = _raw()
    features, names = build_causal_spectral_features(raw, station)
    rng = np.random.default_rng(37)
    target = rng.normal(0.0, 0.2, size=(30, 6))
    perturbed = target.copy()
    train_ids = np.arange(24, dtype=np.int64)
    prediction_ids = np.arange(24, 30, dtype=np.int64)
    perturbed[prediction_ids] = np.nan
    weight = np.ones(30, dtype=np.float64)
    first, first_model = fit_and_predict_causal_spectral_kernel(
        features,
        target[train_ids],
        weight[train_ids],
        train_ids,
        prediction_ids,
        seed=20260816,
        names=names,
    )
    second, second_model = fit_and_predict_causal_spectral_kernel(
        features,
        perturbed[train_ids],
        weight[train_ids],
        train_ids,
        prediction_ids,
        seed=20260816,
        names=names,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first_model.frequency, second_model.frequency)
    assert np.array_equal(first_model.coefficient, second_model.coefficient)
    assert first_model.median_squared_distance == second_model.median_squared_distance
