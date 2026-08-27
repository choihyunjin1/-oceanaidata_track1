from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from p3_wave.stable_energy_state_space import (
    StableEnergyStateSpaceConfig,
    TrainOnlyStateScaler,
    build_wave_energy_state_sequences,
    fit_stable_energy_state_space,
    load_fitted_stable_energy_state_space,
    predict_stable_energy_state_space,
    save_fitted_stable_energy_state_space,
)


def _raw(batch: int = 36) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(101)
    time = np.arange(145, dtype=np.float32)
    raw = np.full((batch, 289, 10), np.nan, dtype=np.float32)
    for case in range(batch):
        phase = 0.08 * time + case * 0.03
        raw[case, ::2, 0] = 2.0 + 0.2 * np.sin(phase)
        raw[case, ::2, 1] = 7.0 + 0.3 * np.cos(phase * 0.7)
        raw[case, ::2, 2] = 3.2 + 0.25 * np.sin(phase * 1.1)
        raw[case, ::2, 3] = np.mod(40.0 + time * 1.2 + case, 360.0)
        raw[case, :, 4] = 5.0 + 0.5 * np.sin(np.arange(289) * 0.04 + case)
        raw[case, :, 5] = raw[case, :, 4] + 1.0
        raw[case, :, 6] = np.mod(70.0 + np.arange(289) * 0.7, 360.0)
        raw[case, :, 7] = 16.0 + rng.normal(0.0, 0.1, 289)
        raw[case, :, 8] = 70.0 + rng.normal(0.0, 0.3, 289)
        raw[case, :, 9] = 1010.0 + rng.normal(0.0, 0.2, 289)
    station = np.arange(batch, dtype=np.int64) % 3
    anchor_time_ns = (
        np.arange(batch, dtype=np.int64) * 72 * 60 * 60 * 1_000_000_000 + 1_700_000_000_000_000_000
    )
    return raw, station, anchor_time_ns


def test_state_builder_preserves_native_order_and_finite_current() -> None:
    raw, _, _ = _raw()
    states = build_wave_energy_state_sequences(raw)
    assert states.shape == (36, 145, 12)
    assert states.dtype == np.float32
    assert np.isfinite(states[:, -1, 0]).all()
    expected = np.log1p(np.square(raw[:, -1, 0].astype(np.float64))).astype(np.float32)
    np.testing.assert_array_equal(states[:, -1, 0], expected)


def test_structural_wave_row_violation_fails_closed() -> None:
    raw, _, _ = _raw()
    raw[0, 1, 0] = 2.0
    with pytest.raises(ValueError, match="structural ten-minute"):
        build_wave_energy_state_sequences(raw)


def test_scaler_and_transition_fit_reject_validation_overlap() -> None:
    raw, station, times = _raw()
    states = build_wave_energy_state_sequences(raw)
    train_ids = np.arange(30, dtype=np.int64)
    validation_ids = np.arange(30, 36, dtype=np.int64)
    scaler = TrainOnlyStateScaler.fit(states, train_ids, forbidden_ids=validation_ids)
    with pytest.raises(PermissionError, match="overlap"):
        fit_stable_energy_state_space(
            states,
            station,
            times,
            train_ids,
            scaler=scaler,
            forbidden_ids=train_ids[-1:],
        )


def test_fit_is_stable_and_reload_prediction_is_exact(tmp_path: Path) -> None:
    raw, station, times = _raw()
    states = build_wave_energy_state_sequences(raw)
    train_ids = np.arange(30, dtype=np.int64)
    prediction_ids = np.arange(30, 36, dtype=np.int64)
    fitted, receipt = fit_stable_energy_state_space(
        states,
        station,
        times,
        train_ids,
        forbidden_ids=prediction_ids,
    )
    assert receipt["validation_ids_used"] == 0
    assert receipt["unique_station_time_transitions"] > 1_000
    assert np.all(fitted.spectral_radius_after <= 0.995 + 1e-10)
    prediction = predict_stable_energy_state_space(fitted, states, station, prediction_ids)
    assert prediction.shape == (6, 6)
    assert np.isfinite(prediction).all()
    assert np.all((prediction >= 0.0) & (prediction <= 30.0))
    path = tmp_path / "model.npz"
    save_fitted_stable_energy_state_space(fitted, path)
    loaded = load_fitted_stable_energy_state_space(path)
    reproduced = predict_stable_energy_state_space(loaded, states, station, prediction_ids)
    assert np.array_equal(prediction, reproduced)
    assert loaded.transition_key_sha256 == fitted.transition_key_sha256
    with pytest.raises(PermissionError, match="prediction IDs overlap"):
        predict_stable_energy_state_space(loaded, states, station, train_ids[:2])


def test_frozen_config_and_deterministic_repeat() -> None:
    raw, station, times = _raw()
    states = build_wave_energy_state_sequences(raw)
    train_ids = np.arange(30, dtype=np.int64)
    prediction_ids = np.arange(30, 36, dtype=np.int64)
    first, _ = fit_stable_energy_state_space(
        states, station, times, train_ids, forbidden_ids=prediction_ids
    )
    second, _ = fit_stable_energy_state_space(
        states, station, times, train_ids, forbidden_ids=prediction_ids
    )
    assert np.array_equal(first.transition, second.transition)
    assert np.array_equal(first.intercept, second.intercept)
    with pytest.raises(ValueError, match="constants are frozen"):
        StableEnergyStateSpaceConfig(global_ridge=2e-3)
