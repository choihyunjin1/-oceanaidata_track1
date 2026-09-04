from __future__ import annotations

import numpy as np

from p3_wave.nlinear_ridge_residual import (
    StandardizedStationRidge,
    absolute_prediction,
    build_compact_features,
    compact_feature_names,
    protected_long_blend,
)


def test_compact_features_are_finite_and_past_only_shape() -> None:
    raw = np.zeros((2, 289, 10), dtype=np.float32)
    raw[0, :30, 0] = np.nan
    raw[1, :, 3] = np.nan
    features = build_compact_features(raw)
    assert features.shape == (2, len(compact_feature_names()))
    assert np.isfinite(features).all()


def test_station_ridge_fits_multioutput_delta() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(90, 12))
    station = np.repeat(np.arange(3), 30)
    target = np.column_stack([features[:, 0] + lead for lead in range(6)])
    rows = np.arange(90)
    model = StandardizedStationRidge.fit(features, station, target, rows, alpha=1.0)
    prediction = model.predict(features, station, rows)
    assert prediction.shape == (90, 6)
    assert np.isfinite(prediction).all()


def test_absolute_prediction_adds_current_hs() -> None:
    current = np.asarray([1.0, 2.0])
    delta = np.ones((2, 6)) * 0.5
    prediction = absolute_prediction(current, delta)
    assert np.allclose(prediction[0], 1.5)
    assert np.allclose(prediction[1], 2.5)


def test_protected_blend_preserves_early_leads_exactly() -> None:
    incumbent = np.arange(12, dtype=np.float64).reshape(2, 6)
    challenger = incumbent + 2.0
    blended = protected_long_blend(incumbent, challenger, long_weight=0.2)
    assert np.array_equal(blended[:, :3], incumbent[:, :3])
    assert np.allclose(blended[:, 3:], incumbent[:, 3:] + 0.4)
