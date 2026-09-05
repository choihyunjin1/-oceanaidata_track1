"""Synthetic-only geometry, leakage, marginalization and residual contracts."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_profile_v4", ROOT / "scripts/run_p2_profile_copula_residual_20260905_v4.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def observations():
    return pd.DataFrame([{"station": "S-ORS", "time": t, "layer": layer, "nominal_depth": layer * 5., "depth": layer * 5. + .5, "temp": 20 - layer * .3, "psal": 30 + layer * .1} for t in pd.date_range("2024-09-03", periods=8, freq="10min", tz="UTC") for layer in range(1, 9)])


def test_targets_never_enter_features():
    obs = observations()
    a, _ = M.base.public_frame(obs)
    obs.loc[obs.layer.isin([2, 3, 4]), ["temp", "psal"]] = [-1e6, 1e6]
    b, _ = M.base.public_frame(obs)
    np.testing.assert_array_equal(M.physical_features(a, np.ones(len(a))), M.physical_features(b, np.ones(len(b))))


def test_actual_interpolation_linear_temperature_and_gradient():
    frame, _ = M.base.public_frame(observations())
    target, interp, gradient = M.actual_profile(frame)
    np.testing.assert_allclose(interp, 20 - (target - .5) * .06)
    np.testing.assert_allclose(gradient, -.06)


def test_missing_depth_falls_back_to_nominal_duplicate_depth_finite():
    frame, _ = M.base.public_frame(observations())
    frame.target_actual_depth = np.nan
    frame.depth_1 = np.nan
    frame.depth_5 = frame.depth_6
    target, interp, gradient = M.actual_profile(frame)
    np.testing.assert_array_equal(target, frame.target_depth)
    assert np.isfinite(interp).all() and np.isfinite(gradient).all()


def test_episode_halfopen_zero_dependency():
    frame, _ = M.base.public_frame(observations())
    changed, selected = M.episode_frame(frame, {"start": "2024-09-03T00:20:00Z", "stop": "2024-09-03T00:50:00Z"})
    assert selected.sum() == 9
    assert changed.loc[selected, ["temp_5", "psal_5"]].isna().all().all()
    a, b = M.physical_features(frame, np.ones(len(frame))), M.physical_features(changed, np.ones(len(frame)))
    np.testing.assert_array_equal(a[~selected], b[~selected])


def test_latent_ties_constant_and_missing():
    result = M.latent(np.array([1., 1., np.nan, -99, 99]), np.array([1., 1., 1.]))
    np.testing.assert_array_equal(result[:3], np.zeros(3))
    assert result[3] < 0 < result[4] and np.isfinite(result).all()


def test_train_only_finite_marginalized_predict_and_model_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(150, 3))
    r = x[:, 0] * .3 + rng.normal(size=150) * .1
    x[::5, 1] = np.nan
    model = M.fit_copula(x, r)
    before = model["covariance"].copy()
    query = np.array([[0., np.nan, 1.], [np.nan, np.nan, np.nan], [1e6, -1e6, 1e6]])
    predicted = M.predict_copula(model, query)
    assert np.isfinite(predicted).all() and predicted.min() >= r.min() and predicted.max() <= r.max()
    np.testing.assert_array_equal(before, model["covariance"])
    path = tmp_path / "model.npz"
    np.savez(path, **model)
    np.testing.assert_array_equal(predicted, M.predict_copula(dict(np.load(path)), query))


def test_zero_residual_noop_and_fixed_half():
    x = np.arange(80.).reshape(40, 2)
    model = M.fit_copula(x, np.zeros(40))
    np.testing.assert_array_equal(M.predict_copula(model, x), np.zeros(40))
    cfg = M.read_config()
    p = M.policies(np.array([1., 2.]), np.array([2., -4.]), cfg)
    np.testing.assert_array_equal(p["C"], [1, 2])
    np.testing.assert_array_equal(p["half"], [2, 0])
    assert cfg["deterministic_unique_residual_fits"] <= cfg["maximum_new_historical_fits"]


def test_training_rejects_nonfinite_response():
    with pytest.raises(ValueError):
        M.fit_copula(np.ones((3, 2)), np.array([1, 2, np.nan]))


def test_same_denominator_sse():
    value = M.metric_table(np.zeros(3), {"C": np.array([0, 0, 3])}, {"pooled": np.ones(3, bool)})
    assert value["pooled"]["n"] == 3
    assert value["pooled"]["metrics"]["C"]["sse"] == 9
    assert value["pooled"]["metrics"]["C"]["rmse"] == pytest.approx(np.sqrt(3))


def test_purge_and_source_contract():
    cfg = M.read_config()
    original = M.previous.load_config()
    assert cfg["source_sha256"] == original["source_sha256"]
    assert cfg["purge_days"] == 7 and cfg["feature_dependency_hours"] == 0
    assert cfg["new_backbone_fits"] == cfg["new_full_fits"] == 0
    code = Path(M.__file__).read_text(encoding="utf-8")
    assert ".cuda(" not in code and "to_csv(" not in code and "pd.read_csv(" not in code
