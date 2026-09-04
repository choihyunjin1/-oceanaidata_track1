from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_discrete_trajectory_geometry_residual_cycle_20260901_v47.py"
SPEC = importlib.util.spec_from_file_location("p3_v47", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample_sequence() -> np.ndarray:
    rng = np.random.default_rng(20260901)
    values = rng.normal(size=(289, 10))
    values[:, 3] = np.linspace(0.0, 359.0, 289)
    values[:, 6] = np.linspace(359.0, 0.0, 289)
    return values


def test_feature_shape_finite_deterministic() -> None:
    first = MODULE.trajectory_geometry_features(sample_sequence())
    second = MODULE.trajectory_geometry_features(sample_sequence())
    assert first.shape == (72,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_line_circle_helix_geometry_guard() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["line_mean_curvature_log1p"] < 1e-10
    assert receipt["circle_mean_curvature_log1p"] > 0.1
    assert receipt["circle_mean_abs_torsion"] < 1e-10
    assert receipt["helix_mean_abs_torsion"] > 1e-4


def test_translation_rotation_invariance() -> None:
    rng = np.random.default_rng(7)
    path = rng.normal(size=(99, 3)).cumsum(axis=0)
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(MODULE.trajectory_statistics(path), MODULE.trajectory_statistics(path @ rotation.T + 5.0), atol=1e-12)


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(MODULE.trajectory_geometry_features(first[:289]), MODULE.trajectory_geometry_features(second[:289]))


def test_sealed_nonduplicate_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_DISCRETE_3D_TRAJECTORY_GEOMETRY_AXIS"
    assert len(config["encoder"]["trajectories"]) == 4
    assert config["encoder"]["windows_rows"] == [145, 73]
    assert config["encoder"]["feature_count"] == 72
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_fold_purge_and_exact_fit_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, targets, reference, _ = MODULE.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
    calls: list[str] = []

    def fake_fit_predict(features, residual, train, valid, spec):
        calls.append(spec.name)
        return np.zeros((int(valid.sum()), 6)), {"candidate": spec.name, "row_deletion": 0}

    monkeypatch.setattr(MODULE.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v26, "fit_predict", fake_fit_predict)
    original = MODULE.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v28.SPECS
    MODULE.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v28.SPECS = MODULE.SPECS
    try:
        _, receipts = MODULE.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v28.crossfit(cases, np.zeros((len(cases), 72)), targets, reference)
    finally:
        MODULE.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v28.SPECS = original
    assert len(calls) == len(receipts) == 12


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "READY_EXACTLY_ONCE"
        assert value["synthetic"]["feature_count"] == 72
        assert value["historical_support"]["passed"]
        assert value["maximum_model_fits"] == 12
        assert value["official_access"] == 0
