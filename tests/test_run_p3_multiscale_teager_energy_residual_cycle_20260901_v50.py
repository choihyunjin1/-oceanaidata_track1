from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_multiscale_teager_energy_residual_cycle_20260901_v50.py"
SPEC = importlib.util.spec_from_file_location("p3_v50", RUNNER)
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
    first = MODULE.multiscale_teager_features(sample_sequence())
    second = MODULE.multiscale_teager_features(sample_sequence())
    assert first.shape == (60,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_constant_sinusoid_amplitude_guards() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["constant_max_abs"] < 1e-12
    assert receipt["sinusoid_max_abs_error"] < 1e-10
    assert receipt["amplitude_quadratic_scaling_error"] < 1e-10


def test_sign_invariance_and_amplitude_scaling() -> None:
    rng = np.random.default_rng(7)
    path = rng.normal(size=145)
    for lag in MODULE.LAGS:
        energy = MODULE.teager_energy(path, lag)
        assert np.array_equal(energy, MODULE.teager_energy(-path, lag))
        assert np.allclose(MODULE.teager_energy(3.0 * path, lag), 9.0 * energy, atol=1e-12)


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(MODULE.multiscale_teager_features(first[:289]), MODULE.multiscale_teager_features(second[:289]))


def test_sealed_cross_problem_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_MULTISCALE_TEAGER_SIGNAL_ENERGY_AXIS"
    assert "P1 v30" in config["duplication_audit"]["cross_problem_note"]
    assert config["encoder"]["lags_rows"] == [1, 3, 6]
    assert config["encoder"]["feature_count"] == 60
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_fold_purge_and_exact_fit_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, targets, reference, _ = MODULE.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
    calls: list[str] = []

    def fake_fit_predict(features, residual, train, valid, spec):
        calls.append(spec.name)
        return np.zeros((int(valid.sum()), 6)), {"candidate": spec.name, "row_deletion": 0}

    monkeypatch.setattr(MODULE.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v26, "fit_predict", fake_fit_predict)
    original = MODULE.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v28.SPECS
    MODULE.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v28.SPECS = MODULE.SPECS
    try:
        _, receipts = MODULE.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v28.crossfit(cases, np.zeros((len(cases), 60)), targets, reference)
    finally:
        MODULE.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v28.SPECS = original
    assert len(calls) == len(receipts) == 12


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "READY_EXACTLY_ONCE"
        assert value["synthetic"]["feature_count"] == 60
        assert value["historical_support"]["passed"]
        assert value["maximum_model_fits"] == 12
        assert value["official_access"] == 0
