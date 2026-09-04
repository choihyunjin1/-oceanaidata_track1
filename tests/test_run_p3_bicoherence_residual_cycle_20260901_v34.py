from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_bicoherence_residual_cycle_20260901_v34.py"
SPEC = importlib.util.spec_from_file_location("p3_v34", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample_sequence() -> np.ndarray:
    rng = np.random.default_rng(20260901)
    values = rng.normal(size=(289, 10))
    values[:, 3] = np.linspace(0.0, 359.0, 289)
    values[:, 6] = np.linspace(359.0, 0.0, 289)
    return values


def test_feature_shape_finite_and_deterministic() -> None:
    sequence = sample_sequence()
    first = MODULE.bicoherence_features(sequence)
    second = MODULE.bicoherence_features(sequence)
    assert first.shape == (72,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.all(first.reshape(-1, 3)[:, 0] <= 1.0 + 1e-12)


def test_known_quadratic_phase_coupling_is_detected() -> None:
    time = np.arange(145, dtype=np.float64)
    coupled = (
        np.sin(2.0 * np.pi * 2.0 * time / 32.0)
        + np.sin(2.0 * np.pi * 3.0 * time / 32.0)
        + 0.8 * np.sin(2.0 * np.pi * 5.0 * time / 32.0)
    )
    spectra = MODULE.detrended_spectra(coupled, 32)
    value = MODULE.normalized_bispectrum(spectra, 2, 3)
    assert abs(value) > 0.75


def test_positive_affine_invariance_of_hs_features() -> None:
    sequence = sample_sequence()
    transformed = sequence.copy()
    transformed[:, 0] = 3.0 * transformed[:, 0] + 8.0
    assert np.allclose(
        MODULE.bicoherence_features(sequence),
        MODULE.bicoherence_features(transformed),
        atol=1e-11,
        rtol=1e-11,
    )


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(
        MODULE.bicoherence_features(first[:289]),
        MODULE.bicoherence_features(second[:289]),
    )


def test_sealed_nonduplicate_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == (
        "NON_DUPLICATE_QUADRATIC_PHASE_COUPLING_AXIS"
    )
    assert not config["duplication_audit"]["posthoc_prior_cycle_adjustment"]
    assert config["encoder"]["segment_lengths"] == [32, 64]
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_fold_purge_and_exact_fit_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, targets, reference, _ = MODULE.v32.v23.case_surface()
    calls: list[str] = []

    def fake_fit_predict(features, residual, train, valid, spec):
        calls.append(spec.name)
        return np.zeros((int(valid.sum()), 6)), {
            "candidate": spec.name,
            "row_deletion": 0,
        }

    monkeypatch.setattr(MODULE.v32.v26, "fit_predict", fake_fit_predict)
    original = MODULE.v32.v28.SPECS
    MODULE.v32.v28.SPECS = MODULE.SPECS
    try:
        _, receipts = MODULE.v32.v28.crossfit(
            cases, np.zeros((len(cases), 72)), targets, reference
        )
    finally:
        MODULE.v32.v28.SPECS = original
    assert len(calls) == len(receipts) == 12


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "READY_EXACTLY_ONCE"
        assert value["synthetic"]["feature_count"] == 72
        assert value["synthetic"]["finite"]
        assert value["maximum_model_fits"] == 12
        assert value["official_access"] == 0
