from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_detrended_cross_correlation_residual_cycle_20260901_v33.py"
SPEC = importlib.util.spec_from_file_location("p3_v33", RUNNER)
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
    first = MODULE.dcca_features(sequence)
    second = MODULE.dcca_features(sequence)
    assert first.shape == (66,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_pair_symmetry() -> None:
    rng = np.random.default_rng(7)
    first = rng.normal(size=289)
    second = 0.4 * first + rng.normal(size=289)
    assert np.allclose(
        MODULE.dcca_pair_features(first, second),
        MODULE.dcca_pair_features(second, first),
        atol=1e-12,
        rtol=1e-12,
    )


def test_pair_positive_affine_invariance() -> None:
    rng = np.random.default_rng(11)
    first = rng.normal(size=289)
    second = rng.normal(size=289)
    expected = MODULE.dcca_pair_features(first, second)
    observed = MODULE.dcca_pair_features(3.0 * first + 8.0, 2.0 * second - 5.0)
    assert np.allclose(expected, observed, atol=1e-10, rtol=1e-10)


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(
        MODULE.dcca_features(first[:289]), MODULE.dcca_features(second[:289])
    )


def test_sealed_nonduplicate_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == (
        "NON_DUPLICATE_DETRENDED_CROSS_SCALE_COVARIANCE_AXIS"
    )
    assert not config["duplication_audit"]["posthoc_prior_cycle_adjustment"]
    assert config["encoder"]["fixed_scales"] == [12, 24, 48, 96]
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
            cases, np.zeros((len(cases), 66)), targets, reference
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
        assert value["synthetic"]["feature_count"] == 66
        assert value["synthetic"]["finite"]
        assert value["maximum_model_fits"] == 12
        assert value["official_access"] == 0
