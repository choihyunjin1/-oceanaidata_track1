from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_cross_quantilogram_residual_cycle_20260901_v38.py"
SPEC = importlib.util.spec_from_file_location("p3_v38", RUNNER)
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
    first = MODULE.cross_quantilogram_features(sequence)
    second = MODULE.cross_quantilogram_features(sequence)
    assert first.shape == (72,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.all(np.abs(first) <= 1.0)


def test_synthetic_directional_tail_hit_recovery() -> None:
    receipt = MODULE.lead_lag_receipt()
    assert receipt["source_to_target"] > 0.50
    assert receipt["directional_margin"] > 0.25


def test_positive_affine_quantile_hit_invariance() -> None:
    rng = np.random.default_rng(37)
    source = rng.normal(size=145)
    target = rng.normal(size=145)
    original = MODULE.cross_quantilogram(source, target, "upper", 0.8, 6)
    transformed = MODULE.cross_quantilogram(
        3.0 * source + 5.0, 4.0 * target - 2.0, "upper", 0.8, 6
    )
    assert abs(original - transformed) < 1e-12


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(
        MODULE.cross_quantilogram_features(first[:289]),
        MODULE.cross_quantilogram_features(second[:289]),
    )


def test_sealed_nonduplicate_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == (
        "NON_DUPLICATE_DIRECTIONAL_TAIL_HIT_AXIS"
    )
    assert not config["duplication_audit"]["posthoc_prior_cycle_adjustment"]
    assert config["encoder"]["lags_rows"] == [0, 6, 18]
    assert config["encoder"]["upper_quantiles"] == [0.8, 0.9]
    assert config["encoder"]["lower_quantiles"] == [0.2, 0.1]
    assert config["encoder"]["feature_count"] == 72
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_fold_purge_and_exact_fit_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, targets, reference, _ = MODULE.v36.v32.v23.case_surface()
    calls: list[str] = []

    def fake_fit_predict(features, residual, train, valid, spec):
        calls.append(spec.name)
        return np.zeros((int(valid.sum()), 6)), {
            "candidate": spec.name,
            "row_deletion": 0,
        }

    monkeypatch.setattr(MODULE.v36.v32.v26, "fit_predict", fake_fit_predict)
    original = MODULE.v36.v32.v28.SPECS
    MODULE.v36.v32.v28.SPECS = MODULE.SPECS
    try:
        _, receipts = MODULE.v36.v32.v28.crossfit(
            cases, np.zeros((len(cases), 72)), targets, reference
        )
    finally:
        MODULE.v36.v32.v28.SPECS = original
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
