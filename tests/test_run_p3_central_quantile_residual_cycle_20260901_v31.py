from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_central_quantile_residual_cycle_20260901_v31.py"
SPEC = importlib.util.spec_from_file_location("p3_v31", RUNNER)
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
    first = MODULE.case_statistics(sequence)
    second = MODULE.case_statistics(sequence)
    assert first.shape == (108,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_constant_slope_is_zero() -> None:
    assert MODULE._slope(np.ones(24)) == pytest.approx(0.0)


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(
        MODULE.case_statistics(first[:289]), MODULE.case_statistics(second[:289])
    )


def test_row_feature_contract() -> None:
    cases, _, _, _ = MODULE.v23.case_surface()
    matrix = MODULE.row_features(cases, np.zeros((len(cases), 108)))
    assert matrix.shape == (1092, 117)
    assert np.isfinite(matrix).all()


def test_sealed_nonduplicate_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == (
        "NON_DUPLICATE_DISTRIBUTIONAL_TARGET_AXIS"
    )
    assert not config["duplication_audit"]["posthoc_v27_v28_v29_v30_adjustment"]
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_fold_purge_and_exact_fit_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, targets, reference, _ = MODULE.v23.case_surface()
    calls: list[float] = []

    class FakeQuantile:
        def __init__(self, quantile, alpha, fit_intercept, solver):
            del alpha, fit_intercept, solver
            self.quantile = quantile

        def fit(self, features, target):
            calls.append(self.quantile)
            self.value = float(np.median(target))
            return self

        def predict(self, features):
            return np.full(len(features), self.value)

    monkeypatch.setattr(MODULE, "QuantileRegressor", FakeQuantile)
    prediction, receipts = MODULE.crossfit(
        cases, np.zeros((len(cases), 108)), targets, reference
    )
    assert prediction.shape == targets.shape
    assert len(calls) == len(receipts) == 12
    assert all(receipt["row_deletion"] == 0 for receipt in receipts)


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "READY_EXACTLY_ONCE"
        assert value["maximum_model_fits"] == 12
        assert value["official_access"] == 0
