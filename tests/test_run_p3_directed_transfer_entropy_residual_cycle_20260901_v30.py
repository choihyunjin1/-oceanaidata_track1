from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_directed_transfer_entropy_residual_cycle_20260901_v30.py"
SPEC = importlib.util.spec_from_file_location("p3_v30", RUNNER)
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
    first = MODULE.directed_information_features(sequence)
    second = MODULE.directed_information_features(sequence)
    assert first.shape == (180,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_transfer_entropy_nonnegative_and_identical_net_zero() -> None:
    values = MODULE.symbols(np.sin(np.linspace(0.0, 20.0, 289)))
    for lag in MODULE.LAGS:
        forward = MODULE.transfer_entropy(values, values, lag)
        reverse = MODULE.transfer_entropy(values, values, lag)
        assert forward >= -1e-15
        assert forward == reverse


def test_symbols_are_positive_affine_invariant() -> None:
    values = np.random.default_rng(7).normal(size=289)
    assert np.array_equal(MODULE.symbols(values), MODULE.symbols(3.0 * values + 8.0))


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(
        MODULE.directed_information_features(first[:289]),
        MODULE.directed_information_features(second[:289]),
    )


def test_sealed_nonduplicate_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == (
        "NON_DUPLICATE_DIRECTED_INFORMATION_TRANSFER_AXIS"
    )
    assert not config["duplication_audit"]["posthoc_v27_v28_v29_adjustment"]
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_fold_purge_and_exact_fit_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, targets, reference, _ = MODULE.v23.case_surface()
    for block in MODULE.v23.BLOCKS:
        valid = cases["block"].eq(block).to_numpy()
        train = MODULE.v23.purged_train_indices(cases, valid)
        assert not np.any(valid[train])
    calls: list[str] = []

    def fake_fit_predict(features, residual, train, valid, spec):
        calls.append(spec.name)
        return np.zeros((int(valid.sum()), 6)), {
            "candidate": spec.name,
            "row_deletion": 0,
        }

    monkeypatch.setattr(MODULE.v26, "fit_predict", fake_fit_predict)
    original = MODULE.v28.SPECS
    MODULE.v28.SPECS = MODULE.SPECS
    try:
        _, receipts = MODULE.v28.crossfit(
            cases, np.zeros((len(cases), 180)), targets, reference
        )
    finally:
        MODULE.v28.SPECS = original
    assert len(calls) == len(receipts) == 12


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "READY_EXACTLY_ONCE"
        assert value["maximum_model_fits"] == 12
        assert value["official_access"] == 0
