from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_nonlinear_recurrence_ordinal_residual_cycle_20260901_v29.py"
SPEC = importlib.util.spec_from_file_location("p3_v29", RUNNER)
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
    first = MODULE.topology_features(sequence)
    second = MODULE.topology_features(sequence)
    assert first.shape == (360,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_ordinal_probabilities_sum_to_one() -> None:
    values = np.sin(np.linspace(0.0, 20.0, 289))
    for delay in MODULE.DELAYS:
        frequencies = MODULE.ordinal_frequencies(values, delay)
        assert frequencies.shape == (6,)
        assert np.isclose(frequencies.sum(), 1.0)
        assert np.all(frequencies >= 0.0)


def test_positive_affine_invariance_for_direct_channels() -> None:
    sequence = sample_sequence()
    changed = sequence.copy()
    direct = [0, 1, 2, 4, 5, 7, 8, 9]
    changed[:, direct] = changed[:, direct] * 2.0 + 3.0
    assert np.allclose(
        MODULE.topology_features(sequence), MODULE.topology_features(changed), atol=1e-12
    )


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(
        MODULE.topology_features(first[:289]), MODULE.topology_features(second[:289])
    )


def test_sealed_nonduplicate_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_NONLINEAR_RECURRENCE_TOPOLOGY_AXIS"
    assert not config["duplication_audit"]["posthoc_v27_or_v28_adjustment"]
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
        return np.zeros((int(valid.sum()), 6)), {"candidate": spec.name, "row_deletion": 0}

    monkeypatch.setattr(MODULE.v26, "fit_predict", fake_fit_predict)
    original = MODULE.v28.SPECS
    MODULE.v28.SPECS = MODULE.SPECS
    try:
        _, receipts = MODULE.v28.crossfit(
            cases, np.zeros((len(cases), 360)), targets, reference
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
