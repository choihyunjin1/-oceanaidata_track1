from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_fixed_echo_state_residual_cycle_20260901_v41.py"
SPEC = importlib.util.spec_from_file_location("p3_v41", RUNNER)
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
    first = MODULE.echo_state_features(sample_sequence())
    second = MODULE.echo_state_features(sample_sequence())
    assert first.shape == (96,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_fixed_operator_and_fading_memory() -> None:
    receipt = MODULE.fading_memory_receipt()
    assert abs(receipt["operator_spectral_radius"] - 0.9) < 1e-12
    assert receipt["initial_impulse_difference"] > 0
    assert receipt["terminal_to_initial_ratio"] < 1e-3


def test_positive_affine_case_scaling_invariance() -> None:
    sequence = sample_sequence()
    transformed = 3.0 * sequence + 5.0
    transformed[:, 3] = sequence[:, 3]
    transformed[:, 6] = sequence[:, 6]
    assert np.allclose(
        MODULE.echo_state_features(sequence),
        MODULE.echo_state_features(transformed),
        atol=1e-12,
    )


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(
        MODULE.echo_state_features(first[:289]),
        MODULE.echo_state_features(second[:289]),
    )


def test_sealed_nonduplicate_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == (
        "NON_DUPLICATE_FIXED_NONLINEAR_RECURRENT_STATE_AXIS"
    )
    assert config["encoder"]["reservoir_states"] == 32
    assert config["encoder"]["seed"] == 20260901
    assert config["encoder"]["spectral_radius"] == 0.9
    assert config["encoder"]["leak_rate"] == 0.3
    assert config["encoder"]["feature_count"] == 96
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_fold_purge_and_exact_fit_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, targets, reference, _ = MODULE.v40.v39.v38.v36.v32.v23.case_surface()
    calls: list[str] = []

    def fake_fit_predict(features, residual, train, valid, spec):
        calls.append(spec.name)
        return np.zeros((int(valid.sum()), 6)), {"candidate": spec.name, "row_deletion": 0}

    monkeypatch.setattr(MODULE.v40.v39.v38.v36.v32.v26, "fit_predict", fake_fit_predict)
    original = MODULE.v40.v39.v38.v36.v32.v28.SPECS
    MODULE.v40.v39.v38.v36.v32.v28.SPECS = MODULE.SPECS
    try:
        _, receipts = MODULE.v40.v39.v38.v36.v32.v28.crossfit(cases, np.zeros((len(cases), 96)), targets, reference)
    finally:
        MODULE.v40.v39.v38.v36.v32.v28.SPECS = original
    assert len(calls) == len(receipts) == 12


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "READY_EXACTLY_ONCE"
        assert value["synthetic"]["feature_count"] == 96
        assert value["synthetic"]["finite"]
        assert value["maximum_model_fits"] == 12
        assert value["official_access"] == 0
