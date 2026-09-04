from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_kramers_moyal_residual_cycle_20260901_v42.py"
SPEC = importlib.util.spec_from_file_location("p3_v42", RUNNER)
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
    first = MODULE.kramers_moyal_features(sample_sequence())
    second = MODULE.kramers_moyal_features(sample_sequence())
    assert first.shape == (80,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_mean_reverting_synthetic_drift() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["mean_reverting_low_bin_drift"] > 0
    assert receipt["mean_reverting_high_bin_drift"] < 0


def test_positive_affine_invariance() -> None:
    sequence = sample_sequence()
    transformed = sequence.copy()
    for channel in MODULE.CHANNELS:
        transformed[:, channel] = 3.0 * transformed[:, channel] + 5.0
    assert np.allclose(MODULE.kramers_moyal_features(sequence), MODULE.kramers_moyal_features(transformed), atol=1e-12)


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(MODULE.kramers_moyal_features(first[:289]), MODULE.kramers_moyal_features(second[:289]))


def test_sealed_nonduplicate_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_STATE_CONDITIONAL_INCREMENT_MOMENT_AXIS"
    assert config["encoder"]["feature_count"] == 80
    assert config["encoder"]["increment_lag_rows"] == 1
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_fold_purge_and_exact_fit_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, targets, reference, _ = MODULE.v41.v40.v39.v38.v36.v32.v23.case_surface()
    calls: list[str] = []

    def fake_fit_predict(features, residual, train, valid, spec):
        calls.append(spec.name)
        return np.zeros((int(valid.sum()), 6)), {"candidate": spec.name, "row_deletion": 0}

    monkeypatch.setattr(MODULE.v41.v40.v39.v38.v36.v32.v26, "fit_predict", fake_fit_predict)
    original = MODULE.v41.v40.v39.v38.v36.v32.v28.SPECS
    MODULE.v41.v40.v39.v38.v36.v32.v28.SPECS = MODULE.SPECS
    try:
        _, receipts = MODULE.v41.v40.v39.v38.v36.v32.v28.crossfit(cases, np.zeros((len(cases), 80)), targets, reference)
    finally:
        MODULE.v41.v40.v39.v38.v36.v32.v28.SPECS = original
    assert len(calls) == len(receipts) == 12


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] in {"READY_EXACTLY_ONCE", "STOP_SUPPORT_GATE"}
        assert value["synthetic"]["feature_count"] == 80
        assert value["real_support"]["target_used_for_feature_gate"] is False
        assert value["official_access"] == 0
