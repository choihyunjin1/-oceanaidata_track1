from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_rosenstein_local_divergence_residual_cycle_20260901_v55.py"
SPEC = importlib.util.spec_from_file_location("p3_v55", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample_sequence() -> np.ndarray:
    rng = np.random.default_rng(55)
    values = rng.normal(size=(289, 10))
    values[:, 3] = np.linspace(0.0, 359.0, 289)
    values[:, 6] = np.linspace(359.0, 0.0, 289)
    return values


def test_feature_shape_finite_and_deterministic() -> None:
    sequence = sample_sequence()
    first = MODULE.local_divergence_features(sequence)
    second = MODULE.local_divergence_features(sequence)
    assert first.shape == (72,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_periodic_vs_chaotic_direction_and_neighbor_support() -> None:
    receipt = MODULE.synthetic_receipt()
    assert receipt["chaotic_slope"] > receipt["periodic_slope"] + 0.01
    assert receipt["periodic_neighbor_fraction"] >= 0.70
    assert receipt["chaotic_neighbor_fraction"] >= 0.70


def test_positive_affine_invariance() -> None:
    rng = np.random.default_rng(550)
    values = rng.normal(size=145)
    assert np.allclose(MODULE.local_divergence_statistics(values), MODULE.local_divergence_statistics(3.0 * values + 4.0), atol=1e-10, rtol=1e-10)


def test_future_perturbation_invariance() -> None:
    history = sample_sequence()
    first = np.vstack([history, np.zeros((11, 10))])
    second = first.copy()
    second[289:] = 1e9
    assert np.array_equal(MODULE.local_divergence_features(first[:289]), MODULE.local_divergence_features(second[:289]))


def test_sealed_nonduplicate_contract() -> None:
    config = MODULE.load_config()
    encoder = config["encoder"]
    assert config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_FORWARD_NEIGHBOR_DIVERGENCE_AXIS"
    assert (encoder["embedding_dimension"], encoder["embedding_delay_rows"], encoder["temporal_exclusion_rows"]) == (3, 2, 12)
    assert tuple(encoder["forward_horizons_rows"]) == (1, 2, 3, 4, 5, 6)
    assert encoder["feature_count"] == 72
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())


def test_fold_purge_and_exact_fit_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    base = MODULE.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v28
    cases, targets, reference, _ = MODULE.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
    calls: list[str] = []

    def fake_fit_predict(features, residual, train, valid, spec):
        calls.append(spec.name)
        return np.zeros((int(valid.sum()), 6)), {"candidate": spec.name, "row_deletion": 0}

    monkeypatch.setattr(base.v26, "fit_predict", fake_fit_predict)
    original = base.SPECS
    base.SPECS = MODULE.SPECS
    try:
        _, receipts = base.crossfit(cases, np.zeros((len(cases), 72)), targets, reference)
    finally:
        base.SPECS = original
    assert len(calls) == len(receipts) == 12


def test_preflight_or_consumed_namespace() -> None:
    if MODULE.ARTIFACT.exists() or MODULE.LOCK.exists():
        with pytest.raises(MODULE.ContractError, match="consumed"):
            MODULE.preflight_payload()
    else:
        value = MODULE.preflight_payload()
        assert value["status"] == "READY_EXACTLY_ONCE"
        assert value["synthetic"]["feature_count"] == 72
        assert value["maximum_model_fits"] == 12
        assert value["official_access"] == 0
