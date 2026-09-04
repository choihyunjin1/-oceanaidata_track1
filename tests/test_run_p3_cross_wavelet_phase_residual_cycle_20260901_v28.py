from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_cross_wavelet_phase_residual_cycle_20260901_v28.py"
SPEC = importlib.util.spec_from_file_location("p3_v28", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_feature_shape_and_determinism() -> None:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    first = MODULE.phase_features(sequence)
    second = MODULE.phase_features(sequence)
    assert first.shape == (330,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_complex_wavelets_are_zero_mean_and_deterministic() -> None:
    for scale in MODULE.SCALES:
        first = MODULE.complex_wavelet(scale)
        second = MODULE.complex_wavelet(scale)
        assert np.allclose(first.mean(), 0.0, atol=1e-15)
        assert np.array_equal(first, second)


def test_circular_moments_are_common_phase_invariant() -> None:
    phase = np.linspace(-np.pi, np.pi, 97)
    reference = np.exp(1j * phase)
    other = np.exp(1j * (phase + 0.4))
    rotation = np.exp(1j * 1.7)
    assert np.allclose(
        MODULE.circular_moments(reference, other),
        MODULE.circular_moments(reference * rotation, other * rotation),
        atol=1e-15,
    )


def test_direction_wrapping_and_future_perturbation_invariance() -> None:
    rng = np.random.default_rng(20260901)
    full = rng.normal(size=(300, 10))
    full[:289, 3] = np.linspace(0.0, 359.0, 289)
    full[:289, 6] = np.linspace(359.0, 0.0, 289)
    wrapped = full.copy()
    wrapped[:289, (3, 6)] += 360.0
    future_changed = full.copy()
    future_changed[289:] = 1e9
    expected = MODULE.phase_features(full[:289])
    assert np.allclose(expected, MODULE.phase_features(wrapped[:289]), atol=1e-12)
    assert np.array_equal(expected, MODULE.phase_features(future_changed[:289]))


def test_sealed_contract() -> None:
    config = MODULE.load_config()
    assert config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_CROSS_WAVELET_PHASE_AXIS"
    assert not config["duplication_audit"]["posthoc_v27_adjustment"]
    assert config["validation"]["maximum_total_fits"] == 12
    assert all(value == 0 for value in config["official_policy"].values())
    assert config["validation"]["tail_guards"]["reference_tail_quantile_within_block"] == 0.80


def test_candidate_budget_is_exact() -> None:
    assert [(item.name, item.alpha) for item in MODULE.SPECS] == [
        ("P3_1_XWPHASE330_RIDGE512_ADD10", 512.0),
        ("P3_2_XWPHASE330_RIDGE2048_ADD10", 2048.0),
    ]
    assert MODULE.BLEND == 0.10


def test_fold_purge_and_exactly_twelve_fit_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, targets, reference, _ = MODULE.v23.case_surface()
    for block in MODULE.v23.BLOCKS:
        valid = cases["block"].eq(block).to_numpy()
        train = MODULE.v23.purged_train_indices(cases, valid)
        train_mask = np.zeros(len(cases), dtype=bool)
        train_mask[train] = True
        assert not np.any(train_mask & valid)
        for station in cases.loc[valid, "station"].unique():
            valid_time = cases.loc[valid & cases["station"].eq(station), "anchor_time"]
            train_time = cases.loc[
                train_mask & cases["station"].eq(station), "anchor_time"
            ]
            if len(valid_time) and len(train_time):
                minimum = min(
                    abs((left - right).total_seconds()) / 3600.0
                    for left in valid_time
                    for right in train_time
                )
                assert minimum >= 78.0

    calls: list[tuple[str, int]] = []

    def fake_fit_predict(features, residual, train, valid, spec):
        calls.append((spec.name, int(valid.sum())))
        return np.zeros((int(valid.sum()), 6)), {"candidate": spec.name}

    monkeypatch.setattr(MODULE.v26, "fit_predict", fake_fit_predict)
    features = np.zeros((len(cases), MODULE.FEATURE_COUNT))
    _, receipts = MODULE.crossfit(cases, features, targets, reference)
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
