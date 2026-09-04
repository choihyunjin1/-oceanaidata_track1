from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from p1_qc.causal_scar_pu import (
    ContractError,
    chronological_inner_split,
    correct_selection_probability,
    estimate_scar_propensity,
    select_add_only_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_public_transport_repair_cycle_20260831_v21.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("p1_v21_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chronological_split_keeps_equal_timestamps_together() -> None:
    times = np.repeat(np.arange(12, dtype=np.int64), 3)
    split = chronological_inner_split(times, np.ones(len(times), dtype=bool), fit_fraction=0.75)
    assert times[split.fit_mask].max() < times[split.calibration_mask].min()
    assert not np.any(split.fit_mask & split.calibration_mask)


def test_propensity_uses_positive_support_and_correction_clips() -> None:
    probability = np.array([0.1, 0.4, 0.6, 0.8, 0.7, 0.9, 0.5, 0.3, 0.2, 0.1])
    labels = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.int8)
    propensity = estimate_scar_propensity(probability, labels, minimum_positive_support=8)
    assert propensity == pytest.approx(probability[labels == 1].mean())
    corrected = correct_selection_probability(probability, propensity)
    assert np.isfinite(corrected).all()
    assert corrected.min() >= 0 and corrected.max() <= 1
    with pytest.raises(ContractError):
        estimate_scar_propensity(probability, np.zeros(10, dtype=np.int8), minimum_positive_support=1)


def test_inner_threshold_is_add_only_and_precision_guarded() -> None:
    score = np.array([0.99, 0.98, 0.70, 0.20, 0.10, 0.05])
    labels = np.array([1, 1, 0, 0, 1, 0], dtype=np.int8)
    anchor = np.array([0, 0, 0, 0, 1, 0], dtype=np.int8)
    selected = select_add_only_threshold(
        score,
        labels,
        anchor,
        maximum_changed_fraction=0.5,
    )
    assert selected.additions == 2
    assert selected.true_positives == 2
    assert selected.false_positives == 0
    assert selected.candidate_f1 > selected.incumbent_f1


def test_preflight_has_zero_historical_official_and_lock_access() -> None:
    runner = load_runner()
    report = json.loads(runner.REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["historical_fits_executed"] == 0
    assert report["fit_budget_if_later_authorized"] == 2
    assert all(value == 0 for value in report["access"].values())


def test_execute_authorization_and_exactly_once_state_are_consistent() -> None:
    runner = load_runner()
    contract = runner.load_contract()
    assert contract["authorization"]["historical_execution"] is True
    assert contract["authorization"]["attempt_lock_creation"] is True
    if runner.ARTIFACT.exists():
        assert (runner.ARTIFACT / "attempt_lock.json").is_file()
        assert (runner.ARTIFACT / "result.json").is_file()
