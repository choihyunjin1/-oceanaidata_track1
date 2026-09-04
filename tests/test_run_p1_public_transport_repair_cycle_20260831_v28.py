from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from p1_qc.prequential_label_shift_em import (
    frozen_logit_matrix,
    label_shift_em,
    select_inner_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_public_transport_repair_cycle_20260831_v28.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("p1_v28_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_logit_matrix_has_exactly_three_finite_columns() -> None:
    matrix = frozen_logit_matrix(
        np.array([0.0, 0.5, 1.0, np.nan]),
        np.array([0.2, 0.4, 0.6, 0.8]),
        np.array([0.9, 0.7, 0.3, 0.1]),
    )
    assert matrix.shape == (4, 3)
    assert np.isfinite(matrix).all()


def test_em_is_deterministic_finite_and_does_not_accept_labels() -> None:
    source = np.linspace(0.01, 0.39, 200)
    first, receipt = label_shift_em(source, 0.2)
    second, repeated = label_shift_em(source, 0.2)
    assert receipt.converged and receipt == repeated
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert 0 <= first.min() <= first.max() <= 1


def test_inner_selector_is_add_only_central_precision_guarded() -> None:
    probability = np.array([0.99, 0.98, 0.8, 0.2, 0.1, 0.05])
    labels = np.array([1, 1, 0, 0, 1, 0], dtype=np.int8)
    anchor = np.array([0, 0, 0, 0, 1, 0], dtype=np.int8)
    receipt = select_inner_threshold(
        probability,
        labels,
        anchor,
        maximum_changed_fraction=0.5,
    )
    assert receipt.additions == 2
    assert receipt.true_positives == 2
    assert receipt.false_positives == 0
    assert receipt.candidate_f1 > receipt.incumbent_f1


def test_synthetic_preflight_converges_with_future_labels_zero() -> None:
    runner = load_runner()
    report = json.loads(runner.REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["historical_fits_executed"] == 0
    assert report["access"]["outer_future_label_reads"] == 0
    assert all(value == 0 for value in report["access"].values())


def test_historical_authorization_and_exactly_once_state_are_consistent() -> None:
    runner = load_runner()
    contract = runner.load_contract()
    assert contract["authorization"]["historical_execution"] is True
    assert contract["authorization"]["attempt_lock_creation"] is True
    if runner.ARTIFACT.exists():
        assert (runner.ARTIFACT / "attempt_lock.json").is_file()
        assert (runner.ARTIFACT / "result.json").is_file()
