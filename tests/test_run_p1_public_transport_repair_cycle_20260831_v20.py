from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v20 as cycle  # noqa: E402

from src.p1_qc.robust_student_t_llr import (  # noqa: E402
    calibrate_threshold,
    fit_student_t,
    score_llr,
)


def test_contract_exact_two_fit_and_inner_calibration() -> None:
    config = cycle.load_contract()
    assert config["fit_budget"]["maximum"] == 2
    assert config["inner_calibration"]["fit_fraction"] == 0.75
    assert config["inner_calibration"]["outer_labels_used"] is False
    assert config["model"]["retuning"] is False


def test_student_t_score_is_finite_with_missing_values() -> None:
    x = np.asarray([[0, 0], [0.1, np.nan], [3, 2], [4, 3]], dtype=float)
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = score_llr(fit_student_t(x, y), x)
    assert np.isfinite(scores).all()
    assert scores[-1] > scores[0]


def test_threshold_calibration_is_deterministic_and_capped() -> None:
    scores = np.arange(1000, dtype=float)
    truth = np.zeros(1000, dtype=np.int8)
    truth[-4:] = 1
    anchor = np.zeros(1000, dtype=np.int8)
    first = calibrate_threshold(scores, truth, anchor)
    assert first == calibrate_threshold(scores, truth, anchor)
    assert first["additions"] <= 5


def test_historical_execution_exactly_once_authorized() -> None:
    config = cycle.load_contract()
    assert config["authorization"]["historical_execution"] is True
    assert config["authorization"]["attempt_lock_creation"] is True
