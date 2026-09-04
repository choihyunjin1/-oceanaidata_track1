from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v22 as cycle  # noqa: E402

from src.p1_qc.robust_student_t_llr import calibrate_threshold_central  # noqa: E402


def test_contract_is_prospective_and_outer_gate_unchanged() -> None:
    config = cycle.load_contract()
    assert config["lineage"]["v20r1_status_unchanged"] == "NO_PASS"
    assert config["inner_selector"]["wilson_lcb_gate"] is False
    assert config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == 0.13168209161000616
    assert config["authorization"]["historical_execution"] is True


def test_central_selector_requires_positive_delta_and_precision() -> None:
    scores = np.arange(1000, dtype=float)
    truth = np.zeros(1000, dtype=np.int8)
    truth[-4:] = 1
    selected = calibrate_threshold_central(scores, truth, np.zeros(1000, dtype=np.int8))
    assert selected["additions"] == 4
    assert selected["inner_delta_f1"] > 0
    assert selected["precision"] == 1.0


def test_no_benefit_means_abstain() -> None:
    selected = calibrate_threshold_central(np.arange(100), np.zeros(100), np.zeros(100))
    assert selected["additions"] == 0
    assert np.isinf(selected["threshold"])
