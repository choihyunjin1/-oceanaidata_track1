import numpy as np

from scripts import run_p1_public_transport_repair_cycle_20260831_v31 as runner
from src.p1_qc.logit_shrunk_label_shift import (
    correct_to_prior,
    shrink_lambda,
    shrunk_target_prevalence,
)


def test_lambda_is_closed_form_and_bounded():
    assert np.isclose(shrink_lambda(0.1, 0.3, 0.2), 0.600721658468336)
    assert shrink_lambda(0.1, 0.3, 0.9) == 1.0


def test_zero_and_full_shrink_endpoints():
    assert np.isclose(shrunk_target_prevalence(0.1, 0.3, 0), 0.1)
    assert np.isclose(shrunk_target_prevalence(0.1, 0.3, 1), 0.3)
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(correct_to_prior(p, 0.1, 0.1), p)


def test_contract_marks_slice_checks_diagnostic_and_history_off():
    cfg = runner.load_contract()
    assert cfg["safety"]["maximum_changed_fraction_any_kst_day"] == "diagnostic_only"
    assert cfg["authorization"]["historical_execution"] is False


def test_preflight_passes():
    assert runner.preflight()["status"] == "PASS"
