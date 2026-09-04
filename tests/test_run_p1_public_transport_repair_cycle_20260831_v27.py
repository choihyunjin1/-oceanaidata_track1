import numpy as np

from scripts import run_p1_public_transport_repair_cycle_20260831_v27 as runner


def test_ecdf_right_continuous_and_bounded():
    ref = runner.fit_ecdf(np.array([0.1, 0.2, 0.2, 0.9]))
    got = runner.apply_ecdf(ref, np.array([0.0, 0.2, 1.0]))
    assert np.allclose(got, [0.0, 0.75, 1.0])


def test_consensus_is_minimum_of_two_prefix_ranks():
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    got = runner.consensus_score(ref, ref, np.array([0.4]), np.array([0.2]))
    assert np.allclose(got, [0.5])


def test_contract_is_exact_and_no_em():
    cfg = runner.load_contract()
    assert cfg["score"]["no_em"] is True
    assert cfg["fit_budget"]["maximum"] == 2
    assert cfg["decision_policy"]["minimum_calibrated_expected_point_delta_inclusive"] == 0.01


def test_synthetic_preflight_passes():
    assert runner.preflight()["status"] == "PASS"
