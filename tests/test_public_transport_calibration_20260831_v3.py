from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_public_transport_calibration_20260831_v3 as calibration  # noqa: E402


def test_p1_penalty_uses_only_p1_official_pair() -> None:
    result = calibration.build()
    assert result["p1"]["observed_pair_count"] == 1
    assert result["p1"]["prospective_unseen_family_or_tier_penalty_points"] == 0.005383691373120247
    assert result["p1"]["prospective_minimum_raw_expected_points_delta"] == 0.015383691373120248


def test_plus_point_zero_one_and_no_retroactive_pass_are_invariants() -> None:
    result = calibration.build()
    assert result["minimum_calibrated_expected_points_delta"] == 0.01
    assert result["policy"]["minimum_plus_0_01_points_maintained"] is True
    assert result["policy"]["retroactive_reclassification_forbidden"] is True
    assert result["policy"]["existing_v5_v22_results_unchanged"] is True


def test_n1_is_explicitly_not_a_confidence_interval() -> None:
    result = calibration.build()
    assert result["p1"]["uncertainty_status"] == "PROVISIONAL_N1_NOT_A_CONFIDENCE_INTERVAL"
    assert result["policy"]["no_future_penalty_reduction_before_three_same_problem_pairs"] is True
    assert result["provenance"]["official_row_reads"] == 0
    assert result["provenance"]["hidden_truth_reads"] == 0
