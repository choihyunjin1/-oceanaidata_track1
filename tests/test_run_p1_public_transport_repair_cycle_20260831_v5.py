from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v5 as cycle  # noqa: E402


def test_root_transport_calibration_is_exact_and_inclusive() -> None:
    config = cycle.load_contract()
    policy = config["decision_policy"]
    assert policy["minimum_calibrated_expected_point_delta_inclusive"] == 0.01
    assert np.isclose(
        policy["worst_observed_public_transport_residual_points"],
        -0.005383691373120247,
    )
    assert np.isclose(
        policy["minimum_raw_expected_point_delta_inclusive"],
        0.015383691373120248,
    )


def test_deployment_eligibility_is_add_only_gors_layer1() -> None:
    frame = pd.DataFrame(
        {
            "station": ["G-ORS", "I-ORS", "G-ORS", "G-ORS"],
            "layer": [1, 1, 2, 1],
            "e150_prediction": [0, 0, 0, 1],
            "pmax": [0.02, 0.02, 0.02, 0.02],
        }
    )
    assert cycle.deployment_eligibility(frame).tolist() == [True, False, False, False]


def test_threshold_is_selected_only_from_inner_calibration_rows() -> None:
    config = copy.deepcopy(cycle.load_contract())
    config["threshold_selection"]["score_grid"] = [0.3, 0.5, 0.7]
    config["threshold_selection"]["minimum_calibration_additions"] = 2
    config["threshold_selection"]["minimum_calibration_precision"] = 0.5
    frame = pd.DataFrame(
        {
            "label_base": [1, 1, 0, 0, 1, 0],
            "e150_prediction": [0, 0, 0, 0, 0, 0],
            "layer": [1, 1, 1, 1, 1, 1],
            "pmax": [0.1] * 6,
        }
    )
    calibration = np.array([True, True, True, True, False, False])
    score = np.array([0.9, 0.8, 0.2, 0.1, 0.99, 0.99])
    selected = cycle.select_threshold(frame, calibration, score, config)
    assert selected["status"] == "SELECTED_ON_INNER_FUTURE_BLOCK"
    assert selected["additions"] == 2
    assert selected["true_positive_additions"] == 2


def test_materialization_is_skipped_when_no_candidate_passes() -> None:
    outputs, access, fits = cycle.materialize_passes(
        pd.DataFrame(), [{"strict_internal_pass": False}], cycle.load_contract()
    )
    assert outputs == []
    assert access["official_covariate_reads"] == 0
    assert fits == 0


def test_score_gate_uses_calibrated_points_and_preserves_worst_block() -> None:
    config = copy.deepcopy(cycle.load_contract())
    config["validation"]["bootstrap_replicates"] = 100
    rows = 40
    frame = pd.DataFrame(
        {
            "fold": ["2025_q3"] * 20 + ["2025_q4"] * 20,
            "time": pd.date_range("2025-07-01", periods=rows, freq="24h", tz="UTC"),
            "label_base": [1, 1, 1, 1] + [0] * 16 + [1, 1] + [0] * 18,
            "e150_prediction": [1, 1, 0, 0] + [0] * 16 + [1, 1] + [0] * 18,
        }
    )
    anchor = frame["e150_prediction"].to_numpy(np.int8)
    candidate = anchor.copy()
    candidate[2:4] = 1
    record = cycle.score_record(
        "candidate",
        frame,
        candidate,
        [candidate.copy(), candidate.copy(), candidate.copy()],
        {},
        config,
    )
    expected = (
        record["raw_expected_points_delta"]
        + config["decision_policy"]["worst_observed_public_transport_residual_points"]
    )
    assert np.isclose(
        record["calibrated_conservative_expected_points_delta"], expected
    )
    assert record["by_fold"]["2025_q4"]["delta_f1"] == 0.0
    assert record["anchor_removals"] == 0
