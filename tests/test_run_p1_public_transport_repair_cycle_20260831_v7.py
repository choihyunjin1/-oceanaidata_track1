from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v7 as cycle  # noqa: E402


def test_scope_is_layer2_drift_only_and_add_only() -> None:
    frame = pd.DataFrame(
        {
            "layer": [2, 1, 2, 2],
            "e150_prediction": [0, 0, 1, 0],
            "pmax": [0.02, 0.02, 0.02, 0.001],
        }
    )
    assert cycle.source_fit_scope(frame).tolist() == [True, True, False, True]
    assert cycle.drift_calibration_scope(frame).tolist() == [True, True, False, False]
    assert cycle.deployment_scope(frame).tolist() == [True, False, False, False]


def test_exact_precision_lower_bound_is_conservative() -> None:
    lower = cycle.precision_lower_bound(tp=90, fp=10, confidence=0.9)
    assert 0.8 < lower < 0.9
    assert cycle.precision_lower_bound(tp=0, fp=10, confidence=0.9) == 0.0


def test_root_transport_and_bootstrap_lcb_are_exact() -> None:
    config = cycle.load_contract()
    policy = config["decision_policy"]
    assert np.isclose(
        policy["minimum_raw_expected_point_delta_inclusive"],
        0.015383691373120248,
    )
    assert np.isclose(
        policy["bootstrap_ci90_low_minimum"],
        0.0005788103467134221,
    )
    assert policy["worst_forward_block_delta_f1_minimum"] == 0.0


def test_native_converts_numpy_boole_for_terminal_json() -> None:
    payload = cycle.native({"value": np.bool_(True), "count": np.int64(3)})
    assert payload == {"value": True, "count": 3}
    assert isinstance(payload["value"], bool)


def test_precision_receipt_requires_lcb_above_f1_half() -> None:
    payload = {
        "2025_q3": {
            "calibration_true_positive_additions": 90,
            "calibration_false_positive_additions": 10,
            "theoretical_add_only_threshold_strict": 0.45,
            "outer_test_fold": "2025_q3",
        }
    }
    receipt = cycle.component_precision_receipts(payload, confidence=0.9)[0]
    assert receipt["pass"] is True
    assert receipt["exact_one_sided_precision_lower_bound"] > 0.45
