from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v24 as cycle  # noqa: E402


def test_exact_gce_contract_and_v16_closure() -> None:
    config = cycle.load_contract()
    assert config["model"]["gce_q"] == 0.7
    assert config["model"]["l2"] == 0.001
    assert config["features"]["encoded_feature_count"] == 165
    assert config["lineage"]["v16_fixed_threshold_result"] == "CLOSED_NO_PASS_UNCHANGED"


def test_inner_selector_and_v3_outer_gate_are_frozen() -> None:
    config = cycle.load_contract()
    assert config["inner_selector"]["outer_labels_used"] is False
    assert config["inner_selector"]["tie_break"] == "higher threshold then fewer additions"
    assert config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == 0.015383691373120248
    assert config["decision_policy"]["minimum_calibrated_expected_point_delta_inclusive"] == 0.01


def test_exactly_once_historical_execution_authorized() -> None:
    config = cycle.load_contract()
    assert config["fit_budget"]["maximum"] == 2
    assert config["authorization"]["historical_execution"] is True
    assert config["authorization"]["attempt_lock_creation"] is True
