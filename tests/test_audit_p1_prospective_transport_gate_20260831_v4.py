from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_p1_prospective_transport_gate_20260831_v4.py"
SPEC = importlib.util.spec_from_file_location("p1_gate_v4", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_diagnostics_cannot_reverse_a_hard_pass() -> None:
    state = gate.prospective_decision_state(
        validity_pass=True,
        hard_gate_results={"pooled": True, "bootstrap": True, "transport": True},
        diagnostic_results={"daily": False, "station_layer": False},
    )
    assert state == "PASS_PRIMARY_WITH_TRANSPORT_WARNING"


def test_hard_or_validity_failure_still_blocks() -> None:
    assert (
        gate.prospective_decision_state(
            validity_pass=True,
            hard_gate_results={"pooled": False},
            diagnostic_results={"daily": True},
        )
        == "NO_PASS_PRIMARY_GATE"
    )
    assert (
        gate.prospective_decision_state(
            validity_pass=False,
            hard_gate_results={"pooled": True},
            diagnostic_results={"daily": True},
        )
        == "QA_BLOCKED"
    )


def test_old_daily_fraction_cap_has_integer_discontinuity() -> None:
    rows = 133
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * rows,
            "year": [2025] * rows,
            "layer": [1] * rows,
            "time": pd.date_range("2025-07-01", periods=rows, freq="10min", tz="UTC"),
            "fold": ["2025_q3"] * rows,
            "label_base": np.resize(np.array([0, 0, 1], dtype=np.int8), rows),
            "current_router_prediction": np.resize(
                np.array([0, 1, 0], dtype=np.int8), rows
            ),
        }
    )
    summary = gate.distribution_summary(frame.copy(), frame.copy())
    old = summary["former_daily_0_005_cap_integer_audit"]
    assert old["days_forcing_zero_integer_actions"] >= 1
    assert old["minimum_nonzero_changed_fraction"] > 0.005


def test_policy_changes_only_the_two_diagnostic_roles() -> None:
    policy = gate.load_policy()
    assert policy["effective_scope"]["future_preregistrations_only"] is True
    assert policy["effective_scope"]["p1_v28_retroactive_pass_forbidden"] is True
    assert set(policy["audit_isolation"]["only_role_changes_in_v4"]) == {
        "maximum_changed_fraction_any_kst_day",
        "minimum_each_supported_station_layer_delta_f1",
    }
    hard = policy["hard_scientific_and_transport_gates"]
    assert hard["minimum_calibrated_expected_point_delta_inclusive"] == 0.01
    assert hard["anchor_removals_required"] == 0


def test_numpy_boole_serialize_in_qa_receipts(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    gate.write_json(output, {"check": np.bool_(True)})
    assert json.loads(output.read_text(encoding="utf-8")) == {"check": True}


def test_workspace_audit_passes_without_reclassifying_v28() -> None:
    result = gate.audit_workspace()
    assert result["status"] == "PASS"
    assert result["prospective_decision"]["v28_original_decision"] == "NO_GO_SAFETY_GATES"
    assert result["prospective_decision"]["v28_retroactive_reclassification"] is False
    assert result["operations"]["model_fits"] == 0
    assert result["operations"]["official_rows_read"] == 0
    assert result["operations"]["hidden_truth_rows_read"] == 0
    assert result["operations"]["uploads"] == 0
