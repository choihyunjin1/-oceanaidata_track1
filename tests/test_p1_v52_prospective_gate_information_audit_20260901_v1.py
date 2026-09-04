from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v52_prospective_gate_information_audit_20260901_v1.py"
DATA = Path(os.environ["P1_DATA_DIR"])


def _module():
    spec = importlib.util.spec_from_file_location("p1_v52_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()


def test_f1_formula() -> None:
    truth = np.array([1, 1, 0, 0], dtype=np.int8)
    prediction = np.array([1, 0, 1, 0], dtype=np.int8)
    assert mod._f1(truth, prediction) == 0.5


def test_parent_selection_is_pretarget_direct_and_not_v35_parent() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    rule = config["parent_selection_rule_sealed_before_q2_target"]
    assert rule["selected_parent"] == "p1_v38_causal_focal_loss_crossquarter_addonly_20260901_v1"
    assert rule["target_information_used_for_parent_selection"] == 0
    assert "non-recovery" in rule["rule"]
    assert config["prior_audit_boundary"]["v35_candidate_or_result_reused"] == 0


def test_contract_fixes_max_lcb_candidate_without_recompute_or_reselection() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    fixed = config["selection_fixed_before_q2_target"]
    assert fixed["candidate_index"] == 1
    assert fixed["quantile"] == 0.9975
    assert fixed["action_key"] == "actions_candidate_1"
    assert fixed["q2_label_blind_action_count"] == 77
    contract = config["contract"]
    assert contract["fits"] == contract["refits"] == 0
    assert contract["score_recomputation"] == contract["threshold_reselection"] == 0


def test_q3_q4_promotion_and_csv_remain_closed() -> None:
    contract = json.loads(mod.CONFIG.read_text(encoding="utf-8"))["contract"]
    assert contract["q2_target_windows"] == 1
    assert contract["q3_q4_target_reads"] == 0
    assert contract["candidate_promotion_rescue_retune"] == 0
    assert contract["csv_materialization"] == 0
    assert contract["parent_decision_unchanged"]


def test_real_preflight_is_zero_fit_target_free_and_selects_77_actions() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    assert ready["status"] == "PASS_ZERO_FIT_PRETARGET"
    assert all(value == 0 for value in ready["counters"].values())
    assert not ready["bundle_receipt"]["target_values_present"]
    assert ready["bundle_receipt"]["actions"] == 77


def test_wilson_source_is_method_only() -> None:
    source = json.loads(mod.CONFIG.read_text(encoding="utf-8"))["primary_source"]
    assert source["doi"] == "10.1080/01621459.1927.10502953"
    assert "does not validate" in source["claim_boundary"]
