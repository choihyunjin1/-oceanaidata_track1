from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v35_v34_rejected_candidate_q2_adjudication_20260901_v1.py"
DATA = Path(r"C:\Users\cedis\Downloads\데이터셋_P1\P1_qc_anomaly")


def _module():
    spec = importlib.util.spec_from_file_location("p1_v35_tested", RUNNER)
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


def test_contract_fixes_max_lcb_candidate_without_refit_or_reselection() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    fixed = config["selection_fixed_before_q2_target"]
    assert fixed["candidate_index"] == 0
    assert fixed["quantile"] == 0.995
    assert fixed["action_key"] == "actions_candidate_0"
    assert config["contract"]["fits"] == config["contract"]["refits"] == 0
    assert config["contract"]["score_recomputation"] == config["contract"]["threshold_reselection"] == 0


def test_q3_q4_and_promotion_remain_closed() -> None:
    contract = json.loads(mod.CONFIG.read_text(encoding="utf-8"))["contract"]
    assert contract["q2_target_windows"] == 1
    assert contract["q3_q4_target_reads"] == 0
    assert contract["candidate_promotion_rescue_retune"] == 0
    assert contract["parent_decision_unchanged"]


def test_real_preflight_is_zero_fit_and_target_free() -> None:
    receipt = mod.ARTIFACT / "preflight.json"
    ready = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else mod.preflight(DATA)
    assert ready["status"] == "PASS_ZERO_FIT_PRETARGET"
    assert all(value == 0 for value in ready["counters"].values())
    assert not ready["bundle_receipt"]["target_values_present"]
    assert ready["bundle_receipt"]["actions"] == 332
