from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p1_robust_subspace_block_conformal_add_only_falsification_20260901_v1.json"
SCRIPT = ROOT / "scripts/run_p1_robust_subspace_block_conformal_add_only_falsification_20260901_v1.py"


def test_contract_is_audit_bound_add_only_and_max_three_fits() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["audit"]["required_decision"] == "ROBUST_SUBSPACE_CONFORMAL_FEASIBLE_RESEARCH_ONLY"
    assert payload["frozen_rule"]["anchor_operation"] == "bitwise_or_no_removal"
    assert payload["frozen_rule"]["label_tuning"] == 0
    assert payload["operations"]["maximum_fits"] == 3
    assert payload["operations"]["supervised_fits"] == 0


def test_runner_seals_before_target_open_and_has_no_official_paths() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index('"predictions_complete.json"') < source.index('usecols=["label", "anomaly_type"]')
    assert "test.csv" not in source
    assert "sample_submission" not in source


def test_single_frozen_evalue_and_budget() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["frozen_rule"]["e_value"] == "0.5/sqrt(p)"
    assert payload["frozen_rule"]["e_bh_q"] == 0.01
    assert payload["frozen_rule"]["block_length_rows"] == 144
