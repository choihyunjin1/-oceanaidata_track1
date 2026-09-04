from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p1_v48_identity_routing_moe_semantic_support_gate_20260901_v1.json"
RUNNER = ROOT / "scripts/run_p1_v48_identity_routing_moe_semantic_support_gate_20260901_v1.py"
LOCK = ROOT / "artifacts/p1_v48_identity_routing_moe_semantic_support_gate_20260901_v1.ATTEMPT_LOCK.json"


def test_zero_fit_semantic_and_identifiability_closure() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "CLOSED_ZERO_FIT_SEMANTIC_AND_IDENTIFIABILITY_GATE"
    assert config["semantic_audit"]["decision"] == "REJECT_ZERO_FIT"
    assert config["semantic_audit"]["exact_duplicate"] is True
    assert config["identifiability_gate"]["target_free_row_occupancy"] == "PASS"
    assert config["identifiability_gate"]["target_free_identity_action_coverage"] == "NOT_IDENTIFIABLE"
    assert config["counters"]["fits"] == 0
    assert config["counters"]["target_reads"] == 0
    assert not RUNNER.exists()
    assert not LOCK.exists()


def test_fixed_recipe_was_bounded_before_rejection() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    recipe = config["candidate"]["prospective_fixed_recipe_if_identifiable"]
    assert recipe["experts"] == 4
    assert recipe["top_k"] == 1
    assert recipe["fits"] == 3
    assert recipe["maximum_fits"] == 9
    assert recipe["sweep"] == 0
    assert recipe["threshold_quantiles"] == [0.995, 0.9975, 0.999]
    assert recipe["anchor_removals"] == 0


def test_train_only_root_cause_receipt_has_no_target_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    audit = config["root_cause_audit"]
    assert audit["prefix_rows"] == 294278
    assert audit["station_layer_cells"] == 11
    assert audit["minimum_station_layer_rows"] == 1971
    assert config["source"]["train_columns_read"] == ["station", "layer", "time"]
    assert config["source"]["target_columns_read"] == 0
    assert config["source"]["official_test_sample_submission_hidden_reads"] == 0


def test_v28_v33_add_only_and_access_zero_preserved() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract = config["contracts_preserved"]
    assert contract["cross_quarter_guard"]["sha256"] == "a051d4d0837b395f9c1d42d71d65572efb99d6c1aeae6aabdb52c8427539c8d6"
    assert contract["auditability_amendment"]["sha256"] == "a20cf248c3c4cd4ced858deccca1fbb52f4e1ed114582988d9957018a7e43128"
    assert contract["add_only"] is True
    assert contract["anchor_removals"] == 0
    assert all(config["counters"][key] == 0 for key in ["fits", "optimizer_steps", "target_reads", "actions", "removals", "official", "csv", "uploads"])
