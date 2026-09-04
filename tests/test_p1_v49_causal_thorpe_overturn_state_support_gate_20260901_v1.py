from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v49_causal_thorpe_overturn_state_support_gate_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def test_single_station_profile_support_closes_before_readiness() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    support = config["target_free_support"]
    assert support["stations_with_ge3_level_profiles"] == 1
    assert support["by_station"]["G-ORS"]["maximum_levels"] == 1
    assert support["by_station"]["I-ORS"]["maximum_levels"] == 2
    assert support["by_station"]["S-ORS"]["profiles_ge3_levels"] == 53319
    assert support["gate"] == "FAIL_ONLY_ONE_STATION_HAS_RESOLVABLE_VERTICAL_PROFILES"
    assert config["decision"]["ready"] is False


def test_fixed_candidate_and_budget_were_sealed_before_rejection() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    recipe = config["candidate"]["fixed_recipe_if_supported"]
    assert recipe["minimum_finite_distinct_depths"] == 3
    assert recipe["minimum_profiles_per_station"] == 10000
    assert recipe["minimum_supported_stations"] == 2
    assert recipe["fits"] == 3
    assert recipe["maximum_fits"] == 9
    assert recipe["sweep"] == 0
    assert recipe["anchor_removals"] == 0


def test_reconstructibility_and_primary_claim_boundary_are_explicit() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    audit = config["semantic_reconstructibility_audit"]
    assert audit["semantic_duplicate"] is True
    assert set(audit) >= {"physical_consistency", "stratification_peer_gate", "vertical_rank"}
    source = config["candidate"]["primary_source"]
    assert source["doi"] == "10.1029/2001JC001154"
    assert "does not validate" in source["claim_boundary"]


def test_zero_fit_no_lock_and_access_zero() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["source"]["target_columns_read"] == 0
    assert config["source"]["official_test_sample_submission_hidden_reads"] == 0
    assert all(config["counters"][key] == 0 for key in ["fits", "optimizer_steps", "preflights", "target_reads", "actions", "removals", "official", "csv", "uploads"])
    assert not RUNNER.exists()
    assert not LOCK.exists()


def test_v28_v33_and_add_only_contract_preserved() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract = config["contracts_preserved"]
    assert contract["cross_quarter_guard"]["sha256"] == "a051d4d0837b395f9c1d42d71d65572efb99d6c1aeae6aabdb52c8427539c8d6"
    assert contract["auditability_amendment"]["sha256"] == "a20cf248c3c4cd4ced858deccca1fbb52f4e1ed114582988d9957018a7e43128"
    assert contract["add_only"] is True
    assert contract["anchor_removals"] == 0
