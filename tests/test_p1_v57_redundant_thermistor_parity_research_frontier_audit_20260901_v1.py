from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v57_redundant_thermistor_parity_research_frontier_audit_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RESULT = ROOT / f"artifacts/{EXPERIMENT_ID}/result.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exactly_one_redundancy_mechanism_was_audited() -> None:
    candidate = _config()["single_mechanism"]
    assert candidate["candidate_count"] == 1
    assert "redundant-thermistor" in candidate["name"]
    assert candidate["result_based_retune_or_residual_sweep"] == 0
    assert candidate["anchor_removals"] == 0


def test_primary_source_identity_and_claim_boundary() -> None:
    source = _config()["primary_source"]
    assert source["doi"] == "10.1109/TAC.1984.1103593"
    assert source["publication_date"] == "1984-07"
    assert "does not establish" in source["claim_boundary"]


def test_schema_lacks_redundant_channel_contract() -> None:
    support = _config()["schema_support_audit"]
    absent = " ".join(support["required_but_absent"])
    assert "probe or instrument identifier" in absent
    assert "colocated temperature channels" in absent
    assert "redundancy/parity relation" in absent
    assert support["identifiability"] == "FAIL"
    assert support["train_rows_read"] == support["target_rows_read"] == 0


def test_invalid_layer_proxy_is_reconstructible() -> None:
    audit = _config()["repository_negative_fingerprint"]
    assert audit["exact_p1_analytical_redundancy_parity_space_or_redundant_probe_hits"] == 0
    assert audit["exact_duplicate"] is False
    assert audit["support_qualified_new_observable"] is False
    assert audit["invalid_layer_proxy_semantic_duplicate_or_reconstructible"] is True
    assert "peer means" in audit["base_cross_layer_features"]["overlap"]
    assert "change coherence" in audit["stratification_peer_family"]["overlap"]
    assert "other-layer residuals" in audit["long_event_cross_layer_family"]["overlap"]


def test_all_nine_evidence_hashes_match() -> None:
    config = _config()
    support = config["schema_support_audit"]
    audit = config["repository_negative_fingerprint"]
    frontier = config["frontier_evidence"]
    pins = [
        (support["layer_identity_warning"]["path"], support["layer_identity_warning"]["sha256"]),
        (audit["base_cross_layer_features"]["path"], audit["base_cross_layer_features"]["sha256"]),
        (audit["stratification_peer_family"]["path"], audit["stratification_peer_family"]["sha256"]),
        (audit["long_event_cross_layer_family"]["path"], audit["long_event_cross_layer_family"]["sha256"]),
        (audit["physical_profile_closures"]["physical_path"], audit["physical_profile_closures"]["physical_sha256"]),
        (audit["physical_profile_closures"]["profile_path"], audit["physical_profile_closures"]["profile_sha256"]),
        (frontier["candidate_registry_audit"]["path"], frontier["candidate_registry_audit"]["sha256"]),
        (frontier["event_hazard_audit"]["path"], frontier["event_hazard_audit"]["sha256"]),
        (frontier["adc_contract_audit"]["path"], frontier["adc_contract_audit"]["sha256"]),
    ]
    assert all(_sha(ROOT / path) == digest for path, digest in pins)
    assert _sha(Path(support["readme_path"])) == support["readme_sha256"]


def test_terminal_result_is_consistent_and_zero_fit() -> None:
    config = _config()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == config["status"]
    assert result["ready"] is config["decision"]["ready"] is False
    assert result["fit_count"] == result["optimizer_steps"] == 0
    assert result["historical_metrics_computed"] is False
    assert result["candidate_actions"] == result["anchor_removals"] == 0


def test_contracts_unchanged_and_no_forbidden_operation() -> None:
    config = _config()
    contracts = config["contracts_preserved"]
    assert contracts["cross_quarter_guard"] == "UNCHANGED_V28"
    assert contracts["auditability_amendment"] == "UNCHANGED_V33"
    assert contracts["feature_contract"] == contracts["split_contract"] == "UNCHANGED"
    assert all(value == 0 for value in config["operations"].values())
    assert not RUNNER.exists()
    assert not LOCK.exists()
