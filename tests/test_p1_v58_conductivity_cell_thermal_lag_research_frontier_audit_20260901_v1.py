from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v58_conductivity_cell_thermal_lag_research_frontier_audit_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RESULT = ROOT / f"artifacts/{EXPERIMENT_ID}/result.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exactly_one_thermal_lag_mechanism_was_audited() -> None:
    candidate = _config()["single_mechanism"]
    assert candidate["candidate_count"] == 1
    assert "thermal-inertia" in candidate["name"]
    assert candidate["result_based_retune_or_response_constant_sweep"] == 0
    assert candidate["anchor_removals"] == 0


def test_primary_source_identity_and_claim_boundary() -> None:
    source = _config()["primary_source"]
    assert source["doi"] == "10.1175/1520-0426(1990)007<0741:TIOCCT>2.0.CO;2"
    assert source["publication_date"] == "1990-10-01"
    assert "does not establish" in source["claim_boundary"]


def test_schema_lacks_raw_ctd_response_contract() -> None:
    support = _config()["schema_support_audit"]
    absent = " ".join(support["required_but_absent"])
    assert "raw conductivity-cell output" in absent
    assert "time alignment" in absent
    assert "flow rate" in absent
    assert "prior thermal-lag correction status" in absent
    assert support["identifiability"] == "FAIL"
    assert support["train_rows_read"] == support["target_rows_read"] == 0


def test_processed_proxy_is_reconstructible() -> None:
    audit = _config()["repository_negative_fingerprint"]
    assert audit["exact_p1_conductivity_cell_thermal_lag_or_inertia_hits"] == 0
    assert audit["exact_duplicate"] is False
    assert audit["support_qualified_new_observable"] is False
    assert audit["processed_proxy_semantic_duplicate_or_reconstructible"] is True
    assert "processed-salinity" in audit["temperature_salinity_matched_filter"]["overlap"]
    assert "signed path area" in audit["causal_path_crossmoment"]["overlap"]
    assert "raw/missing/difference/rolling" in audit["base_temporal_features"]["overlap"]


def test_all_nine_repository_hashes_and_readme_match() -> None:
    config = _config()
    audit = config["repository_negative_fingerprint"]
    frontier = config["frontier_evidence"]
    pins = [
        (audit["temperature_salinity_matched_filter"]["path"], audit["temperature_salinity_matched_filter"]["sha256"]),
        (audit["causal_path_crossmoment"]["path"], audit["causal_path_crossmoment"]["sha256"]),
        (audit["base_temporal_features"]["path"], audit["base_temporal_features"]["sha256"]),
        (audit["physical_consistency_closure"]["path"], audit["physical_consistency_closure"]["sha256"]),
        (frontier["candidate_registry_audit"]["path"], frontier["candidate_registry_audit"]["sha256"]),
        (frontier["event_hazard_audit"]["path"], frontier["event_hazard_audit"]["sha256"]),
        (frontier["adc_contract_audit"]["path"], frontier["adc_contract_audit"]["sha256"]),
        (frontier["redundant_sensor_audit"]["path"], frontier["redundant_sensor_audit"]["sha256"]),
    ]
    assert all(_sha(ROOT / path) == digest for path, digest in pins)
    support = config["schema_support_audit"]
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
