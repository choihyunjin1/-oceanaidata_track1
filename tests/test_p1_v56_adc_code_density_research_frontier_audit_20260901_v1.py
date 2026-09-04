from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v56_adc_code_density_research_frontier_audit_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RESULT = ROOT / f"artifacts/{EXPERIMENT_ID}/result.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exactly_one_adc_mechanism_was_audited() -> None:
    candidate = _config()["single_mechanism"]
    assert candidate["candidate_count"] == 1
    assert "ADC" in candidate["name"]
    assert candidate["result_based_retune_or_bin_width_sweep"] == 0
    assert candidate["anchor_removals"] == 0


def test_primary_source_and_claim_boundary_are_explicit() -> None:
    source = _config()["primary_source"]
    assert source["standard"] == "IEEE 1241-2023"
    assert source["publication_date"] == "2023-10-06"
    assert "does not establish" in source["claim_boundary"]


def test_schema_cannot_identify_adc_code_density() -> None:
    support = _config()["schema_support_audit"]
    absent = " ".join(support["required_but_absent"])
    assert "raw integer ADC output code" in absent
    assert "LSB/code width" in absent
    assert "code transition levels" in absent
    assert support["identifiability"] == "FAIL"
    assert support["train_rows_read"] == support["target_rows_read"] == 0


def test_repository_proxy_overlap_is_fail_closed() -> None:
    audit = _config()["repository_negative_fingerprint"]
    assert audit["exact_p1_adc_dnl_or_code_density_implementation_hits"] == 0
    assert audit["exact_duplicate"] is False
    assert audit["support_qualified_new_observable"] is False
    assert audit["proxy_semantic_duplicate_or_reconstructible"] is True
    assert "plateaus" in audit["plateau_and_zero_step_family"]["overlap"]
    assert "quantize" in audit["soft_symbolic_family"]["overlap"]
    assert "occupancy" in audit["state_occupancy_family"]["overlap"]


def test_all_evidence_hashes_match() -> None:
    config = _config()
    audit = config["repository_negative_fingerprint"]
    pins = [
        (audit["plateau_and_zero_step_family"]["path"], audit["plateau_and_zero_step_family"]["sha256"]),
        (audit["soft_symbolic_family"]["path"], audit["soft_symbolic_family"]["sha256"]),
        (audit["state_occupancy_family"]["path"], audit["state_occupancy_family"]["sha256"]),
        (config["frontier_evidence"]["candidate_registry_audit"]["path"], config["frontier_evidence"]["candidate_registry_audit"]["sha256"]),
        (config["frontier_evidence"]["event_hazard_frontier_audit"]["path"], config["frontier_evidence"]["event_hazard_frontier_audit"]["sha256"]),
    ]
    assert all(_sha(ROOT / path) == digest for path, digest in pins)
    readme = Path(config["schema_support_audit"]["readme_path"])
    assert _sha(readme) == config["schema_support_audit"]["readme_sha256"]


def test_terminal_result_is_consistent_and_zero_fit() -> None:
    config = _config()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == config["status"]
    assert result["ready"] is config["decision"]["ready"] is False
    assert result["fit_count"] == result["optimizer_steps"] == 0
    assert result["historical_metrics_computed"] is False
    assert result["candidate_actions"] == result["anchor_removals"] == 0


def test_no_runner_lock_or_forbidden_operation() -> None:
    config = _config()
    assert all(value == 0 for value in config["operations"].values())
    assert not RUNNER.exists()
    assert not LOCK.exists()
