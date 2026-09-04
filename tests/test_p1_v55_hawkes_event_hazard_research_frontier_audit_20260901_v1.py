from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v55_hawkes_event_hazard_research_frontier_audit_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RESULT = ROOT / f"artifacts/{EXPERIMENT_ID}/result.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exactly_one_mechanism_was_audited() -> None:
    candidate = _config()["single_mechanism"]
    assert candidate["candidate_count"] == 1
    assert "Hawkes" in candidate["name"]
    assert candidate["result_based_retune_or_kernel_sweep"] == 0
    assert candidate["anchor_removals"] == 0


def test_primary_source_claim_boundary_is_explicit() -> None:
    source = _config()["primary_source"]
    assert source["doi"] == "10.1093/biomet/58.1.83"
    assert source["date"] == "1971-04"
    assert "does not establish P1" in source["claim_boundary"]


def test_repository_semantic_overlap_is_pinned() -> None:
    audit = _config()["repository_negative_fingerprint"]
    assert audit["exact_hawkes_term_or_implementation_hits"] == 0
    assert audit["exact_duplicate"] is False
    assert audit["semantic_duplicate_or_reconstructible_substitute"] is True
    assert "time since the most recent anchor" in audit["anchor_history_features"]["overlap"]
    assert "memory kernel" in audit["anchor_history_features"]["overlap"]


def test_all_repository_evidence_hashes_match() -> None:
    config = _config()
    audit = config["repository_negative_fingerprint"]
    pins = [
        (audit["anchor_history_features"]["config_path"], audit["anchor_history_features"]["config_sha256"]),
        (audit["anchor_history_features"]["runner_path"], audit["anchor_history_features"]["runner_sha256"]),
        (audit["event_duration_and_decoder_family"]["typed_duration_path"], audit["event_duration_and_decoder_family"]["typed_duration_sha256"]),
        (audit["event_duration_and_decoder_family"]["typed_factorial_path"], audit["event_duration_and_decoder_family"]["typed_factorial_sha256"]),
        (audit["recurrence_and_return_time_family"]["recurrence_path"], audit["recurrence_and_return_time_family"]["recurrence_sha256"]),
        (audit["recurrence_and_return_time_family"]["return_time_path"], audit["recurrence_and_return_time_family"]["return_time_sha256"]),
        (audit["negative_registry"]["path"], audit["negative_registry"]["sha256"]),
        (config["frontier_evidence"]["fresh_window_audit"]["path"], config["frontier_evidence"]["fresh_window_audit"]["sha256"]),
        (config["frontier_evidence"]["candidate_registry_audit"]["path"], config["frontier_evidence"]["candidate_registry_audit"]["sha256"]),
    ]
    assert all(_sha(ROOT / path) == digest for path, digest in pins)


def test_deployment_support_fails_without_reading_targets() -> None:
    support = _config()["deployment_support_preflight"]
    assert support["true_anomaly_onset_history"]["available_during_unlabeled_deployment"] is False
    assert support["true_anomaly_onset_history"]["causal_feature_authorized"] is False
    assert support["anchor_onset_substitute"]["new_information_beyond_closed_families"] is False
    assert support["support_gate"].startswith("FAIL_")
    assert support["target_rows_read"] == support["train_rows_read"] == 0


def test_terminal_result_is_consistent_and_zero_fit() -> None:
    config = _config()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == config["status"]
    assert result["ready"] is config["decision"]["ready"] is False
    assert result["fit_count"] == result["optimizer_steps"] == 0
    assert result["historical_metrics_computed"] is False
    assert result["candidate_actions"] == result["anchor_removals"] == 0


def test_no_executable_preflight_runner_lock_or_forbidden_access() -> None:
    config = _config()
    assert all(value == 0 for value in config["operations"].values())
    assert not RUNNER.exists()
    assert not LOCK.exists()
