from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v53_fresh_historical_transport_window_availability_audit_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_station_quarters_are_ineligible_or_exposed() -> None:
    config = _config()
    quarters = config["station_quarter_inventory"]
    assert [item["period"] for item in quarters] == [
        "2024Q1", "2024Q2", "2024Q3", "2024Q4",
        "2025Q1", "2025Q2", "2025Q3", "2025Q4",
    ]
    assert all(item["freshness"] != "FRESH_ELIGIBLE" for item in quarters)
    assert config["decision"]["fresh_eligible_station_quarters"] == 0


def test_early_quarters_cannot_supply_multi_station_causal_transport() -> None:
    quarters = {item["period"]: item for item in _config()["station_quarter_inventory"]}
    assert all(quarters[f"2024Q{q}"]["stations"] == 1 for q in range(1, 5))
    assert "no earlier distributed train prefix" in quarters["2024Q1"]["reason"]
    assert all(quarters[f"2025Q{q}"]["stations"] >= 2 for q in range(1, 5))


def test_all_multi_station_quarters_have_prior_label_exposure() -> None:
    quarters = {item["period"]: item for item in _config()["station_quarter_inventory"]}
    assert quarters["2025Q1"]["freshness"] == "EXPOSED"
    assert all(quarters[f"2025Q{q}"]["freshness"] == "GLOBALLY_EXPOSED" for q in range(2, 5))
    ledger = _config()["exposure_ledger"]
    assert ledger["canonical_outer"]["windows"] == ["2025Q2", "2025Q3", "2025Q4"]
    assert "globally exposed" in ledger["global_exposure_receipt"]["evidence"]


def test_pinned_exposure_evidence_hashes_match() -> None:
    ledger = _config()["exposure_ledger"]
    pins = [
        ledger["canonical_outer"],
        ledger["canonical_inner_selection"],
        {"path": ledger["q2_q3_q4_metrics_and_predictions"]["result_path"], "sha256": ledger["q2_q3_q4_metrics_and_predictions"]["result_sha256"]},
        {"path": ledger["q2_q3_q4_metrics_and_predictions"]["manifest_path"], "sha256": ledger["q2_q3_q4_metrics_and_predictions"]["manifest_sha256"]},
        ledger["global_exposure_receipt"],
        ledger["prior_research_ledger"],
        {"path": ledger["v33_bundle_registry"]["amendment_path"], "sha256": ledger["v33_bundle_registry"]["amendment_sha256"]},
    ]
    for pin in pins:
        assert _sha(ROOT / pin["path"]) == pin["sha256"]


def test_v33_registry_is_auditability_not_fresh_truth() -> None:
    registry = _config()["exposure_ledger"]["v33_bundle_registry"]
    assert registry["complete_q2_manifests"] == 11
    assert registry["all_q2_rows"] == 133170
    assert registry["all_q2_target_reads_before_seal"] == 0
    assert "do not create a fresh labeled window" in registry["meaning"]


def test_terminal_is_zero_fit_without_runner_lock_or_candidate() -> None:
    config = _config()
    assert config["status"] == "NO_FRESH_HISTORICAL_TRANSPORT_WINDOW"
    assert config["decision"]["ready"] is False
    assert config["decision"]["candidate_selected"] is False
    assert config["decision"]["sealed_future_candidate_split_written"] is False
    assert all(value == 0 for value in config["operations"].values())
    assert not RUNNER.exists()
    assert not LOCK.exists()


def test_access_accounting_distinguishes_required_aggregate_label_audit() -> None:
    config = _config()
    source = config["source"]
    assert source["train_label_aggregate_reads"] == 1
    assert source["candidate_evaluation_target_reads"] == 0
    assert source["raw_rows_or_values_materialized_in_report"] == 0
    forbidden = ["official", "test", "sample_submission", "submission", "hidden", "csv", "uploads"]
    assert all(config["operations"][key] == 0 for key in forbidden)
