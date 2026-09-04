from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v51_cross_station_predictive_causality_support_gate_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_target_free_timestamp_support_passes_but_information_stability_fails() -> None:
    config = _config()
    support = config["target_free_support"]
    gate = config["target_free_support_gate"]
    assert support["exact_three_station_rows_before_cutoff"] >= gate["minimum_exact_common_layer1_rows_before_cutoff"]
    assert all(value >= gate["minimum_restricted_fit_rows_per_station"] for value in [1827, 1830, 1830])
    assert all(value >= gate["minimum_validation_rows_per_station"] for value in [7567, 7578, 7578])
    assert support["stations_with_positive_overall_increment_gain"] == 2
    assert support["stations_with_strict_majority_positive_days"] == 0
    assert support["support_gate"].startswith("FAIL_")


def test_no_station_has_strict_daily_majority() -> None:
    config = _config()
    support = config["target_free_support"]
    days = support["pre_q2_validation_days"]
    assert days == 61
    assert all(item["positive_days"] * 2 <= days for item in support["increment_predictive_gain"].values())
    assert all(item["median_daily_mse_gain"] < 0.0 for item in support["increment_predictive_gain"].values())


def test_fixed_candidate_is_past_only_and_bounded() -> None:
    config = _config()
    candidate = config["candidate_sealed_before_target_access"]
    assert candidate["lags_rows"] == [1, 6, 36]
    assert candidate["ridge_alpha"] == 10.0
    assert candidate["past_only"] is True and candidate["future_interpolation"] == 0
    assert candidate["fits_if_ready"] == 3 and candidate["maximum_fits"] == 9
    assert candidate["threshold_quantiles"] == [0.995, 0.9975, 0.999]
    assert candidate["sweep"] == 0 and candidate["anchor_removals"] == 0


def test_repository_wide_predictive_causality_overlap_is_explicit() -> None:
    config = _config()
    audit = config["repository_negative_fingerprint"]
    assert audit["exact_p1_implementation"] is False
    assert audit["repository_wide_semantic_duplicate"] is True
    assert "restricted-versus-unrestricted" in audit["p3_pairwise_predictive_causality"]["relation"]
    assert "unrelated-regime contamination" in audit["p1_cross_station_peer_residual"]["relation"]


def test_zero_fit_no_runner_lock_or_forbidden_access() -> None:
    config = _config()
    assert config["decision"]["ready"] is False
    assert config["status"].startswith("NO_GO_ZERO_FIT")
    assert all(value == 0 for value in config["operations"].values())
    assert config["source"]["target_columns_read"] == 0
    assert not RUNNER.exists()
    assert not LOCK.exists()


def test_v28_v33_add_only_contract_preserved() -> None:
    config = _config()
    contract = config["contracts_preserved"]
    assert contract["cross_quarter_guard"]["sha256"] == "a051d4d0837b395f9c1d42d71d65572efb99d6c1aeae6aabdb52c8427539c8d6"
    assert contract["auditability_amendment"]["sha256"] == "a20cf248c3c4cd4ced858deccca1fbb52f4e1ed114582988d9957018a7e43128"
    assert contract["add_only"] is True and contract["anchor_removals"] == 0
    assert contract["maximum_fits"] == 9
    assert contract["result_based_threshold_or_gate_changes"] == 0
