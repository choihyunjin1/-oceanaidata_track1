"""Independent aggregate-only QA for the sealed P3 Bayesian RFF experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_selection_matched_sparse_gp_abstention_20260830_v1"
CONFIG_RELATIVE = f"configs/experiments/{EXPERIMENT_ID}.json"
EXPECTED_CONFIG_SHA256 = "d941e3516bc295e8e10c7dffbd4df4ccee0ed276d98a0b67f8e7875764717611"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _expected_evidence_state(point: float, interval: list[float], fatal: bool) -> str:
    if not fatal:
        return "QA_BLOCKED"
    low, high = map(float, interval)
    if point > 0.0 and low > 0.0:
        return "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
    if point > 0.0:
        return "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
    if point < 0.0 and high < 0.0:
        return "PRIMARY_HARM_RESEARCH_ONLY"
    return "INCONCLUSIVE_RESEARCH_ONLY"


def run_qa(config_path: Path | None = None) -> dict[str, Any]:
    canonical_config = (ROOT / CONFIG_RELATIVE).resolve(strict=True)
    requested = (config_path or canonical_config).resolve(strict=True)
    if requested != canonical_config:
        raise ValueError("non-canonical config path is forbidden")
    if sha256_file(canonical_config) != EXPECTED_CONFIG_SHA256:
        raise ValueError("config SHA changed")
    config = json.loads(canonical_config.read_text(encoding="utf-8"))
    paths = config["canonical_paths"]
    result_path = ROOT / paths["result"]
    lock_path = ROOT / paths["attempt_lock"]
    failure_path = ROOT / paths["failure_receipt"]
    qa_path = ROOT / paths["qa_result"]
    if qa_path.exists():
        raise ValueError("independent QA receipt already exists")
    if not result_path.is_file() or not lock_path.is_file():
        raise ValueError("sealed result or one-shot lock is missing")
    if failure_path.exists():
        raise ValueError("failure receipt exists; completed result is not admissible")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    seal = result.get("seal", {})
    result_without_seal = {key: value for key, value in result.items() if key != "seal"}
    primary = result["primary_evaluation"]
    metrics = primary["metrics"]
    overall = metrics["overall"]
    interval = primary["dependence_aware_interval"]
    fatal = result["validity"]["fatal_hard_gates"]
    benefit = float(overall["benefit_incumbent_minus_candidate_rmse_m"])
    checks = {
        "result_schema_and_research_only_status": bool(
            result["schema_version"]
            == "p3.selection_matched_sparse_gp_abstention.result.20260830.v1"
            and result["status"] == "COMPLETE_TRAIN_ONLY_ONE_SHOT_RESEARCH_ONLY"
            and result["official_action_authorized"] is False
        ),
        "result_seal_recomputes": bool(
            seal.get("algorithm") == "sha256"
            and seal.get("payload_without_seal_sha256") == _payload_sha256(result_without_seal)
        ),
        "attempt_lock_hash_and_identity_match": bool(
            result["one_shot_attempt"]["sha256"] == sha256_file(lock_path)
            and lock["experiment_id"] == EXPERIMENT_ID
            and lock["status"] == "ATTEMPT_CONSUMED_ONE_SHOT"
            and lock["rerun_forbidden"] is True
        ),
        "implementation_snapshot_still_matches": bool(
            all(
                sha256_file(ROOT / paths[name]) == digest
                for name, digest in result["provenance"]["implementation_sha256"].items()
            )
        ),
        "governing_policy_hash_matches": bool(
            sha256_file(ROOT / result["provenance"]["governing_policy_path"])
            == result["provenance"]["governing_policy_sha256"]
        ),
        "source_and_dependency_hashes_stable": bool(
            result["provenance"]["source_sha256_before"]
            == result["provenance"]["source_sha256_after"]
            and result["provenance"]["dependency_sha256_before"]
            == result["provenance"]["dependency_sha256_after"]
        ),
        "candidate_and_incumbent_keys_match": bool(
            result["provenance"]["candidate_key_sha256"]
            == result["provenance"]["incumbent_key_sha256"]
            == result["cohort"]["validation_key_sha256"]
        ),
        "cohort_and_six_lead_row_counts_exact": bool(
            result["cohort"]["canonical_anchor_count"] == 8121
            and result["cohort"]["selection_matched_dense_count"] == 2131
            and result["cohort"]["validation_union_independent_count"] == 157
            and result["cohort"]["validation_by_window"]
            == {"2024_h2_storm": 41, "winter_transition": 65, "2025_h1": 51}
            and overall["cases"] == 157
            and overall["rows"] == 942
            and sum(item["rows"] for item in metrics["by_window"].values()) == 942
            and sum(item["rows"] for item in metrics["by_station"].values()) == 942
            and set(metrics["by_lead"]) == {"3", "6", "9", "12", "18", "24"}
            and all(item["rows"] == 157 for item in metrics["by_lead"].values())
        ),
        "pooled_benefit_arithmetic_and_interval_match": bool(
            np.isclose(
                benefit,
                float(overall["paired_incumbent_rmse_m"])
                - float(overall["candidate_rmse_m"]),
                rtol=0.0,
                atol=1.0e-15,
            )
            and np.isclose(
                benefit,
                float(interval["benefit_incumbent_minus_candidate_point_m"]),
                rtol=0.0,
                atol=1.0e-15,
            )
            and interval["unit"]
            == "forward_window_stratified_contiguous_anchor_day_block_with_six_leads_intact"
            and interval["replicates"] == 5000
            and interval["block_length_anchor_days"] == 3
        ),
        "evidence_state_recomputes_without_numeric_margin": bool(
            result["evidence_state"]
            == _expected_evidence_state(benefit, interval["benefit_ci90_m"], all(fatal.values()))
        ),
        "all_fatal_integrity_gates_pass": bool(
            fatal and all(fatal.values()) and result["validity"]["all_fatal_hard_gates_pass"]
        ),
        "fit_hpo_and_abstention_contract_match": bool(
            result["fit_budget"]["actual_candidate_fits"] == 3
            and result["fit_budget"]["hyperparameter_search_count"] == 0
            and result["fit_budget"]["reference_router_fits"] == 2
            and result["fit_budget"]["catboost_refits"] == 0
            and result["abstention"]["total_rows"] == 942
            and result["abstention"]["active_correction_rows"] <= 942
            and result["abstention"]["exact_incumbent_rows"] <= 942
            and result["abstention"]["maximum_absolute_correction_m"] <= 0.1 + 1.0e-15
        ),
        "no_legacy_or_arbitrary_performance_veto": bool(
            primary["decision_basis"]["arbitrary_numeric_magnitude_tier_applied"] is False
            and primary["decision_basis"]["legacy_minimum_0_01m_applied"] is False
            and primary["decision_basis"]["legacy_two_of_three_windows_applied"] is False
            and primary["decision_basis"]["legacy_worst_lead_0_02m_cap_applied"] is False
            and primary["decision_basis"]["window_station_lead_results_diagnostic_only"] is True
        ),
        "forbidden_reads_outputs_deletions_and_retry_zero": bool(
            result["data_access"]["opened_source_basenames"]
            == ["README.md", "train_wave.csv", "train_atmos.csv"]
            and result["data_access"]["forbidden_source_basenames_opened"] == []
            and all(
                result["data_access"][key] == 0
                for key in (
                    "official_test_rows_read",
                    "official_context_rows_read",
                    "test_index_rows_read",
                    "sample_rows_read",
                    "baseline_rows_read",
                    "score_rows_read",
                    "submission_rows_read_or_written",
                    "hidden_or_answer_rows_read",
                    "csv_output_count",
                    "upload_attempt_count",
                    "raw_prediction_rows_persisted",
                    "source_rows_modified_or_deleted",
                )
            )
            and result["sensor_error_flags"]["flags_used_for_cohort_membership_or_weighting"]
            is False
            and result["sensor_error_flags"]["extreme_storm_rows_deleted"] == 0
            and result["execution"]["result_based_tuning_or_retry"] is False
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": "p3.selection_matched_sparse_gp_abstention.independent_qa.20260830.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "status": "INDEPENDENT_AGGREGATE_QA_PASS" if all(checks.values()) else "INDEPENDENT_QA_FAIL",
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "result_sha256": sha256_file(result_path),
        "attempt_lock_sha256": sha256_file(lock_path),
        "config_sha256": sha256_file(canonical_config),
        "evidence_state": result["evidence_state"],
        "primary": overall,
        "benefit_ci90_m": interval["benefit_ci90_m"],
        "candidate_fits": result["fit_budget"]["actual_candidate_fits"],
        "hyperparameter_search_count": result["fit_budget"]["hyperparameter_search_count"],
        "raw_rows_opened_by_qa": 0,
        "official_interface_rows_opened_by_qa": 0,
    }
    payload["seal"] = {
        "algorithm": "sha256",
        "payload_without_seal_sha256": _payload_sha256(payload),
    }
    digest = _write_exclusive_json(qa_path, payload)
    return {
        "status": payload["status"],
        "all_checks_pass": payload["all_checks_pass"],
        "qa_result": paths["qa_result"],
        "qa_result_sha256": digest,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_qa(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0 if result["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
