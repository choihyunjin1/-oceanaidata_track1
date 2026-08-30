"""Independently verify aggregate-only receipts from the 2026-08-30 preflights.

This script has a fixed, explicit read surface.  It never searches directories and
never opens competition test, sample, submission, baseline, score, or hidden files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/original_dataset_breakthrough_research_20260830_v1"

P1_RESULT = ROOT / "artifacts/p1_heterogeneous_event_utility_preflight_20260830_v1/terminal_result.json"
P2_RESULT = ROOT / "artifacts/p2_state_conditioned_copula_preflight_20260830_v1/result.json"
P3_RESULT = ROOT / "artifacts/p3_selection_matched_cohort_preflight_20260830_v1/preflight.json"
DATASET_AUDIT = REPORT_DIR / "dataset-audit.json"

CONFIGS = {
    "p1": ROOT / "configs/experiments/p1_heterogeneous_event_utility_preflight_20260830_v1.json",
    "p2": ROOT / "configs/experiments/p2_state_conditioned_copula_preflight_20260830_v1.json",
    "p3": ROOT / "configs/experiments/p3_selection_matched_cohort_preflight_20260830_v1.json",
}
RUNNERS = {
    "p1": ROOT / "scripts/run_p1_heterogeneous_event_utility_preflight_20260830_v1.py",
    "p2": ROOT / "scripts/run_p2_state_conditioned_copula_preflight_20260830_v1.py",
    "p3": ROOT / "scripts/run_p3_selection_matched_cohort_preflight_20260830_v1.py",
}
TESTS = {
    "audit": ROOT / "tests/test_audit_original_training_structure_20260830_v1.py",
    "p1": ROOT / "tests/test_p1_heterogeneous_event_utility_preflight_20260830_v1.py",
    "p2": ROOT / "tests/test_p2_state_conditioned_copula_preflight_20260830_v1.py",
    "p3": ROOT / "tests/test_p3_selection_matched_cohort_preflight_20260830_v1.py",
}


class QAError(RuntimeError):
    """Raised when a terminal aggregate receipt cannot be independently verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QAError(f"expected JSON object: {path.name}")
    return value


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QAError(message)


def _qa_p1(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["support_gates"]
    first = result["prefixes"]["q2_to_q3"]
    second = result["prefixes"]["q2_q3_to_q4"]
    q4 = second["calibration"]
    q4_event_count = int(q4["utility_positive_events"])
    q4_minimum = int(gates["minimum_calibration_utility_positive_events"])
    q4_precision = float(q4["proposal_precision"])
    q4_requirement = float(q4["proposal_precision_requirement_f1_over_2"])
    recomputed_no_go = bool(
        first["pass"] is True
        and q4_event_count < q4_minimum
        and q4_precision < q4_requirement
        and result["provenance"]["complete"] is True
    )
    execution = result["execution_audit"]
    zero_execution = all(
        int(execution[key]) == 0
        for key in (
            "model_fit_count",
            "threshold_search_count",
            "prediction_materialization_count",
            "prediction_csv_count",
            "official_interface_rows_read",
            "raw_training_rows_read",
            "raw_temp_rows_read",
            "auxiliary_psal_depth_rows_read",
            "target_positive_rows_removed",
            "anchor_positive_removed_rows",
            "upload_count",
        )
    )
    _require(result["status"] == "NO_GO_ZERO_FIT_SUPPORT_PREFLIGHT", "P1 status mismatch")
    _require(recomputed_no_go, "P1 NO_GO gate did not recompute")
    _require(zero_execution, "P1 zero-execution contract failed")
    _require(result["config_sha256"] == _sha256(CONFIGS["p1"]), "P1 config hash mismatch")
    runner_postdates_receipt = RUNNERS["p1"].stat().st_mtime_ns > P1_RESULT.stat().st_mtime_ns
    _require(runner_postdates_receipt, "P1 post-run hardening chronology changed")
    return {
        "status": result["status"],
        "recomputed_no_go": recomputed_no_go,
        "q2_to_q3_pass": bool(first["pass"]),
        "q4_utility_positive_events": q4_event_count,
        "q4_minimum_events": q4_minimum,
        "q4_proposal_precision": q4_precision,
        "q4_precision_requirement_f1_over_2": q4_requirement,
        "anchor_positive_rows_removed": int(result["proposal_bank"]["anchor_positive_removed_rows"]),
        "zero_execution_and_official_access": zero_execution,
        "receipt_sha256": _sha256(P1_RESULT),
        "current_runner_sha256": _sha256(RUNNERS["p1"]),
        "current_runner_postdates_receipt": runner_postdates_receipt,
        "post_run_hardening_scope": (
            "exclusive receipt creation, exact rational concentration boundary, and strict "
            "boolean provenance checks; observed values are away from those boundaries, so "
            "the Q4 3<4 NO_GO verdict is invariant"
        ),
        "execution_runner_byte_sha256_available": False,
    }


def _qa_p2(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    supported = [
        item for item in result["state_cell_support"] if item["passes_overlap_support_gate"]
    ]
    passing_edges = list(result["kendall_heterogeneity"]["passing_edges"])
    required_edges = int(config["gates"]["minimum_passing_edges"])
    checks_recomputed = {
        "supported_cells_gte_2": len(supported) >= 2,
        "all_supported_cells_meet_profile_gate": all(
            int(item["profiles"]) >= int(config["gates"]["minimum_profiles_per_state_cell"])
            for item in supported
        ),
        "all_supported_cells_meet_day_gate": all(
            int(item["kst_days"]) >= int(config["gates"]["minimum_kst_days_per_state_cell"])
            for item in supported
        ),
        "all_supported_cells_meet_block_gate": all(
            int(item["chronological_blocks"])
            >= int(config["gates"]["minimum_chronological_blocks_per_state_cell"])
            for item in supported
        ),
        "passing_edges_gte_2": len(passing_edges) >= required_edges,
    }
    outlier = result["outlier_weight_diagnostic"]["global"]
    zero_execution = bool(
        int(result["model_fit_count"]) == 0
        and int(result["official_input_rows_read"]) == 0
        and int(result["csv_output_count"]) == 0
        and result["submission_generated"] is False
        and int(result["upload_count"]) == 0
    )
    _require(result["status"] == "TRAIN_ONLY_ZERO_FIT_PREFLIGHT_PASS", "P2 status mismatch")
    _require(all(checks_recomputed.values()), "P2 support gates did not recompute")
    _require(all(bool(value) for value in result["checks"].values()), "P2 recorded check failed")
    _require(len(passing_edges) == 7 and required_edges == 2, "P2 edge count changed")
    _require(int(outlier["hard_deleted_profiles"]) == 0, "P2 outlier deletion detected")
    _require(zero_execution, "P2 zero-execution contract failed")
    _require(result["config_sha256"] == _sha256(CONFIGS["p2"]), "P2 config hash mismatch")
    return {
        "status": result["status"],
        "supported_state_cells": len(supported),
        "passing_edges": passing_edges,
        "passing_edge_count": len(passing_edges),
        "minimum_passing_edges": required_edges,
        "checks_recomputed": checks_recomputed,
        "sensor_suspect_profiles_diagnostic_only": int(outlier["sensor_suspect_profiles"]),
        "physical_extreme_profiles_preserved": int(outlier["preserved_physical_extreme_profiles"]),
        "hard_deleted_profiles": int(outlier["hard_deleted_profiles"]),
        "zero_execution_and_official_access": zero_execution,
        "receipt_sha256": _sha256(P2_RESULT),
    }


def _qa_p3(
    result: dict[str, Any], config: dict[str, Any], dataset_audit: dict[str, Any]
) -> dict[str, Any]:
    support = result["support"]
    station_minimum = int(config["support_gates"]["minimum_global_independent_cases_per_station"])
    window_minimum = int(
        config["support_gates"]["minimum_independent_cases_per_complete_historical_window"]
    )
    applicable_minimum = int(
        config["support_gates"]["minimum_scientifically_applicable_complete_windows"]
    )
    station_counts = {
        str(key): int(value)
        for key, value in support["selection_matched_global_independent_by_station"].items()
    }
    windows = list(support["forward_windows"].values())
    applicable = [item for item in windows if item["scientifically_applicable_complete_footprint"]]
    recomputed_gates = {
        "all_station_counts_gte_30": all(value >= station_minimum for value in station_counts.values()),
        "applicable_windows_gte_2": len(applicable) >= applicable_minimum,
        "all_applicable_windows_gte_20": all(
            int(item["validation_selection_matched_independent_count"]) >= window_minimum
            for item in applicable
        ),
        "all_train_cutoffs_pass": all(
            bool(item["train_cutoff_strictly_respected"]) for item in windows
        ),
    }
    without_seal = copy.deepcopy(result)
    recorded_seal = without_seal.pop("seal")["payload_without_seal_sha256"]
    seal_recomputed = _canonical_payload_sha256(without_seal)
    audit_p3 = dataset_audit["p3"]
    canonical_count_match = bool(
        int(audit_p3["selection_matched_dense_anchor_count"])
        == int(support["selection_matched_dense_count"])
        and int(audit_p3["selection_matched_78h_independent_count"])
        == int(support["selection_matched_global_independent_count"])
        and audit_p3["selection_matched_78h_by_station"] == station_counts
    )
    strict = audit_p3["strict_hs_lag48_selection_diagnostic"]
    zero_execution = bool(
        int(result["execution"]["model_fit_count"]) == 0
        and int(result["execution"]["prediction_row_count"]) == 0
        and int(result["execution"]["csv_output_count"]) == 0
        and result["execution"]["submission_or_upload_attempted"] is False
        and int(result["data_access"]["official_test_rows_read"]) == 0
        and int(result["data_access"]["sample_rows_read"]) == 0
        and int(result["data_access"]["baseline_rows_read"]) == 0
        and int(result["data_access"]["submission_rows_read"]) == 0
        and int(result["data_access"]["hidden_or_answer_rows_read"]) == 0
    )
    _require(result["gates"]["overall_preflight_pass"] is True, "P3 recorded gate failed")
    _require(all(recomputed_gates.values()), "P3 support gates did not recompute")
    _require(recorded_seal == seal_recomputed, "P3 payload seal mismatch")
    _require(canonical_count_match, "P3 canonical audit/preflight count mismatch")
    _require(int(result["sensor_error_flags"]["rows_deleted_or_masked"]) == 0, "P3 deletion detected")
    _require(zero_execution, "P3 zero-execution contract failed")
    _require(result["provenance"]["config_sha256"] == _sha256(CONFIGS["p3"]), "P3 config hash mismatch")
    return {
        "status": "TRAIN_ONLY_ZERO_FIT_PREFLIGHT_PASS",
        "overall_preflight_pass": True,
        "canonical_selection_matched_dense_count": int(support["selection_matched_dense_count"]),
        "canonical_global_independent_count": int(support["selection_matched_global_independent_count"]),
        "canonical_global_by_station": station_counts,
        "applicable_window_counts": {
            name: int(item["validation_selection_matched_independent_count"])
            for name, item in support["forward_windows"].items()
            if item["scientifically_applicable_complete_footprint"]
        },
        "strict_t_minus_48h_finite_diagnostic": strict,
        "recomputed_gates": recomputed_gates,
        "canonical_dataset_audit_match": canonical_count_match,
        "payload_seal_match": recorded_seal == seal_recomputed,
        "storm_extreme_anchor_count_preserved": int(
            result["sensor_error_flags"]["canonical_hs_ge_2_2_anchor_count_preserved"]
        ),
        "rows_deleted_or_masked": int(result["sensor_error_flags"]["rows_deleted_or_masked"]),
        "zero_execution_and_official_access": zero_execution,
        "receipt_sha256": _sha256(P3_RESULT),
    }


def run() -> dict[str, Any]:
    p1 = _load(P1_RESULT)
    p2 = _load(P2_RESULT)
    p3 = _load(P3_RESULT)
    configs = {name: _load(path) for name, path in CONFIGS.items()}
    dataset_audit = _load(DATASET_AUDIT)
    result = {
        "schema_version": "original_dataset_breakthrough_independent_qa.20260830.v1",
        "status": "PASS_WITH_DISCLOSED_P1_POST_RUN_HARDENING",
        "p1": _qa_p1(p1, configs["p1"]),
        "p2": _qa_p2(p2, configs["p2"]),
        "p3": _qa_p3(p3, configs["p3"], dataset_audit),
        "current_code_hashes": {
            "configs": {name: _sha256(path) for name, path in CONFIGS.items()},
            "runners": {name: _sha256(path) for name, path in RUNNERS.items()},
            "tests": {name: _sha256(path) for name, path in TESTS.items()},
            "dataset_audit": _sha256(DATASET_AUDIT),
        },
        "qa_read_surface": [
            path.resolve().relative_to(ROOT).as_posix()
            for path in (
                P1_RESULT,
                P2_RESULT,
                P3_RESULT,
                DATASET_AUDIT,
                *CONFIGS.values(),
                *RUNNERS.values(),
                *TESTS.values(),
            )
        ],
        "data_access": {
            "raw_training_rows_read_by_qa": 0,
            "official_test_rows_read": 0,
            "test_index_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_rows_read": 0,
            "score_rows_read": 0,
            "submission_rows_read": 0,
            "hidden_or_answer_rows_read": 0,
            "csv_created": 0,
            "uploads": 0,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".json":
        raise QAError("QA output must be JSON")
    payload = json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
