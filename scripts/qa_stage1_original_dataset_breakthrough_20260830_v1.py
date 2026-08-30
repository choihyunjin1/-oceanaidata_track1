"""Independently verify the aggregate-only P2/P3 Stage-1 receipts.

The read surface is deliberately fixed.  This QA never searches directories and
never opens raw training data or competition test, sample, baseline, score,
submission, context, query-support, hidden-label, prediction, or checkpoint files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

P2_RESULT = ROOT / "artifacts/p2_state_conditioned_copula_20260830_v1/result.json"
P2_CONFIG = ROOT / "configs/experiments/p2_state_conditioned_copula_20260830_v1.json"
P2_RUNNER = ROOT / "scripts/run_p2_state_conditioned_copula_20260830_v1.py"
P2_TEST = ROOT / "tests/test_p2_state_conditioned_copula_20260830_v1.py"

P3_RESULT = ROOT / "artifacts/p3_selection_matched_masked_ssl_20260830_v1/result.json"
P3_LOCK = ROOT / "artifacts/p3_selection_matched_masked_ssl_20260830_v1.ATTEMPT_LOCK.json"
P3_CONFIG = ROOT / "configs/experiments/p3_selection_matched_masked_ssl_20260830_v1.json"
P3_RUNNER = ROOT / "scripts/run_p3_selection_matched_masked_ssl_20260830_v1.py"
P3_MODULE = ROOT / "src/p3_wave/selection_matched_masked_ssl_20260830_v1.py"
P3_TEST = ROOT / "tests/test_p3_selection_matched_masked_ssl_20260830_v1.py"


class QAError(RuntimeError):
    """Raised when a sealed aggregate receipt fails independent verification."""


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


def _canonical_sha256(payload: dict[str, Any]) -> str:
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


def _qa_p2(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gate = config["promotion_gate"]
    metrics = result["metrics"]
    correction = result["correction"]
    bootstrap = result["bootstrap"]
    window_deltas = [float(item["delta_rmse"]) for item in metrics["by_window"].values()]
    layer_deltas = [float(item["delta_rmse"]) for item in metrics["by_layer"].values()]
    season_deltas = [float(item["delta_rmse"]) for item in metrics["by_season"].values()]
    recomputed_checks = {
        "at_least_two_of_three_windows_improve": sum(delta < 0.0 for delta in window_deltas)
        >= int(gate["minimum_improved_windows"]),
        "correction_p99_lte_0_2_c": float(correction["p99_absolute_c"])
        <= float(gate["maximum_correction_p99_c"]),
        "correction_rms_lte_0_075_c": float(correction["rms_c"])
        <= float(gate["maximum_correction_rms_c"]),
        "no_layer_worse_by_more_than_0_001_c": max(layer_deltas)
        <= float(gate["maximum_layer_regression_c"]),
        "paired_kst_day_bootstrap_ci90_upper_lt_0": float(bootstrap["ci90_high"])
        < float(gate["bootstrap_ci90_upper_max_c"]),
        "pooled_delta_rmse_lt_0": float(metrics["pooled"]["delta_rmse"])
        < float(gate["pooled_delta_rmse_max_c"]),
        "worst_season_regression_lte_0_003_c": max(season_deltas)
        <= float(gate["maximum_worst_season_regression_c"]),
    }
    recorded_checks = {key: bool(value) for key, value in result["promotion_checks"].items()}
    fits = result["fit_counts"]
    resource = config["resource_contract"]
    fit_contract = bool(
        int(fits["outer_dependence_model_fits"])
        == int(resource["outer_dependence_model_fits"])
        and int(fits["inner_selection_fits"]) == int(resource["inner_selection_fits"])
        and int(fits["state_cell_correlation_estimations"])
        <= int(resource["maximum_state_cell_correlation_estimations"])
    )
    access = result["access_receipt"]
    access_zero = bool(
        int(access["official_interface_rows_read"]) == 0
        and int(access["query_support_rows_read"]) == 0
        and int(access["csv_output_count"]) == 0
        and access["submission_generated"] is False
        and int(access["upload_count"]) == 0
        and int(access["hard_deleted_training_profiles"]) == 0
    )
    execution = result["execution_receipt"]
    one_shot = bool(
        int(execution["attempts"]) == 1
        and execution["result_based_retry"] is False
        and execution["result_based_tuning"] is False
        and execution["technical_failure_retry"] is False
    )
    _require(result["decision"] == "NO_GO_STATE_CONDITIONED_COPULA_STAGE1", "P2 decision mismatch")
    _require(recorded_checks == recomputed_checks, "P2 promotion checks do not recompute")
    _require(not all(recomputed_checks.values()), "P2 unexpectedly passes every promotion gate")
    _require(fit_contract, "P2 fit contract failed")
    _require(access_zero, "P2 access/deletion contract failed")
    _require(one_shot, "P2 one-shot contract failed")
    _require(result["config_sha256"] == _sha256(P2_CONFIG), "P2 config byte hash mismatch")
    _require(result["config_canonical_sha256"] == _canonical_sha256(config), "P2 config canonical hash mismatch")
    return {
        "decision": result["decision"],
        "classification": result["classification"],
        "pooled_reference_rmse_c": float(metrics["pooled"]["reference_rmse"]),
        "pooled_candidate_rmse_c": float(metrics["pooled"]["candidate_rmse"]),
        "pooled_delta_candidate_minus_reference_c": float(metrics["pooled"]["delta_rmse"]),
        "window_deltas_c": {
            name: float(item["delta_rmse"]) for name, item in metrics["by_window"].items()
        },
        "season_deltas_c": {
            name: float(item["delta_rmse"]) for name, item in metrics["by_season"].items()
        },
        "bootstrap_ci90_c": [float(bootstrap["ci90_low"]), float(bootstrap["ci90_high"])],
        "recomputed_promotion_checks": recomputed_checks,
        "failed_promotion_checks": [key for key, value in recomputed_checks.items() if not value],
        "fit_counts": fits,
        "elapsed_seconds": float(result["runtime"]["elapsed_seconds"]),
        "hard_deleted_training_profiles": int(access["hard_deleted_training_profiles"]),
        "access_csv_submission_upload_zero": access_zero,
        "one_shot_no_tuning_or_retry": one_shot,
        "one_shot_observed_from_receipt": True,
        "one_shot_fail_closed_enforced_before_fits": False,
        "execution_runner_cryptographically_bound_in_receipt": False,
        "provenance_disclosure": (
            "The receipt records one observed execution with no tuning or retry, but the "
            "runner creates its exclusive result only after fitting and has no pre-fit "
            "attempt lock; the Stage-1 runner hash is therefore not bound into the receipt."
        ),
        "result_sha256": _sha256(P2_RESULT),
        "config_sha256": _sha256(P2_CONFIG),
        "runner_sha256": _sha256(P2_RUNNER),
        "test_sha256": _sha256(P2_TEST),
    }


def _qa_p3(
    result: dict[str, Any], config: dict[str, Any], attempt_lock: dict[str, Any]
) -> dict[str, Any]:
    comparison = result["paired_comparison"]
    metrics = comparison["metrics"]
    overall = metrics["overall"]
    bootstrap = comparison["paired_case_bootstrap"]
    gate = config["evaluation"]["promotion_gate"]
    integrity = {key: bool(value) for key, value in result["integrity_checks"].items()}
    window_deltas = [
        float(item["delta_candidate_minus_incumbent_m"])
        for item in metrics["by_window"].values()
    ]
    lead_deltas = [
        float(item["delta_candidate_minus_incumbent_m"])
        for item in metrics["by_lead"].values()
    ]
    pooled_delta = float(overall["delta_candidate_minus_incumbent_m"])
    recomputed_checks = {
        "all_integrity_checks_pass": all(integrity.values()),
        "pooled_improvement_at_least_preregistered_margin": pooled_delta
        <= -float(gate["minimum_pooled_improvement_vs_paired_incumbent_m"]),
        "paired_case_ci90_upper_below_zero": float(
            bootstrap["delta_candidate_minus_incumbent_ci90_m"][1]
        )
        < 0.0,
        "minimum_improved_forward_windows": sum(delta < 0.0 for delta in window_deltas)
        >= int(gate["minimum_improved_forward_windows"]),
        "worst_lead_regression_within_cap": max(lead_deltas)
        <= float(gate["maximum_worst_lead_regression_m"]),
    }
    promotion = result["promotion_gate"]
    recorded_checks = {key: bool(value) for key, value in promotion["checks"].items()}
    fits = result["fit_budget"]
    actual = fits["actual"]
    preregistered = fits["preregistered"]
    fit_contract = bool(
        actual == {
            "masked_encoder_fits": int(preregistered["masked_encoder_fits"]),
            "huber_head_fits": int(preregistered["huber_head_fits"]),
            "reference_router_fits": int(preregistered["reference_router_fits"]),
            "catboost_fits": int(preregistered["catboost_fits"]),
            "total_fit_calls": int(preregistered["total_fit_calls"]),
        }
        and int(fits["actual_encoder_optimizer_steps"])
        <= int(preregistered["maximum_encoder_optimizer_steps"])
    )
    access = result["data_access"]
    access_zero = bool(
        int(access["official_test_rows_read"]) == 0
        and int(access["official_context_rows_read"]) == 0
        and int(access["sample_rows_read"]) == 0
        and int(access["baseline_rows_read"]) == 0
        and int(access["score_rows_read"]) == 0
        and int(access["submission_rows_read"]) == 0
        and int(access["hidden_or_answer_rows_read"]) == 0
        and int(access["csv_output_count"]) == 0
        and int(access["upload_attempt_count"]) == 0
        and int(access["source_rows_modified_or_deleted"]) == 0
    )
    sensor_flags = result["sensor_error_flags"]
    outlier_zero = bool(
        int(sensor_flags["rows_deleted_or_masked"]) == 0
        and int(sensor_flags["high_wave_and_rapid_rise_rows_deleted"]) == 0
        and int(sensor_flags["extreme_storm_rows_deleted"]) == 0
    )
    without_seal = copy.deepcopy(result)
    recorded_seal = without_seal.pop("seal")["payload_without_seal_sha256"]
    recomputed_seal = _canonical_sha256(without_seal)
    embedded_attempt = copy.deepcopy(result["one_shot_attempt"])
    embedded_lock_sha256 = embedded_attempt.pop("sha256")
    attempt_lock_matches_result = bool(
        embedded_attempt == attempt_lock and embedded_lock_sha256 == _sha256(P3_LOCK)
    )
    implementation = result["one_shot_attempt"]["implementation_sha256"]
    implementation_hashes_match = bool(
        implementation["config"] == _sha256(P3_CONFIG)
        and implementation["runner"] == _sha256(P3_RUNNER)
        and implementation["implementation"] == _sha256(P3_MODULE)
        and implementation["focused_tests"] == _sha256(P3_TEST)
    )
    _require(result["decision"] == "NO_GO_CLOSE_THIS_EXACT_RECIPE", "P3 decision mismatch")
    _require(promotion["passed"] is False, "P3 promotion status mismatch")
    _require(recorded_checks == recomputed_checks, "P3 promotion checks do not recompute")
    _require(int(promotion["improved_forward_window_count"]) == 0, "P3 improved-window count changed")
    _require(fit_contract, "P3 fit contract failed")
    _require(access_zero, "P3 access contract failed")
    _require(outlier_zero, "P3 sensor/extreme deletion detected")
    _require(recorded_seal == recomputed_seal, "P3 payload seal mismatch")
    _require(attempt_lock_matches_result, "P3 attempt lock/result binding mismatch")
    _require(implementation_hashes_match, "P3 execution implementation hash mismatch")
    _require(result["execution"]["result_based_tuning_or_retry"] is False, "P3 tuning/retry detected")
    return {
        "decision": result["decision"],
        "candidate_rmse_m": float(overall["candidate_rmse_m"]),
        "paired_incumbent_rmse_m": float(overall["paired_incumbent_rmse_m"]),
        "persistence_rmse_m": float(overall["persistence_rmse_m"]),
        "pooled_delta_candidate_minus_incumbent_m": pooled_delta,
        "window_deltas_m": {
            name: float(item["delta_candidate_minus_incumbent_m"])
            for name, item in metrics["by_window"].items()
        },
        "paired_case_bootstrap_ci90_m": [
            float(bootstrap["delta_candidate_minus_incumbent_ci90_m"][0]),
            float(bootstrap["delta_candidate_minus_incumbent_ci90_m"][1]),
        ],
        "recomputed_promotion_checks": recomputed_checks,
        "failed_promotion_checks": [key for key, value in recomputed_checks.items() if not value],
        "integrity_failed_checks": [key for key, value in integrity.items() if not value],
        "fit_counts": actual,
        "encoder_optimizer_steps": int(fits["actual_encoder_optimizer_steps"]),
        "elapsed_seconds": float(fits["total_elapsed_seconds"]),
        "sensor_or_extreme_rows_deleted_or_masked": 0,
        "access_csv_submission_upload_zero": access_zero,
        "payload_seal_match": recorded_seal == recomputed_seal,
        "attempt_lock_matches_result_one_shot_attempt": attempt_lock_matches_result,
        "execution_implementation_hashes_match_current_files": implementation_hashes_match,
        "result_sha256": _sha256(P3_RESULT),
        "attempt_lock_sha256": _sha256(P3_LOCK),
        "config_sha256": _sha256(P3_CONFIG),
        "runner_sha256": _sha256(P3_RUNNER),
        "module_sha256": _sha256(P3_MODULE),
        "test_sha256": _sha256(P3_TEST),
    }


def run() -> dict[str, Any]:
    p2_result = _load(P2_RESULT)
    p2_config = _load(P2_CONFIG)
    p3_result = _load(P3_RESULT)
    p3_lock = _load(P3_LOCK)
    p3_config = _load(P3_CONFIG)
    p2 = _qa_p2(p2_result, p2_config)
    p3 = _qa_p3(p3_result, p3_config, p3_lock)
    return {
        "schema_version": "original_dataset_breakthrough_stage1_independent_qa.20260830.v1",
        "status": "PASS_BOTH_TERMINAL_NO_GO_WITH_P2_PROVENANCE_DISCLOSURE",
        "conclusion": "No Stage-1 candidate passed its preregistered local promotion gate.",
        "p2": p2,
        "p3": p3,
        "qa_read_surface": [
            path.resolve().relative_to(ROOT).as_posix()
            for path in (
                P2_RESULT,
                P2_CONFIG,
                P2_RUNNER,
                P2_TEST,
                P3_RESULT,
                P3_LOCK,
                P3_CONFIG,
                P3_RUNNER,
                P3_MODULE,
                P3_TEST,
            )
        ],
        "data_access": {
            "raw_training_rows_read_by_qa": 0,
            "official_test_rows_read": 0,
            "test_index_rows_read": 0,
            "official_context_rows_read": 0,
            "query_support_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_rows_read": 0,
            "score_rows_read": 0,
            "submission_rows_read": 0,
            "hidden_or_answer_rows_read": 0,
            "prediction_rows_read": 0,
            "checkpoint_files_read": 0,
            "csv_created": 0,
            "uploads": 0,
        },
    }


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
