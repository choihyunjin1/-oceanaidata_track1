"""Independent aggregate-only QA for the sealed P2 continuous copula v2 result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_availability_aware_continuous_sparse_copula_20260830_v2"
RESULT_PATH = ROOT / "reports" / EXPERIMENT_ID / "result.json"
OUTPUT_PATH = ROOT / "reports" / EXPERIMENT_ID / "independent-qa.json"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
CORE_PATH = ROOT / "src/p2_restore" / f"{EXPERIMENT_ID}.py"
RUNNER_PATH = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
EXPECTED_CONFIG_CANONICAL_SHA256 = (
    "b67f7017603f4556624396a12407df34d11a7ede1360104a3d0444ef58c8cb0b"
)


class QaError(RuntimeError):
    """Raised when independent aggregate QA fails."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_decision(delta: float, ci_low: float, ci_high: float) -> str:
    if delta < 0.0 and ci_high < 0.0:
        return "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
    if delta < 0.0 and ci_low <= 0.0 <= ci_high:
        return "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
    if delta > 0.0 and ci_low > 0.0:
        return "PRIMARY_HARM_RESEARCH_ONLY"
    return "INCONCLUSIVE_RESEARCH_ONLY"


def qa() -> dict[str, Any]:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    pooled = result["metrics"]["pooled"]
    bootstrap = result["dependence_aware_bootstrap"]
    delta = float(pooled["delta_rmse"])
    recomputed_delta = float(pooled["candidate_rmse"]) - float(pooled["reference_rmse"])
    decision = _expected_decision(
        delta,
        float(bootstrap["ci90_low"]),
        float(bootstrap["ci90_high"]),
    )
    folds = result["fold_receipts"]
    domain_receipts = [fold["physical_domain_guard"] for fold in folds.values()]
    no_op_partitions = [
        int(fold["inactive_profiles"])
        == sum(int(value) for value in fold["no_op_profile_counts"].values())
        for fold in folds.values()
    ]
    access = result["access_receipt"]
    execution = result["execution_receipt"]
    fit = result["fit_counts"]
    implementation = result["implementation_hashes"]
    checks = {
        "experiment_id": result["experiment_id"] == EXPERIMENT_ID,
        "research_only": result["classification"]
        == "RESEARCH_ONLY_EXPOSED_HISTORICAL_SURFACE",
        "config_canonical": _canonical_sha256(config)
        == EXPECTED_CONFIG_CANONICAL_SHA256
        == result["config_canonical_sha256"],
        "config_file_hash": _sha256_file(CONFIG_PATH) == result["config_sha256"],
        "core_file_hash": _sha256_file(CORE_PATH) == implementation["core"],
        "runner_file_hash": _sha256_file(RUNNER_PATH) == implementation["runner"],
        "base_config_hash": config["base_experiment"]["config_sha256"]
        == result["base_config_sha256"],
        "pooled_delta_arithmetic": abs(delta - recomputed_delta) <= 1e-15,
        "pooled_rows": int(pooled["rows"]) == 69_850,
        "window_rows_partition": sum(
            int(record["rows"]) for record in result["metrics"]["by_window"].values()
        )
        == 69_850,
        "layer_rows_partition": sum(
            int(record["rows"]) for record in result["metrics"]["by_layer"].values()
        )
        == 69_850,
        "season_rows_partition": sum(
            int(record["rows"]) for record in result["metrics"]["by_season"].values()
        )
        == 69_850,
        "decision_recomputed": decision == result["decision"]
        == result["primary_decision_receipt"]["evidence_state"],
        "primary_point_flag": result["primary_decision_receipt"][
            "pooled_point_favorable"
        ]
        == (delta < 0.0),
        "primary_interval_flag": result["primary_decision_receipt"][
            "paired_interval_wholly_favorable"
        ]
        == (float(bootstrap["ci90_high"]) < 0.0),
        "bootstrap_contract": int(bootstrap["replicates"]) == 5000
        and int(bootstrap["block_length_days"]) == 7
        and float(bootstrap["confidence"]) == 0.9
        and int(bootstrap["seed"]) == 20260830
        and bootstrap["layers_preserved_together_within_day"]
        and bootstrap["windows_resampled_separately"],
        "fit_budget": int(fit["outer_dependence_model_fits"]) == 3
        and int(fit["continuous_edge_estimations"]) == 21
        and int(fit["inner_selection_fits"]) == 0
        and int(fit["hpo_trials"]) == 0,
        "fixed_edges": len(result["stage0_exposed_edges"]) == 7
        and len(set(result["stage0_exposed_edges"])) == 7,
        "correction_bound": float(result["correction"]["maximum_absolute_c"])
        <= 0.2 + 1e-12
        and result["correction"]["structural_bound_c"] == [-0.2, 0.2],
        "relative_domain_guard": all(
            receipt["candidate_all_finite"]
            and int(receipt["new_candidate_outside_count"]) == 0
            and int(receipt["changed_preexisting_extreme_count"]) == 0
            and int(receipt["active_candidate_outside_count"]) == 0
            and int(receipt["inactive_changed_count"]) == 0
            and not receipt["prediction_values_changed_by_overlay"]
            for receipt in domain_receipts
        ),
        "no_op_partitions": all(no_op_partitions),
        "tail_is_diagnostic_only": result["diagnostics"]["tail_risk"]["role"]
        == "DIAGNOSTIC_SENSITIVITY_ONLY_NOT_A_PROMOTION_GATE",
        "no_diagnostic_hard_veto": int(
            result["primary_decision_receipt"]["diagnostic_slice_hard_veto_count"]
        )
        == 0,
        "source_only_observations": result["source_basenames_opened"]
        == ["observations.csv"]
        and result["source_open_counts"] == {"observations.csv": 1},
        "zero_official_csv_upload": int(access["official_interface_rows_read"]) == 0
        and int(access["query_support_rows_read"]) == 0
        and int(access["csv_output_count"]) == 0
        and not access["submission_generated"]
        and int(access["upload_count"]) == 0,
        "zero_hard_deletion": int(access["hard_deleted_training_profiles"]) == 0,
        "one_shot_no_tuning_retry": int(execution["attempts"]) == 1
        and not execution["v1_reexecution"]
        and not execution["result_based_tuning"]
        and not execution["result_based_retry"]
        and not execution["technical_failure_retry"]
        and execution["aggregate_json_only"],
        "runtime_bound": float(result["runtime"]["elapsed_seconds"])
        <= float(result["runtime"]["maximum_wall_seconds"]),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise QaError(f"independent QA checks failed: {failed}")
    return {
        "schema_version": "p2.availability_aware_continuous_sparse_copula.independent_qa.20260830.v2",
        "experiment_id": EXPERIMENT_ID,
        "qa_status": "PASS",
        "result": {
            "path": str(RESULT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "bytes": RESULT_PATH.stat().st_size,
            "sha256": _sha256_file(RESULT_PATH),
        },
        "recomputed_primary": {
            "rows": int(pooled["rows"]),
            "reference_rmse_c": float(pooled["reference_rmse"]),
            "candidate_rmse_c": float(pooled["candidate_rmse"]),
            "delta_candidate_minus_reference_rmse_c": recomputed_delta,
            "ci90_low_c": float(bootstrap["ci90_low"]),
            "ci90_high_c": float(bootstrap["ci90_high"]),
            "decision": decision,
        },
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": failed,
        "official_interface_rows_read": 0,
        "query_support_rows_read": 0,
        "csv_output_count": 0,
        "upload_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.result_json.resolve(strict=True) != RESULT_PATH.resolve(strict=True):
        raise QaError("--result-json must equal the sealed v2 result path")
    output = args.output_json.resolve(strict=False)
    if output != OUTPUT_PATH.resolve(strict=False):
        raise QaError("--output-json must equal the sealed v2 QA path")
    if output.exists():
        raise FileExistsError(output)
    receipt = qa()
    payload = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(
        json.dumps(
            {
                "qa_status": receipt["qa_status"],
                "checks": receipt["check_count"],
                "decision": receipt["recomputed_primary"]["decision"],
                "result_sha256": receipt["result"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
