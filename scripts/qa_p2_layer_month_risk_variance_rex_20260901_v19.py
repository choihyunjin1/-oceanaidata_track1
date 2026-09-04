"""Independent QA for the P2 v19 layer-month V-REx run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_layer_month_risk_variance_rex_20260901_v19 as runner  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(abs(left - right) <= tolerance)


def main() -> None:
    result_path = runner.REPORT / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    prediction_path = runner.ARTIFACT / f"{runner.PREDICTION_NAME}.npz"
    prediction = np.load(prediction_path, allow_pickle=False)
    time_ns = prediction["time_ns"].astype(np.int64)
    layer = prediction["layer"].astype(int)
    reference = prediction["reference"].astype(float)
    candidate = prediction["candidate"].astype(float)

    data_dir = os.environ.get("P2_DATA_DIR")
    if not data_dir:
        raise RuntimeError("P2_DATA_DIR is required")
    observations_path = Path(data_dir).resolve() / "observations.csv"
    observations = pd.read_csv(
        observations_path,
        dtype={"station": "string", "time": "string"},
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    design = build_normalized_curvature_design(
        build_training_features(observations).frame,
    )
    design_index = pd.MultiIndex.from_arrays(
        [
            runner.v12.metric_engine.canonical_time_ns(design.keys["time"]),
            design.keys["layer"],
        ]
    )
    positions = design_index.get_indexer(
        pd.MultiIndex.from_arrays([time_ns, layer])
    )
    if np.any(positions < 0):
        raise RuntimeError("independent truth alignment failed")
    truth = design.truth[positions]
    reference_rmse = runner.v12.metric_engine.rmse(truth, reference)
    candidate_rmse = runner.v12.metric_engine.rmse(truth, candidate)
    delta = candidate_rmse - reference_rmse
    slope = float(config["evaluation"]["points_per_rmse_C"])
    penalty = float(config["evaluation"]["transport_penalty_points"])

    record = result["candidate"]
    semantic = result["semantic_audit"]
    comparison = result["comparison_to_preserved_candidates"]
    counters = result["operation_counters"]
    prefix_ok = True
    group_mass_ok = True
    fit_receipts_ok = True
    for fold, receipt in result["training"]["folds"].items():
        expected = pd.Timestamp(config["training"]["fold_starts_kst"][fold])
        expected -= pd.Timedelta(days=int(config["training"]["embargo_days"]))
        prefix_ok &= pd.Timestamp(receipt["training_cutoff_exclusive_kst"]) == expected
        group_items = receipt["environment_receipt"]["groups"].values()
        masses = [value["raw_weight_sum"] for value in group_items]
        group_mass_ok &= bool(np.max(masses) - np.min(masses) <= 1e-12)
        expected_count = receipt["environment_receipt"]["group_count"]
        for fit in receipt["fit_receipts"]:
            fit_receipts_ok &= fit["epochs"] == 60
            fit_receipts_ok &= fit["environment_count"] == expected_count
            fit_receipts_ok &= fit["risk_variance_coefficient"] == 10.0
            fit_receipts_ok &= fit["weight_decay"] == 0.001
            fit_receipts_ok &= fit["minimum_environments_present_per_batch"] >= 2
            fit_receipts_ok &= len(fit["final_epoch_environment_risks"]) == expected_count
            fit_receipts_ok &= bool(
                np.isfinite(fit["final_epoch_environment_risk_population_variance"])
            )

    v13_record = json.loads(
        (
            ROOT
            / "reports/p2_prefix_safe_domain_balanced_deepset_20260901_v13/result.json"
        ).read_text(encoding="utf-8")
    )["candidate"]
    v18_record = json.loads(
        (
            ROOT
            / "reports/p2_regularized_layer_month_group_dro_20260901_v18/result.json"
        ).read_text(encoding="utf-8")
    )["candidate"]
    comparison_ok = (
        comparison["use"] == "ledger_only_no_posthoc_selection_or_ensemble"
        and close(comparison["v13_delta_rmse"], v13_record["delta_rmse"])
        and close(comparison["v18_delta_rmse"], v18_record["delta_rmse"])
        and close(
            comparison["v19_minus_v13_delta_rmse"],
            delta - float(v13_record["delta_rmse"]),
        )
        and close(
            comparison["v19_minus_v18_delta_rmse"],
            delta - float(v18_record["delta_rmse"]),
        )
    )
    official_names = (
        "official_test_index_rows_read",
        "sample_rows_read",
        "baseline_file_rows_read",
        "score_file_rows_read",
        "query_support_rows_read",
        "hidden_truth_rows_read",
        "submission_csv_created",
        "uploads",
    )
    checks = {
        "terminal": result["status"].startswith("EXPLORATORY_"),
        "exactly_nine_fits": result["fit_count"] == 9,
        "new_p2_vrex_axis": (
            semantic["classification"]
            == "NEW_P2_LAYER_MONTH_RISK_VARIANCE_REX_OBJECTIVE"
        ),
        "no_prior_p2_vrex_execution": (
            semantic["prior_p2_vrex_runners"] == []
            and semantic["prior_p2_vrex_artifacts"] == []
            and semantic["prior_p2_vrex_reports"] == []
        ),
        "v18_minimax_distinguished": semantic[
            "v18_minimax_adversary_not_risk_variance"
        ],
        "fixed_objective": (
            result["training"]["risk_variance_coefficient"] == 10.0
            and result["training"]["weight_decay"] == 0.001
            and result["training"]["environment_definition"]
            == "target_layer_x_calendar_month"
        ),
        "fit_receipts_valid": fit_receipts_ok,
        "prediction_rows": len(candidate) == record["prediction_commitment"]["rows"],
        "prediction_hash": (
            runner.v12.sha256_file(prediction_path)
            == record["prediction_commitment"]["sha256"]
        ),
        "config_hash": (
            runner.v12.sha256_file(runner.CONFIG) == result["hashes"]["config"]
        ),
        "runner_hash": (
            runner.v12.sha256_file(runner.RUNNER) == result["hashes"]["runner"]
        ),
        "v13_runner_hash": (
            runner.v12.sha256_file(runner.v13.RUNNER)
            == result["hashes"]["v13_runner"]
        ),
        "v18_runner_hash": (
            runner.v12.sha256_file(runner.v18.RUNNER)
            == result["hashes"]["v18_runner"]
        ),
        "reference_rmse": close(reference_rmse, record["reference_rmse"]),
        "candidate_rmse": close(candidate_rmse, record["candidate_rmse"]),
        "delta_rmse": close(delta, record["delta_rmse"]),
        "canonical_nominal": close(
            -delta * slope,
            record["canonical_nominal_pooled_points_delta"],
        ),
        "canonical_transport": close(
            -delta * slope - penalty,
            record["canonical_transport_adjusted_pooled_points_delta"],
        ),
        "fold_metrics": set(record["by_fold"]) == set(config["evaluation"]["folds"]),
        "month_metrics": len(record["by_month"]) == 6,
        "layer_metrics": set(record["by_layer"]) == {"2", "3", "4"},
        "bootstrap_ci": (
            record["bootstrap"]["ci90_low"] <= record["bootstrap"]["ci90_high"]
        ),
        "prefix_cutoffs": prefix_ok,
        "equal_environment_mass": group_mass_ok,
        "permutation_invariance": (
            result["permutation_invariance"]["maximum_abs_error"] <= 1e-6
        ),
        "comparison_v13_v18": comparison_ok,
        "action_bound": float(np.max(np.abs(candidate - reference))) <= 0.5 + 1e-12,
        "row_deletion_zero": result["training"]["row_deletion"] == 0,
        "official_access_zero": all(int(counters[name]) == 0 for name in official_names),
    }
    qa = {
        "schema_version": "p2.layer_month_risk_variance_rex.independent_qa.20260901.v19",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "recomputed": {
            "rows": len(candidate),
            "reference_rmse": reference_rmse,
            "candidate_rmse": candidate_rmse,
            "delta_rmse": delta,
            "canonical_nominal_points": -delta * slope,
            "canonical_transport_adjusted_points": -delta * slope - penalty,
            "abs_action_p99_C": float(np.quantile(np.abs(candidate - reference), 0.99)),
            "abs_action_max_C": float(np.max(np.abs(candidate - reference))),
        },
        "access": {
            "observations_rows_read": len(observations),
            "official_test_index_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "result": runner.v12.sha256_file(result_path),
            "prediction_npz": runner.v12.sha256_file(prediction_path),
            "config": runner.v12.sha256_file(runner.CONFIG),
            "runner": runner.v12.sha256_file(runner.RUNNER),
            "v13_runner": runner.v12.sha256_file(runner.v13.RUNNER),
            "v18_runner": runner.v12.sha256_file(runner.v18.RUNNER),
        },
    }
    runner.v12.atomic_json(runner.REPORT / "independent-qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
