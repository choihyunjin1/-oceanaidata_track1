"""Independent QA for the P2 v17 local masked-public auxiliary candidate."""

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

import run_p2_local_prefix_masked_public_auxiliary_20260901_v17 as runner  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    """Return whether two deterministic scalar receipts agree."""
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
        ],
    )
    positions = design_index.get_indexer(
        pd.MultiIndex.from_arrays([time_ns, layer]),
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
    architecture = result["architecture_contract"]
    semantic = result["semantic_audit"]
    counters = result["operation_counters"]
    prefix_ok = True
    group_mass_ok = True
    fit_receipts_ok = True
    for fold, receipt in result["training"]["folds"].items():
        expected = pd.Timestamp(config["training"]["fold_starts_kst"][fold])
        expected -= pd.Timedelta(days=int(config["training"]["embargo_days"]))
        prefix_ok &= pd.Timestamp(receipt["training_cutoff_exclusive_kst"]) == expected
        masses = [
            value["raw_weight_sum"]
            for value in receipt["weight_receipt"]["groups"].values()
        ]
        group_mass_ok &= bool(np.max(masses) - np.min(masses) <= 1e-12)
        for fit in receipt["fit_receipts"]:
            fit_receipts_ok &= fit["auxiliary_weight"] == 0.25
            fit_receipts_ok &= fit["mask_cycle_counts"] == {
                str(index): 12 for index in range(5)
            }

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
        "new_local_masked_public_axis": (
            semantic["classification"]
            == "NEW_LOCAL_PREFIX_MASKED_PUBLIC_AUXILIARY_REPRESENTATION"
        ),
        "external_depth_query_unexecuted": (
            not any(semantic["external_depth_query_execution"].values())
            and not semantic["external_depth_query_artifact_exists"]
            and not semantic["external_depth_query_report_exists"]
        ),
        "no_target_masked_tcn_csdi_overlap": (
            not semantic["target_masked_tcn_or_csdi_overlap"]
            and not semantic["uses_temporal_masking"]
            and not semantic["reconstructs_target_layer_values"]
        ),
        "no_external_data": (
            not semantic["uses_external_data"]
            and not architecture["external_data_used"]
            and counters["external_rows_read"] == 0
        ),
        "fixed_auxiliary_weight": result["training"]["auxiliary_weight"] == 0.25,
        "fixed_mask_cycle_receipts": fit_receipts_ok,
        "one_masked_slot": (
            architecture["masked_slot_tokens_zero"]
            and architecture["masked_slot_mask_zero"]
        ),
        "ordered_public_slots": architecture["ordered_public_slots"],
        "future_row_isolation": (
            architecture["future_batch_perturbation_maximum_abs_error_on_prior_rows"]
            == 0
        ),
        "target_values_absent": (
            not architecture["current_target_temp_or_psal_feature_present"]
        ),
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
        "module_hash": (
            runner.v12.sha256_file(runner.MODEL_MODULE)
            == result["hashes"]["model_module"]
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
        "balanced_group_mass": group_mass_ok,
        "action_bound": float(np.max(np.abs(candidate - reference))) <= 0.5 + 1e-12,
        "row_deletion_zero": result["training"]["row_deletion"] == 0,
        "official_access_zero": all(int(counters[name]) == 0 for name in official_names),
    }
    qa = {
        "schema_version": (
            "p2.local_prefix_masked_public_auxiliary.independent_qa.20260901.v17"
        ),
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
            "external_rows_read": 0,
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
            "model_module": runner.v12.sha256_file(runner.MODEL_MODULE),
        },
    }
    runner.v12.atomic_json(runner.REPORT / "independent-qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
