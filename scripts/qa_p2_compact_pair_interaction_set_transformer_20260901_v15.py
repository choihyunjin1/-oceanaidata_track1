"""Independent metric, prefix, architecture, hash, and access QA for P2 v15."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for item in (SCRIPTS, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_compact_pair_interaction_set_transformer_20260901_v15 as runner  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import build_normalized_curvature_design  # noqa: E402


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
    observations_dir = os.environ.get("P2_DATA_DIR")
    if not observations_dir:
        raise RuntimeError("P2_DATA_DIR is required")
    observations_path = Path(observations_dir).resolve() / "observations.csv"
    observations = pd.read_csv(observations_path, dtype={"station": "string", "time": "string"})
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    design = build_normalized_curvature_design(build_training_features(observations).frame)
    design_index = pd.MultiIndex.from_arrays(
        [runner.v12.metric_engine.canonical_time_ns(design.keys["time"]), design.keys["layer"]]
    )
    positions = design_index.get_indexer(pd.MultiIndex.from_arrays([time_ns, layer]))
    if np.any(positions < 0):
        raise RuntimeError("independent truth alignment failed")
    truth = design.truth[positions]
    reference_rmse = runner.v12.metric_engine.rmse(truth, reference)
    candidate_rmse = runner.v12.metric_engine.rmse(truth, candidate)
    delta = candidate_rmse - reference_rmse
    slope = float(config["evaluation"]["points_per_rmse_C"])
    penalty = float(config["evaluation"]["transport_penalty_points"])
    record = result["candidate"]
    counters = result["operation_counters"]
    prefix_ok = True
    group_mass_ok = True
    for fold, receipt in result["training"]["folds"].items():
        expected = pd.Timestamp(config["training"]["fold_starts_kst"][fold]) - pd.Timedelta(
            days=int(config["training"]["embargo_days"])
        )
        prefix_ok &= pd.Timestamp(receipt["training_cutoff_exclusive_kst"]) == expected
        masses = [value["raw_weight_sum"] for value in receipt["weight_receipt"]["groups"].values()]
        group_mass_ok &= bool(np.max(masses) - np.min(masses) <= 1e-12)
    architecture = result["architecture_contract"]
    checks = {
        "terminal": result["status"].startswith("EXPLORATORY_"),
        "exactly_nine_fits": int(result["fit_count"]) == 9,
        "new_pair_interaction_representation": result["semantic_audit"]["classification"]
        == "NEW_SAME_TIME_PAIR_INTERACTION_SET_REPRESENTATION",
        "one_attention_block": architecture["attention_blocks"] == 1,
        "two_attention_heads": architecture["attention_heads"] == 2,
        "no_positional_or_temporal_attention": not architecture["positional_encoding"]
        and not architecture["temporal_attention"]
        and not architecture["time_axis_present"],
        "permutation_equivariance": architecture[
            "encoder_equivariance_maximum_abs_error"
        ]
        <= 1e-6,
        "permutation_invariance": architecture["prediction_invariance_maximum_abs_error"]
        <= 1e-6,
        "future_row_isolation": architecture[
            "future_batch_perturbation_maximum_abs_error_on_prior_rows"
        ]
        == 0.0,
        "target_values_absent_from_features": not architecture[
            "current_target_temp_or_psal_feature_present"
        ],
        "prediction_rows_match": len(candidate) == int(record["prediction_commitment"]["rows"]),
        "prediction_hash_match": runner.v12.sha256_file(prediction_path)
        == record["prediction_commitment"]["sha256"],
        "config_hash_match": runner.v12.sha256_file(runner.CONFIG) == result["hashes"]["config"],
        "runner_hash_match": runner.v12.sha256_file(runner.RUNNER) == result["hashes"]["runner"],
        "model_module_hash_match": runner.v12.sha256_file(runner.MODEL_MODULE)
        == result["hashes"]["model_module"],
        "reference_rmse_recomputed": close(reference_rmse, record["reference_rmse"]),
        "candidate_rmse_recomputed": close(candidate_rmse, record["candidate_rmse"]),
        "delta_rmse_recomputed": close(delta, record["delta_rmse"]),
        "canonical_nominal_points_recomputed": close(
            -delta * slope, record["canonical_nominal_pooled_points_delta"]
        ),
        "canonical_transport_points_recomputed": close(
            -delta * slope - penalty,
            record["canonical_transport_adjusted_pooled_points_delta"],
        ),
        "all_fold_metrics_present": set(record["by_fold"]) == set(config["evaluation"]["folds"]),
        "all_month_metrics_present": len(record["by_month"]) == 6,
        "all_layer_metrics_present": set(record["by_layer"]) == {"2", "3", "4"},
        "bootstrap_ci_present": float(record["bootstrap"]["ci90_low"])
        <= float(record["bootstrap"]["ci90_high"]),
        "fold_prefix_cutoffs_exact": prefix_ok,
        "domain_group_masses_equal": group_mass_ok,
        "action_bound_half_degree": float(np.max(np.abs(candidate - reference))) <= 0.5 + 1e-12,
        "row_deletion_zero": result["training"]["row_deletion"] == 0,
        "official_access_zero": all(
            int(counters[name]) == 0
            for name in (
                "official_test_index_rows_read",
                "sample_rows_read",
                "baseline_file_rows_read",
                "score_file_rows_read",
                "query_support_rows_read",
                "hidden_truth_rows_read",
                "submission_csv_created",
                "uploads",
            )
        ),
    }
    qa = {
        "schema_version": "p2.compact_pair_interaction_set_transformer.independent_qa.20260901.v15",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "recomputed": {
            "rows": int(len(candidate)),
            "reference_rmse": reference_rmse,
            "candidate_rmse": candidate_rmse,
            "delta_rmse": delta,
            "canonical_nominal_points": -delta * slope,
            "canonical_transport_adjusted_points": -delta * slope - penalty,
            "abs_action_p99_C": float(np.quantile(np.abs(candidate - reference), 0.99)),
            "abs_action_max_C": float(np.max(np.abs(candidate - reference))),
        },
        "access": {
            "observations_rows_read": int(len(observations)),
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
