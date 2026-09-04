"""Independent QA for sealed P2 v43 target-layer FiLM run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_target_layer_film_conditioned_deepset_20260901_v43 as runner  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def metric_tree_close(left: Any, right: Any) -> bool:
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(
            metric_tree_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            metric_tree_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return close(left, right)
    return left == right


def main() -> None:
    result_path = runner.REPORT / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    prediction_path = runner.ARTIFACT / f"{runner.PREDICTION_NAME}.npz"
    prediction = np.load(prediction_path, allow_pickle=False)
    time_ns = prediction["time_ns"].astype(np.int64)
    layer = prediction["layer"].astype(int)
    fold = prediction["fold"].astype(str)
    reference = prediction["reference"].astype(float)
    candidate = prediction["candidate"].astype(float)

    data_dir = os.environ.get("P2_DATA_DIR")
    if not data_dir:
        raise RuntimeError("P2_DATA_DIR is required")
    observations = pd.read_csv(
        Path(data_dir).resolve() / "observations.csv",
        dtype={"station": "string", "time": "string"},
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    design = build_normalized_curvature_design(
        build_training_features(observations).frame
    )
    design_index = pd.MultiIndex.from_arrays(
        [
            runner.v12.metric_engine.canonical_time_ns(design.keys["time"]),
            design.keys["layer"],
        ]
    )
    positions = design_index.get_indexer(pd.MultiIndex.from_arrays([time_ns, layer]))
    if np.any(positions < 0):
        raise RuntimeError("independent truth alignment failed")
    truth = design.truth[positions]
    blind = pd.DataFrame(
        {
            "time": pd.to_datetime(time_ns, unit="ns", utc=True),
            "layer": layer,
            "fold": fold,
        }
    )
    blind["kst_date"] = blind["time"].dt.tz_convert("Asia/Seoul").dt.date
    spec = runner.v12.metric_engine.CandidateSpec(
        name=runner.PREDICTION_NAME,
        objective=config["training"]["objective"],
        conditional=False,
    )
    recomputed = runner.v12.metric_engine.evaluate_candidate(
        spec, blind, truth, reference, candidate, config
    )
    recomputed["by_month"] = runner.v12.by_month_metrics(
        blind, truth, reference, candidate
    )
    recomputed["action_geometry"] = runner.v12.action_geometry(
        truth, reference, candidate
    )
    delta = float(recomputed["delta_rmse"])
    nominal = -delta * float(config["evaluation"]["points_per_rmse_C"])
    transport = nominal - float(config["evaluation"]["transport_penalty_points"])
    record = result["candidate"]
    local_gate = runner.v37.prospective_fold_layer_gate(recomputed, config)

    prefix_ok = True
    group_mass_ok = True
    fit_receipts_ok = True
    film_updated = True
    for fold_name, receipt in result["training"]["folds"].items():
        expected = pd.Timestamp(config["training"]["fold_starts_kst"][fold_name])
        expected -= pd.Timedelta(days=int(config["training"]["embargo_days"]))
        prefix_ok &= pd.Timestamp(receipt["training_cutoff_exclusive_kst"]) == expected
        masses = [
            value["raw_weight_sum"]
            for value in receipt["weight_receipt"]["groups"].values()
        ]
        group_mass_ok &= bool(np.max(masses) - np.min(masses) <= 1e-12)
        for fit in receipt["fit_receipts"]:
            fit_receipts_ok &= bool(
                np.isfinite([fit["loss_first"], fit["loss_last"]]).all()
            )
            fit_receipts_ok &= fit["epochs"] == 60
            fit_receipts_ok &= fit["parameters"] == 5057
            fit_receipts_ok &= fit["film_generator_count"] == 1
            fit_receipts_ok &= fit["film_added_parameters"] == 192
            fit_receipts_ok &= fit["gamma_beta_broadcast_to_all_tokens"] is True
            fit_receipts_ok &= fit["normalization_modules"] == 0
            fit_receipts_ok &= fit["attention_modules"] == 0
            fit_receipts_ok &= fit["extra_loss_terms"] == 0
            fit_receipts_ok &= fit["loss"] == config["training"]["objective"]
            fit_receipts_ok &= fit["row_deletion"] == 0
            fit_receipts_ok &= fit["loss_finite"] is True
            film_updated &= fit["initial_film_state_sha256"] != fit[
                "final_film_state_sha256"
            ]

    semantic = runner.semantic_audit(config)
    contract = runner._film_contract_receipt()
    isolation = runner._isolation_receipt()
    amendment_path = ROOT / config["authorization_evidence"][
        "prospective_gate_amendment"
    ]
    counters = result["operation_counters"]
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
        "new_p2_target_layer_film_axis": semantic["classification"]
        == "NEW_P2_TARGET_LAYER_CONDITIONED_ELEMENT_FILM",
        "repository_execution_hits_zero": semantic["repository_execution_hits"] == 0,
        "semantic_distinctions": all(
            semantic[name]
            for name in (
                "v15_attention_distinguished",
                "v16_depth_graph_distinguished",
                "v31_domain_adversary_distinguished",
                "v40_v42_consistency_distinguished",
                "v41_weight_norm_distinguished",
            )
        ),
        "official_feedback_selection_zero": not semantic[
            "official_v23_feedback_used_for_selection"
        ],
        "film_generator_contract": contract["generator_count"] == 1
        and contract["generator_shape"] == [64, 3]
        and not contract["generator_bias"]
        and contract["added_parameters"] == 192,
        "identity_initial_function": contract[
            "gamma_initialization_maximum_abs_error"
        ]
        == 0.0
        and contract["beta_initialization_maximum_abs_error"] == 0.0
        and contract["initial_function_maximum_abs_error"] <= 1e-6,
        "no_norm_attention": contract["normalization_module_count"] == 0
        and contract["attention_module_count"] == 0,
        "permutation_invariance": isolation["permutation_maximum_abs_error"] <= 1e-6,
        "masked_future_isolation": isolation[
            "masked_or_future_token_maximum_abs_error"
        ]
        <= 1e-6,
        "repeat_determinism": isolation["repeat_maximum_abs_error"] == 0.0,
        "fit_receipts": fit_receipts_ok,
        "film_updated_by_training": film_updated,
        "prediction_rows": len(candidate) == record["prediction_commitment"]["rows"],
        "prediction_hash": runner.v12.sha256_file(prediction_path)
        == record["prediction_commitment"]["sha256"],
        "config_hash": runner.v12.sha256_file(runner.CONFIG)
        == result["hashes"]["config"],
        "runner_hash": runner.v12.sha256_file(runner.RUNNER)
        == result["hashes"]["runner"],
        "v13_runner_hash": runner.v12.sha256_file(runner._V13_RUNNER)
        == result["hashes"]["v13_runner"],
        "gate_amendment_hash": runner.v12.sha256_file(amendment_path)
        == result["hashes"]["prospective_gate_amendment"],
        "reference_rmse": close(recomputed["reference_rmse"], record["reference_rmse"]),
        "candidate_rmse": close(recomputed["candidate_rmse"], record["candidate_rmse"]),
        "delta_rmse": close(delta, record["delta_rmse"]),
        "canonical_nominal": close(
            nominal, record["canonical_nominal_pooled_points_delta"]
        ),
        "canonical_transport": close(
            transport, record["canonical_transport_adjusted_pooled_points_delta"]
        ),
        "fold_metrics": metric_tree_close(recomputed["by_fold"], record["by_fold"]),
        "month_metrics": metric_tree_close(recomputed["by_month"], record["by_month"]),
        "layer_metrics": metric_tree_close(recomputed["by_layer"], record["by_layer"]),
        "fold_layer_metrics": metric_tree_close(
            recomputed["by_fold_layer"], record["by_fold_layer"]
        ),
        "bootstrap": metric_tree_close(recomputed["bootstrap"], record["bootstrap"]),
        "official_like_bootstrap": metric_tree_close(
            recomputed["official_like_bootstrap"], record["official_like_bootstrap"]
        ),
        "action_geometry": metric_tree_close(
            recomputed["action_geometry"], record["action_geometry"]
        ),
        "prospective_gate": metric_tree_close(
            local_gate, record["prospective_fold_layer_gate"]
        ),
        "effective_safety_pass": record["safety_pass_with_v26a_amendment"]
        == bool(
            record["legacy_safety_pass_without_v26a_amendment"]
            and local_gate["pass"]
        ),
        "prefix_cutoffs": prefix_ok,
        "equal_group_mass": group_mass_ok,
        "action_bound": float(np.max(np.abs(candidate - reference))) <= 0.5 + 1e-12,
        "row_deletion_zero": result["training"]["row_deletion"] == 0,
        "official_access_zero": all(int(counters[name]) == 0 for name in official_names),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    qa = {
        "schema_version": "p2.target_layer_film_conditioned_deepset.independent_qa.20260901.v43",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "recomputed": {
            "rows": len(candidate),
            "reference_rmse": recomputed["reference_rmse"],
            "candidate_rmse": recomputed["candidate_rmse"],
            "delta_rmse": delta,
            "canonical_nominal_points": nominal,
            "canonical_transport_adjusted_points": transport,
            "by_fold": recomputed["by_fold"],
            "by_month": recomputed["by_month"],
            "by_layer": recomputed["by_layer"],
            "by_fold_layer": recomputed["by_fold_layer"],
            "bootstrap": recomputed["bootstrap"],
            "action_geometry": recomputed["action_geometry"],
            "prospective_fold_layer_gate": local_gate,
        },
        "hashes": {
            "result": runner.v12.sha256_file(result_path),
            "prediction": runner.v12.sha256_file(prediction_path),
            "config": runner.v12.sha256_file(runner.CONFIG),
            "runner": runner.v12.sha256_file(runner.RUNNER),
        },
        "official_rows": 0,
        "hidden_rows": 0,
        "csv_materializations": 0,
        "uploads": 0,
    }
    runner.v12.atomic_json(runner.REPORT / "independent-qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
