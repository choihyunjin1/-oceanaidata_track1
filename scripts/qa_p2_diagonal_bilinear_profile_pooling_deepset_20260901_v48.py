"""Independent QA for sealed P2 v48 diagonal-bilinear profile pooling."""

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

import qa_p2_all_linear_dropconnect_deepset_20260901_v45 as v45qa  # noqa: E402
import run_p2_diagonal_bilinear_profile_pooling_deepset_20260901_v48 as runner  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)


def _recompute_comparison(
    record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    evidence = config["authorization_evidence"]
    comparisons: dict[str, Any] = {}
    for name in ("v45", "v45c", "v46", "v47"):
        source = json.loads(
            (ROOT / evidence[f"{name}_result"]).read_text(encoding="utf-8")
        )["candidate"]
        comparisons[name] = {
            "source_result_sha256": evidence[f"{name}_result_sha256"],
            "source_delta_rmse_C": float(source["delta_rmse"]),
            "v48_delta_rmse_C": float(record["delta_rmse"]),
            "v48_minus_source_delta_rmse_C": float(
                record["delta_rmse"] - source["delta_rmse"]
            ),
            "source_canonical_transport_adjusted_points": float(
                source["canonical_transport_adjusted_pooled_points_delta"]
            ),
            "v48_canonical_transport_adjusted_points": float(
                record["canonical_transport_adjusted_pooled_points_delta"]
            ),
        }
    return {
        "use": (
            "post_terminal_ledger_only_no_selection_router_retune_or_ensemble"
        ),
        "comparisons": comparisons,
        "v45_original_commitment_preserved": True,
        "v45c_confirmation_not_used_as_seed_selection": True,
        "v46_no_go_preserved": True,
        "v47_no_go_preserved": True,
    }


def _load_truth(
    time_ns: np.ndarray, layer: np.ndarray
) -> tuple[np.ndarray, int]:
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
        raise RuntimeError("v48 independent truth alignment failed")
    return design.truth[positions], len(observations)


def _fit_receipts(
    result: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    prefix_ok = True
    group_mass_ok = True
    receipts_ok = True
    seeds_ok = True
    training_rows_ok = True
    expected_rows = {
        "2024_sep_oct": 45935,
        "2025_jul_aug": 119667,
        "2025_nov_dec": 149384,
    }
    for fold_name, receipt in result["training"]["folds"].items():
        expected_cutoff = pd.Timestamp(config["training"]["fold_starts_kst"][fold_name])
        expected_cutoff -= pd.Timedelta(days=int(config["training"]["embargo_days"]))
        prefix_ok &= (
            pd.Timestamp(receipt["training_cutoff_exclusive_kst"])
            == expected_cutoff
        )
        training_rows_ok &= receipt["training_rows"] == expected_rows[fold_name]
        masses = [
            value["raw_weight_sum"]
            for value in receipt["weight_receipt"]["groups"].values()
        ]
        group_mass_ok &= bool(np.max(masses) - np.min(masses) <= 1e-12)
        seeds_ok &= [fit["seed"] for fit in receipt["fit_receipts"]] == config[
            "training"
        ]["seeds"]
        for fit in receipt["fit_receipts"]:
            receipts_ok &= bool(
                np.isfinite([fit["loss_first"], fit["loss_last"]]).all()
            )
            receipts_ok &= fit["epochs"] == 60
            receipts_ok &= fit["parameters"] == 5889
    return {
        "fixed_prefix_cutoffs": prefix_ok,
        "historical_training_rows": training_rows_ok,
        "equal_group_mass": group_mass_ok,
        "fixed_seeds": seeds_ok,
        "finite_fit_receipts": receipts_ok,
    }


def main() -> None:
    result_path = runner.REPORT / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    prediction_path = runner.ARTIFACT / f"{runner.PREDICTION_NAME}.npz"
    with np.load(prediction_path, allow_pickle=False) as prediction:
        time_ns = prediction["time_ns"].astype(np.int64)
        layer = prediction["layer"].astype(int)
        fold = prediction["fold"].astype(str)
        reference = prediction["reference"].astype(float)
        candidate = prediction["candidate"].astype(float)

    truth, observations_rows = _load_truth(time_ns, layer)
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
    comparison = _recompute_comparison(record, config)
    semantic = runner.semantic_audit(config)
    contract = runner._bilinear_contract_receipt()
    isolation = runner._isolation_receipt()
    fit_checks = _fit_receipts(result, config)

    expected_pass = bool(
        recomputed["strict_exploratory_pass"]
        and record["legacy_safety_pass_without_v26a_amendment"]
        and local_gate["pass"]
    )
    expected_status = (
        "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if expected_pass
        else "EXPLORATORY_NO_GO_DIAGONAL_BILINEAR_PROFILE_POOLING"
    )
    evidence = config["authorization_evidence"]
    counters = result["operation_counters"]
    zero_access_names = (
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
        "terminal_status": result["status"] == expected_status,
        "exactly_nine_fits": result["fit_count"] == 9,
        **fit_checks,
        "new_bilinear_axis": semantic["classification"]
        == "NEW_P2_MASKED_DIAGONAL_BILINEAR_PROFILE_POOLING",
        "negative_fingerprint_zero": semantic[
            "exact_execution_hits_before_preregistration"
        ]
        == 0
        and semantic["semantic_execution_hits_before_preregistration"] == 0,
        "semantic_distinctions": all(
            semantic[name]
            for name in (
                "v15_attention_distinguished",
                "v16_depth_graph_distinguished",
                "v20_coral_alignment_distinguished",
                "v43_film_distinguished",
                "v45_v45c_dropconnect_distinguished",
                "v46_layernorm_distinguished",
                "v47_cross_network_distinguished",
            )
        ),
        "no_performance_transfer_or_adaptation": not semantic[
            "primary_source_p2_performance_transfer"
        ]
        and not semantic["result_adaptive_search"]
        and not semantic["seed_trio_selection_or_ensemble"]
        and not semantic["router_or_retune"]
        and not semantic["official_v23_feedback_used_for_selection"],
        "bilinear_contract": contract["descriptor"]
        == "masked_mean(element_embedding_squared)"
        and contract["embedding_width"] == 32
        and contract["descriptor_width"] == 107
        and contract["new_head_columns"] == 32
        and contract["new_head_columns_initial_maximum_abs"] == 0.0
        and contract["parameters"] == 5889
        and contract["parameter_tensors"] == 10
        and contract["buffers"] == 0
        and contract["linear_count"] == 5
        and contract["unchanged_parameter_maximum_abs_error_vs_v13"] == 0.0
        and contract["initial_function_maximum_abs_error_vs_v13"] == 0.0
        and contract["descriptor_forward_maximum_abs_error"] == 0.0
        and contract["diagonal_second_moment_maximum_abs_error"] == 0.0
        and contract["all_missing_second_moment_maximum_abs"] == 0.0
        and contract["learned_bilinear_columns_change_function_maximum_abs"]
        > 0.0
        and contract["batch_composition_maximum_abs_error"] <= 1e-6
        and contract["gradients_finite"]
        and contract["new_head_column_gradient_finite_nonzero"]
        and contract["normalization_count"] == 0
        and contract["dropout_count"] == 0
        and contract["attention_count"] == 0,
        "permutation_invariance": isolation["permutation_maximum_abs_error"]
        <= 1e-6,
        "masked_future_isolation": isolation[
            "masked_or_future_token_maximum_abs_error"
        ]
        <= 1e-6,
        "repeat_determinism": isolation["repeat_maximum_abs_error"] == 0.0,
        "prediction_rows": len(candidate)
        == record["prediction_commitment"]["rows"]
        == 69850,
        "prediction_hash": runner.v12.sha256_file(prediction_path)
        == record["prediction_commitment"]["sha256"],
        "config_hash": runner.v12.sha256_file(runner.CONFIG)
        == result["hashes"]["config"],
        "runner_hash": runner.v12.sha256_file(runner.RUNNER)
        == result["hashes"]["runner"],
        "v13_runner_hash": runner.v12.sha256_file(runner._V13_RUNNER)
        == result["hashes"]["v13_runner"],
        "evidence_hashes": all(
            runner.v12.sha256_file(ROOT / evidence[name])
            == result["hashes"][name]
            for name in runner.EVIDENCE_NAMES
        ),
        "reference_rmse": v45qa.close(
            recomputed["reference_rmse"], record["reference_rmse"]
        ),
        "candidate_rmse": v45qa.close(
            recomputed["candidate_rmse"], record["candidate_rmse"]
        ),
        "delta_rmse": v45qa.close(delta, record["delta_rmse"]),
        "canonical_nominal": v45qa.close(
            nominal, record["canonical_nominal_pooled_points_delta"]
        ),
        "canonical_transport": v45qa.close(
            transport, record["canonical_transport_adjusted_pooled_points_delta"]
        ),
        "fold_metrics": v45qa.metric_tree_close(
            recomputed["by_fold"], record["by_fold"]
        ),
        "month_metrics": v45qa.metric_tree_close(
            recomputed["by_month"], record["by_month"]
        ),
        "layer_metrics": v45qa.metric_tree_close(
            recomputed["by_layer"], record["by_layer"]
        ),
        "fold_layer_metrics": v45qa.metric_tree_close(
            recomputed["by_fold_layer"], record["by_fold_layer"]
        ),
        "bootstrap": v45qa.metric_tree_close(
            recomputed["bootstrap"], record["bootstrap"]
        ),
        "official_like_bootstrap": v45qa.metric_tree_close(
            recomputed["official_like_bootstrap"],
            record["official_like_bootstrap"],
        ),
        "action_geometry": v45qa.metric_tree_close(
            recomputed["action_geometry"], record["action_geometry"]
        ),
        "prospective_gate": v45qa.metric_tree_close(
            local_gate, record["prospective_fold_layer_gate"]
        ),
        "comparison_after_terminal": v45qa.metric_tree_close(
            comparison, result["comparison_to_frozen_candidates"]
        ),
        "effective_safety_pass": record["safety_pass_with_v26a_amendment"]
        == bool(
            record["legacy_safety_pass_without_v26a_amendment"]
            and local_gate["pass"]
        ),
        "action_bound": float(np.max(np.abs(candidate - reference)))
        <= 0.5 + 1e-12,
        "row_deletion_zero": result["training"]["row_deletion"] == 0,
        "observations_rows": observations_rows
        == counters["observations_rows_read"]
        == 789408,
        "official_access_zero": all(
            int(counters[name]) == 0 for name in zero_access_names
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    qa = {
        "schema_version": (
            "p2.diagonal_bilinear_profile_pooling_deepset.independent_qa.20260901.v48"
        ),
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
            "comparison_to_frozen_candidates": comparison,
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
