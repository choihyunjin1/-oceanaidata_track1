"""Independent QA for sealed P2 v46 LayerNorm DeepSets."""

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
import run_p2_hidden_activation_layer_normalized_deepset_20260901_v46 as runner  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)


def _recompute_comparison(
    record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    evidence = config["authorization_evidence"]
    comparisons: dict[str, Any] = {}
    for name in ("v45", "v45c"):
        source = json.loads(
            (ROOT / evidence[f"{name}_result"]).read_text(encoding="utf-8")
        )["candidate"]
        comparisons[name] = {
            "source_result_sha256": evidence[f"{name}_result_sha256"],
            "source_delta_rmse_C": float(source["delta_rmse"]),
            "v46_delta_rmse_C": float(record["delta_rmse"]),
            "v46_minus_source_delta_rmse_C": float(
                record["delta_rmse"] - source["delta_rmse"]
            ),
            "source_canonical_transport_adjusted_points": float(
                source["canonical_transport_adjusted_pooled_points_delta"]
            ),
            "v46_canonical_transport_adjusted_points": float(
                record["canonical_transport_adjusted_pooled_points_delta"]
            ),
        }
    return {
        "use": "post_terminal_ledger_only_no_selection_router_or_ensemble",
        "comparisons": comparisons,
        "v45_original_commitment_preserved": True,
        "v45c_confirmation_not_used_as_seed_selection": True,
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
        raise RuntimeError("v46 independent truth alignment failed")
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
    seeds_ok = True
    for fold_name, receipt in result["training"]["folds"].items():
        expected_cutoff = pd.Timestamp(config["training"]["fold_starts_kst"][fold_name])
        expected_cutoff -= pd.Timedelta(days=int(config["training"]["embargo_days"]))
        prefix_ok &= (
            pd.Timestamp(receipt["training_cutoff_exclusive_kst"]) == expected_cutoff
        )
        masses = [
            value["raw_weight_sum"]
            for value in receipt["weight_receipt"]["groups"].values()
        ]
        group_mass_ok &= bool(np.max(masses) - np.min(masses) <= 1e-12)
        seeds_ok &= [fit["seed"] for fit in receipt["fit_receipts"]] == config[
            "training"
        ]["seeds"]
        for fit in receipt["fit_receipts"]:
            fit_receipts_ok &= bool(
                np.isfinite([fit["loss_first"], fit["loss_last"]]).all()
            )
            fit_receipts_ok &= fit["epochs"] == 60
            fit_receipts_ok &= fit["parameters"] == 5121
            fit_receipts_ok &= fit["parameter_tensors"] == 18
            fit_receipts_ok &= fit["buffers"] == 0
            fit_receipts_ok &= fit["layernorm_count"] == 4
            fit_receipts_ok &= fit["layernorm_names"] == [
                "element.1",
                "element.4",
                "head.1",
                "head.4",
            ]
            fit_receipts_ok &= fit["layernorm_eps"] == [1e-5, 1e-5, 1e-5, 1e-5]
            fit_receipts_ok &= all(fit["layernorm_affine"])
            fit_receipts_ok &= fit["batchnorm_count"] == 0
            fit_receipts_ok &= fit["dropout_count"] == 0
            fit_receipts_ok &= fit["loss_finite"] is True
            fit_receipts_ok &= fit["row_deletion"] == 0

    comparison = _recompute_comparison(record, config)
    semantic = runner.semantic_audit(config)
    contract = runner._layernorm_contract_receipt()
    isolation = runner._isolation_receipt()
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
    expected_pass = bool(
        recomputed["strict_exploratory_pass"]
        and record["legacy_safety_pass_without_v26a_amendment"]
        and local_gate["pass"]
    )
    expected_status = (
        "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if expected_pass
        else "EXPLORATORY_NO_GO_HIDDEN_ACTIVATION_LAYERNORM"
    )
    evidence = config["authorization_evidence"]
    checks = {
        "terminal_status": result["status"] == expected_status,
        "exactly_nine_fits": result["fit_count"] == 9,
        "fixed_seeds": seeds_ok,
        "new_layernorm_axis": semantic["classification"]
        == "NEW_P2_HIDDEN_ACTIVATION_LAYER_NORMALIZATION",
        "negative_fingerprint_zero": semantic[
            "exact_execution_hits_before_preregistration"
        ]
        == 0,
        "semantic_distinctions": all(
            semantic[name]
            for name in (
                "v20_coral_distinguished",
                "v27_spectral_norm_distinguished",
                "v37_cmd_distinguished",
                "v40_dropout_consistency_distinguished",
                "v41_weight_norm_distinguished",
                "v45_v45c_dropconnect_distinguished",
            )
        ),
        "primary_source_no_performance_transfer": not semantic[
            "primary_source_p2_performance_transfer"
        ],
        "no_selection_or_ensemble": not semantic["seed_trio_selection_or_ensemble"]
        and not semantic["official_v23_feedback_used_for_selection"],
        "layernorm_contract": contract["layernorm_count"] == 4
        and contract["layernorm_names"]
        == ["element.1", "element.4", "head.1", "head.4"]
        and contract["normalized_shapes"] == [[32], [32], [32], [32]]
        and contract["eps"] == [1e-5, 1e-5, 1e-5, 1e-5]
        and all(contract["elementwise_affine"])
        and max(contract["initial_weight_errors"]) == 0.0
        and max(contract["initial_bias_errors"]) == 0.0
        and contract["parameters"] == 5121
        and contract["parameter_tensors"] == 18
        and contract["buffers"] == 0
        and contract["linear_parameter_maximum_abs_error_vs_v13"] == 0.0
        and contract["initial_function_maximum_abs_difference_vs_v13"] > 0.0
        and contract["batch_composition_maximum_abs_error"] <= 1e-6
        and contract["gradients_finite"]
        and contract["batchnorm_count"] == 0
        and contract["dropout_count"] == 0,
        "permutation_invariance": isolation["permutation_maximum_abs_error"] <= 1e-6,
        "masked_future_isolation": isolation[
            "masked_or_future_token_maximum_abs_error"
        ]
        <= 1e-6,
        "repeat_determinism": isolation["repeat_maximum_abs_error"] == 0.0,
        "fit_receipts": fit_receipts_ok,
        "prediction_rows": len(candidate) == record["prediction_commitment"]["rows"],
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
            for name in (
                "negative_fingerprint",
                "design_fingerprint",
                "execution_decision",
                "v13_result",
                "v45_result",
                "v45c_result",
                "v45c_independent_qa",
                "prospective_gate_amendment",
            )
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
            recomputed["official_like_bootstrap"], record["official_like_bootstrap"]
        ),
        "action_geometry": v45qa.metric_tree_close(
            recomputed["action_geometry"], record["action_geometry"]
        ),
        "prospective_gate": v45qa.metric_tree_close(
            local_gate, record["prospective_fold_layer_gate"]
        ),
        "comparison_after_terminal": v45qa.metric_tree_close(
            comparison, result["comparison_to_frozen_dropconnect"]
        ),
        "effective_safety_pass": record["safety_pass_with_v26a_amendment"]
        == bool(record["legacy_safety_pass_without_v26a_amendment"] and local_gate["pass"]),
        "prefix_cutoffs": prefix_ok,
        "equal_group_mass": group_mass_ok,
        "action_bound": float(np.max(np.abs(candidate - reference))) <= 0.5 + 1e-12,
        "row_deletion_zero": result["training"]["row_deletion"] == 0,
        "official_access_zero": all(int(counters[name]) == 0 for name in official_names),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    qa = {
        "schema_version": (
            "p2.hidden_activation_layer_normalized_deepset.independent_qa."
            "20260901.v46"
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
            "comparison_to_frozen_dropconnect": comparison,
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
