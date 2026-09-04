"""Independent QA for P2 v45c stochastic DropConnect confirmation."""

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
import run_p2_all_linear_dropconnect_stochastic_confirmation_20260901_v45c as runner  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name].copy() for name in bundle.files}


def _recompute_comparison(
    result: dict[str, Any], config: dict[str, Any], current: dict[str, np.ndarray]
) -> dict[str, Any]:
    evidence = config["authorization_evidence"]
    source_result = json.loads(
        (ROOT / evidence["v45_result"]).read_text(encoding="utf-8")
    )
    source = _load_npz(ROOT / evidence["v45_prediction"])
    for name in ("time_ns", "layer", "fold", "reference"):
        if not np.array_equal(current[name], source[name]):
            raise RuntimeError(f"v45c independent commitment alignment failed: {name}")
    current_candidate = current["candidate"].astype(float)
    source_candidate = source["candidate"].astype(float)
    difference = current_candidate - source_candidate
    record = result["candidate"]
    source_record = source_result["candidate"]
    return {
        "use": "diagnostic_only_no_seed_trio_selection_or_ensemble",
        "v45_result_sha256": evidence["v45_result_sha256"],
        "v45_prediction_sha256": evidence["v45_prediction_sha256"],
        "v45_status": source_result["status"],
        "v45_delta_rmse_C": float(source_record["delta_rmse"]),
        "confirmation_delta_rmse_C": float(record["delta_rmse"]),
        "confirmation_minus_v45_delta_rmse_C": float(
            record["delta_rmse"] - source_record["delta_rmse"]
        ),
        "v45_candidate_rmse_C": float(source_record["candidate_rmse"]),
        "confirmation_candidate_rmse_C": float(record["candidate_rmse"]),
        "confirmation_minus_v45_candidate_rmse_C": float(
            record["candidate_rmse"] - source_record["candidate_rmse"]
        ),
        "candidate_prediction_difference": {
            "rows": int(len(difference)),
            "mean_C": float(np.mean(difference)),
            "rms_C": float(np.sqrt(np.mean(np.square(difference)))),
            "mean_abs_C": float(np.mean(np.abs(difference))),
            "p99_abs_C": float(np.quantile(np.abs(difference), 0.99)),
            "max_abs_C": float(np.max(np.abs(difference))),
            "pearson_correlation": float(
                np.corrcoef(current_candidate, source_candidate)[0, 1]
            ),
        },
        "fold_layer_delta_rmse_difference_C": {
            fold: {
                layer: float(
                    record["by_fold_layer"][fold][layer]["delta_rmse"]
                    - source_record["by_fold_layer"][fold][layer]["delta_rmse"]
                )
                for layer in ("2", "3", "4")
            }
            for fold in runner.v12.metric_engine.FOLD_ORDER
        },
        "v45_original_commitment_remains_representative": True,
    }


def main() -> None:
    result_path = runner.REPORT / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    prediction_path = runner.ARTIFACT / f"{runner.PREDICTION_NAME}.npz"
    prediction = _load_npz(prediction_path)
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
        raise RuntimeError("v45c independent truth alignment failed")
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
    masks_used = True
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
            stats = fit["dropconnect_statistics"]
            fit_receipts_ok &= bool(
                np.isfinite([fit["loss_first"], fit["loss_last"]]).all()
            )
            fit_receipts_ok &= fit["epochs"] == 60
            fit_receipts_ok &= fit["parameters"] == 4865
            fit_receipts_ok &= fit["parameter_tensors"] == 10
            fit_receipts_ok &= fit["buffers"] == 0
            fit_receipts_ok &= fit["drop_probability"] == 0.1
            fit_receipts_ok &= fit["keep_probability"] == 0.9
            fit_receipts_ok &= stats["module_count"] == 5
            fit_receipts_ok &= stats["mask_calls"] == fit["expected_mask_calls"]
            fit_receipts_ok &= stats["mask_calls"] == fit["optimizer_steps"] * 5
            fit_receipts_ok &= stats["total_weight_draws"] > 0
            fit_receipts_ok &= 0.89 <= stats["keep_share"] <= 0.91
            fit_receipts_ok &= fit["initial_rng_sha256"] != fit["final_rng_sha256"]
            fit_receipts_ok &= fit["evaluation_rng_unchanged"] is True
            fit_receipts_ok &= fit["dropout_module_count"] == 0
            fit_receipts_ok &= fit["prediction_consistency_loss"] == 0
            fit_receipts_ok &= fit["monte_carlo_inference"] == 0
            fit_receipts_ok &= fit["ensemble_models"] == 1
            fit_receipts_ok &= fit["row_deletion"] == 0
            fit_receipts_ok &= fit["loss_finite"] is True
            masks_used &= all(
                value["mask_calls"] == fit["optimizer_steps"]
                and value["last_mask_sha256"] is not None
                and value["bias_present"]
                for value in stats["modules"].values()
            )

    comparison = _recompute_comparison(result, config, prediction)
    semantic = runner.semantic_audit(config)
    contract = runner.v45._dropconnect_contract_receipt()
    isolation = runner.v45._isolation_receipt()
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
        "STOCHASTIC_CONFIRMATION_PASS_EXPOSED_BLOCKS_ONLY"
        if expected_pass
        else "STOCHASTIC_CONFIRMATION_NO_GO_ALL_LINEAR_DROPCONNECT"
    )
    expected_priority = (
        "V45_FAMILY_FIRST_INTERNAL_DEPLOYMENT_PREFLIGHT_PRIORITY"
        if expected_pass
        else "V23_REMAINS_ONLY_DEPLOYMENT_PRIORITY"
    )
    evidence = config["authorization_evidence"]
    checks = {
        "terminal_status": result["status"] == expected_status,
        "exposed_surface_claim": result["claim_level"]
        == "STOCHASTIC_REPLICATION_ON_EXPOSED_BLOCKS_ONLY",
        "exactly_nine_fits": result["fit_count"] == 9,
        "disjoint_confirmation_seeds": semantic["seed_sets_disjoint"]
        and seeds_ok,
        "same_science_as_v45": semantic[
            "same_architecture_objective_optimizer_blend_caps"
        ],
        "no_seed_trio_selection": not semantic[
            "candidate_selection_between_seed_trios"
        ],
        "v45_original_representative": semantic[
            "v45_original_commitment_remains_representative"
        ],
        "not_fresh_temporal_claim": semantic[
            "same_exposed_folds_not_fresh_temporal_surface"
        ],
        "dropconnect_contract": contract["module_count"] == 5
        and contract["parameters"] == 4865
        and contract["buffers"] == 0
        and contract["evaluation_initial_function_maximum_abs_error"] == 0.0
        and contract["deterministic_same_seed_training_maximum_abs_error"] == 0.0
        and contract["deterministic_same_seed_mask_hashes"]
        and contract["consecutive_step_masks_distinct"]
        and contract["evaluation_rng_unchanged"]
        and contract["dropout_module_count"] == 0,
        "permutation_invariance": isolation["permutation_maximum_abs_error"] <= 1e-6,
        "masked_future_isolation": isolation[
            "masked_or_future_token_maximum_abs_error"
        ]
        <= 1e-6,
        "repeat_determinism": isolation["repeat_maximum_abs_error"] == 0.0,
        "fit_receipts": fit_receipts_ok,
        "masks_used_all_fits": masks_used,
        "prediction_rows": len(candidate) == record["prediction_commitment"]["rows"],
        "prediction_hash": runner.v12.sha256_file(prediction_path)
        == record["prediction_commitment"]["sha256"],
        "config_hash": runner.v12.sha256_file(runner.CONFIG)
        == result["hashes"]["config"],
        "runner_hash": runner.v12.sha256_file(runner.RUNNER)
        == result["hashes"]["runner"],
        "v45_result_hash": runner.v12.sha256_file(ROOT / evidence["v45_result"])
        == result["hashes"]["v45_result"],
        "v45_prediction_hash": runner.v12.sha256_file(
            ROOT / evidence["v45_prediction"]
        )
        == result["hashes"]["v45_prediction"],
        "v45_config_hash": runner.v12.sha256_file(ROOT / evidence["v45_config"])
        == result["hashes"]["v45_config"],
        "v45_runner_hash": runner.v12.sha256_file(ROOT / evidence["v45_runner"])
        == result["hashes"]["v45_runner"],
        "scientific_contract_hash": runner._scientific_contract_sha256(config)
        == result["hashes"]["scientific_contract"],
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
        "action_geometry": v45qa.metric_tree_close(
            recomputed["action_geometry"], record["action_geometry"]
        ),
        "prospective_gate": v45qa.metric_tree_close(
            local_gate, record["prospective_fold_layer_gate"]
        ),
        "confirmation_vs_v45": v45qa.metric_tree_close(
            comparison, result["confirmation_vs_v45"]
        ),
        "confirmation_pass_definition": record[
            "safety_pass_with_v26a_amendment"
        ]
        == bool(record["legacy_safety_pass_without_v26a_amendment"] and local_gate["pass"]),
        "priority_vs_v23": result["priority_vs_v23"]["decision"]
        == expected_priority
        and result["priority_vs_v23"]["v23_position"]
        == "OFFICIAL_ANCHOR_REMAINS_UNCHANGED"
        and not result["priority_vs_v23"][
            "official_v23_feedback_used_for_settings_gate_or_slice"
        ],
        "prefix_cutoffs": prefix_ok,
        "equal_group_mass": group_mass_ok,
        "action_bound": float(np.max(np.abs(candidate - reference))) <= 0.5 + 1e-12,
        "row_deletion_zero": result["training"]["row_deletion"] == 0,
        "official_access_zero": all(int(counters[name]) == 0 for name in official_names),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    qa = {
        "schema_version": (
            "p2.all_linear_dropconnect_stochastic_confirmation.independent_qa."
            "20260901.v45c"
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
            "confirmation_vs_v45": comparison,
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
