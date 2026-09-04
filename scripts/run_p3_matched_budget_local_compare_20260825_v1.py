from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p3_wave.matched_budget_local_compare import (  # noqa: E402
    KEY_COLUMNS,
    apply_fixed_long_lead_shrink,
    compare_aligned_surface,
    complete_case_bootstrap_delta,
    exclusive_write_json,
    metric_summary,
    read_json,
    residual_correlation,
    sha256_file,
    slice_delta,
    validate_surface,
)


def _kst_now() -> str:
    return datetime.now().astimezone().isoformat()


def _resolve(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents and path != ROOT.resolve():
        raise ValueError(f"Path escapes workspace: {relative}")
    return path


def _verify_hash(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"Pinned input hash differs for {path}: {observed} != {expected}")


def _model_fit_receipts(metrics: dict[str, Any]) -> int:
    return sum(
        1
        for receipt in metrics.get("training_receipts", [])
        if isinstance(receipt, dict)
        and "fold" in receipt
        and ("seed" in receipt or "seed_replicate" in receipt)
    )


def _zero_external_access(metrics: dict[str, Any]) -> bool:
    counters = metrics.get("access_counters", {})
    return all(
        not isinstance(value, (int, float)) or float(value) == 0.0
        for value in counters.values()
    )


def execute(config_path: Path) -> dict[str, Any]:
    started = perf_counter()
    config = read_json(config_path)
    output_dir = _resolve(config["output_dir"])
    seal_path = output_dir / "preexecution_seal.json"
    if not seal_path.is_file():
        raise FileNotFoundError("Append-only preexecution seal must exist before scoring")
    seal = read_json(seal_path)
    if seal.get("status") != "SEALED_BEFORE_AGGREGATE_RESULTS":
        raise ValueError("Preexecution seal status differs")
    if seal.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Preexecution seal does not pin the executed config")

    protected_outputs = [
        output_dir / "metrics.json",
        output_dir / "execution_receipt.json",
        output_dir / "independent_qa.json",
        output_dir / "technical_report_ko.md",
        output_dir / "manifest.json",
        output_dir / "manifest.sha256",
    ]
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError("Matched-budget outputs are append-only and already exist")

    common = config["common_protocol"]
    _verify_hash(_resolve(common["path"]), common["sha256"])
    for receipt in config["input_files"].values():
        _verify_hash(_resolve(receipt["path"]), receipt["sha256"])
    for family in config["structural_families"]:
        _verify_hash(_resolve(family["oof_path"]), family["oof_sha256"])
        _verify_hash(_resolve(family["metrics_path"]), family["metrics_sha256"])

    surface = config["surface"]
    canonical_path = _resolve(config["input_files"]["canonical_oof"]["path"])
    canonical = pd.read_parquet(canonical_path)
    surface_audit = validate_surface(
        canonical,
        expected_cases=surface["expected_cases"],
        expected_rows=surface["expected_rows"],
        expected_leads=surface["leads_h"],
    )
    if surface_audit["folds"] != sorted(surface["fixed_windows"]):
        raise ValueError("Canonical OOF fold set differs from the sealed fixed windows")
    if not np.array_equal(
        canonical["current_hs"].to_numpy(dtype=np.float64),
        canonical["persistence"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("Canonical persistence differs from current_hs")

    active_leads = config["coefficient_family"]["active_leads_h"]
    canonical["incumbent_router_unshrunk"] = canonical["routed_prediction"].to_numpy(
        dtype=np.float64
    )
    for setting in config["coefficient_family"]["settings"]:
        canonical[setting["id"]] = apply_fixed_long_lead_shrink(
            canonical["routed_prediction"],
            canonical["persistence"],
            canonical["lead_h"],
            weight=float(setting["persistence_weight"]),
            active_leads=active_leads,
        )
    current_difference = float(
        np.max(
            np.abs(
                canonical["current_shrink_20pct"].to_numpy(dtype=np.float64)
                - canonical["final_prediction"].to_numpy(dtype=np.float64)
            )
        )
    )
    if current_difference > 1e-12:
        raise ValueError("Reconstructed 20 percent shrink differs from canonical final OOF")

    round_a = pd.read_parquet(
        _resolve(config["input_files"]["round_a_oof_reference"]["path"])
    )
    round_a_alignment = compare_aligned_surface(
        canonical,
        round_a,
        float_columns=("target_hs", "current_hs", "persistence"),
        tolerance=0.0,
    )
    canonical_sorted = canonical.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    round_a_sorted = round_a.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    round_a_max_abs = float(
        np.max(
            np.abs(
                canonical_sorted["round_a_shrink_25pct"].to_numpy(dtype=np.float64)
                - round_a_sorted["candidate_prediction"].to_numpy(dtype=np.float64)
            )
        )
    )
    if round_a_max_abs > 1e-12:
        raise ValueError("Reconstructed Round A OOF differs from sealed Round A OOF")

    canonical_ids = [
        "incumbent_router_unshrunk",
        "current_shrink_20pct",
        "round_b_shrink_22p5pct",
        "round_a_shrink_25pct",
    ]
    canonical_metrics = {
        prediction: metric_summary(canonical, prediction) for prediction in canonical_ids
    }
    bootstrap = config["evaluation"]
    coefficient_comparisons: dict[str, Any] = {}
    for candidate_id in (
        "incumbent_router_unshrunk",
        "round_b_shrink_22p5pct",
        "round_a_shrink_25pct",
    ):
        coefficient_comparisons[candidate_id] = {
            "reference": "current_shrink_20pct",
            "delta_rmse_m": slice_delta(
                canonical_metrics[candidate_id], canonical_metrics["current_shrink_20pct"]
            ),
            "complete_case_bootstrap": complete_case_bootstrap_delta(
                canonical,
                candidate=candidate_id,
                reference="current_shrink_20pct",
                replicates=bootstrap["bootstrap_replicates"],
                seed=bootstrap["bootstrap_seed"],
            ),
            "residual_correlation": residual_correlation(
                canonical, candidate_id, "current_shrink_20pct"
            ),
        }

    weights = {
        setting["id"]: float(setting["persistence_weight"])
        for setting in config["coefficient_family"]["settings"]
    }
    w20 = canonical_metrics["current_shrink_20pct"]["rmse_m"]
    w225 = canonical_metrics["round_b_shrink_22p5pct"]["rmse_m"]
    w25 = canonical_metrics["round_a_shrink_25pct"]["rmse_m"]
    coefficient_sensitivity = {
        "weights": weights,
        "rmse_m": {
            "current_shrink_20pct": w20,
            "round_b_shrink_22p5pct": w225,
            "round_a_shrink_25pct": w25,
        },
        "delta_vs_current_m": {
            "current_shrink_20pct": 0.0,
            "round_b_shrink_22p5pct": float(w225 - w20),
            "round_a_shrink_25pct": float(w25 - w20),
        },
        "local_slope_rmse_m_per_unit_weight": {
            "20_to_22p5": float((w225 - w20) / 0.025),
            "22p5_to_25": float((w25 - w225) / 0.025),
        },
        "equal_step_second_difference_m": float(w25 - 2.0 * w225 + w20),
        "grid_extension_or_adaptive_optimum_search_run": False,
    }

    structural_results: dict[str, Any] = {}
    for family in config["structural_families"]:
        metrics_receipt = read_json(_resolve(family["metrics_path"]))
        structural = pd.read_parquet(_resolve(family["oof_path"]))
        full = structural.loc[
            np.isclose(
                structural["prefix_fraction"].to_numpy(dtype=np.float64),
                float(family["prefix_fraction"]),
                rtol=0.0,
                atol=0.0,
            )
        ].copy()
        structural_surface = validate_surface(
            full,
            expected_cases=surface["expected_cases"],
            expected_rows=surface["expected_rows"],
            expected_leads=surface["leads_h"],
        )
        alignment = compare_aligned_surface(
            canonical,
            full,
            float_columns=("target_hs", "current_hs", "persistence"),
            tolerance=0.0,
        )
        fit_receipts = _model_fit_receipts(metrics_receipt)
        if fit_receipts != int(family["fit_cells_each_side"]):
            raise ValueError(
                f"Structural fit receipt count differs for {family['id']}: {fit_receipts}"
            )
        point = metrics_receipt["points"]["1.0"]
        if len(point["incumbent_seed_metrics"]) != int(family["seed_count_each_side"]):
            raise ValueError("Structural incumbent seed count differs")
        if len(point["challenger_seed_metrics"]) != int(family["seed_count_each_side"]):
            raise ValueError("Structural challenger seed count differs")
        incumbent_summary = metric_summary(full, "incumbent_prediction")
        challenger_summary = metric_summary(full, "challenger_prediction")
        if abs(incumbent_summary["rmse_m"] - float(point["incumbent_rmse_m"])) > 1e-12:
            raise ValueError("Structural incumbent metric does not reproduce receipt")
        if abs(challenger_summary["rmse_m"] - float(point["challenger_rmse_m"])) > 1e-12:
            raise ValueError("Structural challenger metric does not reproduce receipt")
        structural_results[family["id"]] = {
            "hypothesis": family["hypothesis"],
            "surface": structural_surface,
            "alignment_to_canonical_truth_surface": alignment,
            "budget": {
                "seed_count_each_side": int(family["seed_count_each_side"]),
                "fit_cells_each_side": int(family["fit_cells_each_side"]),
                "observed_challenger_fit_receipts": fit_receipts,
                "fixed_postprocess": family["fixed_postprocess"],
                "same_surface_metric_postprocess": True,
            },
            "incumbent_matched_refit": incumbent_summary,
            "challenger": challenger_summary,
            "delta_challenger_minus_incumbent_m": slice_delta(
                challenger_summary, incumbent_summary
            ),
            "complete_case_bootstrap": complete_case_bootstrap_delta(
                full,
                candidate="challenger_prediction",
                reference="incumbent_prediction",
                replicates=bootstrap["bootstrap_replicates"],
                seed=bootstrap["bootstrap_seed"],
            ),
            "residual_correlation": residual_correlation(
                full, "challenger_prediction", "incumbent_prediction"
            ),
            "receipt_metric_reproduction": True,
            "external_access_counters_all_zero": _zero_external_access(metrics_receipt),
        }

    density = read_json(_resolve(config["input_files"]["density_gate_metrics"]["path"]))
    failed_receipts = [
        receipt for receipt in density["domain_receipts"] if receipt.get("passed") is False
    ]
    density_gate = {
        "status": density["status"],
        "selection_reason": density["candidate"]["selection_reason"],
        "failed_outer_folds": [receipt["outer_fold"] for receipt in failed_receipts],
        "minimum_ratio_ess_fraction": float(
            min(receipt["ratio_ess_fraction"] for receipt in density["domain_receipts"])
        ),
        "minimum_combined_ess_fraction": float(
            min(receipt["combined_ess_fraction"] for receipt in density["domain_receipts"])
        ),
        "minimum_station_combined_ess_fraction": float(
            min(
                min(receipt["station_combined_ess_fraction"].values())
                for receipt in density["domain_receipts"]
            )
        ),
        "override_attempted": False,
        "density_model_trained_in_this_run": False,
    }
    if density_gate["selection_reason"] != "NO_GO_LABEL_FREE_DOMAIN_GATE":
        raise ValueError("Density exclusion receipt does not preserve NO_GO")
    if not failed_receipts:
        raise ValueError("Density gate unexpectedly has no failed fold")

    structural_deltas = {
        family_id: values["delta_challenger_minus_incumbent_m"]["pooled"]
        for family_id, values in structural_results.items()
    }
    coefficient_deltas = {
        setting_id: float(values["rmse_m"] - w20)
        for setting_id, values in canonical_metrics.items()
        if setting_id in weights
    }
    result = {
        "schema_version": "p3.matched_budget_local_compare.metrics.v1",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _kst_now(),
        "status": "LOCAL_COMPARE_COMPLETE_NO_PROMOTION",
        "common_protocol_sha256": common["sha256"],
        "selection_quarantine": {
            "official_public_score_used_for_selection": False,
            "result_driven_candidate_additions": 0,
            "result_driven_reruns": 0,
            "promotion_authorized": False,
        },
        "surface": surface_audit,
        "additional_windows": {
            "count": 0,
            "reason": surface["additional_window_reason"],
        },
        "canonical_family": {
            "settings": canonical_metrics,
            "comparisons_vs_current_20pct": coefficient_comparisons,
            "coefficient_sensitivity": coefficient_sensitivity,
            "current_20pct_reproduction_max_abs_m": current_difference,
            "round_a_25pct_reference_alignment": round_a_alignment,
            "round_a_25pct_reference_prediction_max_abs_m": round_a_max_abs,
        },
        "structural_families": structural_results,
        "density_gate": density_gate,
        "summary": {
            "coefficient_delta_vs_current_m": coefficient_deltas,
            "structural_delta_vs_matched_incumbent_m": structural_deltas,
            "all_structural_defaults_improve": all(value < 0.0 for value in structural_deltas.values()),
            "structure_gain_supported": any(value < 0.0 for value in structural_deltas.values()),
            "coefficient_only_gain_is_small": min(coefficient_deltas.values()) > -0.005,
            "maturity_bias_supported": False,
            "promotion_authorized": False,
        },
        "resource_isolation": {
            "model_fits": 0,
            "aggregation_thread_cap": config["runtime"]["aggregation_cpu_thread_cap"],
            "era5_path_or_process_accesses": 0,
        },
        "elapsed_seconds": float(perf_counter() - started),
    }
    exclusive_write_json(output_dir / "metrics.json", result)
    receipt = {
        "schema_version": "p3.matched_budget_local_compare.execution_receipt.v1",
        "created_at_kst": _kst_now(),
        "status": "EXECUTED_ONCE_FROM_PRESEALED_INPUTS",
        "preexecution_seal_sha256": sha256_file(seal_path),
        "config_sha256": sha256_file(config_path),
        "metrics_sha256": sha256_file(output_dir / "metrics.json"),
        "input_sha256_after": {
            key: sha256_file(_resolve(value["path"]))
            for key, value in config["input_files"].items()
        }
        | {
            family["id"] + "_oof": sha256_file(_resolve(family["oof_path"]))
            for family in config["structural_families"]
        }
        | {
            family["id"] + "_metrics": sha256_file(_resolve(family["metrics_path"]))
            for family in config["structural_families"]
        },
        "external_actions": {
            "official_evaluation_value_reads": 0,
            "submission_value_reads": 0,
            "submission_files_generated": 0,
            "uploads": 0,
            "era5_path_or_process_accesses": 0,
            "model_fits": 0,
        },
    }
    exclusive_write_json(output_dir / "execution_receipt.json", receipt)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/p3_matched_budget_local_compare_20260825_v1.json",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to score without --execute")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    result = execute(args.config.resolve())
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
