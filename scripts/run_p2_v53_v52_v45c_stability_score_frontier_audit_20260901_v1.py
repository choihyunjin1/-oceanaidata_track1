"""Zero-fit, leakage-safe P2 v52/v45c stability-score frontier audit.

The only model predictions opened are hash-pinned historical OOF commitments.
Truth is reconstructed from organizer-distributed ``observations.csv`` on the
same already-exposed historical rows.  Official inputs, submissions and uploads
have no code path in this executable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402

EXPERIMENT_ID = "p2_v53_v52_v45c_stability_score_frontier_audit_20260901_v1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
RESULT_SCHEMA = "p2.v53_v52_v45c_stability_score_frontier_audit.result.20260901.v1"
FOLDS = ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")
LAYERS = (2, 3, 4)
ALIGNMENT_FIELDS = ("time_ns", "layer", "fold", "reference")


class ContractError(RuntimeError):
    """Raised when a sealed v53 audit contract drifts."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name].copy() for name in bundle.files}


def _verify_evidence(config: dict[str, Any]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for name, relative in config["evidence"].items():
        if name.endswith("_sha256"):
            continue
        path = ROOT / relative
        expected = config["evidence"][f"{name}_sha256"]
        if not path.is_file() or v12.sha256_file(path) != expected:
            raise ContractError(f"v53 evidence hash drift: {name}")
        output[name] = path
    return output


def load_config() -> tuple[dict[str, Any], dict[str, Path]]:
    config = _load_json(CONFIG)
    evidence = _verify_evidence(config)
    source = config["source_contract"]
    policy = config["sealed_selection_policy"]
    evaluation = config["evaluation"]
    limits = config["operation_limits"]
    forbidden_source_flags = (
        "external_observation_allowed",
        "external_reanalysis_allowed",
        "external_forecast_allowed",
        "pretrained_weights_allowed",
        "official_test_index_allowed",
        "official_sample_allowed",
        "official_baseline_allowed",
        "query_support_allowed",
        "hidden_truth_allowed",
        "submission_csv_allowed",
        "upload_allowed",
    )
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_ZERO_FIT_NOT_EXECUTED"
        or not source["organizer_distributed_data_only"]
        or source["only_source_filename"] != "observations.csv"
        or any(source[name] for name in forbidden_source_flags)
        or tuple(policy["fold_order"]) != FOLDS
        or tuple(policy["target_layers"]) != LAYERS
        or policy["v52_weight_grid"] != [0.0, 0.25, 0.5, 0.75, 1.0]
        or policy["selection_unit"] != "heldout_fold_x_target_layer"
        or policy["heldout_outcomes_visible_during_selection"]
        or policy["outcome_adaptive_grid_expansion"]
        or policy["result_adaptive_refit"]
        or policy["candidate_refit"]
        or evaluation["minimum_fold_layer_non_harm_cells"] != 8
        or evaluation["total_fold_layer_cells"] != 9
        or evaluation["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or limits["maximum_model_fit_count"] != 0
        or limits["automatic_retry_count"] != 0
        or any(
            limits[name] != 0
            for name in (
                "official_test_index_rows_read",
                "sample_rows_read",
                "baseline_file_rows_read",
                "query_support_rows_read",
                "hidden_truth_rows_read",
                "submission_csv_created",
                "uploads",
            )
        )
    ):
        raise ContractError("v53 preregistered contract drift")
    return config, evidence


def preflight() -> dict[str, Any]:
    config, evidence = load_config()
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_FIT_FRONTIER_PREFLIGHT_READY",
        "selection_policy": config["sealed_selection_policy"],
        "source_hashes": {
            name: v12.sha256_file(path) for name, path in sorted(evidence.items())
        },
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "data_rows_read": 0,
        "model_fits": 0,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    payload["preflight_sha256"] = v12.sha256_json(payload)
    return payload


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def _metric(
    mask: np.ndarray,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int]:
    ref = _rmse(truth[mask], reference[mask])
    cand = _rmse(truth[mask], candidate[mask])
    return {
        "rows": int(mask.sum()),
        "reference_rmse_C": ref,
        "candidate_rmse_C": cand,
        "delta_rmse_C": cand - ref,
    }


def _metrics(
    blind: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    all_rows = np.ones(len(truth), dtype=bool)
    by_fold: dict[str, Any] = {}
    by_layer: dict[str, Any] = {}
    by_fold_layer: dict[str, Any] = {}
    for fold in FOLDS:
        fold_mask = blind["fold"].eq(fold).to_numpy()
        by_fold[fold] = _metric(fold_mask, truth, reference, candidate)
        by_fold_layer[fold] = {}
        for layer in LAYERS:
            mask = fold_mask & blind["layer"].eq(layer).to_numpy()
            by_fold_layer[fold][str(layer)] = _metric(
                mask, truth, reference, candidate
            )
    for layer in LAYERS:
        mask = blind["layer"].eq(layer).to_numpy()
        by_layer[str(layer)] = _metric(mask, truth, reference, candidate)
    return {
        "pooled": _metric(all_rows, truth, reference, candidate),
        "by_fold": by_fold,
        "by_layer": by_layer,
        "by_fold_layer": by_fold_layer,
    }


def _reconstruct_truth(
    config: dict[str, Any], scoring_path: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, int]]:
    observations_path = v12.resolve_observations(
        {
            "source_contract": {
                "environment_variable": config["source_contract"]["environment_variable"],
                "only_source_filename": config["source_contract"]["only_source_filename"],
                "observations_sha256": config["source_contract"]["observations_sha256"],
            }
        }
    )
    observations = pd.read_csv(
        observations_path, dtype={"station": "string", "time": "string"}
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    if observations.duplicated(["time", "layer"]).any():
        raise ContractError("distributed observation keys are duplicated")
    scored = pd.read_parquet(scoring_path)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    blind, reference = v12.metric_engine.make_reference(observations, scored)
    feature_table = v12.build_training_features(observations)
    design = v12.build_normalized_curvature_design(feature_table.frame)
    design_index = pd.MultiIndex.from_arrays(
        [
            v12.metric_engine.canonical_time_ns(design.keys["time"]),
            design.keys["layer"],
        ],
        names=("time", "layer"),
    )
    query_index = pd.MultiIndex.from_arrays(
        [
            v12.metric_engine.canonical_time_ns(blind["time"]),
            blind["layer"],
        ],
        names=("time", "layer"),
    )
    positions = design_index.get_indexer(query_index)
    if design_index.has_duplicates or np.any(positions < 0):
        raise ContractError("v53 historical truth alignment failed")
    truth = design.truth[positions]
    if not (np.isfinite(truth).all() and np.isfinite(reference).all()):
        raise ContractError("v53 truth/reference contains non-finite values")
    return blind, truth, reference, {
        "observations_rows": int(len(observations)),
        "historical_scoring_rows": int(len(scored)),
    }


def _verify_prediction_alignment(
    score: dict[str, np.ndarray],
    stable: dict[str, np.ndarray],
    blind: pd.DataFrame,
    reference: np.ndarray,
) -> dict[str, Any]:
    if set(score) != set(stable) or set(score) != {
        "time_ns",
        "layer",
        "fold",
        "reference",
        "candidate",
    }:
        raise ContractError("v53 OOF schema drift")
    for name in ALIGNMENT_FIELDS:
        if not np.array_equal(score[name], stable[name]):
            raise ContractError(f"v53 source OOF alignment drift: {name}")
    expected_time = v12.metric_engine.canonical_time_ns(blind["time"])
    if (
        not np.array_equal(score["time_ns"], expected_time)
        or not np.array_equal(score["layer"], blind["layer"].to_numpy(np.int16))
        or not np.array_equal(score["fold"], blind["fold"].to_numpy(str))
        or not np.array_equal(score["reference"], reference)
        or not np.isfinite(score["candidate"]).all()
        or not np.isfinite(stable["candidate"]).all()
    ):
        raise ContractError("v53 OOF-to-distributed-truth alignment drift")
    return {
        "rows": int(len(reference)),
        "schema": ["time_ns", "layer", "fold", "reference", "candidate"],
        "time_equal": True,
        "layer_equal": True,
        "fold_equal": True,
        "reference_byte_equal": True,
        "finite_predictions": True,
    }


def _select_lofo(
    config: dict[str, Any],
    blind: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    stable: np.ndarray,
    score: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    grid = [float(value) for value in config["sealed_selection_policy"]["v52_weight_grid"]]
    limit = float(config["evaluation"]["maximum_any_fold_layer_delta_rmse_C"])
    candidate = np.empty_like(reference)
    receipts: list[dict[str, Any]] = []
    choices: dict[int, list[float]] = {layer: [] for layer in LAYERS}
    fold_values = blind["fold"].to_numpy(str)
    layer_values = blind["layer"].to_numpy(int)
    for heldout in FOLDS:
        for layer in LAYERS:
            train_folds = tuple(fold for fold in FOLDS if fold != heldout)
            rows = (layer_values == layer) & np.isin(fold_values, train_folds)
            eligible: list[dict[str, Any]] = []
            all_grid: list[dict[str, Any]] = []
            for alpha in grid:
                mixed = stable + alpha * (score - stable)
                train_metric = _metric(rows, truth, reference, mixed)
                cell_deltas: dict[str, float] = {}
                for train_fold in train_folds:
                    cell = (layer_values == layer) & (fold_values == train_fold)
                    cell_deltas[train_fold] = float(
                        _metric(cell, truth, reference, mixed)["delta_rmse_C"]
                    )
                item = {
                    "v52_weight": alpha,
                    "training_rows": int(rows.sum()),
                    "training_pooled_rmse_C": float(train_metric["candidate_rmse_C"]),
                    "training_fold_layer_delta_rmse_C": cell_deltas,
                    "training_safety_eligible": max(cell_deltas.values()) <= limit,
                }
                all_grid.append(item)
                if item["training_safety_eligible"]:
                    eligible.append(item)
            if not eligible:
                raise ContractError("sealed LOFO grid has no training-safe weight")
            selected = min(
                eligible,
                key=lambda item: (
                    item["training_pooled_rmse_C"],
                    item["v52_weight"],
                ),
            )
            alpha = float(selected["v52_weight"])
            heldout_rows = (layer_values == layer) & (fold_values == heldout)
            candidate[heldout_rows] = stable[heldout_rows] + alpha * (
                score[heldout_rows] - stable[heldout_rows]
            )
            choices[layer].append(alpha)
            receipts.append(
                {
                    "heldout_fold": heldout,
                    "layer": layer,
                    "training_folds": list(train_folds),
                    "selected_v52_weight": alpha,
                    "selected_without_heldout_outcomes": True,
                    "sealed_grid_receipt": all_grid,
                }
            )
    if not np.isfinite(candidate).all():
        raise ContractError("LOFO frontier candidate is incomplete or non-finite")
    deployment = {
        str(layer): float(np.median(np.asarray(weights, dtype=float)))
        for layer, weights in choices.items()
    }
    return candidate, receipts, deployment


def _bootstrap(
    config: dict[str, Any],
    blind: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    return v12.metric_engine.grouped_bootstrap(
        blind,
        truth,
        reference,
        candidate,
        seed=int(config["evaluation"]["bootstrap_seed"]),
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
    )


def _result_hash(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("result_payload_sha256", None)
    return v12.sha256_json(copy)


def write_report(result: dict[str, Any]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    pooled = result["candidate_metrics"]["pooled"]
    gate = result["promotion_gate"]
    REPORT.joinpath("report-source.md").write_text(
        "# P2 v53 v52/v45c stability-score frontier audit\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. LOFO pooled delta RMSE "
        f"`{pooled['delta_rmse_C']:+.9f} C`, nominal "
        f"`{result['score_translation']['nominal_points']:+.6f}` points, "
        f"transport `{result['score_translation']['transport_adjusted_points']:+.6f}` points.\n\n"
        f"v23 대비 delta RMSE 차이는 `{result['v23_comparison']['delta_rmse_difference_C']:+.9f} C`, "
        f"transport 점수 차이는 `{result['v23_comparison']['transport_points_difference']:+.6f}`다. "
        f"fold-layer non-harm `{gate['non_harm_cells']}/9`, worst cell "
        f"`{gate['maximum_cell_delta_rmse_C']:+.9f} C`, 최종 gate `{gate['pass']}`.\n\n"
        "가중치는 각 held-out fold×layer의 결과를 보지 않고 다른 두 fold의 같은 layer에서만 "
        "고른 cross-fitted 진단이다. median deployment weight는 진단용일 뿐 새 제출 후보가 아니며, "
        "fresh deployment preflight 전 materialization하지 않는다. model fits=0; "
        "official/test/sample/baseline/query/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    started = time.perf_counter()
    if ARTIFACT.exists():
        raise FileExistsError(ARTIFACT)
    config, evidence = load_config()
    ARTIFACT.mkdir(parents=True)
    v12.atomic_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": v12.sha256_file(CONFIG),
            "runner_sha256": v12.sha256_file(RUNNER),
            "fit_count": 0,
            "automatic_retry_count": 0,
            "sealed_selection_policy_sha256": v12.sha256_json(
                config["sealed_selection_policy"]
            ),
        },
    )
    scoring_path = ROOT / config["source_contract"]["truth_free_scoring_frame"]
    if v12.sha256_file(scoring_path) != config["source_contract"][
        "truth_free_scoring_frame_sha256"
    ]:
        raise ContractError("v53 truth-free scoring frame hash drift")
    blind, truth, reference, row_counts = _reconstruct_truth(config, scoring_path)
    score_bundle = _load_npz(evidence["v52_prediction"])
    stable_bundle = _load_npz(evidence["v45c_prediction"])
    alignment = _verify_prediction_alignment(
        score_bundle, stable_bundle, blind, reference
    )
    score_prediction = score_bundle["candidate"].astype(float)
    stable_prediction = stable_bundle["candidate"].astype(float)
    component_metrics = {
        "v52": _metrics(blind, truth, reference, score_prediction),
        "v45c": _metrics(blind, truth, reference, stable_prediction),
    }
    v52_result = _load_json(evidence["v52_result"])
    v45c_result = _load_json(evidence["v45c_result"])
    if (
        abs(
            float(component_metrics["v52"]["pooled"]["delta_rmse_C"])
            - float(v52_result["candidate"]["delta_rmse"])
        )
        > 1e-12
        or abs(
            float(component_metrics["v45c"]["pooled"]["delta_rmse_C"])
            - float(v45c_result["candidate"]["delta_rmse"])
        )
        > 1e-12
    ):
        raise ContractError("component result did not independently reproduce")
    candidate, receipts, deployment_weights = _select_lofo(
        config,
        blind,
        truth,
        reference,
        stable_prediction,
        score_prediction,
    )
    metrics = _metrics(blind, truth, reference, candidate)
    bootstrap = _bootstrap(config, blind, truth, reference, candidate)
    cells = [
        metrics["by_fold_layer"][fold][str(layer)]["delta_rmse_C"]
        for fold in FOLDS
        for layer in LAYERS
    ]
    evaluation = config["evaluation"]
    delta = float(metrics["pooled"]["delta_rmse_C"])
    nominal = -delta * float(evaluation["points_per_rmse_C"])
    transport = nominal - float(evaluation["transport_penalty_points"])
    non_harm = sum(float(value) <= 0.0 for value in cells)
    maximum_cell = max(float(value) for value in cells)
    checks = {
        "beats_v23_pooled_rmse": delta < float(evaluation["v23_delta_rmse_C"]),
        "minimum_eight_of_nine_non_harm": non_harm
        >= int(evaluation["minimum_fold_layer_non_harm_cells"]),
        "all_cells_within_plus_0_003C": maximum_cell
        <= float(evaluation["maximum_any_fold_layer_delta_rmse_C"]),
        "all_three_folds_improve": all(
            float(metrics["by_fold"][fold]["delta_rmse_C"]) < 0.0
            for fold in FOLDS
        ),
        "pooled_bootstrap_ci90_high_below_zero": float(bootstrap["ci90_high"])
        < 0.0,
        "transport_points_beat_v23": transport
        > float(evaluation["v23_transport_adjusted_points"]),
    }
    passed = bool(all(checks.values()))
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "PASS_LOFO_STABILITY_SCORE_FRONTIER_BEATS_V23"
            if passed
            else "NO_GO_NO_LOFO_FRONTIER_CANDIDATE_BEATS_V23_AND_SAFETY"
        ),
        "claim_level": config["claim_level"],
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 0,
        "candidate_count": 1,
        "source_alignment": alignment,
        "source_component_metrics": component_metrics,
        "selection_receipts": receipts,
        "diagnostic_median_deployment_v52_weights_by_layer": deployment_weights,
        "candidate_metrics": metrics,
        "bootstrap": bootstrap,
        "score_translation": {
            "points_per_rmse_C": evaluation["points_per_rmse_C"],
            "transport_penalty_points": evaluation["transport_penalty_points"],
            "nominal_points": nominal,
            "transport_adjusted_points": transport,
        },
        "v23_comparison": {
            "v23_delta_rmse_C": evaluation["v23_delta_rmse_C"],
            "v23_candidate_rmse_C": evaluation["v23_candidate_rmse_C"],
            "v23_nominal_points": evaluation["v23_nominal_points"],
            "v23_transport_adjusted_points": evaluation[
                "v23_transport_adjusted_points"
            ],
            "delta_rmse_difference_C": delta - float(evaluation["v23_delta_rmse_C"]),
            "candidate_rmse_difference_C": float(
                metrics["pooled"]["candidate_rmse_C"]
            )
            - float(evaluation["v23_candidate_rmse_C"]),
            "nominal_points_difference": nominal
            - float(evaluation["v23_nominal_points"]),
            "transport_points_difference": transport
            - float(evaluation["v23_transport_adjusted_points"]),
        },
        "promotion_gate": {
            "checks": checks,
            "non_harm_cells": non_harm,
            "total_cells": 9,
            "maximum_cell_delta_rmse_C": maximum_cell,
            "pass": passed,
        },
        "recommendation": (
            "FRESH_DEPLOYMENT_PREFLIGHT_FOR_SEALED_MEDIAN_WEIGHT_POLICY_ONLY"
            if passed
            else "KEEP_V23_OFFICIAL_ANCHOR_AND_V45C_STABLE_RESEARCH_FALLBACK"
        ),
        "caveats": {
            "historical_folds_already_exposed": True,
            "lofo_selection_never_saw_heldout_fold_layer_outcomes": True,
            "median_deployment_weights_are_diagnostic_only": True,
            "fresh_temporal_confirmation_available": False,
            "official_performance_claimed": False,
        },
        "source_hashes": {
            name: v12.sha256_file(path) for name, path in sorted(evidence.items())
        },
        "hashes": {
            "config": v12.sha256_file(CONFIG),
            "runner": v12.sha256_file(RUNNER),
            "selection_policy": v12.sha256_json(config["sealed_selection_policy"]),
        },
        "operation_counters": {
            "observations_rows_read": row_counts["observations_rows"],
            "historical_scoring_rows_read": row_counts["historical_scoring_rows"],
            "v52_committed_oof_rows_read": int(len(score_prediction)),
            "v45c_committed_oof_rows_read": int(len(stable_prediction)),
            "model_fits": 0,
            "external_observation_rows_read": 0,
            "external_reanalysis_rows_read": 0,
            "external_forecast_rows_read": 0,
            "pretrained_weight_files_loaded": 0,
            "official_test_index_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_file_rows_read": 0,
            "query_support_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
    }
    result["result_payload_sha256"] = _result_hash(result)
    v12.atomic_json(ARTIFACT / "result.json", result)
    v12.atomic_json(REPORT / "result.json", result)
    write_report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight or --execute")
    value = preflight() if args.preflight else run()
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
