"""Sealed technical overlay for the P2 continuous sparse copula v1 guard bug."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p2_restore import (
    p2_availability_aware_continuous_sparse_copula_20260830_v1 as sealed,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "p2_availability_aware_continuous_sparse_copula_20260830_v2"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RUNNER_PATH = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
SEALED_CONFIG_CANONICAL_SHA256 = "b67f7017603f4556624396a12407df34d11a7ede1360104a3d0444ef58c8cb0b"


class OverlayContractError(sealed.ExperimentContractError):
    """Raised when the v2 technical overlay contract is violated."""


def _validate_pinned_record(record: dict[str, Any]) -> None:
    path = ROOT / record["path"]
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sealed._sha256_file(path) != record["sha256"]
    ):
        raise OverlayContractError(f"sealed overlay evidence changed: {path}")


def load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if sealed._canonical_sha256(overlay) != SEALED_CONFIG_CANONICAL_SHA256:
        raise OverlayContractError("v2 preregistered overlay changed")
    if overlay.get("experiment_id") != EXPERIMENT_ID:
        raise OverlayContractError("v2 experiment id changed")
    base_record = overlay["base_experiment"]
    base_path = ROOT / base_record["config_path"]
    if (
        base_path.stat().st_size != int(base_record["config_bytes"])
        or sealed._sha256_file(base_path) != base_record["config_sha256"]
    ):
        raise OverlayContractError("v1 base config changed")
    base_core = ROOT / base_record["core_path"]
    if (
        base_core.stat().st_size != int(base_record["core_bytes"])
        or sealed._sha256_file(base_core) != base_record["core_sha256"]
    ):
        raise OverlayContractError("v1 base core changed")
    for record in overlay["sealed_failure_evidence"].values():
        if isinstance(record, dict) and {"path", "bytes", "sha256"}.issubset(record):
            _validate_pinned_record(record)
    repair = overlay["technical_overlay"]
    changed = [
        name
        for name, value in repair.items()
        if name.endswith("_changed") and bool(value)
    ]
    if changed or repair["changed_component"] != "post_prediction_physical_domain_guard_only":
        raise OverlayContractError("v2 overlay expands beyond the physical guard")
    new_guard = repair["new_guard"]
    if (
        list(map(float, new_guard["absolute_domain_c"])) != [-5.0, 45.0]
        or int(new_guard["newly_out_of_domain_candidate_count_must_equal"]) != 0
        or int(new_guard["active_out_of_domain_candidate_count_must_equal"]) != 0
        or not new_guard["preexisting_out_of_domain_reference_must_be_exact_unchanged"]
        or not new_guard["inactive_rows_must_equal_reference_exactly"]
    ):
        raise OverlayContractError("v2 relative-domain guard changed")
    policy = overlay["execution_policy"]
    if (
        not policy["real_training_execution_authorized"]
        or int(policy["maximum_executions"]) != 1
        or policy["v1_reexecution_allowed"]
        or any(
            policy[name]
            for name in (
                "result_based_tuning",
                "result_based_retry",
                "technical_failure_retry",
                "official_interface_reads_allowed",
                "query_support_reads_allowed",
                "csv_output_allowed",
                "submission_generation_allowed",
                "upload_allowed",
            )
        )
        or not policy["aggregate_json_only"]
    ):
        raise OverlayContractError("v2 execution policy changed")
    diagnostic_record = overlay["sealed_failure_evidence"]["guard_diagnostic"]
    diagnostic = json.loads((ROOT / diagnostic_record["path"]).read_text(encoding="utf-8"))
    predicates = diagnostic["guard_operands"]["guard_predicates"]
    if (
        diagnostic["diagnostic_scope"]["performance_metrics_computed"]
        or not predicates["candidate_all_finite"]
        or int(predicates["reference_outside_count"]) <= 0
        or int(predicates["new_candidate_outside_count"]) != 0
        or int(predicates["active_candidate_outside_count"]) != 0
    ):
        raise OverlayContractError("sealed diagnostic does not prove a guard-only bug")
    base = sealed.load_config()
    if base_record["config_canonical_sha256"] != sealed.SEALED_CONFIG_CANONICAL_SHA256:
        raise OverlayContractError("v1 base canonical hash changed")
    merged = copy.deepcopy(base)
    merged["experiment_id"] = EXPERIMENT_ID
    merged["artifact_path"] = overlay["artifact_path"]
    merged["classification"] = overlay["classification"]
    return merged, overlay


def _relative_physical_domain_guard(
    *,
    reference: np.ndarray,
    candidate: np.ndarray,
    correction: np.ndarray,
    active_rows: np.ndarray,
    lower_c: float = -5.0,
    upper_c: float = 45.0,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    active = np.asarray(active_rows, dtype=bool)
    if not (reference.shape == candidate.shape == correction.shape == active.shape):
        raise OverlayContractError("relative-domain guard operand shapes differ")
    if not np.isfinite(candidate).all():
        raise OverlayContractError("candidate became nonfinite")
    outside_reference = (reference < lower_c) | (reference > upper_c)
    outside_candidate = (candidate < lower_c) | (candidate > upper_c)
    newly_outside = outside_candidate & ~outside_reference
    changed_preexisting = outside_reference & (candidate != reference)
    active_outside = active & outside_candidate
    inactive_changed = (~active) & (candidate != reference)
    if newly_outside.any():
        raise OverlayContractError("candidate created a new physical-domain violation")
    if changed_preexisting.any():
        raise OverlayContractError("pre-existing reference extreme was not an exact no-op")
    if active_outside.any():
        raise OverlayContractError("an out-of-domain candidate was marked active")
    if inactive_changed.any():
        raise OverlayContractError("an inactive candidate changed from its reference")
    return {
        "contract": "relative_to_frozen_reference_preserve_preexisting_extrema_exact_noop",
        "absolute_domain_c": [lower_c, upper_c],
        "reference_outside_count": int(outside_reference.sum()),
        "candidate_outside_count": int(outside_candidate.sum()),
        "new_candidate_outside_count": int(newly_outside.sum()),
        "changed_preexisting_extreme_count": int(changed_preexisting.sum()),
        "active_candidate_outside_count": int(active_outside.sum()),
        "inactive_changed_count": int(inactive_changed.sum()),
        "candidate_all_finite": True,
        "prediction_values_changed_by_overlay": False,
    }


def _predict_outer(
    *,
    fold: str,
    fold_spec: dict[str, Any],
    config: dict[str, Any],
    observations: pd.DataFrame,
    state_table: pd.DataFrame,
    profile_flags: pd.DataFrame,
    anchor_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    query, reference, training, _endpoints, reference_receipt = sealed.legacy._reference_frames(
        fold=fold,
        fold_spec=fold_spec,
        config=config,
        observations=observations,
        anchor_path=anchor_path,
    )
    model, model_receipt = sealed._fit_outer_model(training, state_table, config)
    profile = query[["station", "time"]].drop_duplicates()
    counts = query.groupby(["station", "time"], sort=False)["layer"].nunique().rename(
        "target_layer_count"
    )
    profile = profile.merge(
        counts.reset_index(), on=["station", "time"], how="left", validate="one_to_one"
    )
    profile = profile.merge(
        state_table, on=["station", "time"], how="left", validate="one_to_one"
    )
    complete_target = profile["target_layer_count"].eq(len(sealed.legacy.TARGET_LAYERS)).to_numpy()
    raw_profile = np.zeros((len(profile), len(sealed.RESPONSES)), dtype=np.float64)
    active_profile = np.zeros(len(profile), dtype=bool)
    no_op_reasons = {
        "incomplete_target_profile": int((~complete_target).sum()),
        "missing_public_state": 0,
        "ood_public_state": 0,
        "correlation_matrix_guard": 0,
        "outer_model_guard": 0,
    }
    if model is None:
        no_op_reasons["outer_model_guard"] = int(complete_target.sum())
    else:
        prediction = model.predict(profile)
        eligible_active = prediction.active & complete_target
        raw_profile[eligible_active] = prediction.correction[eligible_active]
        active_profile[eligible_active] = True
        no_op_reasons["missing_public_state"] = int(
            (prediction.missing & complete_target).sum()
        )
        no_op_reasons["ood_public_state"] = int((prediction.ood & complete_target).sum())
        no_op_reasons["correlation_matrix_guard"] = int(
            (prediction.matrix_guard & complete_target).sum()
        )
    profile_correction = profile[["station", "time"]].copy()
    for column, response in enumerate(sealed.RESPONSES):
        profile_correction[response] = raw_profile[:, column]
    profile_correction["active_profile"] = active_profile
    long = profile_correction.melt(
        id_vars=["station", "time", "active_profile"],
        value_vars=list(sealed.RESPONSES),
        var_name="response",
        value_name="raw_correction",
    )
    long["layer"] = long["response"].str.removeprefix("residual_l").astype(int)
    row_prediction = query[["station", "time", "layer", "current_blend50"]].copy()
    row_prediction["reference"] = reference
    row_prediction = row_prediction.merge(
        long.drop(columns="response"),
        on=["station", "time", "layer"],
        how="left",
        validate="one_to_one",
    )
    row_prediction["active_profile"] = row_prediction["active_profile"].fillna(False)
    row_prediction["raw_correction"] = row_prediction["raw_correction"].fillna(0.0)
    lower = float(config["correction"]["structural_minimum_c"])
    upper = float(config["correction"]["structural_maximum_c"])
    correction = np.clip(
        row_prediction["raw_correction"].to_numpy(dtype=np.float64), lower, upper
    )
    active_rows = row_prediction["active_profile"].to_numpy(dtype=bool)
    correction[~active_rows] = 0.0
    candidate = reference + correction
    if not np.array_equal(candidate[~active_rows], reference[~active_rows]):
        raise OverlayContractError("exact no-op rows changed")
    domain_receipt = _relative_physical_domain_guard(
        reference=reference,
        candidate=candidate,
        correction=correction,
        active_rows=active_rows,
    )
    row_prediction["candidate"] = candidate
    row_prediction["correction"] = candidate - reference
    realized = row_prediction["correction"].to_numpy(dtype=np.float64)
    if np.max(realized) > upper + 1e-12 or np.min(realized) < lower - 1e-12:
        raise OverlayContractError("structural correction bound failed")
    receipt = {
        "fold": fold,
        "training_blocks": list(fold_spec["training_blocks"]),
        "training_rows": int(len(training)),
        "query_rows": int(len(query)),
        "query_profiles": int(len(profile)),
        "model": model_receipt,
        "reference": reference_receipt,
        "no_op_profile_counts": no_op_reasons,
        "active_profiles": int(active_profile.sum()),
        "inactive_profiles": int((~active_profile).sum()),
        "maximum_absolute_inactive_correction": float(
            np.max(np.abs(realized[~active_rows])) if (~active_rows).any() else 0.0
        ),
        "structural_correction_bound_c": [lower, upper],
        "correction_rms_c_diagnostic": float(np.sqrt(np.mean(np.square(realized)))),
        "correction_p99_c_diagnostic": float(np.quantile(np.abs(realized), 0.99)),
        "outlier_diagnostic": sealed.legacy._outlier_receipt(profile_flags, training),
        "physical_domain_guard": domain_receipt,
        "prediction_sha256": sealed.legacy._prediction_hash(row_prediction),
    }
    return row_prediction, receipt


def run(config: dict[str, Any], overlay: dict[str, Any], p2_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    observations, source_receipt, access = sealed._read_training_source(p2_dir, config)
    state_table = sealed._public_state_table(observations, config)
    block_observations = sealed.legacy._assign_blocks(observations, config)
    diagnostic_rows = block_observations.loc[block_observations["block"].notna()].copy()
    marked = sealed.legacy.stage0._mark_row_diagnostics(diagnostic_rows, config)
    profile_flags = sealed.legacy.stage0._profile_flag_table(marked, config)
    anchor_record = config["immutable_training_inputs"]["alpha50_oof_anchor"]
    anchor_path = ROOT / anchor_record["path"]
    predictions: dict[str, pd.DataFrame] = {}
    fold_receipts: dict[str, Any] = {}
    with sealed.threadpool_limits(limits=int(config["resource_contract"]["blas_threads"])):
        for fold, fold_spec in config["frozen_historical_windows"].items():
            prediction, receipt = _predict_outer(
                fold=fold,
                fold_spec=fold_spec,
                config=config,
                observations=observations,
                state_table=state_table,
                profile_flags=profile_flags,
                anchor_path=anchor_path,
            )
            predictions[fold] = prediction
            fold_receipts[fold] = receipt
            if time.perf_counter() - started > float(
                config["resource_contract"]["maximum_wall_seconds"]
            ):
                raise OverlayContractError("bounded runtime exceeded before scoring")
    outer_fits = sum(
        int(receipt["model"]["outer_dependence_model_fits"])
        for receipt in fold_receipts.values()
    )
    edge_estimations = sum(
        int(receipt["model"]["continuous_edge_estimations"])
        for receipt in fold_receipts.values()
    )
    if outer_fits != int(config["resource_contract"]["outer_dependence_model_fits"]):
        raise OverlayContractError("outer fit count drifted")
    if edge_estimations != int(config["resource_contract"]["continuous_edge_estimations"]):
        raise OverlayContractError("continuous edge estimation count drifted")
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "truth_metrics_computed": False,
        "prediction_hashes": {
            fold: receipt["prediction_sha256"] for fold, receipt in fold_receipts.items()
        },
        "overlay_config_sha256": sealed._sha256_file(CONFIG_PATH),
        "base_config_sha256": overlay["base_experiment"]["config_sha256"],
        "observations_sha256": source_receipt["sha256"],
        "outer_dependence_model_fits": outer_fits,
        "continuous_edge_estimations": edge_estimations,
    }
    commitment_sha256 = sealed._canonical_sha256(commitment)
    scored_parts: list[pd.DataFrame] = []
    for fold, prediction in predictions.items():
        if sealed.legacy._prediction_hash(prediction) != fold_receipts[fold]["prediction_sha256"]:
            raise OverlayContractError("in-memory prediction changed before truth binding")
        truth = sealed.legacy.base.block_anchor(anchor_path, fold, include_truth=True)[
            ["time", "layer", "truth"]
        ]
        scored = prediction.merge(
            truth, on=["time", "layer"], how="left", validate="one_to_one"
        )
        if scored["truth"].isna().any() or len(scored) != len(prediction):
            raise OverlayContractError("late historical truth binding failed")
        scored["window"] = fold
        scored_parts.append(scored)
    scored = pd.concat(scored_parts, ignore_index=True)
    if len(scored) != int(overlay["inherited_scientific_contract"]["primary_rows"]):
        raise OverlayContractError("primary pooled row count drifted")
    scored["season"] = sealed.legacy._season_labels(scored["time"], config)
    metrics = {
        "pooled": sealed.legacy._metric_record(scored),
        "by_window": {
            str(key): sealed.legacy._metric_record(group)
            for key, group in scored.groupby("window", sort=True)
        },
        "by_layer": {
            str(int(key)): sealed.legacy._metric_record(group)
            for key, group in scored.groupby("layer", sort=True)
        },
        "by_season": {
            str(key): sealed.legacy._metric_record(group)
            for key, group in scored.groupby("season", sort=True)
        },
    }
    bootstrap = sealed._moving_block_bootstrap(scored, config)
    correction = scored["correction"].to_numpy(dtype=np.float64)
    pooled_delta = float(metrics["pooled"]["delta_rmse"])
    evidence_state = sealed._evidence_state(pooled_delta, bootstrap)
    tail_diagnostic = sealed._tail_risk_diagnostic(
        scored,
        int(config["primary_decision"]["paired_interval"]["block_length_days"]),
    )
    elapsed = time.perf_counter() - started
    if elapsed > float(config["resource_contract"]["maximum_wall_seconds"]):
        raise OverlayContractError("bounded runtime exceeded")
    result = {
        "schema_version": "p2.availability_aware_continuous_sparse_copula.result.20260830.v2",
        "experiment_id": EXPERIMENT_ID,
        "decision": evidence_state,
        "classification": overlay["classification"],
        "governing_policy": config["governing_policy"],
        "comparator": config["comparator"],
        "comparator_disclosure": config["comparator_disclosure"],
        "closed_family_rerun": False,
        "stage0_exposed_edges": list(sealed.EXPOSED_EDGES),
        "technical_overlay": overlay["technical_overlay"],
        "sealed_failure_evidence": overlay["sealed_failure_evidence"],
        "config_sha256": sealed._sha256_file(CONFIG_PATH),
        "config_canonical_sha256": SEALED_CONFIG_CANONICAL_SHA256,
        "base_config_sha256": overlay["base_experiment"]["config_sha256"],
        "implementation_hashes": {
            "core": sealed._sha256_file(Path(__file__)),
            "runner": sealed._sha256_file(RUNNER_PATH),
            "inherited_v1_core": overlay["base_experiment"]["core_sha256"],
        },
        "source": source_receipt,
        "source_open_counts": access.open_counts,
        "source_basenames_opened": ["observations.csv"],
        "immutable_training_input_hashes": {
            name: record["sha256"]
            for name, record in config["immutable_training_inputs"].items()
        },
        "prediction_commitment_sha256": commitment_sha256,
        "fold_receipts": fold_receipts,
        "metrics": metrics,
        "dependence_aware_bootstrap": bootstrap,
        "primary_decision_receipt": {
            "metric": config["primary_decision"]["metric"],
            "pooled_point_favorable": pooled_delta < 0.0,
            "paired_interval_wholly_favorable": float(bootstrap["ci90_high"]) < 0.0,
            "evidence_state": evidence_state,
            "diagnostic_slice_hard_veto_count": 0,
        },
        "diagnostics": {
            "tail_risk": tail_diagnostic,
            "transport_slices_are_not_hard_vetoes": True,
            "support_is_not_a_performance_veto": True,
            "correction_magnitude_is_not_a_performance_veto": True,
        },
        "correction": {
            "structural_bound_c": [
                float(config["correction"]["structural_minimum_c"]),
                float(config["correction"]["structural_maximum_c"]),
            ],
            "rms_c_diagnostic": float(np.sqrt(np.mean(np.square(correction)))),
            "p99_absolute_c_diagnostic": float(np.quantile(np.abs(correction), 0.99)),
            "maximum_absolute_c": float(np.max(np.abs(correction))),
        },
        "fit_counts": {
            "outer_dependence_model_fits": outer_fits,
            "inner_selection_fits": 0,
            "hpo_trials": 0,
            "continuous_edge_estimations": edge_estimations,
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "maximum_wall_seconds": float(config["resource_contract"]["maximum_wall_seconds"]),
            "python": platform.python_version(),
            "maximum_total_threads": int(config["resource_contract"]["maximum_total_threads"]),
        },
        "access_receipt": {
            "historical_truth_rows_read_after_prediction_commitment": int(len(scored)),
            "official_interface_rows_read": 0,
            "query_support_rows_read": 0,
            "csv_output_count": 0,
            "submission_generated": False,
            "upload_count": 0,
            "hard_deleted_training_profiles": 0,
        },
        "execution_receipt": {
            "attempts": 1,
            "v1_reexecution": False,
            "result_based_tuning": False,
            "result_based_retry": False,
            "technical_failure_retry": False,
            "aggregate_json_only": True,
            "prediction_values_changed_by_overlay": False,
        },
    }
    return result


def _write_result(
    result: dict[str, Any],
    output: Path,
    p2_dir: Path,
    overlay: dict[str, Any],
) -> Path:
    expected = (ROOT / overlay["artifact_path"]).resolve(strict=False)
    target = output.resolve(strict=False)
    if target != expected or target.suffix.lower() != ".json":
        raise OverlayContractError("--output-json must equal the sealed v2 artifact path")
    if target.is_relative_to(p2_dir.resolve(strict=True)):
        raise OverlayContractError("output cannot be written inside --p2-dir")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    config, overlay = load_config()
    output = args.output_json.resolve(strict=False)
    if output.exists():
        raise FileExistsError(output)
    result = run(config, overlay, args.p2_dir)
    written = _write_result(result, args.output_json, args.p2_dir, overlay)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "decision": result["decision"],
                "output_json": str(written),
                "pooled_delta_rmse": result["metrics"]["pooled"]["delta_rmse"],
                "ci90_high": result["dependence_aware_bootstrap"]["ci90_high"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "official_interface_rows_read": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
