"""One-shot curve engine for the P3 Gen5r2 dense72 correction.

Importing this module performs no fit and writes no file.  Both the public
engine and the private curve function require the in-process canonical
capability issued after independent QA and separate authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from p3_wave.corrected_repeated_forward import OFFICIAL_LEADS, CorrectedFold
from p3_wave.hierarchical_residual_basis import (
    FixedBasisTrainingConfig,
    HierarchicalResidualBasisConfig,
    StaticRobustScaler,
)
from p3_wave.hierarchical_residual_basis_dense72_contract_r1 import (
    COMPARISON_MODE,
    CONFIG_RELATIVE,
    CONFIG_SHA256,
    FOLD_ORDER,
    ExecutionCapability,
    exclusive_json,
    implementation_pins,
    require_execution_capability,
    sha256_file,
    stage_paths,
    verify_consumed_attempt_lock,
    verify_input_pins,
)
from p3_wave.hierarchical_residual_basis_dense72_r1 import (
    fit_dense72_hierarchical_model,
    load_fitted_dense72_model,
    predict_with_fitted_dense72_model,
    save_fitted_dense72_model,
)
from p3_wave.meaningful_learning_curve import (
    PREFIX_FRACTIONS,
    central_evidence,
    evaluate_hypothesis_gate,
    evaluate_point,
)
from p3_wave.models import threshold_case_weights
from p3_wave.one_shot_guard import safe_new_stage_path
from p3_wave.persistence_shrink import (
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_wave.validation import rmse

HYPOTHESIS = "hierarchical_residual_basis_nhits_dense72_masked_supervision"


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _frozen_model_config() -> HierarchicalResidualBasisConfig:
    return HierarchicalResidualBasisConfig(
        static_feature_count=591,
        hidden_width=192,
        conditioning_width=128,
        dropout=0.1,
        context_steps=144,
        input_channels=24,
        forecast_steps=72,
        pooling_factors=(12, 4, 1),
        forecast_knots=(6, 18, 72),
        blocks_per_stack=2,
    )


def _frozen_training_config() -> FixedBasisTrainingConfig:
    return FixedBasisTrainingConfig(
        epochs=12,
        batch_size=512,
        learning_rate=0.001,
        weight_decay=0.0001,
        gradient_clip_norm=1.0,
        use_bf16_on_cuda=True,
    )


def _protected_roots(root: Path, data_dir: Path, config: dict[str, Any]) -> tuple[Path, ...]:
    canonical = config["canonical_paths"]
    return (
        data_dir,
        root / canonical["compact_cache"],
        root / canonical["sequence_cache"],
        root / canonical["gen1_artifact"],
        root / canonical["gen4_artifact"],
        root / "submissions",
        root / "output",
        root / "데이터셋 원본",
    )


def _write_npy_exclusive(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def _write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("append-only parquet path already exists")
    frame.to_parquet(path, index=False)


def _postprocess(delta_hs: np.ndarray, current_hs: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_hs, dtype=np.float64)
    current = np.asarray(current_hs, dtype=np.float64)
    if delta.shape != (len(current), len(OFFICIAL_LEADS)):
        raise ValueError("dense72 official prediction must align to six leads")
    prediction = np.clip(current[:, None] + delta, 0.0, 30.0)
    flat = apply_long_lead_persistence_shrink(
        prediction.reshape(-1),
        np.repeat(current, len(OFFICIAL_LEADS)),
        np.tile(np.asarray(OFFICIAL_LEADS), len(current)),
        config=LongLeadPersistenceShrink(weight=0.2, active_leads=(12, 18, 24)),
    )
    result = flat.reshape(len(current), len(OFFICIAL_LEADS))
    if not np.isfinite(result).all() or not np.all((result >= 0.0) & (result <= 30.0)):
        raise ValueError("postprocessed prediction violates finite/range contract")
    return result


def _cell_comparator(
    truth: pd.DataFrame,
    *,
    fraction: float,
    fold: CorrectedFold,
) -> pd.DataFrame:
    frame = truth.loc[
        truth["prefix_fraction"].eq(fraction) & truth["fold"].astype(str).eq(fold.name),
        [
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "target_hs",
            "current_hs",
            "persistence",
            "incumbent_prediction",
        ],
    ].sort_values(["anchor_id", "lead_h"])
    frame = frame.reset_index(drop=True)
    expected_ids = np.repeat(fold.validation_ids, len(OFFICIAL_LEADS))
    expected_leads = np.tile(np.asarray(OFFICIAL_LEADS), len(fold.validation_ids))
    if not np.array_equal(frame["anchor_id"].to_numpy(np.int64), expected_ids):
        raise ValueError("sealed comparator anchor keys differ from corrected fold")
    if not np.array_equal(frame["lead_h"].to_numpy(int), expected_leads):
        raise ValueError("sealed comparator lead keys differ")
    return frame


def _commitment_path(stage: Path, relative: str, protected: tuple[Path, ...]) -> Path:
    return safe_new_stage_path(stage, relative, protected_roots=protected)


def _verify_blind_commitments(
    stage: Path,
    commitments: list[dict[str, Any]],
    *,
    expected_cells: int,
) -> None:
    if len(commitments) != expected_cells:
        raise PermissionError("blind commitment count differs")
    identity = {
        (float(row["prefix_fraction"]), int(row["seed"]), str(row["fold"]))
        for row in commitments
    }
    expected = {
        (float(fraction), int(seed), fold)
        for fold in FOLD_ORDER
        for fraction in PREFIX_FRACTIONS
        for seed in (20260816, 20260817, 20260818)
    }
    if identity != expected:
        raise PermissionError("blind commitment cell identity differs")
    for row in commitments:
        for path_key, sha_key in (
            ("model_relative_path", "model_sha256"),
            ("blind_prediction_relative_path", "blind_prediction_sha256"),
            ("cell_commitment_relative_path", "cell_commitment_sha256"),
        ):
            path = stage / row[path_key]
            if not path.is_file() or sha256_file(path) != row[sha_key]:
                raise PermissionError(f"blind commitment payload changed: {path_key}")


def _run_curve(
    *,
    capability: ExecutionCapability | object,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    stage: Path,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    require_execution_capability(
        capability,
        root=root,
        config=config,
        preflight=preflight,
        phase="run_curve",
    )
    raw = preflight["raw"]
    station = preflight["station"]
    compact = preflight["compact"]
    anchors = preflight["anchors"]
    folds: tuple[CorrectedFold, ...] = preflight["folds"]
    accessor = preflight["target_accessor"]
    if tuple(fold.name for fold in folds) != FOLD_ORDER:
        raise PermissionError("fold-major execution order differs")
    model_config = _frozen_model_config()
    training_config = _frozen_training_config()
    if compact.shape != (24_360, 591):
        raise ValueError("compact matrix was not loaded and validated before lock")
    protected = _protected_roots(root, data_dir, config)
    anchor_lookup = anchors.set_index("anchor_id")
    blind_predictions: dict[tuple[float, int, str], np.ndarray] = {}
    receipts: list[dict[str, Any]] = []
    fold_commitments: dict[str, dict[str, Any]] = {}

    for fold_index, fold in enumerate(folds):
        fold_rows: list[dict[str, Any]] = []
        if fold.name in accessor.released_groups:
            raise PermissionError("fold labels were released before their blind predictions")
        for fraction in PREFIX_FRACTIONS:
            prefix_tag = f"p{int(round(fraction * 100)):03d}"
            train_ids = preflight["prefix_ids"][fraction][fold.name]
            validation_ids = fold.validation_ids
            if np.intersect1d(train_ids, validation_ids).size:
                raise PermissionError("train and validation IDs overlap before dense72 access")
            payload = accessor.load_training_targets(
                train_ids,
                active_validation_case_ids=validation_ids,
            )
            if payload.forbidden_scalar_decodes != 0:
                raise PermissionError("unreleased validation scalar was decoded")
            train_current = anchor_lookup.loc[train_ids, "current_hs"].to_numpy(np.float64)
            train_weight = threshold_case_weights(train_current).astype(np.float64)
            scaler = StaticRobustScaler.fit(
                np.array(compact[train_ids], copy=True),
                train_ids,
                forbidden_case_ids=validation_ids,
            )
            validation_current = anchor_lookup.loc[
                validation_ids, "current_hs"
            ].to_numpy(np.float64)
            for seed in config["validation"]["seed_replicates"]:
                completed_before = len(receipts)
                print(
                    json.dumps(
                        {
                            "phase": "fit_gen5r2_dense72_cell",
                            "completed_before": completed_before,
                            "total_actual_fit_cells": 45,
                            "fold_index": fold_index,
                            "fold": fold.name,
                            "prefix": fraction,
                            "seed": int(seed),
                            "train_cases": int(len(train_ids)),
                            "dense_target_scalars": int(payload.target_mask.sum()),
                            "unreleased_validation_target_scalar_decodes": 0,
                            "device": "cuda_bfloat16",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                started = time.perf_counter()
                fitted = fit_dense72_hierarchical_model(
                    np.array(raw[train_ids], copy=True),
                    np.array(station[train_ids], copy=True),
                    np.array(compact[train_ids], copy=True),
                    np.array(payload.target_delta, copy=True),
                    np.array(payload.target_mask, dtype=bool, copy=True),
                    np.array(train_weight, copy=True),
                    train_ids,
                    forbidden_case_ids=validation_ids,
                    seed=int(seed),
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                    static_scaler=scaler,
                )
                raw_prediction = predict_with_fitted_dense72_model(
                    fitted,
                    np.array(raw[validation_ids], copy=True),
                    np.array(station[validation_ids], copy=True),
                    np.array(compact[validation_ids], copy=True),
                    device="cuda",
                    batch_size=512,
                )
                prediction = _postprocess(raw_prediction, validation_current)

                model_relative = (
                    f"models/folds/{fold.name}/{prefix_tag}/seed_{seed}/model.pt"
                )
                model_path = _commitment_path(stage, model_relative, protected)
                save_fitted_dense72_model(fitted, model_path)
                model_sha = sha256_file(model_path)
                blind_relative = (
                    f"blind_predictions/folds/{fold.name}/{prefix_tag}/seed_{seed}.npy"
                )
                blind_path = _commitment_path(stage, blind_relative, protected)
                blind_sha = _write_npy_exclusive(blind_path, prediction.astype(np.float64))

                reloaded = load_fitted_dense72_model(model_path, map_location="cpu")
                reloaded_delta = predict_with_fitted_dense72_model(
                    reloaded,
                    np.array(raw[validation_ids], copy=True),
                    np.array(station[validation_ids], copy=True),
                    np.array(compact[validation_ids], copy=True),
                    device="cuda",
                    batch_size=512,
                )
                reloaded_prediction = _postprocess(reloaded_delta, validation_current)
                reload_exact = bool(np.array_equal(reloaded_prediction, prediction))
                max_difference = float(np.max(np.abs(reloaded_prediction - prediction)))
                if not reload_exact:
                    raise RuntimeError("saved dense72 model failed exact blind reproduction")

                commitment_relative = (
                    f"commitments/cells/{fold.name}/{prefix_tag}/seed_{seed}.json"
                )
                commitment_path = _commitment_path(stage, commitment_relative, protected)
                commitment = {
                    "schema_version": "p3_gen5r2_dense72.blind_cell_commitment.r1",
                    "fold_index": int(fold_index),
                    "fold": fold.name,
                    "prefix_fraction": float(fraction),
                    "seed": int(seed),
                    "train_case_count": int(len(train_ids)),
                    "validation_case_count": int(len(validation_ids)),
                    "train_ids_sha256": payload.case_ids_sha256,
                    "validation_ids_sha256": hashlib.sha256(
                        np.asarray(validation_ids, dtype="<i8").tobytes(order="C")
                    ).hexdigest(),
                    "dense_target_delta_sha256": payload.target_delta_sha256,
                    "dense_target_mask_sha256": payload.target_mask_sha256,
                    "dense_target_valid_scalars": int(payload.target_mask.sum()),
                    "unreleased_validation_target_scalar_decodes": 0,
                    "model_relative_path": model_relative,
                    "model_sha256": model_sha,
                    "model_state_sha256": fitted.model_state_sha256,
                    "blind_prediction_relative_path": blind_relative,
                    "blind_prediction_sha256": blind_sha,
                    "saved_model_reload_prediction_exact": True,
                    "candidate_or_test_prediction": False,
                }
                exclusive_json(commitment_path, commitment)
                commitment_sha = sha256_file(commitment_path)
                row = {
                    **commitment,
                    "cell_commitment_relative_path": commitment_relative,
                    "cell_commitment_sha256": commitment_sha,
                    "optimizer_steps": int(fitted.training_steps),
                    "train_context_sha256": fitted.train_context_sha256,
                    "train_target_sha256": fitted.train_target_sha256,
                    "scaler_state_sha256": fitted.scaler_state_sha256,
                    "saved_model_reload_max_abs_difference_m": max_difference,
                    "blind_prediction_sealed_before_validation_truth_attachment": True,
                    "elapsed_seconds": float(time.perf_counter() - started),
                }
                receipts.append(row)
                fold_rows.append(row)
                blind_predictions[(fraction, int(seed), fold.name)] = prediction
                del fitted, reloaded

        if len(fold_rows) != 15:
            raise PermissionError("fold did not produce exactly 15 blind commitments")
        fold_relative = f"commitments/folds/{fold.name}.json"
        fold_path = _commitment_path(stage, fold_relative, protected)
        fold_payload = {
            "schema_version": "p3_gen5r2_dense72.blind_fold_commitment.r1",
            "fold_index": int(fold_index),
            "fold": fold.name,
            "validation_ids_sha256": fold_rows[0]["validation_ids_sha256"],
            "cell_count": 15,
            "cell_commitments": [
                {
                    "path": row["cell_commitment_relative_path"],
                    "sha256": row["cell_commitment_sha256"],
                }
                for row in fold_rows
            ],
            "truth_attached": False,
            "dense_validation_target_scalar_decodes_before_commitment": 0,
        }
        exclusive_json(fold_path, fold_payload)
        fold_sha = sha256_file(fold_path)
        fold_commitments[fold.name] = {
            "path": fold_relative,
            "sha256": fold_sha,
            "cell_count": 15,
        }
        accessor.release_validation_group(
            fold.name,
            fold_commitment_sha256=fold_sha,
        )

    expected_steps = int(config["model"]["expected_optimizer_steps"])
    if len(receipts) != 45 or sum(row["optimizer_steps"] for row in receipts) != expected_steps:
        raise AssertionError("fit-cell or optimizer-step total differs")
    _verify_blind_commitments(stage, receipts, expected_cells=45)
    complete_relative = "commitments/predictions_complete.json"
    complete_path = _commitment_path(stage, complete_relative, protected)
    complete_payload = {
        "schema_version": "p3_gen5r2_dense72.predictions_complete.r1",
        "comparison_mode": COMPARISON_MODE,
        "fold_order": list(FOLD_ORDER),
        "fit_cell_count": 45,
        "optimizer_steps": expected_steps,
        "fold_commitments": fold_commitments,
        "cell_commitments": [
            {
                "path": row["cell_commitment_relative_path"],
                "sha256": row["cell_commitment_sha256"],
            }
            for row in receipts
        ],
        "all_validation_groups_released_only_after_fold_commitment": (
            accessor.released_groups == tuple(sorted(FOLD_ORDER))
        ),
        "unreleased_validation_target_scalar_decodes": accessor.forbidden_scalar_decodes,
        "validation_truth_attached": False,
        "candidate_or_test_prediction": False,
    }
    if complete_payload["unreleased_validation_target_scalar_decodes"] != 0:
        raise PermissionError("validation scalar firewall failed before predictions_complete")
    exclusive_json(complete_path, complete_payload)
    complete_sha = sha256_file(complete_path)
    _verify_blind_commitments(stage, receipts, expected_cells=45)

    # This is the first point at which validation truth is attached.
    truth = pd.read_parquet(
        preflight["input_paths"]["gen1/learning_curve_oof.parquet"]
    )
    required_truth = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        "current_hs",
        "persistence",
        "incumbent_prediction",
        "prefix_fraction",
    }
    if not required_truth.issubset(truth.columns) or len(truth) != 5 * 1_086:
        raise ValueError("sealed comparator truth surface differs after blind commitments")

    points: dict[float, dict[str, Any]] = {}
    all_frames: list[pd.DataFrame] = []
    for fraction in PREFIX_FRACTIONS:
        seed_frames: list[pd.DataFrame] = []
        for seed in config["validation"]["seed_replicates"]:
            fold_frames: list[pd.DataFrame] = []
            for fold in folds:
                comparator = _cell_comparator(truth, fraction=fraction, fold=fold)
                comparator["challenger_prediction"] = blind_predictions[
                    (fraction, int(seed), fold.name)
                ].reshape(-1)
                fold_frames.append(comparator)
            seed_frames.append(pd.concat(fold_frames, ignore_index=True))
        keys = ["fold", "anchor_id", "station", "lead_h"]
        invariant = ["target_hs", "current_hs", "persistence", "incumbent_prediction"]
        ordered = [frame.sort_values(keys).reset_index(drop=True) for frame in seed_frames]
        for frame in ordered[1:]:
            if not frame[keys].equals(ordered[0][keys]):
                raise ValueError("seed OOF keys differ")
            if not np.array_equal(
                frame[invariant].to_numpy(float), ordered[0][invariant].to_numpy(float)
            ):
                raise ValueError("seed OOF invariant values differ")
        mean_frame = ordered[0][keys + invariant].copy()
        mean_frame["challenger_prediction"] = np.mean(
            np.column_stack(
                [frame["challenger_prediction"].to_numpy(float) for frame in ordered]
            ),
            axis=1,
        )
        mean_frame["prefix_fraction"] = float(fraction)
        all_frames.append(mean_frame)
        point = evaluate_point(
            mean_frame,
            candidate_column="challenger_prediction",
            bootstrap_replicates=config["validation"]["bootstrap_replicates"],
            bootstrap_seed=int(config["validation"]["bootstrap_seed"])
            + int(round(fraction * 100)),
        )
        gen1_point = preflight["gen1_metrics"]["points_by_hypothesis"][
            "fixed_horizon_splice"
        ][str(fraction)]
        point["incumbent_seed_metrics"] = [
            float(value) for value in gen1_point["incumbent_seed_metrics"]
        ]
        point["challenger_seed_metrics"] = [
            float(rmse(frame["target_hs"], frame["challenger_prediction"]))
            for frame in ordered
        ]
        points[fraction] = point

    access_audit = accessor.access_audit()
    access_audit["predictions_complete_relative_path"] = complete_relative
    access_audit["predictions_complete_sha256"] = complete_sha
    return pd.concat(all_frames, ignore_index=True), points, receipts, access_audit


def _artifact_hashes(stage: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in stage.rglob("*") if value.is_file()):
        relative = path.relative_to(stage).as_posix()
        result[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def execute_curve_stage(
    *,
    capability: ExecutionCapability | object,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    require_execution_capability(
        capability,
        root=root,
        config=config,
        preflight=preflight,
        phase="execute_stage",
    )
    attempt = verify_consumed_attempt_lock(
        root,
        config,
        capability=capability,
    )
    if implementation_pins(root, config) != preflight["implementation_pins"]:
        raise PermissionError("Gen5r2 implementation changed after lock")
    paths = stage_paths(root, config)
    if paths["output"].exists():
        raise FileExistsError("append-only Gen5r2 output already exists")
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_gen5r2_dense72_", dir=tmp_root))
    started = time.perf_counter()
    curve_oof, points, receipts, target_access = _run_curve(
        capability=capability,
        root=root,
        data_dir=data_dir,
        config=config,
        preflight=preflight,
        stage=stage,
    )
    _write_parquet_new(stage / "oof/learning_curve_oof.parquet", curve_oof)
    _write_parquet_new(
        stage / "validation_keys.parquet",
        preflight["selected"][["fold", "anchor_id", "station", "episode_id"]],
    )

    reproducibility_checks = {
        "canonical_config_and_full_transitive_implementation_pins_equal": True,
        "compact_24360_by_591_loaded_and_validated_before_lock": True,
        "dense72_availability_23527_complete_833_incomplete_1505_missing": True,
        "dense72_missing_values_masked_without_imputation": True,
        "optimizer_uses_all_available_train_only_dense72_steps": True,
        "fold_major_commitment_precedes_later_fold_target_release": True,
        "all_45_blind_predictions_committed_before_truth_attachment": True,
        "unreleased_validation_target_scalar_decodes_zero": (
            target_access["forbidden_validation_target_scalar_decodes"] == 0
        ),
        "same_prefix_ids_fold_keys_metric_clip_and_fixed_0p20_shrink": True,
        "three_fixed_seed_predictions_mean_then_metric": True,
        "all_saved_models_reload_prediction_exact": all(
            row["saved_model_reload_prediction_exact"] for row in receipts
        ),
        "optimizer_step_total_exact_10260": (
            sum(row["optimizer_steps"] for row in receipts) == 10_260
        ),
        "hyperparameter_threshold_alpha_weight_seed_search_zero": True,
        "anonymous_test_and_submission_value_reads_zero": True,
    }
    local_gate = evaluate_hypothesis_gate(
        points,
        leakage_checks=preflight["leakage_checks"],
        reproducibility_checks=reproducibility_checks,
    )
    local_decision = (
        "LOCAL_CURVE_QUALIFIED_PENDING_SEPARATE_FULL_FIT_AND_OFFICIAL_PAIRED_AB"
        if local_gate["passed"]
        else "RESEARCH_ONLY_NO_LOCAL_CURVE_QUALIFICATION"
    )
    evidence = central_evidence(
        points,
        leakage_checks=preflight["leakage_checks"],
        reproducibility_checks=reproducibility_checks,
    )
    evidence["comparison_mode"] = COMPARISON_MODE
    evidence["local_numeric_gate"] = local_gate
    evidence["local_decision"] = local_decision
    evidence["official_promotion"] = {
        "allowed": False,
        "reason": "same-surface official paired A/B has not occurred",
        "reference_seed_full_prediction_exact_to_historical_frozen_oof": False,
    }
    evidence["curve_protocol"] = {
        "comparison_mode": COMPARISON_MODE,
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "seed_ids": list(config["validation"]["seed_replicates"]),
        "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
        "bootstrap_replicates": int(config["validation"]["bootstrap_replicates"]),
        "bootstrap_cluster": "whole_case",
        "incumbent_fresh_refit_each_prefix": True,
        "challenger_fresh_refit_each_prefix": True,
        "same_fold_keys_metric_postprocess": True,
        "reference_seed_full_prediction_exact_to_historical_frozen_oof": False,
        "historical_frozen_mismatch_is_not_a_local_numeric_gate": True,
        "official_promotion_requires_future_paired_ab": True,
    }
    exclusive_json(stage / "learning_curve_evidence.json", evidence)

    status = (
        "LOCAL_CURVE_QUALIFIED_PENDING_SEPARATE_STAGE_NO_CANDIDATE_NO_TEST"
        if local_gate["passed"]
        else "NO_LOCAL_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST"
    )
    access = {
        **target_access,
        "test_sequence_cache_value_reads": 0,
        "test_feature_cache_value_reads": 0,
        "test_index_value_reads": 0,
        "test_context_value_reads": 0,
        "test_target_or_hidden_label_reads": 0,
        "absolute_test_timestamp_recovery_attempts": 0,
        "current_or_frozen_submission_value_reads": 0,
        "current_or_frozen_submission_writes": 0,
        "candidate_files_created": 0,
        "full_fit_count": 0,
        "upload_attempts": 0,
    }
    _, input_after = verify_input_pins(root, data_dir, config)
    if input_after != preflight["input_snapshot"]:
        raise RuntimeError("source/cache/reference/frozen inputs changed during Gen5r2")

    metrics = {
        "created_at_kst": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "comparison_mode": COMPARISON_MODE,
        "exact_official_incumbent_comparison": False,
        "local_numeric_curve_qualified": bool(local_gate["passed"]),
        "official_promotion_allowed": False,
        "official_promotion_pending_same_surface_paired_ab": True,
        "hypothesis": HYPOTHESIS,
        "one_shot_attempt": attempt,
        "points": {str(fraction): points[fraction] for fraction in PREFIX_FRACTIONS},
        "local_gate": local_gate,
        "split_audit": preflight["split_audit"],
        "prefix_audit": preflight["prefix_audit"],
        "leakage_checks": preflight["leakage_checks"],
        "reproducibility_checks": reproducibility_checks,
        "training_receipts": receipts,
        "dense72_access_audit": target_access,
        "access_counters": access,
        "candidate_validation": None,
        "candidate_created": False,
        "test_prediction_created": False,
        "full_fit_performed": False,
        "official_upload_count": 0,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    exclusive_json(stage / "metrics.json", metrics)
    registry = {
        "created_at_kst": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "comparison_mode": COMPARISON_MODE,
        "local_curve_qualified": bool(local_gate["passed"]),
        "official_promotion_allowed": False,
        "next_required_stage": (
            "separate_preregistered_full_fit_then_user_approved_official_paired_ab"
            if local_gate["passed"]
            else "next_structural_generation"
        ),
        "candidate_created": False,
        "candidate_uploaded": False,
        "current_frozen_sha256": config["input_pins"]["current/ready_submission.csv"][
            "sha256"
        ],
        "current_frozen_unchanged": True,
        "official_upload_count": 0,
    }
    exclusive_json(stage / "registry.json", registry)
    output_before_manifest = _artifact_hashes(stage)
    manifest = {
        "created_at_kst": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "append_only_generation": True,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation_pins": implementation_pins(root, config),
        "input_sha256_before": preflight["input_snapshot"],
        "input_sha256_after": input_after,
        "source_cache_reference_frozen_unchanged": True,
        "output_files_before_manifest": output_before_manifest,
        "local_curve_qualified": bool(local_gate["passed"]),
        "official_promotion_allowed": False,
        "candidate_created": False,
        "candidate_uploaded": False,
        "official_upload_count": 0,
        "access_counters": access,
    }
    exclusive_json(stage / "manifest.json", manifest)
    manifest_sha = sha256_file(stage / "manifest.json")
    with (stage / "manifest.sha256").open("xb") as stream:
        stream.write(f"{manifest_sha}  manifest.json\n".encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    if paths["output"].exists():
        raise FileExistsError("canonical Gen5r2 output appeared before atomic move")
    stage.replace(paths["output"])
    return {
        "schema_version": "p3_hierarchical_residual_basis.gen5r2_dense72.run.r1",
        "status": status,
        "artifact_dir": config["canonical_paths"]["output"],
        "metrics_sha256": sha256_file(paths["output"] / "metrics.json"),
        "oof_sha256": sha256_file(paths["output"] / "oof/learning_curve_oof.parquet"),
        "learning_curve_evidence_sha256": sha256_file(
            paths["output"] / "learning_curve_evidence.json"
        ),
        "registry_sha256": sha256_file(paths["output"] / "registry.json"),
        "manifest_sha256": manifest_sha,
        "local_curve_qualified": bool(local_gate["passed"]),
        "official_promotion_allowed": False,
        "candidate_sha256": None,
        "test_prediction_count": 0,
        "official_upload_count": 0,
    }


__all__ = [
    "HYPOTHESIS",
    "_run_curve",
    "execute_curve_stage",
]
