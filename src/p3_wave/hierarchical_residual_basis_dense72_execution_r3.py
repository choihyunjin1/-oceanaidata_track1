"""Single-use Gen5r3 curve engine; import is side-effect free."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import p3_wave.hierarchical_residual_basis_dense72_contract_r1 as predecessor_guard
from p3_wave.corrected_repeated_forward import CorrectedFold
from p3_wave.hierarchical_residual_basis import StaticRobustScaler
from p3_wave.hierarchical_residual_basis_dense72_contract_r3 import (
    COMPARISON_MODE,
    CONFIG_RELATIVE,
    CONFIG_SHA256,
    FOLD_ORDER,
    ExecutionCapability,
    authorize_curve_phase,
    begin_execution_stage,
    exclusive_json,
    implementation_pins,
    require_curve_capability,
    sha256_file,
    stage_paths,
)
from p3_wave.hierarchical_residual_basis_dense72_execution_r1 import (
    HYPOTHESIS,
    _artifact_hashes,
    _cell_comparator,
    _commitment_path,
    _frozen_model_config,
    _frozen_training_config,
    _postprocess,
    _protected_roots,
    _verify_blind_commitments,
    _write_npy_exclusive,
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
from p3_wave.validation import rmse


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _write_all_exclusive(
    path: Path,
    payload: bytes,
    *,
    write_fn: Callable[[int, memoryview], int] | None = None,
) -> None:
    """Create one file exclusively and tolerate short/interrupted OS writes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    writer = write_fn or os.write
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            try:
                written = int(writer(descriptor, view[offset:]))
            except InterruptedError:
                continue
            if written <= 0 or written > len(view) - offset:
                raise OSError("exclusive write made invalid progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_parquet_exclusive(path: Path, frame: pd.DataFrame) -> str:
    """Serialize completely in memory before the one O_EXCL filesystem mutation."""

    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    if not payload:
        raise ValueError("parquet serialization produced no bytes")
    _write_all_exclusive(path, payload)
    return sha256_file(path)


def _run_curve(
    *,
    capability: ExecutionCapability | object,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    stage: Path,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    require_curve_capability(
        capability,
        root=root,
        config=config,
        preflight=preflight,
        temporary_stage=stage,
    )
    raw = preflight["raw"]
    station = preflight["station"]
    compact = preflight["compact"]
    folds: tuple[CorrectedFold, ...] = preflight["folds"]
    accessor = preflight["target_accessor"]
    current_accessor = preflight["current_hs_accessor"]
    predecessor = preflight["predecessor"]
    current_index = int(preflight["current_hs_feature_index"])
    if tuple(fold.name for fold in folds) != FOLD_ORDER or compact.shape != (24_360, 591):
        raise PermissionError("r3 operational fold or compact surface changed")
    protected = _protected_roots(root, data_dir, predecessor)
    model_config = _frozen_model_config()
    training_config = _frozen_training_config()
    blind_predictions: dict[tuple[float, int, str], np.ndarray] = {}
    receipts: list[dict[str, Any]] = []
    fold_commitments: dict[str, dict[str, Any]] = {}
    raw_fold_commitments: dict[str, dict[str, Any]] = {}
    released_validation_current: dict[str, np.ndarray] = {}
    fitted_cell_count = 0

    for fold_index, fold in enumerate(folds):
        raw_fold_rows: list[dict[str, Any]] = []
        raw_records: list[dict[str, Any]] = []
        if (
            fold.name in accessor.released_groups
            or fold.name in current_accessor.released_groups
            or current_accessor.validation_group_process_scalar_decodes(fold.name) != 0
        ):
            raise PermissionError("validation group was released or decoded before raw cells")
        for fraction in PREFIX_FRACTIONS:
            prefix_tag = f"p{int(round(fraction * 100)):03d}"
            train_ids = preflight["prefix_ids"][fraction][fold.name]
            validation_ids = fold.validation_ids
            current_accessor.assert_training_target_current_isolation(train_ids)
            payload = accessor.load_training_targets(
                train_ids,
                active_validation_case_ids=validation_ids,
            )
            if payload.forbidden_scalar_decodes != 0 or not np.all(payload.current_hs == 0.0):
                raise PermissionError("selective r3 target accessor contract changed")
            train_current_exact = current_accessor.load_training_current_hs(
                train_ids,
                active_validation_case_ids=validation_ids,
            )
            if (
                current_accessor.forbidden_scalar_decodes != 0
                or current_accessor.validation_group_process_scalar_decodes(fold.name)
                != 0
                or train_current_exact.shape != (len(train_ids),)
                or not np.isfinite(train_current_exact).all()
            ):
                raise PermissionError("selective r3 current-hs accessor contract changed")
            train_current_float32 = train_current_exact.astype(np.float32)
            if not np.array_equal(
                train_current_float32, compact[train_ids, current_index]
            ):
                raise PermissionError("selected train current hs differs from input cache")
            train_delta = (
                np.asarray(payload.target_delta, dtype=np.float32)
                - train_current_float32[:, None]
            )
            train_delta[~payload.target_mask] = 0.0
            train_weight = threshold_case_weights(train_current_exact).astype(np.float64)
            scaler = StaticRobustScaler.fit(
                np.array(compact[train_ids], copy=True),
                train_ids,
                forbidden_case_ids=validation_ids,
            )
            for seed in predecessor["validation"]["seed_replicates"]:
                print(
                    json.dumps(
                        {
                            "phase": "fit_gen5r3_dense72_cell",
                            "completed_before": fitted_cell_count,
                            "total_actual_fit_cells": 45,
                            "fold": fold.name,
                            "prefix": fraction,
                            "seed": int(seed),
                            "train_cases": len(train_ids),
                            "train_future_hs_float_decodes": int(payload.target_mask.sum()),
                            "train_current_hs_decode_mode": "selective_train_ids_only",
                            "active_validation_current_hs_float_decodes": 0,
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
                    np.array(train_delta, copy=True),
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
                model_relative = (
                    f"models/folds/{fold.name}/{prefix_tag}/seed_{seed}/model.pt"
                )
                model_path = _commitment_path(stage, model_relative, protected)
                save_fitted_dense72_model(fitted, model_path)
                model_sha = sha256_file(model_path)
                raw_relative = (
                    f"blind_raw_deltas/folds/{fold.name}/{prefix_tag}/seed_{seed}.npy"
                )
                raw_path = _commitment_path(stage, raw_relative, protected)
                raw_sha = _write_npy_exclusive(raw_path, raw_prediction.astype(np.float32))
                reloaded = load_fitted_dense72_model(model_path, map_location="cpu")
                reloaded_delta = predict_with_fitted_dense72_model(
                    reloaded,
                    np.array(raw[validation_ids], copy=True),
                    np.array(station[validation_ids], copy=True),
                    np.array(compact[validation_ids], copy=True),
                    device="cuda",
                    batch_size=512,
                )
                sealed_raw = np.load(raw_path, allow_pickle=False)
                if (
                    not np.array_equal(reloaded_delta, raw_prediction)
                    or not np.array_equal(sealed_raw, raw_prediction)
                    or current_accessor.validation_group_process_scalar_decodes(fold.name)
                    != 0
                ):
                    raise RuntimeError(
                        "saved r3 model or raw-delta seal failed exact blind reproduction"
                    )
                raw_commitment_relative = (
                    f"commitments/raw_cells/{fold.name}/{prefix_tag}/seed_{seed}.json"
                )
                raw_commitment_path = _commitment_path(
                    stage, raw_commitment_relative, protected
                )
                raw_commitment = {
                    "schema_version": "p3_gen5r3_dense72.raw_delta_cell_commitment.r1",
                    "fold_index": fold_index,
                    "fold": fold.name,
                    "prefix_fraction": float(fraction),
                    "seed": int(seed),
                    "train_case_count": len(train_ids),
                    "validation_case_count": len(validation_ids),
                    "train_ids_sha256": payload.case_ids_sha256,
                    "validation_ids_sha256": hashlib.sha256(
                        np.asarray(validation_ids, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "dense_target_mask_sha256": payload.target_mask_sha256,
                    "dense_target_valid_scalars": int(payload.target_mask.sum()),
                    "train_target_delta_sha256": fitted.train_target_sha256,
                    "train_current_hs_float64_sha256": hashlib.sha256(
                        np.asarray(train_current_exact, dtype="<f8").tobytes()
                    ).hexdigest(),
                    "active_validation_current_train_wave_hs_scalar_decodes": 0,
                    "unreleased_validation_future_hs_scalar_decodes": 0,
                    "model_relative_path": model_relative,
                    "model_sha256": model_sha,
                    "model_state_sha256": fitted.model_state_sha256,
                    "raw_delta_relative_path": raw_relative,
                    "raw_delta_sha256": raw_sha,
                    "saved_model_reload_raw_delta_exact": True,
                    "candidate_or_test_prediction": False,
                }
                exclusive_json(raw_commitment_path, raw_commitment)
                row = {
                    **raw_commitment,
                    "raw_cell_commitment_relative_path": raw_commitment_relative,
                    "raw_cell_commitment_sha256": sha256_file(raw_commitment_path),
                    "optimizer_steps": fitted.training_steps,
                    "train_context_sha256": fitted.train_context_sha256,
                    "scaler_state_sha256": fitted.scaler_state_sha256,
                    "saved_model_reload_raw_delta_max_abs_difference_m": 0.0,
                    "raw_delta_sealed_before_validation_current_source_decode": True,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                raw_fold_rows.append(row)
                raw_records.append(row)
                fitted_cell_count += 1
                del fitted, reloaded
        if (
            len(raw_fold_rows) != 15
            or current_accessor.validation_group_process_scalar_decodes(fold.name) != 0
            or current_accessor.forbidden_scalar_decodes != 0
            or accessor.forbidden_scalar_decodes != 0
        ):
            raise PermissionError("r3 fold did not seal 15 target/current-blind raw cells")
        raw_fold_relative = f"commitments/raw_folds/{fold.name}.json"
        raw_fold_path = _commitment_path(stage, raw_fold_relative, protected)
        raw_fold_payload = {
            "schema_version": "p3_gen5r3_dense72.raw_delta_fold_commitment.r1",
            "fold_index": fold_index,
            "fold": fold.name,
            "validation_ids_sha256": raw_fold_rows[0]["validation_ids_sha256"],
            "cell_count": 15,
            "raw_cell_commitments": [
                {
                    "path": row["raw_cell_commitment_relative_path"],
                    "sha256": row["raw_cell_commitment_sha256"],
                }
                for row in raw_fold_rows
            ],
            "truth_attached": False,
            "validation_current_train_wave_hs_scalar_decodes_before_commitment": 0,
            "validation_future_train_wave_hs_scalar_decodes_before_commitment": 0,
            "disclosed_cached_current_hs_model_inputs_allowed": True,
        }
        exclusive_json(raw_fold_path, raw_fold_payload)
        raw_fold_sha = sha256_file(raw_fold_path)
        raw_fold_commitments[fold.name] = {
            "path": raw_fold_relative,
            "sha256": raw_fold_sha,
            "cell_count": 15,
        }
        accessor.release_validation_group(
            fold.name, fold_commitment_sha256=raw_fold_sha
        )
        current_accessor.release_validation_group(
            fold.name, fold_commitment_sha256=raw_fold_sha
        )
        validation_current = current_accessor.load_released_validation_current_hs(
            fold.name
        )
        if (
            validation_current.shape != (len(fold.validation_ids),)
            or not np.isfinite(validation_current).all()
            or current_accessor.validation_group_process_scalar_decodes(fold.name)
            != len(fold.validation_ids)
            or not np.array_equal(
                validation_current.astype(np.float32),
                compact[fold.validation_ids, current_index],
            )
        ):
            raise PermissionError("released validation current-hs payload changed")
        released_validation_current[fold.name] = validation_current.copy()

        fold_rows: list[dict[str, Any]] = []
        for raw_row in raw_records:
            fraction = float(raw_row["prefix_fraction"])
            seed = int(raw_row["seed"])
            prefix_tag = f"p{int(round(fraction * 100)):03d}"
            raw_path = (stage / str(raw_row["raw_delta_relative_path"])).resolve(
                strict=True
            )
            if stage.resolve(strict=True) not in raw_path.parents or sha256_file(
                raw_path
            ) != raw_row["raw_delta_sha256"]:
                raise PermissionError("sealed raw delta changed after fold commitment")
            raw_prediction = np.load(raw_path, allow_pickle=False)
            prediction = _postprocess(raw_prediction, validation_current)
            blind_relative = (
                f"blind_predictions/folds/{fold.name}/{prefix_tag}/seed_{seed}.npy"
            )
            blind_path = _commitment_path(stage, blind_relative, protected)
            blind_sha = _write_npy_exclusive(blind_path, prediction.astype(np.float64))
            commitment_relative = (
                f"commitments/cells/{fold.name}/{prefix_tag}/seed_{seed}.json"
            )
            commitment_path = _commitment_path(stage, commitment_relative, protected)
            commitment = {
                "schema_version": "p3_gen5r3_dense72.blind_cell_commitment.r1",
                "fold_index": fold_index,
                "fold": fold.name,
                "prefix_fraction": fraction,
                "seed": seed,
                "train_case_count": raw_row["train_case_count"],
                "validation_case_count": raw_row["validation_case_count"],
                "train_ids_sha256": raw_row["train_ids_sha256"],
                "validation_ids_sha256": raw_row["validation_ids_sha256"],
                "dense_target_mask_sha256": raw_row["dense_target_mask_sha256"],
                "dense_target_valid_scalars": raw_row["dense_target_valid_scalars"],
                "train_target_delta_sha256": raw_row["train_target_delta_sha256"],
                "raw_fold_commitment_relative_path": raw_fold_relative,
                "raw_fold_commitment_sha256": raw_fold_sha,
                "raw_cell_commitment_relative_path": raw_row[
                    "raw_cell_commitment_relative_path"
                ],
                "raw_cell_commitment_sha256": raw_row[
                    "raw_cell_commitment_sha256"
                ],
                "validation_current_hs_float64_sha256": hashlib.sha256(
                    np.asarray(validation_current, dtype="<f8").tobytes()
                ).hexdigest(),
                "model_relative_path": raw_row["model_relative_path"],
                "model_sha256": raw_row["model_sha256"],
                "model_state_sha256": raw_row["model_state_sha256"],
                "blind_prediction_relative_path": blind_relative,
                "blind_prediction_sha256": blind_sha,
                "saved_model_reload_prediction_exact": True,
                "blind_prediction_sealed_before_validation_truth_attachment": True,
                "candidate_or_test_prediction": False,
            }
            exclusive_json(commitment_path, commitment)
            row = {
                **commitment,
                "cell_commitment_relative_path": commitment_relative,
                "cell_commitment_sha256": sha256_file(commitment_path),
                "optimizer_steps": raw_row["optimizer_steps"],
                "train_context_sha256": raw_row["train_context_sha256"],
                "scaler_state_sha256": raw_row["scaler_state_sha256"],
                "saved_model_reload_max_abs_difference_m": 0.0,
                "elapsed_seconds": raw_row["elapsed_seconds"],
            }
            receipts.append(row)
            fold_rows.append(row)
            blind_predictions[(fraction, seed, fold.name)] = prediction

        if len(fold_rows) != 15:
            raise PermissionError("r3 fold did not seal exactly 15 final blind cells")
        fold_relative = f"commitments/folds/{fold.name}.json"
        fold_path = _commitment_path(stage, fold_relative, protected)
        fold_payload = {
            "schema_version": "p3_gen5r3_dense72.blind_fold_commitment.r1",
            "fold_index": fold_index,
            "fold": fold.name,
            "validation_ids_sha256": fold_rows[0]["validation_ids_sha256"],
            "raw_fold_commitment": {
                "path": raw_fold_relative,
                "sha256": raw_fold_sha,
            },
            "cell_count": 15,
            "cell_commitments": [
                {
                    "path": row["cell_commitment_relative_path"],
                    "sha256": row["cell_commitment_sha256"],
                }
                for row in fold_rows
            ],
            "truth_attached": False,
            "validation_current_source_decode_after_raw_commitment": True,
        }
        exclusive_json(fold_path, fold_payload)
        fold_sha = sha256_file(fold_path)
        fold_commitments[fold.name] = {
            "path": fold_relative,
            "sha256": fold_sha,
            "cell_count": 15,
        }

    expected_steps = int(predecessor["model"]["expected_optimizer_steps"])
    if len(receipts) != 45 or sum(row["optimizer_steps"] for row in receipts) != expected_steps:
        raise AssertionError("r3 fit-cell or optimizer-step accounting changed")
    _verify_blind_commitments(stage, receipts, expected_cells=45)
    complete_relative = "commitments/predictions_complete.json"
    complete_path = _commitment_path(stage, complete_relative, protected)
    complete_payload = {
        "schema_version": "p3_gen5r3_dense72.predictions_complete.r1",
        "comparison_mode": COMPARISON_MODE,
        "fold_order": list(FOLD_ORDER),
        "fit_cell_count": 45,
        "optimizer_steps": expected_steps,
        "raw_fold_commitments": raw_fold_commitments,
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
            and current_accessor.released_groups == tuple(sorted(FOLD_ORDER))
        ),
        "unreleased_validation_train_wave_hs_float_decodes": (
            accessor.forbidden_scalar_decodes
        ),
        "unreleased_validation_current_hs_float_decodes": (
            current_accessor.forbidden_scalar_decodes
        ),
        "raw_delta_fold_commitment_preceded_validation_current_source_decode": True,
        "validation_truth_attached": False,
        "candidate_or_test_prediction": False,
    }
    if (
        complete_payload["unreleased_validation_train_wave_hs_float_decodes"] != 0
        or complete_payload["unreleased_validation_current_hs_float_decodes"] != 0
        or not complete_payload["all_validation_groups_released_only_after_fold_commitment"]
    ):
        raise PermissionError("r3 validation target firewall failed")
    exclusive_json(complete_path, complete_payload)
    complete_sha = sha256_file(complete_path)
    _verify_blind_commitments(stage, receipts, expected_cells=45)

    truth = pd.read_parquet(preflight["input_paths"]["gen1/learning_curve_oof.parquet"])
    if len(truth) != 5 * 1_086 or "target_hs" not in truth:
        raise ValueError("sealed comparator truth surface changed")
    points: dict[float, dict[str, Any]] = {}
    all_frames: list[pd.DataFrame] = []
    for fraction in PREFIX_FRACTIONS:
        seed_frames: list[pd.DataFrame] = []
        for seed in predecessor["validation"]["seed_replicates"]:
            by_fold: list[pd.DataFrame] = []
            for fold in folds:
                comparator = _cell_comparator(truth, fraction=fraction, fold=fold)
                expected_current = np.repeat(
                    released_validation_current[fold.name], 6
                )
                if not np.array_equal(
                    comparator["current_hs"].to_numpy(np.float64), expected_current
                ):
                    raise ValueError("released source current hs differs from sealed comparator")
                comparator["challenger_prediction"] = blind_predictions[
                    (fraction, int(seed), fold.name)
                ].reshape(-1)
                by_fold.append(comparator)
            seed_frames.append(pd.concat(by_fold, ignore_index=True))
        keys = ["fold", "anchor_id", "station", "lead_h"]
        invariant = ["target_hs", "current_hs", "persistence", "incumbent_prediction"]
        ordered = [frame.sort_values(keys).reset_index(drop=True) for frame in seed_frames]
        for frame in ordered[1:]:
            if not frame[keys].equals(ordered[0][keys]) or not np.array_equal(
                frame[invariant].to_numpy(float), ordered[0][invariant].to_numpy(float)
            ):
                raise ValueError("r3 seed OOF alignment changed")
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
            bootstrap_replicates=predecessor["validation"]["bootstrap_replicates"],
            bootstrap_seed=int(predecessor["validation"]["bootstrap_seed"])
            + int(round(fraction * 100)),
        )
        gen1_point = preflight["gen1_metrics"]["points_by_hypothesis"][
            "fixed_horizon_splice"
        ][str(fraction)]
        point["incumbent_seed_metrics"] = list(gen1_point["incumbent_seed_metrics"])
        point["challenger_seed_metrics"] = [
            float(rmse(frame["target_hs"], frame["challenger_prediction"]))
            for frame in ordered
        ]
        points[fraction] = point
    access = accessor.access_audit()
    current_access = current_accessor.access_audit()
    access["current_hs_accessor"] = current_access
    access["forbidden_validation_current_hs_scalar_decodes"] = current_accessor.forbidden_scalar_decodes
    access["raw_fold_commitments"] = raw_fold_commitments
    access["predictions_complete_relative_path"] = complete_relative
    access["predictions_complete_sha256"] = complete_sha
    access["process_preflight_train_wave_hs_float_decodes"] = 0
    return pd.concat(all_frames, ignore_index=True), points, receipts, access


def execute_curve_stage(
    *,
    capability: ExecutionCapability | object,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    begin_execution_stage(
        capability,
        root=root,
        config=config,
        preflight=preflight,
    )
    paths = stage_paths(root, config)
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_gen5r3_dense72_", dir=tmp_root))
    authorize_curve_phase(
        capability,
        root=root,
        config=config,
        preflight=preflight,
        temporary_stage=stage,
    )
    started = time.perf_counter()
    curve_oof, points, receipts, target_access = _run_curve(
        capability=capability,
        root=root,
        data_dir=data_dir,
        config=config,
        preflight=preflight,
        stage=stage,
    )
    oof_sha = _write_parquet_exclusive(stage / "oof/learning_curve_oof.parquet", curve_oof)
    keys_sha = _write_parquet_exclusive(
        stage / "validation_keys.parquet",
        preflight["selected"][["fold", "anchor_id", "station", "episode_id"]],
    )
    reproducibility = {
        "r2_scientific_structure_deep_equal": True,
        "lock_sha_stage_and_operational_snapshot_bound_before_capability": True,
        "direct_engine_and_private_curve_single_use_phase_enforced": True,
        "target_free_pinned_split_no_r3_hs_recompute": True,
        "preflight_train_wave_hs_float_decodes_zero": True,
        "unreleased_validation_target_scalar_decodes_zero": (
            target_access["forbidden_validation_target_scalar_decodes"] == 0
        ),
        "validation_current_source_decode_zero_before_raw_fold_commitment": (
            target_access["forbidden_validation_current_hs_scalar_decodes"] == 0
            and all(
                row[
                    "validation_current_train_wave_hs_scalar_decodes_before_commitment"
                ]
                == 0
                for row in (
                    json.loads((stage / item["path"]).read_text(encoding="utf-8"))
                    for item in target_access["raw_fold_commitments"].values()
                )
            )
        ),
        "compact_24360_by_591_loaded_before_lock": True,
        "dense72_masked_train_only_supervision": True,
        "same_prefix_fold_metric_postprocess_and_three_seeds": True,
        "all_saved_models_reload_prediction_exact": all(
            row["saved_model_reload_prediction_exact"] for row in receipts
        ),
        "optimizer_step_total_exact_10260": (
            sum(row["optimizer_steps"] for row in receipts) == 10_260
        ),
        "all_parquet_outputs_o_excl_full_write": True,
        "anonymous_test_and_submission_value_reads_zero": True,
    }
    local_gate = evaluate_hypothesis_gate(
        points,
        leakage_checks=preflight["leakage_checks"],
        reproducibility_checks=reproducibility,
    )
    status = (
        "LOCAL_CURVE_QUALIFIED_PENDING_SEPARATE_STAGE_NO_CANDIDATE_NO_TEST"
        if local_gate["passed"]
        else "NO_LOCAL_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST"
    )
    evidence = central_evidence(
        points,
        leakage_checks=preflight["leakage_checks"],
        reproducibility_checks=reproducibility,
    )
    evidence.update(
        {
            "comparison_mode": COMPARISON_MODE,
            "local_numeric_gate": local_gate,
            "official_promotion": {
                "allowed": False,
                "reason": "same-surface official paired A/B has not occurred",
                "reference_seed_full_prediction_exact_to_historical_frozen_oof": False,
            },
            "historical_split_lineage": config["target_free_split"]["historical_lineage"],
        }
    )
    exclusive_json(stage / "learning_curve_evidence.json", evidence)
    access = {
        **target_access,
        "test_value_reads": 0,
        "candidate_files_created": 0,
        "full_fit_count": 0,
        "upload_attempts": 0,
    }
    _, input_after = predecessor_guard.verify_input_pins(
        root, data_dir, preflight["predecessor"]
    )
    if input_after != preflight["input_snapshot"]:
        raise RuntimeError("r3 immutable inputs changed during execution")
    metrics = {
        "created_at_kst": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "comparison_mode": COMPARISON_MODE,
        "local_numeric_curve_qualified": bool(local_gate["passed"]),
        "official_promotion_allowed": False,
        "hypothesis": HYPOTHESIS,
        "points": {str(fraction): points[fraction] for fraction in PREFIX_FRACTIONS},
        "local_gate": local_gate,
        "split_audit": preflight["split_audit"],
        "prefix_audit": preflight["prefix_audit"],
        "leakage_checks": preflight["leakage_checks"],
        "reproducibility_checks": reproducibility,
        "training_receipts": receipts,
        "dense72_access_audit": target_access,
        "access_counters": access,
        "candidate_created": False,
        "test_prediction_created": False,
        "full_fit_performed": False,
        "official_upload_count": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    exclusive_json(stage / "metrics.json", metrics)
    exclusive_json(
        stage / "registry.json",
        {
            "created_at_kst": _now(),
            "experiment_id": config["experiment_id"],
            "status": status,
            "local_curve_qualified": bool(local_gate["passed"]),
            "official_promotion_allowed": False,
            "candidate_created": False,
            "candidate_uploaded": False,
            "official_upload_count": 0,
        },
    )
    manifest = {
        "created_at_kst": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation_pins": implementation_pins(root, config),
        "operational_snapshot_sha256": preflight["operational_snapshot_sha256"],
        "input_sha256_before": preflight["input_snapshot"],
        "input_sha256_after": input_after,
        "output_files_before_manifest": _artifact_hashes(stage),
        "oof_parquet_sha256": oof_sha,
        "validation_keys_parquet_sha256": keys_sha,
        "local_curve_qualified": bool(local_gate["passed"]),
        "official_promotion_allowed": False,
        "candidate_created": False,
        "official_upload_count": 0,
        "access_counters": access,
    }
    exclusive_json(stage / "manifest.json", manifest)
    manifest_sha = sha256_file(stage / "manifest.json")
    _write_all_exclusive(
        stage / "manifest.sha256", f"{manifest_sha}  manifest.json\n".encode("ascii")
    )
    if paths["output"].exists():
        raise FileExistsError("canonical r3 output appeared before atomic move")
    stage.replace(paths["output"])
    return {
        "schema_version": "p3_hierarchical_residual_basis.gen5r3_dense72.run.r1",
        "status": status,
        "artifact_dir": config["canonical_paths"]["output"],
        "metrics_sha256": sha256_file(paths["output"] / "metrics.json"),
        "oof_sha256": oof_sha,
        "manifest_sha256": manifest_sha,
        "local_curve_qualified": bool(local_gate["passed"]),
        "official_promotion_allowed": False,
        "candidate_sha256": None,
        "test_prediction_count": 0,
        "official_upload_count": 0,
    }


__all__ = [
    "_run_curve",
    "_write_all_exclusive",
    "_write_parquet_exclusive",
    "execute_curve_stage",
]
