"""Run the append-only P3 hierarchical residual-basis curve exactly once.

This fixed-protocol CUDA runner is train/OOF-only. It seals all 45 blind
validation predictions before attaching the already-scored comparator truth,
never opens anonymous-test values, and cannot create or upload a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ocean_goal.meaningful_score_ledger_v5 import validate_ledger
from ocean_goal.meaningful_score_v3 import evaluate_learning_curve, load_contract
from p3_wave.corrected_repeated_forward import CorrectedFold
from p3_wave.hierarchical_residual_basis import (
    FixedBasisTrainingConfig,
    HierarchicalResidualBasisConfig,
    HierarchicalResidualBasisForecaster,
    StaticRobustScaler,
    fit_fixed_epoch_hierarchical_model,
    load_fitted_hierarchical_model,
    predict_with_fitted_hierarchical_model,
    prepare_hierarchical_context,
    save_fitted_hierarchical_model,
)
from p3_wave.meaningful_learning_curve import (
    PREFIX_FRACTIONS,
    central_evidence,
    evaluate_hypothesis_gate,
    evaluate_point,
)
from p3_wave.one_shot_guard import acquire_persistent_attempt_lock, safe_new_stage_path
from p3_wave.validation import rmse

_GEN4_RUNNER_PATH = Path(__file__).with_name("run_p3_station_stable_energy_state_space_v1.py")
_GEN4_SPEC = importlib.util.spec_from_file_location("p3_gen5_gen4_helpers", _GEN4_RUNNER_PATH)
if _GEN4_SPEC is None or _GEN4_SPEC.loader is None:
    raise ImportError("failed to load pinned Gen4 runner helpers")
gen4 = importlib.util.module_from_spec(_GEN4_SPEC)
sys.modules[_GEN4_SPEC.name] = gen4
_GEN4_SPEC.loader.exec_module(gen4)

EXPECTED_CONFIG_SHA256 = "628ff908d6c8885b8461f46846d3029408c4b83df56ca95e35a2dbdb90f3b981"
EXPECTED_CONFIG_DEEP_SHA256 = "74a34a38d503cd340e68fdf8cc48d846a76aacd3a7ee567f82da9f2049ba6bdb"
GEN4_CONFIG_SHA256 = "2d2df3d2b566f795fe005368e7294ff0d9493e84c8c17cc813ff076d24b4fd03"
GEN4_CONFIG_DEEP_SHA256 = "ae99ea1e9499205e1ad594b7c715cf2d99f43f2b49b1bd426d848d964f56293d"
CANONICAL_CONFIG_RELATIVE = "configs/experiments/p3_hierarchical_residual_basis_v1.json"
CANONICAL_GEN4_CONFIG_RELATIVE = "configs/experiments/p3_station_stable_energy_state_space_v1.json"
CANONICAL_GOAL_RELATIVE = "configs/goals/meaningful_score_maximization_v3.json"
CANONICAL_COMPACT_CACHE_RELATIVE = "artifacts/p3/features_all20_v1"
CANONICAL_SEQUENCE_CACHE_RELATIVE = "artifacts/p3/sequences_all20_v1"
CANONICAL_GEN1_RELATIVE = "artifacts/p3_meaningful_learning_curve_20260823_v1"
CANONICAL_GEN4_RELATIVE = "artifacts/p3_station_stable_energy_state_space_20260823_v1"
CANONICAL_V5_LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v5/registry.jsonl"
CANONICAL_OUTPUT_RELATIVE = "artifacts/p3_hierarchical_residual_basis_20260823_v1"
CANONICAL_LOCK_RELATIVE = "artifacts/p3_hierarchical_residual_basis_20260823_v1.ATTEMPT_LOCK.json"
CANONICAL_EXECUTION_CLAIM_RELATIVE = (
    "artifacts/p3_hierarchical_residual_basis_20260823_v1.EXECUTION_CLAIM.json"
)
CANONICAL_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DATA_DIR = Path(r"C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast")
HYPOTHESIS = "hierarchical_residual_basis_nhits"


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


def _now() -> str:
    return gen4._now()


def _deep_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_paths(root: Path) -> dict[str, Path]:
    workspace = root.resolve(strict=True)
    return {
        "config": workspace / CANONICAL_CONFIG_RELATIVE,
        "gen4_config": workspace / CANONICAL_GEN4_CONFIG_RELATIVE,
        "goal": workspace / CANONICAL_GOAL_RELATIVE,
        "compact_cache": workspace / CANONICAL_COMPACT_CACHE_RELATIVE,
        "sequence_cache": workspace / CANONICAL_SEQUENCE_CACHE_RELATIVE,
        "gen1": workspace / CANONICAL_GEN1_RELATIVE,
        "gen4": workspace / CANONICAL_GEN4_RELATIVE,
        "v5_ledger": workspace / CANONICAL_V5_LEDGER_RELATIVE,
        "output": workspace / CANONICAL_OUTPUT_RELATIVE,
        "lock": workspace / CANONICAL_LOCK_RELATIVE,
        "claim": workspace / CANONICAL_EXECUTION_CLAIM_RELATIVE,
    }


def _implementation_paths(root: Path, paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "gen4_runner": _GEN4_RUNNER_PATH,
        "gen4_config": paths["gen4_config"],
        "hierarchical_residual_basis_module": root / "src/p3_wave/hierarchical_residual_basis.py",
        "corrected_split_module": root / "src/p3_wave/corrected_repeated_forward.py",
        "learning_curve_module": root / "src/p3_wave/meaningful_learning_curve.py",
        "persistence_shrink_module": root / "src/p3_wave/persistence_shrink.py",
        "one_shot_guard_module": root / "src/p3_wave/one_shot_guard.py",
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score_v3.py",
        "v5_ledger_contract": root / "configs/goals/meaningful_score_ledger_v5.json",
        "v5_ledger_evaluator": root / "src/ocean_goal/meaningful_score_ledger_v5.py",
    }


def _reference_paths(root: Path, paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "gen1_metrics": paths["gen1"] / "metrics.json",
        "gen1_learning_curve_oof": paths["gen1"] / "oof/learning_curve_oof.parquet",
        "gen1_manifest": paths["gen1"] / "manifest.json",
        "gen1_independent_qa": root
        / "artifacts/p3_meaningful_learning_curve_20260823_v1_QA/independent_aggregate_audit.json",
        "gen4_metrics": paths["gen4"] / "metrics.json",
        "gen4_learning_curve_evidence": paths["gen4"] / "learning_curve_evidence.json",
        "gen4_manifest": paths["gen4"] / "manifest.json",
        "gen4_validation_keys": paths["gen4"] / "validation_keys.parquet",
    }


def _verify_v5_anchor(root: Path, paths: dict[str, Path], config: dict[str, Any]) -> None:
    records = validate_ledger(root, paths["v5_ledger"])
    expected = config["central_ledger_anchor"]
    ledger_sha = gen4.gen3.gen2.gen1.base.sha256_file(paths["v5_ledger"])
    if (
        ledger_sha != expected["ledger_sha256"]
        or paths["v5_ledger"].stat().st_size != expected["ledger_bytes"]
        or len(records) != expected["event_count"]
        or records[-1].get("event_sha256") != expected["head_event_sha256"]
    ):
        raise PermissionError("canonical v5 ledger full-file anchor differs")
    matching = [record for record in records if record.get("seq") == expected["gen4_event_seq"]]
    if len(matching) != 1:
        raise PermissionError("canonical v5 Gen4 event is absent or duplicated")
    event = matching[0]
    if event.get("event_sha256") != expected["gen4_event_sha256"]:
        raise PermissionError("canonical v5 Gen4 event SHA differs")
    payload = event.get("payload", {})
    if (
        payload.get("evidence", {}).get("sha256") != expected["gen4_evidence_sha256"]
        or payload.get("decision", {}).get("decision") != expected["gen4_decision"]
        or payload.get("upload_performed") is not False
        or any(record.get("payload", {}).get("upload_performed") is not False for record in records)
    ):
        raise PermissionError("canonical v5 Gen4 event semantics differ")


def authorize_entry(
    *,
    root: Path,
    data_dir: Path,
    requested_config: Path,
    requested_output: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """First action: bind canonical paths, bytes, deep JSON, and immutable pins."""

    if root.resolve(strict=True) != CANONICAL_WORKSPACE_ROOT:
        raise PermissionError("non-canonical workspace root is forbidden")
    paths = _canonical_paths(root)
    if data_dir.resolve(strict=True) != CANONICAL_DATA_DIR.resolve(strict=True):
        raise PermissionError("non-canonical P3 data directory is forbidden")
    if requested_config.resolve(strict=True) != paths["config"].resolve(strict=True):
        raise PermissionError("non-canonical config path is forbidden")
    if requested_output.resolve(strict=False) != paths["output"].resolve(strict=False):
        raise PermissionError("non-canonical output path is forbidden")
    if paths["output"].exists():
        raise FileExistsError("canonical append-only output already exists")

    content = paths["config"].read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise PermissionError("canonical config byte SHA differs")
    config = json.loads(content)
    if _deep_sha(config) != EXPECTED_CONFIG_DEEP_SHA256:
        raise PermissionError("canonical config fails compiled deep-JSON equality")
    expected_paths = {
        "config": CANONICAL_CONFIG_RELATIVE,
        "goal_contract": CANONICAL_GOAL_RELATIVE,
        "compact_cache": CANONICAL_COMPACT_CACHE_RELATIVE,
        "sequence_cache": CANONICAL_SEQUENCE_CACHE_RELATIVE,
        "gen1_artifact": CANONICAL_GEN1_RELATIVE,
        "gen4_artifact": CANONICAL_GEN4_RELATIVE,
        "v5_ledger": CANONICAL_V5_LEDGER_RELATIVE,
        "output": CANONICAL_OUTPUT_RELATIVE,
        "attempt_lock": CANONICAL_LOCK_RELATIVE,
        "execution_claim": CANONICAL_EXECUTION_CLAIM_RELATIVE,
    }
    if config.get("canonical_paths") != expected_paths:
        raise PermissionError("canonical path fields differ")
    if config.get("experiment_id") != "p3_hierarchical_residual_basis_v1":
        raise PermissionError("experiment identity differs")
    if config.get("created_before_first_fit") is not True:
        raise PermissionError("preregistration timing declaration differs")
    if tuple(item["id"] for item in config["hypotheses"]) != (HYPOTHESIS,):
        raise PermissionError("single structural hypothesis differs")
    if config["validation"]["training_prefix_fractions"] != list(PREFIX_FRACTIONS):
        raise PermissionError("prefix curve differs")
    if config["model"] != {
        "class": "HierarchicalResidualBasisForecaster",
        "raw_context_steps": 289,
        "prepared_context_steps": 144,
        "prepared_channels": 24,
        "stack_pooling_factors": [12, 4, 1],
        "blocks_per_stack": 2,
        "hidden_width": 192,
        "conditioning_width": 128,
        "dropout": 0.1,
        "forecast_steps": 72,
        "forecast_knot_counts": [6, 18, 72],
        "official_target_indices_zero_based": [8, 17, 26, 35, 53, 71],
        "seed_replicates": [20260816, 20260817, 20260818],
        "expected_trainable_parameter_count": 4125120,
        "hyperparameter_search": False,
        "expected_actual_fit_cells": 45,
        "expected_optimizer_steps": 10260,
    }:
        raise PermissionError("frozen hierarchical residual-basis model contract differs")
    if config["training"] != {
        "epochs": 12,
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "device": "cuda",
        "precision": "bfloat16_autocast_float32_optimizer",
        "early_stopping": False,
        "warm_start_across_prefix_seed_or_fold": False,
        "expected_actual_fit_cells": 45,
        "expected_optimizer_steps": 10260,
    }:
        raise PermissionError("fixed training contract differs")
    if config["postprocess"] != {
        "clip_m": [0.0, 30.0],
        "fixed_persistence_shrink_active_leads_h": [12, 18, 24],
        "fixed_persistence_weight": 0.2,
        "identical_to_gen1_comparator": True,
        "new_router_or_blend": False,
    }:
        raise PermissionError("fixed postprocess differs")
    if not all(config["prohibitions"].values()):
        raise PermissionError("all prohibitions must remain enabled")

    for name, path in _implementation_paths(root, paths).items():
        if gen4.gen3.gen2.gen1.base.sha256_file(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    for name, path in _reference_paths(root, paths).items():
        if gen4.gen3.gen2.gen1.base.sha256_file(path) != config["reference_evidence_sha256"][name]:
            raise PermissionError(f"reference evidence SHA differs: {name}")
    _verify_v5_anchor(root, paths, config)
    return config, paths


def _gen4_paths(root: Path) -> dict[str, Path]:
    return gen4._canonical_paths(root)


def _preflight(
    *, root: Path, data_dir: Path, config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    gen4_config_bytes = paths["gen4_config"].read_bytes()
    if hashlib.sha256(gen4_config_bytes).hexdigest() != GEN4_CONFIG_SHA256:
        raise PermissionError("pinned Gen4 config byte SHA differs")
    gen4_config = json.loads(gen4_config_bytes)
    if _deep_sha(gen4_config) != GEN4_CONFIG_DEEP_SHA256:
        raise PermissionError("pinned Gen4 config deep JSON differs")
    preflight = gen4._preflight(
        root=root,
        data_dir=data_dir,
        config=gen4_config,
        paths=_gen4_paths(root),
    )
    gen4_metrics = json.loads((paths["gen4"] / "metrics.json").read_text(encoding="utf-8"))
    gen4_full = gen4_metrics["points"]["1.0"]
    diagnosis = config["gen4_failure_diagnosis"]
    observed_prefix_deltas = [
        float(gen4_metrics["points"][str(fraction)]["delta_candidate_minus_incumbent_m"])
        for fraction in PREFIX_FRACTIONS
    ]
    if (
        gen4_metrics["central_goal_evaluator"]["decision"] != "RESEARCH_ONLY"
        or float(gen4_full["delta_candidate_minus_incumbent_m"])
        != diagnosis["full_delta_candidate_minus_incumbent_m"]
        or list(gen4_full["delta_ci90_m"]) != diagnosis["full_ci90_m"]
        or observed_prefix_deltas != diagnosis["prefix_deltas_candidate_minus_incumbent_m"]
        or gen4_full["fold_deltas_candidate_minus_incumbent_m"] != diagnosis["fold_delta_m"]
        or gen4_full["slice_deltas_candidate_minus_incumbent_m"]
        != diagnosis["critical_slice_delta_m"]
    ):
        raise ValueError("sealed Gen4 aggregate diagnosis differs")
    context_started = time.perf_counter()
    prepared_context = prepare_hierarchical_context(
        torch.from_numpy(np.array(preflight["raw"][:2], dtype=np.float32, copy=True))
    )
    context_elapsed = time.perf_counter() - context_started
    if tuple(prepared_context.values.shape) != (2, 144, 24):
        raise ValueError("hierarchical context contract differs")
    parameter_count = HierarchicalResidualBasisForecaster(
        _frozen_model_config()
    ).trainable_parameter_count
    if parameter_count != config["model"]["expected_trainable_parameter_count"]:
        raise ValueError("trainable parameter count differs")
    expected_steps = sum(
        math.ceil(
            len(preflight["prefix_ids"][fraction][fold.name]) / config["training"]["batch_size"]
        )
        * config["training"]["epochs"]
        for fraction in PREFIX_FRACTIONS
        for _seed in config["model"]["seed_replicates"]
        for fold in preflight["folds"]
    )
    if expected_steps != config["training"]["expected_optimizer_steps"]:
        raise ValueError("optimizer-step accounting differs")
    preflight.update(
        {
            "context_probe_shape": list(prepared_context.values.shape),
            "context_build_elapsed_seconds": float(context_elapsed),
            "trainable_parameter_count": int(parameter_count),
            "expected_optimizer_steps": int(expected_steps),
        }
    )
    return preflight


def _protected_roots(root: Path, data_dir: Path, paths: dict[str, Path]) -> tuple[Path, ...]:
    return (
        data_dir,
        paths["compact_cache"],
        paths["sequence_cache"],
        paths["gen1"],
        paths["gen4"],
        root / "submissions",
        root / "output",
        root / "데이터셋 원본",
    )


def _prefix_id_sha(ids: np.ndarray) -> str:
    return gen4._prefix_id_sha(ids)


def _write_npy_exclusive(path: Path, values: np.ndarray) -> str:
    return gen4._write_npy_exclusive(path, values)


def _postprocess(delta_hs: np.ndarray, current_hs: np.ndarray) -> np.ndarray:
    return gen4.gen3._postprocess(
        np.asarray(delta_hs, dtype=np.float64),
        np.asarray(current_hs, dtype=np.float64),
    )


def _run_curve(
    *,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    preflight: dict[str, Any],
    stage: Path,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]], list[dict[str, Any]]]:
    raw = preflight["raw"]
    station = preflight["station"]
    compact = preflight["compact"]
    anchors = preflight["anchors"]
    folds: tuple[CorrectedFold, ...] = preflight["folds"]
    protected = _protected_roots(root, data_dir, paths)
    model_config = _frozen_model_config()
    training_config = _frozen_training_config()
    if compact.shape != (24_360, model_config.static_feature_count):
        raise ValueError("static feature matrix differs from frozen model contract")
    anchor_lookup = anchors.set_index("anchor_id")
    blind_predictions: dict[tuple[float, int, str], np.ndarray] = {}
    receipts: list[dict[str, Any]] = []
    completed = 0
    expected_steps = preflight["expected_optimizer_steps"]

    for fraction in PREFIX_FRACTIONS:
        prefix_tag = f"p{int(round(fraction * 100)):03d}"
        prefix_scalers: dict[str, StaticRobustScaler] = {}
        train_payloads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for fold in folds:
            train_ids = preflight["prefix_ids"][fraction][fold.name]
            validation_ids = fold.validation_ids
            prefix_scalers[fold.name] = StaticRobustScaler.fit(
                np.array(compact[train_ids], copy=True),
                train_ids,
                forbidden_case_ids=validation_ids,
            )
            train_payloads[fold.name] = gen4.gen3._load_train_targets(
                preflight["anchor_path"], anchors, train_ids
            )
        for seed in config["model"]["seed_replicates"]:
            for fold in folds:
                train_ids = preflight["prefix_ids"][fraction][fold.name]
                validation_ids = fold.validation_ids
                if np.intersect1d(train_ids, validation_ids).size:
                    raise AssertionError("train/validation IDs overlap before fit")
                train_target, train_weight = train_payloads[fold.name]
                current_hs = anchor_lookup.loc[validation_ids, "current_hs"].to_numpy(np.float64)
                print(
                    json.dumps(
                        {
                            "phase": "fit_hierarchical_residual_basis_cell",
                            "completed_before": completed,
                            "total_actual_fit_cells": 45,
                            "prefix": fraction,
                            "seed": int(seed),
                            "fold": fold.name,
                            "train_cases": len(train_ids),
                            "validation_target_values_read_by_model": 0,
                            "device": "cuda_bfloat16",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                started = time.perf_counter()
                fitted = fit_fixed_epoch_hierarchical_model(
                    np.array(raw[train_ids], copy=True),
                    np.array(station[train_ids], copy=True),
                    np.array(compact[train_ids], copy=True),
                    np.array(train_target, copy=True),
                    np.array(train_weight, copy=True),
                    train_ids,
                    forbidden_case_ids=validation_ids,
                    seed=int(seed),
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                    static_scaler=prefix_scalers[fold.name],
                )
                raw_prediction = predict_with_fitted_hierarchical_model(
                    fitted,
                    np.array(raw[validation_ids], copy=True),
                    np.array(station[validation_ids], copy=True),
                    np.array(compact[validation_ids], copy=True),
                    device="cuda",
                    batch_size=512,
                )
                prediction = _postprocess(raw_prediction, current_hs)
                model_relative = f"models/{prefix_tag}/seed_{seed}/folds/{fold.name}/model.pt"
                model_path = safe_new_stage_path(stage, model_relative, protected_roots=protected)
                save_fitted_hierarchical_model(fitted, model_path)
                model_sha = gen4.gen3.gen2.gen1.base.sha256_file(model_path)
                blind_relative = f"blind_predictions/{prefix_tag}/seed_{seed}/{fold.name}.npy"
                blind_path = safe_new_stage_path(stage, blind_relative, protected_roots=protected)
                blind_sha = _write_npy_exclusive(blind_path, prediction.astype(np.float64))

                reloaded = load_fitted_hierarchical_model(model_path, map_location="cpu")
                reload_delta = predict_with_fitted_hierarchical_model(
                    reloaded,
                    np.array(raw[validation_ids], copy=True),
                    np.array(station[validation_ids], copy=True),
                    np.array(compact[validation_ids], copy=True),
                    device="cuda",
                    batch_size=512,
                )
                reload_prediction = _postprocess(reload_delta, current_hs)
                reload_exact = bool(np.array_equal(reload_prediction, prediction))
                maximum_reload_difference = float(np.max(np.abs(reload_prediction - prediction)))
                if not reload_exact:
                    raise RuntimeError("saved hierarchical model reload failed exact reproduction")
                blind_predictions[(fraction, int(seed), fold.name)] = prediction
                completed += 1
                receipts.append(
                    {
                        "prefix_fraction": float(fraction),
                        "seed": int(seed),
                        "fold": fold.name,
                        "train_cases": int(len(train_ids)),
                        "validation_cases": int(len(validation_ids)),
                        "optimizer_steps": int(fitted.training_steps),
                        "train_id_sha256": _prefix_id_sha(train_ids),
                        "validation_id_sha256": _prefix_id_sha(validation_ids),
                        "train_context_sha256": fitted.train_context_sha256,
                        "scaler_fit_id_sha256": fitted.scaler.fit_ids_sha256,
                        "scaler_state_sha256": fitted.scaler_state_sha256,
                        "model_state_sha256": fitted.model_state_sha256,
                        "model_relative_path": model_relative,
                        "model_sha256": model_sha,
                        "blind_prediction_relative_path": blind_relative,
                        "blind_prediction_sha256": blind_sha,
                        "blind_prediction_sealed_before_validation_truth_attachment": True,
                        "saved_model_reload_prediction_exact": reload_exact,
                        "saved_model_reload_max_abs_difference_m": maximum_reload_difference,
                        "elapsed_seconds": float(time.perf_counter() - started),
                        "validation_target_values_read_by_model": 0,
                        "test_or_hidden_value_reads": 0,
                    }
                )

    if completed != 45 or sum(row["optimizer_steps"] for row in receipts) != expected_steps:
        raise AssertionError("fit-cell or optimizer-step count differs")
    comparator_truth = gen4.gen3._load_comparator_truth_after_blind(_gen4_paths(root))
    points: dict[float, dict[str, Any]] = {}
    all_frames: list[pd.DataFrame] = []
    for fraction in PREFIX_FRACTIONS:
        seed_frames: list[pd.DataFrame] = []
        for seed in config["model"]["seed_replicates"]:
            fold_frames: list[pd.DataFrame] = []
            for fold in folds:
                comparator = gen4.gen3.gen2._cell_comparator(
                    comparator_truth, fraction=fraction, fold=fold
                )
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
            np.column_stack([frame["challenger_prediction"].to_numpy(float) for frame in ordered]),
            axis=1,
        )
        mean_frame["prefix_fraction"] = float(fraction)
        all_frames.append(mean_frame)
        point = evaluate_point(
            mean_frame,
            candidate_column="challenger_prediction",
            bootstrap_replicates=config["validation"]["bootstrap_replicates"],
            bootstrap_seed=int(config["validation"]["bootstrap_seed"]) + int(round(fraction * 100)),
        )
        gen1_point = preflight["gen1_metrics"]["points_by_hypothesis"]["fixed_horizon_splice"][
            str(fraction)
        ]
        point["incumbent_seed_metrics"] = [
            float(value) for value in gen1_point["incumbent_seed_metrics"]
        ]
        point["challenger_seed_metrics"] = [
            float(rmse(frame["target_hs"], frame["challenger_prediction"])) for frame in ordered
        ]
        points[fraction] = point
        print(
            json.dumps(
                {
                    "phase": "prefix_scored_after_all_45_blind_predictions_sealed",
                    "prefix": fraction,
                    "completed_actual_fit_cells": completed,
                    "optimizer_steps": expected_steps,
                    "delta_candidate_minus_incumbent_m": point["delta_candidate_minus_incumbent_m"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return pd.concat(all_frames, ignore_index=True), points, receipts


def _artifact_hashes(stage: Path) -> dict[str, dict[str, Any]]:
    return gen4._artifact_hashes(stage)


def _verify_persisted_attempt(
    *, paths: dict[str, Path], config: dict[str, Any], attempt: dict[str, Any]
) -> None:
    expected_keys = {
        "created_at",
        "status",
        "experiment_id",
        "canonical_config_sha256",
        "o_excl",
        "rerun_forbidden",
        "sha256",
    }
    if set(attempt) != expected_keys:
        raise PermissionError("attempt receipt fields differ")
    if not paths["lock"].is_file():
        raise FileNotFoundError("canonical persistent attempt lock is absent")
    if gen4.gen3.gen2.gen1.base.sha256_file(paths["lock"]) != attempt["sha256"]:
        raise PermissionError("canonical persistent attempt lock SHA differs")
    persisted = json.loads(paths["lock"].read_text(encoding="utf-8"))
    if persisted != {key: value for key, value in attempt.items() if key != "sha256"}:
        raise PermissionError("in-memory attempt differs from canonical persisted lock")
    if (
        attempt["status"] != "ATTEMPT_CONSUMED_ONE_SHOT"
        or attempt["experiment_id"] != config["experiment_id"]
        or attempt["canonical_config_sha256"] != EXPECTED_CONFIG_SHA256
        or attempt["o_excl"] is not True
        or attempt["rerun_forbidden"] is not True
    ):
        raise PermissionError("persistent attempt lock semantics differ")


def _run_after_lock(
    *,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    authorized_config, authorized_paths = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_output=paths["output"],
    )
    if authorized_config != config or authorized_paths != paths:
        raise PermissionError("direct-call config or canonical path context differs")
    _verify_persisted_attempt(paths=paths, config=config, attempt=attempt)
    execution_claim = acquire_persistent_attempt_lock(
        paths["claim"],
        experiment_id=config["experiment_id"],
        config_sha256=EXPECTED_CONFIG_SHA256,
        created_at=_now(),
    )
    started = time.perf_counter()
    preflight = _preflight(root=root, data_dir=data_dir, config=config, paths=paths)
    print(
        json.dumps(
            {
                "phase": "preflight_pass",
                "validation_cases": 181,
                "actual_fit_cells": 45,
                "optimizer_steps": 10260,
                "device": "cuda_bfloat16",
                "test_value_reads": 0,
                "upload_count": 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_hierarchical_residual_basis_v1_", dir=tmp_root))
    curve_oof, points, receipts = _run_curve(
        root=root,
        data_dir=data_dir,
        config=config,
        paths=paths,
        preflight=preflight,
        stage=stage,
    )
    gen4.gen3.gen2.gen1.base._atomic_parquet(stage / "oof/learning_curve_oof.parquet", curve_oof)
    gen4.gen3.gen2.gen1.base._atomic_parquet(
        stage / "validation_keys.parquet",
        preflight["selected"][["fold", "anchor_id", "station", "episode_id"]],
    )
    gen4.gen3.gen2.gen1.base._atomic_json(
        stage / "hierarchical_basis_contract.json",
        {
            "source": "case_local_past_only_raw_48h_context",
            "raw_context_steps": 289,
            "raw_cadence_minutes": 10,
            "prepared_context_steps": 144,
            "prepared_cadence_minutes": 20,
            "prepared_channels": 24,
            "pooling_factors": [12, 4, 1],
            "blocks_per_stack": 2,
            "forecast_knot_counts": [6, 18, 72],
            "dense_forecast_steps": 72,
            "optimizer_target_indices_zero_based": [8, 17, 26, 35, 53, 71],
            "dense_future_labels_constructed": False,
            "validation_target_values_used_before_all_blind_seals": 0,
        },
    )

    reproducibility_checks = {
        "canonical_config_path_sha_and_deep_json_equal": True,
        "sealed_gen1_comparator_fresh_refit_each_prefix_and_seed": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
        "same_prefix_ids_for_comparator_and_challenger": True,
        "challenger_fresh_refit_each_prefix_fold_and_seed": True,
        "three_fixed_stochastic_seed_predictions_mean_then_metric": True,
        "same_metric_clip_and_fixed_0p20_shrink": True,
        "hyperparameter_alpha_shrink_weight_and_seed_search_zero": True,
        "all_45_models_and_blind_predictions_saved_and_hashed": len(receipts) == 45,
        "all_saved_models_reload_prediction_exact": all(
            row["saved_model_reload_prediction_exact"] for row in receipts
        ),
        "blind_predictions_sealed_before_validation_truth_attachment": all(
            row["blind_prediction_sealed_before_validation_truth_attachment"] for row in receipts
        ),
        "model_fit_targets_are_train_only_six_official_deltas": True,
        "dense_future_target_surface_not_constructed": True,
        "fixed_epoch_training_without_validation_selection": True,
        "optimizer_step_total_exact_10260": (
            sum(row["optimizer_steps"] for row in receipts) == 10260
        ),
    }
    gate = evaluate_hypothesis_gate(
        points,
        leakage_checks=preflight["leakage_checks"],
        reproducibility_checks=reproducibility_checks,
    )
    evidence = central_evidence(
        points,
        leakage_checks=preflight["leakage_checks"],
        reproducibility_checks=reproducibility_checks,
    )
    evidence["comparison_mode"] = "EXACT_OFFICIAL_PREFIX_REFIT"
    evidence["preregistration"] = {
        "generation_id": config["experiment_id"],
        "config_path": CANONICAL_CONFIG_RELATIVE,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "created_before_first_fit": True,
        "hypothesis_count": 1,
        "hypothesis_count_at_most_3": True,
        "score_derived_tuning": False,
    }
    evidence["curve_protocol"] = {
        "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT",
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "seed_ids": [20260816, 20260817, 20260818],
        "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
        "bootstrap_replicates": int(config["validation"]["bootstrap_replicates"]),
        "bootstrap_cluster": "whole_case",
        "incumbent_fresh_refit_each_prefix": True,
        "challenger_fresh_refit_each_prefix": True,
        "same_fold_keys_metric_postprocess": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
        "inherited_from_gen1_fail_closed": True,
        "challenger_seed_policy": (
            "three independently initialized fixed-seed fits per prefix-fold; arithmetic "
            "prediction mean is scored once and all three seed RMSE values are retained"
        ),
    }
    for point in evidence["points"]:
        fraction = float(point["fraction"])
        point["incumbent_seed_metrics"] = list(points[fraction]["incumbent_seed_metrics"])
        point["challenger_seed_metrics"] = list(points[fraction]["challenger_seed_metrics"])
    central = evaluate_learning_curve(load_contract(root, CANONICAL_GOAL_RELATIVE), evidence)
    if central["passed"]:
        raise AssertionError("known false exact-reference check must fail closed")
    gen4.gen3.gen2.gen1.base._atomic_json(stage / "learning_curve_evidence.json", evidence)

    access = {
        "test_sequence_cache_value_reads": 0,
        "test_feature_cache_value_reads": 0,
        "test_index_value_reads": 0,
        "test_context_value_reads": 0,
        "test_target_or_hidden_label_reads": 0,
        "absolute_test_timestamp_recovery_attempts": 0,
        "current_or_frozen_submission_value_reads": 0,
        "current_or_frozen_submission_writes": 0,
        "upload_attempts": 0,
    }
    inputs = gen4.gen3._input_paths(root, data_dir, _gen4_paths(root))
    input_after = gen4.gen3.gen2._verify_input_hashes(inputs, config["expected_sha256"])
    if input_after != preflight["snapshot"]:
        raise RuntimeError("source/cache/current/frozen inputs changed during run")
    status = "NO_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST_READS"
    metrics = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "interpretation": (
            "Corrected same-surface Gen5 research evidence for a genuinely distinct "
            "N-HiTS-style hierarchical residual-basis sequence model. It is not an "
            "official hidden score or upload authorization; the inherited Gen1 "
            "exact-reference mismatch forces fail-close."
        ),
        "one_shot_attempt": attempt,
        "one_shot_execution_claim": execution_claim,
        "hypothesis": HYPOTHESIS,
        "gen4_failure_diagnosis": config["gen4_failure_diagnosis"],
        "points": {str(fraction): points[fraction] for fraction in PREFIX_FRACTIONS},
        "local_gate": gate,
        "central_goal_evaluator": central,
        "split_audit": preflight["split_audit"],
        "prefix_audit": preflight["prefix_audit"],
        "leakage_checks": preflight["leakage_checks"],
        "reproducibility_checks": reproducibility_checks,
        "context_preflight": {
            "probe_shape": preflight["context_probe_shape"],
            "elapsed_seconds": preflight["context_build_elapsed_seconds"],
            "dense_future_labels_constructed": False,
            "validation_target_values_used": 0,
        },
        "training_receipts": receipts,
        "access_counters": access,
        "candidate_validation": None,
        "invariants": {
            "append_only": True,
            "fixed_model_training_run": True,
            "hyperparameter_search_run": False,
            "shrink_alpha_or_weight_micro_tuning_run": False,
            "test_target_or_hidden_label_reads": 0,
            "absolute_test_timestamp_recovered": False,
            "current_or_frozen_submission_mutated": False,
            "official_submission_uploads": 0,
            "team_wide_daily_upload_limit_assumed": True,
            "source_cache_current_frozen_sha_unchanged": True,
            "fixed_three_seed_cuda_training": True,
            "dense_future_label_builder_used": False,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    gen4.gen3.gen2.gen1.base._atomic_json(stage / "metrics.json", metrics)
    registry = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "hypotheses": [
            {
                "id": HYPOTHESIS,
                "curve_qualified": False,
                "local_gate_passed": bool(gate["passed"]),
                "central_decision": central["decision"],
                "full_delta_m": float(points[1.0]["delta_candidate_minus_incumbent_m"]),
                "promotion_status": "research_only_do_not_promote",
            }
        ],
        "candidate_created": False,
        "candidate_uploaded": False,
        "current_frozen_sha256": config["expected_sha256"]["current/ready_submission.csv"],
        "current_frozen_unchanged": True,
        "official_upload_count": 0,
    }
    gen4.gen3.gen2.gen1.base._atomic_json(stage / "registry.json", registry)

    implementation_paths = {
        "config": paths["config"],
        "runner": Path(__file__).resolve(),
        "runner_tests": root / "tests/test_p3_hierarchical_residual_basis_runner.py",
        "module_tests": root / "tests/test_p3_hierarchical_residual_basis.py",
        **_implementation_paths(root, paths),
    }
    manifest = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "append_only_generation": True,
        "canonical_contract": {
            "config_path": CANONICAL_CONFIG_RELATIVE,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
            "config_full_deep_equality": True,
            "gen1_path": CANONICAL_GEN1_RELATIVE,
            "gen4_path": CANONICAL_GEN4_RELATIVE,
            "v5_gen4_event_sha256": config["central_ledger_anchor"]["gen4_event_sha256"],
            "v5_ledger_sha256": config["central_ledger_anchor"]["ledger_sha256"],
            "v5_ledger_event_count": config["central_ledger_anchor"]["event_count"],
            "v5_ledger_head_event_sha256": config["central_ledger_anchor"]["head_event_sha256"],
            "output_path": CANONICAL_OUTPUT_RELATIVE,
            "attempt_lock_path": CANONICAL_LOCK_RELATIVE,
            "attempt_lock_sha256": attempt["sha256"],
            "execution_claim_path": CANONICAL_EXECUTION_CLAIM_RELATIVE,
            "execution_claim_sha256": execution_claim["sha256"],
        },
        "implementation_sha256": {
            name: gen4.gen3.gen2.gen1.base.sha256_file(path)
            for name, path in implementation_paths.items()
        },
        "git": gen4.gen3.gen2.gen1.base._git_state(root),
        "input_sha256_before": preflight["snapshot"],
        "input_sha256_after": input_after,
        "source_cache_current_frozen_unchanged": True,
        "output_files": _artifact_hashes(stage),
        "curve_qualified": False,
        "local_gate_passed": bool(gate["passed"]),
        "candidate_created": False,
        "candidate_uploaded": False,
        "official_upload_count": 0,
        "access_counters": access,
    }
    gen4.gen3.gen2.gen1.base._atomic_json(stage / "manifest.json", manifest)
    manifest_sha = gen4.gen3.gen2.gen1.base.sha256_file(stage / "manifest.json")
    (stage / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii", newline="\n"
    )
    if paths["output"].exists():
        raise FileExistsError("canonical output appeared before atomic move")
    stage.replace(paths["output"])
    result = {
        "status": status,
        "artifact_dir": CANONICAL_OUTPUT_RELATIVE,
        "metrics_sha256": gen4.gen3.gen2.gen1.base.sha256_file(paths["output"] / "metrics.json"),
        "oof_sha256": gen4.gen3.gen2.gen1.base.sha256_file(
            paths["output"] / "oof/learning_curve_oof.parquet"
        ),
        "learning_curve_evidence_sha256": gen4.gen3.gen2.gen1.base.sha256_file(
            paths["output"] / "learning_curve_evidence.json"
        ),
        "registry_sha256": gen4.gen3.gen2.gen1.base.sha256_file(paths["output"] / "registry.json"),
        "manifest_sha256": manifest_sha,
        "candidate_sha256": None,
        "local_gate_passed": bool(gate["passed"]),
        "central_decision": central["decision"],
        "elapsed_seconds": float(time.perf_counter() - started),
        "official_upload_count": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    paths = _canonical_paths(root)
    config, paths = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_output=paths["output"],
    )
    preflight = _preflight(root=root, data_dir=data_dir, config=config, paths=paths)
    return {
        "status": "CANONICAL_CHECK_ONLY_PASS",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "validation_cases": int(len(preflight["selected"])),
        "validation_rows": int(preflight["split_audit"]["validation_row_count"]),
        "actual_fit_cells": 45,
        "optimizer_steps": preflight["expected_optimizer_steps"],
        "context_probe_shape": preflight["context_probe_shape"],
        "trainable_parameter_count": preflight["trainable_parameter_count"],
        "hypotheses": [HYPOTHESIS],
        "leakage_checks": preflight["leakage_checks"],
        "inherited_exact_reference_check": False,
        "output_absent": not paths["output"].exists(),
        "attempt_lock_absent": not paths["lock"].exists(),
        "execution_claim_absent": not paths["claim"].exists(),
        "test_value_reads": 0,
        "upload_count": 0,
    }


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    paths = _canonical_paths(root)
    config, paths = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_output=paths["output"],
    )
    attempt = acquire_persistent_attempt_lock(
        paths["lock"],
        experiment_id=config["experiment_id"],
        config_sha256=EXPECTED_CONFIG_SHA256,
        created_at=_now(),
    )
    return _run_after_lock(
        root=root,
        data_dir=data_dir,
        config=config,
        paths=paths,
        attempt=attempt,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    result = (
        check_only(root=root, data_dir=data_dir)
        if args.check_only
        else run_experiment(root=root, data_dir=data_dir)
    )
    if args.check_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
