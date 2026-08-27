"""Run the append-only P1 masked-pretrain binary-event Gen4 curve once."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocean_goal.meaningful_score_v3 import evaluate_learning_curve, load_contract
from p1_qc.masked_pretrain_binary_tcn import (
    MaskedPretrainModelConfig,
    MaskedPretrainTrainingConfig,
    fit_masked_pretrain_binary_event_model,
    load_fitted_masked_pretrain_model,
    predict_masked_pretrain_binary_probability,
    save_fitted_masked_pretrain_model,
)

_GEN3_ADAPTER = PROJECT_ROOT / "scripts/run_p1_binary_event_tcn_dense_natural_v3.py"
_GEN3_SPEC = importlib.util.spec_from_file_location("p1_gen4_shared_curve_adapter", _GEN3_ADAPTER)
if _GEN3_SPEC is None or _GEN3_SPEC.loader is None:
    raise ImportError("failed to load pinned P1 Gen3 curve adapter")
gen3 = importlib.util.module_from_spec(_GEN3_SPEC)
sys.modules[_GEN3_SPEC.name] = gen3
_GEN3_SPEC.loader.exec_module(gen3)
shared = gen3.shared

EXPECTED_CONFIG_SHA256 = "a4af6f97d1102752444edf92c8145886b44b6c5175cef0c74beba11009a20767"
EXPECTED_CONFIG_DEEP_SHA256 = "bd7168f889159fb1d4cc414623cf516d653de669178aba22494b7ca4b8af2293"
CANONICAL_CONFIG = "configs/experiments/p1_masked_pretrain_binary_event_v4.json"
CANONICAL_ARTIFACT = "artifacts/p1_masked_pretrain_binary_event_v4"
CANONICAL_LOCK = "artifacts/p1_masked_pretrain_binary_event_v4.ATTEMPT_LOCK.json"
HYPOTHESIS = "masked_sequence_pretraining_then_binary_event_finetune"
FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
SEEDS = (20260813, 20260829, 20260847)
_SHARED_JSON_NEW = shared._json_new
EXECUTION_TOMBSTONE = "artifacts/p1_masked_pretrain_binary_event_v4/EXECUTION_TOMBSTONE.json"
EXECUTION_TOMBSTONE_SHA256 = "28563cd076df5cc617545ccbc079c50df02cfc05bde746324127f8513bc6cda1"


def _enforce_execution_tombstone(root: Path) -> None:
    path = (root.resolve(strict=True) / EXECUTION_TOMBSTONE).resolve(strict=True)
    if shared._sha(path) != EXECUTION_TOMBSTONE_SHA256:
        raise PermissionError("invalidated Gen4 execution tombstone SHA differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not (
        value.get("generation") == "p1_masked_pretrain_binary_event_v4"
        and value.get("execution_prohibited") is True
        and value.get("authorization_must_fail_before_attempt_lock") is True
        and value.get("attempt_lock_created") is False
        and value.get("curve_model_fits") == 0
        and value.get("uploads") == 0
    ):
        raise PermissionError("invalidated Gen4 execution tombstone semantics differ")
    raise PermissionError("p1_masked_pretrain_binary_event_v4 is superseded and non-executable")


def _paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / CANONICAL_CONFIG,
        "artifact": root / CANONICAL_ARTIFACT,
        "lock": root / CANONICAL_LOCK,
        "base_config": root / "configs/p1.toml",
        "goal": root / "configs/goals/meaningful_score_maximization_v3.json",
        "feature_cache": root / "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
        "feature_metadata": root / "artifacts/cache/train_offline_e9fe1eb46cb7431f.json",
        "gen1": root / "artifacts/p1_meaningful_learning_curve_generation_v1",
        "gen2": root / "artifacts/p1_station_layer_temporal_convolution_event_v2",
        "gen3": root / "artifacts/p1_binary_event_tcn_dense_natural_v3",
        "frozen_oof": root / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
    }


def _model_config(
    config: dict[str, Any], feature_count: int, group_count: int
) -> MaskedPretrainModelConfig:
    model = config["model"]
    result = MaskedPretrainModelConfig(
        input_feature_count=feature_count,
        group_count=group_count,
        width=int(model["width"]),
        group_embedding_width=int(model["group_embedding_width"]),
        dilations=tuple(int(value) for value in model["dilations"]),
        kernel_size=int(model["kernel_size"]),
        dropout=float(model["dropout"]),
        norm_groups=int(model["norm_groups"]),
    )
    result.validate()
    if result.receptive_field_rows != model["receptive_field_rows"]:
        raise ValueError("registered receptive field differs from implementation")
    return result


def _training_config(config: dict[str, Any]) -> MaskedPretrainTrainingConfig:
    training = config["training"]
    result = MaskedPretrainTrainingConfig(
        optimizer_steps=int(training["optimizer_steps_per_cell"]),
        pretrain_steps=int(training["pretrain_steps_per_cell"]),
        finetune_steps=int(training["finetune_steps_per_cell"]),
        batch_size=int(training["batch_size"]),
        pretrain_learning_rate=float(training["pretrain_learning_rate"]),
        finetune_learning_rate=float(training["finetune_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        mask_probability=float(training["mask_probability"]),
        auxiliary_loss_weight=float(training["auxiliary_loss_weight"]),
        boundary_band_rows=int(training["boundary_band_rows"]),
    )
    result.validate()
    return result


def authorize_entry(
    *, root: Path, data_dir: Path, requested_config: Path, requested_artifact: Path
) -> tuple[dict[str, Any], dict[str, Path], dict[str, dict[str, Any]]]:
    _enforce_execution_tombstone(root)
    root = root.resolve(strict=True)
    paths = _paths(root)
    if requested_config.resolve(strict=True) != paths["config"].resolve(strict=True):
        raise PermissionError("non-canonical config path is forbidden")
    if requested_artifact.resolve(strict=False) != paths["artifact"].resolve(strict=False):
        raise PermissionError("non-canonical artifact path is forbidden")
    content = paths["config"].read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise PermissionError("canonical config byte SHA differs")
    config = json.loads(content)
    if shared._deep_sha(config) != EXPECTED_CONFIG_DEEP_SHA256:
        raise PermissionError("canonical config deep JSON differs")
    if config["experiment_id"] != "p1_masked_pretrain_binary_event_v4":
        raise PermissionError("experiment identity differs")
    if config.get("comparison_mode") != "EXACT_OFFICIAL_PREFIX_REFIT":
        raise PermissionError("P1 Gen4 comparison mode must remain exact")
    if config["canonical_paths"] != {
        "config": CANONICAL_CONFIG,
        "base_config": "configs/p1.toml",
        "goal_contract": "configs/goals/meaningful_score_maximization_v3.json",
        "feature_cache": "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
        "feature_metadata": "artifacts/cache/train_offline_e9fe1eb46cb7431f.json",
        "gen1_artifact": "artifacts/p1_meaningful_learning_curve_generation_v1",
        "gen2_artifact": "artifacts/p1_station_layer_temporal_convolution_event_v2",
        "gen3_artifact": "artifacts/p1_binary_event_tcn_dense_natural_v3",
        "frozen_oof": "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
        "artifact": CANONICAL_ARTIFACT,
        "attempt_lock": CANONICAL_LOCK,
    }:
        raise PermissionError("canonical path contract differs")
    if [item["id"] for item in config["hypotheses"]] != [HYPOTHESIS]:
        raise PermissionError("single registered hypothesis differs")
    if tuple(config["prefix_fractions"]) != FRACTIONS or tuple(config["seeds"]) != SEEDS:
        raise PermissionError("prefix or seed contract differs")
    training = config["training"]
    if not (
        training["optimizer_steps_per_cell"] == 120
        and training["pretrain_steps_per_cell"] == 30
        and training["finetune_steps_per_cell"] == 90
        and training["batch_size"] == 8192
        and training["pretraining_label_reads"] == 0
        and training["expected_curve_fit_cells"] == 45
        and training["expected_curve_optimizer_steps"] == 5400
        and training["expected_curve_pretrain_steps"] == 1350
        and training["expected_curve_finetune_steps"] == 4050
        and training["main_event_loss"]
        == "unweighted_BCEWithLogits_on_dense_natural_prefix_rows"
        and training["phase_balanced_resampling"] is False
        and training["natural_prior_probability_correction"] is False
        and training["hyperparameter_search"] is False
    ):
        raise PermissionError("fixed masked-pretrain/fine-tune contract differs")
    model = config["model"]
    if not (
        model["pretraining_head"] == "masked_center_feature_reconstruction"
        and model["probability_rule"] == "sigmoid(binary_event_logit)_only"
        and model["reconstruction_or_auxiliary_probability_use_forbidden"] is True
        and model["inference_head"] == "binary_event"
        and model["auxiliary_heads"] == ["onset", "offset"]
    ):
        raise PermissionError("Gen4 head/inference contract differs")
    if not all(value is True for value in config["prohibitions"].values()):
        raise PermissionError("all prohibitions must remain enabled")
    features = config["features"]
    if not (
        features["within_prefix_unlabeled_centered_context_allowed"] is True
        and features["out_of_prefix_context_zero_masked"] is True
        and features["scaler"]
        == "exact_prefix_train_only_componentwise_median_iqr_plus_finite_mask"
    ):
        raise PermissionError("Gen4 exact-prefix feature context contract differs")
    implementations = {
        "masked_pretrain_module": root / "src/p1_qc/masked_pretrain_binary_tcn.py",
        "binary_event_module": root / "src/p1_qc/binary_event_tcn.py",
        "shared_temporal_layout_module": root / "src/p1_qc/temporal_event_tcn.py",
        "shared_gen3_adapter": _GEN3_ADAPTER,
        "shared_gen2_runner": root
        / "scripts/run_p1_station_layer_temporal_convolution_event_v2.py",
        "gen1_runner": root / "scripts/run_p1_meaningful_learning_curve_generation_v1.py",
        "base_config": paths["base_config"],
        "pipeline": root / "src/p1_qc/pipeline.py",
        "validation": root / "src/p1_qc/validation.py",
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score_v3.py",
    }
    for name, path in implementations.items():
        if shared._sha(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    pins = shared._verify_input_pins(root, data_dir, config)
    return config, paths, pins


def _json_new(path: Path, value: Any) -> None:
    if isinstance(value, dict) and path.name == "predictions_complete.json":
        value = {
            **value,
            "pretrain_optimizer_steps": 1350,
            "finetune_optimizer_steps": 4050,
            "pretraining_label_reads": 0,
        }
    elif isinstance(value, dict) and path.name == "learning_curve_evidence.json":
        leakage = {
            **value["leakage_checks"],
            "masked_pretraining_completed_before_prefix_label_target_construction": True,
            "masked_pretraining_uses_exact_prefix_features_only": True,
            "reconstruction_and_auxiliary_heads_excluded_from_inference": True,
        }
        reproducibility = {
            **value["reproducibility_checks"],
            "fixed_1350_pretrain_and_4050_finetune_optimizer_steps": True,
        }
        value = {
            **value,
            "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT",
            "leakage_checks": leakage,
            "reproducibility_checks": reproducibility,
        }
    elif isinstance(value, dict) and path.name == "result.json":
        value = {
            **value,
            "operation_counters": {
                **value["operation_counters"],
                "curve_pretrain_optimizer_steps": 1350,
                "curve_finetune_optimizer_steps": 4050,
                "pretraining_label_reads": 0,
            },
        }
    _SHARED_JSON_NEW(path, value)


def _deferred_prefix_labels(train: Any, train_ids: np.ndarray) -> tuple[Any, dict[str, bool]]:
    state = {"materialized": False}

    def load() -> np.ndarray:
        if state["materialized"]:
            raise RuntimeError("prefix labels were requested more than once")
        labels = np.full(len(train), -1, dtype=np.int8)
        labels[train_ids] = shared.pd.to_numeric(
            train.iloc[train_ids]["label"], errors="raise"
        ).to_numpy(np.int8)
        state["materialized"] = True
        return labels

    return load, state


def _run_curve(
    *,
    root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: Any,
    features: np.ndarray,
    feature_columns: list[str],
    layout: Any,
    folds: list[dict[str, Any]],
    prefix_ids: dict[tuple[str, float], np.ndarray],
    comparator_parts: dict[tuple[str, float], Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del root
    model_config = _model_config(config, len(feature_columns), layout.group_count)
    training_config = _training_config(config)
    receipts: list[dict[str, Any]] = []
    part_receipts: list[dict[str, Any]] = []
    completed = 0
    for fraction in FRACTIONS:
        for fold in folds:
            fold_name = str(fold["name"])
            train_ids = prefix_ids[(fold_name, fraction)]
            validation_ids = fold["val_idx"]
            comparator = shared._comparator_frame(
                comparator_parts[(fold_name, fraction)], fold, fraction
            )
            scaler = shared.PrefixRobustScaler.fit(
                features, train_ids, forbidden_ids=validation_ids
            )
            seed_probabilities: list[np.ndarray] = []
            for seed in SEEDS:
                shared._emit(
                    "fit_cell_start",
                    completed_before=completed,
                    total=45,
                    fraction=fraction,
                    fold=fold_name,
                    seed=seed,
                    train_rows=len(train_ids),
                    validation_rows=len(validation_ids),
                    pretrain_steps=training_config.pretrain_steps,
                    finetune_steps=training_config.finetune_steps,
                    prefix_labels_materialized=False,
                )
                started = time.perf_counter()
                deferred_labels, label_state = _deferred_prefix_labels(train, train_ids)
                fitted = fit_masked_pretrain_binary_event_model(
                    features,
                    train.loc[:, ["station", "layer", "time"]],
                    deferred_labels,
                    layout,
                    train_ids,
                    forbidden_ids=validation_ids,
                    seed=seed,
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                    scaler=scaler,
                )
                if not label_state["materialized"] or not fitted.labels_materialized_after_pretraining:
                    raise RuntimeError("prefix labels were not materialized after pretraining")
                probability = predict_masked_pretrain_binary_probability(
                    fitted,
                    features,
                    layout,
                    validation_ids,
                    device="cuda",
                    batch_size=4096,
                )
                model_relative = f"models/{shared._tag(fraction)}/{fold_name}/seed_{seed}.pt"
                model_path = shared._safe_path(paths["artifact"], model_relative)
                save_fitted_masked_pretrain_model(fitted, model_path)
                blind_relative = (
                    f"blind_predictions/{shared._tag(fraction)}/{fold_name}/seed_{seed}.npy"
                )
                blind_path = shared._safe_path(paths["artifact"], blind_relative)
                blind_sha = shared._npy_new(blind_path, probability)
                reloaded = load_fitted_masked_pretrain_model(model_path)
                reproduced = predict_masked_pretrain_binary_probability(
                    reloaded,
                    features,
                    layout,
                    validation_ids,
                    device="cuda",
                    batch_size=4096,
                )
                reload_exact = bool(np.array_equal(probability, reproduced))
                if not reload_exact:
                    raise RuntimeError("saved Gen4 model did not reproduce blind probabilities")
                seed_probabilities.append(probability)
                completed += 1
                receipts.append(
                    {
                        "fraction": fraction,
                        "fold": fold_name,
                        "seed": seed,
                        "train_rows": int(len(train_ids)),
                        "validation_rows": int(len(validation_ids)),
                        "optimizer_steps": training_config.optimizer_steps,
                        "pretrain_optimizer_steps": training_config.pretrain_steps,
                        "finetune_optimizer_steps": training_config.finetune_steps,
                        "pretraining_label_reads": 0,
                        "labels_materialized_after_pretraining": True,
                        "train_ids_sha256": fitted.train_ids_sha256,
                        "validation_ids_sha256": shared.ids_sha256(validation_ids),
                        "phase_counts": list(fitted.phase_counts),
                        "natural_priors": fitted.natural_priors.tolist(),
                        "sampling_priors": fitted.sampling_priors.tolist(),
                        "mean_training_loss": fitted.mean_training_loss,
                        "mean_pretrain_loss": fitted.mean_pretrain_loss,
                        "mean_finetune_loss": fitted.mean_finetune_loss,
                        "mean_event_loss": fitted.mean_event_loss,
                        "mean_auxiliary_loss": fitted.mean_auxiliary_loss,
                        "model_relative_path": model_relative,
                        "model_sha256": shared._sha(model_path),
                        "model_state_sha256": fitted.model_state_sha256,
                        "scaler_sha256": fitted.scaler.state_sha256,
                        "blind_prediction_relative_path": blind_relative,
                        "blind_prediction_sha256": blind_sha,
                        "blind_prediction_sealed_before_validation_target_read": True,
                        "saved_model_reload_prediction_exact": reload_exact,
                        "elapsed_seconds": float(time.perf_counter() - started),
                        "validation_target_reads": 0,
                        "test_value_reads": 0,
                    }
                )
                shared._emit(
                    "fit_cell_complete",
                    completed=completed,
                    total=45,
                    fraction=fraction,
                    fold=fold_name,
                    seed=seed,
                    elapsed_seconds=receipts[-1]["elapsed_seconds"],
                )
            part = comparator.copy()
            for seed, probability in zip(SEEDS, seed_probabilities, strict=True):
                part[f"challenger__seed_{seed}__probability"] = probability
                part[f"challenger__seed_{seed}__prediction"] = shared.apply_postprocess(
                    train.iloc[validation_ids],
                    probability,
                    comparator["plateau"].to_numpy(bool),
                    comparator["spike_candidate"].to_numpy(bool),
                    config["fixed_fold_postprocess"][fold_name],
                )
            mean_probability = np.mean(np.column_stack(seed_probabilities), axis=1)
            part["challenger_probability"] = mean_probability.astype(np.float32)
            part["challenger_prediction"] = shared.apply_postprocess(
                train.iloc[validation_ids],
                mean_probability,
                comparator["plateau"].to_numpy(bool),
                comparator["spike_candidate"].to_numpy(bool),
                config["fixed_fold_postprocess"][fold_name],
            )
            part_relative = f"prediction_parts/{fold_name}_{shared._tag(fraction)}.parquet"
            part_path = shared._safe_path(paths["artifact"], part_relative)
            part_sha = shared._parquet_new(part_path, part)
            part_receipts.append(
                {
                    "fraction": fraction,
                    "fold": fold_name,
                    "rows": int(len(part)),
                    "path": part_relative,
                    "sha256": part_sha,
                    "key_order_sha256": hashlib.sha256(
                        shared.pd.util.hash_pandas_object(
                            part.loc[:, [*shared.KEY_COLUMNS, "fold"]], index=False
                        ).to_numpy("<u8").tobytes()
                    ).hexdigest(),
                }
            )
    total_steps = sum(row["optimizer_steps"] for row in receipts)
    total_pretrain = sum(row["pretrain_optimizer_steps"] for row in receipts)
    total_finetune = sum(row["finetune_optimizer_steps"] for row in receipts)
    if (completed, total_steps, total_pretrain, total_finetune) != (45, 5400, 1350, 4050):
        raise AssertionError("Gen4 fit-cell or stage optimizer-step count differs")
    completion = {
        "schema_version": "p1_masked_pretrain_predictions_complete.v4",
        "created_at": shared._now(),
        "fit_cells": completed,
        "optimizer_steps": total_steps,
        "pretrain_optimizer_steps": total_pretrain,
        "finetune_optimizer_steps": total_finetune,
        "pretraining_label_reads": 0,
        "prediction_parts": part_receipts,
        "model_receipts": receipts,
        "all_blind_predictions_sealed_before_validation_target_read": True,
        "aggregate_scores_computed_before_completion": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    shared._json_new(paths["artifact"] / "predictions_complete.json", completion)
    return receipts, completion


def _full_fit_models(
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: Any,
    features: np.ndarray,
    feature_columns: list[str],
    layout: Any,
) -> dict[str, Any]:
    full_ids = np.arange(len(train), dtype=np.int64)
    scaler = shared.PrefixRobustScaler.fit(features, full_ids)
    model_config = _model_config(config, len(feature_columns), layout.group_count)
    training_config = _training_config(config)
    models: list[dict[str, Any]] = []
    for seed in SEEDS:
        deferred_labels, label_state = _deferred_prefix_labels(train, full_ids)
        fitted = fit_masked_pretrain_binary_event_model(
            features,
            train.loc[:, ["station", "layer", "time"]],
            deferred_labels,
            layout,
            full_ids,
            forbidden_ids=None,
            seed=seed,
            device="cuda",
            model_config=model_config,
            training_config=training_config,
            scaler=scaler,
        )
        if not label_state["materialized"] or not fitted.labels_materialized_after_pretraining:
            raise RuntimeError("full-fit labels were not deferred until after pretraining")
        relative = f"full_fit/seed_{seed}.pt"
        path = shared._safe_path(paths["artifact"], relative)
        save_fitted_masked_pretrain_model(fitted, path)
        loaded = load_fitted_masked_pretrain_model(path)
        if loaded.model_state_sha256 != fitted.model_state_sha256:
            raise RuntimeError("full-fit Gen4 model state differs after reload")
        models.append(
            {
                "seed": seed,
                "path": relative,
                "sha256": shared._sha(path),
                "model_state_sha256": fitted.model_state_sha256,
                "scaler_sha256": fitted.scaler.state_sha256,
                "train_ids_sha256": fitted.train_ids_sha256,
                "optimizer_steps": training_config.optimizer_steps,
                "pretrain_optimizer_steps": training_config.pretrain_steps,
                "finetune_optimizer_steps": training_config.finetune_steps,
                "pretraining_label_reads": 0,
                "labels_materialized_after_pretraining": True,
            }
        )
    full_steps = sum(row["optimizer_steps"] for row in models)
    full_pretrain = sum(row["pretrain_optimizer_steps"] for row in models)
    full_finetune = sum(row["finetune_optimizer_steps"] for row in models)
    if (len(models), full_steps, full_pretrain, full_finetune) != (3, 360, 90, 270):
        raise AssertionError("Gen4 full-fit model or stage optimizer-step count differs")
    receipt = {
        "performed": True,
        "model_count": len(models),
        "optimizer_steps": full_steps,
        "pretrain_optimizer_steps": full_pretrain,
        "finetune_optimizer_steps": full_finetune,
        "pretraining_label_reads": 0,
        "models": models,
        "feature_columns": feature_columns,
        "test_value_reads": 0,
        "test_prediction_generations": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    shared._json_new(paths["artifact"] / "full_fit_models.json", receipt)
    return receipt


def _patch_shared_engine() -> None:
    shared.__file__ = str(Path(__file__).resolve())
    shared.EXPECTED_CONFIG_SHA256 = EXPECTED_CONFIG_SHA256
    shared.EXPECTED_CONFIG_DEEP_SHA256 = EXPECTED_CONFIG_DEEP_SHA256
    shared.CANONICAL_CONFIG = CANONICAL_CONFIG
    shared.CANONICAL_ARTIFACT = CANONICAL_ARTIFACT
    shared.CANONICAL_LOCK = CANONICAL_LOCK
    shared.HYPOTHESIS = HYPOTHESIS
    shared._paths = _paths
    shared._json_new = _json_new
    shared.authorize_entry = authorize_entry
    shared._model_config = _model_config
    shared._training_config = _training_config
    shared._run_curve = _run_curve
    shared._full_fit_models = _full_fit_models
    shared.TemporalEventModelConfig = MaskedPretrainModelConfig
    shared.FixedStepTrainingConfig = MaskedPretrainTrainingConfig
    shared.fit_fixed_step_temporal_event_model = fit_masked_pretrain_binary_event_model
    shared.predict_temporal_event_probability = predict_masked_pretrain_binary_probability
    shared.save_fitted_temporal_event_model = save_fitted_masked_pretrain_model
    shared.load_fitted_temporal_event_model = load_fitted_masked_pretrain_model
    shared.evaluate_learning_curve = evaluate_learning_curve
    shared.load_contract = load_contract


_patch_shared_engine()


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    result = shared.check_only(root=root, data_dir=data_dir)
    config = json.loads((root / CANONICAL_CONFIG).read_text(encoding="utf-8"))
    return {
        **result,
        "experiment_id": config["experiment_id"],
        "comparison_mode": config["comparison_mode"],
        "pretrain_steps": config["training"]["pretrain_steps_per_cell"],
        "finetune_steps": config["training"]["finetune_steps_per_cell"],
        "pretraining_label_reads": 0,
        "event_probability_only": True,
        "reconstruction_or_auxiliary_probability_use": False,
        "batch_size": config["training"]["batch_size"],
    }


def seal(*, root: Path, data_dir: Path) -> dict[str, Any]:
    return shared.seal(root=root, data_dir=data_dir)


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    return shared.run_experiment(root=root, data_dir=data_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--seal-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    if args.check_only:
        result = check_only(root=root, data_dir=data_dir)
    elif args.seal_only:
        result = seal(root=root, data_dir=data_dir)
    else:
        result = run_experiment(root=root, data_dir=data_dir)
    if not args.run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
