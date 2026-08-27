"""Run the append-only P1 station-layer temporal-convolution Gen2 curve once."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocean_goal.meaningful_score import evaluate_learning_curve, load_contract
from p1_qc.config import load_config
from p1_qc.data import KEY_COLUMNS, load_dataset
from p1_qc.pipeline import apply_postprocess
from p1_qc.temporal_event_tcn import (
    FixedStepTrainingConfig,
    PrefixRobustScaler,
    SequenceLayout,
    TemporalEventModelConfig,
    fit_fixed_step_temporal_event_model,
    ids_sha256,
    load_fitted_temporal_event_model,
    predict_temporal_event_probability,
    save_fitted_temporal_event_model,
)
from p1_qc.validation import paired_block_bootstrap

_GEN1_RUNNER = PROJECT_ROOT / "scripts/run_p1_meaningful_learning_curve_generation_v1.py"
_GEN1_SPEC = importlib.util.spec_from_file_location("p1_gen2_gen1_helpers", _GEN1_RUNNER)
if _GEN1_SPEC is None or _GEN1_SPEC.loader is None:
    raise ImportError("failed to load pinned P1 Gen1 runner")
gen1 = importlib.util.module_from_spec(_GEN1_SPEC)
sys.modules[_GEN1_SPEC.name] = gen1
_GEN1_SPEC.loader.exec_module(gen1)

EXPECTED_CONFIG_SHA256 = "a80109430785f9cad3674ba706fc4455149f0089ffe778a174622e923f9698f1"
EXPECTED_CONFIG_DEEP_SHA256 = "b108f71857b2b81d45647c53abe112a029e9f2577000c019ab88f060465c1711"
CANONICAL_CONFIG = "configs/experiments/p1_station_layer_temporal_convolution_event_v2.json"
CANONICAL_ARTIFACT = "artifacts/p1_station_layer_temporal_convolution_event_v2"
CANONICAL_LOCK = "artifacts/p1_station_layer_temporal_convolution_event_v2.ATTEMPT_LOCK.json"
HYPOTHESIS = "station_layer_centered_tcn_phase_heads"
FOLDS = ("2025_q2", "2025_q3", "2025_q4")
FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
SEEDS = (20260813, 20260829, 20260847)
STATIONS = ("G-ORS", "I-ORS", "S-ORS")


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deep_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _emit(event: str, **values: Any) -> None:
    print(json.dumps({"time_kst": _now(), "event": event, **values}, ensure_ascii=False), flush=True)


def _tag(fraction: float) -> str:
    return f"p{int(round(100 * fraction)):03d}"


def _paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / CANONICAL_CONFIG,
        "artifact": root / CANONICAL_ARTIFACT,
        "lock": root / CANONICAL_LOCK,
        "base_config": root / "configs/p1.toml",
        "goal": root / "configs/goals/meaningful_score_maximization_v2.json",
        "feature_cache": root / "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
        "feature_metadata": root / "artifacts/cache/train_offline_e9fe1eb46cb7431f.json",
        "gen1": root / "artifacts/p1_meaningful_learning_curve_generation_v1",
        "frozen_oof": root / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
    }


def _resolve_input(root: Path, data_dir: Path, name: str) -> Path:
    if name in {"train.csv", "test.csv", "sample_submission.csv", "baseline_rule.csv"}:
        return data_dir / name
    result = (root / name).resolve(strict=True)
    if not result.is_relative_to(root):
        raise PermissionError(f"input path escapes workspace: {name}")
    return result


def _verify_input_pins(
    root: Path, data_dir: Path, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for name, expected in config["immutable_inputs"].items():
        path = _resolve_input(root, data_dir, str(name))
        digest = _sha(path)
        if digest != expected:
            raise PermissionError(f"immutable input SHA differs: {name}")
        observed[str(name)] = {"sha256": digest, "bytes": int(path.stat().st_size)}
    return observed


def authorize_entry(
    *, root: Path, data_dir: Path, requested_config: Path, requested_artifact: Path
) -> tuple[dict[str, Any], dict[str, Path], dict[str, dict[str, Any]]]:
    """Bind canonical paths, full config bytes/deep JSON, code pins, and inputs."""

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
    if _deep_sha(config) != EXPECTED_CONFIG_DEEP_SHA256:
        raise PermissionError("canonical config deep JSON differs")
    if config["experiment_id"] != "p1_station_layer_temporal_convolution_event_v2":
        raise PermissionError("experiment identity differs")
    if config["canonical_paths"] != {
        "config": CANONICAL_CONFIG,
        "base_config": "configs/p1.toml",
        "goal_contract": "configs/goals/meaningful_score_maximization_v2.json",
        "feature_cache": "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
        "feature_metadata": "artifacts/cache/train_offline_e9fe1eb46cb7431f.json",
        "gen1_artifact": "artifacts/p1_meaningful_learning_curve_generation_v1",
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
    if (
        training["optimizer_steps_per_cell"] != 120
        or training["batch_size"] != 1024
        or training["expected_curve_fit_cells"] != 45
        or training["expected_curve_optimizer_steps"] != 5400
        or training["hyperparameter_search"] is not False
    ):
        raise PermissionError("fixed training contract differs")
    if not all(config["prohibitions"].values()):
        raise PermissionError("all prohibitions must remain enabled")
    implementations = {
        "temporal_event_module": root / "src/p1_qc/temporal_event_tcn.py",
        "gen1_runner": _GEN1_RUNNER,
        "base_config": paths["base_config"],
        "pipeline": root / "src/p1_qc/pipeline.py",
        "validation": root / "src/p1_qc/validation.py",
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score.py",
    }
    for name, path in implementations.items():
        if _sha(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    pins = _verify_input_pins(root, data_dir, config)
    return config, paths, pins


def _verify_gen1_parts(root: Path, paths: dict[str, Path]) -> dict[tuple[str, float], Path]:
    completion = _json(paths["gen1"] / "predictions_complete.json")
    if completion["part_count"] != 15 or len(completion["parts"]) != 15:
        raise ValueError("sealed Gen1 comparator part count differs")
    expected_cells = {(fold, fraction) for fold in FOLDS for fraction in FRACTIONS}
    observed: dict[tuple[str, float], Path] = {}
    for item in completion["parts"]:
        cell = (str(item["fold"]), float(item["fraction"]))
        relative = Path(str(item["parquet"]).replace("\\", "/"))
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(paths["gen1"].resolve(strict=True)):
            raise PermissionError("Gen1 comparator part escapes its artifact")
        if _sha(path) != item["parquet_sha256"]:
            raise PermissionError(f"Gen1 comparator parquet SHA differs: {cell}")
        audit_path = path.with_suffix(".json")
        if _sha(audit_path) != item["audit_sha256"]:
            raise PermissionError(f"Gen1 comparator audit SHA differs: {cell}")
        observed[cell] = path
    if set(observed) != expected_cells:
        raise ValueError("sealed Gen1 comparator cell surface differs")
    evidence = _json(paths["gen1"] / "learning_curve_evidence.json")
    protocol = evidence["curve_protocol"]
    if not (
        protocol["incumbent_fresh_refit_each_prefix"]
        and protocol["same_fold_keys_metric_postprocess"]
        and protocol["incumbent_reference_seed_full_prediction_exact_to_frozen_oof"]
    ):
        raise PermissionError("sealed Gen1 exact-comparator facts differ")
    return observed


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    paths = _paths(root)
    config, paths, pins = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_artifact=paths["artifact"],
    )
    parts = _verify_gen1_parts(root, paths)
    metadata = _json(paths["feature_metadata"])
    selected = config["features"]["selected_numeric_columns"]
    if any(column not in metadata["feature_columns"] for column in selected):
        raise ValueError("registered temporal feature is absent from the cache")
    return {
        "status": "CANONICAL_CHECK_ONLY_PASS",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "immutable_pin_count": len(pins),
        "gen1_comparator_parts": len(parts),
        "curve_fit_cells": 45,
        "curve_optimizer_steps": 5400,
        "feature_count": len(selected),
        "receptive_field_rows": 31,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
        "artifact_absent": not paths["artifact"].exists(),
        "attempt_lock_absent": not paths["lock"].exists(),
    }


def seal(*, root: Path, data_dir: Path) -> dict[str, Any]:
    paths = _paths(root)
    config, paths, pins = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_artifact=paths["artifact"],
    )
    if paths["artifact"].exists() or paths["lock"].exists():
        raise FileExistsError("canonical artifact or attempt lock already exists")
    _verify_gen1_parts(root, paths)
    paths["artifact"].mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": "p1_temporal_event_preregistration.v2",
        "created_at": _now(),
        "created_before_first_fit": True,
        "experiment_id": config["experiment_id"],
        "config_path": CANONICAL_CONFIG,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": _sha(Path(__file__).resolve()),
        "implementation_sha256": config["implementation_sha256"],
        "immutable_inputs": pins,
        "hypotheses": config["hypotheses"],
        "prefix_fractions": config["prefix_fractions"],
        "seeds": config["seeds"],
        "model": config["model"],
        "training": config["training"],
        "fixed_fold_postprocess": config["fixed_fold_postprocess"],
        "bootstrap": config["bootstrap"],
        "pass_gates": config["pass_gates"],
        "operation_counters_at_seal": {
            "curve_model_fits": 0,
            "target_fold_scores": 0,
            "test_value_reads": 0,
            "candidate_files": 0,
            "uploads": 0,
        },
    }
    _json_new(paths["artifact"] / "preregistration.json", receipt)
    result = {
        "status": "PREREGISTRATION_SEALED",
        "path": str(paths["artifact"] / "preregistration.json"),
        "sha256": _sha(paths["artifact"] / "preregistration.json"),
    }
    _emit("preregistration_sealed", **result)
    return result


def _acquire_lock(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        "created_at": _now(),
        "status": "ATTEMPT_CONSUMED_ONE_SHOT",
        "experiment_id": config["experiment_id"],
        "canonical_config_sha256": EXPECTED_CONFIG_SHA256,
        "o_excl": True,
        "rerun_forbidden": True,
    }
    _json_new(path, receipt)
    receipt["sha256"] = _sha(path)
    return receipt


def _verify_lock(path: Path, attempt: dict[str, Any]) -> None:
    if not path.is_file() or _sha(path) != attempt["sha256"]:
        raise PermissionError("canonical persistent attempt lock differs")
    persisted = _json(path)
    if persisted != {key: value for key, value in attempt.items() if key != "sha256"}:
        raise PermissionError("in-memory attempt differs from persistent lock")
    if not (attempt["o_excl"] and attempt["rerun_forbidden"]):
        raise PermissionError("one-shot lock semantics differ")


def _safe_path(artifact: Path, relative: str) -> Path:
    path = (artifact / relative).resolve()
    if not path.is_relative_to(artifact.resolve()):
        raise PermissionError("artifact path traversal is forbidden")
    if path.exists():
        raise FileExistsError(path)
    return path


def _npy_new(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha(path)


def _parquet_new(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        frame.to_parquet(handle, index=False)
    return _sha(path)


def _prefixes(
    train: pd.DataFrame, folds: list[dict[str, Any]], cadence_minutes: int
) -> tuple[dict[tuple[str, float], np.ndarray], dict[str, Any]]:
    result: dict[tuple[str, float], np.ndarray] = {}
    audit: dict[str, Any] = {}
    for fraction in FRACTIONS:
        audit[_tag(fraction)] = {}
        for fold in folds:
            ids, row = gen1._safe_prefix(
                train,
                fold["train_idx"],
                fold["time_ns"],
                fraction,
                cadence_minutes=cadence_minutes,
            )
            if np.intersect1d(ids, fold["val_idx"]).size:
                raise AssertionError("prefix training IDs overlap validation IDs")
            result[(fold["name"], fraction)] = ids
            audit[_tag(fraction)][fold["name"]] = {
                **row,
                "id_sha256_little_endian_int64": ids_sha256(ids),
                "validation_id_sha256_little_endian_int64": ids_sha256(fold["val_idx"]),
            }
    return result, audit


def _comparator_frame(path: Path, fold: dict[str, Any], fraction: float) -> pd.DataFrame:
    columns = [
        *KEY_COLUMNS,
        "row_position",
        "fold",
        "fraction",
        "baseline_probability",
        "baseline_prediction",
        *[
            value
            for seed in SEEDS
            for value in (
                f"baseline__seed_{seed}__probability",
                f"baseline__seed_{seed}__prediction",
            )
        ],
        "plateau",
        "spike_candidate",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if len(frame) != len(fold["val_idx"]):
        raise ValueError("comparator validation row count differs")
    if not np.array_equal(frame["row_position"].to_numpy(np.int64), fold["val_idx"]):
        raise ValueError("comparator row IDs differ from the corrected fold")
    if not frame["fold"].eq(fold["name"]).all() or not frame["fraction"].eq(fraction).all():
        raise ValueError("comparator fold or fraction tags differ")
    return frame


def _model_config(config: dict[str, Any], feature_count: int, group_count: int) -> TemporalEventModelConfig:
    model = config["model"]
    result = TemporalEventModelConfig(
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


def _training_config(config: dict[str, Any]) -> FixedStepTrainingConfig:
    training = config["training"]
    result = FixedStepTrainingConfig(
        optimizer_steps=int(training["optimizer_steps_per_cell"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )
    result.validate()
    return result


def _run_curve(
    *,
    root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: pd.DataFrame,
    features: np.ndarray,
    feature_columns: list[str],
    layout: SequenceLayout,
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
            comparator = _comparator_frame(
                comparator_parts[(fold_name, fraction)], fold, fraction
            )
            prefix_labels = np.full(len(train), -1, dtype=np.int8)
            prefix_labels[train_ids] = pd.to_numeric(
                train.iloc[train_ids]["label"], errors="raise"
            ).to_numpy(np.int8)
            scaler = PrefixRobustScaler.fit(features, train_ids, forbidden_ids=validation_ids)
            seed_probabilities: list[np.ndarray] = []
            for seed in SEEDS:
                _emit(
                    "fit_cell_start",
                    completed_before=completed,
                    total=45,
                    fraction=fraction,
                    fold=fold_name,
                    seed=seed,
                    train_rows=len(train_ids),
                    validation_rows=len(validation_ids),
                )
                started = time.perf_counter()
                fitted = fit_fixed_step_temporal_event_model(
                    features,
                    train.loc[:, ["station", "layer", "time"]],
                    prefix_labels,
                    layout,
                    train_ids,
                    forbidden_ids=validation_ids,
                    seed=seed,
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                    scaler=scaler,
                )
                probability = predict_temporal_event_probability(
                    fitted,
                    features,
                    layout,
                    validation_ids,
                    device="cuda",
                    batch_size=4096,
                )
                model_relative = f"models/{_tag(fraction)}/{fold_name}/seed_{seed}.pt"
                model_path = _safe_path(paths["artifact"], model_relative)
                save_fitted_temporal_event_model(fitted, model_path)
                blind_relative = f"blind_predictions/{_tag(fraction)}/{fold_name}/seed_{seed}.npy"
                blind_path = _safe_path(paths["artifact"], blind_relative)
                blind_sha = _npy_new(blind_path, probability)
                reloaded = load_fitted_temporal_event_model(model_path)
                reproduced = predict_temporal_event_probability(
                    reloaded,
                    features,
                    layout,
                    validation_ids,
                    device="cuda",
                    batch_size=4096,
                )
                reload_exact = bool(np.array_equal(probability, reproduced))
                if not reload_exact:
                    raise RuntimeError("saved model did not exactly reproduce blind probabilities")
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
                        "train_ids_sha256": fitted.train_ids_sha256,
                        "validation_ids_sha256": ids_sha256(validation_ids),
                        "phase_counts": list(fitted.phase_counts),
                        "natural_priors": fitted.natural_priors.tolist(),
                        "sampling_priors": fitted.sampling_priors.tolist(),
                        "mean_training_loss": fitted.mean_training_loss,
                        "model_relative_path": model_relative,
                        "model_sha256": _sha(model_path),
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
                _emit(
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
                part[f"challenger__seed_{seed}__prediction"] = apply_postprocess(
                    train.iloc[validation_ids],
                    probability,
                    comparator["plateau"].to_numpy(bool),
                    comparator["spike_candidate"].to_numpy(bool),
                    config["fixed_fold_postprocess"][fold_name],
                )
            mean_probability = np.mean(np.column_stack(seed_probabilities), axis=1)
            part["challenger_probability"] = mean_probability.astype(np.float32)
            part["challenger_prediction"] = apply_postprocess(
                train.iloc[validation_ids],
                mean_probability,
                comparator["plateau"].to_numpy(bool),
                comparator["spike_candidate"].to_numpy(bool),
                config["fixed_fold_postprocess"][fold_name],
            )
            part_relative = f"prediction_parts/{fold_name}_{_tag(fraction)}.parquet"
            part_path = _safe_path(paths["artifact"], part_relative)
            part_sha = _parquet_new(part_path, part)
            part_receipts.append(
                {
                    "fraction": fraction,
                    "fold": fold_name,
                    "rows": int(len(part)),
                    "path": part_relative,
                    "sha256": part_sha,
                    "key_order_sha256": hashlib.sha256(
                        pd.util.hash_pandas_object(
                            part.loc[:, [*KEY_COLUMNS, "fold"]], index=False
                        ).to_numpy("<u8").tobytes()
                    ).hexdigest(),
                }
            )
    if completed != 45 or sum(row["optimizer_steps"] for row in receipts) != 5400:
        raise AssertionError("curve fit-cell or optimizer-step count differs")
    completion = {
        "schema_version": "p1_temporal_event_predictions_complete.v2",
        "created_at": _now(),
        "fit_cells": completed,
        "optimizer_steps": 5400,
        "prediction_parts": part_receipts,
        "model_receipts": receipts,
        "all_blind_predictions_sealed_before_validation_target_read": True,
        "aggregate_scores_computed_before_completion": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    _json_new(paths["artifact"] / "predictions_complete.json", completion)
    return receipts, completion


def _score(
    *,
    root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: pd.DataFrame,
    frozen_oof: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not (paths["artifact"] / "predictions_complete.json").is_file():
        raise FileNotFoundError("complete blind-prediction receipt is absent")
    points: list[dict[str, Any]] = []
    all_keys_exact = True
    full_fold_deltas: list[float] = []
    full_station_deltas: dict[str, float] = {}
    full_station_layer: dict[str, Any] = {}
    reference_seed_exact = False
    for fraction_index, fraction in enumerate(FRACTIONS):
        frames = [
            pd.read_parquet(paths["artifact"] / f"prediction_parts/{fold}_{_tag(fraction)}.parquet")
            for fold in FOLDS
        ]
        combined = pd.concat(frames, ignore_index=True)
        expected_keys = frozen_oof.loc[:, [*KEY_COLUMNS, "fold"]].reset_index(drop=True)
        observed_keys = combined.loc[:, [*KEY_COLUMNS, "fold"]].reset_index(drop=True)
        keys_exact = observed_keys.equals(expected_keys)
        all_keys_exact &= keys_exact
        if not keys_exact:
            raise RuntimeError("candidate OOF key/order differs from frozen OOF")
        row_positions = combined["row_position"].to_numpy(np.int64)
        truth = pd.to_numeric(train.iloc[row_positions]["label"], errors="raise").to_numpy(np.int8)
        incumbent = combined["baseline_prediction"].to_numpy(np.int8)
        challenger = combined["challenger_prediction"].to_numpy(np.int8)
        incumbent_f1 = gen1._binary_f1(truth, incumbent)
        challenger_f1 = gen1._binary_f1(truth, challenger)
        bootstrap = paired_block_bootstrap(
            truth,
            challenger,
            incumbent,
            combined.loc[:, ["station", "layer", "time"]],
            replicates=int(config["bootstrap"]["replicates"]),
            seed=int(config["bootstrap"]["seed"]) + 100 + fraction_index,
            cadence_minutes=10,
            normal_day_timezone="Asia/Seoul",
        )
        fold_metrics = gen1._metric_slices(combined, truth, challenger, incumbent, ["fold"])
        station_metrics = gen1._metric_slices(combined, truth, challenger, incumbent, ["station"])
        station_layer_metrics = gen1._metric_slices(
            combined, truth, challenger, incumbent, ["station", "layer"]
        )
        point = {
            "fraction": fraction,
            "rows": int(len(combined)),
            "incumbent": incumbent_f1,
            "challenger": challenger_f1,
            "delta_candidate_minus_incumbent": challenger_f1 - incumbent_f1,
            "delta_ci90": bootstrap["difference_ci90"],
            "incumbent_seed_metrics": [
                gen1._binary_f1(
                    truth,
                    combined[f"baseline__seed_{seed}__prediction"].to_numpy(np.int8),
                )
                for seed in SEEDS
            ],
            "challenger_seed_metrics": [
                gen1._binary_f1(
                    truth,
                    combined[f"challenger__seed_{seed}__prediction"].to_numpy(np.int8),
                )
                for seed in SEEDS
            ],
            "paired_cluster_bootstrap": bootstrap,
            "folds": fold_metrics,
            "stations": station_metrics,
            "station_layers": station_layer_metrics,
            "key_order_exact": keys_exact,
        }
        points.append(point)
        if fraction == 1.0:
            reference_seed_exact = bool(
                np.array_equal(
                    combined[f"baseline__seed_{SEEDS[0]}__prediction"].to_numpy(np.int8),
                    frozen_oof["prediction"].to_numpy(np.int8),
                )
            )
            full_fold_deltas = [
                float(fold_metrics[name]["delta_candidate_minus_incumbent"]) for name in FOLDS
            ]
            full_station_deltas = {
                station: float(station_metrics[station]["delta_candidate_minus_incumbent"])
                for station in STATIONS
            }
            full_station_layer = station_layer_metrics
    leakage_checks = {
        "validation_target_labels_not_read_before_all_blind_predictions_sealed": True,
        "phase_targets_constructed_only_from_explicit_prefix_train_ids": True,
        "prefix_scaler_fitted_only_on_exact_prefix_train_ids": True,
        "fold_train_validation_positions_disjoint": True,
        "prefix_target_scope_never_after_registered_cutoff": True,
        "feature_cache_excludes_label_and_anomaly_type": True,
        "centered_context_uses_unlabeled_offline_features_only": True,
        "test_values_not_read": True,
        "fixed_postprocess_not_retuned": True,
    }
    completion = _json(paths["artifact"] / "predictions_complete.json")
    reproducibility_checks = {
        "canonical_config_byte_and_deep_json_exact": True,
        "exact_registered_prefixes": list(FRACTIONS) == config["prefix_fractions"],
        "exact_three_registered_seeds": list(SEEDS) == config["seeds"],
        "sealed_gen1_incumbent_fresh_refits_reused_byte_for_byte": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": reference_seed_exact,
        "challenger_fresh_refit_each_prefix_fold_seed": True,
        "same_fold_keys_metric_postprocess": all_keys_exact,
        "all_45_models_and_blind_predictions_saved_and_hashed": len(
            completion["model_receipts"]
        )
        == 45,
        "all_saved_models_reload_probability_exact": all(
            row["saved_model_reload_prediction_exact"] for row in completion["model_receipts"]
        ),
        "fixed_5400_optimizer_steps": completion["optimizer_steps"] == 5400,
        "paired_bootstrap_replicates_exact": int(config["bootstrap"]["replicates"]) == 5000,
    }
    late = {point["fraction"]: point for point in points if point["fraction"] in {0.7, 0.85, 1.0}}
    full = late[1.0]
    gate_checks = {
        "late_fractions_all_improve": all(
            point["delta_candidate_minus_incumbent"] > 0 for point in late.values()
        ),
        "full_fraction_ci90_excludes_zero": float(full["delta_ci90"][0]) > 0,
        "another_late_fraction_ci90_excludes_zero": sum(
            float(late[value]["delta_ci90"][0]) > 0 for value in (0.7, 0.85)
        )
        >= 1,
        "full_effect_at_least_0_020_f1": float(full["delta_candidate_minus_incumbent"]) >= 0.02,
        "minimum_two_of_three_folds_improve": sum(value > 0 for value in full_fold_deltas) >= 2,
        "worst_station_regression_within_0_005": min(full_station_deltas.values()) >= -0.005,
        "all_leakage_checks": all(leakage_checks.values()),
        "all_reproducibility_checks": all(reproducibility_checks.values()),
    }
    report = {
        "schema_version": "p1_temporal_event_curve_metrics.v2",
        "experiment_id": config["experiment_id"],
        "hypothesis": HYPOTHESIS,
        "points": points,
        "full_fraction_fold_deltas_candidate_minus_incumbent": full_fold_deltas,
        "full_fraction_station_deltas_candidate_minus_incumbent": full_station_deltas,
        "full_fraction_station_layer_metrics": full_station_layer,
        "leakage_checks": leakage_checks,
        "reproducibility_checks": reproducibility_checks,
        "gate_checks": gate_checks,
        "passed": all(gate_checks.values()),
        "decision": "PASS" if all(gate_checks.values()) else "RESEARCH_ONLY",
    }
    _json_new(paths["artifact"] / "metrics.json", report)
    prereg = _json(paths["artifact"] / "preregistration.json")
    evidence = {
        "problem": "P1",
        "selected_hypothesis": HYPOTHESIS,
        "selection_status": "QUALIFIED_WINNER" if report["passed"] else "RESEARCH_ONLY_DIAGNOSTIC",
        "preregistration": {
            "generation_id": config["experiment_id"],
            "config_path": CANONICAL_CONFIG,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "created_before_first_fit": prereg["created_before_first_fit"],
            "hypothesis_count": 1,
            "hypothesis_count_at_most_3": True,
            "score_derived_tuning": False,
        },
        "curve_protocol": {
            "prefix_fractions": list(FRACTIONS),
            "seed_ids": list(SEEDS),
            "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
            "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
            "bootstrap_cluster": "event_or_normal_day_by_station_layer",
            "incumbent_fresh_refit_each_prefix": True,
            "challenger_fresh_refit_each_prefix": True,
            "same_fold_keys_metric_postprocess": True,
            "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": reference_seed_exact,
            "frozen_reproduction_reference_seed": SEEDS[0],
        },
        "points": [
            {
                "fraction": point["fraction"],
                "incumbent": point["incumbent"],
                "challenger": point["challenger"],
                "delta_ci90": point["delta_ci90"],
                "incumbent_seed_metrics": point["incumbent_seed_metrics"],
                "challenger_seed_metrics": point["challenger_seed_metrics"],
            }
            for point in points
        ],
        "fold_deltas_candidate_minus_incumbent": full_fold_deltas,
        "slice_deltas_candidate_minus_incumbent": full_station_deltas,
        "leakage_checks": leakage_checks,
        "reproducibility_checks": reproducibility_checks,
        "operation_counters": {"uploads": 0, "source_mutations": 0, "frozen_mutations": 0},
        "validation_scope_caveat": {
            "present": True,
            "meaning": "Frozen event-protected validation assignment can retain a complete positive event tail outside a nominal quarter; train and validation positions remain disjoint and every prefix cutoff is earlier than validation start.",
        },
    }
    _json_new(paths["artifact"] / "learning_curve_evidence.json", evidence)
    central = evaluate_learning_curve(load_contract(root, config["canonical_paths"]["goal_contract"]), evidence)
    _json_new(paths["artifact"] / "canonical_curve_decision.json", central)
    if bool(central["passed"]) != bool(report["passed"]):
        raise RuntimeError("local and canonical gates disagree")
    return report, evidence, central


def _full_fit_models(
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: pd.DataFrame,
    features: np.ndarray,
    feature_columns: list[str],
    layout: SequenceLayout,
) -> dict[str, Any]:
    full_ids = np.arange(len(train), dtype=np.int64)
    labels = pd.to_numeric(train["label"], errors="raise").to_numpy(np.int8)
    scaler = PrefixRobustScaler.fit(features, full_ids)
    model_config = _model_config(config, len(feature_columns), layout.group_count)
    training_config = _training_config(config)
    models: list[dict[str, Any]] = []
    for seed in SEEDS:
        fitted = fit_fixed_step_temporal_event_model(
            features,
            train.loc[:, ["station", "layer", "time"]],
            labels,
            layout,
            full_ids,
            forbidden_ids=None,
            seed=seed,
            device="cuda",
            model_config=model_config,
            training_config=training_config,
            scaler=scaler,
        )
        relative = f"full_fit/seed_{seed}.pt"
        path = _safe_path(paths["artifact"], relative)
        save_fitted_temporal_event_model(fitted, path)
        loaded = load_fitted_temporal_event_model(path)
        if loaded.model_state_sha256 != fitted.model_state_sha256:
            raise RuntimeError("full-fit saved model state differs after reload")
        models.append(
            {
                "seed": seed,
                "path": relative,
                "sha256": _sha(path),
                "model_state_sha256": fitted.model_state_sha256,
                "scaler_sha256": fitted.scaler.state_sha256,
                "train_ids_sha256": fitted.train_ids_sha256,
            }
        )
    receipt = {
        "performed": True,
        "model_count": 3,
        "models": models,
        "feature_columns": feature_columns,
        "test_value_reads": 0,
        "test_prediction_generations": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    _json_new(paths["artifact"] / "full_fit_models.json", receipt)
    return receipt


def _artifact_hashes(artifact: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(artifact).as_posix(): {
            "sha256": _sha(path),
            "bytes": int(path.stat().st_size),
        }
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }


def _run_after_lock(
    *, root: Path, data_dir: Path, config: dict[str, Any], paths: dict[str, Path], attempt: dict[str, Any]
) -> dict[str, Any]:
    authorized, authorized_paths, pins_before = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_artifact=paths["artifact"],
    )
    if authorized != config or authorized_paths != paths:
        raise PermissionError("private/direct call context differs from canonical authorization")
    _verify_lock(paths["lock"], attempt)
    prereg_path = paths["artifact"] / "preregistration.json"
    prereg = _json(prereg_path)
    if (
        prereg["config_sha256"] != EXPECTED_CONFIG_SHA256
        or prereg["runner_sha256"] != _sha(Path(__file__).resolve())
        or prereg["created_before_first_fit"] is not True
    ):
        raise PermissionError("pre-fit preregistration seal differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("canonical P1 Gen2 requires exactly one visible CUDA GPU")
    comparator_parts = _verify_gen1_parts(root, paths)
    started = time.perf_counter()
    train = load_dataset(data_dir / "train.csv", kind="train", audit=False)
    feature_metadata = _json(paths["feature_metadata"])
    feature_columns = [str(value) for value in config["features"]["selected_numeric_columns"]]
    if any(column not in feature_metadata["feature_columns"] for column in feature_columns):
        raise ValueError("registered feature column is absent")
    feature_frame = pd.read_parquet(paths["feature_cache"], columns=feature_columns)
    if len(feature_frame) != len(train) or {"label", "anomaly_type"}.intersection(feature_frame):
        raise ValueError("feature cache row or target-exclusion contract differs")
    features = feature_frame.to_numpy(np.float32)
    layout = SequenceLayout.build(train.loc[:, ["station", "layer", "time"]])
    p1_config = load_config(paths["base_config"])
    frozen_oof = pd.read_parquet(paths["frozen_oof"])
    folds, scope_audit = gen1._fold_runtime(
        train,
        p1_config,
        frozen_oof.loc[:, [*KEY_COLUMNS, "fold", "prediction"]],
    )
    prefix_ids, prefix_audit = _prefixes(train, folds, int(config["features"]["cadence_minutes"]))
    _json_new(
        paths["artifact"] / "split_audit.json",
        {
            "folds": scope_audit,
            "prefixes": prefix_audit,
            "strict_clock_caveat_present": any(
                not value["nominal_wall_clock_scope_exact"] for value in scope_audit.values()
            ),
            "train_validation_positions_disjoint": True,
        },
    )
    receipts, completion = _run_curve(
        root=root,
        config=config,
        paths=paths,
        train=train,
        features=features,
        feature_columns=feature_columns,
        layout=layout,
        folds=folds,
        prefix_ids=prefix_ids,
        comparator_parts=comparator_parts,
    )
    report, evidence, central = _score(
        root=root,
        config=config,
        paths=paths,
        train=train,
        frozen_oof=frozen_oof,
    )
    if central["passed"]:
        full_fit = _full_fit_models(
            config=config,
            paths=paths,
            train=train,
            features=features,
            feature_columns=feature_columns,
            layout=layout,
        )
        next_generation = None
    else:
        full_fit = {
            "performed": False,
            "reason": "curve did not satisfy every preregistered meaningful-improvement gate",
            "model_count": 0,
            "test_value_reads": 0,
            "candidate_files": 0,
            "uploads": 0,
        }
        next_generation = config["on_no_pass"]["exactly_one_next_structural_diagnosis"]
    pins_after = _verify_input_pins(root, data_dir, config)
    if pins_before != pins_after:
        raise RuntimeError("immutable source/cache/current/frozen pins changed during run")
    result = {
        "schema_version": "p1_temporal_event_result.v2",
        "experiment_id": config["experiment_id"],
        "completed_at": _now(),
        "status": "CURVE_QUALIFIED_FULL_FIT_MODELS_SAVED" if central["passed"] else "RESEARCH_ONLY_NO_PASS",
        "decision": central["decision"],
        "passed": bool(central["passed"]),
        "hypothesis": HYPOTHESIS,
        "points": report["points"],
        "gate_checks": report["gate_checks"],
        "full_fit": full_fit,
        "exactly_one_next_structural_diagnosis": next_generation,
        "attempt": attempt,
        "prediction_completion_sha256": _sha(paths["artifact"] / "predictions_complete.json"),
        "operation_counters": {
            "curve_model_fits": len(receipts),
            "curve_optimizer_steps": completion["optimizer_steps"],
            "full_fit_model_fits": int(full_fit["model_count"]),
            "target_fold_scores": len(FRACTIONS),
            "test_value_reads": 0,
            "test_prediction_generations": 0,
            "candidate_files": 0,
            "uploads": 0,
            "source_mutations": 0,
            "frozen_mutations": 0,
        },
        "protected_input_sha256_unchanged": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    _json_new(paths["artifact"] / "result.json", result)
    registry = {
        "schema_version": "p1_temporal_event_registry.v2",
        "experiment_id": config["experiment_id"],
        "registered_at": _now(),
        "decision": central["decision"],
        "passed": bool(central["passed"]),
        "hypothesis": HYPOTHESIS,
        "learning_curve_evidence_sha256": _sha(paths["artifact"] / "learning_curve_evidence.json"),
        "canonical_decision_sha256": _sha(paths["artifact"] / "canonical_curve_decision.json"),
        "result_sha256": _sha(paths["artifact"] / "result.json"),
        "full_fit_models": full_fit.get("models"),
        "candidate": None,
        "test_value_reads": 0,
        "uploads": 0,
    }
    _json_new(paths["artifact"] / "registry.json", registry)
    manifest = {
        "schema_version": "p1_temporal_event_manifest.v2",
        "experiment_id": config["experiment_id"],
        "created_at": _now(),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "runner_sha256": _sha(Path(__file__).resolve()),
        "implementation_sha256": config["implementation_sha256"],
        "attempt_lock_path": CANONICAL_LOCK,
        "attempt_lock_sha256": attempt["sha256"],
        "immutable_inputs_before": pins_before,
        "immutable_inputs_after": pins_after,
        "source_cache_current_frozen_unchanged": True,
        "artifacts": _artifact_hashes(paths["artifact"]),
        "candidate_created": False,
        "uploaded": False,
    }
    _json_new(paths["artifact"] / "manifest.json", manifest)
    manifest_sha = _sha(paths["artifact"] / "manifest.json")
    with (paths["artifact"] / "manifest.sha256").open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{manifest_sha}  manifest.json\n")
    final = {
        "status": result["status"],
        "decision": central["decision"],
        "passed": bool(central["passed"]),
        "artifact": CANONICAL_ARTIFACT,
        "metrics_sha256": _sha(paths["artifact"] / "metrics.json"),
        "evidence_sha256": _sha(paths["artifact"] / "learning_curve_evidence.json"),
        "result_sha256": _sha(paths["artifact"] / "result.json"),
        "manifest_sha256": manifest_sha,
        "candidate_sha256": None,
        "test_value_reads": 0,
        "uploads": 0,
        "elapsed_seconds": result["elapsed_seconds"],
    }
    _emit("generation_complete", **final)
    return final


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    paths = _paths(root)
    config, paths, _ = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_artifact=paths["artifact"],
    )
    if not (paths["artifact"] / "preregistration.json").is_file():
        raise FileNotFoundError("preregistration seal is absent")
    if (paths["artifact"] / "result.json").exists():
        raise FileExistsError("append-only result already exists")
    attempt = _acquire_lock(paths["lock"], config)
    return _run_after_lock(
        root=root,
        data_dir=data_dir,
        config=config,
        paths=paths,
        attempt=attempt,
    )


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
