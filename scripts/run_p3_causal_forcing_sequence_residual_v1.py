"""Run the append-only P3 causal-forcing sequence learning curve once.

The runner is intentionally train-only unless every sealed learning-curve gate passes.
This generation inherits a known false historical-frozen-OOF reproduction check from
Gen1, so the central evaluator must fail closed as research evidence and anonymous-test
arrays remain unopened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from ocean_goal.meaningful_score import evaluate_learning_curve, load_contract
from p3_wave.causal_forcing_sequence import (
    CausalForcingSequenceConfig,
    CompactRobustScaler,
    FixedEpochTrainingConfig,
    build_causal_forcing_sequence,
    fit_fixed_epoch_and_predict,
    load_fitted_sequence_model,
    predict_with_fitted_sequence_model,
    save_fitted_sequence_model,
)
from p3_wave.corrected_repeated_forward import (
    OFFICIAL_LEADS,
    CorrectedFold,
    build_corrected_repeated_forward_folds,
)
from p3_wave.meaningful_learning_curve import (
    PREFIX_FRACTIONS,
    central_evidence,
    chronological_prefix_ids,
    evaluate_hypothesis_gate,
    evaluate_point,
)
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.one_shot_guard import acquire_persistent_attempt_lock, safe_new_stage_path
from p3_wave.persistence_shrink import (
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_wave.revin_patch import assign_storm_episodes_from_wave, validate_raw_context
from p3_wave.validation import rmse

_GEN1_RUNNER_PATH = Path(__file__).with_name("run_p3_meaningful_learning_curve_v1.py")
_GEN1_SPEC = importlib.util.spec_from_file_location("p3_gen2_gen1_helpers", _GEN1_RUNNER_PATH)
if _GEN1_SPEC is None or _GEN1_SPEC.loader is None:
    raise ImportError("failed to load pinned Gen1 runner helpers")
gen1 = importlib.util.module_from_spec(_GEN1_SPEC)
sys.modules[_GEN1_SPEC.name] = gen1
_GEN1_SPEC.loader.exec_module(gen1)

EXPECTED_CONFIG_SHA256 = "f9b7b0eb76ca0d152a0e87f4eb4fc30b3d2a1cc929e6697cd1324cd9a59e84ca"
EXPECTED_CONFIG_DEEP_SHA256 = "c71d8743d66bea605017cd88fcdaef769b7bd5e0d5d19ae8c49c4f7fb8b9168e"
CANONICAL_CONFIG_RELATIVE = "configs/experiments/p3_causal_forcing_sequence_residual_v1.json"
CANONICAL_GOAL_RELATIVE = "configs/goals/meaningful_score_maximization_v2.json"
CANONICAL_COMPACT_CACHE_RELATIVE = "artifacts/p3/features_all20_v1"
CANONICAL_SEQUENCE_CACHE_RELATIVE = "artifacts/p3/sequences_all20_v1"
CANONICAL_GEN1_RELATIVE = "artifacts/p3_meaningful_learning_curve_20260823_v1"
CANONICAL_OUTPUT_RELATIVE = "artifacts/p3_causal_forcing_sequence_residual_20260823_v1"
CANONICAL_LOCK_RELATIVE = (
    "artifacts/p3_causal_forcing_sequence_residual_20260823_v1.ATTEMPT_LOCK.json"
)
CANONICAL_EXECUTION_CLAIM_RELATIVE = (
    "artifacts/p3_causal_forcing_sequence_residual_20260823_v1.EXECUTION_CLAIM.json"
)
CANONICAL_DATA_DIR = Path(r"C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast")
HYPOTHESIS = "lead_coupled_causal_48h_sequence_forcing_encoder"
TARGET_COLUMNS = tuple(f"target_{lead}" for lead in OFFICIAL_LEADS)
REFERENCE_PATHS = {
    "gen1_metrics": f"{CANONICAL_GEN1_RELATIVE}/metrics.json",
    "gen1_learning_curve_oof": f"{CANONICAL_GEN1_RELATIVE}/oof/learning_curve_oof.parquet",
    "gen1_manifest": f"{CANONICAL_GEN1_RELATIVE}/manifest.json",
    "gen1_independent_qa": (
        "artifacts/p3_meaningful_learning_curve_20260823_v1_QA/independent_aggregate_audit.json"
    ),
}


def _now() -> str:
    return gen1._now()


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
        "goal": workspace / CANONICAL_GOAL_RELATIVE,
        "compact_cache": workspace / CANONICAL_COMPACT_CACHE_RELATIVE,
        "sequence_cache": workspace / CANONICAL_SEQUENCE_CACHE_RELATIVE,
        "gen1": workspace / CANONICAL_GEN1_RELATIVE,
        "output": workspace / CANONICAL_OUTPUT_RELATIVE,
        "lock": workspace / CANONICAL_LOCK_RELATIVE,
        "claim": workspace / CANONICAL_EXECUTION_CLAIM_RELATIVE,
    }


def authorize_entry(
    *,
    root: Path,
    data_dir: Path,
    requested_config: Path,
    requested_compact_cache: Path,
    requested_sequence_cache: Path,
    requested_gen1: Path,
    requested_output: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """First action: bind exact paths, config bytes/deep JSON, and code/evidence pins."""

    paths = _canonical_paths(root)
    if data_dir.resolve(strict=True) != CANONICAL_DATA_DIR.resolve(strict=True):
        raise PermissionError("non-canonical P3 data directory is forbidden")
    requested = {
        "config": requested_config.resolve(strict=True),
        "compact_cache": requested_compact_cache.resolve(strict=True),
        "sequence_cache": requested_sequence_cache.resolve(strict=True),
        "gen1": requested_gen1.resolve(strict=True),
        "output": requested_output.resolve(strict=False),
    }
    for name in requested:
        if requested[name] != paths[name].resolve(strict=name != "output"):
            raise PermissionError(f"non-canonical {name} path is forbidden")
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
        "output": CANONICAL_OUTPUT_RELATIVE,
        "attempt_lock": CANONICAL_LOCK_RELATIVE,
        "execution_claim": CANONICAL_EXECUTION_CLAIM_RELATIVE,
    }
    if config.get("canonical_paths") != expected_paths:
        raise PermissionError("canonical path fields differ")
    if config.get("experiment_id") != "p3_causal_forcing_sequence_residual_v1":
        raise PermissionError("experiment identity differs")
    if tuple(item["id"] for item in config["hypotheses"]) != (HYPOTHESIS,):
        raise PermissionError("single structural hypothesis differs")
    if config["validation"]["training_prefix_fractions"] != list(PREFIX_FRACTIONS):
        raise PermissionError("prefix curve differs")
    if config["model"]["seed_replicates"] != [20260816, 20260817, 20260818]:
        raise PermissionError("fixed seed contract differs")
    if config["training"] != {
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0002,
        "batch_size": 512,
        "fixed_epochs": 8,
        "precision": "bf16_amp_on_cuda",
        "gradient_clip_norm": 1.0,
        "num_workers": 0,
        "fixed_epoch_permutation": "seed_plus_epoch",
        "warm_start_between_cells": False,
        "early_stopping": False,
        "checkpoint_or_epoch_selection": False,
        "auxiliary_loss": False,
        "hyperparameter_search": False,
        "expected_fit_cells": 45,
        "expected_optimizer_steps": 6840,
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

    implementation_paths = {
        "gen1_runner": root / "scripts/run_p3_meaningful_learning_curve_v1.py",
        "corrected_split_module": root / "src/p3_wave/corrected_repeated_forward.py",
        "learning_curve_module": root / "src/p3_wave/meaningful_learning_curve.py",
        "sequence_module": root / "src/p3_wave/sequences.py",
        "revin_preparation_module": root / "src/p3_wave/revin_patch.py",
        "causal_forcing_analog_module": root / "src/p3_wave/causal_forcing_analog.py",
        "causal_forcing_sequence_module": root / "src/p3_wave/causal_forcing_sequence.py",
        "models_module": root / "src/p3_wave/models.py",
        "persistence_shrink_module": root / "src/p3_wave/persistence_shrink.py",
        "one_shot_guard_module": root / "src/p3_wave/one_shot_guard.py",
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score.py",
    }
    for name, path in implementation_paths.items():
        if gen1.base.sha256_file(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    for name, relative in REFERENCE_PATHS.items():
        if gen1.base.sha256_file(root / relative) != config["reference_evidence_sha256"][name]:
            raise PermissionError(f"reference evidence SHA differs: {name}")
    return config, paths


def _resolved_input_paths(*, root: Path, data_dir: Path, paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "source/train_wave.csv": data_dir / "train_wave.csv",
        "source/train_atmos.csv": data_dir / "train_atmos.csv",
        "source/test_context.parquet": data_dir / "test_context.parquet",
        "source/test_index.csv": data_dir / "test_index.csv",
        "source/sample_submission.csv": data_dir / "sample_submission.csv",
        "source/baseline_persistence.csv": data_dir / "baseline_persistence.csv",
        "compact_cache/manifest.json": paths["compact_cache"] / "manifest.json",
        "compact_cache/train_features.parquet": paths["compact_cache"] / "train_features.parquet",
        "compact_cache/train_anchors.parquet": paths["compact_cache"] / "train_anchors.parquet",
        "sequence_cache/manifest.json": paths["sequence_cache"] / "manifest.json",
        "sequence_cache/train_values.npy": paths["sequence_cache"] / "train_values.npy",
        "sequence_cache/train_station.npy": paths["sequence_cache"] / "train_station.npy",
        "gen1/metrics.json": paths["gen1"] / "metrics.json",
        "gen1/learning_curve_oof.parquet": paths["gen1"] / "oof/learning_curve_oof.parquet",
        "gen1/manifest.json": paths["gen1"] / "manifest.json",
        "gen1/independent_qa.json": root
        / "artifacts/p3_meaningful_learning_curve_20260823_v1_QA/independent_aggregate_audit.json",
        "frozen/current_submission.csv": root
        / "submissions/p3_long_persistence_shrink/submission.csv",
        "frozen/current_manifest.json": root
        / "submissions/p3_long_persistence_shrink/manifest.json",
        "current/ready_submission.csv": root / "output/2026-08-20/ready/P3_submission.csv",
    }


def _verify_input_hashes(inputs: dict[str, Path], expected: dict[str, str]) -> dict[str, str]:
    if set(inputs) != set(expected):
        raise ValueError("input hash contract keys differ")
    observed: dict[str, str] = {}
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"pinned input missing: {name}")
        observed[name] = gen1.base.sha256_file(path)
        if observed[name] != expected[name]:
            raise PermissionError(f"pinned input SHA differs: {name}")
    return observed


def _prefix_id_sha(ids: np.ndarray) -> str:
    values = np.asarray(ids, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _preflight(
    *, root: Path, data_dir: Path, config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    inputs = _resolved_input_paths(root=root, data_dir=data_dir, paths=paths)
    snapshot = _verify_input_hashes(inputs, config["expected_sha256"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("canonical Gen2 requires exactly one visible CUDA GPU")

    features = pd.read_parquet(paths["compact_cache"] / "train_features.parquet")
    anchors = pd.read_parquet(paths["compact_cache"] / "train_anchors.parquet")
    if len(features) != 24_360 or len(anchors) != 24_360:
        raise ValueError("training cache row contract differs")
    if not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
        raise ValueError("training feature/anchor keys differ")
    expected_ids = np.arange(len(anchors), dtype=np.int64)
    if not np.array_equal(anchors["anchor_id"].to_numpy(np.int64), expected_ids):
        raise ValueError("anchor_id must be the exact sequence-cache row index")
    feature_columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    if len(feature_columns) != config["features"]["compact_feature_count"]:
        raise ValueError("compact feature count differs")

    raw = np.load(paths["sequence_cache"] / "train_values.npy", mmap_mode="r")
    station = np.load(paths["sequence_cache"] / "train_station.npy", mmap_mode="r")
    if raw.shape != (24_360, 289, 10) or raw.dtype != np.float32:
        raise ValueError("train raw-sequence contract differs")
    if station.shape != (24_360,) or station.dtype != np.int64:
        raise ValueError("train station-code contract differs")
    validate_raw_context(torch.from_numpy(np.array(raw[:8], copy=True)))
    if not np.array_equal(station, anchors["station"].map({"G-ORS": 0, "I-ORS": 1, "S-ORS": 2})):
        raise ValueError("sequence station codes differ from anchor keys")

    wave = pd.read_csv(data_dir / "train_wave.csv")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split_audit = build_corrected_repeated_forward_folds(
        anchors,
        windows=config["validation"]["windows"],
        gap_hours=config["validation"]["gap_hours"],
        footprint_hours=config["validation"]["footprint_hours"],
    )
    if len(selected) != 181 or split_audit["validation_row_count"] != 1_086:
        raise ValueError("corrected validation surface differs")

    prefix_ids: dict[float, dict[str, np.ndarray]] = {}
    prefix_audit: dict[str, Any] = {}
    anchor_lookup = anchors.set_index("anchor_id")
    for fraction in PREFIX_FRACTIONS:
        prefix_ids[fraction] = {}
        tag = f"{int(round(fraction * 100)):03d}"
        prefix_audit[tag] = {}
        for fold in folds:
            ids = chronological_prefix_ids(anchors, fold.train_ids, fraction)
            prefix_ids[fraction][fold.name] = ids
            times = pd.to_datetime(anchor_lookup.loc[ids, "anchor_time"], utc=True)
            gap = float(
                (pd.Timestamp(fold.validation_start) - times.max()).total_seconds() / 3600.0
            )
            prefix_audit[tag][fold.name] = {
                "fraction": float(fraction),
                "count": int(len(ids)),
                "full_count": int(len(fold.train_ids)),
                "id_sha256_little_endian_int64": _prefix_id_sha(ids),
                "nested_subset_of_safe_outer_train": bool(np.isin(ids, fold.train_ids).all()),
                "maximum_anchor_before_validation_start_hours": gap,
            }
    leakage_checks = {
        "station_global_validation_gap_at_least_78h": all(
            value >= 78.0 for value in split_audit["station_global_minimum_gap_hours"].values()
        ),
        "validation_station_episode_reuse_zero": split_audit["repeated_station_episode_count"] == 0,
        "validation_72h_footprint_overlap_zero": split_audit[
            "context48_plus_target24_footprint_overlap_pairs"
        ]
        == 0,
        "outer_train_validation_episode_overlap_zero": all(
            row["shared_train_validation_station_episode_count"] == 0
            for row in split_audit["folds"].values()
        ),
        "outer_train_validation_gap_at_least_78h": all(
            row["minimum_train_validation_anchor_gap_hours"] >= 78.0
            for row in split_audit["folds"].values()
        ),
        "all_prefixes_nested_in_safe_outer_train": all(
            row["nested_subset_of_safe_outer_train"]
            for current in prefix_audit.values()
            for row in current.values()
        ),
    }
    if not all(leakage_checks.values()):
        raise AssertionError("preflight leakage checks failed")

    gen1_oof = pd.read_parquet(paths["gen1"] / "oof/learning_curve_oof.parquet")
    required_oof = {
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
    if not required_oof.issubset(gen1_oof.columns) or len(gen1_oof) != 5 * 1_086:
        raise ValueError("sealed Gen1 comparator OOF contract differs")
    keys = ["prefix_fraction", "fold", "anchor_id", "station", "lead_h"]
    if gen1_oof.duplicated(keys).any():
        raise ValueError("sealed Gen1 comparator keys are duplicated")
    for fraction in PREFIX_FRACTIONS:
        current = gen1_oof.loc[gen1_oof["prefix_fraction"].eq(fraction)]
        if len(current) != 1_086 or current["anchor_id"].nunique() != 181:
            raise ValueError("sealed Gen1 prefix surface differs")

    target = anchors.loc[:, TARGET_COLUMNS].to_numpy(np.float32)
    current_hs = anchors["current_hs"].to_numpy(np.float32)
    delta = target - current_hs[:, None]
    if not np.isfinite(delta).all():
        raise ValueError("training target deltas are non-finite")
    weights = threshold_case_weights(current_hs).astype(np.float32)
    compact = features.loc[:, feature_columns].to_numpy(np.float32)
    gen1_metrics = json.loads((paths["gen1"] / "metrics.json").read_text(encoding="utf-8"))
    gen1_protocol = json.loads(
        (paths["gen1"] / "learning_curve_evidence.json").read_text(encoding="utf-8")
    )["curve_protocol"]
    if gen1_protocol["incumbent_reference_seed_full_prediction_exact_to_frozen_oof"] is not False:
        raise ValueError("Gen1 inherited fail-close reference fact unexpectedly changed")

    expected_steps = sum(
        math.ceil(len(prefix_ids[fraction][fold.name]) / config["training"]["batch_size"])
        * config["training"]["fixed_epochs"]
        * len(config["model"]["seed_replicates"])
        for fraction in PREFIX_FRACTIONS
        for fold in folds
    )
    if expected_steps != config["training"]["expected_optimizer_steps"]:
        raise ValueError("optimizer-step preregistration differs from exact split counts")
    return {
        "inputs": inputs,
        "snapshot": snapshot,
        "features": features,
        "anchors": anchors,
        "feature_columns": feature_columns,
        "compact": compact,
        "raw": raw,
        "station": station,
        "target_delta": delta,
        "case_weight": weights,
        "folds": folds,
        "selected": selected,
        "split_audit": split_audit,
        "prefix_ids": prefix_ids,
        "prefix_audit": prefix_audit,
        "leakage_checks": leakage_checks,
        "gen1_oof": gen1_oof,
        "gen1_metrics": gen1_metrics,
        "gen1_protocol": gen1_protocol,
        "expected_optimizer_steps": expected_steps,
    }


def _protected_roots(root: Path, data_dir: Path, paths: dict[str, Path]) -> tuple[Path, ...]:
    return (
        data_dir,
        paths["compact_cache"],
        paths["sequence_cache"],
        paths["gen1"],
        root / "submissions",
        root / "output",
        root / "데이터셋 원본",
    )


def _write_npy_exclusive(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return gen1.base.sha256_file(path)


def _postprocess(delta: np.ndarray, current_hs: np.ndarray) -> np.ndarray:
    values = np.asarray(delta, dtype=np.float64)
    current = np.asarray(current_hs, dtype=np.float64)
    if values.shape != (len(current), len(OFFICIAL_LEADS)):
        raise ValueError("sequence prediction must align to six leads")
    prediction = np.clip(current[:, None] + values, 0.0, 30.0)
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
    gen1_oof: pd.DataFrame, *, fraction: float, fold: CorrectedFold
) -> pd.DataFrame:
    frame = gen1_oof.loc[
        gen1_oof["prefix_fraction"].eq(fraction) & gen1_oof["fold"].astype(str).eq(fold.name),
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
        raise ValueError("Gen1 comparator anchor keys differ from corrected fold")
    if not np.array_equal(frame["lead_h"].to_numpy(int), expected_leads):
        raise ValueError("Gen1 comparator lead keys differ")
    return frame


def _run_curve(
    *,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    preflight: dict[str, Any],
    stage: Path,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]], list[dict[str, Any]]]:
    del data_dir
    raw = preflight["raw"]
    station = preflight["station"]
    compact = preflight["compact"]
    target_delta = preflight["target_delta"]
    case_weight = preflight["case_weight"]
    anchors = preflight["anchors"]
    folds: tuple[CorrectedFold, ...] = preflight["folds"]
    protected = _protected_roots(root, CANONICAL_DATA_DIR, paths)
    model_config = CausalForcingSequenceConfig()
    training_config = FixedEpochTrainingConfig(
        epochs=8,
        batch_size=512,
        learning_rate=0.0003,
        weight_decay=0.0002,
        gradient_clip_norm=1.0,
    )
    print(json.dumps({"phase": "build_train_only_causal_forcing_cache"}), flush=True)
    forcing_started = time.perf_counter()
    forcing = build_causal_forcing_sequence(raw, window_steps=37)
    if forcing.shape != (24_360, 289, 12) or forcing.dtype != np.float32:
        raise ValueError("causal forcing cache contract differs")
    forcing_receipt = {
        "shape": list(forcing.shape),
        "dtype": str(forcing.dtype),
        "elapsed_seconds": float(time.perf_counter() - forcing_started),
        "train_only": True,
    }

    points: dict[float, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []
    completed = 0
    for fraction in PREFIX_FRACTIONS:
        prefix_tag = f"p{int(round(fraction * 100)):03d}"
        seed_frames: list[pd.DataFrame] = []
        prefix_scalers = {
            fold.name: CompactRobustScaler.fit(
                compact,
                preflight["prefix_ids"][fraction][fold.name],
                forbidden_ids=fold.validation_ids,
            )
            for fold in folds
        }
        for seed in config["model"]["seed_replicates"]:
            fold_frames: list[pd.DataFrame] = []
            for fold in folds:
                train_ids = preflight["prefix_ids"][fraction][fold.name]
                validation_ids = fold.validation_ids
                if np.intersect1d(train_ids, validation_ids).size:
                    raise AssertionError("train/validation IDs overlap before fit")
                current_hs = (
                    anchors.set_index("anchor_id").loc[validation_ids, "current_hs"].to_numpy(float)
                )
                print(
                    json.dumps(
                        {
                            "phase": "fit_sequence_cell",
                            "completed_before": completed,
                            "total": 45,
                            "prefix": fraction,
                            "seed": seed,
                            "fold": fold.name,
                            "train_cases": len(train_ids),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                started = time.perf_counter()
                prediction_delta, fitted = fit_fixed_epoch_and_predict(
                    raw,
                    station,
                    compact,
                    target_delta,
                    case_weight,
                    train_ids,
                    validation_ids,
                    seed=int(seed),
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                    forcing=forcing,
                    compact_scaler=prefix_scalers[fold.name],
                )
                prediction = _postprocess(prediction_delta, current_hs)
                model_relative = f"models/{prefix_tag}/seed_{seed}/folds/{fold.name}/model.pt"
                model_path = safe_new_stage_path(stage, model_relative, protected_roots=protected)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                save_fitted_sequence_model(fitted, model_path)
                model_sha = gen1.base.sha256_file(model_path)
                blind_relative = f"blind_predictions/{prefix_tag}/seed_{seed}/{fold.name}.npy"
                blind_path = safe_new_stage_path(stage, blind_relative, protected_roots=protected)
                blind_sha = _write_npy_exclusive(blind_path, prediction.astype(np.float64))

                reloaded = load_fitted_sequence_model(model_path, map_location="cpu")
                reload_delta = predict_with_fitted_sequence_model(
                    reloaded,
                    raw,
                    station,
                    compact,
                    validation_ids,
                    device="cuda",
                    batch_size=512,
                    forcing=forcing,
                )
                reload_prediction = _postprocess(reload_delta, current_hs)
                maximum_reload_difference = float(np.max(np.abs(reload_prediction - prediction)))
                reload_exact = bool(np.array_equal(reload_prediction, prediction))
                if not reload_exact:
                    raise RuntimeError(
                        "saved-model reload did not exactly reproduce blind prediction: "
                        f"{maximum_reload_difference}"
                    )

                comparator = _cell_comparator(preflight["gen1_oof"], fraction=fraction, fold=fold)
                comparator["challenger_prediction"] = prediction.reshape(-1)
                fold_frames.append(comparator)
                completed += 1
                steps = (
                    math.ceil(len(train_ids) / training_config.batch_size) * training_config.epochs
                )
                receipts.append(
                    {
                        "prefix_fraction": float(fraction),
                        "seed": int(seed),
                        "fold": fold.name,
                        "train_cases": int(len(train_ids)),
                        "validation_cases": int(len(validation_ids)),
                        "optimizer_steps": int(steps),
                        "train_id_sha256": _prefix_id_sha(train_ids),
                        "validation_id_sha256": _prefix_id_sha(validation_ids),
                        "model_relative_path": model_relative,
                        "model_sha256": model_sha,
                        "blind_prediction_relative_path": blind_relative,
                        "blind_prediction_sha256": blind_sha,
                        "blind_prediction_sealed_before_target_attachment": True,
                        "saved_model_reload_prediction_exact": reload_exact,
                        "saved_model_reload_max_abs_difference_m": maximum_reload_difference,
                        "elapsed_seconds": float(time.perf_counter() - started),
                        "test_or_hidden_value_reads": 0,
                    }
                )
            seed_frames.append(pd.concat(fold_frames, ignore_index=True))

        keys = ["fold", "anchor_id", "station", "lead_h"]
        ordered = [frame.sort_values(keys).reset_index(drop=True) for frame in seed_frames]
        invariant = ["target_hs", "current_hs", "persistence", "incumbent_prediction"]
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
                    "phase": "prefix_complete",
                    "prefix": fraction,
                    "completed_cells": completed,
                    "delta_candidate_minus_incumbent_m": point["delta_candidate_minus_incumbent_m"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if completed != 45 or sum(row["optimizer_steps"] for row in receipts) != 6840:
        raise AssertionError("fit-cell or optimizer-step count differs")
    receipts.insert(0, {"forcing_cache": forcing_receipt})
    return pd.concat(all_frames, ignore_index=True), points, receipts


def _artifact_hashes(stage: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(stage).as_posix(): {
            "bytes": int(path.stat().st_size),
            "sha256": gen1.base.sha256_file(path),
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }


def _verify_persisted_attempt(
    *, paths: dict[str, Path], config: dict[str, Any], attempt: dict[str, Any]
) -> None:
    """Reject private/direct execution unless the canonical O_EXCL receipt exists."""

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
    lock = paths["lock"]
    if not lock.is_file():
        raise FileNotFoundError("canonical persistent attempt lock is absent")
    if gen1.base.sha256_file(lock) != attempt["sha256"]:
        raise PermissionError("canonical persistent attempt lock SHA differs")
    persisted = json.loads(lock.read_text(encoding="utf-8"))
    if persisted != {key: value for key, value in attempt.items() if key != "sha256"}:
        raise PermissionError("in-memory attempt does not equal persisted canonical lock")
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
        requested_compact_cache=paths["compact_cache"],
        requested_sequence_cache=paths["sequence_cache"],
        requested_gen1=paths["gen1"],
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
                "fit_cells": 45,
                "optimizer_steps": preflight["expected_optimizer_steps"],
                "test_value_reads": 0,
                "upload_count": 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_causal_forcing_sequence_v1_", dir=tmp_root))
    curve_oof, points, receipts = _run_curve(
        root=root,
        data_dir=data_dir,
        config=config,
        paths=paths,
        preflight=preflight,
        stage=stage,
    )
    gen1.base._atomic_parquet(stage / "oof/learning_curve_oof.parquet", curve_oof)
    gen1.base._atomic_parquet(
        stage / "validation_keys.parquet",
        preflight["selected"][["fold", "anchor_id", "station", "episode_id"]],
    )

    reproducibility_checks = {
        "canonical_config_path_sha_and_deep_json_equal": True,
        "sealed_gen1_comparator_fresh_refit_each_prefix_and_seed": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
        "same_prefix_ids_for_comparator_and_challenger": True,
        "challenger_fresh_refit_each_prefix_fold_seed": True,
        "fixed_three_seed_prediction_mean_reducer": True,
        "same_metric_clip_and_fixed_0p20_shrink": True,
        "hyperparameter_alpha_shrink_and_weight_search_zero": True,
        "complete_unique_181_case_1086_row_surface_each_prefix": all(
            len(curve_oof.loc[curve_oof["prefix_fraction"].eq(fraction)]) == 1_086
            for fraction in PREFIX_FRACTIONS
        ),
        "all_45_models_and_blind_predictions_saved_and_hashed": len(receipts) == 46,
        "all_saved_models_reload_prediction_exact": all(
            row.get("saved_model_reload_prediction_exact") is True for row in receipts[1:]
        ),
        "blind_predictions_sealed_before_target_attachment": all(
            row.get("blind_prediction_sealed_before_target_attachment") is True
            for row in receipts[1:]
        ),
        "deterministic_fixed_epoch_no_validation_selection": True,
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
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "seed_ids": [int(value) for value in config["model"]["seed_replicates"]],
        "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
        "bootstrap_replicates": int(config["validation"]["bootstrap_replicates"]),
        "bootstrap_cluster": "whole_case",
        "incumbent_fresh_refit_each_prefix": True,
        "challenger_fresh_refit_each_prefix": True,
        "same_fold_keys_metric_postprocess": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
        "inherited_from_gen1_fail_closed": True,
    }
    for point in evidence["points"]:
        fraction = float(point["fraction"])
        point["incumbent_seed_metrics"] = list(points[fraction]["incumbent_seed_metrics"])
        point["challenger_seed_metrics"] = list(points[fraction]["challenger_seed_metrics"])
    central = evaluate_learning_curve(load_contract(root, CANONICAL_GOAL_RELATIVE), evidence)
    if central["passed"] or gate["passed"]:
        raise AssertionError("known false exact-reference check must fail closed")
    gen1.base._atomic_json(stage / "learning_curve_evidence.json", evidence)

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
    input_after = _verify_input_hashes(preflight["inputs"], config["expected_sha256"])
    if input_after != preflight["snapshot"]:
        raise RuntimeError("source/cache/current/frozen inputs changed during run")
    status = "NO_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST_READS"
    metrics = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "interpretation": (
            "Corrected same-surface Gen2 research evidence. This is neither an official hidden "
            "score nor upload authorization; the inherited Gen1 exact-reference mismatch is "
            "preserved and forces the central contract to fail closed."
        ),
        "one_shot_attempt": attempt,
        "one_shot_execution_claim": execution_claim,
        "hypothesis": HYPOTHESIS,
        "points": {str(fraction): points[fraction] for fraction in PREFIX_FRACTIONS},
        "local_gate": gate,
        "central_goal_evaluator": central,
        "split_audit": preflight["split_audit"],
        "prefix_audit": preflight["prefix_audit"],
        "leakage_checks": preflight["leakage_checks"],
        "reproducibility_checks": reproducibility_checks,
        "training_receipts": receipts,
        "access_counters": access,
        "candidate_validation": None,
        "invariants": {
            "append_only": True,
            "model_or_hyperparameter_search_run": False,
            "shrink_alpha_or_weight_micro_tuning_run": False,
            "test_target_or_hidden_label_reads": 0,
            "absolute_test_timestamp_recovered": False,
            "current_or_frozen_submission_mutated": False,
            "official_submission_uploads": 0,
            "team_wide_daily_upload_limit_assumed": True,
            "source_cache_current_frozen_sha_unchanged": True,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    gen1.base._atomic_json(stage / "metrics.json", metrics)
    registry = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "hypotheses": [
            {
                "id": HYPOTHESIS,
                "curve_qualified": False,
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
    gen1.base._atomic_json(stage / "registry.json", registry)

    implementation_paths = {
        "config": paths["config"],
        "runner": Path(__file__).resolve(),
        "gen1_runner": _GEN1_RUNNER_PATH,
        "corrected_split_module": root / "src/p3_wave/corrected_repeated_forward.py",
        "learning_curve_module": root / "src/p3_wave/meaningful_learning_curve.py",
        "causal_forcing_sequence_module": root / "src/p3_wave/causal_forcing_sequence.py",
        "one_shot_guard_module": root / "src/p3_wave/one_shot_guard.py",
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score.py",
        "runner_tests": root / "tests/test_p3_causal_forcing_sequence_residual_v1.py",
        "module_tests": root / "tests/test_p3_causal_forcing_sequence_module.py",
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
            "compact_cache_path": CANONICAL_COMPACT_CACHE_RELATIVE,
            "sequence_cache_path": CANONICAL_SEQUENCE_CACHE_RELATIVE,
            "gen1_path": CANONICAL_GEN1_RELATIVE,
            "output_path": CANONICAL_OUTPUT_RELATIVE,
            "attempt_lock_path": CANONICAL_LOCK_RELATIVE,
            "attempt_lock_sha256": attempt["sha256"],
            "execution_claim_path": CANONICAL_EXECUTION_CLAIM_RELATIVE,
            "execution_claim_sha256": execution_claim["sha256"],
        },
        "implementation_sha256": {
            name: gen1.base.sha256_file(path) for name, path in implementation_paths.items()
        },
        "git": gen1.base._git_state(root),
        "input_sha256_before": preflight["snapshot"],
        "input_sha256_after": input_after,
        "source_cache_current_frozen_unchanged": True,
        "output_files": _artifact_hashes(stage),
        "curve_qualified": False,
        "candidate_created": False,
        "candidate_uploaded": False,
        "official_upload_count": 0,
        "access_counters": access,
    }
    gen1.base._atomic_json(stage / "manifest.json", manifest)
    manifest_sha = gen1.base.sha256_file(stage / "manifest.json")
    (stage / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii", newline="\n"
    )
    if paths["output"].exists():
        raise FileExistsError("canonical output appeared before atomic move")
    stage.replace(paths["output"])
    result = {
        "status": status,
        "artifact_dir": CANONICAL_OUTPUT_RELATIVE,
        "metrics_sha256": gen1.base.sha256_file(paths["output"] / "metrics.json"),
        "oof_sha256": gen1.base.sha256_file(paths["output"] / "oof/learning_curve_oof.parquet"),
        "learning_curve_evidence_sha256": gen1.base.sha256_file(
            paths["output"] / "learning_curve_evidence.json"
        ),
        "registry_sha256": gen1.base.sha256_file(paths["output"] / "registry.json"),
        "manifest_sha256": manifest_sha,
        "candidate_sha256": None,
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
        requested_compact_cache=paths["compact_cache"],
        requested_sequence_cache=paths["sequence_cache"],
        requested_gen1=paths["gen1"],
        requested_output=paths["output"],
    )
    preflight = _preflight(root=root, data_dir=data_dir, config=config, paths=paths)
    return {
        "status": "CANONICAL_CHECK_ONLY_PASS",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "validation_cases": int(len(preflight["selected"])),
        "validation_rows": int(preflight["split_audit"]["validation_row_count"]),
        "fit_cells": 45,
        "optimizer_steps": int(preflight["expected_optimizer_steps"]),
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
        requested_compact_cache=paths["compact_cache"],
        requested_sequence_cache=paths["sequence_cache"],
        requested_gen1=paths["gen1"],
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
