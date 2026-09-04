"""Run the append-only, train-only P3 nested checkpoint audit.

The command is inert unless exactly one of ``--check-only`` or ``--execute`` is
provided.  It never enumerates or opens anonymous evaluation or submission inputs.
All outer predictions are committed before the historical outer-reference values are
opened for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

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
from p3_wave.causal_forcing_sequence_checkpoint import (
    fit_inner_checkpoint_curve,
    ids_sha256,
    postprocess_sequence_delta,
    select_earliest_ensemble_epoch,
)
from p3_wave.corrected_repeated_forward import (
    OFFICIAL_LEADS,
    CorrectedFold,
    build_corrected_repeated_forward_folds,
)
from p3_wave.meaningful_learning_curve import (
    chronological_prefix_ids,
    evaluate_point,
)
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.one_shot_guard import acquire_persistent_attempt_lock, safe_new_stage_path
from p3_wave.revin_patch import (
    InnerEpisodeSplit,
    assign_storm_episodes_from_wave,
    build_inner_episode_split,
    validate_raw_context,
)
from p3_wave.validation import rmse


EXPECTED_CONFIG_SHA256 = "a11537916b52eb8793ba003177b679c0139d0934c5c1354d0af62892bc611b35"
EXPECTED_CONFIG_DEEP_SHA256 = "5c0c954ffd8ba8e57caa903b357c67dfe93d94055161032205fa0bbc7ab0f8bf"
CANONICAL_CONFIG_RELATIVE = (
    "configs/experiments/p3_causal_forcing_sequence_checkpoint_nested_v2.json"
)
CANONICAL_COMPACT_CACHE_RELATIVE = "artifacts/p3/features_all20_v1"
CANONICAL_SEQUENCE_CACHE_RELATIVE = "artifacts/p3/sequences_all20_v1"
CANONICAL_REFERENCE_RELATIVE = "artifacts/p3_causal_forcing_sequence_residual_20260823_v1"
CANONICAL_OUTPUT_RELATIVE = (
    "artifacts/p3_causal_forcing_sequence_checkpoint_nested_20260827_v2"
)
CANONICAL_LOCK_RELATIVE = f"{CANONICAL_OUTPUT_RELATIVE}.ATTEMPT_LOCK.json"
CANONICAL_CLAIM_RELATIVE = f"{CANONICAL_OUTPUT_RELATIVE}.EXECUTION_CLAIM.json"
TARGET_COLUMNS = tuple(f"target_{lead}" for lead in OFFICIAL_LEADS)
KEYS = ["fold", "anchor_id", "station", "lead_h"]
REGISTERED_PREFIX_FRACTIONS = (1.0,)
EXPLICIT_SOURCE_NAMES = ("README.md", "train_wave.csv", "train_atmos.csv")
EXPLICIT_INPUT_KEYS = (
    "source/README.md",
    "source/train_wave.csv",
    "source/train_atmos.csv",
    "compact_cache/manifest.json",
    "compact_cache/train_features.parquet",
    "compact_cache/train_anchors.parquet",
    "sequence_cache/manifest.json",
    "sequence_cache/train_values.npy",
    "sequence_cache/train_station.npy",
    "fixed8_reference/metrics.json",
    "fixed8_reference/oof/learning_curve_oof.parquet",
    "fixed8_reference/validation_keys.parquet",
    "fixed8_reference/manifest.json",
)


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deep_sha(value: Any) -> str:
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _atomic_json_exclusive(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(content).hexdigest()


def _write_npy_exclusive(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_file(path)


def _write_parquet_exclusive(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_parquet(path, index=False)
    return _sha256_file(path)


def _git_state(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _paths(root: Path) -> dict[str, Path]:
    workspace = root.resolve(strict=True)
    return {
        "config": workspace / CANONICAL_CONFIG_RELATIVE,
        "compact_cache": workspace / CANONICAL_COMPACT_CACHE_RELATIVE,
        "sequence_cache": workspace / CANONICAL_SEQUENCE_CACHE_RELATIVE,
        "reference": workspace / CANONICAL_REFERENCE_RELATIVE,
        "output": workspace / CANONICAL_OUTPUT_RELATIVE,
        "lock": workspace / CANONICAL_LOCK_RELATIVE,
        "claim": workspace / CANONICAL_CLAIM_RELATIVE,
    }


def _implementation_paths(root: Path) -> dict[str, Path]:
    return {
        "checkpoint_module": root / "src/p3_wave/causal_forcing_sequence_checkpoint.py",
        "causal_forcing_sequence_module": root / "src/p3_wave/causal_forcing_sequence.py",
        "corrected_split_module": root / "src/p3_wave/corrected_repeated_forward.py",
        "meaningful_learning_curve_module": root / "src/p3_wave/meaningful_learning_curve.py",
        "inner_split_module": root / "src/p3_wave/revin_patch.py",
        "models_module": root / "src/p3_wave/models.py",
        "persistence_shrink_module": root / "src/p3_wave/persistence_shrink.py",
        "one_shot_guard_module": root / "src/p3_wave/one_shot_guard.py",
    }


def _input_paths(data_dir: Path, paths: dict[str, Path]) -> dict[str, Path]:
    """Return the only source/cache/reference paths this runner may touch."""

    result = {
        "source/README.md": data_dir / "README.md",
        "source/train_wave.csv": data_dir / "train_wave.csv",
        "source/train_atmos.csv": data_dir / "train_atmos.csv",
        "compact_cache/manifest.json": paths["compact_cache"] / "manifest.json",
        "compact_cache/train_features.parquet": paths["compact_cache"]
        / "train_features.parquet",
        "compact_cache/train_anchors.parquet": paths["compact_cache"]
        / "train_anchors.parquet",
        "sequence_cache/manifest.json": paths["sequence_cache"] / "manifest.json",
        "sequence_cache/train_values.npy": paths["sequence_cache"] / "train_values.npy",
        "sequence_cache/train_station.npy": paths["sequence_cache"] / "train_station.npy",
        "fixed8_reference/metrics.json": paths["reference"] / "metrics.json",
        "fixed8_reference/oof/learning_curve_oof.parquet": paths["reference"]
        / "oof/learning_curve_oof.parquet",
        "fixed8_reference/validation_keys.parquet": paths["reference"]
        / "validation_keys.parquet",
        "fixed8_reference/manifest.json": paths["reference"] / "manifest.json",
    }
    if tuple(result) != EXPLICIT_INPUT_KEYS:
        raise AssertionError("explicit train-only input order changed")
    if tuple(path.name for path in result.values() if path.parent == data_dir) != EXPLICIT_SOURCE_NAMES:
        raise AssertionError("source allowlist changed")
    return result


def authorize_entry(
    *, root: Path, data_dir: Path
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Path]]:
    paths = _paths(root)
    if paths["output"].exists():
        raise FileExistsError("append-only v2 output already exists")
    content = paths["config"].read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise PermissionError("v2 config byte SHA differs")
    config = json.loads(content)
    if _deep_sha(config) != EXPECTED_CONFIG_DEEP_SHA256:
        raise PermissionError("v2 config deep JSON differs")
    if config.get("experiment_id") != "p3_causal_forcing_sequence_checkpoint_nested_v2":
        raise PermissionError("experiment identity differs")
    expected_canonical = {
        "config": CANONICAL_CONFIG_RELATIVE,
        "compact_cache": CANONICAL_COMPACT_CACHE_RELATIVE,
        "sequence_cache": CANONICAL_SEQUENCE_CACHE_RELATIVE,
        "fixed8_reference": CANONICAL_REFERENCE_RELATIVE,
        "output": CANONICAL_OUTPUT_RELATIVE,
        "attempt_lock": CANONICAL_LOCK_RELATIVE,
        "execution_claim": CANONICAL_CLAIM_RELATIVE,
    }
    if config.get("canonical_paths") != expected_canonical:
        raise PermissionError("canonical path contract differs")
    if not all(config.get("prohibitions", {}).values()):
        raise PermissionError("all v2 prohibitions must remain enabled")
    if config["access_policy"] != {
        "train_only_preflight": True,
        "allowed_source_files": list(EXPLICIT_SOURCE_NAMES),
        "directory_enumeration": False,
        "anonymous_evaluation_file_reads": 0,
        "submission_artifact_reads": 0,
        "submission_artifact_writes": 0,
        "upload_attempts": 0,
    }:
        raise PermissionError("train-only access policy differs")
    if config["validation"]["training_prefix_fractions"] != list(
        REGISTERED_PREFIX_FRACTIONS
    ):
        raise PermissionError("outer prefix curve differs")
    if config["model"]["seed_replicates"] != [20260816, 20260817, 20260818]:
        raise PermissionError("registered seed ensemble differs")
    if config["training"]["maximum_inner_epochs"] != 8:
        raise PermissionError("registered checkpoint horizon differs")
    for name, path in _implementation_paths(root).items():
        if _sha256_file(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"pinned implementation SHA differs: {name}")
    inputs = _input_paths(data_dir.resolve(strict=True), paths)
    if set(inputs) != set(config["expected_sha256"]):
        raise PermissionError("train-only input hash key set differs")
    return config, paths, inputs


def _verify_input_hashes(
    inputs: dict[str, Path], expected: dict[str, str]
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name in EXPLICIT_INPUT_KEYS:
        path = inputs[name]
        if not path.is_file():
            raise FileNotFoundError(f"pinned train-only input missing: {name}")
        observed[name] = _sha256_file(path)
        if observed[name] != expected[name]:
            raise PermissionError(f"pinned train-only input SHA differs: {name}")
    return observed


def _minimum_same_station_gap_hours(
    anchors: pd.DataFrame, left_ids: np.ndarray, right_ids: np.ndarray
) -> float:
    lookup = anchors.set_index("anchor_id")
    left = lookup.loc[left_ids]
    right = lookup.loc[right_ids]
    minima: list[float] = []
    for station, validation in right.groupby("station", observed=True):
        train_times = left.loc[left["station"].eq(station), "anchor_time"]
        if train_times.empty:
            continue
        for timestamp in validation["anchor_time"]:
            minima.append(
                float((train_times - pd.Timestamp(timestamp)).abs().min().total_seconds() / 3600.0)
            )
    if not minima:
        raise ValueError("inner split has no common station support")
    return float(min(minima))


def _preflight(
    *, root: Path, config: dict[str, Any], paths: dict[str, Path], inputs: dict[str, Path]
) -> dict[str, Any]:
    snapshot = _verify_input_hashes(inputs, config["expected_sha256"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("nested checkpoint execution requires exactly one visible CUDA GPU")

    features = pd.read_parquet(inputs["compact_cache/train_features.parquet"])
    anchors = pd.read_parquet(
        inputs["compact_cache/train_anchors.parquet"],
        columns=["anchor_id", "station", "anchor_time", "current_hs"],
    )
    if len(features) != 24_360 or len(anchors) != 24_360:
        raise ValueError("training cache row contract differs")
    if not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
        raise ValueError("feature and metadata keys differ")
    expected_ids = np.arange(len(anchors), dtype=np.int64)
    if not np.array_equal(anchors["anchor_id"].to_numpy(np.int64), expected_ids):
        raise ValueError("anchor IDs no longer equal cache row indices")
    feature_columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    if len(feature_columns) != config["features"]["compact_feature_count"]:
        raise ValueError("compact feature count differs")
    compact = features.loc[:, feature_columns].to_numpy(np.float32)

    raw = np.load(inputs["sequence_cache/train_values.npy"], mmap_mode="r")
    station = np.load(inputs["sequence_cache/train_station.npy"], mmap_mode="r")
    if raw.shape != (24_360, 289, 10) or raw.dtype != np.float32:
        raise ValueError("raw sequence cache contract differs")
    if station.shape != (24_360,) or station.dtype != np.int64:
        raise ValueError("station sequence cache contract differs")
    validate_raw_context(torch.from_numpy(np.array(raw[:8], copy=True)))
    station_expected = anchors["station"].map({"G-ORS": 0, "I-ORS": 1, "S-ORS": 2})
    if not np.array_equal(station, station_expected.to_numpy(np.int64)):
        raise ValueError("station codes differ from anchor metadata")

    wave = pd.read_csv(inputs["source/train_wave.csv"])
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split_audit = build_corrected_repeated_forward_folds(
        anchors,
        windows=config["validation"]["windows"],
        gap_hours=int(config["validation"]["outer_gap_hours"]),
        footprint_hours=int(config["validation"]["outer_footprint_hours"]),
    )
    if len(selected) != config["validation"]["expected_outer_cases"]:
        raise ValueError("outer case surface differs")
    validation_keys = pd.read_parquet(inputs["fixed8_reference/validation_keys.parquet"])
    expected_selected = selected[["fold", "anchor_id", "station", "episode_id"]].sort_values(
        ["fold", "anchor_id"]
    ).reset_index(drop=True)
    observed_selected = validation_keys[
        ["fold", "anchor_id", "station", "episode_id"]
    ].sort_values(["fold", "anchor_id"]).reset_index(drop=True)
    if not observed_selected.equals(expected_selected):
        raise ValueError("fixed8 validation keys differ from rebuilt outer surface")
    outer_union = np.sort(selected["anchor_id"].to_numpy(np.int64))

    prefix_ids: dict[float, dict[str, np.ndarray]] = {}
    inner_splits: dict[tuple[float, str], InnerEpisodeSplit] = {}
    inner_audit: dict[str, Any] = {}
    for fraction in REGISTERED_PREFIX_FRACTIONS:
        prefix_ids[fraction] = {}
        prefix_tag = f"p{int(round(fraction * 100)):03d}"
        inner_audit[prefix_tag] = {}
        for fold in folds:
            original = chronological_prefix_ids(anchors, fold.train_ids, fraction)
            if not len(original) or np.intersect1d(original, fold.validation_ids).size:
                raise AssertionError("outer training prefix overlaps its own validation fold")
            try:
                inner = build_inner_episode_split(
                    anchors,
                    original,
                    validation_days=int(config["validation"]["inner_validation_days"]),
                    embargo_hours=int(config["validation"]["inner_embargo_hours"]),
                )
            except ValueError as exc:
                raise ValueError(
                    f"inner split failed for prefix={fraction:.2f}, fold={fold.name}: {exc}"
                ) from exc
            if not np.isin(inner.train_ids, original).all() or not np.isin(
                inner.validation_ids, original
            ).all():
                raise AssertionError("inner split escaped full outer-training prefix")
            if np.intersect1d(inner.train_ids, inner.validation_ids).size:
                raise AssertionError("inner split overlaps")
            minimum_gap = _minimum_same_station_gap_hours(
                anchors, inner.train_ids, inner.validation_ids
            )
            if minimum_gap < 78.0:
                raise AssertionError("inner split violates the 78-hour embargo")
            prefix_ids[fraction][fold.name] = original
            inner_splits[(fraction, fold.name)] = inner
            inner_audit[prefix_tag][fold.name] = {
                "outer_train_cases": int(len(original)),
                "own_outer_validation_overlap_cases": 0,
                "inner_train_cases": int(len(inner.train_ids)),
                "inner_validation_cases": int(len(inner.validation_ids)),
                "minimum_inner_train_validation_gap_hours": minimum_gap,
                "outer_train_ids_sha256": ids_sha256(original),
                "inner_train_ids_sha256": ids_sha256(inner.train_ids),
                "inner_validation_ids_sha256": ids_sha256(inner.validation_ids),
            }
    if len(inner_splits) != config["validation"]["expected_inner_selection_cells"]:
        raise AssertionError("inner selection cell count differs")
    return {
        "snapshot": snapshot,
        "features": features,
        "anchors": anchors,
        "feature_columns": feature_columns,
        "compact": compact,
        "raw": raw,
        "station": station,
        "folds": folds,
        "selected": selected,
        "outer_union": outer_union,
        "split_audit": split_audit,
        "prefix_ids": prefix_ids,
        "inner_splits": inner_splits,
        "inner_audit": inner_audit,
        "outer_reference_values_opened": False,
    }


def _load_training_targets(
    *, preflight: dict[str, Any], inputs: dict[str, Path]
) -> np.ndarray:
    """Materialize train-source residual targets; fitters index only registered train IDs."""

    anchors: pd.DataFrame = preflight["anchors"]
    target_frame = pd.read_parquet(
        inputs["compact_cache/train_anchors.parquet"],
        columns=["anchor_id", *TARGET_COLUMNS],
    )
    if not np.array_equal(
        target_frame["anchor_id"].to_numpy(np.int64), np.arange(len(anchors), dtype=np.int64)
    ):
        raise ValueError("target cache IDs differ")
    current = anchors["current_hs"].to_numpy(np.float32)
    absolute = target_frame.loc[:, TARGET_COLUMNS].to_numpy(np.float32)
    delta = absolute - current[:, None]
    del absolute, target_frame
    if not np.isfinite(delta).all():
        raise ValueError("train-source target is non-finite")
    return delta


def _protected_roots(root: Path, data_dir: Path, paths: dict[str, Path]) -> tuple[Path, ...]:
    return (
        data_dir,
        paths["compact_cache"],
        paths["sequence_cache"],
        paths["reference"],
        root / "submissions",
        root / "output",
    )


def _blind_frame(
    anchors: pd.DataFrame,
    ids: np.ndarray,
    prediction: np.ndarray,
    *,
    fold: str,
) -> pd.DataFrame:
    lookup = anchors.set_index("anchor_id")
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != (len(ids), len(OFFICIAL_LEADS)) or not np.isfinite(values).all():
        raise ValueError("blind prediction surface differs")
    return pd.DataFrame(
        {
            "fold": fold,
            "anchor_id": np.repeat(ids, len(OFFICIAL_LEADS)),
            "station": np.repeat(lookup.loc[ids, "station"].astype(str).to_numpy(), len(OFFICIAL_LEADS)),
            "lead_h": np.tile(np.asarray(OFFICIAL_LEADS, dtype=int), len(ids)),
            "checkpoint_nested_prediction": values.reshape(-1),
        }
    )


def _artifact_hashes(stage: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(stage).as_posix(): {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }


def _run_training(
    *,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    inputs: dict[str, Path],
    preflight: dict[str, Any],
    stage: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    raw = preflight["raw"]
    station = preflight["station"]
    compact = preflight["compact"]
    anchors: pd.DataFrame = preflight["anchors"]
    target_delta = _load_training_targets(preflight=preflight, inputs=inputs)
    case_weight = threshold_case_weights(anchors["current_hs"].to_numpy(float)).astype(np.float32)
    print(json.dumps({"phase": "build_train_only_forcing_cache"}), flush=True)
    forcing = build_causal_forcing_sequence(raw, window_steps=37)
    if forcing.shape != (24_360, 289, 12) or forcing.dtype != np.float32:
        raise ValueError("forcing cache contract differs")

    model_config = CausalForcingSequenceConfig()
    maximum_training = FixedEpochTrainingConfig(
        epochs=int(config["training"]["maximum_inner_epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
    )
    protected = _protected_roots(root, data_dir, paths)
    seeds = tuple(int(seed) for seed in config["model"]["seed_replicates"])
    inner_receipts: list[dict[str, Any]] = []
    outer_receipts: list[dict[str, Any]] = []
    blind_frames: dict[tuple[float, int, str], pd.DataFrame] = {}
    blind_records: list[dict[str, Any]] = []
    completed_inner = 0
    completed_outer = 0

    expected_seed_fits = int(config["training"]["expected_inner_seed_fits"])
    expected_outer_refits = int(config["training"]["expected_outer_seed_refits"])
    for fraction in REGISTERED_PREFIX_FRACTIONS:
        prefix_tag = f"p{int(round(fraction * 100)):03d}"
        for fold in preflight["folds"]:
            outer_train_ids = preflight["prefix_ids"][fraction][fold.name]
            inner: InnerEpisodeSplit = preflight["inner_splits"][(fraction, fold.name)]
            inner_scaler = CompactRobustScaler.fit(
                compact,
                inner.train_ids,
                forbidden_ids=inner.validation_ids,
            )
            curves = []
            seed_checkpoint_receipts: dict[str, Any] = {}
            for seed in seeds:
                print(
                    json.dumps(
                        {
                            "phase": "fit_inner_checkpoint_curve",
                            "completed_before": completed_inner,
                            "total": expected_seed_fits,
                            "prefix": fraction,
                            "fold": fold.name,
                            "seed": seed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                checkpoints: list[dict[str, Any]] = []

                def commit_checkpoint(epoch: int, fitted: Any) -> None:
                    relative = (
                        f"inner_checkpoints/{prefix_tag}/{fold.name}/seed_{seed}/"
                        f"epoch_{epoch:02d}.pt"
                    )
                    target = safe_new_stage_path(stage, relative, protected_roots=protected)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    save_fitted_sequence_model(fitted, target)
                    checkpoints.append(
                        {
                            "epoch": int(epoch),
                            "relative_path": relative,
                            "bytes": int(target.stat().st_size),
                            "sha256": _sha256_file(target),
                            "model_state_sha256": fitted.model_state_sha256,
                        }
                    )

                curve = fit_inner_checkpoint_curve(
                    raw,
                    station,
                    compact,
                    target_delta,
                    case_weight,
                    inner.train_ids,
                    inner.validation_ids,
                    seed=seed,
                    device="cuda",
                    model_config=model_config,
                    training_config=maximum_training,
                    forcing=forcing,
                    compact_scaler=inner_scaler,
                    checkpoint_callback=commit_checkpoint,
                )
                if len(checkpoints) != 8:
                    raise AssertionError("inner checkpoint count differs")
                curves.append(curve)
                seed_checkpoint_receipts[str(seed)] = {
                    "optimizer_steps": int(curve.optimizer_steps),
                    "scaler_state_sha256": curve.scaler_state_sha256,
                    "train_ids_sha256": curve.train_ids_sha256,
                    "validation_ids_sha256": curve.validation_ids_sha256,
                    "checkpoint_model_state_sha256": list(curve.model_state_sha256_by_epoch),
                    "checkpoints": checkpoints,
                }
                completed_inner += 1

            inner_current = anchors.set_index("anchor_id").loc[
                inner.validation_ids, "current_hs"
            ].to_numpy(np.float32)
            inner_target = target_delta[inner.validation_ids] + inner_current[:, None]
            selection = select_earliest_ensemble_epoch(
                curves,
                current_hs=inner_current,
                target_hs=inner_target,
            )
            selection_receipt = {
                "prefix_fraction": float(fraction),
                "fold": fold.name,
                "outer_train_cases": int(len(outer_train_ids)),
                "inner_train_cases": int(len(inner.train_ids)),
                "inner_validation_cases": int(len(inner.validation_ids)),
                "selected_epoch": int(selection.selected_epoch),
                "rmse_by_epoch_m": list(selection.rmse_by_epoch),
                "selection_prediction_sha256_by_epoch": list(
                    selection.selection_prediction_sha256_by_epoch
                ),
                "tie_break": "earliest_exact_minimum",
                "seed_ids": list(selection.seed_ids),
                "outer_reference_values_opened": False,
                "seed_checkpoint_receipts": seed_checkpoint_receipts,
            }
            selection_relative = f"epoch_selection/{prefix_tag}/{fold.name}.json"
            selection_path = safe_new_stage_path(
                stage, selection_relative, protected_roots=protected
            )
            selection_receipt["receipt_sha256"] = _atomic_json_exclusive(
                selection_path, selection_receipt
            )
            inner_receipts.append(selection_receipt)

            outer_scaler = CompactRobustScaler.fit(
                compact,
                outer_train_ids,
                forbidden_ids=fold.validation_ids,
            )
            selected_training = FixedEpochTrainingConfig(
                epochs=int(selection.selected_epoch),
                batch_size=maximum_training.batch_size,
                learning_rate=maximum_training.learning_rate,
                weight_decay=maximum_training.weight_decay,
                gradient_clip_norm=maximum_training.gradient_clip_norm,
            )
            current_outer = anchors.set_index("anchor_id").loc[
                fold.validation_ids, "current_hs"
            ].to_numpy(np.float32)
            for seed in seeds:
                print(
                    json.dumps(
                        {
                            "phase": "refit_selected_epoch_outer_train",
                            "completed_before": completed_outer,
                            "total": expected_outer_refits,
                            "prefix": fraction,
                            "fold": fold.name,
                            "seed": seed,
                            "selected_epoch": selection.selected_epoch,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                cell_started = time.perf_counter()
                prediction_delta, fitted = fit_fixed_epoch_and_predict(
                    raw,
                    station,
                    compact,
                    target_delta,
                    case_weight,
                    outer_train_ids,
                    fold.validation_ids,
                    seed=seed,
                    device="cuda",
                    model_config=model_config,
                    training_config=selected_training,
                    forcing=forcing,
                    compact_scaler=outer_scaler,
                )
                prediction = postprocess_sequence_delta(prediction_delta, current_outer)
                model_relative = (
                    f"outer_models/{prefix_tag}/{fold.name}/seed_{seed}/model.pt"
                )
                model_path = safe_new_stage_path(stage, model_relative, protected_roots=protected)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                save_fitted_sequence_model(fitted, model_path)
                model_sha = _sha256_file(model_path)
                reloaded = load_fitted_sequence_model(model_path, map_location="cpu")
                reload_delta = predict_with_fitted_sequence_model(
                    reloaded,
                    raw,
                    station,
                    compact,
                    fold.validation_ids,
                    device="cuda",
                    batch_size=maximum_training.batch_size,
                    forcing=forcing,
                )
                reload_prediction = postprocess_sequence_delta(reload_delta, current_outer)
                if not np.array_equal(reload_prediction, prediction):
                    raise RuntimeError("outer saved-model reload prediction differs")
                blind_relative = (
                    f"blind_predictions/{prefix_tag}/{fold.name}/seed_{seed}.npy"
                )
                blind_path = safe_new_stage_path(
                    stage, blind_relative, protected_roots=protected
                )
                blind_sha = _write_npy_exclusive(blind_path, prediction.astype(np.float64))
                frame = _blind_frame(
                    anchors,
                    fold.validation_ids,
                    prediction,
                    fold=fold.name,
                )
                blind_frames[(fraction, seed, fold.name)] = frame
                record = {
                    "prefix_fraction": float(fraction),
                    "fold": fold.name,
                    "seed": int(seed),
                    "selected_epoch": int(selection.selected_epoch),
                    "train_cases": int(len(outer_train_ids)),
                    "validation_cases": int(len(fold.validation_ids)),
                    "train_ids_sha256": ids_sha256(outer_train_ids),
                    "validation_ids_sha256": ids_sha256(fold.validation_ids),
                    "model_relative_path": model_relative,
                    "model_sha256": model_sha,
                    "model_state_sha256": fitted.model_state_sha256,
                    "blind_prediction_relative_path": blind_relative,
                    "blind_prediction_sha256": blind_sha,
                    "saved_model_reload_prediction_exact": True,
                    "outer_reference_values_opened": False,
                    "elapsed_seconds": float(time.perf_counter() - cell_started),
                }
                outer_receipts.append(record)
                blind_records.append(record)
                completed_outer += 1

    if completed_inner != expected_seed_fits or completed_outer != expected_outer_refits:
        raise AssertionError("registered inner/outer fit count differs")
    if (
        sum(len(value["seed_checkpoint_receipts"]) for value in inner_receipts)
        != expected_seed_fits
    ):
        raise AssertionError("inner seed receipt count differs")
    checkpoint_count = sum(
        len(seed["checkpoints"])
        for receipt in inner_receipts
        for seed in receipt["seed_checkpoint_receipts"].values()
    )
    if checkpoint_count != config["training"]["expected_inner_checkpoint_files"]:
        raise AssertionError("inner checkpoint file count differs")

    blind_commit = {
        "schema_version": "p3_blind_outer_commit.v1",
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "prediction_files": blind_records,
        "prediction_file_count": len(blind_records),
        "outer_reference_values_opened_before_commit": False,
        "anonymous_evaluation_value_reads": 0,
        "submission_reads_or_writes": 0,
    }
    blind_commit_path = safe_new_stage_path(
        stage, "blind_outer_commit.json", protected_roots=protected
    )
    blind_commit_sha = _atomic_json_exclusive(blind_commit_path, blind_commit)

    # This is intentionally the first parsed read of the historical outer-reference values.
    reference = pd.read_parquet(inputs["fixed8_reference/oof/learning_curve_oof.parquet"])
    required = {
        *KEYS,
        "prefix_fraction",
        "target_hs",
        "current_hs",
        "persistence",
        "incumbent_prediction",
        "challenger_prediction",
    }
    if not required.issubset(reference.columns):
        raise ValueError("fixed8 reference OOF columns differ")
    all_oof: list[pd.DataFrame] = []
    points_vs_incumbent: dict[str, Any] = {}
    points_vs_fixed8: dict[str, Any] = {}
    prefix_metrics: dict[str, Any] = {}
    for fraction in REGISTERED_PREFIX_FRACTIONS:
        reference_prefix = reference.loc[
            np.isclose(reference["prefix_fraction"].to_numpy(float), fraction)
        ].sort_values(KEYS).reset_index(drop=True)
        if len(reference_prefix) != config["validation"]["expected_outer_rows_per_prefix"]:
            raise ValueError("fixed8 reference prefix row count differs")
        seed_frames: list[pd.DataFrame] = []
        for seed in seeds:
            current = pd.concat(
                [blind_frames[(fraction, seed, fold.name)] for fold in preflight["folds"]],
                ignore_index=True,
            ).sort_values(KEYS).reset_index(drop=True)
            if not current[KEYS].equals(reference_prefix[KEYS]):
                raise ValueError("blind prediction keys differ from fixed8 reference")
            seed_frames.append(current)
        evaluated = reference_prefix[
            [
                *KEYS,
                "target_hs",
                "current_hs",
                "persistence",
                "incumbent_prediction",
                "challenger_prediction",
            ]
        ].copy()
        evaluated = evaluated.rename(columns={"challenger_prediction": "fixed8_prediction"})
        evaluated["checkpoint_nested_prediction"] = np.mean(
            np.column_stack(
                [frame["checkpoint_nested_prediction"].to_numpy(float) for frame in seed_frames]
            ),
            axis=1,
        )
        evaluated["prefix_fraction"] = float(fraction)
        point_incumbent = evaluate_point(
            evaluated,
            candidate_column="checkpoint_nested_prediction",
            incumbent_column="incumbent_prediction",
            bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
            bootstrap_seed=int(config["evaluation"]["bootstrap_seed"])
            + int(round(fraction * 100)),
        )
        point_fixed8 = evaluate_point(
            evaluated,
            candidate_column="checkpoint_nested_prediction",
            incumbent_column="fixed8_prediction",
            bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
            bootstrap_seed=int(config["evaluation"]["bootstrap_seed"])
            + 1000
            + int(round(fraction * 100)),
        )
        tag = f"{fraction:.2f}"
        points_vs_incumbent[tag] = point_incumbent
        points_vs_fixed8[tag] = point_fixed8
        prefix_metrics[tag] = {
            "rows": int(len(evaluated)),
            "cases": int(evaluated["anchor_id"].nunique()),
            "incumbent_rmse_m": float(rmse(evaluated["target_hs"], evaluated["incumbent_prediction"])),
            "fixed8_rmse_m": float(rmse(evaluated["target_hs"], evaluated["fixed8_prediction"])),
            "checkpoint_nested_rmse_m": float(
                rmse(evaluated["target_hs"], evaluated["checkpoint_nested_prediction"])
            ),
            "delta_checkpoint_minus_fixed8_m": float(
                point_fixed8["delta_candidate_minus_incumbent_m"]
            ),
            "delta_checkpoint_minus_incumbent_m": float(
                point_incumbent["delta_candidate_minus_incumbent_m"]
            ),
        }
        all_oof.append(evaluated)
    oof = pd.concat(all_oof, ignore_index=True)
    oof_sha = _write_parquet_exclusive(stage / "oof/learning_curve_oof.parquet", oof)
    access = {
        "outer_reference_parsed_reads_before_blind_commit": 0,
        "outer_reference_parsed_reads_after_blind_commit": 1,
        "anonymous_evaluation_value_reads": 0,
        "hidden_target_reads": 0,
        "submission_artifact_reads": 0,
        "submission_artifact_writes": 0,
        "upload_attempts": 0,
        "era5_artifact_or_process_reads": 0,
        "era5_artifact_or_process_writes": 0,
    }
    metrics = {
        "schema_version": "p3_causal_forcing_sequence_checkpoint_nested.metrics.v2",
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": "TRAIN_ONLY_NESTED_CHECKPOINT_RESEARCH_COMPLETE_NO_CANDIDATE",
        "interpretation": (
            "Inner seed-ensemble epoch selection inside each registered outer-training prefix, "
            "followed by full outer-train refits and a blind prediction commit before the "
            "historical outer-reference OOF values are parsed. "
            "This is local research evidence, not official promotion or upload authorization."
        ),
        "prefix_metrics": prefix_metrics,
        "points_vs_incumbent": points_vs_incumbent,
        "points_vs_fixed8": points_vs_fixed8,
        "inner_selection_receipts": inner_receipts,
        "outer_refit_receipts": outer_receipts,
        "split_audit": preflight["split_audit"],
        "inner_audit": preflight["inner_audit"],
        "blind_outer_commit_sha256": blind_commit_sha,
        "oof_sha256": oof_sha,
        "access_counters": access,
        "candidate_created": False,
        "candidate_uploaded": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return metrics, oof, access


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    config, paths, inputs = authorize_entry(root=root, data_dir=data_dir)
    preflight = _preflight(root=root, config=config, paths=paths, inputs=inputs)
    counts = [
        row
        for prefix in preflight["inner_audit"].values()
        for row in prefix.values()
    ]
    return {
        "status": "CHECK_ONLY_PASS_TRAIN_ONLY_NO_LOCK_CONSUMED",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "outer_cases": int(len(preflight["selected"])),
        "inner_selection_cells": int(len(preflight["inner_splits"])),
        "minimum_inner_train_cases": int(min(row["inner_train_cases"] for row in counts)),
        "minimum_inner_validation_cases": int(
            min(row["inner_validation_cases"] for row in counts)
        ),
        "minimum_inner_gap_hours": float(
            min(row["minimum_inner_train_validation_gap_hours"] for row in counts)
        ),
        "full_outer_train_refit": True,
        "own_outer_validation_forbidden_from_fit": True,
        "reference_oof_values_opened": False,
        "anonymous_evaluation_value_reads": 0,
        "submission_reads_or_writes": 0,
        "lock_absent": not paths["lock"].exists(),
        "claim_absent": not paths["claim"].exists(),
        "output_absent": not paths["output"].exists(),
    }


def execute(*, root: Path, data_dir: Path) -> dict[str, Any]:
    config, paths, inputs = authorize_entry(root=root, data_dir=data_dir)
    preflight = _preflight(root=root, config=config, paths=paths, inputs=inputs)
    attempt = acquire_persistent_attempt_lock(
        paths["lock"],
        experiment_id=config["experiment_id"],
        config_sha256=EXPECTED_CONFIG_SHA256,
        created_at=_now(),
    )
    claim = acquire_persistent_attempt_lock(
        paths["claim"],
        experiment_id=config["experiment_id"],
        config_sha256=EXPECTED_CONFIG_SHA256,
        created_at=_now(),
    )
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_checkpoint_nested_v2_", dir=tmp_root))
    metrics, oof, access = _run_training(
        root=root,
        data_dir=data_dir,
        config=config,
        paths=paths,
        inputs=inputs,
        preflight=preflight,
        stage=stage,
    )
    metrics["attempt_lock"] = attempt
    metrics["execution_claim"] = claim
    metrics_sha = _atomic_json_exclusive(stage / "metrics.json", metrics)
    registry = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": metrics["status"],
        "full_prefix_delta_checkpoint_minus_fixed8_m": metrics["prefix_metrics"]["1.00"][
            "delta_checkpoint_minus_fixed8_m"
        ],
        "full_prefix_delta_checkpoint_minus_incumbent_m": metrics["prefix_metrics"]["1.00"][
            "delta_checkpoint_minus_incumbent_m"
        ],
        "candidate_created": False,
        "candidate_uploaded": False,
        "official_upload_count": 0,
    }
    registry_sha = _atomic_json_exclusive(stage / "registry.json", registry)
    qa_scaffold = {
        "schema_version": "p3_checkpoint_nested_independent_qa_scaffold.v1",
        "status": "PENDING_INDEPENDENT_QA",
        "verifier": "scripts/verify_p3_causal_forcing_sequence_checkpoint_nested_v2.py",
        "required_checks": [
            "manifest_and_output_file_sha256",
            "72_inner_checkpoints_and_9_outer_models",
            "9_blind_prediction_hashes",
            "selected_epoch_in_closed_interval_1_8",
            "earliest_exact_minimum_recomputed",
            "one_complete_1086_row_full_prefix_surface",
            "outer_reference_opened_only_after_blind_commit",
            "all_forbidden_access_counters_zero",
            "candidate_and_upload_absent",
        ],
    }
    qa_sha = _atomic_json_exclusive(stage / "independent_qa_scaffold.json", qa_scaffold)
    after = _verify_input_hashes(inputs, config["expected_sha256"])
    if after != preflight["snapshot"]:
        raise RuntimeError("train-only input snapshot changed during execution")
    manifest = {
        "schema_version": "p3_causal_forcing_sequence_checkpoint_nested.manifest.v2",
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": metrics["status"],
        "append_only": True,
        "canonical_contract": {
            "config_path": CANONICAL_CONFIG_RELATIVE,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
            "output_path": CANONICAL_OUTPUT_RELATIVE,
            "attempt_lock_path": CANONICAL_LOCK_RELATIVE,
            "attempt_lock_sha256": attempt["sha256"],
            "execution_claim_path": CANONICAL_CLAIM_RELATIVE,
            "execution_claim_sha256": claim["sha256"],
        },
        "git": _git_state(root),
        "implementation_sha256": {
            **{name: _sha256_file(path) for name, path in _implementation_paths(root).items()},
            "runner": _sha256_file(Path(__file__).resolve()),
            "verifier": _sha256_file(
                root / "scripts/verify_p3_causal_forcing_sequence_checkpoint_nested_v2.py"
            ),
        },
        "input_sha256_before": preflight["snapshot"],
        "input_sha256_after": after,
        "output_files": _artifact_hashes(stage),
        "metrics_sha256": metrics_sha,
        "registry_sha256": registry_sha,
        "qa_scaffold_sha256": qa_sha,
        "oof_rows": int(len(oof)),
        "access_counters": access,
        "candidate_created": False,
        "candidate_uploaded": False,
    }
    _atomic_json_exclusive(stage / "manifest.json", manifest)
    manifest_sha = _sha256_file(stage / "manifest.json")
    with (stage / "manifest.sha256").open("xb") as handle:
        handle.write(f"{manifest_sha}  manifest.json\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
    if paths["output"].exists():
        raise FileExistsError("append-only output appeared before final move")
    stage.replace(paths["output"])
    result = {
        "status": metrics["status"],
        "artifact_dir": CANONICAL_OUTPUT_RELATIVE,
        "metrics_sha256": metrics_sha,
        "manifest_sha256": manifest_sha,
        "registry_sha256": registry_sha,
        "candidate_created": False,
        "candidate_uploaded": False,
        "official_upload_count": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    if args.check_only:
        print(json.dumps(check_only(root=root, data_dir=data_dir), ensure_ascii=False, indent=2))
    else:
        execute(root=root, data_dir=data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
