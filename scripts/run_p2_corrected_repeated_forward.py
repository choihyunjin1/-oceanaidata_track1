"""Run the sealed append-only corrected repeated-forward P2 v2 generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from p2_restore.corrected_repeated_forward import (
    build_fixed_lean_arm,
    build_joint_masked_population,
    fit_fixed_blend,
    forward_training_mask,
    joint_mask_target_context,
    metric_report,
    nominal_target_rows,
    paired_fold_day_bootstrap,
    predict_scored_window,
    public_endpoints_from_masked_context,
    window_mask,
)
from p2_restore.data import KEYS, P2Data, load_p2_data, resolve_data_dir
from p2_restore.features import FeatureTable, build_test_features
from p2_restore.profile_projection import project_profiles_vectorized
from p2_restore.submission import build_submission, validate_submission

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/p2_corrected_repeated_forward_v2.json"
DEFAULT_OUTPUT = ROOT / "artifacts/p2_corrected_repeated_forward_v2"
ATTEMPT_LOCK = ROOT / "artifacts/p2_corrected_repeated_forward_v2_control/attempt.lock"
CANONICAL_CONFIG_SHA256 = "cd0f88fd12fa7900be7c39cd8566aa455dae6ffbf4da4077adc52bd10ced70ca"
KST = ZoneInfo("Asia/Seoul")

EXPECTED_FOLDS = [
    {
        "name": "outer_2024_sep_oct",
        "outer": ["2024-09-01T00:00:00+09:00", "2024-11-01T00:00:00+09:00"],
        "inner": ["2024-07-01T00:00:00+09:00", "2024-09-01T00:00:00+09:00"],
        "same_season_priority": True,
    },
    {
        "name": "outer_2025_may_jun",
        "outer": ["2025-05-01T00:00:00+09:00", "2025-07-01T00:00:00+09:00"],
        "inner": ["2024-09-01T00:00:00+09:00", "2024-11-01T00:00:00+09:00"],
        "same_season_priority": False,
    },
    {
        "name": "outer_2025_jul_aug",
        "outer": ["2025-07-01T00:00:00+09:00", "2025-09-01T00:00:00+09:00"],
        "inner": ["2025-05-01T00:00:00+09:00", "2025-07-01T00:00:00+09:00"],
        "same_season_priority": False,
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _now() -> str:
    return datetime.now(KST).isoformat()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    _atomic_bytes(path, payload + b"\n")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_joblib(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _status(output_dir: Path, progress: float, phase: str, detail: str) -> None:
    status_path = _contained_path(output_dir / "status.json", DEFAULT_OUTPUT, role="status")
    _atomic_json(
        status_path,
        {
            "experiment_id": "p2_corrected_repeated_forward_v2",
            "status": "complete" if progress >= 100 else "running",
            "progress": float(progress),
            "phase": phase,
            "detail": detail,
            "updated_at_kst": _now(),
        },
    )


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("P2 corrected config must be a JSON object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "p2_corrected_repeated_forward.v2":
        raise ValueError("unexpected corrected P2 schema version")
    if config.get("experiment_id") != "p2_corrected_repeated_forward_v2":
        raise ValueError("unexpected corrected P2 experiment id")
    if config.get("status") != "authorized_local_corrected_retraining":
        raise ValueError("corrected P2 generation is not locally authorized")
    if config.get("research_only") is not True or config.get("upload_allowed") is not False:
        raise ValueError("corrected P2 generation must remain research-only and non-uploadable")
    candidate = config["candidate"]
    expected_parameters = {
        "objective": "regression_l2",
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 200,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "deterministic": True,
        "force_row_wise": True,
    }
    if candidate.get("model_parameters") != expected_parameters:
        raise ValueError("fixed LightGBM parameters changed")
    if candidate.get("blend_weights") != {"base": 0.5, "lean": 0.5}:
        raise ValueError("fixed 50:50 blend changed")
    if candidate.get("hyperparameter_searches") != 0:
        raise ValueError("hyperparameter search is forbidden in this generation")
    if candidate.get("seed") != 20260822:
        raise ValueError("fixed training seed changed")
    if candidate.get("target_layer_inputs") != []:
        raise ValueError("target-layer inputs must remain empty")
    if candidate.get("public_input_layers") != [1, 5, 6, 7, 8]:
        raise ValueError("public input-layer contract changed")
    if candidate.get("target_layers") != [2, 3, 4]:
        raise ValueError("target-layer contract changed")
    masking = config["masking"]
    if masking.get("jointly_null_columns") != ["temp", "psal"]:
        raise ValueError("joint target-variable mask changed")
    if masking.get("jointly_null_layers") != [2, 3, 4]:
        raise ValueError("joint target-layer mask changed")
    if masking.get("hidden_window_kst_half_open") != [
        "2025-09-01T00:00:00+09:00",
        "2025-11-01T00:00:00+09:00",
    ]:
        raise ValueError("hidden target window changed")
    if masking.get("hidden_target_non_null_allowed") is not False:
        raise ValueError("hidden target values must fail closed")
    validation = config["validation"]
    if int(validation.get("embargo_days", 0)) != 7:
        raise ValueError("the corrected validation embargo must remain seven days")
    if (
        float(validation.get("minimum_inner_target_coverage", 0.0)) != 0.96
        or float(validation.get("minimum_outer_target_coverage", 0.0)) != 0.96
    ):
        raise ValueError("the pre-fit official-like coverage floors must remain 0.96")
    if sum(int(value) for value in validation["official_layer_counts"].values()) != 26_061:
        raise ValueError("official layer denominator no longer sums to 26,061")
    if validation.get("official_layer_counts") != {"2": 8713, "3": 8712, "4": 8636}:
        raise ValueError("official layer counts changed")
    if validation.get("folds") != EXPECTED_FOLDS:
        raise ValueError("exact nested-forward fold membership changed")
    if validation.get("bootstrap") != {
        "unit": "KST calendar day within fold",
        "replicates": 2000,
        "interval": 0.9,
        "seed": 20260822,
    }:
        raise ValueError("fixed bootstrap contract changed")
    output = config["output_contract"]
    if output.get("append_only") is not True:
        raise ValueError("corrected outputs must remain append-only")
    if output.get("current_frozen_submission_mutation_allowed") is not False:
        raise ValueError("current frozen submission mutation must remain forbidden")
    if output.get("default_directory") != "artifacts/p2_corrected_repeated_forward_v2":
        raise ValueError("canonical output contract changed")
    if output.get("attempt_lock") != (
        "artifacts/p2_corrected_repeated_forward_v2_control/attempt.lock"
    ):
        raise ValueError("attempt-lock contract changed")
    if output.get("candidate_relative_path") != (
        "candidate/P2_CORRECTED_REPEATED_FORWARD_V2.csv"
    ):
        raise ValueError("candidate path contract changed")
    if output.get("test_rows") != 26_061 or output.get("columns") != [
        "station",
        "layer",
        "time",
        "temp",
    ]:
        raise ValueError("candidate schema or row contract changed")
    if config.get("qa_controls") != {
        "canonical_config_path_required": True,
        "canonical_config_sha256_required": True,
        "deep_config_equality_required_on_direct_call": True,
        "canonical_output_path_required": True,
        "persistent_o_excl_attempt_lock": True,
        "rerun_allowed": False,
    }:
        raise ValueError("QA control contract changed")


def _canonical_preflight(
    config: Mapping[str, Any], config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Fail closed on path copies, mutated mappings, or canonical-file drift."""

    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("only the canonical v2 config path is accepted")
    if output_dir.resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError("only the canonical v2 output directory is accepted")
    actual_sha = _sha256(DEFAULT_CONFIG)
    if actual_sha != CANONICAL_CONFIG_SHA256:
        raise ValueError("canonical v2 config SHA-256 changed")
    canonical = _load_config(DEFAULT_CONFIG)
    _validate_config(canonical)
    if dict(config) != canonical:
        raise ValueError("passed config differs from the reloaded canonical v2 config")
    return canonical


def _contained_path(path: Path, root: Path, *, role: str) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{role} escapes the canonical output directory") from error
    if not relative.parts:
        raise ValueError(f"{role} must be a file below the canonical output directory")
    return resolved


def _planned_write_paths(
    config: Mapping[str, Any], output_dir: Path
) -> dict[str, Path]:
    """Resolve every v2 write target and reject absolute/traversal escape."""

    paths: dict[str, Path] = {
        "status": output_dir / "status.json",
        "inner_oof": output_dir / "oof/inner_diagnostic.parquet",
        "outer_oof": output_dir / "oof/outer_repeated_forward.parquet",
        "metrics": output_dir / "metrics.json",
        "final_model": output_dir / "models/final_full_train.joblib",
        "candidate": output_dir / str(config["output_contract"]["candidate_relative_path"]),
        "result": output_dir / "result.json",
        "manifest": output_dir / "manifest.json",
        "seal": output_dir / "seal.json",
    }
    for fold in config["validation"]["folds"]:
        name = str(fold["name"])
        paths[f"model_{name}_inner"] = output_dir / "models" / f"{name}_inner.joblib"
        paths[f"model_{name}_outer"] = output_dir / "models" / f"{name}_outer.joblib"
    resolved = {
        role: _contained_path(path, output_dir, role=role) for role, path in paths.items()
    }
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("planned v2 write paths collide")
    return resolved


def _prewrite_guard(
    config: Mapping[str, Any], data_dir: Path, output_dir: Path
) -> dict[str, Path]:
    """Prove all writes are new, contained, and disjoint from protected inputs."""

    if output_dir.exists():
        raise FileExistsError("canonical v2 output already exists")
    if ATTEMPT_LOCK.exists():
        raise FileExistsError("canonical v2 one-shot attempt lock already exists")
    planned = _planned_write_paths(config, output_dir)
    protected_roots = [
        data_dir.resolve(),
        (ROOT / "submissions").resolve(),
        (ROOT / "output/2026-08-20/ready").resolve(),
    ]
    protected_files = {
        (data_dir / name).resolve() for name in config["source_contract"]
    } | {path.resolve() for path in (ROOT / "submissions/p2").glob("*.csv")}
    ready = ROOT / "output/2026-08-20/ready/P2_submission.csv"
    if ready.is_file():
        protected_files.add(ready.resolve())
    for role, path in planned.items():
        if path.exists():
            raise FileExistsError(f"planned write target already exists: {role}")
        if path in protected_files:
            raise ValueError(f"planned write target aliases a protected file: {role}")
        for protected_root in protected_roots:
            try:
                path.relative_to(protected_root)
            except ValueError:
                continue
            raise ValueError(f"planned write target enters a protected tree: {role}")
    return planned


def _acquire_attempt_lock(path: Path = ATTEMPT_LOCK) -> dict[str, Any]:
    """Atomically consume the sole actual-run attempt for this generation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": "p2_corrected_repeated_forward_v2",
        "attempt": 1,
        "canonical_config_sha256": CANONICAL_CONFIG_SHA256,
        "created_at_kst": _now(),
        "status": "ATTEMPT_CONSUMED",
        "rerun_allowed": False,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError("v2 actual attempt was already consumed") from error
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": _logical(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _verify_sources(config: Mapping[str, Any], data_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, expected in config["source_contract"].items():
        path = data_dir / name
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"source hash mismatch for {name}")
        records[name] = {"logical_id": name, "sha256": actual, "bytes": path.stat().st_size}
    return records


def _frozen_snapshot() -> dict[str, str]:
    paths = sorted((ROOT / "submissions/p2").glob("*.csv"))
    ready = ROOT / "output/2026-08-20/ready/P2_submission.csv"
    if ready.is_file():
        paths.append(ready)
    return {_logical(path): _sha256(path) for path in paths if path.is_file()}


def _git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {
        "head": head,
        "dirty": bool(status),
        "status_entry_count": len(status),
        "modified_entry_count": sum(not line.startswith("??") for line in status),
        "untracked_entry_count": sum(line.startswith("??") for line in status),
    }


def _package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "lightgbm", "scikit-learn", "joblib", "pyarrow"]
    return {name: metadata.version(name) for name in names}


def _coverage(population: FeatureTable, start: str, stop: str) -> dict[str, Any]:
    window = window_mask(population.frame, start, stop)
    scored = window & np.isfinite(population.frame["target"].to_numpy(float))
    nominal = nominal_target_rows(start, stop)
    by_layer = {
        str(layer): int((scored & population.frame["layer"].eq(layer).to_numpy(bool)).sum())
        for layer in (2, 3, 4)
    }
    return {
        "nominal_rows": int(nominal),
        "eligible_feature_rows": int(window.sum()),
        "finite_target_rows": int(scored.sum()),
        "target_coverage": float(scored.sum() / nominal),
        "rows_by_layer": by_layer,
    }


def _training_summary(
    frame: pd.DataFrame, selected: np.ndarray, cutoff: pd.Timestamp
) -> dict[str, Any]:
    time = pd.to_datetime(frame.loc[selected, "time"], utc=True)
    return {
        "rows": int(selected.sum()),
        "rows_by_layer": {
            str(layer): int((selected & frame["layer"].eq(layer).to_numpy(bool)).sum())
            for layer in (2, 3, 4)
        },
        "first_label_time_utc": time.min().isoformat(),
        "last_label_time_utc": time.max().isoformat(),
        "exclusive_cutoff_utc": cutoff.isoformat(),
        "max_label_precedes_cutoff": bool(time.max() < cutoff),
    }


def _predict_official_candidate(
    model: object,
    data: P2Data,
    masked_context: pd.DataFrame,
    population_base: FeatureTable,
    population_lean: FeatureTable,
    endpoints: pd.DataFrame,
    *,
    hidden_start: str,
    hidden_stop: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    masked_data = P2Data(
        observations=masked_context,
        test_index=data.test_index,
        sample_submission=data.sample_submission,
        baseline=data.baseline,
    )
    official_base = build_test_features(masked_data)
    official_lean = build_fixed_lean_arm(official_base, masked_context)
    official_unprojected = model.predict(official_base, official_lean)

    hidden = window_mask(population_base.frame, hidden_start, hidden_stop)
    full_base = FeatureTable(
        population_base.frame.loc[hidden].reset_index(drop=True), population_base.feature_columns
    )
    full_lean = FeatureTable(
        population_lean.frame.loc[hidden].reset_index(drop=True), population_lean.feature_columns
    )
    if full_base.frame["target"].notna().any():
        raise AssertionError("hidden target temperature entered full-grid inference")
    full_unprojected = model.predict(full_base, full_lean)

    full_keys = full_base.frame.loc[:, ["station", "layer", "time"]].copy()
    full_keys["_utc"] = pd.to_datetime(full_keys["time"], utc=True)
    official_keys = official_base.frame.loc[:, ["station", "layer", "time"]].copy()
    official_keys["_utc"] = pd.to_datetime(official_keys["time"], utc=True)
    official_keys["_official_row"] = np.arange(len(official_keys))
    official_keys["_official_prediction"] = official_unprojected
    aligned = full_keys.merge(
        official_keys.loc[
            :, ["station", "layer", "_utc", "_official_row", "_official_prediction"]
        ],
        on=["station", "layer", "_utc"],
        how="left",
        validate="one_to_one",
    )
    matched = aligned["_official_row"].notna().to_numpy(bool)
    if int(matched.sum()) != len(data.test_index):
        raise ValueError("full hidden profile population does not cover every official test key")
    combined = full_unprojected.copy()
    combined[matched] = aligned.loc[matched, "_official_prediction"].to_numpy(float)
    projection = project_profiles_vectorized(full_base.frame, combined, endpoints)
    official_rows = aligned.loc[matched, "_official_row"].to_numpy(int)
    official_prediction = np.empty(len(data.test_index), dtype=float)
    official_prediction[official_rows] = projection.prediction[matched]
    submission = build_submission(data.test_index, official_prediction)
    diagnostics = {
        "official_rows": int(len(submission)),
        "full_hidden_profile_rows": int(len(full_base.frame)),
        "full_hidden_truth_non_null": int(full_base.frame["target"].notna().sum()),
        "projection_eligible_full_rows": int(projection.eligible_mask.sum()),
        "projection_active_full_rows": int(projection.active_mask.sum()),
        "projection_active_official_rows": int(projection.active_mask[matched].sum()),
        "prediction_min_c": float(official_prediction.min()),
        "prediction_max_c": float(official_prediction.max()),
        "prediction_mean_c": float(official_prediction.mean()),
    }
    return submission, diagnostics


def _dry_run(
    config: Mapping[str, Any], config_path: Path, data_dir: Path, output_dir: Path
) -> int:
    config = _canonical_preflight(config, config_path, output_dir)
    _prewrite_guard(config, data_dir, output_dir)
    sources = _verify_sources(config, data_dir)
    data = load_p2_data(data_dir)
    masked, mask_audit = joint_mask_target_context(data.observations)
    population = build_joint_masked_population(data.observations, masked)
    lean = build_fixed_lean_arm(population, masked)
    forbidden = [
        column
        for column in lean.feature_columns
        if column.startswith(("temp_2", "temp_3", "temp_4", "psal_2", "psal_3", "psal_4"))
    ]
    if forbidden:
        raise AssertionError(f"target-layer features entered dry-run schema: {forbidden}")
    inner_floor = float(config["validation"]["minimum_inner_target_coverage"])
    outer_floor = float(config["validation"]["minimum_outer_target_coverage"])
    coverage: dict[str, Any] = {}
    for fold in config["validation"]["folds"]:
        inner = _coverage(population, *fold["inner"])
        outer = _coverage(population, *fold["outer"])
        if (
            inner["target_coverage"] < inner_floor
            or outer["target_coverage"] < outer_floor
        ):
            raise ValueError(f"fold {fold['name']} fails the fixed coverage floor")
        coverage[str(fold["name"])] = {"inner": inner, "outer": outer}
    print(
        json.dumps(
            {
                "dry_run": "PASS",
                "experiment_id": config["experiment_id"],
                "source_contracts": len(sources),
                "joint_mask": mask_audit.__dict__,
                "base_feature_count": len(population.feature_columns),
                "lean_feature_count": len(lean.feature_columns),
                "finite_training_labels": int(population.frame["target"].notna().sum()),
                "fold_coverage": coverage,
                "model_fits": 0,
                "candidate_writes": 0,
                "output_directory_created": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run(config: Mapping[str, Any], config_path: Path, data_dir: Path, output_dir: Path) -> int:
    config = _canonical_preflight(config, config_path, output_dir)
    planned = _prewrite_guard(config, data_dir, output_dir)
    attempt_record = _acquire_attempt_lock()
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    frozen_before = _frozen_snapshot()
    git_before = _git_state()
    _status(output_dir, 2, "preflight", "source hashes, immutable baselines, and config contract")
    source_records = _verify_sources(config, data_dir)
    data = load_p2_data(data_dir)

    _status(output_dir, 8, "mask_and_features", "joint target temp+psal mask and public-only features")
    masked, mask_audit = joint_mask_target_context(data.observations)
    population_base = build_joint_masked_population(data.observations, masked)
    population_lean = build_fixed_lean_arm(population_base, masked)
    endpoints = public_endpoints_from_masked_context(masked)
    if int(population_base.frame["target"].notna().sum()) == 0:
        raise ValueError("no legally distributed target labels are available")

    folds = config["validation"]["folds"]
    inner_floor = float(config["validation"]["minimum_inner_target_coverage"])
    outer_floor = float(config["validation"]["minimum_outer_target_coverage"])
    embargo_days = int(config["validation"]["embargo_days"])
    seed = int(config["candidate"]["seed"])
    inner_frames: list[pd.DataFrame] = []
    outer_frames: list[pd.DataFrame] = []
    fold_records: dict[str, Any] = {}
    model_records: dict[str, Any] = {}

    for number, fold in enumerate(folds):
        name = str(fold["name"])
        progress = 15 + number * (52 / len(folds))
        _status(output_dir, progress, "nested_forward_fit", f"{name}: fixed inner diagnostic")
        inner_coverage = _coverage(population_base, *fold["inner"])
        outer_coverage = _coverage(population_base, *fold["outer"])
        if (
            inner_coverage["target_coverage"] < inner_floor
            or outer_coverage["target_coverage"] < outer_floor
        ):
            raise ValueError(f"{name} violates the predeclared target-coverage floor")

        inner_train, inner_cutoff = forward_training_mask(
            population_base.frame, fold["inner"][0], embargo_days=embargo_days
        )
        inner_model = fit_fixed_blend(
            population_base, population_lean, inner_train, seed=seed + number * 10
        )
        inner_prediction = predict_scored_window(
            inner_model,
            population_base,
            population_lean,
            endpoints,
            start=fold["inner"][0],
            stop=fold["inner"][1],
            fold=name,
            stage="inner",
        )
        inner_path = planned[f"model_{name}_inner"]
        _atomic_joblib(inner_path, inner_model)
        inner_frames.append(inner_prediction)

        _status(output_dir, progress + 8, "nested_forward_fit", f"{name}: sealed outer fit")
        outer_train, outer_cutoff = forward_training_mask(
            population_base.frame, fold["outer"][0], embargo_days=embargo_days
        )
        outer_model = fit_fixed_blend(
            population_base, population_lean, outer_train, seed=seed + number * 10 + 1
        )
        outer_prediction = predict_scored_window(
            outer_model,
            population_base,
            population_lean,
            endpoints,
            start=fold["outer"][0],
            stop=fold["outer"][1],
            fold=name,
            stage="outer",
        )
        outer_path = planned[f"model_{name}_outer"]
        _atomic_joblib(outer_path, outer_model)
        outer_frames.append(outer_prediction)
        fold_records[name] = {
            "inner_window_kst": fold["inner"],
            "outer_window_kst": fold["outer"],
            "same_season_priority": bool(fold["same_season_priority"]),
            "inner_coverage": inner_coverage,
            "outer_coverage": outer_coverage,
            "inner_training": _training_summary(
                population_base.frame, inner_train, inner_cutoff
            ),
            "outer_training": _training_summary(
                population_base.frame, outer_train, outer_cutoff
            ),
        }
        model_records[f"{name}_inner"] = {
            "path": _logical(inner_path),
            "sha256": _sha256(inner_path),
            "bytes": inner_path.stat().st_size,
        }
        model_records[f"{name}_outer"] = {
            "path": _logical(outer_path),
            "sha256": _sha256(outer_path),
            "bytes": outer_path.stat().st_size,
        }

    inner_oof = pd.concat(inner_frames, ignore_index=True)
    outer_oof = pd.concat(outer_frames, ignore_index=True)
    if outer_oof.duplicated(["station", "layer", "time"]).any():
        raise ValueError("outer repeated-forward OOF keys overlap")
    inner_path = planned["inner_oof"]
    outer_path = planned["outer_oof"]
    _atomic_parquet(inner_path, inner_oof)
    _atomic_parquet(outer_path, outer_oof)

    _status(output_dir, 72, "metrics", "fold-equal official-denominator metrics and day bootstrap")
    layer_counts = config["validation"]["official_layer_counts"]
    bootstrap = config["validation"]["bootstrap"]
    metrics = {
        "experiment_id": config["experiment_id"],
        "research_only": True,
        "adaptive_research": True,
        "fresh_holdout_claimed": False,
        "interpretation": "corrected repeated-forward research evidence only; not absolute hidden calibration",
        "inner_role": config["validation"]["inner_role"],
        "hyperparameter_searches": 0,
        "mask_audit": mask_audit.__dict__,
        "fold_contracts": fold_records,
        "inner_diagnostic": {
            "baseline": metric_report(
                inner_oof,
                prediction_column="baseline",
                official_layer_counts=layer_counts,
            ),
            "unprojected_blend50": metric_report(
                inner_oof,
                prediction_column="blend_prediction",
                official_layer_counts=layer_counts,
            ),
            "candidate": metric_report(
                inner_oof,
                prediction_column="prediction",
                official_layer_counts=layer_counts,
            ),
        },
        "outer_repeated_forward": {
            "baseline": metric_report(
                outer_oof,
                prediction_column="baseline",
                official_layer_counts=layer_counts,
            ),
            "unprojected_blend50": metric_report(
                outer_oof,
                prediction_column="blend_prediction",
                official_layer_counts=layer_counts,
            ),
            "candidate": metric_report(
                outer_oof,
                prediction_column="prediction",
                official_layer_counts=layer_counts,
            ),
            "candidate_vs_baseline_bootstrap": paired_fold_day_bootstrap(
                outer_oof,
                reference_column="baseline",
                candidate_column="prediction",
                official_layer_counts=layer_counts,
                replicates=int(bootstrap["replicates"]),
                seed=int(bootstrap["seed"]),
                interval=float(bootstrap["interval"]),
            ),
            "projection_vs_unprojected_bootstrap": paired_fold_day_bootstrap(
                outer_oof,
                reference_column="blend_prediction",
                candidate_column="prediction",
                official_layer_counts=layer_counts,
                replicates=int(bootstrap["replicates"]),
                seed=int(bootstrap["seed"]) + 1,
                interval=float(bootstrap["interval"]),
            ),
        },
    }
    metrics_path = planned["metrics"]
    _atomic_json(metrics_path, metrics)

    _status(output_dir, 84, "full_train", "all legal finite labels, fixed 400-round arms")
    full_train = np.isfinite(population_base.frame["residual"].to_numpy(float))
    final_model = fit_fixed_blend(
        population_base, population_lean, full_train, seed=seed + 10_000
    )
    final_model_path = planned["final_model"]
    _atomic_joblib(final_model_path, final_model)
    model_records["final_full_train"] = {
        "path": _logical(final_model_path),
        "sha256": _sha256(final_model_path),
        "bytes": final_model_path.stat().st_size,
        "training": _training_summary(
            population_base.frame,
            full_train,
            pd.Timestamp("2262-01-01", tz="UTC"),
        ),
    }

    _status(output_dir, 92, "candidate", "full hidden profile projection and official key extraction")
    hidden_start, hidden_stop = config["masking"]["hidden_window_kst_half_open"]
    submission, test_diagnostics = _predict_official_candidate(
        final_model,
        data,
        masked,
        population_base,
        population_lean,
        endpoints,
        hidden_start=hidden_start,
        hidden_stop=hidden_stop,
    )
    candidate_path = planned["candidate"]
    _atomic_csv(candidate_path, submission)
    candidate_validation = validate_submission(candidate_path, data.test_index)
    reloaded = pd.read_csv(candidate_path, dtype={"station": "string", "time": "string"})
    if not reloaded[KEYS].equals(data.test_index[KEYS]):
        raise AssertionError("candidate reload changed official key order")

    frozen_after = _frozen_snapshot()
    if frozen_before != frozen_after:
        raise AssertionError("an existing frozen/current P2 submission changed")
    result = {
        "experiment_id": config["experiment_id"],
        "completed_at_kst": _now(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "decision_scope": "CORRECTED_REPEATED_FORWARD_RESEARCH_EVIDENCE_ONLY",
        "promotion_or_upload_authorized": False,
        "current_frozen_submission_modified": False,
        "hidden_target_temperature_values_accessed": 0,
        "hidden_target_salinity_values_accessed": 0,
        "model_fits": int(2 * len(folds) * 2 + 2),
        "outer_metrics": metrics["outer_repeated_forward"],
        "candidate": {
            "path": _logical(candidate_path),
            "sha256": _sha256(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "validation": candidate_validation,
            "diagnostics": test_diagnostics,
        },
    }
    result_path = planned["result"]
    _atomic_json(result_path, result)

    implementation_paths = [
        ROOT / "src/p2_restore/corrected_repeated_forward.py",
        ROOT / "scripts/run_p2_corrected_repeated_forward.py",
        ROOT / "src/p2_restore/features.py",
        ROOT / "src/p2_restore/model.py",
        ROOT / "src/p2_restore/research.py",
        ROOT / "src/p2_restore/profile_projection.py",
    ]
    artifacts = {
        "inner_oof": {
            "path": _logical(inner_path),
            "sha256": _sha256(inner_path),
            "bytes": inner_path.stat().st_size,
            "rows": int(len(inner_oof)),
        },
        "outer_oof": {
            "path": _logical(outer_path),
            "sha256": _sha256(outer_path),
            "bytes": outer_path.stat().st_size,
            "rows": int(len(outer_oof)),
        },
        "metrics": {
            "path": _logical(metrics_path),
            "sha256": _sha256(metrics_path),
            "bytes": metrics_path.stat().st_size,
        },
        "result": {
            "path": _logical(result_path),
            "sha256": _sha256(result_path),
            "bytes": result_path.stat().st_size,
        },
        "candidate": result["candidate"],
    }
    manifest = {
        "schema_version": "p2_corrected_repeated_forward.manifest.v1",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _now(),
        "research_only": True,
        "upload_allowed": False,
        "adaptive_research": True,
        "fresh_holdout_claimed": False,
        "config": {
            "path": _logical(config_path),
            "sha256": _sha256(config_path),
            "bytes": config_path.stat().st_size,
        },
        "sources": source_records,
        "implementation": {
            _logical(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in implementation_paths
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "git_before": git_before,
        "git_after": _git_state(),
        "attempt_lock": attempt_record,
        "joint_mask": mask_audit.__dict__,
        "feature_counts": {
            "base": len(population_base.feature_columns),
            "lean": len(population_lean.feature_columns),
        },
        "hyperparameter_searches": 0,
        "model_records": model_records,
        "artifacts": artifacts,
        "frozen_submission_snapshot_count": len(frozen_before),
        "frozen_submission_snapshot_unchanged": True,
        "elapsed_seconds": result["elapsed_seconds"],
    }
    manifest_path = planned["manifest"]
    _atomic_json(manifest_path, manifest)
    seal = {
        "experiment_id": config["experiment_id"],
        "manifest_path": _logical(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "candidate_path": _logical(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "outer_oof_sha256": _sha256(outer_path),
        "sealed_at_kst": _now(),
        "upload_performed": False,
    }
    seal_path = planned["seal"]
    _atomic_json(seal_path, seal)
    _status(output_dir, 100, "complete", "corrected research candidate sealed; no upload")
    print(json.dumps({"result": result, "seal": seal}, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = _load_config(config_path)
    _validate_config(config)
    data_dir = resolve_data_dir(args.data_dir)
    if args.dry_run:
        return _dry_run(config, config_path, data_dir, output_dir)
    return _run(config, config_path, data_dir, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
