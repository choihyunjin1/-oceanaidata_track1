"""Run one append-only P2 conservative-stack improvement cycle.

The three stack weights are frozen in the canonical config.  Every corrected
inner/outer underlying model is refit before branch evaluation.  The hidden
target-layer temperature and salinity remain jointly masked for every feature
calculation, and no upload or frozen/current submission mutation is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
DEFAULT_CONFIG = ROOT / "configs/experiments/p2_conservative_stack_improvement_v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts/p2_conservative_stack_improvement_v1"
ATTEMPT_LOCK = ROOT / "artifacts/p2_conservative_stack_improvement_v1_control/attempt.lock"
CANONICAL_CONFIG_SHA256 = "4917f8daf1a84e3a63d2cb5ab884b483d331eed3f0279afb339e944786842212"
PREDECESSOR_ROOT = ROOT / "artifacts/p2_corrected_repeated_forward_v2"
PREDECESSOR_MANIFEST = PREDECESSOR_ROOT / "manifest.json"
PREDECESSOR_METRICS = PREDECESSOR_ROOT / "metrics.json"
PREDECESSOR_OUTER_OOF = PREDECESSOR_ROOT / "oof/outer_repeated_forward.parquet"
PREDECESSOR_INNER_OOF = PREDECESSOR_ROOT / "oof/inner_diagnostic.parquet"
PREDECESSOR_CANDIDATE = PREDECESSOR_ROOT / "candidate/P2_CORRECTED_REPEATED_FORWARD_V2.csv"
PREDECESSOR_FINAL_MODEL = PREDECESSOR_ROOT / "models/final_full_train.joblib"
KST = ZoneInfo("Asia/Seoul")

EXPECTED_BRANCHES = [
    {"id": "STACK_W0500", "candidate_weight": 0.5},
    {"id": "STACK_W0625", "candidate_weight": 0.625},
    {"id": "STACK_W0750", "candidate_weight": 0.75},
]
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"append-only target already exists: {_logical(path)}") from error
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    _exclusive_bytes(path, encoded + b"\n")


def _exclusive_csv(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False, encoding="utf-8", lineterminator="\n").encode("utf-8")
    _exclusive_bytes(path, payload)


def _exclusive_parquet(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    _exclusive_bytes(path, buffer.getvalue())


def _exclusive_joblib(path: Path, value: object) -> None:
    buffer = io.BytesIO()
    joblib.dump(value, buffer)
    _exclusive_bytes(path, buffer.getvalue())


def _progress(number: int, progress: float, phase: str, detail: str) -> None:
    _exclusive_json(
        DEFAULT_OUTPUT / "progress" / f"{number:03d}_{phase}.json",
        {
            "experiment_id": "p2_conservative_stack_improvement_v1",
            "progress": float(progress),
            "phase": phase,
            "detail": detail,
            "updated_at_kst": _now(),
        },
    )


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "p2_conservative_stack_improvement.v1":
        raise ValueError("unexpected P2 stack schema version")
    if config.get("experiment_id") != "p2_conservative_stack_improvement_v1":
        raise ValueError("unexpected P2 stack experiment id")
    if config.get("status") != "authorized_local_improvement_cycle":
        raise ValueError("local improvement cycle is not authorized")
    if config.get("research_only") is not True or config.get("upload_allowed") is not False:
        raise ValueError("generation must remain research-only and non-uploadable")
    if (
        config.get("adaptive_research") is not True
        or config.get("fresh_holdout_claimed") is not False
    ):
        raise ValueError("adaptive/fresh-holdout disclosure changed")
    if config["stack"].get("branches") != EXPECTED_BRANCHES:
        raise ValueError("the three predeclared stack branches changed")
    if config["stack"].get("maximum_branches") != 3:
        raise ValueError("branch budget changed")
    if config["validation"].get("folds") != EXPECTED_FOLDS:
        raise ValueError("corrected fold definitions changed")
    if config["validation"].get("official_layer_counts") != {
        "2": 8713,
        "3": 8712,
        "4": 8636,
    }:
        raise ValueError("official layer denominator changed")
    if int(config["validation"].get("embargo_days", 0)) != 7:
        raise ValueError("seven-day embargo changed")
    if config["validation"].get("bootstrap") != {
        "unit": "KST calendar day within fold",
        "replicates": 2000,
        "interval": 0.9,
        "seed": 20260822,
    }:
        raise ValueError("bootstrap contract changed")
    if config["masking"].get("jointly_null_columns") != ["temp", "psal"]:
        raise ValueError("joint masking columns changed")
    if config["masking"].get("jointly_null_layers") != [2, 3, 4]:
        raise ValueError("joint masking layers changed")
    if config["masking"].get("hidden_target_non_null_allowed") is not False:
        raise ValueError("hidden target values must fail closed")
    if config["base_model"].get("target_layer_inputs") != []:
        raise ValueError("target-layer inputs must remain empty")
    if config["base_model"].get("public_input_layers") != [1, 5, 6, 7, 8]:
        raise ValueError("public input-layer contract changed")
    if config["base_model"].get("fold_seed_base") != 20260822:
        raise ValueError("fold seed changed")
    if config["base_model"].get("full_train_seed") != 20270822:
        raise ValueError("full-training seed changed")
    predecessor = config["predecessor"]
    if predecessor.get("experiment_id") != "p2_corrected_repeated_forward_v2":
        raise ValueError("predecessor experiment changed")
    if predecessor.get("outer_primary_rmse_c") != 1.1158878559665548:
        raise ValueError("predecessor outer metric changed")
    guards = config["winner_guards"]
    if guards.get("minimum_outer_fold_improvements_vs_predecessor") != 2:
        raise ValueError("fold-improvement guard changed")
    if guards.get("maximum_outer_fold_regression_vs_predecessor_c") != 0.15:
        raise ValueError("fold-regression guard changed")
    output = config["output_contract"]
    if output.get("append_only") is not True:
        raise ValueError("output must remain append-only")
    if output.get("default_directory") != "artifacts/p2_conservative_stack_improvement_v1":
        raise ValueError("canonical output directory changed")
    if output.get("attempt_lock") != (
        "artifacts/p2_conservative_stack_improvement_v1_control/attempt.lock"
    ):
        raise ValueError("canonical attempt lock changed")
    if output.get("current_frozen_submission_mutation_allowed") is not False:
        raise ValueError("frozen/current mutation must remain forbidden")
    if output.get("test_rows") != 26_061 or output.get("columns") != [
        "station",
        "layer",
        "time",
        "temp",
    ]:
        raise ValueError("candidate contract changed")


def _canonical_preflight(
    config: Mapping[str, Any], config_path: Path, output_dir: Path
) -> dict[str, Any]:
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("only the canonical stack config path is accepted")
    if output_dir.resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError("only the canonical stack output directory is accepted")
    if _sha256(DEFAULT_CONFIG) != CANONICAL_CONFIG_SHA256:
        raise ValueError("canonical stack config SHA-256 changed")
    canonical = _load_json(DEFAULT_CONFIG)
    _validate_config(canonical)
    if dict(config) != canonical:
        raise ValueError("passed config differs from the canonical stack config")
    return canonical


def _contained(path: Path, root: Path, *, role: str) -> Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{role} escapes the canonical output directory") from error
    if not relative.parts:
        raise ValueError(f"{role} must be below the output directory")
    return resolved


def _planned_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    output = config["output_contract"]
    paths = {
        "inner_oof": DEFAULT_OUTPUT / "oof/inner_stack.parquet",
        "outer_oof": DEFAULT_OUTPUT / "oof/outer_stack.parquet",
        "metrics": DEFAULT_OUTPUT / "metrics.json",
        "result": DEFAULT_OUTPUT / "result.json",
        "manifest": DEFAULT_OUTPUT / "manifest.json",
        "seal": DEFAULT_OUTPUT / "seal.json",
        "candidate": DEFAULT_OUTPUT / str(output["candidate_relative_path"]),
        "reproduction": DEFAULT_OUTPUT / str(output["reproduction_relative_path"]),
        "final_underlying_model": DEFAULT_OUTPUT / "models/final_underlying_blend.joblib",
        "final_stack_model": DEFAULT_OUTPUT / "models/final_stack_model.joblib",
    }
    for fold in EXPECTED_FOLDS:
        for stage in ("inner", "outer"):
            role = f"model_{fold['name']}_{stage}"
            paths[role] = DEFAULT_OUTPUT / "models" / f"{fold['name']}_{stage}.joblib"
    resolved = {role: _contained(path, DEFAULT_OUTPUT, role=role) for role, path in paths.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("planned output paths collide")
    return resolved


def _verify_predecessor(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    predecessor = config["predecessor"]
    expected = {
        "manifest": (PREDECESSOR_MANIFEST, predecessor["manifest_sha256"]),
        "metrics": (PREDECESSOR_METRICS, predecessor["metrics_sha256"]),
        "inner_oof": (PREDECESSOR_INNER_OOF, predecessor["inner_oof_sha256"]),
        "outer_oof": (PREDECESSOR_OUTER_OOF, predecessor["outer_oof_sha256"]),
        "candidate": (PREDECESSOR_CANDIDATE, predecessor["candidate_sha256"]),
        "final_model": (PREDECESSOR_FINAL_MODEL, predecessor["final_model_sha256"]),
    }
    records: dict[str, dict[str, Any]] = {}
    for role, (path, pinned) in expected.items():
        actual = _sha256(path)
        if actual != pinned:
            raise ValueError(f"predecessor {role} SHA-256 mismatch")
        records[role] = {"path": _logical(path), "sha256": actual, "bytes": path.stat().st_size}
    return records


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


def _prewrite_guard(config: Mapping[str, Any], data_dir: Path) -> dict[str, Path]:
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError("canonical stack output already exists")
    if ATTEMPT_LOCK.exists():
        raise FileExistsError("canonical stack attempt was already consumed")
    planned = _planned_paths(config)
    protected_roots = [
        data_dir.resolve(),
        (ROOT / "submissions").resolve(),
        (ROOT / "output/2026-08-20/ready").resolve(),
        PREDECESSOR_ROOT.resolve(),
    ]
    for role, path in planned.items():
        if path.exists():
            raise FileExistsError(f"planned target already exists: {role}")
        for protected in protected_roots:
            try:
                path.relative_to(protected)
            except ValueError:
                continue
            raise ValueError(f"planned target enters protected tree: {role}")
    return planned


def _acquire_attempt_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "experiment_id": config["experiment_id"],
        "attempt": 1,
        "canonical_config_sha256": CANONICAL_CONFIG_SHA256,
        "status": "ATTEMPT_CONSUMED",
        "rerun_allowed": False,
        "created_at_kst": _now(),
    }
    _exclusive_json(ATTEMPT_LOCK, payload)
    return {
        "path": _logical(ATTEMPT_LOCK),
        "sha256": _sha256(ATTEMPT_LOCK),
        "bytes": ATTEMPT_LOCK.stat().st_size,
    }


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


def _coverage(population: FeatureTable, start: str, stop: str) -> dict[str, Any]:
    window = window_mask(population.frame, start, stop)
    scored = window & np.isfinite(population.frame["target"].to_numpy(float))
    nominal = nominal_target_rows(start, stop)
    return {
        "nominal_rows": int(nominal),
        "eligible_feature_rows": int(window.sum()),
        "finite_target_rows": int(scored.sum()),
        "target_coverage": float(scored.sum() / nominal),
        "rows_by_layer": {
            str(layer): int((scored & population.frame["layer"].eq(layer).to_numpy(bool)).sum())
            for layer in (2, 3, 4)
        },
    }


def _training_summary(
    frame: pd.DataFrame, selected: np.ndarray, cutoff: pd.Timestamp
) -> dict[str, Any]:
    time_values = pd.to_datetime(frame.loc[selected, "time"], utc=True)
    return {
        "rows": int(selected.sum()),
        "rows_by_layer": {
            str(layer): int((selected & frame["layer"].eq(layer).to_numpy(bool)).sum())
            for layer in (2, 3, 4)
        },
        "first_label_time_utc": time_values.min().isoformat(),
        "last_label_time_utc": time_values.max().isoformat(),
        "exclusive_cutoff_utc": cutoff.isoformat(),
        "max_label_precedes_cutoff": bool(time_values.max() < cutoff),
    }


def stack_prediction(
    baseline: np.ndarray | pd.Series,
    corrected_prediction: np.ndarray | pd.Series,
    candidate_weight: float,
) -> np.ndarray:
    if not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("candidate weight must be in [0, 1]")
    baseline_values = np.asarray(baseline, dtype=float)
    corrected_values = np.asarray(corrected_prediction, dtype=float)
    if baseline_values.shape != corrected_values.shape:
        raise ValueError("stack inputs have different shapes")
    prediction = baseline_values + candidate_weight * (corrected_values - baseline_values)
    if not np.isfinite(prediction).all():
        raise ValueError("stack produced non-finite predictions")
    return prediction


def _add_stack_columns(frame: pd.DataFrame, branches: list[dict[str, Any]]) -> pd.DataFrame:
    result = frame.copy()
    for branch in branches:
        result[f"prediction_{branch['id']}"] = stack_prediction(
            result["baseline"], result["incumbent_prediction"], branch["candidate_weight"]
        )
    return result


def _fold_rmse_map(report: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(name): float(value["official_layer_weighted_rmse_c"])
        for name, value in report["by_fold"].items()
    }


def branch_guard_report(
    *,
    branch_id: str,
    outer_report: Mapping[str, Any],
    inner_report: Mapping[str, Any],
    outer_baseline_rmse: float,
    inner_baseline_rmse: float,
    predecessor_outer: Mapping[str, Any],
    predecessor_inner_rmse: float,
    guards: Mapping[str, Any],
) -> dict[str, Any]:
    outer_rmse = float(outer_report["fold_equal_official_layer_weighted_rmse_c"])
    inner_rmse = float(inner_report["fold_equal_official_layer_weighted_rmse_c"])
    predecessor_outer_rmse = float(predecessor_outer["fold_equal_official_layer_weighted_rmse_c"])
    previous_folds = _fold_rmse_map(predecessor_outer)
    branch_folds = _fold_rmse_map(outer_report)
    fold_deltas = {name: branch_folds[name] - previous_folds[name] for name in previous_folds}
    improvement_count = sum(delta < 0.0 for delta in fold_deltas.values())
    maximum_regression = max(fold_deltas.values())
    predecessor_inner_excess = predecessor_inner_rmse - inner_baseline_rmse
    branch_inner_excess = inner_rmse - inner_baseline_rmse
    directionality_gap = abs(
        (outer_rmse - outer_baseline_rmse) - (inner_rmse - inner_baseline_rmse)
    )
    checks = {
        "outer_strict_improvement": outer_rmse < predecessor_outer_rmse,
        "minimum_two_of_three_fold_improvements": improvement_count
        >= int(guards["minimum_outer_fold_improvements_vs_predecessor"]),
        "maximum_fold_regression": maximum_regression
        <= float(guards["maximum_outer_fold_regression_vs_predecessor_c"]),
        "inner_excess_reduced": branch_inner_excess < predecessor_inner_excess,
        "directionality_gap_reduced": directionality_gap
        < float(guards["inner_outer_directionality_gap_strictly_below_predecessor_c"]),
    }
    return {
        "branch_id": branch_id,
        "eligible": all(checks.values()),
        "checks": checks,
        "outer_primary_rmse_c": outer_rmse,
        "outer_delta_vs_predecessor_c": outer_rmse - predecessor_outer_rmse,
        "outer_fold_improvement_count": int(improvement_count),
        "outer_fold_delta_vs_predecessor_c": fold_deltas,
        "maximum_outer_fold_regression_c": float(maximum_regression),
        "inner_primary_rmse_c": inner_rmse,
        "inner_excess_over_baseline_c": float(branch_inner_excess),
        "predecessor_inner_excess_over_baseline_c": float(predecessor_inner_excess),
        "inner_outer_directionality_gap_c": float(directionality_gap),
    }


def select_winner(branches: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [branch for branch in branches if branch["guard"]["eligible"]]
    if not eligible:
        raise RuntimeError("no predeclared stack branch passed every winner guard")
    return min(eligible, key=lambda branch: branch["guard"]["outer_primary_rmse_c"])


def _predict_official_underlying(
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
    full_keys = full_base.frame.loc[:, KEYS].copy()
    full_keys["_utc"] = pd.to_datetime(full_keys["time"], utc=True)
    official_keys = official_base.frame.loc[:, KEYS].copy()
    official_keys["_utc"] = pd.to_datetime(official_keys["time"], utc=True)
    official_keys["_official_row"] = np.arange(len(official_keys))
    official_keys["_official_prediction"] = official_unprojected
    aligned = full_keys.merge(
        official_keys.loc[:, ["station", "layer", "_utc", "_official_row", "_official_prediction"]],
        on=["station", "layer", "_utc"],
        how="left",
        validate="one_to_one",
    )
    matched = aligned["_official_row"].notna().to_numpy(bool)
    if int(matched.sum()) != len(data.test_index):
        raise ValueError("full hidden profile does not cover every official key")
    combined = full_unprojected.copy()
    combined[matched] = aligned.loc[matched, "_official_prediction"].to_numpy(float)
    projection = project_profiles_vectorized(full_base.frame, combined, endpoints)
    official_rows = aligned.loc[matched, "_official_row"].to_numpy(int)
    official_prediction = np.empty(len(data.test_index), dtype=float)
    official_prediction[official_rows] = projection.prediction[matched]
    return build_submission(data.test_index, official_prediction), {
        "official_rows": int(len(data.test_index)),
        "full_hidden_profile_rows": int(len(full_base.frame)),
        "full_hidden_truth_non_null": int(full_base.frame["target"].notna().sum()),
        "projection_eligible_full_rows": int(projection.eligible_mask.sum()),
        "projection_active_full_rows": int(projection.active_mask.sum()),
        "projection_active_official_rows": int(projection.active_mask[matched].sum()),
        "prediction_min_c": float(official_prediction.min()),
        "prediction_max_c": float(official_prediction.max()),
    }


def _official_stack_submission(
    data: P2Data, underlying: pd.DataFrame, weight: float
) -> pd.DataFrame:
    if not underlying[KEYS].equals(data.test_index[KEYS]):
        raise AssertionError("underlying candidate keys differ from official test index")
    if not data.baseline[KEYS].equals(data.test_index[KEYS]):
        raise AssertionError("official interpolation baseline keys differ from test index")
    prediction = stack_prediction(data.baseline["temp"], underlying["temp"], weight)
    return build_submission(data.test_index, prediction)


def _build_runtime(data_dir: Path) -> dict[str, Any]:
    data = load_p2_data(data_dir)
    hidden_start, hidden_stop = _load_json(DEFAULT_CONFIG)["masking"]["hidden_window_kst_half_open"]
    masked, mask_audit = joint_mask_target_context(
        data.observations, hidden_start=hidden_start, hidden_stop=hidden_stop
    )
    base = build_joint_masked_population(data.observations, masked)
    lean = build_fixed_lean_arm(base, masked)
    endpoints = public_endpoints_from_masked_context(masked)
    forbidden = [
        column
        for column in lean.feature_columns
        if column.startswith(("temp_2", "temp_3", "temp_4", "psal_2", "psal_3", "psal_4"))
    ]
    if forbidden:
        raise AssertionError(f"target-layer features entered schema: {forbidden}")
    return {
        "data": data,
        "masked": masked,
        "mask_audit": mask_audit,
        "base": base,
        "lean": lean,
        "endpoints": endpoints,
    }


def _dry_run(config: Mapping[str, Any], config_path: Path, data_dir: Path, output_dir: Path) -> int:
    config = _canonical_preflight(config, config_path, output_dir)
    _prewrite_guard(config, data_dir)
    sources = _verify_sources(config, data_dir)
    predecessor = _verify_predecessor(config)
    runtime = _build_runtime(data_dir)
    coverage: dict[str, Any] = {}
    for fold in EXPECTED_FOLDS:
        inner = _coverage(runtime["base"], *fold["inner"])
        outer = _coverage(runtime["base"], *fold["outer"])
        if inner["target_coverage"] < 0.96 or outer["target_coverage"] < 0.96:
            raise ValueError(f"fold {fold['name']} fails the fixed coverage floor")
        coverage[fold["name"]] = {"inner": inner, "outer": outer}
    print(
        json.dumps(
            {
                "dry_run": "PASS",
                "experiment_id": config["experiment_id"],
                "config_sha256": CANONICAL_CONFIG_SHA256,
                "predeclared_branches": EXPECTED_BRANCHES,
                "sources_verified": len(sources),
                "predecessor_artifacts_verified": len(predecessor),
                "joint_mask": runtime["mask_audit"].__dict__,
                "feature_counts": {
                    "base": len(runtime["base"].feature_columns),
                    "lean": len(runtime["lean"].feature_columns),
                },
                "fold_coverage": coverage,
                "planned_underlying_estimator_fits": 14,
                "actual_model_fits": 0,
                "candidate_writes": 0,
                "attempt_lock_created": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run(config: Mapping[str, Any], config_path: Path, data_dir: Path, output_dir: Path) -> int:
    config = _canonical_preflight(config, config_path, output_dir)
    planned = _prewrite_guard(config, data_dir)
    predecessor_before = _verify_predecessor(config)
    source_records = _verify_sources(config, data_dir)
    frozen_before = _frozen_snapshot()
    ready_key = "output/2026-08-20/ready/P2_submission.csv"
    if frozen_before.get(ready_key) != config["predecessor"]["frozen_submission_sha256"]:
        raise ValueError("frozen P2 submission SHA-256 changed before fit")
    attempt_record = _acquire_attempt_lock(config)
    started = time.perf_counter()
    git_before = _git_state()
    _progress(1, 2, "preflight", "canonical config, source, predecessor, and frozen hashes passed")
    runtime = _build_runtime(data_dir)
    data: P2Data = runtime["data"]
    base: FeatureTable = runtime["base"]
    lean: FeatureTable = runtime["lean"]
    endpoints: pd.DataFrame = runtime["endpoints"]
    _progress(2, 8, "features", "joint target temp+psal mask and public-only features complete")

    inner_frames: list[pd.DataFrame] = []
    outer_frames: list[pd.DataFrame] = []
    fold_records: dict[str, Any] = {}
    model_records: dict[str, Any] = {}
    embargo_days = int(config["validation"]["embargo_days"])
    seed_base = int(config["base_model"]["fold_seed_base"])
    event = 3
    for number, fold in enumerate(EXPECTED_FOLDS):
        name = str(fold["name"])
        inner_coverage = _coverage(base, *fold["inner"])
        outer_coverage = _coverage(base, *fold["outer"])
        if inner_coverage["target_coverage"] < 0.96 or outer_coverage["target_coverage"] < 0.96:
            raise ValueError(f"fold {name} violates the coverage floor")
        fold_record = {
            "inner_window_kst": fold["inner"],
            "outer_window_kst": fold["outer"],
            "same_season_priority": bool(fold["same_season_priority"]),
            "inner_coverage": inner_coverage,
            "outer_coverage": outer_coverage,
        }
        for stage, offset in (("inner", 0), ("outer", 1)):
            validation_window = fold[stage]
            selected, cutoff = forward_training_mask(
                base.frame, validation_window[0], embargo_days=embargo_days
            )
            model_seed = seed_base + number * 10 + offset
            _progress(
                event,
                15 + (number * 2 + offset) * 9,
                f"fit_{name}_{stage}",
                f"fresh corrected {stage} fit with seed {model_seed}",
            )
            event += 1
            model = fit_fixed_blend(base, lean, selected, seed=model_seed)
            prediction = predict_scored_window(
                model,
                base,
                lean,
                endpoints,
                start=validation_window[0],
                stop=validation_window[1],
                fold=name,
                stage=stage,
            ).rename(columns={"prediction": "incumbent_prediction"})
            model_path = planned[f"model_{name}_{stage}"]
            _exclusive_joblib(model_path, model)
            model_records[f"{name}_{stage}"] = {
                "path": _logical(model_path),
                "sha256": _sha256(model_path),
                "bytes": model_path.stat().st_size,
                "seed": model_seed,
                "estimator_fits": 2,
            }
            fold_record[f"{stage}_training"] = _training_summary(base.frame, selected, cutoff)
            (inner_frames if stage == "inner" else outer_frames).append(prediction)
        fold_records[name] = fold_record

    inner_oof = pd.concat(inner_frames, ignore_index=True)
    outer_oof = pd.concat(outer_frames, ignore_index=True)
    if outer_oof.duplicated(KEYS).any():
        raise ValueError("fresh outer OOF keys overlap")
    previous_inner = pd.read_parquet(PREDECESSOR_INNER_OOF)
    previous_outer = pd.read_parquet(PREDECESSOR_OUTER_OOF)
    comparison_columns = [*KEYS, "fold", "stage"]
    reproduction_error: dict[str, float] = {}
    for stage, fresh, previous in (
        ("inner", inner_oof, previous_inner),
        ("outer", outer_oof, previous_outer),
    ):
        if not fresh[comparison_columns].equals(previous[comparison_columns]):
            raise AssertionError(f"fresh {stage} OOF keys differ from predecessor")
        for fresh_column, previous_column in (
            ("truth", "truth"),
            ("baseline", "baseline"),
            ("incumbent_prediction", "prediction"),
        ):
            error = float(
                np.max(
                    np.abs(
                        fresh[fresh_column].to_numpy(float)
                        - previous[previous_column].to_numpy(float)
                    )
                )
            )
            reproduction_error[f"{stage}_{fresh_column}"] = error
            if error > 1e-12:
                raise AssertionError(f"fresh {stage} {fresh_column} reproduction drift: {error}")

    inner_oof = _add_stack_columns(inner_oof, config["stack"]["branches"])
    outer_oof = _add_stack_columns(outer_oof, config["stack"]["branches"])
    _exclusive_parquet(planned["inner_oof"], inner_oof)
    _exclusive_parquet(planned["outer_oof"], outer_oof)
    _progress(
        9, 70, "oof", "fresh corrected OOF reproduced; three frozen stack branches materialized"
    )

    layer_counts = config["validation"]["official_layer_counts"]
    previous_metrics = _load_json(PREDECESSOR_METRICS)
    predecessor_outer = previous_metrics["outer_repeated_forward"]["candidate"]
    predecessor_inner = previous_metrics["inner_diagnostic"]["candidate"]
    outer_baseline = metric_report(
        outer_oof, prediction_column="baseline", official_layer_counts=layer_counts
    )
    inner_baseline = metric_report(
        inner_oof, prediction_column="baseline", official_layer_counts=layer_counts
    )
    branch_records: list[dict[str, Any]] = []
    for branch in config["stack"]["branches"]:
        column = f"prediction_{branch['id']}"
        outer_report = metric_report(
            outer_oof, prediction_column=column, official_layer_counts=layer_counts
        )
        inner_report = metric_report(
            inner_oof, prediction_column=column, official_layer_counts=layer_counts
        )
        guard = branch_guard_report(
            branch_id=branch["id"],
            outer_report=outer_report,
            inner_report=inner_report,
            outer_baseline_rmse=float(outer_baseline["fold_equal_official_layer_weighted_rmse_c"]),
            inner_baseline_rmse=float(inner_baseline["fold_equal_official_layer_weighted_rmse_c"]),
            predecessor_outer=predecessor_outer,
            predecessor_inner_rmse=float(
                predecessor_inner["fold_equal_official_layer_weighted_rmse_c"]
            ),
            guards=config["winner_guards"],
        )
        branch_records.append(
            {
                "id": branch["id"],
                "candidate_weight": float(branch["candidate_weight"]),
                "inner": inner_report,
                "outer": outer_report,
                "guard": guard,
            }
        )
    winner = select_winner(branch_records)
    winner_column = f"prediction_{winner['id']}"
    bootstrap = config["validation"]["bootstrap"]
    winner_vs_predecessor = paired_fold_day_bootstrap(
        outer_oof,
        reference_column="incumbent_prediction",
        candidate_column=winner_column,
        official_layer_counts=layer_counts,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]) + 50,
        interval=float(bootstrap["interval"]),
    )
    winner_vs_baseline = paired_fold_day_bootstrap(
        outer_oof,
        reference_column="baseline",
        candidate_column=winner_column,
        official_layer_counts=layer_counts,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]) + 51,
        interval=float(bootstrap["interval"]),
    )
    metrics = {
        "schema_version": "p2_conservative_stack_improvement.metrics.v1",
        "experiment_id": config["experiment_id"],
        "research_only": True,
        "adaptive_research": True,
        "fresh_holdout_claimed": False,
        "primary_metric": config["validation"]["primary_metric"],
        "predecessor": {
            "outer_candidate": predecessor_outer,
            "inner_candidate": predecessor_inner,
        },
        "fresh_reproduction_max_abs_error_c": reproduction_error,
        "inner_baseline": inner_baseline,
        "outer_baseline": outer_baseline,
        "branches": branch_records,
        "winner": {
            "id": winner["id"],
            "candidate_weight": winner["candidate_weight"],
            "guard": winner["guard"],
            "inner": winner["inner"],
            "outer": winner["outer"],
            "winner_vs_predecessor_bootstrap": winner_vs_predecessor,
            "winner_vs_baseline_bootstrap": winner_vs_baseline,
        },
        "fold_contracts": fold_records,
        "mask_audit": runtime["mask_audit"].__dict__,
    }
    _exclusive_json(planned["metrics"], metrics)
    _progress(10, 78, "winner", f"{winner['id']} passed all guards and won the fixed branch screen")

    full_train = np.isfinite(base.frame["residual"].to_numpy(float))
    full_seed = int(config["base_model"]["full_train_seed"])
    final_underlying = fit_fixed_blend(base, lean, full_train, seed=full_seed)
    _exclusive_joblib(planned["final_underlying_model"], final_underlying)
    stack_model = {
        "schema_version": "p2_conservative_stack_model.v1",
        "experiment_id": config["experiment_id"],
        "winner_id": winner["id"],
        "candidate_weight": float(winner["candidate_weight"]),
        "formula": config["stack"]["formula"],
        "underlying_model": final_underlying,
        "base_feature_columns": list(base.feature_columns),
        "lean_feature_columns": list(lean.feature_columns),
        "source_contract": dict(config["source_contract"]),
        "hidden_target_inputs": [],
    }
    _exclusive_joblib(planned["final_stack_model"], stack_model)
    model_records["final_underlying_blend"] = {
        "path": _logical(planned["final_underlying_model"]),
        "sha256": _sha256(planned["final_underlying_model"]),
        "bytes": planned["final_underlying_model"].stat().st_size,
        "seed": full_seed,
        "estimator_fits": 2,
        "training": _training_summary(base.frame, full_train, pd.Timestamp("2262-01-01", tz="UTC")),
    }
    model_records["final_stack_model"] = {
        "path": _logical(planned["final_stack_model"]),
        "sha256": _sha256(planned["final_stack_model"]),
        "bytes": planned["final_stack_model"].stat().st_size,
        "winner_id": winner["id"],
        "candidate_weight": winner["candidate_weight"],
    }
    _progress(
        11, 88, "full_train", "full legal-label underlying model and stored stack model fitted"
    )

    hidden_start, hidden_stop = config["masking"]["hidden_window_kst_half_open"]
    underlying_submission, inference_diagnostics = _predict_official_underlying(
        final_underlying,
        data,
        runtime["masked"],
        base,
        lean,
        endpoints,
        hidden_start=hidden_start,
        hidden_stop=hidden_stop,
    )
    submission = _official_stack_submission(
        data, underlying_submission, float(winner["candidate_weight"])
    )
    _exclusive_csv(planned["candidate"], submission)
    candidate_validation = validate_submission(planned["candidate"], data.test_index)

    loaded_stack = joblib.load(planned["final_stack_model"])
    if loaded_stack["base_feature_columns"] != list(base.feature_columns):
        raise AssertionError("saved stack base feature schema changed")
    if loaded_stack["lean_feature_columns"] != list(lean.feature_columns):
        raise AssertionError("saved stack lean feature schema changed")
    reproduced_underlying, _ = _predict_official_underlying(
        loaded_stack["underlying_model"],
        data,
        runtime["masked"],
        base,
        lean,
        endpoints,
        hidden_start=hidden_start,
        hidden_stop=hidden_stop,
    )
    reproduced = _official_stack_submission(
        data, reproduced_underlying, float(loaded_stack["candidate_weight"])
    )
    _exclusive_csv(planned["reproduction"], reproduced)
    if planned["candidate"].read_bytes() != planned["reproduction"].read_bytes():
        raise AssertionError("saved stack-model reinference is not byte-identical")
    if not pd.read_csv(planned["candidate"], dtype={"station": "string", "time": "string"})[
        KEYS
    ].equals(data.test_index[KEYS]):
        raise AssertionError("candidate reload changed official key order")

    predecessor_candidate = pd.read_csv(
        PREDECESSOR_CANDIDATE, dtype={"station": "string", "time": "string"}
    )
    if not predecessor_candidate[KEYS].equals(data.test_index[KEYS]):
        raise AssertionError("predecessor candidate keys changed")
    difference = submission["temp"].to_numpy(float) - predecessor_candidate["temp"].to_numpy(float)
    frozen_after = _frozen_snapshot()
    predecessor_after = _verify_predecessor(config)
    source_after = _verify_sources(config, data_dir)
    if frozen_before != frozen_after:
        raise AssertionError("a frozen/current P2 submission changed")
    if predecessor_before != predecessor_after:
        raise AssertionError("a predecessor artifact changed")
    if source_records != source_after:
        raise AssertionError("a source artifact changed")

    candidate_record = {
        "path": _logical(planned["candidate"]),
        "sha256": _sha256(planned["candidate"]),
        "bytes": planned["candidate"].stat().st_size,
        "validation": candidate_validation,
        "minimum_c": float(submission["temp"].min()),
        "maximum_c": float(submission["temp"].max()),
        "changed_rows_vs_predecessor": int(np.count_nonzero(difference)),
        "mean_absolute_difference_vs_predecessor_c": float(np.mean(np.abs(difference))),
        "rmse_distance_vs_predecessor_c": float(np.sqrt(np.mean(difference**2))),
        "saved_model_reproduction": {
            "path": _logical(planned["reproduction"]),
            "sha256": _sha256(planned["reproduction"]),
            "byte_identical": True,
        },
        "inference_diagnostics": inference_diagnostics,
    }
    result = {
        "schema_version": "p2_conservative_stack_improvement.result.v1",
        "experiment_id": config["experiment_id"],
        "completed_at_kst": _now(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "status": "WINNER_FULL_MODEL_AND_CANDIDATE_COMPLETE",
        "winner_id": winner["id"],
        "winner_candidate_weight": winner["candidate_weight"],
        "winner_outer_primary_rmse_c": winner["guard"]["outer_primary_rmse_c"],
        "winner_outer_delta_vs_predecessor_c": winner["guard"]["outer_delta_vs_predecessor_c"],
        "winner_outer_fold_improvement_count": winner["guard"]["outer_fold_improvement_count"],
        "all_winner_guards_passed": winner["guard"]["eligible"],
        "underlying_lightgbm_estimator_fits": 14,
        "hidden_target_temperature_values_accessed": 0,
        "hidden_target_salinity_values_accessed": 0,
        "current_frozen_submission_modified": False,
        "predecessor_artifacts_modified": False,
        "promotion_or_upload_authorized": False,
        "upload_performed": False,
        "candidate": candidate_record,
    }
    _exclusive_json(planned["result"], result)

    implementation_paths = [
        DEFAULT_CONFIG,
        Path(__file__).resolve(),
        ROOT / "src/p2_restore/corrected_repeated_forward.py",
        ROOT / "src/p2_restore/data.py",
        ROOT / "src/p2_restore/features.py",
        ROOT / "src/p2_restore/model.py",
        ROOT / "src/p2_restore/research.py",
        ROOT / "src/p2_restore/profile_projection.py",
        ROOT / "src/p2_restore/submission.py",
    ]
    artifacts = {
        "inner_oof": {
            "path": _logical(planned["inner_oof"]),
            "sha256": _sha256(planned["inner_oof"]),
            "bytes": planned["inner_oof"].stat().st_size,
            "rows": int(len(inner_oof)),
        },
        "outer_oof": {
            "path": _logical(planned["outer_oof"]),
            "sha256": _sha256(planned["outer_oof"]),
            "bytes": planned["outer_oof"].stat().st_size,
            "rows": int(len(outer_oof)),
        },
        "metrics": {
            "path": _logical(planned["metrics"]),
            "sha256": _sha256(planned["metrics"]),
            "bytes": planned["metrics"].stat().st_size,
        },
        "result": {
            "path": _logical(planned["result"]),
            "sha256": _sha256(planned["result"]),
            "bytes": planned["result"].stat().st_size,
        },
        "candidate": candidate_record,
    }
    manifest = {
        "schema_version": "p2_conservative_stack_improvement.manifest.v1",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _now(),
        "research_only": True,
        "upload_allowed": False,
        "adaptive_research": True,
        "fresh_holdout_claimed": False,
        "config": {
            "path": _logical(DEFAULT_CONFIG),
            "sha256": _sha256(DEFAULT_CONFIG),
            "bytes": DEFAULT_CONFIG.stat().st_size,
        },
        "sources": source_records,
        "predecessor_pins": predecessor_before,
        "implementation": {
            _logical(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in implementation_paths
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                name: metadata.version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "lightgbm",
                    "scikit-learn",
                    "joblib",
                    "pyarrow",
                ]
            },
        },
        "git_before": git_before,
        "git_after": _git_state(),
        "attempt_lock": attempt_record,
        "joint_mask": runtime["mask_audit"].__dict__,
        "feature_counts": {"base": len(base.feature_columns), "lean": len(lean.feature_columns)},
        "branch_budget": 3,
        "branch_order": [branch["id"] for branch in config["stack"]["branches"]],
        "winner": {
            "id": winner["id"],
            "candidate_weight": winner["candidate_weight"],
            "guard": winner["guard"],
        },
        "model_records": model_records,
        "artifacts": artifacts,
        "frozen_submission_snapshot_count": len(frozen_before),
        "frozen_submission_snapshot_unchanged": True,
        "predecessor_artifacts_unchanged": True,
        "source_artifacts_unchanged": True,
        "hidden_target_temperature_values_accessed": 0,
        "hidden_target_salinity_values_accessed": 0,
        "upload_performed": False,
        "elapsed_seconds": result["elapsed_seconds"],
    }
    _exclusive_json(planned["manifest"], manifest)
    seal = {
        "schema_version": "p2_conservative_stack_improvement.seal.v1",
        "experiment_id": config["experiment_id"],
        "winner_id": winner["id"],
        "winner_outer_primary_rmse_c": winner["guard"]["outer_primary_rmse_c"],
        "manifest_path": _logical(planned["manifest"]),
        "manifest_sha256": _sha256(planned["manifest"]),
        "candidate_path": _logical(planned["candidate"]),
        "candidate_sha256": _sha256(planned["candidate"]),
        "reproduction_sha256": _sha256(planned["reproduction"]),
        "final_stack_model_sha256": _sha256(planned["final_stack_model"]),
        "outer_oof_sha256": _sha256(planned["outer_oof"]),
        "sealed_at_kst": _now(),
        "current_frozen_submission_modified": False,
        "upload_performed": False,
    }
    _exclusive_json(planned["seal"], seal)
    _progress(12, 100, "complete", "winner full model, candidate, reproduction, and seal complete")
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
    config = _load_json(config_path)
    _validate_config(config)
    data_dir = resolve_data_dir(args.data_dir)
    if args.dry_run:
        return _dry_run(config, config_path, data_dir, output_dir)
    return _run(config, config_path, data_dir, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
