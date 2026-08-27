"""Resume the same P2 stack attempt with a serialization-only correction.

The initial run completed all 14 LightGBM estimator fits, exposed the fixed
three-branch metrics, selected STACK_W0625, trained the full model, and wrote
the candidate.  It then failed closed while reloading a joblib payload because
binary bytes had been sent through a Windows text-mode descriptor.  This
resume performs no fit and changes no model behavior.  It reverses the exact
LF-to-CRLF expansion into new append-only paths, validates every saved model,
re-materializes all six OOF predictions, and seals only after the repaired
stack model reproduces the already-written candidate byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
    metric_report,
    paired_fold_day_bootstrap,
    predict_scored_window,
)
from p2_restore.data import KEYS, P2Data, resolve_data_dir
from p2_restore.submission import validate_submission

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/p2_conservative_stack_improvement_v1"
CONTROL = ROOT / "artifacts/p2_conservative_stack_improvement_v1_control"
ORIGINAL_CONFIG = ROOT / "configs/experiments/p2_conservative_stack_improvement_v1.json"
RESUME_CONFIG = ROOT / "configs/experiments/p2_conservative_stack_improvement_v1_resume.json"
ORIGINAL_RUNNER = ROOT / "scripts/run_p2_conservative_stack_improvement_v1.py"
RESUME_LOCK = CONTROL / "resume.lock"
CANONICAL_RESUME_CONFIG_SHA256 = "a060e81e0aa3e159fe259a984d2d7e4f7e63b2f87a7ed0ccb36ce3896d8ece64"
KST = ZoneInfo("Asia/Seoul")

EXPECTED_BRANCHES = [
    {"id": "STACK_W0500", "candidate_weight": 0.5},
    {"id": "STACK_W0625", "candidate_weight": 0.625},
    {"id": "STACK_W0750", "candidate_weight": 0.75},
]
MODEL_ROLES = [
    "outer_2024_sep_oct_inner",
    "outer_2024_sep_oct_outer",
    "outer_2025_may_jun_inner",
    "outer_2025_may_jun_outer",
    "outer_2025_jul_aug_inner",
    "outer_2025_jul_aug_outer",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _exclusive_binary(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
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
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    _exclusive_binary(path, payload + b"\n")


def repair_windows_text_expansion(payload: bytes) -> bytes:
    """Reverse the exact Windows text-mode LF -> CRLF byte expansion."""

    repaired = payload.replace(b"\r\n", b"\n")
    if repaired == payload:
        raise ValueError("affected binary contains no reversible CRLF expansion")
    return repaired


def _load_original_runner(config: Mapping[str, Any]):
    expected = config["initial_pins"]["initial_runner"]
    if _sha256(ORIGINAL_RUNNER) != expected["sha256"]:
        raise ValueError("initial runner SHA-256 changed")
    spec = importlib.util.spec_from_file_location("p2_stack_initial_runner", ORIGINAL_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the pinned initial runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_resume_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "p2_conservative_stack_improvement.resume.v1":
        raise ValueError("unexpected resume schema")
    if config.get("experiment_id") != "p2_conservative_stack_improvement_v1":
        raise ValueError("unexpected resume experiment")
    if config.get("status") != "authorized_same_attempt_serialization_correction":
        raise ValueError("same-attempt correction is not authorized")
    if config.get("same_original_attempt") != 1:
        raise ValueError("resume must use the original attempt")
    if config.get("new_generation_or_attempt") is not False:
        raise ValueError("resume cannot open a new generation or attempt")
    if config.get("research_only") is not True or config.get("upload_allowed") is not False:
        raise ValueError("resume must remain research-only and non-uploadable")
    if config.get("metrics_exposed_before_correction") is not True:
        raise ValueError("initial metric exposure must be disclosed")
    if config.get("winner_exposed_before_correction") != {
        "id": "STACK_W0625",
        "candidate_weight": 0.625,
        "outer_primary_rmse_c": 1.042512377552349,
    }:
        raise ValueError("exposed winner contract changed")
    if any(int(value) != 0 for value in config["behavior_changes"].values()):
        raise ValueError("serialization correction cannot change behavior")
    if config.get("fit_accounting") != {
        "initial_underlying_lightgbm_estimator_fits": 14,
        "correction_model_refits": 0,
        "total_underlying_lightgbm_estimator_fits": 14,
    }:
        raise ValueError("fit accounting changed")
    if len(config.get("affected_binary_pins", {})) != 10:
        raise ValueError("exactly ten affected binaries must be pinned")
    output = config["correction_output"]
    if output.get("append_only") is not True:
        raise ValueError("correction must remain append-only")
    if output.get("original_files_may_be_overwritten") is not False:
        raise ValueError("initial artifacts must remain immutable")
    if output.get("resume_lock") != (
        "artifacts/p2_conservative_stack_improvement_v1_control/resume.lock"
    ):
        raise ValueError("resume lock changed")


def _canonical_preflight(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    if config_path.resolve() != RESUME_CONFIG.resolve():
        raise ValueError("only the canonical resume config is accepted")
    if _sha256(RESUME_CONFIG) != CANONICAL_RESUME_CONFIG_SHA256:
        raise ValueError("canonical resume config SHA-256 changed")
    canonical = _load_json(RESUME_CONFIG)
    _validate_resume_config(canonical)
    if dict(config) != canonical:
        raise ValueError("passed config differs from canonical resume config")
    original_pin = canonical["initial_pins"]["canonical_config"]
    if _sha256(ORIGINAL_CONFIG) != original_pin["sha256"]:
        raise ValueError("original canonical config SHA-256 changed")
    original = _load_json(ORIGINAL_CONFIG)
    if original["stack"]["branches"] != EXPECTED_BRANCHES:
        raise ValueError("initial branches changed")
    return canonical


def _path_from_logical(value: str) -> Path:
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("resume path escapes repository root") from error
    return candidate


def _verify_initial_pins(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for role, pin in config["initial_pins"].items():
        path = _path_from_logical(pin["path"])
        actual = _sha256(path)
        if actual != pin["sha256"] or path.stat().st_size != int(pin["bytes"]):
            raise ValueError(f"initial {role} pin mismatch")
        records[role] = {"path": _logical(path), "sha256": actual, "bytes": path.stat().st_size}
    for relative, expected in config["affected_binary_pins"].items():
        path = (OUTPUT / relative).resolve()
        try:
            path.relative_to(OUTPUT.resolve())
        except ValueError as error:
            raise ValueError("affected binary path escapes output") from error
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"affected binary pin mismatch: {relative}")
        records[f"affected:{relative}"] = {
            "path": _logical(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return records


def _repaired_path(config: Mapping[str, Any], relative: str) -> Path:
    root = _path_from_logical(config["correction_output"]["repaired_root"])
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("repaired target escapes repaired root") from error
    return path


def _planned_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    output = config["correction_output"]
    paths = {
        f"repaired:{relative}": _repaired_path(config, relative)
        for relative in config["affected_binary_pins"]
    }
    paths.update(
        {
            "reproduction": _path_from_logical(output["reproduction_path"]),
            "failure_receipt": _path_from_logical(output["failure_receipt"]),
            "resume_completion": _path_from_logical(output["resume_completion"]),
            "result": _path_from_logical(output["result"]),
            "manifest": _path_from_logical(output["manifest"]),
            "seal": _path_from_logical(output["seal"]),
            "resume_lock": _path_from_logical(output["resume_lock"]),
            "progress_start": OUTPUT / "progress/012_resume_serialization_correction.json",
            "progress_repair": OUTPUT / "progress/013_repaired_binary_loads.json",
            "progress_oof": OUTPUT / "progress/014_saved_model_oof_reinference.json",
            "progress_candidate": OUTPUT / "progress/015_candidate_byte_match.json",
            "progress_complete": OUTPUT / "progress/016_complete.json",
        }
    )
    if len(set(path.resolve() for path in paths.values())) != len(paths):
        raise ValueError("resume write paths collide")
    return paths


def _prewrite_guard(config: Mapping[str, Any]) -> dict[str, Path]:
    if not OUTPUT.is_dir():
        raise FileNotFoundError("initial output directory is missing")
    planned = _planned_paths(config)
    for role, path in planned.items():
        if path.exists():
            raise FileExistsError(f"resume target already exists: {role}")
        if role.startswith("repaired:"):
            path.relative_to(_path_from_logical(config["correction_output"]["repaired_root"]))
        elif role != "resume_lock":
            path.relative_to(OUTPUT.resolve())
    return planned


def _acquire_resume_lock(config: Mapping[str, Any], path: Path) -> dict[str, Any]:
    payload = {
        "experiment_id": config["experiment_id"],
        "same_original_attempt": 1,
        "new_generation_or_attempt": False,
        "canonical_resume_config_sha256": CANONICAL_RESUME_CONFIG_SHA256,
        "status": "RESUME_CONSUMED",
        "rerun_allowed": False,
        "created_at_kst": _now(),
    }
    _exclusive_json(path, payload)
    return {"path": _logical(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


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


def _progress(path: Path, progress: float, phase: str, detail: str) -> None:
    _exclusive_json(
        path,
        {
            "experiment_id": "p2_conservative_stack_improvement_v1",
            "same_original_attempt": 1,
            "new_generation_or_attempt": False,
            "progress": float(progress),
            "phase": phase,
            "detail": detail,
            "updated_at_kst": _now(),
        },
    )


def _repair_in_memory(config: Mapping[str, Any]) -> dict[str, bytes]:
    repaired: dict[str, bytes] = {}
    for relative in config["affected_binary_pins"]:
        repaired[relative] = repair_windows_text_expansion((OUTPUT / relative).read_bytes())
    return repaired


def _validate_repaired_payloads(repaired: Mapping[str, bytes]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for relative, payload in repaired.items():
        if relative.endswith(".joblib"):
            import io

            value = joblib.load(io.BytesIO(payload))
            kind = type(value).__name__
        else:
            import io

            value = pd.read_parquet(io.BytesIO(payload))
            kind = f"DataFrame[{len(value)}x{len(value.columns)}]"
        records[relative] = {
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
            "load": "PASS",
            "kind": kind,
        }
    return records


def _write_repaired(
    config: Mapping[str, Any], repaired: Mapping[str, bytes]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative, payload in repaired.items():
        source = OUTPUT / relative
        target = _repaired_path(config, relative)
        _exclusive_binary(target, payload)
        if target.read_bytes() != payload:
            raise AssertionError(f"binary writer changed repaired payload: {relative}")
        records[relative] = {
            "initial_path": _logical(source),
            "initial_sha256": _sha256(source),
            "initial_bytes": source.stat().st_size,
            "repaired_path": _logical(target),
            "repaired_sha256": _sha256(target),
            "repaired_bytes": target.stat().st_size,
            "removed_text_mode_cr_bytes": source.stat().st_size - target.stat().st_size,
            "initial_file_immutable": True,
        }
    return records


def _assert_float_equal(actual: float, expected: float, *, role: str, atol: float = 1e-12) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=atol):
        raise AssertionError(f"{role} differs: {actual} != {expected}")


def _recompute_branch_metrics(
    initial_runner,
    original_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    inner_oof: pd.DataFrame,
    outer_oof: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layer_counts = original_config["validation"]["official_layer_counts"]
    outer_baseline = metric_report(
        outer_oof, prediction_column="baseline", official_layer_counts=layer_counts
    )
    inner_baseline = metric_report(
        inner_oof, prediction_column="baseline", official_layer_counts=layer_counts
    )
    predecessor_outer = metrics["predecessor"]["outer_candidate"]
    predecessor_inner = metrics["predecessor"]["inner_candidate"]
    recomputed: list[dict[str, Any]] = []
    for expected in EXPECTED_BRANCHES:
        branch_id = expected["id"]
        column = f"prediction_{branch_id}"
        outer_report = metric_report(
            outer_oof, prediction_column=column, official_layer_counts=layer_counts
        )
        inner_report = metric_report(
            inner_oof, prediction_column=column, official_layer_counts=layer_counts
        )
        guard = initial_runner.branch_guard_report(
            branch_id=branch_id,
            outer_report=outer_report,
            inner_report=inner_report,
            outer_baseline_rmse=float(outer_baseline["fold_equal_official_layer_weighted_rmse_c"]),
            inner_baseline_rmse=float(inner_baseline["fold_equal_official_layer_weighted_rmse_c"]),
            predecessor_outer=predecessor_outer,
            predecessor_inner_rmse=float(
                predecessor_inner["fold_equal_official_layer_weighted_rmse_c"]
            ),
            guards=original_config["winner_guards"],
        )
        recorded = next(item for item in metrics["branches"] if item["id"] == branch_id)
        _assert_float_equal(
            outer_report["fold_equal_official_layer_weighted_rmse_c"],
            recorded["outer"]["fold_equal_official_layer_weighted_rmse_c"],
            role=f"{branch_id} outer primary",
        )
        _assert_float_equal(
            inner_report["fold_equal_official_layer_weighted_rmse_c"],
            recorded["inner"]["fold_equal_official_layer_weighted_rmse_c"],
            role=f"{branch_id} inner primary",
        )
        if guard != recorded["guard"]:
            raise AssertionError(f"{branch_id} recomputed guard differs")
        for fold, values in outer_report["by_fold"].items():
            _assert_float_equal(
                values["official_layer_weighted_rmse_c"],
                recorded["outer"]["by_fold"][fold]["official_layer_weighted_rmse_c"],
                role=f"{branch_id} {fold}",
            )
        recomputed.append(
            {
                "id": branch_id,
                "candidate_weight": expected["candidate_weight"],
                "inner": inner_report,
                "outer": outer_report,
                "guard": guard,
            }
        )
    winner = initial_runner.select_winner(recomputed)
    if winner["id"] != "STACK_W0625" or winner["candidate_weight"] != 0.625:
        raise AssertionError("recomputed winner changed")
    bootstrap = original_config["validation"]["bootstrap"]
    winner_column = f"prediction_{winner['id']}"
    winner_vs_predecessor = paired_fold_day_bootstrap(
        outer_oof,
        reference_column="incumbent_prediction",
        candidate_column=winner_column,
        official_layer_counts=layer_counts,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]) + 50,
        interval=float(bootstrap["interval"]),
    )
    recorded_bootstrap = metrics["winner"]["winner_vs_predecessor_bootstrap"]
    for key in ("reference_rmse_c", "candidate_rmse_c", "delta_rmse_c"):
        _assert_float_equal(
            winner_vs_predecessor[key], recorded_bootstrap[key], role=f"bootstrap {key}"
        )
    for index in (0, 1):
        _assert_float_equal(
            winner_vs_predecessor["delta_interval"][index],
            recorded_bootstrap["delta_interval"][index],
            role=f"bootstrap interval {index}",
        )
    return recomputed, winner


def _verify_saved_oof_models(
    initial_runner,
    original_config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    config: Mapping[str, Any],
    inner_oof: pd.DataFrame,
    outer_oof: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    folds = {fold["name"]: fold for fold in original_config["validation"]["folds"]}
    records: dict[str, dict[str, Any]] = {}
    for role in MODEL_ROLES:
        stage = "inner" if role.endswith("_inner") else "outer"
        fold_name = role[: -len(f"_{stage}")]
        model_path = _repaired_path(config, f"models/{role}.joblib")
        model = joblib.load(model_path)
        window = folds[fold_name][stage]
        prediction = predict_scored_window(
            model,
            runtime["base"],
            runtime["lean"],
            runtime["endpoints"],
            start=window[0],
            stop=window[1],
            fold=fold_name,
            stage=stage,
        ).rename(columns={"prediction": "incumbent_prediction"})
        expected_source = inner_oof if stage == "inner" else outer_oof
        expected = expected_source.loc[expected_source["fold"].eq(fold_name)].reset_index(drop=True)
        if not prediction[[*KEYS, "fold", "stage"]].equals(expected[[*KEYS, "fold", "stage"]]):
            raise AssertionError(f"saved {role} reinference keys differ")
        maximum_error = float(
            np.max(
                np.abs(
                    prediction["incumbent_prediction"].to_numpy(float)
                    - expected["incumbent_prediction"].to_numpy(float)
                )
            )
        )
        if maximum_error > 1e-12:
            raise AssertionError(f"saved {role} reinference drift: {maximum_error}")
        records[role] = {
            "path": _logical(model_path),
            "sha256": _sha256(model_path),
            "rows": int(len(prediction)),
            "maximum_absolute_prediction_error_c": maximum_error,
            "model_refits_during_correction": 0,
        }
    return records


def _candidate_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, encoding="utf-8", lineterminator="\r\n").encode("utf-8")


def _dry_run(config: Mapping[str, Any], config_path: Path, data_dir: Path) -> int:
    config = _canonical_preflight(config, config_path)
    _prewrite_guard(config)
    initial_records = _verify_initial_pins(config)
    initial_runner = _load_original_runner(config)
    original_config = _load_json(ORIGINAL_CONFIG)
    initial_runner._verify_sources(original_config, data_dir)
    repaired = _repair_in_memory(config)
    load_records = _validate_repaired_payloads(repaired)
    metrics = _load_json(OUTPUT / "metrics.json")
    if metrics["winner"]["id"] != "STACK_W0625":
        raise ValueError("initial exposed winner changed")
    candidate_validation = validate_submission(
        OUTPUT / original_config["output_contract"]["candidate_relative_path"],
        initial_runner.load_p2_data(data_dir).test_index,
    )
    print(
        json.dumps(
            {
                "dry_run": "PASS",
                "experiment_id": config["experiment_id"],
                "same_original_attempt": 1,
                "new_generation_or_attempt": False,
                "initial_pins_verified": len(initial_records),
                "affected_binaries_repaired_in_memory": len(repaired),
                "repaired_load_checks": load_records,
                "exposed_winner": config["winner_exposed_before_correction"],
                "correction_model_refits": 0,
                "candidate_validation": candidate_validation,
                "writes": 0,
                "resume_lock_created": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run(config: Mapping[str, Any], config_path: Path, data_dir: Path) -> int:
    config = _canonical_preflight(config, config_path)
    planned = _prewrite_guard(config)
    initial_before = _verify_initial_pins(config)
    initial_runner = _load_original_runner(config)
    original_config = _load_json(ORIGINAL_CONFIG)
    source_before = initial_runner._verify_sources(original_config, data_dir)
    predecessor_before = initial_runner._verify_predecessor(original_config)
    frozen_before = initial_runner._frozen_snapshot()
    started = time.perf_counter()
    git_before = _git_state()
    resume_lock = _acquire_resume_lock(config, planned["resume_lock"])
    _progress(
        planned["progress_start"],
        89,
        "resume_serialization_correction",
        "same attempt; exposed winner unchanged; zero refits",
    )
    failure_receipt = {
        "schema_version": "p2_conservative_stack_improvement.serialization_failure.v1",
        "experiment_id": config["experiment_id"],
        "same_original_attempt": 1,
        "new_generation_or_attempt": False,
        "failure_phase": config["failure"]["phase"],
        "exception_type": config["failure"]["exception_type"],
        "root_cause": config["failure"]["root_cause"],
        "metrics_exposed_before_failure": True,
        "winner_exposed_before_failure": config["winner_exposed_before_correction"],
        "initial_underlying_lightgbm_estimator_fits": 14,
        "candidate_created_before_failure": True,
        "result_manifest_seal_created_before_failure": False,
        "correction_model_refits": 0,
        "branch_weight_parameter_feature_fold_threshold_postprocess_changes": 0,
        "initial_artifacts_preserved": True,
        "recorded_at_kst": _now(),
    }
    _exclusive_json(planned["failure_receipt"], failure_receipt)

    repaired = _repair_in_memory(config)
    in_memory_loads = _validate_repaired_payloads(repaired)
    repair_records = _write_repaired(config, repaired)
    for relative, record in in_memory_loads.items():
        repair_records[relative]["load_check"] = record
    _progress(
        planned["progress_repair"],
        92,
        "repaired_binary_loads",
        "8 joblib and 2 parquet payloads load from append-only repaired paths",
    )

    runtime = initial_runner._build_runtime(data_dir)
    inner_oof = pd.read_parquet(_repaired_path(config, "oof/inner_stack.parquet"))
    outer_oof = pd.read_parquet(_repaired_path(config, "oof/outer_stack.parquet"))
    metrics = _load_json(OUTPUT / "metrics.json")
    recomputed_branches, winner = _recompute_branch_metrics(
        initial_runner, original_config, metrics, inner_oof, outer_oof
    )
    saved_oof_model_checks = _verify_saved_oof_models(
        initial_runner,
        original_config,
        runtime,
        config,
        inner_oof,
        outer_oof,
    )
    _progress(
        planned["progress_oof"],
        96,
        "saved_model_oof_reinference",
        "all six repaired fold models reproduce sealed OOF with zero refits",
    )

    data: P2Data = runtime["data"]
    repaired_stack_path = _repaired_path(config, "models/final_stack_model.joblib")
    repaired_underlying_path = _repaired_path(config, "models/final_underlying_blend.joblib")
    stack_model = joblib.load(repaired_stack_path)
    standalone_model = joblib.load(repaired_underlying_path)
    if stack_model["winner_id"] != winner["id"] or stack_model["candidate_weight"] != 0.625:
        raise AssertionError("repaired stack model winner or weight changed")
    if stack_model["base_feature_columns"] != list(runtime["base"].feature_columns):
        raise AssertionError("repaired stack base feature schema changed")
    if stack_model["lean_feature_columns"] != list(runtime["lean"].feature_columns):
        raise AssertionError("repaired stack lean feature schema changed")
    hidden_start, hidden_stop = original_config["masking"]["hidden_window_kst_half_open"]
    embedded_underlying, embedded_diagnostics = initial_runner._predict_official_underlying(
        stack_model["underlying_model"],
        data,
        runtime["masked"],
        runtime["base"],
        runtime["lean"],
        runtime["endpoints"],
        hidden_start=hidden_start,
        hidden_stop=hidden_stop,
    )
    standalone_underlying, _ = initial_runner._predict_official_underlying(
        standalone_model,
        data,
        runtime["masked"],
        runtime["base"],
        runtime["lean"],
        runtime["endpoints"],
        hidden_start=hidden_start,
        hidden_stop=hidden_stop,
    )
    underlying_error = float(
        np.max(
            np.abs(
                embedded_underlying["temp"].to_numpy(float)
                - standalone_underlying["temp"].to_numpy(float)
            )
        )
    )
    if underlying_error > 1e-12:
        raise AssertionError("standalone and embedded full models differ")
    reproduced = initial_runner._official_stack_submission(
        data, embedded_underlying, float(stack_model["candidate_weight"])
    )
    reproduction_bytes = _candidate_bytes(reproduced)
    candidate_path = _path_from_logical(config["initial_pins"]["candidate"]["path"])
    candidate_bytes = candidate_path.read_bytes()
    if reproduction_bytes != candidate_bytes:
        raise AssertionError("repaired saved stack model did not reproduce candidate bytes")
    _exclusive_binary(planned["reproduction"], reproduction_bytes)
    if planned["reproduction"].read_bytes() != candidate_bytes:
        raise AssertionError("stored reproduction differs from candidate bytes")
    candidate_validation = validate_submission(candidate_path, data.test_index)
    reproduction_validation = validate_submission(planned["reproduction"], data.test_index)
    if candidate_validation != reproduction_validation:
        raise AssertionError("candidate and reproduction validation differ")
    _progress(
        planned["progress_candidate"],
        99,
        "candidate_byte_match",
        "repaired saved stack model reproduces the 26,061-row candidate byte-identically",
    )

    initial_after = _verify_initial_pins(config)
    source_after = initial_runner._verify_sources(original_config, data_dir)
    predecessor_after = initial_runner._verify_predecessor(original_config)
    frozen_after = initial_runner._frozen_snapshot()
    if initial_before != initial_after:
        raise AssertionError("an initial failure artifact changed during correction")
    if source_before != source_after:
        raise AssertionError("a source artifact changed during correction")
    if predecessor_before != predecessor_after:
        raise AssertionError("a predecessor artifact changed during correction")
    if frozen_before != frozen_after:
        raise AssertionError("a frozen/current P2 submission changed during correction")

    result = {
        "schema_version": "p2_conservative_stack_improvement.result.v1_resumed_same_attempt",
        "experiment_id": config["experiment_id"],
        "status": "WINNER_FULL_MODEL_AND_CANDIDATE_COMPLETE",
        "completed_at_kst": _now(),
        "same_original_attempt": 1,
        "new_generation_or_attempt": False,
        "initial_serialization_failure_disclosed": True,
        "metrics_exposed_before_correction": True,
        "winner_id": winner["id"],
        "winner_candidate_weight": winner["candidate_weight"],
        "winner_outer_primary_rmse_c": winner["guard"]["outer_primary_rmse_c"],
        "winner_outer_delta_vs_predecessor_c": winner["guard"]["outer_delta_vs_predecessor_c"],
        "winner_outer_fold_improvement_count": winner["guard"]["outer_fold_improvement_count"],
        "winner_bootstrap": metrics["winner"]["winner_vs_predecessor_bootstrap"],
        "initial_underlying_lightgbm_estimator_fits": 14,
        "correction_model_refits": 0,
        "total_underlying_lightgbm_estimator_fits": 14,
        "saved_fold_model_reinference_checks": 6,
        "final_saved_model_inference_checks": 2,
        "hidden_target_temperature_values_accessed": 0,
        "hidden_target_salinity_values_accessed": 0,
        "candidate": {
            "path": _logical(candidate_path),
            "sha256": _sha256(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "validation": candidate_validation,
            "reproduction_path": _logical(planned["reproduction"]),
            "reproduction_sha256": _sha256(planned["reproduction"]),
            "byte_identical_saved_model_reproduction": True,
        },
        "final_stack_model": {
            "path": _logical(repaired_stack_path),
            "sha256": _sha256(repaired_stack_path),
            "bytes": repaired_stack_path.stat().st_size,
            "load_check": "PASS",
            "feature_schema_check": "PASS",
        },
        "current_frozen_submission_modified": False,
        "predecessor_artifacts_modified": False,
        "source_artifacts_modified": False,
        "promotion_or_upload_authorized": False,
        "upload_performed": False,
        "elapsed_resume_seconds": float(time.perf_counter() - started),
    }
    _exclusive_json(planned["result"], result)

    manifest = {
        "schema_version": "p2_conservative_stack_improvement.manifest.v1_resumed_same_attempt",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _now(),
        "same_original_attempt": 1,
        "new_generation_or_attempt": False,
        "research_only": True,
        "upload_allowed": False,
        "adaptive_research": True,
        "fresh_holdout_claimed": False,
        "interruption_and_correction": {
            "initial_failure_receipt": {
                "path": _logical(planned["failure_receipt"]),
                "sha256": _sha256(planned["failure_receipt"]),
            },
            "metrics_exposed_before_correction": True,
            "winner_exposed_before_correction": config["winner_exposed_before_correction"],
            "root_cause": config["failure"]["root_cause"],
            "correction": config["failure"]["correction"],
            "behavior_changes": config["behavior_changes"],
            "initial_underlying_lightgbm_estimator_fits": 14,
            "correction_model_refits": 0,
            "total_underlying_lightgbm_estimator_fits": 14,
        },
        "configs": {
            "initial": {
                "path": _logical(ORIGINAL_CONFIG),
                "sha256": _sha256(ORIGINAL_CONFIG),
                "bytes": ORIGINAL_CONFIG.stat().st_size,
            },
            "resume": {
                "path": _logical(RESUME_CONFIG),
                "sha256": _sha256(RESUME_CONFIG),
                "bytes": RESUME_CONFIG.stat().st_size,
            },
        },
        "implementation": {
            _logical(ORIGINAL_RUNNER): {
                "sha256": _sha256(ORIGINAL_RUNNER),
                "bytes": ORIGINAL_RUNNER.stat().st_size,
                "role": "initial_fit_runner_preserved",
            },
            _logical(Path(__file__).resolve()): {
                "sha256": _sha256(Path(__file__).resolve()),
                "bytes": Path(__file__).resolve().stat().st_size,
                "role": "same_attempt_serialization_correction",
            },
        },
        "initial_pins": initial_before,
        "source_pins": source_before,
        "predecessor_pins": predecessor_before,
        "resume_lock": resume_lock,
        "repair_records": repair_records,
        "saved_oof_model_checks": saved_oof_model_checks,
        "recomputed_branches": recomputed_branches,
        "winner": {
            "id": winner["id"],
            "candidate_weight": winner["candidate_weight"],
            "guard": winner["guard"],
            "bootstrap": metrics["winner"]["winner_vs_predecessor_bootstrap"],
        },
        "final_saved_model_checks": {
            "stack_model_path": _logical(repaired_stack_path),
            "stack_model_sha256": _sha256(repaired_stack_path),
            "standalone_model_path": _logical(repaired_underlying_path),
            "standalone_model_sha256": _sha256(repaired_underlying_path),
            "embedded_vs_standalone_max_abs_error_c": underlying_error,
            "candidate_byte_identical": True,
            "inference_diagnostics": embedded_diagnostics,
        },
        "candidate": result["candidate"],
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
        "initial_artifacts_unchanged": True,
        "source_artifacts_unchanged": True,
        "predecessor_artifacts_unchanged": True,
        "frozen_submission_snapshot_count": len(frozen_before),
        "frozen_submission_snapshot_unchanged": True,
        "hidden_target_temperature_values_accessed": 0,
        "hidden_target_salinity_values_accessed": 0,
        "upload_performed": False,
        "result": {
            "path": _logical(planned["result"]),
            "sha256": _sha256(planned["result"]),
            "bytes": planned["result"].stat().st_size,
        },
    }
    _exclusive_json(planned["manifest"], manifest)
    seal = {
        "schema_version": "p2_conservative_stack_improvement.seal.v1_resumed_same_attempt",
        "experiment_id": config["experiment_id"],
        "same_original_attempt": 1,
        "new_generation_or_attempt": False,
        "winner_id": winner["id"],
        "winner_candidate_weight": winner["candidate_weight"],
        "winner_outer_primary_rmse_c": winner["guard"]["outer_primary_rmse_c"],
        "manifest_path": _logical(planned["manifest"]),
        "manifest_sha256": _sha256(planned["manifest"]),
        "result_sha256": _sha256(planned["result"]),
        "candidate_path": _logical(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "reproduction_sha256": _sha256(planned["reproduction"]),
        "candidate_byte_identical_saved_model_reproduction": True,
        "final_stack_model_path": _logical(repaired_stack_path),
        "final_stack_model_sha256": _sha256(repaired_stack_path),
        "metrics_sha256": _sha256(OUTPUT / "metrics.json"),
        "repaired_outer_oof_sha256": _sha256(_repaired_path(config, "oof/outer_stack.parquet")),
        "initial_underlying_lightgbm_estimator_fits": 14,
        "correction_model_refits": 0,
        "initial_artifacts_unchanged": True,
        "frozen_submission_snapshot_unchanged": True,
        "sealed_at_kst": _now(),
        "upload_performed": False,
    }
    _exclusive_json(planned["seal"], seal)
    completion = {
        "schema_version": "p2_conservative_stack_improvement.resume_completion_status.v1",
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "same_original_attempt": 1,
        "new_generation_or_attempt": False,
        "completed_at_kst": _now(),
        "failure_receipt": {
            "path": _logical(planned["failure_receipt"]),
            "sha256": _sha256(planned["failure_receipt"]),
        },
        "manifest": {
            "path": _logical(planned["manifest"]),
            "sha256": _sha256(planned["manifest"]),
        },
        "seal": {"path": _logical(planned["seal"]), "sha256": _sha256(planned["seal"])},
        "candidate": {
            "path": _logical(candidate_path),
            "sha256": _sha256(candidate_path),
            "reproduction_sha256": _sha256(planned["reproduction"]),
            "byte_identical": True,
        },
        "final_stack_model": {
            "path": _logical(repaired_stack_path),
            "sha256": _sha256(repaired_stack_path),
            "load_check": "PASS",
        },
        "initial_underlying_lightgbm_estimator_fits": 14,
        "correction_model_refits": 0,
        "upload_performed": False,
    }
    _exclusive_json(planned["resume_completion"], completion)
    _progress(
        planned["progress_complete"],
        100,
        "complete",
        "repaired saved-model reload, OOF reinference, and candidate byte match sealed",
    )
    print(
        json.dumps(
            {"result": result, "seal": seal, "completion": completion}, ensure_ascii=False, indent=2
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path, default=RESUME_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.resolve()
    config = _load_json(config_path)
    _validate_resume_config(config)
    data_dir = resolve_data_dir(args.data_dir)
    if args.dry_run:
        return _dry_run(config, config_path, data_dir)
    return _run(config, config_path, data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
