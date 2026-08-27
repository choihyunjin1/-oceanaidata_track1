"""Deterministically resume the interrupted P2 v2 one-shot attempt.

This is not a new experiment, generation, or attempt.  It binds the original
v2 code/config/attempt lock and three completed blend models, loads those models
without refitting or overwriting them, and can create only artifacts that were
missing when the original process was interrupted.  Actual resume remains
blocked behind an explicit reviewer authorization token.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p2_corrected_repeated_forward_v2.json"
HELPER_PATH = ROOT / "src/p2_restore/corrected_repeated_forward.py"
RUNNER_PATH = ROOT / "scripts/run_p2_corrected_repeated_forward.py"
OUTPUT_DIR = ROOT / "artifacts/p2_corrected_repeated_forward_v2"
CONTROL_DIR = ROOT / "artifacts/p2_corrected_repeated_forward_v2_control"
ATTEMPT_LOCK = CONTROL_DIR / "attempt.lock"
RESUME_LOCK = CONTROL_DIR / "resume.lock"
RESUME_COMPLETION_RECEIPT = OUTPUT_DIR / "resume_completion_status.json"
AUTHORIZATION_TOKEN = "REVIEWER_GO_P2_V2_DETERMINISTIC_RESUME"
KST = ZoneInfo("Asia/Seoul")

PINNED_CORE_SHA256 = {
    "config": "cd0f88fd12fa7900be7c39cd8566aa455dae6ffbf4da4077adc52bd10ced70ca",
    "helper": "796bb06143e517426dff2fb754ed083b1fe5f2084788b7c574ca85895de47402",
    "runner": "5a58f2bbf0b1702912f2ef81802f02d60fe0aacf45d3882178302e4bde724b1d",
    "attempt_lock": "9a3a0067ed7296bf14ab31e26df29501f34bdaf936dda4d09238a8fdeeeffdac",
    "interrupted_status": "e6b9c245a12e421d8c77894105ee61b32da1cdc13b70e4cec8370db52e50e4cb",
}

PINNED_RUNTIME_MODULE_SHA256 = {
    "src/p2_restore/data.py": "e4d3644e3609575b3cea22f027d0f8a7f5a7f0ff403200bd7f0c190f3de679d9",
    "src/p2_restore/features.py": "b23e19ec55120f6144e693f9da24ba78b85b6191c55de1a2889b0d26fd8d8ee7",
    "src/p2_restore/model.py": "5a1b4af4f7092e490741f576b1bf7595ff7e2f38a0d1d3605db9c865b5c20020",
    "src/p2_restore/research.py": "84716e1e8836a1f6ffaee74309aff317b53e6bc3e5bfb264d0df554ca474ca02",
    "src/p2_restore/profile_projection.py": (
        "fb1615ea1b0b67aad8a35daaef416eaff3dcd9d5b9cd498e3631c5b0b88d74e6"
    ),
    "src/p2_restore/submission.py": (
        "ac04940e1643d3ca6d933b39f74be03cfee953464aec5ab76d8329903dc4dcc7"
    ),
}

EXPECTED_LGBM_PARAMETERS = {
    "boosting_type": "gbdt",
    "class_weight": None,
    "colsample_bytree": 0.85,
    "deterministic": True,
    "force_row_wise": True,
    "importance_type": "split",
    "learning_rate": 0.04,
    "max_depth": 7,
    "min_child_samples": 200,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "n_estimators": 400,
    "n_jobs": 8,
    "num_leaves": 31,
    "objective": "regression_l2",
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "subsample": 0.85,
    "subsample_for_bin": 200_000,
    "subsample_freq": 0,
    "verbosity": -1,
}

FIT_ORDER = [
    {
        "role": "outer_2024_sep_oct_inner",
        "fold": "outer_2024_sep_oct",
        "stage": "inner",
        "seed": 20260822,
        "origin": "initial_attempt_load_only",
        "sha256": "165837560bb6ace50e53939f7c73e09728366f506285bc8279625e06f8405b72",
    },
    {
        "role": "outer_2024_sep_oct_outer",
        "fold": "outer_2024_sep_oct",
        "stage": "outer",
        "seed": 20260823,
        "origin": "initial_attempt_load_only",
        "sha256": "104c57707c9ed5c79243d34c797141c99311214e66179cdc35dbf0a6954de41e",
    },
    {
        "role": "outer_2025_may_jun_inner",
        "fold": "outer_2025_may_jun",
        "stage": "inner",
        "seed": 20260832,
        "origin": "initial_attempt_load_only",
        "sha256": "5c0fbbd8b0db5600235b2bbb43296f550c3036ef31e0cb942c5284b4e2b89691",
    },
    {
        "role": "outer_2025_may_jun_outer",
        "fold": "outer_2025_may_jun",
        "stage": "outer",
        "seed": 20260833,
        "origin": "resume_fit_missing_only",
        "sha256": None,
    },
    {
        "role": "outer_2025_jul_aug_inner",
        "fold": "outer_2025_jul_aug",
        "stage": "inner",
        "seed": 20260842,
        "origin": "resume_fit_missing_only",
        "sha256": None,
    },
    {
        "role": "outer_2025_jul_aug_outer",
        "fold": "outer_2025_jul_aug",
        "stage": "outer",
        "seed": 20260843,
        "origin": "resume_fit_missing_only",
        "sha256": None,
    },
    {
        "role": "final_full_train",
        "fold": None,
        "stage": "final",
        "seed": 20270822,
        "origin": "resume_fit_missing_only",
        "sha256": None,
    },
]

PARTIAL_FILES = {
    "status.json": PINNED_CORE_SHA256["interrupted_status"],
    "models/outer_2024_sep_oct_inner.joblib": FIT_ORDER[0]["sha256"],
    "models/outer_2024_sep_oct_outer.joblib": FIT_ORDER[1]["sha256"],
    "models/outer_2025_may_jun_inner.joblib": FIT_ORDER[2]["sha256"],
}

INTERRUPTION_RESUME_DISCLOSURE = {
    "same_attempt": 1,
    "new_generation_or_attempt": False,
    "initial_completed_estimators": 6,
    "resume_completed_estimators": 8,
    "initial_in_memory_predictions_discarded": True,
    "initial_predictions_used_for_selection": False,
    "parameter_threshold_or_weight_changes": 0,
    "literal_inference_once_claimed": False,
    "fold1_outer": {
        "initial_ephemeral_unexposed_inference_invocations": 1,
        "resume_saved_model_materialization_inference_invocations": 1,
        "total_inference_invocations": 2,
        "blend_model_fits_initial_attempt": 1,
        "blend_model_refits_during_resume": 0,
        "outer_metric_exposures": 1,
        "metric_exposure_timing": "final resumed aggregate only",
        "persisted_oof_before_resume": False,
        "resume_role": "missing persisted materialization from the pinned saved model",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(KST).isoformat()


def _logical(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _verify_hash(path: Path, expected: str, *, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {role}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"pinned {role} SHA-256 changed")


def _load_original_runner() -> Any:
    spec = importlib.util.spec_from_file_location("p2_corrected_v2_original_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the pinned original v2 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_runtime_module_sources() -> None:
    """Pin every local module imported transitively by the original runner."""

    for relative, expected in PINNED_RUNTIME_MODULE_SHA256.items():
        _verify_hash(ROOT / relative, expected, role=f"runtime module {relative}")


def _verify_core_and_attempt() -> tuple[Any, dict[str, Any]]:
    _verify_hash(CONFIG_PATH, PINNED_CORE_SHA256["config"], role="v2 config")
    _verify_hash(HELPER_PATH, PINNED_CORE_SHA256["helper"], role="v2 helper")
    _verify_hash(RUNNER_PATH, PINNED_CORE_SHA256["runner"], role="v2 original runner")
    _verify_hash(ATTEMPT_LOCK, PINNED_CORE_SHA256["attempt_lock"], role="v2 attempt lock")
    _verify_runtime_module_sources()
    attempt = json.loads(ATTEMPT_LOCK.read_text(encoding="utf-8"))
    expected_attempt = {
        "experiment_id": "p2_corrected_repeated_forward_v2",
        "attempt": 1,
        "canonical_config_sha256": PINNED_CORE_SHA256["config"],
        "created_at_kst": "2026-08-22T19:11:59.860420+09:00",
        "status": "ATTEMPT_CONSUMED",
        "rerun_allowed": False,
    }
    if attempt != expected_attempt:
        raise ValueError("v2 attempt-lock semantics changed")
    runner = _load_original_runner()
    config = runner._load_config(CONFIG_PATH)
    canonical = runner._canonical_preflight(config, CONFIG_PATH, OUTPUT_DIR)
    return runner, canonical


def _verify_partial_inventory() -> None:
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError("interrupted v2 output directory is absent")
    actual = {
        path.relative_to(OUTPUT_DIR).as_posix(): path
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file()
    }
    if set(actual) != set(PARTIAL_FILES):
        extra = sorted(set(actual).difference(PARTIAL_FILES))
        missing = sorted(set(PARTIAL_FILES).difference(actual))
        raise ValueError(f"partial v2 inventory changed; extra={extra}, missing={missing}")
    for relative, expected in PARTIAL_FILES.items():
        _verify_hash(actual[relative], str(expected), role=f"partial artifact {relative}")
    if RESUME_LOCK.exists():
        raise FileExistsError("v2 deterministic resume lock already exists")


def _model_path(role: str) -> Path:
    return OUTPUT_DIR / "models" / f"{role}.joblib"


def _validate_model(model: object, base: object, lean: object, *, seed: int, role: str) -> None:
    if float(model.weight) != 0.5:
        raise ValueError(f"{role} blend weight changed")
    if tuple(model.base_model.feature_columns) != tuple(base.feature_columns):
        raise ValueError(f"{role} base feature schema changed")
    if tuple(model.lean_model.feature_columns) != tuple(lean.feature_columns):
        raise ValueError(f"{role} lean feature schema changed")
    for arm in (model.base_model.estimator, model.lean_model.estimator):
        parameters = arm.get_params()
        if type(arm).__name__ != "LGBMRegressor" or type(arm).__module__ != "lightgbm.sklearn":
            raise ValueError(f"{role} estimator class changed")
        expected = {**EXPECTED_LGBM_PARAMETERS, "random_state": seed}
        if parameters != expected:
            changed = {
                key: {"expected": expected.get(key), "actual": parameters.get(key)}
                for key in sorted(set(expected) | set(parameters))
                if expected.get(key) != parameters.get(key)
            }
            raise ValueError(f"{role} full estimator parameters changed: {changed}")
        if int(arm.n_estimators_) != 400:
            raise ValueError(f"{role} estimator tree count changed")


def _verify_existing_models(base: object, lean: object) -> dict[str, object]:
    loaded: dict[str, object] = {}
    for item in FIT_ORDER:
        if item["origin"] != "initial_attempt_load_only":
            continue
        path = _model_path(str(item["role"]))
        _verify_hash(path, str(item["sha256"]), role=str(item["role"]))
        model = joblib.load(path)
        _validate_model(model, base, lean, seed=int(item["seed"]), role=str(item["role"]))
        loaded[str(item["role"])] = model
    return loaded


def _obtain_model(
    item: Mapping[str, Any],
    models: Mapping[str, object],
    runner: Any,
    base: object,
    lean: object,
    selected: np.ndarray,
) -> tuple[object, bool]:
    """Load a completed model or fit exactly one missing role, never both."""

    role = str(item["role"])
    if item["origin"] == "initial_attempt_load_only":
        if role not in models:
            raise KeyError(f"pinned initial model was not loaded: {role}")
        return models[role], False
    path = _model_path(role)
    if path.exists():
        raise FileExistsError(f"missing-only role unexpectedly exists: {role}")
    model = runner.fit_fixed_blend(base, lean, selected, seed=int(item["seed"]))
    _validate_model(model, base, lean, seed=int(item["seed"]), role=role)
    _exclusive_joblib(path, model)
    return model, True


def _exclusive_materialize(path: Path, writer: Callable[[Path], None]) -> None:
    """Create one missing artifact without any overwrite-capable operation."""

    if path.exists():
        raise FileExistsError(f"resume target already exists: {_logical(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".resume.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        writer(temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )

    def write(temporary: Path) -> None:
        temporary.write_bytes(payload)

    _exclusive_materialize(path, write)


def _exclusive_joblib(path: Path, value: object) -> None:
    _exclusive_materialize(path, lambda temporary: joblib.dump(value, temporary))


def _exclusive_parquet(path: Path, frame: pd.DataFrame) -> None:
    _exclusive_materialize(path, lambda temporary: frame.to_parquet(temporary, index=False))


def _exclusive_csv(path: Path, frame: pd.DataFrame) -> None:
    _exclusive_materialize(
        path,
        lambda temporary: frame.to_csv(
            temporary, index=False, encoding="utf-8", lineterminator="\n"
        ),
    )


def _acquire_resume_lock(token: str | None, path: Path = RESUME_LOCK) -> dict[str, Any]:
    if token != AUTHORIZATION_TOKEN:
        raise PermissionError("independent reviewer authorization token is required")
    payload = {
        "experiment_id": "p2_corrected_repeated_forward_v2",
        "original_attempt": 1,
        "role": "CONCURRENCY_GUARD_FOR_SAME_ATTEMPT_RESUME_ONLY",
        "new_experiment_or_attempt": False,
        "created_at_kst": _now(),
        "authorization": "INDEPENDENT_REVIEWER_GO",
    }
    _exclusive_json(path, payload)
    return {"path": _logical(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _validate_fold_coverage(
    runner: Any, config: Mapping[str, Any], population: object
) -> dict[str, dict[str, Any]]:
    """Reapply the original v2 pre-fit coverage gate without modification."""

    inner_floor = float(config["validation"]["minimum_inner_target_coverage"])
    outer_floor = float(config["validation"]["minimum_outer_target_coverage"])
    if inner_floor != 0.96 or outer_floor != 0.96:
        raise ValueError("resume coverage floors differ from the pinned v2 contract")
    coverage: dict[str, dict[str, Any]] = {}
    for fold in config["validation"]["folds"]:
        inner = runner._coverage(population, *fold["inner"])
        outer = runner._coverage(population, *fold["outer"])
        if (
            float(inner["target_coverage"]) < inner_floor
            or float(outer["target_coverage"]) < outer_floor
        ):
            raise ValueError(f"fold {fold['name']} fails the fixed coverage floor")
        coverage[str(fold["name"])] = {"inner": inner, "outer": outer}
    return coverage


def _build_runtime(runner: Any, config: Mapping[str, Any], data_dir: Path) -> dict[str, Any]:
    source_records = runner._verify_sources(config, data_dir)
    data = runner.load_p2_data(data_dir)
    masked, mask_audit = runner.joint_mask_target_context(data.observations)
    base = runner.build_joint_masked_population(data.observations, masked)
    lean = runner.build_fixed_lean_arm(base, masked)
    endpoints = runner.public_endpoints_from_masked_context(masked)
    fold_coverage = _validate_fold_coverage(runner, config, base)
    existing = _verify_existing_models(base, lean)
    return {
        "source_records": source_records,
        "data": data,
        "masked": masked,
        "mask_audit": mask_audit,
        "base": base,
        "lean": lean,
        "endpoints": endpoints,
        "fold_coverage": fold_coverage,
        "existing_models": existing,
    }


def _dry_run(data_dir: Path) -> int:
    runner, config = _verify_core_and_attempt()
    _verify_partial_inventory()
    runtime = _build_runtime(runner, config, data_dir)
    planned = runner._planned_write_paths(config, OUTPUT_DIR)
    existing_planned = {"status", *[f"model_{item['role']}" for item in FIT_ORDER[:3]]}
    for role, path in planned.items():
        if role in existing_planned:
            continue
        if path.exists():
            raise FileExistsError(f"missing-stage target unexpectedly exists: {role}")
    print(
        json.dumps(
            {
                "dry_run": "PASS",
                "experiment_id": config["experiment_id"],
                "same_original_attempt": 1,
                "new_generation_or_attempt": False,
                "core_sha256": PINNED_CORE_SHA256,
                "partial_inventory": PARTIAL_FILES,
                "completed_blend_models_load_only": list(runtime["existing_models"]),
                "missing_blend_models_to_fit_after_reviewer_go": [
                    item["role"] for item in FIT_ORDER if item["origin"] == "resume_fit_missing_only"
                ],
                "fit_order": FIT_ORDER,
                "joint_mask": runtime["mask_audit"].__dict__,
                "fold_coverage": runtime["fold_coverage"],
                "model_fits": 0,
                "prediction_or_metric_writes": 0,
                "resume_lock_created": False,
                "actual_resume_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _resume(data_dir: Path, authorization_token: str | None) -> int:
    runner, config = _verify_core_and_attempt()
    _verify_partial_inventory()
    resume_lock = _acquire_resume_lock(authorization_token)
    started = time.perf_counter()
    frozen_before = runner._frozen_snapshot()
    git_before = runner._git_state()
    runtime = _build_runtime(runner, config, data_dir)
    data = runtime["data"]
    masked = runtime["masked"]
    mask_audit = runtime["mask_audit"]
    base = runtime["base"]
    lean = runtime["lean"]
    endpoints = runtime["endpoints"]
    fold_coverage = runtime["fold_coverage"]
    models = dict(runtime["existing_models"])
    folds_by_name = {fold["name"]: fold for fold in config["validation"]["folds"]}
    embargo_days = int(config["validation"]["embargo_days"])
    inner_frames: list[pd.DataFrame] = []
    outer_frames: list[pd.DataFrame] = []
    model_records: dict[str, Any] = {}
    fold_records: dict[str, Any] = {}

    for item in FIT_ORDER[:-1]:
        role = str(item["role"])
        fold = folds_by_name[str(item["fold"])]
        stage = str(item["stage"])
        window = fold[stage]
        selected, cutoff = runner.forward_training_mask(
            base.frame, window[0], embargo_days=embargo_days
        )
        model, fitted_during_resume = _obtain_model(
            item, models, runner, base, lean, selected
        )
        if fitted_during_resume:
            models[role] = model
        prediction = runner.predict_scored_window(
            model,
            base,
            lean,
            endpoints,
            start=window[0],
            stop=window[1],
            fold=str(item["fold"]),
            stage=stage,
        )
        (inner_frames if stage == "inner" else outer_frames).append(prediction)
        path = _model_path(role)
        model_records[role] = {
            "path": _logical(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "seed": int(item["seed"]),
            "origin": item["origin"],
            "refit_during_resume": fitted_during_resume,
        }
        record = fold_records.setdefault(
            str(item["fold"]),
            {
                "inner_window_kst": fold["inner"],
                "outer_window_kst": fold["outer"],
                "same_season_priority": bool(fold["same_season_priority"]),
                "inner_coverage": fold_coverage[str(item["fold"])]["inner"],
                "outer_coverage": fold_coverage[str(item["fold"])]["outer"],
            },
        )
        record[f"{stage}_training"] = runner._training_summary(base.frame, selected, cutoff)

    inner_oof = pd.concat(inner_frames, ignore_index=True)
    outer_oof = pd.concat(outer_frames, ignore_index=True)
    if outer_oof.duplicated(["station", "layer", "time"]).any():
        raise ValueError("resumed outer OOF keys overlap")
    planned = runner._planned_write_paths(config, OUTPUT_DIR)
    _exclusive_parquet(planned["inner_oof"], inner_oof)
    _exclusive_parquet(planned["outer_oof"], outer_oof)

    layer_counts = config["validation"]["official_layer_counts"]
    bootstrap = config["validation"]["bootstrap"]
    metrics = {
        "experiment_id": config["experiment_id"],
        "research_only": True,
        "adaptive_research": True,
        "fresh_holdout_claimed": False,
        "interpretation": "corrected repeated-forward research evidence only; not absolute hidden calibration",
        "interruption_resume_disclosure": INTERRUPTION_RESUME_DISCLOSURE,
        "inner_role": config["validation"]["inner_role"],
        "hyperparameter_searches": 0,
        "mask_audit": mask_audit.__dict__,
        "fold_contracts": fold_records,
        "inner_diagnostic": {
            column: runner.metric_report(
                inner_oof, prediction_column=prediction, official_layer_counts=layer_counts
            )
            for column, prediction in {
                "baseline": "baseline",
                "unprojected_blend50": "blend_prediction",
                "candidate": "prediction",
            }.items()
        },
        "outer_repeated_forward": {
            column: runner.metric_report(
                outer_oof, prediction_column=prediction, official_layer_counts=layer_counts
            )
            for column, prediction in {
                "baseline": "baseline",
                "unprojected_blend50": "blend_prediction",
                "candidate": "prediction",
            }.items()
        },
    }
    metrics["outer_repeated_forward"]["candidate_vs_baseline_bootstrap"] = (
        runner.paired_fold_day_bootstrap(
            outer_oof,
            reference_column="baseline",
            candidate_column="prediction",
            official_layer_counts=layer_counts,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
            interval=float(bootstrap["interval"]),
        )
    )
    metrics["outer_repeated_forward"]["projection_vs_unprojected_bootstrap"] = (
        runner.paired_fold_day_bootstrap(
            outer_oof,
            reference_column="blend_prediction",
            candidate_column="prediction",
            official_layer_counts=layer_counts,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]) + 1,
            interval=float(bootstrap["interval"]),
        )
    )
    _exclusive_json(planned["metrics"], metrics)

    full_train = np.isfinite(base.frame["residual"].to_numpy(float))
    final_item = FIT_ORDER[-1]
    final_model = runner.fit_fixed_blend(
        base, lean, full_train, seed=int(final_item["seed"])
    )
    _validate_model(
        final_model, base, lean, seed=int(final_item["seed"]), role="final_full_train"
    )
    _exclusive_joblib(planned["final_model"], final_model)
    model_records["final_full_train"] = {
        "path": _logical(planned["final_model"]),
        "sha256": _sha256(planned["final_model"]),
        "bytes": planned["final_model"].stat().st_size,
        "seed": int(final_item["seed"]),
        "origin": final_item["origin"],
        "refit_during_resume": True,
        "training": runner._training_summary(
            base.frame, full_train, pd.Timestamp("2262-01-01", tz="UTC")
        ),
    }

    hidden_start, hidden_stop = config["masking"]["hidden_window_kst_half_open"]
    submission, test_diagnostics = runner._predict_official_candidate(
        final_model,
        data,
        masked,
        base,
        lean,
        endpoints,
        hidden_start=hidden_start,
        hidden_stop=hidden_stop,
    )
    _exclusive_csv(planned["candidate"], submission)
    candidate_validation = runner.validate_submission(planned["candidate"], data.test_index)
    reloaded = pd.read_csv(
        planned["candidate"], dtype={"station": "string", "time": "string"}
    )
    if not reloaded[runner.KEYS].equals(data.test_index[runner.KEYS]):
        raise AssertionError("resumed candidate reload changed official key order")
    frozen_after = runner._frozen_snapshot()
    if frozen_before != frozen_after:
        raise AssertionError("a frozen/current P2 submission changed during resume")

    elapsed = float(time.perf_counter() - started)
    result = {
        "experiment_id": config["experiment_id"],
        "completed_at_kst": _now(),
        "elapsed_resume_seconds": elapsed,
        "decision_scope": "CORRECTED_REPEATED_FORWARD_RESEARCH_EVIDENCE_ONLY",
        "promotion_or_upload_authorized": False,
        "current_frozen_submission_modified": False,
        "hidden_target_temperature_values_accessed": 0,
        "hidden_target_salinity_values_accessed": 0,
        "same_original_attempt": 1,
        "new_generation_or_attempt": False,
        "initial_completed_estimator_fits": 6,
        "resume_completed_estimator_fits": 8,
        "interruption_resume_disclosure": INTERRUPTION_RESUME_DISCLOSURE,
        "outer_metrics": metrics["outer_repeated_forward"],
        "candidate": {
            "path": _logical(planned["candidate"]),
            "sha256": _sha256(planned["candidate"]),
            "bytes": planned["candidate"].stat().st_size,
            "validation": candidate_validation,
            "diagnostics": test_diagnostics,
        },
    }
    _exclusive_json(planned["result"], result)

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
        "candidate": result["candidate"],
    }
    resume_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "p2_corrected_repeated_forward.manifest.v2_resumed_same_attempt",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _now(),
        "research_only": True,
        "upload_allowed": False,
        "adaptive_research": True,
        "fresh_holdout_claimed": False,
        "same_original_attempt": 1,
        "new_generation_or_attempt": False,
        "interruption_resume_disclosure": metrics["interruption_resume_disclosure"],
        "pinned_pre_resume_state": {
            "core_sha256": PINNED_CORE_SHA256,
            "runtime_module_sha256": PINNED_RUNTIME_MODULE_SHA256,
            "partial_files": PARTIAL_FILES,
            "fit_order": FIT_ORDER,
        },
        "status_reconciliation": {
            "historical_interrupted_status": {
                "path": _logical(OUTPUT_DIR / "status.json"),
                "sha256": _sha256(OUTPUT_DIR / "status.json"),
                "immutable": True,
                "semantic_role": "historical_interrupted_running_state_only",
            },
            "final_resume_completion_receipt": {
                "path": _logical(RESUME_COMPLETION_RECEIPT),
                "write_mode": "append-only O_EXCL materialization after manifest and seal",
                "semantic_role": "authoritative final resumed state",
            },
        },
        "config": {
            "path": _logical(CONFIG_PATH),
            "sha256": _sha256(CONFIG_PATH),
            "bytes": CONFIG_PATH.stat().st_size,
        },
        "sources": runtime["source_records"],
        "implementation": {
            _logical(HELPER_PATH): {
                "sha256": _sha256(HELPER_PATH),
                "bytes": HELPER_PATH.stat().st_size,
            },
            _logical(RUNNER_PATH): {
                "sha256": _sha256(RUNNER_PATH),
                "bytes": RUNNER_PATH.stat().st_size,
            },
            _logical(resume_path): {
                "sha256": _sha256(resume_path),
                "bytes": resume_path.stat().st_size,
            },
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
        "git_after": runner._git_state(),
        "attempt_lock": {
            "path": _logical(ATTEMPT_LOCK),
            "sha256": _sha256(ATTEMPT_LOCK),
            "bytes": ATTEMPT_LOCK.stat().st_size,
        },
        "resume_concurrency_lock": resume_lock,
        "joint_mask": mask_audit.__dict__,
        "feature_counts": {"base": len(base.feature_columns), "lean": len(lean.feature_columns)},
        "hyperparameter_searches": 0,
        "parameter_threshold_or_weight_changes": 0,
        "model_records": model_records,
        "artifacts": artifacts,
        "frozen_submission_snapshot_count": len(frozen_before),
        "frozen_submission_snapshot_unchanged": True,
        "elapsed_resume_seconds": elapsed,
    }
    _exclusive_json(planned["manifest"], manifest)
    seal = {
        "experiment_id": config["experiment_id"],
        "same_original_attempt": 1,
        "new_generation_or_attempt": False,
        "manifest_path": _logical(planned["manifest"]),
        "manifest_sha256": _sha256(planned["manifest"]),
        "manifest_bytes": planned["manifest"].stat().st_size,
        "candidate_path": _logical(planned["candidate"]),
        "candidate_sha256": _sha256(planned["candidate"]),
        "outer_oof_sha256": _sha256(planned["outer_oof"]),
        "sealed_at_kst": _now(),
        "upload_performed": False,
    }
    _exclusive_json(planned["seal"], seal)
    completion_receipt = {
        "schema_version": "p2_corrected_repeated_forward.resume_completion_status.v1",
        "experiment_id": config["experiment_id"],
        "same_original_attempt": 1,
        "new_generation_or_attempt": False,
        "status": "complete",
        "completed_at_kst": _now(),
        "historical_interrupted_status": {
            "path": _logical(OUTPUT_DIR / "status.json"),
            "sha256": _sha256(OUTPUT_DIR / "status.json"),
            "immutable": True,
            "semantic_role": "historical_interrupted_running_state_only",
        },
        "manifest": {
            "path": _logical(planned["manifest"]),
            "sha256": _sha256(planned["manifest"]),
        },
        "seal": {"path": _logical(planned["seal"]), "sha256": _sha256(planned["seal"])},
        "candidate": {
            "path": _logical(planned["candidate"]),
            "sha256": _sha256(planned["candidate"]),
        },
        "resume_concurrency_lock": resume_lock,
        "interruption_resume_disclosure": INTERRUPTION_RESUME_DISCLOSURE,
        "upload_performed": False,
    }
    _exclusive_json(RESUME_COMPLETION_RECEIPT, completion_receipt)
    completion_receipt["receipt_sha256"] = _sha256(RESUME_COMPLETION_RECEIPT)
    print(
        json.dumps(
            {"result": result, "seal": seal, "resume_completion": completion_receipt},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorization-token")
    return parser


def main() -> int:
    args = _parser().parse_args()
    data_dir = args.data_dir.resolve()
    if args.dry_run:
        if args.authorization_token is not None:
            raise ValueError("dry-run must not receive an authorization token")
        return _dry_run(data_dir)
    return _resume(data_dir, args.authorization_token)


if __name__ == "__main__":
    raise SystemExit(main())
