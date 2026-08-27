"""Append-only recovery for the P1 v1 full-fit inference interruption.

The original one-shot outer winner and valid full XGBoost fit are reused.  The
partial causal fit is never overwritten: a corrected causal LightGBM is fit in
a new directory with ``features.mode='causal'`` explicitly forced, then the
candidate and saved-model reproduction are completed.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from p1_qc.config import load_config
from p1_qc.data import load_dataset
from p1_qc.improvement_cycle import (
    PROJECT_ROOT,
    P1ImprovementEnsemble,
    _bootstrap_day_delta,
    _guard_report,
    _predict_ensemble,
    _resolve,
    load_cycle_config,
    sha256_file,
    write_json,
    write_json_exclusive,
)
from p1_qc.pipeline import load_or_build_features, train_full_model
from p1_qc.submission import validate_submission, write_submission

KST = ZoneInfo("Asia/Seoul")
ROOT_RELATIVE = "artifacts/p1_full_improvement_cycle_20260822_v1"


def _resume_paths() -> dict[str, Path]:
    return {
        "module": _resolve("src/p1_qc/improvement_cycle_resume.py"),
        "runner": _resolve("scripts/resume_p1_full_improvement_cycle.py"),
        "tests": _resolve("tests/test_p1_full_improvement_cycle_resume.py"),
    }


def seal_resume(config_path: str | Path) -> dict[str, Any]:
    resolved_config, config = load_cycle_config(config_path)
    root = _resolve(ROOT_RELATIVE)
    original_seal = json.loads((root / "preexecution_seal.json").read_text(encoding="utf-8"))
    for name, expected in original_seal["implementation_sha256"].items():
        path_map = {
            "config": _resolve("configs/experiments/p1_full_improvement_cycle_v1.json"),
            "module": _resolve("src/p1_qc/improvement_cycle.py"),
            "runner": _resolve("scripts/run_p1_full_improvement_cycle.py"),
            "tests": _resolve("tests/test_p1_full_improvement_cycle.py"),
        }
        if sha256_file(path_map[name]) != expected:
            raise RuntimeError(f"original sealed implementation changed: {name}")
    required = {
        "original_config": resolved_config,
        "original_seal": root / "preexecution_seal.json",
        "original_attempt_lock": root / "attempt_lock.json",
        "original_failure": root / "execution_failure.json",
        "winner_oof": root / "winner_oof.parquet",
        "valid_offline_xgboost": root / "models/offline_xgboost_full.joblib",
        "invalid_partial_causal": root / "models/causal_lightgbm_full.joblib",
        "invalid_partial_ensemble": root / "models/p1_improved_ensemble.joblib",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    forbidden = [
        root / "resume_lock.json",
        root / "resume_models",
        root / "candidate",
        root / "result.json",
    ]
    if any(path.exists() for path in forbidden):
        raise FileExistsError("resume output already exists")
    failure = json.loads(required["original_failure"].read_text(encoding="utf-8"))
    if failure.get("error") != "model and inference feature modes differ":
        raise RuntimeError("resume is limited to the registered feature-mode interruption")
    receipt = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "sealed_at_kst": datetime.now(KST).isoformat(),
        "decision": "RESUME_ONLY_CAUSAL_REFIT_AND_INFERENCE",
        "root_cause": "load_config retained the default features.mode=offline despite top-level P1QC_MODE=causal; the initial partial causal full fit is ineligible",
        "original_artifact_sha256": {name: sha256_file(path) for name, path in required.items()},
        "resume_implementation_sha256": {
            name: sha256_file(path) for name, path in _resume_paths().items()
        },
        "registered_input_sha256": config["expected_sha256"],
        "allowed_new_model_fits": 1,
        "allowed_new_test_prediction_generations": 2,
        "test_label_reads": 0,
        "submission_uploads": 0,
    }
    write_json_exclusive(root / "resume_preexecution_seal.json", receipt)
    return receipt


def _verify_resume_seal(config: dict[str, Any], root: Path) -> dict[str, Any]:
    seal_path = root / "resume_preexecution_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for name, expected in seal["resume_implementation_sha256"].items():
        if sha256_file(_resume_paths()[name]) != expected:
            raise RuntimeError(f"resume implementation changed: {name}")
    for name, expected in config["expected_sha256"].items():
        if sha256_file(_resolve(config["paths"][name])) != expected:
            raise RuntimeError(f"registered input changed before resume: {name}")
    return seal


def resume_cycle(config_path: str | Path) -> dict[str, Any]:
    resolved_config, config = load_cycle_config(config_path)
    root = _resolve(ROOT_RELATIVE)
    resume_seal = _verify_resume_seal(config, root)
    write_json_exclusive(
        root / "resume_lock.json",
        {
            "schema_version": "1.0",
            "started_at_kst": datetime.now(KST).isoformat(),
            "pid": os.getpid(),
            "resume_seal_sha256": sha256_file(root / "resume_preexecution_seal.json"),
        },
    )
    train_path = _resolve(config["paths"]["train_csv"])
    test_path = _resolve(config["paths"]["test_csv"])
    frozen_path = _resolve(config["paths"]["frozen_submission"])
    protected_before = {
        "train": sha256_file(train_path),
        "test": sha256_file(test_path),
        "frozen": sha256_file(frozen_path),
    }
    try:
        original_partial = joblib.load(root / "models/p1_improved_ensemble.joblib")
        if original_partial.winner_branch != "causal_event_rescue_walk_forward":
            raise RuntimeError("unexpected sealed outer winner")
        if original_partial.causal_model.feature_mode != "causal":
            raise RuntimeError("partial causal model metadata is unexpected")

        data_dir = train_path.parent
        raw_causal_config = load_config(
            _resolve(config["paths"]["p1_config"]),
            env={"P1_DATA_DIR": str(data_dir), "P1QC_MODE": "causal"},
        )
        causal_config = replace(
            raw_causal_config,
            mode="causal",
            features=replace(raw_causal_config.features, mode="causal"),
        )
        offline_config = replace(
            raw_causal_config,
            mode="offline",
            features=replace(raw_causal_config.features, mode="offline"),
        )
        train = load_dataset(train_path, kind="train", audit=True)
        causal_train_features = load_or_build_features(
            train, causal_config, kind="train", use_cache=True
        )
        selection = json.loads(
            _resolve(config["paths"]["causal_selection"]).read_text(encoding="utf-8")
        )
        corrected_causal = train_full_model(train, causal_train_features, causal_config, selection)
        if corrected_causal.feature_mode != "causal":
            raise RuntimeError("corrective causal refit did not retain causal mode")
        resume_models = root / "resume_models"
        resume_models.mkdir(parents=True, exist_ok=False)
        causal_path = resume_models / "causal_lightgbm_full_corrected.joblib"
        joblib.dump(corrected_causal, causal_path, compress=3)
        ensemble = P1ImprovementEnsemble(
            incumbent_model=original_partial.incumbent_model,
            causal_model=corrected_causal,
            winner_branch=original_partial.winner_branch,
            deployment_parameters=original_partial.deployment_parameters,
            logistic_model=original_partial.logistic_model,
            config_sha256=sha256_file(resolved_config),
        )
        ensemble_path = resume_models / "p1_improved_ensemble_corrected.joblib"
        joblib.dump(ensemble, ensemble_path, compress=3)

        test = load_dataset(test_path, kind="test", audit=True)
        offline_test_features = load_or_build_features(
            test, offline_config, kind="test", use_cache=True
        )
        causal_test_features = load_or_build_features(
            test, causal_config, kind="test", use_cache=True
        )
        candidate = _predict_ensemble(ensemble, test, offline_test_features, causal_test_features)
        candidate_dir = root / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=False)
        candidate_path = candidate_dir / "P1_IMPROVED_ENSEMBLE_V1.csv"
        write_submission(candidate, candidate_path)
        validation = validate_submission(candidate_path, test)
        loaded = joblib.load(ensemble_path)
        reproduced = _predict_ensemble(loaded, test, offline_test_features, causal_test_features)
        reproduced_path = candidate_dir / "reproduced.csv"
        write_submission(reproduced, reproduced_path)
        if candidate_path.read_bytes() != reproduced_path.read_bytes():
            raise RuntimeError("corrected saved ensemble did not reproduce candidate bytes")

        oof = pd.read_parquet(root / "winner_oof.parquet")
        truth = oof["label"].to_numpy(dtype=np.int8)
        incumbent = oof["incumbent_prediction"].to_numpy(dtype=np.int8)
        winner = oof["candidate_prediction"].to_numpy(dtype=np.int8)
        frame = oof.loc[:, [*config["key_columns"], "fold"]]
        guard = _guard_report(frame, truth, incumbent, winner, config["success_guards"])
        if not guard["passed"]:
            raise RuntimeError("sealed winner no longer passes success guards")
        bootstrap = _bootstrap_day_delta(
            frame,
            truth,
            incumbent,
            winner,
            replicates=int(config["bootstrap"]["replicates"]),
            seed=int(config["bootstrap"]["seed"]),
        )
        protected_after = {
            "train": sha256_file(train_path),
            "test": sha256_file(test_path),
            "frozen": sha256_file(frozen_path),
        }
        if protected_before != protected_after:
            raise RuntimeError("protected file changed during resume")
        metrics = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "winner": ensemble.winner_branch,
            "outer_rows": len(oof),
            "outer_key_duplicates": int(oof.duplicated(config["key_columns"]).sum()),
            "outer_key_alignment_exact": True,
            "guard": guard,
            "bootstrap": bootstrap,
            "target_fold_label_reads_before_prediction": 0,
        }
        write_json(root / "metrics.json", metrics)
        result = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "status": "COMPLETE_AFTER_APPEND_ONLY_RESUME_WINNER_TRAINED_CANDIDATE_REPRODUCED_NOT_UPLOADED",
            "completed_at_kst": datetime.now(KST).isoformat(),
            "winner": ensemble.winner_branch,
            "outer_incumbent_f1": guard["incumbent"]["f1"],
            "outer_candidate_f1": guard["candidate"]["f1"],
            "outer_f1_delta": guard["micro_f1_delta"],
            "guard_passed": True,
            "bootstrap": bootstrap,
            "full_fit": {
                "train_rows": len(train),
                "valid_offline_xgboost_reused_from_initial_attempt": {
                    "path": str(root / "models/offline_xgboost_full.joblib"),
                    "sha256": sha256_file(root / "models/offline_xgboost_full.joblib"),
                },
                "invalid_partial_causal_preserved": {
                    "path": str(root / "models/causal_lightgbm_full.joblib"),
                    "sha256": sha256_file(root / "models/causal_lightgbm_full.joblib"),
                    "eligible": False,
                },
                "corrected_causal_lightgbm": {
                    "path": str(causal_path),
                    "sha256": sha256_file(causal_path),
                    "feature_mode": corrected_causal.feature_mode,
                },
                "corrected_ensemble": {
                    "path": str(ensemble_path),
                    "sha256": sha256_file(ensemble_path),
                },
                "new_resume_model_fits": 1,
            },
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
                "bytes": candidate_path.stat().st_size,
                "rows": len(candidate),
                "positive_rows": int(candidate["label"].sum()),
                "positive_rate": float(candidate["label"].mean()),
            },
            "reproduction": {
                "path": str(reproduced_path),
                "sha256": sha256_file(reproduced_path),
                "byte_identical": True,
            },
            "strict_validation": validation,
            "protected_hashes": {
                "before": protected_before,
                "after": protected_after,
            },
            "operation_counters": {
                "initial_valid_xgboost_full_fits": 1,
                "initial_invalid_causal_full_fits_preserved": 1,
                "resume_corrective_causal_full_fits": 1,
                "test_prediction_generations": 2,
                "test_label_reads": 0,
                "submission_uploads": 0,
                "source_mutations": 0,
                "frozen_submission_mutations": 0,
            },
        }
        write_json(root / "result.json", result)
        manifest = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "created_at_kst": datetime.now(KST).isoformat(),
            "resume_seal_sha256": sha256_file(root / "resume_preexecution_seal.json"),
            "resume_lock_sha256": sha256_file(root / "resume_lock.json"),
            "resume_implementation_sha256": resume_seal["resume_implementation_sha256"],
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "artifacts": {},
        }
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                manifest["artifacts"][str(path.relative_to(PROJECT_ROOT))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        write_json(root / "manifest.json", manifest)
        return result
    except Exception as exc:
        failure_path = root / "resume_failure.json"
        if not failure_path.exists():
            write_json(
                failure_path,
                {
                    "status": "RESUME_FAILED",
                    "failed_at_kst": datetime.now(KST).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "submission_uploads": 0,
                },
            )
        raise


__all__ = ["resume_cycle", "seal_resume"]
