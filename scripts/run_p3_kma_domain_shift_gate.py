"""Run the pre-registered KMA-versus-P3-train domain-shift gate.

The runner is deliberately unable to accept a P3 test file, model, OOF table,
or submission.  It writes only an aggregate JSON result under ignored
``artifacts/`` and a small local progress gauge.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow
import sklearn

from p3_wave.domain_shift import (
    ALLOWED_P3_TRAIN_FILES,
    FORBIDDEN_P3_FILES,
    DomainShiftError,
    aggregate_input_receipts,
    build_sampled_representation,
    density_ratio_summary,
    evaluate_domain_classifier,
    gate_decision,
    load_external_canonical,
    load_target_train_canonical,
    resolve_p3_train_paths,
    sha256_file,
    standardized_drift_summary,
    validate_external_manifest,
)

DEFAULT_CONFIG = Path("configs/experiments/p3_kma_domain_shift_gate_v1.json")
STATUS_RELATIVE_PATH = Path("artifacts/status/p3_kma_domain_shift_v1.json")
KST = ZoneInfo("Asia/Seoul")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class Gauge:
    """Atomic local-only progress record with no paths, rows, or credentials."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()

    def update(self, phase: str, progress: float, *, status: str = "running") -> None:
        bounded = float(np.clip(progress, 0.0, 1.0))
        elapsed = time.perf_counter() - self.started
        eta = 0.0 if bounded >= 1.0 else (elapsed * (1.0 - bounded) / bounded if bounded else 0.0)
        _atomic_json(
            self.path,
            {
                "status": status,
                "phase": phase,
                "progress": round(bounded, 6),
                "elapsed_seconds": round(elapsed, 3),
                "eta_seconds": round(eta, 3),
            },
        )


def _resolve_repo_path(repo_root: Path, configured: str, *, must_exist: bool) -> Path:
    candidate = Path(configured)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.expanduser().resolve(strict=must_exist)
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise DomainShiftError("configured repository input/output escapes the repository") from exc
    return resolved


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DomainShiftError("unable to read the domain-shift preregistration") from exc
    if config.get("experiment_id") != "P3_kma_domain_shift_gate_v1":
        raise DomainShiftError("unexpected experiment_id")
    if int(config.get("seed", -1)) != 20260821:
        raise DomainShiftError("pre-registered seed changed")
    inputs = config.get("inputs", {})
    if tuple(inputs.get("allowed_p3_train_files", ())) != ALLOWED_P3_TRAIN_FILES:
        raise DomainShiftError("P3 training allowlist changed")
    if set(inputs.get("forbidden_p3_files", ())) != set(FORBIDDEN_P3_FILES):
        raise DomainShiftError("P3 forbidden-file list changed")
    prohibitions = config.get("prohibitions", {})
    required_false = (
        "read_test_context",
        "read_hidden_labels",
        "read_frozen_models_oof_or_submissions",
        "train_forecast_model",
        "write_submission",
    )
    if any(prohibitions.get(key) is not False for key in required_false):
        raise DomainShiftError("one or more safety prohibitions are not fixed false")
    return config


def _git_summary(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        porcelain = run("status", "--short")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "dirty_entry_count": None}
    entries = [line for line in porcelain.splitlines() if line]
    return {"commit": commit, "dirty": bool(entries), "dirty_entry_count": len(entries)}


def _fixed_output_dir(repo_root: Path, config: Mapping[str, Any]) -> Path:
    configured = str(config["output"]["directory"])
    output_dir = _resolve_repo_path(repo_root, configured, must_exist=False)
    artifacts_root = (repo_root / "artifacts").resolve()
    try:
        output_dir.relative_to(artifacts_root)
    except ValueError as exc:
        raise DomainShiftError("aggregate output must stay below ignored artifacts/") from exc
    return output_dir


def run_gate(
    *,
    repo_root: Path,
    config_path: Path,
    p3_data_dir: Path,
    replace: bool = False,
) -> dict[str, Any]:
    status_path = (repo_root / STATUS_RELATIVE_PATH).resolve()
    gauge = Gauge(status_path)
    gauge.update("preflight", 0.01)
    config = _load_config(config_path)
    config_hash = sha256_file(config_path)
    output_dir = _fixed_output_dir(repo_root, config)
    result_path = output_dir / str(config["output"]["result_file"])
    if result_path.exists() and not replace:
        raise DomainShiftError(
            "aggregate result already exists; use --replace for an explicit rerun"
        )

    external_path = _resolve_repo_path(
        repo_root, str(config["inputs"]["external_parquet"]), must_exist=True
    )
    manifest_path = _resolve_repo_path(
        repo_root, str(config["inputs"]["external_manifest"]), must_exist=True
    )
    train_paths = resolve_p3_train_paths(p3_data_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_external_manifest(manifest)
    gauge.update("hash_and_source_validation", 0.05)
    input_paths = {
        "preregistration": config_path,
        "external_manifest": manifest_path,
        "external_source": external_path,
        "p3_train_wave": train_paths["train_wave.csv"],
        "p3_train_atmos": train_paths["train_atmos.csv"],
    }
    input_receipts = aggregate_input_receipts(input_paths)
    external_hash = next(
        item["sha256"] for item in input_receipts if item["role"] == "external_source"
    )
    if manifest.get("file_sha256") != external_hash:
        raise DomainShiftError("KMA external Parquet hash does not match its manifest")

    gauge.update("load_external_source", 0.1)
    source = load_external_canonical(
        external_path, maximum_time_kst=str(config["inputs"]["maximum_external_time_kst"])
    )
    gauge.update("load_p3_train_only", 0.16)
    target = load_target_train_canonical(train_paths)
    gauge.update("build_causal_representation", 0.2)

    def feature_progress(completed: int, total: int) -> None:
        fraction = completed / total if total else 1.0
        gauge.update("build_causal_representation", 0.2 + 0.36 * fraction)

    sampled = build_sampled_representation(
        source,
        target,
        config["representation"],
        seed=int(config["seed"]),
        progress_callback=feature_progress,
    )
    del source, target
    gauge.update("group_blocked_domain_classifier", 0.58)

    def classifier_progress(completed: int, total: int) -> None:
        gauge.update("group_blocked_domain_classifier", 0.58 + 0.32 * completed / total)

    evaluation = evaluate_domain_classifier(
        sampled,
        config["classifier"],
        config["representation"],
        seed=int(config["seed"]),
        progress_callback=classifier_progress,
    )
    gauge.update("aggregate_diagnostics", 0.92)
    decision = gate_decision(evaluation.auc, config["gate"])
    density = density_ratio_summary(sampled.domain, evaluation.probabilities, config["gate"])
    drift = standardized_drift_summary(sampled, evaluation.normalized_oof)
    elapsed = time.perf_counter() - gauge.started
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "created_at_kst": datetime.now(KST).isoformat(),
        "objective": "source_vs_target_domain_discrimination_only",
        "does_not_test_transfer_utility": True,
        "interpretation_scope": {
            "direct_row_concat_or_full_finetune_gate_only": True,
            "kma_source_permanently_rejected": False,
            "future_independent_hypothesis": (
                "source_prediction_meta_only_plus_target_inner_utility_gate"
            ),
            "forecast_model_trained": False,
            "test_inference_run": False,
            "submission_written": False,
        },
        "decision": decision,
        "domain_classifier": {
            "oof_auc": evaluation.auc,
            "fold_auc": list(evaluation.fold_auc),
            "folds": list(evaluation.fold_summaries),
            "classifier": config["classifier"],
            "grouping": "station_year_stratified_group_kfold",
            "normalization": config["representation"]["stationwise_normalization"],
        },
        "density_ratio": density,
        "standardized_drift": drift,
        "representation": {
            **sampled.summary,
            "grid_minutes": config["representation"]["grid_minutes"],
            "history_windows_hours": config["representation"]["history_windows_hours"],
            "causal": True,
            "row_level_artifact_written": False,
        },
        "input_receipts": input_receipts,
        "preregistration_sha256": config_hash,
        "access_audit": {
            "p3_files_read": list(ALLOWED_P3_TRAIN_FILES),
            "forbidden_p3_file_reads": 0,
            "test_context_read": False,
            "test_index_read": False,
            "hidden_labels_read": False,
            "frozen_model_oof_or_submission_read": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
            "platform": platform.platform(),
        },
        "git": _git_summary(repo_root),
        "runtime_seconds": elapsed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(result_path, result)
    result_hash = sha256_file(result_path)
    gauge.update("completed", 1.0, status="completed")
    return {
        "status": "completed",
        "decision": decision["tier"],
        "oof_auc": evaluation.auc,
        "does_not_test_transfer_utility": True,
        "result_sha256": result_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--p3-data-dir",
        type=Path,
        default=None,
        help="distributed P3 source directory; defaults to P3_DATA_DIR",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="explicitly replace an existing aggregate result; never changes source data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    try:
        config_path = config_path.resolve(strict=True)
        p3_data_dir = args.p3_data_dir
        if p3_data_dir is None:
            configured = os.environ.get("P3_DATA_DIR")
            if not configured:
                raise DomainShiftError("P3_DATA_DIR is required")
            p3_data_dir = Path(configured)
        summary = run_gate(
            repo_root=repo_root,
            config_path=config_path,
            p3_data_dir=p3_data_dir,
            replace=args.replace,
        )
    except (DomainShiftError, OSError, ValueError, json.JSONDecodeError) as exc:
        Gauge((repo_root / STATUS_RELATIVE_PATH).resolve()).update(
            "failed_closed", 0.0, status="failed"
        )
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "forecast_model_trained": False,
                    "submission_written": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
