"""Run the preregistered aggregate-only P2 NCR_LGBM Stage 1 gate.

Without ``--execute-stage1`` this program performs hash/firewall preflight
only.  The execution path reads only ``observations.csv`` plus the frozen
historical incumbent OOF artifact.  It cannot read official test/sample files,
create row predictions or CSVs, create submission candidates, or upload.
"""

# ruff: noqa: E402 -- thread limits must be set before numerical imports.

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "8"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KST = ZoneInfo("Asia/Seoul")
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_normalized_curvature_residual_lgbm_stage1_v1.json"
)
EXPECTED_CONFIG_SHA256 = "074ab4b55cb1a0efa8fdeec631d72070b2362352e20cf2d4eb07cc0d139e2206"
RUNNER_HASH_MODE = "sha256_config_hash_literal_normalized_v1"
RUNNER_HASH_CANONICAL_LINE = (
    b'EXPECTED_CONFIG_SHA256 = "__CONFIG_SHA256_CANONICAL__"'
)
RUNNER_HASH_PATTERN = re.compile(
    rb'(?m)^EXPECTED_CONFIG_SHA256 = "[^"\r\n]+"$'
)
FORBIDDEN_FILE_NAMES = {
    "test_index.csv",
    "sample_submission.csv",
    "baseline_interp.csv",
}
FORBIDDEN_PATH_TOKENS = ("submission", "candidate")


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _normalized_runner_sha256(path: Path) -> str:
    raw = path.read_bytes()
    normalized, replacements = RUNNER_HASH_PATTERN.subn(RUNNER_HASH_CANONICAL_LINE, raw)
    if replacements != 1:
        raise RuntimeError("runner config-hash literal normalization failed closed")
    return hashlib.sha256(normalized).hexdigest()


def _assert_safe_input_path(path: Path) -> None:
    lowered_parts = tuple(part.lower() for part in path.parts)
    if path.name.lower() in FORBIDDEN_FILE_NAMES:
        raise RuntimeError(f"forbidden P2 input path: {path}")
    if any(token in part for part in lowered_parts for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"submission/candidate-like input path is forbidden: {path}")


def _repo_file(relative_text: str) -> Path:
    relative = Path(relative_text)
    _assert_safe_input_path(relative)
    resolved = (PROJECT_ROOT / relative).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT) or not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _validate_runtime_pins(config: dict[str, Any]) -> dict[str, str]:
    pins = config["runtime_pins"]
    observed: dict[str, str] = {"python": platform.python_version()}
    if observed["python"] != str(pins["python"]):
        raise RuntimeError("Python runtime version drift")
    for distribution, expected in pins["packages"].items():
        observed[str(distribution)] = importlib.metadata.version(str(distribution))
        if observed[str(distribution)] != str(expected):
            raise RuntimeError(f"runtime package version drift: {distribution}")
    return observed


def _verify_static_bundle(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the preregistration and implementation before numerical imports."""

    config_path = config_path.resolve()
    if not config_path.is_relative_to(PROJECT_ROOT) or not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_sha256 = _sha256(config_path)
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("NCR_LGBM preregistration hash drift")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("preregistration must be a JSON object")
    if config.get("schema_version") != (
        "p2_normalized_curvature_residual_lgbm_stage1.prereg.v1"
    ):
        raise ValueError("unexpected NCR_LGBM preregistration schema")
    if config.get("status") != "PREREGISTERED_NOT_EXECUTED":
        raise ValueError("NCR_LGBM Stage 1 is not preregistered")
    policy = config["selection_policy"]
    if bool(policy["official_score_used_for_gate_or_tuning"]):
        raise ValueError("official-score calibration is forbidden")
    if bool(policy["result_based_retuning"]):
        raise ValueError("result-based tuning is forbidden")
    if int(policy["candidate_grid_size"]) != 1:
        raise ValueError("Stage 1 candidate grid changed")
    if int(policy["stage1_model_fit_count"]) != 3:
        raise ValueError("Stage 1 fit budget changed")
    output = config["output"]
    if not bool(output["aggregate_only"]):
        raise ValueError("aggregate-only output contract changed")
    if bool(output["row_predictions_written"]) or bool(output["csv_files_written"]):
        raise ValueError("row-level output is forbidden")
    if int(output["submission_files_generated"]) or int(output["uploads"]):
        raise ValueError("external action contract changed")
    if sorted(output["allowed_files"]) != ["manifest.json", "result.json"]:
        raise ValueError("output allow-list changed")

    runner_path = Path(__file__).resolve()
    implementations: dict[str, dict[str, Any]] = {}
    for name, pin in config["implementation_pins"].items():
        path = _repo_file(str(pin["path"]))
        if name == "runner":
            if path != runner_path or str(pin["hash_mode"]) != RUNNER_HASH_MODE:
                raise RuntimeError("runner pin contract drift")
            pinned_digest = _normalized_runner_sha256(path)
        else:
            if str(pin["hash_mode"]) != "sha256_raw_v1":
                raise RuntimeError(f"implementation hash mode drift: {name}")
            pinned_digest = _sha256(path)
        if pinned_digest != str(pin["sha256"]):
            raise RuntimeError(f"implementation hash drift: {name}")
        if path.stat().st_size != int(pin["bytes"]):
            raise RuntimeError(f"implementation size drift: {name}")
        implementations[name] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "hash_mode": str(pin["hash_mode"]),
            "pinned_sha256": pinned_digest,
            "raw_sha256": _sha256(path),
        }

    references: dict[str, dict[str, Any]] = {}
    for name, reference in config["immutable_references"].items():
        path = _repo_file(str(reference["path"]))
        digest = _sha256(path)
        if digest != str(reference["sha256"]):
            raise RuntimeError(f"immutable reference hash drift: {name}")
        if path.stat().st_size != int(reference["bytes"]):
            raise RuntimeError(f"immutable reference size drift: {name}")
        references[name] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    runtime = _validate_runtime_pins(config)
    return config, {
        "config": {
            "path": config_path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": config_path.stat().st_size,
            "sha256": config_sha256,
        },
        "implementation_pins": implementations,
        "immutable_references": references,
        "runtime": runtime,
    }


def _assert_static_bundle_unchanged(
    config_path: Path,
    expected_config: dict[str, Any],
    expected_bundle: dict[str, Any],
) -> None:
    observed_config, observed_bundle = _verify_static_bundle(config_path)
    if observed_config != expected_config or observed_bundle != expected_bundle:
        raise RuntimeError("NCR_LGBM static bundle changed after preflight")


def _git_state() -> dict[str, Any]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    head = run("rev-parse", "HEAD").stdout.strip()
    dirty = bool(run("status", "--porcelain=v1", "--untracked-files=normal").stdout)
    return {"head": head, "dirty": dirty}


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _repo_reference(config: dict[str, Any], name: str) -> Path:
    return _repo_file(str(config["immutable_references"][name]["path"]))


def _output_directory(config: dict[str, Any]) -> Path:
    relative = Path(str(config["output"]["directory"]))
    resolved = (PROJECT_ROOT / relative).resolve()
    artifact_root = (PROJECT_ROOT / "artifacts").resolve()
    if not resolved.is_relative_to(artifact_root) or resolved == artifact_root:
        raise RuntimeError("NCR_LGBM output must be a specific artifacts subdirectory")
    if resolved.exists():
        raise FileExistsError(
            f"append-only NCR_LGBM output already exists; refusing overwrite: {resolved}"
        )
    return resolved


def _assert_exact_metric_pin(
    incumbent_metrics: dict[str, Any], reference: dict[str, Any]
) -> None:
    tolerance = 1e-12
    if int(incumbent_metrics["rows"]) != int(reference["rows"]):
        raise RuntimeError("exact same-season incumbent row count drift")
    for name in ("row_pooled_rmse_c", "layer_equal_rmse_c"):
        if abs(float(incumbent_metrics[name]) - float(reference[name])) > tolerance:
            raise RuntimeError(f"exact same-season incumbent {name} drift")
    for layer, expected in reference["by_layer_rmse_c"].items():
        observed = float(incumbent_metrics["by_layer_rmse_c"][str(layer)])
        if abs(observed - float(expected)) > tolerance:
            raise RuntimeError(f"exact same-season incumbent layer {layer} RMSE drift")


def _preflight(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config, verified_bundle = _verify_static_bundle(config_path)
    if int(config["execution"]["maximum_new_model_fits"]) != 3:
        raise ValueError("NCR_LGBM fit ceiling changed")
    if config["execution"]["default_mode"] != "preflight_only":
        raise ValueError("NCR_LGBM default execution mode changed")
    return config, verified_bundle


def _execute(
    config_path: Path,
    config: dict[str, Any],
    verified_bundle: dict[str, Any],
) -> Path:
    started_at = _now_kst()
    output_directory = _output_directory(config)
    _assert_static_bundle_unchanged(config_path, config, verified_bundle)

    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import pandas as pd

    from p2_restore.features import build_training_features
    from p2_restore.normalized_curvature_residual import (
        align_exact_incumbent,
        build_normalized_curvature_design,
        evaluate_stage1_gate,
        fit_lgbm_seed_ensemble,
        make_stage1_split,
        metric_report,
        paired_day_bootstrap,
        resolve_observations_path,
        subset_design,
    )

    _assert_static_bundle_unchanged(config_path, config, verified_bundle)
    data_dir_text = os.environ.get(str(config["data_contract"]["source_environment_variable"]))
    if not data_dir_text:
        raise RuntimeError("P2_DATA_DIR is required only for --execute-stage1")
    observations_path = resolve_observations_path(
        Path(data_dir_text), str(config["data_contract"]["observations_csv_sha256"])
    )

    observations = pd.read_csv(observations_path)
    feature_table = build_training_features(observations)
    full_design = build_normalized_curvature_design(
        feature_table.frame,
        scale_floor_c=float(config["target_contract"]["scale_floor_c"]),
        salinity_scale_floor=float(config["feature_contract"]["salinity_scale_floor"]),
        depth_scale_floor_m=float(config["feature_contract"]["depth_scale_floor_m"]),
    )
    split_contract = config["stage1_split"]
    split = make_stage1_split(
        full_design.keys["time"],
        validation_start=str(split_contract["validation_start_inclusive"]),
        validation_end=str(split_contract["validation_end_exclusive"]),
        embargo_days=int(split_contract["embargo_days"]),
    )
    train_design = subset_design(full_design, split.train_mask)
    validation_design = subset_design(full_design, split.validation_mask)
    _assert_static_bundle_unchanged(config_path, config, verified_bundle)

    exact_contract = config["immutable_references"]["exact_same_season_incumbent_oof"]
    exact_oof_path = _repo_reference(config, "exact_same_season_incumbent_oof")
    exact_oof = pd.read_parquet(
        exact_oof_path,
        columns=["time", "layer", "block", "truth", "prediction"],
    )
    alignment = align_exact_incumbent(
        validation_design,
        exact_oof,
        block=str(exact_contract["block"]),
        expected_rows=int(split_contract["expected_validation_rows_after_exact_alignment"]),
        truth_column=str(exact_contract["truth_column"]),
        prediction_column=str(exact_contract["prediction_column"]),
    )
    del exact_oof

    exact_aggregate = config["immutable_references"]["exact_same_season_aggregate"]
    incumbent_metrics = metric_report(
        alignment.truth,
        alignment.incumbent_prediction,
        alignment.layer,
    )
    _assert_exact_metric_pin(incumbent_metrics, exact_aggregate)

    _assert_static_bundle_unchanged(config_path, config, verified_bundle)
    candidate_all_validation = fit_lgbm_seed_ensemble(
        train_design,
        validation_design,
        seeds=[int(value) for value in config["model"]["seeds"]],
        parameters=dict(config["model"]["parameters"]),
    )
    _assert_static_bundle_unchanged(config_path, config, verified_bundle)
    candidate_prediction = candidate_all_validation[alignment.candidate_positions]
    candidate_metrics = metric_report(
        alignment.truth,
        candidate_prediction,
        alignment.layer,
    )
    bootstrap_contract = config["metrics"]["paired_day_bootstrap"]
    bootstrap = paired_day_bootstrap(
        alignment.truth,
        alignment.incumbent_prediction,
        candidate_prediction,
        alignment.time,
        replicates=int(bootstrap_contract["replicates"]),
        seed=int(bootstrap_contract["seed"]),
        confidence=float(bootstrap_contract["confidence"]),
    )
    gate = evaluate_stage1_gate(
        incumbent_metrics,
        candidate_metrics,
        bootstrap,
        dict(config["stage1_gate"]),
    )

    result: dict[str, Any] = {
        "schema_version": "p2_normalized_curvature_residual_lgbm_stage1.result.v1",
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE_STAGE1_PASS" if gate["passed"] else "COMPLETE_STAGE1_FAIL",
        "started_at_kst": started_at,
        "completed_at_kst": _now_kst(),
        "family": config["family"],
        "population": exact_aggregate["population"],
        "rows": int(len(alignment.truth)),
        "fit_count": len(config["model"]["seeds"]),
        "train_rows": int(len(train_design.features)),
        "validation_rows_before_exact_alignment": int(len(validation_design.features)),
        "feature_count": int(len(train_design.features.columns)),
        "incumbent_metrics": incumbent_metrics,
        "candidate_metrics": candidate_metrics,
        "paired_day_bootstrap": bootstrap,
        "stage1_gate": gate,
        "stage2_executed": False,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
        "submission_files_generated": 0,
        "uploads": 0,
        "row_predictions_written": False,
        "csv_files_written": False,
        "seal_limitations": list(config["known_seal_limitations"]),
    }

    _assert_static_bundle_unchanged(config_path, config, verified_bundle)
    git_state = _git_state()
    staging_directory = output_directory.parent / (
        f".{output_directory.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if not staging_directory.resolve().is_relative_to((PROJECT_ROOT / "artifacts").resolve()):
        raise RuntimeError("NCR_LGBM staging path escaped artifacts")
    staging_directory.mkdir(parents=False, exist_ok=False)
    try:
        result_path = staging_directory / "result.json"
        _write_json_new(result_path, result)
        _assert_static_bundle_unchanged(config_path, config, verified_bundle)
        manifest: dict[str, Any] = {
            "schema_version": "p2_normalized_curvature_residual_lgbm_stage1.manifest.v1",
            "experiment_id": config["experiment_id"],
            "created_at_kst": _now_kst(),
            "status": result["status"],
            "append_only": True,
            "aggregate_only": True,
            "transactional_publish": {
                "final_directory_never_contains_one_of_two_required_files": True,
                "same_filesystem_staging_then_directory_rename": True,
                "hard_crash_may_leave_orphan_staging_directory": True,
            },
            "config": verified_bundle["config"],
            "implementation_pins": verified_bundle["implementation_pins"],
            "source": {
                "observations.csv": {
                    "bytes": observations_path.stat().st_size,
                    "sha256": _sha256(observations_path),
                }
            },
            "immutable_references": verified_bundle["immutable_references"],
            "result": {
                "path": "result.json",
                "bytes": result_path.stat().st_size,
                "sha256": _sha256(result_path),
            },
            "environment": {
                "python_full": sys.version,
                "platform": platform.platform(),
                "pinned_runtime_versions": verified_bundle["runtime"],
                "git": git_state,
            },
            "seal_rechecks": {
                "before_numerical_import": True,
                "after_numerical_import": True,
                "before_model_fit": True,
                "after_model_fit": True,
                "before_publish": True,
                "os_atomic_with_import": False,
            },
            "known_seal_limitations": list(config["known_seal_limitations"]),
            "external_actions": {
                "official_test_reads": 0,
                "sample_submission_reads": 0,
                "submission_candidate_reads": 0,
                "submission_files_generated": 0,
                "uploads": 0,
            },
        }
        _write_json_new(staging_directory / "manifest.json", manifest)
        _assert_static_bundle_unchanged(config_path, config, verified_bundle)
        staging_directory.replace(output_directory)
    except BaseException:
        if staging_directory.exists():
            resolved_staging = staging_directory.resolve()
            artifact_root = (PROJECT_ROOT / "artifacts").resolve()
            expected_prefix = f".{output_directory.name}.staging-"
            if (
                resolved_staging.is_relative_to(artifact_root)
                and resolved_staging.name.startswith(expected_prefix)
            ):
                shutil.rmtree(resolved_staging)
        raise
    return output_directory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Sealed NCR_LGBM Stage 1 preregistration (hash must match).",
    )
    parser.add_argument(
        "--execute-stage1",
        action="store_true",
        help="Explicitly authorize the three-fit historical Stage 1 computation.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config, verified_bundle = _preflight(config_path)
    if not args.execute_stage1:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_PASS_NOT_EXECUTED",
                    "experiment_id": config["experiment_id"],
                    "config_sha256": _sha256(config_path),
                    "verified_implementation_count": len(
                        verified_bundle["implementation_pins"]
                    ),
                    "verified_reference_count": len(
                        verified_bundle["immutable_references"]
                    ),
                    "verified_runtime": verified_bundle["runtime"],
                    "maximum_new_model_fits": 3,
                    "numerical_execution": False,
                    "official_or_submission_reads": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    output_directory = _execute(config_path, config, verified_bundle)
    print(output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
