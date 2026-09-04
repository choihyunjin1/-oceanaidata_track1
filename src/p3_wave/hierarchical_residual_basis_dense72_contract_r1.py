"""Fail-closed contract for the append-only P3 Gen5r2 dense72 correction."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from ocean_goal.meaningful_score_ledger_v5 import validate_ledger
from p3_wave.corrected_repeated_forward import build_corrected_repeated_forward_folds
from p3_wave.dense72_targets_r1 import Dense72TargetAccessor, sha256_file
from p3_wave.hierarchical_residual_basis import (
    HierarchicalResidualBasisConfig,
    HierarchicalResidualBasisForecaster,
    prepare_hierarchical_context,
)
from p3_wave.meaningful_learning_curve import PREFIX_FRACTIONS, chronological_prefix_ids
from p3_wave.models import compact_feature_columns
from p3_wave.revin_patch import assign_storm_episodes_from_wave, validate_raw_context

CONFIG_RELATIVE = "configs/experiments/p3_hierarchical_residual_basis_dense72_r1.json"
CONFIG_SHA256 = "7e085bdac8c5460205ce7909090d59ccec3e876ad9bc50473af343bbe363c91c"
STAGE = "P3_HIERARCHICAL_RESIDUAL_BASIS_GEN5R2_DENSE72_R1"
COMPARISON_MODE = "STRUCTURE_MATCHED_FRESH_REFIT_PENDING_OFFICIAL_PAIRED_AB"
FOLD_ORDER = ("2024_h2_storm", "winter_transition", "2025_h1")
EXPECTED_IMPLEMENTATION_ROLES = frozenset(
    {
        "CONFIG",
        "TARGET_ACCESSOR",
        "MODEL",
        "GUARD",
        "ENGINE",
        "RUNNER",
        "MODEL_TESTS",
        "RUNNER_TESTS",
        "P3_PACKAGE",
        "P3_DATA",
        "P3_FEATURES",
        "P3_CAUSAL_FORCING_ANALOG",
        "P3_EPISODE_DISTINCT_ANALOG",
        "P3_CAUSAL_FORCING_SEQUENCE",
        "P3_REVIN_PATCH",
        "P3_MODELS",
        "P3_CORRECTED_SPLIT",
        "P3_LOSS_ROUTER",
        "P3_MEANINGFUL_CURVE",
        "P3_PERSISTENCE_SHRINK",
        "P3_ONE_SHOT_GUARD",
        "P3_VALIDATION",
        "FAILED_V1_MODEL",
        "OCEAN_GOAL_PACKAGE",
        "OCEAN_GOAL_V2",
        "OCEAN_GOAL_V3",
        "OCEAN_LEDGER_V5",
        "GOAL_V2_CONFIG",
        "GOAL_V3_CONFIG",
        "LEDGER_V5_CONFIG",
        "FAILED_V1_CONFIG",
        "FAILED_V1_RUNNER",
        "FAILED_V1_MODEL_TESTS",
        "FAILED_V1_RUNNER_TESTS",
        "ENVIRONMENT_LOCK",
    }
)
QA_BOUND_DYNAMIC_ROLES = frozenset({"CONFIG", "GUARD", "RUNNER"})


class Dense72ContractError(RuntimeError):
    """Raised when a frozen correction contract drifts."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def strict_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Dense72ContractError("JSON document must be an object")
    return value


def _pin(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_pin(path: Path, expected: dict[str, Any], *, label: str) -> dict[str, Any]:
    observed = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if observed != {"sha256": expected["sha256"], "bytes": expected["bytes"]}:
        raise Dense72ContractError(f"{label} pin changed")
    return observed


def _origin_url(root: Path) -> str:
    completed = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _canonical_workspace(root: Path, config: dict[str, Any] | None = None) -> Path:
    workspace = root.resolve(strict=True)
    if Path.cwd().resolve(strict=True) != workspace:
        raise PermissionError("Gen5r2 must run from the supplied canonical workspace root")
    if not (workspace / ".git").is_dir():
        raise PermissionError("canonical workspace lacks its Git boundary")
    expected = config or _read_config_without_workspace_check(workspace)
    identity = expected["canonical_workspace_identity"]
    stat = workspace.stat()
    if (
        int(stat.st_dev) != int(identity["root_st_dev"])
        or int(stat.st_ino) != int(identity["root_st_ino"])
    ):
        raise PermissionError("canonical workspace filesystem identity differs")
    if sha256_file(workspace / ".git/config") != identity["git_config_sha256"]:
        raise PermissionError("canonical workspace Git config pin differs")
    if _origin_url(workspace) != identity["origin_url"]:
        raise PermissionError("canonical workspace origin identity differs")
    return workspace


def workspace_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise PermissionError("Gen5r2 path must be non-traversing and workspace-relative")
    workspace = root.resolve(strict=True)
    path = (workspace / candidate).resolve(strict=must_exist)
    if path != workspace and workspace not in path.parents:
        raise PermissionError("Gen5r2 path escaped the canonical workspace")
    return path


def _read_config_without_workspace_check(root: Path) -> dict[str, Any]:
    path = workspace_path(root, CONFIG_RELATIVE)
    if sha256_file(path) != CONFIG_SHA256:
        raise Dense72ContractError("canonical Gen5r2 config byte SHA differs")
    return strict_json_object(path)


def _validate_config(config: dict[str, Any]) -> None:
    required_top = {
        "schema_version",
        "experiment_id",
        "created_at_kst",
        "status",
        "problem",
        "hypothesis",
        "comparison_mode",
        "exact_official_incumbent_comparison",
        "local_numeric_curve_qualification_allowed",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
        "canonical_paths",
        "canonical_workspace_identity",
        "runtime_environment",
        "failed_v1_lineage",
        "implementation_roles",
        "frozen_transitive_sha256",
        "input_pins",
        "central_ledger_anchor",
        "gen4_failure_diagnosis",
        "validation",
        "compact_preflight",
        "dense72_supervision",
        "model",
        "training",
        "postprocess",
        "comparator",
        "gate",
        "execution_policy",
        "qa_receipt_contract",
        "authorization_contract",
        "resource_estimate",
    }
    if set(config) != required_top:
        raise Dense72ContractError("Gen5r2 config top-level fields changed")
    if (
        config["schema_version"]
        != "p3_hierarchical_residual_basis.gen5r2_dense72.r1"
        or config["experiment_id"] != "p3_hierarchical_residual_basis_dense72_r1"
        or config["problem"] != "P3"
        or config["comparison_mode"] != COMPARISON_MODE
        or config["exact_official_incumbent_comparison"] is not False
        or config["local_numeric_curve_qualification_allowed"] is not True
        or config["official_promotion_allowed"] is not False
        or config["candidate_or_test_prediction_allowed"] is not False
        or config["upload_allowed"] is not False
    ):
        raise Dense72ContractError("Gen5r2 identity or comparison semantics changed")
    if set(config["implementation_roles"]) != EXPECTED_IMPLEMENTATION_ROLES:
        raise Dense72ContractError("full transitive implementation role set changed")
    expected_frozen_paths = {
        relative
        for role, relative in config["implementation_roles"].items()
        if role not in QA_BOUND_DYNAMIC_ROLES
    }
    frozen = config["frozen_transitive_sha256"]
    if set(frozen) != expected_frozen_paths or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in frozen.values()
    ):
        raise Dense72ContractError("frozen transitive SHA surface changed")
    runtime = config["runtime_environment"]
    if runtime != {
        "python_implementation": "CPython",
        "python_version": "3.12.10",
        "requirements_lock_sha256": (
            "a1aa280b03af38c1920a8a171da1fdb568b8310edf3bfea66fd8ab2f71c470ba"
        ),
        "requirements_lock_line_count": 83,
        "noneditable_freeze_sha256": (
            "cfe46bea056591d9f551f22828056857875ba1ae9d4ce31c20e9a29f3eff8acf"
        ),
        "allowed_editable_distribution": (
            "-e git+https://github.com/choihyunjin1/-oceanaidata_track1.git@"
            "eb29ea194aabce58f1c024d106da70ce5a075d45#egg=p1_qc"
        ),
        "installed_noneditable_distributions_must_exactly_match_lock": True,
    }:
        raise Dense72ContractError("exact runtime environment contract changed")
    if config["validation"]["fold_execution_order"] != list(FOLD_ORDER):
        raise Dense72ContractError("fold-major execution order changed")
    if config["validation"]["training_prefix_fractions"] != list(PREFIX_FRACTIONS):
        raise Dense72ContractError("learning-curve prefix fractions changed")
    dense = config["dense72_supervision"]
    if (
        dense["steps"] != 72
        or dense["complete_cases"] != 23_527
        or dense["incomplete_cases"] != 833
        or dense["missing_scalars"] != 1_505
        or dense["official_six_missing_scalars"] != 0
        or dense["validation_scalar_decode_before_fold_commit"] != 0
        or dense["dense_target_array_materialized_for_all_cases"] is not False
    ):
        raise Dense72ContractError("dense72 selective-supervision contract changed")
    compact = config["compact_preflight"]
    if (
        compact["source_rows"] != 24_360
        or compact["source_columns"] != 1_277
        or compact["selected_rows"] != 24_360
        or compact["selected_columns"] != 591
        or compact["load_and_validate_before_attempt_lock"] is not True
        or compact["target_columns_selected"] != 0
    ):
        raise Dense72ContractError("compact preflight contract changed")
    model = config["model"]
    if (
        model["optimizer_target_surface"] != "all_available_dense72_train_only_steps"
        or model["expected_actual_fit_cells"] != 45
        or model["expected_optimizer_steps"] != 10_260
        or model["hyperparameter_search"] is not False
    ):
        raise Dense72ContractError("dense72 model contract changed")
    comparator = config["comparator"]
    if (
        comparator["reference_seed_full_prediction_exact_to_historical_frozen_oof"]
        is not False
        or comparator["comparison_label"] != COMPARISON_MODE
        or comparator["may_support_local_numeric_qualification"] is not True
        or comparator["may_support_official_promotion_without_paired_ab"] is not False
    ):
        raise Dense72ContractError("structure-matched comparator semantics changed")
    policy = config["execution_policy"]
    required_true = (
        "check_only_is_default",
        "static_preflight_before_qa_authorization_or_lock",
        "compact_and_dense72_availability_validated_before_lock",
        "actual_curve_requires_independent_static_qa",
        "actual_curve_requires_separate_append_only_authorization",
        "qa_and_authorization_verified_before_attempt_lock",
        "engine_imported_after_attempt_lock",
        "canonical_capability_required_for_engine_and_curve",
        "fold_major_blind_commitment_before_target_release",
        "all_45_blind_predictions_committed_before_truth_attachment",
        "output_and_commitments_use_o_excl_semantics",
        "post_lock_exception_writes_aggregate_only_o_excl_failure_receipt",
    )
    required_false = (
        "rerun_allowed",
        "resume_allowed",
        "candidate_or_test_prediction_allowed",
        "full_fit_allowed_in_current_stage",
        "frozen_or_submission_mutation_allowed",
        "registry_append_allowed",
        "automatic_upload_allowed",
    )
    if not all(policy.get(name) is True for name in required_true) or not all(
        policy.get(name) is False for name in required_false
    ):
        raise Dense72ContractError("Gen5r2 execution policy changed")


def load_canonical_config(
    root: Path,
    requested_config: Path | None = None,
    *,
    supplied_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provisional = _read_config_without_workspace_check(root.resolve(strict=True))
    workspace = _canonical_workspace(root, provisional)
    config = _read_config_without_workspace_check(workspace)
    _validate_config(config)
    for relative, expected_sha256 in config["frozen_transitive_sha256"].items():
        if sha256_file(workspace_path(workspace, relative)) != expected_sha256:
            raise PermissionError(f"frozen Gen5r2 transitive dependency changed: {relative}")
    if requested_config is not None:
        if requested_config.resolve(strict=True) != workspace_path(workspace, CONFIG_RELATIVE):
            raise PermissionError("noncanonical Gen5r2 config was requested")
    if supplied_config is not None and canonical_json_bytes(supplied_config) != canonical_json_bytes(
        config
    ):
        raise PermissionError("supplied Gen5r2 config differs from canonical bytes")
    return config


def implementation_pins(root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    canonical = config or load_canonical_config(root)
    workspace = _canonical_workspace(root, canonical)
    return {
        role: _pin(workspace_path(workspace, relative), workspace)
        for role, relative in canonical["implementation_roles"].items()
    }


def verify_runtime_environment(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Require the executing interpreter to match the sealed lock exactly."""

    workspace = _canonical_workspace(root, config)
    contract = config["runtime_environment"]
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.implementation.name != "cpython" or version != contract["python_version"]:
        raise PermissionError("Gen5r2 Python runtime identity differs")
    lock_path = workspace_path(workspace, config["implementation_roles"]["ENVIRONMENT_LOCK"])
    if sha256_file(lock_path) != contract["requirements_lock_sha256"]:
        raise PermissionError("Gen5r2 requirements lock SHA differs")
    expected = [
        line.strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(expected) != contract["requirements_lock_line_count"] or any(
        line.startswith("-e ") for line in expected
    ):
        raise PermissionError("Gen5r2 requirements lock line surface differs")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    installed = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    editable = [line for line in installed if line.startswith("-e ")]
    noneditable = [line for line in installed if not line.startswith("-e ")]
    if editable != [contract["allowed_editable_distribution"]] or noneditable != expected:
        raise PermissionError("Gen5r2 installed dependency closure differs from lock")
    freeze_sha = hashlib.sha256(canonical_json_bytes(noneditable)).hexdigest()
    if freeze_sha != contract["noneditable_freeze_sha256"]:
        raise PermissionError("Gen5r2 installed dependency closure SHA differs")
    return {
        "python_implementation": contract["python_implementation"],
        "python_version": version,
        "requirements_lock_sha256": contract["requirements_lock_sha256"],
        "requirements_lock_line_count": len(expected),
        "noneditable_freeze_sha256": freeze_sha,
        "editable_distribution_count": len(editable),
        "exact_match": True,
    }


def stage_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    workspace = _canonical_workspace(root, config)
    return {
        key: workspace_path(workspace, config["canonical_paths"][key], must_exist=False)
        for key in (
            "output",
            "control",
            "pre_execution_qa",
            "authorization",
            "attempt_lock",
            "run_failure_receipt",
        )
    }


def _input_paths(root: Path, data_dir: Path, config: dict[str, Any]) -> dict[str, Path]:
    canonical = config["canonical_paths"]
    compact = workspace_path(root, canonical["compact_cache"])
    sequence = workspace_path(root, canonical["sequence_cache"])
    gen1 = workspace_path(root, canonical["gen1_artifact"])
    gen4 = workspace_path(root, canonical["gen4_artifact"])
    return {
        "source/train_wave.csv": data_dir / "train_wave.csv",
        "source/train_atmos.csv": data_dir / "train_atmos.csv",
        "source/test_context.parquet": data_dir / "test_context.parquet",
        "source/test_index.csv": data_dir / "test_index.csv",
        "source/sample_submission.csv": data_dir / "sample_submission.csv",
        "source/baseline_persistence.csv": data_dir / "baseline_persistence.csv",
        "source/README.md": data_dir / "README.md",
        "compact_cache/manifest.json": compact / "manifest.json",
        "compact_cache/train_features.parquet": compact / "train_features.parquet",
        "compact_cache/train_anchors.parquet": compact / "train_anchors.parquet",
        "sequence_cache/manifest.json": sequence / "manifest.json",
        "sequence_cache/train_values.npy": sequence / "train_values.npy",
        "sequence_cache/train_station.npy": sequence / "train_station.npy",
        "gen1/metrics.json": gen1 / "metrics.json",
        "gen1/learning_curve_oof.parquet": gen1 / "oof/learning_curve_oof.parquet",
        "gen1/manifest.json": gen1 / "manifest.json",
        "gen1/learning_curve_evidence.json": gen1 / "learning_curve_evidence.json",
        "gen1/independent_qa.json": root
        / "artifacts/p3_meaningful_learning_curve_20260823_v1_QA/independent_aggregate_audit.json",
        "gen4/metrics.json": gen4 / "metrics.json",
        "gen4/learning_curve_evidence.json": gen4 / "learning_curve_evidence.json",
        "gen4/manifest.json": gen4 / "manifest.json",
        "gen4/validation_keys.parquet": gen4 / "validation_keys.parquet",
        "v5/registry.jsonl": workspace_path(root, canonical["v5_ledger"]),
        "frozen/current_submission.csv": root
        / "submissions/p3_long_persistence_shrink/submission.csv",
        "frozen/current_manifest.json": root
        / "submissions/p3_long_persistence_shrink/manifest.json",
        "current/ready_submission.csv": root / "output/2026-08-20/ready/P3_submission.csv",
    }


def verify_input_pins(
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    source = data_dir.resolve(strict=True)
    if source == root.resolve(strict=True) or root.resolve(strict=True) in source.parents:
        raise PermissionError("P3 source data must remain outside the repository workspace")
    paths = _input_paths(root, source, config)
    if set(paths) != set(config["input_pins"]):
        raise Dense72ContractError("input pin key surface changed")
    observed: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"pinned P3 input is missing: {name}")
        record = config["input_pins"][name]
        observed[name] = _verify_pin(path, record, label=name)
    return paths, observed


def _verify_failed_v1(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    lineage = config["failed_v1_lineage"]
    verified: dict[str, Any] = {}
    for name, expected in lineage["files"].items():
        path = workspace_path(root, expected["path"])
        observed = _pin(path, root)
        if observed != expected:
            raise Dense72ContractError(f"failed v1 preservation pin changed: {name}")
        verified[name] = observed
    old_output = root / "artifacts/p3_hierarchical_residual_basis_20260823_v1"
    old_lock = root / "artifacts/p3_hierarchical_residual_basis_20260823_v1.ATTEMPT_LOCK.json"
    old_claim = root / "artifacts/p3_hierarchical_residual_basis_20260823_v1.EXECUTION_CLAIM.json"
    if old_output.exists() or old_lock.exists() or old_claim.exists():
        raise Dense72ContractError("failed static v1 unexpectedly acquired execution state")
    if any(lineage[key] != 0 for key in ("fit_count", "prediction_count", "score_count")):
        raise Dense72ContractError("failed v1 no-fit lineage changed")
    return verified


def _sha_array(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _id_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _build_preflight(
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    workspace = _canonical_workspace(root, config)
    paths = stage_paths(workspace, config)
    if paths["output"].exists() or paths["attempt_lock"].exists() or paths[
        "run_failure_receipt"
    ].exists():
        raise FileExistsError("Gen5r2 append-only curve state is already consumed")
    runtime_environment = verify_runtime_environment(workspace, config)
    input_paths, input_snapshot = verify_input_pins(workspace, data_dir, config)
    failed_v1 = _verify_failed_v1(workspace, config)

    records = validate_ledger(workspace, input_paths["v5/registry.jsonl"])
    ledger = config["central_ledger_anchor"]
    if (
        len(records) != ledger["event_count"]
        or records[-1].get("event_sha256") != ledger["head_event_sha256"]
        or any(record.get("payload", {}).get("upload_performed") is not False for record in records)
    ):
        raise Dense72ContractError("v5 ledger anchor or no-upload lineage changed")

    anchors = pd.read_parquet(
        input_paths["compact_cache/train_anchors.parquet"],
        columns=["anchor_id", "station", "anchor_time", "current_hs"],
    )
    features = pd.read_parquet(input_paths["compact_cache/train_features.parquet"])
    if anchors.shape != (24_360, 4) or features.shape != (24_360, 1_277):
        raise ValueError("anchor or compact feature cache shape differs")
    expected_ids = np.arange(24_360, dtype=np.int64)
    if not np.array_equal(anchors["anchor_id"].to_numpy(np.int64), expected_ids):
        raise ValueError("anchor IDs differ from exact sequence row identity")
    if not np.array_equal(features["anchor_id"].to_numpy(np.int64), expected_ids):
        raise ValueError("feature IDs differ from exact sequence row identity")
    if not anchors["station"].astype(str).equals(features["station"].astype(str)):
        raise ValueError("anchor and feature station identities differ")
    feature_columns = compact_feature_columns(list(features.columns))
    if (
        len(feature_columns) != 591
        or len(set(feature_columns)) != 591
        or any(str(name).startswith("target_") for name in feature_columns)
    ):
        raise ValueError("compact 591-column input-only surface differs")
    compact = features.loc[:, feature_columns].to_numpy(np.float32)
    if compact.shape != (24_360, 591):
        raise ValueError("loaded compact matrix differs from 24360x591")
    feature_names_sha = hashlib.sha256(
        canonical_json_bytes([str(value) for value in feature_columns])
    ).hexdigest()
    compact_sha = _sha_array(compact, dtype="<f4")
    del features

    raw = np.load(input_paths["sequence_cache/train_values.npy"], mmap_mode="r")
    station = np.load(input_paths["sequence_cache/train_station.npy"], mmap_mode="r")
    if raw.shape != (24_360, 289, 10) or raw.dtype != np.float32:
        raise ValueError("raw sequence cache differs from 24360x289x10 float32")
    if station.shape != (24_360,) or station.dtype != np.int64:
        raise ValueError("station sequence cache differs from 24360 int64")
    validate_raw_context(torch.from_numpy(np.array(raw[:8], copy=True)))
    encoded = anchors["station"].map({"G-ORS": 0, "I-ORS": 1, "S-ORS": 2})
    if encoded.isna().any() or not np.array_equal(station, encoded.to_numpy(np.int64)):
        raise ValueError("station sequence codes differ from anchor identities")

    wave = pd.read_csv(
        input_paths["source/train_wave.csv"],
        usecols=["station", "time", "hs"],
    )
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    anchors_with_episode = assign_storm_episodes_from_wave(anchors, wave)
    del wave
    folds, selected, split_audit = build_corrected_repeated_forward_folds(
        anchors_with_episode,
        windows=config["validation"]["windows"],
        gap_hours=config["validation"]["gap_hours"],
        footprint_hours=config["validation"]["footprint_hours"],
    )
    if tuple(fold.name for fold in folds) != FOLD_ORDER:
        raise ValueError("corrected fold order differs")
    if len(selected) != 181 or split_audit["validation_row_count"] != 1_086:
        raise ValueError("corrected validation surface differs")

    prefix_ids: dict[float, dict[str, np.ndarray]] = {}
    prefix_audit: dict[str, Any] = {}
    lookup = anchors_with_episode.set_index("anchor_id")
    for fraction in PREFIX_FRACTIONS:
        prefix_ids[fraction] = {}
        tag = f"{int(round(fraction * 100)):03d}"
        prefix_audit[tag] = {}
        for fold in folds:
            ids = chronological_prefix_ids(anchors_with_episode, fold.train_ids, fraction)
            prefix_ids[fraction][fold.name] = ids
            times = pd.to_datetime(lookup.loc[ids, "anchor_time"], utc=True)
            gap = float(
                (pd.Timestamp(fold.validation_start) - times.max()).total_seconds() / 3600.0
            )
            prefix_audit[tag][fold.name] = {
                "count": int(len(ids)),
                "full_count": int(len(fold.train_ids)),
                "id_sha256_little_endian_int64": _id_sha(ids),
                "nested_subset_of_safe_outer_train": bool(np.isin(ids, fold.train_ids).all()),
                "maximum_anchor_before_validation_start_hours": gap,
            }
    leakage_checks = {
        "station_global_validation_gap_at_least_78h": all(
            value >= 78.0
            for value in split_audit["station_global_minimum_gap_hours"].values()
        ),
        "validation_station_episode_reuse_zero": split_audit[
            "repeated_station_episode_count"
        ]
        == 0,
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
        raise AssertionError("corrected split leakage checks failed")

    validation_groups = {fold.name: fold.validation_ids for fold in folds}
    accessor = Dense72TargetAccessor(
        input_paths["source/train_wave.csv"],
        anchors.loc[:, ["anchor_id", "station", "anchor_time", "current_hs"]],
        validation_groups=validation_groups,
        expected_source_sha256=config["input_pins"]["source/train_wave.csv"]["sha256"],
        expected_source_bytes=config["input_pins"]["source/train_wave.csv"]["bytes"],
    )
    availability = accessor.availability_audit().as_dict()
    if availability["scalar_decodes"] != 0:
        raise PermissionError("dense72 preflight decoded target scalars")

    gen1_oof = pd.read_parquet(
        input_paths["gen1/learning_curve_oof.parquet"],
        columns=[
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "current_hs",
            "persistence",
            "incumbent_prediction",
            "prefix_fraction",
        ],
    )
    keys = ["prefix_fraction", "fold", "anchor_id", "station", "lead_h"]
    if len(gen1_oof) != 5 * 1_086 or gen1_oof.duplicated(keys).any():
        raise ValueError("sealed Gen1 comparator key surface differs")
    gen1_protocol = strict_json_object(
        input_paths["gen1/learning_curve_evidence.json"]
    )["curve_protocol"]
    if gen1_protocol["incumbent_reference_seed_full_prediction_exact_to_frozen_oof"] is not False:
        raise ValueError("historical frozen-reference fact unexpectedly changed")
    gen1_metrics = strict_json_object(input_paths["gen1/metrics.json"])

    gen4_metrics = strict_json_object(input_paths["gen4/metrics.json"])
    diagnosis = config["gen4_failure_diagnosis"]
    observed_deltas = [
        float(gen4_metrics["points"][str(fraction)]["delta_candidate_minus_incumbent_m"])
        for fraction in PREFIX_FRACTIONS
    ]
    if (
        observed_deltas != diagnosis["prefix_deltas_candidate_minus_incumbent_m"]
        or float(gen4_metrics["points"]["1.0"]["delta_candidate_minus_incumbent_m"])
        != diagnosis["full_delta_candidate_minus_incumbent_m"]
        or list(gen4_metrics["points"]["1.0"]["delta_ci90_m"])
        != diagnosis["full_ci90_m"]
    ):
        raise ValueError("sealed Gen4 aggregate failure diagnosis differs")

    probe = prepare_hierarchical_context(
        torch.from_numpy(np.array(raw[:2], dtype=np.float32, copy=True))
    )
    if tuple(probe.values.shape) != (2, 144, 24):
        raise ValueError("hierarchical context probe shape differs")
    model_config = HierarchicalResidualBasisConfig(
        static_feature_count=591,
        hidden_width=192,
        conditioning_width=128,
        dropout=0.1,
        context_steps=144,
        input_channels=24,
        forecast_steps=72,
        pooling_factors=(12, 4, 1),
        forecast_knots=(6, 18, 72),
        blocks_per_stack=2,
    )
    parameter_count = HierarchicalResidualBasisForecaster(model_config).trainable_parameter_count
    if parameter_count != config["model"]["expected_trainable_parameter_count"]:
        raise ValueError("frozen trainable parameter count differs")
    expected_steps = sum(
        math.ceil(len(prefix_ids[fraction][fold.name]) / config["training"]["batch_size"])
        * config["training"]["epochs"]
        for fold in folds
        for fraction in PREFIX_FRACTIONS
        for _seed in config["validation"]["seed_replicates"]
    )
    if expected_steps != config["model"]["expected_optimizer_steps"]:
        raise ValueError("fixed optimizer-step accounting differs")

    summary = {
        "schema_version": "p3_hierarchical_residual_basis.gen5r2_dense72.preflight.r1",
        "status": "PASS_STATIC_IMPLEMENTATION_ONLY_NO_FIT_NO_LOCK",
        "problem": "P3",
        "comparison_mode": COMPARISON_MODE,
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "runtime_environment": runtime_environment,
        "input_pins": input_snapshot,
        "failed_v1_preserved": failed_v1,
        "compact": {
            "shape": [24_360, 591],
            "dtype": "float32",
            "feature_names_sha256": feature_names_sha,
            "matrix_sha256": compact_sha,
            "target_columns_selected": 0,
            "loaded_before_attempt_lock": True,
        },
        "dense72_availability": availability,
        "validation": {
            "cases": int(len(selected)),
            "rows": int(split_audit["validation_row_count"]),
            "fold_order": list(FOLD_ORDER),
            "selected_anchor_ids_sha256": _id_sha(
                selected.sort_values("anchor_id")["anchor_id"].to_numpy(np.int64)
            ),
            "split_audit": split_audit,
            "prefix_audit": prefix_audit,
            "leakage_checks": leakage_checks,
        },
        "model": {
            "context_probe_shape": list(probe.values.shape),
            "trainable_parameter_count": int(parameter_count),
            "actual_fit_cells": 45,
            "optimizer_steps": int(expected_steps),
            "optimizer_target_surface": "all_available_dense72_train_only_steps",
        },
        "comparator": {
            "rows_without_truth": int(len(gen1_oof)),
            "fresh_refit_each_prefix": True,
            "reference_seed_full_prediction_exact_to_historical_frozen_oof": False,
            "local_numeric_qualification_allowed": True,
            "official_promotion_requires_future_paired_ab": True,
        },
        "preflight_dense_target_scalar_decodes": 0,
        "test_value_decodes": 0,
        "fits": 0,
        "predictions": 0,
        "scores": 0,
        "test_predictions": 0,
        "uploads": 0,
    }
    summary_sha = hashlib.sha256(canonical_json_bytes(summary)).hexdigest()
    frozen_implementation_pins = implementation_pins(workspace, config)
    return {
        "summary": summary,
        "summary_sha256": summary_sha,
        "implementation_pins": frozen_implementation_pins,
        "input_paths": input_paths,
        "input_snapshot": input_snapshot,
        "anchors": anchors_with_episode,
        "anchor_path": input_paths["compact_cache/train_anchors.parquet"],
        "raw": raw,
        "station": station,
        "compact": compact,
        "feature_columns": tuple(feature_columns),
        "folds": folds,
        "selected": selected,
        "split_audit": split_audit,
        "prefix_ids": prefix_ids,
        "prefix_audit": prefix_audit,
        "leakage_checks": leakage_checks,
        "target_accessor": accessor,
        "gen1_metrics": gen1_metrics,
    }


def prepare_execution_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
    supplied_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_canonical_config(
        root,
        requested_config,
        supplied_config=supplied_config,
    )
    preflight = _build_preflight(root.resolve(strict=True), data_dir.resolve(strict=True), config)
    return config, preflight


def static_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
) -> dict[str, Any]:
    config, preflight = prepare_execution_preflight(
        root,
        data_dir,
        requested_config=requested_config,
    )
    paths = stage_paths(root, config)
    return {
        **preflight["summary"],
        "config": _pin(workspace_path(root, CONFIG_RELATIVE), root.resolve(strict=True)),
        "implementation_pins": preflight["implementation_pins"],
        "static_preflight_sha256": preflight["summary_sha256"],
        "control_state": {name: path.exists() for name, path in paths.items()},
        "files_written": 0,
        "attempt_locks_created": 0,
    }


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(canonical_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_pre_execution_qa(
    root: Path,
    config: dict[str, Any],
    *,
    static_preflight_sha256: str,
) -> tuple[dict[str, Any], str]:
    path = stage_paths(root, config)["pre_execution_qa"]
    if not path.is_file():
        raise PermissionError("independent Gen5r2 pre-execution QA receipt is missing")
    receipt = strict_json_object(path)
    expected_keys = {
        "schema_version",
        "created_at_kst",
        "reviewer",
        "decision",
        "p0_count",
        "p1_count",
        "config",
        "implementation_pins",
        "static_preflight_sha256",
        "failed_v1_preserved",
        "dense72_contract_verified",
        "notes",
    }
    contract = config["qa_receipt_contract"]
    checks = {
        "keys": set(receipt) == expected_keys,
        "schema": receipt.get("schema_version") == contract["schema_version"],
        "reviewer": bool(receipt.get("reviewer")),
        "decision": receipt.get("decision") == contract["decision"],
        "p0": receipt.get("p0_count") == 0,
        "p1": receipt.get("p1_count") == 0,
        "config": receipt.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation": receipt.get("implementation_pins")
        == implementation_pins(root, config),
        "preflight": receipt.get("static_preflight_sha256") == static_preflight_sha256,
        "v1": receipt.get("failed_v1_preserved") is True,
        "dense72": receipt.get("dense72_contract_verified") is True,
        "notes": isinstance(receipt.get("notes"), list) and bool(receipt["notes"]),
    }
    if not all(checks.values()):
        raise PermissionError(
            "Gen5r2 QA receipt failed: "
            f"{sorted(name for name, value in checks.items() if not value)}"
        )
    return receipt, sha256_file(path)


def verify_execution_authorization(
    root: Path,
    config: dict[str, Any],
    *,
    qa_sha256: str,
) -> tuple[dict[str, Any], str]:
    paths = stage_paths(root, config)
    if paths["output"].exists() or paths["attempt_lock"].exists() or paths[
        "run_failure_receipt"
    ].exists():
        raise FileExistsError("Gen5r2 one-shot state is already consumed")
    if not paths["authorization"].is_file():
        raise PermissionError("separate Gen5r2 execution authorization is missing")
    authorization = strict_json_object(paths["authorization"])
    expected_keys = {
        "schema_version",
        "created_at_kst",
        "stage",
        "config",
        "authorization",
        "user_message_reference",
        "qa_receipt",
        "implementation_pins",
        "curve_execution_authorized",
        "full_fit_or_candidate_authorized",
        "test_prediction_authorized",
        "upload_authorized",
    }
    contract = config["authorization_contract"]
    checks = {
        "keys": set(authorization) == expected_keys,
        "schema": authorization.get("schema_version") == contract["schema_version"],
        "stage": authorization.get("stage") == STAGE,
        "config": authorization.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "phrase": authorization.get("authorization")
        == contract["authorization_phrase_prefix"] + CONFIG_SHA256,
        "user": bool(authorization.get("user_message_reference")),
        "qa": authorization.get("qa_receipt")
        == {"path": config["canonical_paths"]["pre_execution_qa"], "sha256": qa_sha256},
        "implementation": authorization.get("implementation_pins")
        == implementation_pins(root, config),
        "curve": authorization.get("curve_execution_authorized") is True,
        "full_fit": authorization.get("full_fit_or_candidate_authorized") is False,
        "test": authorization.get("test_prediction_authorized") is False,
        "upload": authorization.get("upload_authorized") is False,
    }
    if not all(checks.values()):
        raise PermissionError(
            "Gen5r2 authorization failed: "
            f"{sorted(name for name, value in checks.items() if not value)}"
        )
    return authorization, sha256_file(paths["authorization"])


@dataclass(frozen=True)
class ExecutionCapability:
    root_st_dev: int
    root_st_ino: int
    config_sha256: str
    static_preflight_sha256: str
    qa_sha256: str
    authorization_sha256: str
    nonce: str


_LIVE_CAPABILITY: ExecutionCapability | None = None


def issue_execution_capability(
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    *,
    qa_sha256: str,
    authorization_sha256: str,
) -> ExecutionCapability:
    global _LIVE_CAPABILITY
    workspace = _canonical_workspace(root, config)
    if _LIVE_CAPABILITY is not None:
        raise PermissionError("a canonical Gen5r2 execution capability is already live")
    if hashlib.sha256(canonical_json_bytes(preflight["summary"])).hexdigest() != preflight[
        "summary_sha256"
    ]:
        raise PermissionError("Gen5r2 preflight summary changed before capability issue")
    if implementation_pins(workspace, config) != preflight["implementation_pins"]:
        raise PermissionError("Gen5r2 implementation changed before capability issue")
    _qa, observed_qa_sha256 = verify_pre_execution_qa(
        workspace,
        config,
        static_preflight_sha256=preflight["summary_sha256"],
    )
    if observed_qa_sha256 != qa_sha256:
        raise PermissionError("Gen5r2 QA receipt SHA differs at capability issue")
    _authorization, observed_authorization_sha256 = verify_execution_authorization(
        workspace,
        config,
        qa_sha256=observed_qa_sha256,
    )
    if observed_authorization_sha256 != authorization_sha256:
        raise PermissionError("Gen5r2 authorization SHA differs at capability issue")
    nonce = hashlib.sha256(
        canonical_json_bytes(
            {
                "root_st_dev": workspace.stat().st_dev,
                "root_st_ino": workspace.stat().st_ino,
                "config": CONFIG_SHA256,
                "preflight": preflight["summary_sha256"],
                "qa": qa_sha256,
                "authorization": authorization_sha256,
                "pid": os.getpid(),
            }
        )
    ).hexdigest()
    capability = ExecutionCapability(
        root_st_dev=int(workspace.stat().st_dev),
        root_st_ino=int(workspace.stat().st_ino),
        config_sha256=CONFIG_SHA256,
        static_preflight_sha256=preflight["summary_sha256"],
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
        nonce=nonce,
    )
    _LIVE_CAPABILITY = capability
    return capability


def require_execution_capability(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    phase: str,
) -> ExecutionCapability:
    workspace = _canonical_workspace(root, config)
    if capability is not _LIVE_CAPABILITY or not isinstance(capability, ExecutionCapability):
        raise PermissionError("canonical live Gen5r2 execution capability is required")
    if phase not in {"consume_lock", "execute_stage", "run_curve"}:
        raise PermissionError("unknown Gen5r2 capability phase")
    if (
        capability.root_st_dev != int(workspace.stat().st_dev)
        or capability.root_st_ino != int(workspace.stat().st_ino)
        or capability.config_sha256 != CONFIG_SHA256
        or capability.static_preflight_sha256 != preflight["summary_sha256"]
    ):
        raise PermissionError("forged or stale Gen5r2 execution capability")
    if hashlib.sha256(canonical_json_bytes(preflight["summary"])).hexdigest() != preflight[
        "summary_sha256"
    ]:
        raise PermissionError("Gen5r2 preflight summary changed after capability issue")
    if implementation_pins(workspace, config) != preflight["implementation_pins"]:
        raise PermissionError("Gen5r2 implementation changed after capability issue")
    paths = stage_paths(workspace, config)
    if (
        not paths["pre_execution_qa"].is_file()
        or sha256_file(paths["pre_execution_qa"]) != capability.qa_sha256
        or not paths["authorization"].is_file()
        or sha256_file(paths["authorization"]) != capability.authorization_sha256
    ):
        raise PermissionError("Gen5r2 QA or authorization changed after capability issue")
    return capability


def revoke_execution_capability(capability: ExecutionCapability) -> None:
    global _LIVE_CAPABILITY
    if capability is not _LIVE_CAPABILITY:
        raise PermissionError("cannot revoke a noncanonical Gen5r2 capability")
    _LIVE_CAPABILITY = None


def _lock_payload(
    root: Path,
    config: dict[str, Any],
    *,
    capability: ExecutionCapability,
) -> dict[str, Any]:
    return {
        "schema_version": "p3_hierarchical_residual_basis.gen5r2_dense72.attempt_lock.r1",
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "stage": STAGE,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "static_preflight_sha256": capability.static_preflight_sha256,
        "qa_receipt_sha256": capability.qa_sha256,
        "authorization_sha256": capability.authorization_sha256,
        "implementation_pins": implementation_pins(root, config),
        "comparison_mode": COMPARISON_MODE,
        "exact_official_incumbent_comparison": False,
        "candidate_or_test_prediction_allowed": False,
        "full_fit_allowed": False,
        "rerun_allowed": False,
        "resume_allowed": False,
        "upload_allowed": False,
    }


def consume_attempt_lock(
    root: Path,
    config: dict[str, Any],
    *,
    capability: ExecutionCapability,
    preflight: dict[str, Any],
) -> Path:
    require_execution_capability(
        capability,
        root=root,
        config=config,
        preflight=preflight,
        phase="consume_lock",
    )
    paths = stage_paths(root, config)
    if paths["output"].exists() or paths["run_failure_receipt"].exists():
        raise FileExistsError("Gen5r2 output or failure receipt already exists")
    payload = _lock_payload(root, config, capability=capability)
    exclusive_json(paths["attempt_lock"], payload)
    return paths["attempt_lock"]


def verify_consumed_attempt_lock(
    root: Path,
    config: dict[str, Any],
    *,
    capability: ExecutionCapability,
) -> dict[str, Any]:
    path = stage_paths(root, config)["attempt_lock"]
    if not path.is_file():
        raise PermissionError("Gen5r2 attempt lock is missing")
    observed = strict_json_object(path)
    expected = _lock_payload(root, config, capability=capability)
    expected["created_at_kst"] = observed.get("created_at_kst")
    if observed != expected:
        raise PermissionError("Gen5r2 attempt lock fails full deep equality")
    return observed


def write_run_failure_receipt(
    root: Path,
    config: dict[str, Any],
    *,
    exception: BaseException,
) -> Path:
    paths = stage_paths(root, config)
    lock = paths["attempt_lock"].resolve(strict=True)
    output_entries: list[str] = []
    if paths["output"].is_dir():
        output_entries = sorted(
            path.relative_to(paths["output"]).as_posix()
            for path in paths["output"].rglob("*")
        )
    receipt = {
        "schema_version": "p3_hierarchical_residual_basis.gen5r2_dense72.failure.r1",
        "stage": STAGE,
        "classification": "POST_LOCK_FAILURE_NO_AUTOMATIC_RETRY",
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "attempt_lock": _pin(lock, root.resolve(strict=True)),
        "implementation_pins": implementation_pins(root, config),
        "exception_type": type(exception).__name__,
        "exception_message_sha256": hashlib.sha256(str(exception).encode("utf-8")).hexdigest(),
        "raw_exception_message_persisted": False,
        "output_recursive_entry_count": len(output_entries),
        "output_registered_relative_entries": output_entries,
        "candidate_or_test_prediction_allowed": False,
        "upload_allowed": False,
        "rerun_allowed": False,
        "resume_allowed": False,
        "uploads": 0,
    }
    exclusive_json(paths["run_failure_receipt"], receipt)
    return paths["run_failure_receipt"]


__all__ = [
    "COMPARISON_MODE",
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "Dense72ContractError",
    "ExecutionCapability",
    "FOLD_ORDER",
    "QA_BOUND_DYNAMIC_ROLES",
    "STAGE",
    "canonical_json_bytes",
    "consume_attempt_lock",
    "exclusive_json",
    "implementation_pins",
    "issue_execution_capability",
    "load_canonical_config",
    "prepare_execution_preflight",
    "require_execution_capability",
    "revoke_execution_capability",
    "sha256_file",
    "stage_paths",
    "static_preflight",
    "strict_json_object",
    "verify_consumed_attempt_lock",
    "verify_execution_authorization",
    "verify_input_pins",
    "verify_pre_execution_qa",
    "verify_runtime_environment",
    "workspace_path",
    "write_run_failure_receipt",
]
