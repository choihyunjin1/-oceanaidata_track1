"""Run the append-only P1 Gen5r5 portable blind-boundary correction once."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from p1_qc import incumbent_residual_experiment_v5r3 as engine
from p1_qc.causal_raw_features_v4r2 import build_exact_prefix_causal_matrix
from p1_qc.selective_targets_v5r2 import (
    SelectiveTargetAccessor,
    load_frozen_oof_keys_only,
    load_input_only_train,
)

EXPECTED_CONFIG_SHA256 = (
    "a80bec2fa23d256d12c7575f6118cdcadffcb4deea9ef10d8aa19f890d799817"
)
EXPECTED_CONFIG_DEEP_SHA256 = (
    "636975841e1805602b3f4404a1f500eae48adee17c13558fb899fef8a6e79083"
)
CANONICAL_CONFIG = (
    "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5r5.json"
)
CANONICAL_RUNNER = (
    "scripts/run_p1_incumbent_rule_distillation_neural_residual_v5r5.py"
)
CANONICAL_ARTIFACT = "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r5"
CANONICAL_CONTROL = "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r5_control"
CANONICAL_PREREGISTRATION = f"{CANONICAL_CONTROL}/preregistration.json"
CANONICAL_PRESEAL = f"{CANONICAL_CONTROL}/preseal_static_qa.json"
CANONICAL_QA_RECEIPT = f"{CANONICAL_CONTROL}/independent_static_qa.json"
CANONICAL_AUTHORIZATION = f"{CANONICAL_CONTROL}/execution_authorization.json"
CANONICAL_LOCK = f"{CANONICAL_CONTROL}/attempt.lock"
WORKSPACE_ENV = "P1_WORKSPACE_ROOT"
DATA_ENV = "P1_DATA_DIR"
R4_CONFIG_SHA256 = "2c81d5bf945f8d1e1900824e0daf983dc0f79892e9e4b07d370480b97e0d8f2a"
R4_RUNNER_SHA256 = "4febfba797666c933756529afb776d2d619b5dbeb99ac15cbbad5b5255777e04"
R4_TEST_SHA256 = "20bc87911ebc1e778f5fe64aac52b3f6c73c23f56c3dbea22edd8d4897765e71"
R4_PREREGISTRATION_SHA256 = (
    "1dc43e8d94f06f3741199bcd7e2a106292c92296d6ef4f3c799295bf30692d4b"
)
R4_PRESEAL_SHA256 = (
    "f4dd2a39866090ec5ff36794ccf71fc4e6ee20d44d11a7759fa1eecb08126bd1"
)
R4_OWNER_NO_GO_SHA256 = (
    "c07cc8ffe817a8fcc8230a45a005c9766b8d22768af429f6285a38aeb9bb305f"
)
R4_TOMBSTONE_SHA256 = (
    "1cc874629d3033cb057f64f82027cac9ebb8514fd3de01ffdb4eb63f216d1271"
)
DATA_IDENTITY_OBSERVATION_1_SHA256 = (
    "7f0d568920fd5ecc82ce753cf872113accab9975297d934fe2f438973c1e223c"
)
DATA_IDENTITY_OBSERVATION_2_SHA256 = (
    "84460f1b7f49fd5da6fac5362dfd8ff828664b8b6927be0c5f4862b888dc0890"
)
SCIENCE_PROJECTION_SHA256 = (
    "1b571ee9b755b1e0ff791bfd72adc47b22d7ae571f34affbf2bccfcab4eaa72b"
)
SCIENCE_SOURCE_CONFIG_SHA256 = (
    "da7427dcfa58daff7d9825653c34296aeb6c4d0648d0d2295715c5e8c0179396"
)
HYPOTHESIS = engine.HYPOTHESIS
FRACTIONS = engine.FRACTIONS
SEEDS = engine.SEEDS
FOLD_ORDER = engine.FOLD_ORDER
_ALLOWED_SCIENCE_KEYS = (
    "problem",
    "metric",
    "direction",
    "comparison_mode",
    "hypotheses",
    "prefix_fractions",
    "seeds",
    "prefix_protocol",
    "comparator",
    "inner_cross_fit",
    "features",
    "model",
    "training",
    "train_only_no_op_gate",
    "fixed_fold_postprocess",
    "bootstrap",
    "pass_gates",
    "on_pass",
    "on_no_pass",
    "prohibitions",
)
_ANCHOR_FILES = {
    "repository_policy": ("workspace", "AGENTS.md"),
    "mandatory_policy": ("workspace", "00_MUST_READ_FIRST.md"),
    "requirements_lock": ("workspace", "requirements-lock.txt"),
    "source_readme": ("data", "README.md"),
    "train_source": ("data", "train.csv"),
    "test_source": ("data", "test.csv"),
    "sample_submission_source": ("data", "sample_submission.csv"),
}
_FORBIDDEN_EXECUTABLE_CLOSURE = {
    "scripts/run_p1_incumbent_rule_distillation_neural_residual_v5.py",
    "scripts/run_p1_incumbent_rule_distillation_neural_residual_v5r2.py",
    "scripts/run_p1_masked_pretrain_binary_event_v4r4.py",
    "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5.json",
    "configs/experiments/p1_masked_pretrain_binary_event_v4r4.json",
}
_REQUIRED_EXECUTABLE_CLOSURE = {
    "src/p1_qc/incumbent_residual_experiment_v5r3.py",
    "src/p1_qc/selective_targets_v5r2.py",
    "src/p1_qc/incumbent_residual_tcn.py",
    "src/p1_qc/causal_raw_features_v4r2.py",
    "src/p1_qc/binary_event_tcn.py",
    "src/p1_qc/temporal_event_tcn.py",
    "src/p1_qc/config.py",
    "src/p1_qc/data.py",
    "src/p1_qc/features.py",
    "src/p1_qc/pipeline.py",
    "src/p1_qc/models_tabular.py",
    "src/p1_qc/rules.py",
    "src/p1_qc/validation.py",
    "src/ocean_goal/meaningful_score_v3.py",
    "src/ocean_goal/meaningful_score_ledger_v5.py",
    "configs/p1.toml",
    "configs/goals/meaningful_score_maximization_v3.json",
    "configs/goals/meaningful_score_ledger_v5.json",
    "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5_science_projection.json",
    "requirements-lock.txt",
}


def _paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / CANONICAL_CONFIG,
        "runner": root / CANONICAL_RUNNER,
        "artifact": root / CANONICAL_ARTIFACT,
        "control": root / CANONICAL_CONTROL,
        "preregistration": root / CANONICAL_PREREGISTRATION,
        "preseal": root / CANONICAL_PRESEAL,
        "qa_receipt": root / CANONICAL_QA_RECEIPT,
        "authorization": root / CANONICAL_AUTHORIZATION,
        "lock": root / CANONICAL_LOCK,
        "r4_config": root
        / "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5r4.json",
        "r4_runner": root
        / "scripts/run_p1_incumbent_rule_distillation_neural_residual_v5r4.py",
        "r4_tests": root
        / "tests/test_run_p1_incumbent_rule_distillation_neural_residual_v5r4.py",
        "r4_preregistration": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r4_control/preregistration.json",
        "r4_preseal": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r4_control/preseal_static_qa.json",
        "r4_owner_no_go": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r4_control/OWNER_STATIC_QA_NO_GO_20260823.json",
        "r4_tombstone": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r4_control/EXECUTION_TOMBSTONE.json",
        "r4_artifact": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r4",
        "r4_qa": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r4_control/independent_static_qa.json",
        "r4_authorization": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r4_control/execution_authorization.json",
        "r4_lock": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r4_control/attempt.lock",
        "identity_observation_1": root
        / f"{CANONICAL_CONTROL}/data_identity_observation_1.json",
        "identity_observation_2": root
        / f"{CANONICAL_CONTROL}/data_identity_observation_2.json",
        "science_projection": root
        / "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5_science_projection.json",
        "base_config": root / "configs/p1.toml",
        "goal": root / "configs/goals/meaningful_score_maximization_v3.json",
        "feature_cache": root / "artifacts/cache/train_causal_raw_prefix_safe_v4r2.parquet",
        "feature_metadata": root / "artifacts/cache/train_causal_raw_prefix_safe_v4r2.json",
        "gen1": root / "artifacts/p1_meaningful_learning_curve_generation_v1",
        "frozen_oof": root
        / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
        "ledger": root / "artifacts/meaningful_score_goal_v5/registry.jsonl",
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deep_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token in {path.name}: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _reject_reparse_chain(path: Path, *, role: str) -> None:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    absolute = Path(os.path.abspath(path))
    chain = [absolute, *absolute.parents]
    for item in chain:
        if not item.exists():
            continue
        info = os.lstat(item)
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
            raise PermissionError(f"reparse-point identity anchor is forbidden: {role}")


def _file_identity(
    path: Path,
    *,
    role: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    _reject_reparse_chain(path, role=role)
    resolved = path.resolve(strict=True)
    info = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise PermissionError(f"identity anchor is not a regular file: {role}")
    if info.st_nlink != 1:
        raise PermissionError(f"hardlinked identity anchor is forbidden: {role}")
    digest = _sha(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise PermissionError(f"identity anchor SHA differs: {role}")
    return {
        "sha256": digest,
        "bytes": int(info.st_size),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "nlink": int(info.st_nlink),
        "non_reparse": True,
    }


def _directory_identity(path: Path, *, role: str) -> dict[str, Any]:
    _reject_reparse_chain(path, role=role)
    resolved = path.resolve(strict=True)
    info = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError(f"identity anchor is not a directory: {role}")
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "non_reparse": True,
    }


def _environment_paths(
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    source = os.environ if environ is None else environ
    workspace_raw = source.get(WORKSPACE_ENV)
    data_raw = source.get(DATA_ENV)
    if not workspace_raw or not data_raw:
        raise PermissionError(
            f"{WORKSPACE_ENV} and {DATA_ENV} are both required; path fallback is forbidden"
        )
    workspace_lexical = Path(workspace_raw)
    data_lexical = Path(data_raw)
    if not workspace_lexical.is_absolute() or not data_lexical.is_absolute():
        raise PermissionError("injected workspace and data directories must be absolute")
    _reject_reparse_chain(workspace_lexical, role="workspace_environment")
    _reject_reparse_chain(data_lexical, role="data_environment")
    root = workspace_lexical.resolve(strict=True)
    data_dir = data_lexical.resolve(strict=True)
    if not os.path.samefile(root / CANONICAL_RUNNER, Path(__file__).resolve(strict=True)):
        raise PermissionError("injected workspace does not own the executing runner")
    return root, data_dir


def _capture_anchor_snapshot(
    root: Path,
    data_dir: Path,
    overlay: dict[str, Any],
) -> dict[str, Any]:
    expected_files = overlay["anchor_file_identity"]
    if set(expected_files) != set(_ANCHOR_FILES):
        raise PermissionError("registered anchor-file roles differ")
    files: dict[str, dict[str, Any]] = {}
    for role, (base_name, relative) in _ANCHOR_FILES.items():
        base = root if base_name == "workspace" else data_dir
        files[role] = _file_identity(base / relative, role=role)
        if files[role] != expected_files[role]:
            raise PermissionError(f"registered anchor-file identity differs: {role}")
    directories = {
        "workspace": _directory_identity(root, role="workspace"),
        "data": _directory_identity(data_dir, role="data"),
    }
    if directories != overlay["anchor_directory_identity"]:
        raise PermissionError("registered anchor-directory identity differs")
    runner = _file_identity(root / CANONICAL_RUNNER, role="runner")
    config = _file_identity(
        root / CANONICAL_CONFIG,
        role="config",
        expected_sha256=EXPECTED_CONFIG_SHA256,
    )
    return {
        "directories": directories,
        "files": files,
        "runner": runner,
        "config": config,
    }


def _verify_r4_tombstone(paths: dict[str, Path]) -> dict[str, Any]:
    expected = {
        "r4_config": R4_CONFIG_SHA256,
        "r4_runner": R4_RUNNER_SHA256,
        "r4_tests": R4_TEST_SHA256,
        "r4_preregistration": R4_PREREGISTRATION_SHA256,
        "r4_preseal": R4_PRESEAL_SHA256,
        "r4_owner_no_go": R4_OWNER_NO_GO_SHA256,
        "r4_tombstone": R4_TOMBSTONE_SHA256,
    }
    observed = {name: _sha(paths[name]) for name in expected}
    if observed != expected:
        raise PermissionError("Gen5r4 reviewed bytes, owner NO-GO, or tombstone differ")
    if any(
        paths[name].exists()
        for name in ("r4_artifact", "r4_qa", "r4_authorization", "r4_lock")
    ):
        raise PermissionError("invalidated Gen5r4 gained a run-control or result artifact")
    tombstone = _strict_json(paths["r4_tombstone"])
    no_go = _strict_json(paths["r4_owner_no_go"])
    if not (
        tombstone.get("generation")
        == "p1_incumbent_rule_distillation_neural_residual_v5r4"
        and tombstone.get("successor_generation")
        == "p1_incumbent_rule_distillation_neural_residual_v5r5"
        and tombstone.get("execution_prohibited") is True
        and tombstone.get("authorization_must_fail_before_attempt_lock") is True
        and tombstone.get("identity_reversion_does_not_clear_tombstone") is True
        and tombstone.get("attempt_lock_created") is False
        and tombstone.get("model_fits") == 0
        and tombstone.get("target_fold_scores") == 0
        and tombstone.get("test_value_reads") == 0
        and tombstone.get("candidate_files") == 0
        and tombstone.get("uploads") == 0
    ):
        raise PermissionError("Gen5r4 execution tombstone semantics differ")
    reviewer_snapshot = no_go.get("reviewer_snapshot", {})
    expected_inodes = {
        "data_directory": (844424930448122, 844424930448107),
        "README.md": (844424930448125, 844424930448109),
        "train.csv": (844424930448126, 844424930448113),
        "test.csv": (844424930448124, 844424930448112),
        "sample_submission.csv": (844424930448128, 844424930448110),
    }
    if not (
        no_go.get("verdict") == "P0=0_P1=1_NO_GO"
        and no_go.get("p0_count") == 0
        and no_go.get("p1_count") == 1
        and reviewer_snapshot.get("device") == "not_observed_in_reviewer_payload"
        and all(
            reviewer_snapshot.get(role, {}).get("frozen_inode") == pair[0]
            and reviewer_snapshot.get(role, {}).get("observed_inode") == pair[1]
            for role, pair in expected_inodes.items()
        )
        and no_go.get("post_review_live_recheck", {}).get(
            "identity_returned_to_original_r4_frozen_values"
        )
        is True
        and no_go.get("post_review_live_recheck", {}).get(
            "reversion_treated_as_provenance_risk_not_clearance"
        )
        is True
    ):
        raise PermissionError("Gen5r4 transient identity-drift provenance differs")
    return {**observed, "execution_prohibited": True}


def _verify_r4_source_content_equivalence(
    paths: dict[str, Path],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    r4 = _strict_json(paths["r4_config"])
    role_names = {
        "source_readme": "README.md",
        "train_source": "train.csv",
        "test_source": "test.csv",
        "sample_submission_source": "sample_submission.csv",
    }
    rows: dict[str, dict[str, Any]] = {}
    for role, filename in role_names.items():
        previous = r4["anchor_file_identity"][role]
        current = overlay["anchor_file_identity"][role]
        if (
            previous["sha256"] != current["sha256"]
            or previous["bytes"] != current["bytes"]
        ):
            raise PermissionError(f"Gen5r5 source content differs from r4: {filename}")
        rows[filename] = {
            "r4_sha256": previous["sha256"],
            "current_sha256": current["sha256"],
            "r4_bytes": int(previous["bytes"]),
            "current_bytes": int(current["bytes"]),
            "sha256_equal": True,
            "bytes_equal": True,
        }
    if set(rows) != {"README.md", "train.csv", "test.csv", "sample_submission.csv"}:
        raise AssertionError("Gen5r5 four-source equivalence set differs")
    return {
        "r4_config_sha256": R4_CONFIG_SHA256,
        "sources": rows,
        "all_four_sha256_and_bytes_equal": True,
    }


def _verify_identity_observations(
    paths: dict[str, Path],
    anchors: dict[str, Any],
) -> dict[str, Any]:
    pins = (
        (1, paths["identity_observation_1"], DATA_IDENTITY_OBSERVATION_1_SHA256),
        (2, paths["identity_observation_2"], DATA_IDENTITY_OBSERVATION_2_SHA256),
    )
    source_roles = {
        "README.md": "source_readme",
        "train.csv": "train_source",
        "test.csv": "test_source",
        "sample_submission.csv": "sample_submission_source",
    }
    observed: list[dict[str, Any]] = []
    for ordinal, path, expected_sha in pins:
        if _sha(path) != expected_sha:
            raise PermissionError(f"Gen5r5 identity observation {ordinal} SHA differs")
        value = _strict_json(path)
        expected_sources = {
            filename: {
                **anchors["files"][role],
                "sha256_and_bytes_equal_r4": True,
            }
            for filename, role in source_roles.items()
        }
        if not (
            value.get("schema_version")
            == "p1_gen5r5_data_identity_observation.v1"
            and value.get("observation") == ordinal
            and value.get("problem") == "P1"
            and value.get("generation")
            == "p1_incumbent_rule_distillation_neural_residual_v5r5"
            and value.get("path_recorded") is False
            and value.get("environment_contract")
            == {
                "workspace": WORKSPACE_ENV,
                "data": DATA_ENV,
                "fallback_allowed": False,
            }
            and value.get("data_directory_identity") == anchors["directories"]["data"]
            and value.get("source_files") == expected_sources
            and value.get("r4_config_sha256") == R4_CONFIG_SHA256
            and value.get("all_four_source_sha256_and_bytes_equal_r4") is True
            and value.get("identity_matches_r4_frozen_snapshot_at_this_observation")
            is True
            and value.get("model_fits") == 0
            and value.get("target_fold_scores") == 0
            and value.get("test_value_reads") == 0
            and value.get("candidate_files") == 0
            and value.get("uploads") == 0
        ):
            raise PermissionError(f"Gen5r5 identity observation {ordinal} differs")
        observed.append(
            {
                "path": path.relative_to(paths["config"].parents[2]).as_posix(),
                "sha256": expected_sha,
                "captured_at_kst": value["captured_at_kst"],
            }
        )
    if not observed[0]["captured_at_kst"] < observed[1]["captured_at_kst"]:
        raise PermissionError("Gen5r5 identity observations are not time-separated")
    return {
        "observations": observed,
        "same_identity_across_both_observations_and_runtime": True,
        "all_four_source_sha256_and_bytes_equal_r4": True,
    }


def _load_scientific_projection(
    paths: dict[str, Path],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    projection_path = paths["science_projection"]
    if _sha(projection_path) != SCIENCE_PROJECTION_SHA256:
        raise PermissionError("scientific allowlist projection SHA differs")
    projection = _strict_json(projection_path)
    if not (
        projection.get("schema_version")
        == "p1_gen5_scientific_allowlist_projection.v1"
        and projection.get("source_config_sha256") == SCIENCE_SOURCE_CONFIG_SHA256
        and tuple(projection.get("allowed_keys", ())) == _ALLOWED_SCIENCE_KEYS
        and set(projection.get("science", {})) == set(_ALLOWED_SCIENCE_KEYS)
    ):
        raise PermissionError("scientific allowlist projection identity differs")
    science = projection["science"]
    if _deep_sha(science) != overlay["scientific_projection"]["science_deep_sha256"]:
        raise PermissionError("scientific allowlist projection deep SHA differs")
    forbidden_keys = {
        "canonical_paths",
        "canonical_identity",
        "immutable_inputs",
        "execution_closure_sha256",
        "authorization_contract",
        "v5_ledger_binding",
        "predecessor_config_path",
        "predecessor_config_sha256",
    }
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if forbidden_keys.intersection(value):
                raise PermissionError("path-bearing predecessor metadata entered science")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(science)
    return copy.deepcopy(science)


def _verify_execution_closure(
    root: Path,
    overlay: dict[str, Any],
) -> dict[str, str]:
    expected = overlay["execution_closure_sha256"]
    if not isinstance(expected, dict) or not expected:
        raise PermissionError("Gen5r5 executable closure is empty")
    names = set(expected)
    if _FORBIDDEN_EXECUTABLE_CLOSURE.intersection(names):
        raise PermissionError("path-bearing predecessor runner/config entered closure")
    if any(name.startswith("scripts/") for name in names):
        raise PermissionError("runner scripts are excluded from the portable pure closure")
    if not _REQUIRED_EXECUTABLE_CLOSURE.issubset(names):
        raise PermissionError("portable transitive executable closure is incomplete")
    observed: dict[str, str] = {}
    for relative, expected_sha in expected.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise PermissionError("Gen5r5 executable closure path is non-portable")
        path = (root / rel).resolve(strict=True)
        if not path.is_relative_to(root):
            raise PermissionError("Gen5r5 executable closure escaped the workspace")
        observed[relative] = _sha(path)
        if observed[relative] != expected_sha:
            raise PermissionError(f"Gen5r5 executable closure SHA differs: {relative}")
    return observed


def _runtime_config(
    science: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    runtime = copy.deepcopy(science)
    runtime.update(
        {
            "schema_version": overlay["schema_version"],
            "experiment_id": overlay["experiment_id"],
            "status": overlay["status"],
            "canonical_paths": {
                "config": CANONICAL_CONFIG,
                "artifact": CANONICAL_ARTIFACT,
                "attempt_lock": CANONICAL_LOCK,
                "goal_contract": "configs/goals/meaningful_score_maximization_v3.json",
                "ledger_contract": "configs/goals/meaningful_score_ledger_v5.json",
            },
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "immutable_inputs": copy.deepcopy(overlay["immutable_inputs"]),
            "v5_ledger_binding": copy.deepcopy(overlay["v5_ledger_binding"]),
            "diagnosis_binding": copy.deepcopy(overlay["diagnosis_binding"]),
        }
    )
    return runtime


def _verify_owner_seals(
    root: Path,
    overlay: dict[str, Any],
    paths: dict[str, Path],
    anchors: dict[str, Any],
) -> dict[str, str]:
    if not paths["preregistration"].is_file() or not paths["preseal"].is_file():
        raise PermissionError("Gen5r5 owner preregistration or preseal is missing")
    runner_pin = {"path": CANONICAL_RUNNER, "sha256": _sha(paths["runner"])}
    config_pin = {"path": CANONICAL_CONFIG, "sha256": EXPECTED_CONFIG_SHA256}
    observations = _verify_identity_observations(paths, anchors)
    content_equivalence = _verify_r4_source_content_equivalence(paths, overlay)
    prereg = _strict_json(paths["preregistration"])
    if not (
        prereg.get("schema_version") == "p1_gen5r5_preregistration.v1"
        and prereg.get("problem") == "P1"
        and prereg.get("generation") == overlay["experiment_id"]
        and prereg.get("config") == config_pin
        and prereg.get("runner") == runner_pin
        and prereg.get("scientific_projection")
        == {
            "path": "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5_science_projection.json",
            "sha256": SCIENCE_PROJECTION_SHA256,
        }
        and prereg.get("anchor_snapshot_sha256") == _deep_sha(anchors)
        and prereg.get("data_identity_observations") == observations
        and prereg.get("r4_source_content_equivalence_sha256")
        == _deep_sha(content_equivalence)
        and prereg.get("created_before_first_fit") is True
        and prereg.get("score_derived_tuning") is False
        and prereg.get("hypothesis_count") == 1
        and prereg.get("qa_receipts") == 0
        and prereg.get("execution_authorizations") == 0
        and prereg.get("attempt_locks") == 0
        and prereg.get("model_fits") == 0
        and prereg.get("target_fold_scores") == 0
        and prereg.get("test_value_reads") == 0
        and prereg.get("candidate_files") == 0
        and prereg.get("uploads") == 0
    ):
        raise PermissionError("Gen5r5 owner preregistration semantics differ")
    preseal = _strict_json(paths["preseal"])
    if not (
        preseal.get("schema_version") == "p1_gen5r5_owner_preseal_static_qa.v1"
        and preseal.get("problem") == "P1"
        and preseal.get("generation") == overlay["experiment_id"]
        and preseal.get("verdict") == "READY_FOR_INDEPENDENT_STATIC_QA"
        and preseal.get("config") == config_pin
        and preseal.get("runner") == runner_pin
        and preseal.get("preregistration")
        == {"path": CANONICAL_PREREGISTRATION, "sha256": _sha(paths["preregistration"])}
        and preseal.get("anchor_snapshot_sha256") == _deep_sha(anchors)
        and preseal.get("data_identity_observations") == observations
        and preseal.get("r4_source_content_equivalence_sha256")
        == _deep_sha(content_equivalence)
        and preseal.get("p0_count") == preseal.get("p1_count") == 0
        and preseal.get("actual_run_performed") is False
        and preseal.get("independent_qa_receipt_created") is False
        and preseal.get("execution_authorization_created") is False
        and preseal.get("attempt_lock_created") is False
        and preseal.get("model_fits") == 0
        and preseal.get("target_fold_scores") == 0
        and preseal.get("test_value_reads") == 0
        and preseal.get("candidate_files") == 0
        and preseal.get("uploads") == 0
    ):
        raise PermissionError("Gen5r5 owner preseal semantics differ")
    return {
        "preregistration_sha256": _sha(paths["preregistration"]),
        "preseal_sha256": _sha(paths["preseal"]),
    }


class _OperationalContext:
    __slots__ = (
        "root",
        "data_dir",
        "overlay",
        "config",
        "paths",
        "closure",
        "anchors",
        "input_pins",
        "owner_seals",
        "ledger_binding",
        "r4_content_equivalence",
        "identity_observations",
        "core_snapshot",
    )

    def __init__(
        self,
        *,
        root: Path,
        data_dir: Path,
        overlay: dict[str, Any],
        config: dict[str, Any],
        paths: dict[str, Path],
        closure: dict[str, str],
        anchors: dict[str, Any],
        input_pins: dict[str, dict[str, Any]],
        owner_seals: dict[str, str],
        ledger_binding: dict[str, Any],
        r4_content_equivalence: dict[str, Any],
        identity_observations: dict[str, Any],
    ) -> None:
        self.root = root
        self.data_dir = data_dir
        self.overlay = overlay
        self.config = config
        self.paths = paths
        self.closure = closure
        self.anchors = anchors
        self.input_pins = input_pins
        self.owner_seals = owner_seals
        self.ledger_binding = ledger_binding
        self.r4_content_equivalence = r4_content_equivalence
        self.identity_observations = identity_observations
        self.core_snapshot = {
            "schema_version": "p1_gen5r5_operational_core_snapshot.v1",
            "config": {
                "path": CANONICAL_CONFIG,
                "sha256": EXPECTED_CONFIG_SHA256,
                "deep_sha256": EXPECTED_CONFIG_DEEP_SHA256,
            },
            "runner": {
                "path": CANONICAL_RUNNER,
                "sha256": _sha(paths["runner"]),
            },
            "scientific_projection": {
                "sha256": SCIENCE_PROJECTION_SHA256,
                "science_deep_sha256": overlay["scientific_projection"][
                    "science_deep_sha256"
                ],
            },
            "owner_seals": owner_seals,
            "r4_tombstone": _verify_r4_tombstone(paths),
            "r4_source_content_equivalence": r4_content_equivalence,
            "data_identity_observations": identity_observations,
            "anchor_snapshot": anchors,
            "execution_closure_sha256": closure,
            "immutable_input_snapshot": input_pins,
            "v5_ledger_binding": ledger_binding,
            "environment_contract": {
                "workspace": WORKSPACE_ENV,
                "data": DATA_ENV,
                "fallback_allowed": False,
            },
        }


def _load_operational_context(
    *,
    environ: Mapping[str, str] | None = None,
    require_owner_seals: bool,
) -> _OperationalContext:
    root, data_dir = _environment_paths(environ)
    paths = _paths(root)
    content = paths["config"].read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise PermissionError("canonical Gen5r5 config byte SHA differs")
    overlay = _strict_json(paths["config"])
    if _deep_sha(overlay) != EXPECTED_CONFIG_DEEP_SHA256:
        raise PermissionError("canonical Gen5r5 config deep JSON differs")
    if not (
        overlay.get("schema_version")
        == "p1_incumbent_rule_distillation_neural_residual.v5r5"
        and overlay.get("experiment_id")
        == "p1_incumbent_rule_distillation_neural_residual_v5r5"
        and overlay.get("problem") == "P1"
        and overlay.get("comparison_mode") == "EXACT_OFFICIAL_PREFIX_REFIT"
        and overlay.get("environment_contract")
        == {
            "workspace": WORKSPACE_ENV,
            "data": DATA_ENV,
            "fallback_allowed": False,
        }
        and overlay["v5_ledger_binding"]["head_seq"] == 9
        and overlay["v5_ledger_binding"]["semantic_upload_count"] == 0
        and overlay["correction_contract"]["fold_major_order"] == list(FOLD_ORDER)
        and overlay["correction_contract"]["prefix_order"] == list(FRACTIONS)
        and overlay["correction_contract"][
            "authorize_entry_returns_opaque_prelock_capability"
        ]
        is True
        and overlay["reanchor_contract"]["scientific_structure_changed"] is False
        and overlay["reanchor_contract"]["time_separated_observation_count"] == 2
        and overlay["reanchor_contract"][
            "all_four_source_sha256_and_bytes_equal_r4"
        ]
        is True
        and overlay["data_identity_observations"]
        == [
            {
                "path": f"{CANONICAL_CONTROL}/data_identity_observation_1.json",
                "sha256": DATA_IDENTITY_OBSERVATION_1_SHA256,
            },
            {
                "path": f"{CANONICAL_CONTROL}/data_identity_observation_2.json",
                "sha256": DATA_IDENTITY_OBSERVATION_2_SHA256,
            },
        ]
    ):
        raise PermissionError("Gen5r5 identity, environment, comparator, or capability contract differs")
    _verify_r4_tombstone(paths)
    science = _load_scientific_projection(paths, overlay)
    closure = _verify_execution_closure(root, overlay)
    anchors = _capture_anchor_snapshot(root, data_dir, overlay)
    r4_content_equivalence = _verify_r4_source_content_equivalence(paths, overlay)
    identity_observations = _verify_identity_observations(paths, anchors)
    config = _runtime_config(science, overlay)
    input_pins = engine.verify_relative_input_pins(
        root,
        data_dir,
        config["immutable_inputs"],
    )
    ledger_binding = engine._verify_v5_ledger_binding(
        root,
        config,
        paths["ledger"],
    )
    owner_seals = (
        _verify_owner_seals(root, overlay, paths, anchors)
        if require_owner_seals
        else {}
    )
    return _OperationalContext(
        root=root,
        data_dir=data_dir,
        overlay=overlay,
        config=config,
        paths=paths,
        closure=closure,
        anchors=anchors,
        input_pins=input_pins,
        owner_seals=owner_seals,
        ledger_binding=ledger_binding,
        r4_content_equivalence=r4_content_equivalence,
        identity_observations=identity_observations,
    )


def _context_environment(context: _OperationalContext) -> dict[str, str]:
    return {
        WORKSPACE_ENV: str(context.root),
        DATA_ENV: str(context.data_dir),
    }


def _reload_context(context: _OperationalContext) -> _OperationalContext:
    current = _load_operational_context(
        environ=_context_environment(context),
        require_owner_seals=True,
    )
    if current.core_snapshot != context.core_snapshot:
        raise PermissionError("Gen5r5 operational core snapshot changed")
    return current


def _verify_independent_qa(
    context: _OperationalContext,
) -> tuple[dict[str, Any], str]:
    paths = context.paths
    if not paths["qa_receipt"].is_file():
        raise PermissionError("canonical Gen5r5 independent-QA receipt is missing")
    qa = _strict_json(paths["qa_receipt"])
    required = {
        "schema_version",
        "problem",
        "generation",
        "verdict",
        "p0_count",
        "p1_count",
        "config",
        "runner",
        "scientific_projection",
        "owner_seals",
        "anchor_snapshot_sha256",
        "execution_closure_sha256",
        "checks",
        "reviewer",
        "actual_run_performed",
        "files_modified",
        "test_value_reads",
        "candidate_files",
        "uploads",
    }
    if set(qa) != required:
        raise PermissionError("Gen5r5 independent-QA receipt keys differ")
    if not (
        qa["schema_version"] == "p1_gen5r5_independent_static_qa.v1"
        and qa["problem"] == "P1"
        and qa["generation"] == context.overlay["experiment_id"]
        and qa["verdict"] == "GO"
        and qa["p0_count"] == qa["p1_count"] == 0
        and qa["config"]
        == {"path": CANONICAL_CONFIG, "sha256": EXPECTED_CONFIG_SHA256}
        and qa["runner"]
        == {"path": CANONICAL_RUNNER, "sha256": _sha(paths["runner"])}
        and qa["scientific_projection"]
        == {
            "path": (
                "configs/experiments/"
                "p1_incumbent_rule_distillation_neural_residual_v5_science_projection.json"
            ),
            "sha256": SCIENCE_PROJECTION_SHA256,
        }
        and qa["owner_seals"] == context.owner_seals
        and qa["anchor_snapshot_sha256"] == _deep_sha(context.anchors)
        and qa["execution_closure_sha256"] == context.closure
        and isinstance(qa["checks"], dict)
        and qa["checks"]
        and all(value is True for value in qa["checks"].values())
        and bool(qa["reviewer"])
        and qa["actual_run_performed"] is False
        and qa["files_modified"] == 0
        and qa["test_value_reads"] == qa["candidate_files"] == qa["uploads"] == 0
    ):
        raise PermissionError("Gen5r5 independent-QA receipt did not authorize GO")
    return qa, _sha(paths["qa_receipt"])


def _verify_execution_authorization(
    context: _OperationalContext,
    *,
    qa_sha256: str,
    require_lock_absent: bool = True,
) -> tuple[dict[str, Any], str]:
    paths = context.paths
    if require_lock_absent and paths["lock"].exists():
        raise FileExistsError("Gen5r5 one-shot attempt lock already exists")
    if not paths["authorization"].is_file():
        raise PermissionError("separate Gen5r5 execution authorization is missing")
    authorization = _strict_json(paths["authorization"])
    required = {
        "schema_version",
        "problem",
        "generation",
        "authorization",
        "user_message_reference",
        "config",
        "runner",
        "qa_receipt",
        "owner_seals",
        "anchor_snapshot_sha256",
        "execution_closure_sha256",
        "execution_authorized",
        "test_prediction_allowed",
        "candidate_creation_allowed",
        "upload_allowed",
    }
    if set(authorization) != required:
        raise PermissionError("Gen5r5 execution authorization keys differ")
    runner_pin = {
        "path": CANONICAL_RUNNER,
        "sha256": _sha(paths["runner"]),
    }
    phrase = (
        f"AUTHORIZE_P1_GEN5R5_EXECUTION:{EXPECTED_CONFIG_SHA256}:"
        f"{runner_pin['sha256']}"
    )
    if not (
        authorization["schema_version"]
        == "p1_gen5r5_execution_authorization.v1"
        and authorization["problem"] == "P1"
        and authorization["generation"] == context.overlay["experiment_id"]
        and authorization["authorization"] == phrase
        and bool(authorization["user_message_reference"])
        and authorization["config"]
        == {"path": CANONICAL_CONFIG, "sha256": EXPECTED_CONFIG_SHA256}
        and authorization["runner"] == runner_pin
        and authorization["qa_receipt"]
        == {"path": CANONICAL_QA_RECEIPT, "sha256": qa_sha256}
        and authorization["owner_seals"] == context.owner_seals
        and authorization["anchor_snapshot_sha256"] == _deep_sha(context.anchors)
        and authorization["execution_closure_sha256"] == context.closure
        and authorization["execution_authorized"] is True
        and authorization["test_prediction_allowed"] is False
        and authorization["candidate_creation_allowed"] is False
        and authorization["upload_allowed"] is False
    ):
        raise PermissionError("Gen5r5 execution authorization failed")
    return authorization, _sha(paths["authorization"])


def _full_operational_snapshot(
    context: _OperationalContext,
    *,
    qa_sha256: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "p1_gen5r5_full_operational_snapshot.v1",
        "core": context.core_snapshot,
        "independent_qa_receipt": {
            "path": CANONICAL_QA_RECEIPT,
            "sha256": qa_sha256,
        },
        "execution_authorization": {
            "path": CANONICAL_AUTHORIZATION,
            "sha256": authorization_sha256,
        },
    }


_PRELOCK_MINT = object()
_EXECUTION_MINT = object()
_PRELOCK_STATE_MINT = object()
_EXECUTION_STATE_MINT = object()
_LIVE_PRELOCK: dict[object, dict[str, Any]] = {}
_LIVE_EXECUTION: dict[object, dict[str, Any]] = {}


class _PreLockCapability:
    __slots__ = ("_token",)

    def __init__(self, mint: object, token: object) -> None:
        if mint is not _PRELOCK_MINT:
            raise PermissionError("canonical Gen5r5 pre-lock capability mint is required")
        self._token = token


class _ExecutionCapability:
    __slots__ = ("_token",)

    def __init__(self, mint: object, token: object) -> None:
        if mint is not _EXECUTION_MINT:
            raise PermissionError("canonical Gen5r5 post-lock capability mint is required")
        self._token = token


def _mint_prelock_capability(
    context: _OperationalContext,
    *,
    qa_sha256: str,
    authorization_sha256: str,
) -> _PreLockCapability:
    if _LIVE_PRELOCK or _LIVE_EXECUTION:
        raise PermissionError("a canonical Gen5r5 capability is already live")
    token = object()
    nonce = os.urandom(32).hex()
    snapshot = _full_operational_snapshot(
        context,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    capability = _PreLockCapability(_PRELOCK_MINT, token)
    _LIVE_PRELOCK[token] = {
        "mint": _PRELOCK_STATE_MINT,
        "capability": capability,
        "context": context,
        "qa_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "snapshot": snapshot,
        "snapshot_sha256": _deep_sha(snapshot),
        "nonce": nonce,
        "nonce_sha256": hashlib.sha256(bytes.fromhex(nonce)).hexdigest(),
        "stage": "QA_AUTH_VERIFIED_PRELOCK",
    }
    return capability


def authorize_entry(
    *,
    environ: Mapping[str, str] | None = None,
) -> _PreLockCapability:
    context = _load_operational_context(
        environ=environ,
        require_owner_seals=True,
    )
    if context.paths["artifact"].exists() or context.paths["lock"].exists():
        raise FileExistsError("Gen5r5 artifact or attempt lock already exists")
    _, qa_sha256 = _verify_independent_qa(context)
    _, authorization_sha256 = _verify_execution_authorization(
        context,
        qa_sha256=qa_sha256,
    )
    current = _reload_context(context)
    _, reloaded_qa_sha256 = _verify_independent_qa(current)
    _, reloaded_authorization_sha256 = _verify_execution_authorization(
        current,
        qa_sha256=reloaded_qa_sha256,
    )
    if (
        reloaded_qa_sha256 != qa_sha256
        or reloaded_authorization_sha256 != authorization_sha256
    ):
        raise PermissionError("Gen5r5 QA or authorization changed before pre-lock mint")
    return _mint_prelock_capability(
        current,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )


def _require_prelock_capability(
    capability: object,
    *,
    expected_stage: str,
) -> tuple[dict[str, Any], _OperationalContext]:
    if not isinstance(capability, _PreLockCapability):
        raise PermissionError("canonical opaque Gen5r5 pre-lock capability is required")
    state = _LIVE_PRELOCK.get(capability._token)
    if (
        state is None
        or state.get("mint") is not _PRELOCK_STATE_MINT
        or state.get("capability") is not capability
    ):
        raise PermissionError("forged or stale Gen5r5 pre-lock capability")
    if state["stage"] != expected_stage:
        raise PermissionError(f"Gen5r5 pre-lock stage must be {expected_stage}")
    current = _reload_context(state["context"])
    _, qa_sha256 = _verify_independent_qa(current)
    _, authorization_sha256 = _verify_execution_authorization(
        current,
        qa_sha256=qa_sha256,
        require_lock_absent=expected_stage != "LOCK_WRITTEN",
    )
    snapshot = _full_operational_snapshot(
        current,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    if not (
        qa_sha256 == state["qa_sha256"]
        and authorization_sha256 == state["authorization_sha256"]
        and snapshot == state["snapshot"]
        and _deep_sha(snapshot) == state["snapshot_sha256"]
    ):
        raise PermissionError("Gen5r5 pre-lock operational snapshot changed")
    return state, current


def _canonical_lock_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "p1_gen5r5_attempt_lock.v1",
        "status": "ATTEMPT_CONSUMED_ONE_SHOT",
        "experiment_id": "p1_incumbent_rule_distillation_neural_residual_v5r5",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "runner_sha256": state["snapshot"]["core"]["runner"]["sha256"],
        "qa_receipt_sha256": state["qa_sha256"],
        "authorization_sha256": state["authorization_sha256"],
        "full_operational_snapshot": state["snapshot"],
        "full_operational_snapshot_sha256": state["snapshot_sha256"],
        "prelock_capability_nonce_sha256": state["nonce_sha256"],
        "o_excl": True,
        "rerun_forbidden": True,
    }


def _verify_consumed_lock(
    state: dict[str, Any],
    context: _OperationalContext,
) -> dict[str, Any]:
    if state.get("mint") not in {_PRELOCK_STATE_MINT, _EXECUTION_STATE_MINT}:
        raise PermissionError("forged Gen5r5 capability state")
    _, qa_sha256 = _verify_independent_qa(context)
    _, authorization_sha256 = _verify_execution_authorization(
        context,
        qa_sha256=qa_sha256,
        require_lock_absent=False,
    )
    if (
        qa_sha256 != state["qa_sha256"]
        or authorization_sha256 != state["authorization_sha256"]
    ):
        raise PermissionError("Gen5r5 post-lock QA or authorization differs")
    lock_path = context.paths["lock"]
    if not lock_path.is_file():
        raise PermissionError("canonical consumed Gen5r5 attempt lock is missing")
    payload = _strict_json(lock_path)
    expected = _canonical_lock_payload(state)
    if payload != expected:
        raise PermissionError("canonical consumed Gen5r5 attempt lock payload differs")
    digest = _sha(lock_path)
    if state.get("lock_sha256") not in {None, digest}:
        raise PermissionError("canonical consumed Gen5r5 attempt lock SHA differs")
    return {"payload": payload, "sha256": digest}


def _mint_execution_capability(
    capability: _PreLockCapability,
) -> _ExecutionCapability:
    state, context = _require_prelock_capability(
        capability,
        expected_stage="LOCK_WRITTEN",
    )
    lock = _verify_consumed_lock(state, context)
    token = object()
    execution = _ExecutionCapability(_EXECUTION_MINT, token)
    execution_state = {
        **state,
        "mint": _EXECUTION_STATE_MINT,
        "capability": execution,
        "context": context,
        "lock_sha256": lock["sha256"],
        "stage": "LOCK_CONSUMED",
        "runtime": None,
    }
    del _LIVE_PRELOCK[capability._token]
    _LIVE_EXECUTION[token] = execution_state
    return execution


def _acquire_lock(capability: object) -> _ExecutionCapability:
    state, context = _require_prelock_capability(
        capability,
        expected_stage="QA_AUTH_VERIFIED_PRELOCK",
    )
    if context.paths["lock"].exists():
        raise FileExistsError("Gen5r5 one-shot attempt lock already exists")
    payload = _canonical_lock_payload(state)
    _json_new(context.paths["lock"], payload)
    state["lock_sha256"] = _sha(context.paths["lock"])
    state["stage"] = "LOCK_WRITTEN"
    return _mint_execution_capability(capability)


def _require_execution_capability(
    capability: object,
    *,
    expected_stage: str,
    transition_to: str | None = None,
) -> tuple[dict[str, Any], _OperationalContext]:
    if not isinstance(capability, _ExecutionCapability):
        raise PermissionError("canonical live post-lock Gen5r5 capability is required")
    state = _LIVE_EXECUTION.get(capability._token)
    if (
        state is None
        or state.get("mint") is not _EXECUTION_STATE_MINT
        or state.get("capability") is not capability
    ):
        raise PermissionError("forged, stale, or replayed Gen5r5 execution capability")
    if state["stage"] != expected_stage:
        raise PermissionError(f"Gen5r5 execution stage must be {expected_stage}")
    current = _reload_context(state["context"])
    lock = _verify_consumed_lock(state, current)
    if lock["sha256"] != state["lock_sha256"]:
        raise PermissionError("Gen5r5 consumed lock changed")
    if transition_to is not None:
        state["stage"] = transition_to
    return state, current


class _BlindCommitmentLedger:
    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self._expected = [(fold, fraction) for fold in FOLD_ORDER for fraction in FRACTIONS]
        self._cells: list[dict[str, Any]] = []
        self._folds: dict[str, dict[str, Any]] = {}
        self._head = "0" * 64
        self._global: dict[str, Any] | None = None

    def is_fold_committed(self, fold: str) -> bool:
        return fold in self._folds

    def is_global_committed(self) -> bool:
        return self._global is not None

    def commit_cell(
        self,
        *,
        fold: str,
        fraction: float,
        validation_ids_sha256: str,
        seed_blind_predictions: list[dict[str, Any]],
        prediction_part: dict[str, Any],
        target_accessor: SelectiveTargetAccessor,
    ) -> dict[str, Any]:
        sequence = len(self._cells) + 1
        if self._expected[sequence - 1] != (fold, fraction):
            raise PermissionError("Gen5r5 cell commitment order is not fold-major monotone")
        if [row["seed"] for row in seed_blind_predictions] != list(SEEDS):
            raise PermissionError("Gen5r5 cell commitment seed order differs")
        for row in [*seed_blind_predictions, prediction_part]:
            target = self.artifact / row["path"]
            if not target.is_file() or _sha(target) != row["sha256"]:
                raise PermissionError("Gen5r5 blind prediction pin differs before commitment")
        validation_ids = target_accessor.validation_rows(fold)
        if engine.ids_sha256(validation_ids) != validation_ids_sha256:
            raise PermissionError("Gen5r5 active validation ID pin differs")
        target_counts = target_accessor.validation_target_decode_counts(fold)
        if target_counts != {"label": 0, "anomaly_type": 0}:
            raise PermissionError(
                "Gen5r5 active outer-validation target decoded before commitment"
            )
        fraction_tag = engine._tag(fraction)
        relative = f"blind_commitments/cell_{sequence:02d}_{fold}_{fraction_tag}.json"
        path = self.artifact / relative
        payload = {
            "schema_version": "p1_gen5r5_blind_cell_commitment.v1",
            "sequence": sequence,
            "previous_event_sha256": self._head,
            "fold": fold,
            "fraction": fraction,
            "validation_ids_sha256": validation_ids_sha256,
            "seed_blind_predictions": seed_blind_predictions,
            "prediction_part": prediction_part,
            "active_outer_validation_labels_decoded_before_commitment": target_counts[
                "label"
            ],
            "active_outer_validation_anomaly_types_decoded_before_commitment": target_counts[
                "anomaly_type"
            ],
            "o_excl": True,
        }
        _json_new(path, payload)
        event = {"path": relative, "sha256": _sha(path), "payload": payload}
        self._head = event["sha256"]
        self._cells.append(event)
        fold_cells = [row for row in self._cells if row["payload"]["fold"] == fold]
        if len(fold_cells) == len(FRACTIONS):
            fold_relative = f"blind_commitments/fold_{fold}.json"
            fold_path = self.artifact / fold_relative
            fold_payload = {
                "schema_version": "p1_gen5r5_blind_fold_commitment.v1",
                "fold": fold,
                "fractions": list(FRACTIONS),
                "cell_commitments": [
                    {"path": row["path"], "sha256": row["sha256"]}
                    for row in fold_cells
                ],
                "chain_head_sha256": self._head,
                "all_five_cells_committed": True,
                "rolling_origin_training_reuse_now_allowed": True,
                "o_excl": True,
            }
            _json_new(fold_path, fold_payload)
            self._folds[fold] = {
                "path": fold_relative,
                "sha256": _sha(fold_path),
                "payload": fold_payload,
            }
        return event

    def finalize_ledger(self) -> dict[str, Any]:
        if len(self._cells) != 15 or tuple(self._folds) != FOLD_ORDER:
            raise PermissionError("Gen5r5 blind ledger is incomplete")
        relative = "blind_commitments/ledger_complete.json"
        path = self.artifact / relative
        payload = {
            "schema_version": "p1_gen5r5_blind_commitment_ledger.v1",
            "order": "fold_major_then_prefix_fraction",
            "fold_order": list(FOLD_ORDER),
            "prefix_order": list(FRACTIONS),
            "cell_count": 15,
            "seed_blind_prediction_count": 45,
            "cell_commitments": [
                {"path": row["path"], "sha256": row["sha256"]} for row in self._cells
            ],
            "fold_commitments": [
                {"path": self._folds[fold]["path"], "sha256": self._folds[fold]["sha256"]}
                for fold in FOLD_ORDER
            ],
            "chain_head_sha256": self._head,
            "target_fold_scores": 0,
            "o_excl": True,
        }
        _json_new(path, payload)
        return {"path": relative, "sha256": _sha(path), "payload": payload}

    def mark_predictions_complete(self, path: Path) -> None:
        value = _strict_json(path)
        if not (
            value.get("fit_cells") == 225
            and value.get("blind_commitment_ledger", {}).get("sha256")
            == _sha(self.artifact / "blind_commitments/ledger_complete.json")
            and len(value.get("prediction_parts", [])) == 15
            and len(value.get("model_receipts", [])) == 45
        ):
            raise PermissionError("Gen5r5 predictions_complete did not bind every blind prediction")
        self._global = {"path": path, "sha256": _sha(path)}

    @property
    def cell_receipts(self) -> list[dict[str, Any]]:
        return [
            {"path": row["path"], "sha256": row["sha256"]} for row in self._cells
        ]

    @property
    def fold_receipts(self) -> list[dict[str, Any]]:
        return [
            {"path": self._folds[fold]["path"], "sha256": self._folds[fold]["sha256"]}
            for fold in FOLD_ORDER
        ]


def _run_curve(
    *,
    capability: _ExecutionCapability,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    state, context = _require_execution_capability(
        capability,
        expected_stage="BLIND_CURVE_READY",
        transition_to="CURVE_RUNNING",
    )
    runtime = state.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("sealed") is not True:
        raise PermissionError("Gen5r5 bound curve runtime is absent")
    config = context.config
    paths = context.paths
    train = runtime["train"]
    targets = runtime["targets"]
    commitment = runtime["commitment"]
    features = runtime["features"]
    feature_columns = runtime["feature_columns"]
    layout = runtime["layout"]
    folds = runtime["folds"]
    prefix_ids = runtime["prefix_ids"]
    comparator_parts = runtime["comparator_parts"]
    if tuple(str(fold["name"]) for fold in folds) != FOLD_ORDER:
        raise PermissionError("Gen5r5 outer fold order differs")
    p1_config = engine.load_config(paths["base_config"])
    model_config = engine._model_config(config, len(feature_columns), layout.group_count)
    training_config = engine._training_config(config)
    working_train = train.copy()
    working_train["label"] = pd.Series(pd.NA, index=working_train.index, dtype="Int8")
    all_receipts: list[dict[str, Any]] = []
    primary_receipts: list[dict[str, Any]] = []
    teacher_receipts: list[dict[str, Any]] = []
    gate_receipts: list[dict[str, Any]] = []
    part_receipts: list[dict[str, Any]] = []
    cell_commitments: list[dict[str, Any]] = []
    completed_primary = 0
    for fold in folds:
        fold_name = str(fold["name"])
        validation_ids = np.asarray(fold["val_idx"], dtype=np.int64)
        if not np.array_equal(validation_ids, targets.validation_rows(fold_name)):
            raise PermissionError("Gen5r5 selective accessor validation IDs differ")
        for fraction in FRACTIONS:
            working_train["label"] = pd.Series(
                pd.NA, index=working_train.index, dtype="Int8"
            )
            fraction_tag = engine._tag(fraction)
            train_ids = prefix_ids[(fold_name, fraction)]
            if np.intersect1d(train_ids, validation_ids).size:
                raise PermissionError("Gen5r5 active train/validation IDs overlap")
            planned_splits = engine.build_three_block_inner_splits(
                train.loc[:, ["time"]],
                train_ids,
                purge_days=int(config["inner_cross_fit"]["purge_days"]),
            )
            pre_gate_label_ids = np.unique(
                np.concatenate(
                    [
                        *(split.teacher_train_ids for split in planned_splits),
                        planned_splits[0].teacher_prediction_ids,
                        planned_splits[1].teacher_prediction_ids,
                    ]
                )
            )
            selected_labels = targets.labels_for(
                pre_gate_label_ids,
                commitment=commitment,
                purpose=f"curve_pre_gate_training:{fold_name}:{fraction_tag}",
                active_fold=fold_name,
            )
            label_column = working_train.columns.get_loc("label")
            working_train.iloc[pre_gate_label_ids, label_column] = selected_labels
            comparator = engine._comparator_frame(
                comparator_parts[(fold_name, fraction)], fold, fraction
            )
            prefix_features, prefix_feature_sha = build_exact_prefix_causal_matrix(
                train,
                train_ids,
                full_reference=features,
            )
            teacher, cell_teacher_receipts = engine._teacher_oof(
                config=config,
                paths=paths,
                train=working_train,
                p1_config=p1_config,
                outer_prefix_ids=train_ids,
                outer_forbidden_ids=validation_ids,
                fold_name=fold_name,
                fold_ordinal=int(fold["ordinal"]),
                fraction_tag=fraction_tag,
                scope="curve",
            )
            teacher_receipts.extend(cell_teacher_receipts)
            all_receipts.extend(cell_teacher_receipts)
            splits = teacher["splits"]
            if any(
                left.block != right.block
                or not np.array_equal(left.teacher_train_ids, right.teacher_train_ids)
                or not np.array_equal(left.teacher_prediction_ids, right.teacher_prediction_ids)
                for left, right in zip(planned_splits, splits, strict=True)
            ):
                raise PermissionError("Gen5r5 planned and executed inner splits differ")
            residual_fit_ids = np.concatenate(
                [splits[0].teacher_prediction_ids, splits[1].teacher_prediction_ids]
            )
            gate_ids = splits[2].teacher_prediction_ids
            if working_train.iloc[gate_ids]["label"].notna().any():
                raise PermissionError("Gen5r5 gate labels were decoded before blind prediction")
            refit_ids = teacher["oof_ids"]
            outer_context = np.unique(np.concatenate([train_ids, validation_ids]))
            prediction_features = prefix_features.copy()
            prediction_features[validation_ids] = features[validation_ids]
            seed_final_probabilities: list[np.ndarray] = []
            seed_final_predictions: list[np.ndarray] = []
            seed_blind_pins: list[dict[str, Any]] = []
            cell_gate_passes: list[bool] = []
            gate_stages: dict[int, dict[str, Any]] = {}
            gate_forbidden = np.unique(np.concatenate([validation_ids, gate_ids]))
            residual_fit_labels = pd.to_numeric(
                working_train.iloc[residual_fit_ids]["label"], errors="raise"
            ).to_numpy(np.int8)
            for gate_seed in SEEDS:
                gate_started = time.perf_counter()
                gate_model = engine.fit_incumbent_residual_model(
                    prefix_features,
                    layout,
                    residual_fit_ids,
                    residual_fit_labels,
                    teacher["seed_probability"][gate_seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][gate_seed],
                    context_ids=train_ids,
                    forbidden_ids=gate_forbidden,
                    seed=int(gate_seed) + int(fold["ordinal"]),
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                )
                gate_probability = engine.predict_incumbent_residual_probability(
                    gate_model,
                    prefix_features,
                    layout,
                    gate_ids,
                    teacher["seed_probability"][gate_seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][gate_seed],
                    context_ids=train_ids,
                    device="cuda",
                )
                gate_model_relative = (
                    f"gate_models/{fraction_tag}/{fold_name}/seed_{gate_seed}.pt"
                )
                gate_model_path = engine._safe_path(paths["artifact"], gate_model_relative)
                engine.save_fitted_incumbent_residual_model(gate_model, gate_model_path)
                loaded_gate = engine.load_fitted_incumbent_residual_model(gate_model_path)
                reproduced_gate = engine.predict_incumbent_residual_probability(
                    loaded_gate,
                    prefix_features,
                    layout,
                    gate_ids,
                    teacher["seed_probability"][gate_seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][gate_seed],
                    context_ids=train_ids,
                    device="cuda",
                )
                if not np.array_equal(gate_probability, reproduced_gate):
                    raise RuntimeError("saved Gen5r5 gate residual did not reproduce")
                gate_blind_relative = (
                    f"gate_blind_predictions/{fraction_tag}/{fold_name}/seed_{gate_seed}.npy"
                )
                gate_blind_path = engine._safe_path(paths["artifact"], gate_blind_relative)
                gate_blind_sha = engine._npy_new(gate_blind_path, gate_probability)
                gate_residual_prediction = engine._postprocess_ids(
                    working_train,
                    gate_ids,
                    gate_probability,
                    config["fixed_fold_postprocess"][fold_name],
                )
                gate_stages[gate_seed] = {
                    "model": gate_model,
                    "prediction": gate_residual_prediction,
                    "blind_relative": gate_blind_relative,
                    "blind_sha256": gate_blind_sha,
                    "model_relative": gate_model_relative,
                    "model_path": gate_model_path,
                    "elapsed_seconds": float(time.perf_counter() - gate_started),
                }
            if tuple(gate_stages) != SEEDS or any(
                not (paths["artifact"] / stage["blind_relative"]).is_file()
                or _sha(paths["artifact"] / stage["blind_relative"])
                != stage["blind_sha256"]
                for stage in gate_stages.values()
            ):
                raise PermissionError("Gen5r5 did not seal all three gate predictions")
            if working_train.iloc[gate_ids]["label"].notna().any():
                raise PermissionError("Gen5r5 gate labels changed before all blind seals")
            gate_truth = targets.labels_for(
                gate_ids,
                commitment=commitment,
                purpose=f"gate_after_three_blind_seals:{fold_name}:{fraction_tag}",
                active_fold=fold_name,
            )
            working_train.iloc[gate_ids, label_column] = gate_truth
            for seed in SEEDS:
                stage = gate_stages[seed]
                gate_model = stage["model"]
                gate_residual_prediction = stage["prediction"]
                gate_blind_relative = stage["blind_relative"]
                gate_blind_sha = stage["blind_sha256"]
                gate_model_relative = stage["model_relative"]
                gate_model_path = stage["model_path"]
                gate_result = engine._gate_decision(
                    train=working_train,
                    gate_ids=gate_ids,
                    truth=gate_truth,
                    base_prediction=teacher["seed_decision"][seed][gate_ids],
                    residual_prediction=gate_residual_prediction,
                    config=config,
                )
                cell_gate_passes.append(bool(gate_result["passed"]))
                gate_receipt = {
                    "role": "residual_gate",
                    "fraction": fraction,
                    "fold": fold_name,
                    "seed": seed,
                    "train_ids_sha256": gate_model.train_ids_sha256,
                    "gate_ids_sha256": engine.ids_sha256(gate_ids),
                    "optimizer_steps": training_config.optimizer_steps,
                    "gate_prediction_sealed_before_gate_label_read": True,
                    "all_three_gate_predictions_sealed_before_gate_label_read": True,
                    "gate_blind_prediction_relative_path": gate_blind_relative,
                    "gate_blind_prediction_sha256": gate_blind_sha,
                    "model_relative_path": gate_model_relative,
                    "model_sha256": _sha(gate_model_path),
                    "model_state_sha256": gate_model.model_state_sha256,
                    "saved_model_reload_prediction_exact": True,
                    "gate": gate_result,
                    "outer_validation_target_reads": 0,
                    "elapsed_seconds": stage["elapsed_seconds"],
                    "test_value_reads": 0,
                }
                gate_receipts.append(gate_receipt)
                all_receipts.append(gate_receipt)

                refit_started = time.perf_counter()
                refit_labels = pd.to_numeric(
                    working_train.iloc[refit_ids]["label"], errors="raise"
                ).to_numpy(np.int8)
                refit_model = engine.fit_incumbent_residual_model(
                    prefix_features,
                    layout,
                    refit_ids,
                    refit_labels,
                    teacher["seed_probability"][seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][seed],
                    context_ids=train_ids,
                    forbidden_ids=validation_ids,
                    seed=int(seed) + int(fold["ordinal"]),
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                )
                n_rows = len(working_train)
                outer_seed = np.full(n_rows, 0.5, dtype=np.float32)
                outer_mean = np.full(n_rows, 0.5, dtype=np.float32)
                outer_std = np.zeros(n_rows, dtype=np.float32)
                outer_decision = np.zeros(n_rows, dtype=np.int8)
                seed_column = f"baseline__seed_{seed}__probability"
                decision_column = f"baseline__seed_{seed}__prediction"
                outer_seed[validation_ids] = comparator[seed_column].to_numpy(np.float32)
                outer_mean[validation_ids] = comparator["baseline_probability"].to_numpy(
                    np.float32
                )
                comparator_seed_matrix = np.column_stack(
                    [
                        comparator[
                            f"baseline__seed_{registered}__probability"
                        ].to_numpy(np.float32)
                        for registered in SEEDS
                    ]
                )
                outer_std[validation_ids] = comparator_seed_matrix.std(axis=1).astype(
                    np.float32
                )
                outer_decision[validation_ids] = comparator[decision_column].to_numpy(np.int8)
                residual_probability = engine.predict_incumbent_residual_probability(
                    refit_model,
                    prediction_features,
                    layout,
                    validation_ids,
                    outer_seed,
                    outer_mean,
                    outer_std,
                    outer_decision,
                    context_ids=outer_context,
                    device="cuda",
                )
                model_relative = f"models/{fraction_tag}/{fold_name}/seed_{seed}.pt"
                model_path = engine._safe_path(paths["artifact"], model_relative)
                engine.save_fitted_incumbent_residual_model(refit_model, model_path)
                loaded = engine.load_fitted_incumbent_residual_model(model_path)
                reproduced = engine.predict_incumbent_residual_probability(
                    loaded,
                    prediction_features,
                    layout,
                    validation_ids,
                    outer_seed,
                    outer_mean,
                    outer_std,
                    outer_decision,
                    context_ids=outer_context,
                    device="cuda",
                )
                reload_exact = bool(np.array_equal(residual_probability, reproduced))
                if not reload_exact:
                    raise RuntimeError("saved Gen5r5 refit did not reproduce blind probability")
                final_probability = engine.exact_identity_or_residual(
                    comparator[seed_column].to_numpy(np.float32),
                    residual_probability,
                    gate_passed=bool(gate_result["passed"]),
                )
                if gate_result["passed"]:
                    final_prediction = engine.apply_postprocess(
                        working_train.iloc[validation_ids],
                        final_probability,
                        comparator["plateau"].to_numpy(bool),
                        comparator["spike_candidate"].to_numpy(bool),
                        config["fixed_fold_postprocess"][fold_name],
                    ).astype(np.int8)
                else:
                    final_prediction = comparator[decision_column].to_numpy(np.int8).copy()
                    if not (
                        np.array_equal(
                            final_probability,
                            comparator[seed_column].to_numpy(np.float32),
                        )
                        and np.array_equal(
                            final_prediction,
                            comparator[decision_column].to_numpy(np.int8),
                        )
                    ):
                        raise AssertionError("failed Gen5r5 gate changed incumbent seed")
                blind_relative = (
                    f"blind_predictions/{fraction_tag}/{fold_name}/seed_{seed}.npy"
                )
                blind_path = engine._safe_path(paths["artifact"], blind_relative)
                blind_sha = engine._npy_new(blind_path, final_probability)
                seed_blind_pins.append(
                    {"seed": seed, "path": blind_relative, "sha256": blind_sha}
                )
                seed_final_probabilities.append(final_probability)
                seed_final_predictions.append(final_prediction)
                completed_primary += 1
                primary_receipt = {
                    "role": "residual_refit",
                    "fraction": fraction,
                    "fold": fold_name,
                    "seed": seed,
                    "train_rows": int(len(refit_ids)),
                    "validation_rows": int(len(validation_ids)),
                    "optimizer_steps": training_config.optimizer_steps,
                    "raw_prefix_causal_feature_sha256": prefix_feature_sha,
                    "gate_passed": bool(gate_result["passed"]),
                    "failed_gate_exact_incumbent_identity": not gate_result["passed"],
                    "train_ids_sha256": refit_model.train_ids_sha256,
                    "validation_ids_sha256": engine.ids_sha256(validation_ids),
                    "model_relative_path": model_relative,
                    "model_sha256": _sha(model_path),
                    "model_state_sha256": refit_model.model_state_sha256,
                    "scaler_sha256": refit_model.scaler.state_sha256,
                    "blind_prediction_relative_path": blind_relative,
                    "blind_prediction_sha256": blind_sha,
                    "blind_prediction_sealed_before_validation_target_read": True,
                    "saved_model_reload_prediction_exact": reload_exact,
                    "elapsed_seconds": float(time.perf_counter() - refit_started),
                    "validation_target_reads": 0,
                    "test_value_reads": 0,
                }
                primary_receipts.append(primary_receipt)
                all_receipts.append(primary_receipt)
                engine._emit(
                    "gen5r5_primary_fit_complete",
                    completed=completed_primary,
                    total=45,
                    fraction=fraction,
                    fold=fold_name,
                    seed=seed,
                    gate_passed=bool(gate_result["passed"]),
                )
            part = comparator.copy()
            for seed, probability, prediction in zip(
                SEEDS,
                seed_final_probabilities,
                seed_final_predictions,
                strict=True,
            ):
                part[f"challenger__seed_{seed}__probability"] = probability
                part[f"challenger__seed_{seed}__prediction"] = prediction
            if not any(cell_gate_passes):
                mean_probability = comparator["baseline_probability"].to_numpy(
                    np.float32
                ).copy()
                mean_prediction = comparator["baseline_prediction"].to_numpy(np.int8).copy()
                if not (
                    np.array_equal(
                        mean_probability,
                        comparator["baseline_probability"].to_numpy(np.float32),
                    )
                    and np.array_equal(
                        mean_prediction,
                        comparator["baseline_prediction"].to_numpy(np.int8),
                    )
                ):
                    raise AssertionError("all-failed Gen5r5 gates changed incumbent ensemble")
            else:
                mean_probability = np.mean(
                    np.column_stack(seed_final_probabilities), axis=1
                ).astype(np.float32)
                mean_prediction = engine.apply_postprocess(
                    working_train.iloc[validation_ids],
                    mean_probability,
                    comparator["plateau"].to_numpy(bool),
                    comparator["spike_candidate"].to_numpy(bool),
                    config["fixed_fold_postprocess"][fold_name],
                ).astype(np.int8)
            part["challenger_probability"] = mean_probability
            part["challenger_prediction"] = mean_prediction
            part_relative = f"prediction_parts/{fold_name}_{fraction_tag}.parquet"
            part_path = engine._safe_path(paths["artifact"], part_relative)
            part_sha = engine._parquet_new(part_path, part)
            part_receipt = {
                "fraction": fraction,
                "fold": fold_name,
                "rows": int(len(part)),
                "path": part_relative,
                "sha256": part_sha,
                "gate_pass_count": int(sum(cell_gate_passes)),
                "key_order_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(
                        part.loc[:, [*engine.KEY_COLUMNS, "fold"]], index=False
                    ).to_numpy("<u8").tobytes()
                ).hexdigest(),
            }
            part_receipts.append(part_receipt)
            event = commitment.commit_cell(
                fold=fold_name,
                fraction=fraction,
                validation_ids_sha256=engine.ids_sha256(validation_ids),
                seed_blind_predictions=seed_blind_pins,
                prediction_part={
                    "path": part_relative,
                    "sha256": part_sha,
                    "key_order_sha256": part_receipt["key_order_sha256"],
                },
                target_accessor=targets,
            )
            cell_commitments.append(
                {"path": event["path"], "sha256": event["sha256"]}
            )
        if not commitment.is_fold_committed(fold_name):
            raise PermissionError("Gen5r5 fold did not reach its five-cell commitment")
    if not (
        len(teacher_receipts) == 135
        and len(gate_receipts) == 45
        and len(primary_receipts) == 45
        and len(all_receipts) == 225
        and sum(row["optimizer_steps"] for row in gate_receipts) == 5400
        and sum(row["optimizer_steps"] for row in primary_receipts) == 5400
        and len(part_receipts) == len(cell_commitments) == 15
    ):
        raise AssertionError("Gen5r5 curve fit, part, or optimizer-step count differs")
    ledger_receipt = commitment.finalize_ledger()
    completion = {
        "schema_version": "p1_incumbent_residual_predictions_complete.v5r5",
        "created_at": engine._now(),
        "fit_cells": 225,
        "teacher_fit_count": 135,
        "residual_gate_fit_count": 45,
        "residual_refit_count": 45,
        "optimizer_steps": 5400,
        "gate_optimizer_steps": 5400,
        "total_residual_optimizer_steps": 10800,
        "teacher_model_receipts": teacher_receipts,
        "gate_model_receipts": gate_receipts,
        "model_receipts": primary_receipts,
        "prediction_parts": part_receipts,
        "cell_commitments": cell_commitments,
        "fold_commitments": commitment.fold_receipts,
        "blind_commitment_ledger": {
            "path": ledger_receipt["path"],
            "sha256": ledger_receipt["sha256"],
        },
        "execution_order": "fold_major_then_prefix_fraction",
        "rolling_origin_reuse": (
            "earlier-fold validation labels become later-fold training history only after "
            "all five blind cells for the earlier fold are O_EXCL committed"
        ),
        "all_inner_and_outer_blind_predictions_sealed_before_their_target_reads": True,
        "aggregate_scores_computed_before_completion": 0,
        "frozen_oof_target_columns_decoded": 0,
        "train_anomaly_type_rows_decoded": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    completion_path = paths["artifact"] / "predictions_complete.json"
    _json_new(completion_path, completion)
    commitment.mark_predictions_complete(completion_path)
    state["stage"] = "BLIND_COMMITTED"
    return all_receipts, completion, working_train


def _full_fit_models(
    *,
    capability: _ExecutionCapability,
) -> dict[str, Any]:
    state, context = _require_execution_capability(
        capability,
        expected_stage="FULL_FIT_AUTHORIZED",
        transition_to="FULL_FIT_RUNNING",
    )
    runtime = state.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("labels_committed") is not True:
        raise PermissionError("Gen5r5 bound full-fit runtime is absent")
    return engine._full_fit_models(
        config=context.config,
        paths=context.paths,
        train=runtime["working_train"],
        features=runtime["features"],
        feature_columns=runtime["feature_columns"],
        layout=runtime["layout"],
    )


_ENGINE_JSON_NEW = engine._json_new


def _score_json_new(path: Path, value: Any) -> None:
    if isinstance(value, dict) and path.name == "learning_curve_evidence.json":
        value = {
            **value,
            "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT",
            "leakage_checks": {
                **value["leakage_checks"],
                "input_frame_loaded_without_label_or_anomaly_type": True,
                "frozen_oof_projected_to_keys_fold_prediction_only": True,
                "target_csv_indexed_as_opaque_byte_spans": True,
                "active_cell_validation_targets_undecoded_until_its_o_excl_commitment": True,
                "earlier_fold_validation_reused_only_after_all_five_fold_commitments": True,
                "aggregate_scoring_targets_undecoded_until_predictions_complete": True,
                "anomaly_type_scalars_never_decoded": True,
                "inner_teacher_fit_and_prediction_rows_disjoint": True,
                "inner_teacher_seven_day_purge_exact": True,
                "teacher_raw_rows_restricted_to_exact_outer_prefix": True,
                "residual_raw_context_future_half_zero_masked": True,
                "gate_predictions_sealed_before_gate_block_labels_read": True,
                "failed_gate_returns_exact_incumbent_seed_probability_and_prediction": True,
            },
            "reproducibility_checks": {
                **value["reproducibility_checks"],
                "fold_major_then_fraction_commitment_order_fixed": True,
                "fifteen_cell_and_three_fold_commitments_hash_chained": True,
                "predictions_complete_binds_all_forty_five_seed_blind_files": True,
                "fixed_135_teacher_45_gate_45_refit_curve_fits": True,
                "fixed_10800_total_residual_optimizer_steps": True,
                "all_teacher_and_residual_saved_models_reload_exact": True,
                "full_transitive_executable_closure_same_at_start_and_end": True,
            },
        }
    elif isinstance(value, dict) and path.name == "result.json":
        value = {
            **value,
            "operation_counters": {
                **value["operation_counters"],
                "curve_teacher_model_fits": 135,
                "curve_residual_gate_model_fits": 45,
                "curve_residual_refit_model_fits": 45,
                "curve_total_model_fits": 225,
                "curve_gate_optimizer_steps": 5400,
                "curve_refit_optimizer_steps": 5400,
                "curve_total_residual_optimizer_steps": 10800,
                "frozen_oof_target_columns_decoded": 0,
                "train_anomaly_type_rows_decoded": 0,
                "test_value_reads": 0,
                "uploads": 0,
            },
        }
    _ENGINE_JSON_NEW(path, value)


def _patch_pure_engine_writer() -> None:
    engine._json_new = _score_json_new


def _artifact_hashes(artifact: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(artifact).as_posix(): {
            "sha256": _sha(path),
            "bytes": int(path.stat().st_size),
        }
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }


def _run_after_lock(
    *,
    capability: _ExecutionCapability,
) -> dict[str, Any]:
    state, context = _require_execution_capability(
        capability,
        expected_stage="LOCK_CONSUMED",
        transition_to="PREPARING_RUNTIME",
    )
    root = context.root
    data_dir = context.data_dir
    overlay = context.overlay
    config = context.config
    paths = context.paths
    closure_before = context.closure
    anchors_before = context.anchors
    input_pins_before = context.input_pins
    qa_sha256 = state["qa_sha256"]
    authorization_sha256 = state["authorization_sha256"]
    lock_payload = _strict_json(paths["lock"])
    lock = {**lock_payload, "sha256": state["lock_sha256"]}
    if paths["artifact"].exists():
        raise FileExistsError("Gen5r5 artifact already exists")
    _patch_pure_engine_writer()
    if _verify_execution_closure(root, overlay) != closure_before:
        raise PermissionError("Gen5r5 closure changed after lock")
    if _capture_anchor_snapshot(root, data_dir, overlay) != anchors_before:
        raise PermissionError("Gen5r5 anchor snapshot changed after lock")
    pins_before = engine.verify_relative_input_pins(
        root,
        data_dir,
        config["immutable_inputs"],
    )
    if pins_before != input_pins_before:
        raise PermissionError("Gen5r5 immutable-input snapshot changed after lock")
    paths["artifact"].mkdir(parents=True, exist_ok=False)
    _json_new(
        paths["artifact"] / "preregistration.json",
        {
            "schema_version": "p1_gen5r5_runtime_preregistration.v1",
            "generation_id": overlay["experiment_id"],
            "config_path": CANONICAL_CONFIG,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "created_before_first_fit": True,
            "hypothesis_count": 1,
            "score_derived_tuning": False,
            "scientific_projection_sha256": SCIENCE_PROJECTION_SHA256,
            "owner_preregistration_sha256": _sha(paths["preregistration"]),
            "anchor_snapshot_sha256": _deep_sha(anchors_before),
            "model_fits": 0,
            "target_fold_scores": 0,
            "test_value_reads": 0,
            "candidate_files": 0,
            "uploads": 0,
        },
    )
    started = time.perf_counter()
    train = load_input_only_train(data_dir / "train.csv")
    if len(train) == 0 or {"label", "anomaly_type"}.intersection(train.columns):
        raise PermissionError("Gen5r5 input-only train load differs")
    feature_metadata = _strict_json(paths["feature_metadata"])
    feature_columns = [str(value) for value in config["features"]["selected_numeric_columns"]]
    if any(column not in feature_metadata["feature_columns"] for column in feature_columns):
        raise ValueError("registered Gen5r5 causal feature column is absent")
    feature_frame = pd.read_parquet(paths["feature_cache"], columns=feature_columns)
    if len(feature_frame) != len(train) or {"label", "anomaly_type"}.intersection(
        feature_frame.columns
    ):
        raise PermissionError("Gen5r5 feature cache row or target exclusion differs")
    features = feature_frame.to_numpy(np.float32)
    layout = engine.SequenceLayout.build(train.loc[:, ["station", "layer", "time"]])
    p1_config = engine.load_config(paths["base_config"])
    frozen_oof = load_frozen_oof_keys_only(paths["frozen_oof"])
    folds, scope_audit = engine._fold_runtime(
        train,
        p1_config,
        frozen_oof,
    )
    if tuple(str(fold["name"]) for fold in folds) != FOLD_ORDER:
        raise PermissionError("Gen5r5 frozen outer fold order differs")
    prefix_ids, prefix_audit = engine._pinned_label_free_prefixes(
        root,
        train,
        folds,
        int(config["features"]["cadence_minutes"]),
    )
    comparator_parts = engine._verify_gen1_parts(root, paths)
    validation_rows_by_fold = {
        str(fold["name"]): np.asarray(fold["val_idx"], dtype=np.int64) for fold in folds
    }
    targets = SelectiveTargetAccessor(
        data_dir / "train.csv",
        expected_sha256=config["immutable_inputs"]["train.csv"],
        expected_rows=len(train),
        validation_rows_by_fold=validation_rows_by_fold,
        fold_order=FOLD_ORDER,
    )
    if targets.decoded_target_scalars != 0:
        raise PermissionError("Gen5r5 target accessor decoded targets during indexing")
    _json_new(
        paths["artifact"] / "split_audit.json",
        {
            "folds": scope_audit,
            "prefixes": prefix_audit,
            "execution_order": "fold_major_then_prefix_fraction",
            "rolling_origin_reuse": True,
            "input_columns": list(train.columns),
            "input_target_columns_decoded": 0,
            "frozen_oof_columns": list(frozen_oof.columns),
            "frozen_oof_target_columns_decoded": 0,
            "target_accessor_opaque_index_rows": targets.row_count,
            "target_accessor_target_scalars_decoded_at_split_seal": 0,
        },
    )
    commitment = _BlindCommitmentLedger(paths["artifact"])
    state["runtime"] = {
        "sealed": True,
        "labels_committed": False,
        "train": train,
        "targets": targets,
        "commitment": commitment,
        "features": features,
        "feature_columns": feature_columns,
        "layout": layout,
        "folds": folds,
        "prefix_ids": prefix_ids,
        "comparator_parts": comparator_parts,
        "frozen_oof": frozen_oof,
    }
    state["stage"] = "BLIND_CURVE_READY"
    receipts, completion, working_train = _run_curve(
        capability=capability,
    )
    state, context = _require_execution_capability(
        capability,
        expected_stage="BLIND_COMMITTED",
    )
    full_ids = np.arange(len(train), dtype=np.int64)
    all_labels = targets.labels_for(
        full_ids,
        commitment=commitment,
        purpose="post_predictions_complete_scoring_and_optional_full_fit",
        active_fold=None,
        require_global=True,
    )
    working_train.iloc[:, working_train.columns.get_loc("label")] = all_labels
    state["runtime"]["working_train"] = working_train
    state["runtime"]["labels_committed"] = True
    if targets.decoded_anomaly_rows != 0:
        raise PermissionError("Gen5r5 decoded anomaly_type despite binary-only structure")
    target_audit = targets.audit()
    _json_new(paths["artifact"] / "selective_target_audit.json", target_audit)
    report, evidence, central = engine._score(
        root=root,
        config=config,
        paths=paths,
        train=working_train,
        frozen_oof=frozen_oof,
    )
    if central["passed"]:
        state["stage"] = "FULL_FIT_AUTHORIZED"
        full_fit = _full_fit_models(
            capability=capability,
        )
        next_generation = None
    else:
        state["stage"] = "SCORED_NO_PASS"
        full_fit = {
            "performed": False,
            "reason": "curve did not satisfy every preregistered meaningful-improvement gate",
            "model_count": 0,
            "test_value_reads": 0,
            "candidate_files": 0,
            "uploads": 0,
        }
        next_generation = config["on_no_pass"]["exactly_one_next_structural_diagnosis"]
    closure_after = _verify_execution_closure(root, overlay)
    if closure_before != closure_after:
        raise RuntimeError("Gen5r5 executable closure changed during the run")
    pins_after = engine.verify_relative_input_pins(
        root,
        data_dir,
        config["immutable_inputs"],
    )
    if pins_before != pins_after:
        raise RuntimeError("Gen5r5 immutable inputs changed during the run")
    anchors_after = _capture_anchor_snapshot(root, data_dir, overlay)
    if anchors_before != anchors_after:
        raise RuntimeError("Gen5r5 anchor identities changed during the run")
    current = _reload_context(context)
    verified_lock = _verify_consumed_lock(state, current)
    if verified_lock["sha256"] != lock["sha256"]:
        raise RuntimeError("Gen5r5 consumed attempt lock changed during the run")
    result = {
        "schema_version": "p1_incumbent_residual_result.v5r5",
        "experiment_id": overlay["experiment_id"],
        "completed_at": engine._now(),
        "status": (
            "CURVE_QUALIFIED_FULL_FIT_MODELS_SAVED"
            if central["passed"]
            else "RESEARCH_ONLY_NO_PASS"
        ),
        "decision": central["decision"],
        "passed": bool(central["passed"]),
        "hypothesis": HYPOTHESIS,
        "points": report["points"],
        "gate_checks": report["gate_checks"],
        "full_fit": full_fit,
        "exactly_one_next_structural_diagnosis": next_generation,
        "attempt": lock,
        "qa_receipt_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "prediction_completion_sha256": _sha(
            paths["artifact"] / "predictions_complete.json"
        ),
        "blind_commitment_ledger_sha256": completion["blind_commitment_ledger"][
            "sha256"
        ],
        "selective_target_audit_sha256": _sha(
            paths["artifact"] / "selective_target_audit.json"
        ),
        "operation_counters": {
            "curve_model_fits": len(receipts),
            "curve_optimizer_steps": completion["optimizer_steps"],
            "full_fit_model_fits": int(full_fit["model_count"]),
            "target_fold_scores": len(FRACTIONS),
            "frozen_oof_target_columns_decoded": 0,
            "train_anomaly_type_rows_decoded": 0,
            "test_value_reads": 0,
            "test_prediction_generations": 0,
            "candidate_files": 0,
            "uploads": 0,
            "source_mutations": 0,
            "frozen_mutations": 0,
        },
        "protected_input_sha256_unchanged": True,
        "executable_closure_sha256_unchanged": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    _score_json_new(paths["artifact"] / "result.json", result)
    registry = {
        "schema_version": "p1_incumbent_residual_registry.v5r5",
        "experiment_id": overlay["experiment_id"],
        "registered_at": engine._now(),
        "decision": central["decision"],
        "passed": bool(central["passed"]),
        "hypothesis": HYPOTHESIS,
        "learning_curve_evidence_sha256": _sha(
            paths["artifact"] / "learning_curve_evidence.json"
        ),
        "canonical_decision_sha256": _sha(
            paths["artifact"] / "canonical_curve_decision.json"
        ),
        "result_sha256": _sha(paths["artifact"] / "result.json"),
        "full_fit_models": full_fit.get("models"),
        "candidate": None,
        "test_value_reads": 0,
        "uploads": 0,
    }
    _json_new(paths["artifact"] / "registry.json", registry)
    manifest = {
        "schema_version": "p1_incumbent_residual_manifest.v5r5",
        "experiment_id": overlay["experiment_id"],
        "created_at": engine._now(),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "runner_sha256": _sha(Path(__file__).resolve()),
        "execution_closure_before": closure_before,
        "execution_closure_after": closure_after,
        "attempt_lock_path": CANONICAL_LOCK,
        "attempt_lock_sha256": lock["sha256"],
        "qa_receipt_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "immutable_inputs_before": pins_before,
        "immutable_inputs_after": pins_after,
        "anchor_snapshot_before": anchors_before,
        "anchor_snapshot_after": anchors_after,
        "source_cache_current_frozen_unchanged": True,
        "artifacts": _artifact_hashes(paths["artifact"]),
        "candidate_created": False,
        "uploaded": False,
    }
    _json_new(paths["artifact"] / "manifest.json", manifest)
    manifest_sha = _sha(paths["artifact"] / "manifest.json")
    with (paths["artifact"] / "manifest.sha256").open(
        "x", encoding="ascii", newline="\n"
    ) as handle:
        handle.write(f"{manifest_sha}  manifest.json\n")
    final = {
        "status": result["status"],
        "decision": central["decision"],
        "passed": bool(central["passed"]),
        "artifact": CANONICAL_ARTIFACT,
        "metrics_sha256": _sha(paths["artifact"] / "metrics.json"),
        "evidence_sha256": _sha(paths["artifact"] / "learning_curve_evidence.json"),
        "result_sha256": _sha(paths["artifact"] / "result.json"),
        "manifest_sha256": manifest_sha,
        "candidate_sha256": None,
        "test_value_reads": 0,
        "uploads": 0,
        "elapsed_seconds": result["elapsed_seconds"],
    }
    engine._emit("gen5r5_generation_complete", **final)
    state["stage"] = "COMPLETE"
    del _LIVE_EXECUTION[capability._token]
    return final


def check_only(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    context = _load_operational_context(
        environ=environ,
        require_owner_seals=True,
    )
    root = context.root
    data_dir = context.data_dir
    overlay = context.overlay
    config = context.config
    paths = context.paths
    closure = context.closure
    anchors = context.anchors
    input_pins = context.input_pins
    if paths["artifact"].exists() or paths["lock"].exists():
        raise FileExistsError("Gen5r5 artifact or attempt lock already exists")
    owner_seals = context.owner_seals
    comparator = engine._verify_gen1_parts(root, paths)
    train = load_input_only_train(data_dir / "train.csv")
    frozen_oof = load_frozen_oof_keys_only(paths["frozen_oof"])
    p1_config = engine.load_config(paths["base_config"])
    folds, scope_audit = engine._fold_runtime(train, p1_config, frozen_oof)
    prefix_ids, prefix_audit = engine._pinned_label_free_prefixes(
        root,
        train,
        folds,
        int(config["features"]["cadence_minutes"]),
    )
    validation_rows_by_fold = {
        str(fold["name"]): np.asarray(fold["val_idx"], dtype=np.int64)
        for fold in folds
    }
    targets = SelectiveTargetAccessor(
        data_dir / "train.csv",
        expected_sha256=config["immutable_inputs"]["train.csv"],
        expected_rows=len(train),
        validation_rows_by_fold=validation_rows_by_fold,
        fold_order=FOLD_ORDER,
    )
    split_cells: list[dict[str, Any]] = []
    for fold in folds:
        fold_name = str(fold["name"])
        validation_ids = np.asarray(fold["val_idx"], dtype=np.int64)
        for fraction in FRACTIONS:
            train_ids = prefix_ids[(fold_name, fraction)]
            splits = engine.build_three_block_inner_splits(
                train.loc[:, ["time"]],
                train_ids,
                purge_days=int(config["inner_cross_fit"]["purge_days"]),
            )
            if (
                len(splits) != 3
                or np.intersect1d(train_ids, validation_ids).size
                or any(
                    np.intersect1d(
                        split.teacher_train_ids,
                        split.teacher_prediction_ids,
                    ).size
                    for split in splits
                )
            ):
                raise PermissionError("Gen5r5 static split firewall differs")
            split_cells.append(
                {
                    "fold": fold_name,
                    "fraction": fraction,
                    "prefix_rows": int(len(train_ids)),
                    "validation_rows": int(len(validation_ids)),
                    "inner_split_count": len(splits),
                    "active_validation_target_decodes": 0,
                }
            )
    if targets.decoded_target_scalars != 0 or targets.decoded_anomaly_rows != 0:
        raise PermissionError("Gen5r5 check-only decoded a target scalar")
    causal_audit = engine.causal_feature_audit(
        train,
        cached_path=paths["feature_cache"],
    )
    ledger_binding = engine._verify_v5_ledger_binding(
        root,
        config,
        paths["ledger"],
    )

    missing_qa_rejected = False
    try:
        authorize_entry(environ=environ)
    except PermissionError:
        missing_qa_rejected = True
    if not missing_qa_rejected:
        raise PermissionError("Gen5r5 authorize_entry did not reject missing QA/auth")

    direct_guards: dict[str, bool] = {}
    guarded_calls = {
        "prelock_requirement_rejected_without_capability": lambda: (
            _require_prelock_capability(
                None,
                expected_stage="QA_AUTH_VERIFIED_PRELOCK",
            )
        ),
        "lock_acquisition_rejected_without_capability": lambda: _acquire_lock(
            object()
        ),
        "postlock_requirement_rejected_without_capability": lambda: (
            _require_execution_capability(
                None,
                expected_stage="LOCK_CONSUMED",
            )
        ),
        "run_after_lock_rejected_without_capability": lambda: _run_after_lock(
            capability=object()  # type: ignore[arg-type]
        ),
        "curve_rejected_without_capability": lambda: _run_curve(
            capability=object()  # type: ignore[arg-type]
        ),
        "full_fit_rejected_without_capability": lambda: _full_fit_models(
            capability=object()  # type: ignore[arg-type]
        ),
    }
    for name, call in guarded_calls.items():
        try:
            call()
        except PermissionError:
            direct_guards[name] = True
        else:
            raise PermissionError(f"Gen5r5 direct-call guard failed: {name}")
    if _LIVE_PRELOCK or _LIVE_EXECUTION:
        raise PermissionError("Gen5r5 check-only left a live capability")
    if paths["lock"].exists() or paths["artifact"].exists():
        raise PermissionError("Gen5r5 check-only created an execution output")

    control_state = {
        "preregistration": paths["preregistration"].is_file(),
        "owner_preseal": paths["preseal"].is_file(),
        "independent_qa_receipt": paths["qa_receipt"].exists(),
        "execution_authorization": paths["authorization"].exists(),
        "attempt_lock": paths["lock"].exists(),
    }
    if control_state != {
        "preregistration": True,
        "owner_preseal": True,
        "independent_qa_receipt": False,
        "execution_authorization": False,
        "attempt_lock": False,
    }:
        raise PermissionError("Gen5r5 static-only control state differs")
    return {
        "status": "CANONICAL_GEN5R5_CHECK_ONLY_PASS",
        "experiment_id": overlay["experiment_id"],
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "runner_sha256": _sha(paths["runner"]),
        "scientific_projection_sha256": SCIENCE_PROJECTION_SHA256,
        "scientific_projection_deep_sha256": overlay["scientific_projection"][
            "science_deep_sha256"
        ],
        "execution_closure_sha256": closure,
        "execution_closure_count": len(closure),
        "anchor_snapshot_sha256": _deep_sha(anchors),
        "anchor_files_single_link_non_reparse": all(
            value["nlink"] == 1 and value["non_reparse"] is True
            for value in anchors["files"].values()
        ),
        "runner_and_config_single_link_non_reparse": (
            anchors["runner"]["nlink"] == 1
            and anchors["config"]["nlink"] == 1
            and anchors["runner"]["non_reparse"] is True
            and anchors["config"]["non_reparse"] is True
        ),
        "environment_contract": overlay["environment_contract"],
        "owner_seals": owner_seals,
        "r4_tombstone": _verify_r4_tombstone(paths),
        "r4_source_content_equivalence": context.r4_content_equivalence,
        "data_identity_observations": context.identity_observations,
        "immutable_input_snapshot_sha256": _deep_sha(input_pins),
        "input_only_columns": list(train.columns),
        "input_target_columns_decoded": 0,
        "frozen_oof_columns": list(frozen_oof.columns),
        "frozen_oof_target_columns_decoded": 0,
        "opaque_target_index_rows": targets.row_count,
        "opaque_target_index_decoded_scalars": targets.decoded_target_scalars,
        "outer_fold_scope": scope_audit,
        "prefix_audit": prefix_audit,
        "split_cells": split_cells,
        "split_cell_count": len(split_cells),
        "execution_order": "fold_major_then_prefix_fraction",
        "planned_cell_commitments": 15,
        "planned_fold_commitments": 3,
        "planned_seed_blind_predictions": 45,
        "curve_fit_cells": 225,
        "curve_total_residual_optimizer_steps": 10800,
        "maximum_full_fit_fits_on_pass": 18,
        "gen1_comparator_parts": len(comparator),
        "causal_feature_audit": causal_audit,
        "v5_ledger_binding": ledger_binding,
        "authorize_entry_missing_qa_auth_rejected_before_capability_mint": (
            missing_qa_rejected
        ),
        "opaque_capability_direct_call_guards": direct_guards,
        "live_prelock_capability_count": len(_LIVE_PRELOCK),
        "live_postlock_capability_count": len(_LIVE_EXECUTION),
        "lock_api_caller_supplied_sha_parameters": 0,
        "lock_binds_full_operational_snapshot": True,
        "postlock_capability_requires_reloaded_qa_auth_and_consumed_lock": True,
        "control_files_present": control_state,
        "artifact_absent": not paths["artifact"].exists(),
        "attempt_lock_absent": not paths["lock"].exists(),
        "model_fits": 0,
        "target_fold_scores": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }


def run_experiment(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    prelock_capability = authorize_entry(environ=environ)
    execution_capability = _acquire_lock(prelock_capability)
    return _run_after_lock(capability=execution_capability)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.check_only:
        result = check_only()
    elif args.run:
        result = run_experiment()
    else:
        raise AssertionError("unreachable Gen5r5 mode")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
