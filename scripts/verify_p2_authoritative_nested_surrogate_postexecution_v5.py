#!/usr/bin/env python
"""Fail-closed post-terminal verifier for the sealed P2 v5 execution.

The verifier is deliberately separate from the training runner.  It never
loads model/checkpoint payloads and never opens official P2 test, sample, or
submission files.  Aggregate prediction values are read only from the five
terminal ``evaluated_oof_*.parquet`` files, and only after the immutable start,
terminal result, terminal receipt, authorization, and execution binding have
all been verified.

No report is published unless every check succeeds.  The command-line entry
point additionally requires ``--post-terminal`` and the canonical sealed v5
paths, so importing or dry-running this module cannot inspect a live namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep the verifier independent of every mutable training module.  These
# literals are re-derived from the sealed plan and checked against it below;
# no project module is imported before its bytes have been authenticated.
TARGET_LAYERS = (2, 3, 4)
COMPONENTS = (
    "router_400",
    "depth_query_bitcn",
    "moment_units_scratch",
    "lsti_style",
    "timemixerpp_style",
)
OUTER_KEY_COLUMNS = ("fold", "station", "layer", "time")
FAMILY_SETTINGS = (
    "INCUMBENT_NOOP",
    "STACK_W0500",
    "STACK_W0625",
    "STACK_W0750",
    "CAUSAL_RESIDUAL_SCALE025",
    "FALLBACK_BLEND50_A0625",
    "CAUSAL_ON_FALLBACK",
)
STATE_FEATURES = (
    "abs_t1_t5",
    "public_temp_range",
    "public_psal_range",
    "contrast_delta_past_24h",
    "temp_range_delta_past_24h",
    "contrast_center_change_24h",
    "temp_range_center_change_24h",
    "m2_amplitude_mean",
    "m2_amplitude_spread",
    "m2_phase_coherence",
)
SEED_NAMESPACE = "P2_AUTH_NESTED_V1"
GATE_REGULARIZATION = 10.0

CONFIG_RELATIVE = "configs/experiments/p2_authoritative_nested_surrogate_execution_20260825_v5.json"
READY_RELATIVE = "artifacts/p2_authoritative_nested_surrogate_execution_ready_20260825_v5"
ACTUAL_RELATIVE = "artifacts/p2_authoritative_nested_surrogate_actual_20260825_v5"
OUTPUT_RELATIVE = "artifacts/p2_authoritative_nested_surrogate_postexecution_qa_20260825_v5"
VERIFIER_RELATIVE = "scripts/verify_p2_authoritative_nested_surrogate_postexecution_v5.py"
TEST_RELATIVE = "tests/test_verify_p2_authoritative_nested_surrogate_postexecution_v5.py"

AUTHORIZATION_NAME = "EXECUTION_AUTHORIZATION.json"
AUTHORIZATION_SCHEMA = "p2_authoritative_nested_surrogate_execution_authorization.v5"
TERMINAL_STATUS = "COMPLETE_LOCAL_AUTHORITATIVE_SURROGATE_V5_NO_PROMOTION"
REGISTERED_LAYERS = (2, 3, 4)
EXPECTED_SEEDS = (20260823, 20260824, 20260825)
EXPECTED_FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
EXPECTED_TOKENS = ("040", "055", "070", "085", "100")
EXPECTED_OUTER_FOLDS = (
    "outer_2024_sep_oct",
    "outer_2025_may_jun",
    "outer_2025_jul_aug",
)
EXPECTED_ROWS_PER_FRACTION = 78_156
EXPECTED_FOLD_ROWS = {
    "outer_2024_sep_oct": 26_167,
    "outer_2025_may_jun": 25_338,
    "outer_2025_jul_aug": 26_651,
}
EXPECTED_LAYER_ROWS = {2: 26_085, 3: 26_050, 4: 26_021}
EXPECTED_SCOPE_COUNT = 15
EXPECTED_CELL_COUNT = 45
EXPECTED_JOB_COUNT = 900
EXPECTED_BASE_FITS = 1_440
EXPECTED_DEEP_FITS = 720
EXPECTED_LIGHTGBM_FITS = 720
EXPECTED_META_OPTIMIZATIONS = 405
MAXIMUM_RESUME_ATTEMPTS = 2
MAXIMUM_TOTAL_ATTEMPTS = 3

EXPECTED_RESULT_FIELDS = {
    "status",
    "outer_prefix_cells",
    "seeded_cells",
    "component_jobs_new_this_invocation",
    "component_jobs_reused_this_invocation",
    "cell_jobs_new_this_invocation",
    "cell_jobs_reused_this_invocation",
    "metrics_by_prefix",
    "top_level_component_jobs_total",
    "underlying_base_estimator_fits_total",
    "underlying_deep_fits_total",
    "underlying_lightgbm_fits_total",
    "meta_optimizations_total",
    "same_population_digest_across_fractions",
    "submission_files_generated",
    "uploads",
    "scientific_surface_inherited_byte_pinned_from_v3",
    "resume_or_result_based_tuning_performed",
    "execution_contract_revision",
    "v4_resume_engine_byte_pinned",
    "foreign_v1_v2_v3_v4_job_or_cell_reuse",
    "execution_binding_sha256",
    "preexecution_seal_sha256",
    "semantic_preflight_sha256",
    "initial_start_attempt_count",
    "resume_attempts_started",
    "total_attempts_started",
    "official_test_sample_submission_reads",
}

# These are the independently QAed pre-execution v5 bytes.  The post-verifier
# must not silently follow a later mutable manifest or runner.
SEALED_SHA256 = {
    "config": "a954511ed0a01ef08d7c3762d0444bad9f00d8bb59b084cd0be15f178b74d2a0",
    "recipe": "3ff52057567185f40537c182032b8d4079609fefda132535526e4762c362a520",
    "recipe_seal": "17b6d613c772c17e799b8684133673c600cca7befa8b6e2292ccd398f7e001a6",
    "module_v5": "c2e4ca7a624eea003f7212e4f034e63e8b026930b55229cd493b38a8a45143e3",
    "resume_engine_v4": "cfe48c1fd4566feca620724cf4616ec0cbdf2bf2d8ae7a5896f9038d69638967",
    "runner_v5": "a472101ffe5cc22881658266ac0208a85445d13347aa670970f865b9bffa5201",
    "preexecution_seal": "6b10fde914d540138017d2c68d4ee22c22b4a8f56e3ecc6441cc2826ee780485",
    "semantic_preflight": "f47869d2900375ed0c0b8bbc6e9e03cdaf7cf69a233bfb3bb28251e3a696b761",
    "exact_command": "356ab94fd27e7adf1733790e68de9f4f287f671186225de536339c8ee9672bf7",
    "readiness_manifest": "1355eac26f6f4257817472585b1401ee81b093db18e40ce76c1fa7e1709b4bad",
    # V3 inherits this v2 science engine.  Its hash is independently sealed in
    # the v2 readiness manifest even though the later 30-file v5 ledger omitted
    # the transitive module; post-QA therefore pins it explicitly.
    "base_execution_v2": "10b3890d96f12a4e3afc4d752533c01c9365cf4c1172972383d7508ebdea3729",
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RESUME_NAME = re.compile(r"^resume_attempt_(\d{3})\.json$")
_ATTEMPT_TERMINAL_NAME = re.compile(r"^attempt_(\d{3})_terminal\.json$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def child_seed(
    complete_seed: int,
    component: str,
    outer_fold: str,
    prefix_fraction: float,
    inner_fold_or_full: str,
) -> int:
    _require(component in COMPONENTS, f"unknown component: {component}")
    _require(
        inner_fold_or_full in {"inner_1", "inner_2", "inner_3", "full"},
        "unknown child-seed phase",
    )
    preimage = "|".join(
        (
            SEED_NAMESPACE,
            str(int(complete_seed)),
            component,
            outer_fold,
            f"{float(prefix_fraction):.2f}",
            inner_fold_or_full,
        )
    )
    return int(hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16], 16) % 2_147_483_647


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"unsafe or missing JSON: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def _require_sha(value: object, message: str) -> str:
    digest = str(value)
    _require(_HEX64.fullmatch(digest) is not None, message)
    return digest


def _require_nonnegative_int(value: object, message: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, message)
    return int(value)


def _require_finite_float(value: object, message: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    _require(math.isfinite(converted), message)
    return converted


def _verify_self_hash(value: Mapping[str, Any], field: str, message: str) -> str:
    preimage = dict(value)
    claimed = _require_sha(preimage.pop(field, ""), message)
    _require(canonical_sha256(preimage) == claimed, message)
    return claimed


def _resolve_under(root: Path, relative: str) -> Path:
    root = root.resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    return path


def _verify_file_pin(path: Path, pin: Mapping[str, Any], label: str) -> dict[str, Any]:
    _require(isinstance(pin, Mapping) and set(pin) == {"sha256", "bytes"}, f"{label} pin changed")
    _require(path.is_file() and not path.is_symlink(), f"{label} is unsafe or missing")
    size = path.stat().st_size
    digest = sha256_file(path)
    _require_nonnegative_int(pin["bytes"], f"{label} byte pin invalid")
    _require_sha(pin["sha256"], f"{label} SHA-256 pin invalid")
    _require(size == pin["bytes"], f"{label} bytes changed")
    _require(digest == pin["sha256"], f"{label} SHA-256 changed")
    return {"bytes": size, "sha256": digest}


def _verify_atomic_publish_receipt(
    value: object,
    *,
    expected_sha256: str,
    expected_bytes: int,
    label: str,
    allow_finalization_recovery: bool = False,
) -> None:
    _require(isinstance(value, dict), f"{label} atomic publish receipt missing")
    status = value.get("status")
    common = {"status", "sha256", "bytes", "partial_created", "partial_policy"}
    if status == "COMMITTED_BY_FSYNC_AND_ATOMIC_RENAME":
        _require(set(value) == {*common, "partial_consumed_by_rename"}, f"{label} atomic receipt fields changed")
        _require(value.get("partial_created") is True, f"{label} atomic partial flag changed")
        _require(value.get("partial_consumed_by_rename") is True, f"{label} atomic rename was not consumed")
        _require(
            value.get("partial_policy")
            == "SUCCESSFUL_CURRENT_PARTIAL_CONSUMED_OLD_PARTIALS_PRESERVED",
            f"{label} atomic policy changed",
        )
    elif status == "REUSED_VERIFIED_FINAL":
        _require(set(value) == common, f"{label} reused receipt fields changed")
        _require(value.get("partial_created") is False, f"{label} reused receipt created a partial")
        _require(
            value.get("partial_policy")
            == "PREEXISTING_PARTIALS_IGNORED_AND_PRESERVED_FOR_AUDIT",
            f"{label} reused policy changed",
        )
    elif status == "RECOVERED_VERIFIED_ATOMIC_FINAL" and allow_finalization_recovery:
        _require(set(value) == common, f"{label} recovered receipt fields changed")
        _require(value.get("partial_created") is False, f"{label} recovery created a partial")
        _require(
            value.get("partial_policy")
            == "STALE_PARTIALS_IGNORED_AND_PRESERVED_FOR_AUDIT",
            f"{label} recovery policy changed",
        )
    else:
        raise ValueError(f"{label} was not committed/reused without a racing partial")
    _require(value.get("sha256") == expected_sha256, f"{label} atomic SHA differs")
    _require(value.get("bytes") == expected_bytes, f"{label} atomic bytes differ")


@dataclass(frozen=True)
class ScopeSpec:
    scope_id: str
    outer_fold: str
    fraction: float
    cutoff_kst: str
    inner_folds: tuple[str, ...]
    prefix_time_count: int = 0
    inner_details: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    scope_id: str
    outer_fold: str
    fraction: float
    cutoff_kst: str
    pipeline_seed: int
    phase: str
    component: str
    child_seed: int
    train_last_kst: str | None = None
    validation_start_kst: str | None = None
    embargo_threshold_kst: str | None = None
    training_supervised_time_count: int | None = None
    validation_supervised_time_count: int | None = None
    prefix_supervised_time_count: int | None = None


@dataclass(frozen=True)
class CellSpec:
    directory_id: str
    receipt_id: str
    scope_id: str
    outer_fold: str
    fraction: float
    cutoff_kst: str
    pipeline_seed: int
    inner_folds: tuple[str, ...]
    expected_outer_rows: int = 0
    expected_inner_oof_rows: int = 0


@dataclass(frozen=True)
class PostExecutionContract:
    namespace: str
    binding: dict[str, Any]
    scopes: tuple[ScopeSpec, ...]
    jobs: dict[str, JobSpec]
    cells: dict[str, CellSpec]
    seeds: tuple[int, ...]
    components: tuple[str, ...]
    layers: tuple[int, ...]
    outer_folds: tuple[str, ...]
    tokens: tuple[str, ...]
    family_settings: tuple[str, ...]
    expected_rows_per_fraction: int
    expected_fold_rows: dict[str, int]
    expected_layer_rows: dict[int, int]
    expected_fold_layer_rows: dict[str, dict[int, int]]
    expected_outer_key_sha256: dict[str, str]
    expected_base_fits: int
    expected_deep_fits: int
    expected_lightgbm_fits: int
    expected_meta_optimizations: int
    readiness_receipt: dict[str, Any]


def _verify_authorization(
    path: Path,
    *,
    preexecution_seal_sha256: str,
    exact_command_sha256: str,
) -> dict[str, Any]:
    value = _read_json(path)
    _require(
        set(value)
        == {
            "schema_version",
            "status",
            "training_authorized",
            "authorized_at_kst",
            "authorization_basis",
            "preexecution_seal_sha256",
            "exact_command_sha256",
            "same_contract_crash_resume_authorized",
            "maximum_automatic_resume_attempts_after_initial",
            "maximum_total_attempts",
            "result_based_rerun_or_tuning_authorized",
            "v1_v2_v3_v4_job_or_cell_reuse_authorized",
            "single_scientific_contract_only",
            "single_execution_only",
            "official_test_access_authorized",
            "sample_submission_access_authorized",
            "submission_generation_authorized",
            "public_score_selection_authorized",
            "upload_authorized",
            "p3_process_mutation_authorized",
        },
        "authorization field set changed",
    )
    _require(value.get("schema_version") == AUTHORIZATION_SCHEMA, "authorization schema changed")
    _require(value.get("status") == "APPROVED_EXACT_P2_45_CELL_COMMAND", "authorization status changed")
    _require(value.get("training_authorized") is True, "training authorization is absent")
    authorized_at = datetime.fromisoformat(str(value.get("authorized_at_kst")))
    _require(authorized_at.tzinfo is not None, "authorization timestamp lacks timezone")
    _require(isinstance(value.get("authorization_basis"), str) and value["authorization_basis"], "authorization basis missing")
    _require(
        value.get("preexecution_seal_sha256") == preexecution_seal_sha256,
        "authorization seal binding changed",
    )
    _require(
        value.get("exact_command_sha256") == exact_command_sha256,
        "authorization command binding changed",
    )
    for key in (
        "official_test_access_authorized",
        "sample_submission_access_authorized",
        "submission_generation_authorized",
        "public_score_selection_authorized",
        "upload_authorized",
        "p3_process_mutation_authorized",
        "result_based_rerun_or_tuning_authorized",
        "v1_v2_v3_v4_job_or_cell_reuse_authorized",
    ):
        _require(value.get(key) is False, f"authorization expands forbidden scope: {key}")
    _require(value.get("same_contract_crash_resume_authorized") is True, "crash resume not authorized")
    _require(
        value.get("maximum_automatic_resume_attempts_after_initial") == MAXIMUM_RESUME_ATTEMPTS,
        "authorization resume budget changed",
    )
    _require(value.get("maximum_total_attempts") == MAXIMUM_TOTAL_ATTEMPTS, "total attempt budget changed")
    _require(value.get("single_scientific_contract_only") is True, "authorization is not single-contract")
    _require(value.get("single_execution_only") is False, "authorization contradicts bounded resume")
    return value


def _verify_readiness_manifest(
    project_root: Path,
    readiness_dir: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    project_root = project_root.resolve(strict=True)
    readiness_dir = readiness_dir.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    _require(readiness_dir.is_dir() and not readiness_dir.is_symlink(), "readiness directory is unsafe")
    _require(
        sha256_file(readiness_dir / "manifest.json") == SEALED_SHA256["readiness_manifest"],
        "readiness manifest is not the independently sealed v5 manifest",
    )
    manifest = _read_json(readiness_dir / "manifest.json")
    _require(
        manifest.get("schema_version")
        == "p2_authoritative_nested_surrogate_execution_ready_manifest.v5",
        "readiness manifest schema changed",
    )
    _require(manifest.get("status") == "EXECUTION_READY_NOT_AUTHORIZED", "readiness status changed")

    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict) and len(outputs) == 14, "readiness output ledger changed")
    expected_entries = {"manifest.json", AUTHORIZATION_NAME, *outputs.keys()}
    actual_entries = {path.name for path in readiness_dir.iterdir()}
    _require(actual_entries == expected_entries, "unexpected readiness entry after authorization")
    _require(all(".partial." not in name for name in actual_entries), "readiness partial exists")
    for name, pin in outputs.items():
        _verify_file_pin(readiness_dir / name, pin, f"readiness output {name}")

    config_sha = sha256_file(config_path)
    _require(config_sha == SEALED_SHA256["config"], "v5 config changed")
    _require(manifest["config"]["sha256"] == config_sha, "manifest config pin changed")
    _require(
        _resolve_under(project_root, str(manifest["config"]["path"])) == config_path,
        "manifest config path changed",
    )
    config = _read_json(config_path)
    _require(config.get("schema_version") == "p2_authoritative_nested_surrogate_execution.v5", "config schema changed")
    _require(config.get("status") == "SEALED_EXECUTION_READY_DRY_RUN_NO_ACTUAL_FIT", "config status changed")

    module_path = _resolve_under(project_root, str(manifest["module"]["path"]))
    runner_path = _resolve_under(project_root, str(manifest["runner"]["path"]))
    base_execution_path = _resolve_under(
        project_root, "src/p2_restore/authoritative_nested_surrogate_execution.py"
    )
    _require(sha256_file(module_path) == SEALED_SHA256["module_v5"], "v5 module changed")
    _require(sha256_file(runner_path) == SEALED_SHA256["runner_v5"], "v5 runner changed")
    _require(manifest["module"]["sha256"] == SEALED_SHA256["module_v5"], "manifest module pin changed")
    _require(manifest["runner"]["sha256"] == SEALED_SHA256["runner_v5"], "manifest runner pin changed")
    _require(
        sha256_file(base_execution_path) == SEALED_SHA256["base_execution_v2"],
        "transitive v2 science engine changed",
    )

    preseal_path = readiness_dir / "preexecution_seal.json"
    preseal_sha = sha256_file(preseal_path)
    _require(preseal_sha == SEALED_SHA256["preexecution_seal"], "preexecution seal changed")
    preseal = _read_json(preseal_path)
    _require(
        preseal.get("schema_version") == "p2_authoritative_nested_surrogate_preexecution_seal.v5",
        "preexecution seal schema changed",
    )
    _require(preseal.get("config_sha256") == config_sha, "preexecution config pin changed")
    _require(preseal.get("module_sha256") == SEALED_SHA256["module_v5"], "preexecution module pin changed")
    _require(preseal.get("runner_sha256") == SEALED_SHA256["runner_v5"], "preexecution runner pin changed")
    _require(
        preseal.get("v4_resume_engine_sha256") == SEALED_SHA256["resume_engine_v4"],
        "resume engine pin changed",
    )
    _require(
        preseal.get("semantic_preflight_sha256") == SEALED_SHA256["semantic_preflight"],
        "semantic preflight pin changed",
    )
    exact_command_sha = hashlib.sha256(str(config["exact_command"]).encode()).hexdigest()
    _require(exact_command_sha == SEALED_SHA256["exact_command"], "exact command changed")
    _require(preseal.get("exact_command_sha256") == exact_command_sha, "preexecution command pin changed")
    _require(
        preseal.get("required_authorization", {}).get("schema_version") == AUTHORIZATION_SCHEMA,
        "preexecution authorization schema changed",
    )

    parent = config["parent_contract"]
    recipe_path = _resolve_under(project_root, str(parent["path"]))
    recipe_seal_path = _resolve_under(project_root, str(parent["contract_seal_path"]))
    _require(sha256_file(recipe_path) == SEALED_SHA256["recipe"], "parent recipe changed")
    _require(sha256_file(recipe_seal_path) == SEALED_SHA256["recipe_seal"], "recipe seal changed")
    _require(parent["sha256"] == SEALED_SHA256["recipe"], "config parent recipe pin changed")
    _require(parent["contract_seal_sha256"] == SEALED_SHA256["recipe_seal"], "config recipe seal pin changed")

    static = _read_json(readiness_dir / "static_verification.json")
    pins = static.get("verified_inputs")
    _require(isinstance(pins, dict) and len(pins) == 30, "recursive source pin ledger changed")
    for relative, pin in pins.items():
        _verify_file_pin(_resolve_under(project_root, str(relative)), pin, f"recursive pin {relative}")

    semantic = _read_json(readiness_dir / "actual_data_semantic_preflight.json")
    semantic_preimage = dict(semantic)
    semantic_claim = _require_sha(
        semantic_preimage.pop("semantic_receipt_sha256", ""), "semantic self-hash missing"
    )
    _require(canonical_sha256(semantic_preimage) == semantic_claim, "semantic self-hash changed")
    _require(semantic_claim == SEALED_SHA256["semantic_preflight"], "semantic identity changed")
    semantic_outer = semantic.get("outer_validation_by_fold")
    _require(isinstance(semantic_outer, dict), "semantic outer population ledger missing")
    _require(set(semantic_outer) == set(EXPECTED_OUTER_FOLDS), "semantic outer folds changed")
    for fold, row in semantic_outer.items():
        _require(isinstance(row, dict), f"semantic outer row invalid: {fold}")
        _require(row.get("router_deep_key_support_equal") is True, f"semantic router/deep keys differ: {fold}")
        _require(row.get("rows") == EXPECTED_FOLD_ROWS[fold], f"semantic outer rows changed: {fold}")
        _require_sha(row.get("ordered_key_sha256"), f"semantic outer key hash missing: {fold}")
        rows_by_layer = row.get("rows_by_layer")
        _require(
            isinstance(rows_by_layer, dict)
            and set(rows_by_layer) == {str(layer) for layer in REGISTERED_LAYERS}
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in rows_by_layer.values()
            )
            and sum(rows_by_layer.values()) == row["rows"],
            f"semantic fold/layer population changed: {fold}",
        )
    _require(
        {
            layer: sum(
                int(semantic_outer[fold]["rows_by_layer"][str(layer)])
                for fold in EXPECTED_OUTER_FOLDS
            )
            for layer in REGISTERED_LAYERS
        }
        == EXPECTED_LAYER_ROWS,
        "semantic global layer population changed",
    )

    authorization_path = readiness_dir / AUTHORIZATION_NAME
    _verify_authorization(
        authorization_path,
        preexecution_seal_sha256=preseal_sha,
        exact_command_sha256=exact_command_sha,
    )
    return config, manifest, preseal, {
        "authorization_sha256": sha256_file(authorization_path),
        "config_sha256": config_sha,
        "exact_command_sha256": exact_command_sha,
        "module_sha256": sha256_file(module_path),
        "transitive_base_execution_sha256": sha256_file(base_execution_path),
        "runner_sha256": sha256_file(runner_path),
        "preexecution_seal_sha256": preseal_sha,
        "readiness_manifest_sha256": sha256_file(readiness_dir / "manifest.json"),
        "recipe_sha256": sha256_file(recipe_path),
        "recipe_seal_sha256": sha256_file(recipe_seal_path),
        "semantic_preflight_sha256": semantic_claim,
        "semantic_outer_validation_by_fold": semantic_outer,
        "recursive_source_pins_verified": len(pins),
    }


def load_sealed_contract(
    *,
    project_root: Path = PROJECT_ROOT,
    readiness_dir: Path | None = None,
    config_path: Path | None = None,
) -> PostExecutionContract:
    """Load only the known pre-execution v5 contract; never inspect actual output."""

    project_root = project_root.resolve(strict=True)
    canonical_ready = (project_root / READY_RELATIVE).resolve(strict=True)
    canonical_config = (project_root / CONFIG_RELATIVE).resolve(strict=True)
    selected_ready = (readiness_dir or canonical_ready).resolve(strict=True)
    selected_config = (config_path or canonical_config).resolve(strict=True)
    _require(selected_ready == canonical_ready, "noncanonical readiness directory is forbidden")
    _require(selected_config == canonical_config, "noncanonical v5 config is forbidden")
    config, _, preseal, ready_receipt = _verify_readiness_manifest(
        project_root, selected_ready, selected_config
    )

    plan = _read_json(selected_ready / "execution_plan.json")
    _require(plan.get("outer_prefix_cells") == EXPECTED_SCOPE_COUNT, "outer scope count changed")
    _require(plan.get("seeded_cells") == EXPECTED_CELL_COUNT, "seeded cell count changed")
    _require(plan.get("top_level_component_jobs") == EXPECTED_JOB_COUNT, "job count changed")
    seeds = tuple(int(value) for value in plan["complete_pipeline_seeds"])
    components = tuple(str(value) for value in plan["component_order"])
    _require(seeds == EXPECTED_SEEDS, "pipeline seeds changed")
    _require(components == tuple(COMPONENTS), "component order changed")
    _require(tuple(TARGET_LAYERS) == REGISTERED_LAYERS, "registered target layers changed")
    _require(
        int(plan["outer_evaluation_rows_per_fraction"]) == EXPECTED_ROWS_PER_FRACTION,
        "outer metric denominator changed",
    )

    scopes: list[ScopeSpec] = []
    for raw in plan["prefix_plans"]:
        inner = tuple(str(item["inner_fold"]) for item in raw["inner_folds"])
        _require(inner == ("inner_1", "inner_2", "inner_3"), "inner fold identities changed")
        scopes.append(
            ScopeSpec(
                scope_id=str(raw["scope_id"]),
                outer_fold=str(raw["outer_fold"]),
                fraction=float(raw["prefix_fraction"]),
                cutoff_kst=str(raw["cutoff_kst"]),
                inner_folds=inner,
                prefix_time_count=int(raw["prefix_time_count"]),
                inner_details={
                    str(item["inner_fold"]): dict(item) for item in raw["inner_folds"]
                },
            )
        )
    _require(len(scopes) == EXPECTED_SCOPE_COUNT, "execution plan scope count changed")
    _require(len({scope.scope_id for scope in scopes}) == EXPECTED_SCOPE_COUNT, "duplicate scope id")
    _require(
        tuple(sorted({scope.outer_fold for scope in scopes})) == tuple(sorted(EXPECTED_OUTER_FOLDS)),
        "outer fold identities changed",
    )
    for fold in EXPECTED_OUTER_FOLDS:
        fractions = tuple(sorted(scope.fraction for scope in scopes if scope.outer_fold == fold))
        _require(fractions == EXPECTED_FRACTIONS, f"prefix fractions changed for {fold}")

    seed_ledger = [
        {
            "scope_id": scope.scope_id,
            "complete_seed": pipeline_seed,
            "component": component,
            "phase": phase,
            "child_seed": child_seed(
                pipeline_seed,
                component,
                scope.outer_fold,
                scope.fraction,
                phase,
            ),
        }
        for scope in scopes
        for pipeline_seed in seeds
        for component in components
        for phase in ("inner_1", "inner_2", "inner_3", "full")
    ]
    _require(len(seed_ledger) == EXPECTED_JOB_COUNT, "child-seed ledger count changed")
    _require(
        len({row["child_seed"] for row in seed_ledger}) == EXPECTED_JOB_COUNT,
        "child-seed collision detected",
    )
    _require(plan.get("unique_child_seeds") == EXPECTED_JOB_COUNT, "sealed unique child-seed count changed")
    _require(
        plan.get("child_seed_ledger_sha256") == canonical_sha256(seed_ledger),
        "sealed child-seed ledger changed",
    )

    jobs: dict[str, JobSpec] = {}
    cells: dict[str, CellSpec] = {}
    for scope in scopes:
        for pipeline_seed in seeds:
            receipt_id = f"{scope.scope_id}__s{pipeline_seed}"
            directory_id = f"cell__{receipt_id}"
            cells[directory_id] = CellSpec(
                directory_id=directory_id,
                receipt_id=receipt_id,
                scope_id=scope.scope_id,
                outer_fold=scope.outer_fold,
                fraction=scope.fraction,
                cutoff_kst=scope.cutoff_kst,
                pipeline_seed=pipeline_seed,
                inner_folds=scope.inner_folds,
                expected_outer_rows=EXPECTED_FOLD_ROWS[scope.outer_fold],
                expected_inner_oof_rows=sum(
                    int(scope.inner_details[phase]["validation_time_count"])
                    for phase in scope.inner_folds
                )
                * len(REGISTERED_LAYERS),
            )
            for phase in (*scope.inner_folds, "full"):
                for component in components:
                    inner_detail = scope.inner_details.get(phase)
                    job_id = f"{scope.scope_id}__s{pipeline_seed}__{phase}__{component}"
                    jobs[job_id] = JobSpec(
                        job_id=job_id,
                        scope_id=scope.scope_id,
                        outer_fold=scope.outer_fold,
                        fraction=scope.fraction,
                        cutoff_kst=scope.cutoff_kst,
                        pipeline_seed=pipeline_seed,
                        phase=phase,
                        component=component,
                        child_seed=int(
                            child_seed(
                                pipeline_seed,
                                component,
                                scope.outer_fold,
                                scope.fraction,
                                phase,
                            )
                        ),
                        train_last_kst=None
                        if inner_detail is None
                        else str(inner_detail["train_last_kst"]),
                        validation_start_kst=None
                        if inner_detail is None
                        else str(inner_detail["validation_start_kst"]),
                        embargo_threshold_kst=None
                        if inner_detail is None
                        else str(inner_detail["embargo_threshold_kst"]),
                        training_supervised_time_count=None
                        if inner_detail is None
                        else int(inner_detail["train_time_count"]),
                        validation_supervised_time_count=None
                        if inner_detail is None
                        else int(inner_detail["validation_time_count"]),
                        prefix_supervised_time_count=scope.prefix_time_count,
                    )
    _require(len(cells) == EXPECTED_CELL_COUNT, "expected cell id set changed")
    _require(len(jobs) == EXPECTED_JOB_COUNT, "expected job id set changed")

    actual_relative = str(config["output"]["actual_directory"])
    _require(actual_relative == ACTUAL_RELATIVE, "actual namespace path changed")
    namespace = Path(actual_relative).name
    binding = {
        "namespace": namespace,
        "execution_contract_sha256": ready_receipt["config_sha256"],
        "parent_recipe_sha256": ready_receipt["recipe_sha256"],
        "preexecution_seal_sha256": ready_receipt["preexecution_seal_sha256"],
        "semantic_preflight_sha256": ready_receipt["semantic_preflight_sha256"],
        "exact_command_sha256": ready_receipt["exact_command_sha256"],
        "authorization_sha256": ready_receipt["authorization_sha256"],
        "module_sha256": ready_receipt["module_sha256"],
        "runner_sha256": ready_receipt["runner_sha256"],
        "job_store_contract_sha256": ready_receipt["preexecution_seal_sha256"],
        "expected_terminal_status": TERMINAL_STATUS,
        "maximum_resume_attempts": MAXIMUM_RESUME_ATTEMPTS,
        "maximum_total_attempts": MAXIMUM_TOTAL_ATTEMPTS,
        "execution_contract_revision": "v5",
        "control_engine_schema_version": "v4",
    }
    _require(preseal["parent_recipe_sha256"] == binding["parent_recipe_sha256"], "recipe binding changed")
    return PostExecutionContract(
        namespace=namespace,
        binding=binding,
        scopes=tuple(scopes),
        jobs=jobs,
        cells=cells,
        seeds=seeds,
        components=components,
        layers=REGISTERED_LAYERS,
        outer_folds=EXPECTED_OUTER_FOLDS,
        tokens=EXPECTED_TOKENS,
        family_settings=tuple(FAMILY_SETTINGS),
        expected_rows_per_fraction=EXPECTED_ROWS_PER_FRACTION,
        expected_fold_rows=dict(EXPECTED_FOLD_ROWS),
        expected_layer_rows=dict(EXPECTED_LAYER_ROWS),
        expected_fold_layer_rows={
            fold: {
                int(layer): int(rows)
                for layer, rows in ready_receipt["semantic_outer_validation_by_fold"][fold][
                    "rows_by_layer"
                ].items()
            }
            for fold in EXPECTED_OUTER_FOLDS
        },
        expected_outer_key_sha256={
            fold: str(
                ready_receipt["semantic_outer_validation_by_fold"][fold][
                    "ordered_key_sha256"
                ]
            )
            for fold in EXPECTED_OUTER_FOLDS
        },
        expected_base_fits=EXPECTED_BASE_FITS,
        expected_deep_fits=EXPECTED_DEEP_FITS,
        expected_lightgbm_fits=EXPECTED_LIGHTGBM_FITS,
        expected_meta_optimizations=EXPECTED_META_OPTIMIZATIONS,
        readiness_receipt=ready_receipt,
    )


def _verify_control_plane(actual_dir: Path, contract: PostExecutionContract) -> dict[str, Any]:
    """Verify terminal closure before any aggregate prediction value is opened."""

    _require(actual_dir.is_dir() and not actual_dir.is_symlink(), "actual namespace is unsafe")
    _require(actual_dir.name == contract.namespace, "actual namespace identity changed")
    start = _read_json(actual_dir / "execution_start.json")
    result_path = actual_dir / "result.json"
    terminal_path = actual_dir / "terminal_receipt.json"
    result = _read_json(result_path)
    terminal = _read_json(terminal_path)

    _require(start.get("schema_version") == "p2_authoritative_execution_start.v4", "start schema changed")
    _require(
        set(start)
        == {
            "schema_version",
            "status",
            "created_at_kst",
            "binding",
            "initial_start_attempt_count",
            "resume_attempt_budget",
            "total_attempt_budget",
            "result_based_tuning_allowed",
            "cross_v1_v2_v3_job_reuse_allowed",
            "execution_start_sha256",
        },
        "start receipt field set changed",
    )
    _require(start.get("status") == "STARTED_INCOMPLETE_RESUMABLE", "start status changed")
    _require(start.get("binding") == contract.binding, "execution-start binding changed")
    _require(start.get("initial_start_attempt_count") == 1, "initial attempt count changed")
    _require(start.get("resume_attempt_budget") == MAXIMUM_RESUME_ATTEMPTS, "start resume budget changed")
    _require(start.get("total_attempt_budget") == MAXIMUM_TOTAL_ATTEMPTS, "start total budget changed")
    _require(start.get("result_based_tuning_allowed") is False, "start permits result tuning")
    _require(start.get("cross_v1_v2_v3_job_reuse_allowed") is False, "start permits foreign JobStore reuse")
    start_sha = _verify_self_hash(start, "execution_start_sha256", "start self-hash changed")

    attempts_dir = actual_dir / "attempts"
    _require(attempts_dir.is_dir() and not attempts_dir.is_symlink(), "attempt ledger is unsafe")
    attempt_entries = sorted(attempts_dir.iterdir(), key=lambda path: path.name)
    _require(all(path.is_file() and not path.is_symlink() for path in attempt_entries), "attempt entry is unsafe")
    _require(all(".partial." not in path.name for path in attempt_entries), "attempt partial exists")
    resume_paths = [path for path in attempt_entries if _RESUME_NAME.fullmatch(path.name)]
    terminal_paths = [path for path in attempt_entries if _ATTEMPT_TERMINAL_NAME.fullmatch(path.name)]
    _require(len(resume_paths) + len(terminal_paths) == len(attempt_entries), "unexpected attempt entry")
    _require(len(resume_paths) <= MAXIMUM_RESUME_ATTEMPTS, "resume budget exceeded")
    expected_resume_names = [
        f"resume_attempt_{number:03d}.json" for number in range(2, 2 + len(resume_paths))
    ]
    _require([path.name for path in resume_paths] == expected_resume_names, "resume sequence changed")
    resume_audits: list[dict[str, Any]] = []
    previous_jobs = 0
    previous_cells = 0
    for number, path in enumerate(resume_paths, start=2):
        value = _read_json(path)
        _require(
            set(value)
            == {
                "schema_version",
                "status",
                "created_at_kst",
                "attempt_number",
                "resume_attempt_number",
                "remaining_resume_budget_after_start",
                "execution_start_sha256",
                "binding",
                "read_only_namespace_audit",
                "result_based_tuning_allowed",
                "resume_attempt_sha256",
            },
            "resume receipt field set changed",
        )
        _require(value.get("schema_version") == "p2_authoritative_resume_attempt.v4", "resume schema changed")
        _require(value.get("status") == "RESUME_STARTED", "resume status changed")
        _require(value.get("attempt_number") == number, "resume attempt number changed")
        _require(value.get("resume_attempt_number") == number - 1, "resume ordinal changed")
        _require(
            value.get("remaining_resume_budget_after_start") == MAXIMUM_TOTAL_ATTEMPTS - number,
            "resume remaining budget changed",
        )
        _require(value.get("execution_start_sha256") == start_sha, "resume start binding changed")
        _require(value.get("binding") == contract.binding, "resume execution binding changed")
        audit = value.get("read_only_namespace_audit")
        _require(isinstance(audit, dict), "resume namespace audit missing")
        _require(
            set(audit)
            == {
                "jobs_completed",
                "cells_completed",
                "job_manifest_ledger_sha256",
                "cell_manifest_ledger_sha256",
                "terminal_result_absent",
                "exclusive_lock_acquired_before_this_receipt",
            },
            "resume namespace audit field set changed",
        )
        _require(audit.get("terminal_result_absent") is True, "resume started after terminal result")
        _require(
            audit.get("exclusive_lock_acquired_before_this_receipt") is True,
            "resume receipt was not lock-protected",
        )
        _require_sha(audit.get("job_manifest_ledger_sha256"), "resume job ledger hash missing")
        _require_sha(audit.get("cell_manifest_ledger_sha256"), "resume cell ledger hash missing")
        jobs_completed = _require_nonnegative_int(audit.get("jobs_completed"), "resume job count invalid")
        cells_completed = _require_nonnegative_int(audit.get("cells_completed"), "resume cell count invalid")
        _require(jobs_completed <= len(contract.jobs), "resume job count exceeds graph")
        _require(cells_completed <= len(contract.cells), "resume cell count exceeds graph")
        _require(jobs_completed >= previous_jobs, "resume job count decreased")
        _require(cells_completed >= previous_cells, "resume cell count decreased")
        jobs_per_cell = len(contract.jobs) // len(contract.cells)
        _require(
            jobs_completed >= jobs_per_cell * cells_completed,
            "resume job count cannot support completed cells",
        )
        previous_jobs = jobs_completed
        previous_cells = cells_completed
        resume_audits.append(
            {
                "attempt_number": number,
                "jobs_completed": jobs_completed,
                "cells_completed": cells_completed,
                "job_manifest_ledger_sha256": audit["job_manifest_ledger_sha256"],
                "cell_manifest_ledger_sha256": audit["cell_manifest_ledger_sha256"],
            }
        )
        _require(value.get("result_based_tuning_allowed") is False, "resume permits result tuning")
        _verify_self_hash(value, "resume_attempt_sha256", "resume self-hash changed")

    terminal_by_attempt: dict[int, dict[str, Any]] = {}
    total_attempts = 1 + len(resume_paths)
    for path in terminal_paths:
        value = _read_json(path)
        _require(
            set(value)
            == {
                "schema_version",
                "status",
                "recorded_at_kst",
                "attempt_number",
                "execution_start_sha256",
                "binding",
                "classification",
                "automatic_resume_permitted",
                "exception_type",
                "exception_message",
                "traceback_sha256",
                "traceback_text_recorded",
                "raw_observation_values_recorded",
                "attempt_terminal_sha256",
            },
            "attempt-terminal field set changed",
        )
        _require(
            value.get("schema_version") == "p2_authoritative_attempt_terminal.v4",
            "attempt-terminal schema changed",
        )
        number = int(value.get("attempt_number", -1))
        _require(1 <= number <= total_attempts, "attempt-terminal has no start")
        _require(number not in terminal_by_attempt, "duplicate attempt-terminal receipt")
        _require(value.get("execution_start_sha256") == start_sha, "attempt-terminal start changed")
        _require(value.get("binding") == contract.binding, "attempt-terminal binding changed")
        status = value.get("status")
        if number == total_attempts:
            _require(status == "COMPLETE_TERMINAL", "latest attempt is not terminal-success")
            _require(value.get("automatic_resume_permitted") is False, "terminal attempt permits resume")
            _require(value.get("classification") == "SUCCESS", "terminal attempt classification changed")
            _require(
                value.get("exception_type") is None
                and value.get("exception_message") is None
                and value.get("traceback_sha256") is None,
                "terminal success contains failure evidence",
            )
        else:
            _require(status == "FAILED_TRANSIENT_RESUMABLE", "prior closed attempt was not transient")
            _require(value.get("automatic_resume_permitted") is True, "transient attempt did not permit resume")
            _require(
                value.get("classification") == "TRANSIENT_RUNTIME_EXPLICIT",
                "transient attempt classification changed",
            )
            _require(isinstance(value.get("exception_type"), str) and value["exception_type"], "transient exception type missing")
            _require(isinstance(value.get("exception_message"), str), "transient exception message missing")
            _require_sha(value.get("traceback_sha256"), "transient traceback hash missing")
        _require(value.get("traceback_text_recorded") is False, "raw traceback text was recorded")
        _require(value.get("raw_observation_values_recorded") is False, "raw observation values were recorded")
        _verify_self_hash(value, "attempt_terminal_sha256", "attempt-terminal self-hash changed")
        terminal_by_attempt[number] = value
    _require(total_attempts in terminal_by_attempt, "latest attempt-terminal receipt is missing")

    result_sha = sha256_file(result_path)
    result_bytes = result_path.stat().st_size
    binding_sha = canonical_sha256(contract.binding)
    _require(result.get("status") == TERMINAL_STATUS, "result is not v5 terminal")
    _require(result.get("execution_binding_sha256") == binding_sha, "result binding hash changed")
    _require(
        result.get("preexecution_seal_sha256") == contract.binding["preexecution_seal_sha256"],
        "result seal binding changed",
    )
    _require(
        result.get("semantic_preflight_sha256") == contract.binding["semantic_preflight_sha256"],
        "result semantic binding changed",
    )
    _require(result.get("initial_start_attempt_count") == 1, "result initial count changed")
    _require(result.get("resume_attempts_started") == len(resume_paths), "result resume count changed")
    _require(result.get("total_attempts_started") == total_attempts, "result total count changed")
    _require(result.get("submission_files_generated") == 0, "result generated a submission")
    _require(result.get("uploads") == 0, "result reports an upload")
    _require(result.get("official_test_sample_submission_reads") == 0, "result reports official reads")
    _require(result.get("resume_or_result_based_tuning_performed") is False, "result-based tuning occurred")
    _require(result.get("foreign_v1_v2_v3_v4_job_or_cell_reuse") == 0, "foreign JobStore reuse occurred")
    _require(result.get("execution_contract_revision") == "v5", "result revision changed")
    _require(result.get("v4_resume_engine_byte_pinned") is True, "result lost resume-engine pin")

    _require(
        terminal.get("schema_version") == "p2_authoritative_terminal_receipt.v4",
        "terminal receipt schema changed",
    )
    _require(
        set(terminal)
        == {
            "schema_version",
            "status",
            "completed_at_kst",
            "binding",
            "result_sha256",
            "result_bytes",
            "result_atomic_publish",
            "initial_start_attempts",
            "resume_attempts_started",
            "total_attempts_started",
            "automatic_resume_budget_remaining",
            "terminal_rerun_allowed",
            "terminal_receipt_sha256",
        },
        "terminal receipt field set changed",
    )
    _require(terminal.get("status") == "TERMINAL_COMPLETE_NO_RERUN", "terminal receipt is not closed")
    _require(terminal.get("binding") == contract.binding, "terminal receipt binding changed")
    _require(terminal.get("result_sha256") == result_sha, "terminal/result SHA pin differs")
    _require(terminal.get("result_bytes") == result_bytes, "terminal/result byte pin differs")
    _require(terminal.get("initial_start_attempts") == 1, "terminal initial count changed")
    _require(terminal.get("resume_attempts_started") == len(resume_paths), "terminal resume count changed")
    _require(terminal.get("total_attempts_started") == total_attempts, "terminal total count changed")
    _require(
        terminal.get("automatic_resume_budget_remaining") == MAXIMUM_TOTAL_ATTEMPTS - total_attempts,
        "terminal remaining budget changed",
    )
    _require(terminal.get("terminal_rerun_allowed") is False, "terminal rerun is allowed")
    publish = terminal.get("result_atomic_publish")
    _verify_atomic_publish_receipt(
        publish,
        expected_sha256=result_sha,
        expected_bytes=result_bytes,
        label="terminal result",
        allow_finalization_recovery=True,
    )
    terminal_self_sha = _verify_self_hash(
        terminal, "terminal_receipt_sha256", "terminal receipt self-hash changed"
    )
    return {
        "result": result,
        "result_sha256": result_sha,
        "result_bytes": result_bytes,
        "terminal_receipt_sha256": sha256_file(terminal_path),
        "terminal_receipt_self_sha256": terminal_self_sha,
        "execution_start_sha256": start_sha,
        "execution_binding_sha256": binding_sha,
        "initial_attempts": 1,
        "resume_attempts": len(resume_paths),
        "total_attempts": total_attempts,
        "prior_transient_receipts": sum(
            value.get("status") == "FAILED_TRANSIENT_RESUMABLE"
            for number, value in terminal_by_attempt.items()
            if number < total_attempts
        ),
        "abrupt_prior_attempts_without_terminal_receipt": sum(
            number not in terminal_by_attempt for number in range(1, total_attempts)
        ),
        "resume_audits": resume_audits,
        "final_invocation_start_jobs_completed": previous_jobs,
        "final_invocation_start_cells_completed": previous_cells,
    }


def _verify_store_manifest(
    directory: Path,
    *,
    identifier: str,
    contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(directory.is_dir() and not directory.is_symlink(), f"unsafe store entry: {identifier}")
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    _require(
        set(manifest)
        == {
            "schema_version",
            "job_id",
            "contract_sha256",
            "files",
            "payload_files",
            "complete",
        },
        f"manifest field set changed: {identifier}",
    )
    _require(
        manifest.get("schema_version") == "p2_authoritative_nested_surrogate_job.v1",
        f"manifest schema changed: {identifier}",
    )
    _require(manifest.get("job_id") == identifier, f"manifest id changed: {identifier}")
    _require(manifest.get("contract_sha256") == contract_sha256, f"contract changed: {identifier}")
    _require(manifest.get("complete") is True, f"incomplete store entry: {identifier}")
    files = manifest.get("files")
    payload_files = manifest.get("payload_files")
    _require(isinstance(files, dict) and files, f"file pins missing: {identifier}")
    _require(isinstance(payload_files, list), f"payload ledger missing: {identifier}")
    _require({"prediction.parquet", "receipt.json"}.issubset(files), f"core payload missing: {identifier}")
    _require(set(payload_files).issubset(files), f"payload pin missing: {identifier}")
    _require(len(payload_files) == len(set(payload_files)), f"duplicate payload name: {identifier}")
    expected_entries = {"manifest.json", *files.keys()}
    actual_entries = {path.name for path in directory.iterdir()}
    _require(actual_entries == expected_entries, f"unexpected store file: {identifier}")
    _require(all(".partial." not in name for name in actual_entries), f"partial store file: {identifier}")
    for name, pin in files.items():
        _require(Path(str(name)).name == name, f"nested payload name: {identifier}")
        _verify_file_pin(directory / name, pin, f"{identifier}/{name}")
    receipt = _read_json(directory / "receipt.json")
    ledger = {
        "job_id": identifier,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "payload_count": len(files),
        "payload_files": list(payload_files),
        "files": {
            name: {"sha256": str(pin["sha256"]), "bytes": int(pin["bytes"])}
            for name, pin in files.items()
        },
    }
    return receipt, ledger


def _verify_job_receipt(receipt: Mapping[str, Any], spec: JobSpec, payload_files: Sequence[str]) -> None:
    _require(receipt.get("component") == spec.component, f"job component changed: {spec.job_id}")
    _require(receipt.get("phase") == spec.phase, f"job phase changed: {spec.job_id}")
    _require(receipt.get("seed") == spec.child_seed, f"job seed changed: {spec.job_id}")
    _require(
        receipt.get("future_or_outer_labels_in_fit") is False,
        f"job reports future/outer label use: {spec.job_id}",
    )
    if spec.component == "router_400":
        _require(
            set(receipt)
            == {
                "component",
                "phase",
                "seed",
                "composite_lightgbm_estimators",
                "rounds_per_estimator",
                "cpu_threads_per_estimator",
                "training_timestamp_count",
                "prediction_timestamp_count",
                "future_or_outer_labels_in_fit",
            },
            f"router receipt fields changed: {spec.job_id}",
        )
        _require(set(payload_files) == {"model.joblib"}, f"router payload changed: {spec.job_id}")
        _require(receipt.get("composite_lightgbm_estimators") == 4, f"router estimator count changed: {spec.job_id}")
        _require(receipt.get("rounds_per_estimator") == 400, f"router rounds changed: {spec.job_id}")
        _require(receipt.get("cpu_threads_per_estimator") == 4, f"router threads changed: {spec.job_id}")
        _require(_require_nonnegative_int(receipt.get("training_timestamp_count"), f"router train count invalid: {spec.job_id}") > 0, f"router train count empty: {spec.job_id}")
        _require(_require_nonnegative_int(receipt.get("prediction_timestamp_count"), f"router prediction count invalid: {spec.job_id}") > 0, f"router prediction count empty: {spec.job_id}")
        if spec.phase == "full":
            _require(
                receipt.get("training_timestamp_count")
                == spec.prefix_supervised_time_count,
                f"router full training count differs from sealed prefix: {spec.job_id}",
            )
        else:
            _require(
                receipt.get("training_timestamp_count")
                == spec.training_supervised_time_count,
                f"router inner training count differs from sealed plan: {spec.job_id}",
            )
            _require(
                receipt.get("prediction_timestamp_count")
                == spec.validation_supervised_time_count,
                f"router inner prediction count differs from sealed plan: {spec.job_id}",
            )
    else:
        expected_fields = (
            {
                "component",
                "phase",
                "seed",
                "epochs",
                "parameter_count",
                "final_train_mse_c",
                "adapter",
                "future_or_outer_labels_in_fit",
            }
            if spec.phase == "full"
            else {
                "component",
                "phase",
                "seed",
                "best_epoch",
                "best_rmse_c",
                "parameter_count",
                "history",
                "adapter",
                "future_or_outer_labels_in_fit",
            }
        )
        _require(set(receipt) == expected_fields, f"deep receipt fields changed: {spec.job_id}")
        _require(set(payload_files) == {"checkpoint.pt"}, f"deep payload changed: {spec.job_id}")
        _require(int(receipt.get("parameter_count", 0)) > 0, f"deep parameter count invalid: {spec.job_id}")
        adapter = receipt.get("adapter")
        _require(isinstance(adapter, dict), f"deep adapter missing: {spec.job_id}")
        _require(
            adapter.get("continuous_public_covariates_preserved") is True,
            f"deep public context changed: {spec.job_id}",
        )
        _require(
            adapter.get("labels_restricted_to_registered_common_ledger") is True,
            f"deep label ledger changed: {spec.job_id}",
        )
        if spec.phase == "full":
            _require(
                set(adapter)
                == {
                    "schema_version",
                    "scope_id",
                    "continuous_public_time_count",
                    "supervised_time_count",
                    "cutoff_kst",
                    "later_public_time_count",
                    "continuous_public_covariates_preserved",
                    "labels_restricted_to_registered_common_ledger",
                },
                f"full adapter fields changed: {spec.job_id}",
            )
            _require(adapter.get("schema_version") == "p2_authoritative_deep_full_adapter.v3", f"full adapter schema changed: {spec.job_id}")
            _require(adapter.get("scope_id") == spec.scope_id, f"full adapter scope changed: {spec.job_id}")
            _require(adapter.get("cutoff_kst") == spec.cutoff_kst, f"full adapter cutoff changed: {spec.job_id}")
            _require(_require_nonnegative_int(adapter.get("continuous_public_time_count"), f"full public count invalid: {spec.job_id}") > 0, f"full public context empty: {spec.job_id}")
            _require(_require_nonnegative_int(adapter.get("supervised_time_count"), f"full supervised count invalid: {spec.job_id}") > 0, f"full supervision empty: {spec.job_id}")
            _require(
                adapter.get("supervised_time_count") == spec.prefix_supervised_time_count,
                f"full supervised count differs from sealed prefix: {spec.job_id}",
            )
            _require(
                adapter.get("continuous_public_time_count")
                >= adapter.get("supervised_time_count"),
                f"full public context is smaller than supervision: {spec.job_id}",
            )
            _require(int(receipt.get("epochs", 0)) >= 1, f"full deep epochs invalid: {spec.job_id}")
            final_train_mse = _require_finite_float(receipt.get("final_train_mse_c"), f"full train metric invalid: {spec.job_id}")
            _require(final_train_mse >= 0.0, f"full train metric is negative: {spec.job_id}")
            _require(adapter.get("later_public_time_count") == 0, f"full adapter uses later context: {spec.job_id}")
        else:
            _require(
                set(adapter)
                == {
                    "schema_version",
                    "inner_fold",
                    "panel_time_count",
                    "continuous_training_public_time_count",
                    "continuous_validation_public_time_count",
                    "training_supervised_time_count",
                    "validation_supervised_time_count",
                    "masked_nonregistered_target_values",
                    "training_context_last_kst",
                    "validation_context_first_kst",
                    "strict_embargo_pass",
                    "continuous_public_covariates_preserved",
                    "labels_restricted_to_registered_common_ledger",
                },
                f"inner adapter fields changed: {spec.job_id}",
            )
            _require(adapter.get("schema_version") == "p2_authoritative_deep_inner_adapter.v3", f"inner adapter schema changed: {spec.job_id}")
            _require(adapter.get("inner_fold") == spec.phase, f"inner adapter fold changed: {spec.job_id}")
            for name in (
                "panel_time_count",
                "continuous_training_public_time_count",
                "continuous_validation_public_time_count",
                "training_supervised_time_count",
                "validation_supervised_time_count",
            ):
                _require(_require_nonnegative_int(adapter.get(name), f"inner adapter count invalid: {spec.job_id}/{name}") > 0, f"inner adapter support empty: {spec.job_id}/{name}")
            _require_nonnegative_int(adapter.get("masked_nonregistered_target_values"), f"inner masked count invalid: {spec.job_id}")
            _require(
                adapter.get("training_supervised_time_count")
                == spec.training_supervised_time_count,
                f"inner training support differs from sealed plan: {spec.job_id}",
            )
            _require(
                adapter.get("validation_supervised_time_count")
                == spec.validation_supervised_time_count,
                f"inner validation support differs from sealed plan: {spec.job_id}",
            )
            _require(
                adapter.get("panel_time_count")
                == adapter.get("continuous_training_public_time_count")
                + adapter.get("continuous_validation_public_time_count"),
                f"inner continuous context partition changed: {spec.job_id}",
            )
            _require(
                adapter.get("training_context_last_kst") == spec.train_last_kst,
                f"inner training boundary changed: {spec.job_id}",
            )
            _require(
                adapter.get("validation_context_first_kst")
                == spec.validation_start_kst,
                f"inner validation boundary changed: {spec.job_id}",
            )
            _require(
                spec.train_last_kst is not None
                and spec.embargo_threshold_kst is not None
                and spec.validation_start_kst is not None,
                f"sealed inner boundaries missing: {spec.job_id}",
            )
            train_last = datetime.fromisoformat(spec.train_last_kst)
            embargo = datetime.fromisoformat(spec.embargo_threshold_kst)
            validation_start = datetime.fromisoformat(spec.validation_start_kst)
            _require(
                train_last.tzinfo is not None
                and embargo.tzinfo is not None
                and validation_start.tzinfo is not None,
                f"sealed inner timezone missing: {spec.job_id}",
            )
            _require(train_last < embargo, f"inner training crossed embargo: {spec.job_id}")
            _require(
                validation_start - embargo == timedelta(days=7),
                f"inner seven-day embargo changed: {spec.job_id}",
            )
            _require(int(receipt.get("best_epoch", 0)) >= 1, f"inner best epoch invalid: {spec.job_id}")
            _require_finite_float(receipt.get("best_rmse_c"), f"inner RMSE invalid: {spec.job_id}")
            _require(adapter.get("strict_embargo_pass") is True, f"inner embargo failed: {spec.job_id}")
            history = receipt.get("history")
            _require(isinstance(history, list) and history, f"inner history missing: {spec.job_id}")
            _require(
                all(
                    isinstance(row, dict)
                    and set(row) == {"epoch", "train_mse_c", "validation_rmse", "learning_rate"}
                    and isinstance(row["epoch"], int)
                    and row["epoch"] >= 1
                    and math.isfinite(float(row["train_mse_c"]))
                    and float(row["train_mse_c"]) >= 0.0
                    and math.isfinite(float(row["validation_rmse"]))
                    and float(row["validation_rmse"]) >= 0.0
                    and math.isfinite(float(row["learning_rate"]))
                    and float(row["learning_rate"]) > 0.0
                    for row in history
                ),
                f"inner history invalid: {spec.job_id}",
            )
            history_epochs = [int(row["epoch"]) for row in history]
            _require(history_epochs[0] == 1, f"inner history does not begin at epoch one: {spec.job_id}")
            _require(
                all(
                    left < right
                    for left, right in zip(
                        history_epochs, history_epochs[1:], strict=False
                    )
                ),
                f"inner history epochs are not strictly increasing: {spec.job_id}",
            )
            selected_epoch = -1
            selected_rmse = float("inf")
            for row in history:
                current_rmse = float(row["validation_rmse"])
                if current_rmse < selected_rmse - 1e-6:
                    selected_epoch = int(row["epoch"])
                    selected_rmse = current_rmse
            _require(receipt.get("best_epoch") == selected_epoch, f"inner best epoch differs from history: {spec.job_id}")
            _require(
                math.isclose(
                    float(receipt["best_rmse_c"]),
                    selected_rmse,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ),
                f"inner best RMSE differs from history: {spec.job_id}",
            )


def _verify_job_store(actual_dir: Path, contract: PostExecutionContract) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = actual_dir / "jobs"
    _require(root.is_dir() and not root.is_symlink(), "job store is unsafe")
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    _require(all(path.is_dir() and not path.is_symlink() for path in entries), "job store entry is unsafe")
    actual_ids = {path.name for path in entries}
    _require(actual_ids == set(contract.jobs), "completed job id set is not exactly sealed 900-job graph")
    receipts: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    for path in entries:
        spec = contract.jobs[path.name]
        receipt, row = _verify_store_manifest(
            path,
            identifier=path.name,
            contract_sha256=str(contract.binding["job_store_contract_sha256"]),
        )
        manifest = _read_json(path / "manifest.json")
        payload_name = "model.joblib" if spec.component == "router_400" else "checkpoint.pt"
        _require(
            set(manifest["files"]) == {"prediction.parquet", "receipt.json", payload_name},
            f"job manifest file set changed: {path.name}",
        )
        _verify_job_receipt(receipt, spec, tuple(str(value) for value in manifest["payload_files"]))
        receipts[path.name] = receipt
        ledger.append(row)
    return {
        "completed_jobs": len(entries),
        "expected_jobs": len(contract.jobs),
        "ordered_manifest_ledger_sha256": canonical_sha256(ledger),
        "ordered_manifest_ledger": ledger,
        "model_or_checkpoint_payload_values_opened": 0,
        "job_prediction_values_opened": 0,
        "manifest_file_hashes_verified": sum(row["payload_count"] for row in ledger),
    }, receipts


def _verify_cell_receipt(
    receipt: Mapping[str, Any],
    spec: CellSpec,
    *,
    job_receipts: Mapping[str, Mapping[str, Any]],
    components: Sequence[str],
) -> None:
    _require(
        receipt.get("schema_version") == "p2_authoritative_nested_surrogate_cell.v1",
        f"cell schema changed: {spec.directory_id}",
    )
    _require(receipt.get("cell_id") == spec.receipt_id, f"cell id changed: {spec.directory_id}")
    _require(receipt.get("outer_fold") == spec.outer_fold, f"cell outer fold changed: {spec.directory_id}")
    _require(math.isclose(float(receipt.get("prefix_fraction")), spec.fraction), f"cell fraction changed: {spec.directory_id}")
    _require(receipt.get("pipeline_seed") == spec.pipeline_seed, f"cell seed changed: {spec.directory_id}")
    _require(receipt.get("prefix_cutoff_kst") == spec.cutoff_kst, f"cell cutoff changed: {spec.directory_id}")
    expected_cell_fields = {
        "schema_version",
        "cell_id",
        "outer_fold",
        "prefix_fraction",
        "pipeline_seed",
        "prefix_cutoff_kst",
        "inner_oof_ledger",
        "selected_inner_epochs",
        "full_prefix_epochs",
        "meta",
        "postprocess",
        "component_receipts",
        "full_receipts",
        "guards",
    }
    _require(set(receipt) == expected_cell_fields, f"cell receipt field set changed: {spec.directory_id}")
    ledger = receipt.get("inner_oof_ledger")
    _require(isinstance(ledger, dict), f"inner OOF ledger missing: {spec.directory_id}")
    _require(
        set(ledger)
        == {
            "rows",
            "ordered_key_sha256",
            "ordered_key_truth_sha256",
            "component_prediction_sha256",
            "component_count",
            "duplicate_keys",
            "nonfinite_truth",
            "nonfinite_predictions",
            "same_ordered_key_and_truth_across_components",
        },
        f"inner OOF ledger fields changed: {spec.directory_id}",
    )
    _require(_require_nonnegative_int(ledger.get("rows"), f"inner OOF row count invalid: {spec.directory_id}") > 0, f"inner OOF ledger empty: {spec.directory_id}")
    _require(
        ledger.get("rows") == spec.expected_inner_oof_rows,
        f"inner OOF rows differ from sealed fold supports: {spec.directory_id}",
    )
    _require_sha(ledger.get("ordered_key_sha256"), f"inner OOF key hash missing: {spec.directory_id}")
    _require_sha(ledger.get("ordered_key_truth_sha256"), f"inner OOF truth hash missing: {spec.directory_id}")
    prediction_digests = ledger.get("component_prediction_sha256")
    _require(
        isinstance(prediction_digests, dict)
        and set(prediction_digests) == set(components)
        and all(_HEX64.fullmatch(str(value)) is not None for value in prediction_digests.values()),
        f"inner OOF component digest ledger changed: {spec.directory_id}",
    )
    _require(ledger.get("component_count") == len(components), f"inner OOF component count changed: {spec.directory_id}")
    _require(ledger.get("duplicate_keys") == 0, f"inner OOF duplicate keys reported: {spec.directory_id}")
    _require(ledger.get("nonfinite_truth") == 0, f"inner OOF nonfinite truth reported: {spec.directory_id}")
    _require(ledger.get("nonfinite_predictions") == 0, f"inner OOF nonfinite predictions reported: {spec.directory_id}")
    _require(ledger.get("same_ordered_key_and_truth_across_components") is True, f"inner OOF component population mismatch: {spec.directory_id}")
    meta = receipt.get("meta")
    _require(isinstance(meta, dict), f"cell meta receipt missing: {spec.directory_id}")
    _require(
        set(meta)
        == {
            "scope_id",
            "oof_rows",
            "oof_key_truth_sha256",
            "stack_method",
            "stack_weights",
            "gate",
            "parameter_source",
            "frozen_epoch_reused",
            "frozen_stack_reused",
            "frozen_gate_reused",
        },
        f"cell meta receipt fields changed: {spec.directory_id}",
    )
    _require(meta.get("scope_id") == spec.receipt_id, f"cell meta scope changed: {spec.directory_id}")
    _require(meta.get("parameter_source") == "CURRENT_SCOPE_INNER_OOF_ONLY", f"cell meta source changed: {spec.directory_id}")
    _require(meta.get("frozen_epoch_reused") is False, f"cell meta reused frozen epoch: {spec.directory_id}")
    _require(meta.get("frozen_stack_reused") is False, f"cell meta reused frozen stack: {spec.directory_id}")
    _require(meta.get("frozen_gate_reused") is False, f"cell meta reused frozen gate: {spec.directory_id}")
    _require(meta.get("oof_rows") == ledger["rows"], f"cell meta/ledger row count differs: {spec.directory_id}")
    _require_sha(meta.get("oof_key_truth_sha256"), f"cell meta OOF hash missing: {spec.directory_id}")
    _require(meta.get("stack_method") == "SCIPY_NNLS_THEN_SUM_NORMALIZE_UNIFORM_IF_ALL_ZERO", f"cell stack method changed: {spec.directory_id}")
    stack_weights = meta.get("stack_weights")
    _require(isinstance(stack_weights, dict) and set(stack_weights) == {str(layer) for layer in REGISTERED_LAYERS}, f"cell stack layers changed: {spec.directory_id}")
    for layer, weights in stack_weights.items():
        _require(isinstance(weights, dict) and set(weights) == set(components), f"cell stack components changed: {spec.directory_id}/{layer}")
        numeric = [_require_finite_float(value, f"cell stack weight invalid: {spec.directory_id}/{layer}") for value in weights.values()]
        _require(all(value >= 0.0 for value in numeric), f"cell stack has negative weight: {spec.directory_id}/{layer}")
        _require(math.isclose(sum(numeric), 1.0, rel_tol=0.0, abs_tol=1e-10), f"cell stack weights do not sum to one: {spec.directory_id}/{layer}")
    gate = meta.get("gate")
    _require(
        isinstance(gate, dict)
        and set(gate) == {"feature_names", "prediction_columns", "regularization", "layers"},
        f"cell gate receipt fields changed: {spec.directory_id}",
    )
    _require(gate.get("feature_names") == list(STATE_FEATURES), f"cell gate features changed: {spec.directory_id}")
    _require(gate.get("prediction_columns") == list(components), f"cell gate contributors changed: {spec.directory_id}")
    _require(
        math.isclose(
            _require_finite_float(gate.get("regularization"), f"cell gate regularization invalid: {spec.directory_id}"),
            GATE_REGULARIZATION,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        f"cell gate regularization changed: {spec.directory_id}",
    )
    gate_layers = gate.get("layers")
    _require(isinstance(gate_layers, dict) and set(gate_layers) == {str(layer) for layer in REGISTERED_LAYERS}, f"cell gate layers changed: {spec.directory_id}")
    for layer, fitted in gate_layers.items():
        _require(
            isinstance(fitted, dict)
            and set(fitted)
            == {"prior", "coefficient_sha256", "optimizer_iterations", "objective_mse"},
            f"cell fitted gate fields changed: {spec.directory_id}/{layer}",
        )
        prior = fitted.get("prior")
        _require(isinstance(prior, list) and len(prior) == len(components), f"cell gate prior changed: {spec.directory_id}/{layer}")
        prior_values = [_require_finite_float(value, f"cell gate prior invalid: {spec.directory_id}/{layer}") for value in prior]
        _require(all(value >= 0.0 for value in prior_values), f"cell gate prior negative: {spec.directory_id}/{layer}")
        _require(math.isclose(sum(prior_values), 1.0, rel_tol=0.0, abs_tol=1e-10), f"cell gate prior does not sum to one: {spec.directory_id}/{layer}")
        _require_sha(fitted.get("coefficient_sha256"), f"cell gate coefficient hash missing: {spec.directory_id}/{layer}")
        _require_nonnegative_int(fitted.get("optimizer_iterations"), f"cell gate optimizer count invalid: {spec.directory_id}/{layer}")
        objective = _require_finite_float(fitted.get("objective_mse"), f"cell gate objective invalid: {spec.directory_id}/{layer}")
        _require(objective >= 0.0, f"cell gate objective negative: {spec.directory_id}/{layer}")
    postprocess = receipt.get("postprocess")
    _require(
        isinstance(postprocess, dict)
        and set(postprocess)
        == {
            "rows",
            "deep_projection_active_rows",
            "soft_route_rows",
            "final_projection_active_rows",
            "minimum",
            "maximum",
        },
        f"cell postprocess receipt changed: {spec.directory_id}",
    )
    post_rows = _require_nonnegative_int(postprocess.get("rows"), f"cell postprocess rows invalid: {spec.directory_id}")
    _require(post_rows > 0, f"cell postprocess is empty: {spec.directory_id}")
    _require(
        post_rows == spec.expected_outer_rows,
        f"cell outer rows differ from sealed population: {spec.directory_id}",
    )
    for name in ("deep_projection_active_rows", "soft_route_rows", "final_projection_active_rows"):
        count = _require_nonnegative_int(postprocess.get(name), f"cell postprocess count invalid: {spec.directory_id}/{name}")
        _require(count <= post_rows, f"cell postprocess count exceeds rows: {spec.directory_id}/{name}")
    minimum = _require_finite_float(postprocess.get("minimum"), f"cell postprocess minimum invalid: {spec.directory_id}")
    maximum = _require_finite_float(postprocess.get("maximum"), f"cell postprocess maximum invalid: {spec.directory_id}")
    _require(minimum <= maximum, f"cell postprocess range invalid: {spec.directory_id}")
    guards = receipt.get("guards")
    _require(isinstance(guards, dict), f"cell guards missing: {spec.directory_id}")
    expected_guards = {
        "joint_temp_psal_mask": True,
        "seven_day_embargo": True,
        "future_or_outer_labels_in_fit": False,
        "current_scope_meta_only": True,
        "frozen_epoch_stack_gate_reuse": False,
    }
    _require(set(guards) == set(expected_guards), f"cell guard field set changed: {spec.directory_id}")
    _require(all(guards.get(key) is value for key, value in expected_guards.items()), f"cell guard changed: {spec.directory_id}")
    inner_receipts = receipt.get("component_receipts")
    full_receipts = receipt.get("full_receipts")
    _require(isinstance(inner_receipts, list), f"cell inner receipts missing: {spec.directory_id}")
    _require(isinstance(full_receipts, list), f"cell full receipts missing: {spec.directory_id}")
    _require(len(inner_receipts) == len(spec.inner_folds) * len(components), f"cell inner receipt count changed: {spec.directory_id}")
    _require(len(full_receipts) == len(components), f"cell full receipt count changed: {spec.directory_id}")
    deep_components = tuple(component for component in components if component != "router_400")
    selected_epochs = receipt.get("selected_inner_epochs")
    full_epochs = receipt.get("full_prefix_epochs")
    _require(isinstance(selected_epochs, dict), f"selected epochs missing: {spec.directory_id}")
    _require(isinstance(full_epochs, dict), f"full epochs missing: {spec.directory_id}")
    _require(set(selected_epochs) == set(deep_components), f"selected epoch components changed: {spec.directory_id}")
    _require(set(full_epochs) == set(deep_components), f"full epoch components changed: {spec.directory_id}")
    for component in deep_components:
        epochs = selected_epochs[component]
        _require(
            isinstance(epochs, list)
            and len(epochs) == len(spec.inner_folds)
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 1 for value in epochs),
            f"selected epoch ledger invalid: {spec.directory_id}/{component}",
        )
        expected_selected_epochs = [
            job_receipts[
                f"{spec.scope_id}__s{spec.pipeline_seed}__{phase}__{component}"
            ]["best_epoch"]
            for phase in spec.inner_folds
        ]
        _require(
            epochs == expected_selected_epochs,
            f"selected epoch/job receipts differ: {spec.directory_id}/{component}",
        )
        _require(
            full_epochs[component] == sorted(epochs)[len(epochs) // 2],
            f"full epoch is not the registered median: {spec.directory_id}/{component}",
        )
        _require(
            job_receipts[
                f"{spec.scope_id}__s{spec.pipeline_seed}__full__{component}"
            ]["epochs"]
            == full_epochs[component],
            f"full epoch/job receipt differs: {spec.directory_id}/{component}",
        )
    expected_inner = [
        job_receipts[f"{spec.scope_id}__s{spec.pipeline_seed}__{phase}__{component}"]
        for phase in spec.inner_folds
        for component in components
    ]
    expected_full = [
        job_receipts[f"{spec.scope_id}__s{spec.pipeline_seed}__full__{component}"]
        for component in components
    ]
    _require(inner_receipts == expected_inner, f"cell inner/job receipt binding changed: {spec.directory_id}")
    _require(full_receipts == expected_full, f"cell full/job receipt binding changed: {spec.directory_id}")


def _verify_cell_store(
    actual_dir: Path,
    contract: PostExecutionContract,
    job_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    root = actual_dir / "cells"
    _require(root.is_dir() and not root.is_symlink(), "cell store is unsafe")
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    _require(all(path.is_dir() and not path.is_symlink() for path in entries), "cell store entry is unsafe")
    _require({path.name for path in entries} == set(contract.cells), "completed cell id set is not exactly sealed 45-cell graph")
    ledger: list[dict[str, Any]] = []
    for path in entries:
        spec = contract.cells[path.name]
        receipt, row = _verify_store_manifest(
            path,
            identifier=path.name,
            contract_sha256=str(contract.binding["job_store_contract_sha256"]),
        )
        manifest = _read_json(path / "manifest.json")
        _require(set(manifest["payload_files"]) == {"meta.joblib"}, f"cell payload changed: {path.name}")
        _require(
            set(manifest["files"]) == {"prediction.parquet", "receipt.json", "meta.joblib"},
            f"cell manifest file set changed: {path.name}",
        )
        _verify_cell_receipt(
            receipt,
            spec,
            job_receipts=job_receipts,
            components=contract.components,
        )
        ledger.append(row)
    return {
        "completed_cells": len(entries),
        "expected_cells": len(contract.cells),
        "ordered_manifest_ledger_sha256": canonical_sha256(ledger),
        "ordered_manifest_ledger": ledger,
        "cell_prediction_values_opened": 0,
        "meta_joblib_values_opened": 0,
        "manifest_file_hashes_verified": sum(row["payload_count"] for row in ledger),
    }


def _recompute_metric(frame: pd.DataFrame, prediction: np.ndarray, layers: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(prediction, dtype=float)
    truth = frame["truth"].to_numpy(float)
    _require(values.shape == truth.shape and np.isfinite(values).all(), "metric prediction is invalid")
    _require(np.isfinite(truth).all(), "metric truth is invalid")
    work = frame.loc[:, ["fold", "layer"]].copy()
    work["error2"] = (values - truth) ** 2
    by_fold: dict[str, Any] = {}
    fold_layer_mse: list[float] = []
    for fold, part in work.groupby("fold", sort=True):
        layer_mse: list[float] = []
        layer_report: dict[str, float] = {}
        for layer in layers:
            selected = part["layer"].astype(int).eq(layer)
            _require(bool(selected.any()), f"fold {fold} lacks registered layer {layer}")
            mse = float(part.loc[selected, "error2"].mean())
            layer_mse.append(mse)
            fold_layer_mse.append(mse)
            layer_report[str(layer)] = float(np.sqrt(mse))
        by_fold[str(fold)] = {
            "rows": int(len(part)),
            "row_pooled_rmse_c": float(np.sqrt(part["error2"].mean())),
            "layer_equal_rmse_c": float(np.sqrt(np.mean(layer_mse))),
            "by_layer_rmse_c": layer_report,
        }
    by_layer = {
        str(layer): float(
            np.sqrt(work.loc[work["layer"].astype(int).eq(layer), "error2"].mean())
        )
        for layer in layers
    }
    return {
        "rows": int(len(frame)),
        "fold_equal_layer_equal_rmse_c": float(np.sqrt(np.mean(fold_layer_mse))),
        "fixed_historical_row_weighted_rmse_c": float(np.sqrt(work["error2"].mean())),
        "by_fold": by_fold,
        "by_layer_rmse_c": by_layer,
        "maximum_absolute_error_c": float(np.max(np.abs(values - truth), initial=0.0)),
    }


def _compare_metric(expected: Any, actual: Any, path: str = "metric") -> None:
    if isinstance(expected, dict):
        _require(isinstance(actual, dict), f"{path} type changed")
        _require(set(expected) == set(actual), f"{path} keys changed")
        for key in expected:
            _compare_metric(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, bool) or isinstance(expected, str) or expected is None:
        _require(actual == expected, f"{path} changed")
    elif isinstance(expected, (int, np.integer)):
        _require(int(actual) == int(expected), f"{path} changed")
    else:
        _require(
            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12),
            f"{path} changed",
        )


def _verify_aggregates(
    actual_dir: Path,
    contract: PostExecutionContract,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    metrics_by_prefix = result.get("metrics_by_prefix")
    _require(isinstance(metrics_by_prefix, dict), "result prefix metrics missing")
    _require(set(metrics_by_prefix) == set(contract.tokens), "result prefix token set changed")
    expected_columns = [
        *OUTER_KEY_COLUMNS,
        "truth",
        *(f"seed_{seed}" for seed in contract.seeds),
        *contract.family_settings,
    ]
    expected_prefix_fields = {
        "rows",
        "seed_columns",
        "metrics_by_setting",
        "paired_day_bootstrap_vs_incumbent",
        "complementarity",
        "materialization_diagnostics",
        "per_seed_setting_count",
        "ordered_key_truth_sha256",
        "evaluated_oof_publish",
    }
    population_digest: str | None = None
    summary: dict[str, Any] = {}
    for token in contract.tokens:
        path = actual_dir / f"evaluated_oof_{token}.parquet"
        _require(path.is_file() and not path.is_symlink(), f"aggregate missing: {token}")
        prefix_result = metrics_by_prefix[token]
        _require(isinstance(prefix_result, dict), f"prefix result invalid: {token}")
        _require(set(prefix_result) == expected_prefix_fields, f"prefix result fields changed: {token}")
        _require(prefix_result.get("rows") == contract.expected_rows_per_fraction, f"prefix result rows changed: {token}")
        expected_seed_columns = [f"seed_{seed}" for seed in contract.seeds]
        _require(prefix_result.get("seed_columns") == expected_seed_columns, f"seed column ledger changed: {token}")
        per_seed_count = prefix_result.get("per_seed_setting_count")
        _require(
            isinstance(per_seed_count, dict)
            and set(per_seed_count) == set(contract.family_settings)
            and all(value == len(contract.seeds) for value in per_seed_count.values()),
            f"per-seed setting count changed: {token}",
        )
        _require(
            isinstance(prefix_result.get("paired_day_bootstrap_vs_incumbent"), dict)
            and set(prefix_result["paired_day_bootstrap_vs_incumbent"]) == set(contract.family_settings),
            f"bootstrap setting ledger changed: {token}",
        )
        _require(
            isinstance(prefix_result.get("complementarity"), dict)
            and set(prefix_result["complementarity"]) == set(contract.family_settings),
            f"complementarity setting ledger changed: {token}",
        )
        _require(isinstance(prefix_result.get("materialization_diagnostics"), dict), f"materialization diagnostics missing: {token}")
        published = prefix_result.get("evaluated_oof_publish")
        _verify_atomic_publish_receipt(
            published,
            expected_sha256=sha256_file(path),
            expected_bytes=path.stat().st_size,
            label=f"aggregate {token}",
        )

        # This is the first prediction-value read in the verifier.  Control,
        # binding, terminal closure, and all recursive file pins have already
        # passed before this function is called.
        frame = pd.read_parquet(path)
        _require(list(frame.columns) == expected_columns, f"aggregate columns/order changed: {token}")
        _require(len(frame) == contract.expected_rows_per_fraction, f"aggregate row count changed: {token}")
        _require(not frame.loc[:, expected_columns].isna().any().any(), f"aggregate null value found: {token}")
        _require(pd.api.types.is_integer_dtype(frame["layer"].dtype), f"aggregate layer dtype changed: {token}")
        for name in ("truth", *(f"seed_{seed}" for seed in contract.seeds), *contract.family_settings):
            _require(pd.api.types.is_numeric_dtype(frame[name].dtype), f"aggregate numeric dtype changed: {token}/{name}")
        _require(
            set(frame["layer"].astype(int).unique()) == set(contract.layers),
            f"aggregate registered layers changed: {token}",
        )
        _require(set(frame["fold"].astype(str).unique()) == set(contract.outer_folds), f"outer folds changed: {token}")
        actual_fold_rows = frame["fold"].astype(str).value_counts().to_dict()
        _require(actual_fold_rows == contract.expected_fold_rows, f"outer fold denominators changed: {token}")
        actual_layer_rows = frame["layer"].astype(int).value_counts().to_dict()
        _require(actual_layer_rows == contract.expected_layer_rows, f"layer denominators changed: {token}")
        for fold in contract.outer_folds:
            selected_fold = frame.loc[frame["fold"].astype(str).eq(fold)].copy()
            fold_layer_rows = selected_fold["layer"].astype(int).value_counts().to_dict()
            _require(
                fold_layer_rows == contract.expected_fold_layer_rows[fold],
                f"fold/layer population changed: {token}/{fold}",
            )
            outer_keys = selected_fold.loc[:, ["station", "layer", "time"]].copy()
            outer_keys["station"] = outer_keys["station"].astype(str)
            outer_keys["layer"] = outer_keys["layer"].astype(int)
            outer_keys["time"] = outer_keys["time"].astype(str)
            _require(
                canonical_sha256(outer_keys.to_dict("records"))
                == contract.expected_outer_key_sha256[fold],
                f"sealed semantic outer key population changed: {token}/{fold}",
            )
        _require(not frame.duplicated(list(OUTER_KEY_COLUMNS)).any(), f"duplicate aggregate key: {token}")
        numeric_columns = [
            "truth",
            *(f"seed_{seed}" for seed in contract.seeds),
            *contract.family_settings,
        ]
        _require(np.isfinite(frame.loc[:, numeric_columns].to_numpy(float)).all(), f"nonfinite aggregate: {token}")
        digest = canonical_sha256(
            frame.loc[:, [*OUTER_KEY_COLUMNS, "truth"]].astype(str).to_dict("records")
        )
        _require(
            prefix_result.get("ordered_key_truth_sha256") == digest,
            f"aggregate population digest changed: {token}",
        )
        if population_digest is None:
            population_digest = digest
        else:
            _require(digest == population_digest, "outer population differs across prefix fractions")
        reported_metrics = prefix_result.get("metrics_by_setting")
        _require(isinstance(reported_metrics, dict), f"reported metrics missing: {token}")
        _require(set(reported_metrics) == set(contract.family_settings), f"metric setting set changed: {token}")
        recomputed: dict[str, Any] = {}
        for setting in contract.family_settings:
            metric = _recompute_metric(frame, frame[setting].to_numpy(float), contract.layers)
            _compare_metric(metric, reported_metrics[setting], f"{token}.{setting}")
            recomputed[setting] = metric
        summary[token] = {
            "rows": len(frame),
            "registered_layers": list(contract.layers),
            "outer_folds": sorted(contract.outer_folds),
            "ordered_key_truth_sha256": digest,
            "aggregate_sha256": sha256_file(path),
            "aggregate_bytes": path.stat().st_size,
            "recomputed_metrics_by_setting": recomputed,
        }
    _require(
        result.get("same_population_digest_across_fractions") == population_digest,
        "result population digest differs from aggregates",
    )
    return {
        "status": "PASS_AGGREGATE_SCORE_RECOMPUTATION_EXACT",
        "expected_rows_per_fraction": contract.expected_rows_per_fraction,
        "registered_layers": list(contract.layers),
        "prefixes": summary,
        "same_population_digest_across_fractions": population_digest,
        "official_test_sample_submission_values_read": 0,
        "job_or_model_prediction_values_read": 0,
        "aggregate_prediction_files_read": len(contract.tokens),
    }


def verify_terminal_namespace(
    actual_dir: Path,
    *,
    contract: PostExecutionContract,
) -> dict[str, Any]:
    """Return a QA receipt without writing any file."""

    actual_dir = actual_dir.resolve(strict=True)
    control = _verify_control_plane(actual_dir, contract)

    expected_root_entries = {
        "execution.lock",
        "execution_start.json",
        "attempts",
        "jobs",
        "cells",
        "result.json",
        "terminal_receipt.json",
        *(f"evaluated_oof_{token}.parquet" for token in contract.tokens),
    }
    root_entries = {path.name for path in actual_dir.iterdir()}
    _require(root_entries == expected_root_entries, "actual root entry set changed")
    all_entries = list(actual_dir.rglob("*"))
    _require(all(not path.is_symlink() for path in all_entries), "actual namespace contains a symlink")
    _require(
        all(".partial." not in path.name and not path.name.endswith(".partial") for path in all_entries),
        "actual namespace contains a partial",
    )

    result = control["result"]
    _require(set(result) == EXPECTED_RESULT_FIELDS, "terminal result field set changed")
    _require(result.get("outer_prefix_cells") == len(contract.scopes), "result scope count changed")
    _require(result.get("seeded_cells") == len(contract.cells), "result cell count changed")
    _require(result.get("top_level_component_jobs_total") == len(contract.jobs), "result job count changed")
    component_new = _require_nonnegative_int(
        result.get("component_jobs_new_this_invocation"), "component new count invalid"
    )
    component_reused = _require_nonnegative_int(
        result.get("component_jobs_reused_this_invocation"), "component reused count invalid"
    )
    cell_new = _require_nonnegative_int(
        result.get("cell_jobs_new_this_invocation"), "cell new count invalid"
    )
    cell_reused = _require_nonnegative_int(
        result.get("cell_jobs_reused_this_invocation"), "cell reused count invalid"
    )
    _require(len(contract.jobs) % len(contract.cells) == 0, "job graph is not cell-regular")
    jobs_per_new_cell = len(contract.jobs) // len(contract.cells)
    invocation_start_jobs = int(control["final_invocation_start_jobs_completed"])
    invocation_start_cells = int(control["final_invocation_start_cells_completed"])
    _require(
        component_new == len(contract.jobs) - invocation_start_jobs,
        "component new count differs from final-attempt start audit",
    )
    _require(
        component_reused
        == invocation_start_jobs - jobs_per_new_cell * invocation_start_cells,
        "component reused count differs from final-attempt start audit",
    )
    _require(
        cell_new == len(contract.cells) - invocation_start_cells,
        "cell new count differs from final-attempt start audit",
    )
    _require(
        cell_reused == invocation_start_cells,
        "cell reused count differs from final-attempt start audit",
    )
    _require(
        component_new + component_reused == jobs_per_new_cell * cell_new,
        "final invocation component count is inconsistent with newly executed cells",
    )
    _require(
        cell_new + cell_reused == len(contract.cells),
        "final invocation did not materialize exact cell graph",
    )
    _require(component_new + component_reused <= len(contract.jobs), "component invocation count exceeds graph")
    _require(result.get("underlying_base_estimator_fits_total") == contract.expected_base_fits, "base estimator fit contract changed")
    _require(result.get("underlying_deep_fits_total") == contract.expected_deep_fits, "deep fit contract changed")
    _require(result.get("underlying_lightgbm_fits_total") == contract.expected_lightgbm_fits, "LightGBM fit contract changed")
    _require(result.get("meta_optimizations_total") == contract.expected_meta_optimizations, "meta optimization contract changed")
    _require(result.get("scientific_surface_inherited_byte_pinned_from_v3") is True, "v3 science pin missing")

    jobs, job_receipts = _verify_job_store(actual_dir, contract)
    cells = _verify_cell_store(actual_dir, contract, job_receipts)
    aggregates = _verify_aggregates(actual_dir, contract, result)
    return {
        "schema_version": "p2_authoritative_nested_surrogate_postexecution_qa.v5",
        "status": "PASS_TERMINAL_V5_RECURSIVE_GRAPH_AND_SCORE_QA",
        "verified_at_kst": datetime.now().astimezone().isoformat(),
        "actual_namespace": contract.namespace,
        "actual_namespace_path": str(actual_dir),
        "readiness": contract.readiness_receipt,
        "execution_binding_sha256": control["execution_binding_sha256"],
        "control_plane": {key: value for key, value in control.items() if key != "result"},
        "attempt_budget": {
            "initial": 1,
            "maximum_resume_attempts": MAXIMUM_RESUME_ATTEMPTS,
            "maximum_total_attempts": MAXIMUM_TOTAL_ATTEMPTS,
            "observed_total_attempts": control["total_attempts"],
        },
        "graph": {
            "outer_scopes": len(contract.scopes),
            "seeded_cells": len(contract.cells),
            "top_level_component_jobs": len(contract.jobs),
            "seeds": list(contract.seeds),
            "fractions": list(EXPECTED_FRACTIONS),
            "components": list(contract.components),
            "registered_layers": list(contract.layers),
        },
        "job_store": jobs,
        "cell_store": cells,
        "aggregate_score_recomputation": aggregates,
        "partials_found": 0,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
        "submission_files_generated": 0,
        "uploads": 0,
        "p3_process_mutations": 0,
        "model_or_checkpoint_values_opened": 0,
    }


def _report_ko(qa: Mapping[str, Any]) -> str:
    control = qa["control_plane"]
    graph = qa["graph"]
    aggregate = qa["aggregate_score_recomputation"]
    lines = [
        "# P2 authoritative nested surrogate v5 사후 독립 QA",
        "",
        "## 결론",
        "",
        "판정은 `PASS_TERMINAL_V5_RECURSIVE_GRAPH_AND_SCORE_QA`다. terminal result와 terminal receipt의 상호 SHA/bytes pin, 실행 binding, 승인·config·runner·module·seal, attempt 이력, 전체 JobStore 및 aggregate metric 재계산이 모두 일치했다.",
        "",
        "## 실행·그래프 무결성",
        "",
        f"- 총 실행 attempt: {control['total_attempts']} (초기 1 + resume {control['resume_attempts']}, 허용 최대 3)",
        f"- outer scope / seeded cell / component job: {graph['outer_scopes']} / {graph['seeded_cells']} / {graph['top_level_component_jobs']}",
        f"- 등록 layer: {graph['registered_layers']}",
        f"- partial: {qa['partials_found']}",
        "",
        "## aggregate 점수 재계산",
        "",
    ]
    for token, row in aggregate["prefixes"].items():
        primary = row["recomputed_metrics_by_setting"]["INCUMBENT_NOOP"]
        lines.append(
            f"- p{token}: {row['rows']:,}행, INCUMBENT_NOOP fold-equal/layer-equal RMSE {primary['fold_equal_layer_equal_rmse_c']:.12f} ℃, aggregate SHA-256 `{row['aggregate_sha256']}`"
        )
    lines.extend(
        [
            "",
            "공식 P2 test/sample/submission 또는 submission candidate는 읽지 않았고, submission 생성·업로드와 P3 변경은 모두 0회다. 모델·checkpoint·개별 component prediction 값은 열지 않고 파일 hash만 검증했다.",
            "",
        ]
    )
    return "\n".join(lines)


def _publish_verified_artifacts(
    qa: dict[str, Any],
    *,
    output_dir: Path,
    verifier_path: Path,
    test_path: Path | None,
) -> dict[str, Any]:
    """Atomically publish QA only after an in-memory terminal PASS exists."""

    _require(qa.get("status") == "PASS_TERMINAL_V5_RECURSIVE_GRAPH_AND_SCORE_QA", "non-PASS QA cannot publish")
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), "postexecution QA output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = output_dir.parent / f".{output_dir.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
    partial.mkdir(exist_ok=False)
    qa_path = partial / "postexecution_qa.json"
    report_path = partial / "REPORT_KO.md"
    qa_path.write_bytes(_json_bytes(qa))
    report_path.write_text(_report_ko(qa), encoding="utf-8", newline="\n")
    outputs = {
        qa_path.name: {"sha256": sha256_file(qa_path), "bytes": qa_path.stat().st_size},
        report_path.name: {"sha256": sha256_file(report_path), "bytes": report_path.stat().st_size},
    }
    manifest = {
        "schema_version": "p2_authoritative_nested_surrogate_postexecution_manifest.v5",
        "status": qa["status"],
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "verifier": {
            "path": VERIFIER_RELATIVE,
            "sha256": sha256_file(verifier_path),
            "bytes": verifier_path.stat().st_size,
        },
        "tests": None
        if test_path is None
        else {
            "path": TEST_RELATIVE,
            "sha256": sha256_file(test_path),
            "bytes": test_path.stat().st_size,
        },
        "readiness_manifest_sha256": qa["readiness"]["readiness_manifest_sha256"],
        "preexecution_seal_sha256": qa["readiness"]["preexecution_seal_sha256"],
        "authorization_sha256": qa["readiness"]["authorization_sha256"],
        "execution_binding_sha256": qa["execution_binding_sha256"],
        "terminal_result_sha256": qa["control_plane"]["result_sha256"],
        "terminal_receipt_sha256": qa["control_plane"]["terminal_receipt_sha256"],
        "job_manifest_ledger_sha256": qa["job_store"]["ordered_manifest_ledger_sha256"],
        "cell_manifest_ledger_sha256": qa["cell_store"]["ordered_manifest_ledger_sha256"],
        "outputs": outputs,
        "official_test_sample_submission_reads": 0,
        "submission_files_generated": 0,
        "uploads": 0,
        "p3_process_mutations": 0,
    }
    manifest_path = partial / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    os.rename(partial, output_dir)
    return {
        "status": qa["status"],
        "output_dir": str(output_dir),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "qa_sha256": sha256_file(output_dir / "postexecution_qa.json"),
        "report_sha256": sha256_file(output_dir / "REPORT_KO.md"),
    }


def run_postterminal_verification(
    *,
    project_root: Path,
    readiness_dir: Path,
    config_path: Path,
    actual_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    contract = load_sealed_contract(
        project_root=project_root,
        readiness_dir=readiness_dir,
        config_path=config_path,
    )
    canonical_actual = (project_root.resolve(strict=True) / ACTUAL_RELATIVE).resolve()
    _require(actual_dir.resolve(strict=True) == canonical_actual, "noncanonical actual namespace is forbidden")
    canonical_output = (project_root.resolve(strict=True) / OUTPUT_RELATIVE).resolve()
    _require(output_dir.resolve() == canonical_output, "noncanonical postexecution output is forbidden")
    qa = verify_terminal_namespace(actual_dir, contract=contract)
    test_path = project_root / TEST_RELATIVE
    return _publish_verified_artifacts(
        qa,
        output_dir=output_dir,
        verifier_path=Path(__file__).resolve(),
        test_path=test_path if test_path.is_file() else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-terminal", action="store_true")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--readiness-dir", type=Path, default=PROJECT_ROOT / READY_RELATIVE)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / CONFIG_RELATIVE)
    parser.add_argument("--actual-dir", type=Path, default=PROJECT_ROOT / ACTUAL_RELATIVE)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / OUTPUT_RELATIVE)
    args = parser.parse_args()
    _require(args.post_terminal, "explicit --post-terminal is required; live namespaces are never inspected")
    result = run_postterminal_verification(
        project_root=args.project_root,
        readiness_dir=args.readiness_dir,
        config_path=args.config,
        actual_dir=args.actual_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
