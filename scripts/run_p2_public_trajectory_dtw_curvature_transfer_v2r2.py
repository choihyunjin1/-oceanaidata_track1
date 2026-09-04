"""Sealed single-attempt runner for P2 public-trajectory DTW Cycle 1 v2r2.

Default invocation is a read-only preflight. Numerical materialization requires
an external authorization file whose raw SHA-256 is supplied through the
sealed environment variable. This source performs no model fit in any mode.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r2_design.json"
DESIGN_SHA256 = "e90a866e69d01731704db878332c121a4f42a0258e61d8489afa9344fa8264a1"
TRIGGER_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2_trigger_resolution.json"
TRIGGER_SHA256 = "36c9a2943b14d56867784c5e1e91d2bee086eebb9247f90e14a11db6f4eaf9e9"
EXECUTION_CONFIG_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r2_execution.json"
EXECUTION_CONFIG_SHA256 = "f28b5764b13728e1db779ef767b37e1032d2ab9e41bd40821d82f1737fb88a4b"
MODULE_PATH = REPO_ROOT / "src/p2_restore/public_trajectory_dtw_v2r2.py"
TEST_PATH = REPO_ROOT / "tests/test_p2_public_trajectory_dtw_curvature_transfer_v2r2.py"
SEAL_PATH = REPO_ROOT / "artifacts/p2_public_trajectory_dtw_curvature_transfer_v2r2_preexecution/preexecution_seal.json"
AUTHORIZATION_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r2_execution_authorization.json"
AUTHORIZATION_ENV = "P2_TRAJECTORY_V2R2_EXECUTION_AUTHORIZATION_SHA256"
EXPERIMENT_ID = "p2_public_trajectory_dtw_curvature_transfer_20260826_v2r2"
RUNNER_RELATIVE = "scripts/run_p2_public_trajectory_dtw_curvature_transfer_v2r2.py"
MODULE_RELATIVE = "src/p2_restore/public_trajectory_dtw_v2r2.py"
TEST_RELATIVE = "tests/test_p2_public_trajectory_dtw_curvature_transfer_v2r2.py"
DESIGN_RELATIVE = "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r2_design.json"
TRIGGER_RELATIVE = "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2_trigger_resolution.json"
EXECUTION_CONFIG_RELATIVE = "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r2_execution.json"
SEAL_RELATIVE = "artifacts/p2_public_trajectory_dtw_curvature_transfer_v2r2_preexecution/preexecution_seal.json"
CLOSURE_MATRIX_RELATIVE = (
    "reports/p2_public_trajectory_dtw_curvature_transfer_v2r2_qa_closure_matrix_20260826.json"
)
V2_QA_RELATIVE = (
    "reports/p2_public_trajectory_dtw_curvature_transfer_v2_independent_preexecution_qa_20260826.json"
)

OBSERVATIONS_PATH = Path(
    "C:/Users/cedis/Downloads/p2/데이터셋_P2/P2_profile_restore/observations.csv"
)
EXACT_ANCHOR_PATH = REPO_ROOT / "artifacts/p2_extrapolated_soft_gate_v2/oof.parquet"
P100_ANCHOR_PATH = (
    REPO_ROOT
    / "artifacts/p2_authoritative_nested_surrogate_actual_20260825_v5/evaluated_oof_100.parquet"
)

CLAIM_PATH = REPO_ROOT / f"artifacts/_p2_trajectory_claims/{EXPERIMENT_ID}.claim.json"
JOURNAL_PATH = REPO_ROOT / f"artifacts/_p2_trajectory_attempt_journals/{EXPERIMENT_ID}.ndjson"
OOB_PATH = REPO_ROOT / f"artifacts/_p2_trajectory_terminal_receipts/{EXPERIMENT_ID}.terminal_failure.json"
FINAL_DIR = REPO_ROOT / f"artifacts/{EXPERIMENT_ID}"
STAGING_ROOT = REPO_ROOT / "artifacts/_p2_trajectory_staging"

HARD_WALL_SECONDS = 7200
MAX_PROCESS_TREE_RSS_BYTES = 1536 * 1024 * 1024
PHYSICAL_FIT_CALLS = 0
MATERIALIZATION_SLOTS = 22
EXPECTED_INNER_MATERIALIZATIONS = 18
EXPECTED_EXACT_MATERIALIZATIONS = 1
EXPECTED_P100_MATERIALIZATIONS = 3
FORBIDDEN_PATH_TOKENS = (
    "test_index",
    "sample_submission",
    "submission_candidate",
    "candidate.csv",
)


class AuthorizationError(RuntimeError):
    pass


class ExistingAttemptError(RuntimeError):
    pass


class HardWallTimeout(RuntimeError):
    pass


class MemoryCeilingExceeded(RuntimeError):
    pass


def _now_kst() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value


def _safe_project_path(path: Path, *, allow_external_observations: bool = False) -> Path:
    resolved = path.resolve(strict=False)
    text = str(resolved).lower().replace("\\", "/")
    if any(token in text for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError("forbidden official or candidate path token")
    if allow_external_observations:
        if resolved != OBSERVATIONS_PATH.resolve(strict=False) or resolved.name != "observations.csv":
            raise ValueError("only the sealed historical observations path is allowed externally")
    elif REPO_ROOT.resolve() not in resolved.parents and resolved != REPO_ROOT.resolve():
        raise ValueError("project path escaped repository root")
    return resolved


def _verify_design() -> dict[str, Any]:
    if _sha256(DESIGN_PATH) != DESIGN_SHA256:
        raise RuntimeError("sealed Cycle-1-v2r2 design hash changed")
    design = _read_json(DESIGN_PATH)
    if design.get("schema") != "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED":
        raise ValueError("design schema changed")
    if design.get("status") != "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED":
        raise ValueError("design status changed")
    if design.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("design experiment id changed")
    supersession = design["supersession"]
    if (
        supersession["superseded_experiment_id"]
        != "p2_public_trajectory_dtw_curvature_transfer_20260826_v2"
        or supersession["superseded_status"]
        != "SUPERSEDED_CRITICAL_P0_DO_NOT_USE"
        or supersession["superseded_files_mutated"] is not False
        or supersession["scientific_results_from_v2"] != 0
        or supersession["v2_execution_or_materialization_occurred"] is not False
    ):
        raise ValueError("v2 critical-P0 supersession lineage changed")
    lineage = design["lineage"]
    base = lineage["base_v1_design"]
    if (
        base["sha256"]
        != "341b41b79f867208de0d1494d3ea6c45108b648e87f6c347178de955897779fb"
        or lineage["scientific_hypothesis_search_splits_metrics_and_gates_changed"]
        is not False
        or lineage["operational_fix_only"] is not True
    ):
        raise ValueError("v1 scientific lineage changed")
    if _sha256(TRIGGER_PATH) != TRIGGER_SHA256:
        raise RuntimeError("sealed v2 trigger-resolution hash changed")
    trigger = _read_json(TRIGGER_PATH)
    if (
        trigger.get("status") != "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
        or trigger.get("experiment_id")
        != "p2_public_trajectory_dtw_curvature_transfer_20260826_v2"
        or trigger["predecessor_evidence"]["scientific_outcome"]
        != "UNEVALUATED_INFRASTRUCTURE_FAILURE"
        or trigger["prospective_branch_resolution"]["ncr_predictions_or_metrics_consumed"]
        is not False
    ):
        raise ValueError("trigger resolution does not preserve the fail-closed frozen-exact branch")
    if _sha256(EXECUTION_CONFIG_PATH) != EXECUTION_CONFIG_SHA256:
        raise RuntimeError("sealed v2r2 execution-config hash changed")
    execution_config = _read_json(EXECUTION_CONFIG_PATH)
    if (
        execution_config.get("status") != "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
        or execution_config.get("authorized") is not False
        or execution_config["operation_ceiling"]["physical_fit_calls"] != 0
        or execution_config["operation_ceiling"]["total_materialization_slots"] != 22
    ):
        raise ValueError("v2r2 execution config authorization or ceiling changed")
    scientific = design["unchanged_scientific_contract"]
    if scientific["cells"] != [
        "d1_k3",
        "d1_k7",
        "d3_k3",
        "d3_k7",
        "d7_k3",
        "d7_k7",
    ]:
        raise ValueError("six-cell search changed")
    if (
        scientific["inner_windows"] != ["2024-03", "2024-05", "2024-07"]
        or scientific["exact_rows"] != 26273
        or scientific["p100_rows"] != 78156
        or scientific["model_fit_calls"] != 0
        or scientific["maximum_materializations"] != 22
    ):
        raise ValueError("scientific split or zero-fit budget changed")
    if scientific["exact_RESEARCH_GO"] != {
        "delta_rmse_c_lte": -0.060,
        "ci90_upper_c_lte": -0.040,
        "each_layer_delta_lte": 0.003,
        "layer4_delta_lt": 0.0,
        "weekly_p90_lte": 0.015,
    }:
        raise ValueError("exact research gate changed")
    if scientific["p100_RESEARCH_GO"] != {
        "delta_rmse_c_lt": 0.0,
        "ci90_upper_c_lt": 0.0,
        "minimum_improving_folds": 2,
        "minimum_improving_layers": 2,
        "worst_fold_lte": 0.010,
        "worst_layer_lte": 0.005,
    }:
        raise ValueError("p100 research gate changed")
    binding = design["p100_canonical_metadata_binding"]
    if (
        binding["canonical_key"] != ["fold", "station", "UTC-ns time", "layer"]
        or binding["positional_assignment_between_different_sort_orders"] is not False
        or binding["ordering_invariance_required"] is not True
    ):
        raise ValueError("p100 canonical metadata binding changed")
    if design["resource_ceiling"]["hard_wall_seconds"] != HARD_WALL_SECONDS:
        raise ValueError("hard wall changed")
    if design["trigger_resolution"] != {
        "branch": "FROZEN_EXACT_INCUMBENT_ONLY",
        "ncr_v1r7_scientific_outcome": "UNEVALUATED_INFRASTRUCTURE_FAILURE",
        "ncr_predictions_metrics_or_fit_outputs_consumed": False,
        "independent_qa_must_accept_trigger_resolution": True,
        "fail_closed_if_any_transitive_pin_or_lineage_fails": True,
    }:
        raise ValueError("v2r2 trigger branch changed")
    prohibitions = design["prohibitions"]
    for key in (
        "official_test_reads",
        "sample_submission_reads",
        "submission_candidate_reads",
        "candidate_files",
        "uploads",
        "result_driven_reruns",
    ):
        if prohibitions.get(key) != 0:
            raise ValueError(f"design prohibition changed: {key}")
    inputs = design["transitive_static_inputs"]
    paths = [str(item["path"]) for item in inputs]
    if len(inputs) != 14 or len(set(paths)) != len(paths):
        raise ValueError("transitive static input inventory is incomplete or duplicated")
    for item in inputs:
        _verify_pin(_safe_project_path(REPO_ROOT / item["path"]), item)
    drift = design["documented_nonconsumed_drift"]
    if (
        drift["current_v1r6_path_state"]["sha256"]
        != "a0492740b5aea4c441c58d112a5ee9d0ec1b95e02bbc3df4d850e5412ab2c99a"
        or "prohibited" not in drift["effect"].lower()
    ):
        raise ValueError("obsolete v1r6 nonconsumed-drift record changed")
    _verify_pin(
        _safe_project_path(REPO_ROOT / drift["current_v1r6_path_state"]["path"]),
        drift["current_v1r6_path_state"],
    )
    return design


def _verify_pin(path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"sealed file missing: {path.name}")
    size = path.stat().st_size
    digest = _sha256(path)
    if size != int(pin["bytes"]) or digest != str(pin["sha256"]):
        raise RuntimeError(f"sealed file pin mismatch: {path.name}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def _verify_seal() -> tuple[dict[str, Any], dict[str, Any]]:
    design = _verify_design()
    seal = _read_json(SEAL_PATH)
    if seal.get("status") != "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA":
        raise ValueError("preexecution seal status changed")
    if seal.get("experiment_id") != EXPERIMENT_ID or seal.get("design_sha256") != DESIGN_SHA256:
        raise ValueError("preexecution seal lineage changed")
    if (
        seal.get("base_v1_design_sha256")
        != "341b41b79f867208de0d1494d3ea6c45108b648e87f6c347178de955897779fb"
        or seal.get("trigger_resolution_sha256") != TRIGGER_SHA256
        or seal.get("execution_config_sha256") != EXECUTION_CONFIG_SHA256
    ):
        raise ValueError("preexecution seal transitive lineage changed")
    bundle = seal["bundle"]
    expected_bundle = {
        DESIGN_RELATIVE,
        RUNNER_RELATIVE,
        MODULE_RELATIVE,
        TEST_RELATIVE,
        TRIGGER_RELATIVE,
        EXECUTION_CONFIG_RELATIVE,
        CLOSURE_MATRIX_RELATIVE,
        V2_QA_RELATIVE,
    }
    if set(bundle) != expected_bundle:
        raise ValueError("preexecution seal bundle is incomplete or contains extras")
    verified = {
        relative: _verify_pin(
            Path(__file__).resolve()
            if relative == RUNNER_RELATIVE
            else _safe_project_path(REPO_ROOT / relative),
            bundle[relative],
        )
        for relative in sorted(expected_bundle)
    }
    sealed_transitive = seal["transitive_static_inputs"]
    designed_transitive = design["transitive_static_inputs"]
    if sealed_transitive != designed_transitive:
        raise ValueError("seal transitive inventory differs from prospective design")
    transitive_verified = {
        str(item["path"]): _verify_pin(
            _safe_project_path(REPO_ROOT / item["path"]),
            item,
        )
        for item in sealed_transitive
    }
    predecessor = _verify_predecessor_evidence()
    expected_versions = seal["runtime_versions"]
    if expected_versions["python"] != ".".join(map(str, sys.version_info[:3])):
        raise RuntimeError("Python runtime differs from seal")
    for package, expected in expected_versions["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"runtime package differs from seal: {package}")
    return seal, {
        "bundle": verified,
        "transitive_static_inputs": transitive_verified,
        "predecessor_semantics": predecessor,
        "runtime_versions": expected_versions,
    }


def _verify_predecessor_evidence() -> dict[str, Any]:
    """Semantically verify the pinned infrastructure-failure trigger branch."""

    claim_path = REPO_ROOT / (
        "artifacts/_ncr_stage1_claims/"
        "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r7.claim.json"
    )
    journal_path = REPO_ROOT / (
        "artifacts/_ncr_stage1_attempt_journals/"
        "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r7.ndjson"
    )
    claim = _read_json(claim_path)
    events = _read_journal(journal_path)
    names = [event.get("event") for event in events]
    if (
        claim.get("experiment_id")
        != "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r7"
        or claim.get("permanent_no_rerun_claim") is not True
        or names.count("FIT_SLOT_RESERVED") != 3
        or names.count("FIT_SLOT_COMPLETED") != 3
        or names[-1:] != ["ATTEMPT_TERMINAL_FAILED"]
        or "WORKER_FAILED" not in names
    ):
        raise RuntimeError("predecessor does not prove one terminal infrastructure failure")
    worker_failure = next(event for event in events if event.get("event") == "WORKER_FAILED")
    terminal = events[-1]
    if (
        worker_failure.get("phase") != "ALL_FITS_COMPLETED_BUILDING_RESULT"
        or worker_failure.get("error_type") != "ValueError"
        or terminal.get("automatic_rerun_allowed") is not False
        or terminal.get("final_exists") is not False
        or terminal.get("worker_returncode") != 1
        or terminal.get("reason") != "WORKER_NONZERO_EXIT"
    ):
        raise RuntimeError("predecessor failure phase or no-rerun semantics changed")
    forbidden_result_fields = {"rmse", "metrics", "predictions", "research_go"}
    if any(forbidden_result_fields.intersection(event) for event in events):
        raise RuntimeError("predecessor journal unexpectedly contains scientific result fields")
    return {
        "experiment_id": claim["experiment_id"],
        "terminal_event": terminal["event"],
        "failure_phase": worker_failure["phase"],
        "physical_fit_slots_completed": 3,
        "scientific_metrics_consumed": 0,
        "trigger_branch": "FROZEN_EXACT_INCUMBENT_ONLY",
    }


def _authorization_state(*, require_authorized: bool) -> tuple[dict[str, Any], str]:
    authorization = _read_json(AUTHORIZATION_PATH)
    raw_hash = _sha256(AUTHORIZATION_PATH)
    required = {
        "schema_version",
        "status",
        "experiment_id",
        "authorized",
        "design_sha256",
        "trigger_resolution_sha256",
        "seal_sha256",
        "bundle",
        "operation_ceiling",
        "independent_qa",
        "blockers",
    }
    if set(authorization) != required:
        raise ValueError("external authorization schema drift")
    if authorization["experiment_id"] != EXPERIMENT_ID or authorization["design_sha256"] != DESIGN_SHA256:
        raise AuthorizationError("external authorization lineage mismatch")
    if authorization["trigger_resolution_sha256"] != TRIGGER_SHA256:
        raise AuthorizationError("external authorization trigger-resolution mismatch")
    if authorization["seal_sha256"] != _sha256(SEAL_PATH):
        raise AuthorizationError("external authorization seal hash mismatch")
    bundle = authorization["bundle"]
    if set(bundle) != {"static_files", "independent_qa_report"}:
        raise ValueError("authorization bundle schema drift")
    static_files = bundle["static_files"]
    expected_static = {
        DESIGN_RELATIVE,
        TRIGGER_RELATIVE,
        EXECUTION_CONFIG_RELATIVE,
        RUNNER_RELATIVE,
        MODULE_RELATIVE,
        TEST_RELATIVE,
        SEAL_RELATIVE,
        CLOSURE_MATRIX_RELATIVE,
    }
    if set(static_files) != expected_static:
        raise AuthorizationError("authorization static bundle is incomplete or contains extras")
    for relative, pin in static_files.items():
        _verify_pin(_safe_project_path(REPO_ROOT / relative), pin)
    ceiling = authorization["operation_ceiling"]
    expected_ceiling = {
        "attempts": 1,
        "physical_fit_calls": 0,
        "inner_materializations": 18,
        "exact_materializations": 1,
        "conditional_p100_materializations": 3,
        "total_materialization_slots": 22,
        "result_driven_reruns": 0,
        "candidate_files": 0,
        "uploads": 0,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
    }
    if ceiling != expected_ceiling:
        raise AuthorizationError("authorization operation ceiling changed")
    if require_authorized:
        supplied = os.environ.get(AUTHORIZATION_ENV, "").strip().lower()
        if supplied != raw_hash:
            raise AuthorizationError("external authorization SHA-256 environment binding failed")
        if authorization["authorized"] is not True or authorization["status"] != "AUTHORIZED_AFTER_INDEPENDENT_QA":
            raise AuthorizationError("execution is not authorized after independent QA")
        independent = authorization["independent_qa"]
        qa_pin = bundle["independent_qa_report"]
        if independent != qa_pin:
            raise AuthorizationError("independent QA pin is not identical to its bundle entry")
        required_qa = {"path", "bytes", "sha256", "verdict", "design_sha256", "seal_sha256"}
        if set(qa_pin) != required_qa:
            raise AuthorizationError("independent QA receipt pin schema changed")
        if (
            not isinstance(qa_pin["path"], str)
            or not isinstance(qa_pin["bytes"], int)
            or qa_pin["bytes"] <= 0
            or not isinstance(qa_pin["sha256"], str)
            or len(qa_pin["sha256"]) != 64
            or qa_pin["verdict"] != "PASS"
            or qa_pin["design_sha256"] != DESIGN_SHA256
            or qa_pin["seal_sha256"] != _sha256(SEAL_PATH)
        ):
            raise AuthorizationError("independent QA proof is absent or has wrong lineage")
        qa_path = _safe_project_path(REPO_ROOT / qa_pin["path"])
        _verify_pin(qa_path, qa_pin)
        qa_report = _read_json(qa_path)
        if (
            qa_report.get("experiment_id") != EXPERIMENT_ID
            or qa_report.get("verdict") != "PASS"
            or qa_report.get("design_sha256") != DESIGN_SHA256
            or qa_report.get("seal_sha256") != _sha256(SEAL_PATH)
        ):
            raise AuthorizationError("independent QA report content or lineage mismatch")
        if authorization["blockers"]:
            raise AuthorizationError("execution authorization still records blockers")
    else:
        pending = bundle["independent_qa_report"]
        if authorization["independent_qa"] != pending:
            raise AuthorizationError("pending QA pin differs from authorization bundle")
        if pending != {
            "path": None,
            "bytes": None,
            "sha256": None,
            "verdict": None,
            "design_sha256": DESIGN_SHA256,
            "seal_sha256": None,
        }:
            raise AuthorizationError("pending authorization unexpectedly carries QA authority")
        if authorization["authorized"] is not False or authorization["status"] != "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA":
            raise AuthorizationError("default authorization template is not fail-closed")
    return authorization, raw_hash


@contextmanager
def _held_verified_bytes(path: Path, pin: Mapping[str, Any]) -> Iterator[tuple[BinaryIO, bytes, str]]:
    handle = path.open("rb")
    try:
        payload = handle.read()
        digest = _sha256_bytes(payload)
        if len(payload) != int(pin["bytes"]) or digest != str(pin["sha256"]):
            raise RuntimeError(f"held-byte source pin mismatch: {path.name}")
        if handle.closed:
            raise RuntimeError("held source handle closed before parsing")
        yield handle, payload, digest
        if handle.closed:
            raise RuntimeError("held source handle closed during parsing")
        if _sha256_bytes(payload) != digest:
            raise RuntimeError("captured held bytes changed in memory")
    finally:
        handle.close()


def _preexact_source_pins(
    seal: Mapping[str, Any],
) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    """Resolve only sources permitted before the durable exact gate."""

    sources = seal["historical_sources"]
    return {
        "observations": (
            _safe_project_path(OBSERVATIONS_PATH, allow_external_observations=True),
            sources["observations"],
        ),
        "exact_anchor": (
            _safe_project_path(EXACT_ANCHOR_PATH),
            sources["exact_anchor"],
        ),
    }


def _p100_literal_pin(seal: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return sealed metadata without constructing, resolving, or probing its path."""

    pin = seal["historical_sources"]["p100_anchor"]
    if set(pin) != {"path_literal", "bytes", "sha256"}:
        raise ValueError("p100 literal-pin schema changed")
    if pin["path_literal"] != str(P100_ANCHOR_PATH).replace("\\", "/"):
        raise ValueError("p100 literal path changed")
    return pin


def _source_readiness(seal: Mapping[str, Any]) -> dict[str, Any]:
    """Verify only sources legal before the exact gate; never touch p100."""

    pins = _preexact_source_pins(seal)
    p100_pin = _p100_literal_pin(seal)
    readiness = {
        "observations": _verify_pin(*pins["observations"]),
        "exact_anchor": _verify_pin(*pins["exact_anchor"]),
        "p100_anchor": {
            "state": "DEFERRED_UNREAD_UNHASHED_UNPARSED_UNTIL_EXACT_RESEARCH_GO",
            "bytes_pin": int(p100_pin["bytes"]),
            "sha256_pin": str(p100_pin["sha256"]),
            "filesystem_accesses": 0,
        },
    }
    return readiness


def _control_state() -> dict[str, Any]:
    staging = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in STAGING_ROOT.parent.glob(f"{STAGING_ROOT.name}/{EXPERIMENT_ID}.*")
        if path.exists()
    )
    return {
        "claim_exists": CLAIM_PATH.exists(),
        "journal_exists": JOURNAL_PATH.exists(),
        "oob_terminal_exists": OOB_PATH.exists(),
        "final_exists": FINAL_DIR.exists(),
        "staging_paths": staging,
        "clean_for_single_attempt": not any(
            (CLAIM_PATH.exists(), JOURNAL_PATH.exists(), OOB_PATH.exists(), FINAL_DIR.exists(), bool(staging))
        ),
    }


def read_only_preflight() -> dict[str, Any]:
    seal, verification = _verify_seal()
    authorization, authorization_hash = _authorization_state(require_authorized=False)
    sources = _source_readiness(seal)
    state = _control_state()
    status = "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
    if not state["clean_for_single_attempt"]:
        status = "BLOCKED_EXISTING_SINGLE_ATTEMPT_STATE"
    return {
        "schema_version": "p2_public_trajectory_dtw_v2r2.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "read_only": True,
        "design_sha256": DESIGN_SHA256,
        "seal_sha256": _sha256(SEAL_PATH),
        "authorization_sha256": authorization_hash,
        "authorization_status": authorization["status"],
        "authorized": authorization["authorized"],
        "blockers": authorization["blockers"],
        "static_verification": verification,
        "historical_source_readiness": sources,
        "control_state": state,
        "operation_counters": {
            "attempts": 0,
            "physical_fit_calls": 0,
            "materializations": 0,
            "scores": 0,
            "candidate_files": 0,
            "uploads": 0,
            "official_test_reads": 0,
            "sample_submission_reads": 0,
            "submission_candidate_reads": 0,
        },
    }


def _fsync_directory(path: Path) -> bool:
    """Durably flush directory metadata or fail closed."""

    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return True

    import ctypes
    from ctypes import wintypes

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x80000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x02000000,
        None,
    )
    invalid = wintypes.HANDLE(-1).value
    if handle == invalid:
        raise ctypes.WinError()
    try:
        if ctypes.windll.kernel32.FlushFileBuffers(handle):
            return True
        error_code = int(ctypes.windll.kernel32.GetLastError())
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    if error_code not in {1, 5, 6, 87}:
        raise ctypes.WinError(error_code)
    # Windows does not guarantee FlushFileBuffers on a directory handle.
    # A file created in that directory with a write-through fsync is the
    # durable metadata barrier used when the native directory call refuses.
    barrier = path / ".p2_dtw_v2r2_directory_fsync"
    descriptor = os.open(barrier, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle_file:
            handle_file.write(b"p2-dtw-v2r2-directory-durability-barrier\n")
            handle_file.flush()
            os.fsync(handle_file.fileno())
    finally:
        os.close(descriptor)
    return True


def _flush_file_path(path: Path) -> bool:
    """Flush a regular file using a handle valid on the supported platform."""

    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return True

    import ctypes
    from ctypes import wintypes

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x80000000 | 0x40000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x80000000,
        None,
    )
    invalid = wintypes.HANDLE(-1).value
    if handle == invalid:
        raise ctypes.WinError()
    try:
        if not ctypes.windll.kernel32.FlushFileBuffers(handle):
            raise ctypes.WinError()
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    return True


def _exclusive_create_bytes(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return _fsync_directory(path.parent)


def _exclusive_create_json(path: Path, value: Mapping[str, Any]) -> bool:
    return _exclusive_create_bytes(path, _canonical_bytes(value))


def _append_journal(path: Path, event: Mapping[str, Any]) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(event)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, "ab", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _atomic_replace_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    """Write+fsync a unique temp, atomically install it, and fsync the directory."""

    if path.exists():
        raise FileExistsError(f"atomic destination already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    token = _sha256_bytes(payload + os.urandom(16))[:20]
    temporary = path.parent / f".{path.name}.{token}.tmp"
    _exclusive_create_bytes(temporary, payload)
    if path.exists():
        raise FileExistsError(f"atomic destination appeared concurrently: {path.name}")
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    observed = path.read_bytes()
    if observed != payload:
        raise RuntimeError("atomic result byte identity verification failed")
    return {
        "bytes": len(observed),
        "sha256": _sha256_bytes(observed),
        "file_fsync": True,
        "directory_fsync": True,
        "atomic_replace": True,
    }


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return _atomic_replace_bytes(path, _canonical_bytes(value))


def _read_journal(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("journal line must be a JSON object")
        events.append(value)
    return events


def _inventory_paths(paths: Sequence[Path]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        else:
            expanded.append(path)
    for path in sorted(set(expanded), key=lambda item: str(item).lower()):
        safe = _safe_project_path(path)
        if safe.is_file():
            inventory.append(
                {
                    "path": str(safe.relative_to(REPO_ROOT)),
                    "exists": True,
                    "bytes": safe.stat().st_size,
                    "sha256": _sha256(safe),
                }
            )
        else:
            inventory.append(
                {
                    "path": str(safe.relative_to(REPO_ROOT)),
                    "exists": False,
                    "bytes": 0,
                    "sha256": None,
                }
            )
    return inventory


def _attempt_inventory() -> list[dict[str, Any]]:
    paths = [CLAIM_PATH, JOURNAL_PATH, FINAL_DIR]
    if STAGING_ROOT.exists():
        paths.extend(STAGING_ROOT.glob(f"{EXPERIMENT_ID}.*"))
    return _inventory_paths(paths)


def _verify_oob_receipt(
    receipt: Mapping[str, Any],
    *,
    attempt_id: str,
    authorization_hash: str,
) -> None:
    if (
        receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("attempt_id") != attempt_id
        or receipt.get("design_sha256") != DESIGN_SHA256
        or receipt.get("trigger_resolution_sha256") != TRIGGER_SHA256
        or receipt.get("authorization_sha256") != authorization_hash
        or receipt.get("status") != "TERMINAL_FAILURE_NO_RERUN"
    ):
        raise RuntimeError("OOB terminal identity verification failed")
    inventory = receipt.get("observed_inventory")
    if not isinstance(inventory, list):
        raise RuntimeError("OOB observed inventory is absent")
    for item in inventory:
        path = _safe_project_path(REPO_ROOT / str(item["path"]))
        if bool(item["exists"]) != path.is_file():
            raise RuntimeError("OOB inventory existence changed")
        if path.is_file() and (
            path.stat().st_size != int(item["bytes"])
            or _sha256(path) != str(item["sha256"])
        ):
            raise RuntimeError("OOB inventory byte identity changed")


def _write_oob_receipt(
    *,
    attempt_id: str,
    authorization_hash: str,
    reason: str,
    error: BaseException,
    phase: str,
    accounting: Mapping[str, Any],
    secondary_error: BaseException | None = None,
) -> dict[str, Any]:
    if FINAL_DIR.joinpath("terminal_success.json").exists():
        raise RuntimeError("refusing failure OOB after durable terminal success")
    if OOB_PATH.exists():
        existing = _read_json(OOB_PATH)
        _verify_oob_receipt(
            existing,
            attempt_id=attempt_id,
            authorization_hash=authorization_hash,
        )
        return existing
    receipt = {
        "schema_version": "p2_public_trajectory_dtw_v2r2.terminal_failure.v2",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "design_sha256": DESIGN_SHA256,
        "trigger_resolution_sha256": TRIGGER_SHA256,
        "authorization_sha256": authorization_hash,
        "status": "TERMINAL_FAILURE_NO_RERUN",
        "reason": reason,
        "phase": phase,
        "at_kst": _now_kst(),
        "physical_fit_calls": 0,
        "materialization_accounting": dict(accounting),
        "observed_inventory": _attempt_inventory(),
        "error": sanitized_error_provenance(error, phase=phase),
        "secondary_error": (
            sanitized_error_provenance(
                secondary_error,
                phase=f"{phase}_SECONDARY",
            )
            if secondary_error is not None
            else None
        ),
        "process_tree_termination": getattr(error, "termination_evidence", None),
    }
    if _exclusive_create_json(OOB_PATH, receipt) is not True:
        raise RuntimeError("OOB durable creation was not confirmed")
    observed = _read_json(OOB_PATH)
    _verify_oob_receipt(
        observed,
        attempt_id=attempt_id,
        authorization_hash=authorization_hash,
    )
    return observed


def _terminal_state() -> dict[str, Any]:
    """Return one conflict-free state, including an incomplete commit boundary."""

    sources: list[dict[str, str]] = []
    commit_ready_events: list[dict[str, Any]] = []
    if JOURNAL_PATH.exists():
        for event in _read_journal(JOURNAL_PATH):
            if event.get("event") == "ATTEMPT_TERMINAL_FAILED":
                sources.append({"source": "journal", "state": "FAILURE"})
            elif event.get("event") == "ATTEMPT_COMMIT_READY":
                commit_ready_events.append(event)
    if OOB_PATH.exists():
        sources.append({"source": "oob", "state": "FAILURE"})
    success_path = FINAL_DIR / "terminal_success.json"
    if success_path.exists():
        if len(commit_ready_events) != 1:
            raise RuntimeError("terminal success lacks exactly one durable commit-ready event")
        terminal = _read_json(success_path)
        ready_hash = _sha256_bytes(_canonical_bytes(commit_ready_events[0]))
        if (
            terminal.get("experiment_id") != EXPERIMENT_ID
            or terminal.get("attempt_id") != commit_ready_events[0].get("attempt_id")
            or terminal.get("commit_ready_event_sha256") != ready_hash
        ):
            raise RuntimeError("terminal success identity differs from commit-ready event")
        sources.append({"source": "final", "state": "SUCCESS"})
    states = {row["state"] for row in sources}
    if len(states) > 1:
        raise RuntimeError("conflicting durable terminal success and failure states")
    if len(commit_ready_events) > 1:
        raise RuntimeError("multiple commit-ready records")
    journal_failures = [row for row in sources if row["source"] == "journal"]
    if len(journal_failures) > 1:
        raise RuntimeError("multiple journal terminal records")
    logical_state = next(iter(states), "NONE")
    if logical_state == "NONE" and commit_ready_events:
        logical_state = "COMMIT_INCOMPLETE"
    return {
        "logical_state": logical_state,
        "sources": sources,
        "commit_ready": bool(commit_ready_events),
        "conflict_free": True,
    }


def _event(event: str, attempt_id: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": "p2_public_trajectory_dtw_v2r2.attempt_journal.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "event": event,
        "at_kst": _now_kst(),
        "physical_fit_calls": 0,
        **fields,
    }


def _sanitize_message(message: str) -> str:
    value = message.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"[A-Za-z]:[\\/][^\s]+", "<ABSOLUTE_PATH_REDACTED>", value)
    value = re.sub(r"/(?:[^\s/]+/)+[^\s]+", "<ABSOLUTE_PATH_REDACTED>", value)
    return value[:512]


def sanitized_error_provenance(error: BaseException, *, phase: str) -> dict[str, Any]:
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        raw_message = str(current)
        chain.append(
            {
                "type": type(current).__name__,
                "module": type(current).__module__,
                "message_sanitized": _sanitize_message(raw_message),
                "message_sha256": _sha256_bytes(raw_message.encode("utf-8", errors="replace")),
            }
        )
        current = current.__cause__ or current.__context__
    extracted = traceback.extract_tb(error.__traceback__)
    frames = [
        {
            "file": Path(frame.filename).name,
            "module": Path(frame.filename).stem,
            "function": frame.name,
            "line": int(frame.lineno),
        }
        for frame in extracted
    ]
    raw_traceback = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return {
        "phase": phase,
        "chain": chain,
        "frames": frames,
        "traceback_sha256": _sha256_bytes(raw_traceback.encode("utf-8", errors="replace")),
        "traceback_frame_count": len(frames),
        "locals_captured": False,
        "raw_traceback_persisted": False,
    }


def _attempt_id(authorization_hash: str) -> str:
    payload = f"{EXPERIMENT_ID}|{DESIGN_SHA256}|{authorization_hash}".encode()
    return _sha256_bytes(payload)[:24]


def _acquire_attempt(
    authorization_hash: str,
    *,
    create_json: Callable[[Path, Mapping[str, Any]], bool] = _exclusive_create_json,
    append_event: Callable[[Path, Mapping[str, Any]], None] = _append_journal,
) -> dict[str, Any]:
    state = _control_state()
    if not state["clean_for_single_attempt"]:
        raise ExistingAttemptError("single-attempt namespace is already consumed")
    attempt_id = _attempt_id(authorization_hash)
    claim = {
        "schema_version": "p2_public_trajectory_dtw_v2r2.claim.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "claimed_at_kst": _now_kst(),
        "design_sha256": DESIGN_SHA256,
        "seal_sha256": _sha256(SEAL_PATH),
        "authorization_sha256": authorization_hash,
        "physical_fit_calls_ceiling": 0,
        "materialization_slots": MATERIALIZATION_SLOTS,
        "automatic_reruns": 0,
    }
    if create_json(CLAIM_PATH, claim) is not True:
        raise RuntimeError("permanent claim durable creation was not confirmed")
    try:
        append_event(
            JOURNAL_PATH,
            _event(
                "ATTEMPT_CLAIMED",
                attempt_id,
                design_sha256=DESIGN_SHA256,
                authorization_sha256=authorization_hash,
                materialization_slots=MATERIALIZATION_SLOTS,
            ),
        )
    except BaseException as error:
        _write_oob_receipt(
            attempt_id=attempt_id,
            authorization_hash=authorization_hash,
            reason="JOURNAL_INITIALIZATION_FAILED_AFTER_CLAIM",
            error=error,
            phase="JOURNAL_INITIALIZATION",
            accounting={
                "states": {},
                "reserved": 0,
                "completed": 0,
                "failed": 0,
                "skipped_gate": 0,
                "physical_fit_calls": 0,
            },
        )
        raise
    return claim


def _materialization_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    states: dict[int, str] = {}
    for event in events:
        name = event.get("event")
        if name not in {
            "MATERIALIZATION_RESERVED",
            "MATERIALIZATION_COMPLETED",
            "MATERIALIZATION_FAILED",
            "MATERIALIZATION_SKIPPED_GATE",
        }:
            continue
        slot = int(event["slot"])
        if not 1 <= slot <= MATERIALIZATION_SLOTS:
            raise RuntimeError("journal materialization slot outside sealed range")
        previous = states.get(slot)
        if name == "MATERIALIZATION_RESERVED":
            if previous is not None:
                raise RuntimeError("materialization slot reserved more than once")
            states[slot] = "RESERVED"
        elif name == "MATERIALIZATION_COMPLETED":
            if previous != "RESERVED":
                raise RuntimeError("materialization completed without reservation")
            states[slot] = "COMPLETED"
        elif name == "MATERIALIZATION_FAILED":
            if previous != "RESERVED":
                raise RuntimeError("materialization failed without reservation")
            states[slot] = "FAILED"
        elif name == "MATERIALIZATION_SKIPPED_GATE":
            if previous is not None:
                raise RuntimeError("skipped materialization slot was already consumed")
            states[slot] = "SKIPPED_GATE"
    return {
        "states": {str(key): value for key, value in sorted(states.items())},
        "reserved": sum(value in {"RESERVED", "COMPLETED", "FAILED"} for value in states.values()),
        "completed": sum(value == "COMPLETED" for value in states.values()),
        "failed": sum(value == "FAILED" for value in states.values()),
        "skipped_gate": sum(value == "SKIPPED_GATE" for value in states.values()),
        "physical_fit_calls": 0,
    }


def _record_terminal_failure(
    error: BaseException,
    *,
    attempt_id: str,
    phase: str,
    authorization_hash: str | None = None,
) -> None:
    provenance = sanitized_error_provenance(error, phase=phase)
    if authorization_hash is None:
        claim = _read_json(CLAIM_PATH)
        authorization_hash = str(claim["authorization_sha256"])
    success_marker = FINAL_DIR / "terminal_success.json"
    if success_marker.exists():
        state = _terminal_state()
        if state["logical_state"] != "SUCCESS":
            raise RuntimeError("durable success marker conflicts with terminal state")
        return
    counts: dict[str, Any]
    try:
        events = _read_journal(JOURNAL_PATH)
        if any(event.get("event") == "ATTEMPT_TERMINAL_FAILED" for event in events):
            _terminal_state()
            return
        counts = _materialization_counts(events)
        _append_journal(
            JOURNAL_PATH,
            _event(
                "ATTEMPT_TERMINAL_FAILED",
                attempt_id,
                status="TERMINAL_FAILURE_NO_RERUN",
                error=provenance,
                process_tree_termination=getattr(error, "termination_evidence", None),
                materialization_accounting=counts,
            ),
        )
        if _terminal_state()["logical_state"] != "FAILURE":
            raise RuntimeError("journal terminal failure did not become durable")
        return
    except BaseException as journal_error:
        counts = {
            "states": {},
            "reserved": 0,
            "completed": 0,
            "failed": 0,
            "skipped_gate": 0,
            "physical_fit_calls": 0,
        }
        _write_oob_receipt(
            attempt_id=attempt_id,
            authorization_hash=authorization_hash,
            reason="JOURNAL_UNAVAILABLE_OR_TORN",
            error=error,
            phase="TERMINAL_FAILURE_LOGGING",
            accounting=counts,
            secondary_error=journal_error,
        )
        _terminal_state()


def _hardlink_create_only(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise RuntimeError("create-only hardlink destination exists") from error
    except OSError as error:
        if error.errno in {errno.EEXIST, errno.EACCES, errno.EPERM}:
            raise RuntimeError("create-only hardlink publication refused") from error
        raise
    if not os.path.samefile(source, destination):
        raise RuntimeError("hardlink publication did not preserve file identity")
    _flush_file_path(destination)
    if source.stat().st_size != destination.stat().st_size or _sha256(source) != _sha256(destination):
        raise RuntimeError("hardlink publication byte identity mismatch")


def _terminal_hardlink_last(source: Path, destination: Path) -> None:
    """Create the terminal link and perform deliberately no follow-up I/O."""

    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise RuntimeError("create-only terminal hardlink destination exists") from error
    except OSError as error:
        if error.errno in {errno.EEXIST, errno.EACCES, errno.EPERM}:
            raise RuntimeError("create-only terminal hardlink publication refused") from error
        raise


def _publish_aggregate(
    result: Mapping[str, Any],
    *,
    attempt_id: str,
    link_fn: Callable[[Path, Path], None] = _hardlink_create_only,
    terminal_link_fn: Callable[[Path, Path], None] = _terminal_hardlink_last,
    append_event: Callable[[Path, Mapping[str, Any]], None] = _append_journal,
    sync_directory: Callable[[Path], bool] = _fsync_directory,
) -> dict[str, Any]:
    if FINAL_DIR.exists():
        raise RuntimeError("final directory already exists")
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    staging = STAGING_ROOT / f"{EXPERIMENT_ID}.{attempt_id}"
    staging.mkdir(exist_ok=False)
    sync_directory(STAGING_ROOT)
    result_path = staging / "result.json"
    result_write = _atomic_replace_json(result_path, result)
    result_hash = _sha256(result_path)
    manifest = {
        "schema_version": "p2_public_trajectory_dtw_v2r2.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "status": "COMPLETE_LOCAL_RESEARCH_ONLY",
        "design_sha256": DESIGN_SHA256,
        "trigger_resolution_sha256": TRIGGER_SHA256,
        "seal_sha256": _sha256(SEAL_PATH),
        "physical_fit_calls": 0,
        "materialization_slots": MATERIALIZATION_SLOTS,
        "aggregate_only": True,
        "row_predictions_written": False,
        "csv_files_written": False,
        "candidate_files": 0,
        "uploads": 0,
        "files": {"result.json": {"bytes": result_path.stat().st_size, "sha256": result_hash}},
        "atomic_result_write": result_write,
    }
    manifest_path = staging / "manifest.json"
    _atomic_replace_json(manifest_path, manifest)
    manifest_hash = _sha256(manifest_path)
    sync_directory(staging)
    FINAL_DIR.mkdir(parents=False, exist_ok=False)
    sync_directory(FINAL_DIR.parent)
    link_fn(result_path, FINAL_DIR / "result.json")
    link_fn(manifest_path, FINAL_DIR / "manifest.json")
    sync_directory(FINAL_DIR)
    sync_directory(FINAL_DIR.parent)
    counts = _materialization_counts(_read_journal(JOURNAL_PATH))
    publication = {
        "result_sha256": result_hash,
        "manifest_sha256": manifest_hash,
        "publication": "CREATE_ONLY_HARDLINK_COMMIT_READY_TERMINAL_LAST",
        "post_terminal_repository_io": 0,
    }
    ready_event = _event(
        "ATTEMPT_COMMIT_READY",
        attempt_id,
        status="COMMIT_READY_NOT_YET_SUCCESS",
        materialization_accounting=counts,
        publication=dict(publication),
    )
    append_event(JOURNAL_PATH, ready_event)
    ready_hash = _sha256_bytes(_canonical_bytes(ready_event))
    terminal = {
        "schema_version": "p2_public_trajectory_dtw_v2r2.terminal_success.v2",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "result_sha256": result_hash,
        "manifest_sha256": manifest_hash,
        "commit_ready_event_sha256": ready_hash,
        "committed_at_kst": _now_kst(),
    }
    terminal_path = staging / "terminal_success.json"
    _atomic_replace_json(terminal_path, terminal)
    terminal_hash = _sha256(terminal_path)
    publication["terminal_success_sha256"] = terminal_hash
    publication["commit_ready_event_sha256"] = ready_hash
    sync_directory(staging)
    sync_directory(FINAL_DIR)
    sync_directory(FINAL_DIR.parent)
    # This call is intentionally the last repository filesystem operation.
    terminal_link_fn(terminal_path, FINAL_DIR / "terminal_success.json")
    return publication


def _validate_worker_prelaunch(attempt_id: str, authorization_hash: str) -> None:
    events = _read_journal(JOURNAL_PATH)
    if [event.get("event") for event in events] != [
        "ATTEMPT_CLAIMED",
        "PARENT_LAUNCHING_SINGLE_WORKER",
    ]:
        raise RuntimeError("worker prelaunch journal order differs from sealed two-event contract")
    for event in events:
        if event.get("attempt_id") != attempt_id or event.get("physical_fit_calls") != 0:
            raise RuntimeError("worker prelaunch attempt or fit-zero contract mismatch")
    claim = _read_json(CLAIM_PATH)
    if claim.get("attempt_id") != attempt_id or claim.get("authorization_sha256") != authorization_hash:
        raise RuntimeError("worker claim binding mismatch")


def _load_sealed_numerical_module() -> Any:
    """Load the verified file directly, bypassing mutable package ``__init__``."""

    module_name = "_sealed_p2_public_trajectory_dtw_v2r2"
    specification = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("sealed numerical module could not be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _run_internal_worker(attempt_id: str, authorization_hash: str) -> dict[str, Any]:
    seal, _ = _verify_seal()
    _, observed_auth_hash = _authorization_state(require_authorized=True)
    if authorization_hash != observed_auth_hash:
        raise AuthorizationError("worker authorization hash argument mismatch")
    _validate_worker_prelaunch(attempt_id, authorization_hash)
    pins = _preexact_source_pins(seal)
    p100_pin = _p100_literal_pin(seal)
    with _held_verified_bytes(*pins["observations"]) as (_, observation_bytes, _), _held_verified_bytes(
        *pins["exact_anchor"]
    ) as (_, exact_bytes, _):
        import pandas as pd

        numerical_module = _load_sealed_numerical_module()

        observations = pd.read_csv(
            io.BytesIO(observation_bytes),
            dtype={"station": "string", "time": "string"},
        )
        exact = pd.read_parquet(io.BytesIO(exact_bytes))
        materializer = numerical_module.HistoricalTrajectoryMaterializer(observations, exact)
        deadline = float(_read_json(CLAIM_PATH)["deadline_monotonic"])
        p100_load_count = 0

        def on_slot(slot: int, state: str, details: Mapping[str, Any]) -> None:
            names = {
                "RESERVED": "MATERIALIZATION_RESERVED",
                "COMPLETED": "MATERIALIZATION_COMPLETED",
                "FAILED": "MATERIALIZATION_FAILED",
                "SKIPPED_GATE": "MATERIALIZATION_SKIPPED_GATE",
            }
            if state not in names:
                raise ValueError("unknown materialization accounting state")
            _append_journal(
                JOURNAL_PATH,
                _event(names[state], attempt_id, slot=int(slot), details=dict(details)),
            )

        def load_p100_after_exact_gate() -> None:
            nonlocal p100_load_count
            if p100_load_count != 0:
                raise RuntimeError("locked p100 lazy loader called more than once")
            events = _read_journal(JOURNAL_PATH)
            exact_events = [
                event
                for event in events
                if event.get("event") == "MATERIALIZATION_COMPLETED"
                and event.get("slot") == 19
                and event.get("details", {}).get("gate") == "RESEARCH_GO"
            ]
            if len(exact_events) != 1:
                raise RuntimeError("p100 access attempted before one durable exact RESEARCH_GO")
            _append_journal(
                JOURNAL_PATH,
                _event("P100_LAZY_ACCESS_STARTED", attempt_id, after_exact_slot=19),
            )
            p100_path = _safe_project_path(P100_ANCHOR_PATH)
            with _held_verified_bytes(p100_path, p100_pin) as (_, p100_bytes, digest):
                p100 = pd.read_parquet(io.BytesIO(p100_bytes))
                certificate = materializer.load_p100_anchor(p100)
            p100_load_count += 1
            _append_journal(
                JOURNAL_PATH,
                _event(
                    "P100_LAZY_ACCESS_COMPLETED",
                    attempt_id,
                    bytes=int(p100_pin["bytes"]),
                    sha256=digest,
                    truth_key_sha256=certificate["key_sha256"],
                ),
            )

        result = numerical_module.execute_zero_fit_protocol(
            materializer,
            deadline_monotonic=deadline,
            on_slot=on_slot,
            on_exact_research_go=load_p100_after_exact_gate,
        )
        if result["physical_fit_calls"] != 0 or result["materialization_slots_total"] != 22:
            raise RuntimeError("worker zero-fit/materialization accounting mismatch")
        counts = _materialization_counts(_read_journal(JOURNAL_PATH))
        if counts["reserved"] + counts["skipped_gate"] != 22 or counts["failed"] != 0:
            raise RuntimeError("worker journal does not account for all 22 slots")
        _append_journal(
            JOURNAL_PATH,
            _event(
                "WORKER_COMPLETED",
                attempt_id,
                materialization_accounting=counts,
                overall_gate=result["overall_gate"],
            ),
        )
        return result


def _process_tree_rss(pid: int) -> int:
    import psutil

    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        return int(sum(process.memory_info().rss for process in processes if process.is_running()))
    except psutil.Error:
        return 0


def _enumerate_process_tree(pid: int) -> list[dict[str, Any]]:
    import psutil

    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return []
    captured: list[dict[str, Any]] = []
    for item in processes:
        try:
            captured.append({"pid": int(item.pid), "create_time": float(item.create_time())})
        except psutil.Error:
            continue
    return sorted(captured, key=lambda row: int(row["pid"]))


def _captured_processes_still_present(captured: Sequence[Mapping[str, Any]]) -> list[int]:
    import psutil

    remaining: list[int] = []
    for item in captured:
        pid = int(item["pid"])
        try:
            observed = psutil.Process(pid)
            if abs(float(observed.create_time()) - float(item["create_time"])) < 1e-6:
                remaining.append(pid)
        except psutil.Error:
            continue
    return sorted(remaining)


def _terminate_process_tree(process: subprocess.Popen[str]) -> dict[str, Any]:
    pid = process.pid
    captured = _enumerate_process_tree(pid)
    captured_pids = [int(item["pid"]) for item in captured]
    if pid not in captured_pids:
        captured.append({"pid": pid, "create_time": -1.0})
        captured_pids.append(pid)
    evidence: dict[str, Any] = {
        "root_pid": pid,
        "platform": os.name,
        "pretermination_processes": captured,
        "pretermination_descendant_pids": sorted(value for value in captured_pids if value != pid),
        "descendants_enumerated_before_termination": True,
    }
    if os.name == "nt":
        killed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        evidence.update(
            {
                "mechanism": "taskkill_T_F",
                "returncode": killed.returncode,
                "stdout_sha256": _sha256_bytes(killed.stdout.encode("utf-8", errors="replace")),
                "stderr_sha256": _sha256_bytes(killed.stderr.encode("utf-8", errors="replace")),
            }
        )
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            evidence.update({"mechanism": "killpg_SIGKILL", "returncode": 0})
        except ProcessLookupError:
            evidence.update({"mechanism": "killpg_SIGKILL", "returncode": 0, "already_absent": True})
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired as error:
        evidence["root_absence_verified"] = False
        raise RuntimeError("worker process tree remained after forced termination") from error
    remaining = _captured_processes_still_present(
        [item for item in captured if float(item["create_time"]) >= 0.0]
    )
    evidence["posttermination_remaining_captured_pids"] = remaining
    evidence["root_absence_verified"] = process.poll() is not None and pid not in remaining
    evidence["all_captured_descendants_absent"] = not remaining
    if not evidence["root_absence_verified"] or remaining:
        raise RuntimeError("captured worker descendants remained after forced termination")
    return evidence


def _launch_worker(attempt_id: str, authorization_hash: str, deadline: float) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--attempt-id",
        attempt_id,
        "--authorization-sha256",
        authorization_hash,
    ]
    kwargs: dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": dict(os.environ),
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    peak_rss = 0
    while process.poll() is None:
        peak_rss = max(peak_rss, _process_tree_rss(process.pid))
        if peak_rss > MAX_PROCESS_TREE_RSS_BYTES:
            evidence = _terminate_process_tree(process)
            error = MemoryCeilingExceeded("worker process tree exceeded sealed RSS ceiling")
            error.termination_evidence = evidence  # type: ignore[attr-defined]
            raise error
        if time.monotonic() >= deadline:
            evidence = _terminate_process_tree(process)
            error = HardWallTimeout("worker exceeded sealed 7200-second wall")
            error.termination_evidence = evidence  # type: ignore[attr-defined]
            raise error
        time.sleep(0.25)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            "worker failed: "
            f"returncode={process.returncode},stdout_sha256={_sha256_bytes(stdout.encode())},"
            f"stderr_sha256={_sha256_bytes(stderr.encode())}"
        )
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("worker stdout protocol must contain one JSON line")
    payload = json.loads(lines[0])
    if not isinstance(payload, dict) or payload.get("physical_fit_calls") != 0:
        raise RuntimeError("worker result violates zero-fit protocol")
    payload["peak_process_tree_rss_bytes"] = peak_rss
    return payload


def _build_parent_result(worker_result: Mapping[str, Any], *, attempt_id: str) -> dict[str, Any]:
    allowed = {
        "status",
        "selected_cell",
        "inner_selection_records",
        "exact",
        "exact_gate",
        "p100",
        "p100_gate",
        "overall_gate",
        "physical_fit_calls",
        "materialization_slots_total",
        "result_driven_reruns",
        "peak_process_tree_rss_bytes",
    }
    if set(worker_result) != allowed:
        raise ValueError("worker aggregate result schema drift")
    if worker_result["physical_fit_calls"] != 0 or worker_result["materialization_slots_total"] != 22:
        raise ValueError("worker result fit/materialization ceiling drift")
    serialized = json.dumps(worker_result, ensure_ascii=False).lower()
    if any(token in serialized for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError("worker result contains a forbidden path token")
    return {
        "schema_version": "p2_public_trajectory_dtw_v2r2.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "completed_at_kst": _now_kst(),
        "status": worker_result["status"],
        "design_sha256": DESIGN_SHA256,
        "scientific_result": dict(worker_result),
        "operation_counters": {
            "attempts": 1,
            "physical_fit_calls": 0,
            "materialization_slots": 22,
            "result_driven_reruns": 0,
            "candidate_files": 0,
            "uploads": 0,
            "official_test_reads": 0,
            "sample_submission_reads": 0,
            "submission_candidate_reads": 0,
        },
        "aggregate_only": True,
    }


def _execute_parent() -> dict[str, Any]:
    phase = "STATIC_AUTHORIZATION"
    seal, _ = _verify_seal()
    _, authorization_hash = _authorization_state(require_authorized=True)
    _source_readiness(seal)
    deadline = time.monotonic() + HARD_WALL_SECONDS
    phase = "ATTEMPT_CLAIM"
    claim = _acquire_attempt(authorization_hash)
    attempt_id = str(claim["attempt_id"])
    try:
        claim_with_deadline = dict(claim)
        claim_with_deadline["deadline_monotonic"] = deadline
        # Claim is immutable. Deadline binding is a create-only companion event.
        _append_journal(
            JOURNAL_PATH,
            _event(
                "PARENT_LAUNCHING_SINGLE_WORKER",
                attempt_id,
                deadline_monotonic=deadline,
                hard_wall_seconds=HARD_WALL_SECONDS,
                max_process_tree_rss_bytes=MAX_PROCESS_TREE_RSS_BYTES,
            ),
        )
        # Worker reads the deadline from this exact launch event, not a rewritten claim.
        phase = "WORKER"
        worker_result = _launch_worker(attempt_id, authorization_hash, deadline)
        phase = "RESULT_BUILD"
        result = _build_parent_result(worker_result, attempt_id=attempt_id)
        phase = "RESULT_PUBLICATION"
        publication = _publish_aggregate(result, attempt_id=attempt_id)
        return {"result": result, "publication": publication}
    except BaseException as error:
        _record_terminal_failure(
            error,
            attempt_id=attempt_id,
            phase=phase,
            authorization_hash=authorization_hash,
        )
        raise


def _internal_worker_deadline() -> float:
    events = _read_journal(JOURNAL_PATH)
    launches = [event for event in events if event.get("event") == "PARENT_LAUNCHING_SINGLE_WORKER"]
    if len(launches) != 1:
        raise RuntimeError("worker launch deadline event missing or duplicated")
    return float(launches[0]["deadline_monotonic"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-local", action="store_true")
    mode.add_argument("--internal-worker", action="store_true")
    parser.add_argument("--attempt-id")
    parser.add_argument("--authorization-sha256")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.internal_worker:
        if not args.attempt_id or not args.authorization_sha256:
            raise RuntimeError("internal worker requires sealed attempt and authorization arguments")
        # Bind the helper expected by the numerical worker without mutating the claim.
        original_read_json = globals()["_read_json"]

        def read_json_with_deadline(path: Path) -> dict[str, Any]:
            value = original_read_json(path)
            if path.resolve() == CLAIM_PATH.resolve():
                value = dict(value)
                value["deadline_monotonic"] = _internal_worker_deadline()
            return value

        globals()["_read_json"] = read_json_with_deadline
        try:
            result = _run_internal_worker(args.attempt_id, args.authorization_sha256)
        finally:
            globals()["_read_json"] = original_read_json
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        return 0
    if args.execute_local:
        output = _execute_parent()
        print(json.dumps(output, sort_keys=True, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(read_only_preflight(), sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
