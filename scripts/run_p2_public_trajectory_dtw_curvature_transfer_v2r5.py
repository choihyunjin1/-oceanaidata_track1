"""Sealed single-attempt runner for P2 public-trajectory DTW Cycle 1 v2r5.

Default invocation is a read-only preflight. Numerical materialization requires
an external authorization file whose raw SHA-256 is supplied through the
sealed environment variable. This source performs no model fit in any mode.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.metadata
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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, BinaryIO
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r5_design.json"
DESIGN_SHA256 = "c044ae23d14f85c634d8145cbd8f85b004536e378277dc91dc00097dc78f7fe4"
TRIGGER_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2_trigger_resolution.json"
TRIGGER_SHA256 = "36c9a2943b14d56867784c5e1e91d2bee086eebb9247f90e14a11db6f4eaf9e9"
EXECUTION_CONFIG_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r5_execution.json"
EXECUTION_CONFIG_SHA256 = "32484a1118eda01d900d87f687b05766585d42cbd7c5c6c02621e28f94aa4e8e"
MODULE_PATH = REPO_ROOT / "src/p2_restore/public_trajectory_dtw_v2r5.py"
TEST_PATH = REPO_ROOT / "tests/test_p2_public_trajectory_dtw_curvature_transfer_v2r5.py"
SEAL_PATH = REPO_ROOT / "artifacts/p2_public_trajectory_dtw_curvature_transfer_v2r5_preexecution/preexecution_seal.json"
AUTHORIZATION_PATH = REPO_ROOT / "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r5_execution_authorization.json"
AUTHORIZATION_ENV = "P2_TRAJECTORY_V2R5_EXECUTION_AUTHORIZATION_SHA256"
EXPERIMENT_ID = "p2_public_trajectory_dtw_curvature_transfer_20260826_v2r5"
RUNNER_RELATIVE = "scripts/run_p2_public_trajectory_dtw_curvature_transfer_v2r5.py"
MODULE_RELATIVE = "src/p2_restore/public_trajectory_dtw_v2r5.py"
TEST_RELATIVE = "tests/test_p2_public_trajectory_dtw_curvature_transfer_v2r5.py"
DESIGN_RELATIVE = "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r5_design.json"
TRIGGER_RELATIVE = "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2_trigger_resolution.json"
EXECUTION_CONFIG_RELATIVE = "configs/experiments/p2_public_trajectory_dtw_curvature_transfer_v2r5_execution.json"
SEAL_RELATIVE = "artifacts/p2_public_trajectory_dtw_curvature_transfer_v2r5_preexecution/preexecution_seal.json"
CLOSURE_MATRIX_RELATIVE = (
    "reports/p2_public_trajectory_dtw_curvature_transfer_v2r5_qa_closure_matrix_20260826.json"
)
V2R4_QA_RELATIVE = (
    "reports/p2_public_trajectory_dtw_curvature_transfer_v2r4_independent_preexecution_qa_20260826.json"
)
DIAGNOSIS_RELATIVE = (
    "reports/p2_public_trajectory_dtw_curvature_transfer_v2r5_exact_key_metadata_20260826.json"
)

P2_DATA_DIR_ENV = "P2_DATA_DIR"
OBSERVATIONS_RELATIVE = Path("observations.csv")
EXACT_ANCHOR_PATH = REPO_ROOT / "artifacts/p2_extrapolated_soft_gate_v2/oof.parquet"
P100_ANCHOR_RELATIVE = Path(
    "artifacts/p2_authoritative_nested_surrogate_actual_20260825_v5/evaluated_oof_100.parquet"
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


@dataclass(frozen=True)
class VerifiedSnapshot:
    """Immutable bytes retained from the one allowed read of sealed identities."""

    design_bytes: bytes
    design_digest: str
    execution_bytes: bytes
    execution_digest: str
    seal_bytes: bytes
    seal_digest: str
    module_bytes: bytes
    module_digest: str
    bundle_bytes: Mapping[str, bytes]
    verification: Mapping[str, Any]


def _now_kst() -> str:
    return datetime.now(UTC).astimezone(ZoneInfo("Asia/Seoul")).isoformat()


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


def _json_object_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {label}")
    return value


def _read_byte_snapshot(path: Path) -> tuple[bytes, str]:
    """Open once and bind every downstream operation to the captured bytes."""

    with path.open("rb") as handle:
        payload = handle.read()
    return payload, _sha256_bytes(payload)


def _verify_snapshot_pin(
    payload: bytes,
    pin: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    digest = _sha256_bytes(payload)
    if len(payload) != int(pin["bytes"]) or digest != str(pin["sha256"]):
        raise RuntimeError(f"held-byte pin mismatch: {label}")
    return {"path": label, "bytes": len(payload), "sha256": digest}


def _read_pinned_snapshot(
    path: Path,
    pin: Mapping[str, Any],
    *,
    label: str,
) -> tuple[bytes, str, dict[str, Any]]:
    payload, digest = _read_byte_snapshot(path)
    verification = _verify_snapshot_pin(payload, pin, label=label)
    if verification["sha256"] != digest:
        raise RuntimeError(f"held-byte digest split: {label}")
    return payload, digest, verification


def _safe_project_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    text = str(resolved).lower().replace("\\", "/")
    if any(token in text for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError("forbidden official or candidate path token")
    if REPO_ROOT.resolve() not in resolved.parents and resolved != REPO_ROOT.resolve():
        raise ValueError("project path escaped repository root")
    return resolved


def _resolve_observations_path() -> Path:
    """Resolve exactly ``P2_DATA_DIR/observations.csv`` without enumeration."""

    raw_root = os.environ.get(P2_DATA_DIR_ENV, "")
    if not raw_root:
        raise FileNotFoundError("P2_DATA_DIR is required for historical observations readiness")
    root_literal = Path(raw_root)
    if not root_literal.is_absolute():
        raise ValueError("P2_DATA_DIR must be an absolute directory")
    root = root_literal.resolve(strict=True)
    candidate = (root / OBSERVATIONS_RELATIVE).resolve(strict=True)
    candidate_text = str(candidate).lower().replace("\\", "/")
    if (
        candidate.parent != root
        or candidate.name != OBSERVATIONS_RELATIVE.name
        or any(token in candidate_text for token in FORBIDDEN_PATH_TOKENS)
    ):
        raise ValueError("portable observations resolution violated the official-path firewall")
    return candidate


def _parse_aware_datetime(value: str, *, label: str) -> datetime:
    observed = datetime.fromisoformat(value)
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return observed


def _verify_design() -> tuple[
    dict[str, Any],
    bytes,
    str,
    dict[str, Any],
    bytes,
    str,
]:
    design_bytes, design_digest = _read_byte_snapshot(DESIGN_PATH)
    if len(design_bytes) != 16087 or design_digest != DESIGN_SHA256:
        raise RuntimeError("sealed Cycle-1-v2r5 recovery design snapshot changed")
    design = _json_object_from_bytes(design_bytes, label=DESIGN_PATH.name)
    if (
        design.get("schema") != "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
        or design.get("status") != "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
        or design.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("v2r5 recovery design identity or status changed")
    created = _parse_aware_datetime(design["created_at_kst"], label="design created_at_kst")
    if created.astimezone(UTC) > datetime.now(UTC):
        raise ValueError("design timestamp is in the future")
    supersession = design["supersession"]
    if (
        supersession["superseded_experiment_id"]
        != "p2_public_trajectory_dtw_curvature_transfer_20260826_v2r4"
        or supersession["superseded_status"]
        != "TERMINAL_INFRASTRUCTURE_FAILURE_AFTER_AUTHORIZATION_BEFORE_FIRST_MATERIALIZATION_DO_NOT_REUSE"
        or supersession["superseded_files_mutated"] is not False
        or supersession["superseded_authorization"]["authorized"] is not True
        or supersession["superseded_journal"]["reserved"] != 0
        or supersession["superseded_journal"]["completed"] != 0
        or supersession["superseded_journal"]["failed"] != 0
        or supersession["v2r4_execution_attempt_consumed"] is not True
        or supersession["v2r4_scientific_result_or_gate_consumed"] is not False
    ):
        raise ValueError("v2r4 zero-materialization supersession lineage changed")
    diagnosis = design["exact_failure_diagnosis"]
    if (
        diagnosis["sha256"]
        != "a450a642987e6c940519b7b5767da22ebcf63f5ef69b66fe8425486d31de7909"
        or diagnosis["bytes"] != 3963
        or diagnosis["failure_phase"]
        != "HistoricalTrajectoryMaterializer.__init__ before slot 1 reservation"
        or diagnosis["scientific_or_score_fields_read_from_failed_attempt"] != 0
    ):
        raise ValueError("v2r5 exact-key diagnosis lineage changed")
    portable = design["portable_source_contract"]
    observations = portable["observations"]
    p100 = portable["p100_anchor"]
    if (
        observations["root_environment"] != P2_DATA_DIR_ENV
        or observations["relative_path"] != "observations.csv"
        or observations["fallback_absolute_path"] is not None
        or observations["bytes"] != 49058719
        or observations["sha256"]
        != "cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a"
        or Path(observations["relative_path"]).is_absolute()
        or Path(p100["literal_relative_path"]).is_absolute()
        or p100["pre_exact_resolve_stat_open_hash_parse"] != 0
    ):
        raise ValueError("portable historical-source contract changed")
    amendment = design["integrity_adapter_amendment"]
    metadata = design["exact_key_metadata_freeze"]
    if (
        amendment["only_permitted_numerical_module_change"]
        != "Replace continuous-presence validation for the sparse exact anchor with canonical-grid membership and explicit sparse-key integrity diagnostics."
        or amendment["registered_rows_required"] != 26273
        or amendment["layers_required"] != [2, 3, 4]
        or amendment["canonical_step_ns"] != 600000000000
        or amendment["full_continuous_presence_required"] is not False
        or amendment["missing_union_timestamps_allowed"] is not True
        or amendment["per_layer_missing_keys_allowed"] is not True
        or amendment["independent_observation_key_truth_equality_before_scoring"]
        is not True
        or amendment["all_other_v2r4_numerical_logic_changed"] is not False
        or metadata["registered_exact_rows"] != 26273
        or metadata["unique_time_layer_keys"] != 26273
        or metadata["duplicate_time_layer_keys"] != 0
        or metadata["unique_timestamps"] != 8779
        or metadata["missing_union_timestamps"] != 5
        or metadata["gap_intervals"] != 2
        or metadata["off_grid_timestamps"] != 0
        or metadata["layer_rows"] != {"2": 8777, "3": 8774, "4": 8722}
        or metadata["kst_days"] != 61
    ):
        raise ValueError("sparse exact-key integrity amendment changed")
    scientific = design["unchanged_scientific_contract"]
    if (
        scientific["base_v1_design_sha256"]
        != "341b41b79f867208de0d1494d3ea6c45108b648e87f6c347178de955897779fb"
        or scientific["v2r4_design_sha256"]
        != "9863a283f8be657f93f710a19e88754a6ea7fcfbf7dfa51329c8c1ea848ecae5"
        or scientific["cells"]
        != ["d1_k3", "d1_k7", "d3_k3", "d3_k7", "d7_k3", "d7_k7"]
        or scientific["inner_windows"] != ["2024-03", "2024-05", "2024-07"]
        or scientific["exact_rows"] != 26273
        or scientific["p100_rows"] != 78156
        or scientific["model_fit_calls"] != 0
        or scientific["maximum_materializations"] != 22
        or scientific["result_driven_tuning_or_rerun"] is not False
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

    execution_bytes, execution_digest = _read_byte_snapshot(EXECUTION_CONFIG_PATH)
    if len(execution_bytes) != 2606 or execution_digest != EXECUTION_CONFIG_SHA256:
        raise RuntimeError("sealed v2r5 execution snapshot changed")
    execution = _json_object_from_bytes(
        execution_bytes,
        label=EXECUTION_CONFIG_PATH.name,
    )
    if (
        execution.get("schema_version")
        != "p2_public_trajectory_dtw.v2r5.execution.v1"
        or execution.get("experiment_id") != EXPERIMENT_ID
        or execution.get("status") != "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
        or execution.get("authorized") is not False
        or execution["design"]
        != {"path": DESIGN_RELATIVE, "bytes": len(design_bytes), "sha256": design_digest}
        or execution["operation_ceiling"]["physical_fit_calls"] != 0
        or execution["operation_ceiling"]["total_materialization_slots"] != 22
        or execution["portable_observations"]["fallback_absolute_path"] is not None
        or execution["predecessor_failure"]["materialization_reserved"] != 0
        or execution["predecessor_failure"]["materialization_completed"] != 0
        or execution["predecessor_failure"]["scientific_results"] != 0
        or execution["integrity_adapter"]["full_continuous_presence_required"]
        is not False
        or execution["integrity_adapter"]["all_scientific_logic_changed"] is not False
        or execution["real_initialization_readiness"]["required_before_claim"]
        is not True
        or execution["snapshot_chain"]["post_verify_seal_path_hash_or_reopen"] != 0
        or execution["snapshot_chain"]["post_verify_module_path_reopen"] != 0
    ):
        raise ValueError("v2r5 execution config changed")
    execution_created = _parse_aware_datetime(
        execution["created_at_kst"],
        label="execution created_at_kst",
    )
    if not created <= execution_created <= datetime.now(UTC).astimezone(
        ZoneInfo("Asia/Seoul")
    ):
        raise ValueError("design-to-implementation chronology is invalid")
    inputs = design["transitive_static_inputs"]
    paths = [str(item["path"]) for item in inputs]
    if len(inputs) != 22 or len(set(paths)) != len(paths):
        raise ValueError("transitive static input inventory is incomplete or duplicated")
    for key in (
        "official_test_reads",
        "sample_submission_reads",
        "submission_candidate_reads",
        "candidate_files",
        "uploads",
        "physical_fit_calls",
        "result_driven_reruns",
        "p100_access_before_exact_RESEARCH_GO",
    ):
        if design["prohibitions"].get(key) != 0:
            raise ValueError(f"design prohibition changed: {key}")
    return (
        design,
        design_bytes,
        design_digest,
        execution,
        execution_bytes,
        execution_digest,
    )

def _verify_pin(path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"sealed file missing: {path.name}")
    size = path.stat().st_size
    digest = _sha256(path)
    if size != int(pin["bytes"]) or digest != str(pin["sha256"]):
        raise RuntimeError(f"sealed file pin mismatch: {path.name}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def _verify_chronology(
    design: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    chronology = seal["chronology"]
    design_created = _parse_aware_datetime(
        str(design["created_at_kst"]),
        label="design created_at_kst",
    ).astimezone(UTC)
    implementation_completed = _parse_aware_datetime(
        str(chronology["implementation_completed_at_kst"]),
        label="implementation_completed_at_kst",
    ).astimezone(UTC)
    tests_completed = _parse_aware_datetime(
        str(chronology["tests_completed_at_kst"]),
        label="tests_completed_at_kst",
    ).astimezone(UTC)
    seal_created = _parse_aware_datetime(
        str(chronology["seal_created_at_kst"]),
        label="seal_created_at_kst",
    ).astimezone(UTC)
    now_utc = datetime.now(UTC)
    if not design_created <= implementation_completed <= tests_completed <= seal_created <= now_utc:
        raise RuntimeError("design-to-seal recorded chronology is invalid or future-dated")
    expected_paths = {
        DIAGNOSIS_RELATIVE: REPO_ROOT / DIAGNOSIS_RELATIVE,
        DESIGN_RELATIVE: DESIGN_PATH,
        EXECUTION_CONFIG_RELATIVE: EXECUTION_CONFIG_PATH,
        MODULE_RELATIVE: MODULE_PATH,
        RUNNER_RELATIVE: Path(__file__).resolve(),
        TEST_RELATIVE: TEST_PATH,
        CLOSURE_MATRIX_RELATIVE: REPO_ROOT / CLOSURE_MATRIX_RELATIVE,
    }
    recorded_mtime_ns = chronology["filesystem_mtime_ns"]
    if set(recorded_mtime_ns) != set(expected_paths):
        raise RuntimeError("chronology filesystem inventory is incomplete or contains extras")
    actual_mtime_ns: dict[str, int] = {}
    for relative, path in expected_paths.items():
        observed = int(path.stat().st_mtime_ns)
        if observed != int(recorded_mtime_ns[relative]):
            raise RuntimeError(f"chronology mtime pin changed: {relative}")
        actual_mtime_ns[relative] = observed
    design_mtime = datetime.fromtimestamp(
        actual_mtime_ns[DESIGN_RELATIVE] / 1_000_000_000,
        tz=UTC,
    )
    diagnosis_mtime = datetime.fromtimestamp(
        actual_mtime_ns[DIAGNOSIS_RELATIVE] / 1_000_000_000,
        tz=UTC,
    )
    implementation_mtimes = [
        datetime.fromtimestamp(
            actual_mtime_ns[relative] / 1_000_000_000,
            tz=UTC,
        )
        for relative in (EXECUTION_CONFIG_RELATIVE, MODULE_RELATIVE, RUNNER_RELATIVE)
    ]
    test_mtime = datetime.fromtimestamp(
        actual_mtime_ns[TEST_RELATIVE] / 1_000_000_000,
        tz=UTC,
    )
    closure_mtime = datetime.fromtimestamp(
        actual_mtime_ns[CLOSURE_MATRIX_RELATIVE] / 1_000_000_000,
        tz=UTC,
    )
    seal_mtime = datetime.fromtimestamp(SEAL_PATH.stat().st_mtime_ns / 1_000_000_000, tz=UTC)
    if not (
        diagnosis_mtime
        <= design_created
        <= design_mtime
        <= min(implementation_mtimes)
        <= max(implementation_mtimes)
        <= implementation_completed
        <= test_mtime
        <= tests_completed
        <= closure_mtime
        <= seal_created
        <= seal_mtime
        <= now_utc
    ):
        raise RuntimeError("actual filesystem chronology is nonmonotonic or future-dated")
    return {
        "timezone": "Asia/Seoul",
        "order": ["diagnosis", "design", "implementation", "tests", "seal", "preflight"],
        "filesystem_mtime_ns": actual_mtime_ns,
        "seal_mtime_utc": seal_mtime.isoformat(),
        "future_timestamp_count": 0,
        "nondecreasing": True,
    }


def _verify_seal() -> tuple[dict[str, Any], str, VerifiedSnapshot]:
    (
        design,
        design_bytes,
        design_digest,
        _execution,
        execution_bytes,
        execution_digest,
    ) = _verify_design()
    seal_bytes, seal_digest = _read_byte_snapshot(SEAL_PATH)
    seal = _json_object_from_bytes(seal_bytes, label=SEAL_PATH.name)
    if (
        seal.get("schema_version")
        != "p2_public_trajectory_dtw.preexecution_seal.v2r5"
        or seal.get("status") != "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
        or seal.get("experiment_id") != EXPERIMENT_ID
        or seal.get("design_sha256") != design_digest
        or seal.get("trigger_resolution_sha256") != TRIGGER_SHA256
        or seal.get("execution_config_sha256") != execution_digest
    ):
        raise ValueError("preexecution seal identity, status, or lineage changed")
    bundle = seal["bundle"]
    expected_bundle = {
        DESIGN_RELATIVE,
        RUNNER_RELATIVE,
        MODULE_RELATIVE,
        TEST_RELATIVE,
        TRIGGER_RELATIVE,
        EXECUTION_CONFIG_RELATIVE,
        CLOSURE_MATRIX_RELATIVE,
        V2R4_QA_RELATIVE,
        DIAGNOSIS_RELATIVE,
    }
    if set(bundle) != expected_bundle:
        raise ValueError("preexecution seal bundle is incomplete or contains extras")

    bundle_bytes: dict[str, bytes] = {
        DESIGN_RELATIVE: design_bytes,
        EXECUTION_CONFIG_RELATIVE: execution_bytes,
    }
    verified: dict[str, dict[str, Any]] = {}
    for relative in sorted(expected_bundle):
        if relative in bundle_bytes:
            payload = bundle_bytes[relative]
            verified[relative] = _verify_snapshot_pin(
                payload,
                bundle[relative],
                label=relative,
            )
            continue
        path = (
            Path(__file__).resolve()
            if relative == RUNNER_RELATIVE
            else _safe_project_path(REPO_ROOT / relative)
        )
        payload, _, proof = _read_pinned_snapshot(
            path,
            bundle[relative],
            label=relative,
        )
        bundle_bytes[relative] = payload
        verified[relative] = proof

    if seal["transitive_static_inputs"] != design["transitive_static_inputs"]:
        raise ValueError("seal transitive inventory differs from prospective design")
    transitive_bytes: dict[str, bytes] = {}
    transitive_verified: dict[str, dict[str, Any]] = {}
    for item in seal["transitive_static_inputs"]:
        relative = str(item["path"])
        if relative in bundle_bytes:
            payload = bundle_bytes[relative]
            proof = _verify_snapshot_pin(payload, item, label=relative)
        else:
            payload, _, proof = _read_pinned_snapshot(
                _safe_project_path(REPO_ROOT / relative),
                item,
                label=relative,
            )
        transitive_bytes[relative] = payload
        transitive_verified[relative] = proof

    qa_report = _json_object_from_bytes(
        bundle_bytes[V2R4_QA_RELATIVE],
        label=V2R4_QA_RELATIVE,
    )
    if (
        qa_report.get("verdict") != "PASS"
        or qa_report.get("severity_counts") != {"P0": 0, "P1": 0, "P2": 1}
        or qa_report.get("blocking_finding_count") != 0
        or qa_report.get("design_sha256")
        != "9863a283f8be657f93f710a19e88754a6ea7fcfbf7dfa51329c8c1ea848ecae5"
        or qa_report.get("seal_sha256")
        != "e19a60ee1798d2dea9f76e1902de66bbdbf44f4a743902d5528337652ff3eaf9"
    ):
        raise RuntimeError("v2r4 independent-QA evidence changed")
    diagnosis = _json_object_from_bytes(
        bundle_bytes[DIAGNOSIS_RELATIVE],
        label=DIAGNOSIS_RELATIVE,
    )
    exact_metadata = diagnosis.get("exact_2024_sep_oct", {})
    if (
        diagnosis.get("status") != "READ_ONLY_METADATA_DIAGNOSIS_NO_MATERIALIZATION"
        or exact_metadata.get("rows") != 26273
        or exact_metadata.get("unique_time_layer_keys") != 26273
        or exact_metadata.get("duplicate_time_layer_keys") != 0
        or exact_metadata.get("unique_timestamps") != 8779
        or exact_metadata.get("missing_union_timestamps") != 5
        or exact_metadata.get("gap_intervals_in_observed_union") != 2
        or exact_metadata.get("off_canonical_grid_timestamps") != 0
        or diagnosis.get("safety_accounting", {}).get("materializations") != 0
    ):
        raise RuntimeError("v2r5 sparse-key diagnosis evidence changed")
    sources = seal["historical_sources"]
    if (
        sources["observations"]
        != {
            "relative_path": "observations.csv",
            "bytes": 49058719,
            "sha256": "cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a",
        }
        or Path(sources["observations"]["relative_path"]).is_absolute()
        or Path(sources["p100_anchor"]["literal_relative_path"]).is_absolute()
    ):
        raise RuntimeError("portable sealed historical-source paths changed")
    expected_versions = seal["runtime_versions"]
    if expected_versions["python"] != ".".join(map(str, sys.version_info[:3])):
        raise RuntimeError("Python runtime differs from seal")
    for package, expected in expected_versions["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"runtime package differs from seal: {package}")
    chronology = _verify_chronology(design, seal)
    predecessor = _verify_predecessor_evidence(transitive_bytes)
    verification: dict[str, Any] = {
        "bundle": verified,
        "transitive_static_inputs": transitive_verified,
        "predecessor_semantics": predecessor,
        "runtime_versions": expected_versions,
        "chronology": chronology,
        "snapshot_binding": {
            "design_sha256": design_digest,
            "execution_sha256": execution_digest,
            "seal_sha256": seal_digest,
            "module_sha256": _sha256_bytes(bundle_bytes[MODULE_RELATIVE]),
            "seal_path_reopens_after_snapshot": 0,
            "module_path_reopens_after_snapshot": 0,
        },
    }
    snapshot = VerifiedSnapshot(
        design_bytes=design_bytes,
        design_digest=design_digest,
        execution_bytes=execution_bytes,
        execution_digest=execution_digest,
        seal_bytes=seal_bytes,
        seal_digest=seal_digest,
        module_bytes=bundle_bytes[MODULE_RELATIVE],
        module_digest=_sha256_bytes(bundle_bytes[MODULE_RELATIVE]),
        bundle_bytes=MappingProxyType(dict(bundle_bytes)),
        verification=MappingProxyType(verification),
    )
    return seal, seal_digest, snapshot


def _verify_predecessor_evidence(
    transitive_bytes: Mapping[str, bytes],
) -> dict[str, Any]:
    """Verify the consumed v2r4 attempt ended before any materialization."""

    claim_relative = (
        "artifacts/_p2_trajectory_claims/"
        "p2_public_trajectory_dtw_curvature_transfer_20260826_v2r4.claim.json"
    )
    journal_relative = (
        "artifacts/_p2_trajectory_attempt_journals/"
        "p2_public_trajectory_dtw_curvature_transfer_20260826_v2r4.ndjson"
    )
    claim = _json_object_from_bytes(
        transitive_bytes[claim_relative],
        label=claim_relative,
    )
    events = [
        json.loads(line)
        for line in transitive_bytes[journal_relative].decode("utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(event, dict) for event in events):
        raise ValueError("predecessor journal event must be an object")
    names = [event.get("event") for event in events]
    if (
        claim.get("experiment_id")
        != "p2_public_trajectory_dtw_curvature_transfer_20260826_v2r4"
        or claim.get("attempt_id") != "908becd9672518ad93afb039"
        or claim.get("materialization_slots") != 22
        or claim.get("physical_fit_calls_ceiling") != 0
        or names
        != [
            "ATTEMPT_CLAIMED",
            "PARENT_LAUNCHING_SINGLE_WORKER",
            "ATTEMPT_TERMINAL_FAILED",
        ]
    ):
        raise RuntimeError("v2r4 predecessor claim or three-event terminal lineage changed")
    terminal = events[-1]
    accounting = terminal.get("materialization_accounting", {})
    if (
        terminal.get("status") != "TERMINAL_FAILURE_NO_RERUN"
        or terminal.get("physical_fit_calls") != 0
        or accounting.get("reserved") != 0
        or accounting.get("completed") != 0
        or accounting.get("failed") != 0
        or accounting.get("skipped_gate") != 0
        or accounting.get("states") != {}
    ):
        raise RuntimeError("v2r4 predecessor is not a zero-materialization terminal failure")
    forbidden_result_fields = {"rmse", "metrics", "predictions", "research_go"}
    if any(forbidden_result_fields.intersection(event) for event in events) or any(
        isinstance(name, str) and name.startswith("MATERIALIZATION_") for name in names
    ):
        raise RuntimeError("v2r4 journal unexpectedly contains scientific result fields")
    return {
        "experiment_id": claim["experiment_id"],
        "attempt_id": claim["attempt_id"],
        "terminal_event": terminal["event"],
        "terminal_status": terminal["status"],
        "physical_fit_calls": 0,
        "materialization_reserved": 0,
        "materialization_completed": 0,
        "scientific_metrics_consumed": 0,
        "trigger_branch": "USER_AUTHORIZED_INFRASTRUCTURE_RECOVERY",
    }


def _authorization_state(
    *,
    require_authorized: bool,
    seal_digest: str,
    snapshot: VerifiedSnapshot,
) -> tuple[dict[str, Any], str]:
    if (
        snapshot.seal_digest != seal_digest
        or _sha256_bytes(snapshot.seal_bytes) != seal_digest
        or snapshot.design_digest != DESIGN_SHA256
        or snapshot.execution_digest != EXECUTION_CONFIG_SHA256
    ):
        raise AuthorizationError("validated snapshot lineage is internally inconsistent")
    authorization_bytes, raw_hash = _read_byte_snapshot(AUTHORIZATION_PATH)
    authorization = _json_object_from_bytes(
        authorization_bytes,
        label=AUTHORIZATION_PATH.name,
    )
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
    if authorization["schema_version"] != "p2_public_trajectory_dtw_v2r5.authorization.v1":
        raise ValueError("external authorization schema version changed")
    if (
        authorization["experiment_id"] != EXPERIMENT_ID
        or authorization["design_sha256"] != snapshot.design_digest
    ):
        raise AuthorizationError("external authorization lineage mismatch")
    if authorization["trigger_resolution_sha256"] != TRIGGER_SHA256:
        raise AuthorizationError("external authorization trigger-resolution mismatch")
    if authorization["seal_sha256"] != seal_digest:
        raise AuthorizationError("external authorization seal snapshot mismatch")
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
        V2R4_QA_RELATIVE,
        DIAGNOSIS_RELATIVE,
    }
    if set(static_files) != expected_static:
        raise AuthorizationError("authorization static bundle is incomplete or contains extras")
    for relative, pin in static_files.items():
        payload = (
            snapshot.seal_bytes
            if relative == SEAL_RELATIVE
            else snapshot.bundle_bytes[relative]
        )
        _verify_snapshot_pin(payload, pin, label=relative)
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
        if (
            authorization["authorized"] is not True
            or authorization["status"] != "AUTHORIZED_AFTER_INDEPENDENT_QA"
        ):
            raise AuthorizationError("execution is not authorized after independent QA")
        independent = authorization["independent_qa"]
        qa_pin = bundle["independent_qa_report"]
        if independent != qa_pin:
            raise AuthorizationError("independent QA pin is not identical to its bundle entry")
        required_qa = {
            "path",
            "bytes",
            "sha256",
            "verdict",
            "design_sha256",
            "seal_sha256",
        }
        if set(qa_pin) != required_qa:
            raise AuthorizationError("independent QA receipt pin schema changed")
        if (
            not isinstance(qa_pin["path"], str)
            or not isinstance(qa_pin["bytes"], int)
            or qa_pin["bytes"] <= 0
            or not isinstance(qa_pin["sha256"], str)
            or len(qa_pin["sha256"]) != 64
            or qa_pin["verdict"] != "PASS"
            or qa_pin["design_sha256"] != snapshot.design_digest
            or qa_pin["seal_sha256"] != seal_digest
        ):
            raise AuthorizationError("independent QA proof is absent or has wrong lineage")
        qa_path = _safe_project_path(REPO_ROOT / qa_pin["path"])
        with _held_verified_bytes(qa_path, qa_pin) as (_, qa_bytes, _):
            qa_report = _json_object_from_bytes(qa_bytes, label=qa_path.name)
        if (
            qa_report.get("experiment_id") != EXPERIMENT_ID
            or qa_report.get("verdict") != "PASS"
            or qa_report.get("design_sha256") != snapshot.design_digest
            or qa_report.get("seal_sha256") != seal_digest
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
            "design_sha256": snapshot.design_digest,
            "seal_sha256": None,
        }:
            raise AuthorizationError("pending authorization unexpectedly carries QA authority")
        if (
            authorization["authorized"] is not False
            or authorization["status"] != "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
        ):
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
            _resolve_observations_path(),
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
    if set(pin) != {"literal_relative_path", "bytes", "sha256"}:
        raise ValueError("p100 literal-pin schema changed")
    if pin["literal_relative_path"] != P100_ANCHOR_RELATIVE.as_posix():
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


def _real_exact_initialization_readiness(
    seal: Mapping[str, Any],
    snapshot: VerifiedSnapshot,
) -> dict[str, Any]:
    """Construct the real pinned materializer without scoring or materializing."""

    pins = _preexact_source_pins(seal)
    with _held_verified_bytes(*pins["observations"]) as (
        _,
        observation_bytes,
        observation_digest,
    ), _held_verified_bytes(*pins["exact_anchor"]) as (
        _,
        exact_bytes,
        exact_digest,
    ):
        import pandas as pd

        numerical_module = _load_sealed_numerical_module(snapshot)
        observations = pd.read_csv(
            io.BytesIO(observation_bytes),
            dtype={"station": "string", "time": "string"},
        )
        exact = pd.read_parquet(io.BytesIO(exact_bytes))
        materializer = numerical_module.HistoricalTrajectoryMaterializer(
            observations,
            exact,
        )
        certificate = dict(materializer.exact_truth_certificate)
    contract = certificate.get("time_contract", {})
    if (
        len(observations) != 789408
        or len(exact) != 69850
        or certificate.get("rows") != 26273
        or certificate.get("anchor_truth_equal_observations") is not True
        or contract.get("unique_time_layer_keys") != 26273
        or contract.get("duplicate_time_layer_keys") != 0
        or contract.get("unique_timestamps") != 8779
        or contract.get("missing_union_timestamps") != 5
        or contract.get("gap_intervals") != 2
        or contract.get("off_grid_timestamps") != 0
        or contract.get("layer_rows") != {"2": 8777, "3": 8774, "4": 8722}
        or contract.get("kst_days") != 61
    ):
        raise RuntimeError("real pinned exact-anchor initialization certificate changed")
    return {
        "status": "PASS_REAL_PINNED_INITIALIZATION_ONLY",
        "observations_rows": int(len(observations)),
        "full_exact_anchor_rows": int(len(exact)),
        "registered_exact_rows": int(certificate["rows"]),
        "observation_sha256": observation_digest,
        "exact_anchor_sha256": exact_digest,
        "anchor_truth_equal_observations": True,
        "time_contract": contract,
        "materialize_calls": 0,
        "score_calls": 0,
        "p100_accesses": 0,
        "physical_fit_calls": 0,
    }


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
    seal, seal_digest, snapshot = _verify_seal()
    authorization, authorization_hash = _authorization_state(
        require_authorized=False,
        seal_digest=seal_digest,
        snapshot=snapshot,
    )
    sources = _source_readiness(seal)
    real_initialization = _real_exact_initialization_readiness(seal, snapshot)
    state = _control_state()
    status = "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
    if not state["clean_for_single_attempt"]:
        status = "BLOCKED_EXISTING_SINGLE_ATTEMPT_STATE"
    return {
        "schema_version": "p2_public_trajectory_dtw_v2r5.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "read_only": True,
        "design_sha256": DESIGN_SHA256,
        "seal_sha256": seal_digest,
        "authorization_sha256": authorization_hash,
        "authorization_status": authorization["status"],
        "authorized": authorization["authorized"],
        "blockers": authorization["blockers"],
        "static_verification": dict(snapshot.verification),
        "historical_source_readiness": sources,
        "real_exact_initialization_readiness": real_initialization,
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
    barrier = path / ".p2_dtw_v2r5_directory_fsync"
    descriptor = os.open(barrier, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle_file:
            handle_file.write(b"p2-dtw-v2r5-directory-durability-barrier\n")
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
    seal_digest: str,
) -> None:
    if (
        receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("attempt_id") != attempt_id
        or receipt.get("design_sha256") != DESIGN_SHA256
        or receipt.get("trigger_resolution_sha256") != TRIGGER_SHA256
        or receipt.get("seal_sha256") != seal_digest
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
    seal_digest: str,
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
            seal_digest=seal_digest,
        )
        return existing
    receipt = {
        "schema_version": "p2_public_trajectory_dtw_v2r5.terminal_failure.v2",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "design_sha256": DESIGN_SHA256,
        "trigger_resolution_sha256": TRIGGER_SHA256,
        "seal_sha256": seal_digest,
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
        seal_digest=seal_digest,
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
            or terminal.get("seal_sha256")
            != commit_ready_events[0].get("seal_sha256")
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
        "schema_version": "p2_public_trajectory_dtw_v2r5.attempt_journal.v1",
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


def _attempt_id(authorization_hash: str, seal_digest: str) -> str:
    payload = (
        f"{EXPERIMENT_ID}|{DESIGN_SHA256}|{seal_digest}|{authorization_hash}"
    ).encode()
    return _sha256_bytes(payload)[:24]


def _acquire_attempt(
    authorization_hash: str,
    seal_digest: str,
    *,
    create_json: Callable[[Path, Mapping[str, Any]], bool] = _exclusive_create_json,
    append_event: Callable[[Path, Mapping[str, Any]], None] = _append_journal,
) -> dict[str, Any]:
    state = _control_state()
    if not state["clean_for_single_attempt"]:
        raise ExistingAttemptError("single-attempt namespace is already consumed")
    attempt_id = _attempt_id(authorization_hash, seal_digest)
    claim = {
        "schema_version": "p2_public_trajectory_dtw_v2r5.claim.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "claimed_at_kst": _now_kst(),
        "design_sha256": DESIGN_SHA256,
        "seal_sha256": seal_digest,
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
                seal_sha256=seal_digest,
                authorization_sha256=authorization_hash,
                materialization_slots=MATERIALIZATION_SLOTS,
            ),
        )
    except BaseException as error:
        _write_oob_receipt(
            attempt_id=attempt_id,
            authorization_hash=authorization_hash,
            seal_digest=seal_digest,
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
    seal_digest: str | None = None,
) -> None:
    provenance = sanitized_error_provenance(error, phase=phase)
    if authorization_hash is None:
        claim = _read_json(CLAIM_PATH)
        authorization_hash = str(claim["authorization_sha256"])
    if seal_digest is None:
        claim = _read_json(CLAIM_PATH)
        seal_digest = str(claim["seal_sha256"])
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
                seal_sha256=seal_digest,
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
            seal_digest=seal_digest,
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
    seal_digest: str,
    link_fn: Callable[[Path, Path], None] = _hardlink_create_only,
    terminal_link_fn: Callable[[Path, Path], None] = _terminal_hardlink_last,
    append_event: Callable[[Path, Mapping[str, Any]], None] = _append_journal,
    sync_directory: Callable[[Path], bool] = _fsync_directory,
) -> dict[str, Any]:
    if result.get("seal_sha256") != seal_digest:
        raise RuntimeError("result seal snapshot lineage changed before publication")
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
        "schema_version": "p2_public_trajectory_dtw_v2r5.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "status": "COMPLETE_LOCAL_RESEARCH_ONLY",
        "design_sha256": DESIGN_SHA256,
        "trigger_resolution_sha256": TRIGGER_SHA256,
        "seal_sha256": seal_digest,
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
        "seal_sha256": seal_digest,
        "result_sha256": result_hash,
        "manifest_sha256": manifest_hash,
        "publication": "CREATE_ONLY_HARDLINK_COMMIT_READY_TERMINAL_LAST",
        "post_terminal_repository_io": 0,
    }
    ready_event = _event(
        "ATTEMPT_COMMIT_READY",
        attempt_id,
        status="COMMIT_READY_NOT_YET_SUCCESS",
        seal_sha256=seal_digest,
        materialization_accounting=counts,
        publication=dict(publication),
    )
    append_event(JOURNAL_PATH, ready_event)
    ready_hash = _sha256_bytes(_canonical_bytes(ready_event))
    terminal = {
        "schema_version": "p2_public_trajectory_dtw_v2r5.terminal_success.v2",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "seal_sha256": seal_digest,
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


def _validate_worker_prelaunch(
    attempt_id: str,
    authorization_hash: str,
    seal_digest: str,
) -> None:
    events = _read_journal(JOURNAL_PATH)
    if [event.get("event") for event in events] != [
        "ATTEMPT_CLAIMED",
        "PARENT_LAUNCHING_SINGLE_WORKER",
    ]:
        raise RuntimeError("worker prelaunch journal order differs from sealed two-event contract")
    for event in events:
        if (
            event.get("attempt_id") != attempt_id
            or event.get("physical_fit_calls") != 0
            or event.get("seal_sha256") != seal_digest
        ):
            raise RuntimeError("worker prelaunch attempt, seal, or fit-zero contract mismatch")
    claim = _read_json(CLAIM_PATH)
    if (
        claim.get("attempt_id") != attempt_id
        or claim.get("authorization_sha256") != authorization_hash
        or claim.get("seal_sha256") != seal_digest
    ):
        raise RuntimeError("worker claim binding mismatch")


def _load_sealed_numerical_module(snapshot: VerifiedSnapshot) -> Any:
    """Compile and execute the module bytes retained by seal verification."""

    if (
        _sha256_bytes(snapshot.module_bytes) != snapshot.module_digest
        or snapshot.module_digest
        != _sha256_bytes(snapshot.bundle_bytes[MODULE_RELATIVE])
    ):
        raise RuntimeError("retained numerical-module snapshot changed")
    module_name = "_sealed_p2_public_trajectory_dtw_v2r5"
    code = compile(snapshot.module_bytes.decode("utf-8"), str(MODULE_PATH), "exec")
    module = ModuleType(module_name)
    module.__file__ = str(MODULE_PATH)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _run_internal_worker(
    attempt_id: str,
    authorization_hash: str,
    expected_seal_digest: str,
) -> dict[str, Any]:
    seal, seal_digest, snapshot = _verify_seal()
    if seal_digest != expected_seal_digest:
        raise AuthorizationError("worker validated seal digest differs from parent binding")
    _, observed_auth_hash = _authorization_state(
        require_authorized=True,
        seal_digest=seal_digest,
        snapshot=snapshot,
    )
    if authorization_hash != observed_auth_hash:
        raise AuthorizationError("worker authorization hash argument mismatch")
    _validate_worker_prelaunch(attempt_id, authorization_hash, seal_digest)
    pins = _preexact_source_pins(seal)
    p100_pin = _p100_literal_pin(seal)
    with _held_verified_bytes(*pins["observations"]) as (_, observation_bytes, _), _held_verified_bytes(
        *pins["exact_anchor"]
    ) as (_, exact_bytes, _):
        import pandas as pd

        numerical_module = _load_sealed_numerical_module(snapshot)

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
            p100_path = _safe_project_path(REPO_ROOT / P100_ANCHOR_RELATIVE)
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
        bound_result = dict(result)
        bound_result["seal_sha256"] = seal_digest
        return bound_result


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


def _launch_worker(
    attempt_id: str,
    authorization_hash: str,
    seal_digest: str,
    deadline: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--attempt-id",
        attempt_id,
        "--authorization-sha256",
        authorization_hash,
        "--seal-sha256",
        seal_digest,
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
    if (
        not isinstance(payload, dict)
        or payload.get("physical_fit_calls") != 0
        or payload.get("seal_sha256") != seal_digest
    ):
        raise RuntimeError("worker result violates zero-fit protocol")
    payload["peak_process_tree_rss_bytes"] = peak_rss
    return payload


def _build_parent_result(
    worker_result: Mapping[str, Any],
    *,
    attempt_id: str,
    seal_digest: str,
) -> dict[str, Any]:
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
        "seal_sha256",
    }
    if set(worker_result) != allowed:
        raise ValueError("worker aggregate result schema drift")
    if worker_result["physical_fit_calls"] != 0 or worker_result["materialization_slots_total"] != 22:
        raise ValueError("worker result fit/materialization ceiling drift")
    if worker_result["seal_sha256"] != seal_digest:
        raise ValueError("worker result seal snapshot lineage drift")
    serialized = json.dumps(worker_result, ensure_ascii=False).lower()
    if any(token in serialized for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError("worker result contains a forbidden path token")
    return {
        "schema_version": "p2_public_trajectory_dtw_v2r5.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "completed_at_kst": _now_kst(),
        "status": worker_result["status"],
        "design_sha256": DESIGN_SHA256,
        "seal_sha256": seal_digest,
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
    seal, seal_digest, snapshot = _verify_seal()
    _, authorization_hash = _authorization_state(
        require_authorized=True,
        seal_digest=seal_digest,
        snapshot=snapshot,
    )
    _source_readiness(seal)
    _real_exact_initialization_readiness(seal, snapshot)
    deadline = time.monotonic() + HARD_WALL_SECONDS
    phase = "ATTEMPT_CLAIM"
    claim = _acquire_attempt(authorization_hash, seal_digest)
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
                seal_sha256=seal_digest,
                deadline_monotonic=deadline,
                hard_wall_seconds=HARD_WALL_SECONDS,
                max_process_tree_rss_bytes=MAX_PROCESS_TREE_RSS_BYTES,
            ),
        )
        # Worker reads the deadline from this exact launch event, not a rewritten claim.
        phase = "WORKER"
        worker_result = _launch_worker(
            attempt_id,
            authorization_hash,
            seal_digest,
            deadline,
        )
        phase = "RESULT_BUILD"
        result = _build_parent_result(
            worker_result,
            attempt_id=attempt_id,
            seal_digest=seal_digest,
        )
        phase = "RESULT_PUBLICATION"
        publication = _publish_aggregate(
            result,
            attempt_id=attempt_id,
            seal_digest=seal_digest,
        )
        return {"result": result, "publication": publication}
    except BaseException as error:
        _record_terminal_failure(
            error,
            attempt_id=attempt_id,
            phase=phase,
            authorization_hash=authorization_hash,
            seal_digest=seal_digest,
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
    parser.add_argument("--seal-sha256")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.internal_worker:
        if not args.attempt_id or not args.authorization_sha256 or not args.seal_sha256:
            raise RuntimeError(
                "internal worker requires sealed attempt, authorization, and seal arguments"
            )
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
            result = _run_internal_worker(
                args.attempt_id,
                args.authorization_sha256,
                args.seal_sha256,
            )
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
