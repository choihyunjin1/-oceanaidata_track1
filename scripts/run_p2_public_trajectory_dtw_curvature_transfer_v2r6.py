"""Read-only terminal preflight for blocked P2 DTW Cycle-1 v2r6.

The registered March inner window contains no finite target truth.  This runner
can never create a claim or launch a worker.  Its only data-bearing operation is
an aggregate-only finite-mask certificate over pinned historical inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_public_trajectory_dtw_curvature_transfer_20260826_v2r6"

DESIGN_RELATIVE = (
    "configs/experiments/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_design.json"
)
TRIGGER_RELATIVE = (
    "configs/experiments/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_trigger_resolution.json"
)
EXECUTION_RELATIVE = (
    "configs/experiments/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_execution.json"
)
MODULE_RELATIVE = "src/p2_restore/public_trajectory_dtw_v2r6.py"
RUNNER_RELATIVE = "scripts/run_p2_public_trajectory_dtw_curvature_transfer_v2r6.py"
TEST_RELATIVE = "tests/test_p2_public_trajectory_dtw_curvature_transfer_v2r6.py"
SLOT_DIAGNOSTIC_RELATIVE = (
    "reports/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_"
    "slot1_error_only_diagnostic_20260826.json"
)
IDENTIFIABILITY_RELATIVE = (
    "reports/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_"
    "inner_identifiability_certificate_20260826.json"
)
CLOSURE_RELATIVE = (
    "reports/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_"
    "qa_closure_matrix_20260826.json"
)
SEAL_RELATIVE = (
    "artifacts/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_preexecution/"
    "preexecution_seal.json"
)
AUTHORIZATION_RELATIVE = (
    "configs/experiments/"
    "p2_public_trajectory_dtw_curvature_transfer_v2r6_execution_authorization.json"
)

DESIGN_PATH = REPO_ROOT / DESIGN_RELATIVE
TRIGGER_PATH = REPO_ROOT / TRIGGER_RELATIVE
EXECUTION_PATH = REPO_ROOT / EXECUTION_RELATIVE
MODULE_PATH = REPO_ROOT / MODULE_RELATIVE
TEST_PATH = REPO_ROOT / TEST_RELATIVE
SLOT_DIAGNOSTIC_PATH = REPO_ROOT / SLOT_DIAGNOSTIC_RELATIVE
IDENTIFIABILITY_PATH = REPO_ROOT / IDENTIFIABILITY_RELATIVE
CLOSURE_PATH = REPO_ROOT / CLOSURE_RELATIVE
SEAL_PATH = REPO_ROOT / SEAL_RELATIVE
AUTHORIZATION_PATH = REPO_ROOT / AUTHORIZATION_RELATIVE

V2R5_MODULE_RELATIVE = "src/p2_restore/public_trajectory_dtw_v2r5.py"
V2R5_CLAIM_RELATIVE = (
    "artifacts/_p2_trajectory_claims/"
    "p2_public_trajectory_dtw_curvature_transfer_20260826_v2r5.claim.json"
)
V2R5_JOURNAL_RELATIVE = (
    "artifacts/_p2_trajectory_attempt_journals/"
    "p2_public_trajectory_dtw_curvature_transfer_20260826_v2r5.ndjson"
)
V2R5_MODULE_PATH = REPO_ROOT / V2R5_MODULE_RELATIVE
V2R5_CLAIM_PATH = REPO_ROOT / V2R5_CLAIM_RELATIVE
V2R5_JOURNAL_PATH = REPO_ROOT / V2R5_JOURNAL_RELATIVE

P2_DATA_DIR_ENV = "P2_DATA_DIR"
OBSERVATIONS_RELATIVE = Path("observations.csv")
EXACT_ANCHOR_PATH = REPO_ROOT / "artifacts/p2_extrapolated_soft_gate_v2/oof.parquet"

CLAIM_PATH = REPO_ROOT / f"artifacts/_p2_trajectory_claims/{EXPERIMENT_ID}.claim.json"
JOURNAL_PATH = REPO_ROOT / (
    f"artifacts/_p2_trajectory_attempt_journals/{EXPERIMENT_ID}.ndjson"
)
TERMINAL_PATH = REPO_ROOT / (
    "artifacts/_p2_trajectory_terminal_receipts/"
    f"{EXPERIMENT_ID}.terminal_failure.json"
)
ERROR_RECEIPT_PATH = REPO_ROOT / (
    f"artifacts/_p2_trajectory_worker_errors/{EXPERIMENT_ID}.error.json"
)
FINAL_DIR = REPO_ROOT / f"artifacts/{EXPERIMENT_ID}"
STAGING_PATH = REPO_ROOT / f"artifacts/_p2_trajectory_staging/{EXPERIMENT_ID}"

FORBIDDEN_PATH_TOKENS = (
    "test_index",
    "sample_submission",
    "submission_candidate",
    "candidate.csv",
)

STATIC_PINS = MappingProxyType(
    {
        DESIGN_RELATIVE: (14_145, "dc8684950bf7466038270a33ab2f2c6425820433aa205790bd40c1d5e842004e"),
        TRIGGER_RELATIVE: (3_557, "3348f8bf10335a1da414fcbb8f04f7fb7e00ec798ff774bba63edbc13af73f87"),
        EXECUTION_RELATIVE: (3_480, "3a2c03c84143c8cf5e395241605f3506efbc00e3d60841a4b05099eb054bd70d"),
        MODULE_RELATIVE: (15_820, "8c507a385ab2846ed27698366cbc96c8454aa0fd6821a27c90287932c06b5145"),
        SLOT_DIAGNOSTIC_RELATIVE: (1_599, "c091f5afb612015f2f8390462aa557117154f0468e8afbc7306699c5c80504d7"),
        IDENTIFIABILITY_RELATIVE: (3_802, "a84f09e87322b3561e8e7109515d576f27e73e3063ae846159876d045b54a864"),
        V2R5_MODULE_RELATIVE: (78_348, "e523d15d6113fdd4861826786be38fef97a68f0ef5732321c196bd67acf1f05f"),
        V2R5_CLAIM_RELATIVE: (561, "00a8f84de695475eaec52ec691ab219448408c6dc90f5e9879ace71cc09c232f"),
        V2R5_JOURNAL_RELATIVE: (3_239, "710a4eb202467aca26b13c24f74aff71378a0c23cea0e1e890c5b07f0e8f2e4f"),
    }
)

SOURCE_PINS = MappingProxyType(
    {
        "observations": (
            49_058_719,
            "cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a",
        ),
        "exact_anchor": (
            2_477_660,
            "dab52579e99a20cc0444bf13bc3a1400191024a10303cb996ba59a89509c9cb4",
        ),
    }
)


class BlockedExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class StaticSnapshot:
    bytes_by_relative: Mapping[str, bytes]
    digest_by_relative: Mapping[str, str]
    seal: Mapping[str, Any]
    seal_bytes: bytes
    seal_digest: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {label}")
    return value


def _read_snapshot(path: Path) -> tuple[bytes, str]:
    with path.open("rb") as handle:
        payload = handle.read()
    return payload, _sha256_bytes(payload)


def _verify_pin(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> tuple[bytes, str]:
    payload, digest = _read_snapshot(path)
    if len(payload) != expected_bytes or digest != expected_sha256:
        raise RuntimeError(f"held-byte pin mismatch: {path.name}")
    return payload, digest


def _safe_repo_path(relative: str) -> Path:
    candidate = (REPO_ROOT / relative).resolve(strict=True)
    root = REPO_ROOT.resolve(strict=True)
    if root != candidate and root not in candidate.parents:
        raise ValueError("static path escaped repository root")
    text = str(candidate).lower().replace("\\", "/")
    if any(token in text for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError("forbidden official or candidate path token")
    return candidate


def _resolve_observations_path() -> Path:
    raw_root = os.environ.get(P2_DATA_DIR_ENV, "")
    if not raw_root:
        raise FileNotFoundError("P2_DATA_DIR is required for read-only P2 readiness")
    literal_root = Path(raw_root)
    if not literal_root.is_absolute():
        raise ValueError("P2_DATA_DIR must be an absolute directory")
    root = literal_root.resolve(strict=True)
    candidate = (root / OBSERVATIONS_RELATIVE).resolve(strict=True)
    text = str(candidate).lower().replace("\\", "/")
    if (
        candidate.parent != root
        or candidate.name != OBSERVATIONS_RELATIVE.name
        or any(token in text for token in FORBIDDEN_PATH_TOKENS)
    ):
        raise ValueError("portable observations resolution violated its firewall")
    return candidate


def _load_module_from_snapshot(
    payload: bytes,
    *,
    digest: str,
    expected_digest: str,
    module_name: str,
    source_path: Path,
) -> ModuleType:
    if _sha256_bytes(payload) != digest or digest != expected_digest:
        raise RuntimeError("held numerical-module snapshot changed")
    code = compile(payload.decode("utf-8"), str(source_path), "exec")
    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _verify_v2r5_lineage(
    claim_bytes: bytes,
    journal_bytes: bytes,
) -> dict[str, Any]:
    claim = _json_object(claim_bytes, label=V2R5_CLAIM_PATH.name)
    events = [
        json.loads(line)
        for line in journal_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(event, dict) for event in events):
        raise ValueError("v2r5 journal event schema changed")
    expected_events = [
        "ATTEMPT_CLAIMED",
        "PARENT_LAUNCHING_SINGLE_WORKER",
        "MATERIALIZATION_RESERVED",
        "MATERIALIZATION_FAILED",
        "ATTEMPT_TERMINAL_FAILED",
    ]
    if [event.get("event") for event in events] != expected_events:
        raise RuntimeError("v2r5 terminal event lineage changed")
    terminal = events[-1]
    accounting = terminal.get("materialization_accounting", {})
    if (
        claim.get("experiment_id")
        != "p2_public_trajectory_dtw_curvature_transfer_20260826_v2r5"
        or claim.get("attempt_id") != "77cfa4f6c3a15ae62c2b2a61"
        or accounting.get("reserved") != 1
        or accounting.get("failed") != 1
        or accounting.get("completed") != 0
        or accounting.get("skipped_gate") != 0
        or accounting.get("physical_fit_calls") != 0
        or terminal.get("status") != "TERMINAL_FAILURE_NO_RERUN"
    ):
        raise RuntimeError("v2r5 failure accounting changed")
    return {
        "status": "PASS_IMMUTABLE_V2R5_FAILURE_LINEAGE",
        "attempt_id": claim["attempt_id"],
        "events": expected_events,
        "reserved": 1,
        "failed": 1,
        "completed": 0,
        "scores": 0,
        "physical_fit_calls": 0,
        "p100_accesses": 0,
    }


def _verify_static_semantics(snapshot: StaticSnapshot) -> dict[str, Any]:
    design = _json_object(
        snapshot.bytes_by_relative[DESIGN_RELATIVE],
        label=DESIGN_RELATIVE,
    )
    trigger = _json_object(
        snapshot.bytes_by_relative[TRIGGER_RELATIVE],
        label=TRIGGER_RELATIVE,
    )
    execution = _json_object(
        snapshot.bytes_by_relative[EXECUTION_RELATIVE],
        label=EXECUTION_RELATIVE,
    )
    slot_diagnostic = _json_object(
        snapshot.bytes_by_relative[SLOT_DIAGNOSTIC_RELATIVE],
        label=SLOT_DIAGNOSTIC_RELATIVE,
    )
    certificate = _json_object(
        snapshot.bytes_by_relative[IDENTIFIABILITY_RELATIVE],
        label=IDENTIFIABILITY_RELATIVE,
    )
    if (
        design.get("status") != "CONDITIONAL_PREREGISTERED_NOT_AUTHORIZED"
        or design.get("experiment_id") != EXPERIMENT_ID
        or design.get("trigger_and_stop_relation", {}).get("automatic_reruns_after_v2r6")
        != 0
    ):
        raise RuntimeError("v2r6 prospective design semantics changed")
    resolution = trigger.get("resolution", {})
    if (
        resolution.get("terminal_status")
        != "NO_GO_INNER_WINDOW_UNIDENTIFIABLE"
        or resolution.get("claim_may_be_created") is not False
        or resolution.get("worker_may_launch") is not False
        or resolution.get("materialization_may_run") is not False
        or resolution.get("score_may_run") is not False
        or resolution.get("authorization_may_become_true") is not False
    ):
        raise RuntimeError("v2r6 trigger resolution no longer fails closed")
    if (
        execution.get("status")
        != "PRECLAIM_BLOCKED_NO_GO_INNER_WINDOW_UNIDENTIFIABLE"
        or execution.get("authorized") is not False
        or execution.get("execution_permitted") is not False
        or any(int(value) != 0 for value in execution["operation_ceiling"].values())
    ):
        raise RuntimeError("v2r6 execution contract no longer blocks all operations")
    attempt = slot_diagnostic.get("diagnostic_attempt", {})
    if (
        attempt.get("ordinal") != 1
        or attempt.get("attempt_ceiling") != 1
        or attempt.get("materializer_status")
        != "RETURNED_WITHOUT_EXCEPTION_PAYLOAD_IMMEDIATELY_DISCARDED"
        or attempt.get("additional_real_slot_diagnostic_authorized") is not False
    ):
        raise RuntimeError("v2r6 one-shot diagnostic accounting changed")
    if (
        certificate.get("all_three_registered_windows_identifiable") is not False
        or certificate.get("frozen_eighteen_cell_selection_complete") is not False
        or certificate["windows"][0]["rows_after_finite_truth_anchor_mask"] != 0
    ):
        raise RuntimeError("v2r6 frozen identifiability blocker changed")
    lineage = _verify_v2r5_lineage(
        snapshot.bytes_by_relative[V2R5_CLAIM_RELATIVE],
        snapshot.bytes_by_relative[V2R5_JOURNAL_RELATIVE],
    )
    return {
        "status": "PASS_STATIC_BLOCKED_CONTRACT",
        "v2r5_lineage": lineage,
        "scientific_contract": {
            "cells": design["unchanged_scientific_contract"]["cells"],
            "inner_windows": design["unchanged_scientific_contract"]["inner_windows"],
            "materialization_slots": design["unchanged_scientific_contract"]["maximum_materializations"],
            "physical_fit_calls": 0,
            "changed": False,
        },
    }


def _verify_seal() -> StaticSnapshot:
    seal_bytes, seal_digest = _read_snapshot(SEAL_PATH)
    seal = _json_object(seal_bytes, label=SEAL_RELATIVE)
    if (
        seal.get("schema_version") != "p2_dtw_v2r6.preexecution_seal.v1"
        or seal.get("experiment_id") != EXPERIMENT_ID
        or seal.get("status")
        != "SEALED_PRECLAIM_BLOCKED_NO_GO_INNER_WINDOW_UNIDENTIFIABLE"
    ):
        raise RuntimeError("v2r6 preexecution seal semantics changed")
    bundle = seal.get("bundle")
    if not isinstance(bundle, dict):
        raise ValueError("v2r6 seal bundle is missing")
    expected_bundle = {
        DESIGN_RELATIVE,
        TRIGGER_RELATIVE,
        EXECUTION_RELATIVE,
        MODULE_RELATIVE,
        RUNNER_RELATIVE,
        TEST_RELATIVE,
        SLOT_DIAGNOSTIC_RELATIVE,
        IDENTIFIABILITY_RELATIVE,
        CLOSURE_RELATIVE,
        V2R5_MODULE_RELATIVE,
        V2R5_CLAIM_RELATIVE,
        V2R5_JOURNAL_RELATIVE,
    }
    if set(bundle) != expected_bundle:
        raise ValueError("v2r6 seal bundle inventory changed")
    bytes_by_relative: dict[str, bytes] = {}
    digest_by_relative: dict[str, str] = {}
    for relative in sorted(expected_bundle):
        pin = bundle[relative]
        payload, digest = _verify_pin(
            _safe_repo_path(relative),
            expected_bytes=int(pin["bytes"]),
            expected_sha256=str(pin["sha256"]),
        )
        bytes_by_relative[relative] = payload
        digest_by_relative[relative] = digest
    for relative, (expected_bytes, expected_digest) in STATIC_PINS.items():
        if (
            len(bytes_by_relative[relative]) != expected_bytes
            or digest_by_relative[relative] != expected_digest
        ):
            raise RuntimeError(f"v2r6 built-in static pin changed: {relative}")
    snapshot = StaticSnapshot(
        bytes_by_relative=MappingProxyType(bytes_by_relative),
        digest_by_relative=MappingProxyType(digest_by_relative),
        seal=MappingProxyType(seal),
        seal_bytes=seal_bytes,
        seal_digest=seal_digest,
    )
    _verify_static_semantics(snapshot)
    return snapshot


def _verify_authorization(snapshot: StaticSnapshot) -> dict[str, Any]:
    payload, digest = _read_snapshot(AUTHORIZATION_PATH)
    authorization = _json_object(payload, label=AUTHORIZATION_RELATIVE)
    if (
        authorization.get("schema_version")
        != "p2_dtw_v2r6.execution_authorization.v1"
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("status")
        != "PERMANENTLY_NOT_AUTHORIZED_PRECLAIM_BLOCKED"
        or authorization.get("authorized") is not False
        or authorization.get("execution_permitted") is not False
        or authorization.get("seal", {}).get("sha256") != snapshot.seal_digest
        or authorization.get("design_sha256")
        != snapshot.digest_by_relative[DESIGN_RELATIVE]
    ):
        raise BlockedExecutionError("v2r6 authorization is not the sealed permanent false state")
    return {
        "status": authorization["status"],
        "authorized": False,
        "execution_permitted": False,
        "bytes": len(payload),
        "sha256": digest,
    }


def _source_snapshots() -> tuple[bytes, bytes, dict[str, Any]]:
    observations_path = _resolve_observations_path()
    observations_bytes, observations_digest = _verify_pin(
        observations_path,
        expected_bytes=SOURCE_PINS["observations"][0],
        expected_sha256=SOURCE_PINS["observations"][1],
    )
    exact_bytes, exact_digest = _verify_pin(
        _safe_repo_path(EXACT_ANCHOR_PATH.relative_to(REPO_ROOT).as_posix()),
        expected_bytes=SOURCE_PINS["exact_anchor"][0],
        expected_sha256=SOURCE_PINS["exact_anchor"][1],
    )
    return observations_bytes, exact_bytes, {
        "observations": {
            "bytes": len(observations_bytes),
            "sha256": observations_digest,
        },
        "exact_anchor": {
            "bytes": len(exact_bytes),
            "sha256": exact_digest,
        },
        "p100": {
            "state": "UNRESOLVED_UNSTATTED_UNOPENED_UNHASHED_UNPARSED",
            "filesystem_accesses": 0,
        },
    }


def _runtime_identifiability(snapshot: StaticSnapshot) -> dict[str, Any]:
    observations_bytes, exact_bytes, source_state = _source_snapshots()
    import pandas as pd

    v2r5_module = _load_module_from_snapshot(
        snapshot.bytes_by_relative[V2R5_MODULE_RELATIVE],
        digest=snapshot.digest_by_relative[V2R5_MODULE_RELATIVE],
        expected_digest=STATIC_PINS[V2R5_MODULE_RELATIVE][1],
        module_name="_sealed_p2_dtw_v2r5_for_v2r6_readiness",
        source_path=V2R5_MODULE_PATH,
    )
    guard_module = _load_module_from_snapshot(
        snapshot.bytes_by_relative[MODULE_RELATIVE],
        digest=snapshot.digest_by_relative[MODULE_RELATIVE],
        expected_digest=STATIC_PINS[MODULE_RELATIVE][1],
        module_name="_sealed_p2_dtw_v2r6_guard",
        source_path=MODULE_PATH,
    )
    observations = pd.read_csv(
        io.BytesIO(observations_bytes),
        dtype={"station": "string", "time": "string"},
    )
    exact = pd.read_parquet(io.BytesIO(exact_bytes))
    materializer = v2r5_module.HistoricalTrajectoryMaterializer(observations, exact)
    certificate = guard_module.build_inner_identifiability_certificate(
        materializer.panel
    )
    stop = guard_module.verify_frozen_certificate(certificate)
    frozen = _json_object(
        snapshot.bytes_by_relative[IDENTIFIABILITY_RELATIVE],
        label=IDENTIFIABILITY_RELATIVE,
    )
    expected_windows = [
        {
            key: row[key]
            for key in (
                "window_id",
                "time_keys",
                "layer_finite_masks",
                "rows_after_finite_truth_anchor_mask",
                "kst_days_after_mask",
                "identifiable_for_scoring",
            )
        }
        for row in frozen["windows"]
    ]
    if certificate["windows"] != expected_windows:
        raise RuntimeError("runtime identifiability certificate differs from frozen diagnosis")
    return {
        "status": stop["status"],
        "blocking_window": stop["blocking_window"],
        "blocking_rows_after_mask": stop["blocking_rows_after_mask"],
        "windows": certificate["windows"],
        "claim_permitted": False,
        "worker_launch_permitted": False,
        "materialization_permitted": False,
        "score_permitted": False,
        "operation_counters": {
            "claims": 0,
            "worker_launches": 0,
            "materializations": 0,
            "scores": 0,
            "physical_fit_calls": 0,
            "p100_accesses": 0,
        },
        "historical_sources": source_state,
    }


def _control_state() -> dict[str, Any]:
    paths = {
        "claim": CLAIM_PATH,
        "journal": JOURNAL_PATH,
        "terminal": TERMINAL_PATH,
        "worker_error_receipt": ERROR_RECEIPT_PATH,
        "final_directory": FINAL_DIR,
        "staging_path": STAGING_PATH,
    }
    exists = {name: path.exists() for name, path in paths.items()}
    return {
        "exists": exists,
        "clean": not any(exists.values()),
    }


def _durable_directory_barrier(directory: Path) -> None:
    """Best-effort Windows-safe directory barrier without fsyncing read handles."""

    marker = directory / ".p2_dtw_v2r6_directory_fsync"
    binary = getattr(os, "O_BINARY", 0)
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | binary, 0o600)
    try:
        os.write(descriptor, b"p2_dtw_v2r6_directory_barrier\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if os.name != "nt":
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _publish_error_envelope_create_only(
    destination: Path,
    envelope: Mapping[str, Any],
    *,
    guard_module: ModuleType,
) -> dict[str, Any]:
    """Publish a bounded sanitized receipt; used only by synthetic fault tests."""

    verification = guard_module.validate_error_envelope(envelope)
    payload = guard_module.canonical_json_bytes(envelope)
    if (
        len(payload) != int(verification["bytes"])
        or _sha256_bytes(payload) != str(verification["sha256"])
    ):
        raise RuntimeError("error envelope changed between validation and publication")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    binary = getattr(os, "O_BINARY", 0)
    descriptor = os.open(
        temp,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | binary,
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temp, destination)
        _durable_directory_barrier(destination.parent)
    finally:
        temp.unlink(missing_ok=True)
    held, digest = _read_snapshot(destination)
    if held != payload or digest != verification["sha256"]:
        raise RuntimeError("published error envelope identity changed")
    return {
        "path_name": destination.name,
        "bytes": len(held),
        "sha256": digest,
        "durable_create_only": True,
    }


def _bind_worker_error_receipt(
    path: Path,
    *,
    expected_sha256: str,
    guard_module: ModuleType,
) -> dict[str, Any]:
    held, digest = _read_snapshot(path)
    if digest != expected_sha256:
        raise RuntimeError("worker error receipt digest differs from parent binding")
    envelope = _json_object(held, label=path.name)
    guard_module.validate_error_envelope(envelope)
    return {
        "bytes": len(held),
        "sha256": digest,
        "error": envelope,
        "held_snapshot_bound": True,
    }


def read_only_preflight() -> dict[str, Any]:
    before = _control_state()
    if not before["clean"]:
        raise RuntimeError("v2r6 blocked namespace is not clean")
    snapshot = _verify_seal()
    static = _verify_static_semantics(snapshot)
    authorization = _verify_authorization(snapshot)
    identifiability = _runtime_identifiability(snapshot)
    after = _control_state()
    if before != after or not after["clean"]:
        raise RuntimeError("read-only preflight changed blocked control state")
    return {
        "schema_version": "p2_dtw_v2r6.read_only_preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "NO_GO_INNER_WINDOW_UNIDENTIFIABLE",
        "read_only": True,
        "preclaim_blocked": True,
        "authorized": False,
        "execution_permitted": False,
        "design_sha256": snapshot.digest_by_relative[DESIGN_RELATIVE],
        "seal_sha256": snapshot.seal_digest,
        "static_verification": static,
        "authorization": authorization,
        "identifiability": identifiability,
        "control_state": after,
        "operation_counters": {
            "claims": 0,
            "worker_launches": 0,
            "materializations": 0,
            "scores": 0,
            "physical_fit_calls": 0,
            "p100_accesses": 0,
            "official_test_reads": 0,
            "sample_submission_reads": 0,
            "submission_candidate_reads": 0,
            "candidate_files": 0,
            "uploads": 0,
        },
        "next_action": "ROUTE_ONLY_TO_SEPARATELY_PREREGISTERED_CYCLE2",
    }


def execute_blocked() -> None:
    preflight = read_only_preflight()
    raise BlockedExecutionError(
        f"{preflight['status']}: v2r6 cannot create a claim or launch a worker"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-local", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.execute_local:
        execute_blocked()
    print(
        json.dumps(
            read_only_preflight(),
            sort_keys=True,
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
