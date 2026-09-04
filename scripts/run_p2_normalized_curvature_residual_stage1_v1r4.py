"""Crash-safe, literal-path-sealed P2 NCR_LGBM Stage 1 v1r4 runner.

The default mode is read-only strict preflight.  Before ``--execute-stage1``
can create any permanent claim, control, or staging path, the parent verifies
the approved literal observations path and all sealed implementation/runtime
references.  It then launches this same sealed file as a numerical worker
under a 1,800-second parent-enforced wall clock.  There is no retry or resume
path.  Official test, sample-submission, submission-candidate, CSV, and upload
code paths do not exist.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import secrets
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
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
    / "configs/experiments/p2_normalized_curvature_residual_lgbm_stage1_v1r4.json"
)
EXPECTED_CONFIG_SHA256 = "a7120e557709cc083682dbca3c27e4874c1b9bfc4ff9e28a59a7f7204adaea29"
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
WORKER_TOKEN_ENV = "P2_NCR_V1R4_ATTEMPT_TOKEN"
WORKER_DEADLINE_ENV = "P2_NCR_V1R4_DEADLINE_EPOCH"
WORKER_PARENT_PID_ENV = "P2_NCR_V1R4_PARENT_PID"
PRELAUNCH_JOURNAL_SCHEMA = "p2_ncr_prelaunch_journal.v1r4"


class HardWallTimeout(RuntimeError):
    """The sealed numerical worker exceeded its remaining hard wall clock."""


class ExistingAttemptError(RuntimeError):
    """A stable claim/journal/final/staging state forbids another attempt."""


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise RuntimeError("runner config-hash normalization failed closed")
    return _sha256_bytes(normalized)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _assert_safe_input_path(path: Path) -> None:
    lowered_parts = tuple(part.lower() for part in path.parts)
    if path.name.lower() in FORBIDDEN_FILE_NAMES:
        raise RuntimeError(f"forbidden P2 input path: {path}")
    if any(token in part for part in lowered_parts for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"submission/candidate-like path is forbidden: {path}")


def _repo_file(relative_text: str) -> Path:
    relative = Path(relative_text)
    _assert_safe_input_path(relative)
    resolved = (PROJECT_ROOT / relative).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT) or not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _repo_control_path(relative_text: str, required_parent: str) -> Path:
    relative = Path(relative_text)
    resolved = (PROJECT_ROOT / relative).resolve()
    required_root = (PROJECT_ROOT / "artifacts" / required_parent).resolve()
    if not resolved.is_relative_to(required_root) or resolved == required_root:
        raise RuntimeError(f"control path escaped {required_parent}: {relative_text}")
    return resolved


def _native_distribution_fingerprint(
    distribution_name: str, native_pins: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    distribution = importlib.metadata.distribution(distribution_name)
    observed: list[dict[str, Any]] = []
    for pin in native_pins:
        relative = Path(str(pin["distribution_relative_path"]))
        path = Path(distribution.locate_file(relative)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        digest = _sha256(path)
        if size != int(pin["bytes"]) or digest != str(pin["sha256"]):
            raise RuntimeError(f"native runtime fingerprint drift: {relative.as_posix()}")
        observed.append(
            {
                "distribution_relative_path": relative.as_posix(),
                "bytes": size,
                "sha256": digest,
            }
        )
    return observed


def _validate_runtime_pins(config: Mapping[str, Any]) -> dict[str, Any]:
    pins = config["runtime_pins"]
    observed_packages: dict[str, str] = {"python": platform.python_version()}
    if observed_packages["python"] != str(pins["python"]):
        raise RuntimeError("Python runtime version drift")
    for distribution, expected in pins["packages"].items():
        version = importlib.metadata.version(str(distribution))
        observed_packages[str(distribution)] = version
        if version != str(expected):
            raise RuntimeError(f"runtime package version drift: {distribution}")
    native = _native_distribution_fingerprint(
        "lightgbm", list(pins["lightgbm_native_files"])
    )
    return {"packages": observed_packages, "lightgbm_native_files": native}


def _verify_pinned_group(
    pins: Mapping[str, Mapping[str, Any]], *, runner_name: str | None = None
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    runner_path = Path(__file__).resolve()
    for name, pin in pins.items():
        path = _repo_file(str(pin["path"]))
        mode = str(pin.get("hash_mode", "sha256_raw_v1"))
        if name == runner_name:
            if path != runner_path or mode != RUNNER_HASH_MODE:
                raise RuntimeError("v1r4 runner pin contract drift")
            pinned_digest = _normalized_runner_sha256(path)
        else:
            if mode != "sha256_raw_v1":
                raise RuntimeError(f"unexpected hash mode for {name}")
            pinned_digest = _sha256(path)
        if path.stat().st_size != int(pin["bytes"]):
            raise RuntimeError(f"pinned file size drift: {name}")
        if pinned_digest != str(pin["sha256"]):
            raise RuntimeError(f"pinned file hash drift: {name}")
        observed[name] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "hash_mode": mode,
            "pinned_sha256": pinned_digest,
            "raw_sha256": _sha256(path),
        }
    return observed


def _verify_static_bundle(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed before importing NumPy, pandas, LightGBM, or p2_restore."""

    config_path = config_path.resolve()
    if not config_path.is_relative_to(PROJECT_ROOT) or not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_sha256 = _sha256(config_path)
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("NCR_LGBM v1r4 preregistration hash drift")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("v1r4 preregistration must be a JSON object")
    if config.get("schema_version") != (
        "p2_normalized_curvature_residual_lgbm_stage1.prereg.v1r4"
    ):
        raise ValueError("unexpected v1r4 preregistration schema")
    if config.get("experiment_id") != (
        "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r4"
    ):
        raise ValueError("unexpected v1r4 experiment id")
    if config.get("status") != "PREREGISTERED_NOT_EXECUTED_SUPERSEDES_V1_V1R2_V1R3":
        raise ValueError("v1r4 is not in its sealed preregistered state")

    prohibition = config.get("superseded_execution_forbidden", {})
    if prohibition.get("enforced") is not True:
        raise ValueError("superseded-runner execution prohibition is not enforced")
    if prohibition.get("experiment_ids") != [
        "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1",
        "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r2",
        "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r3",
    ]:
        raise ValueError("superseded experiment prohibition lineage drift")

    data_contract = config["data_contract"]
    if data_contract.get("source_environment_variable") != "P2_DATA_DIR":
        raise ValueError("P2_DATA_DIR execution-environment contract drift")
    if data_contract.get("approved_literal_data_directory") != (
        r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore"
    ):
        raise ValueError("approved literal P2 directory drift")
    if data_contract.get("approved_literal_observations_path") != (
        r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv"
    ):
        raise ValueError("approved literal observations path drift")
    observations_pin = data_contract.get("observations_csv", {})
    if observations_pin != {
        "bytes": 49058719,
        "sha256": "cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a",
    }:
        raise ValueError("approved literal observations byte pin drift")

    policy = config["selection_policy"]
    if bool(policy["official_score_used_for_gate_or_tuning"]):
        raise ValueError("official-score calibration is forbidden")
    if bool(policy["result_based_retuning"]):
        raise ValueError("result-based tuning is forbidden")
    if int(policy["candidate_grid_size"]) != 1:
        raise ValueError("candidate grid changed")
    if int(policy["stage1_model_fit_count"]) != 3:
        raise ValueError("fit budget changed")
    if [int(seed) for seed in config["model"]["seeds"]] != [
        20260823,
        20260824,
        20260825,
    ]:
        raise ValueError("seed contract changed")
    execution = config["execution"]
    if int(execution["maximum_physical_fit_calls_lifetime"]) != 3:
        raise ValueError("physical fit lifetime ceiling changed")
    if int(execution["worker_restart_count"]) != 0:
        raise ValueError("worker restart is forbidden")
    if float(execution["hard_wall_seconds"]) != 1800.0:
        raise ValueError("hard wall changed")
    command = str(execution.get("command", ""))
    if "run_p2_normalized_curvature_residual_stage1_v1r4.py" not in command:
        raise ValueError("execution command is not the v1r4 sealed runner")
    if "run_p2_normalized_curvature_residual_stage1.py" in command or (
        "run_p2_normalized_curvature_residual_stage1_v1r2.py" in command
    ) or (
        "run_p2_normalized_curvature_residual_stage1_v1r3.py" in command
    ):
        raise ValueError("superseded v1/v1r2/v1r3 runner execution is forbidden")

    journal_contract = config.get("worker_prelaunch_journal_contract", {})
    if journal_contract != {
        "schema": PRELAUNCH_JOURNAL_SCHEMA,
        "exact_event_order": [
            "ATTEMPT_CLAIMED",
            "PARENT_LAUNCHING_SINGLE_WORKER",
        ],
        "exact_event_count": 2,
        "worker_restart_count": 0,
        "missing_extra_or_reordered_policy": "FAIL_CLOSED_BEFORE_WORKER_STARTED",
    }:
        raise ValueError("v1r4 worker prelaunch journal contract drift")
    allowed_features = [str(value) for value in config["feature_contract"]["allowed_feature_columns"]]
    if len(allowed_features) != 55 or len(set(allowed_features)) != 55:
        raise ValueError("exact NCR feature allow-list must contain 55 unique columns")
    if not {
        "log1p_profile_scale",
        "log1p_psal_scale",
        "log1p_depth_scale",
    }.issubset(allowed_features):
        raise ValueError("scale-magnitude feature allow-list is incomplete")

    output = config["output"]
    if not bool(output["aggregate_only"]):
        raise ValueError("aggregate-only output contract changed")
    if bool(output["row_predictions_written"]) or bool(output["csv_files_written"]):
        raise ValueError("row-level output is forbidden")
    if sorted(output["allowed_final_files"]) != ["manifest.json", "result.json"]:
        raise ValueError("final output allow-list changed")
    if int(output["submission_files_generated"]) or int(output["uploads"]):
        raise ValueError("external action contract changed")

    implementations = _verify_pinned_group(
        config["implementation_pins"], runner_name="runner"
    )
    lineage = _verify_pinned_group(config["superseded_lineage_pins"])
    references = _verify_pinned_group(config["immutable_references"])
    runtime = _validate_runtime_pins(config)
    bundle: dict[str, Any] = {
        "config": {
            "path": config_path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": config_path.stat().st_size,
            "sha256": config_sha256,
        },
        "implementation_pins": implementations,
        "superseded_lineage_pins": lineage,
        "immutable_references": references,
        "runtime_pins": runtime,
    }
    bundle["bundle_sha256"] = _canonical_sha256(bundle)
    return config, bundle


def _assert_static_bundle_unchanged(
    config_path: Path,
    expected_config: Mapping[str, Any],
    expected_bundle: Mapping[str, Any],
) -> None:
    observed_config, observed_bundle = _verify_static_bundle(config_path)
    if observed_config != expected_config or observed_bundle != expected_bundle:
        raise RuntimeError("v1r4 static bundle changed after preflight")


def _resolved_same_path(left: Path, right: Path) -> bool:
    """Compare already approved local paths with Windows case normalization."""

    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
        os.path.normpath(str(right))
    )


def _execution_environment_readiness(
    config: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    variable = str(config["data_contract"]["source_environment_variable"])
    approved_parent = Path(
        str(config["data_contract"]["approved_literal_data_directory"])
    ).resolve(strict=True)
    raw_value = environment.get(variable)
    if raw_value is None or not str(raw_value).strip():
        return {
            "variable": variable,
            "present": False,
            "matches_approved_literal_parent": False,
            "value_sha256": None,
        }
    supplied = Path(str(raw_value)).resolve(strict=False)
    return {
        "variable": variable,
        "present": True,
        "matches_approved_literal_parent": _resolved_same_path(supplied, approved_parent),
        "value_sha256": _sha256_bytes(str(raw_value).encode("utf-8")),
    }


def _literal_source_readiness(
    config: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    require_environment: bool,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify literal observations bytes before any permanent control mutation.

    The returned receipt exists only in memory until an exclusive claim succeeds.
    Execution-mode environment mismatch is deliberately checked before opening the
    49 MB source so missing/wrong-environment tests can prove zero claim attempts.
    """

    data_contract = config["data_contract"]
    approved_parent_text = str(data_contract["approved_literal_data_directory"])
    approved_path_text = str(data_contract["approved_literal_observations_path"])
    environment = _execution_environment_readiness(config, environ=environ)
    if require_environment and not environment["present"]:
        raise RuntimeError("P2_DATA_DIR is required before any v1r4 claim")
    if require_environment and not environment["matches_approved_literal_parent"]:
        raise RuntimeError("P2_DATA_DIR differs from the approved literal parent")

    approved_parent = Path(approved_parent_text).resolve(strict=True)
    approved_path = Path(approved_path_text).resolve(strict=True)
    _assert_safe_input_path(approved_path)
    if approved_path.name != "observations.csv":
        raise RuntimeError("approved literal observations filename drift")
    if not _resolved_same_path(approved_path.parent, approved_parent):
        raise RuntimeError("approved literal observations parent drift")
    if not approved_parent.is_dir() or not approved_path.is_file():
        raise FileNotFoundError(approved_path)
    observations_pin = data_contract["observations_csv"]
    with _held_verified_bytes(
        approved_path,
        expected_bytes=int(observations_pin["bytes"]),
        expected_sha256=str(observations_pin["sha256"]),
    ) as (_held_handle, captured, observed):
        _verify_captured_source(
            captured,
            expected_bytes=int(observations_pin["bytes"]),
            expected_sha256=str(observations_pin["sha256"]),
            label="approved literal observations.csv readiness",
        )
        receipt: dict[str, Any] = {
            "schema_version": "p2_ncr_literal_data_readiness.v1r4",
            "experiment_id": config["experiment_id"],
            "contract_sha256": str(bundle["config"]["sha256"]),
            "bundle_sha256": str(bundle["bundle_sha256"]),
            "verified_at_kst": _now_kst(),
            "parent_pid": os.getpid(),
            "approved_literal_data_directory": approved_parent_text,
            "approved_literal_observations_path": approved_path_text,
            "resolved_parent_matches": True,
            "filename": approved_path.name,
            "bytes": int(observed["bytes"]),
            "sha256": str(observed["sha256"]),
            "hash_read_binding": "held-handle bytes captured and SHA-256 verified",
            "execution_environment": environment,
            "permanent_claim_created": False,
        }
    receipt["readiness_sha256"] = _canonical_sha256(receipt)
    return receipt


def _prepare_readiness_then_acquire(
    config: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    claim_fn: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]
    ]
    | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The sole parent mutation boundary: readiness must return before claim_fn."""

    readiness = _literal_source_readiness(
        config,
        bundle,
        require_environment=True,
        environ=environ,
    )
    acquire = _acquire_attempt if claim_fn is None else claim_fn
    attempt = acquire(config, bundle, readiness)
    return readiness, attempt


def _control_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    controls = config["controls"]
    final = _repo_control_path(str(config["output"]["directory"]), "")
    claim = _repo_control_path(str(controls["claim_path"]), "_ncr_stage1_claims")
    journal = _repo_control_path(
        str(controls["journal_path"]), "_ncr_stage1_attempt_journals"
    )
    artifact_root = (PROJECT_ROOT / "artifacts").resolve()
    if not final.is_relative_to(artifact_root) or final == artifact_root:
        raise RuntimeError("final artifact path escaped artifacts")
    if len({final, claim, journal}) != 3:
        raise RuntimeError("final, claim, and journal paths must be distinct")
    return {"final": final, "claim": claim, "journal": journal}


def _staging_paths(final: Path) -> list[Path]:
    if not final.parent.exists():
        return []
    return sorted(final.parent.glob(f".{final.name}.staging-*"))


def _read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise RuntimeError(f"attempt journal has a torn line at {line_number}")
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise RuntimeError("attempt journal event is not an object")
            events.append(parsed)
    return events


def _inspect_paths(paths: Mapping[str, Path]) -> dict[str, Any]:
    staging = _staging_paths(paths["final"])
    final_exists = paths["final"].exists()
    claim_exists = paths["claim"].exists()
    journal_exists = paths["journal"].exists()
    journal_error: str | None = None
    try:
        journal_events = _read_journal(paths["journal"]) if journal_exists else []
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        journal_events = []
        journal_error = type(error).__name__
    terminal_events = [
        event
        for event in journal_events
        if str(event.get("event", "")).startswith("ATTEMPT_TERMINAL_")
    ]
    clean = not (final_exists or claim_exists or journal_exists or staging)
    if journal_error is not None:
        status = "CORRUPT_OR_TORN_JOURNAL_NO_RERUN"
    elif clean:
        status = "CLEAN_ELIGIBLE_FOR_SINGLE_ATTEMPT"
    elif final_exists and terminal_events:
        status = "TERMINAL_FINAL_EXISTS_NO_RERUN"
    elif final_exists:
        status = "FINAL_EXISTS_JOURNAL_NONTERMINAL_NO_RERUN"
    else:
        status = "INCOMPLETE_OR_CONCURRENT_ATTEMPT_NO_RERUN"
    return {
        "status": status,
        "eligible": clean,
        "final_exists": final_exists,
        "claim_exists": claim_exists,
        "journal_exists": journal_exists,
        "journal_event_count": len(journal_events),
        "journal_parse_error": journal_error,
        "terminal_event_count": len(terminal_events),
        "orphan_or_active_staging_count": len(staging),
    }


def _inspect_control_state(config: Mapping[str, Any]) -> dict[str, Any]:
    return _inspect_paths(_control_paths(config))


def _fsync_directory_best_effort(path: Path) -> bool:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(descriptor)
    except OSError:
        return False
    finally:
        os.close(descriptor)
    return True


def _exclusive_create_bytes(path: Path, payload: bytes) -> bool:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(str(path), flags, 0o600)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise RuntimeError("exclusive durable write was partial")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _fsync_directory_best_effort(path.parent)


def _exclusive_create_json(path: Path, value: Mapping[str, Any]) -> bool:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return _exclusive_create_bytes(path, payload)


def _journal_line(event: Mapping[str, Any]) -> bytes:
    payload = (
        json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(payload) > 8192:
        raise ValueError("attempt journal event exceeds the atomic append ceiling")
    return payload


def _append_journal_event(path: Path, event: Mapping[str, Any]) -> None:
    payload = _journal_line(event)
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(str(path), flags)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise RuntimeError("durable attempt journal append was partial")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _journal_event(
    event: str,
    *,
    contract_sha256: str,
    bundle_sha256: str,
    attempt_token_sha256: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "event": event,
        "at_kst": _now_kst(),
        "contract_sha256": contract_sha256,
        "bundle_sha256": bundle_sha256,
        "attempt_token_sha256": attempt_token_sha256,
        **details,
    }


def _fit_slot_states(events: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    states = {1: "UNRESERVED", 2: "UNRESERVED", 3: "UNRESERVED"}
    for event in events:
        name = str(event.get("event", ""))
        if name not in {
            "FIT_SLOT_RESERVED",
            "FIT_SLOT_COMPLETED",
            "FIT_SLOT_FAILED",
            "FIT_SLOT_ABORTED_BEFORE_CALL",
        }:
            continue
        slot = int(event["slot"])
        if slot not in states:
            raise RuntimeError("journal contains an invalid physical fit slot")
        if name == "FIT_SLOT_RESERVED":
            if states[slot] != "UNRESERVED":
                raise RuntimeError("physical fit slot was reserved more than once")
            states[slot] = "RESERVED"
        elif name == "FIT_SLOT_COMPLETED":
            if states[slot] != "RESERVED":
                raise RuntimeError("physical fit slot completed without reservation")
            states[slot] = "COMPLETED"
        elif name == "FIT_SLOT_FAILED":
            if states[slot] != "RESERVED":
                raise RuntimeError("physical fit slot failed without reservation")
            states[slot] = "FAILED"
        else:
            if states[slot] != "RESERVED":
                raise RuntimeError("physical fit slot aborted without reservation")
            states[slot] = "ABORTED_BEFORE_CALL"
    return states


def _reserve_fit_slot(
    journal_path: Path,
    *,
    slot: int,
    seed: int,
    contract_sha256: str,
    bundle_sha256: str,
    attempt_token_sha256: str,
) -> None:
    states = _fit_slot_states(_read_journal(journal_path))
    if states.get(slot) != "UNRESERVED":
        raise RuntimeError(f"physical fit slot {slot} is already consumed")
    if slot > 1 and states[slot - 1] != "COMPLETED":
        raise RuntimeError("physical fit slots must be completed in order")
    _append_journal_event(
        journal_path,
        _journal_event(
            "FIT_SLOT_RESERVED",
            contract_sha256=contract_sha256,
            bundle_sha256=bundle_sha256,
            attempt_token_sha256=attempt_token_sha256,
            slot=slot,
            seed=seed,
            status="DURABLE_RESERVATION_BEFORE_MODEL_FIT_CALL",
        ),
    )


def _complete_fit_slot(
    journal_path: Path,
    *,
    slot: int,
    seed: int,
    elapsed_seconds: float,
    contract_sha256: str,
    bundle_sha256: str,
    attempt_token_sha256: str,
) -> None:
    if _fit_slot_states(_read_journal(journal_path)).get(slot) != "RESERVED":
        raise RuntimeError("cannot complete an unreserved physical fit slot")
    _append_journal_event(
        journal_path,
        _journal_event(
            "FIT_SLOT_COMPLETED",
            contract_sha256=contract_sha256,
            bundle_sha256=bundle_sha256,
            attempt_token_sha256=attempt_token_sha256,
            slot=slot,
            seed=seed,
            elapsed_seconds=float(elapsed_seconds),
        ),
    )


def _acquire_attempt(
    config: Mapping[str, Any],
    bundle: Mapping[str, Any],
    readiness_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if readiness_receipt.get("experiment_id") != config["experiment_id"]:
        raise RuntimeError("literal readiness experiment mismatch before claim")
    if readiness_receipt.get("contract_sha256") != bundle["config"]["sha256"]:
        raise RuntimeError("literal readiness contract mismatch before claim")
    if readiness_receipt.get("bundle_sha256") != bundle["bundle_sha256"]:
        raise RuntimeError("literal readiness bundle mismatch before claim")
    expected_readiness_sha256 = _canonical_sha256(
        {key: value for key, value in readiness_receipt.items() if key != "readiness_sha256"}
    )
    if readiness_receipt.get("readiness_sha256") != expected_readiness_sha256:
        raise RuntimeError("literal readiness receipt integrity mismatch before claim")
    environment = readiness_receipt.get("execution_environment", {})
    if environment.get("present") is not True or (
        environment.get("matches_approved_literal_parent") is not True
    ):
        raise RuntimeError("literal readiness execution environment is not approved")
    state = _inspect_control_state(config)
    if not state["eligible"]:
        raise ExistingAttemptError(state["status"])
    paths = _control_paths(config)
    paths["claim"].parent.mkdir(parents=True, exist_ok=True)
    paths["journal"].parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    token_sha256 = _sha256_bytes(token.encode("ascii"))
    started_epoch = time.time()
    deadline_epoch = started_epoch + float(config["execution"]["hard_wall_seconds"])
    contract_sha256 = str(bundle["config"]["sha256"])
    bundle_sha256 = str(bundle["bundle_sha256"])
    claim = {
        "schema_version": "p2_ncr_stage1_execution_claim.v1r4",
        "experiment_id": config["experiment_id"],
        "contract_sha256": contract_sha256,
        "bundle_sha256": bundle_sha256,
        "attempt_token_sha256": token_sha256,
        "parent_pid": os.getpid(),
        "started_at_kst": _now_kst(),
        "started_epoch": started_epoch,
        "deadline_epoch": deadline_epoch,
        "hard_wall_seconds": float(config["execution"]["hard_wall_seconds"]),
        "literal_readiness_receipt": dict(readiness_receipt),
        "literal_readiness_sha256": str(readiness_receipt["readiness_sha256"]),
        "permanent_no_rerun_claim": True,
    }
    claim_directory_fsync = _exclusive_create_json(paths["claim"], claim)
    initial_event = _journal_event(
        "ATTEMPT_CLAIMED",
        contract_sha256=contract_sha256,
        bundle_sha256=bundle_sha256,
        attempt_token_sha256=token_sha256,
        journal_schema=PRELAUNCH_JOURNAL_SCHEMA,
        parent_pid=os.getpid(),
        started_epoch=started_epoch,
        deadline_epoch=deadline_epoch,
        literal_readiness_sha256=str(readiness_receipt["readiness_sha256"]),
        claim_directory_fsync_supported=claim_directory_fsync,
        physical_fit_slots=[
            {"slot": slot, "seed": int(seed), "status": "UNRESERVED"}
            for slot, seed in enumerate(config["model"]["seeds"], start=1)
        ],
    )
    _exclusive_create_bytes(paths["journal"], _journal_line(initial_event))
    return {
        "token": token,
        "token_sha256": token_sha256,
        "contract_sha256": contract_sha256,
        "bundle_sha256": bundle_sha256,
        "started_epoch": started_epoch,
        "deadline_epoch": deadline_epoch,
        "readiness_receipt": dict(readiness_receipt),
        "paths": paths,
    }


@contextlib.contextmanager
def _held_verified_bytes(
    path: Path, *, expected_bytes: int, expected_sha256: str
) -> Iterator[tuple[Any, bytes, dict[str, Any]]]:
    """Yield bytes hashed from the same held file handle later used via BytesIO."""

    handle = path.open("rb", buffering=0)
    try:
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while block := handle.read(1 << 20):
            digest.update(block)
            chunks.append(block)
            total += len(block)
        observed_sha256 = digest.hexdigest()
        if total != expected_bytes or observed_sha256 != expected_sha256:
            raise RuntimeError(f"hash-bound source drift: {path.name}")
        payload = b"".join(chunks)
        yield handle, payload, {"bytes": total, "sha256": observed_sha256}
    finally:
        handle.close()


def _verify_captured_source(
    payload: bytes, *, expected_bytes: int, expected_sha256: str, label: str
) -> None:
    if len(payload) != expected_bytes or _sha256_bytes(payload) != expected_sha256:
        raise RuntimeError(f"captured source digest changed before fit: {label}")


def _validate_prelaunch_journal_events(
    events: Sequence[Mapping[str, Any]],
    *,
    contract_sha256: str,
    bundle_sha256: str,
    attempt_token_sha256: str,
    parent_pid: int,
    deadline_epoch: float,
    literal_readiness_sha256: str,
    seeds: Sequence[int],
    hard_wall_seconds: float,
) -> None:
    """Accept exactly the two durable events written before worker startup."""

    if len(events) != 2:
        raise RuntimeError("internal worker requires exactly two prelaunch journal events")
    claimed, launching = events
    if [claimed.get("event"), launching.get("event")] != [
        "ATTEMPT_CLAIMED",
        "PARENT_LAUNCHING_SINGLE_WORKER",
    ]:
        raise RuntimeError("internal worker prelaunch journal order drift")

    claimed_keys = {
        "event",
        "at_kst",
        "journal_schema",
        "contract_sha256",
        "bundle_sha256",
        "attempt_token_sha256",
        "parent_pid",
        "started_epoch",
        "deadline_epoch",
        "literal_readiness_sha256",
        "claim_directory_fsync_supported",
        "physical_fit_slots",
    }
    launching_keys = {
        "event",
        "at_kst",
        "journal_schema",
        "contract_sha256",
        "bundle_sha256",
        "attempt_token_sha256",
        "parent_pid",
        "worker_restart_count",
        "hard_wall_seconds",
        "literal_readiness_sha256",
    }
    if set(claimed) != claimed_keys or set(launching) != launching_keys:
        raise RuntimeError("internal worker prelaunch journal event schema drift")

    common = {
        "journal_schema": PRELAUNCH_JOURNAL_SCHEMA,
        "contract_sha256": contract_sha256,
        "bundle_sha256": bundle_sha256,
        "attempt_token_sha256": attempt_token_sha256,
        "parent_pid": int(parent_pid),
        "literal_readiness_sha256": literal_readiness_sha256,
    }
    for index, event in enumerate((claimed, launching), start=1):
        if not isinstance(event.get("at_kst"), str) or not event["at_kst"]:
            raise RuntimeError(f"internal worker prelaunch event {index} timestamp drift")
        for name, expected in common.items():
            if event.get(name) != expected:
                raise RuntimeError(
                    f"internal worker prelaunch event {index} mismatch: {name}"
                )

    expected_slots = [
        {"slot": slot, "seed": int(seed), "status": "UNRESERVED"}
        for slot, seed in enumerate(seeds, start=1)
    ]
    if claimed.get("physical_fit_slots") != expected_slots:
        raise RuntimeError("internal worker prelaunch physical fit slot schema drift")
    if not isinstance(claimed.get("claim_directory_fsync_supported"), bool):
        raise RuntimeError("internal worker prelaunch claim durability field drift")
    if abs(float(claimed.get("deadline_epoch")) - float(deadline_epoch)) > 1e-6:
        raise RuntimeError("internal worker prelaunch deadline drift")
    started_epoch = float(claimed.get("started_epoch"))
    if not (started_epoch < float(deadline_epoch)):
        raise RuntimeError("internal worker prelaunch start/deadline order drift")
    if launching.get("worker_restart_count") != 0:
        raise RuntimeError("internal worker prelaunch restart count drift")
    if float(launching.get("hard_wall_seconds")) != float(hard_wall_seconds):
        raise RuntimeError("internal worker prelaunch hard wall drift")


def _validate_worker_claim(
    config: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    token = os.environ.get(WORKER_TOKEN_ENV)
    deadline_text = os.environ.get(WORKER_DEADLINE_ENV)
    parent_pid_text = os.environ.get(WORKER_PARENT_PID_ENV)
    if not token or not deadline_text or not parent_pid_text:
        raise RuntimeError("internal worker authorization environment is incomplete")
    token_sha256 = _sha256_bytes(token.encode("ascii"))
    paths = _control_paths(config)
    if paths["final"].exists() or not paths["claim"].is_file() or not paths["journal"].is_file():
        raise RuntimeError("internal worker control state is invalid")
    claim = json.loads(paths["claim"].read_text(encoding="utf-8"))
    if claim.get("schema_version") != "p2_ncr_stage1_execution_claim.v1r4":
        raise RuntimeError("internal worker claim schema mismatch")
    required = {
        "contract_sha256": str(bundle["config"]["sha256"]),
        "bundle_sha256": str(bundle["bundle_sha256"]),
        "attempt_token_sha256": token_sha256,
        "parent_pid": int(parent_pid_text),
    }
    for name, expected in required.items():
        if claim.get(name) != expected:
            raise RuntimeError(f"internal worker claim mismatch: {name}")
    readiness = claim.get("literal_readiness_receipt")
    if not isinstance(readiness, dict):
        raise RuntimeError("internal worker claim lacks literal readiness receipt")
    readiness_sha256 = _canonical_sha256(
        {key: value for key, value in readiness.items() if key != "readiness_sha256"}
    )
    if readiness.get("readiness_sha256") != readiness_sha256:
        raise RuntimeError("internal worker literal readiness receipt integrity mismatch")
    if claim.get("literal_readiness_sha256") != readiness_sha256:
        raise RuntimeError("internal worker claim literal readiness digest mismatch")
    deadline_epoch = float(deadline_text)
    if abs(deadline_epoch - float(claim["deadline_epoch"])) > 1e-6:
        raise RuntimeError("internal worker deadline mismatch")
    events = _read_journal(paths["journal"])
    _validate_prelaunch_journal_events(
        events,
        contract_sha256=str(bundle["config"]["sha256"]),
        bundle_sha256=str(bundle["bundle_sha256"]),
        attempt_token_sha256=token_sha256,
        parent_pid=int(parent_pid_text),
        deadline_epoch=deadline_epoch,
        literal_readiness_sha256=readiness_sha256,
        seeds=[int(seed) for seed in config["model"]["seeds"]],
        hard_wall_seconds=float(config["execution"]["hard_wall_seconds"]),
    )
    return {
        "token_sha256": token_sha256,
        "deadline_epoch": deadline_epoch,
        "parent_pid": int(parent_pid_text),
        "paths": paths,
        "contract_sha256": str(bundle["config"]["sha256"]),
        "bundle_sha256": str(bundle["bundle_sha256"]),
        "literal_readiness_receipt": readiness,
        "literal_readiness_sha256": readiness_sha256,
    }


def _runtime_thread_fingerprint() -> list[dict[str, Any]]:
    from threadpoolctl import threadpool_info

    allowed = {
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "num_threads",
        "threading_layer",
        "architecture",
    }
    return [
        {name: value for name, value in item.items() if name in allowed}
        for item in threadpool_info()
    ]


def _loaded_lightgbm_library_fingerprint(
    lightgbm_module: Any, config: Mapping[str, Any]
) -> dict[str, Any]:
    loaded = Path(str(lightgbm_module.basic._LIB._name)).resolve()
    if not loaded.is_file():
        raise FileNotFoundError("loaded LightGBM native library is unavailable")
    runtime_pin = next(
        (
            pin
            for pin in config["runtime_pins"]["lightgbm_native_files"]
            if str(pin["distribution_relative_path"]).lower().endswith(".dll")
        ),
        None,
    )
    if runtime_pin is None:
        raise RuntimeError("LightGBM runtime DLL pin is missing")
    size = loaded.stat().st_size
    digest = _sha256(loaded)
    if size != int(runtime_pin["bytes"]) or digest != str(runtime_pin["sha256"]):
        raise RuntimeError("loaded LightGBM native library differs from the sealed DLL")
    return {
        "basename": loaded.name,
        "bytes": size,
        "sha256": digest,
        "distribution_relative_pin": str(runtime_pin["distribution_relative_path"]),
    }


def _assert_exact_metric_pin(
    incumbent_metrics: Mapping[str, Any], reference: Mapping[str, Any]
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
            raise RuntimeError(f"exact same-season layer {layer} RMSE drift")


def _run_internal_worker(
    config_path: Path,
    config: dict[str, Any],
    bundle: dict[str, Any],
    verified_observations_path: str | None,
) -> dict[str, Any]:
    authorization = _validate_worker_claim(config, bundle)
    journal_path = authorization["paths"]["journal"]
    event_base = {
        "contract_sha256": authorization["contract_sha256"],
        "bundle_sha256": authorization["bundle_sha256"],
        "attempt_token_sha256": authorization["token_sha256"],
    }
    _append_journal_event(
        journal_path,
        _journal_event(
            "WORKER_STARTED",
            **event_base,
            worker_pid=os.getpid(),
            parent_pid=authorization["parent_pid"],
        ),
    )

    if not verified_observations_path:
        raise RuntimeError("internal worker requires the sealed observations path argument")
    approved_literal = str(config["data_contract"]["approved_literal_observations_path"])
    if verified_observations_path != approved_literal:
        raise RuntimeError("internal worker observations argument differs from sealed literal")
    readiness = authorization["literal_readiness_receipt"]
    if readiness.get("approved_literal_observations_path") != approved_literal:
        raise RuntimeError("internal worker claim path differs from sealed literal")
    observations_path = Path(verified_observations_path).resolve(strict=True)
    approved_parent = Path(
        str(config["data_contract"]["approved_literal_data_directory"])
    ).resolve(strict=True)
    _assert_safe_input_path(observations_path)
    if (
        observations_path.name != "observations.csv"
        or not observations_path.is_file()
        or not _resolved_same_path(observations_path.parent, approved_parent)
    ):
        raise FileNotFoundError(observations_path)
    observations_pin = config["data_contract"]["observations_csv"]
    oof_pin = config["immutable_references"]["exact_same_season_incumbent_oof"]
    oof_path = _repo_file(str(oof_pin["path"]))

    with _held_verified_bytes(
        observations_path,
        expected_bytes=int(observations_pin["bytes"]),
        expected_sha256=str(observations_pin["sha256"]),
    ) as (_observations_handle, observations_bytes, observations_digest), _held_verified_bytes(
        oof_path,
        expected_bytes=int(oof_pin["bytes"]),
        expected_sha256=str(oof_pin["sha256"]),
    ) as (_oof_handle, oof_bytes, oof_digest):
        if observations_digest != {
            "bytes": int(readiness["bytes"]),
            "sha256": str(readiness["sha256"]),
        }:
            raise RuntimeError("worker observations bytes differ from parent readiness receipt")
        if str(PROJECT_ROOT / "src") not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
        import lightgbm
        import numpy as np
        import pandas as pd
        from lightgbm import LGBMRegressor

        from p2_restore.features import build_training_features
        from p2_restore.normalized_curvature_residual import (
            align_exact_incumbent,
            build_normalized_curvature_design,
            decode_normalized_curvature,
            evaluate_stage1_gate,
            make_stage1_split,
            metric_report,
            paired_day_bootstrap,
            subset_design,
        )

        _assert_static_bundle_unchanged(config_path, config, bundle)
        observations = pd.read_csv(io.BytesIO(observations_bytes))
        exact_oof = pd.read_parquet(
            io.BytesIO(oof_bytes),
            columns=["time", "layer", "block", "truth", "prediction"],
        )
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
        expected_feature_columns = [
            str(value) for value in config["feature_contract"]["allowed_feature_columns"]
        ]
        if list(train_design.features.columns) != expected_feature_columns:
            raise RuntimeError("training NCR design columns differ from the sealed exact allow-list")
        if list(validation_design.features.columns) != expected_feature_columns:
            raise RuntimeError(
                "validation NCR design columns differ from the sealed exact allow-list"
            )
        alignment = align_exact_incumbent(
            validation_design,
            exact_oof,
            block=str(oof_pin["block"]),
            expected_rows=int(split_contract["expected_validation_rows_after_exact_alignment"]),
            truth_column=str(oof_pin["truth_column"]),
            prediction_column=str(oof_pin["prediction_column"]),
        )
        del exact_oof, observations, feature_table, full_design

        exact_aggregate = config["immutable_references"]["exact_same_season_aggregate"]
        incumbent_metrics = metric_report(
            alignment.truth, alignment.incumbent_prediction, alignment.layer
        )
        _assert_exact_metric_pin(incumbent_metrics, exact_aggregate)
        runtime_fingerprint = {
            "lightgbm_version": str(lightgbm.__version__),
            "loaded_lightgbm_native_library": _loaded_lightgbm_library_fingerprint(
                lightgbm, config
            ),
            "threadpools": _runtime_thread_fingerprint(),
            "preimport_pins": bundle["runtime_pins"],
        }

        predictions: list[Any] = []
        for slot, seed_value in enumerate(config["model"]["seeds"], start=1):
            seed = int(seed_value)
            if time.time() >= float(authorization["deadline_epoch"]):
                raise HardWallTimeout(f"deadline reached before physical fit slot {slot}")
            _verify_captured_source(
                observations_bytes,
                expected_bytes=int(observations_pin["bytes"]),
                expected_sha256=str(observations_pin["sha256"]),
                label="observations.csv",
            )
            _verify_captured_source(
                oof_bytes,
                expected_bytes=int(oof_pin["bytes"]),
                expected_sha256=str(oof_pin["sha256"]),
                label="exact incumbent OOF parquet",
            )
            _assert_static_bundle_unchanged(config_path, config, bundle)
            _reserve_fit_slot(
                journal_path,
                slot=slot,
                seed=seed,
                **event_base,
            )
            if time.time() >= float(authorization["deadline_epoch"]):
                _append_journal_event(
                    journal_path,
                    _journal_event(
                        "FIT_SLOT_ABORTED_BEFORE_CALL",
                        **event_base,
                        slot=slot,
                        seed=seed,
                        reason="HARD_DEADLINE_REACHED_AFTER_RESERVATION",
                    ),
                )
                raise HardWallTimeout(f"deadline reached after reserving fit slot {slot}")
            parameters = dict(config["model"]["parameters"])
            parameters["random_state"] = seed
            model = LGBMRegressor(**parameters)
            fit_started = time.monotonic()
            try:
                model.fit(train_design.features, train_design.normalized_target)
            except BaseException:
                _append_journal_event(
                    journal_path,
                    _journal_event(
                        "FIT_SLOT_FAILED",
                        **event_base,
                        slot=slot,
                        seed=seed,
                        error_type="MODEL_FIT_EXCEPTION",
                    ),
                )
                raise
            _complete_fit_slot(
                journal_path,
                slot=slot,
                seed=seed,
                elapsed_seconds=time.monotonic() - fit_started,
                **event_base,
            )
            predicted_curvature = np.asarray(
                model.predict(validation_design.features), dtype=float
            )
            predictions.append(
                decode_normalized_curvature(
                    predicted_curvature,
                    validation_design.baseline,
                    validation_design.profile_scale,
                )
            )
            _assert_static_bundle_unchanged(config_path, config, bundle)

        if _fit_slot_states(_read_journal(journal_path)) != {
            1: "COMPLETED",
            2: "COMPLETED",
            3: "COMPLETED",
        }:
            raise RuntimeError("three physical fit slots did not complete exactly once")
        candidate_all_validation = np.mean(np.vstack(predictions), axis=0)
        candidate_prediction = candidate_all_validation[alignment.candidate_positions]
        candidate_metrics = metric_report(
            alignment.truth, candidate_prediction, alignment.layer
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
        _verify_captured_source(
            observations_bytes,
            expected_bytes=int(observations_pin["bytes"]),
            expected_sha256=str(observations_pin["sha256"]),
            label="observations.csv",
        )
        _verify_captured_source(
            oof_bytes,
            expected_bytes=int(oof_pin["bytes"]),
            expected_sha256=str(oof_pin["sha256"]),
            label="exact incumbent OOF parquet",
        )
        _assert_static_bundle_unchanged(config_path, config, bundle)
        worker_result: dict[str, Any] = {
            "schema_version": "p2_ncr_stage1_worker.result.v1r4",
            "experiment_id": config["experiment_id"],
            "contract_sha256": authorization["contract_sha256"],
            "bundle_sha256": authorization["bundle_sha256"],
            "physical_fit_calls": 3,
            "seeds": [int(value) for value in config["model"]["seeds"]],
            "train_rows": int(len(train_design.features)),
            "validation_rows_before_exact_alignment": int(len(validation_design.features)),
            "feature_count": int(len(train_design.features.columns)),
            "rows": int(len(alignment.truth)),
            "incumbent_metrics": incumbent_metrics,
            "candidate_metrics": candidate_metrics,
            "paired_day_bootstrap": bootstrap,
            "stage1_gate": gate,
            "bound_sources": {
                "observations.csv": observations_digest,
                "exact_incumbent_oof.parquet": oof_digest,
            },
            "literal_readiness_sha256": authorization["literal_readiness_sha256"],
            "runtime_fingerprint": runtime_fingerprint,
            "official_or_submission_reads": 0,
            "row_predictions_written": False,
        }
    _append_journal_event(
        journal_path,
        _journal_event("WORKER_COMPLETED", **event_base, physical_fit_calls=3),
    )
    return worker_result


def _run_subprocess_with_deadline(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    deadline_epoch: float,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now_fn: Callable[[], float] = time.time,
) -> subprocess.CompletedProcess[str]:
    remaining = deadline_epoch - now_fn()
    if remaining <= 0:
        raise HardWallTimeout("hard wall elapsed before worker launch")
    try:
        return run_fn(
            list(command),
            cwd=PROJECT_ROOT,
            env=dict(environment),
            check=False,
            capture_output=True,
            text=True,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as error:
        raise HardWallTimeout("numerical worker exceeded the 1800-second hard wall") from error


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

    return {
        "head": run("rev-parse", "HEAD").stdout.strip(),
        "dirty": bool(run("status", "--porcelain=v1", "--untracked-files=normal").stdout),
    }


def _durable_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _exclusive_create_bytes(path, payload)


def _publish_aggregate(
    final: Path,
    result: Mapping[str, Any],
    manifest_without_result_hash: Mapping[str, Any],
) -> dict[str, Any]:
    if final.exists() or _staging_paths(final):
        raise ExistingAttemptError("final or orphan staging exists before publish")
    staging = final.parent / f".{final.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    result_path = staging / "result.json"
    _durable_new_json(result_path, result)
    staging_fsync_after_result = _fsync_directory_best_effort(staging)
    manifest = dict(manifest_without_result_hash)
    manifest["result"] = {
        "path": "result.json",
        "bytes": result_path.stat().st_size,
        "sha256": _sha256(result_path),
    }
    manifest["transactional_publish"] = {
        "same_filesystem_staging_then_atomic_directory_rename": True,
        "staging_directory_fsync_after_result_supported": staging_fsync_after_result,
        "directory_fsync_is_best_effort_on_this_platform": True,
        "orphan_staging_is_never_auto_resumed": True,
    }
    _durable_new_json(staging / "manifest.json", manifest)
    staging_fsync_complete = _fsync_directory_best_effort(staging)
    os.replace(staging, final)
    parent_fsync = _fsync_directory_best_effort(final.parent)
    return {
        "staging_directory_fsync_complete_supported": staging_fsync_complete,
        "final_parent_directory_fsync_supported": parent_fsync,
    }


def _execute_parent(
    config_path: Path, config: dict[str, Any], bundle: dict[str, Any]
) -> Path:
    _assert_static_bundle_unchanged(config_path, config, bundle)
    readiness, attempt = _prepare_readiness_then_acquire(config, bundle)
    journal_path = attempt["paths"]["journal"]
    event_base = {
        "contract_sha256": attempt["contract_sha256"],
        "bundle_sha256": attempt["bundle_sha256"],
        "attempt_token_sha256": attempt["token_sha256"],
    }
    environment = dict(os.environ)
    environment[WORKER_TOKEN_ENV] = attempt["token"]
    environment[WORKER_DEADLINE_ENV] = repr(attempt["deadline_epoch"])
    environment[WORKER_PARENT_PID_ENV] = str(os.getpid())
    environment.pop(str(config["data_contract"]["source_environment_variable"]), None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--internal-worker",
        "--verified-observations-path",
        str(config["data_contract"]["approved_literal_observations_path"]),
    ]
    _append_journal_event(
        journal_path,
        _journal_event(
            "PARENT_LAUNCHING_SINGLE_WORKER",
            **event_base,
            journal_schema=PRELAUNCH_JOURNAL_SCHEMA,
            parent_pid=os.getpid(),
            worker_restart_count=0,
            hard_wall_seconds=float(config["execution"]["hard_wall_seconds"]),
            literal_readiness_sha256=str(readiness["readiness_sha256"]),
        ),
    )
    try:
        completed = _run_subprocess_with_deadline(
            command,
            environment=environment,
            deadline_epoch=float(attempt["deadline_epoch"]),
        )
    except BaseException as error:
        _append_journal_event(
            journal_path,
            _journal_event(
                "ATTEMPT_TERMINAL_FAILED",
                **event_base,
                reason=type(error).__name__,
                automatic_rerun_allowed=False,
            ),
        )
        raise
    if completed.returncode != 0:
        _append_journal_event(
            journal_path,
            _journal_event(
                "ATTEMPT_TERMINAL_FAILED",
                **event_base,
                reason="WORKER_NONZERO_EXIT",
                worker_returncode=int(completed.returncode),
                worker_stdout_sha256=_sha256_bytes(completed.stdout.encode("utf-8")),
                worker_stderr_sha256=_sha256_bytes(completed.stderr.encode("utf-8")),
                automatic_rerun_allowed=False,
            ),
        )
        raise RuntimeError("sealed numerical worker failed; same experiment cannot rerun")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("sealed worker stdout protocol drift")
    worker_result = json.loads(lines[0])
    if not isinstance(worker_result, dict):
        raise RuntimeError("sealed worker result is not a JSON object")
    required_worker_values = {
        "schema_version": "p2_ncr_stage1_worker.result.v1r4",
        "experiment_id": config["experiment_id"],
        "contract_sha256": attempt["contract_sha256"],
        "bundle_sha256": attempt["bundle_sha256"],
        "physical_fit_calls": 3,
        "literal_readiness_sha256": str(readiness["readiness_sha256"]),
    }
    for name, expected in required_worker_values.items():
        if worker_result.get(name) != expected:
            raise RuntimeError(f"sealed worker result contract mismatch: {name}")
    if _fit_slot_states(_read_journal(journal_path)) != {
        1: "COMPLETED",
        2: "COMPLETED",
        3: "COMPLETED",
    }:
        raise RuntimeError("journal does not prove exactly three completed fit slots")

    _assert_static_bundle_unchanged(config_path, config, bundle)
    _append_journal_event(
        journal_path,
        _journal_event("PARENT_ACCEPTED_WORKER_RESULT", **event_base),
    )
    result: dict[str, Any] = {
        "schema_version": "p2_normalized_curvature_residual_lgbm_stage1.result.v1r4",
        "experiment_id": config["experiment_id"],
        "status": (
            "COMPLETE_STAGE1_PASS"
            if worker_result["stage1_gate"]["passed"]
            else "COMPLETE_STAGE1_FAIL"
        ),
        "started_at_kst": _read_journal(journal_path)[0]["at_kst"],
        "completed_at_kst": _now_kst(),
        "family": config["family"],
        "population": config["immutable_references"]["exact_same_season_aggregate"][
            "population"
        ],
        "rows": worker_result["rows"],
        "physical_fit_calls": worker_result["physical_fit_calls"],
        "seeds": worker_result["seeds"],
        "train_rows": worker_result["train_rows"],
        "validation_rows_before_exact_alignment": worker_result[
            "validation_rows_before_exact_alignment"
        ],
        "feature_count": worker_result["feature_count"],
        "incumbent_metrics": worker_result["incumbent_metrics"],
        "candidate_metrics": worker_result["candidate_metrics"],
        "paired_day_bootstrap": worker_result["paired_day_bootstrap"],
        "stage1_gate": worker_result["stage1_gate"],
        "bound_sources": worker_result["bound_sources"],
        "stage2_executed": False,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
        "submission_files_generated": 0,
        "uploads": 0,
        "row_predictions_written": False,
        "csv_files_written": False,
        "scientific_limitations": list(config["scientific_limitations"]),
    }
    journal_prefix = journal_path.read_bytes()
    manifest: dict[str, Any] = {
        "schema_version": "p2_normalized_curvature_residual_lgbm_stage1.manifest.v1r4",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _now_kst(),
        "status": result["status"],
        "append_only": True,
        "aggregate_only": True,
        "config": bundle["config"],
        "bundle_sha256": bundle["bundle_sha256"],
        "implementation_pins": bundle["implementation_pins"],
        "superseded_lineage_pins": bundle["superseded_lineage_pins"],
        "immutable_references": bundle["immutable_references"],
        "runtime_pins": bundle["runtime_pins"],
        "worker_runtime_fingerprint": worker_result["runtime_fingerprint"],
        "source_hash_read_binding": worker_result["bound_sources"],
        "literal_data_readiness": readiness,
        "attempt_control": {
            "claim_path": attempt["paths"]["claim"].relative_to(PROJECT_ROOT).as_posix(),
            "claim_sha256": _sha256(attempt["paths"]["claim"]),
            "journal_path": journal_path.relative_to(PROJECT_ROOT).as_posix(),
            "journal_prefix_bytes_at_publish": len(journal_prefix),
            "journal_prefix_sha256_at_publish": _sha256_bytes(journal_prefix),
            "automatic_rerun_allowed": False,
            "maximum_physical_fit_calls_lifetime": 3,
            "worker_restart_count": 0,
            "hard_wall_seconds": 1800,
        },
        "environment": {
            "python_full": sys.version,
            "platform": platform.platform(),
            "git": _git_state(),
        },
        "external_actions": {
            "official_test_reads": 0,
            "sample_submission_reads": 0,
            "submission_candidate_reads": 0,
            "submission_files_generated": 0,
            "uploads": 0,
        },
        "known_limitations": list(config["known_limitations"]),
        "scientific_limitations": list(config["scientific_limitations"]),
    }
    _append_journal_event(
        journal_path,
        _journal_event("PUBLISH_STAGING_STARTED", **event_base),
    )
    _assert_static_bundle_unchanged(config_path, config, bundle)
    publish_durability = _publish_aggregate(attempt["paths"]["final"], result, manifest)
    _append_journal_event(
        journal_path,
        _journal_event(
            "ATTEMPT_TERMINAL_COMPLETE",
            **event_base,
            final_status=result["status"],
            publish_durability=publish_durability,
            automatic_rerun_allowed=False,
        ),
    )
    return attempt["paths"]["final"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-stage1", action="store_true")
    mode.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--verified-observations-path",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config, bundle = _verify_static_bundle(config_path)
    if args.internal_worker:
        try:
            result = _run_internal_worker(
                config_path,
                config,
                bundle,
                args.verified_observations_path,
            )
        except BaseException as error:
            with contextlib.suppress(Exception):
                authorization = _validate_worker_claim(config, bundle)
                _append_journal_event(
                    authorization["paths"]["journal"],
                    _journal_event(
                        "WORKER_FAILED",
                        contract_sha256=authorization["contract_sha256"],
                        bundle_sha256=authorization["bundle_sha256"],
                        attempt_token_sha256=authorization["token_sha256"],
                        error_type=type(error).__name__,
                    ),
                )
            raise
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0

    if args.verified_observations_path is not None:
        raise RuntimeError("sealed observations argument is internal-worker-only")

    state = _inspect_control_state(config)
    if not args.execute_stage1:
        literal_readiness = _literal_source_readiness(
            config,
            bundle,
            require_environment=False,
        )
        environment_readiness = literal_readiness["execution_environment"]
        print(
            json.dumps(
                {
                    "status": "STRICT_PREFLIGHT_PASS_NOT_EXECUTED",
                    "experiment_id": config["experiment_id"],
                    "config_sha256": bundle["config"]["sha256"],
                    "bundle_sha256": bundle["bundle_sha256"],
                    "control_state": state,
                    "verified_implementation_count": len(bundle["implementation_pins"]),
                    "verified_lineage_count": len(bundle["superseded_lineage_pins"]),
                    "verified_reference_count": len(bundle["immutable_references"]),
                    "verified_runtime": bundle["runtime_pins"],
                    "literal_data_ready": True,
                    "literal_data_readiness": literal_readiness,
                    "execution_env_ready": bool(
                        environment_readiness["present"]
                        and environment_readiness["matches_approved_literal_parent"]
                    ),
                    "superseded_v1_v1r2_v1r3_execution_forbidden": True,
                    "maximum_physical_fit_calls_lifetime": 3,
                    "worker_restart_count": 0,
                    "hard_wall_seconds": 1800,
                    "numerical_execution": False,
                    "official_or_submission_reads": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if not state["eligible"]:
        raise ExistingAttemptError(state["status"])
    final = _execute_parent(config_path, config, bundle)
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
