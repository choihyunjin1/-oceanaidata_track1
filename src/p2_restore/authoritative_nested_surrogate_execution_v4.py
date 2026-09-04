"""Crash-resumable v4 control plane for the sealed P2 v3 science contract.

V4 deliberately does not change a scientific surface.  The supervised ledger,
splits, components, hyperparameters, meta refit, postprocess, metrics, and
78,156-row outer population are inherited from v3.  This module only makes an
authorized v4 run safely resumable in its own namespace.

The coordinator has two transaction paths:

* clean start: run the exact actual-data zero-fit semantic preflight before the
  actual namespace exists, then acquire the exclusive lock and create an
  immutable execution-start receipt;
* crash resume: inspect the existing namespace without writing, acquire the
  same non-blocking lock, repeat the inspection and semantic preflight, consume
  one of at most two resume attempts, then reuse only fully verified v4 jobs.

A final result is written through a unique fsynced partial and atomic rename.
Failed/stale partials are preserved but never count as terminal completion.
"""

from __future__ import annotations

import hashlib
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from p2_restore import authoritative_nested_surrogate_execution as v2
from p2_restore import authoritative_nested_surrogate_execution_v3 as v3
from p2_restore.authoritative_nested_surrogate_conformance import PrefixPlan

MAXIMUM_RESUME_ATTEMPTS = 2
MAXIMUM_TOTAL_ATTEMPTS = 1 + MAXIMUM_RESUME_ATTEMPTS
START_RECEIPT_NAME = "execution_start.json"
LOCK_NAME = "execution.lock"
RESULT_NAME = "result.json"
TERMINAL_RECEIPT_NAME = "terminal_receipt.json"
ATTEMPTS_DIRECTORY_NAME = "attempts"
TERMINAL_STATUS = "COMPLETE_LOCAL_AUTHORITATIVE_SURROGATE_V4_NO_PROMOTION"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def _publish_control_json_atomic(path: Path, value: Any) -> dict[str, Any]:
    """Atomically publish one binding-critical receipt; preserve failed partials."""

    return v2.atomic_write_or_verify(path, _json_bytes(value))


def _self_hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = dict(value)
    _require(field not in payload, f"self-hash field already exists: {field}")
    payload[field] = v2.canonical_sha256(payload)
    return payload


def _verify_self_hash(value: Mapping[str, Any], field: str, message: str) -> str:
    payload = dict(value)
    claimed = str(payload.pop(field, ""))
    _require(len(claimed) == 64, message)
    _require(v2.canonical_sha256(payload) == claimed, message)
    return claimed


@dataclass(frozen=True)
class ExecutionBindingV4:
    """Every immutable input that a first start and every resume must share."""

    namespace: str
    execution_contract_sha256: str
    parent_recipe_sha256: str
    preexecution_seal_sha256: str
    semantic_preflight_sha256: str
    exact_command_sha256: str
    authorization_sha256: str
    module_sha256: str
    runner_sha256: str
    expected_terminal_status: str = TERMINAL_STATUS
    maximum_resume_attempts: int = MAXIMUM_RESUME_ATTEMPTS

    def as_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "execution_contract_sha256": self.execution_contract_sha256,
            "parent_recipe_sha256": self.parent_recipe_sha256,
            "preexecution_seal_sha256": self.preexecution_seal_sha256,
            "semantic_preflight_sha256": self.semantic_preflight_sha256,
            "exact_command_sha256": self.exact_command_sha256,
            "authorization_sha256": self.authorization_sha256,
            "module_sha256": self.module_sha256,
            "runner_sha256": self.runner_sha256,
            "job_store_contract_sha256": self.preexecution_seal_sha256,
            "expected_terminal_status": self.expected_terminal_status,
            "maximum_resume_attempts": self.maximum_resume_attempts,
            "maximum_total_attempts": 1 + self.maximum_resume_attempts,
        }

    def validate(self) -> None:
        _require(self.namespace != "", "v4 namespace is empty")
        _require(Path(self.namespace).name == self.namespace, "v4 namespace is unsafe")
        for name, digest in self.as_dict().items():
            if name.endswith("sha256"):
                _require(
                    isinstance(digest, str)
                    and len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest),
                    f"invalid v4 binding digest: {name}",
                )
        _require(
            self.maximum_resume_attempts == MAXIMUM_RESUME_ATTEMPTS,
            "v4 resume budget changed",
        )
        _require(
            self.expected_terminal_status == TERMINAL_STATUS,
            "v4 terminal status changed",
        )


@dataclass(frozen=True)
class SemanticPreflightOutcomeV4:
    semantic_sha256: str
    execution_context: Any


class TerminalExecutionClosed(RuntimeError):
    """Raised when an atomic terminal result already closes this namespace."""


class ResumeBudgetExhausted(RuntimeError):
    """Raised when the initial start plus two crash resumes were consumed."""


class DeterministicExecutionClosed(RuntimeError):
    """Raised when a graceful deterministic failure forbids automatic resume."""


class TransientExecutionError(RuntimeError):
    """An explicitly classified runtime failure that may consume one resume."""


def semantic_preflight_actual_data_v4(
    observations: Any,
    *,
    recipe: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[tuple[PrefixPlan, ...], dict[str, Any]]:
    """Run the exact v3 scientific preflight and wrap its unchanged evidence."""

    plans, scientific = v3.semantic_preflight_actual_data(
        observations, recipe=recipe, config=config
    )
    scientific_sha = str(scientific["semantic_receipt_sha256"])
    receipt = dict(scientific)
    receipt.pop("semantic_receipt_sha256")
    receipt["schema_version"] = "p2_authoritative_actual_data_semantic_preflight.v4"
    receipt["scientific_preflight_v3_sha256"] = scientific_sha
    receipt["operational_revision"] = {
        "science_surface_changed": False,
        "clean_start_preflight_before_actual_namespace": True,
        "resume_read_only_namespace_audit_before_lock": True,
        "resume_preflight_under_exclusive_lock_before_fit": True,
        "maximum_resume_attempts_after_initial": MAXIMUM_RESUME_ATTEMPTS,
        "terminal_result_atomic_publish": True,
    }
    receipt["semantic_receipt_sha256"] = v2.canonical_sha256(receipt)
    return plans, receipt


def execute_authorized_curve_v4(
    *,
    observations: Any,
    plans: Sequence[PrefixPlan],
    parent_recipe: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    """Execute the byte-pinned v3 science graph in the isolated v4 JobStores."""

    result = v3.execute_authorized_curve_v3(
        observations=observations,
        plans=plans,
        parent_recipe=parent_recipe,
        config=config,
        output_dir=output_dir,
        contract_sha256=contract_sha256,
    )
    result = dict(result)
    _require(
        result["status"] == "COMPLETE_LOCAL_AUTHORITATIVE_SURROGATE_V3_NO_PROMOTION",
        "inherited v3 execution status changed",
    )
    result["status"] = TERMINAL_STATUS
    result["scientific_surface_inherited_byte_pinned_from_v3"] = True
    result["resume_or_result_based_tuning_performed"] = False
    return result


def _validate_job_manifest(
    directory: Path, *, job_id: str, contract_sha256: str
) -> dict[str, Any]:
    _require(directory.is_dir() and not directory.is_symlink(), "job target is unsafe")
    manifest_path = directory / "manifest.json"
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "job manifest missing")
    manifest = _read_json(manifest_path)
    _require(manifest.get("complete") is True, "job manifest is incomplete")
    _require(manifest.get("job_id") == job_id, "job id changed on resume")
    _require(
        manifest.get("contract_sha256") == contract_sha256,
        "job contract hash changed on resume",
    )
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "job file pins are missing")
    _require(
        {"prediction.parquet", "receipt.json"}.issubset(files),
        "job core payloads are missing",
    )
    for name, pin in files.items():
        _require(Path(str(name)).name == name, "nested job payload is forbidden")
        path = directory / name
        _require(path.is_file() and not path.is_symlink(), f"job payload missing: {name}")
        _require(path.stat().st_size == int(pin["bytes"]), f"job bytes changed: {name}")
        _require(v2.sha256_file(path) == pin["sha256"], f"job hash changed: {name}")
    payload_files = manifest.get("payload_files")
    _require(isinstance(payload_files, list), "job payload file ledger changed")
    _require(set(payload_files).issubset(files), "job payload file pin missing")
    expected_entries = {"manifest.json", *files.keys()}
    actual_entries = {path.name for path in directory.iterdir()}
    _require(actual_entries == expected_entries, "unexpected file inside completed job")
    return {
        "job_id": job_id,
        "manifest_sha256": v2.sha256_file(manifest_path),
        "payload_count": len(files),
    }


_PARTIAL_DIRECTORY = re.compile(r"^\.[A-Za-z0-9_.-]+\.partial\.\d+\.[0-9a-f]+$")


def validate_job_store_read_only(root: Path, *, contract_sha256: str) -> dict[str, Any]:
    """Verify every committed v4 job/cell; audit partials without reusing them."""

    if not root.exists():
        return {
            "root_exists": False,
            "completed_jobs": 0,
            "preserved_partial_directories": 0,
            "ordered_manifest_ledger_sha256": v2.canonical_sha256([]),
        }
    _require(root.is_dir() and not root.is_symlink(), "JobStore root is unsafe")
    manifests: list[dict[str, Any]] = []
    partials = 0
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        _require(not path.is_symlink(), "JobStore symlink is forbidden")
        if path.name.startswith("."):
            _require(
                path.is_dir() and _PARTIAL_DIRECTORY.fullmatch(path.name) is not None,
                "unexpected JobStore partial entry",
            )
            partials += 1
            continue
        manifests.append(
            _validate_job_manifest(
                path, job_id=path.name, contract_sha256=contract_sha256
            )
        )
    return {
        "root_exists": True,
        "completed_jobs": len(manifests),
        "preserved_partial_directories": partials,
        "ordered_manifest_ledger_sha256": v2.canonical_sha256(manifests),
    }


def _verify_start_receipt(
    path: Path, binding: ExecutionBindingV4
) -> tuple[dict[str, Any], str]:
    value = _read_json(path)
    _require(
        value.get("schema_version") == "p2_authoritative_execution_start.v4",
        "v4 execution-start schema changed",
    )
    _require(value.get("status") == "STARTED_INCOMPLETE_RESUMABLE", "v4 start status changed")
    _require(value.get("binding") == binding.as_dict(), "v4 execution binding changed")
    _require(value.get("initial_start_attempt_count") == 1, "v4 initial count changed")
    _require(value.get("resume_attempt_budget") == MAXIMUM_RESUME_ATTEMPTS, "v4 budget changed")
    digest = _verify_self_hash(value, "execution_start_sha256", "v4 start self-hash changed")
    return value, digest


def _verify_attempt_ledger(
    attempts_dir: Path,
    *,
    execution_start_sha256: str,
    binding: ExecutionBindingV4,
) -> dict[str, Any]:
    if not attempts_dir.exists():
        return {
            "resume_starts": [],
            "terminal_by_attempt": {},
            "latest_attempt_terminal_status": None,
            "stale_control_partials": 0,
        }
    _require(attempts_dir.is_dir() and not attempts_dir.is_symlink(), "attempt ledger is unsafe")
    files = sorted(attempts_dir.iterdir(), key=lambda item: item.name)
    _require(all(path.is_file() and not path.is_symlink() for path in files), "attempt entry is unsafe")
    stale_control_partials = [
        path
        for path in files
        if path.name.startswith(".resume_attempt_")
        or path.name.startswith(".attempt_")
    ]
    _require(
        all(".json.partial." in path.name for path in stale_control_partials),
        "unexpected attempt control partial",
    )
    files = [path for path in files if path not in stale_control_partials]
    resume_paths = sorted(
        (path for path in files if re.fullmatch(r"resume_attempt_\d{3}\.json", path.name)),
        key=lambda item: item.name,
    )
    terminal_paths = sorted(
        (path for path in files if re.fullmatch(r"attempt_\d{3}_terminal\.json", path.name)),
        key=lambda item: item.name,
    )
    _require(
        len(resume_paths) + len(terminal_paths) == len(files),
        "unexpected attempt-ledger entry",
    )
    expected_names = [
        f"resume_attempt_{number:03d}.json"
        for number in range(2, 2 + len(resume_paths))
    ]
    _require([path.name for path in resume_paths] == expected_names, "resume attempt sequence changed")
    starts: list[dict[str, Any]] = []
    for number, path in enumerate(resume_paths, start=2):
        value = _read_json(path)
        _require(
            value.get("schema_version") == "p2_authoritative_resume_attempt.v4",
            "resume attempt schema changed",
        )
        _require(value.get("status") == "RESUME_STARTED", "resume attempt status changed")
        _require(value.get("attempt_number") == number, "resume attempt number changed")
        _require(
            value.get("resume_attempt_number") == number - 1,
            "resume attempt ordinal changed",
        )
        _require(
            value.get("execution_start_sha256") == execution_start_sha256,
            "resume attempt start binding changed",
        )
        _require(value.get("binding") == binding.as_dict(), "resume attempt binding changed")
        _verify_self_hash(value, "resume_attempt_sha256", "resume attempt self-hash changed")
        starts.append(value)
    _require(len(starts) <= MAXIMUM_RESUME_ATTEMPTS, "v4 resume budget was exceeded")
    terminal_by_attempt: dict[int, dict[str, Any]] = {}
    started_numbers = {1, *(int(value["attempt_number"]) for value in starts)}
    allowed_statuses = {
        "FAILED_TRANSIENT_RESUMABLE",
        "FAILED_DETERMINISTIC_CLOSED",
        "COMPLETE_TERMINAL",
    }
    for path in terminal_paths:
        value = _read_json(path)
        _require(
            value.get("schema_version") == "p2_authoritative_attempt_terminal.v4",
            "attempt terminal schema changed",
        )
        number = int(value.get("attempt_number", -1))
        _require(number in started_numbers, "terminal receipt has no started attempt")
        _require(number not in terminal_by_attempt, "duplicate attempt terminal receipt")
        _require(value.get("status") in allowed_statuses, "attempt terminal status changed")
        _require(
            value.get("execution_start_sha256") == execution_start_sha256,
            "attempt terminal start binding changed",
        )
        _require(value.get("binding") == binding.as_dict(), "attempt terminal binding changed")
        _verify_self_hash(value, "attempt_terminal_sha256", "attempt terminal self-hash changed")
        terminal_by_attempt[number] = value
    _require(
        all(number <= 1 + len(starts) for number in terminal_by_attempt),
        "attempt terminal sequence changed",
    )
    latest_number = 1 + len(starts)
    latest = terminal_by_attempt.get(latest_number)
    return {
        "resume_starts": starts,
        "terminal_by_attempt": terminal_by_attempt,
        "latest_attempt_terminal_status": None if latest is None else latest["status"],
        "stale_control_partials": len(stale_control_partials),
    }


def _failure_evidence(error: BaseException) -> dict[str, Any]:
    message = str(error).replace("\r", " ").replace("\n", " ")[:1000]
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return {
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "exception_message": message,
        "traceback_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
        "traceback_text_recorded": False,
        "raw_observation_values_recorded": False,
    }


def _create_attempt_terminal(
    actual_dir: Path,
    *,
    binding: ExecutionBindingV4,
    execution_start_sha256: str,
    attempt_number: int,
    status: str,
    now: Callable[[], str],
    error: BaseException | None = None,
) -> dict[str, Any]:
    _require(
        status
        in {
            "FAILED_TRANSIENT_RESUMABLE",
            "FAILED_DETERMINISTIC_CLOSED",
            "COMPLETE_TERMINAL",
        },
        "invalid attempt terminal status",
    )
    evidence = (
        {
            "exception_type": None,
            "exception_message": None,
            "traceback_sha256": None,
            "traceback_text_recorded": False,
            "raw_observation_values_recorded": False,
        }
        if error is None
        else _failure_evidence(error)
    )
    value = _self_hashed(
        {
            "schema_version": "p2_authoritative_attempt_terminal.v4",
            "status": status,
            "recorded_at_kst": now(),
            "attempt_number": int(attempt_number),
            "execution_start_sha256": execution_start_sha256,
            "binding": binding.as_dict(),
            "classification": (
                "TRANSIENT_RUNTIME_EXPLICIT"
                if status == "FAILED_TRANSIENT_RESUMABLE"
                else "DETERMINISTIC_FAIL_CLOSED"
                if status == "FAILED_DETERMINISTIC_CLOSED"
                else "SUCCESS"
            ),
            "automatic_resume_permitted": status == "FAILED_TRANSIENT_RESUMABLE",
            **evidence,
        },
        "attempt_terminal_sha256",
    )
    _publish_control_json_atomic(
        actual_dir
        / ATTEMPTS_DIRECTORY_NAME
        / f"attempt_{attempt_number:03d}_terminal.json",
        value,
    )
    return value


def _create_gate_failure(
    actual_dir: Path,
    *,
    binding: ExecutionBindingV4,
    phase: str,
    error: BaseException,
    now: Callable[[], str],
) -> dict[str, Any]:
    _require(phase in {"prestart", "resume"}, "invalid v4 gate failure phase")
    value = _self_hashed(
        {
            "schema_version": "p2_authoritative_gate_failure.v4",
            "status": "FAILED_DETERMINISTIC_CLOSED",
            "phase": phase,
            "recorded_at_kst": now(),
            "binding": binding.as_dict(),
            "attempt_started": False,
            "automatic_resume_permitted": False,
            **_failure_evidence(error),
        },
        "gate_failure_sha256",
    )
    v2.atomic_write_or_verify(
        actual_dir / f"{phase}_gate_failure.json", _json_bytes(value)
    )
    return value


def _terminal_state(actual_dir: Path, binding: ExecutionBindingV4) -> dict[str, Any] | None:
    result_path = actual_dir / RESULT_NAME
    receipt_path = actual_dir / TERMINAL_RECEIPT_NAME
    if not result_path.exists():
        _require(not receipt_path.exists(), "terminal receipt exists without atomic result")
        return None
    _require(result_path.is_file() and not result_path.is_symlink(), "terminal result is unsafe")
    result = _read_json(result_path)
    _require(
        result.get("status") == binding.expected_terminal_status,
        "terminal result status changed",
    )
    expected_binding_sha = v2.canonical_sha256(binding.as_dict())
    _require(
        result.get("execution_binding_sha256") == expected_binding_sha,
        "terminal result execution binding changed",
    )
    _require(
        result.get("preexecution_seal_sha256") == binding.preexecution_seal_sha256,
        "terminal result seal binding changed",
    )
    _require(
        result.get("semantic_preflight_sha256") == binding.semantic_preflight_sha256,
        "terminal result semantic binding changed",
    )
    _require(result.get("initial_start_attempt_count") == 1, "terminal initial count changed")
    total_attempts = int(result.get("total_attempts_started", -1))
    resume_attempts = int(result.get("resume_attempts_started", -1))
    _require(
        total_attempts == 1 + resume_attempts
        and 1 <= total_attempts <= MAXIMUM_TOTAL_ATTEMPTS,
        "terminal attempt counts changed",
    )
    _require(
        result.get("submission_files_generated") == 0 and result.get("uploads") == 0,
        "terminal result expanded submission scope",
    )
    _require(
        result.get("official_test_sample_submission_reads") == 0,
        "terminal result reports forbidden official reads",
    )
    result_sha = v2.sha256_file(result_path)
    state = {
        "status": "TERMINAL_COMPLETE_NO_RERUN",
        "result_sha256": result_sha,
        "result_bytes": result_path.stat().st_size,
        "terminal_receipt_present": receipt_path.exists(),
        "result_total_attempts_started": total_attempts,
    }
    if receipt_path.exists():
        _require(receipt_path.is_file() and not receipt_path.is_symlink(), "terminal receipt is unsafe")
        receipt = _read_json(receipt_path)
        _require(
            receipt.get("schema_version") == "p2_authoritative_terminal_receipt.v4",
            "terminal receipt schema changed",
        )
        _require(receipt.get("status") == "TERMINAL_COMPLETE_NO_RERUN", "terminal receipt status changed")
        _require(receipt.get("binding") == binding.as_dict(), "terminal binding changed")
        _require(receipt.get("result_sha256") == result_sha, "terminal result hash changed")
        _require(
            receipt.get("result_bytes") == result_path.stat().st_size,
            "terminal result bytes changed",
        )
        _verify_self_hash(receipt, "terminal_receipt_sha256", "terminal receipt self-hash changed")
    return state


def inspect_actual_namespace_read_only(
    actual_dir: Path, *, binding: ExecutionBindingV4
) -> dict[str, Any]:
    """Audit an existing v4 namespace without creating, deleting, or repairing it."""

    binding.validate()
    actual_dir = actual_dir.resolve()
    _require(actual_dir.exists(), "v4 actual namespace does not exist")
    _require(actual_dir.name == binding.namespace, "v4 actual namespace differs")
    _require(actual_dir.is_dir() and not actual_dir.is_symlink(), "v4 actual namespace is unsafe")
    terminal = _terminal_state(actual_dir, binding)
    start_path = actual_dir / START_RECEIPT_NAME
    if not start_path.exists():
        _require(terminal is None, "terminal result exists without immutable execution start")
        allowed = {LOCK_NAME, "prestart_gate_failure.json"}
        entries = list(actual_dir.iterdir())
        stale_prestart_control_partials = 0
        for path in entries:
            _require(not path.is_symlink(), "unbound v4 namespace symlink is forbidden")
            if path.name in allowed:
                continue
            if (
                path.name.startswith(f".{START_RECEIPT_NAME}.partial.")
                or path.name.startswith(".prestart_gate_failure.json.partial.")
            ):
                _require(path.is_file(), "prestart control partial is not a file")
                stale_prestart_control_partials += 1
                continue
            raise ValueError("unbound v4 namespace contains execution state")
        gate_failure = actual_dir / "prestart_gate_failure.json"
        if gate_failure.exists():
            value = _read_json(gate_failure)
            _require(
                value.get("schema_version") == "p2_authoritative_gate_failure.v4"
                and value.get("status") == "FAILED_DETERMINISTIC_CLOSED"
                and value.get("phase") == "prestart"
                and value.get("binding") == binding.as_dict(),
                "prestart gate failure changed",
            )
            _verify_self_hash(
                value, "gate_failure_sha256", "prestart gate failure self-hash changed"
            )
            return {
                "status": "FAILED_DETERMINISTIC_CLOSED",
                "terminal": terminal,
                "resume_attempts_started": 0,
                "total_attempts_started": 0,
                "remaining_resume_budget": 0,
                "automatic_resume_permitted": False,
                "latest_attempt_terminal_status": "FAILED_DETERMINISTIC_CLOSED",
                "jobs": validate_job_store_read_only(
                    actual_dir / "jobs",
                    contract_sha256=binding.preexecution_seal_sha256,
                ),
                "cells": validate_job_store_read_only(
                    actual_dir / "cells",
                    contract_sha256=binding.preexecution_seal_sha256,
                ),
                "stale_terminal_partials": 0,
                "stale_control_partials": stale_prestart_control_partials,
            }
        return {
            "status": "EMPTY_PRESTART_NAMESPACE",
            "terminal": terminal,
            "resume_attempts_started": 0,
            "total_attempts_started": 0,
            "automatic_resume_permitted": True,
            "latest_attempt_terminal_status": None,
            "jobs": validate_job_store_read_only(
                actual_dir / "jobs", contract_sha256=binding.preexecution_seal_sha256
            ),
            "cells": validate_job_store_read_only(
                actual_dir / "cells", contract_sha256=binding.preexecution_seal_sha256
            ),
            "stale_terminal_partials": 0,
            "stale_control_partials": stale_prestart_control_partials,
        }
    _require((actual_dir / LOCK_NAME).is_file(), "v4 persistent lock file is missing")
    _, start_sha = _verify_start_receipt(start_path, binding)
    attempt_ledger = _verify_attempt_ledger(
        actual_dir / ATTEMPTS_DIRECTORY_NAME,
        execution_start_sha256=start_sha,
        binding=binding,
    )
    attempts = attempt_ledger["resume_starts"]
    jobs = validate_job_store_read_only(
        actual_dir / "jobs", contract_sha256=binding.preexecution_seal_sha256
    )
    cells = validate_job_store_read_only(
        actual_dir / "cells", contract_sha256=binding.preexecution_seal_sha256
    )
    allowed_exact = {
        LOCK_NAME,
        START_RECEIPT_NAME,
        ATTEMPTS_DIRECTORY_NAME,
        "jobs",
        "cells",
        RESULT_NAME,
        TERMINAL_RECEIPT_NAME,
        "evaluated_oof_040.parquet",
        "evaluated_oof_055.parquet",
        "evaluated_oof_070.parquet",
        "evaluated_oof_085.parquet",
        "evaluated_oof_100.parquet",
        "resume_gate_failure.json",
    }
    stale_terminal_partials = 0
    evaluated_partials = 0
    for path in actual_dir.iterdir():
        _require(not path.is_symlink(), "v4 actual namespace symlink is forbidden")
        if path.name in allowed_exact:
            continue
        if path.name.startswith(f".{RESULT_NAME}.partial.") or path.name.startswith(
            f".{TERMINAL_RECEIPT_NAME}.partial."
        ):
            _require(path.is_file(), "terminal partial is not a file")
            stale_terminal_partials += 1
            continue
        if path.name.startswith(".resume_gate_failure.json.partial."):
            _require(path.is_file(), "resume gate control partial is not a file")
            continue
        if path.name.startswith(f".{START_RECEIPT_NAME}.partial.") or path.name.startswith(
            ".prestart_gate_failure.json.partial."
        ):
            _require(path.is_file(), "start control partial is not a file")
            continue
        if path.name.startswith(".evaluated_oof_") and ".parquet.partial." in path.name:
            _require(path.is_file(), "evaluated OOF partial is not a file")
            evaluated_partials += 1
            continue
        raise ValueError(f"unexpected v4 actual namespace entry: {path.name}")
    gate_failure_path = actual_dir / "resume_gate_failure.json"
    gate_failure: dict[str, Any] | None = None
    if gate_failure_path.exists():
        gate_failure = _read_json(gate_failure_path)
        _require(
            gate_failure.get("schema_version") == "p2_authoritative_gate_failure.v4"
            and gate_failure.get("status") == "FAILED_DETERMINISTIC_CLOSED"
            and gate_failure.get("phase") == "resume"
            and gate_failure.get("binding") == binding.as_dict(),
            "resume gate failure changed",
        )
        _verify_self_hash(
            gate_failure, "gate_failure_sha256", "resume gate failure self-hash changed"
        )
    latest_terminal_status = attempt_ledger["latest_attempt_terminal_status"]
    terminal_needs_finalization = terminal is not None and (
        not terminal["terminal_receipt_present"]
        or latest_terminal_status != "COMPLETE_TERMINAL"
    )
    if terminal is not None:
        _require(
            int(terminal["result_total_attempts_started"]) == 1 + len(attempts),
            "terminal result attempt count differs from ledger",
        )
    deterministic_closed = gate_failure is not None or (
        terminal is None
        and latest_terminal_status in {
            "FAILED_DETERMINISTIC_CLOSED",
            "COMPLETE_TERMINAL",
        }
    )
    automatic_resume_permitted = (
        terminal is None
        and not deterministic_closed
        and latest_terminal_status in {None, "FAILED_TRANSIENT_RESUMABLE"}
        and len(attempts) < MAXIMUM_RESUME_ATTEMPTS
    )
    return {
        "status": (
            "TERMINAL_RESULT_NEEDS_FINALIZATION"
            if terminal_needs_finalization
            else "TERMINAL_COMPLETE_NO_RERUN"
            if terminal
            else "FAILED_DETERMINISTIC_CLOSED"
            if deterministic_closed
            else "INTERRUPTED_INCOMPLETE_RESUMABLE"
        ),
        "terminal": terminal,
        "execution_start_sha256": start_sha,
        "resume_attempts_started": len(attempts),
        "total_attempts_started": 1 + len(attempts),
        "remaining_resume_budget": MAXIMUM_RESUME_ATTEMPTS - len(attempts),
        "latest_attempt_terminal_status": latest_terminal_status,
        "automatic_resume_permitted": automatic_resume_permitted,
        "jobs": jobs,
        "cells": cells,
        "stale_terminal_partials": stale_terminal_partials,
        "stale_evaluated_oof_partials": evaluated_partials,
        "stale_control_partials": attempt_ledger["stale_control_partials"],
    }


def _create_execution_start(
    actual_dir: Path,
    *,
    binding: ExecutionBindingV4,
    now: Callable[[], str],
) -> tuple[dict[str, Any], str]:
    value = _self_hashed(
        {
            "schema_version": "p2_authoritative_execution_start.v4",
            "status": "STARTED_INCOMPLETE_RESUMABLE",
            "created_at_kst": now(),
            "binding": binding.as_dict(),
            "initial_start_attempt_count": 1,
            "resume_attempt_budget": MAXIMUM_RESUME_ATTEMPTS,
            "total_attempt_budget": MAXIMUM_TOTAL_ATTEMPTS,
            "result_based_tuning_allowed": False,
            "cross_v1_v2_v3_job_reuse_allowed": False,
        },
        "execution_start_sha256",
    )
    _publish_control_json_atomic(actual_dir / START_RECEIPT_NAME, value)
    attempts_dir = actual_dir / ATTEMPTS_DIRECTORY_NAME
    attempts_dir.mkdir(exist_ok=False)
    return value, str(value["execution_start_sha256"])


def _create_resume_attempt(
    actual_dir: Path,
    *,
    binding: ExecutionBindingV4,
    execution_start_sha256: str,
    existing_resume_count: int,
    namespace_audit: Mapping[str, Any],
    now: Callable[[], str],
) -> dict[str, Any]:
    if existing_resume_count >= MAXIMUM_RESUME_ATTEMPTS:
        raise ResumeBudgetExhausted(
            "v4 automatic resume budget exhausted after initial plus two resumes"
        )
    number = 2 + existing_resume_count
    attempts_dir = actual_dir / ATTEMPTS_DIRECTORY_NAME
    attempts_dir.mkdir(exist_ok=True)
    value = _self_hashed(
        {
            "schema_version": "p2_authoritative_resume_attempt.v4",
            "status": "RESUME_STARTED",
            "created_at_kst": now(),
            "attempt_number": number,
            "resume_attempt_number": number - 1,
            "remaining_resume_budget_after_start": MAXIMUM_TOTAL_ATTEMPTS - number,
            "execution_start_sha256": execution_start_sha256,
            "binding": binding.as_dict(),
            "read_only_namespace_audit": {
                "jobs_completed": namespace_audit["jobs"]["completed_jobs"],
                "cells_completed": namespace_audit["cells"]["completed_jobs"],
                "job_manifest_ledger_sha256": namespace_audit["jobs"][
                    "ordered_manifest_ledger_sha256"
                ],
                "cell_manifest_ledger_sha256": namespace_audit["cells"][
                    "ordered_manifest_ledger_sha256"
                ],
                "terminal_result_absent": True,
                "exclusive_lock_acquired_before_this_receipt": True,
            },
            "result_based_tuning_allowed": False,
        },
        "resume_attempt_sha256",
    )
    _publish_control_json_atomic(
        attempts_dir / f"resume_attempt_{number:03d}.json", value
    )
    return value


def _publish_terminal(
    actual_dir: Path,
    *,
    binding: ExecutionBindingV4,
    result: Mapping[str, Any],
    namespace_audit: Mapping[str, Any],
    now: Callable[[], str],
) -> dict[str, Any]:
    _require(result.get("status") == binding.expected_terminal_status, "v4 result status invalid")
    total_attempts = int(namespace_audit["total_attempts_started"])
    bound_result = dict(result)
    _require(
        bound_result.get("submission_files_generated") == 0
        and bound_result.get("uploads") == 0,
        "v4 execution result expanded submission scope",
    )
    bound_result.update(
        {
            "execution_binding_sha256": v2.canonical_sha256(binding.as_dict()),
            "preexecution_seal_sha256": binding.preexecution_seal_sha256,
            "semantic_preflight_sha256": binding.semantic_preflight_sha256,
            "initial_start_attempt_count": 1,
            "resume_attempts_started": total_attempts - 1,
            "total_attempts_started": total_attempts,
            "official_test_sample_submission_reads": 0,
        }
    )
    result_payload = _json_bytes(bound_result)
    result_publish = v2.atomic_write_or_verify(actual_dir / RESULT_NAME, result_payload)
    result_sha = hashlib.sha256(result_payload).hexdigest()
    _require(result_publish["sha256"] == result_sha, "v4 atomic result hash differs")
    receipt = _terminal_receipt_value(
        binding=binding,
        result_sha=result_sha,
        result_bytes=len(result_payload),
        result_publish=result_publish,
        total_attempts=total_attempts,
        now=now,
    )
    receipt_payload = _json_bytes(receipt)
    receipt_publish = v2.atomic_write_or_verify(
        actual_dir / TERMINAL_RECEIPT_NAME, receipt_payload
    )
    return {
        "status": "TERMINAL_COMPLETE_NO_RERUN",
        "result_sha256": result_sha,
        "result_bytes": len(result_payload),
        "terminal_receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "terminal_receipt_bytes": len(receipt_payload),
        "result_publish": result_publish,
        "terminal_receipt_publish": receipt_publish,
        "total_attempts_started": total_attempts,
    }


def _terminal_receipt_value(
    *,
    binding: ExecutionBindingV4,
    result_sha: str,
    result_bytes: int,
    result_publish: Mapping[str, Any],
    total_attempts: int,
    now: Callable[[], str],
) -> dict[str, Any]:
    return _self_hashed(
        {
            "schema_version": "p2_authoritative_terminal_receipt.v4",
            "status": "TERMINAL_COMPLETE_NO_RERUN",
            "completed_at_kst": now(),
            "binding": binding.as_dict(),
            "result_sha256": result_sha,
            "result_bytes": int(result_bytes),
            "result_atomic_publish": dict(result_publish),
            "initial_start_attempts": 1,
            "resume_attempts_started": total_attempts - 1,
            "total_attempts_started": total_attempts,
            "automatic_resume_budget_remaining": MAXIMUM_TOTAL_ATTEMPTS - total_attempts,
            "terminal_rerun_allowed": False,
        },
        "terminal_receipt_sha256",
    )


def _finalize_existing_terminal(
    actual_dir: Path,
    *,
    binding: ExecutionBindingV4,
    namespace_audit: Mapping[str, Any],
    now: Callable[[], str],
) -> dict[str, Any]:
    """Backfill receipts only; never call the science executor or spend a resume."""

    terminal = namespace_audit.get("terminal")
    _require(isinstance(terminal, dict), "verified terminal result is required")
    total_attempts = int(namespace_audit["total_attempts_started"])
    result_path = actual_dir / RESULT_NAME
    result_sha = v2.sha256_file(result_path)
    result_bytes = result_path.stat().st_size
    if not (actual_dir / TERMINAL_RECEIPT_NAME).exists():
        receipt = _terminal_receipt_value(
            binding=binding,
            result_sha=result_sha,
            result_bytes=result_bytes,
            result_publish={
                "status": "RECOVERED_VERIFIED_ATOMIC_FINAL",
                "sha256": result_sha,
                "bytes": result_bytes,
                "partial_created": False,
                "partial_policy": "STALE_PARTIALS_IGNORED_AND_PRESERVED_FOR_AUDIT",
            },
            total_attempts=total_attempts,
            now=now,
        )
        v2.atomic_write_or_verify(
            actual_dir / TERMINAL_RECEIPT_NAME, _json_bytes(receipt)
        )
    if namespace_audit.get("latest_attempt_terminal_status") != "COMPLETE_TERMINAL":
        _create_attempt_terminal(
            actual_dir,
            binding=binding,
            execution_start_sha256=str(namespace_audit["execution_start_sha256"]),
            attempt_number=total_attempts,
            status="COMPLETE_TERMINAL",
            now=now,
        )
    verified = inspect_actual_namespace_read_only(actual_dir, binding=binding)
    _require(
        verified["status"] == "TERMINAL_COMPLETE_NO_RERUN",
        "v4 finalization-only recovery did not close packaging",
    )
    terminal_receipt_path = actual_dir / TERMINAL_RECEIPT_NAME
    return {
        "status": "TERMINAL_COMPLETE_NO_RERUN",
        "finalization_only_recovery": True,
        "execute_curve_called": False,
        "resume_budget_consumed": False,
        "result_sha256": result_sha,
        "result_bytes": result_bytes,
        "terminal_receipt_sha256": v2.sha256_file(terminal_receipt_path),
        "total_attempts_started": total_attempts,
    }


def run_resumable_execution_v4(
    *,
    actual_dir: Path,
    binding: ExecutionBindingV4,
    semantic_preflight: Callable[[], SemanticPreflightOutcomeV4],
    execute_curve: Callable[[Any, str], Mapping[str, Any]],
    now: Callable[[], str] | None = None,
    before_terminal_publish: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a clean start or one same-command resume under the v4 contract."""

    binding.validate()
    actual_dir = actual_dir.resolve()
    _require(actual_dir.name == binding.namespace, "v4 actual directory namespace changed")
    clock = now or (lambda: datetime.now().astimezone().isoformat())

    if actual_dir.exists():
        # Advisory only: an active owner can atomically rename a partial between
        # enumeration and inspection.  Any advisory audit error is retried
        # strictly after acquiring the persistent lock; it never authorizes a
        # mutation or classifies the namespace by itself.
        try:
            preliminary = inspect_actual_namespace_read_only(actual_dir, binding=binding)
        except (OSError, ValueError, json.JSONDecodeError):
            preliminary = None
        if preliminary is not None:
            if preliminary["status"] == "TERMINAL_COMPLETE_NO_RERUN":
                raise TerminalExecutionClosed(
                    f"v4 terminal result already exists: {preliminary['terminal']['result_sha256']}"
                )
            if preliminary["status"] == "FAILED_DETERMINISTIC_CLOSED":
                raise DeterministicExecutionClosed(
                    "v4 namespace has a deterministic failure receipt; automatic resume is forbidden"
                )
            if (
                preliminary["status"]
                not in {"EMPTY_PRESTART_NAMESPACE", "TERMINAL_RESULT_NEEDS_FINALIZATION"}
                and not preliminary["automatic_resume_permitted"]
            ):
                raise ResumeBudgetExhausted(
                    "v4 automatic resume budget exhausted after initial plus two resumes"
                )
        clean_preflight: SemanticPreflightOutcomeV4 | None = None
    else:
        # The clean-start preflight is the last operation before namespace creation.
        clean_preflight = semantic_preflight()
        _require(
            clean_preflight.semantic_sha256 == binding.semantic_preflight_sha256,
            "clean-start v4 semantic preflight differs from seal",
        )
        try:
            actual_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # A racing invocation won namespace creation.  Do no mutation here;
            # the ordinary read-only resume path will reject or lock safely.
            return run_resumable_execution_v4(
                actual_dir=actual_dir,
                binding=binding,
                semantic_preflight=semantic_preflight,
                execute_curve=execute_curve,
                now=clock,
                before_terminal_publish=before_terminal_publish,
            )

    with v2.process_lock(actual_dir / LOCK_NAME):
        audit = inspect_actual_namespace_read_only(actual_dir, binding=binding)
        if audit["status"] == "TERMINAL_COMPLETE_NO_RERUN":
            raise TerminalExecutionClosed(
                f"v4 terminal result already exists: {audit['terminal']['result_sha256']}"
            )
        if audit["status"] == "TERMINAL_RESULT_NEEDS_FINALIZATION":
            return _finalize_existing_terminal(
                actual_dir, binding=binding, namespace_audit=audit, now=clock
            )
        if audit["status"] == "FAILED_DETERMINISTIC_CLOSED":
            raise DeterministicExecutionClosed(
                "v4 namespace has a deterministic failure receipt; automatic resume is forbidden"
            )
        if audit["status"] == "EMPTY_PRESTART_NAMESPACE":
            try:
                outcome = clean_preflight or semantic_preflight()
                _require(
                    outcome.semantic_sha256 == binding.semantic_preflight_sha256,
                    "v4 initial semantic preflight differs from seal",
                )
            except Exception as error:
                _create_gate_failure(
                    actual_dir,
                    binding=binding,
                    phase="prestart",
                    error=error,
                    now=clock,
                )
                raise DeterministicExecutionClosed(
                    "v4 prestart semantic gate failed closed"
                ) from error
            _, start_sha = _create_execution_start(
                actual_dir, binding=binding, now=clock
            )
            audit = inspect_actual_namespace_read_only(actual_dir, binding=binding)
            _require(audit["execution_start_sha256"] == start_sha, "v4 start publication failed")
            attempt_number = 1
        else:
            # Resume preflight happens under the already-acquired lock and before
            # the first resume mutation or model fit in this invocation.
            try:
                outcome = semantic_preflight()
                _require(
                    outcome.semantic_sha256 == binding.semantic_preflight_sha256,
                    "v4 resume semantic preflight differs from seal",
                )
            except Exception as error:
                _create_gate_failure(
                    actual_dir,
                    binding=binding,
                    phase="resume",
                    error=error,
                    now=clock,
                )
                raise DeterministicExecutionClosed(
                    "v4 resume semantic gate failed closed"
                ) from error
            if int(audit["resume_attempts_started"]) >= MAXIMUM_RESUME_ATTEMPTS:
                raise ResumeBudgetExhausted(
                    "v4 automatic resume budget exhausted after initial plus two resumes"
                )
            _create_resume_attempt(
                actual_dir,
                binding=binding,
                execution_start_sha256=str(audit["execution_start_sha256"]),
                existing_resume_count=int(audit["resume_attempts_started"]),
                namespace_audit=audit,
                now=clock,
            )
            audit = inspect_actual_namespace_read_only(actual_dir, binding=binding)
            attempt_number = int(audit["total_attempts_started"])
        try:
            result = dict(
                execute_curve(outcome.execution_context, binding.preexecution_seal_sha256)
            )
            _require(
                result.get("status") == binding.expected_terminal_status,
                "v4 execute callback returned a nonterminal result",
            )
            if before_terminal_publish is not None:
                before_terminal_publish(actual_dir, result)
            final_audit = inspect_actual_namespace_read_only(actual_dir, binding=binding)
            _require(final_audit["terminal"] is None, "v4 result appeared before publication")
            terminal_summary = _publish_terminal(
                actual_dir,
                binding=binding,
                result=result,
                namespace_audit=final_audit,
                now=clock,
            )
        except TransientExecutionError as error:
            _create_attempt_terminal(
                actual_dir,
                binding=binding,
                execution_start_sha256=str(audit["execution_start_sha256"]),
                attempt_number=attempt_number,
                status="FAILED_TRANSIENT_RESUMABLE",
                now=clock,
                error=error,
            )
            raise
        except Exception as error:
            # If result.json already committed atomically, the namespace is
            # terminal even if a later receipt write failed.  Record success
            # when possible and never authorize a second scientific run.
            if (actual_dir / RESULT_NAME).is_file():
                _create_attempt_terminal(
                    actual_dir,
                    binding=binding,
                    execution_start_sha256=str(audit["execution_start_sha256"]),
                    attempt_number=attempt_number,
                    status="COMPLETE_TERMINAL",
                    now=clock,
                )
            else:
                _create_attempt_terminal(
                    actual_dir,
                    binding=binding,
                    execution_start_sha256=str(audit["execution_start_sha256"]),
                    attempt_number=attempt_number,
                    status="FAILED_DETERMINISTIC_CLOSED",
                    now=clock,
                    error=error,
                )
            raise
        else:
            _create_attempt_terminal(
                actual_dir,
                binding=binding,
                execution_start_sha256=str(audit["execution_start_sha256"]),
                attempt_number=attempt_number,
                status="COMPLETE_TERMINAL",
                now=clock,
            )
            return terminal_summary
        finally:
            # The surrounding process_lock context releases the OS lock.  No
            # lock file, partial, job, or receipt is deleted here.
            pass
