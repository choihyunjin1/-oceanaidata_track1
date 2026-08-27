"""Capability and static-integrity contract for P1 Gen6r3.

This module is compiled only from the byte buffer authenticated by the Gen6r3
bootstrap.  It deliberately does not expose a normal-import execution route.
The frozen Gen6r2 science is reused byte-for-byte; only the execution protocol
and trust boundary change.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

try:
    _CONTEXT = _P1_V6R3_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r3 contract requires the authenticated bootstrap") from exc

if not isinstance(_CONTEXT, dict) or _CONTEXT.get("all_owner_roles_authenticated") is not True:
    raise RuntimeError("P1 Gen6r3 contract loaded before source authentication")

GENERATION = "p1_multiscale_cross_layer_offset_drift_unary_v6r3"
FOLDS = ("2025_q2", "2025_q3", "2025_q4")
FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
FRACTION_TAGS = ("p040", "p055", "p070", "p085", "p100")
BLOCKS = (1, 2, 3)
SEEDS = (20260813, 20260829, 20260847)
CELL_COUNT = 15
INNER_COMMITMENTS = 45
TEACHER_RECEIPTS = 135
SCORE_DECOMPOSITION = {
    "inner_blocks": 45,
    "inner_gate_aggregates": 15,
    "fraction_aggregates": 5,
    "fold_aggregates": 3,
    "total": 68,
}
V9_PIN = {
    "path": "artifacts/meaningful_score_goal_v9/registry.jsonl",
    "bytes": 15812,
    "sha256": "232b6ed3133de11ee05150ec439efe05baa315bbb64ea0f319ffcbddd421b965",
}
V9_HEAD = "1b3e01be70c6f8ed2df04038deac3b3642804f70f9f17a238826c64d68090317"


class ContractError(RuntimeError):
    """A static, authorization, ordering, or integrity invariant failed."""


class CapabilityError(PermissionError):
    """An entry point was called without the exact live process capability."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def deep_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    if type(value) is not bytes:
        raise TypeError("bytes_sha256 requires exact bytes")
    return hashlib.sha256(value).hexdigest()


def _is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def parse_json_bytes(payload: bytes, *, label: str) -> Any:
    if type(payload) is not bytes:
        raise ContractError(f"{label} is not an authenticated byte buffer")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ContractError(f"{label} contains non-finite JSON constant {value}")

    try:
        text = payload.decode("utf-8", errors="strict")
        return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not strict UTF-8 JSON") from exc


def _relative_parts(relative: str) -> tuple[str, ...]:
    if type(relative) is not str or not relative or "\\" in relative:
        raise ContractError("relative path must be nonempty canonical POSIX text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError(f"unsafe relative path: {relative!r}")
    return pure.parts


def _has_reparse(path: Path) -> bool:
    info = path.lstat()
    return bool(int(getattr(info, "st_file_attributes", 0)) & 0x400)


def contained_path(
    root: Path,
    relative: str,
    *,
    must_exist: bool,
    kind: str | None = None,
) -> Path:
    root = root.resolve(strict=True)
    if _has_reparse(root):
        raise ContractError("workspace root is a reparse point")
    current = root
    parts = _relative_parts(relative)
    for index, part in enumerate(parts):
        current = current / part
        exists = os.path.lexists(current)
        if exists and _has_reparse(current):
            raise ContractError(f"reparse point forbidden in path: {relative}")
        if not exists and (must_exist or index < len(parts) - 1):
            raise ContractError(f"required path is absent: {relative}")
    if must_exist and not current.exists():
        raise ContractError(f"required path is absent: {relative}")
    if current.exists():
        resolved = current.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"path escapes workspace: {relative}") from exc
        info = current.lstat()
        if kind == "file" and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise ContractError(f"file is not a unique plain file: {relative}")
        if kind == "directory" and not stat.S_ISDIR(info.st_mode):
            raise ContractError(f"path is not a directory: {relative}")
    return current


def _authenticated_bytes(pin: dict[str, Any], label: str) -> bytes:
    reader = _CONTEXT.get("authenticated_bytes_for_pin")
    if not callable(reader):
        raise ContractError("authenticated byte reader is unavailable")
    if (
        type(pin) is not dict
        or set(pin) != {"path", "bytes", "sha256"}
        or type(pin["path"]) is not str
        or type(pin["bytes"]) is not int
        or pin["bytes"] < 0
        or not _is_sha(pin["sha256"])
    ):
        raise ContractError(f"{label} pin domain differs")
    payload = reader(pin, label)
    if type(payload) is not bytes or len(payload) != pin["bytes"] or bytes_sha256(payload) != pin["sha256"]:
        raise ContractError(f"{label} authenticated bytes differ")
    return payload


def _authenticated_json(pin: dict[str, Any], label: str) -> Any:
    return parse_json_bytes(_authenticated_bytes(pin, label), label=label)


def cell_identity(cell: int) -> tuple[str, float, str]:
    if type(cell) is not int or not 1 <= cell <= CELL_COUNT:
        raise ContractError("cell must be an exact integer from 1 through 15")
    fold_index, fraction_index = divmod(cell - 1, len(FRACTIONS))
    return FOLDS[fold_index], FRACTIONS[fraction_index], FRACTION_TAGS[fraction_index]


def expected_teacher_keys() -> set[tuple[str, str, int, int]]:
    return {
        (fold, fraction_tag, block, seed)
        for fold in FOLDS
        for fraction_tag in FRACTION_TAGS
        for block in BLOCKS
        for seed in SEEDS
    }


def verify_teacher_receipt_catalog(
    predictions_complete: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    if type(predictions_complete) is not dict or type(predictions_complete.get("teacher_model_receipts")) is not list:
        raise ContractError("Gen5r6 predictions_complete teacher receipts are absent")
    receipts = predictions_complete["teacher_model_receipts"]
    if len(receipts) != TEACHER_RECEIPTS:
        raise ContractError("Gen5r6 teacher receipt count differs")
    artifacts = manifest.get("artifacts") if type(manifest) is dict else None
    if type(artifacts) is not dict:
        raise ContractError("Gen5r6 manifest artifact map is absent")
    catalog: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for receipt in receipts:
        if type(receipt) is not dict:
            raise ContractError("Gen5r6 teacher receipt is not an object")
        required = {
            "fold",
            "fraction_tag",
            "block",
            "seed",
            "role",
            "scope",
            "blind_prediction_relative_path",
            "blind_prediction_sha256",
            "prediction_ids_sha256",
            "train_ids_sha256",
            "prediction_rows",
            "train_rows",
            "teacher_fit_and_prediction_rows_disjoint",
            "outer_validation_rows_touched",
            "test_value_reads",
            "saved_model_reload_prediction_exact",
        }
        if not required.issubset(receipt):
            raise ContractError("Gen5r6 teacher receipt fields differ")
        key = (receipt["fold"], receipt["fraction_tag"], receipt["block"], receipt["seed"])
        if key in catalog:
            raise ContractError("duplicate Gen5r6 teacher receipt identity")
        if key not in expected_teacher_keys():
            raise ContractError("unexpected Gen5r6 teacher receipt identity")
        expected_path = (
            f"teacher_blind_predictions/curve/{key[1]}/{key[0]}/"
            f"block_{key[2]}/seed_{key[3]}.npy"
        )
        artifact = artifacts.get(expected_path)
        if (
            receipt["role"] != "inner_teacher"
            or receipt["scope"] != "curve"
            or receipt["blind_prediction_relative_path"] != expected_path
            or not _is_sha(receipt["blind_prediction_sha256"])
            or not _is_sha(receipt["prediction_ids_sha256"])
            or not _is_sha(receipt["train_ids_sha256"])
            or type(receipt["prediction_rows"]) is not int
            or receipt["prediction_rows"] <= 0
            or type(receipt["train_rows"]) is not int
            or receipt["train_rows"] <= 0
            or receipt["teacher_fit_and_prediction_rows_disjoint"] is not True
            or receipt["outer_validation_rows_touched"] != 0
            or receipt["test_value_reads"] != 0
            or receipt["saved_model_reload_prediction_exact"] is not True
            or type(artifact) is not dict
            or set(artifact) != {"bytes", "sha256"}
            or artifact["sha256"] != receipt["blind_prediction_sha256"]
        ):
            raise ContractError(f"Gen5r6 teacher receipt binding differs: {key}")
        catalog[key] = receipt
    if set(catalog) != expected_teacher_keys():
        raise ContractError("Gen5r6 teacher receipt identity set differs")
    return catalog


def verify_teacher_request(
    catalog: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    fold: str,
    fraction_tag: str,
    block: int,
    seed: int,
    prediction_ids_sha256: str,
    train_ids_sha256: str,
    prediction_rows: int,
    train_rows: int,
) -> dict[str, Any]:
    receipt = catalog.get((fold, fraction_tag, block, seed))
    if receipt is None:
        raise ContractError("teacher receipt identity is absent")
    if (
        receipt["prediction_ids_sha256"] != prediction_ids_sha256
        or receipt["train_ids_sha256"] != train_ids_sha256
        or receipt["prediction_rows"] != prediction_rows
        or receipt["train_rows"] != train_rows
    ):
        raise ContractError("teacher receipt row-identity binding differs")
    return receipt


def _verify_v9() -> dict[str, Any]:
    payload = _authenticated_bytes(V9_PIN, "meaningful score registry v9")
    lines = payload.splitlines()
    if len(lines) != 3:
        raise ContractError("v9 event count differs")
    events = [parse_json_bytes(line, label=f"v9 event {index}") for index, line in enumerate(lines, 3)]
    if [event.get("seq") for event in events] != [3, 4, 5]:
        raise ContractError("v9 sequence differs")
    previous = "2f4a16abb2213ed0e517967ae5782dfbcb5ab1b1bd1f08f9e5852cfba15c4c20"
    for event in events:
        if event.get("previous_event_sha256") != previous or not _is_sha(event.get("event_sha256")):
            raise ContractError("v9 chain differs")
        claimed = event["event_sha256"]
        body = dict(event)
        body.pop("event_sha256")
        if deep_sha256(body) != claimed:
            raise ContractError("v9 event hash differs")
        previous = claimed
    if previous != V9_HEAD or any(event.get("payload", {}).get("upload_performed") is True for event in events):
        raise ContractError("v9 head or upload state differs")
    return {"events": 3, "head_seq": 5, "head_event_sha256": previous, "official_uploads": 0}


def _verify_lineage(config: dict[str, Any]) -> dict[str, Any]:
    lineage = config.get("frozen_v6r2_lineage")
    if type(lineage) is not dict or lineage.get("must_remain_byte_exact") is not True:
        raise ContractError("frozen Gen6r2 lineage declaration differs")
    roles = lineage.get("roles")
    if type(roles) is not dict or set(roles) != {
        "CONFIG",
        "SCIENCE_PROJECTION",
        "BOOTSTRAP",
        "CONTRACT",
        "ENGINE",
        "SCIENCE",
        "SELECTIVE_TARGET_HELPER",
        "RUNNER",
        "TESTS",
    }:
        raise ContractError("frozen Gen6r2 role set differs")
    return {name: {"bytes": len(_authenticated_bytes(pin, f"frozen v6r2 {name}")), "sha256": pin["sha256"]} for name, pin in roles.items()}


def _verify_disposition(config: dict[str, Any]) -> dict[str, Any]:
    disposition = config.get("v6r2_no_go_disposition")
    if type(disposition) is not dict or set(disposition) != {"owner_no_go", "execution_tombstone"}:
        raise ContractError("Gen6r2 disposition pins differ")
    owner = _authenticated_json(disposition["owner_no_go"], "Gen6r2 owner NO-GO")
    tombstone = _authenticated_json(disposition["execution_tombstone"], "Gen6r2 tombstone")
    codes = [item.get("code") for item in owner.get("findings", [])]
    expected = [
        "CROSS_FRACTION_TARGET_VAULT_CONTAMINATION",
        "AUTHENTICATED_NATIVE_BYTES_NOT_BOUND_TO_LOADED_BYTES",
        "SCORE_CALL_ACCOUNTING_53_VS_ACTUAL_68",
        "DIRECT_WINAPI_AND_PYC_MUTATION_FIREWALL_BYPASS",
        "GEN5R6_TEACHER_RECEIPT_PREDICTION_IDS_NOT_ENFORCED",
    ]
    if (
        owner.get("reviewer") != "/root/p1_gen6r2_independent_qa"
        or owner.get("verdict") != "P0=2_P1=3_NO_GO"
        or owner.get("p0_count") != 2
        or owner.get("p1_count") != 3
        or owner.get("independent_qa_receipt", {}).get("present") is not False
        or codes != expected
        or tombstone.get("status") != "PERMANENTLY_TOMBSTONED_NEVER_EXECUTE"
        or tombstone.get("owner_no_go_receipt") != disposition["owner_no_go"]
    ):
        raise ContractError("Gen6r2 NO-GO semantics differ")
    return {"verdict": owner["verdict"], "qa_receipt_present": False, "finding_codes": codes}


def _verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    protocol = config.get("process_isolation_protocol")
    if type(protocol) is not dict:
        raise ContractError("process-isolation protocol is absent")
    required = {
        "workers": 15,
        "sequential_windows_spawn": True,
        "one_cell_per_worker": True,
        "fresh_empty_target_vault_per_worker": True,
        "parent_never_receives_inner_labels": True,
        "worker_exits_after_cell_receipt": True,
        "persisted_o_excl_chain_replaces_in_memory_order": True,
        "fold_commit_before_parent_outer_decode": True,
    }
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ContractError("process-isolation protocol semantics differ")
    score = config.get("score_accounting")
    if score != SCORE_DECOMPOSITION:
        raise ContractError("exact 68-call score decomposition differs")
    return {"workers": CELL_COUNT, "score_calls": SCORE_DECOMPOSITION["total"]}


def _verify_owner_roles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    roles = config.get("owner_roles")
    expected = {"STARTUP_TRUST", "CONTRACT", "ENGINE", "VERIFIER", "RUNNER", "TESTS"}
    if type(roles) is not dict or set(roles) != expected:
        raise ContractError("Gen6r3 owner role set differs")
    observed: dict[str, dict[str, Any]] = {}
    for role in sorted(expected):
        payload = _authenticated_bytes(roles[role], f"Gen6r3 owner role {role}")
        observed[role] = {"bytes": len(payload), "sha256": bytes_sha256(payload)}
    return observed


def _verify_future_state(config: dict[str, Any]) -> dict[str, bool]:
    workspace = Path(_CONTEXT["workspace"])
    future = config.get("future_state_must_be_absent")
    if type(future) is not list or not future or any(type(item) is not str for item in future):
        raise ContractError("future-state absence list differs")
    observed = {relative: os.path.lexists(contained_path(workspace, relative, must_exist=False)) for relative in future}
    if any(observed.values()):
        present = [path for path, exists in observed.items() if exists]
        raise ContractError(f"future Gen6r3 state already exists: {present}")
    return observed


def static_preflight(*, require_future_state_absent: bool = True) -> dict[str, Any]:
    if type(require_future_state_absent) is not bool:
        raise ContractError("future-state preflight selector must be exact bool")
    config = _CONTEXT.get("config")
    base = _CONTEXT.get("base_config")
    if type(config) is not dict or type(base) is not dict:
        raise ContractError("authenticated configuration is unavailable")
    lineage = _verify_lineage(config)
    owner_roles = _verify_owner_roles(config)
    disposition = _verify_disposition(config)
    protocol = _verify_protocol(config)
    teacher = config.get("teacher_receipt_binding")
    if type(teacher) is not dict or teacher.get("receipt_count") != TEACHER_RECEIPTS:
        raise ContractError("teacher receipt configuration differs")
    predictions_complete = _authenticated_json(
        teacher["gen5r6_predictions_complete"], "Gen5r6 predictions_complete"
    )
    manifest = _authenticated_json(base["inner_incumbent_binding"]["gen5r6_manifest"], "Gen5r6 manifest")
    catalog = verify_teacher_receipt_catalog(predictions_complete, manifest)
    _CONTEXT["teacher_receipt_catalog"] = catalog
    v9 = _verify_v9()
    future = _verify_future_state(config) if require_future_state_absent else {}
    return {
        "schema_version": "p1_multiscale_cross_layer_offset_drift_unary.v6r3.static_preflight.v1",
        "generation": GENERATION,
        "lineage_roles": lineage,
        "owner_roles": owner_roles,
        "v6r2_disposition": disposition,
        "protocol": protocol,
        "teacher_receipts": len(catalog),
        "v9": v9,
        "future_state_absent": None if not require_future_state_absent else not any(future.values()),
        "execution_authorized": False,
        "actual_run_performed": False,
    }


def require_engine_capability(capability: object, entry_name: str) -> Any:
    """Delegate to the bootstrap-owned, non-exported capability registry.

    There is intentionally no mint in this module or in ``_CONTEXT``.  The
    bootstrap constructs the opaque object only after live QA, authorization,
    lock, argv, session, and (for workers) exact cell-chain verification.
    """

    guard = _CONTEXT.get("capability_guard")
    if not callable(guard) or type(entry_name) is not str or not entry_name:
        raise CapabilityError(f"live Gen6r3 capability required for {entry_name}")
    live = guard(capability, entry_name)
    if live is not capability:
        raise CapabilityError(f"bootstrap capability guard rejected {entry_name}")
    return live


def bump_counter(capability: object, name: str, amount: int = 1) -> int:
    require_engine_capability(capability, "bump_counter")
    if type(name) is not str or not name or type(amount) is not int or amount < 0:
        raise CapabilityError("counter mutation domain differs")
    counter = _CONTEXT.get("capability_counter_bump")
    if not callable(counter):
        raise CapabilityError("bootstrap counter closure is unavailable")
    value = counter(capability, name, amount)
    if type(value) is not int or value < amount:
        raise CapabilityError("bootstrap counter result differs")
    return value


def close_capability(capability: object) -> dict[str, int]:
    require_engine_capability(capability, "close_capability")
    closer = _CONTEXT.get("capability_close")
    if not callable(closer):
        raise CapabilityError("bootstrap capability closer is unavailable")
    counters = closer(capability)
    if type(counters) is not dict or any(type(key) is not str or type(value) is not int for key, value in counters.items()):
        raise CapabilityError("bootstrap capability close result differs")
    return counters


def capability_snapshot(capability: object) -> dict[str, Any]:
    require_engine_capability(capability, "capability_snapshot")
    snapshot = _CONTEXT.get("capability_snapshot")
    if not callable(snapshot):
        raise CapabilityError("bootstrap capability snapshot is unavailable")
    value = snapshot(capability)
    if (
        type(value) is not dict
        or value.get("role") not in {"parent", "cell_worker"}
        or not _is_sha(value.get("session_sha256"))
        or type(value.get("counters")) is not dict
    ):
        raise CapabilityError("bootstrap capability snapshot differs")
    return value


def _allowed_worker_path(cell: int, relative: str) -> bool:
    prefix = f"cells/cell_{cell:02d}/"
    if not relative.startswith(prefix):
        return False
    tail = relative[len(prefix) :]
    allowed = {
        *(f"models/inner_{block}.json" for block in BLOCKS),
        "models/outer.json",
        *(f"inner_predictions/block_{block}.bin" for block in BLOCKS),
        *(f"commitments/inner_{block}.json" for block in BLOCKS),
        "outer_prediction.bin",
        "cell_receipt.json",
    }
    return tail in allowed


def _allowed_parent_path(relative: str) -> bool:
    return (
        relative == "commitments/session.json"
        or relative == "commitments/predictions_complete.json"
        or relative in {f"commitments/fold_{fold}.json" for fold in FOLDS}
        or relative in {f"metrics/fold_{fold}.json" for fold in FOLDS}
        or relative in {f"metrics/fraction_{tag}.json" for tag in FRACTION_TAGS}
        or relative in {
            "split_audit.json",
            "selective_target_audit.json",
            "metrics.json",
            "learning_curve_evidence.json",
            "result.json",
            "resource_audit.json",
            "manifest.json",
            "manifest.sha256",
            "final_seal.json",
        }
    )


def write_output_exclusive(capability: object, relative: str, value: Any) -> dict[str, Any]:
    require_engine_capability(capability, "write_output_exclusive")
    live = capability_snapshot(capability)
    _relative_parts(relative)
    allowed = (
        live["role"] == "cell_worker"
        and live.get("cell") is not None
        and _allowed_worker_path(live["cell"], relative)
    ) or (live["role"] == "parent" and _allowed_parent_path(relative))
    if not allowed:
        raise CapabilityError(f"output path is outside {live['role']} capability: {relative}")
    payload = value if type(value) is bytes else canonical_json_bytes(value) + b"\n"
    writer: Callable[[object, str, bytes], dict[str, Any]] | None = _CONTEXT.get(
        "exclusive_output_writer"
    )
    if not callable(writer):
        raise ContractError("exclusive output writer is unavailable")
    # The bootstrap writer re-verifies live QA/auth/lock/session/cell/chain on
    # every call before temporarily opening its otherwise-closed write scope.
    pin = writer(capability, relative, payload)
    expected = {"path": relative, "bytes": len(payload), "sha256": bytes_sha256(payload)}
    if pin != expected:
        raise ContractError("exclusive output writer pin differs")
    bump_counter(capability, "files_written")
    return pin


def exact_completion_counters() -> dict[str, int]:
    return {
        "worker_processes": 15,
        "baseline_fits": 60,
        "unary_fits": 60,
        "top_level_fits": 120,
        "predictions": 60,
        "inner_commitments": 45,
        "cell_commitments": 15,
        "fold_commitments": 3,
        "predictions_complete": 1,
        "bootstrap_replicates": 25_000,
        "scores": 68,
        "test_value_reads": 0,
        "candidate_files": 0,
        "ledger_appends": 0,
        "uploads": 0,
    }


def verify_completion_counters(observed: dict[str, Any]) -> dict[str, int]:
    expected = exact_completion_counters()
    if type(observed) is not dict or any(type(observed.get(key)) is not int or observed[key] != value for key, value in expected.items()):
        raise ContractError("Gen6r3 exact completion counters differ")
    return expected
