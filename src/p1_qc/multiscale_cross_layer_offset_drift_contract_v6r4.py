"""Append-only static contract for P1 multiscale Gen6r4.

Gen6r4 keeps the frozen Gen6r2 scientific projection and replaces the Gen6r3
Python target-vault convention with an inherited-handle target broker.  This
module contains no file, process, target, or scoring authority.  The external
bootstrap compiles these authenticated bytes and injects read-only callbacks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

try:
    _CONTEXT = _P1_V6R4_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r4 contract requires the authenticated bootstrap") from exc

if not isinstance(_CONTEXT, dict) or _CONTEXT.get("all_owner_roles_authenticated") is not True:
    raise RuntimeError("P1 Gen6r4 contract loaded before source authentication")

GENERATION = "p1_multiscale_cross_layer_offset_drift_unary_v6r4"
FOLDS = ("2025_q2", "2025_q3", "2025_q4")
FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
FRACTION_TAGS = ("p040", "p055", "p070", "p085", "p100")
BLOCKS = (1, 2, 3)
SEEDS = (20260813, 20260814, 20260815)
CELL_COUNT = 15
FILES_PER_CELL = 12
OUTPUT_FILES = 202
TEACHER_RECEIPTS = 135
GENESIS_SHA256 = hashlib.sha256(b"p1_v6r4_brokered_commitment_genesis").hexdigest()

SCORE_DECOMPOSITION = {
    "inner_blocks": 45,
    "inner_gate_aggregates": 15,
    "fraction_aggregates": 5,
    "fold_aggregates": 3,
    "total": 68,
}

EXACT_COMPLETION_COUNTERS = {
    "broker_processes": 1,
    "worker_processes": 15,
    "sanitized_cell_buffers": 15,
    "teacher_receipts_consumed": 135,
    "baseline_fits": 60,
    "unary_fits": 60,
    "top_level_fits": 120,
    "predictions": 60,
    "inner_commitments": 45,
    "cell_commitments": 15,
    "fold_commitments": 3,
    "predictions_complete": 1,
    "bootstrap_replicates": 25000,
    "scores": 68,
    "output_files": 202,
    "test_value_reads": 0,
    "candidate_files": 0,
    "ledger_appends": 0,
    "uploads": 0,
}

PHASES = (
    "inner_1_train",
    "inner_1_gate",
    "inner_2_train",
    "inner_2_gate",
    "inner_3_train",
    "inner_3_gate",
    "outer_train",
    "outer_seal",
)


class ContractError(RuntimeError):
    """A frozen identity or semantic contract differs."""


class CapabilityError(PermissionError):
    """An inherited private capability is absent, stale, or out of scope."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def bytes_sha256(value: bytes) -> str:
    if type(value) is not bytes:
        raise ContractError("SHA-256 input is not exact bytes")
    return hashlib.sha256(value).hexdigest()


def deep_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json_bytes(value))


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def strict_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be a non-bool integer >= {minimum}")
    return value


def parse_json_bytes(payload: bytes, *, label: str) -> Any:
    if type(payload) is not bytes:
        raise ContractError(f"{label} is not exact bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ContractError(f"{label} contains non-finite constant {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not strict UTF-8 JSON") from exc


def require_exact_keys(value: object, keys: Iterable[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ContractError(f"{label} is not an object")
    expected = set(keys)
    if set(value) != expected:
        raise ContractError(f"{label} keys differ: {sorted(set(value) ^ expected)}")
    return value


def validate_pin(value: object, *, label: str, expected_path: str | None = None) -> dict[str, Any]:
    pin = require_exact_keys(value, ("path", "bytes", "sha256"), label=label)
    path = pin["path"]
    if (
        type(path) is not str
        or not path
        or path.startswith(("/", "\\"))
        or "\\" in path
        or ".." in path.split("/")
        or (expected_path is not None and path != expected_path)
    ):
        raise ContractError(f"{label} path differs")
    strict_int(pin["bytes"], label=f"{label}.bytes")
    if not is_sha256(pin["sha256"]):
        raise ContractError(f"{label}.sha256 differs")
    return pin


def cell_identity(cell: object) -> tuple[str, float, str]:
    number = strict_int(cell, label="cell", minimum=1)
    if number > CELL_COUNT:
        raise ContractError("cell exceeds frozen cell count")
    fold_index, fraction_index = divmod(number - 1, len(FRACTIONS))
    return FOLDS[fold_index], FRACTIONS[fraction_index], FRACTION_TAGS[fraction_index]


def expected_teacher_keys() -> set[tuple[str, str, int, int]]:
    return {
        (fold, fraction, block, seed)
        for fold in FOLDS
        for fraction in FRACTION_TAGS
        for block in BLOCKS
        for seed in SEEDS
    }


def verify_teacher_receipt_catalog(
    predictions_complete: object,
    manifest: object,
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    complete = require_exact_keys(
        predictions_complete,
        ("schema_version", "generation", "receipts", "receipt_count"),
        label="teacher predictions_complete",
    )
    strict_int(complete["receipt_count"], label="teacher receipt_count")
    receipts = complete["receipts"]
    if type(receipts) is not list or len(receipts) != TEACHER_RECEIPTS:
        raise ContractError("teacher receipt count differs")
    if complete["receipt_count"] != len(receipts):
        raise ContractError("teacher receipt_count field differs")
    manifest_object = require_exact_keys(
        manifest,
        ("schema_version", "generation", "files"),
        label="teacher manifest",
    )
    files = manifest_object["files"]
    if type(files) is not list:
        raise ContractError("teacher manifest files differ")
    manifest_pins: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(files):
        pin = validate_pin(item, label=f"teacher manifest file {index}")
        if pin["path"] in manifest_pins:
            raise ContractError("teacher manifest path repeats")
        manifest_pins[pin["path"]] = pin
    result: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for index, raw in enumerate(receipts):
        receipt = require_exact_keys(
            raw,
            (
                "fold",
                "fraction_tag",
                "block",
                "seed",
                "blind_prediction_relative_path",
                "blind_prediction_sha256",
                "prediction_ids_sha256",
                "train_ids_sha256",
                "prediction_rows",
                "train_rows",
            ),
            label=f"teacher receipt {index}",
        )
        key = (
            receipt["fold"],
            receipt["fraction_tag"],
            strict_int(receipt["block"], label="teacher block", minimum=1),
            strict_int(receipt["seed"], label="teacher seed", minimum=1),
        )
        if key in result or key not in expected_teacher_keys():
            raise ContractError("teacher receipt identity differs")
        for name in ("blind_prediction_sha256", "prediction_ids_sha256", "train_ids_sha256"):
            if not is_sha256(receipt[name]):
                raise ContractError(f"teacher receipt {name} differs")
        strict_int(receipt["prediction_rows"], label="teacher prediction_rows", minimum=1)
        strict_int(receipt["train_rows"], label="teacher train_rows", minimum=1)
        path = receipt["blind_prediction_relative_path"]
        pin = manifest_pins.get(path)
        if pin is None or pin["sha256"] != receipt["blind_prediction_sha256"]:
            raise ContractError("teacher receipt prediction pin differs from manifest")
        result[key] = receipt
    if set(result) != expected_teacher_keys():
        raise ContractError("teacher receipt identity product is incomplete")
    return result


def row_ids_sha256(values: object, *, size: int, label: str) -> str:
    if type(values) is not list or not values:
        raise ContractError(f"{label} row IDs are not a non-empty list")
    prior = -1
    canonical: list[int] = []
    for index, raw in enumerate(values):
        item = strict_int(raw, label=f"{label}[{index}]")
        if item >= size or item <= prior:
            raise ContractError(f"{label} row IDs are not sorted, unique, and in range")
        prior = item
        canonical.append(item)
    return deep_sha256(canonical)


def verify_disjoint(left: list[int], right: list[int], *, label: str) -> None:
    if set(left) & set(right):
        raise ContractError(f"{label} row scopes overlap")


def event_body(schema: str, prior: str, fields: dict[str, Any]) -> dict[str, Any]:
    if type(schema) is not str or not schema or not is_sha256(prior) or type(fields) is not dict:
        raise ContractError("event construction arguments differ")
    body = {
        "schema_version": schema,
        "generation": GENERATION,
        "prior_event_sha256": prior,
        **fields,
    }
    body["event_sha256"] = deep_sha256(body)
    return body


def verify_event(value: object, *, schema: str, prior: str, label: str) -> str:
    if type(value) is not dict:
        raise ContractError(f"{label} event is not an object")
    if (
        value.get("schema_version") != schema
        or value.get("generation") != GENERATION
        or value.get("prior_event_sha256") != prior
        or not is_sha256(value.get("event_sha256"))
    ):
        raise ContractError(f"{label} event identity differs")
    body = dict(value)
    claimed = body.pop("event_sha256")
    if deep_sha256(body) != claimed:
        raise ContractError(f"{label} event hash differs")
    return claimed


def validate_launch_envelope(
    envelope: object,
    *,
    role: str,
    inherited_handle_identity: str,
    challenge_sha256: str,
) -> dict[str, Any]:
    value = require_exact_keys(
        envelope,
        (
            "schema_version",
            "generation",
            "role",
            "cell",
            "session_sha256",
            "inherited_handle_identity",
            "challenge_sha256",
            "encoded_command_sha256",
            "launcher_sha256",
            "public_cli_fields_absent",
        ),
        label="private launch envelope",
    )
    if (
        value["schema_version"] != "p1_v6r4_private_inherited_launch.v1"
        or value["generation"] != GENERATION
        or value["role"] != role
        or value["inherited_handle_identity"] != inherited_handle_identity
        or value["challenge_sha256"] != challenge_sha256
        or value["public_cli_fields_absent"] is not True
        or not is_sha256(value["session_sha256"])
        or not is_sha256(value["encoded_command_sha256"])
        or not is_sha256(value["launcher_sha256"])
    ):
        raise CapabilityError("private inherited launch provenance differs")
    if role == "cell_worker":
        cell_identity(value["cell"])
    elif role in {"parent", "target_broker"}:
        if value["cell"] is not None:
            raise CapabilityError("non-worker launch unexpectedly names a cell")
    else:
        raise CapabilityError("private launch role differs")
    return value


def expected_cell_paths(cell: object) -> tuple[str, ...]:
    number = strict_int(cell, label="cell", minimum=1)
    cell_identity(number)
    prefix = f"cells/cell_{number:02d}"
    paths: list[str] = []
    for block in BLOCKS:
        paths.extend(
            (
                f"{prefix}/models/inner_{block}.json",
                f"{prefix}/inner_predictions/block_{block}.bin",
                f"{prefix}/commitments/inner_{block}.json",
            )
        )
    paths.extend(
        (
            f"{prefix}/models/outer.json",
            f"{prefix}/outer_prediction.bin",
            f"{prefix}/cell_receipt.json",
        )
    )
    if len(paths) != FILES_PER_CELL or len(set(paths)) != FILES_PER_CELL:
        raise ContractError("cell path arithmetic differs")
    return tuple(paths)


def expected_parent_paths() -> tuple[str, ...]:
    paths: list[str] = ["commitments/session.json"]
    for fold in FOLDS:
        paths.extend((f"commitments/fold_{fold}.json", f"metrics/fold_{fold}.json"))
    paths.append("commitments/predictions_complete.json")
    paths.extend(f"metrics/fraction_{tag}.json" for tag in FRACTION_TAGS)
    paths.extend(
        (
            "split_audit.json",
            "target_broker_audit.json",
            "metrics.json",
            "learning_curve_evidence.json",
            "result.json",
            "manifest.json",
            "manifest.sha256",
            "preseal.json",
            "final_seal.json",
        )
    )
    if len(paths) != 22 or len(set(paths)) != 22:
        raise ContractError("parent path arithmetic differs")
    return tuple(paths)


def expected_output_paths() -> tuple[str, ...]:
    paths = [
        path
        for cell in range(1, CELL_COUNT + 1)
        for path in expected_cell_paths(cell)
    ]
    paths.extend(expected_parent_paths())
    if len(paths) != OUTPUT_FILES or len(set(paths)) != OUTPUT_FILES:
        raise ContractError("exact output inventory arithmetic differs")
    if any(
        not path
        or path.startswith(("/", "\\"))
        or "\\" in path
        or ".." in path.split("/")
        or "*" in path
        for path in paths
    ):
        raise ContractError("output inventory contains unsafe path")
    return tuple(paths)


def verify_completion_counters(value: object) -> dict[str, int]:
    counters = require_exact_keys(value, EXACT_COMPLETION_COUNTERS, label="completion counters")
    for name, expected in EXACT_COMPLETION_COUNTERS.items():
        actual = strict_int(counters[name], label=f"counter {name}")
        if actual != expected:
            raise ContractError(f"counter {name} differs: {actual} != {expected}")
    if counters["scores"] != sum(
        counters[name]
        for name in ()
    ) + SCORE_DECOMPOSITION["total"]:
        raise ContractError("score accounting differs from 45+15+5+3")
    return dict(counters)


def verify_sanitized_buffer(payload: bytes, *, cell: int, expected_pin: object) -> dict[str, Any]:
    if type(payload) is not bytes or b"\x00" in payload:
        raise ContractError("sanitized cell buffer is not canonical text bytes")
    pin = validate_pin(expected_pin, label="sanitized buffer pin")
    if pin["bytes"] != len(payload) or pin["sha256"] != bytes_sha256(payload):
        raise ContractError("sanitized cell buffer pin differs")
    lines = payload.splitlines()
    if len(lines) < 3:
        raise ContractError("sanitized cell buffer has no rows")
    prefix = lines[0].decode("ascii", errors="strict")
    expected_prefix = f"P1V6R4-CELL:{cell:02d}:"
    base_sha = prefix.removeprefix(expected_prefix)
    if not prefix.startswith(expected_prefix) or not is_sha256(base_sha):
        raise ContractError("sanitized cell buffer channel prefix differs")
    csv_payload = b"\n".join(lines[1:]) + b"\n"
    if bytes_sha256(csv_payload) != base_sha:
        raise ContractError("sanitized cell buffer base digest differs")
    header = lines[1].decode("utf-8", errors="strict").split(",")
    if header != ["station", "year", "layer", "time", "temp", "psal", "depth"]:
        raise ContractError("sanitized cell buffer header differs or contains targets")
    if any(len(line.split(b",")) != 7 for line in lines[2:]):
        raise ContractError("sanitized cell buffer field count differs")
    cell_identity(cell)
    return {"cell": cell, "rows": len(lines) - 2, "pin": pin}


def static_contract_summary() -> dict[str, Any]:
    return {
        "generation": GENERATION,
        "cells": CELL_COUNT,
        "sanitized_buffers": CELL_COUNT,
        "target_brokers": 1,
        "teacher_receipts": TEACHER_RECEIPTS,
        "score_calls": dict(SCORE_DECOMPOSITION),
        "output_files": len(expected_output_paths()),
        "execution_authority_present": False,
    }


__all__ = [
    "BLOCKS",
    "CELL_COUNT",
    "ContractError",
    "EXACT_COMPLETION_COUNTERS",
    "FILES_PER_CELL",
    "FOLDS",
    "FRACTIONS",
    "FRACTION_TAGS",
    "GENERATION",
    "GENESIS_SHA256",
    "OUTPUT_FILES",
    "PHASES",
    "SCORE_DECOMPOSITION",
    "TEACHER_RECEIPTS",
    "bytes_sha256",
    "canonical_json_bytes",
    "cell_identity",
    "deep_sha256",
    "event_body",
    "expected_cell_paths",
    "expected_output_paths",
    "expected_parent_paths",
    "expected_teacher_keys",
    "is_sha256",
    "parse_json_bytes",
    "require_exact_keys",
    "row_ids_sha256",
    "static_contract_summary",
    "strict_int",
    "validate_launch_envelope",
    "validate_pin",
    "verify_completion_counters",
    "verify_disjoint",
    "verify_event",
    "verify_sanitized_buffer",
    "verify_teacher_receipt_catalog",
]
