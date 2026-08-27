"""Binary-safe replay ledger that supersedes the failed Windows v4 genesis.

The valid chain predecessor remains the immutable one-event v3 ledger.  The
failed v4 registry, its independent QA receipt, and its fail-closed receipt are
preserved as pinned recovery evidence, never as valid predecessor events.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ocean_goal import meaningful_score_v3 as scoring

ContractError = scoring.ContractError
canonical_json_bytes = scoring.canonical_json_bytes
sha256_file = scoring.sha256_file

CONTRACT_RELATIVE = "configs/goals/meaningful_score_ledger_v5.json"
CONTRACT_SHA256 = "6c61e71016d3c4cf9b8a34e818ca865b7246e3b02ad2cb22348dd2a6e71d6b35"
LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v5/registry.jsonl"
PRE_INIT_QA_RELATIVE = "artifacts/meaningful_score_goal_v5/pre_init_qa.json"

V3_SCORING_CONTRACT_RELATIVE = "configs/goals/meaningful_score_maximization_v3.json"
V3_SCORING_CONTRACT_SHA256 = (
    "76d7cc6e10277ae6c06b1109a15f4dcc8413216f73ac3f479ec265a065d756c0"
)
V3_SCORING_EVALUATOR_RELATIVE = "src/ocean_goal/meaningful_score_v3.py"
V3_SCORING_EVALUATOR_SHA256 = (
    "720839224209cf487e500650ecf34c1e8cd3e5fb26f2395c60e76d2050ed973a"
)
V3_LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v3/registry.jsonl"
V3_LEDGER_SHA256 = "51817d6e4b0c4a9a8ef4b5207e2d852107371ebe7da457e2007fca9ce1adeb2c"
V3_LEDGER_BYTES = 5137
V3_LEDGER_EVENT_COUNT = 1
V3_LEDGER_HEAD_SHA256 = "5e3e6b56bee7df5324d31ef0dcc0f698ce7c2e28ad4be95e9d855bb69132f99a"

V4_FAILED_REGISTRY = {
    "path": "artifacts/meaningful_score_goal_v4/registry.jsonl",
    "sha256": "13c48b8d0e3169b272bd624d83cb34fb92079c511dcd727e07f1da024164df1c",
    "bytes": 2470,
}
V4_PRE_INIT_QA = {
    "path": "artifacts/meaningful_score_goal_v4/pre_init_qa.json",
    "sha256": "303e22cdf807f73e5846138447011a063ca25658b9666e602209aee7281c7450",
    "bytes": 1815,
}
V4_FAILURE_RECEIPT = {
    "path": "artifacts/meaningful_score_goal_v4/initialization_failure_20260823.json",
    "sha256": "1adc2ef84e10fc41730655a85ddda3e3e58946f4d8d67c15efa8bdbe60c1b18a",
    "bytes": 2406,
}

GENESIS_EVENT_TYPE = "GOAL_INITIALIZED"
LATER_EVENT_TYPES = frozenset(
    {"CURVE_RESULT", "OFFICIAL_SCORE_RESULT", "UPLOAD_READINESS", "GOAL_COMPLETION"}
)
ALL_EVENT_TYPES = frozenset({GENESIS_EVENT_TYPE, *LATER_EVENT_TYPES})
O_BINARY = getattr(os, "O_BINARY", 0)

IMPLEMENTATION_RELATIVES = {
    "V5_CONTRACT": CONTRACT_RELATIVE,
    "V5_EVALUATOR": "src/ocean_goal/meaningful_score_ledger_v5.py",
    "V5_CLI": "scripts/run_meaningful_score_ledger_v5.py",
    "V5_TESTS": "tests/test_meaningful_score_ledger_v5.py",
}
IMPLEMENTATION_ROLES = frozenset(IMPLEMENTATION_RELATIVES)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _workspace(root: Path) -> Path:
    return root.resolve(strict=True)


def _workspace_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    workspace = _workspace(root)
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ContractError("path must be workspace-relative and non-traversing")
    resolved = (workspace / candidate).resolve(strict=must_exist)
    if not resolved.is_relative_to(workspace):
        raise ContractError("path escapes workspace")
    return resolved


def _canonical_ledger_path(root: Path, requested: Path, *, must_exist: bool) -> Path:
    workspace = _workspace(root)
    lexical = workspace / LEDGER_RELATIVE
    expected = lexical.resolve(strict=must_exist)
    if expected != lexical:
        raise ContractError("canonical v5 ledger path must not use symlinks")
    candidate = requested.resolve(strict=must_exist)
    if not candidate.is_relative_to(workspace):
        raise ContractError("v5 ledger escapes workspace")
    if candidate != expected:
        raise ContractError("only the canonical v5 ledger path is accepted")
    return candidate


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ContractError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)


def _json_object(path: Path) -> dict[str, Any]:
    value = _strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _file_pin_for_path(root: Path, path: Path) -> dict[str, Any]:
    workspace = _workspace(root)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(workspace):
        raise ContractError("pinned file escapes workspace")
    return {
        "path": resolved.relative_to(workspace).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _verify_file_pin(
    root: Path, value: Mapping[str, Any], *, role: str
) -> tuple[dict[str, Any], Path]:
    if set(value) != {"path", "sha256", "bytes"}:
        raise ContractError(f"{role} pin keys changed")
    relative = value.get("path")
    expected_sha = value.get("sha256")
    expected_bytes = value.get("bytes")
    if not isinstance(relative, str) or not _is_sha256(expected_sha):
        raise ContractError(f"{role} pin identity is invalid")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
        raise ContractError(f"{role} byte count is invalid")
    path = _workspace_path(root, relative)
    observed = _file_pin_for_path(root, path)
    if observed != dict(value):
        raise ContractError(f"{role} SHA or size mismatch")
    return observed, path


def _predecessor_anchor() -> dict[str, Any]:
    return {
        "path": V3_LEDGER_RELATIVE,
        "sha256": V3_LEDGER_SHA256,
        "bytes": V3_LEDGER_BYTES,
        "event_count": V3_LEDGER_EVENT_COUNT,
        "head_event_sha256": V3_LEDGER_HEAD_SHA256,
    }


def _failed_v4_contract() -> dict[str, Any]:
    return {
        "failed_registry": V4_FAILED_REGISTRY,
        "pre_init_qa": V4_PRE_INIT_QA,
        "failure_receipt": V4_FAILURE_RECEIPT,
        "valid_v4_event_count": 0,
        "v4_retry_allowed": False,
        "failure_reason": "WINDOWS_TEXT_MODE_TRANSLATED_TERMINAL_LF_TO_CRLF",
    }


def load_contract(root: Path, relative: str = CONTRACT_RELATIVE) -> dict[str, Any]:
    if Path(relative).as_posix() != CONTRACT_RELATIVE:
        raise ContractError("only the canonical v5 ledger contract is accepted")
    path = _workspace_path(root, relative)
    if _file_pin_for_path(root, path)["path"] != CONTRACT_RELATIVE:
        raise ContractError("canonical v5 ledger contract path uses a symlink")
    if sha256_file(path) != CONTRACT_SHA256:
        raise ContractError("canonical v5 ledger contract SHA mismatch")
    contract = _json_object(path)
    if contract.get("schema_version") != "meaningful_score_ledger.v5":
        raise ContractError("v5 ledger schema identity changed")
    if contract.get("scoring_contract") != {
        "path": V3_SCORING_CONTRACT_RELATIVE,
        "sha256": V3_SCORING_CONTRACT_SHA256,
    }:
        raise ContractError("v3 scoring-contract pin changed")
    if contract.get("scoring_evaluator") != {
        "path": V3_SCORING_EVALUATOR_RELATIVE,
        "sha256": V3_SCORING_EVALUATOR_SHA256,
    }:
        raise ContractError("v3 scoring-evaluator pin changed")
    if contract.get("predecessor_ledger") != _predecessor_anchor():
        raise ContractError("valid v3 predecessor anchor changed")
    if contract.get("failed_v4_lineage") != _failed_v4_contract():
        raise ContractError("failed v4 recovery lineage changed")
    if contract.get("canonical_paths") != {
        "ledger": LEDGER_RELATIVE,
        "pre_init_qa": PRE_INIT_QA_RELATIVE,
    }:
        raise ContractError("v5 canonical paths changed")
    protocol = contract.get("event_protocol")
    if not isinstance(protocol, Mapping):
        raise ContractError("v5 event protocol is missing")
    if protocol.get("genesis_event_type") != GENESIS_EVENT_TYPE or protocol.get(
        "later_event_types"
    ) != ["CURVE_RESULT", "OFFICIAL_SCORE_RESULT", "UPLOAD_READINESS", "GOAL_COMPLETION"]:
        raise ContractError("v5 typed event allowlist changed")
    for key in (
        "unknown_event_types_forbidden",
        "payload_must_deep_equal_recomputed_payload",
        "evidence_must_be_workspace_relative_sha256_and_size_pinned",
        "replay_every_event_on_every_validation",
        "replay_before_append_lock_or_write",
        "all_os_open_calls_use_o_binary",
        "all_os_write_calls_use_robust_write_loop",
    ):
        if protocol.get(key) is not True:
            raise ContractError(f"v5 protocol weakened: {key}")
    if protocol.get("canonical_line_ending") != "LF_ONLY":
        raise ContractError("v5 canonical line ending changed")
    return contract


def verify_predecessor(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("predecessor_ledger") != _predecessor_anchor():
        raise ContractError("v3 predecessor differs from v5 contract")
    paths = {
        "V3_SCORING_CONTRACT": _workspace_path(root, V3_SCORING_CONTRACT_RELATIVE),
        "V3_SCORING_EVALUATOR": _workspace_path(root, V3_SCORING_EVALUATOR_RELATIVE),
        "V3_PREDECESSOR_LEDGER": _workspace_path(root, V3_LEDGER_RELATIVE),
    }
    pins = {role: _file_pin_for_path(root, path) for role, path in paths.items()}
    if pins["V3_SCORING_CONTRACT"] != {
        "path": V3_SCORING_CONTRACT_RELATIVE,
        "sha256": V3_SCORING_CONTRACT_SHA256,
        "bytes": paths["V3_SCORING_CONTRACT"].stat().st_size,
    }:
        raise ContractError("v3 scoring-contract bytes drifted")
    if pins["V3_SCORING_EVALUATOR"] != {
        "path": V3_SCORING_EVALUATOR_RELATIVE,
        "sha256": V3_SCORING_EVALUATOR_SHA256,
        "bytes": paths["V3_SCORING_EVALUATOR"].stat().st_size,
    }:
        raise ContractError("v3 scoring-evaluator bytes drifted")
    if pins["V3_PREDECESSOR_LEDGER"] != {
        "path": V3_LEDGER_RELATIVE,
        "sha256": V3_LEDGER_SHA256,
        "bytes": V3_LEDGER_BYTES,
    }:
        raise ContractError("v3 predecessor-ledger bytes drifted")
    scoring.load_contract(root)
    records = scoring.validate_goal_ledger(root, paths["V3_PREDECESSOR_LEDGER"])
    if len(records) != 1 or records[-1].get("event_sha256") != V3_LEDGER_HEAD_SHA256:
        raise ContractError("v3 predecessor count or head drifted")
    return pins


def verify_failed_v4_lineage(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("failed_v4_lineage") != _failed_v4_contract():
        raise ContractError("failed v4 lineage differs from v5 contract")
    pins: dict[str, Any] = {}
    for role, expected in (
        ("V4_FAILED_REGISTRY", V4_FAILED_REGISTRY),
        ("V4_PRE_INIT_QA", V4_PRE_INIT_QA),
        ("V4_FAILURE_RECEIPT", V4_FAILURE_RECEIPT),
    ):
        pin, _path = _verify_file_pin(root, expected, role=role)
        pins[role] = pin
    receipt = _json_object(_workspace_path(root, V4_FAILURE_RECEIPT["path"]))
    if (
        receipt.get("schema_version") != "meaningful_score_ledger_v4.initialization_failure.v1"
        or receipt.get("status") != "FAIL_CLOSED_WINDOWS_TEXT_MODE_CRLF_CANONICAL_PATH_CONSUMED"
        or receipt.get("failed_ledger") != V4_FAILED_REGISTRY
        or receipt.get("pre_init_qa") != V4_PRE_INIT_QA
        or receipt.get("predecessor_ledger_anchor") != _predecessor_anchor()
        or receipt.get("observed", {}).get("valid_v4_event_count") != 0
        or receipt.get("observed", {}).get("retry_performed") is not False
        or receipt.get("required_recovery", {}).get("create_new_v5_canonical_ledger") is not True
        or receipt.get("required_recovery", {}).get("use_binary_mode_for_genesis_lock_and_append")
        is not True
    ):
        raise ContractError("v4 failure receipt does not authorize the v5 recovery lineage")
    return pins


def current_implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    pins = {
        role: _file_pin_for_path(root, _workspace_path(root, relative))
        for role, relative in IMPLEMENTATION_RELATIVES.items()
    }
    if set(pins) != IMPLEMENTATION_ROLES:
        raise ContractError("v5 implementation role set changed")
    return pins


def _canonical_qa_pin(root: Path, requested: Path) -> dict[str, Any]:
    workspace = _workspace(root)
    lexical = workspace / PRE_INIT_QA_RELATIVE
    expected = lexical.resolve(strict=False)
    if expected != lexical:
        raise ContractError("canonical v5 QA path must not use symlinks")
    resolved = requested.resolve(strict=True)
    if not resolved.is_relative_to(workspace) or resolved != expected:
        raise ContractError("v5 pre-init QA receipt path must be canonical")
    return _file_pin_for_path(root, resolved)


def _ledger_contract_pin(root: Path) -> dict[str, Any]:
    return _file_pin_for_path(root, _workspace_path(root, CONTRACT_RELATIVE))


def build_genesis_payload(root: Path, qa_receipt: Path) -> dict[str, Any]:
    contract = load_contract(root)
    predecessor_pins = verify_predecessor(root, contract)
    failed_v4_pins = verify_failed_v4_lineage(root, contract)
    implementation_pins = current_implementation_pins(root)
    qa_pin = _canonical_qa_pin(root, qa_receipt)
    receipt = _json_object(_workspace_path(root, qa_pin["path"]))
    if (
        receipt.get("schema_version") != "meaningful_score_ledger_v5.pre_init_qa.v1"
        or receipt.get("decision") != "GO_INITIALIZE_V5_LEDGER"
        or receipt.get("p0_count") != 0
        or receipt.get("p1_count") != 0
        or receipt.get("ledger_contract") != _ledger_contract_pin(root)
        or receipt.get("predecessor_ledger_anchor") != _predecessor_anchor()
        or receipt.get("failed_v4_lineage_pins") != failed_v4_pins
        or receipt.get("implementation_pins") != implementation_pins
    ):
        raise ContractError("v5 pre-init QA receipt does not bind current genesis")
    initial = contract["initial_state"]
    return {
        "ledger_id": contract["ledger_id"],
        "status": initial["status"],
        "ledger_contract": _ledger_contract_pin(root),
        "verified_predecessor_pins": predecessor_pins,
        "predecessor_ledger_anchor": _predecessor_anchor(),
        "failed_v4_lineage_pins": failed_v4_pins,
        "implementation_pins": implementation_pins,
        "independent_pre_init_qa": qa_pin,
        "official_uploads": initial["official_uploads"],
        "score_promotions": initial["score_promotions"],
        "meaningful_promotions": initial["meaningful_promotions"],
        "execution_counts": initial["execution_counts"],
        "upload_performed": False,
    }


def _verify_genesis_payload(root: Path, payload: Mapping[str, Any]) -> None:
    qa = payload.get("independent_pre_init_qa")
    if not isinstance(qa, Mapping) or qa.get("path") != PRE_INIT_QA_RELATIVE:
        raise ContractError("v5 genesis canonical QA pin is missing")
    _, qa_path = _verify_file_pin(root, qa, role="v5 pre-init QA")
    if dict(payload) != build_genesis_payload(root, qa_path):
        raise ContractError("v5 genesis payload differs from canonical recomputation")


def _source_json_from_pin(
    root: Path, value: Any, *, role: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{role} pin is missing")
    pin, path = _verify_file_pin(root, value, role=role)
    return _json_object(path), pin, path


def _scoring_contract(root: Path) -> dict[str, Any]:
    return scoring.load_contract(root)


def build_curve_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    pin = _file_pin_for_path(root, evidence_path)
    evidence = _json_object(_workspace_path(root, pin["path"]))
    contract = _scoring_contract(root)
    return {
        "schema_version": "meaningful_score_ledger_v5.curve_result.v1",
        "evidence": pin,
        "evidence_pins": scoring.verify_curve_evidence_pins(root, evidence),
        "decision": scoring.evaluate_learning_curve(contract, evidence),
        "upload_performed": False,
    }


def build_official_score_payload(
    root: Path, evidence_path: Path, curve_decision_path: Path
) -> dict[str, Any]:
    evidence_pin = _file_pin_for_path(root, evidence_path)
    curve_pin = _file_pin_for_path(root, curve_decision_path)
    evidence = _json_object(_workspace_path(root, evidence_pin["path"]))
    curve = _json_object(_workspace_path(root, curve_pin["path"]))
    contract = _scoring_contract(root)
    return {
        "schema_version": "meaningful_score_ledger_v5.official_score_result.v1",
        "evidence": evidence_pin,
        "curve_decision": curve_pin,
        "evidence_pins": scoring.verify_official_evidence_pins(root, contract, evidence),
        "decision": scoring.evaluate_official_score(contract, curve, evidence),
        "upload_performed": False,
    }


def build_upload_readiness_payload(
    root: Path, receipt_path: Path, curve_decision_path: Path | None
) -> dict[str, Any]:
    receipt_pin = _file_pin_for_path(root, receipt_path)
    receipt = _json_object(_workspace_path(root, receipt_pin["path"]))
    if curve_decision_path is None:
        curve_pin = None
        curve = None
    else:
        curve_pin = _file_pin_for_path(root, curve_decision_path)
        curve = _json_object(_workspace_path(root, curve_pin["path"]))
    return {
        "schema_version": "meaningful_score_ledger_v5.upload_readiness.v1",
        "receipt": receipt_pin,
        "curve_decision": curve_pin,
        "readiness": scoring.validate_upload_approval(
            root, _scoring_contract(root), receipt, curve_decision=curve
        ),
        "upload_performed": False,
    }


def build_goal_completion_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    pin = _file_pin_for_path(root, evidence_path)
    evidence = _json_object(_workspace_path(root, pin["path"]))
    return {
        "schema_version": "meaningful_score_ledger_v5.goal_completion.v1",
        "evidence": pin,
        "decision": scoring.evaluate_goal_completion(_scoring_contract(root), evidence),
        "upload_performed": False,
    }


def recompute_later_payload(
    root: Path, event_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v5 event type is not allowlisted: {event_type}")
    if not isinstance(payload, Mapping):
        raise ContractError("v5 event payload must be an object")
    if event_type == "CURVE_RESULT":
        _, _, evidence = _source_json_from_pin(root, payload.get("evidence"), role="curve evidence")
        expected = build_curve_payload(root, evidence)
    elif event_type == "OFFICIAL_SCORE_RESULT":
        _, _, evidence = _source_json_from_pin(
            root, payload.get("evidence"), role="official evidence"
        )
        _, _, curve = _source_json_from_pin(
            root, payload.get("curve_decision"), role="curve decision"
        )
        expected = build_official_score_payload(root, evidence, curve)
    elif event_type == "UPLOAD_READINESS":
        _, _, receipt = _source_json_from_pin(
            root, payload.get("receipt"), role="upload receipt"
        )
        curve_value = payload.get("curve_decision")
        if curve_value is None:
            curve = None
        else:
            _, _, curve = _source_json_from_pin(
                root, curve_value, role="upload curve decision"
            )
        expected = build_upload_readiness_payload(root, receipt, curve)
    else:
        _, _, evidence = _source_json_from_pin(
            root, payload.get("evidence"), role="completion evidence"
        )
        expected = build_goal_completion_payload(root, evidence)
    if dict(payload) != expected:
        raise ContractError(f"{event_type} payload differs from replay recomputation")
    return expected


def _ledger_record(
    *, seq: int, previous: str, event_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if event_type not in ALL_EVENT_TYPES:
        raise ContractError(f"v5 event type is not allowlisted: {event_type}")
    base = {
        "seq": seq,
        "recorded_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "event_type": event_type,
        "previous_event_sha256": previous,
        "payload": dict(payload),
    }
    return {**base, "event_sha256": hashlib.sha256(canonical_json_bytes(base)).hexdigest()}


def validate_ledger(root: Path, path: Path) -> list[dict[str, Any]]:
    contract = load_contract(root)
    verify_predecessor(root, contract)
    verify_failed_v4_lineage(root, contract)
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    if not ledger.exists():
        return []
    raw = ledger.read_bytes()
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ContractError("v5 ledger must be non-empty canonical LF-terminated JSONL")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("v5 ledger is not UTF-8") from error
    records: list[dict[str, Any]] = []
    previous = V3_LEDGER_HEAD_SHA256
    for line_number, line in enumerate(text.split("\n")[:-1], 1):
        if not line.strip():
            raise ContractError(f"blank v5 ledger line {line_number}")
        record = _strict_json_loads(line)
        if not isinstance(record, dict) or set(record) != {
            "seq",
            "recorded_at_kst",
            "event_type",
            "previous_event_sha256",
            "payload",
            "event_sha256",
        }:
            raise ContractError(f"v5 ledger record keys changed at line {line_number}")
        if (
            not isinstance(record.get("seq"), int)
            or isinstance(record.get("seq"), bool)
            or record["seq"] != len(records) + 1
        ):
            raise ContractError(f"v5 ledger sequence mismatch at line {line_number}")
        if record.get("previous_event_sha256") != previous:
            raise ContractError(f"v5 ledger chain mismatch at line {line_number}")
        timestamp = record.get("recorded_at_kst")
        if not isinstance(timestamp, str):
            raise ContractError(f"v5 ledger timestamp is invalid at line {line_number}")
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ContractError(f"v5 ledger timestamp is invalid at line {line_number}") from error
        if parsed.utcoffset() != timedelta(hours=9):
            raise ContractError(f"v5 ledger timestamp is not KST at line {line_number}")
        event_type = record.get("event_type")
        if event_type not in ALL_EVENT_TYPES:
            raise ContractError(f"v5 ledger event is not allowlisted at line {line_number}")
        claimed = record.get("event_sha256")
        if not _is_sha256(claimed):
            raise ContractError(f"v5 event SHA is invalid at line {line_number}")
        base = {key: value for key, value in record.items() if key != "event_sha256"}
        if hashlib.sha256(canonical_json_bytes(base)).hexdigest() != claimed:
            raise ContractError(f"v5 event hash mismatch at line {line_number}")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise ContractError(f"v5 payload is invalid at line {line_number}")
        if line_number == 1:
            if event_type != GENESIS_EVENT_TYPE:
                raise ContractError("v5 first event is not genesis")
            _verify_genesis_payload(root, payload)
        else:
            if event_type == GENESIS_EVENT_TYPE:
                raise ContractError("v5 ledger contains duplicate genesis")
            recompute_later_payload(root, event_type, payload)
        previous = claimed
        records.append(record)
    return records


def _write_all(descriptor: int, payload: bytes, *, role: str) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError(f"short v5 {role} write")
        offset += written


def initialize_ledger(
    root: Path, path: Path, *, payload: Mapping[str, Any]
) -> dict[str, Any]:
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    _verify_genesis_payload(root, payload)
    record = _ledger_record(
        seq=1,
        previous=V3_LEDGER_HEAD_SHA256,
        event_type=GENESIS_EVENT_TYPE,
        payload=payload,
    )
    encoded = canonical_json_bytes(record) + b"\n"
    if not ledger.parent.is_dir():
        raise ContractError("canonical v5 ledger directory must already exist via QA receipt")
    descriptor = os.open(
        ledger,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_BINARY,
        0o600,
    )
    try:
        _write_all(descriptor, encoded, role="genesis")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    persisted = validate_ledger(root, ledger)
    if persisted != [record]:
        raise ContractError("v5 genesis failed replay round-trip validation")
    return record


def append_ledger_event(
    root: Path,
    path: Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v5 later event is not allowlisted: {event_type}")
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    canonical_payload = recompute_later_payload(root, event_type, payload)
    existing = validate_ledger(root, ledger)
    if not existing:
        raise ContractError("v5 ledger must be initialized before append")
    snapshot = (len(existing), existing[-1]["event_sha256"])
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    descriptor = os.open(
        lock,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_BINARY,
        0o600,
    )
    try:
        try:
            lock_payload = canonical_json_bytes(
                {
                    "schema_version": "meaningful_score_ledger_v5.append_lock.v1",
                    "ledger": ledger.name,
                    "pid": os.getpid(),
                    "expected_event_count": snapshot[0],
                    "expected_head_event_sha256": snapshot[1],
                }
            )
            _write_all(descriptor, lock_payload, role="append-lock")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        locked = validate_ledger(root, ledger)
        if (len(locked), locked[-1]["event_sha256"]) != snapshot:
            raise ContractError("v5 ledger changed before lock acquisition")
        replayed = recompute_later_payload(root, event_type, canonical_payload)
        record = _ledger_record(
            seq=snapshot[0] + 1,
            previous=snapshot[1],
            event_type=event_type,
            payload=replayed,
        )
        encoded = canonical_json_bytes(record) + b"\n"
        descriptor = os.open(ledger, os.O_WRONLY | os.O_APPEND | O_BINARY)
        try:
            _write_all(descriptor, encoded, role="ledger-append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        persisted = validate_ledger(root, ledger)
        if len(persisted) != snapshot[0] + 1 or persisted[-1] != record:
            raise ContractError("v5 append failed replay round-trip validation")
        return record
    finally:
        lock.unlink(missing_ok=True)


__all__ = [
    "ALL_EVENT_TYPES",
    "CONTRACT_RELATIVE",
    "CONTRACT_SHA256",
    "ContractError",
    "GENESIS_EVENT_TYPE",
    "IMPLEMENTATION_RELATIVES",
    "IMPLEMENTATION_ROLES",
    "LATER_EVENT_TYPES",
    "LEDGER_RELATIVE",
    "O_BINARY",
    "PRE_INIT_QA_RELATIVE",
    "V3_LEDGER_HEAD_SHA256",
    "append_ledger_event",
    "build_curve_payload",
    "build_genesis_payload",
    "build_goal_completion_payload",
    "build_official_score_payload",
    "build_upload_readiness_payload",
    "current_implementation_pins",
    "initialize_ledger",
    "load_contract",
    "recompute_later_payload",
    "sha256_file",
    "validate_ledger",
    "verify_failed_v4_lineage",
    "verify_predecessor",
]
