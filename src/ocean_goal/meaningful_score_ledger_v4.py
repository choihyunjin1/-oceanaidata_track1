"""Replay-validated append-only ledger for the pinned v3 scoring contract.

The v3 scoring implementation and its one-event ledger are immutable
predecessors.  This module never trains, predicts, or uploads.  Every later
event is rebuilt from SHA/size-pinned evidence and must deep-equal that rebuild
before an append lock is created and again before bytes are written.
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

CONTRACT_RELATIVE = "configs/goals/meaningful_score_ledger_v4.json"
CONTRACT_SHA256 = "2f8ed835b9cb15f0dcfc6809f8417b1162a86b7aa84e3193e36f9169d609e4e4"
LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v4/registry.jsonl"
PRE_INIT_QA_RELATIVE = "artifacts/meaningful_score_goal_v4/pre_init_qa.json"

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

GENESIS_EVENT_TYPE = "GOAL_INITIALIZED"
LATER_EVENT_TYPES = frozenset(
    {
        "CURVE_RESULT",
        "OFFICIAL_SCORE_RESULT",
        "UPLOAD_READINESS",
        "GOAL_COMPLETION",
    }
)
ALL_EVENT_TYPES = frozenset({GENESIS_EVENT_TYPE, *LATER_EVENT_TYPES})

IMPLEMENTATION_RELATIVES = {
    "V4_CONTRACT": CONTRACT_RELATIVE,
    "V4_EVALUATOR": "src/ocean_goal/meaningful_score_ledger_v4.py",
    "V4_CLI": "scripts/run_meaningful_score_ledger_v4.py",
    "V4_TESTS": "tests/test_meaningful_score_ledger_v4.py",
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
        raise ContractError("canonical v4 ledger path must not use symlinks")
    candidate = requested.resolve(strict=must_exist)
    if not candidate.is_relative_to(workspace):
        raise ContractError("v4 ledger escapes workspace")
    if candidate != expected:
        raise ContractError("only the canonical v4 ledger path is accepted")
    return candidate


def _json_object(path: Path) -> dict[str, Any]:
    value = _strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


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

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


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


def load_contract(root: Path, relative: str = CONTRACT_RELATIVE) -> dict[str, Any]:
    """Load only the canonical ledger contract and enforce its fixed protocol."""

    if Path(relative).as_posix() != CONTRACT_RELATIVE:
        raise ContractError("only the canonical v4 ledger contract is accepted")
    path = _workspace_path(root, relative)
    if _file_pin_for_path(root, path)["path"] != CONTRACT_RELATIVE:
        raise ContractError("canonical v4 ledger contract path uses a symlink")
    if sha256_file(path) != CONTRACT_SHA256:
        raise ContractError("canonical v4 ledger contract SHA mismatch")
    contract = _json_object(path)
    if contract.get("schema_version") != "meaningful_score_ledger.v4":
        raise ContractError("v4 ledger schema identity changed")
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
        raise ContractError("v3 predecessor-ledger anchor changed")
    if contract.get("canonical_paths") != {
        "ledger": LEDGER_RELATIVE,
        "pre_init_qa": PRE_INIT_QA_RELATIVE,
    }:
        raise ContractError("v4 canonical paths changed")
    protocol = contract.get("event_protocol")
    if not isinstance(protocol, Mapping):
        raise ContractError("v4 event protocol is missing")
    if protocol.get("genesis_event_type") != GENESIS_EVENT_TYPE:
        raise ContractError("v4 genesis event type changed")
    if protocol.get("later_event_types") != [
        "CURVE_RESULT",
        "OFFICIAL_SCORE_RESULT",
        "UPLOAD_READINESS",
        "GOAL_COMPLETION",
    ]:
        raise ContractError("v4 later-event allowlist changed")
    required_true = {
        "unknown_event_types_forbidden",
        "payload_must_deep_equal_recomputed_payload",
        "evidence_must_be_workspace_relative_sha256_and_size_pinned",
        "replay_every_event_on_every_validation",
        "replay_before_append_lock_or_write",
    }
    if any(protocol.get(key) is not True for key in required_true):
        raise ContractError("v4 replay protocol weakened")
    return contract


def _predecessor_anchor() -> dict[str, Any]:
    return {
        "path": V3_LEDGER_RELATIVE,
        "sha256": V3_LEDGER_SHA256,
        "bytes": V3_LEDGER_BYTES,
        "event_count": V3_LEDGER_EVENT_COUNT,
        "head_event_sha256": V3_LEDGER_HEAD_SHA256,
    }


def verify_predecessor(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate current v3 contract/evaluator/ledger bytes and full v3 chain."""

    if contract.get("predecessor_ledger") != _predecessor_anchor():
        raise ContractError("v3 predecessor anchor differs from v4 contract")
    contract_path = _workspace_path(root, V3_SCORING_CONTRACT_RELATIVE)
    evaluator_path = _workspace_path(root, V3_SCORING_EVALUATOR_RELATIVE)
    ledger_path = _workspace_path(root, V3_LEDGER_RELATIVE)
    pins = {
        "V3_SCORING_CONTRACT": _file_pin_for_path(root, contract_path),
        "V3_SCORING_EVALUATOR": _file_pin_for_path(root, evaluator_path),
        "V3_PREDECESSOR_LEDGER": _file_pin_for_path(root, ledger_path),
    }
    if pins["V3_SCORING_CONTRACT"]["path"] != V3_SCORING_CONTRACT_RELATIVE:
        raise ContractError("v3 scoring-contract path is noncanonical")
    if pins["V3_SCORING_CONTRACT"]["sha256"] != V3_SCORING_CONTRACT_SHA256:
        raise ContractError("v3 scoring-contract bytes drifted")
    if pins["V3_SCORING_EVALUATOR"]["path"] != V3_SCORING_EVALUATOR_RELATIVE:
        raise ContractError("v3 scoring-evaluator path is noncanonical")
    if pins["V3_SCORING_EVALUATOR"]["sha256"] != V3_SCORING_EVALUATOR_SHA256:
        raise ContractError("v3 scoring-evaluator bytes drifted")
    if pins["V3_PREDECESSOR_LEDGER"] != {
        "path": V3_LEDGER_RELATIVE,
        "sha256": V3_LEDGER_SHA256,
        "bytes": V3_LEDGER_BYTES,
    }:
        raise ContractError("v3 predecessor-ledger bytes drifted")
    scoring.load_contract(root)
    records = scoring.validate_goal_ledger(root, ledger_path)
    if (
        len(records) != V3_LEDGER_EVENT_COUNT
        or records[-1].get("event_sha256") != V3_LEDGER_HEAD_SHA256
    ):
        raise ContractError("v3 predecessor-ledger count or head drifted")
    return pins


def current_implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    pins: dict[str, dict[str, Any]] = {}
    for role, relative in IMPLEMENTATION_RELATIVES.items():
        pin = _file_pin_for_path(root, _workspace_path(root, relative))
        if pin["path"] != relative:
            raise ContractError(f"v4 implementation path is noncanonical: {role}")
        pins[role] = pin
    if set(pins) != IMPLEMENTATION_ROLES:
        raise ContractError("v4 implementation role set changed")
    return pins


def _canonical_qa_pin(root: Path, requested: Path) -> dict[str, Any]:
    workspace = _workspace(root)
    lexical = workspace / PRE_INIT_QA_RELATIVE
    expected = lexical.resolve(strict=False)
    if expected != lexical:
        raise ContractError("canonical pre-init QA path must not use symlinks")
    resolved = requested.resolve(strict=True)
    if not resolved.is_relative_to(workspace):
        raise ContractError("pre-init QA receipt escapes workspace")
    if resolved != expected:
        raise ContractError("pre-init QA receipt path must be canonical")
    expected = expected.resolve(strict=True)
    return _file_pin_for_path(root, resolved)


def _ledger_contract_pin(root: Path) -> dict[str, Any]:
    return _file_pin_for_path(root, _workspace_path(root, CONTRACT_RELATIVE))


def _scoring_contract(root: Path) -> dict[str, Any]:
    return scoring.load_contract(root)


def build_genesis_payload(root: Path, qa_receipt: Path) -> dict[str, Any]:
    """Build the sole canonical genesis payload after validating independent QA."""

    contract = load_contract(root)
    predecessor_pins = verify_predecessor(root, contract)
    implementation_pins = current_implementation_pins(root)
    qa_pin = _canonical_qa_pin(root, qa_receipt)
    receipt = _json_object(_workspace_path(root, qa_pin["path"]))
    if (
        receipt.get("schema_version") != "meaningful_score_ledger_v4.pre_init_qa.v1"
        or receipt.get("decision") != "GO_INITIALIZE_V4_LEDGER"
        or receipt.get("p0_count") != 0
        or receipt.get("p1_count") != 0
        or receipt.get("ledger_contract") != _ledger_contract_pin(root)
        or receipt.get("predecessor_ledger_anchor") != _predecessor_anchor()
        or receipt.get("implementation_pins") != implementation_pins
    ):
        raise ContractError("v4 pre-init QA receipt does not bind current genesis")
    initial = contract["initial_state"]
    return {
        "ledger_id": contract["ledger_id"],
        "status": initial["status"],
        "ledger_contract": _ledger_contract_pin(root),
        "verified_predecessor_pins": predecessor_pins,
        "predecessor_ledger_anchor": _predecessor_anchor(),
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
        raise ContractError("v4 genesis canonical QA pin is missing")
    _, qa_path = _verify_file_pin(root, qa, role="v4 pre-init QA")
    expected = build_genesis_payload(root, qa_path)
    if dict(payload) != expected:
        raise ContractError("v4 genesis payload differs from canonical recomputation")


def _source_json_from_pin(
    root: Path, value: Any, *, role: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{role} pin is missing")
    pin, path = _verify_file_pin(root, value, role=role)
    return _json_object(path), pin, path


def build_curve_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    pin = _file_pin_for_path(root, evidence_path)
    evidence = _json_object(_workspace_path(root, pin["path"]))
    contract = _scoring_contract(root)
    evidence_pins = scoring.verify_curve_evidence_pins(root, evidence)
    decision = scoring.evaluate_learning_curve(contract, evidence)
    return {
        "schema_version": "meaningful_score_ledger_v4.curve_result.v1",
        "evidence": pin,
        "evidence_pins": evidence_pins,
        "decision": decision,
        "upload_performed": False,
    }


def build_official_score_payload(
    root: Path, evidence_path: Path, curve_decision_path: Path
) -> dict[str, Any]:
    evidence_pin = _file_pin_for_path(root, evidence_path)
    curve_pin = _file_pin_for_path(root, curve_decision_path)
    evidence = _json_object(_workspace_path(root, evidence_pin["path"]))
    curve_decision = _json_object(_workspace_path(root, curve_pin["path"]))
    contract = _scoring_contract(root)
    evidence_pins = scoring.verify_official_evidence_pins(root, contract, evidence)
    decision = scoring.evaluate_official_score(contract, curve_decision, evidence)
    return {
        "schema_version": "meaningful_score_ledger_v4.official_score_result.v1",
        "evidence": evidence_pin,
        "curve_decision": curve_pin,
        "evidence_pins": evidence_pins,
        "decision": decision,
        "upload_performed": False,
    }


def build_upload_readiness_payload(
    root: Path, receipt_path: Path, curve_decision_path: Path | None
) -> dict[str, Any]:
    receipt_pin = _file_pin_for_path(root, receipt_path)
    receipt = _json_object(_workspace_path(root, receipt_pin["path"]))
    if curve_decision_path is None:
        curve_decision = None
        curve_pin = None
    else:
        curve_pin = _file_pin_for_path(root, curve_decision_path)
        curve_decision = _json_object(_workspace_path(root, curve_pin["path"]))
    readiness = scoring.validate_upload_approval(
        root,
        _scoring_contract(root),
        receipt,
        curve_decision=curve_decision,
    )
    return {
        "schema_version": "meaningful_score_ledger_v4.upload_readiness.v1",
        "receipt": receipt_pin,
        "curve_decision": curve_pin,
        "readiness": readiness,
        "upload_performed": False,
    }


def build_goal_completion_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    evidence_pin = _file_pin_for_path(root, evidence_path)
    evidence = _json_object(_workspace_path(root, evidence_pin["path"]))
    decision = scoring.evaluate_goal_completion(_scoring_contract(root), evidence)
    return {
        "schema_version": "meaningful_score_ledger_v4.goal_completion.v1",
        "evidence": evidence_pin,
        "decision": decision,
        "upload_performed": False,
    }


def recompute_later_payload(
    root: Path, event_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Reload pinned evidence and rebuild one allowlisted event from source bytes."""

    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v4 event type is not allowlisted: {event_type}")
    if not isinstance(payload, Mapping):
        raise ContractError("v4 event payload must be an object")
    if event_type == "CURVE_RESULT":
        _, _, evidence_path = _source_json_from_pin(
            root, payload.get("evidence"), role="curve evidence"
        )
        expected = build_curve_payload(root, evidence_path)
    elif event_type == "OFFICIAL_SCORE_RESULT":
        _, _, evidence_path = _source_json_from_pin(
            root, payload.get("evidence"), role="official evidence"
        )
        _, _, curve_path = _source_json_from_pin(
            root, payload.get("curve_decision"), role="curve decision"
        )
        expected = build_official_score_payload(root, evidence_path, curve_path)
    elif event_type == "UPLOAD_READINESS":
        _, _, receipt_path = _source_json_from_pin(
            root, payload.get("receipt"), role="upload receipt"
        )
        curve_value = payload.get("curve_decision")
        if curve_value is None:
            curve_path = None
        else:
            _, _, curve_path = _source_json_from_pin(
                root, curve_value, role="upload curve decision"
            )
        expected = build_upload_readiness_payload(root, receipt_path, curve_path)
    else:
        _, _, evidence_path = _source_json_from_pin(
            root, payload.get("evidence"), role="completion evidence"
        )
        expected = build_goal_completion_payload(root, evidence_path)
    if dict(payload) != expected:
        raise ContractError(f"{event_type} payload differs from replay recomputation")
    return expected


def _ledger_record(
    *, seq: int, previous: str, event_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if event_type not in ALL_EVENT_TYPES:
        raise ContractError(f"v4 event type is not allowlisted: {event_type}")
    base = {
        "seq": seq,
        "recorded_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "event_type": event_type,
        "previous_event_sha256": previous,
        "payload": dict(payload),
    }
    return {
        **base,
        "event_sha256": hashlib.sha256(canonical_json_bytes(base)).hexdigest(),
    }


def validate_ledger(root: Path, path: Path) -> list[dict[str, Any]]:
    """Validate the v3-head -> v4 chain and replay every v4 event."""

    contract = load_contract(root)
    verify_predecessor(root, contract)
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    if not ledger.exists():
        return []
    raw = ledger.read_bytes()
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ContractError("v4 ledger must be non-empty canonical LF-terminated JSONL")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("v4 ledger is not UTF-8") from error
    records: list[dict[str, Any]] = []
    previous = V3_LEDGER_HEAD_SHA256
    for line_number, line in enumerate(text.split("\n")[:-1], 1):
        if not line.strip():
            raise ContractError(f"blank v4 ledger line {line_number}")
        record = _strict_json_loads(line)
        if not isinstance(record, dict):
            raise ContractError(f"v4 ledger record is not an object at line {line_number}")
        if set(record) != {
            "seq",
            "recorded_at_kst",
            "event_type",
            "previous_event_sha256",
            "payload",
            "event_sha256",
        }:
            raise ContractError(f"v4 ledger record keys changed at line {line_number}")
        if (
            not isinstance(record.get("seq"), int)
            or isinstance(record.get("seq"), bool)
            or record.get("seq") != len(records) + 1
        ):
            raise ContractError(f"v4 ledger sequence mismatch at line {line_number}")
        if record.get("previous_event_sha256") != previous:
            raise ContractError(f"v4 ledger chain mismatch at line {line_number}")
        timestamp = record.get("recorded_at_kst")
        if not isinstance(timestamp, str):
            raise ContractError(f"v4 ledger timestamp is invalid at line {line_number}")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ContractError(
                f"v4 ledger timestamp is invalid at line {line_number}"
            ) from error
        if parsed_timestamp.utcoffset() != timedelta(hours=9):
            raise ContractError(f"v4 ledger timestamp is not KST at line {line_number}")
        event_type = record.get("event_type")
        if event_type not in ALL_EVENT_TYPES:
            raise ContractError(f"v4 ledger event is not allowlisted at line {line_number}")
        claimed = record.get("event_sha256")
        if not _is_sha256(claimed):
            raise ContractError(f"v4 ledger event SHA is invalid at line {line_number}")
        base = {key: value for key, value in record.items() if key != "event_sha256"}
        observed = hashlib.sha256(canonical_json_bytes(base)).hexdigest()
        if claimed != observed:
            raise ContractError(f"v4 ledger event hash mismatch at line {line_number}")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise ContractError(f"v4 ledger payload is invalid at line {line_number}")
        if line_number == 1:
            if event_type != GENESIS_EVENT_TYPE:
                raise ContractError("v4 ledger first event is not genesis")
            _verify_genesis_payload(root, payload)
        else:
            if event_type == GENESIS_EVENT_TYPE:
                raise ContractError("v4 ledger contains duplicate genesis")
            recompute_later_payload(root, event_type, payload)
        previous = claimed
        records.append(record)
    return records


def initialize_ledger(
    root: Path, path: Path, *, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the canonical v4 genesis exactly once with O_EXCL."""

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
        raise ContractError("canonical v4 ledger directory must already exist via QA receipt")
    descriptor = os.open(ledger, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("short v4 genesis write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    persisted = validate_ledger(root, ledger)
    if persisted != [record]:
        raise ContractError("v4 genesis failed replay round-trip validation")
    return record


def append_ledger_event(
    root: Path,
    path: Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay, then append one allowlisted event under an adjacent O_EXCL lock."""

    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v4 later event is not allowlisted: {event_type}")
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    canonical_payload = recompute_later_payload(root, event_type, payload)
    existing = validate_ledger(root, ledger)
    if not existing:
        raise ContractError("v4 ledger must be initialized before append")
    snapshot = (len(existing), existing[-1]["event_sha256"])
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            lock_payload = canonical_json_bytes(
                {
                    "schema_version": "meaningful_score_ledger_v4.append_lock.v1",
                    "ledger": ledger.name,
                    "pid": os.getpid(),
                    "expected_event_count": snapshot[0],
                    "expected_head_event_sha256": snapshot[1],
                }
            )
            if os.write(descriptor, lock_payload) != len(lock_payload):
                raise OSError("short v4 append-lock write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        locked_records = validate_ledger(root, ledger)
        locked_snapshot = (len(locked_records), locked_records[-1]["event_sha256"])
        if locked_snapshot != snapshot:
            raise ContractError("v4 ledger changed before lock acquisition")
        replayed_payload = recompute_later_payload(root, event_type, canonical_payload)
        record = _ledger_record(
            seq=snapshot[0] + 1,
            previous=snapshot[1],
            event_type=event_type,
            payload=replayed_payload,
        )
        encoded = canonical_json_bytes(record) + b"\n"
        descriptor = os.open(ledger, os.O_WRONLY | os.O_APPEND)
        try:
            if os.write(descriptor, encoded) != len(encoded):
                raise OSError("short v4 ledger append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        persisted = validate_ledger(root, ledger)
        if len(persisted) != snapshot[0] + 1 or persisted[-1] != record:
            raise ContractError("v4 append failed replay round-trip validation")
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
    "verify_predecessor",
]
