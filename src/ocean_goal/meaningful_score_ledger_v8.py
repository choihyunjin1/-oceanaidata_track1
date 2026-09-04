"""Ledger-wide completion-lineage closure over the frozen v7 policy revision.

The v7 curve, official-score, and upload policy remains byte-pinned and is
delegated without payload rewriting.  GOAL_COMPLETION is different: its
standalone organizer evidence is only an input claim and is recomputed with
all prior v8 records.  The current contract has no authorized distinct P2
exact-official-confirmation event, so P2 cannot create a meaningful milestone
or a COMPLETE result in this revision.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ocean_goal import meaningful_score_ledger_v5 as ledger_v5
from ocean_goal import meaningful_score_ledger_v7 as ledger_v7
from ocean_goal import meaningful_score_v3 as scoring

ContractError = scoring.ContractError
canonical_json_bytes = scoring.canonical_json_bytes
sha256_file = scoring.sha256_file

CONTRACT_RELATIVE = "configs/goals/meaningful_score_ledger_v8.json"
CONTRACT_SHA256 = "269b3b3f42c6fe356cee9d5685e268131ff01950b84659b7869234579183f6d1"
LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v8/registry.jsonl"
PRE_INIT_QA_RELATIVE = "artifacts/meaningful_score_goal_v8/pre_init_qa.json"

V5_LEDGER_RELATIVE = ledger_v7.V5_LEDGER_RELATIVE
V5_LEDGER_SHA256 = ledger_v7.V5_LEDGER_SHA256
V5_LEDGER_BYTES = ledger_v7.V5_LEDGER_BYTES
V5_LEDGER_EVENT_COUNT = ledger_v7.V5_LEDGER_EVENT_COUNT
V5_LEDGER_HEAD_SHA256 = ledger_v7.V5_LEDGER_HEAD_SHA256
V3_SCORING_CONTRACT = ledger_v7.V3_SCORING_CONTRACT
V3_SCORING_EVALUATOR = ledger_v7.V3_SCORING_EVALUATOR

V7_IMPLEMENTATION = {
    "V7_CONTRACT": {
        "path": "configs/goals/meaningful_score_ledger_v7.json",
        "sha256": "bd71902bd4d00fa925b6fddb15e76a5d3872d242ccbdd1c8f15fd9d7309efddc",
        "bytes": 7644,
    },
    "V7_EVALUATOR": {
        "path": "src/ocean_goal/meaningful_score_ledger_v7.py",
        "sha256": "6cea123ccc181915667643ff1a2d320368255997bec442ba847bb13d9ec29382",
        "bytes": 34661,
    },
    "V7_CLI": {
        "path": "scripts/run_meaningful_score_ledger_v7.py",
        "sha256": "b898fead7ab7b6cd1edb1896c9d33b5baeb3c963fa5177c0d2390ef56eae2b49",
        "bytes": 7866,
    },
    "V7_TESTS": {
        "path": "tests/test_meaningful_score_ledger_v7.py",
        "sha256": "b6d165d6313957f8aa27018c83123da960cccdc6284848123e439c4d579c2ad0",
        "bytes": 16840,
    },
}

GENESIS_EVENT_TYPE = "GOAL_INITIALIZED"
LATER_EVENT_TYPES = frozenset(
    {"CURVE_RESULT", "OFFICIAL_SCORE_RESULT", "UPLOAD_READINESS", "GOAL_COMPLETION"}
)
ALL_EVENT_TYPES = frozenset({GENESIS_EVENT_TYPE, *LATER_EVENT_TYPES})
O_BINARY = getattr(os, "O_BINARY", 0)
ARCHITECTURE_MODE = scoring.ARCHITECTURE_MODE
REQUIRED_CONFIRMATION_EVENT_TYPE = "P2_EXACT_OFFICIAL_CONFIRMATION"

IMPLEMENTATION_RELATIVES = {
    "V8_CONTRACT": CONTRACT_RELATIVE,
    "V8_EVALUATOR": "src/ocean_goal/meaningful_score_ledger_v8.py",
    "V8_CLI": "scripts/run_meaningful_score_ledger_v8.py",
    "V8_TESTS": "tests/test_meaningful_score_ledger_v8.py",
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
        raise ContractError("canonical v8 ledger path must not use symlinks")
    candidate = requested.resolve(strict=must_exist)
    if not candidate.is_relative_to(workspace):
        raise ContractError("v8 ledger escapes workspace")
    if candidate != expected:
        raise ContractError("only the canonical v8 ledger path is accepted")
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
        "path": V5_LEDGER_RELATIVE,
        "sha256": V5_LEDGER_SHA256,
        "bytes": V5_LEDGER_BYTES,
        "event_count": V5_LEDGER_EVENT_COUNT,
        "head_event_sha256": V5_LEDGER_HEAD_SHA256,
    }


def _expected_revision() -> dict[str, Any]:
    return {
        "kind": "APPEND_ONLY_GOAL_COMPLETION_LINEAGE_CLOSURE",
        "supersedes_static_v7_without_mutation": True,
        "reason": (
            "v7 closed P2 architecture curve, official-score, and upload routes but "
            "still delegated standalone GOAL_COMPLETION evidence to v5 without "
            "reconciling prior event lineage."
        ),
        "inherit_v7_curve_official_and_upload_policy_exactly": True,
        "p1_p3_p2_exact_noncompletion_payloads_delegate_bit_for_bit": True,
    }


def _expected_completion_policy() -> dict[str, Any]:
    return {
        "standalone_completion_evidence_authoritative": False,
        "replay_with_all_prior_v8_events_required": True,
        "p2_architecture_events_cannot_satisfy_meaningful_milestone": True,
        "p2_meaningful_milestone_requires_distinct_event_type": (
            REQUIRED_CONFIRMATION_EVENT_TYPE
        ),
        "required_event_type_currently_allowlisted": False,
        "required_event_type_currently_policy_authorized": False,
        "current_p2_milestone_authorized": False,
        "current_goal_completion_result": "NOT_COMPLETE",
        "future_authorization_requires_new_append_only_contract_revision": True,
    }


def load_contract(root: Path, relative: str = CONTRACT_RELATIVE) -> dict[str, Any]:
    if Path(relative).as_posix() != CONTRACT_RELATIVE:
        raise ContractError("only the canonical v8 ledger contract is accepted")
    path = _workspace_path(root, relative)
    if _file_pin_for_path(root, path)["path"] != CONTRACT_RELATIVE:
        raise ContractError("canonical v8 ledger contract path uses a symlink")
    if sha256_file(path) != CONTRACT_SHA256:
        raise ContractError("canonical v8 ledger contract SHA mismatch")
    contract = _json_object(path)
    if contract.get("schema_version") != "meaningful_score_ledger.v8":
        raise ContractError("v8 ledger schema identity changed")
    if contract.get("revision") != _expected_revision():
        raise ContractError("v8 completion-lineage revision changed")
    if contract.get("completion_lineage_policy") != _expected_completion_policy():
        raise ContractError("v8 completion-lineage policy changed")
    if contract.get("scoring_contract") != V3_SCORING_CONTRACT:
        raise ContractError("v3 scoring-contract pin changed")
    if contract.get("scoring_evaluator") != V3_SCORING_EVALUATOR:
        raise ContractError("v3 scoring-evaluator pin changed")
    if contract.get("predecessor_ledger") != _predecessor_anchor():
        raise ContractError("v5 predecessor anchor changed")
    if contract.get("superseded_v7_implementation") != V7_IMPLEMENTATION:
        raise ContractError("superseded v7 implementation pins changed")
    if contract.get("lineage_source") != {
        "policy_and_p2_stage_a_v3_lineage": "PINNED_V7_CONTRACT_AND_DEEP_REPLAY",
        "v7_canonical_ledger_must_remain_absent": True,
        "v7_pre_init_qa_must_remain_absent": True,
    }:
        raise ContractError("v8 lineage source changed")
    if contract.get("canonical_paths") != {
        "ledger": LEDGER_RELATIVE,
        "pre_init_qa": PRE_INIT_QA_RELATIVE,
    }:
        raise ContractError("v8 canonical paths changed")
    protocol = contract.get("event_protocol")
    if not isinstance(protocol, Mapping):
        raise ContractError("v8 event protocol is missing")
    if protocol.get("genesis_event_type") != GENESIS_EVENT_TYPE or protocol.get(
        "later_event_types"
    ) != ["CURVE_RESULT", "OFFICIAL_SCORE_RESULT", "UPLOAD_READINESS", "GOAL_COMPLETION"]:
        raise ContractError("v8 typed event allowlist changed")
    for key in (
        "unknown_event_types_forbidden",
        "payload_must_deep_equal_recomputed_payload",
        "evidence_must_be_workspace_relative_sha256_and_size_pinned",
        "replay_every_event_on_every_validation",
        "completion_replay_receives_all_prior_records",
        "replay_before_append_lock_or_write",
        "all_os_open_calls_use_o_binary",
        "all_os_write_calls_use_robust_write_loop",
    ):
        if protocol.get(key) is not True:
            raise ContractError(f"v8 protocol weakened: {key}")
    if protocol.get("append_lock") != "O_EXCL_ADJACENT_LOCK":
        raise ContractError("v8 append lock changed")
    if protocol.get("genesis_creation") != "O_EXCL":
        raise ContractError("v8 genesis creation changed")
    if protocol.get("canonical_line_ending") != "LF_ONLY":
        raise ContractError("v8 canonical line ending changed")
    if REQUIRED_CONFIRMATION_EVENT_TYPE in ALL_EVENT_TYPES:
        raise ContractError("v8 accidentally allowlisted the future confirmation event")
    return contract


def _verify_v7_implementation(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if contract.get("superseded_v7_implementation") != V7_IMPLEMENTATION:
        raise ContractError("superseded v7 pins differ from v8 contract")
    verified: dict[str, dict[str, Any]] = {}
    for role, expected in V7_IMPLEMENTATION.items():
        pin, _ = _verify_file_pin(root, expected, role=role)
        verified[role] = pin
    ledger_v7.load_contract(root)
    return verified


def verify_predecessor(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("predecessor_ledger") != _predecessor_anchor():
        raise ContractError("v5 predecessor differs from v8 contract")
    v7_pins = _verify_v7_implementation(root, contract)
    for relative in (
        ledger_v7.LEDGER_RELATIVE,
        ledger_v7.PRE_INIT_QA_RELATIVE,
        f"{ledger_v7.LEDGER_RELATIVE}.append.lock",
    ):
        if _workspace_path(root, relative, must_exist=False).exists():
            raise ContractError("superseded v7 canonical state must remain uninitialized")
    v7_contract = ledger_v7.load_contract(root)
    inherited = ledger_v7.verify_predecessor(root, v7_contract)
    return {"V7_STATIC_IMPLEMENTATION": v7_pins, "INHERITED_V7_LINEAGE": inherited}


def verify_p2_stage_a_v3_lineage(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    _verify_v7_implementation(root, contract)
    return ledger_v7.verify_p2_stage_a_v3_lineage(root, ledger_v7.load_contract(root))


def current_implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    pins = {
        role: _file_pin_for_path(root, _workspace_path(root, relative))
        for role, relative in IMPLEMENTATION_RELATIVES.items()
    }
    if set(pins) != IMPLEMENTATION_ROLES:
        raise ContractError("v8 implementation role set changed")
    return pins


def _canonical_qa_pin(root: Path, requested: Path) -> dict[str, Any]:
    workspace = _workspace(root)
    lexical = workspace / PRE_INIT_QA_RELATIVE
    expected = lexical.resolve(strict=False)
    if expected != lexical:
        raise ContractError("canonical v8 QA path must not use symlinks")
    resolved = requested.resolve(strict=True)
    if not resolved.is_relative_to(workspace) or resolved != expected:
        raise ContractError("v8 pre-init QA receipt path must be canonical")
    return _file_pin_for_path(root, resolved)


def _ledger_contract_pin(root: Path) -> dict[str, Any]:
    return _file_pin_for_path(root, _workspace_path(root, CONTRACT_RELATIVE))


def build_genesis_payload(root: Path, qa_receipt: Path) -> dict[str, Any]:
    contract = load_contract(root)
    predecessor_pins = verify_predecessor(root, contract)
    lineage_pins = verify_p2_stage_a_v3_lineage(root, contract)
    implementation_pins = current_implementation_pins(root)
    qa_pin = _canonical_qa_pin(root, qa_receipt)
    receipt = _json_object(_workspace_path(root, qa_pin["path"]))
    if (
        receipt.get("schema_version") != "meaningful_score_ledger_v8.pre_init_qa.v1"
        or receipt.get("decision") != "GO_INITIALIZE_V8_LEDGER"
        or receipt.get("p0_count") != 0
        or receipt.get("p1_count") != 0
        or receipt.get("ledger_contract") != _ledger_contract_pin(root)
        or receipt.get("predecessor_ledger_anchor") != _predecessor_anchor()
        or receipt.get("verified_predecessor_pins") != predecessor_pins
        or receipt.get("p2_stage_a_v3_lineage_pins") != lineage_pins
        or receipt.get("implementation_pins") != implementation_pins
    ):
        raise ContractError("v8 pre-init QA receipt does not bind current genesis")
    initial = contract["initial_state"]
    return {
        "ledger_id": contract["ledger_id"],
        "status": initial["status"],
        "ledger_contract": _ledger_contract_pin(root),
        "verified_predecessor_pins": predecessor_pins,
        "predecessor_ledger_anchor": _predecessor_anchor(),
        "p2_stage_a_v3_lineage_pins": lineage_pins,
        "implementation_pins": implementation_pins,
        "independent_pre_init_qa": qa_pin,
        "inherited_v5_event_count": initial["inherited_v5_event_count"],
        "official_uploads": initial["official_uploads"],
        "score_promotions": initial["score_promotions"],
        "meaningful_promotions": initial["meaningful_promotions"],
        "execution_counts": initial["execution_counts"],
        "upload_performed": False,
    }


def _verify_genesis_payload(root: Path, payload: Mapping[str, Any]) -> None:
    qa = payload.get("independent_pre_init_qa")
    if not isinstance(qa, Mapping) or qa.get("path") != PRE_INIT_QA_RELATIVE:
        raise ContractError("v8 genesis canonical QA pin is missing")
    _, qa_path = _verify_file_pin(root, qa, role="v8 pre-init QA")
    if dict(payload) != build_genesis_payload(root, qa_path):
        raise ContractError("v8 genesis payload differs from canonical recomputation")


def _source_json_from_pin(
    root: Path, value: Any, *, role: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{role} pin is missing")
    pin, path = _verify_file_pin(root, value, role=role)
    return _json_object(path), pin, path


def _verify_inherited_policy(root: Path) -> dict[str, Any]:
    contract = load_contract(root)
    _verify_v7_implementation(root, contract)
    return contract


def build_curve_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    _verify_inherited_policy(root)
    return ledger_v7.build_curve_payload(root, evidence_path)


def build_official_score_payload(
    root: Path, evidence_path: Path, curve_decision_path: Path
) -> dict[str, Any]:
    _verify_inherited_policy(root)
    return ledger_v7.build_official_score_payload(root, evidence_path, curve_decision_path)


def build_upload_readiness_payload(
    root: Path, receipt_path: Path, curve_decision_path: Path | None
) -> dict[str, Any]:
    _verify_inherited_policy(root)
    return ledger_v7.build_upload_readiness_payload(root, receipt_path, curve_decision_path)


def _record_decision(record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return {}
    if record.get("event_type") == "UPLOAD_READINESS":
        value = payload.get("readiness")
    else:
        value = payload.get("decision")
    return value if isinstance(value, Mapping) else {}


def _completion_lineage(prior_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    architecture_seqs: list[int] = []
    exact_official_score_seqs: list[int] = []
    previous_seq = 0
    head = V5_LEDGER_HEAD_SHA256
    for record in prior_records:
        if not isinstance(record, Mapping):
            raise ContractError("completion lineage record must be an object")
        seq = record.get("seq")
        event_type = record.get("event_type")
        event_sha = record.get("event_sha256")
        if (
            not isinstance(seq, int)
            or isinstance(seq, bool)
            or seq != previous_seq + 1
            or event_type not in ALL_EVENT_TYPES
            or not _is_sha256(event_sha)
        ):
            raise ContractError("completion lineage records are not a valid v8 prefix")
        decision = _record_decision(record)
        if decision.get("problem") == "P2" and decision.get("comparison_mode") == (
            ARCHITECTURE_MODE
        ):
            architecture_seqs.append(seq)
        if (
            event_type == "OFFICIAL_SCORE_RESULT"
            and decision.get("problem") == "P2"
            and decision.get("comparison_mode", scoring.EXACT_MODE) == scoring.EXACT_MODE
        ):
            exact_official_score_seqs.append(seq)
        previous_seq = seq
        head = event_sha
    return {
        "prior_event_count": len(prior_records),
        "prior_head_event_sha256": head,
        "p2_architecture_event_seqs": architecture_seqs,
        "p2_exact_official_score_event_seqs": exact_official_score_seqs,
        "required_confirmation_event_type": REQUIRED_CONFIRMATION_EVENT_TYPE,
        "required_confirmation_event_count": 0,
        "required_confirmation_event_allowlisted": False,
        "required_confirmation_policy_authorized": False,
        "p2_milestone_authorized": False,
    }


def build_goal_completion_payload(
    root: Path,
    evidence_path: Path,
    *,
    prior_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    contract = _verify_inherited_policy(root)
    base = ledger_v5.build_goal_completion_payload(root, evidence_path)
    decision = base.get("decision")
    if not isinstance(decision, Mapping):
        raise ContractError("v5 completion evaluator returned an invalid decision")
    problem_checks = decision.get("problem_checks")
    global_checks = decision.get("global_checks")
    if (
        not isinstance(problem_checks, Mapping)
        or not isinstance(problem_checks.get("P2"), Mapping)
        or not isinstance(global_checks, Mapping)
    ):
        raise ContractError("v5 completion decision omitted required checks")
    lineage = _completion_lineage(prior_records)
    claimed_complete = decision.get("goal_complete") is True
    claimed_p2_milestone = (
        problem_checks["P2"].get("meaningful_promotion_at_least_once") is True
    )
    closed_problem_checks = {
        **problem_checks,
        "P2": {
            **problem_checks["P2"],
            "meaningful_promotion_at_least_once": False,
            "exact_official_confirmation_event_authorized": False,
        },
    }
    closed_global_checks = {
        **global_checks,
        "all_problem_checks_pass": False,
        "p2_milestone_authorized_by_event_lineage": False,
    }
    closed_decision = {
        **decision,
        "schema_version": "goal_completion_decision.v2_lineage_closed.v1",
        "decision": "NOT_COMPLETE",
        "goal_complete": False,
        "problem_checks": closed_problem_checks,
        "global_checks": closed_global_checks,
        "p2_milestone_authorized": False,
    }
    return {
        "schema_version": "meaningful_score_ledger_v8.goal_completion.v1",
        "evidence": base["evidence"],
        "lineage": lineage,
        "lineage_policy": contract["completion_lineage_policy"],
        "standalone_evidence_claims": {
            "goal_complete": claimed_complete,
            "p2_meaningful_promotion_at_least_once": claimed_p2_milestone,
            "authoritative": False,
        },
        "decision": closed_decision,
        "upload_performed": False,
    }


def recompute_later_payload(
    root: Path,
    event_type: str,
    payload: Mapping[str, Any],
    *,
    prior_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v8 event type is not allowlisted: {event_type}")
    if not isinstance(payload, Mapping):
        raise ContractError("v8 event payload must be an object")
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
        if prior_records is None:
            raise ContractError("GOAL_COMPLETION replay requires all prior v8 records")
        _, _, evidence = _source_json_from_pin(
            root, payload.get("evidence"), role="completion evidence"
        )
        expected = build_goal_completion_payload(
            root, evidence, prior_records=prior_records
        )
    if dict(payload) != expected:
        raise ContractError(f"{event_type} payload differs from replay recomputation")
    return expected


def _ledger_record(
    *, seq: int, previous: str, event_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if event_type not in ALL_EVENT_TYPES:
        raise ContractError(f"v8 event type is not allowlisted: {event_type}")
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
    verify_p2_stage_a_v3_lineage(root, contract)
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    if not ledger.exists():
        return []
    raw = ledger.read_bytes()
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ContractError("v8 ledger must be non-empty canonical LF-terminated JSONL")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("v8 ledger is not UTF-8") from error
    records: list[dict[str, Any]] = []
    previous = V5_LEDGER_HEAD_SHA256
    for line_number, line in enumerate(text.split("\n")[:-1], 1):
        if not line.strip():
            raise ContractError(f"blank v8 ledger line {line_number}")
        record = _strict_json_loads(line)
        if not isinstance(record, dict) or set(record) != {
            "seq",
            "recorded_at_kst",
            "event_type",
            "previous_event_sha256",
            "payload",
            "event_sha256",
        }:
            raise ContractError(f"v8 ledger record keys changed at line {line_number}")
        if (
            not isinstance(record.get("seq"), int)
            or isinstance(record.get("seq"), bool)
            or record["seq"] != len(records) + 1
        ):
            raise ContractError(f"v8 ledger sequence mismatch at line {line_number}")
        if record.get("previous_event_sha256") != previous:
            raise ContractError(f"v8 ledger chain mismatch at line {line_number}")
        timestamp = record.get("recorded_at_kst")
        if not isinstance(timestamp, str):
            raise ContractError(f"v8 ledger timestamp is invalid at line {line_number}")
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ContractError(f"v8 ledger timestamp is invalid at line {line_number}") from error
        if parsed.utcoffset() != timedelta(hours=9):
            raise ContractError(f"v8 ledger timestamp is not KST at line {line_number}")
        event_type = record.get("event_type")
        if event_type not in ALL_EVENT_TYPES:
            raise ContractError(f"v8 ledger event is not allowlisted at line {line_number}")
        claimed = record.get("event_sha256")
        if not _is_sha256(claimed):
            raise ContractError(f"v8 event SHA is invalid at line {line_number}")
        base = {key: value for key, value in record.items() if key != "event_sha256"}
        if hashlib.sha256(canonical_json_bytes(base)).hexdigest() != claimed:
            raise ContractError(f"v8 event hash mismatch at line {line_number}")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise ContractError(f"v8 payload is invalid at line {line_number}")
        if line_number == 1:
            if event_type != GENESIS_EVENT_TYPE:
                raise ContractError("v8 first event is not genesis")
            _verify_genesis_payload(root, payload)
        else:
            if event_type == GENESIS_EVENT_TYPE:
                raise ContractError("v8 ledger contains duplicate genesis")
            recompute_later_payload(
                root, event_type, payload, prior_records=tuple(records)
            )
        previous = claimed
        records.append(record)
    return records


def _write_all(descriptor: int, payload: bytes, *, role: str) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError(f"short v8 {role} write")
        offset += written


def initialize_ledger(
    root: Path, path: Path, *, payload: Mapping[str, Any]
) -> dict[str, Any]:
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    _verify_genesis_payload(root, payload)
    record = _ledger_record(
        seq=1,
        previous=V5_LEDGER_HEAD_SHA256,
        event_type=GENESIS_EVENT_TYPE,
        payload=payload,
    )
    encoded = canonical_json_bytes(record) + b"\n"
    if not ledger.parent.is_dir():
        raise ContractError("canonical v8 ledger directory must already exist via QA receipt")
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
        raise ContractError("v8 genesis failed replay round-trip validation")
    return record


def append_ledger_event(
    root: Path,
    path: Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v8 later event is not allowlisted: {event_type}")
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    existing = validate_ledger(root, ledger)
    if not existing:
        raise ContractError("v8 ledger must be initialized before append")
    canonical_payload = recompute_later_payload(
        root, event_type, payload, prior_records=tuple(existing)
    )
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
                    "schema_version": "meaningful_score_ledger_v8.append_lock.v1",
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
            raise ContractError("v8 ledger changed before lock acquisition")
        replayed = recompute_later_payload(
            root, event_type, canonical_payload, prior_records=tuple(locked)
        )
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
            raise ContractError("v8 append failed replay round-trip validation")
        return record
    finally:
        lock.unlink(missing_ok=True)


__all__ = [
    "ALL_EVENT_TYPES",
    "ARCHITECTURE_MODE",
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
    "REQUIRED_CONFIRMATION_EVENT_TYPE",
    "V5_LEDGER_HEAD_SHA256",
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
    "verify_p2_stage_a_v3_lineage",
    "verify_predecessor",
]
