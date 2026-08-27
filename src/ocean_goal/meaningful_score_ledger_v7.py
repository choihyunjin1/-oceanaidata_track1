"""Policy-closed append-only ledger for P2 architecture-matched evidence.

v7 preserves the immutable v3/v5 lineage and the uninitialized v6 static
implementation.  It closes the two post-curve routes that v6 still delegated:
architecture-matched official scores can record an ordinary paired score
observation but can never create a meaningful promotion, and architecture-
matched challenger upload readiness always fails closed.
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

from ocean_goal import meaningful_score_ledger_v5 as ledger_v5
from ocean_goal import meaningful_score_ledger_v6 as ledger_v6
from ocean_goal import meaningful_score_v3 as scoring

ContractError = scoring.ContractError
canonical_json_bytes = scoring.canonical_json_bytes
sha256_file = scoring.sha256_file

CONTRACT_RELATIVE = "configs/goals/meaningful_score_ledger_v7.json"
CONTRACT_SHA256 = "bd71902bd4d00fa925b6fddb15e76a5d3872d242ccbdd1c8f15fd9d7309efddc"
LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v7/registry.jsonl"
PRE_INIT_QA_RELATIVE = "artifacts/meaningful_score_goal_v7/pre_init_qa.json"

V5_LEDGER_RELATIVE = ledger_v6.V5_LEDGER_RELATIVE
V5_LEDGER_SHA256 = ledger_v6.V5_LEDGER_SHA256
V5_LEDGER_BYTES = ledger_v6.V5_LEDGER_BYTES
V5_LEDGER_EVENT_COUNT = ledger_v6.V5_LEDGER_EVENT_COUNT
V5_LEDGER_HEAD_SHA256 = ledger_v6.V5_LEDGER_HEAD_SHA256
V3_SCORING_CONTRACT = ledger_v6.V3_SCORING_CONTRACT
V3_SCORING_EVALUATOR = ledger_v6.V3_SCORING_EVALUATOR

V6_IMPLEMENTATION = {
    "V6_CONTRACT": {
        "path": "configs/goals/meaningful_score_ledger_v6.json",
        "sha256": "a0aa5b04d57cc74552b58f94f6eb79c078ce48722e7351f21f3007f0ca8264f8",
        "bytes": 6560,
    },
    "V6_EVALUATOR": {
        "path": "src/ocean_goal/meaningful_score_ledger_v6.py",
        "sha256": "2162695cb178aec1f639a66d9f07b9156977d4154cafc0449d281836ce7146b3",
        "bytes": 36256,
    },
    "V6_CLI": {
        "path": "scripts/run_meaningful_score_ledger_v6.py",
        "sha256": "2f5065db2e4300afa53375973cb09869623cb9b4fd15cedac890d9e7be392620",
        "bytes": 7766,
    },
    "V6_TESTS": {
        "path": "tests/test_meaningful_score_ledger_v6.py",
        "sha256": "f0872cf214060d1185c7cbb8caeac82c10205ff4f3df4c3752f43570bd768d5a",
        "bytes": 13921,
    },
}

GENESIS_EVENT_TYPE = "GOAL_INITIALIZED"
LATER_EVENT_TYPES = frozenset(
    {"CURVE_RESULT", "OFFICIAL_SCORE_RESULT", "UPLOAD_READINESS", "GOAL_COMPLETION"}
)
ALL_EVENT_TYPES = frozenset({GENESIS_EVENT_TYPE, *LATER_EVENT_TYPES})
O_BINARY = getattr(os, "O_BINARY", 0)
ARCHITECTURE_MODE = scoring.ARCHITECTURE_MODE
ARCHITECTURE_UPLOAD_ROLE = "ARCHITECTURE_MATCHED_CHALLENGER"
P2_OFFICIAL_DECISION = "RESEARCH_ONLY_PAIRED_AB_PENDING"
P2_UPLOAD_STATUS = "FAIL_CLOSED_RESEARCH_ONLY_P2_ARCHITECTURE_MATCHED"

IMPLEMENTATION_RELATIVES = {
    "V7_CONTRACT": CONTRACT_RELATIVE,
    "V7_EVALUATOR": "src/ocean_goal/meaningful_score_ledger_v7.py",
    "V7_CLI": "scripts/run_meaningful_score_ledger_v7.py",
    "V7_TESTS": "tests/test_meaningful_score_ledger_v7.py",
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
        raise ContractError("canonical v7 ledger path must not use symlinks")
    candidate = requested.resolve(strict=must_exist)
    if not candidate.is_relative_to(workspace):
        raise ContractError("v7 ledger escapes workspace")
    if candidate != expected:
        raise ContractError("only the canonical v7 ledger path is accepted")
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
        "kind": "APPEND_ONLY_P2_ARCHITECTURE_POLICY_CLOSURE",
        "supersedes_static_v6_without_mutation": True,
        "reason": (
            "v6 closed the curve-result path but delegated architecture-matched "
            "official-score and upload-readiness paths to v5, allowing a later "
            "local-qualification proof to bypass the research-only policy."
        ),
        "standard_event_types_only": True,
        "p1_p3_exact_payloads_delegate_bit_for_bit_to_v5": True,
        "p2_exact_payloads_delegate_bit_for_bit_to_v5": True,
    }


def _expected_p2_policy() -> dict[str, Any]:
    return {
        "curve_result": {
            "delegate_to_v6_compatibility_replay": True,
            "decision": "RESEARCH_ONLY",
            "passed": False,
            "official_promotion": False,
            "preserve_local_qualification": True,
        },
        "official_score_result": {
            "numeric_and_operational_evaluation": "V3_PAIRED_AB_REPLAY",
            "score_incumbent_policy": "ALLOW_GENERAL_PAIRED_OFFICIAL_SCORE_RECORD_ONLY",
            "score_incumbent_updates_may_reflect_v3_operational_checks": True,
            "score_incumbent_update_is_not_meaningful_promotion": True,
            "decision": P2_OFFICIAL_DECISION,
            "meaningful_incumbent_updates": False,
            "official_promotion": False,
            "goal_problem_milestone_complete": False,
        },
        "upload_readiness": {
            "status": P2_UPLOAD_STATUS,
            "upload_ready": False,
            "upload_performed": False,
            "no_v3_readiness_delegation": True,
            "immutable_baseline_anchor_and_nonarchitecture_routes_delegate_to_v5": True,
        },
    }


def load_contract(root: Path, relative: str = CONTRACT_RELATIVE) -> dict[str, Any]:
    if Path(relative).as_posix() != CONTRACT_RELATIVE:
        raise ContractError("only the canonical v7 ledger contract is accepted")
    path = _workspace_path(root, relative)
    if _file_pin_for_path(root, path)["path"] != CONTRACT_RELATIVE:
        raise ContractError("canonical v7 ledger contract path uses a symlink")
    if sha256_file(path) != CONTRACT_SHA256:
        raise ContractError("canonical v7 ledger contract SHA mismatch")
    contract = _json_object(path)
    if contract.get("schema_version") != "meaningful_score_ledger.v7":
        raise ContractError("v7 ledger schema identity changed")
    if contract.get("revision") != _expected_revision():
        raise ContractError("v7 policy-closure revision changed")
    if contract.get("p2_architecture_matched_policy") != _expected_p2_policy():
        raise ContractError("v7 P2 architecture policy changed")
    if contract.get("scoring_contract") != V3_SCORING_CONTRACT:
        raise ContractError("v3 scoring-contract pin changed")
    if contract.get("scoring_evaluator") != V3_SCORING_EVALUATOR:
        raise ContractError("v3 scoring-evaluator pin changed")
    if contract.get("predecessor_ledger") != _predecessor_anchor():
        raise ContractError("v5 predecessor anchor changed")
    if contract.get("superseded_v6_implementation") != V6_IMPLEMENTATION:
        raise ContractError("superseded v6 implementation pins changed")
    lineage = contract.get("p2_stage_a_v3_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != ledger_v6.STAGE_A_ROLES:
        raise ContractError("P2 Stage-A v3 lineage role set changed")
    if contract.get("canonical_paths") != {
        "ledger": LEDGER_RELATIVE,
        "pre_init_qa": PRE_INIT_QA_RELATIVE,
    }:
        raise ContractError("v7 canonical paths changed")
    protocol = contract.get("event_protocol")
    if not isinstance(protocol, Mapping):
        raise ContractError("v7 event protocol is missing")
    if protocol.get("genesis_event_type") != GENESIS_EVENT_TYPE or protocol.get(
        "later_event_types"
    ) != ["CURVE_RESULT", "OFFICIAL_SCORE_RESULT", "UPLOAD_READINESS", "GOAL_COMPLETION"]:
        raise ContractError("v7 typed event allowlist changed")
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
            raise ContractError(f"v7 protocol weakened: {key}")
    if protocol.get("append_lock") != "O_EXCL_ADJACENT_LOCK":
        raise ContractError("v7 append lock changed")
    if protocol.get("genesis_creation") != "O_EXCL":
        raise ContractError("v7 genesis creation changed")
    if protocol.get("canonical_line_ending") != "LF_ONLY":
        raise ContractError("v7 canonical line ending changed")
    return contract


def _verify_v6_implementation(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if contract.get("superseded_v6_implementation") != V6_IMPLEMENTATION:
        raise ContractError("superseded v6 pins differ from v7 contract")
    verified: dict[str, dict[str, Any]] = {}
    for role, expected in V6_IMPLEMENTATION.items():
        pin, _ = _verify_file_pin(root, expected, role=role)
        verified[role] = pin
    ledger_v6.load_contract(root)
    return verified


def verify_predecessor(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("predecessor_ledger") != _predecessor_anchor():
        raise ContractError("v5 predecessor differs from v7 contract")
    v6_pins = _verify_v6_implementation(root, contract)
    v6_contract = ledger_v6.load_contract(root)
    if contract.get("p2_stage_a_v3_lineage") != v6_contract.get("p2_stage_a_v3_lineage"):
        raise ContractError("v7 lineage differs from frozen v6 compatibility lineage")
    for relative in (
        ledger_v6.LEDGER_RELATIVE,
        ledger_v6.PRE_INIT_QA_RELATIVE,
        f"{ledger_v6.LEDGER_RELATIVE}.append.lock",
    ):
        if _workspace_path(root, relative, must_exist=False).exists():
            raise ContractError("superseded v6 canonical state must remain uninitialized")
    v5_lineage = ledger_v6.verify_predecessor(root, v6_contract)
    return {"V6_STATIC_IMPLEMENTATION": v6_pins, "V5_LINEAGE": v5_lineage}


def verify_p2_stage_a_v3_lineage(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    v6_contract = ledger_v6.load_contract(root)
    if contract.get("p2_stage_a_v3_lineage") != v6_contract.get("p2_stage_a_v3_lineage"):
        raise ContractError("v7 P2 lineage differs from frozen v6")
    return ledger_v6.verify_p2_stage_a_v3_lineage(root, v6_contract)


def current_implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    pins = {
        role: _file_pin_for_path(root, _workspace_path(root, relative))
        for role, relative in IMPLEMENTATION_RELATIVES.items()
    }
    if set(pins) != IMPLEMENTATION_ROLES:
        raise ContractError("v7 implementation role set changed")
    return pins


def _canonical_qa_pin(root: Path, requested: Path) -> dict[str, Any]:
    workspace = _workspace(root)
    lexical = workspace / PRE_INIT_QA_RELATIVE
    expected = lexical.resolve(strict=False)
    if expected != lexical:
        raise ContractError("canonical v7 QA path must not use symlinks")
    resolved = requested.resolve(strict=True)
    if not resolved.is_relative_to(workspace) or resolved != expected:
        raise ContractError("v7 pre-init QA receipt path must be canonical")
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
        receipt.get("schema_version") != "meaningful_score_ledger_v7.pre_init_qa.v1"
        or receipt.get("decision") != "GO_INITIALIZE_V7_LEDGER"
        or receipt.get("p0_count") != 0
        or receipt.get("p1_count") != 0
        or receipt.get("ledger_contract") != _ledger_contract_pin(root)
        or receipt.get("predecessor_ledger_anchor") != _predecessor_anchor()
        or receipt.get("verified_predecessor_pins") != predecessor_pins
        or receipt.get("p2_stage_a_v3_lineage_pins") != lineage_pins
        or receipt.get("implementation_pins") != implementation_pins
    ):
        raise ContractError("v7 pre-init QA receipt does not bind current genesis")
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
        raise ContractError("v7 genesis canonical QA pin is missing")
    _, qa_path = _verify_file_pin(root, qa, role="v7 pre-init QA")
    if dict(payload) != build_genesis_payload(root, qa_path):
        raise ContractError("v7 genesis payload differs from canonical recomputation")


def _source_json_from_pin(
    root: Path, value: Any, *, role: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{role} pin is missing")
    pin, path = _verify_file_pin(root, value, role=role)
    return _json_object(path), pin, path


def _pinned_json_from_path(
    root: Path, path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    pin = _file_pin_for_path(root, path)
    canonical = _workspace_path(root, pin["path"])
    return _json_object(canonical), pin, canonical


def _verify_policy_dependencies(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_contract(root)
    _verify_v6_implementation(root, contract)
    _verify_file_pin(root, V3_SCORING_CONTRACT, role="v3 scoring contract")
    _verify_file_pin(root, V3_SCORING_EVALUATOR, role="v3 scoring evaluator")
    lineage = verify_p2_stage_a_v3_lineage(root, contract)
    return contract, lineage


def _validate_p2_architecture_curve(curve: Mapping[str, Any]) -> None:
    if (
        curve.get("problem") != "P2"
        or curve.get("comparison_mode") != ARCHITECTURE_MODE
        or curve.get("exact_official_incumbent_comparison") is not False
        or curve.get("decision") != "RESEARCH_ONLY"
        or curve.get("passed") is not False
        or curve.get("official_promotion") is not False
        or curve.get("curve_alone_can_promote") is not False
        or not isinstance(curve.get("local_qualification"), bool)
    ):
        raise ContractError("P2 architecture curve does not satisfy the v7 research-only policy")


def _architecture_official_route(
    evidence: Mapping[str, Any], curve: Mapping[str, Any]
) -> bool:
    evidence_mode = evidence.get("comparison_mode", scoring.EXACT_MODE)
    curve_mode = curve.get("comparison_mode", scoring.EXACT_MODE)
    architecture = evidence_mode == ARCHITECTURE_MODE or curve_mode == ARCHITECTURE_MODE
    if not architecture:
        return False
    if (
        evidence_mode != ARCHITECTURE_MODE
        or curve_mode != ARCHITECTURE_MODE
        or evidence.get("problem") != "P2"
        or curve.get("problem") != "P2"
    ):
        raise ContractError("P2 architecture official route cannot be downgraded or mismatched")
    _validate_p2_architecture_curve(curve)
    return True


def build_curve_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    contract = load_contract(root)
    _verify_v6_implementation(root, contract)
    return ledger_v6.build_curve_payload(root, evidence_path)


def build_official_score_payload(
    root: Path, evidence_path: Path, curve_decision_path: Path
) -> dict[str, Any]:
    evidence, evidence_pin, canonical_evidence = _pinned_json_from_path(root, evidence_path)
    curve, curve_pin, canonical_curve = _pinned_json_from_path(root, curve_decision_path)
    if not _architecture_official_route(evidence, curve):
        return ledger_v5.build_official_score_payload(
            root, canonical_evidence, canonical_curve
        )
    contract, lineage = _verify_policy_dependencies(root)
    scoring_contract = scoring.load_contract(root)
    evidence_pins = scoring.verify_official_evidence_pins(root, scoring_contract, evidence)
    evaluated = scoring.evaluate_official_score(scoring_contract, curve, evidence)
    if not isinstance(evaluated, Mapping) or not isinstance(
        evaluated.get("score_incumbent_updates"), bool
    ):
        raise ContractError("v3 official evaluator returned an invalid score-incumbent result")
    meaningful_checks = evaluated.get("meaningful_checks")
    if not isinstance(meaningful_checks, Mapping):
        raise ContractError("v3 official evaluator omitted meaningful checks")
    decision = {
        **evaluated,
        "schema_version": "official_score_decision.v3_policy_closed.v1",
        "decision": P2_OFFICIAL_DECISION,
        "meaningful_incumbent_updates": False,
        "goal_problem_milestone_complete": False,
        "official_promotion": False,
        "local_qualification": curve["local_qualification"],
        "meaningful_checks": {
            **meaningful_checks,
            "v7_research_only_policy_allows_meaningful_promotion": False,
        },
        "score_incumbent_policy": "ALLOW_GENERAL_PAIRED_OFFICIAL_SCORE_RECORD_ONLY",
        "score_incumbent_update_is_not_meaningful_promotion": True,
    }
    return {
        "schema_version": "meaningful_score_ledger_v7.official_score_result.v1",
        "evidence": evidence_pin,
        "curve_decision": curve_pin,
        "evidence_pins": {
            "official_paired_ab": evidence_pins,
            "p2_stage_a_v3_lineage": lineage,
        },
        "policy": contract["p2_architecture_matched_policy"]["official_score_result"],
        "decision": decision,
        "upload_performed": False,
    }


def _architecture_upload_route(
    receipt: Mapping[str, Any], curve: Mapping[str, Any] | None
) -> bool:
    receipt_architecture = (
        receipt.get("role") == ARCHITECTURE_UPLOAD_ROLE
        or receipt.get("comparison_mode") == ARCHITECTURE_MODE
    )
    curve_architecture = curve is not None and curve.get("comparison_mode") == ARCHITECTURE_MODE
    if not receipt_architecture and not curve_architecture:
        return False
    if (
        curve is None
        or receipt.get("role") != ARCHITECTURE_UPLOAD_ROLE
        or receipt.get("problem") != "P2"
        or curve.get("problem") != "P2"
        or curve.get("comparison_mode") != ARCHITECTURE_MODE
    ):
        raise ContractError("P2 architecture upload route is inconsistent and fail-closed")
    _validate_p2_architecture_curve(curve)
    return True


def build_upload_readiness_payload(
    root: Path, receipt_path: Path, curve_decision_path: Path | None
) -> dict[str, Any]:
    receipt, receipt_pin, canonical_receipt = _pinned_json_from_path(root, receipt_path)
    if curve_decision_path is None:
        curve = None
        curve_pin = None
        canonical_curve = None
    else:
        curve, curve_pin, canonical_curve = _pinned_json_from_path(root, curve_decision_path)
    if not _architecture_upload_route(receipt, curve):
        return ledger_v5.build_upload_readiness_payload(
            root, canonical_receipt, canonical_curve
        )
    contract, lineage = _verify_policy_dependencies(root)
    assert curve is not None
    readiness = {
        "schema_version": "upload_readiness_receipt.v3_policy_closed.v1",
        "status": P2_UPLOAD_STATUS,
        "problem": "P2",
        "role": ARCHITECTURE_UPLOAD_ROLE,
        "comparison_mode": ARCHITECTURE_MODE,
        "local_qualification": curve["local_qualification"],
        "upload_ready": False,
        "upload_performed": False,
        "reason": "P2 architecture-matched evidence remains research-only in v7",
    }
    return {
        "schema_version": "meaningful_score_ledger_v7.upload_readiness.v1",
        "receipt": receipt_pin,
        "curve_decision": curve_pin,
        "evidence_pins": {"p2_stage_a_v3_lineage": lineage},
        "policy": contract["p2_architecture_matched_policy"]["upload_readiness"],
        "readiness": readiness,
        "upload_performed": False,
    }


def build_goal_completion_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    return ledger_v5.build_goal_completion_payload(root, evidence_path)


def recompute_later_payload(
    root: Path, event_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v7 event type is not allowlisted: {event_type}")
    if not isinstance(payload, Mapping):
        raise ContractError("v7 event payload must be an object")
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
        raise ContractError(f"v7 event type is not allowlisted: {event_type}")
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
        raise ContractError("v7 ledger must be non-empty canonical LF-terminated JSONL")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("v7 ledger is not UTF-8") from error
    records: list[dict[str, Any]] = []
    previous = V5_LEDGER_HEAD_SHA256
    for line_number, line in enumerate(text.split("\n")[:-1], 1):
        if not line.strip():
            raise ContractError(f"blank v7 ledger line {line_number}")
        record = _strict_json_loads(line)
        if not isinstance(record, dict) or set(record) != {
            "seq",
            "recorded_at_kst",
            "event_type",
            "previous_event_sha256",
            "payload",
            "event_sha256",
        }:
            raise ContractError(f"v7 ledger record keys changed at line {line_number}")
        if (
            not isinstance(record.get("seq"), int)
            or isinstance(record.get("seq"), bool)
            or record["seq"] != len(records) + 1
        ):
            raise ContractError(f"v7 ledger sequence mismatch at line {line_number}")
        if record.get("previous_event_sha256") != previous:
            raise ContractError(f"v7 ledger chain mismatch at line {line_number}")
        timestamp = record.get("recorded_at_kst")
        if not isinstance(timestamp, str):
            raise ContractError(f"v7 ledger timestamp is invalid at line {line_number}")
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ContractError(f"v7 ledger timestamp is invalid at line {line_number}") from error
        if parsed.utcoffset() != timedelta(hours=9):
            raise ContractError(f"v7 ledger timestamp is not KST at line {line_number}")
        event_type = record.get("event_type")
        if event_type not in ALL_EVENT_TYPES:
            raise ContractError(f"v7 ledger event is not allowlisted at line {line_number}")
        claimed = record.get("event_sha256")
        if not _is_sha256(claimed):
            raise ContractError(f"v7 event SHA is invalid at line {line_number}")
        base = {key: value for key, value in record.items() if key != "event_sha256"}
        if hashlib.sha256(canonical_json_bytes(base)).hexdigest() != claimed:
            raise ContractError(f"v7 event hash mismatch at line {line_number}")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise ContractError(f"v7 payload is invalid at line {line_number}")
        if line_number == 1:
            if event_type != GENESIS_EVENT_TYPE:
                raise ContractError("v7 first event is not genesis")
            _verify_genesis_payload(root, payload)
        else:
            if event_type == GENESIS_EVENT_TYPE:
                raise ContractError("v7 ledger contains duplicate genesis")
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
            raise OSError(f"short v7 {role} write")
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
        raise ContractError("canonical v7 ledger directory must already exist via QA receipt")
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
        raise ContractError("v7 genesis failed replay round-trip validation")
    return record


def append_ledger_event(
    root: Path,
    path: Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v7 later event is not allowlisted: {event_type}")
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    canonical_payload = recompute_later_payload(root, event_type, payload)
    existing = validate_ledger(root, ledger)
    if not existing:
        raise ContractError("v7 ledger must be initialized before append")
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
                    "schema_version": "meaningful_score_ledger_v7.append_lock.v1",
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
            raise ContractError("v7 ledger changed before lock acquisition")
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
            raise ContractError("v7 append failed replay round-trip validation")
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
    "P2_OFFICIAL_DECISION",
    "P2_UPLOAD_STATUS",
    "PRE_INIT_QA_RELATIVE",
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
