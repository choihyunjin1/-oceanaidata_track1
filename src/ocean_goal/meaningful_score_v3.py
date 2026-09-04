"""P2-only architecture-matched extension of the sealed v2 score contract.

The v2 evaluator remains byte-for-byte unchanged.  This module preserves its
exact-comparator behavior for P1/P3 (and exact P2 evidence), while adding one
explicitly non-exact P2 research mode.  A local architecture-matched curve can
never promote by itself: promotion requires a same-surface official paired A/B
whose baseline bytes are the immutable P2 CSV.

No function in this module uploads, trains, predicts, or opens target values.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ocean_goal import meaningful_score as v2

ContractError = v2.ContractError
ProblemContract = v2.ProblemContract
canonical_json_bytes = v2.canonical_json_bytes
evaluate_goal_completion = v2.evaluate_goal_completion
sha256_file = v2.sha256_file

CONTRACT_RELATIVE = "configs/goals/meaningful_score_maximization_v3.json"
CONTRACT_SHA256 = "76d7cc6e10277ae6c06b1109a15f4dcc8413216f73ac3f479ec265a065d756c0"
V2_CONTRACT_RELATIVE = "configs/goals/meaningful_score_maximization_v2.json"
V2_CONTRACT_SHA256 = "8fc6b03aba3b88b07b954030759f351644513490f7a9c030b7d7f1b950023549"
V2_EVALUATOR_RELATIVE = "src/ocean_goal/meaningful_score.py"
V2_EVALUATOR_SHA256 = "69b9dc1168a47e0d1b1a50e5590c3d2f0966f2885f0f92730d4c38c4ea92800c"
V2_LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v2/registry.jsonl"
V2_LEDGER_SHA256 = "1decc63e63a3ac5a732035402cb00f8750e1afb64c835f17d8f972a6526fb233"
V2_LEDGER_BYTES = 7222
V2_LEDGER_EVENT_COUNT = 5
V2_LEDGER_HEAD_SHA256 = "19f8395ee85976275ce65f430a13db51eb3a3d21dcd224d17a0ac0cae6c00f70"
GENESIS_EVENT_TYPE = "GOAL_INITIALIZED"
PRE_INIT_QA_RELATIVE = "artifacts/meaningful_score_goal_v3/pre_init_qa.json"
GENESIS_IMPLEMENTATION_ROLES = frozenset(
    {
        "V3_CONTRACT",
        "V3_EVALUATOR",
        "V3_CLI",
        "P2_ARCHITECTURE_CONFIG",
        "P2_ARCHITECTURE_GUARDS",
        "P2_STAGE_A_RUNNER",
        "P2_STAGE_B_RUNNER",
        "V3_TESTS",
        "P2_ARCHITECTURE_TESTS",
    }
)
EXACT_MODE = "EXACT_OFFICIAL_PREFIX_REFIT"
ARCHITECTURE_MODE = "ARCHITECTURE_MATCHED_TIME_SAFE_BASELINE"
ARCHITECTURE_PASS_DECISION = "ARCHITECTURE_MATCHED_CURVE_QUALIFIED_PENDING_OFFICIAL_PAIRED_AB"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _workspace_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    workspace = root.resolve(strict=True)
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ContractError("path must be workspace-relative and non-traversing")
    resolved = (workspace / candidate).resolve(strict=must_exist)
    if not resolved.is_relative_to(workspace):
        raise ContractError("path escapes workspace")
    return resolved


def _verify_file_pin(root: Path, value: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    relative = str(value.get("path", ""))
    expected = value.get("sha256")
    if not _is_sha256(expected):
        raise ContractError(f"{role} SHA is not a lowercase SHA-256")
    path = _workspace_path(root, relative)
    observed = sha256_file(path)
    if observed != expected:
        raise ContractError(f"{role} SHA mismatch")
    return {"path": Path(relative).as_posix(), "sha256": observed, "bytes": path.stat().st_size}


def load_contract(root: Path, relative: str = CONTRACT_RELATIVE) -> dict[str, Any]:
    """Load only the canonical v3 bytes, then apply all inherited v2 checks."""

    if Path(relative).as_posix() != CONTRACT_RELATIVE:
        raise ContractError("only the canonical v3 contract path is accepted")
    path = _workspace_path(root, relative)
    if sha256_file(path) != CONTRACT_SHA256:
        raise ContractError("canonical v3 contract SHA mismatch")
    contract = v2.load_contract(root, relative)
    if contract.get("schema_version") != "meaningful_score_goal.v3":
        raise ContractError("v3 schema identity changed")
    modes = contract.get("comparison_modes")
    if not isinstance(modes, Mapping) or set(modes) != {EXACT_MODE, ARCHITECTURE_MODE}:
        raise ContractError("v3 comparison-mode set changed")
    if modes[ARCHITECTURE_MODE].get("allowed_problems") != ["P2"]:
        raise ContractError("architecture-matched mode must remain P2-only")
    for problem in ("P1", "P3"):
        if contract["problems"][problem].get("allowed_comparison_modes") != [EXACT_MODE]:
            raise ContractError(f"{problem} exact-only rule changed")
    if contract["problems"]["P2"].get("allowed_comparison_modes") != [
        EXACT_MODE,
        ARCHITECTURE_MODE,
    ]:
        raise ContractError("P2 comparison-mode allowlist changed")
    return contract


def verify_initial_pins(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = dict(v2.verify_initial_pins(root, contract))
    predecessor = contract.get("supersedes")
    if not isinstance(predecessor, Mapping):
        raise ContractError("v3 predecessor pin is missing")
    predecessor_pin = {
        "path": predecessor.get("contract"),
        "sha256": predecessor.get("contract_sha256"),
    }
    if predecessor_pin != {"path": V2_CONTRACT_RELATIVE, "sha256": V2_CONTRACT_SHA256}:
        raise ContractError("v2 predecessor identity changed")
    verified["V2_PREDECESSOR"] = _verify_file_pin(root, predecessor_pin, role="v2 predecessor")
    verified["V2_EVALUATOR"] = _verify_file_pin(
        root,
        {"path": V2_EVALUATOR_RELATIVE, "sha256": V2_EVALUATOR_SHA256},
        role="v2 evaluator",
    )
    expected_ledger_anchor = {
        "ledger": V2_LEDGER_RELATIVE,
        "ledger_sha256": V2_LEDGER_SHA256,
        "ledger_bytes": V2_LEDGER_BYTES,
        "ledger_event_count": V2_LEDGER_EVENT_COUNT,
        "ledger_head_event_sha256": V2_LEDGER_HEAD_SHA256,
    }
    observed_ledger_anchor = {key: predecessor.get(key) for key in expected_ledger_anchor}
    if observed_ledger_anchor != expected_ledger_anchor:
        raise ContractError("v2 predecessor ledger anchor changed")
    verified["V2_LEDGER"] = _verify_file_pin(
        root,
        {"path": V2_LEDGER_RELATIVE, "sha256": V2_LEDGER_SHA256},
        role="v2 predecessor ledger",
    )
    ledger_path = _workspace_path(root, V2_LEDGER_RELATIVE)
    ledger_records = v2.validate_ledger(ledger_path)
    if (
        ledger_path.stat().st_size != V2_LEDGER_BYTES
        or len(ledger_records) != V2_LEDGER_EVENT_COUNT
        or ledger_records[-1].get("event_sha256") != V2_LEDGER_HEAD_SHA256
    ):
        raise ContractError("v2 predecessor ledger shape or head changed")
    audit = contract.get("evidence", {}).get("p2_exact_incumbent_prefix_refit_audit")
    if not isinstance(audit, Mapping):
        raise ContractError("P2 exact-prefix audit pin is missing")
    verified["P2_EXACT_PREFIX_AUDIT"] = _verify_file_pin(root, audit, role="P2 audit")
    audit_payload = json.loads(
        _workspace_path(root, str(audit["path"])).read_text(encoding="utf-8")
    )
    if audit_payload.get("verdict") != audit.get("verdict"):
        raise ContractError("P2 audit verdict differs from the v3 contract")
    return verified


def _ledger_record(
    *, seq: int, previous: str, event_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if not event_type or not event_type.replace("_", "").isalnum():
        raise ContractError("event type must be a non-empty identifier")
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


def _verify_genesis_payload(root: Path, payload: Mapping[str, Any]) -> None:
    contract = load_contract(root)
    verify_initial_pins(root, contract)
    initial = contract["initial_state"]
    expected_anchor = {
        "path": V2_LEDGER_RELATIVE,
        "sha256": V2_LEDGER_SHA256,
        "bytes": V2_LEDGER_BYTES,
        "event_count": V2_LEDGER_EVENT_COUNT,
        "head_event_sha256": V2_LEDGER_HEAD_SHA256,
    }
    if payload.get("predecessor_ledger_anchor") != expected_anchor:
        raise ContractError("v3 genesis predecessor anchor mismatch")
    if (
        payload.get("goal_id") != contract["goal_id"]
        or payload.get("status") != initial["status"]
        or payload.get("contract_path") != CONTRACT_RELATIVE
        or payload.get("contract_sha256") != CONTRACT_SHA256
        or payload.get("official_uploads") != 0
        or payload.get("meaningful_promotions") != {"P1": False, "P2": False, "P3": False}
        or payload.get("score_promotions") != {"P1": False, "P2": False, "P3": False}
        or payload.get("upload_performed") is not False
    ):
        raise ContractError("v3 genesis initial state mismatch")
    execution = payload.get("execution_counts")
    if execution != {
        "stage_a_fit": 0,
        "stage_a_prediction": 0,
        "stage_b_fit": 0,
        "stage_b_prediction": 0,
        "upload": 0,
    }:
        raise ContractError("v3 genesis execution counts are not zero")
    pins = payload.get("implementation_pins")
    if not isinstance(pins, Mapping) or set(pins) != GENESIS_IMPLEMENTATION_ROLES:
        raise ContractError("v3 genesis implementation pin roles changed")
    for role, pin in pins.items():
        if not isinstance(pin, Mapping):
            raise ContractError(f"v3 genesis implementation pin is invalid: {role}")
        _verify_file_pin(root, pin, role=f"v3 genesis {role}")
    qa = payload.get("independent_pre_init_qa")
    if not isinstance(qa, Mapping):
        raise ContractError("v3 genesis independent QA pin is missing")
    if qa.get("path") != PRE_INIT_QA_RELATIVE:
        raise ContractError("v3 pre-init QA receipt path is noncanonical")
    qa_pin = _verify_file_pin(root, qa, role="v3 pre-init QA")
    qa_payload = json.loads(_workspace_path(root, str(qa_pin["path"])).read_text(encoding="utf-8"))
    if (
        qa_payload.get("decision") != "GO_INITIALIZE_V3_LEDGER"
        or qa_payload.get("p0_count") != 0
        or qa_payload.get("p1_count") != 0
        or qa_payload.get("predecessor_ledger_anchor") != expected_anchor
        or qa_payload.get("implementation_pins") != pins
    ):
        raise ContractError("v3 pre-init QA receipt does not bind genesis")


def validate_goal_ledger(root: Path, path: Path) -> list[dict[str, Any]]:
    """Validate the cross-version v2-head -> v3-genesis append-only chain."""

    workspace = root.resolve(strict=True)
    ledger = path.resolve(strict=False)
    if not ledger.is_relative_to(workspace):
        raise ContractError("v3 ledger escapes workspace")
    if not ledger.exists():
        return []
    records: list[dict[str, Any]] = []
    previous = V2_LEDGER_HEAD_SHA256
    for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ContractError(f"blank v3 ledger line {line_number}")
        record = json.loads(line)
        if record.get("seq") != len(records) + 1:
            raise ContractError(f"v3 ledger sequence mismatch at line {line_number}")
        if record.get("previous_event_sha256") != previous:
            raise ContractError(f"v3 ledger chain mismatch at line {line_number}")
        claimed = record.get("event_sha256")
        base = {key: value for key, value in record.items() if key != "event_sha256"}
        observed = hashlib.sha256(canonical_json_bytes(base)).hexdigest()
        if claimed != observed:
            raise ContractError(f"v3 ledger event hash mismatch at line {line_number}")
        if line_number == 1:
            if record.get("event_type") != GENESIS_EVENT_TYPE:
                raise ContractError("v3 ledger first event is not genesis")
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                raise ContractError("v3 genesis payload is invalid")
            _verify_genesis_payload(root, payload)
        elif record.get("event_type") == GENESIS_EVENT_TYPE:
            raise ContractError("v3 ledger contains duplicate genesis")
        previous = claimed
        records.append(record)
    return records


def initialize_goal_ledger(root: Path, path: Path, *, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create the v3 genesis exactly once with an O_EXCL filesystem boundary."""

    _verify_genesis_payload(root, payload)
    ledger = path.resolve(strict=False)
    workspace = root.resolve(strict=True)
    if not ledger.is_relative_to(workspace):
        raise ContractError("v3 ledger escapes workspace")
    record = _ledger_record(
        seq=1,
        previous=V2_LEDGER_HEAD_SHA256,
        event_type=GENESIS_EVENT_TYPE,
        payload=payload,
    )
    encoded = canonical_json_bytes(record) + b"\n"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(ledger, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("short v3 genesis write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    persisted = validate_goal_ledger(root, ledger)
    if persisted != [record]:
        raise ContractError("v3 genesis failed round-trip validation")
    return record


def append_goal_ledger_event(
    root: Path,
    path: Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one event only after a valid genesis, serialized by O_EXCL lock."""

    if event_type == GENESIS_EVENT_TYPE:
        raise ContractError("duplicate v3 genesis is forbidden")
    workspace = root.resolve(strict=True)
    ledger = path.resolve(strict=True)
    if not ledger.is_relative_to(workspace):
        raise ContractError("v3 ledger escapes workspace")
    lock = ledger.with_name(f"{ledger.name}.append.lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        lock_payload = canonical_json_bytes(
            {
                "schema_version": "meaningful_score_goal_v3.append_lock.v1",
                "ledger": ledger.name,
                "pid": os.getpid(),
            }
        )
        if os.write(descriptor, lock_payload) != len(lock_payload):
            raise OSError("short v3 append-lock write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        existing = validate_goal_ledger(root, ledger)
        if not existing:
            raise ContractError("v3 ledger must be initialized before append")
        record = _ledger_record(
            seq=len(existing) + 1,
            previous=existing[-1]["event_sha256"],
            event_type=event_type,
            payload=payload,
        )
        encoded = canonical_json_bytes(record) + b"\n"
        descriptor = os.open(ledger, os.O_WRONLY | os.O_APPEND)
        try:
            if os.write(descriptor, encoded) != len(encoded):
                raise OSError("short v3 ledger append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        persisted = validate_goal_ledger(root, ledger)
        if persisted[-1] != record:
            raise ContractError("v3 append failed round-trip validation")
        return record
    finally:
        lock.unlink(missing_ok=True)


def _comparison_mode(contract: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    problem = str(evidence.get("problem", ""))
    if problem not in contract.get("problems", {}):
        raise ContractError(f"unknown problem {problem}")
    mode = str(evidence.get("comparison_mode", EXACT_MODE))
    allowed = contract["problems"][problem].get("allowed_comparison_modes", [])
    if mode not in allowed:
        raise ContractError(f"{mode} is forbidden for {problem}")
    return mode


def verify_architecture_reference_binding(
    root: Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify Stage-A bytes and their deep binding before any Stage-B evaluation."""

    binding = evidence.get("reference_binding")
    if not isinstance(binding, Mapping):
        raise ContractError("architecture-matched evidence requires reference_binding")
    required = {
        "stage_a_config",
        "deployed_graph_manifest",
        "training_recipe",
        "reference_oof_100",
        "reference_seal",
    }
    if set(binding) != required:
        raise ContractError("reference_binding keys differ from the v3 contract")
    verified: dict[str, Any] = {}
    for role in sorted(required):
        item = binding[role]
        if not isinstance(item, Mapping):
            raise ContractError(f"reference binding {role} must be an object")
        verified[role] = _verify_file_pin(root, item, role=role)

    seal_path = _workspace_path(root, str(binding["reference_seal"]["path"]))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema_version") != "p2_architecture_matched_reference.seal.v1":
        raise ContractError("Stage-A seal schema changed")
    if seal.get("comparison_mode") != ARCHITECTURE_MODE:
        raise ContractError("Stage-A seal comparison mode changed")
    if seal.get("exact_official_incumbent_comparison") is not False:
        raise ContractError("Stage-A seal falsely claims exact incumbent comparison")
    if seal.get("complete") is not True or seal.get("all_five_prefixes_sealed") is not True:
        raise ContractError("Stage-A reference curve is incomplete")
    if seal.get("challenger_fit_or_score_count_before_seal") != 0:
        raise ContractError("Stage-A was not sealed before challenger scoring")
    prefix_oof = seal.get("reference_oof_by_fraction")
    expected_fractions = {"0.4", "0.55", "0.7", "0.85", "1.0"}
    if not isinstance(prefix_oof, Mapping) or set(prefix_oof) != expected_fractions:
        raise ContractError("Stage-A seal must bind all five prefix OOF files")
    seal_directory = seal_path.parent
    verified_prefix_oof: dict[str, dict[str, Any]] = {}
    for fraction, pin in prefix_oof.items():
        if not isinstance(pin, Mapping) or not _is_sha256(pin.get("sha256")):
            raise ContractError("Stage-A prefix OOF pin is invalid")
        relative = Path(str(pin.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ContractError("Stage-A prefix OOF path is unsafe")
        path = (seal_directory / relative).resolve(strict=True)
        if not path.is_relative_to(seal_directory.resolve(strict=True)):
            raise ContractError("Stage-A prefix OOF escapes its sealed directory")
        if sha256_file(path) != pin["sha256"]:
            raise ContractError(f"Stage-A prefix OOF SHA mismatch: {fraction}")
        verified_prefix_oof[fraction] = {
            "path": relative.as_posix(),
            "sha256": pin["sha256"],
        }
    reference_100_path = _workspace_path(root, str(binding["reference_oof_100"]["path"]))
    sealed_100_path = (seal_directory / verified_prefix_oof["1.0"]["path"]).resolve(strict=True)
    if (
        sealed_100_path != reference_100_path
        or verified_prefix_oof["1.0"]["sha256"] != binding["reference_oof_100"]["sha256"]
    ):
        raise ContractError("Stage-A 100% OOF binding differs from the sealed curve")
    seal_binding = seal.get("binding")
    expected_binding = {
        "stage_a_config_sha256": binding["stage_a_config"]["sha256"],
        "deployed_graph_manifest_sha256": binding["deployed_graph_manifest"]["sha256"],
        "training_recipe_sha256": binding["training_recipe"]["sha256"],
        "reference_oof_100_sha256": binding["reference_oof_100"]["sha256"],
    }
    if seal_binding != expected_binding:
        raise ContractError("Stage-A seal does not deeply equal the evidence binding")
    return verified


def verify_official_evidence_pins(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the two official receipt files for architecture-matched P2 A/B."""

    if evidence.get("comparison_mode", EXACT_MODE) != ARCHITECTURE_MODE:
        return {}
    if evidence.get("problem") != "P2":
        raise ContractError("architecture-matched official evidence is P2-only")
    pair = evidence.get("official_paired_ab")
    if not isinstance(pair, Mapping):
        raise ContractError("official paired A/B evidence is missing")
    verified: dict[str, Any] = {}
    roles = ("baseline", "challenger")
    for role in roles:
        pin = {
            "path": pair.get(f"{role}_receipt_path"),
            "sha256": pair.get(f"{role}_receipt_sha256"),
        }
        verified[role] = _verify_file_pin(root, pin, role=f"official {role} receipt")
        payload = json.loads(_workspace_path(root, str(pin["path"])).read_text(encoding="utf-8"))
        expected_submission = pair.get(f"{role}_submission_sha256")
        if (
            payload.get("official_receipt_id") != pair.get(f"{role}_receipt_id")
            or payload.get("submission_sha256") != expected_submission
            or payload.get("scoring_version")
            != evidence.get(f"{role if role == 'challenger' else 'incumbent'}_scoring_version")
            or payload.get("split_identity")
            != evidence.get(f"{role if role == 'challenger' else 'incumbent'}_split")
        ):
            raise ContractError(f"official {role} receipt payload binding mismatch")
    return verified


def verify_curve_evidence_pins(root: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    verified = {"preregistration": v2.verify_curve_evidence_pins(root, evidence)}
    if evidence.get("comparison_mode", EXACT_MODE) == ARCHITECTURE_MODE:
        verified["reference_binding"] = verify_architecture_reference_binding(root, evidence)
    return verified


def _architecture_protocol_pass(evidence: Mapping[str, Any]) -> bool:
    protocol = evidence.get("curve_protocol")
    identity = evidence.get("baseline_identity")
    binding = evidence.get("reference_binding")
    if not isinstance(protocol, Mapping) or not isinstance(identity, Mapping):
        return False
    return (
        protocol.get("comparison_mode") == ARCHITECTURE_MODE
        and protocol.get("incumbent_fresh_refit_each_prefix") is True
        and protocol.get("architecture_matched_reference_fresh_refit_each_prefix") is True
        and protocol.get("challenger_fresh_refit_each_prefix") is True
        and protocol.get("same_fold_keys_metric_postprocess") is True
        and protocol.get("incumbent_reference_seed_full_prediction_exact_to_frozen_oof") is False
        and protocol.get("deployed_inference_graph_sha_pinned") is True
        and protocol.get("nested_chronological_component_oof") is True
        and protocol.get("prefix_local_epoch_selection") is True
        and protocol.get("three_complete_pipeline_seeds") is True
        and protocol.get("reference_100_percent_oof_sealed_before_challenger_scoring") is True
        and identity.get("comparison_mode") == ARCHITECTURE_MODE
        and identity.get("explicitly_not_exact_official_incumbent") is True
        and identity.get("training_recipe_origin") == "NEW_PREREGISTERED_TIME_SAFE_RECIPE"
        and identity.get("immutable_csv_used_only_for_official_paired_ab") is True
        and isinstance(binding, Mapping)
        and set(binding)
        == {
            "stage_a_config",
            "deployed_graph_manifest",
            "training_recipe",
            "reference_oof_100",
            "reference_seal",
        }
        and all(
            isinstance(item, Mapping) and _is_sha256(item.get("sha256")) and bool(item.get("path"))
            for item in binding.values()
        )
    )


def evaluate_learning_curve(
    contract: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    mode = _comparison_mode(contract, evidence)
    problem = str(evidence.get("problem"))
    if mode == EXACT_MODE:
        result = v2.evaluate_learning_curve(contract, evidence)
        return {
            **result,
            "schema_version": "meaningful_learning_curve_decision.v2",
            "comparison_mode": EXACT_MODE,
            "exact_official_incumbent_comparison": True,
            "local_qualification": result["passed"],
            "curve_alone_can_promote": False,
            "official_promotion": "PENDING_OFFICIAL_SCORE_EVIDENCE",
        }
    if problem != "P2":
        raise ContractError("architecture-matched mode is P2-only")

    # Reuse v2's exact numeric implementation, but never reuse its semantic
    # protocol label.  The synthetic flag exists only inside this function and
    # is removed from the returned decision.
    numeric_input = deepcopy(dict(evidence))
    numeric_protocol = deepcopy(dict(evidence.get("curve_protocol", {})))
    numeric_protocol["incumbent_reference_seed_full_prediction_exact_to_frozen_oof"] = True
    numeric_input["curve_protocol"] = numeric_protocol
    numeric = v2.evaluate_learning_curve(contract, numeric_input)
    local_gates = dict(numeric["gates"])
    local_gates["curve_protocol_contract_pass"] = _architecture_protocol_pass(evidence)
    local_gates["explicit_non_exact_identity_pass"] = (
        evidence.get("baseline_identity", {}).get("explicitly_not_exact_official_incumbent") is True
    )
    local_gates["stage_a_reference_binding_declared"] = isinstance(
        evidence.get("reference_binding"), Mapping
    )
    qualified = all(local_gates.values())
    return {
        **numeric,
        "schema_version": "meaningful_learning_curve_decision.v2",
        "comparison_mode": ARCHITECTURE_MODE,
        "exact_official_incumbent_comparison": False,
        "decision": ARCHITECTURE_PASS_DECISION if qualified else "RESEARCH_ONLY",
        "passed": False,
        "local_qualification": qualified,
        "curve_alone_can_promote": False,
        "official_promotion": "REQUIRES_ACTUAL_IMMUTABLE_CSV_PAIRED_AB",
        "gates": local_gates,
    }


def _paired_ab_checks(contract: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, bool]:
    pair = evidence.get("official_paired_ab")
    if not isinstance(pair, Mapping):
        return {"official_paired_ab_object_present": False}
    immutable_sha = contract["problems"]["P2"]["immutable_baseline"]["sha256"]
    baseline_receipt = str(pair.get("baseline_receipt_id", ""))
    challenger_receipt = str(pair.get("challenger_receipt_id", ""))
    return {
        "official_paired_ab_object_present": True,
        "actual_immutable_csv_is_baseline": (
            pair.get("baseline_submission_sha256") == immutable_sha
            and evidence.get("incumbent_sha256") == immutable_sha
        ),
        "challenger_sha_matches_pair": (
            pair.get("challenger_submission_sha256") == evidence.get("challenger_sha256")
        ),
        "baseline_scored_first": pair.get("baseline_scored_before_challenger") is True,
        "distinct_official_receipt_ids": (
            bool(baseline_receipt)
            and bool(challenger_receipt)
            and baseline_receipt != challenger_receipt
        ),
        "receipt_hashes_present": (
            _is_sha256(pair.get("baseline_receipt_sha256"))
            and _is_sha256(pair.get("challenger_receipt_sha256"))
        ),
        "receipt_paths_present": (
            bool(pair.get("baseline_receipt_path")) and bool(pair.get("challenger_receipt_path"))
        ),
        "same_surface_pair_declared": pair.get("same_scoring_version_and_split") is True,
        "team_wide_upload_accounting_recorded": (
            pair.get("team_wide_upload_accounting_recorded") is True
        ),
    }


def evaluate_official_score(
    contract: Mapping[str, Any],
    curve_decision: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    mode = str(curve_decision.get("comparison_mode", EXACT_MODE))
    if mode == EXACT_MODE:
        result = v2.evaluate_official_score(contract, curve_decision, evidence)
        return {**result, "schema_version": "official_score_decision.v2", "comparison_mode": mode}
    if mode != ARCHITECTURE_MODE or evidence.get("problem") != "P2":
        raise ContractError("invalid architecture-matched official-score request")
    if evidence.get("comparison_mode") != ARCHITECTURE_MODE:
        raise ContractError("curve and official comparison modes differ")
    if (
        curve_decision.get("local_qualification") is not True
        or curve_decision.get("passed") is not False
    ):
        raise ContractError("P2 architecture-matched curve is not locally qualified")

    delegated_curve = dict(curve_decision)
    delegated_curve["passed"] = True
    base = v2.evaluate_official_score(contract, delegated_curve, evidence)
    pair_checks = _paired_ab_checks(contract, evidence)
    operational_checks = {**base["operational_checks"], **pair_checks}
    score_promoted = all(operational_checks.values())
    meaningful_checks = {
        **operational_checks,
        "architecture_matched_local_qualification": True,
        "actual_immutable_csv_paired_ab": all(pair_checks.values()),
        "official_raw_effect_meets_threshold": (
            base["official_raw_improvement"]
            >= contract["problems"]["P2"]["absolute_effect_threshold"]
        ),
    }
    meaningful_promoted = all(meaningful_checks.values())
    final_confirmed = evidence.get("final_or_private_confirmed") is True
    if meaningful_promoted and final_confirmed:
        decision = "MEANINGFUL_PROMOTED_FINAL_CONFIRMED"
    elif meaningful_promoted:
        decision = "MEANINGFUL_PROMOTED_PROVISIONAL"
    elif score_promoted:
        decision = "SCORE_INCUMBENT_ONLY_SMALL_GAIN"
    else:
        decision = "OFFICIAL_REJECTED_OR_INCOMPARABLE"
    return {
        **base,
        "schema_version": "official_score_decision.v2",
        "comparison_mode": ARCHITECTURE_MODE,
        "decision": decision,
        "score_incumbent_updates": score_promoted,
        "meaningful_incumbent_updates": meaningful_promoted,
        "goal_problem_milestone_complete": meaningful_promoted and final_confirmed,
        "operational_checks": operational_checks,
        "meaningful_checks": meaningful_checks,
        "local_qualification": True,
        "official_promotion": meaningful_promoted,
    }


def validate_upload_approval(
    root: Path,
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    curve_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate readiness only; this function never performs an upload."""

    role = receipt.get("role")
    if role != "ARCHITECTURE_MATCHED_CHALLENGER":
        return v2.validate_upload_approval(root, contract, receipt, curve_decision=curve_decision)
    if curve_decision is None:
        raise ContractError("architecture-matched challenger requires a curve decision")
    if (
        receipt.get("problem") != "P2"
        or curve_decision.get("problem") != "P2"
        or curve_decision.get("comparison_mode") != ARCHITECTURE_MODE
        or curve_decision.get("local_qualification") is not True
        or curve_decision.get("passed") is not False
    ):
        raise ContractError("architecture-matched upload request is not locally qualified P2")
    anchor = receipt.get("baseline_anchor_official_receipt")
    if not isinstance(anchor, Mapping):
        raise ContractError("baseline-first official receipt pin is required")
    verified_anchor = _verify_file_pin(root, anchor, role="baseline official receipt")
    anchor_payload = json.loads(
        _workspace_path(root, str(anchor["path"])).read_text(encoding="utf-8")
    )
    immutable_sha = contract["problems"]["P2"]["immutable_baseline"]["sha256"]
    if (
        anchor_payload.get("submission_sha256") != immutable_sha
        or anchor_payload.get("scoring_version") != receipt.get("scoring_version")
        or anchor_payload.get("split_identity") != receipt.get("split_identity")
        or anchor_payload.get("official_receipt_id") in {None, ""}
    ):
        raise ContractError("baseline official receipt does not bind the paired A/B surface")
    delegated_receipt = dict(receipt)
    delegated_receipt["role"] = "CURVE_QUALIFIED_CHALLENGER"
    delegated_curve = dict(curve_decision)
    delegated_curve["passed"] = True
    result = v2.validate_upload_approval(
        root, contract, delegated_receipt, curve_decision=delegated_curve
    )
    return {
        **result,
        "schema_version": "upload_readiness_receipt.v2",
        "role": role,
        "comparison_mode": ARCHITECTURE_MODE,
        "baseline_anchor_official_receipt": verified_anchor,
        "upload_performed": False,
    }


__all__ = [
    "ARCHITECTURE_MODE",
    "ARCHITECTURE_PASS_DECISION",
    "CONTRACT_RELATIVE",
    "CONTRACT_SHA256",
    "ContractError",
    "EXACT_MODE",
    "append_goal_ledger_event",
    "evaluate_goal_completion",
    "evaluate_learning_curve",
    "evaluate_official_score",
    "initialize_goal_ledger",
    "load_contract",
    "sha256_file",
    "validate_goal_ledger",
    "validate_upload_approval",
    "verify_architecture_reference_binding",
    "verify_curve_evidence_pins",
    "verify_initial_pins",
    "verify_official_evidence_pins",
]
