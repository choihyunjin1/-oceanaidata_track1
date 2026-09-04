"""Append-only v6 compatibility ledger for immutable P2 Stage-A v3 evidence.

The v3 scoring contract and evaluator, the complete nine-event v5 ledger, and
the P2 Stage-A v3 reference lineage remain immutable.  This revision only
adds a fail-closed compatibility verifier for the real v3 seal shape.  Exact
P1/P3 curve payloads and all non-curve payloads are delegated to v5 without
rewriting their schemas or values.
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
from ocean_goal import meaningful_score_v3 as scoring

ContractError = scoring.ContractError
canonical_json_bytes = scoring.canonical_json_bytes
sha256_file = scoring.sha256_file

CONTRACT_RELATIVE = "configs/goals/meaningful_score_ledger_v6.json"
CONTRACT_SHA256 = "a0aa5b04d57cc74552b58f94f6eb79c078ce48722e7351f21f3007f0ca8264f8"
LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v6/registry.jsonl"
PRE_INIT_QA_RELATIVE = "artifacts/meaningful_score_goal_v6/pre_init_qa.json"

V5_LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v5/registry.jsonl"
V5_LEDGER_SHA256 = "729b2eb8c22d3380651e1728a2ac6c96fc3c70305e1631ef7ec1bda5d21c0989"
V5_LEDGER_BYTES = 29320
V5_LEDGER_EVENT_COUNT = 9
V5_LEDGER_HEAD_SHA256 = "ded6a43bdc62fbbdce9b54ede37d882ce3e27b54b193da736ba05ca0303e5066"

V3_SCORING_CONTRACT = {
    "path": "configs/goals/meaningful_score_maximization_v3.json",
    "sha256": "76d7cc6e10277ae6c06b1109a15f4dcc8413216f73ac3f479ec265a065d756c0",
    "bytes": 7922,
}
V3_SCORING_EVALUATOR = {
    "path": "src/ocean_goal/meaningful_score_v3.py",
    "sha256": "720839224209cf487e500650ecf34c1e8cd3e5fb26f2395c60e76d2050ed973a",
    "bytes": 33680,
}
V5_IMPLEMENTATION = {
    "V5_CONTRACT": {
        "path": "configs/goals/meaningful_score_ledger_v5.json",
        "sha256": "6c61e71016d3c4cf9b8a34e818ca865b7246e3b02ad2cb22348dd2a6e71d6b35",
        "bytes": 3253,
    },
    "V5_EVALUATOR": {
        "path": "src/ocean_goal/meaningful_score_ledger_v5.py",
        "sha256": "aaf54e6cafac0cc01af7f3265af42a7a80e2738a4d5a682070bd47b41e4b0053",
        "bytes": 28940,
    },
    "V5_CLI": {
        "path": "scripts/run_meaningful_score_ledger_v5.py",
        "sha256": "bcf9c68a2a050f805c7a7ba59f76163c3c42be92082d3ac669251c8bcf09c74b",
        "bytes": 7701,
    },
    "V5_TESTS": {
        "path": "tests/test_meaningful_score_ledger_v5.py",
        "sha256": "85c7c4a5511dade27daad063220742a21a726740bd30b9db7c6a0b87db64f9c0",
        "bytes": 11479,
    },
}

GENESIS_EVENT_TYPE = "GOAL_INITIALIZED"
LATER_EVENT_TYPES = frozenset(
    {"CURVE_RESULT", "OFFICIAL_SCORE_RESULT", "UPLOAD_READINESS", "GOAL_COMPLETION"}
)
ALL_EVENT_TYPES = frozenset({GENESIS_EVENT_TYPE, *LATER_EVENT_TYPES})
O_BINARY = getattr(os, "O_BINARY", 0)
ARCHITECTURE_MODE = "ARCHITECTURE_MATCHED_TIME_SAFE_BASELINE"
P2_STAGE_B_V3_SCHEMA = "p2_architecture_matched_stage_b.learning_curve_evidence.v3"
OOF_ROLES_BY_FRACTION = {
    "0.4": "OOF_040",
    "0.55": "OOF_055",
    "0.7": "OOF_070",
    "0.85": "OOF_085",
    "1.0": "OOF_100",
}
STAGE_A_ROLES = frozenset(
    {
        "CONFIG",
        "MANIFEST",
        "SEAL",
        "ARCHITECTURE_MANIFEST",
        "TRAINING_RECIPE",
        "TRAINING_RECEIPT",
        "CURVE_METRICS",
        *OOF_ROLES_BY_FRACTION.values(),
    }
)
IMPLEMENTATION_RELATIVES = {
    "V6_CONTRACT": CONTRACT_RELATIVE,
    "V6_EVALUATOR": "src/ocean_goal/meaningful_score_ledger_v6.py",
    "V6_CLI": "scripts/run_meaningful_score_ledger_v6.py",
    "V6_TESTS": "tests/test_meaningful_score_ledger_v6.py",
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
        raise ContractError("canonical v6 ledger path must not use symlinks")
    candidate = requested.resolve(strict=must_exist)
    if not candidate.is_relative_to(workspace):
        raise ContractError("v6 ledger escapes workspace")
    if candidate != expected:
        raise ContractError("only the canonical v6 ledger path is accepted")
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


def load_contract(root: Path, relative: str = CONTRACT_RELATIVE) -> dict[str, Any]:
    if Path(relative).as_posix() != CONTRACT_RELATIVE:
        raise ContractError("only the canonical v6 ledger contract is accepted")
    path = _workspace_path(root, relative)
    if _file_pin_for_path(root, path)["path"] != CONTRACT_RELATIVE:
        raise ContractError("canonical v6 ledger contract path uses a symlink")
    if sha256_file(path) != CONTRACT_SHA256:
        raise ContractError("canonical v6 ledger contract SHA mismatch")
    contract = _json_object(path)
    if contract.get("schema_version") != "meaningful_score_ledger.v6":
        raise ContractError("v6 ledger schema identity changed")
    revision = contract.get("revision")
    if not isinstance(revision, Mapping) or revision != {
        "kind": "APPEND_ONLY_P2_STAGE_A_V3_COMPATIBILITY",
        "reason": (
            "The immutable Stage-A reference uses seal schema v3, while the frozen "
            "central v3 verifier accepts only the obsolete seal schema v1 shape."
        ),
        "standard_event_type": "CURVE_RESULT",
        "p1_p3_exact_payloads_delegate_bit_for_bit_to_v5": True,
        "architecture_matched_p2_central_decision": "RESEARCH_ONLY",
        "architecture_matched_p2_passed": False,
        "architecture_matched_p2_official_promotion": False,
        "architecture_matched_p2_preserve_local_qualification": True,
    }:
        raise ContractError("v6 compatibility revision changed")
    if contract.get("scoring_contract") != V3_SCORING_CONTRACT:
        raise ContractError("v3 scoring-contract pin changed")
    if contract.get("scoring_evaluator") != V3_SCORING_EVALUATOR:
        raise ContractError("v3 scoring-evaluator pin changed")
    if contract.get("predecessor_ledger") != _predecessor_anchor():
        raise ContractError("v5 predecessor anchor changed")
    if contract.get("predecessor_implementation") != V5_IMPLEMENTATION:
        raise ContractError("v5 implementation pins changed")
    lineage = contract.get("p2_stage_a_v3_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != STAGE_A_ROLES:
        raise ContractError("P2 Stage-A v3 lineage role set changed")
    if contract.get("canonical_paths") != {
        "ledger": LEDGER_RELATIVE,
        "pre_init_qa": PRE_INIT_QA_RELATIVE,
    }:
        raise ContractError("v6 canonical paths changed")
    protocol = contract.get("event_protocol")
    if not isinstance(protocol, Mapping):
        raise ContractError("v6 event protocol is missing")
    if protocol.get("genesis_event_type") != GENESIS_EVENT_TYPE or protocol.get(
        "later_event_types"
    ) != ["CURVE_RESULT", "OFFICIAL_SCORE_RESULT", "UPLOAD_READINESS", "GOAL_COMPLETION"]:
        raise ContractError("v6 typed event allowlist changed")
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
            raise ContractError(f"v6 protocol weakened: {key}")
    if protocol.get("append_lock") != "O_EXCL_ADJACENT_LOCK":
        raise ContractError("v6 append lock changed")
    if protocol.get("genesis_creation") != "O_EXCL":
        raise ContractError("v6 genesis creation changed")
    if protocol.get("canonical_line_ending") != "LF_ONLY":
        raise ContractError("v6 canonical line ending changed")
    return contract


def verify_predecessor(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("predecessor_ledger") != _predecessor_anchor():
        raise ContractError("v5 predecessor differs from v6 contract")
    verified: dict[str, Any] = {}
    ledger_pin, ledger_path = _verify_file_pin(
        root,
        {
            "path": V5_LEDGER_RELATIVE,
            "sha256": V5_LEDGER_SHA256,
            "bytes": V5_LEDGER_BYTES,
        },
        role="v5 predecessor ledger",
    )
    verified["V5_PREDECESSOR_LEDGER"] = ledger_pin
    for role, expected in V5_IMPLEMENTATION.items():
        pin, _ = _verify_file_pin(root, expected, role=role)
        verified[role] = pin
    for role, expected in (
        ("V3_SCORING_CONTRACT", V3_SCORING_CONTRACT),
        ("V3_SCORING_EVALUATOR", V3_SCORING_EVALUATOR),
    ):
        pin, _ = _verify_file_pin(root, expected, role=role)
        verified[role] = pin
    ledger_v5.load_contract(root)
    records = ledger_v5.validate_ledger(root, ledger_path)
    if len(records) != V5_LEDGER_EVENT_COUNT:
        raise ContractError("v5 predecessor event count changed")
    if records[-1].get("event_sha256") != V5_LEDGER_HEAD_SHA256:
        raise ContractError("v5 predecessor head changed")
    return verified


def _local_pin(pin: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": Path(str(pin["path"])).name,
        "sha256": pin["sha256"],
        "bytes": pin["bytes"],
    }


def _expected_reference_binding(lineage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage_a_config": dict(lineage["CONFIG"]),
        "deployed_graph_manifest": dict(lineage["ARCHITECTURE_MANIFEST"]),
        "training_recipe": dict(lineage["TRAINING_RECIPE"]),
        "reference_oof_100": dict(lineage["OOF_100"]),
        "reference_seal": dict(lineage["SEAL"]),
    }


def verify_p2_stage_a_v3_lineage(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    lineage = contract.get("p2_stage_a_v3_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != STAGE_A_ROLES:
        raise ContractError("P2 Stage-A v3 lineage role set changed")
    verified: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for role in sorted(STAGE_A_ROLES):
        value = lineage[role]
        if not isinstance(value, Mapping):
            raise ContractError(f"P2 Stage-A v3 {role} pin is not an object")
        pin, path = _verify_file_pin(root, value, role=f"P2 Stage-A v3 {role}")
        verified[role] = pin
        paths[role] = path

    config = _json_object(paths["CONFIG"])
    manifest = _json_object(paths["MANIFEST"])
    seal = _json_object(paths["SEAL"])
    curve = _json_object(paths["CURVE_METRICS"])
    if (
        config.get("schema_version") != "p2_architecture_matched_stage_a_execution.v3"
        or config.get("problem") != "P2"
        or config.get("comparison_mode") != ARCHITECTURE_MODE
        or config.get("exact_official_incumbent_comparison") is not False
        or config.get("official_promotion_allowed") is not False
        or config.get("upload_allowed") is not False
        or config.get("official_submission_count") != 0
    ):
        raise ContractError("P2 Stage-A v3 config semantics changed")
    if (
        manifest.get("schema_version") != "p2_architecture_matched_reference.manifest.v3"
        or manifest.get("problem") != "P2"
        or manifest.get("comparison_mode") != ARCHITECTURE_MODE
        or manifest.get("exact_official_incumbent_comparison") is not False
        or manifest.get("official_promotion_allowed") is not False
        or manifest.get("append_only") is not True
        or manifest.get("challenger_import_fit_or_score_count") != 0
        or manifest.get("uploads") != 0
        or manifest.get("config")
        != {"path": verified["CONFIG"]["path"], "sha256": verified["CONFIG"]["sha256"]}
    ):
        raise ContractError("P2 Stage-A v3 manifest semantics changed")
    manifest_artifacts = manifest.get("artifacts")
    manifest_role_by_name = {
        "architecture_manifest": "ARCHITECTURE_MANIFEST",
        "training_recipe": "TRAINING_RECIPE",
        "training_receipt": "TRAINING_RECEIPT",
        "reference_curve_metrics": "CURVE_METRICS",
        "reference_oof_040": "OOF_040",
        "reference_oof_055": "OOF_055",
        "reference_oof_070": "OOF_070",
        "reference_oof_085": "OOF_085",
        "reference_oof_100": "OOF_100",
    }
    expected_manifest_artifacts = {
        name: _local_pin(verified[role]) for name, role in manifest_role_by_name.items()
    }
    if manifest_artifacts != expected_manifest_artifacts:
        raise ContractError("P2 Stage-A v3 manifest artifact pins changed")
    if (
        seal.get("schema_version") != "p2_architecture_matched_reference.seal.v3"
        or seal.get("comparison_mode") != ARCHITECTURE_MODE
        or seal.get("exact_official_incumbent_comparison") is not False
        or seal.get("official_promotion_allowed") is not False
        or seal.get("complete") is not True
        or seal.get("all_five_prefixes_sealed") is not True
        or seal.get("challenger_import_fit_or_score_count_before_seal") != 0
        or seal.get("upload_count") != 0
        or seal.get("config")
        != {"path": verified["CONFIG"]["path"], "sha256": verified["CONFIG"]["sha256"]}
        or seal.get("manifest") != _local_pin(verified["MANIFEST"])
    ):
        raise ContractError("P2 Stage-A v3 seal semantics changed")
    expected_oof = {
        fraction: _local_pin(verified[role]) for fraction, role in OOF_ROLES_BY_FRACTION.items()
    }
    if seal.get("reference_oof_by_fraction") != expected_oof:
        raise ContractError("P2 Stage-A v3 seal does not bind the exact five OOF files")
    expected_header = (
        b"fold,station,layer,time,seed_20260823,seed_20260824,"
        b"seed_20260825,prediction_mean"
    )
    for role in OOF_ROLES_BY_FRACTION.values():
        with paths[role].open("rb") as stream:
            if stream.readline().rstrip(b"\r\n") != expected_header:
                raise ContractError(f"P2 Stage-A v3 {role} header changed")
    if (
        curve.get("schema_version") != "p2_architecture_matched_reference.curve_metrics.v3"
        or curve.get("problem") != "P2"
        or curve.get("comparison_mode") != ARCHITECTURE_MODE
        or curve.get("exact_official_incumbent_comparison") is not False
        or curve.get("local_qualification_only") is not True
        or curve.get("official_promotion_allowed") is not False
        or curve.get("uploads") != 0
    ):
        raise ContractError("P2 Stage-A v3 curve-metrics semantics changed")
    points = curve.get("points")
    if not isinstance(points, list) or [point.get("fraction") for point in points] != [
        0.4,
        0.55,
        0.7,
        0.85,
        1.0,
    ]:
        raise ContractError("P2 Stage-A v3 curve fractions changed")
    if any(point.get("rows") != 78156 for point in points):
        raise ContractError("P2 Stage-A v3 curve row counts changed")
    if points[-1].get("prediction_mean_metric") != 1.0109798870010898:
        raise ContractError("P2 Stage-A v3 full reference metric changed")
    return verified


def current_implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    pins = {
        role: _file_pin_for_path(root, _workspace_path(root, relative))
        for role, relative in IMPLEMENTATION_RELATIVES.items()
    }
    if set(pins) != IMPLEMENTATION_ROLES:
        raise ContractError("v6 implementation role set changed")
    return pins


def _canonical_qa_pin(root: Path, requested: Path) -> dict[str, Any]:
    workspace = _workspace(root)
    lexical = workspace / PRE_INIT_QA_RELATIVE
    expected = lexical.resolve(strict=False)
    if expected != lexical:
        raise ContractError("canonical v6 QA path must not use symlinks")
    resolved = requested.resolve(strict=True)
    if not resolved.is_relative_to(workspace) or resolved != expected:
        raise ContractError("v6 pre-init QA receipt path must be canonical")
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
        receipt.get("schema_version") != "meaningful_score_ledger_v6.pre_init_qa.v1"
        or receipt.get("decision") != "GO_INITIALIZE_V6_LEDGER"
        or receipt.get("p0_count") != 0
        or receipt.get("p1_count") != 0
        or receipt.get("ledger_contract") != _ledger_contract_pin(root)
        or receipt.get("predecessor_ledger_anchor") != _predecessor_anchor()
        or receipt.get("verified_predecessor_pins") != predecessor_pins
        or receipt.get("p2_stage_a_v3_lineage_pins") != lineage_pins
        or receipt.get("implementation_pins") != implementation_pins
    ):
        raise ContractError("v6 pre-init QA receipt does not bind current genesis")
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
        raise ContractError("v6 genesis canonical QA pin is missing")
    _, qa_path = _verify_file_pin(root, qa, role="v6 pre-init QA")
    if dict(payload) != build_genesis_payload(root, qa_path):
        raise ContractError("v6 genesis payload differs from canonical recomputation")


def _source_json_from_pin(
    root: Path, value: Any, *, role: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{role} pin is missing")
    pin, path = _verify_file_pin(root, value, role=role)
    return _json_object(path), pin, path


def _verify_preregistration(root: Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    registration = evidence.get("preregistration")
    if not isinstance(registration, Mapping):
        raise ContractError("preregistration must be an object")
    relative = registration.get("config_path")
    expected_sha = registration.get("config_sha256")
    if not isinstance(relative, str) or not _is_sha256(expected_sha):
        raise ContractError("preregistration config pin is invalid")
    path = _workspace_path(root, relative)
    observed = _file_pin_for_path(root, path)
    if observed["sha256"] != expected_sha:
        raise ContractError("preregistration config SHA mismatch")
    return {
        "config_path": observed["path"],
        "config_sha256": observed["sha256"],
        "config_bytes": observed["bytes"],
    }


def _build_p2_architecture_curve_payload(
    root: Path, evidence_path: Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    if evidence.get("schema_version") != P2_STAGE_B_V3_SCHEMA:
        raise ContractError("architecture-matched P2 evidence must use Stage-B schema v3")
    if (
        evidence.get("problem") != "P2"
        or evidence.get("comparison_mode") != ARCHITECTURE_MODE
        or evidence.get("exact_official_incumbent_comparison") is not False
    ):
        raise ContractError("architecture-matched P2 evidence identity changed")
    contract = load_contract(root)
    _verify_file_pin(root, V3_SCORING_CONTRACT, role="v3 scoring contract")
    _verify_file_pin(root, V3_SCORING_EVALUATOR, role="v3 scoring evaluator")
    lineage = verify_p2_stage_a_v3_lineage(root, contract)
    expected_binding = _expected_reference_binding(lineage)
    if evidence.get("reference_binding") != expected_binding:
        raise ContractError("P2 evidence reference_binding differs from immutable Stage-A v3")
    preregistration = _verify_preregistration(root, evidence)
    decision = scoring.evaluate_learning_curve(scoring.load_contract(root), evidence)
    if (
        decision.get("problem") != "P2"
        or decision.get("comparison_mode") != ARCHITECTURE_MODE
        or not isinstance(decision.get("local_qualification"), bool)
    ):
        raise ContractError("v3 P2 numeric evaluator returned incompatible semantics")
    central_decision = {
        **decision,
        "decision": "RESEARCH_ONLY",
        "passed": False,
        "official_promotion": False,
        "curve_alone_can_promote": False,
    }
    return {
        "schema_version": "meaningful_score_ledger_v6.curve_result.v1",
        "evidence": _file_pin_for_path(root, evidence_path),
        "evidence_pins": {
            "preregistration": preregistration,
            "reference_binding": expected_binding,
            "p2_stage_a_v3_lineage": lineage,
        },
        "decision": central_decision,
        "upload_performed": False,
    }


def build_curve_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    pin = _file_pin_for_path(root, evidence_path)
    canonical_path = _workspace_path(root, pin["path"])
    evidence = _json_object(canonical_path)
    mode = evidence.get("comparison_mode", scoring.EXACT_MODE)
    if mode == ARCHITECTURE_MODE:
        if evidence.get("problem") != "P2":
            raise ContractError("architecture-matched mode is P2-only")
        return _build_p2_architecture_curve_payload(root, canonical_path, evidence)
    return ledger_v5.build_curve_payload(root, canonical_path)


def build_official_score_payload(
    root: Path, evidence_path: Path, curve_decision_path: Path
) -> dict[str, Any]:
    return ledger_v5.build_official_score_payload(root, evidence_path, curve_decision_path)


def build_upload_readiness_payload(
    root: Path, receipt_path: Path, curve_decision_path: Path | None
) -> dict[str, Any]:
    return ledger_v5.build_upload_readiness_payload(root, receipt_path, curve_decision_path)


def build_goal_completion_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    return ledger_v5.build_goal_completion_payload(root, evidence_path)


def recompute_later_payload(
    root: Path, event_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v6 event type is not allowlisted: {event_type}")
    if not isinstance(payload, Mapping):
        raise ContractError("v6 event payload must be an object")
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
        raise ContractError(f"v6 event type is not allowlisted: {event_type}")
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
        raise ContractError("v6 ledger must be non-empty canonical LF-terminated JSONL")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("v6 ledger is not UTF-8") from error
    records: list[dict[str, Any]] = []
    previous = V5_LEDGER_HEAD_SHA256
    for line_number, line in enumerate(text.split("\n")[:-1], 1):
        if not line.strip():
            raise ContractError(f"blank v6 ledger line {line_number}")
        record = _strict_json_loads(line)
        if not isinstance(record, dict) or set(record) != {
            "seq",
            "recorded_at_kst",
            "event_type",
            "previous_event_sha256",
            "payload",
            "event_sha256",
        }:
            raise ContractError(f"v6 ledger record keys changed at line {line_number}")
        if (
            not isinstance(record.get("seq"), int)
            or isinstance(record.get("seq"), bool)
            or record["seq"] != len(records) + 1
        ):
            raise ContractError(f"v6 ledger sequence mismatch at line {line_number}")
        if record.get("previous_event_sha256") != previous:
            raise ContractError(f"v6 ledger chain mismatch at line {line_number}")
        timestamp = record.get("recorded_at_kst")
        if not isinstance(timestamp, str):
            raise ContractError(f"v6 ledger timestamp is invalid at line {line_number}")
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ContractError(f"v6 ledger timestamp is invalid at line {line_number}") from error
        if parsed.utcoffset() != timedelta(hours=9):
            raise ContractError(f"v6 ledger timestamp is not KST at line {line_number}")
        event_type = record.get("event_type")
        if event_type not in ALL_EVENT_TYPES:
            raise ContractError(f"v6 ledger event is not allowlisted at line {line_number}")
        claimed = record.get("event_sha256")
        if not _is_sha256(claimed):
            raise ContractError(f"v6 event SHA is invalid at line {line_number}")
        base = {key: value for key, value in record.items() if key != "event_sha256"}
        if hashlib.sha256(canonical_json_bytes(base)).hexdigest() != claimed:
            raise ContractError(f"v6 event hash mismatch at line {line_number}")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise ContractError(f"v6 payload is invalid at line {line_number}")
        if line_number == 1:
            if event_type != GENESIS_EVENT_TYPE:
                raise ContractError("v6 first event is not genesis")
            _verify_genesis_payload(root, payload)
        else:
            if event_type == GENESIS_EVENT_TYPE:
                raise ContractError("v6 ledger contains duplicate genesis")
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
            raise OSError(f"short v6 {role} write")
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
        raise ContractError("canonical v6 ledger directory must already exist via QA receipt")
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
        raise ContractError("v6 genesis failed replay round-trip validation")
    return record


def append_ledger_event(
    root: Path,
    path: Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v6 later event is not allowlisted: {event_type}")
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    canonical_payload = recompute_later_payload(root, event_type, payload)
    existing = validate_ledger(root, ledger)
    if not existing:
        raise ContractError("v6 ledger must be initialized before append")
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
                    "schema_version": "meaningful_score_ledger_v6.append_lock.v1",
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
            raise ContractError("v6 ledger changed before lock acquisition")
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
            raise ContractError("v6 append failed replay round-trip validation")
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
