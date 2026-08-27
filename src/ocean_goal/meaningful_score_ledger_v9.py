"""Append-only v9 ledger compatibility for one frozen P3 research curve.

The initialized v8 ledger is an immutable predecessor.  All existing v8 curve,
official-score, upload-readiness, and completion decisions are delegated without
rewriting.  v9 adds only an exact-artifact route for the P3 Gen5r4
structure-matched learning curve.  That route is permanently research-only and
cannot be used for official scoring, upload readiness, or promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ocean_goal import meaningful_score_ledger_v8 as ledger_v8
from ocean_goal import meaningful_score_v3 as scoring

ContractError = scoring.ContractError
canonical_json_bytes = scoring.canonical_json_bytes
sha256_file = scoring.sha256_file

CONTRACT_RELATIVE = "configs/goals/meaningful_score_ledger_v9.json"
CONTRACT_SHA256 = "2f305a27844d7a76ebddd7cba1c29beff6beaf8086f20f42c569484f4c401489"
LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v9/registry.jsonl"
PRE_INIT_QA_RELATIVE = "artifacts/meaningful_score_goal_v9/pre_init_qa.json"

V8_LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v8/registry.jsonl"
V8_LEDGER_SHA256 = "30d0c885833af64c4900f15b9159caf76fb344abd1175ca742902defd9e74326"
V8_LEDGER_BYTES = 13231
V8_LEDGER_EVENT_COUNT = 2
V8_LEDGER_HEAD_SHA256 = "2f4a16abb2213ed0e517967ae5782dfbcb5ab1b1bd1f08f9e5852cfba15c4c20"

V8_PRE_INIT_QA = {
    "path": "artifacts/meaningful_score_goal_v8/pre_init_qa.json",
    "sha256": "f0b04aaae52c0afa361496f282ae6a9b6089f3da220d866753c06c65f44189fb",
    "bytes": 7937,
}
V8_IMPLEMENTATION = {
    "V8_CONTRACT": {
        "path": "configs/goals/meaningful_score_ledger_v8.json",
        "sha256": "269b3b3f42c6fe356cee9d5685e268131ff01950b84659b7869234579183f6d1",
        "bytes": 4773,
    },
    "V8_EVALUATOR": {
        "path": "src/ocean_goal/meaningful_score_ledger_v8.py",
        "sha256": "c1f100d73e2c391ddaf7c61a76f12b2a7745a9ec5a420b9601e7b86de92f37bf",
        "bytes": 32801,
    },
    "V8_CLI": {
        "path": "scripts/run_meaningful_score_ledger_v8.py",
        "sha256": "322eb9775dc369d16cccb400c1c23121af0c91b44ea32f8e2d3341cb347a2b1c",
        "bytes": 7974,
    },
    "V8_TESTS": {
        "path": "tests/test_meaningful_score_ledger_v8.py",
        "sha256": "e8bc6a9dc3012ebee0e4ebefa54ebff84dba0459b09be07ddbde6778ec54c4d7",
        "bytes": 16767,
    },
}

STRUCTURE_MODE = "STRUCTURE_MATCHED_FRESH_REFIT_PENDING_OFFICIAL_PAIRED_AB"
P3_ARTIFACT_ROOT = "artifacts/p3_hierarchical_residual_basis_dense72_20260823_r4"
P3_STATUS = "NO_LOCAL_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST"
P3_CORE_FILES = {
    "CONFIG": {
        "path": "configs/experiments/p3_hierarchical_residual_basis_dense72_r4.json",
        "sha256": "e3eaac2891e1919b6d781812a451e8f40db5e2ef69964ea3d766cfe789943d7d",
        "bytes": 15361,
    },
    "QA": {
        "path": (
            "artifacts/p3_hierarchical_residual_basis_dense72_20260823_r4_control/"
            "pre_execution_qa.json"
        ),
        "sha256": "45ae3fa0cc1f5c79a322a6a857fdbd72cc7d0fdfd7c6909894353ec00bddea51",
        "bytes": 17020,
    },
    "AUTHORIZATION": {
        "path": (
            "artifacts/p3_hierarchical_residual_basis_dense72_20260823_r4_control/"
            "authorization.json"
        ),
        "sha256": "e53d4b1ea189a2bbf27756a545ea647ce15d86163b1ee8bbbd8fffd0eb3a9469",
        "bytes": 16777,
    },
    "ATTEMPT_LOCK": {
        "path": (
            "artifacts/p3_hierarchical_residual_basis_dense72_20260823_r4_control/"
            "attempt.lock"
        ),
        "sha256": "2357fb88329cf5bfc2266f83a13598f983177c7a5a4ce0fec4a739fe1f047486",
        "bytes": 37904,
    },
    "MANIFEST": {
        "path": f"{P3_ARTIFACT_ROOT}/manifest.json",
        "sha256": "c4c104b11679584249588605932a7efd767704addffa5a1dccf72ce7a89d8f0f",
        "bytes": 58154,
    },
    "MANIFEST_SHA256": {
        "path": f"{P3_ARTIFACT_ROOT}/manifest.sha256",
        "sha256": "a195019c3236b6f872c51ec74ea15b1a8132381dc541c199d2ec7b2ccdb374aa",
        "bytes": 80,
    },
    "METRICS": {
        "path": f"{P3_ARTIFACT_ROOT}/metrics.json",
        "sha256": "5a6209de2f6e258a64479844ad3c2ef947d29e9dabdc9b536f8480493cf49969",
        "bytes": 112306,
    },
    "OOF": {
        "path": f"{P3_ARTIFACT_ROOT}/oof/learning_curve_oof.parquet",
        "sha256": "594f53316a308150d022523de46a555483086c0c146f67425f40fc1600f0d619",
        "bytes": 119152,
    },
    "EVIDENCE": {
        "path": f"{P3_ARTIFACT_ROOT}/learning_curve_evidence.json",
        "sha256": "0ee58d63460dd989331a776f705b078f125f033c34fca777f5b630ebbd4bdedd",
        "bytes": 2983,
    },
    "PREDICTIONS_COMPLETE": {
        "path": f"{P3_ARTIFACT_ROOT}/commitments/predictions_complete.json",
        "sha256": "a94c9e6b5f7d30926c42f30e26d1408cc2f362edad168fb7f1790e3ebd7821ee",
        "bytes": 7973,
    },
    "REGISTRY": {
        "path": f"{P3_ARTIFACT_ROOT}/registry.json",
        "sha256": "d2d8ce0a4f2f646efc58c072dcf1ada8f3bed99e674af6ac629ac7566fc95b1e",
        "bytes": 332,
    },
    "VALIDATION_KEYS": {
        "path": f"{P3_ARTIFACT_ROOT}/validation_keys.parquet",
        "sha256": "25cb230e7b161906d5b17c7bb5a3f56e8baa47b833ced7801f419f2e0200429a",
        "bytes": 4864,
    },
}
P3_MANIFEST_OUTPUT_MAP_SHA256 = (
    "6f1d1dc00e7c1f86011db47467ec19a081d3a0b7ae356e68da15288a808aa4e4"
)
P3_FOLDS = ("2024_h2_storm", "winter_transition", "2025_h1")
P3_PREFIXES = (0.4, 0.55, 0.7, 0.85, 1.0)
P3_SEEDS = (20260816, 20260817, 20260818)
P3_SLICE_KEYS = frozenset(
    {"G-ORS", "I-ORS", "S-ORS", "winter", "lead_12", "lead_18", "lead_24"}
)

GENESIS_EVENT_TYPE = "GOAL_INITIALIZED"
LATER_EVENT_TYPES = ledger_v8.LATER_EVENT_TYPES
ALL_EVENT_TYPES = ledger_v8.ALL_EVENT_TYPES
O_BINARY = getattr(os, "O_BINARY", 0)

IMPLEMENTATION_RELATIVES = {
    "V9_CONTRACT": CONTRACT_RELATIVE,
    "V9_EVALUATOR": "src/ocean_goal/meaningful_score_ledger_v9.py",
    "V9_CLI": "scripts/run_meaningful_score_ledger_v9.py",
    "V9_TESTS": "tests/test_meaningful_score_ledger_v9.py",
}
IMPLEMENTATION_ROLES = frozenset(IMPLEMENTATION_RELATIVES)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
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
        raise ContractError("canonical v9 ledger path must not use symlinks")
    candidate = requested.resolve(strict=must_exist)
    if not candidate.is_relative_to(workspace):
        raise ContractError("v9 ledger escapes workspace")
    if candidate != expected:
        raise ContractError("only the canonical v9 ledger path is accepted")
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


def _v8_anchor() -> dict[str, Any]:
    return {
        "path": V8_LEDGER_RELATIVE,
        "sha256": V8_LEDGER_SHA256,
        "bytes": V8_LEDGER_BYTES,
        "event_count": V8_LEDGER_EVENT_COUNT,
        "head_event_sha256": V8_LEDGER_HEAD_SHA256,
    }


def _expected_revision() -> dict[str, Any]:
    return {
        "kind": "APPEND_ONLY_P3_STRUCTURE_MATCHED_CURVE_COMPATIBILITY",
        "supersedes_initialized_v8_without_mutation": True,
        "reason": (
            "v8 correctly failed closed on the P3 Gen5r4 structure-matched research "
            "evidence because that artifact predates the generic preregistration schema; "
            "v9 adds one exact artifact-bound research-only curve route."
        ),
        "inherit_v8_completion_policy_exactly": True,
        "p1_p2_and_exact_routes_delegate_to_v8_without_payload_rewriting": True,
        "p3_structure_matched_official_score_and_upload_routes_fail_closed": True,
    }


def _expected_completion_policy() -> dict[str, Any]:
    return {
        "source": "FROZEN_V8_POLICY_REPLAY_OVER_V8_PREDECESSOR_PLUS_V9_RECORDS",
        "standalone_completion_evidence_authoritative": False,
        "p2_architecture_events_cannot_satisfy_meaningful_milestone": True,
        "p2_meaningful_milestone_requires_distinct_event_type": (
            ledger_v8.REQUIRED_CONFIRMATION_EVENT_TYPE
        ),
        "required_event_type_currently_allowlisted": False,
        "required_event_type_currently_policy_authorized": False,
        "current_p2_milestone_authorized": False,
        "current_goal_completion_result": "NOT_COMPLETE",
    }


def _expected_forced_decision() -> dict[str, Any]:
    return {
        "decision": "RESEARCH_ONLY",
        "local_qualification": False,
        "passed": False,
        "official_promotion": False,
        "curve_alone_can_promote": False,
        "official_score_route_allowed": False,
        "upload_readiness_route_allowed": False,
    }


def load_contract(root: Path, relative: str = CONTRACT_RELATIVE) -> dict[str, Any]:
    if Path(relative).as_posix() != CONTRACT_RELATIVE:
        raise ContractError("only the canonical v9 ledger contract is accepted")
    path = _workspace_path(root, relative)
    if _file_pin_for_path(root, path)["path"] != CONTRACT_RELATIVE:
        raise ContractError("canonical v9 ledger contract path uses a symlink")
    if sha256_file(path) != CONTRACT_SHA256:
        raise ContractError("canonical v9 ledger contract SHA mismatch")
    contract = _json_object(path)
    if contract.get("schema_version") != "meaningful_score_ledger.v9":
        raise ContractError("v9 ledger schema identity changed")
    if contract.get("revision") != _expected_revision():
        raise ContractError("v9 compatibility revision changed")
    if contract.get("predecessor_v8_ledger") != _v8_anchor():
        raise ContractError("v8 predecessor anchor changed")
    if contract.get("v8_pre_init_qa") != V8_PRE_INIT_QA:
        raise ContractError("v8 QA anchor changed")
    if contract.get("superseded_v8_implementation") != V8_IMPLEMENTATION:
        raise ContractError("v8 implementation pins changed")
    route = contract.get("p3_gen5r4_structure_curve_route")
    if not isinstance(route, Mapping):
        raise ContractError("P3 structure route is missing")
    if (
        route.get("problem") != "P3"
        or route.get("comparison_mode") != STRUCTURE_MODE
        or route.get("exact_official_incumbent_comparison") is not False
        or route.get("artifact_root") != P3_ARTIFACT_ROOT
        or route.get("artifact_file_count") != 239
        or route.get("manifest_output_file_count") != 237
        or route.get("manifest_output_map_sha256") != P3_MANIFEST_OUTPUT_MAP_SHA256
        or route.get("core_files") != P3_CORE_FILES
        or route.get("forced_curve_decision") != _expected_forced_decision()
    ):
        raise ContractError("P3 exact research-only route changed")
    if contract.get("completion_lineage_policy") != _expected_completion_policy():
        raise ContractError("v9 completion policy changed")
    if contract.get("canonical_paths") != {
        "ledger": LEDGER_RELATIVE,
        "pre_init_qa": PRE_INIT_QA_RELATIVE,
    }:
        raise ContractError("v9 canonical paths changed")
    protocol = contract.get("event_protocol")
    if not isinstance(protocol, Mapping):
        raise ContractError("v9 event protocol is missing")
    if protocol.get("genesis_event_type") != GENESIS_EVENT_TYPE or protocol.get(
        "later_event_types"
    ) != ["CURVE_RESULT", "OFFICIAL_SCORE_RESULT", "UPLOAD_READINESS", "GOAL_COMPLETION"]:
        raise ContractError("v9 typed event allowlist changed")
    for key in (
        "unknown_event_types_forbidden",
        "payload_must_deep_equal_recomputed_payload",
        "evidence_must_be_workspace_relative_sha256_and_size_pinned",
        "replay_every_event_on_every_validation",
        "completion_replay_includes_frozen_v8_predecessor",
        "replay_before_append_lock_or_write",
        "all_os_open_calls_use_o_binary",
        "all_os_write_calls_use_robust_write_loop",
    ):
        if protocol.get(key) is not True:
            raise ContractError(f"v9 protocol weakened: {key}")
    if protocol.get("append_lock") != "O_EXCL_ADJACENT_LOCK":
        raise ContractError("v9 append lock changed")
    if protocol.get("genesis_creation") != "O_EXCL":
        raise ContractError("v9 genesis creation changed")
    if protocol.get("canonical_line_ending") != "LF_ONLY":
        raise ContractError("v9 canonical line ending changed")
    prohibitions = contract.get("prohibitions")
    if not isinstance(prohibitions, Mapping) or any(value is not True for value in prohibitions.values()):
        raise ContractError("v9 prohibitions were weakened")
    return contract


def _verify_v8_implementation(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if contract.get("superseded_v8_implementation") != V8_IMPLEMENTATION:
        raise ContractError("v8 implementation differs from v9 contract")
    verified: dict[str, dict[str, Any]] = {}
    for role, expected in V8_IMPLEMENTATION.items():
        pin, _ = _verify_file_pin(root, expected, role=role)
        verified[role] = pin
    ledger_v8.load_contract(root)
    return verified


def _validated_v8_records(root: Path) -> list[dict[str, Any]]:
    records = ledger_v8.validate_ledger(root, _workspace_path(root, V8_LEDGER_RELATIVE))
    if len(records) != V8_LEDGER_EVENT_COUNT:
        raise ContractError("v8 predecessor event count changed")
    if records[-1].get("event_sha256") != V8_LEDGER_HEAD_SHA256:
        raise ContractError("v8 predecessor head changed")
    return records


def verify_predecessor(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("predecessor_v8_ledger") != _v8_anchor():
        raise ContractError("v8 predecessor differs from v9 contract")
    v8_pins = _verify_v8_implementation(root, contract)
    registry_expected = {
        key: _v8_anchor()[key] for key in ("path", "sha256", "bytes")
    }
    registry_pin, _ = _verify_file_pin(root, registry_expected, role="v8 registry")
    qa_pin, _ = _verify_file_pin(root, V8_PRE_INIT_QA, role="v8 pre-init QA")
    lock = _workspace_path(root, f"{V8_LEDGER_RELATIVE}.append.lock", must_exist=False)
    if lock.exists():
        raise ContractError("v8 predecessor append lock must remain absent")
    _validated_v8_records(root)
    return {
        "V8_STATIC_IMPLEMENTATION": v8_pins,
        "V8_PRE_INIT_QA": qa_pin,
        "V8_LEDGER": {
            **registry_pin,
            "event_count": V8_LEDGER_EVENT_COUNT,
            "head_event_sha256": V8_LEDGER_HEAD_SHA256,
        },
    }


def current_implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    pins = {
        role: _file_pin_for_path(root, _workspace_path(root, relative))
        for role, relative in IMPLEMENTATION_RELATIVES.items()
    }
    if set(pins) != IMPLEMENTATION_ROLES:
        raise ContractError("v9 implementation role set changed")
    return pins


def _short_pin(pin: Mapping[str, Any]) -> dict[str, Any]:
    return {"path": pin["path"], "sha256": pin["sha256"]}


def _mapping_all_true(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ContractError(f"{role} must be a non-empty object")
    if any(item is not True for item in value.values()):
        raise ContractError(f"{role} contains a failed check")
    return dict(value)


def _contains_structure_identity(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("comparison_mode") == STRUCTURE_MODE:
            return True
        schema = value.get("schema_version")
        if isinstance(schema, str) and schema.startswith(
            "meaningful_score_ledger_v9.p3_structure_matched"
        ):
            return True
        return any(_contains_structure_identity(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_structure_identity(item) for item in value)
    return False


def _output_pin(outputs: Mapping[str, Any], relative: str, *, role: str) -> dict[str, Any]:
    value = outputs.get(relative)
    if not isinstance(value, Mapping) or set(value) != {"sha256", "bytes"}:
        raise ContractError(f"{role} output pin is missing")
    if not _is_sha256(value.get("sha256")):
        raise ContractError(f"{role} output SHA is invalid")
    size = value.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ContractError(f"{role} output byte count is invalid")
    return dict(value)


def _verify_manifest_outputs(
    root: Path,
    artifact_root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_count: int,
) -> tuple[dict[str, Any], set[str]]:
    outputs = manifest.get("output_files_before_manifest")
    if not isinstance(outputs, Mapping) or len(outputs) != expected_count:
        raise ContractError("P3 manifest output map count changed")
    if hashlib.sha256(canonical_json_bytes(outputs)).hexdigest() != (
        P3_MANIFEST_OUTPUT_MAP_SHA256
    ):
        raise ContractError("P3 manifest output map SHA changed")
    verified: dict[str, Any] = {}
    for relative, expected in outputs.items():
        if not isinstance(relative, str) or not isinstance(expected, Mapping):
            raise ContractError("P3 manifest output pin is invalid")
        candidate = Path(relative)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ContractError("P3 manifest output path is unsafe")
        lexical = artifact_root / candidate
        resolved = lexical.resolve(strict=True)
        if resolved != lexical or not resolved.is_relative_to(artifact_root):
            raise ContractError("P3 manifest output path uses indirection")
        pin = _output_pin(outputs, relative, role=relative)
        observed = {"sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}
        if observed != pin:
            raise ContractError(f"P3 manifest output differs: {relative}")
        verified[relative] = observed
    actual: set[str] = set()
    for path in artifact_root.rglob("*"):
        if path.is_symlink():
            raise ContractError("P3 artifact contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(artifact_root).as_posix())
    expected_files = {*verified, "manifest.json", "manifest.sha256"}
    if actual != expected_files or len(actual) != 239:
        raise ContractError("P3 artifact file allowlist differs from exact 239-file lineage")
    return verified, actual


def _verify_manifest_implementation_pins(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    pins = manifest.get("implementation_pins")
    if not isinstance(pins, Mapping) or len(pins) != 71:
        raise ContractError("P3 implementation pin count changed")
    verified: dict[str, Any] = {}
    for role, expected in pins.items():
        if not isinstance(role, str) or not isinstance(expected, Mapping):
            raise ContractError("P3 implementation pin is invalid")
        pin, _ = _verify_file_pin(root, expected, role=f"P3 implementation {role}")
        verified[role] = pin
    return verified


def _validate_input_snapshots(manifest: Mapping[str, Any]) -> dict[str, Any]:
    before = manifest.get("input_sha256_before")
    after = manifest.get("input_sha256_after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping) or before != after:
        raise ContractError("P3 input before/after snapshot differs")
    if len(before) != 26:
        raise ContractError("P3 input pin count changed")
    for role, pin in before.items():
        if (
            not isinstance(role, str)
            or not isinstance(pin, Mapping)
            or set(pin) != {"sha256", "bytes"}
            or not _is_sha256(pin.get("sha256"))
            or not isinstance(pin.get("bytes"), int)
            or isinstance(pin.get("bytes"), bool)
            or pin["bytes"] < 0
        ):
            raise ContractError("P3 input snapshot pin is invalid")
    return dict(before)


def _validate_point_evidence(evidence: Mapping[str, Any]) -> None:
    if evidence.get("problem") != "P3" or evidence.get("comparison_mode") != STRUCTURE_MODE:
        raise ContractError("P3 evidence identity changed")
    points = evidence.get("points")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or len(points) != 5:
        raise ContractError("P3 evidence point count changed")
    seen: list[float] = []
    for point in points:
        if not isinstance(point, Mapping) or set(point) != {
            "fraction",
            "incumbent",
            "challenger",
            "delta_ci90",
        }:
            raise ContractError("P3 evidence point schema changed")
        fraction = point.get("fraction")
        ci = point.get("delta_ci90")
        if not _finite(fraction) or not _finite(point.get("incumbent")) or not _finite(
            point.get("challenger")
        ):
            raise ContractError("P3 evidence point contains a non-finite metric")
        if (
            not isinstance(ci, Sequence)
            or isinstance(ci, (str, bytes))
            or len(ci) != 2
            or not all(_finite(value) for value in ci)
            or float(ci[0]) > float(ci[1])
        ):
            raise ContractError("P3 evidence CI is invalid")
        seen.append(float(fraction))
    if tuple(seen) != P3_PREFIXES:
        raise ContractError("P3 evidence prefixes changed")
    folds = evidence.get("fold_deltas_candidate_minus_incumbent")
    slices = evidence.get("slice_deltas_candidate_minus_incumbent")
    if (
        not isinstance(folds, Sequence)
        or isinstance(folds, (str, bytes))
        or len(folds) != 3
        or not all(_finite(value) for value in folds)
    ):
        raise ContractError("P3 fold deltas changed")
    if (
        not isinstance(slices, Mapping)
        or set(slices) != P3_SLICE_KEYS
        or not all(_finite(value) for value in slices.values())
    ):
        raise ContractError("P3 slice deltas changed")
    _mapping_all_true(evidence.get("leakage_checks"), role="P3 leakage checks")
    _mapping_all_true(
        evidence.get("reproducibility_checks"), role="P3 reproducibility checks"
    )
    gate = evidence.get("local_numeric_gate")
    promotion = evidence.get("official_promotion")
    if (
        not isinstance(gate, Mapping)
        or gate.get("decision") != "RESEARCH_ONLY"
        or gate.get("passed") is not False
        or not isinstance(promotion, Mapping)
        or promotion.get("allowed") is not False
    ):
        raise ContractError("P3 research-only evidence gate changed")


def _validate_training_receipts(
    metrics: Mapping[str, Any], outputs: Mapping[str, Any]
) -> None:
    receipts = metrics.get("training_receipts")
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)) or len(receipts) != 45:
        raise ContractError("P3 training receipt count changed")
    observed_cells: set[tuple[str, float, int]] = set()
    total_steps = 0
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ContractError("P3 training receipt is invalid")
        fold = receipt.get("fold")
        fraction = receipt.get("prefix_fraction")
        seed = receipt.get("seed")
        if fold not in P3_FOLDS or fraction not in P3_PREFIXES or seed not in P3_SEEDS:
            raise ContractError("P3 training receipt cell identity changed")
        cell = (str(fold), float(fraction), int(seed))
        if cell in observed_cells:
            raise ContractError("P3 training receipt cell is duplicated")
        observed_cells.add(cell)
        steps = receipt.get("optimizer_steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ContractError("P3 optimizer step count is invalid")
        total_steps += steps
        if (
            receipt.get("blind_prediction_sealed_before_validation_truth_attachment") is not True
            or receipt.get("saved_model_reload_prediction_exact") is not True
            or receipt.get("saved_model_reload_max_abs_difference_m") != 0.0
            or receipt.get("candidate_or_test_prediction") is not False
        ):
            raise ContractError("P3 blind/reload receipt guard changed")
        for path_key, sha_key in (
            ("model_relative_path", "model_sha256"),
            ("blind_prediction_relative_path", "blind_prediction_sha256"),
            ("cell_commitment_relative_path", "cell_commitment_sha256"),
            ("raw_cell_commitment_relative_path", "raw_cell_commitment_sha256"),
            ("raw_fold_commitment_relative_path", "raw_fold_commitment_sha256"),
        ):
            relative = receipt.get(path_key)
            if not isinstance(relative, str):
                raise ContractError("P3 receipt output path is missing")
            pin = _output_pin(outputs, relative, role=path_key)
            if pin["sha256"] != receipt.get(sha_key):
                raise ContractError("P3 receipt output SHA differs from manifest")
    expected_cells = {
        (fold, fraction, seed)
        for fold in P3_FOLDS
        for fraction in P3_PREFIXES
        for seed in P3_SEEDS
    }
    if observed_cells != expected_cells or total_steps != 10260:
        raise ContractError("P3 45-cell/10260-step execution lineage changed")


def _validate_metrics_curve(
    metrics: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    points = metrics.get("points")
    expected_keys = tuple(str(value) for value in P3_PREFIXES)
    if not isinstance(points, Mapping) or tuple(points) != expected_keys:
        raise ContractError("P3 metrics point keys changed")
    evidence_by_fraction = {
        float(point["fraction"]): point for point in evidence["points"]
    }
    for key, fraction in zip(expected_keys, P3_PREFIXES, strict=True):
        point = points.get(key)
        expected = evidence_by_fraction[fraction]
        if not isinstance(point, Mapping):
            raise ContractError("P3 metrics point is invalid")
        incumbent_seeds = point.get("incumbent_seed_metrics")
        challenger_seeds = point.get("challenger_seed_metrics")
        bootstrap = point.get("paired_whole_case_bootstrap")
        if (
            not isinstance(incumbent_seeds, Sequence)
            or isinstance(incumbent_seeds, (str, bytes))
            or len(incumbent_seeds) != 3
            or not all(_finite(value) for value in incumbent_seeds)
            or not isinstance(challenger_seeds, Sequence)
            or isinstance(challenger_seeds, (str, bytes))
            or len(challenger_seeds) != 3
            or not all(_finite(value) for value in challenger_seeds)
            or not isinstance(bootstrap, Mapping)
            or bootstrap.get("replicates") != 5000
            or bootstrap.get("cases") != 181
        ):
            raise ContractError("P3 3-seed/5000-bootstrap curve protocol changed")
        if (
            point.get("incumbent_rmse_m") != expected["incumbent"]
            or point.get("challenger_rmse_m") != expected["challenger"]
            or point.get("delta_ci90_m") != expected["delta_ci90"]
            or point.get("delta_candidate_minus_incumbent_m")
            != expected["challenger"] - expected["incumbent"]
        ):
            raise ContractError("P3 metrics/evidence point binding changed")
    prefix_audit = metrics.get("prefix_audit")
    if not isinstance(prefix_audit, Mapping) or tuple(prefix_audit) != (
        "040",
        "055",
        "070",
        "085",
        "100",
    ):
        raise ContractError("P3 prefix audit keys changed")
    for folds in prefix_audit.values():
        if not isinstance(folds, Mapping) or set(folds) != set(P3_FOLDS):
            raise ContractError("P3 prefix fold audit changed")
        if any(
            not isinstance(value, Mapping)
            or value.get("nested_subset_of_safe_outer_train") is not True
            for value in folds.values()
        ):
            raise ContractError("P3 prefix safety audit failed")
    split = metrics.get("split_audit")
    if (
        not isinstance(split, Mapping)
        or split.get("validation_case_count") != 181
        or split.get("validation_row_count") != 1086
        or split.get("unique_station_episode_count") != 181
        or split.get("repeated_station_episode_count") != 0
        or split.get("context48_plus_target24_footprint_overlap_pairs") != 0
        or split.get("cross_window_pairs_below_78h") != 0
        or split.get("r3_train_wave_hs_float_decodes") != 0
    ):
        raise ContractError("P3 split/OOF-key audit changed")


def _validate_predictions_complete(
    complete: Mapping[str, Any], outputs: Mapping[str, Any]
) -> None:
    if (
        complete.get("comparison_mode") != STRUCTURE_MODE
        or complete.get("fit_cell_count") != 45
        or complete.get("optimizer_steps") != 10260
        or complete.get("candidate_or_test_prediction") is not False
        or complete.get("validation_truth_attached") is not False
        or complete.get("all_validation_groups_released_only_after_fold_commitment") is not True
        or complete.get(
            "raw_delta_fold_commitment_preceded_validation_current_source_decode"
        )
        is not True
        or complete.get("unreleased_validation_current_hs_float_decodes") != 0
        or complete.get("unreleased_validation_train_wave_hs_float_decodes") != 0
        or complete.get("fold_order") != list(P3_FOLDS)
    ):
        raise ContractError("P3 predictions-complete guards changed")
    cells = complete.get("cell_commitments")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)) or len(cells) != 45:
        raise ContractError("P3 predictions-complete cell count changed")
    paths: set[str] = set()
    for pin in cells:
        if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256"}:
            raise ContractError("P3 cell commitment pin is invalid")
        relative = pin.get("path")
        if not isinstance(relative, str) or relative in paths:
            raise ContractError("P3 cell commitment path is invalid or duplicated")
        paths.add(relative)
        if _output_pin(outputs, relative, role="cell commitment")["sha256"] != pin.get(
            "sha256"
        ):
            raise ContractError("P3 cell commitment SHA differs from manifest")
    for key in ("fold_commitments", "raw_fold_commitments"):
        mapping = complete.get(key)
        if not isinstance(mapping, Mapping) or set(mapping) != set(P3_FOLDS):
            raise ContractError(f"P3 {key} changed")
        for fold, pin in mapping.items():
            if not isinstance(pin, Mapping) or pin.get("cell_count") != 15:
                raise ContractError(f"P3 {key} count changed")
            relative = pin.get("path")
            if not isinstance(relative, str) or _output_pin(
                outputs, relative, role=f"{key} {fold}"
            )["sha256"] != pin.get("sha256"):
                raise ContractError(f"P3 {key} pin differs from manifest")


def _zero_access_guards(value: Any, *, role: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractError(f"{role} access counters are missing")
    for key in (
        "anonymous_test_value_reads",
        "candidate_files_created",
        "forbidden_validation_current_hs_scalar_decodes",
        "forbidden_validation_target_scalar_decodes",
        "process_preflight_train_wave_hs_float_decodes",
        "test_value_reads",
        "upload_attempts",
    ):
        if value.get(key) != 0:
            raise ContractError(f"{role} guard is nonzero: {key}")


def _validate_p3_documents(
    *,
    config: Mapping[str, Any],
    qa: Mapping[str, Any],
    authorization: Mapping[str, Any],
    lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any],
    complete: Mapping[str, Any],
    registry: Mapping[str, Any],
    outputs: Mapping[str, Any],
    implementation_pins: Mapping[str, Any],
) -> None:
    config_short = _short_pin(P3_CORE_FILES["CONFIG"])
    qa_short = _short_pin(P3_CORE_FILES["QA"])
    if (
        config.get("problem") != "P3"
        or config.get("comparison_mode") != STRUCTURE_MODE
        or config.get("exact_official_incumbent_comparison") is not False
        or config.get("official_promotion_allowed") is not False
        or config.get("candidate_or_test_prediction_allowed") is not False
        or config.get("upload_allowed") is not False
        or config.get("execution_policy", {}).get("registry_append_allowed") is not False
    ):
        raise ContractError("P3 config research-only policy changed")
    _validate_point_evidence(evidence)
    if (
        qa.get("decision") != "GO_AUTHORIZE_P3_GEN5R4_DENSE72_R1"
        or qa.get("p0_count") != 0
        or qa.get("p1_count") != 0
        or qa.get("config") != config_short
        or qa.get("implementation_pins") != implementation_pins
    ):
        raise ContractError("P3 independent QA binding changed")
    if (
        authorization.get("config") != config_short
        or authorization.get("qa_receipt") != qa_short
        or authorization.get("implementation_pins") != implementation_pins
        or authorization.get("curve_execution_authorized") is not True
        or authorization.get("full_fit_or_candidate_authorized") is not False
        or authorization.get("test_prediction_authorized") is not False
        or authorization.get("upload_authorized") is not False
    ):
        raise ContractError("P3 authorization binding changed")
    operational = manifest.get("operational_snapshot_sha256")
    if not _is_sha256(operational) or any(
        document.get("operational_snapshot_sha256") != operational
        for document in (qa, authorization, lock)
    ):
        raise ContractError("P3 operational snapshot binding changed")
    if (
        lock.get("config") != config_short
        or lock.get("qa_receipt_sha256") != P3_CORE_FILES["QA"]["sha256"]
        or lock.get("authorization_sha256") != P3_CORE_FILES["AUTHORIZATION"]["sha256"]
        or lock.get("implementation_pins") != implementation_pins
        or lock.get("status") != "ATTEMPT_CONSUMED_BEFORE_CAPABILITY_MINT"
        or lock.get("capability_minted") is not False
        or lock.get("candidate_or_test_prediction_allowed") is not False
        or lock.get("upload_allowed") is not False
        or lock.get("rerun_allowed") is not False
        or lock.get("resume_allowed") is not False
    ):
        raise ContractError("P3 attempt-lock/capability binding changed")
    if (
        manifest.get("config") != config_short
        or manifest.get("status") != P3_STATUS
        or manifest.get("local_curve_qualified") is not False
        or manifest.get("official_promotion_allowed") is not False
        or manifest.get("candidate_created") is not False
        or manifest.get("official_upload_count") != 0
        or manifest.get("implementation_pins") != implementation_pins
        or manifest.get("oof_parquet_sha256") != P3_CORE_FILES["OOF"]["sha256"]
        or manifest.get("validation_keys_parquet_sha256")
        != P3_CORE_FILES["VALIDATION_KEYS"]["sha256"]
    ):
        raise ContractError("P3 manifest scientific/status binding changed")
    if (
        metrics.get("status") != P3_STATUS
        or metrics.get("comparison_mode") != STRUCTURE_MODE
        or metrics.get("candidate_created") is not False
        or metrics.get("test_prediction_created") is not False
        or metrics.get("full_fit_performed") is not False
        or metrics.get("local_numeric_curve_qualified") is not False
        or metrics.get("official_promotion_allowed") is not False
        or metrics.get("official_upload_count") != 0
        or metrics.get("local_gate") != evidence.get("local_numeric_gate")
        or metrics.get("leakage_checks") != evidence.get("leakage_checks")
        or metrics.get("reproducibility_checks") != evidence.get("reproducibility_checks")
    ):
        raise ContractError("P3 metrics research-only decision changed")
    _validate_metrics_curve(metrics, evidence)
    _validate_training_receipts(metrics, outputs)
    _validate_predictions_complete(complete, outputs)
    _zero_access_guards(manifest.get("access_counters"), role="P3 manifest")
    _zero_access_guards(metrics.get("access_counters"), role="P3 metrics")
    if (
        registry.get("status") != P3_STATUS
        or registry.get("candidate_created") is not False
        or registry.get("candidate_uploaded") is not False
        or registry.get("local_curve_qualified") is not False
        or registry.get("official_promotion_allowed") is not False
        or registry.get("official_upload_count") != 0
    ):
        raise ContractError("P3 registry research-only status changed")


def verify_p3_gen5r4_lineage(
    root: Path, contract: Mapping[str, Any], evidence_path: Path | None = None
) -> dict[str, Any]:
    route = contract.get("p3_gen5r4_structure_curve_route")
    if not isinstance(route, Mapping) or route.get("core_files") != P3_CORE_FILES:
        raise ContractError("P3 route core pins differ from the v9 contract")
    verified_core: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for role, expected in P3_CORE_FILES.items():
        pin, path = _verify_file_pin(root, expected, role=f"P3 {role}")
        verified_core[role] = pin
        paths[role] = path
    if evidence_path is not None and evidence_path.resolve(strict=True) != paths["EVIDENCE"]:
        raise ContractError("only the canonical P3 Gen5r4 evidence is accepted")
    artifact_lexical = _workspace(root) / P3_ARTIFACT_ROOT
    artifact_root = artifact_lexical.resolve(strict=True)
    if artifact_root != artifact_lexical or not artifact_root.is_dir():
        raise ContractError("P3 artifact root uses indirection")
    control_root = paths["QA"].parent
    control_files = {path.name for path in control_root.iterdir() if path.is_file()}
    if control_files != {"pre_execution_qa.json", "authorization.json", "attempt.lock"}:
        raise ContractError("P3 control file allowlist changed")
    manifest = _json_object(paths["MANIFEST"])
    outputs, actual = _verify_manifest_outputs(
        root, artifact_root, manifest, expected_count=237
    )
    manifest_sha_text = paths["MANIFEST_SHA256"].read_text(encoding="ascii")
    if manifest_sha_text != f'{P3_CORE_FILES["MANIFEST"]["sha256"]}  manifest.json\n':
        raise ContractError("P3 manifest.sha256 content changed")
    for role, relative in (
        ("METRICS", "metrics.json"),
        ("OOF", "oof/learning_curve_oof.parquet"),
        ("EVIDENCE", "learning_curve_evidence.json"),
        ("PREDICTIONS_COMPLETE", "commitments/predictions_complete.json"),
        ("REGISTRY", "registry.json"),
        ("VALIDATION_KEYS", "validation_keys.parquet"),
    ):
        output_pin = _output_pin(outputs, relative, role=role)
        if output_pin != {
            "sha256": P3_CORE_FILES[role]["sha256"],
            "bytes": P3_CORE_FILES[role]["bytes"],
        }:
            raise ContractError(f"P3 core {role} differs from manifest")
    implementation_pins = _verify_manifest_implementation_pins(root, manifest)
    input_pins = _validate_input_snapshots(manifest)
    config = _json_object(paths["CONFIG"])
    qa = _json_object(paths["QA"])
    authorization = _json_object(paths["AUTHORIZATION"])
    lock = _json_object(paths["ATTEMPT_LOCK"])
    metrics = _json_object(paths["METRICS"])
    evidence = _json_object(paths["EVIDENCE"])
    complete = _json_object(paths["PREDICTIONS_COMPLETE"])
    registry = _json_object(paths["REGISTRY"])
    _validate_p3_documents(
        config=config,
        qa=qa,
        authorization=authorization,
        lock=lock,
        manifest=manifest,
        metrics=metrics,
        evidence=evidence,
        complete=complete,
        registry=registry,
        outputs=outputs,
        implementation_pins=implementation_pins,
    )
    output_paths = set(outputs)
    counts = {
        "models": sum(path.startswith("models/") for path in output_paths),
        "blind_predictions": sum(
            path.startswith("blind_predictions/") for path in output_paths
        ),
        "blind_raw_deltas": sum(
            path.startswith("blind_raw_deltas/") for path in output_paths
        ),
        "cell_commitments": sum(
            path.startswith("commitments/cells/") for path in output_paths
        ),
        "raw_cell_commitments": sum(
            path.startswith("commitments/raw_cells/") for path in output_paths
        ),
        "fold_commitments": sum(
            path.startswith("commitments/folds/") for path in output_paths
        ),
        "raw_fold_commitments": sum(
            path.startswith("commitments/raw_folds/") for path in output_paths
        ),
    }
    if counts != {
        "models": 45,
        "blind_predictions": 45,
        "blind_raw_deltas": 45,
        "cell_commitments": 45,
        "raw_cell_commitments": 45,
        "fold_commitments": 3,
        "raw_fold_commitments": 3,
    }:
        raise ContractError("P3 45-model/prediction/commitment lineage changed")
    return {
        "artifact_root": P3_ARTIFACT_ROOT,
        "core_files": verified_core,
        "artifact_file_count": len(actual),
        "manifest_output_file_count": len(outputs),
        "manifest_output_map_sha256": P3_MANIFEST_OUTPUT_MAP_SHA256,
        "manifest_implementation_pin_count": len(implementation_pins),
        "manifest_implementation_pins_sha256": hashlib.sha256(
            canonical_json_bytes(implementation_pins)
        ).hexdigest(),
        "manifest_input_pin_count": len(input_pins),
        "manifest_input_pins_sha256": hashlib.sha256(
            canonical_json_bytes(input_pins)
        ).hexdigest(),
        "control_file_count": len(control_files),
        "execution_counts": {**counts, "fit_cells": 45, "optimizer_steps": 10260},
        "curve_protocol": {
            "points": 5,
            "folds": 3,
            "seeds": 3,
            "bootstrap_replicates_per_point": 5000,
            "validation_cases": 181,
            "validation_rows_per_fraction": 1086,
            "pinned_oof_rows": 5430,
        },
        "status": P3_STATUS,
        "candidate_created": False,
        "test_prediction_created": False,
        "official_upload_count": 0,
    }


def _canonical_qa_pin(root: Path, requested: Path) -> dict[str, Any]:
    workspace = _workspace(root)
    lexical = workspace / PRE_INIT_QA_RELATIVE
    expected = lexical.resolve(strict=False)
    if expected != lexical:
        raise ContractError("canonical v9 QA path must not use symlinks")
    resolved = requested.resolve(strict=True)
    if not resolved.is_relative_to(workspace) or resolved != expected:
        raise ContractError("v9 pre-init QA receipt path must be canonical")
    return _file_pin_for_path(root, resolved)


def _ledger_contract_pin(root: Path) -> dict[str, Any]:
    return _file_pin_for_path(root, _workspace_path(root, CONTRACT_RELATIVE))


def build_genesis_payload(root: Path, qa_receipt: Path) -> dict[str, Any]:
    contract = load_contract(root)
    predecessor = verify_predecessor(root, contract)
    p3_lineage = verify_p3_gen5r4_lineage(root, contract)
    implementation = current_implementation_pins(root)
    qa_pin = _canonical_qa_pin(root, qa_receipt)
    receipt = _json_object(_workspace_path(root, qa_pin["path"]))
    if (
        receipt.get("schema_version") != "meaningful_score_ledger_v9.pre_init_qa.v1"
        or receipt.get("decision") != "GO_INITIALIZE_V9_LEDGER"
        or receipt.get("p0_count") != 0
        or receipt.get("p1_count") != 0
        or receipt.get("ledger_contract") != _ledger_contract_pin(root)
        or receipt.get("predecessor_v8") != predecessor
        or receipt.get("p3_gen5r4_lineage") != p3_lineage
        or receipt.get("implementation_pins") != implementation
    ):
        raise ContractError("v9 pre-init QA receipt does not bind current genesis")
    initial = contract["initial_state"]
    return {
        "ledger_id": contract["ledger_id"],
        "status": initial["status"],
        "ledger_contract": _ledger_contract_pin(root),
        "predecessor_v8": predecessor,
        "predecessor_v8_anchor": _v8_anchor(),
        "p3_gen5r4_lineage": p3_lineage,
        "implementation_pins": implementation,
        "independent_pre_init_qa": qa_pin,
        "inherited_v8_event_count": initial["inherited_v8_event_count"],
        "inherited_v8_head_event_sha256": initial[
            "inherited_v8_head_event_sha256"
        ],
        "official_uploads": initial["official_uploads"],
        "score_promotions": initial["score_promotions"],
        "meaningful_promotions": initial["meaningful_promotions"],
        "execution_counts": initial["execution_counts"],
        "upload_performed": False,
    }


def _verify_genesis_payload(root: Path, payload: Mapping[str, Any]) -> None:
    qa = payload.get("independent_pre_init_qa")
    if not isinstance(qa, Mapping) or qa.get("path") != PRE_INIT_QA_RELATIVE:
        raise ContractError("v9 genesis canonical QA pin is missing")
    _, qa_path = _verify_file_pin(root, qa, role="v9 pre-init QA")
    if dict(payload) != build_genesis_payload(root, qa_path):
        raise ContractError("v9 genesis payload differs from canonical recomputation")


def _source_json_from_pin(
    root: Path, value: Any, *, role: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{role} pin is missing")
    pin, path = _verify_file_pin(root, value, role=role)
    return _json_object(path), pin, path


def _verify_inherited_policy(root: Path) -> dict[str, Any]:
    contract = load_contract(root)
    verify_predecessor(root, contract)
    return contract


def _p3_structure_decision(evidence: Mapping[str, Any]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for point in evidence["points"]:
        incumbent = float(point["incumbent"])
        challenger = float(point["challenger"])
        points.append(
            {
                "fraction": float(point["fraction"]),
                "incumbent": incumbent,
                "challenger": challenger,
                "delta_candidate_minus_incumbent": challenger - incumbent,
                "delta_ci90_candidate_minus_incumbent": [
                    float(point["delta_ci90"][0]),
                    float(point["delta_ci90"][1]),
                ],
            }
        )
    return {
        "schema_version": "meaningful_learning_curve_decision.v2.p3_structure_research_only.v1",
        "problem": "P3",
        "comparison_mode": STRUCTURE_MODE,
        "exact_official_incumbent_comparison": False,
        "metric": "integrated_RMSE_m",
        "direction": "lower",
        "points": points,
        "fold_deltas_candidate_minus_incumbent": list(
            evidence["fold_deltas_candidate_minus_incumbent"]
        ),
        "slice_deltas_candidate_minus_incumbent": dict(
            evidence["slice_deltas_candidate_minus_incumbent"]
        ),
        "local_numeric_gate": dict(evidence["local_numeric_gate"]),
        "decision": "RESEARCH_ONLY",
        "local_qualification": False,
        "passed": False,
        "official_promotion": False,
        "curve_alone_can_promote": False,
        "official_score_route_allowed": False,
        "upload_readiness_route_allowed": False,
        "official_promotion_reason": "PENDING_OFFICIAL_PAIRED_AB_AND_LOCAL_GATE_FAILED",
    }


def _build_p3_structure_curve_payload(
    root: Path, evidence_path: Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    contract = _verify_inherited_policy(root)
    lineage = verify_p3_gen5r4_lineage(root, contract, evidence_path)
    _validate_point_evidence(evidence)
    return {
        "schema_version": "meaningful_score_ledger_v9.p3_structure_matched_curve_result.v1",
        "evidence": _file_pin_for_path(root, evidence_path),
        "evidence_pins": lineage,
        "decision": _p3_structure_decision(evidence),
        "upload_performed": False,
    }


def build_curve_payload(root: Path, evidence_path: Path) -> dict[str, Any]:
    path = evidence_path.resolve(strict=True)
    evidence = _json_object(path)
    if _contains_structure_identity(evidence):
        if (
            evidence.get("comparison_mode") != STRUCTURE_MODE
            or evidence.get("problem") != "P3"
        ):
            raise ContractError("structure-matched P3 mode cannot be downgraded or mismatched")
        return _build_p3_structure_curve_payload(root, path, evidence)
    _verify_inherited_policy(root)
    return ledger_v8.build_curve_payload(root, evidence_path)


def build_official_score_payload(
    root: Path, evidence_path: Path, curve_decision_path: Path
) -> dict[str, Any]:
    evidence = _json_object(evidence_path.resolve(strict=True))
    curve = _json_object(curve_decision_path.resolve(strict=True))
    if _contains_structure_identity(evidence) or _contains_structure_identity(curve):
        raise ContractError("P3 structure-matched official-score route is forbidden")
    _verify_inherited_policy(root)
    return ledger_v8.build_official_score_payload(root, evidence_path, curve_decision_path)


def build_upload_readiness_payload(
    root: Path, receipt_path: Path, curve_decision_path: Path | None
) -> dict[str, Any]:
    receipt = _json_object(receipt_path.resolve(strict=True))
    curve = (
        None
        if curve_decision_path is None
        else _json_object(curve_decision_path.resolve(strict=True))
    )
    if _contains_structure_identity(receipt) or (
        curve is not None and _contains_structure_identity(curve)
    ):
        raise ContractError("P3 structure-matched upload-readiness route is forbidden")
    _verify_inherited_policy(root)
    return ledger_v8.build_upload_readiness_payload(root, receipt_path, curve_decision_path)


def build_goal_completion_payload(
    root: Path,
    evidence_path: Path,
    *,
    prior_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    _verify_inherited_policy(root)
    inherited = _validated_v8_records(root)
    combined = tuple(inherited) + tuple(prior_records)
    return ledger_v8.build_goal_completion_payload(
        root, evidence_path, prior_records=combined
    )


def recompute_later_payload(
    root: Path,
    event_type: str,
    payload: Mapping[str, Any],
    *,
    prior_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v9 event type is not allowlisted: {event_type}")
    if not isinstance(payload, Mapping):
        raise ContractError("v9 event payload must be an object")
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
            raise ContractError("GOAL_COMPLETION replay requires all prior v9 records")
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
        raise ContractError(f"v9 event type is not allowlisted: {event_type}")
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
    verify_p3_gen5r4_lineage(root, contract)
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    if not ledger.exists():
        return []
    raw = ledger.read_bytes()
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ContractError("v9 ledger must be non-empty canonical LF-terminated JSONL")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("v9 ledger is not UTF-8") from error
    records: list[dict[str, Any]] = []
    previous = V8_LEDGER_HEAD_SHA256
    for line_number, line in enumerate(text.split("\n")[:-1], 1):
        if not line.strip():
            raise ContractError(f"blank v9 ledger line {line_number}")
        record = _strict_json_loads(line)
        if not isinstance(record, dict) or set(record) != {
            "seq",
            "recorded_at_kst",
            "event_type",
            "previous_event_sha256",
            "payload",
            "event_sha256",
        }:
            raise ContractError(f"v9 ledger record keys changed at line {line_number}")
        if (
            not isinstance(record.get("seq"), int)
            or isinstance(record.get("seq"), bool)
            or record["seq"] != V8_LEDGER_EVENT_COUNT + len(records) + 1
        ):
            raise ContractError(f"v9 ledger sequence mismatch at line {line_number}")
        if record.get("previous_event_sha256") != previous:
            raise ContractError(f"v9 ledger chain mismatch at line {line_number}")
        timestamp = record.get("recorded_at_kst")
        if not isinstance(timestamp, str):
            raise ContractError(f"v9 ledger timestamp is invalid at line {line_number}")
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ContractError(f"v9 ledger timestamp is invalid at line {line_number}") from error
        if parsed.utcoffset() != timedelta(hours=9):
            raise ContractError(f"v9 ledger timestamp is not KST at line {line_number}")
        event_type = record.get("event_type")
        if event_type not in ALL_EVENT_TYPES:
            raise ContractError(f"v9 ledger event is not allowlisted at line {line_number}")
        claimed = record.get("event_sha256")
        if not _is_sha256(claimed):
            raise ContractError(f"v9 event SHA is invalid at line {line_number}")
        base = {key: value for key, value in record.items() if key != "event_sha256"}
        if hashlib.sha256(canonical_json_bytes(base)).hexdigest() != claimed:
            raise ContractError(f"v9 event hash mismatch at line {line_number}")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise ContractError(f"v9 payload is invalid at line {line_number}")
        if line_number == 1:
            if event_type != GENESIS_EVENT_TYPE:
                raise ContractError("v9 first event is not genesis")
            _verify_genesis_payload(root, payload)
        else:
            if event_type == GENESIS_EVENT_TYPE:
                raise ContractError("v9 ledger contains duplicate genesis")
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
            raise OSError(f"short v9 {role} write")
        offset += written


def initialize_ledger(
    root: Path, path: Path, *, payload: Mapping[str, Any]
) -> dict[str, Any]:
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    _verify_genesis_payload(root, payload)
    record = _ledger_record(
        seq=V8_LEDGER_EVENT_COUNT + 1,
        previous=V8_LEDGER_HEAD_SHA256,
        event_type=GENESIS_EVENT_TYPE,
        payload=payload,
    )
    encoded = canonical_json_bytes(record) + b"\n"
    if not ledger.parent.is_dir():
        raise ContractError("canonical v9 ledger directory must already exist via QA receipt")
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
        raise ContractError("v9 genesis failed replay round-trip validation")
    return record


def append_ledger_event(
    root: Path,
    path: Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if event_type not in LATER_EVENT_TYPES:
        raise ContractError(f"v9 later event is not allowlisted: {event_type}")
    ledger = _canonical_ledger_path(root, path, must_exist=False)
    existing = validate_ledger(root, ledger)
    if not existing:
        raise ContractError("v9 ledger must be initialized before append")
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
                    "schema_version": "meaningful_score_ledger_v9.append_lock.v1",
                    "ledger": ledger.name,
                    "pid": os.getpid(),
                    "expected_local_event_count": snapshot[0],
                    "expected_global_event_count": V8_LEDGER_EVENT_COUNT + snapshot[0],
                    "expected_head_event_sha256": snapshot[1],
                }
            )
            _write_all(descriptor, lock_payload, role="append-lock")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        locked = validate_ledger(root, ledger)
        if (len(locked), locked[-1]["event_sha256"]) != snapshot:
            raise ContractError("v9 ledger changed before lock acquisition")
        replayed = recompute_later_payload(
            root, event_type, canonical_payload, prior_records=tuple(locked)
        )
        record = _ledger_record(
            seq=V8_LEDGER_EVENT_COUNT + snapshot[0] + 1,
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
            raise ContractError("v9 append failed replay round-trip validation")
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
    "P3_ARTIFACT_ROOT",
    "P3_CORE_FILES",
    "PRE_INIT_QA_RELATIVE",
    "STRUCTURE_MODE",
    "V8_LEDGER_EVENT_COUNT",
    "V8_LEDGER_HEAD_SHA256",
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
    "verify_p3_gen5r4_lineage",
    "verify_predecessor",
]
