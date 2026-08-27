"""Fail-closed static bootstrap for brokered P1 multiscale Gen6r4.

Check-only is the only public CLI.  Execute entry has no role, cell, session,
prior, token, or handle argument: an externally authenticated PowerShell
supervisor must supply a live challenge transcript over inherited standard
handles.  No execution state is created until QA, authorization, provenance,
native-image, and canonical predecessor checks have all passed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import secrets
import struct
import sys
from pathlib import Path
from typing import Any

if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
    raise RuntimeError("canonical Gen6r4 bootstrap requires python -I -S -B")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
GENERATION = "p1_multiscale_cross_layer_offset_drift_unary_v6r4"
CONFIG_PIN = (
    "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r4.json",
    13681,
    "62a5ea4e2be163c4f815beaed6f3fb877dbc3d6091f81148e9437de9a6c27d44",
)

R3_FROZEN_PINS = {
    "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r3_startup_trust.json": (
        8249,
        "7dcc0c4a79eb3d1d67e22a1d9889e0160c082a925f460b884b1b7e50f5d75dfc",
    ),
    "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r3.json": (
        13513,
        "83eba89c97635ebc2bb38e22b7401b38ef980d89c7832cb3aab9abb490fd7cb8",
    ),
    "src/p1_qc/multiscale_cross_layer_offset_drift_contract_v6r3.py": (
        24539,
        "04f1325864ef983b70f1639844483b853067237d696146fe3dab934b0c7b7ef6",
    ),
    "src/p1_qc/multiscale_cross_layer_offset_drift_execution_v6r3.py": (
        46446,
        "7952e709788c5f398861df636680307930b00a98b5fcdfa78b9b6db8b38e08d1",
    ),
    "src/p1_qc/multiscale_cross_layer_offset_drift_verifier_v6r3.py": (
        10051,
        "b016b1f767e2f2b12e79ef086b2193fceab484997531db292fc9a0162867571d",
    ),
    "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6r3.py": (
        1595,
        "85f5dca139d5a98eaba12105e96ff4d3c5606ce2827fe998b384759ef22fea59",
    ),
    "scripts/bootstrap_p1_multiscale_cross_layer_offset_drift_unary_v6r3.py": (
        74549,
        "e9e78d80559e234d6fde88a97e02d87325a2313435d32823956bd37cf65deecc",
    ),
    "scripts/launch_p1_multiscale_cross_layer_offset_drift_unary_v6r3.ps1": (
        16092,
        "4c5064e95ea22f19b7b2154dca4f86b38b91babc888ffdc1520178ad20868783",
    ),
    "scripts/p1_multiscale_cross_layer_offset_drift_unary_v6r3_stage0.ps1.txt": (
        7911,
        "1bb46d84f8325d9d704a6967244aec381a0b86a84df7338c50ffd34556c720eb",
    ),
    "tests/test_run_p1_multiscale_cross_layer_offset_drift_unary_v6r3.py": (
        25475,
        "d0f320178f5dc402ee11f9944bb8fea2e27697e7a479ccf5373d9d1d7719889f",
    ),
}

R3_DISPOSITION_PINS = {
    "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r3_disposition/OWNER_STATIC_QA_NO_GO_20260823.json": (
        9342,
        "190f3955ae0d4e63ba1b70df699ada6011cd1ee933424f0704d6a7bbeb453623",
    ),
    "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r3_disposition/EXECUTION_TOMBSTONE.json": (
        2375,
        "54d84540feb5fac49b4408ab7cf5b45c49f20cfe1acde7526561ebe517c86f56",
    ),
}

OWNER_ROLE_PINS = {
    "CONTRACT": (
        "src/p1_qc/multiscale_cross_layer_offset_drift_contract_v6r4.py",
        17791,
        "1cbbebc844f793bb6e319d80e3d6c73bb68635f351cfabbddf5f588c1cb92c22",
    ),
    "BROKER": (
        "src/p1_qc/multiscale_cross_layer_offset_drift_target_broker_v6r4.py",
        14724,
        "d530ba2fbe27f82882241d5ee5e66d13508306d31d3a0048a2d7319c998889c7",
    ),
    "ENGINE": (
        "src/p1_qc/multiscale_cross_layer_offset_drift_execution_v6r4.py",
        25605,
        "4592226a6964a7d87b695507c1a819592eba25d45c3469fd7772481bc7197c63",
    ),
    "VERIFIER": (
        "src/p1_qc/multiscale_cross_layer_offset_drift_verifier_v6r4.py",
        32612,
        "8734f5d9ae6ea66dbfeaeff3d2ff68d4a646d075d7647a88075bc269088d9cf0",
    ),
    "RUNNER": (
        "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6r4.py",
        1473,
        "f9bde62b4f3d4ecfeaa7d78fcf9e2137d348b50249c491e23c21784ce66319af",
    ),
}

STATIC_ROLE_PINS = {
    "STARTUP_TRUST": (
        "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r4_startup_trust.json",
        6498,
        "9b8148f1aae2dde54adc026dfdde2cdabc3c2b930a6596f2275c9e8169e2c989",
    ),
    "TESTS": (
        "tests/test_run_p1_multiscale_cross_layer_offset_drift_unary_v6r4.py",
        18531,
        "55f6485627548adf0d13717074066047f6916848667ddc707f72e799902baa6e",
    ),
}

FUTURE_STATE = (
    "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r4_control/pre_execution_qa.json",
    "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r4_control/execution_authorization.json",
    "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r4_control/attempt.lock",
    "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r4",
    "artifacts/status/p1_multiscale_cross_layer_offset_drift_unary_v6r4.json",
    "submissions/p1_multiscale_cross_layer_offset_drift_unary_v6r4.csv",
)

FORBIDDEN_IMPORTS_AFTER_HANDOFF = frozenset(
    {
        "_winapi",
        "mmap",
        "ctypes",
        "_ctypes",
        "_xxsubinterpreters",
        "_interpreters",
        "interpreters",
        "multiprocessing",
        "subprocess",
    }
)
FORBIDDEN_CALLS = frozenset(
    {
        "os.pipe",
        "os.pipe2",
        "os.system",
        "os.popen",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
    }
)


class BootstrapError(RuntimeError):
    """Static or inherited launch trust differs."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_json(payload: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise BootstrapError(f"{label} duplicate key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise BootstrapError(f"{label} non-finite constant {value}")

    try:
        return json.loads(payload, object_pairs_hook=pairs, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"{label} is not strict JSON") from exc


def _read_pin(relative: str, expected_bytes: int, expected_sha256: str) -> bytes:
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise BootstrapError(f"pinned regular file is absent: {relative}")
    payload = path.read_bytes()
    if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise BootstrapError(f"pinned bytes differ: {relative}")
    return payload


def _audit_source_ast(relative: str, payload: bytes) -> dict[str, Any]:
    try:
        tree = ast.parse(payload, filename=relative)
    except SyntaxError as exc:
        raise BootstrapError(f"owner source does not parse: {relative}") from exc
    forbidden_imports: list[str] = []
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden_imports.extend(
                alias.name for alias in node.names if alias.name.split(".")[0] in FORBIDDEN_IMPORTS_AFTER_HANDOFF
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in FORBIDDEN_IMPORTS_AFTER_HANDOFF:
                forbidden_imports.append(module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                name = f"{node.func.value.id}.{node.func.attr}"
                if name in FORBIDDEN_CALLS:
                    forbidden_calls.append(name)
    if forbidden_imports or forbidden_calls:
        raise BootstrapError(
            f"owner source uses forbidden capability: {relative}: "
            f"imports={forbidden_imports}, calls={forbidden_calls}"
        )
    return {
        "path": relative,
        "forbidden_imports": 0,
        "forbidden_calls": 0,
        "subinterpreter_entry_present": False,
        "os_pipe_entry_present": False,
    }


def _verify_r3_disposition() -> dict[str, Any]:
    owner = _strict_json(
        _read_pin(
            "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r3_disposition/OWNER_STATIC_QA_NO_GO_20260823.json",
            *R3_DISPOSITION_PINS[
                "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r3_disposition/OWNER_STATIC_QA_NO_GO_20260823.json"
            ],
        ),
        "Gen6r3 owner NO-GO",
    )
    tombstone_relative = (
        "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r3_disposition/EXECUTION_TOMBSTONE.json"
    )
    tombstone = _strict_json(
        _read_pin(tombstone_relative, *R3_DISPOSITION_PINS[tombstone_relative]),
        "Gen6r3 tombstone",
    )
    expected_findings = [
        "RAW_TARGET_BUFFER_BYPASSES_CELL_VAULT",
        "DIRECT_WORKER_NOT_BOUND_TO_PERSISTED_PREDECESSOR",
        "NATIVE_DIGEST_NOT_BOUND_TO_EXTENSION_LOAD",
        "EXTERNAL_POWERSHELL_TCB_NOT_BOUND_TO_CHILD",
        "FIREWALL_CAPABILITY_AND_MAIN_AUTHORITY_REACHABLE",
        "CELL_AND_FINAL_VERIFIERS_NOT_FULL_SEMANTIC_VERIFIERS",
    ]
    if (
        owner.get("reviewer") != "/root/p1_gen6r3_independent_qa"
        or owner.get("verdict") != "P0=3_P1=3_NO_GO"
        or owner.get("p0_count") != 3
        or owner.get("p1_count") != 3
        or owner.get("independent_qa_receipt", {}).get("present") is not False
        or [item.get("code") for item in owner.get("findings", [])] != expected_findings
        or tombstone.get("status") != "PERMANENTLY_TOMBSTONED_NEVER_EXECUTE"
        or tombstone.get("review", {}).get("finding_codes") != expected_findings
    ):
        raise BootstrapError("Gen6r3 NO-GO/tombstone semantics differ")
    return {"reviewer": owner["reviewer"], "p0": 3, "p1": 3, "qa_receipt": False}


def _verify_v9_anchor() -> dict[str, Any]:
    relative = "artifacts/meaningful_score_goal_v9/registry.jsonl"
    payload = _read_pin(
        relative,
        15812,
        "232b6ed3133de11ee05150ec439efe05baa315bbb64ea0f319ffcbddd421b965",
    )
    events = [_strict_json(line, f"v9 event {index}") for index, line in enumerate(payload.splitlines())]
    if (
        [event.get("seq") for event in events] != [3, 4, 5]
        or events[-1].get("event_sha256")
        != "1b3e01be70c6f8ed2df04038deac3b3642804f70f9f17a238826c64d68090317"
        or any(event.get("payload", {}).get("upload_performed") is True for event in events)
    ):
        raise BootstrapError("v9 seq5/upload-zero anchor differs")
    return {"bytes": len(payload), "seq": 5, "uploads": 0}


def _compile_contract(payload: bytes) -> Any:
    namespace = {
        "__name__": "_p1_v6r4_authenticated_contract",
        "_P1_V6R4_BOOTSTRAP_CONTEXT": {"all_owner_roles_authenticated": True},
    }
    exec(compile(payload, OWNER_ROLE_PINS["CONTRACT"][0], "exec"), namespace)
    return type("AuthenticatedContract", (), namespace)


def static_check_only() -> dict[str, Any]:
    r3_pins = []
    for relative, (expected_bytes, expected_sha) in R3_FROZEN_PINS.items():
        _read_pin(relative, expected_bytes, expected_sha)
        r3_pins.append(relative)
    disposition = _verify_r3_disposition()
    config_payload = _read_pin(*CONFIG_PIN)
    trust_payload = _read_pin(*STATIC_ROLE_PINS["STARTUP_TRUST"])
    config = _strict_json(config_payload, "Gen6r4 config")
    trust = _strict_json(trust_payload, "Gen6r4 startup trust")
    if (
        config.get("generation") != GENERATION
        or config.get("status")
        != "STATIC_ONLY_AWAITING_FRESH_INDEPENDENT_QA_AND_EXPLICIT_EXECUTION_AUTHORIZATION"
        or trust.get("generation") != GENERATION
    ):
        raise BootstrapError("Gen6r4 config/startup trust semantics differ")
    declared_roles = config.get("owner_roles")
    expected_roles = {**STATIC_ROLE_PINS, **OWNER_ROLE_PINS}
    if type(declared_roles) is not dict or set(declared_roles) != set(expected_roles):
        raise BootstrapError("Gen6r4 owner role set differs")
    for role, (relative, expected_bytes, expected_sha) in expected_roles.items():
        if declared_roles.get(role) != {
            "path": relative,
            "bytes": expected_bytes,
            "sha256": expected_sha,
        }:
            raise BootstrapError(f"Gen6r4 owner role pin differs: {role}")
        _read_pin(relative, expected_bytes, expected_sha)
    role_audits = []
    role_payloads: dict[str, bytes] = {}
    for role, (relative, expected_bytes, expected_sha) in OWNER_ROLE_PINS.items():
        payload = _read_pin(relative, expected_bytes, expected_sha)
        role_payloads[role] = payload
        role_audits.append(_audit_source_ast(relative, payload))
    contract_module = _compile_contract(role_payloads["CONTRACT"])
    summary = contract_module.static_contract_summary()
    if (
        summary.get("cells") != 15
        or summary.get("sanitized_buffers") != 15
        or summary.get("target_brokers") != 1
        or summary.get("teacher_receipts") != 135
        or summary.get("score_calls", {}).get("total") != 68
        or summary.get("output_files") != 202
        or summary.get("execution_authority_present") is not False
    ):
        raise BootstrapError("Gen6r4 static contract summary differs")
    present = [relative for relative in FUTURE_STATE if (ROOT / relative).exists()]
    if present:
        raise BootstrapError(f"Gen6r4 future execution state exists: {present}")
    return {
        "schema_version": "p1_multiscale_cross_layer_offset_drift_unary.v6r4.static_check.v1",
        "generation": GENERATION,
        "status": "STATIC_CHECK_PASS_AWAITING_FRESH_INDEPENDENT_QA",
        "r3_pins_verified": len(r3_pins),
        "r3_disposition": disposition,
        "owner_roles_verified": sorted(OWNER_ROLE_PINS),
        "capability_ast_audits": role_audits,
        "contract": summary,
        "v9": _verify_v9_anchor(),
        "future_state_absent": len(FUTURE_STATE),
        "qa_receipt_present": False,
        "authorization_present": False,
        "attempt_lock_present": False,
        "run_performed": False,
        "fits": 0,
        "predictions": 0,
        "scores": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "ledger_appends": 0,
        "uploads": 0,
    }


class _InheritedProtocol:
    """Length-framed live supervisor protocol over inherited standard handles."""

    __slots__ = ("_reader", "_writer", "_closed", "_native_images")

    def __init__(self) -> None:
        if os.isatty(0) or os.isatty(1):
            raise BootstrapError("execute entry requires inherited non-terminal IPC handles")
        self._reader = sys.stdin.buffer
        self._writer = sys.stdout.buffer
        self._closed = False
        self._native_images: dict[str, dict[str, Any]] = {}

    def receive(self) -> dict[str, Any]:
        header = self._reader.read(8)
        if len(header) != 8:
            raise BootstrapError("inherited protocol frame header is absent")
        size = struct.unpack("<Q", header)[0]
        if not 1 <= size <= 64 * 1024 * 1024:
            raise BootstrapError("inherited protocol frame size differs")
        payload = self._reader.read(size)
        if len(payload) != size:
            raise BootstrapError("inherited protocol frame is truncated")
        value = _strict_json(payload, "inherited protocol frame")
        if type(value) is not dict:
            raise BootstrapError("inherited protocol frame is not an object")
        return value

    def send(self, value: dict[str, Any]) -> None:
        if self._closed:
            raise BootstrapError("inherited protocol is closed")
        payload = _canonical(value)
        self._writer.write(struct.pack("<Q", len(payload)) + payload)
        self._writer.flush()

    def challenge(self, trust: dict[str, Any]) -> dict[str, Any]:
        envelope = self.receive()
        nonce = envelope.get("nonce")
        if type(nonce) is not str or len(nonce) != 64:
            raise BootstrapError("external launch nonce differs")
        challenge = secrets.token_hex(32)
        self.send({"kind": "child_challenge", "challenge": challenge})
        response = self.receive()
        expected = hashlib.sha256(
            (nonce + challenge + envelope.get("encoded_command_sha256", "") + envelope.get("launcher_sha256", "")).encode(
                "ascii", errors="strict"
            )
        ).hexdigest()
        if (
            response
            != {
                "kind": "supervisor_challenge_response",
                "response_sha256": expected,
            }
            or envelope.get("schema_version") != "p1_v6r4_external_launch_envelope.v1"
            or envelope.get("generation") != GENERATION
            or envelope.get("public_cli_fields_absent") is not True
            or type(envelope.get("encoded_command_sha256")) is not str
            or len(envelope["encoded_command_sha256"]) != 64
            or type(envelope.get("launcher_sha256")) is not str
            or len(envelope["launcher_sha256"]) != 64
            or trust.get("live_child_provenance", {}).get(
                "inherited_redirected_duplex_handles_required"
            )
            is not True
        ):
            raise BootstrapError("live external EncodedCommand provenance challenge differs")
        return envelope

    def stage_native(self, original: str) -> dict[str, Any]:
        self.send({"kind": "stage_native_private_image", "original": original})
        record = self.receive()
        if (
            record.get("kind") != "staged_native_private_image"
            or record.get("source_path") != original
            or record.get("deny_write_delete_handle_held") is not True
            or record.get("o_excl_private_image") is not True
            or record.get("nlink") != 1
            or record.get("reparse") is not False
            or type(record.get("path")) is not str
            or type(record.get("bytes")) is not int
            or type(record.get("sha256")) is not str
        ):
            raise BootstrapError("supervisor could not stage immutable authenticated native image")
        self._native_images[original] = record
        return record

    def rehash_native(self, original: str, phase: str) -> dict[str, Any]:
        record = self._native_images.get(original)
        if record is None:
            raise BootstrapError("native image was not privately staged")
        self.send(
            {
                "kind": "same_handle_native_rehash",
                "path": record["path"],
                "phase": phase,
            }
        )
        response = self.receive()
        if (
            response.get("kind") != "same_handle_native_rehash_result"
            or response.get("path") != record["path"]
            or response.get("bytes") != record["bytes"]
            or response.get("sha256") != record["sha256"]
            or response.get("same_held_handle") is not True
        ):
            raise BootstrapError(f"native same-handle digest differs at {phase}")
        return record

    def verify_native_exit(self) -> None:
        for original in sorted(self._native_images):
            self.rehash_native(original, "process_exit")

    def close(self) -> None:
        self.verify_native_exit()
        self.send({"kind": "child_protocol_close"})
        self._closed = True


class _AuthenticatedPrivateExtensionLoader(importlib.machinery.ExtensionFileLoader):
    """Load only an O_EXCL staged image held deny-write/delete by supervisor."""

    def __init__(self, fullname: str, path: str, protocol: _InheritedProtocol) -> None:
        self._source_path = path
        self._protocol = protocol
        record = protocol.stage_native(path)
        self._staged_path = record["path"]
        super().__init__(fullname, record["path"])

    def create_module(self, spec: Any) -> Any:
        if os.path.normcase(os.path.abspath(spec.origin)) != os.path.normcase(
            os.path.abspath(self._source_path)
        ):
            raise BootstrapError("native import spec does not name the authenticated source")
        # ExtensionFileLoader delegates to _imp.create_dynamic(spec), which
        # consumes spec.origin rather than loader.path.  Bind the OS load to the
        # O_EXCL staged, supervisor-held immutable image explicitly.
        spec.origin = self._staged_path
        self._protocol.rehash_native(self._source_path, "immediately_before_os_load")
        module = super().create_module(spec)
        self._protocol.rehash_native(self._source_path, "immediately_after_os_load")
        if os.path.normcase(os.path.abspath(spec.origin)) != os.path.normcase(
            os.path.abspath(self._staged_path)
        ):
            raise BootstrapError("OS extension loader did not retain the staged image origin")
        return module

    def exec_module(self, module: Any) -> None:
        self._protocol.rehash_native(self._source_path, "immediately_before_exec")
        super().exec_module(module)
        self._protocol.rehash_native(self._source_path, "immediately_after_exec")


def _sealed_audit_hook(event: str, arguments: tuple[object, ...]) -> None:
    if event == "import" and arguments:
        name = arguments[0]
        if type(name) is str and name.split(".")[0] in FORBIDDEN_IMPORTS_AFTER_HANDOFF:
            raise PermissionError(f"fresh dangerous import is forbidden: {name}")
    if event in {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.posix_spawnp",
        "winreg.CreateKey",
        "socket.__new__",
        "socket.connect",
        "socket.bind",
        "ctypes.dlopen",
        "ctypes.dlsym",
    }:
        raise PermissionError(f"unaudited dangerous capability is forbidden: {event}")
    if event == "open" and len(arguments) >= 2:
        mode = arguments[1]
        if type(mode) is str and any(flag in mode for flag in "wax+"):
            raise PermissionError("direct write-capable open is forbidden")


def _private_execute() -> dict[str, Any]:
    trust = _strict_json(
        _read_pin(*STATIC_ROLE_PINS["STARTUP_TRUST"]), "Gen6r4 startup trust"
    )
    protocol = _InheritedProtocol()
    envelope = protocol.challenge(trust)
    role = envelope.get("role")
    if role not in {"parent", "target_broker", "cell_worker"}:
        raise BootstrapError("private inherited role differs")
    if role == "cell_worker" and type(envelope.get("cell")) is not int:
        raise BootstrapError("private worker cell differs")
    if role != "cell_worker" and envelope.get("cell") is not None:
        raise BootstrapError("non-worker private launch names a cell")
    # No control state is created here.  The supervisor must first return exact
    # pre-existing QA and explicit authorization pins; both are absent in the
    # owner-static zero state and therefore execution currently fails closed.
    protocol.send({"kind": "request_preexisting_control_pins"})
    controls = protocol.receive()
    if controls.get("kind") != "preexisting_control_pins":
        raise BootstrapError("pre-existing control pin response differs")
    qa = controls.get("qa_receipt")
    authorization = controls.get("authorization")
    if type(qa) is not dict or type(authorization) is not dict:
        raise BootstrapError("fresh QA and explicit authorization are required before execution")
    raise BootstrapError(
        "Gen6r4 owner-static cycle cannot mint execution authority; a future fresh QA cycle must "
        "supply the audited native supervisor service binding"
    )


def main(argv: list[str]) -> int:
    if argv == ["--check-only"]:
        print(_canonical(static_check_only()).decode("utf-8"))
        return 0
    if argv == ["--execute"]:
        # --execute conveys no role/cell/session/prior/handle.  All such state
        # must arrive over inherited handles and pass the live challenge.
        sys.addaudithook(_sealed_audit_hook)
        result = _private_execute()
        print(_canonical(result).decode("utf-8"))
        return 0
    raise BootstrapError("Gen6r4 accepts only public --check-only or private --execute")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
