from __future__ import annotations

import hashlib
import inspect
import json
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r4.json"
TRUST_PATH = (
    ROOT
    / "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r4_startup_trust.json"
)
CONTRACT_PATH = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift_contract_v6r4.py"
BROKER_PATH = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift_target_broker_v6r4.py"
ENGINE_PATH = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift_execution_v6r4.py"
VERIFIER_PATH = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift_verifier_v6r4.py"
RUNNER_PATH = ROOT / "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6r4.py"
BOOTSTRAP_PATH = ROOT / "scripts/bootstrap_p1_multiscale_cross_layer_offset_drift_unary_v6r4.py"
STAGE1_PATH = ROOT / "scripts/launch_p1_multiscale_cross_layer_offset_drift_unary_v6r4.ps1"
STAGE0_PATH = ROOT / "scripts/p1_multiscale_cross_layer_offset_drift_unary_v6r4_stage0.ps1.txt"
NO_GO_PATH = (
    ROOT
    / "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r3_disposition/"
    "OWNER_STATIC_QA_NO_GO_20260823.json"
)
TOMBSTONE_PATH = NO_GO_PATH.with_name("EXECUTION_TOMBSTONE.json")

R3_PINS = {
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

FINDINGS = [
    "RAW_TARGET_BUFFER_BYPASSES_CELL_VAULT",
    "DIRECT_WORKER_NOT_BOUND_TO_PERSISTED_PREDECESSOR",
    "NATIVE_DIGEST_NOT_BOUND_TO_EXTENSION_LOAD",
    "EXTERNAL_POWERSHELL_TCB_NOT_BOUND_TO_CHILD",
    "FIREWALL_CAPABILITY_AND_MAIN_AUTHORITY_REACHABLE",
    "CELL_AND_FINAL_VERIFIERS_NOT_FULL_SEMANTIC_VERIFIERS",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_contract() -> types.ModuleType:
    module = types.ModuleType("p1_v6r4_contract_test")
    module.__dict__["_P1_V6R4_BOOTSTRAP_CONTEXT"] = {
        "all_owner_roles_authenticated": True
    }
    exec(compile(CONTRACT_PATH.read_bytes(), str(CONTRACT_PATH), "exec"), module.__dict__)
    return module


def _load_verifier(contract: types.ModuleType) -> types.ModuleType:
    module = types.ModuleType("p1_v6r4_verifier_test")
    module.__dict__.update(
        {
            "_P1_V6R4_VERIFIER_CONTEXT": {
                "all_owner_roles_authenticated": True,
                "execution_authority_present": False,
            },
            "_P1_V6R4_AUTH_CONTRACT": contract,
        }
    )
    exec(compile(VERIFIER_PATH.read_bytes(), str(VERIFIER_PATH), "exec"), module.__dict__)
    return module


def _load_broker(contract: types.ModuleType) -> tuple[types.ModuleType, dict[str, Any]]:
    raw = (
        b"station,year,layer,time,temp,psal,depth,label,anomaly_type\n"
        b"A,2024,1,t0,1,2,3,0,normal\n"
        b"A,2024,1,t1,2,3,4,1,offset\n"
        b"A,2024,1,t2,3,4,5,0,normal\n"
        b"A,2024,1,t3,4,5,6,1,drift\n"
    )
    consumed = {"value": False}
    active = {"cell": 1}
    events: dict[str, tuple[bytes, dict[str, Any]]] = {}

    def source_once() -> bytes:
        assert consumed["value"] is False
        consumed["value"] = True
        return raw

    def plan(_sanitized: bytes, _cell: int) -> dict[str, list[int]]:
        return {
            "inner_1_train": [0],
            "inner_1_gate": [1],
            "inner_2_train": [0],
            "inner_2_gate": [2],
            "inner_3_train": [0],
            "inner_3_gate": [3],
            "outer_train": [0, 1, 2],
            "outer_validation": [3],
        }

    def rehash(pin: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
        return events[pin["path"]]

    context = {
        "private_inherited_broker_entry": True,
        "public_cli_fields_absent": True,
        "consume_authenticated_target_source_once": source_once,
        "derive_exact_cell_plan": plan,
        "active_sanitized_channel_cell": lambda: active["cell"],
        "rehash_persisted_event_same_handle": rehash,
        "canonical_predecessor_for_cell": lambda _cell: contract.GENESIS_SHA256,
    }
    module = types.ModuleType("p1_v6r4_broker_test")
    module.__dict__.update(
        {
            "_P1_V6R4_BROKER_CONTEXT": context,
            "_P1_V6R4_AUTH_CONTRACT": contract,
        }
    )
    exec(compile(BROKER_PATH.read_bytes(), str(BROKER_PATH), "exec"), module.__dict__)
    return module, {"events": events, "active": active}


def _register_event(
    store: dict[str, tuple[bytes, dict[str, Any]]],
    *,
    path: str,
    event: dict[str, Any],
) -> dict[str, Any]:
    payload = json.dumps(
        event, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    pin = {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    store[path] = (payload, pin)
    return pin


def test_all_frozen_v6r3_bytes_remain_exact() -> None:
    for relative, (expected_bytes, expected_sha) in R3_PINS.items():
        path = ROOT / relative
        assert path.stat().st_size == expected_bytes
        assert _sha(path) == expected_sha


def test_owner_no_go_and_tombstone_bind_exact_review_without_receipt() -> None:
    owner = json.loads(NO_GO_PATH.read_bytes())
    tombstone = json.loads(TOMBSTONE_PATH.read_bytes())
    assert owner["reviewer"] == "/root/p1_gen6r3_independent_qa"
    assert owner["verdict"] == "P0=3_P1=3_NO_GO"
    assert owner["p0_count"] == 3
    assert owner["p1_count"] == 3
    assert owner["independent_qa_receipt"] == {
        "present": False,
        "path": None,
        "bytes": None,
        "sha256": None,
        "reason": "The independent reviewer reported the verdict conversationally and did not materialize a receipt file.",
    }
    assert [finding["code"] for finding in owner["findings"]] == FINDINGS
    assert tombstone["owner_no_go_receipt"] == {
        "path": NO_GO_PATH.relative_to(ROOT).as_posix(),
        "bytes": NO_GO_PATH.stat().st_size,
        "sha256": _sha(NO_GO_PATH),
    }
    assert tombstone["review"]["finding_codes"] == FINDINGS
    assert tombstone["status"] == "PERMANENTLY_TOMBSTONED_NEVER_EXECUTE"


def test_raw_target_buffer_is_broker_only_and_fifteen_buffers_are_cell_specific() -> None:
    contract = _load_contract()
    broker_module, state = _load_broker(contract)
    broker = broker_module.TargetBroker()
    pins = []
    for cell in range(1, 16):
        state["active"]["cell"] = cell
        envelope = broker.sanitized_cell_buffer(cell)
        verified = contract.verify_sanitized_buffer(
            envelope["payload"], cell=cell, expected_pin=envelope["pin"]
        )
        assert verified["rows"] == 4
        assert envelope["payload"].splitlines()[1] == (
            b"station,year,layer,time,temp,psal,depth"
        )
        pins.append(envelope["pin"]["sha256"])
    assert len(set(pins)) == 15
    engine_text = ENGINE_PATH.read_text(encoding="utf-8")
    assert "authenticated_train_bytes" not in engine_text
    assert "raw_target_bytes_available\") is not False" in engine_text
    assert "parent_received_raw_target_bytes" not in engine_text


def test_broker_has_phase_only_api_and_rejects_arbitrary_ids_and_precommit_gate() -> None:
    contract = _load_contract()
    broker_module, state = _load_broker(contract)
    broker = broker_module.TargetBroker()
    predecessor = {
        "schema_version": "p1_v6r4_session.v1",
        "generation": contract.GENERATION,
        "prior_event_sha256": contract.GENESIS_SHA256,
        "event_sha256": contract.GENESIS_SHA256,
    }
    predecessor_pin = _register_event(
        state["events"], path="commitments/session.json", event=predecessor
    )
    assert broker.bind_predecessor(1, predecessor_pin) == contract.GENESIS_SHA256
    release = broker.release(1, "inner_1_train", predecessor_pin)
    assert release["labels"] == [0]
    assert release["phase"] == "inner_1_train"
    with pytest.raises(TypeError):
        broker.release(1, "inner_1_gate", predecessor_pin, ids=[1])
    with pytest.raises(PermissionError):
        broker.release(1, "inner_1_gate", predecessor_pin)
    parameters = tuple(inspect.signature(broker.release).parameters)
    assert parameters == ("cell", "phase", "evidence_pin")


def test_worker_entry_has_no_public_cell_session_or_prior_cli() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    assert "--cell-worker" not in runner
    assert "--cell-worker" not in bootstrap
    assert "--session" not in runner
    assert "--prior" not in runner
    assert 'argv == ["--execute"]' in bootstrap
    assert "inherited non-terminal IPC handles" in bootstrap
    assert "validate_persisted_canonical_next_cell" in ENGINE_PATH.read_text(encoding="utf-8")
    contract = _load_contract()
    with pytest.raises(contract.CapabilityError):
        contract.validate_launch_envelope(
            {
                "schema_version": "p1_v6r4_private_inherited_launch.v1",
                "generation": contract.GENERATION,
                "role": "cell_worker",
                "cell": 1,
                "session_sha256": "1" * 64,
                "inherited_handle_identity": "2" * 64,
                "challenge_sha256": "3" * 64,
                "encoded_command_sha256": "4" * 64,
                "launcher_sha256": "5" * 64,
                "public_cli_fields_absent": False,
            },
            role="cell_worker",
            inherited_handle_identity="2" * 64,
            challenge_sha256="3" * 64,
        )


def test_native_load_uses_immutable_staged_image_and_same_handle_exit_rehash() -> None:
    bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    launcher = STAGE1_PATH.read_text(encoding="utf-8")
    trust = json.loads(TRUST_PATH.read_bytes())
    assert "stage_native_private_image" in bootstrap
    assert "immediately_before_os_load" in bootstrap
    assert "immediately_after_os_load" in bootstrap
    assert 'rehash_native(original, "process_exit")' in bootstrap
    assert "super().__init__(fullname, record[\"path\"])" in bootstrap
    assert "Stage-AuthenticatedNativeImage" in launcher
    assert "[System.IO.FileMode]::CreateNew" in launcher
    assert "[System.IO.FileShare]::Read" in launcher
    native = trust["native_extension_image_contract"]
    assert native["raw_extension_path_load_forbidden"] is True
    assert native["same_private_image_handle_rehashed_at_process_exit"] is True
    assert native["failure_to_stage_or_hold_must_fail_closed"] is True


def test_external_encoded_command_is_live_bound_to_child_not_json_asserted() -> None:
    bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    launcher = STAGE1_PATH.read_text(encoding="utf-8")
    stage0 = STAGE0_PATH.read_text(encoding="utf-8")
    assert "$MyInvocation.MyCommand.Definition" in stage0
    assert "P1_R4_STAGE0_ENCODED_SHA256" in stage0
    assert "[ScriptBlock]::Create($stage1Text)" in stage0
    assert "RedirectStandardInput = $true" in launcher
    assert "RedirectStandardOutput = $true" in launcher
    assert "child_challenge" in launcher
    assert "supervisor_challenge_response" in launcher
    assert "nonce + challenge" in bootstrap
    assert "inherited_redirected_duplex_handles_required" in bootstrap


def test_firewall_rejects_subinterpreters_fresh_native_imports_and_os_pipe() -> None:
    bootstrap = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    owner_sources = [CONTRACT_PATH, BROKER_PATH, ENGINE_PATH, VERIFIER_PATH, RUNNER_PATH]
    forbidden_import_lines = (
        "import _winapi",
        "import mmap",
        "import ctypes",
        "from ctypes",
        "import _xxsubinterpreters",
        "import _interpreters",
        "import interpreters",
    )
    for path in owner_sources:
        text = path.read_text(encoding="utf-8")
        assert all(line not in text for line in forbidden_import_lines)
        assert "os.pipe(" not in text
        assert "os.pipe2(" not in text
    for name in (
        '"_winapi"',
        '"mmap"',
        '"ctypes"',
        '"_xxsubinterpreters"',
        '"_interpreters"',
        '"interpreters"',
    ):
        assert name in bootstrap
    assert '"os.pipe"' in bootstrap
    assert '"os.pipe2"' in bootstrap
    assert "fresh dangerous import is forbidden" in bootstrap
    assert "direct write-capable open is forbidden" in bootstrap


def test_full_semantic_verifier_rejects_bool_counters_and_prediction_tamper() -> None:
    contract = _load_contract()
    verifier = _load_verifier(contract)
    exact = dict(contract.EXACT_COMPLETION_COUNTERS)
    assert contract.verify_completion_counters(exact) == exact
    exact["scores"] = True
    with pytest.raises(contract.ContractError, match="non-bool integer"):
        contract.verify_completion_counters(exact)
    payload = verifier.encode_predictions([0.0, 0.25, 1.0])
    assert verifier.verify_prediction_bytes(
        payload, expected_rows=3, label="valid"
    )["rows"] == 3
    with pytest.raises(verifier.VerificationError):
        verifier.verify_prediction_bytes(payload[:-1], expected_rows=3, label="truncated")
    mutated = bytearray(payload)
    mutated[-8:] = b"\x00\x00\x00\x00\x00\x00\xf8\x7f"
    with pytest.raises(verifier.VerificationError):
        verifier.verify_prediction_bytes(bytes(mutated), expected_rows=3, label="nan")


def test_exact_202_path_manifest_sidecar_preseal_graph_is_declared_and_verified() -> None:
    contract = _load_contract()
    paths = contract.expected_output_paths()
    assert len(paths) == 202
    assert len(set(paths)) == 202
    assert len(contract.expected_parent_paths()) == 22
    assert all(len(contract.expected_cell_paths(cell)) == 12 for cell in range(1, 16))
    assert "manifest.json" in paths
    assert "manifest.sha256" in paths
    assert "preseal.json" in paths
    assert "final_seal.json" in paths
    verifier_text = VERIFIER_PATH.read_text(encoding="utf-8")
    for evidence in (
        "row_ids_sha256",
        "verify_disjoint",
        "teacher_receipt_catalog",
        "prediction binary shape differs",
        "preseal inventory",
        "final output tree is not exactly the canonical 202 paths",
    ):
        assert evidence in verifier_text


def test_science_score_teacher_and_v9_invariants_remain_exact() -> None:
    config = json.loads(CONFIG_PATH.read_bytes())
    assert config["frozen_science"]["science_bytes_unchanged"] is True
    assert config["frozen_science"]["gate_thresholds_unchanged"] is True
    assert config["frozen_science"]["nonnegative_residual_guards_unchanged"] is True
    assert config["score_accounting"] == {
        "inner_blocks": 45,
        "inner_gate_aggregates": 15,
        "fraction_aggregates": 5,
        "fold_aggregates": 3,
        "total": 68,
    }
    assert config["teacher_receipt_binding"]["receipt_count"] == 135
    teacher = ROOT / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r6/predictions_complete.json"
    assert teacher.stat().st_size == 313087
    assert _sha(teacher) == "32b8a15d5bcc52523ff2483eff50a591ef7497ea87a0dcabbe7279fd468599b0"
    v9 = ROOT / "artifacts/meaningful_score_goal_v9/registry.jsonl"
    assert v9.stat().st_size == 15812
    assert _sha(v9) == "232b6ed3133de11ee05150ec439efe05baa315bbb64ea0f319ffcbddd421b965"
    events = [json.loads(line) for line in v9.read_bytes().splitlines()]
    assert [event["seq"] for event in events] == [3, 4, 5]
    assert events[-1]["event_sha256"] == (
        "1b3e01be70c6f8ed2df04038deac3b3642804f70f9f17a238826c64d68090317"
    )
    assert all(event["payload"].get("upload_performed") is not True for event in events)


def test_no_gen6r4_qa_authorization_lock_run_output_or_candidate_state_exists() -> None:
    config = json.loads(CONFIG_PATH.read_bytes())
    for relative in config["future_state_must_be_absent"]:
        assert not (ROOT / relative).exists()
    gate = config["execution_gate"]
    assert gate["current_qa_receipt_present"] is False
    assert gate["current_execution_authorization_present"] is False
    assert gate["current_attempt_lock_present"] is False
    assert gate["current_actual_run_performed"] is False
    assert config["exact_completion_counters"]["test_value_reads"] == 0
    assert config["exact_completion_counters"]["candidate_files"] == 0
    assert config["exact_completion_counters"]["ledger_appends"] == 0
    assert config["exact_completion_counters"]["uploads"] == 0


def test_direct_import_guards_remain_fail_closed() -> None:
    for path in (CONTRACT_PATH, BROKER_PATH, ENGINE_PATH, VERIFIER_PATH, RUNNER_PATH):
        namespace: dict[str, Any] = {"__name__": "direct_guard_test"}
        with pytest.raises(RuntimeError):
            exec(compile(path.read_bytes(), str(path), "exec"), namespace)
