from __future__ import annotations

import hashlib
import json
import types
from array import array
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift_contract_v6r3.py"
ENGINE_PATH = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift_execution_v6r3.py"
VERIFIER_PATH = ROOT / "src/p1_qc/multiscale_cross_layer_offset_drift_verifier_v6r3.py"
RUNNER_PATH = ROOT / "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6r3.py"
BOOTSTRAP_PATH = ROOT / "scripts/bootstrap_p1_multiscale_cross_layer_offset_drift_unary_v6r3.py"
LAUNCHER_PATH = ROOT / "scripts/launch_p1_multiscale_cross_layer_offset_drift_unary_v6r3.ps1"
STAGE0_PATH = ROOT / "scripts/p1_multiscale_cross_layer_offset_drift_unary_v6r3_stage0.ps1.txt"
STARTUP_TRUST_PATH = (
    ROOT
    / "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r3_startup_trust.json"
)
NO_GO_PATH = (
    ROOT
    / "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r2_disposition/"
    "OWNER_STATIC_QA_NO_GO_20260823.json"
)
TOMBSTONE_PATH = NO_GO_PATH.with_name("EXECUTION_TOMBSTONE.json")

V6R2_PINS = {
    "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r2.json": (
        21292,
        "5343b6d9a15ac7e0b2728b30f84db5974431b80070ea8519d51d9bfd8ad1dc12",
    ),
    "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r2_science_projection.json": (
        9537,
        "c8bc59c7dc78568fbf79a54a8dfdbfe799242d44cfa31d155b31000d4cafcaef",
    ),
    "scripts/bootstrap_p1_multiscale_cross_layer_offset_drift_unary_v6r2.py": (
        52140,
        "40e5e6ccd8be5d9b854715a8b5f39f60f78e4e01b487196564336668dc1de177",
    ),
    "src/p1_qc/multiscale_cross_layer_offset_drift_contract_v6r2.py": (
        63371,
        "b8867e61a54855d78bd3c85ed0bc88f2cce9d51d63be661b111c4136dd3d2bdb",
    ),
    "src/p1_qc/multiscale_cross_layer_offset_drift_execution_v6r2.py": (
        63055,
        "4d46d30cb1895a952e2925cbf6811609d776d133528dd43ff06f520a276fffc9",
    ),
    "src/p1_qc/multiscale_cross_layer_offset_drift_v6r2.py": (
        70846,
        "4b5c74aeb54406416cda09576fac1d7e1569c6cbc46534ad92a9c2ea154a03af",
    ),
    "src/p1_qc/selective_targets_v5r2.py": (
        17435,
        "0c3b5c4d806605c83b2a024443d718868f8ced37900f0a4250b0887d72805e79",
    ),
    "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6r2.py": (
        2609,
        "2d7e1314f4964230189f70505df026b135cb0f348e4dcaab4462b1e25c646817",
    ),
    "tests/test_run_p1_multiscale_cross_layer_offset_drift_unary_v6r2.py": (
        25232,
        "7e1787677905cfe8cd2ec9710fa6a75f42f6a0497d2edc461f6ef813835822b6",
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Capability:
    def __init__(self, role: str, cell: int | None = None) -> None:
        self.role = role
        self.cell = cell
        self.session_sha256 = "1" * 64
        self.counters: dict[str, int] = {}
        self.live = True


def _load_contract() -> tuple[types.ModuleType, dict[str, Any], list[tuple[object, str, bytes]]]:
    registered: set[int] = set()
    writes: list[tuple[object, str, bytes]] = []
    seen_paths: set[str] = set()
    lock_live = {"value": True}

    def guard(capability: object, _entry: str) -> object:
        if id(capability) not in registered or not getattr(capability, "live", False):
            raise PermissionError("not live")
        return capability

    def snapshot(capability: _Capability) -> dict[str, Any]:
        guard(capability, "snapshot")
        return {
            "role": capability.role,
            "cell": capability.cell,
            "session_sha256": capability.session_sha256,
            "counters": dict(capability.counters),
        }

    def bump(capability: _Capability, name: str, amount: int) -> int:
        guard(capability, "bump")
        capability.counters[name] = capability.counters.get(name, 0) + amount
        return capability.counters[name]

    def close(capability: _Capability) -> dict[str, int]:
        guard(capability, "close")
        capability.live = False
        return dict(capability.counters)

    def writer(capability: _Capability, relative: str, payload: bytes) -> dict[str, Any]:
        guard(capability, "writer")
        if not lock_live["value"]:
            raise PermissionError("lock not live")
        if relative in seen_paths:
            raise FileExistsError(relative)
        seen_paths.add(relative)
        writes.append((capability, relative, payload))
        return {
            "path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    context: dict[str, Any] = {
        "all_owner_roles_authenticated": True,
        "capability_guard": guard,
        "capability_snapshot": snapshot,
        "capability_counter_bump": bump,
        "capability_close": close,
        "exclusive_output_writer": writer,
        "_test_register": registered.add,
        "_test_lock": lock_live,
    }
    module = types.ModuleType("p1_v6r3_contract_test")
    module.__dict__["_P1_V6R3_BOOTSTRAP_CONTEXT"] = context
    exec(compile(CONTRACT_PATH.read_bytes(), str(CONTRACT_PATH), "exec"), module.__dict__)
    return module, context, writes


def _load_verifier(contract: types.ModuleType) -> types.ModuleType:
    context = {
        "all_owner_roles_authenticated": True,
    }
    module = types.ModuleType("p1_v6r3_verifier_test")
    module.__dict__.update(
        {
            "_P1_V6R3_BOOTSTRAP_CONTEXT": context,
            "_P1_V6R3_AUTH_CONTRACT": contract,
        }
    )
    exec(compile(VERIFIER_PATH.read_bytes(), str(VERIFIER_PATH), "exec"), module.__dict__)
    return module


def _load_engine_for_vault_tests(
    contract: types.ModuleType, committed: Any
) -> tuple[types.ModuleType, dict[str, Any]]:
    columns = (
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "label",
        "anomaly_type",
    )

    def spans(raw: bytes, expected: int) -> tuple[bytes, tuple[tuple[int, int], ...]]:
        line = raw.rstrip(b"\r\n")
        starts = [0]
        for index, value in enumerate(line):
            if value == 44:
                starts.append(index + 1)
        result = []
        for index, start in enumerate(starts):
            end = starts[index + 1] - 1 if index + 1 < len(starts) else len(line)
            result.append((start, end))
        assert len(result) == expected
        return line, tuple(result)

    legacy = types.SimpleNamespace(
        TRAIN_COLUMNS=columns,
        TARGET_COLUMNS=("label", "anomaly_type"),
        _csv_field_spans=spans,
        _decode_field=lambda line, span: line[span[0] : span[1]].decode("utf-8"),
    )
    context = {
        "mode": "cell_worker",
        "all_owner_roles_authenticated": True,
        "bootstrap_documents_prevalidated": True,
        "verify_numerical_runtime": lambda: {},
        "is_fold_committed": committed,
    }
    module = types.ModuleType("p1_v6r3_engine_vault_test")
    module.__dict__.update(
        {
            "_P1_V6R3_BOOTSTRAP_CONTEXT": context,
            "_P1_V6R3_AUTH_CONTRACT": contract,
            "_P1_V6R3_AUTH_SCIENCE": types.SimpleNamespace(),
            "_P1_V6R3_AUTH_LEGACY_ENGINE": legacy,
            "_P1_V6R3_AUTH_VERIFIER": types.SimpleNamespace(),
        }
    )
    exec(compile(ENGINE_PATH.read_bytes(), str(ENGINE_PATH), "exec"), module.__dict__)
    return module, context


def test_frozen_v6r2_bytes_remain_exact() -> None:
    for relative, (expected_bytes, expected_sha) in V6R2_PINS.items():
        path = ROOT / relative
        assert path.stat().st_size == expected_bytes
        assert _sha(path) == expected_sha


def test_owner_no_go_and_tombstone_bind_exact_review() -> None:
    owner = json.loads(NO_GO_PATH.read_bytes())
    tombstone = json.loads(TOMBSTONE_PATH.read_bytes())
    assert owner["reviewer"] == "/root/p1_gen6r2_independent_qa"
    assert owner["verdict"] == "P0=2_P1=3_NO_GO"
    assert owner["independent_qa_receipt"]["present"] is False
    assert [item["code"] for item in owner["findings"]] == [
        "CROSS_FRACTION_TARGET_VAULT_CONTAMINATION",
        "AUTHENTICATED_NATIVE_BYTES_NOT_BOUND_TO_LOADED_BYTES",
        "SCORE_CALL_ACCOUNTING_53_VS_ACTUAL_68",
        "DIRECT_WINAPI_AND_PYC_MUTATION_FIREWALL_BYPASS",
        "GEN5R6_TEACHER_RECEIPT_PREDICTION_IDS_NOT_ENFORCED",
    ]
    assert tombstone["owner_no_go_receipt"] == {
        "path": NO_GO_PATH.relative_to(ROOT).as_posix(),
        "bytes": NO_GO_PATH.stat().st_size,
        "sha256": _sha(NO_GO_PATH),
    }
    assert tombstone["status"] == "PERMANENTLY_TOMBSTONED_NEVER_EXECUTE"


def test_contract_has_no_reachable_mint_and_rejects_direct_objects() -> None:
    contract, _context, _writes = _load_contract()
    assert "mint_capability" not in contract.__dict__
    assert "capability_mint" not in contract._CONTEXT
    with pytest.raises(PermissionError):
        contract.require_engine_capability(object(), "direct_call")


def test_worker_capability_is_cell_scoped_replay_safe_and_lock_bound() -> None:
    contract, context, writes = _load_contract()
    worker = _Capability("cell_worker", 1)
    context["_test_register"](id(worker))
    pin = contract.write_output_exclusive(
        worker, "cells/cell_01/models/inner_1.json", {"ok": True}
    )
    assert pin["path"] == "cells/cell_01/models/inner_1.json"
    assert len(writes) == 1
    with pytest.raises(PermissionError):
        contract.write_output_exclusive(worker, "cells/cell_02/models/inner_1.json", b"x")
    with pytest.raises(FileExistsError):
        contract.write_output_exclusive(worker, "cells/cell_01/models/inner_1.json", b"x")
    context["_test_lock"]["value"] = False
    with pytest.raises(PermissionError):
        contract.write_output_exclusive(worker, "cells/cell_01/models/inner_2.json", b"x")


def test_parent_cannot_write_worker_or_extra_paths() -> None:
    contract, context, _writes = _load_contract()
    parent = _Capability("parent")
    context["_test_register"](id(parent))
    contract.write_output_exclusive(parent, "commitments/session.json", {"ok": True})
    contract.write_output_exclusive(parent, "final_seal.json", {"ok": True})
    with pytest.raises(PermissionError):
        contract.write_output_exclusive(parent, "cells/cell_01/models/outer.json", b"x")
    with pytest.raises(PermissionError):
        contract.write_output_exclusive(parent, "unexpected.json", b"x")


def test_score_decomposition_is_exactly_68() -> None:
    contract, _context, _writes = _load_contract()
    assert contract.SCORE_DECOMPOSITION == {
        "inner_blocks": 45,
        "inner_gate_aggregates": 15,
        "fraction_aggregates": 5,
        "fold_aggregates": 3,
        "total": 68,
    }
    assert contract.exact_completion_counters()["scores"] == 68


def test_all_fifteen_cells_have_exact_fold_fraction_identity() -> None:
    contract, _context, _writes = _load_contract()
    identities = [contract.cell_identity(cell) for cell in range(1, 16)]
    assert len(set(identities)) == 15
    assert identities[0] == ("2025_q2", 0.4, "p040")
    assert identities[-1] == ("2025_q4", 1.0, "p100")
    with pytest.raises(contract.ContractError):
        contract.cell_identity(True)


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    contract, _context, _writes = _load_contract()
    with pytest.raises(contract.ContractError):
        contract.parse_json_bytes(b'{"a":1,"a":2}', label="duplicate")
    with pytest.raises(contract.ContractError):
        contract.parse_json_bytes(b'{"a":NaN}', label="nonfinite")


def test_all_135_teacher_receipts_are_authenticated_and_unique() -> None:
    contract, _context, _writes = _load_contract()
    complete = contract.parse_json_bytes(
        (
            ROOT
            / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r6/"
            "predictions_complete.json"
        ).read_bytes(),
        label="predictions_complete",
    )
    manifest = contract.parse_json_bytes(
        (
            ROOT
            / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r6/manifest.json"
        ).read_bytes(),
        label="manifest",
    )
    catalog = contract.verify_teacher_receipt_catalog(complete, manifest)
    assert len(catalog) == 135
    first = catalog[("2025_q2", "p040", 1, 20260813)]
    assert (
        contract.verify_teacher_request(
            catalog,
            fold="2025_q2",
            fraction_tag="p040",
            block=1,
            seed=20260813,
            prediction_ids_sha256=first["prediction_ids_sha256"],
            train_ids_sha256=first["train_ids_sha256"],
            prediction_rows=first["prediction_rows"],
            train_rows=first["train_rows"],
        )
        is first
    )
    with pytest.raises(contract.ContractError):
        contract.verify_teacher_request(
            catalog,
            fold="2025_q2",
            fraction_tag="p040",
            block=1,
            seed=20260813,
            prediction_ids_sha256="0" * 64,
            train_ids_sha256=first["train_ids_sha256"],
            prediction_rows=first["prediction_rows"],
            train_rows=first["train_rows"],
        )


def test_predictions_complete_teacher_manifest_pin_is_exact() -> None:
    path = (
        ROOT
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r6/"
        "predictions_complete.json"
    )
    assert path.stat().st_size == 313087
    assert _sha(path) == "32b8a15d5bcc52523ff2483eff50a591ef7497ea87a0dcabbe7279fd468599b0"


def test_expected_output_tree_is_exactly_202_and_has_no_wildcards() -> None:
    contract, _context, _writes = _load_contract()
    verifier = _load_verifier(contract)
    paths = verifier.expected_output_paths()
    assert len(paths) == 202
    assert all("*" not in path and ".." not in path and "\\" not in path for path in paths)
    assert len(verifier.expected_cell_paths(1)) == 12


def test_engine_text_uses_fresh_sequential_workers_and_computed_decode_proofs() -> None:
    text = ENGINE_PATH.read_text(encoding="utf-8")
    assert "for cell in range(1, contract.CELL_COUNT + 1):" in text
    assert "vault = CellTargetVault(capability, train_raw, len(frame), cell)" in text
    assert "already_decoded = int(sum(" in text
    assert "current inner holdout was decoded before commitment" in text
    assert "active outer validation targets were decoded before commitment" in text
    assert "contract.bump_counter(capability, \"scores\")  # the 15 aggregate calls" in text
    assert "verifier.verify_cell_artifact_graph(" in text
    assert text.index("verifier.verify_cell_artifact_graph(") < text.index("loaded = _load_outer_result(")


def test_direct_cell_release_has_no_id_argument_and_commit_phase_denies_decode() -> None:
    np = pytest.importorskip("numpy")
    contract, context, _writes = _load_contract()
    capability = _Capability("cell_worker", 1)
    context["_test_register"](id(capability))
    engine, _engine_context = _load_engine_for_vault_tests(
        contract, lambda _fold, _sha: False
    )
    raw = (
        b"station,year,layer,time,temp,psal,depth,label,anomaly_type\n"
        b"A,2024,1,t0,1,2,3,0,normal\n"
        b"A,2024,1,t1,1,2,3,1,offset\n"
        b"A,2024,1,t2,1,2,3,0,normal\n"
    )
    lines = raw.splitlines(keepends=True)
    vault = object.__new__(engine.CellTargetVault)
    vault._capability = capability
    vault._raw = raw
    vault._cell = 1
    vault._expected_rows = 3
    vault._offsets = array("Q", [len(lines[0]), len(lines[0]) + len(lines[1]), len(lines[0]) + len(lines[1]) + len(lines[2])])
    vault._lengths = array("I", [len(line) for line in lines[1:]])
    vault._columns = {name: index for index, name in enumerate(engine.TRAIN_COLUMNS)}
    vault._labels = {}
    vault._anomalies = {}
    vault._events = []
    vault._blocks = {1: (np.asarray([0], dtype=np.int64), np.asarray([1], dtype=np.int64))}
    vault._outer_train = np.asarray([0, 1], dtype=np.int64)
    vault._outer_validation = np.asarray([2], dtype=np.int64)
    vault._stage = "inner_1_train"
    labels, _anomaly = vault._release()
    assert labels.tolist() == [0]
    assert vault._stage == "inner_1_commit"
    with pytest.raises(TypeError):
        vault._release(np.asarray([2], dtype=np.int64))
    with pytest.raises(PermissionError, match="no target decode capability"):
        vault._release()


def test_direct_parent_release_rejects_arbitrary_ids_and_uncommitted_later_fold() -> None:
    np = pytest.importorskip("numpy")
    contract, context, _writes = _load_contract()
    capability = _Capability("parent")
    context["_test_register"](id(capability))
    committed_folds = {"2025_q2"}
    engine, _engine_context = _load_engine_for_vault_tests(
        contract, lambda fold, _sha: fold in committed_folds
    )
    raw = (
        b"station,year,layer,time,temp,psal,depth,label,anomaly_type\n"
        b"A,2024,1,t0,1,2,3,0,normal\n"
        b"A,2024,1,t1,1,2,3,1,offset\n"
        b"A,2024,1,t2,1,2,3,0,normal\n"
    )
    lines = raw.splitlines(keepends=True)
    vault = object.__new__(engine.ParentOuterTargetVault)
    vault._capability = capability
    vault._raw = raw
    vault._expected_rows = 3
    vault._validation = {
        "2025_q2": np.asarray([0], dtype=np.int64),
        "2025_q3": np.asarray([1], dtype=np.int64),
        "2025_q4": np.asarray([2], dtype=np.int64),
    }
    vault._released = set()
    vault._events = []
    vault._offsets = array("Q", [len(lines[0]), len(lines[0]) + len(lines[1]), len(lines[0]) + len(lines[1]) + len(lines[2])])
    vault._lengths = array("I", [len(line) for line in lines[1:]])
    vault._columns = {name: index for index, name in enumerate(engine.TRAIN_COLUMNS)}
    with pytest.raises(PermissionError, match="preceded persisted fold commitment"):
        vault.release_fold("2025_q3", np.asarray([1], dtype=np.int64), "2" * 64)
    with pytest.raises(PermissionError, match="differ from the frozen fold validation"):
        vault.release_fold("2025_q2", np.asarray([1], dtype=np.int64), "2" * 64)


def test_runner_and_engine_are_direct_import_guarded() -> None:
    for path in (ENGINE_PATH, VERIFIER_PATH, RUNNER_PATH):
        namespace: dict[str, Any] = {"__name__": "direct_guard_test"}
        with pytest.raises(RuntimeError, match="authenticated bootstrap"):
            exec(compile(path.read_bytes(), str(path), "exec"), namespace)


def test_external_startup_trust_declares_pre_script_boundary_exactly() -> None:
    trust = json.loads(STARTUP_TRUST_PATH.read_bytes())
    assert trust["canonical_python"]["version"] == "3.12.10"
    assert trust["canonical_python"]["flags"][:5] == [
        "-I",
        "-S",
        "-B",
        "-X",
        "pycache_prefix=<workspace>/configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6r3_startup_trust.json",
    ]
    assert trust["expected_pre_script_path_backed_modules"] == [
        "encodings",
        "encodings.aliases",
        "encodings.cp949",
        "encodings.utf_8",
    ]
    assert trust["expected_pre_script_builtin_modules"] == ["_codecs_kr", "_multibytecodec"]
    requirements = trust["external_qa_requirements"]
    assert requirements["launcher_must_retain_all_startup_handles_until_child_exit"] is True
    assert requirements["launcher_must_post_child_rehash_every_startup_file_through_the_same_handle"] is True
    assert requirements["launcher_must_hold_regular_file_pycache_sentinel_through_child"] is True
    stage0 = trust["inline_powershell_stage0"]
    assert stage0["reference_source_is_never_invoked_by_path"] is True
    assert stage0["authenticates_stage1_bytes_before_compilation"] is True
    assert stage0["executes_only_authenticated_stage1_text_in_memory"] is True
    inventory = trust["external_powershell_distribution_inventory"]
    assert inventory == {
        "root_source": "directory containing the externally injected absolute P1_POWERSHELL_HOST",
        "files": 983,
        "bytes": 296034085,
        "inventory_sha256": "fcbbc18499e682ca08a0860dcb3b5353099a2a846e9eedc50afbb0c28ed728dc",
        "canonical_row": "ordinal-sorted <relative-forward-slash-path> NUL <bytes> NUL <sha256> LF",
        "fresh_external_qa_must_authenticate_inventory_before_powershell_start": True,
        "launcher_post_start_inventory_check_is_not_retroactive": True,
        "launcher_must_hold_every_inventory_file_share_deny_until_child_exit": True,
        "launcher_must_same_handle_rehash_every_inventory_file_after_child": True,
    }


def test_external_launcher_holds_exact_startup_chain_and_rehashes_after_child() -> None:
    text = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "C:\\Users\\cedis" not in text
    assert "P1_POWERSHELL_HOST" in text
    assert "[Environment]::ProcessPath" in text
    assert "CreateFileW" in text
    assert "$fileShareRead" in text
    assert "$openReparsePoint" in text
    assert "NumberOfLinks -ne 1" in text
    assert "Open-HeldPowerShellDistribution" in text
    assert "POWERSHELL_DISTRIBUTION::$relative" in text
    assert "-ne 983" in text
    assert "296034085" in text
    assert "fcbbc18499e682ca08a0860dcb3b5353099a2a846e9eedc50afbb0c28ed728dc" in text
    for label in (
        "EXTERNAL_LAUNCHER",
        "POWERSHELL_HOST",
        "VENV_STUB",
        "PYVENV_CFG",
        "STARTUP_TRUST",
        "BOOTSTRAP",
        "BASE_PYTHON",
        "PYTHON312_DLL",
        "PYTHON3_DLL",
        "VCRUNTIME140_DLL",
        "ENCODINGS_INIT",
        "ENCODINGS_ALIASES",
        "ENCODINGS_UTF8",
        "ENCODINGS_CP949",
    ):
        assert f"'{label}'" in text
    assert "Assert-HeldFilesUnchanged" in text
    assert text.index("& $stub.Path -I -S -B -X") < text.rindex("Assert-HeldFilesUnchanged")
    assert "$cachePrefix = Get-NormalizedPath $startupTrust.Path" in text
    assert "P1_STAGE0_LAUNCHER_STREAM" in text
    assert "$PSCommandPath" not in text
    assert "pwsh.exe" not in text.casefold()
    assert "Start-Process" not in text


def test_inline_stage0_closes_launcher_hash_to_execution_swap() -> None:
    text = STAGE0_PATH.read_text(encoding="utf-8")
    assert "C:\\Users\\cedis" not in text
    assert "P1_LAUNCH_MODE" in text
    assert "CreateFileW" in text
    assert "$fileShareRead" in text
    assert "$openReparsePoint" in text
    assert "NumberOfLinks -ne 1" in text
    assert "Get-StreamSha256" in text
    assert "P1_STAGE0_LAUNCHER_STREAM" in text
    assert "[ScriptBlock]::Create($launcherText)" in text
    assert "-File" not in text
    assert "Start-Process" not in text


def test_bootstrap_closes_bytecode_and_direct_native_mutation_bypasses() -> None:
    text = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    assert "sys.dont_write_bytecode = True" in text
    assert 'target.casefold().endswith((".pyc", ".pyo"))' in text
    assert "bytecode read/write is forbidden" in text
    assert "ignored_unbound_bytecode_declarations" in text
    assert "class _AuthenticatedSourceLoader" in text
    assert "source is outside authenticated buffers" in text
    assert "class _AuthenticatedExtensionLoader" in text
    assert 'record is None or record["fd"] is None' in text
    assert '"direct_main_authority_absent": True' in text
    assert "dangerous bootstrap authority remains in check-only __main__" in text
    assert "def _verify_external_tcb_qa(" in text
    assert 'authorization.get("external_powershell_tcb") != qa["external_powershell_tcb"]' in text
    assert 'source_text.encode("utf-16-le", errors="strict")' in text
    assert "base64.b64encode(utf16le)" in text
    assert "_RAW_CREATE_FILE =" not in text
    assert "_RAW_CREATE_PROCESS =" not in text
    assert "_CAPABILITY =" not in text
    assert "_CAPABILITY_CALLBACKS =" not in text
    for assignment in (
        "_winapi.CreateFile = _deny_winapi",
        "_winapi.CreateProcess = _guarded_create_process",
        "_winapi.WriteFile = _deny_winapi",
        "os.truncate = _deny_mutation",
        "os.ftruncate = _deny_mutation",
        "os.chmod = _deny_mutation",
        "os.utime = _deny_mutation",
    ):
        assert assignment in text


def test_no_canonical_gen6r3_execution_state_exists() -> None:
    for relative in (
        "artifacts/p1_v6r3_forbidden_pycache",
        "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r3_control",
        "artifacts/p1_multiscale_cross_layer_offset_drift_unary_v6r3",
        "artifacts/status/p1_multiscale_cross_layer_offset_drift_unary_v6r3.json",
        "submissions/p1_multiscale_cross_layer_offset_drift_unary_v6r3.csv",
    ):
        assert not (ROOT / relative).exists()


def test_v9_anchor_remains_exact_and_upload_free() -> None:
    path = ROOT / "artifacts/meaningful_score_goal_v9/registry.jsonl"
    assert path.stat().st_size == 15812
    assert _sha(path) == "232b6ed3133de11ee05150ec439efe05baa315bbb64ea0f319ffcbddd421b965"
    events = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert [event["seq"] for event in events] == [3, 4, 5]
    assert events[-1]["event_sha256"] == "1b3e01be70c6f8ed2df04038deac3b3642804f70f9f17a238826c64d68090317"
    assert all(event["payload"].get("upload_performed") is not True for event in events)
