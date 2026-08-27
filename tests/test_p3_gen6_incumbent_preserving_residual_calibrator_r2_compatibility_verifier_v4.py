from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / (
    "scripts/launch_verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v4.ps1"
)
STAGE0 = ROOT / (
    "scripts/stage0_verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v4.ps1"
)
PYCACHE_SENTINEL = ROOT / "scripts/p3_gen6_r2_compatibility_v4_pycache_block.sentinel"
BOOTSTRAP = ROOT / (
    "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_v4.py"
)
CONFIG = ROOT / (
    "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_"
    "v1r2_compatibility_verifier_v4.json"
)
HELPER = ROOT / (
    "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v4.py"
)
CLI = ROOT / (
    "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v4.py"
)
V3_NO_GO = ROOT / (
    "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
    "v1r2_compatibility_verifier_v3_no_go/OWNER_STATIC_QA_NO_GO_20260823.json"
)
V3_TOMBSTONE = V3_NO_GO.with_name("EXECUTION_TOMBSTONE.json")
V3 = {
    "BOOTSTRAP": (
        ROOT
        / "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v3.py",
        83798,
        "6efb2024319d7743cd8cee54f8f1ac621275081e14bef2e41877c5db13fb22ed",
    ),
    "CONFIG": (
        ROOT
        / "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_v1r2_compatibility_verifier_v3.json",
        11701,
        "5f69a379a03dce661fd55464628aa5adedda9a9bdfa17bab1574501197ed4084",
    ),
    "HELPER": (
        ROOT
        / "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v3.py",
        13069,
        "4943d4db0310a68949d87f112ccca185c9e48920241b2fac93111dc851a1a39e",
    ),
    "CLI": (
        ROOT
        / "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v3.py",
        1007,
        "2863c7b4e5fe5b16bb2df3b753ba8c935f5a8767fafb135782171dcc4d7349bb",
    ),
    "TESTS": (
        ROOT
        / "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v3.py",
        13804,
        "b337173af7b97ea4088bc41717cde2c8192378c72469aa0d4f2c4542086a032d",
    ),
}
FINDINGS = [
    "PRE_SCRIPT_ENCODING_BYTECODE_TRUST_NOT_CLOSED",
    "STDLIB_NATIVE_AUTH_TO_LOAD_TOCTOU",
    "SEMANTIC_BYTES_NOT_ALL_PARSED_FROM_HELD_BUFFERS",
    "HARDLINK_CONTAINMENT_NOT_ENFORCED",
    "DIRECT_WINAPI_AND_NETWORK_FIREWALL_INCOMPLETE",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_bytes())


def _canonical_environment() -> dict[str, str]:
    environment = os.environ.copy()
    powershell = shutil.which("pwsh")
    assert powershell is not None
    environment.update(
        {
            "P3_WORKSPACE_ROOT": str(ROOT),
            "P3_DATA_DIR": str(Path.home() / "Downloads/p3/데이터셋_P3/P3_wave_forecast"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "P3_POWERSHELL_HOST": str(Path(powershell).resolve()),
            "P3_V4_STAGE0_ENCODED_COMMAND": base64.b64encode(
                STAGE0.read_text(encoding="utf-8").encode("utf-16le")
            ).decode("ascii"),
        }
    )
    return environment


def _probe(expression: str, *, before: str = "") -> subprocess.CompletedProcess[str]:
    assert PYCACHE_SENTINEL.is_file()
    script = (
        "import sys;"
        + before
        + (";" if before else "")
        + "p=sys.argv[1];raw=open(p,'rb').read();"
        + "ns={'__name__':'p3_v4_probe','__file__':p};"
        + "exec(compile(raw,p,'exec'),ns);"
        + expression
    )
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={PYCACHE_SENTINEL}",
            "-c",
            script,
            str(BOOTSTRAP),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_v3_bytes_are_preserved_exactly() -> None:
    for path, size, digest in V3.values():
        assert path.stat().st_size == size
        assert _sha(path) == digest


def test_v3_independent_no_go_is_exact_and_has_no_receipt() -> None:
    owner = json.loads(V3_NO_GO.read_bytes())
    tombstone = json.loads(V3_TOMBSTONE.read_bytes())
    review = owner["review"]
    assert review["reviewer"] == "/root/p3_gen6_compat_v3_qa"
    assert review["verdict"] == "NO-GO"
    assert review["p0_count"] == 0
    assert review["p1_count"] == 5
    assert review["qa_receipt_present"] is False
    assert review["independent_qa_receipt_sha256"] is None
    assert [item["id"] for item in review["findings"]] == FINDINGS
    assert tombstone["review"]["finding_ids"] == FINDINGS
    assert tombstone["status"].startswith("PERMANENTLY_TOMBSTONED")


def test_v4_subordinate_pins_are_exact() -> None:
    config = _config()
    roles = config["implementation_roles"]
    pins = config["authenticated_subordinate_pins"]
    for role in ("HELPER", "CLI", "TESTS"):
        path = ROOT / roles[role]
        raw = path.read_bytes()
        assert pins[role] == {
            "path": roles[role],
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }


def test_external_launcher_is_noncyclic_and_canonical() -> None:
    contract = _config()["trusted_launcher"]
    assert contract["path"] == LAUNCHER.relative_to(ROOT).as_posix()
    assert contract["stage0_source_path"] == STAGE0.relative_to(ROOT).as_posix()
    assert contract["self_hash_embedded"] is False
    assert contract["external_exact_pin_required"] is True
    assert contract["holds_authenticated_runtime_files_through_child_exit"] is True
    assert (
        "$env:P3_POWERSHELL_HOST -NoLogo -NoProfile -NonInteractive" in contract["canonical_launch"]
    )
    assert "-EncodedCommand $env:P3_V4_STAGE0_ENCODED_COMMAND" in contract["canonical_launch"]
    assert " -File " not in contract["canonical_launch"]
    assert contract["direct_file_execution_of_stage0_or_stage1_forbidden"] is True
    assert contract["external_host_pin"]["path_source"] == "P3_POWERSHELL_HOST"
    inventory = contract["external_host_distribution_inventory"]
    assert inventory == {
        "root": "DIRECTORY_OF_P3_POWERSHELL_HOST",
        "algorithm": (
            "SHA256_SORTED_DIRECTORIES_D_NUL_RELATIVE_LF_THEN_FILES_"
            "F_NUL_RELATIVE_NUL_BYTES_NUL_FILE_SHA256_LF"
        ),
        "directories": 53,
        "files": 983,
        "file_bytes": 296034085,
        "payload_bytes": 112750,
        "sha256": "eef4626964532f664559724e8ce95b2a95b6cb4729d275ac6cd0da81e0115444",
        "external_authority_prelaunch_verification_required": True,
        "external_authority_holds_every_file_share_read_only_through_target_exit_required": True,
        "launcher_holds_every_file_share_read_only_through_child_exit": True,
        "launcher_final_name_identity_and_same_handle_hash_verification_required": True,
    }


def test_encoded_stage0_authenticates_and_holds_stage1_before_memory_execution() -> None:
    source = STAGE0.read_text(encoding="utf-8")
    assert "CreateFileW" in source
    assert "$OpenReparsePoint = [uint32]0x00200000" in source
    assert "$FileShareRead = [uint32]1" in source
    assert "NumberOfLinks -ne 1" in source
    assert "ScriptBlock]::Create($launcherText)" in source
    assert "Get-Stage0StreamSha $held.Stream" in source
    assert "-AuthenticatedLauncherSha256 $launcher.Sha256" in source


def test_launcher_holds_startup_and_implementation_pins_through_child_exit() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "FILE_SHARE_READ" not in source
    assert "$FileShareRead = [uint32]0x00000001" in source
    assert "$OpenReparsePoint = [uint32]0x00200000" in source
    assert "NumberOfLinks -ne 1" in source
    assert "Assert-HeldFilesUnchanged" in source
    assert "Open-HeldPowerShellDistribution" in source
    assert "Assert-PowerShellDistributionNamesUnchanged" in source
    assert '-X "pycache_prefix=$cacheSentinel"' in source
    for role in ("V4_BOOTSTRAP", "V4_CONFIG", "V4_HELPER", "V4_CLI", "V4_TESTS"):
        assert f"@('{role}'" in source
    assert "@('PYCACHE_BLOCK_SENTINEL'" in source


def test_startup_encoding_contract_uses_held_regular_file_sentinel() -> None:
    runtime = _config()["canonical_runtime_contract"]
    assert runtime["required_cli_flags"] == [
        "-I",
        "-S",
        "-B",
        "-X",
        "pycache_prefix=<pinned_held_regular_file_sentinel>",
    ]
    startup = runtime["external_startup_trust"]
    assert (
        startup["pycache_regular_file_sentinel_relative"]
        == PYCACHE_SENTINEL.relative_to(ROOT).as_posix()
    )
    assert (
        startup[
            "pycache_regular_file_sentinel_must_be_pinned_nlink1_non_reparse_and_held_share_deny"
        ]
        is True
    )
    assert startup["legacy_pycache_not_consulted"] is True
    assert startup["startup_codec_primitives_builtin"] == [
        "_codecs_kr",
        "_multibytecodec",
    ]


def test_regular_file_pycache_sentinel_forces_startup_encodings_source(tmp_path: Path) -> None:
    cache = tmp_path / "pinned-sentinel"
    cache.write_bytes(b"test-only-regular-file-sentinel\n")
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={cache}",
            "-v",
            "-c",
            "pass",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    for name in ("__init__.py", "aliases.py", "utf_8.py", "cp949.py"):
        assert f"encodings\\{name}" in completed.stderr
    assert "encodings\\__pycache__" not in completed.stderr
    assert cache.read_bytes() == b"test-only-regular-file-sentinel\n"


def test_bootstrap_claims_only_script_initiated_hook_boundary() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "pre_script_imports_trusted_by_external_launcher" in source
    assert "audit_hook_installed_before_script_initiated_non_sys_import" in source
    assert "before every import except" not in source
    tree = ast.parse(source)
    hook_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sys"
        and node.func.attr == "addaudithook"
    ]
    assert len(hook_lines) == 1
    imports_before = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and node.lineno < hook_lines[0]
        for alias in node.names
    ]
    assert imports_before == ["sys"]


def test_native_authentication_is_bound_to_held_bytes_and_final_rehash() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "def hold_native(self, path, expected_size, expected_digest)" in source
    assert "held native bytes differ from authenticated inventory" in source
    assert "same-handle final native rehash changed" in source
    assert '"native_same_handle_final_rehashes"' in source


def test_adversarial_wrong_native_digest_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "native.pyd"
    target.write_bytes(b"trusted-native")
    expression = (
        "b=ns['_WINAPI_BROKER_FACTORY']();"
        "r=ns['_HeldRegistry'](os.path.dirname(t),os.path.dirname(t),b);"
        "r.hold_native(t,len(b'trusted-native'),'0'*64)"
    )
    before = f"import os;t={str(target)!r}"
    completed = _probe(expression, before=before)
    assert completed.returncode != 0
    assert "held native bytes differ from authenticated inventory" in completed.stderr


def test_hardlink_is_rejected_by_runtime_containment(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    alias = tmp_path / "alias.bin"
    target.write_bytes(b"x")
    os.link(target, alias)
    completed = _probe("ns['_plain_absolute'](t,'file')", before=f"t={str(target)!r}")
    assert completed.returncode != 0
    assert "single-link regular file required" in completed.stderr


def test_symlink_or_reparse_is_rejected_by_runtime_containment(tmp_path: Path) -> None:
    target = tmp_path / "target"
    alias = tmp_path / "alias"
    target.mkdir()
    (target / "value.bin").write_bytes(b"x")
    created = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    try:
        completed = _probe(
            "ns['_plain_absolute'](t,'file')",
            before=f"t={str(alias / 'value.bin')!r}",
        )
        assert completed.returncode != 0
        assert "link/reparse path is forbidden" in completed.stderr
    finally:
        alias.rmdir()


@pytest.mark.parametrize(
    "name",
    [
        "WriteFile",
        "CopyFile2",
        "CreateFileMapping",
        "OpenFileMapping",
        "MapViewOfFile",
        "CreateJunction",
        "CreateProcess",
        "OpenProcess",
        "TerminateProcess",
        "CreateNamedPipe",
        "CreatePipe",
    ],
)
def test_direct_winapi_escape_functions_are_replaced(name: str) -> None:
    completed = _probe(f"getattr(__import__('_winapi'),{name!r})()")
    assert completed.returncode != 0
    assert f"direct _winapi.{name} is forbidden" in completed.stderr


def test_direct_createfile_read_and_write_are_denied() -> None:
    read_probe = _probe("__import__('_winapi').CreateFile('x',0x80000000,1,0,3,0,0)")
    assert read_probe.returncode != 0
    assert "direct _winapi.CreateFile is forbidden" in read_probe.stderr
    write_probe = _probe("sys.audit('_winapi.CreateFile','x',0x00120116,1,3,0)")
    assert write_probe.returncode != 0
    assert "Windows write/delete handle is forbidden" in write_probe.stderr


@pytest.mark.parametrize(
    ("expression", "name"),
    [
        ("__import__('_winapi').CreateFileMapping(-1,None,4,0,4096,None)", "CreateFileMapping"),
        ("__import__('_winapi').WriteFile(-1,b'x',False)", "WriteFile"),
        ("__import__('_winapi').OpenProcess(0x1fffff,False,0)", "OpenProcess"),
    ],
)
def test_real_direct_winapi_capability_calls_are_removed(expression: str, name: str) -> None:
    completed = _probe(expression)
    assert completed.returncode != 0
    assert f"direct _winapi.{name} is forbidden" in completed.stderr


def test_main_namespace_does_not_retain_original_or_broker_winapi_capability() -> None:
    config = _config()["canonical_runtime_contract"]["comprehensive_firewall"]
    assert config["all_callable_winapi_exports_replaced_after_single_closure_capture"] is True
    assert config["raw_original_or_broker_callable_reachable_from_main_namespace"] is False
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'globals().pop("_WINAPI_BROKER_FACTORY", None)' in source
    assert "_assert_main_namespace_has_no_winapi_capability()" in source
    completed = _probe(
        "assert all(getattr(getattr(ns['_winapi'],name),'__name__','').startswith('blocked_') "
        "for name in dir(ns['_winapi']) if not name.startswith('__') "
        "and callable(getattr(ns['_winapi'],name)))"
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "event", ["socket.sendto", "socket.connect", "socket.bind", "socket.gethostname"]
)
def test_all_network_audit_surfaces_fail_closed(event: str) -> None:
    completed = _probe(f"sys.audit({event!r},None,None)")
    assert completed.returncode != 0
    assert "network access is forbidden" in completed.stderr


@pytest.mark.parametrize("event", ["winreg.OpenKey", "winreg.SetValue"])
def test_registry_access_outside_authenticated_dependency_scope_is_denied(event: str) -> None:
    completed = _probe(f"sys.audit({event!r},None,None)")
    assert completed.returncode != 0
    assert "Windows registry access is forbidden" in completed.stderr


@pytest.mark.parametrize("module_name", ["ctypes", "mmap"])
def test_native_escape_import_is_denied(module_name: str) -> None:
    completed = _probe(f"exec('import {module_name}',ns)")
    assert completed.returncode != 0
    assert "forbidden or preauthentication numerical import" in completed.stderr


@pytest.mark.parametrize("event", ["ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function"])
def test_ctypes_native_events_are_denied_after_authenticated_import_scope(event: str) -> None:
    completed = _probe(
        f"ns['_AUDIT'].__setitem__('phase','DEPENDENCIES_AUTHENTICATED');sys.audit({event!r},None)"
    )
    assert completed.returncode != 0
    assert "ctypes native loading or invocation is forbidden" in completed.stderr


def test_ctypes_import_exception_is_narrow_and_authenticated() -> None:
    firewall = _config()["canonical_runtime_contract"]["comprehensive_firewall"]
    assert firewall["temporary_local_hostname_query_scope"] == ("DEPENDENCIES_AUTHENTICATED_ONLY")
    assert firewall["temporary_read_only_timezone_registry_scope"] == (
        "DEPENDENCIES_AUTHENTICATED_ONLY"
    )
    assert firewall["registry_mutation_always_forbidden"] is True
    assert (
        firewall["network_connect_bind_listen_send_receive_and_socket_creation_forbidden"] is True
    )
    assert set(firewall["temporary_authenticated_native_support_imports"]) == {
        "ctypes",
        "_ctypes",
        "mmap",
    }
    assert firewall["temporary_native_support_scope"] == "DEPENDENCIES_AUTHENTICATED_ONLY"
    assert (
        firewall["ctypes_native_load_symbol_lookup_and_call_events_forbidden_after_scope"] is True
    )
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    assert '_AUDIT["trusted_dependency_import"] = True' in bootstrap
    assert '_AUDIT["trusted_dependency_import"] = False' in bootstrap
    assert 'event.startswith("ctypes.")' in bootstrap


def test_all_semantic_path_apis_are_buffer_adapted() -> None:
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    r2 = (
        ROOT / "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_contract_r2.py"
    ).read_text(encoding="utf-8")
    assert r2.count("pq.ParquetFile(") == 4
    assert "def pq_parquet_file(source, *args, **kwargs):" in bootstrap
    assert "pa.BufferReader(held_bytes(source))" in bootstrap
    assert "pq.ParquetFile = pq_parquet_file" in bootstrap
    assert '"pq_ParquetFile": pq.ParquetFile' in bootstrap
    assert "io.BytesIO(held_bytes(file))" in bootstrap


def test_strict_json_proxy_rejects_duplicates_and_nonfinite() -> None:
    for payload in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        completed = _probe(f"ns['_strict_json_value']({payload!r},'poison')", before="import json")
        assert completed.returncode != 0
        assert (
            "duplicate JSON key" in completed.stderr or "non-finite JSON value" in completed.stderr
        )


def test_exact_two_science_adaptations_and_research_only_are_preserved() -> None:
    config = _config()
    contract = config["compatibility_contract"]
    assert contract["adaptation_count"] == 2
    assert contract["only_scoped_science_adaptations"] == [
        "replace_frozen_r2_verifier_prefix_expectation_with_four_source_consensus",
        "admit_exact_pinned_historical_failure_receipt_to_the_r2_control_inventory",
    ]
    assert contract["bootstrap_replicates_per_point"] == 5000
    assert contract["bootstrap_points"] == 5
    assert config["expected_result"]["gate_decision"] == "RESEARCH_ONLY"
    assert config["expected_result"]["local_gate_passed"] is False


def test_v4_control_receipts_are_absent_and_sentinel_is_exact() -> None:
    config = _config()
    for relative in config["canonical_paths"].values():
        assert not (ROOT / relative).exists()
        assert not (ROOT / relative).is_symlink()
    assert PYCACHE_SENTINEL.stat().st_size == 74
    assert _sha(PYCACHE_SENTINEL) == (
        "ddb8423e21b551829ced83fb63c56d17df123ba9459fc44a3ddfbbb8735c55bd"
    )


@pytest.mark.parametrize("path", [HELPER, CLI])
def test_direct_helper_or_cli_execution_fails_closed(path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "requires the trusted v4 bootstrap" in completed.stderr


def test_direct_bootstrap_without_external_attestation_fails_closed() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(BOOTSTRAP),
            "--root",
            str(ROOT),
        ],
        cwd=ROOT,
        env=_canonical_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "external launcher startup attestation is required" in completed.stderr


def test_all_static_side_effect_counters_are_zero() -> None:
    config = _config()
    assert config["status"].endswith("NO_RECEIPT")
    assert config["static_counters"]
    assert not any(config["static_counters"].values())
    for key in (
        "r2_mutation_allowed",
        "r2_rerun_or_resume_allowed",
        "qa_or_compatibility_receipt_write_allowed",
        "execution_authorization_or_attempt_lock_allowed",
        "fit_prediction_or_new_score_allowed",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "registry_append_allowed",
        "upload_allowed",
    ):
        assert config[key] is False


def test_central_v9_seq5_anchor_is_unchanged() -> None:
    path = ROOT / "artifacts/meaningful_score_goal_v9/registry.jsonl"
    raw = path.read_bytes()
    assert len(raw) == 15812
    assert hashlib.sha256(raw).hexdigest() == (
        "232b6ed3133de11ee05150ec439efe05baa315bbb64ea0f319ffcbddd421b965"
    )
    lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert [line["seq"] for line in lines] == [3, 4, 5]
    assert lines[-1]["event_sha256"] == (
        "1b3e01be70c6f8ed2df04038deac3b3642804f70f9f17a238826c64d68090317"
    )
    assert sum(int(line.get("payload", {}).get("uploads", 0)) for line in lines) == 0


def test_canonical_external_launcher_check_only_passes_without_state_change() -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    environment = _canonical_environment()
    watched = [path for path, unused_size, unused_sha in V3.values()]
    watched.extend([V3_NO_GO, V3_TOMBSTONE])
    watched.append(ROOT / "artifacts/meaningful_score_goal_v9/registry.jsonl")
    before = {path: (path.stat().st_size, _sha(path)) for path in watched}
    completed = subprocess.run(
        [
            pwsh,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            environment["P3_V4_STAGE0_ENCODED_COMMAND"],
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == ("PASS_HELD_BUFFER_R2_COMPATIBILITY_RESEARCH_ONLY_NO_PROMOTION")
    runtime = result["trusted_runtime"]
    assert runtime["external_startup_trust"]["pre_script_bytecode_executed"] is False
    assert runtime["pre_script_imports_trusted_by_external_launcher"] is True
    assert not any(runtime["audit_attempts"].values())
    assert runtime["main_namespace_winapi_capability_scan"] == {
        "factory_deleted_before_authenticated_module_execution": True,
        "original_or_broker_globals": 0,
        "raw_winapi_callables": 0,
    }
    registry = result["stable_semantic_registry"]
    assert registry["same_handle_final_rehashes"] == registry["files"]
    assert registry["native_same_handle_final_rehashes"] > 0
    counts = registry["semantic_parse_counts"]
    assert counts["parquet_file_from_buffer"] >= 4
    assert counts["parquet_read_table_from_buffer"] >= 4
    assert counts["npy_from_buffer"] > 0
    assert counts["strict_legacy_json"] > 0
    assert registry["strict_json_files_preauthenticated"] > 0
    assert result["compatibility_adaptations"]["count"] == 2
    assert before == {path: (path.stat().st_size, _sha(path)) for path in watched}
    assert PYCACHE_SENTINEL.stat().st_size == 74
