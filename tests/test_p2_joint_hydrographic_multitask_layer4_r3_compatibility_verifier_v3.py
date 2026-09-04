from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-p1" / "Scripts" / "python.exe"
BOOTSTRAP = (
    ROOT / "scripts" / "bootstrap_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v3.py"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.json"
)
HELPER = (
    ROOT
    / "src"
    / "p2_restore"
    / "joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.py"
)
CLI = ROOT / "scripts" / "verify_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v3.py"
V2_CONTROL = (
    ROOT
    / "artifacts"
    / "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v2_control"
)
V3_CONTROL = (
    ROOT
    / "artifacts"
    / "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3_control"
)
R3_OUTPUT = ROOT / "artifacts" / "p2_joint_hydrographic_multitask_layer4_execution_r3"
R3_CONTROL = ROOT / "artifacts" / "p2_joint_hydrographic_multitask_layer4_execution_r3_control"
V2_PINS = {
    "BOOTSTRAP": (
        "scripts/bootstrap_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v2.py",
        17996,
        "79dddfca4c177a8b6c30642f169dac33c1dd3601bddd0013f6d62dca41618ed2",
    ),
    "TRUST_ANCHOR": (
        "configs/experiments/"
        "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v2_trust_anchor.json",
        1245,
        "f99393120c31c4f4ffaf2b804c5a22d0b2a469cb8149d00036aaa72dd12fd75e",
    ),
    "CONFIG": (
        "configs/experiments/"
        "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v2.json",
        5568,
        "5244ae78fb85b66a29a3831d8b3ea6cfd2dfa7a784b660dc116b21e4c6014c3a",
    ),
    "HELPER": (
        "src/p2_restore/joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v2.py",
        26842,
        "b4f2455cba231f64f3f749472c6b6f9eb921ad6f14727843bb66a6f8685bb10c",
    ),
    "CLI": (
        "scripts/verify_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v2.py",
        1859,
        "ddbbbddd2151ca2c1eba8077794de4f0eb51a0282c5002d5f402f8b4e7a69454",
    ),
    "TESTS": (
        "tests/test_p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v2.py",
        16296,
        "4171a8c2a90e0ce67913026aefce997350d06ccecb7febe82abaef21596f7030",
    ),
}


def _pin(path: Path) -> tuple[int, str]:
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def _tree_snapshot(root: Path) -> list[tuple[str, str, int, str]]:
    if not root.exists():
        return []
    result: list[tuple[str, str, int, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            result.append((relative, "directory", 0, ""))
        elif path.is_file():
            raw = path.read_bytes()
            result.append((relative, "file", len(raw), hashlib.sha256(raw).hexdigest()))
        else:
            result.append((relative, "special", 0, ""))
    return result


def _canonical_command(root: Path = ROOT, bootstrap: Path = BOOTSTRAP) -> list[str]:
    python = root / ".venv-p1" / "Scripts" / "python.exe"
    return [
        str(python),
        "-I",
        "-S",
        "-B",
        str(bootstrap),
        "--root",
        str(root),
        "--mode",
        "check-only",
    ]


def _run(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


@pytest.fixture(scope="module")
def canonical_report() -> dict[str, Any]:
    before = {
        "output": _tree_snapshot(R3_OUTPUT),
        "control": _tree_snapshot(R3_CONTROL),
        "v2": _tree_snapshot(V2_CONTROL),
        "v3_exists": V3_CONTROL.exists(),
    }
    result = _run(_canonical_command())
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    after = {
        "output": _tree_snapshot(R3_OUTPUT),
        "control": _tree_snapshot(R3_CONTROL),
        "v2": _tree_snapshot(V2_CONTROL),
        "v3_exists": V3_CONTROL.exists(),
    }
    assert after == before
    return report


def test_v2_six_files_are_exact_and_unchanged() -> None:
    for relative, size, digest in V2_PINS.values():
        assert _pin(ROOT / relative) == (size, digest)


def test_owner_v2_no_go_and_tombstone_preserve_v1_lineage() -> None:
    receipt_path = V2_CONTROL / "OWNER_STATIC_QA_NO_GO_20260823.json"
    tombstone_path = V2_CONTROL / "EXECUTION_TOMBSTONE.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    tombstone = json.loads(tombstone_path.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "P0=0_P1=2_OWNER_KNOWN_NO_GO"
    assert receipt["independent_p2_v2_review_claimed"] is False
    assert [item["code"] for item in receipt["findings"]] == [
        "PREIMPORT_AND_DEPENDENCY_TRUST_BOUNDARY_NOT_CLOSED",
        "PINNED_ARTIFACT_BYTES_NOT_BOUND_TO_SEMANTIC_REPLAY",
    ]
    assert receipt["preserved_v1_no_go_lineage"]["verdict"] == "P0=0_P1=3_NO_GO"
    assert tombstone["execution_prohibited"] is True
    assert tombstone["v2_compatibility_pass_must_fail_closed"] is True
    assert tombstone["owner_no_go_receipt"] == {
        "path": receipt_path.relative_to(ROOT).as_posix(),
        "bytes": _pin(receipt_path)[0],
        "sha256": _pin(receipt_path)[1],
    }


def test_v3_config_is_static_hermetic_and_receipt_free() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    runtime = config["canonical_runtime_contract"]
    stable = config["stable_semantic_read_contract"]
    assert runtime["required_cli_flags"] == ["-I", "-S", "-B"]
    assert runtime["third_party_distributions"] == []
    assert runtime["third_party_record_files"] == []
    assert runtime["site_packages_on_sys_path"] is False
    assert runtime["post_hook_pyc_or_pyo_read_allowed"] is False
    assert runtime["direct_winapi_create_or_process_allowed"] is False
    assert runtime["stdlib_inventory"] == {
        "base_root_direct_files": "DLL_ONLY",
        "recursive_roots": ["DLLs", "Lib"],
        "excluded_directory_names": ["__pycache__", "site-packages"],
        "excluded_file_suffixes": [".pyc", ".pyo"],
        "directories": 199,
        "files": 2443,
        "file_bytes": 66487423,
        "algorithm": "SHA256_SORTED_TYPE_NUL_RELATIVE_NUL_BYTES_NUL_FILE_SHA256_LF",
        "sha256": "5cc5d4b2f90199292a4334a6530eaa90c288fd45723ba5290295a3803d13eeba",
    }
    assert stable["concurrent_swap_reopen_gap_allowed"] is False
    assert stable["strict_json_from_single_authenticated_buffer"] is True
    assert stable["npy_and_csv_from_authenticated_buffers"] is True
    assert stable["open_reparse_point_required"] is True
    assert stable["post_open_reparse_and_identity_check_required"] is True
    assert not V3_CONTROL.exists()


@pytest.mark.parametrize("missing_flag", ["-I", "-S", "-B"])
def test_bootstrap_rejects_any_missing_canonical_flag(missing_flag: str) -> None:
    command = [item for item in _canonical_command() if item != missing_flag]
    result = _run(command)
    assert result.returncode != 0
    assert "canonical python -I -S -B runtime is required" in result.stderr


def _copy_minimal_workspace(destination: Path) -> None:
    for relative in (
        ".venv-p1/Scripts/python.exe",
        ".venv-p1/pyvenv.cfg",
        BOOTSTRAP.relative_to(ROOT).as_posix(),
        CONFIG.relative_to(ROOT).as_posix(),
        HELPER.relative_to(ROOT).as_posix(),
        CLI.relative_to(ROOT).as_posix(),
        Path(__file__).resolve().relative_to(ROOT).as_posix(),
    ):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


@pytest.mark.parametrize(
    "relative",
    [
        "configs/experiments/"
        "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.json",
        "src/p2_restore/joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.py",
        "scripts/verify_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v3.py",
        "tests/test_p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.py",
    ],
)
def test_any_v3_role_drift_fails_before_execution(tmp_path: Path, relative: str) -> None:
    workspace = tmp_path / "workspace"
    _copy_minimal_workspace(workspace)
    target = workspace / relative
    target.write_bytes(target.read_bytes() + b"\n# forged drift\n")
    bootstrap = workspace / BOOTSTRAP.relative_to(ROOT)
    result = _run(_canonical_command(workspace, bootstrap), cwd=workspace)
    assert result.returncode != 0
    assert "stable authenticated bytes changed" in result.stderr
    assert not (workspace / "artifacts").exists()


def test_isolated_runtime_ignores_sibling_shadow_module(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _copy_minimal_workspace(workspace)
    marker = workspace / "shadow_marker"
    shadow = workspace / "scripts" / "json.py"
    shadow.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    config = workspace / CONFIG.relative_to(ROOT)
    config.write_bytes(config.read_bytes() + b"\n")
    bootstrap = workspace / BOOTSTRAP.relative_to(ROOT)
    result = _run(_canonical_command(workspace, bootstrap), cwd=workspace)
    assert result.returncode != 0
    assert "stable authenticated bytes changed" in result.stderr
    assert not marker.exists()


def test_reparse_implementation_target_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _copy_minimal_workspace(workspace)
    target = workspace / CONFIG.relative_to(ROOT)
    target.unlink()
    try:
        target.symlink_to(CONFIG)
    except OSError:
        junction = target.parent
        junction.rmdir()
        junction_text = str(junction).replace("'", "''")
        source_text = str(CONFIG.parent).replace("'", "''")
        junction_result = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                (
                    f"New-Item -ItemType Junction -Path '{junction_text}' "
                    f"-Target '{source_text}' | Out-Null"
                ),
            ],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if junction_result.returncode != 0:
            pytest.skip(f"reparse creation is unavailable: {junction_result.stderr}")
    bootstrap = workspace / BOOTSTRAP.relative_to(ROOT)
    result = _run(_canonical_command(workspace, bootstrap), cwd=workspace)
    assert result.returncode != 0
    assert "reparse" in result.stderr.lower()


def test_windows_share_read_handle_denies_concurrent_write_and_replace(tmp_path: Path) -> None:
    target = tmp_path / "protected.bin"
    replacement = tmp_path / "replacement.bin"
    target.write_bytes(b"frozen")
    replacement.write_bytes(b"forged")
    holder = (
        "import _winapi,msvcrt,nt,sys;"
        "p=sys.argv[1];"
        "h=_winapi.CreateFile(p,_winapi.GENERIC_READ,1,0,_winapi.OPEN_EXISTING,0,0);"
        "fd=msvcrt.open_osfhandle(h,nt.O_RDONLY|nt.O_BINARY);"
        "print('READY',flush=True);"
        "sys.stdin.readline();"
        "nt.close(fd)"
    )
    process = subprocess.Popen(
        [str(PYTHON), "-I", "-S", "-B", "-c", holder, str(target)],
        cwd=ROOT,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        with pytest.raises(PermissionError):
            target.write_bytes(b"changed")
        with pytest.raises(PermissionError):
            os.replace(replacement, target)
        assert target.read_bytes() == b"frozen"
    finally:
        if process.stdin is not None:
            process.stdin.write("\n")
            process.stdin.flush()
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stdout + stderr


def test_audit_hook_rejects_direct_winapi_process_and_bytecode_bypasses() -> None:
    tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"), filename=str(BOOTSTRAP))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_is_forbidden_import", "_audit_hook"}
    ]
    namespace: dict[str, Any] = {
        "_FORBIDDEN_ENGINE": "p2_restore.engine",
        "_FORBIDDEN_ROOTS": ("numpy",),
        "_WRITE_FLAGS": 1 | 2 | 8 | 256 | 512 | 1024,
        "_GENERIC_READ": 0x80000000,
        "_OPEN_EXISTING": 3,
        "_MUTATION_OR_PROCESS_EVENTS": {"_winapi.CreateProcess"},
        "_FIREWALL": {
            "installed": True,
            "dependency_ready": True,
            "write_delete_process_attempts": 0,
            "forbidden_import_attempts": 0,
            "unexpected_import_attempts": 0,
            "bytecode_read_attempts": 0,
        },
        "_ALLOWED_IMPORTS": set(),
        "_EXPECTED_ABSENT_PLATFORM_PROBES": set(),
        "_UNEXPECTED_IMPORT_NAMES": [],
        "_sys": types.SimpleNamespace(builtin_module_names=()),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(BOOTSTRAP), "exec"), namespace)  # noqa: S102
    hook = namespace["_audit_hook"]
    hook("_winapi.CreateFile", ("frozen", 0x80000000, 1, 3, 0x00200000))
    with pytest.raises(PermissionError):
        hook("_winapi.CreateFile", ("forged", 0x40000000, 0, 2, 0))
    with pytest.raises(PermissionError):
        hook("_winapi.CreateProcess", (None, "forged", None))
    with pytest.raises(PermissionError):
        hook("open", ("forged.pyc", "r", 0))
    with pytest.raises(ImportError):
        hook("import", ("numpy",))
    firewall = namespace["_FIREWALL"]
    assert firewall["write_delete_process_attempts"] == 2
    assert firewall["bytecode_read_attempts"] == 1
    assert firewall["forbidden_import_attempts"] == 1


def _load_helper_for_unit_contract() -> Any:
    module = types.ModuleType("p2_v3_unit_helper")
    module.__file__ = str(HELPER)
    module.__package__ = "p2_restore"
    module.__dict__["_P2_V3_BOOTSTRAP_CONTEXT"] = {}
    exec(compile(HELPER.read_bytes(), str(HELPER), "exec"), module.__dict__)  # noqa: S102
    return module


def _valid_fold_audits() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    order = ["outer_2024_sep_oct", "outer_2025_may_jun", "outer_2025_jul_aug"]
    forbidden = [
        "active_fold_target_temp_psal_scalar_fields_decoded_or_converted",
        "withheld_target_temp_psal_scalar_fields_decoded_or_converted",
        "anomaly_or_hidden_target_proxy_reads",
    ]
    audits: dict[str, Any] = {}
    for index, fold in enumerate(order):
        audits[fold] = {
            "fold": fold,
            "verified_prior_fold_commitments": index,
            **{field: 0 for field in forbidden},
        }
    contract = {
        "canonical_fold_order": order,
        "verified_prior_fold_commitments_by_canonical_fold": {
            fold: index for index, fold in enumerate(order)
        },
        "forbidden_decode_fields": forbidden,
    }
    persisted = {order[0]: audits[order[0]], order[2]: audits[order[2]], order[1]: audits[order[1]]}
    return persisted, {key: dict(value) for key, value in audits.items()}, contract


def test_corrected_fold_predicate_accepts_only_insertion_order_difference() -> None:
    helper = _load_helper_for_unit_contract()
    persisted, commitments, contract = _valid_fold_audits()
    report = helper._verify_fold_audits(persisted, commitments, contract)
    assert report["mapping_insertion_order_ignored"] is True
    assert [item["verified_prior_fold_commitments"] for item in report["verified"]] == [0, 1, 2]


@pytest.mark.parametrize("forgery", ["extra", "prior", "decode", "commitment"])
def test_corrected_fold_predicate_rejects_every_semantic_forgery(forgery: str) -> None:
    helper = _load_helper_for_unit_contract()
    persisted, commitments, contract = _valid_fold_audits()
    if forgery == "extra":
        persisted["extra_fold"] = dict(next(iter(persisted.values())))
    elif forgery == "prior":
        persisted["outer_2025_may_jun"]["verified_prior_fold_commitments"] = 2
    elif forgery == "decode":
        persisted["outer_2024_sep_oct"][
            "active_fold_target_temp_psal_scalar_fields_decoded_or_converted"
        ] = 1
    else:
        commitments["outer_2025_jul_aug"]["fold"] = "forged"
    with pytest.raises(helper.CompatibilityVerifierV3Error):
        helper._verify_fold_audits(persisted, commitments, contract)


def test_direct_helper_and_cli_execution_require_authenticated_injection() -> None:
    for path in (HELPER, CLI):
        result = _run([str(PYTHON), "-I", "-S", "-B", str(path)], cwd=ROOT)
        assert result.returncode != 0
        assert "requires the authenticated bootstrap" in result.stderr


def test_canonical_report_closes_dependency_and_stable_byte_boundaries(
    canonical_report: dict[str, Any],
) -> None:
    assert canonical_report["status"] == (
        "PASS_STABLE_AUTHENTICATED_R3_COMPATIBILITY_RESEARCH_ONLY_LOCAL_FAIL"
    )
    bootstrap = canonical_report["authenticated_bootstrap"]
    dependency = canonical_report["dependency_trust"]
    stable = canonical_report["stable_registry"]
    assert bootstrap["canonical_python_flags"] == ["-I", "-S", "-B"]
    assert bootstrap["authenticated_buffer_compile_exec_only"] is True
    assert bootstrap["source_file_loader_or_pyc_used_for_v3"] is False
    assert bootstrap["write_delete_process_attempts"] == 0
    assert bootstrap["forbidden_import_attempts"] == 0
    assert bootstrap["unexpected_import_attempts"] == 0
    assert bootstrap["bytecode_read_attempts"] == 0
    assert dependency["stdlib_inventory"]["files"] == 2443
    assert dependency["module_origins"]["third_party_distributions"] == []
    assert dependency["module_origins"]["numerical_distributions_imported"] == []
    assert stable["same_handle_final_rehashes"] == stable["files"]
    assert stable["share_write_allowed"] is False
    assert stable["share_delete_allowed"] is False


def test_canonical_report_retains_exact_failure_and_research_only_result(
    canonical_report: dict[str, Any],
) -> None:
    original = canonical_report["original_r3_verifier"]
    assert original["failure"] == {
        "exception_type": "Layer4ContractError",
        "message": "Layer-4 seal verification failed: ['receipt_fold_audits']",
        "only_failed_check": "receipt_fold_audits",
    }
    assert original["authenticated_buffer_adapters"] == [
        "_csv_header_and_rows",
        "_npy_little_endian_float64_payload",
        "sha256_file",
        "strict_json_object",
    ]
    assert original["all_original_globals_restored"] is True
    assert all(canonical_report["result_checks"].values())
    assert canonical_report["v9"]["head_sequence"] == 5
    assert canonical_report["v9"]["uploads"] == 0


def test_canonical_report_has_zero_action_state(canonical_report: dict[str, Any]) -> None:
    for field in (
        "independent_qa_receipts_created",
        "compatibility_receipts_created",
        "execution_authorizations_created",
        "attempt_locks_created",
        "model_fit_calls",
        "prediction_calls",
        "truth_scalar_decodes",
        "score_calls",
        "candidate_files",
        "test_prediction_files",
        "registry_appends",
        "uploads",
        "files_written",
    ):
        assert canonical_report[field] == 0
    assert canonical_report["v3_control_exists"] is False
    assert canonical_report["pre_execution_qa_exists"] is False
    assert canonical_report["compatibility_receipt_exists"] is False
    assert canonical_report["authenticated_cli"]["r3_engine_imported"] is False
    assert not V3_CONTROL.exists()


def test_no_absolute_personal_path_is_persisted_in_new_sources() -> None:
    personal = ("C:" + "\\Users\\" + "cedis").casefold()
    for path in (BOOTSTRAP, CONFIG, HELPER, CLI, Path(__file__).resolve()):
        assert personal not in path.read_text(encoding="utf-8").casefold()


def test_bootstrap_contains_no_sourcefileloader_or_untrusted_pathfinder_after_trust() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "SourceFileLoader(" not in source
    assert "_sys.meta_path[:]" in source
    assert "_AuthenticatedFinder" in source
    assert "FILE_SHARE_READ" in source
    assert "FILE_FLAG_OPEN_REPARSE_POINT" in source
    assert 'event == "_winapi.CreateFile"' in source
    assert '"_winapi.CreateProcess"' in source
    assert 'endswith((".pyc", ".pyo"))' in source
    assert "same-handle final hash changed" in source
