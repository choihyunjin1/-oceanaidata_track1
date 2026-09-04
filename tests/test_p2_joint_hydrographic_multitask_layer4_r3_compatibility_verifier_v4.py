from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-p1" / "Scripts" / "python.exe"
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v4.json"
)
BOOTSTRAP = (
    ROOT / "scripts" / "bootstrap_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v4.py"
)
LAUNCHER = (
    ROOT / "scripts" / "launch_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v4.ps1"
)
HELPER = (
    ROOT
    / "src"
    / "p2_restore"
    / "joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v4.py"
)
CLI = ROOT / "scripts" / "verify_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v4.py"
V3_CONTROL = (
    ROOT
    / "artifacts"
    / "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3_control"
)
V4_CONTROL = (
    ROOT
    / "artifacts"
    / "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v4_control"
)
R3_OUTPUT = ROOT / "artifacts" / "p2_joint_hydrographic_multitask_layer4_execution_r3"
R3_CONTROL = ROOT / "artifacts" / "p2_joint_hydrographic_multitask_layer4_execution_r3_control"


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


def _canonical_command(root: Path = ROOT) -> list[str]:
    host = Path(os.environ["P2_POWERSHELL_HOST"])
    assert host.is_absolute()
    assert host.is_file()
    return [
        str(host.resolve(strict=True)),
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(root / LAUNCHER.relative_to(ROOT)),
        "-Root",
        str(root),
    ]


def _run(
    command: list[str], *, cwd: Path = ROOT, timeout: int = 240
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="strict",
        capture_output=True,
        check=False,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def canonical_envelope() -> dict[str, Any]:
    before = {
        "output": _tree_snapshot(R3_OUTPUT),
        "control": _tree_snapshot(R3_CONTROL),
        "v3": _tree_snapshot(V3_CONTROL),
        "v4_exists": V4_CONTROL.exists(),
    }
    result = _run(_canonical_command())
    assert result.returncode == 0, result.stderr
    envelope = json.loads(result.stdout)
    after = {
        "output": _tree_snapshot(R3_OUTPUT),
        "control": _tree_snapshot(R3_CONTROL),
        "v3": _tree_snapshot(V3_CONTROL),
        "v4_exists": V4_CONTROL.exists(),
    }
    assert after == before
    return envelope


@pytest.fixture(scope="module")
def canonical_report(canonical_envelope: dict[str, Any]) -> dict[str, Any]:
    return canonical_envelope["child_report"]


def test_v3_five_files_are_exact_and_unchanged() -> None:
    expected = {
        "CONFIG": (
            "configs/experiments/"
            "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.json",
            8236,
            "08bf804b0367951a43a80ffa22dc1067e3c1a1bc516da8351a64e97203a597ec",
        ),
        "BOOTSTRAP": (
            "scripts/bootstrap_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v3.py",
            44253,
            "12835a2f3c0caa6341b5eac0e001935f8466356504d64d5e7fcb23f5c1c1b508",
        ),
        "HELPER": (
            "src/p2_restore/joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.py",
            28473,
            "4f7d327afadac3ce9ff244ada52aca851df8fd39e089cdd310745880643eabc3",
        ),
        "CLI": (
            "scripts/verify_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v3.py",
            1843,
            "a7feac34b70ee3160f98c8bba79aa266a8a4da0e43000edb398211ee463bcd60",
        ),
        "TESTS": (
            "tests/test_p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v3.py",
            21385,
            "1ab2268d59b2bece36e125ce8060105befc7e3aa025ca44c914e993d861c63ce",
        ),
    }
    for relative, size, digest in expected.values():
        assert _pin(ROOT / relative) == (size, digest)


def test_v3_no_go_and_tombstone_have_no_persisted_independent_receipt() -> None:
    owner = json.loads((V3_CONTROL / "OWNER_STATIC_QA_NO_GO_20260823.json").read_text("utf-8"))
    tombstone = json.loads((V3_CONTROL / "EXECUTION_TOMBSTONE.json").read_text("utf-8"))
    assert owner["verdict"] == "P0=0_P1=2_NO_GO"
    assert owner["independent_qa_review_performed"] is True
    assert owner["independent_qa_receipt_created"] is False
    assert owner["independent_qa_receipt_path"] is None
    assert [item["code"] for item in owner["findings"]] == [
        "PREHOOK_STARTUP_TRUST_ROOT_NOT_CLOSED",
        "ZERO_ACTION_AUDIT_FIREWALL_BYPASSABLE",
    ]
    assert tombstone["owner_no_go_receipt"] == {
        "path": (V3_CONTROL / "OWNER_STATIC_QA_NO_GO_20260823.json").relative_to(ROOT).as_posix(),
        "bytes": _pin(V3_CONTROL / "OWNER_STATIC_QA_NO_GO_20260823.json")[0],
        "sha256": _pin(V3_CONTROL / "OWNER_STATIC_QA_NO_GO_20260823.json")[1],
    }
    assert tombstone["v3_compatibility_pass_must_fail_closed"] is True
    assert tombstone["execution_prohibited"] is True


def test_v4_config_declares_noncyclic_external_startup_root() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    runtime = config["canonical_runtime_contract"]
    assert runtime["required_cli_flags"] == [
        "-I",
        "-S",
        "-B",
        "-X pycache_prefix=<canonical-absent-path>",
    ]
    assert runtime["external_launcher_required"] is True
    assert runtime["external_launcher_must_be_independently_pinned"] is True
    assert runtime["external_powershell_host_env"] == "P2_POWERSHELL_HOST"
    assert runtime["external_powershell_host_exact_absolute_path_required"] is True
    assert runtime["external_powershell_host_must_be_pinned_before_execution"] is True
    assert runtime["external_launcher_must_be_pinned_before_execution"] is True
    assert runtime["host_and_launcher_self_authentication_claimed"] is False
    assert runtime["external_launcher_holds_all_startup_files_until_child_exit"] is True
    assert runtime["external_launcher_post_child_same_handle_rehash"] is True
    assert runtime["prehook_encodings_source_count"] == 4
    assert runtime["prehook_native_module_origins"] == {
        "_codecs_kr": "built-in",
        "_multibytecodec": "built-in",
    }
    assert len(runtime["external_startup_file_pins"]) == 11
    assert runtime["ctypes_socket_mmap_or_spawn_allowed"] is False
    assert runtime["write_process_network_registry_attempts_required"] == 0
    assert config["stable_semantic_read_contract"]["regular_file_nlink_required"] == 1
    assert not V4_CONTROL.exists()


def test_externally_injected_powershell_host_pin_matches() -> None:
    host = Path(os.environ["P2_POWERSHELL_HOST"])
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pin = config["canonical_runtime_contract"]["external_powershell_host_pin"]
    assert host.is_absolute()
    assert _pin(host.resolve(strict=True)) == (pin["bytes"], pin["sha256"])
    assert host.stat().st_nlink == 1
    assert not host.is_symlink()


def test_external_startup_file_pins_match_current_files() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pins = config["canonical_runtime_contract"]["external_startup_file_pins"]
    pyvenv = (ROOT / ".venv-p1" / "pyvenv.cfg").read_text(encoding="utf-8")
    home = next(
        line.split("=", 1)[1].strip() for line in pyvenv.splitlines() if line.startswith("home =")
    )
    base = Path(home)
    for pin in pins.values():
        root = ROOT if pin["scope"] == "workspace" else base
        path = root.joinpath(*Path(pin["relative"]).parts)
        assert _pin(path) == (pin["bytes"], pin["sha256"])
        assert path.stat().st_nlink == 1
        assert not path.is_symlink()


def test_canonical_launcher_contains_required_native_handle_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for token in (
        "CreateFileW",
        "GENERIC_READ",
        "FILE_SHARE_READ",
        "OPEN_EXISTING",
        "FILE_FLAG_OPEN_REPARSE_POINT",
        "GetFileInformationByHandle",
        "GetFinalPathNameByHandleW",
        "NumberOfLinks",
        "PostChildRehash",
        "-Xpycache_prefix=",
        "ExternalLauncherAttestation",
    ):
        assert token in source


def test_direct_bootstrap_without_external_attestation_fails() -> None:
    prefix = ROOT / "artifacts" / "p2_v4_direct_probe_absent_pycache"
    assert not prefix.exists()
    result = _run(
        [
            str(PYTHON),
            "-I",
            "-S",
            "-B",
            f"-Xpycache_prefix={prefix}",
            str(BOOTSTRAP),
            "--root",
            str(ROOT),
            "--mode",
            "check-only",
        ],
        timeout=30,
    )
    assert result.returncode != 0
    assert "external launcher attestation" in result.stderr.lower()
    assert not prefix.exists()


@pytest.mark.parametrize("missing_flag", ["-I", "-S", "-B"])
def test_bootstrap_rejects_any_missing_canonical_flag(missing_flag: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    prefix = CONFIG if missing_flag == "-B" else ROOT / config["canonical_absent_pycache_prefix"]
    command = [
        str(PYTHON),
        "-I",
        "-S",
        "-B",
        f"-Xpycache_prefix={prefix}",
        str(BOOTSTRAP),
        "--root",
        str(ROOT),
        "--mode",
        "check-only",
        "--external-launcher-attestation",
        "owner-static-test",
    ]
    command.remove(missing_flag)
    result = _run(command, timeout=30)
    assert result.returncode != 0
    assert "canonical python" in result.stderr.lower()


def test_absent_pycache_prefix_executes_four_sources_not_pyc() -> None:
    prefix = ROOT / "artifacts" / "p2_v4_verbose_probe_absent_pycache"
    assert not prefix.exists()
    result = _run(
        [
            str(PYTHON),
            "-I",
            "-S",
            "-B",
            f"-Xpycache_prefix={prefix}",
            "-v",
            "-c",
            "pass",
        ],
        timeout=30,
    )
    assert result.returncode == 0
    startup = [
        line for line in result.stderr.splitlines() if "encodings" in line and "code object" in line
    ]
    assert len(startup) == 4
    assert all(".pyc" not in line for line in startup)
    assert all(".py" in line for line in startup)
    assert not prefix.exists()


def _load_hook() -> tuple[Any, dict[str, Any]]:
    tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"), filename=str(BOOTSTRAP))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_is_forbidden_import", "_audit_hook"}
    ]
    firewall = {
        "installed": True,
        "dependency_ready": True,
        "write_process_network_registry_attempts": 0,
        "forbidden_import_attempts": 0,
        "unexpected_import_attempts": 0,
        "bytecode_read_attempts": 0,
    }
    namespace: dict[str, Any] = {
        "_FORBIDDEN_ENGINE": "p2_restore.engine",
        "_FORBIDDEN_ROOTS": ("ctypes", "_ctypes", "socket", "_socket", "mmap", "numpy"),
        "_WRITE_FLAGS": 1 | 2 | 8 | 256 | 512 | 1024,
        "_GENERIC_READ": 0x80000000,
        "_OPEN_EXISTING": 3,
        "_MUTATION_OR_PROCESS_EVENTS": {"os.remove", "subprocess.Popen"},
        "_BLOCKED_EVENT_PREFIXES": ("os.spawn", "os.exec", "socket.", "winreg.", "ctypes."),
        "_FIREWALL": firewall,
        "_ALLOWED_IMPORTS": set(sys.modules),
        "_EXPECTED_ABSENT_PLATFORM_PROBES": set(),
        "_UNEXPECTED_IMPORT_NAMES": [],
        "_sys": sys,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(BOOTSTRAP), "exec"), namespace)  # noqa: S102
    return namespace["_audit_hook"], firewall


@pytest.mark.parametrize(
    ("event", "arguments"),
    [
        ("_winapi.CreateFile", ("NUL", 0x40000000, 3, 3, 0)),
        ("_winapi.CreateProcess", (None, "bad", None)),
        ("os.spawn", (0, "bad", ("bad",), None)),
        ("socket.sendto", (object(), b"", ("127.0.0.1", 9))),
        ("winreg.SetValue", (object(), "x", 1, "y")),
        ("ctypes.dlsym", (object(), "CreateFileW")),
        ("open", ("bad.pyc", "r", 0)),
    ],
)
def test_audit_hook_rejects_each_bypass_family(event: str, arguments: tuple[Any, ...]) -> None:
    hook, _firewall = _load_hook()
    with pytest.raises((PermissionError, ImportError)):
        hook(event, arguments)


@pytest.mark.parametrize("name", ["ctypes", "_ctypes", "socket", "_socket", "mmap", "numpy"])
def test_audit_hook_rejects_risky_imports(name: str) -> None:
    hook, _firewall = _load_hook()
    with pytest.raises(ImportError):
        hook("import", (name,))


def _load_helper_for_unit_contract() -> Any:
    module = types.ModuleType("p2_v4_unit_helper")
    module.__file__ = str(HELPER)
    module.__package__ = "p2_restore"
    module.__dict__["_P2_V4_BOOTSTRAP_CONTEXT"] = {}
    exec(compile(HELPER.read_bytes(), str(HELPER), "exec"), module.__dict__)  # noqa: S102
    return module


def _valid_fold_audits() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    order = ["outer_2024_sep_oct", "outer_2025_may_jun", "outer_2025_jul_aug"]
    forbidden = [
        "active_fold_target_temp_psal_scalar_fields_decoded_or_converted",
        "withheld_target_temp_psal_scalar_fields_decoded_or_converted",
        "anomaly_or_hidden_target_proxy_reads",
    ]
    audits = {
        fold: {
            "fold": fold,
            "verified_prior_fold_commitments": index,
            **{field: 0 for field in forbidden},
        }
        for index, fold in enumerate(order)
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


def test_corrected_fold_predicate_accepts_only_mapping_order_difference() -> None:
    helper = _load_helper_for_unit_contract()
    persisted, commitments, contract = _valid_fold_audits()
    report = helper._verify_fold_audits(persisted, commitments, contract)
    assert report["mapping_insertion_order_ignored"] is True
    assert [item["verified_prior_fold_commitments"] for item in report["verified"]] == [0, 1, 2]


@pytest.mark.parametrize("forgery", ["extra", "prior", "decode", "commitment"])
def test_corrected_fold_predicate_rejects_semantic_forgery(forgery: str) -> None:
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
    with pytest.raises(helper.CompatibilityVerifierV4Error):
        helper._verify_fold_audits(persisted, commitments, contract)


def test_direct_helper_and_cli_require_authenticated_injection() -> None:
    for path in (HELPER, CLI):
        result = _run([str(PYTHON), "-I", "-S", "-B", str(path)], timeout=30)
        assert result.returncode != 0
        assert "requires the authenticated bootstrap" in result.stderr


def test_external_launcher_envelope_is_noncyclic_and_stable(
    canonical_envelope: dict[str, Any],
) -> None:
    trust = canonical_envelope["external_startup_trust"]
    assert trust["launcher_self_authentication_claimed"] is False
    assert trust["host_self_authentication_claimed"] is False
    assert trust["host_requires_independent_preexecution_pin"] is True
    assert trust["launcher_requires_independent_preexecution_pin"] is True
    assert trust["launcher_requires_independent_pin"] is True
    assert trust["startup_files"] == 11
    assert trust["startup_files_prehashed"] == 11
    assert trust["startup_files_post_child_rehashed"] == 11
    assert trust["all_handles_held_until_child_exit"] is True
    assert trust["share_write_allowed"] is False
    assert trust["share_delete_allowed"] is False
    assert trust["open_reparse_point"] is True
    assert trust["all_regular_file_nlinks"] == 1
    assert trust["canonical_pycache_prefix_absent_before"] is True
    assert trust["canonical_pycache_prefix_absent_after"] is True


def test_canonical_report_closes_dependency_and_stable_byte_boundaries(
    canonical_report: dict[str, Any],
) -> None:
    assert canonical_report["status"] == (
        "PASS_OWNER_STATIC_EXTERNAL_STARTUP_ROOT_STABLE_R3_COMPATIBILITY_"
        "AWAITING_FRESH_INDEPENDENT_QA_RESEARCH_ONLY_LOCAL_FAIL"
    )
    bootstrap = canonical_report["authenticated_bootstrap"]
    dependency = canonical_report["dependency_trust"]
    stable = canonical_report["stable_registry"]
    assert bootstrap["canonical_python_flags"] == ["-I", "-S", "-B", "-Xpycache_prefix"]
    assert bootstrap["prehook_startup_sources_rehashed"] == 4
    assert bootstrap["write_process_network_registry_attempts"] == 0
    assert bootstrap["forbidden_import_attempts"] == 0
    assert bootstrap["unexpected_import_attempts"] == 0
    assert bootstrap["bytecode_read_attempts"] == 0
    assert dependency["stdlib_inventory"]["files"] == 2443
    assert dependency["module_origins"]["third_party_distributions"] == []
    assert dependency["module_origins"]["numerical_distributions_imported"] == []
    assert stable["same_handle_final_rehashes"] == stable["files"]
    assert stable["all_regular_file_nlinks"] == 1
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
    assert canonical_report["v4_control_exists"] is False
    assert canonical_report["pre_execution_qa_exists"] is False
    assert canonical_report["compatibility_receipt_exists"] is False
    assert canonical_report["authenticated_cli"]["r3_engine_imported"] is False
    assert not V4_CONTROL.exists()


def test_no_absolute_personal_path_is_persisted_in_v4_sources() -> None:
    personal = ("C:" + "\\Users\\" + "cedis").casefold()
    for path in (BOOTSTRAP, CONFIG, HELPER, CLI, LAUNCHER, Path(__file__).resolve()):
        assert personal not in path.read_text(encoding="utf-8").casefold()


def test_bootstrap_uses_only_authenticated_source_loader_after_startup() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "SourceFileLoader(" not in source
    assert "_sys.meta_path[:]" in source
    assert "_AuthenticatedFinder" in source
    assert "FILE_SHARE_READ" in source
    assert "FILE_FLAG_OPEN_REPARSE_POINT" in source
    assert "same-handle final hash changed" in source
    assert '"os.spawn"' in source
    assert '"ctypes."' in source
    assert '"socket."' in source
    assert '"winreg."' in source
