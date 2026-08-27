from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_long_event_segment_proposal_rescore_v8.py"


def load_r8() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p1_segment_r8_adversarial", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def namespace_image(root: Path) -> dict[str, tuple[str, bytes]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            ("dir", b"") if path.is_dir() else ("file", path.read_bytes())
        )
        for path in sorted(root.rglob("*"), key=lambda value: str(value).casefold())
    }


def test_parent_without_canonical_authorization_is_namespace_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r8 = load_r8()
    monkeypatch.delenv(r8.AUTHORIZATION_ENV_VAR, raising=False)
    before = namespace_image(r8.CANONICAL_ARTIFACT_DIR)
    with pytest.raises(r8.AuthorizationError):
        r8.execute_parent()
    assert namespace_image(r8.CANONICAL_ARTIFACT_DIR) == before


@pytest.mark.parametrize("secret_length", [0, 31, 33])
def test_pipe_rejects_non_32_byte_capability(secret_length: int) -> None:
    if os.name != "nt":
        pytest.skip("Windows inherited pipe")
    import msvcrt

    r8 = load_r8()
    read_fd, write_fd = os.pipe()
    read_handle = int(msvcrt.get_osfhandle(read_fd))
    payload = r8._canonical_bytes(
        {
            "schema_version": "p1_segment_rescore.parent_pipe_capability.v8",
            "authorization_sha256": "1" * 64,
            "seal_sha256": "2" * 64,
            "qa_sha256": "3" * 64,
            "claim_sha256": "4" * 64,
            "manifest_sha256": "5" * 64,
            "launch_nonce": "6" * 64,
            "parent_identity": {"pid": 1},
        }
    )
    raw = b"x" * secret_length + len(payload).to_bytes(4, "big") + payload
    os.write(write_fd, raw)
    os.close(write_fd)
    with pytest.raises(r8.AuthorizationError):
        r8._read_pipe_capability(read_handle)


def test_equal_bytes_decoy_claim_has_different_file_id(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows FileId contract")
    r8 = load_r8()
    r7 = r8._load_r7()
    canonical = tmp_path / "canonical.claim"
    decoy = tmp_path / "decoy.claim"
    canonical.write_bytes(b"same")
    decoy.write_bytes(b"same")
    with r7.HeldReadDenyMutation([canonical, decoy]) as guard:
        canonical_record = guard.records[str(canonical.resolve())]
        decoy_handle = r8._native_handle_from_guard(guard, decoy)
        decoy_record = r8._handle_file_identity(decoy_handle)
        assert canonical_record["sha256"] == guard.records[str(decoy.resolve())]["sha256"]
        assert canonical_record["file_id"] != decoy_record["file_id"]


def test_unrelated_live_process_handle_is_not_true_parent() -> None:
    if os.name != "nt":
        pytest.skip("Windows process identity")
    from ctypes import wintypes

    r8 = load_r8()
    sleeper = subprocess.Popen(
        [str(Path(os.environ["SystemRoot"]) / "System32/timeout.exe"), "/T", "20", "/NOBREAK"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = int(kernel32.OpenProcess(0x1000 | 0x00100000, False, sleeper.pid))
    try:
        identity = r8._process_identity_from_handle(handle)
        assert identity["pid"] == sleeper.pid
        assert identity["pid"] != r8._actual_os_parent_pid()
    finally:
        kernel32.CloseHandle(handle)
        sleeper.terminate()
        sleeper.wait(timeout=10)


def test_real_native_load_succeeds_while_concurrent_insertion_is_denied(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows ACL/native-loader contract")
    r8 = load_r8()
    source = r8._base_python_root() / "DLLs/libffi-8.dll"
    root = tmp_path / "native"
    root.mkdir()
    target = root / "sealed_libffi_probe.dll"
    shutil.copyfile(source, target)
    attacker_code = (
        "from pathlib import Path; "
        f"Path({str(root / 'planted_dependency.dll')!r}).write_bytes(b'evil')"
    )
    with r8.HeldDirectoryHandles([root]), r8.DirectoryAclFreeze([root]):
        attacker = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", "-c", attacker_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        library = ctypes.WinDLL(str(target))
        stdout, stderr = attacker.communicate(timeout=20)
        del library, stdout, stderr
        assert attacker.returncode != 0
        assert not (root / "planted_dependency.dll").exists()


def test_copied_interpreter_initial_map_contains_exact_vc_runtime(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows initial module map")
    r8 = load_r8()
    inventory = r8._python_home_inventory()
    home = tmp_path / "python_home"
    for relative, record in inventory["files"].items():
        source = Path(record["source_identity"]["resolved_path"])
        target = home / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    probe = (
        "import ctypes,json; from ctypes import wintypes; "
        "k=ctypes.WinDLL('kernel32'); p=ctypes.WinDLL('psapi'); "
        "k.GetCurrentProcess.restype=wintypes.HANDLE; "
        "mods=(wintypes.HMODULE*512)(); n=wintypes.DWORD(); "
        "p.EnumProcessModulesEx(k.GetCurrentProcess(),mods,ctypes.sizeof(mods),ctypes.byref(n),3); "
        "out=[]; "
        "exec(\"for m in mods[:n.value//ctypes.sizeof(wintypes.HMODULE)]:\\n b=ctypes.create_unicode_buffer(32768)\\n l=p.GetModuleFileNameExW(k.GetCurrentProcess(),m,b,len(b))\\n out.append(b.value) if l else None\"); "
        "print(json.dumps(out))"
    )
    result = subprocess.run(
        [str(home / "python.exe"), "-I", "-S", "-B", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=home,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("PYTHON")
        },
    )
    paths = [Path(value).resolve() for value in json.loads(result.stdout)]
    vc = [path for path in paths if path.name.casefold() == "vcruntime140.dll"]
    assert vc == [(home / "vcruntime140.dll").resolve()]
    assert vc[0].stat().st_size == 120400
    assert hashlib.sha256(vc[0].read_bytes()).hexdigest() == (
        "052ad6a20d375957e82aa6a3c441ea548d89be0981516ca7eb306e063d5027f4"
    )
