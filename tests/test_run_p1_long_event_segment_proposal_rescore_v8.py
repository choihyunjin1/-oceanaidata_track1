from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_long_event_segment_proposal_rescore_v8.py"
MODULE = ROOT / "src/p1_qc/long_event_segment_proposal_rescore_execution_v8.py"
AMENDMENT = ROOT / (
    "configs/experiments/"
    "p1_long_event_segment_proposal_rescore_v8_isolated_python_parent_capability_amendment.json"
)
TEMPLATE = ROOT / (
    "configs/experiments/"
    "p1_long_event_segment_proposal_rescore_v8_execution_authorization_template.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def namespace_image(root: Path) -> dict[str, tuple[str, bytes]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            "dir",
            b"",
        )
        if path.is_dir()
        else ("file", path.read_bytes())
        for path in sorted(root.rglob("*"), key=lambda value: str(value).casefold())
    }


@pytest.fixture(scope="module")
def r8() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p1_segment_r8_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prospective_authority_and_frozen_science(r8: ModuleType) -> None:
    assert sha256(AMENDMENT) == r8.R8_AMENDMENT_SHA256
    assert r8.MAXIMUM_LIFETIME_PHYSICAL_FITS == 72
    assert r8.MAXIMUM_SCIENTIFIC_MATERIALIZATIONS == 21
    assert r8.MAXIMUM_OUTER_SCORES == 1
    assert r8.HARD_WALL_SECONDS == 21600
    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert value["created_before_r8_scientific_execution"] == {
        "actual_authorization_files": 0,
        "canonical_claims": 0,
        "physical_fits": 0,
        "scientific_materializations": 0,
        "outer_scores": 0,
        "candidate_files": 0,
        "official_actions": 0,
    }


def test_bootstrap_top_level_imports_are_stdlib_only(r8: ModuleType) -> None:
    imported = r8._verify_top_level_stdlib_only(RUNNER)
    assert {
        "numpy",
        "pandas",
        "lightgbm",
        "sklearn",
        "scipy",
        "pyarrow",
        "p1_qc",
    }.isdisjoint(imported)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":{"b":1,"b":2}}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
    ],
)
def test_exact_wire_rejects_duplicate_keys_and_nonfinite(
    r8: ModuleType, raw: bytes
) -> None:
    with pytest.raises(r8.WireFormatError):
        r8._json_from_bytes(raw, label="adversarial")


@pytest.mark.parametrize(
    "value",
    [
        "A" * 64,
        "a" * 63,
        "a" * 65,
        "g" * 64,
        " a" * 32,
        b"a" * 64,
        True,
    ],
)
def test_nonce_requires_exact_32_byte_lowercase_hex(
    r8: ModuleType, value: object
) -> None:
    with pytest.raises(r8.AuthorizationError):
        r8._validate_nonce(value)
    assert r8._validate_nonce("0" * 64) == "0" * 64


def test_reused_nonce_is_rejected(r8: ModuleType) -> None:
    seen: set[str] = set()
    r8._validate_nonce("1" * 64, seen)
    with pytest.raises(r8.AuthorizationError, match="reused"):
        r8._validate_nonce("1" * 64, seen)


def test_clock_contract_rejects_future_stale_expired_and_span_drift(
    r8: ModuleType,
) -> None:
    span = 21600 * 1_000_000_000
    now = 1_000_000_000_000
    mono = 2_000_000_000_000
    r8._validate_clock_fields(
        created_epoch_ns=now - 1,
        deadline_epoch_ns=now - 1 + span,
        created_monotonic_ns=mono - 1,
        deadline_monotonic_ns=mono - 1 + span,
        now_epoch_ns=now,
        now_monotonic_ns=mono,
    )
    cases = [
        (now + 1, now + 1 + span, mono, mono + span),
        (now - 301_000_000_000, now - 301_000_000_000 + span, mono, mono + span),
        (now - span, now, mono - span, mono),
        (now, now + span + 1, mono, mono + span),
        (now, now + span, mono, mono + span - 1),
        (True, True + span, mono, mono + span),
    ]
    for created, deadline, created_mono, deadline_mono in cases:
        with pytest.raises(r8.AuthorizationError):
            r8._validate_clock_fields(
                created_epoch_ns=created,
                deadline_epoch_ns=deadline,
                created_monotonic_ns=created_mono,
                deadline_monotonic_ns=deadline_mono,
                now_epoch_ns=now,
                now_monotonic_ns=mono,
            )


def test_execute_start_is_captured_before_delegate(
    r8: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, int] = {}

    def fake(epoch: int, monotonic: int) -> tuple[Path, dict[str, object]]:
        observed.update(epoch=epoch, monotonic=monotonic)
        raise RuntimeError("stop")

    before_epoch = time.time_ns()
    before_mono = time.monotonic_ns()
    monkeypatch.setattr(r8, "_execute_parent_started", fake)
    with pytest.raises(RuntimeError, match="stop"):
        r8.execute_parent()
    assert before_epoch <= observed["epoch"] <= time.time_ns()
    assert before_mono <= observed["monotonic"] <= time.monotonic_ns()


def test_isolated_python_home_inventory_exact_closure(r8: ModuleType) -> None:
    inventory = r8._python_home_inventory()
    assert inventory["file_count"] == 4044
    assert inventory["total_bytes"] == 81631030
    assert "Lib/site-packages" not in "\n".join(inventory["files"])
    assert inventory["files"]["vcruntime140.dll"] == {
        **inventory["files"]["vcruntime140.dll"],
        "bytes": 120400,
        "sha256": "052ad6a20d375957e82aa6a3c441ea548d89be0981516ca7eb306e063d5027f4",
    }
    assert inventory["files"]["vcruntime140_1.dll"]["bytes"] == 49776
    assert inventory["files"]["vcruntime140_1.dll"]["sha256"] == (
        "6a99bc0128e0c7d6cbbf615fcc26909565e17d4ca3451b97f8987f9c6acbc6c8"
    )


@pytest.fixture(scope="module")
def isolated_python_probe(r8: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p1-r8-python-home")
    inventory = r8._python_home_inventory()
    for relative, record in inventory["files"].items():
        source = Path(record["source_identity"]["resolved_path"])
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


def test_real_copied_python_I_S_B_bootstrap(
    r8: ModuleType, isolated_python_probe: Path, tmp_path: Path
) -> None:
    r7 = r8._load_r7()
    runtime = r7._full_runtime_inventory()
    tzroot = tmp_path / "runtime"
    for relative, record in runtime["files"].items():
        if record["owners"] != ["tzdata"]:
            continue
        source = Path(record["source_identity"]["resolved_path"])
        target = tzroot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    probe = (
        "import json,sys; from pathlib import Path; "
        "pre=list(sys.path); sys.path.insert(0,sys.argv[1]); "
        "from zoneinfo import ZoneInfo; "
        "print(json.dumps({'isolated':sys.flags.isolated,'no_site':sys.flags.no_site,"
        "'dont_write':sys.dont_write_bytecode,'exe':sys.executable,'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,'pre':pre,'zone':ZoneInfo('Asia/Seoul').key}))"
    )
    result = subprocess.run(
        [
            str(isolated_python_probe / "python.exe"),
            "-I",
            "-S",
            "-B",
            "-c",
            probe,
            str(tzroot),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("PYTHON")
        },
        cwd=isolated_python_probe,
    )
    payload = json.loads(result.stdout)
    assert payload["isolated"] == 1
    assert payload["no_site"] == 1
    assert payload["dont_write"] is True
    assert Path(payload["exe"]).resolve() == (isolated_python_probe / "python.exe").resolve()
    assert Path(payload["prefix"]).resolve() == isolated_python_probe.resolve()
    assert Path(payload["base_prefix"]).resolve() == isolated_python_probe.resolve()
    assert payload["zone"] == "Asia/Seoul"
    assert all("site-packages" not in value.lower() for value in payload["pre"])


def test_direct_and_copied_worker_fail_namespace_unchanged(
    r8: ModuleType, tmp_path: Path
) -> None:
    before = namespace_image(r8.CANONICAL_ARTIFACT_DIR)
    copied = tmp_path / RUNNER.name
    copied.write_bytes(RUNNER.read_bytes())
    for target in (RUNNER, copied):
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(target), "--worker"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
        )
        assert result.returncode != 0
        assert namespace_image(r8.CANONICAL_ARTIFACT_DIR) == before


def test_capability_is_exactly_pipe_only_bytes(r8: ModuleType) -> None:
    secret = bytes(range(32))
    payload = {
        "schema_version": "p1_segment_rescore.parent_pipe_capability.v8",
        "authorization_sha256": "1" * 64,
        "seal_sha256": "2" * 64,
        "qa_sha256": "3" * 64,
        "claim_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "launch_nonce": "6" * 64,
        "parent_identity": {"pid": 1},
    }
    frame = r8._capability_frame(secret, payload)
    assert frame[:32] == secret
    assert secret.hex().encode() not in frame[32:]
    assert secret not in r8._canonical_bytes(payload)


def test_true_parent_process_handle_identity(r8: ModuleType) -> None:
    if os.name != "nt":
        pytest.skip("Windows process-handle contract")
    handle = r8._open_self_process_handle()
    try:
        assert r8._process_identity_from_handle(handle) == r8._process_identity(os.getpid())
    finally:
        import ctypes

        ctypes.windll.kernel32.CloseHandle(handle)


def test_acl_freeze_denies_real_child_insertion(
    r8: ModuleType, tmp_path: Path
) -> None:
    if os.name != "nt":
        pytest.skip("Windows ACL contract")
    root = tmp_path / "frozen"
    root.mkdir()
    existing = root / "existing.dll"
    existing.write_bytes(b"sealed")
    with r8.DirectoryAclFreeze([root]):
        attacker = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                "from pathlib import Path; "
                f"Path({str(root / 'planted_dependency.dll')!r}).write_bytes(b'x')",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert attacker.returncode != 0
        assert not (root / "planted_dependency.dll").exists()
        assert existing.read_bytes() == b"sealed"
    (root / "released.dll").write_bytes(b"ok")


def test_claim_publication_is_create_only(r8: ModuleType, tmp_path: Path) -> None:
    claim = tmp_path / "canonical_launch.claim"
    r8._atomic_create_bytes(claim, b"partial")
    with pytest.raises(FileExistsError):
        r8._atomic_create_json(claim, {"second": True})
    assert claim.read_bytes() == b"partial"


def test_execution_wrapper_only_delegates_frozen_science() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_authorized_screen"
        for node in calls
    )
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "predict"}
        for node in calls
    )


def test_false_template_and_zero_operation_namespace(r8: ModuleType) -> None:
    assert TEMPLATE.exists()
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert template["authorized"] is False
    assert template["status"] == "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"
    assert template["runner"]["sha256"] == sha256(RUNNER)
    assert template["execution_module"]["sha256"] == sha256(MODULE)
    assert set(template["zero_prior_state"].values()) == {0}
    assert not r8.R8_AUTHORIZATION_PATH.exists()
    assert not r8.LAUNCH_CLAIM_PATH.exists()
    assert not (r8.CANONICAL_ARTIFACT_DIR / "execution.lock").exists()
    assert not (r8.CANONICAL_ARTIFACT_DIR / "result.json").exists()
