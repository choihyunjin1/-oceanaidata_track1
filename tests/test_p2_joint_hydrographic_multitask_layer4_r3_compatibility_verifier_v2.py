from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = (
    ROOT / "scripts" / "bootstrap_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v2.py"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v2.json"
)
ANCHOR = (
    ROOT
    / "configs"
    / "experiments"
    / "p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v2_trust_anchor.json"
)
PYTHON = ROOT / ".venv-p1" / "Scripts" / "python.exe"
V1_PINS = {
    "CONFIG": {
        "path": "configs/experiments/p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v1.json",
        "bytes": 8891,
        "sha256": "b1e30c04801bec2a575ed1cefbf6afd913da17147415176bc432f086b9e87491",
    },
    "HELPER": {
        "path": "src/p2_restore/joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v1.py",
        "bytes": 22145,
        "sha256": "0b58ede0dcb2a3b56cf4272bf62e4da606ce4527f069c2ce876180cce9253050",
    },
    "CLI": {
        "path": "scripts/verify_p2_joint_hydrographic_multitask_layer4_r3_compatibility_v1.py",
        "bytes": 2533,
        "sha256": "ffa80b1ab06fba1fb0082fa1d12352b4212a282c1326518200ad955f6cb6a6df",
    },
    "TESTS": {
        "path": "tests/test_p2_joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v1.py",
        "bytes": 17094,
        "sha256": "098d877833e2f10acfc7288d5a8ee96ddc83ba0c3581af92240b1e4b8d68e6bf",
    },
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _anchor() -> dict[str, object]:
    return json.loads(ANCHOR.read_text(encoding="utf-8"))


def _bootstrap_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "__name__": "p2_v2_bootstrap_test_namespace",
        "__file__": str(BOOTSTRAP),
    }
    raw = BOOTSTRAP.read_bytes()
    exec(compile(raw, str(BOOTSTRAP), "exec"), namespace)  # noqa: S102
    return namespace


def _canonical_command(*extra: str) -> list[str]:
    return [
        str(PYTHON),
        str(BOOTSTRAP),
        "--root",
        str(ROOT),
        "--mode",
        "check-only",
        *extra,
    ]


def _protected_snapshot() -> dict[str, object]:
    config = _config()
    v1 = json.loads((ROOT / config["v1_contract_pin"]["path"]).read_text(encoding="utf-8"))
    return {
        "v1": {role: _pin(ROOT / pin["path"]) for role, pin in V1_PINS.items()},
        "owner": {
            role: _pin(ROOT / pin["path"]) for role, pin in config["v1_disposition_pins"].items()
        },
        "r3_implementation": {
            role: _pin(ROOT / pin["path"]) for role, pin in v1["r3_implementation_pins"].items()
        },
        "r3_controls": {
            role: _pin(ROOT / pin["path"]) for role, pin in v1["r3_control_pins"].items()
        },
        "r3_core": {
            role: _pin(ROOT / pin["path"]) for role, pin in v1["r3_core_artifact_pins"].items()
        },
        "v9": _pin(ROOT / v1["v9_anchor"]["path"]),
        "v2_control_exists": (ROOT / config["canonical_paths"]["v2_control"]).exists(),
    }


def test_v1_four_files_are_byte_exact_and_owner_no_go_is_append_only() -> None:
    config = _config()
    assert config["v1_implementation_pins"] == V1_PINS
    for pin in V1_PINS.values():
        assert _pin(ROOT / pin["path"]) == pin
    owner = json.loads(
        (ROOT / config["v1_disposition_pins"]["OWNER_NO_GO"]["path"]).read_text(encoding="utf-8")
    )
    tombstone = json.loads(
        (ROOT / config["v1_disposition_pins"]["EXECUTION_TOMBSTONE"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert owner["verdict"] == "P0=0_P1=3_NO_GO"
    assert [finding["code"] for finding in owner["findings"]] == config["v1_independent_qa"][
        "required_finding_codes"
    ]
    assert tombstone["owner_no_go_receipt"] == config["v1_disposition_pins"]["OWNER_NO_GO"]
    assert tombstone["execution_prohibited"] is True


def test_v2_config_is_strictly_read_only_and_has_noncyclic_trust_contract() -> None:
    config = _config()
    assert config["identity"] == (
        "P2_JOINT_HYDROGRAPHIC_MULTITASK_LAYER4_R3_COMPATIBILITY_VERIFIER_V2"
    )
    assert config["verifier_only"] is config["check_only_default"] is True
    assert config["append_only_successor_of_v1"] is True
    for key in (
        "r3_mutation_allowed",
        "r3_rerun_or_resume_allowed",
        "execution_authorization_or_lock_allowed",
        "fit_prediction_truth_decode_or_scoring_allowed",
        "compatibility_receipt_write_allowed",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "registry_append_allowed",
        "upload_allowed",
    ):
        assert config[key] is False
    assert all(value == 0 for value in config["static_counters"].values())
    trust = config["trusted_bootstrap_contract"]
    assert trust["external_fresh_qa_must_pin_bootstrap"] is True
    assert trust["trust_anchor_exact_roles"] == ["CONFIG", "HELPER", "CLI", "TESTS"]
    assert trust["source_file_loader_or_pyc_execution_allowed"] is False


def test_trust_anchor_exactly_pins_config_helper_cli_and_tests() -> None:
    anchor = _anchor()
    config = _config()
    trusted = anchor["trusted_files"]
    assert set(trusted) == {"CONFIG", "HELPER", "CLI", "TESTS"}
    for role, pin in trusted.items():
        assert pin["path"] == config["implementation_roles"][role]
        assert _pin(ROOT / pin["path"]) == pin
    namespace = _bootstrap_namespace()
    assert namespace["ANCHOR_BYTES"] == ANCHOR.stat().st_size
    assert namespace["ANCHOR_SHA256"] == _sha(ANCHOR)


@pytest.mark.parametrize("role", ["CONFIG", "HELPER", "CLI", "TESTS"])
def test_bootstrap_rejects_each_trusted_role_drift_before_execution(
    role: str, tmp_path: Path
) -> None:
    namespace = _bootstrap_namespace()
    anchor = copy.deepcopy(_anchor())
    for pin in anchor["trusted_files"].values():
        target = tmp_path / pin["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / pin["path"]).read_bytes())
    observed, _buffers = namespace["verify_trusted_files"](tmp_path, anchor)
    assert observed == anchor["trusted_files"]
    target = tmp_path / anchor["trusted_files"][role]["path"]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(namespace["BootstrapV2Error"], match="authenticated buffer"):
        namespace["verify_trusted_files"](tmp_path, anchor)


def test_drifted_helper_payload_cannot_execute(tmp_path: Path) -> None:
    namespace = _bootstrap_namespace()
    anchor = copy.deepcopy(_anchor())
    marker = tmp_path / "forbidden_marker.txt"
    for pin in anchor["trusted_files"].values():
        target = tmp_path / pin["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / pin["path"]).read_bytes())
    helper = tmp_path / anchor["trusted_files"]["HELPER"]["path"]
    helper.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n")
    with pytest.raises(namespace["BootstrapV2Error"], match="authenticated buffer"):
        namespace["verify_trusted_files"](tmp_path, anchor)
    assert not marker.exists()


def test_single_buffer_json_parser_rejects_duplicate_and_nonfinite_values() -> None:
    namespace = _bootstrap_namespace()
    with pytest.raises(namespace["BootstrapV2Error"], match="duplicate JSON key"):
        namespace["parse_json_buffer"](b'{"a":1,"a":2}', label="duplicate")
    with pytest.raises(namespace["BootstrapV2Error"], match="non-finite"):
        namespace["parse_json_buffer"](b'{"a":NaN}', label="nonfinite")


def test_after_read_identity_recheck_is_mandatory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _bootstrap_namespace()
    target = tmp_path / "payload.bin"
    target.write_bytes(b"authenticated")
    pin = {
        "path": "payload.bin",
        "bytes": target.stat().st_size,
        "sha256": _sha(target),
    }
    monkeypatch.setitem(
        namespace,
        "_current_pin",
        lambda _root, _relative: {**pin, "sha256": "0" * 64},
    )
    with pytest.raises(namespace["BootstrapV2Error"], match="after-read"):
        namespace["authenticated_bytes"](tmp_path, pin, label="identity race")


def test_full_ancestor_chain_rejects_linklike_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _bootstrap_namespace()
    target = tmp_path / "plain" / "payload.bin"
    target.parent.mkdir()
    target.write_bytes(b"x")
    original = Path.is_symlink

    def classify(path: Path) -> bool:
        return path == target.parent or original(path)

    monkeypatch.setattr(Path, "is_symlink", classify)
    with pytest.raises(namespace["BootstrapV2Error"], match="link/reparse ancestor"):
        namespace["contained_path"](tmp_path, "plain/payload.bin", kind="file")


def test_real_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    namespace = _bootstrap_namespace()
    target = tmp_path / "target.bin"
    target.write_bytes(b"x")
    link = tmp_path / "link.bin"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("local Windows policy does not permit test symlinks")
    with pytest.raises(namespace["BootstrapV2Error"], match="link/reparse ancestor"):
        namespace["contained_path"](tmp_path, "link.bin", kind="file")


def test_write_audit_firewall_blocks_mutation_in_a_fresh_process(tmp_path: Path) -> None:
    marker = tmp_path / "blocked.txt"
    code = f"""
import pathlib
namespace={{'__name__':'firewall_test','__file__':{str(BOOTSTRAP)!r}}}
raw=pathlib.Path({str(BOOTSTRAP)!r}).read_bytes()
exec(compile(raw,{str(BOOTSTRAP)!r},'exec'),namespace)
namespace['install_firewall']()
try:
    pathlib.Path({str(marker)!r}).write_text('forbidden')
except PermissionError:
    print('BLOCKED')
print(namespace['_FIREWALL_STATE']['mutation_attempts'])
"""
    completed = subprocess.run(
        [str(PYTHON), "-B", "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["BLOCKED", "1"]
    assert not marker.exists()


@pytest.mark.parametrize("forbidden", ["numpy", "r3_engine"])
def test_forbidden_preloaded_module_fails_before_verification(forbidden: str) -> None:
    preload = (
        "import numpy\n"
        if forbidden == "numpy"
        else (
            "import sys,types\n"
            "sys.modules['p2_restore.joint_hydrographic_multitask_layer4_execution_r3']="
            "types.ModuleType('p2_restore.joint_hydrographic_multitask_layer4_execution_r3')\n"
        )
    )
    code = f"""
{preload}
import pathlib
namespace={{'__name__':'forbidden_test','__file__':{str(BOOTSTRAP)!r}}}
raw=pathlib.Path({str(BOOTSTRAP)!r}).read_bytes()
exec(compile(raw,{str(BOOTSTRAP)!r},'exec'),namespace)
try:
    namespace['install_firewall']()
except namespace['BootstrapV2Error']:
    print('FAIL_CLOSED')
"""
    completed = subprocess.run(
        [str(PYTHON), "-B", "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "FAIL_CLOSED"


def test_helper_and_cli_reject_direct_execution() -> None:
    config = _config()
    for role in ("HELPER", "CLI"):
        completed = subprocess.run(
            [str(PYTHON), "-B", str(ROOT / config["implementation_roles"][role])],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert "requires the authenticated bootstrap" in completed.stderr


def test_canonical_default_check_only_passes_and_preserves_protected_state() -> None:
    before = _protected_snapshot()
    environment = dict(os.environ)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    completed = subprocess.run(
        _canonical_command(),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == ("PASS_AUTHENTICATED_R3_COMPATIBILITY_RESEARCH_ONLY_LOCAL_FAIL")
    assert payload["trusted_implementation_pins"] == _anchor()["trusted_files"]
    assert payload["original_r3_verifier_failure"]["message"] == (
        "Layer-4 seal verification failed: ['receipt_fold_audits']"
    )
    assert payload["corrected_fold_audit_verification"]["verified"] == [
        {"fold": "outer_2024_sep_oct", "verified_prior_fold_commitments": 0},
        {"fold": "outer_2025_may_jun", "verified_prior_fold_commitments": 1},
        {"fold": "outer_2025_jul_aug", "verified_prior_fold_commitments": 2},
    ]
    assert all(payload["result_checks"].values())
    assert (
        payload["authenticated_bootstrap"]["all_trusted_roles_authenticated_before_execution"]
        is True
    )
    assert payload["authenticated_bootstrap"]["source_file_loader_or_pyc_used"] is False
    assert payload["authenticated_bootstrap"]["write_audit_attempts"] == 0
    assert payload["authenticated_bootstrap"]["forbidden_modules"] == []
    assert payload["authenticated_cli"]["r3_engine_imported"] is False
    assert payload["sys_dont_write_bytecode"] is True
    assert payload["v2_control_exists"] is False
    for key in (
        "files_written",
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
    ):
        assert payload[key] == 0
    assert _protected_snapshot() == before


def test_optimized_mode_keeps_authenticated_checks() -> None:
    command = _canonical_command()
    command.insert(1, "-O")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["files_written"] == 0
    assert payload["authenticated_bootstrap"]["forbidden_modules"] == []


def test_source_has_no_loader_pyc_write_execution_or_personal_path_bypass() -> None:
    config = _config()
    sources = [
        ROOT / config["implementation_roles"][role]
        for role in ("BOOTSTRAP", "CONFIG", "HELPER", "CLI")
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "SourceFileLoader" not in text
    assert "importlib" not in text
    assert "sys.dont_write_bytecode = True" in BOOTSTRAP.read_text(encoding="utf-8")
    assert not re.search(
        r"(?i)(?:[A-Z]:[\\/]Users[\\/][A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+)",
        text,
    )
    helper = (ROOT / config["implementation_roles"]["HELPER"]).read_text(encoding="utf-8")
    assert "compile(source" in helper
    assert "exec(code, module.__dict__)" in helper
    assert "reverify_trusted_files" in helper
    for forbidden in (
        "execute_layer4_curve(",
        "_load_metric_truth_after_commitment(",
        "exclusive_json(",
        "exclusive_bytes(",
    ):
        assert forbidden not in helper


def test_v2_control_qa_receipt_authorization_and_lock_are_absent() -> None:
    config = _config()
    control = ROOT / config["canonical_paths"]["v2_control"]
    assert not control.exists()
    for key in ("pre_execution_qa", "compatibility_receipt"):
        assert not (ROOT / config["canonical_paths"][key]).exists()
    assert not (control / "authorization.json").exists()
    assert not (control / "attempt.lock").exists()
