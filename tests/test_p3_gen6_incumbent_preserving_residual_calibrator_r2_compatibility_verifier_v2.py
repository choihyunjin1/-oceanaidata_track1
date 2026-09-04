from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = (
    ROOT / "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_v2.py"
)
BOOT = runpy.run_path(os.fspath(BOOTSTRAP_PATH), run_name="p3_compatibility_v2_test_bootstrap")
PYTHON = ROOT / ".venv-p1/Scripts/python.exe"


def _pin(path: Path, relative: str) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _canonical_environment() -> dict[str, str]:
    data_dir = os.environ.get("P3_DATA_DIR")
    if not data_dir:
        pytest.skip("P3_DATA_DIR is required for the sealed read-only compatibility replay")
    return {
        **os.environ,
        "P3_WORKSPACE_ROOT": os.fspath(ROOT),
        "P3_DATA_DIR": data_dir,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _frozen_snapshot() -> dict[str, object]:
    pins = BOOT["PINNED_SOURCES"]
    pinned = {role: _pin(ROOT / pin["path"], pin["path"]) for role, pin in pins.items()}
    v1_config = json.loads((ROOT / pins["V1_CONFIG"]["path"]).read_text(encoding="utf-8"))
    extra = {
        role: _pin(ROOT / pin["path"], pin["path"])
        for group in ("r2_control_pins", "r2_output_pins")
        for role, pin in v1_config[group].items()
    }
    ledger = v1_config["v9_anchor"]
    return {
        "authenticated_sources": pinned,
        "frozen_control_and_output": extra,
        "v9": _pin(ROOT / ledger["path"], ledger["path"]),
    }


def test_noncyclic_bootstrap_machine_enforces_every_subordinate_pin() -> None:
    assert BOOT["IMPLEMENTATION_ROLES"] == {
        "TRUSTED_BOOTSTRAP": (
            "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_"
            "r2_compatibility_v2.py"
        ),
        "CONFIG": (
            "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_"
            "v1r2_compatibility_verifier_v2.json"
        ),
        "HELPER": (
            "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_"
            "r2_compatibility_verifier_v2.py"
        ),
        "CLI": (
            "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v2.py"
        ),
        "TESTS": (
            "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_"
            "r2_compatibility_verifier_v2.py"
        ),
    }
    pins = BOOT["PINNED_SOURCES"]
    assert set(pins) == {
        "CONFIG",
        "HELPER",
        "CLI",
        "TESTS",
        "R2_CONTRACT",
        "V1_CONFIG",
        "V1_HELPER",
        "V1_CLI",
        "V1_TESTS",
        "V1_OWNER_NO_GO",
        "V1_TOMBSTONE",
    }
    buffers, identities = BOOT["_authenticate_all"](ROOT)
    assert set(buffers) == set(pins) == set(identities)
    for role, pin in pins.items():
        assert len(buffers[role]) == pin["bytes"]
        assert hashlib.sha256(buffers[role]).hexdigest() == pin["sha256"]


@pytest.mark.parametrize("role", ["CONFIG", "HELPER", "CLI", "TESTS"])
def test_each_new_implementation_byte_drift_fails_before_execution(role: str) -> None:
    forged = copy.deepcopy(BOOT["PINNED_SOURCES"])
    forged[role]["sha256"] = "0" * 64
    with pytest.raises(BOOT["BootstrapTrustError"], match=f"pin drift: {role}"):
        BOOT["_authenticate_all"](ROOT, forged)


def test_post_read_file_mutation_is_detected(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"VALUE = 1\n")
    pins = {"SOURCE": _pin(source, "source.py")}
    buffers, identities = BOOT["_authenticate_all"](tmp_path, pins)
    source.write_bytes(b"VALUE = 2\n")
    with pytest.raises(BOOT["BootstrapTrustError"], match="pin drift|mutated"):
        BOOT["_reverify_all"](tmp_path, pins, buffers, identities)


def test_authenticated_buffer_exec_ignores_pyc_decoy(tmp_path: Path) -> None:
    source = tmp_path / "payload.py"
    raw = b"VALUE = 'authenticated-source'\n"
    source.write_bytes(raw)
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "payload.cpython-312.pyc").write_bytes(b"malicious-pyc-decoy")
    name = "p3_compatibility_v2_pyc_decoy_test"
    try:
        module = BOOT["_exec_authenticated_buffer"](
            module_name=name,
            source_path=source,
            raw=raw,
        )
        assert module.VALUE == "authenticated-source"
        assert module.__loader__ is None
        assert module.__spec__ is None
        assert sys.dont_write_bytecode is True
    finally:
        sys.modules.pop(name, None)


def test_direct_helper_and_cli_execution_fail_closed_before_numerical_import() -> None:
    for relative in (
        BOOT["IMPLEMENTATION_ROLES"]["HELPER"],
        BOOT["IMPLEMENTATION_ROLES"]["CLI"],
    ):
        completed = subprocess.run(
            [os.fspath(PYTHON), "-B", os.fspath(ROOT / relative)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert completed.returncode != 0
        assert "trusted bootstrap" in completed.stderr
        assert completed.stdout == ""


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("import numpy", "numerical modules were imported before authentication"),
        (
            (
                "import sys,types;"
                "sys.modules['p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2']="
                "types.ModuleType('p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2')"
            ),
            "protected modules were imported before authentication",
        ),
    ],
)
def test_preimport_numerical_or_engine_state_fails_closed(setup: str, expected: str) -> None:
    code = (
        f"{setup};import runpy,sys;"
        f"sys.argv=['bootstrap','--root',{os.fspath(ROOT)!r}];"
        f"runpy.run_path({os.fspath(BOOTSTRAP_PATH)!r},run_name='__main__')"
    )
    completed = subprocess.run(
        [os.fspath(PYTHON), "-B", "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode != 0
    assert expected in completed.stderr


def test_full_ancestor_symlink_or_reparse_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    linked = tmp_path / "linked"
    try:
        os.symlink(target, linked, target_is_directory=True)
    except OSError:
        pytest.skip("local Windows policy does not permit symlink creation")
    with pytest.raises(BOOT["BootstrapTrustError"], match="link/reparse"):
        BOOT["_checked_path"](linked, "source.py", kind="file")


def test_mocked_intermediate_reparse_is_rejected_without_platform_privilege(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ancestor = tmp_path / "ordinary_ancestor"
    ancestor.mkdir()
    (ancestor / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    original = BOOT["_has_link_or_reparse"]

    def classify(path: Path) -> bool:
        return path == ancestor or original(path)

    monkeypatch.setitem(BOOT["_checked_path"].__globals__, "_has_link_or_reparse", classify)
    with pytest.raises(BOOT["BootstrapTrustError"], match="link/reparse"):
        BOOT["_checked_path"](tmp_path, "ordinary_ancestor/source.py", kind="file")


def test_v2_restoration_guard_repairs_and_rejects_adversarial_global_drift() -> None:
    token = object()
    original_control = object()
    r2 = types.SimpleNamespace(
        PREFIX_FRACTIONS=(0.2, 0.4, 0.6, 0.8, 1.0),
        _control_inventory=original_control,
    )
    v1 = types.SimpleNamespace()
    context = {"token": token, "r2_module": r2, "v1_module": v1}
    namespace = {
        "__name__": "p3_compatibility_v2_restoration_test",
        "__file__": os.fspath(ROOT / BOOT["IMPLEMENTATION_ROLES"]["HELPER"]),
        "__trusted_bootstrap_context__": context,
        "__trusted_bootstrap_token__": token,
        "__trusted_v1_module__": v1,
        "__trusted_r2_module__": r2,
    }
    raw = (ROOT / BOOT["IMPLEMENTATION_ROLES"]["HELPER"]).read_bytes()
    exec(compile(raw, namespace["__file__"], "exec", dont_inherit=True), namespace)
    original_prefixes = r2.PREFIX_FRACTIONS
    with pytest.raises(namespace["CompatibilityV2Error"], match="failed to restore"):
        with namespace["_restoration_guard"]():
            r2.PREFIX_FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
            r2._control_inventory = object()
            raise LookupError("forced after both scoped adaptations")
    assert r2.PREFIX_FRACTIONS is original_prefixes
    assert r2._control_inventory is original_control


def test_config_is_one_authenticated_buffer_and_deep_matches_trust_map() -> None:
    pins = BOOT["PINNED_SOURCES"]
    raw = (ROOT / pins["CONFIG"]["path"]).read_bytes()
    config = json.loads(raw.decode("utf-8"))
    assert len(raw) == pins["CONFIG"]["bytes"]
    assert hashlib.sha256(raw).hexdigest() == pins["CONFIG"]["sha256"]
    assert config["authenticated_source_pins"] == {
        role: pin for role, pin in pins.items() if role != "CONFIG"
    }
    assert config["trusted_bootstrap"]["self_hash_embedded"] is False
    assert config["compatibility_contract"]["adaptation_count"] == 2


def test_canonical_bootstrap_replays_full_verifier_and_preserves_frozen_state() -> None:
    before = _frozen_snapshot()
    completed = subprocess.run(
        [
            os.fspath(PYTHON),
            "-B",
            os.fspath(BOOTSTRAP_PATH),
            "--root",
            os.fspath(ROOT),
            "--mode",
            "check-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_canonical_environment(),
    )
    result = json.loads(completed.stdout)
    assert result["status"] == ("PASS_AUTHENTICATED_R2_COMPATIBILITY_RESEARCH_ONLY_NO_PROMOTION")
    trust = result["trusted_bootstrap"]
    assert trust["authenticated_all_sources_before_module_execution"] is True
    assert trust["authenticated_buffer_compile_exec_only"] is True
    assert trust["config_parse_count"] == 1
    assert trust["post_read_identity_reverified"] is True
    assert trust["final_file_identity_verified"] is True
    assert trust["r2_engine_imported"] is False
    assert trust["reverification_phases"] == [
        "pre_r2_contract_exec",
        "pre_v1_helper_exec",
        "pre_v2_helper_exec",
        "pre_v2_cli_exec",
        "pre_cli_entry",
        "helper_post_config_parse",
        "helper_pre_legacy_verifier_entry",
        "helper_post_legacy_verifier",
        "post_cli_entry",
        "bootstrap_finally",
    ]
    legacy = result["legacy_compatibility_verification"]
    assert legacy["independent_metric_verification"]["bootstrap_replicates_total"] == 25000
    assert legacy["independent_metric_verification"]["gate"]["decision"] == "RESEARCH_ONLY"
    assert legacy["independent_metric_verification"]["gate"]["passed"] is False
    assert result["compatibility_adaptations"]["count"] == 2
    assert result["compatibility_adaptations"]["globals_restored"] is True
    assert _frozen_snapshot() == before


def test_v2_control_qa_and_receipts_remain_absent() -> None:
    control = (
        ROOT / "artifacts/p3_gen6_incumbent_preserving_residual_calibrator_20260823_"
        "v1r2_compatibility_verifier_v2_control"
    )
    assert not control.exists()


def test_static_sources_have_no_write_loader_engine_or_personal_path_surface() -> None:
    paths = [
        ROOT / BOOT["IMPLEMENTATION_ROLES"][role]
        for role in ("TRUSTED_BOOTSTRAP", "CONFIG", "HELPER", "CLI")
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in (
        "importlib.util",
        "spec_from_file_location",
        "SourceFileLoader",
        "robust_write_exclusive(",
        "write_output_exclusive(",
        "write_failure_receipt(",
        "create_attempt_lock(",
        "execute_gen6_curve(",
        "gen6_incumbent_preserving_residual_calibrator_execution_r2 import",
    ):
        assert forbidden not in text
    assert "compile(raw" in text
    assert "exec(code" in text
    assert "sys.dont_write_bytecode = True" in text
    assert "assert " not in "\n".join(
        path.read_text(encoding="utf-8") for path in paths if path.suffix == ".py"
    )
