from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / (
    "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_v3.py"
)
CONFIG = ROOT / (
    "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_"
    "v1r2_compatibility_verifier_v3.json"
)
HELPER = ROOT / (
    "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_verifier_v3.py"
)
CLI = ROOT / (
    "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_"
    "r2_compatibility_v3.py"
)
V2 = {
    "BOOTSTRAP": (
        ROOT
        / "scripts/bootstrap_verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v2.py",
        22071,
        "d2dec0e2d05d53da3d0489f8af9762d7e57524326143b1e9afa91d2a47537733",
    ),
    "CONFIG": (
        ROOT
        / "configs/experiments/p3_gen6_incumbent_preserving_residual_calibrator_v1r2_compatibility_verifier_v2.json",
        11074,
        "a80aedd91cc1ed73d638fcaa2827f73344220d49b3f2c1073458e7040c044cc1",
    ),
    "HELPER": (
        ROOT
        / "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v2.py",
        15462,
        "054c271e2ba0d8aac8fc8f4436884b491ab7c03a397fd10aae0e0478ddcd681b",
    ),
    "CLI": (
        ROOT
        / "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v2.py",
        1417,
        "39faf8b6f6d6a1acb043ed038cce396a85a4265be04678e8a9cdbc134980df13",
    ),
    "TESTS": (
        ROOT
        / "tests/test_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v2.py",
        12901,
        "a64f7e865422d2ace395e17a3717cca1ebec9fa35a57c06c477f0983f23abb67",
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_bytes())


def _canonical_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "P3_WORKSPACE_ROOT": str(ROOT),
            "P3_DATA_DIR": str(Path.home() / "Downloads/p3/데이터셋_P3/P3_wave_forecast"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def test_v2_bytes_are_preserved_exactly() -> None:
    for path, size, digest in V2.values():
        assert path.stat().st_size == size
        assert _sha(path) == digest


def test_v3_subordinate_implementation_pins_are_exact() -> None:
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


def test_adversarial_single_byte_drift_changes_every_pinned_identity() -> None:
    pins = _config()["authenticated_subordinate_pins"]
    for role, pin in pins.items():
        raw = bytearray((ROOT / pin["path"]).read_bytes())
        raw[len(raw) // 2] ^= 1
        assert len(raw) == pin["bytes"]
        assert hashlib.sha256(raw).hexdigest() != pin["sha256"], role


def test_v2_no_go_has_no_fabricated_independent_receipt() -> None:
    config = _config()
    disposition = config["v2_disposition"]
    assert disposition["reviewer"] == "/root/meaningful_improvement_audit/p2_stageb_blind_review"
    assert disposition["independent_qa_receipt_file_exists"] is False
    assert disposition["independent_qa_receipt_hash_exists"] is False
    assert disposition["finding_ids"] == [
        "PREIMPORT_AND_DEPENDENCY_TRUST_BOUNDARY_NOT_CLOSED",
        "PINNED_ARTIFACT_BYTES_NOT_BOUND_TO_SEMANTIC_REPLAY",
    ]


def test_v3_control_and_receipts_are_absent() -> None:
    for relative in _config()["canonical_paths"].values():
        assert not (ROOT / relative).exists()
        assert not (ROOT / relative).is_symlink()


def test_bootstrap_installs_audit_hook_before_non_sys_import() -> None:
    tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    imports: list[tuple[int, str]] = []
    hook_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sys"
            and node.func.attr == "addaudithook"
        ):
            hook_line = node.lineno
    assert hook_line is not None
    assert all(name == "sys" for line, name in imports if line < hook_line)
    assert any(name == "_winapi" and line > hook_line for line, name in imports)


def test_runtime_is_exact_isolated_no_site_no_bytecode() -> None:
    runtime = _config()["canonical_runtime_contract"]
    assert runtime["required_cli_flags"] == ["-I", "-S", "-B"]
    assert runtime["required_sys_flags"] == {
        "isolated": 1,
        "no_site": 1,
        "safe_path": True,
        "ignore_environment": 1,
        "no_user_site": 1,
        "dont_write_bytecode": 1,
    }
    assert runtime["initial_sys_path_roles"] == ["python312.zip", "DLLs", "Lib", "BASE"]
    assert runtime["initial_meta_path"] == [
        "_frozen_importlib.BuiltinImporter",
        "_frozen_importlib.FrozenImporter",
        "_frozen_importlib_external.PathFinder",
    ]


def test_dependency_record_and_native_inventories_are_fully_pinned() -> None:
    distributions = _config()["canonical_runtime_contract"]["third_party_distributions"]
    assert set(distributions) == {
        "numpy",
        "pandas",
        "pyarrow",
        "python_dateutil",
        "six",
        "tzdata",
    }
    assert distributions["numpy"]["files"] == 1332
    assert distributions["pandas"]["files"] == 2989
    assert distributions["pyarrow"]["native_files"] == 33
    assert all(len(item["record_sha256"]) == 64 for item in distributions.values())


def test_semantic_parsers_are_buffer_only() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "io.BytesIO(held_bytes(file))" in source
    assert "pa.BufferReader(held_bytes(source))" in source
    assert '"protected_reopen"' in source
    assert "FILE_SHARE_READ" in source
    stable = _config()["stable_semantic_read_contract"]
    assert stable["protected_path_reopen_allowed"] is False
    assert stable["held_windows_handles_until_final_verification"] is True


def test_only_two_science_compatibility_adaptations_remain() -> None:
    contract = _config()["compatibility_contract"]
    assert contract["adaptation_count"] == 2
    assert contract["only_scoped_science_adaptations"] == [
        "replace_frozen_r2_verifier_prefix_expectation_with_four_source_consensus",
        "admit_exact_pinned_historical_failure_receipt_to_the_r2_control_inventory",
    ]
    assert contract["independent_oof_bootstrap_gate_replay"] is True
    assert contract["bootstrap_replicates_per_point"] == 5000
    assert contract["bootstrap_points"] == 5


def test_global_adapters_have_unconditional_restore_guard() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "finally:" in source
    assert "restore()" in source
    assert "semantic adapters or legacy globals were not restored" in source
    assert "r2_module.PREFIX_FRACTIONS is not original_prefixes" in source
    assert "r2_module._control_inventory is not original_inventory" in source


def test_engine_import_and_process_launch_are_denied() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "r2 execution engine import is forbidden" in source
    assert '"_winapi.CreateProcess"' in source
    assert '"subprocess.Popen"' in source
    assert '"process_launch"' in source


@pytest.mark.parametrize(
    ("event_expression", "message"),
    [
        ("sys.audit('open', 'poison.pyc', 'r', 0)", "bytecode access is forbidden"),
        (
            "sys.audit('_winapi.CreateFile', 'poison', 0x40000000, 0, 3, 0)",
            "Windows write handle is forbidden",
        ),
        (
            "sys.audit('_winapi.CreateProcess', 'poison', (), None)",
            "process launch is forbidden",
        ),
        (
            "sys.audit('import', "
            "'p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2', "
            "None)",
            "r2 execution engine import is forbidden",
        ),
    ],
)
def test_adversarial_audit_events_fail_closed(
    event_expression: str, message: str
) -> None:
    probe = (
        "import sys; p=sys.argv[1]; "
        "ns={'__name__':'p3_v3_audit_probe','__file__':p}; "
        "exec(compile(open(p,'rb').read(),p,'exec'),ns); "
        + event_expression
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", probe, str(BOOTSTRAP)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert message in completed.stderr


def test_bytecode_is_denied_and_sources_compile_from_buffers() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'lowered.endswith((".pyc", ".pyo"))' in source
    assert "compile(self.raw, self.path" in source
    assert "compile(raw, path" in source
    assert "sys.dont_write_bytecode = True" in source


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
    assert "requires the trusted v3 bootstrap" in completed.stderr


def test_missing_isolated_flags_fail_before_semantic_replay() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(BOOTSTRAP), "--root", str(ROOT)],
        cwd=ROOT,
        env=_canonical_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "-I -S -B" in completed.stderr


def test_alternate_config_path_fails_before_any_semantic_replay(tmp_path: Path) -> None:
    alternate = tmp_path / "config.json"
    alternate.write_bytes(CONFIG.read_bytes())
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(BOOTSTRAP),
            "--root",
            str(ROOT),
            "--config",
            str(alternate),
            "--mode",
            "check-only",
        ],
        cwd=ROOT,
        env=_canonical_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "alternate compatibility-v3 config is forbidden" in completed.stderr


def test_canonical_check_only_replay_has_zero_attempts_and_same_handle_rehashes() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(BOOTSTRAP),
            "--root",
            str(ROOT),
            "--mode",
            "check-only",
        ],
        cwd=ROOT,
        env=_canonical_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == (
        "PASS_HELD_BUFFER_R2_COMPATIBILITY_RESEARCH_ONLY_NO_PROMOTION"
    )
    assert not any(result["trusted_runtime"]["audit_attempts"].values())
    registry = result["stable_semantic_registry"]
    assert registry["protected_reopens"] == 0
    assert registry["same_handle_final_rehashes"] == registry["files"]
    assert registry["final_identity_and_hash_verified"] is True
    assert result["compatibility_adaptations"]["count"] == 2
    assert result["legacy_compatibility_verification"]["status"] == (
        "PASS_AUTHENTICATED_R2_COMPATIBILITY_RESEARCH_ONLY_NO_PROMOTION"
    )


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
