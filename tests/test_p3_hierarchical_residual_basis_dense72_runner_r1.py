from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

import p3_wave.hierarchical_residual_basis_dense72_contract_r1 as guard
import p3_wave.hierarchical_residual_basis_dense72_execution_r1 as engine
from p3_wave.dense72_targets_r1 import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / guard.CONFIG_RELATIVE


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_config_identity_workspace_and_full_transitive_roles_are_frozen() -> None:
    config = _config()
    assert sha256_file(CONFIG) == guard.CONFIG_SHA256
    assert config["comparison_mode"] == guard.COMPARISON_MODE
    assert config["exact_official_incumbent_comparison"] is False
    assert config["local_numeric_curve_qualification_allowed"] is True
    assert config["official_promotion_allowed"] is False
    assert set(config["implementation_roles"]) == guard.EXPECTED_IMPLEMENTATION_ROLES
    assert {
        "src/p3_wave/features.py",
        "src/p3_wave/episode_distinct_analog.py",
        "requirements-lock.txt",
    }.issubset(config["implementation_roles"].values())
    frozen = config["frozen_transitive_sha256"]
    expected_frozen_paths = {
        relative
        for role, relative in config["implementation_roles"].items()
        if role not in guard.QA_BOUND_DYNAMIC_ROLES
    }
    assert set(frozen) == expected_frozen_paths
    for relative, expected_sha256 in frozen.items():
        assert sha256_file(ROOT / relative) == expected_sha256
    runtime = guard.verify_runtime_environment(ROOT, config)
    assert runtime["exact_match"] is True
    assert runtime["requirements_lock_line_count"] == 83
    identity = config["canonical_workspace_identity"]
    assert identity["root_st_dev"] == ROOT.stat().st_dev
    assert identity["root_st_ino"] == ROOT.stat().st_ino
    assert identity["personal_absolute_path_stored"] is False
    assert "C:\\Users" not in CONFIG.read_text(encoding="utf-8")


def test_failed_v1_bytes_are_preserved_and_no_fit_state_exists() -> None:
    config = _config()
    lineage = config["failed_v1_lineage"]
    for expected in lineage["files"].values():
        path = ROOT / expected["path"]
        assert path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    assert lineage["attempt_lock_created"] is False
    assert lineage["fit_count"] == lineage["prediction_count"] == 0
    assert not (ROOT / "artifacts/p3_hierarchical_residual_basis_20260823_v1").exists()
    assert not (
        ROOT / "artifacts/p3_hierarchical_residual_basis_20260823_v1.ATTEMPT_LOCK.json"
    ).exists()


def test_dense72_and_local_vs_official_semantics_are_compiled() -> None:
    config = _config()
    dense = config["dense72_supervision"]
    assert dense["steps"] == 72
    assert dense["complete_cases"] == 23_527
    assert dense["incomplete_cases"] == 833
    assert dense["missing_scalars"] == 1_505
    assert dense["official_six_missing_scalars"] == 0
    assert dense["dense_target_array_materialized_for_all_cases"] is False
    assert config["model"]["optimizer_target_surface"] == (
        "all_available_dense72_train_only_steps"
    )
    assert config["comparator"][
        "reference_seed_full_prediction_exact_to_historical_frozen_oof"
    ] is False
    assert config["comparator"]["may_support_local_numeric_qualification"] is True
    assert config["comparator"]["may_support_official_promotion_without_paired_ab"] is False


def test_direct_engine_and_curve_calls_require_unforgeable_capability(tmp_path: Path) -> None:
    config = _config()
    with pytest.raises(PermissionError, match="capability"):
        engine.execute_curve_stage(
            capability=object(),
            root=ROOT,
            data_dir=tmp_path,
            config=config,
            preflight={"summary_sha256": "0" * 64},
        )
    with pytest.raises(PermissionError, match="capability"):
        engine._run_curve(
            capability=object(),
            root=ROOT,
            data_dir=tmp_path,
            config=config,
            preflight={"summary_sha256": "0" * 64},
            stage=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_missing_independent_qa_fails_before_attempt_lock() -> None:
    config = guard.load_canonical_config(ROOT)
    paths = guard.stage_paths(ROOT, config)
    assert not paths["pre_execution_qa"].exists()
    assert not paths["authorization"].exists()
    assert not paths["attempt_lock"].exists()


def test_capability_issue_and_lock_consumption_cannot_bypass_receipts() -> None:
    config = guard.load_canonical_config(ROOT)
    paths = guard.stage_paths(ROOT, config)
    empty_summary: dict[str, object] = {}
    preflight = {
        "summary": empty_summary,
        "summary_sha256": hashlib.sha256(
            guard.canonical_json_bytes(empty_summary)
        ).hexdigest(),
        "implementation_pins": guard.implementation_pins(ROOT, config),
    }
    with pytest.raises(PermissionError, match="QA receipt is missing"):
        guard.issue_execution_capability(
            ROOT,
            config,
            preflight,
            qa_sha256="0" * 64,
            authorization_sha256="1" * 64,
        )
    fake = guard.ExecutionCapability(
        root_st_dev=int(ROOT.stat().st_dev),
        root_st_ino=int(ROOT.stat().st_ino),
        config_sha256=guard.CONFIG_SHA256,
        static_preflight_sha256=preflight["summary_sha256"],
        qa_sha256="0" * 64,
        authorization_sha256="1" * 64,
        nonce="2" * 64,
    )
    with pytest.raises(PermissionError, match="canonical live.*capability"):
        guard.consume_attempt_lock(
            ROOT,
            config,
            capability=fake,
            preflight=preflight,
        )
    assert not paths["attempt_lock"].exists()
    assert not paths["output"].exists()
    with pytest.raises(PermissionError, match="QA receipt is missing"):
        guard.verify_pre_execution_qa(
            ROOT,
            config,
            static_preflight_sha256="0" * 64,
        )
    assert not paths["attempt_lock"].exists()


def test_runner_imports_engine_only_after_qa_authorization_and_lock() -> None:
    source = (ROOT / "scripts/run_p3_hierarchical_residual_basis_dense72_r1.py").read_text(
        encoding="utf-8"
    )
    run_start = source.index("def run_once")
    run_source = source[run_start:]
    qa = run_source.index("verify_pre_execution_qa")
    authorization = run_source.index("verify_execution_authorization")
    capability = run_source.index("issue_execution_capability")
    lock = run_source.index("consume_attempt_lock")
    engine_import = run_source.index("importlib.import_module(ENGINE_MODULE)")
    assert qa < authorization < capability < lock < engine_import
    assert 'default="check-only"' in source
    assert "candidate_generated\": False" in source
    assert "test_prediction_generated\": False" in source
    assert "uploads\": 0" in source


def test_clone_or_alternate_root_is_rejected_without_personal_path_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises((FileNotFoundError, PermissionError, guard.Dense72ContractError)):
        guard.load_canonical_config(tmp_path, supplied_config=config)


def test_runtime_modules_resolve_to_registered_canonical_files() -> None:
    config = _config()
    modules = {
        "GUARD": guard,
        "ENGINE": engine,
        "TARGET_ACCESSOR": importlib.import_module("p3_wave.dense72_targets_r1"),
        "MODEL": importlib.import_module(
            "p3_wave.hierarchical_residual_basis_dense72_r1"
        ),
    }
    for role, module in modules.items():
        assert Path(module.__file__).resolve() == (ROOT / config["implementation_roles"][role]).resolve()
