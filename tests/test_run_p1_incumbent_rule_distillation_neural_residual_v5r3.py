from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts/run_p1_incumbent_rule_distillation_neural_residual_v5r3.py"
)
SPEC = importlib.util.spec_from_file_location("test_p1_gen5r3_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict[str, object]:
    return json.loads((ROOT / runner.CANONICAL_CONFIG).read_text(encoding="utf-8"))


def _environment() -> dict[str, str]:
    return {
        runner.WORKSPACE_ENV: os.environ[runner.WORKSPACE_ENV],
        runner.DATA_ENV: os.environ[runner.DATA_ENV],
    }


def test_config_and_portable_science_projection_are_exact() -> None:
    config = _config()
    assert _sha(ROOT / runner.CANONICAL_CONFIG) == runner.EXPECTED_CONFIG_SHA256
    assert runner._deep_sha(config) == runner.EXPECTED_CONFIG_DEEP_SHA256
    assert config["comparison_mode"] == "EXACT_OFFICIAL_PREFIX_REFIT"
    assert config["environment_contract"] == {
        "workspace": "P1_WORKSPACE_ROOT",
        "data": "P1_DATA_DIR",
        "fallback_allowed": False,
    }
    science = runner._load_scientific_projection(runner._paths(ROOT), config)
    assert science["hypotheses"] == [
        {
            "id": runner.HYPOTHESIS,
            "order": 1,
            "score_derived_tuning": False,
            "structure": (
                "exact incumbent identity base plus three-block inner cross-fitted incumbent "
                "teacher, a causal station-layer TCN bounded to plus or minus 0.5 logit, and "
                "a fixed held-out train-only no-op gate"
            ),
            "why_structurally_distinct": (
                "identity-preserving error correction rather than replacement learning or "
                "threshold, class-weight, seed, feature, or postprocess variation"
            ),
        }
    ]
    assert science["prefix_fractions"] == list(runner.FRACTIONS)
    assert science["seeds"] == list(runner.SEEDS)


def test_code_config_and_tests_contain_no_personal_path_literal() -> None:
    targets = [
        RUNNER_PATH,
        ROOT / "src/p1_qc/incumbent_residual_experiment_v5r3.py",
        ROOT / runner.CANONICAL_CONFIG,
    ]
    drive_pattern = re.compile("[A-Za-z]" + ":" + r"[\\/]")
    personal_segment = "Us" + "ers"
    source_directory_segment = "\ub370\uc774\ud130\uc14b \uc6d0\ubcf8"
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert drive_pattern.search(text) is None
        assert personal_segment not in text
        assert source_directory_segment not in text


def test_environment_injection_is_mandatory_and_has_no_fallback() -> None:
    with pytest.raises(PermissionError, match="both required"):
        runner._environment_paths({})
    with pytest.raises(PermissionError, match="both required"):
        runner._environment_paths({runner.WORKSPACE_ENV: str(ROOT)})
    root, data_dir = runner._environment_paths(_environment())
    assert os.path.samefile(root / runner.CANONICAL_RUNNER, RUNNER_PATH)
    assert os.path.samefile(data_dir, Path(os.environ[runner.DATA_ENV]))


def test_injected_clone_workspace_is_rejected(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    clone_runner = clone / runner.CANONICAL_RUNNER
    clone_runner.parent.mkdir(parents=True)
    clone_runner.write_bytes(RUNNER_PATH.read_bytes())
    with pytest.raises(PermissionError, match="does not own"):
        runner._environment_paths(
            {
                runner.WORKSPACE_ENV: str(clone),
                runner.DATA_ENV: os.environ[runner.DATA_ENV],
            }
        )


def test_hardlink_anchor_is_rejected(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.txt"
    linked = tmp_path / "linked.txt"
    anchor.write_text("anchor", encoding="utf-8")
    try:
        os.link(anchor, linked)
    except OSError as exc:
        pytest.skip(f"hardlink unsupported in test environment: {exc}")
    with pytest.raises(PermissionError, match="hardlinked"):
        runner._file_identity(anchor, role="poison_hardlink")


def test_symlink_or_reparse_anchor_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("anchor", encoding="utf-8")
    original_lstat = runner.os.lstat

    def poisoned_lstat(path: object) -> object:
        observed = original_lstat(path)
        if Path(path) == target:
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_file_attributes=getattr(
                    runner.stat,
                    "FILE_ATTRIBUTE_REPARSE_POINT",
                    0x400,
                ),
            )
        return observed

    monkeypatch.setattr(runner.os, "lstat", poisoned_lstat)
    with pytest.raises(PermissionError, match="reparse-point"):
        runner._file_identity(target, role="poison_reparse")


def test_full_portable_closure_matches_and_excludes_runner_chain() -> None:
    config = _config()
    observed = runner._verify_execution_closure(ROOT, config)
    assert observed == config["execution_closure_sha256"]
    assert runner._REQUIRED_EXECUTABLE_CLOSURE.issubset(observed)
    assert not runner._FORBIDDEN_EXECUTABLE_CLOSURE.intersection(observed)
    assert not any(path.startswith("scripts/") for path in observed)
    assert {
        "src/p1_qc/binary_event_tcn.py",
        "src/p1_qc/data.py",
        "src/p1_qc/config.py",
        "src/p1_qc/models_tabular.py",
        "src/p1_qc/rules.py",
        "src/p1_qc/validation.py",
        "requirements-lock.txt",
    }.issubset(observed)


def test_r2_owner_no_go_and_tombstone_are_exact() -> None:
    result = runner._verify_r2_tombstone(runner._paths(ROOT))
    assert result["execution_prohibited"] is True
    assert result["r2_config"] == runner.R2_CONFIG_SHA256
    assert result["r2_runner"] == runner.R2_RUNNER_TOMBSTONED_SHA256
    assert result["r2_owner_no_go"] == runner.R2_OWNER_NO_GO_SHA256
    assert result["r2_tombstone"] == runner.R2_TOMBSTONE_SHA256


def test_run_curve_and_full_fit_require_live_capability_first() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for name, phase in (
        ("_run_curve", "BLIND_CURVE"),
        ("_full_fit_models", "FULL_FIT_AUTHORIZED"),
    ):
        first = functions[name].body[0]
        assert isinstance(first, ast.Expr)
        assert isinstance(first.value, ast.Call)
        assert isinstance(first.value.func, ast.Name)
        assert first.value.func.id == "_require_capability"
        assert isinstance(first.value.args[1], ast.Constant)
        assert first.value.args[1].value == phase
    with pytest.raises(PermissionError, match="canonical live"):
        runner._require_capability(None, "BLIND_CURVE")


def test_run_orders_qa_authorization_rechecks_before_lock() -> None:
    source = ast.unparse(ast.parse(RUNNER_PATH.read_text(encoding="utf-8")))
    body = source[source.index("def run_experiment") :]
    qa = body.index("_verify_independent_qa(")
    authorization = body.index("_verify_execution_authorization(")
    closure = body.index("_verify_execution_closure(")
    anchors = body.index("_capture_anchor_snapshot(")
    pins = body.index("verify_relative_input_pins(")
    lock = body.index("lock = _acquire_lock(")
    after = body.index("return _run_after_lock(")
    assert qa < authorization < closure < anchors < pins < lock < after


def test_static_state_has_only_owner_seals() -> None:
    paths = runner._paths(ROOT)
    assert paths["preregistration"].is_file()
    assert paths["preseal"].is_file()
    assert not paths["qa_receipt"].exists()
    assert not paths["authorization"].exists()
    assert not paths["lock"].exists()
    assert not paths["artifact"].exists()
    assert _config()["static_counters"] == {
        "owner_preregistrations": 1,
        "owner_preseals": 1,
        "independent_qa_receipts": 0,
        "execution_authorizations": 0,
        "attempt_locks": 0,
        "model_fits": 0,
        "predictions": 0,
        "target_fold_scores": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }


def test_missing_independent_qa_rejects_before_lock_or_artifact() -> None:
    paths = runner._paths(ROOT)
    with pytest.raises(PermissionError, match="independent-QA receipt is missing"):
        runner.run_experiment(environ=_environment())
    assert not paths["lock"].exists()
    assert not paths["artifact"].exists()


def test_canonical_check_only_is_target_free_and_side_effect_free() -> None:
    paths = runner._paths(ROOT)
    before = {
        name: path.exists()
        for name, path in paths.items()
        if name in {"artifact", "qa_receipt", "authorization", "lock"}
    }
    result = runner.check_only(environ=_environment())
    assert result["status"] == "CANONICAL_GEN5R3_CHECK_ONLY_PASS"
    assert result["split_cell_count"] == 15
    assert result["execution_closure_count"] == 29
    assert result["anchor_files_single_link_non_reparse"] is True
    assert result["runner_and_config_single_link_non_reparse"] is True
    assert result["opaque_target_index_decoded_scalars"] == 0
    assert result["frozen_oof_target_columns_decoded"] == 0
    assert result["model_fits"] == result["target_fold_scores"] == 0
    assert result["test_value_reads"] == result["candidate_files"] == 0
    assert result["uploads"] == 0
    after = {
        name: path.exists()
        for name, path in paths.items()
        if name in {"artifact", "qa_receipt", "authorization", "lock"}
    }
    assert after == before
