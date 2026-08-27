from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts/run_p1_incumbent_rule_distillation_neural_residual_v5r6.py"
)
SPEC = importlib.util.spec_from_file_location("test_p1_gen5r6_runner", RUNNER_PATH)
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


def _functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_config_and_scientific_projection_are_exact_and_unchanged() -> None:
    config = _config()
    assert _sha(ROOT / runner.CANONICAL_CONFIG) == runner.EXPECTED_CONFIG_SHA256
    assert runner._deep_sha(config) == runner.EXPECTED_CONFIG_DEEP_SHA256
    assert config["comparison_mode"] == "EXACT_OFFICIAL_PREFIX_REFIT"
    assert config["scientific_projection"] == {
        "path": (
            "configs/experiments/"
            "p1_incumbent_rule_distillation_neural_residual_v5_science_projection.json"
        ),
        "sha256": runner.SCIENCE_PROJECTION_SHA256,
        "science_deep_sha256": (
            "a4e976c5e627d82dd360d5efef5126e0b59ed9cb62ffbe26762ebebc0f22b4c9"
        ),
        "source_generation": "p1_incumbent_rule_distillation_neural_residual_v5",
        "source_config_sha256": runner.SCIENCE_SOURCE_CONFIG_SHA256,
        "explicit_allowlist_only": True,
        "path_bearing_predecessor_metadata_imported": False,
    }
    science = runner._load_scientific_projection(runner._paths(ROOT), config)
    assert science["prefix_fractions"] == list(runner.FRACTIONS)
    assert science["seeds"] == list(runner.SEEDS)
    assert science["hypotheses"][0]["id"] == runner.HYPOTHESIS


def test_config_declares_capability_correction_without_science_change() -> None:
    config = _config()
    correction = config["correction_contract"]
    assert correction["scientific_structure_changed"] is False
    for key in (
        "authorize_entry_returns_opaque_prelock_capability",
        "qa_and_authorization_verified_before_prelock_mint",
        "prelock_capability_registry_state_not_returned",
        "acquire_lock_accepts_only_opaque_prelock_capability",
        "lock_binds_canonical_qa_and_authorization",
        "lock_binds_full_operational_snapshot",
        "postlock_qa_authorization_reload",
        "postlock_capability_after_consumed_canonical_lock_only",
        "run_after_lock_accepts_capability_only",
        "run_curve_accepts_capability_only",
        "full_fit_accepts_capability_only",
        "fixed_internal_stage_transitions",
        "forged_self_consistent_lock_rejected",
        "direct_call_rejected",
        "capability_replay_rejected",
        "caller_selected_stage_rejected",
    ):
        assert correction[key] is True
    assert correction["acquire_lock_caller_hash_parameters"] == 0
    provenance = config["provenance_correction_contract"]
    assert provenance["scientific_structure_changed"] is False
    assert provenance[
        "science_projection_model_folds_seeds_gates_and_capability_closure_unchanged"
    ] is True
    assert provenance["known_data_identity_set_count"] == 2
    assert provenance["accepted_data_identity_sets"] == [
        "r5_frozen",
        "reviewer_alternate",
    ]
    assert provenance["complete_atomic_set_match_only"] is True
    assert provenance["mixed_state_rejected"] is True
    assert provenance["third_state_rejected"] is True
    assert provenance["oscillation_between_known_sets_does_not_change_normalized_snapshot"]
    assert tuple(config["allowed_data_identity_sets"]) == (
        "r5_frozen",
        "reviewer_alternate",
    )


def test_code_config_and_tests_contain_no_personal_path_literal() -> None:
    targets = [
        RUNNER_PATH,
        ROOT / "src/p1_qc/incumbent_residual_experiment_v5r3.py",
        ROOT / runner.CANONICAL_CONFIG,
        Path(__file__),
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


def test_hardlink_and_reparse_anchors_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = tmp_path / "anchor.txt"
    linked = tmp_path / "linked.txt"
    anchor.write_text("anchor", encoding="utf-8")
    try:
        os.link(anchor, linked)
    except OSError as exc:
        pytest.skip(f"hardlink unsupported in test environment: {exc}")
    with pytest.raises(PermissionError, match="hardlinked"):
        runner._file_identity(anchor, role="poison_hardlink")

    reparse_target = tmp_path / "reparse.txt"
    reparse_target.write_text("anchor", encoding="utf-8")
    original_lstat = runner.os.lstat

    def poisoned_lstat(path: object) -> object:
        observed = original_lstat(path)
        if Path(path) == reparse_target:
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
        runner._file_identity(reparse_target, role="poison_reparse")


def test_full_portable_closure_matches_and_excludes_runner_chain() -> None:
    config = _config()
    observed = runner._verify_execution_closure(ROOT, config)
    assert observed == config["execution_closure_sha256"]
    assert len(observed) == 29
    assert runner._REQUIRED_EXECUTABLE_CLOSURE.issubset(observed)
    assert not runner._FORBIDDEN_EXECUTABLE_CLOSURE.intersection(observed)
    assert not any(path.startswith("scripts/") for path in observed)
    assert observed["src/p1_qc/incumbent_residual_experiment_v5r3.py"] == (
        "8ead3a1348543989cb87737672b1e3e130ac3cb0f8b8428e4c0b355d0e9bc9fc"
    )


def test_r5_owner_no_go_and_tombstone_are_exact() -> None:
    result = runner._verify_r5_tombstone(runner._paths(ROOT))
    assert result["execution_prohibited"] is True
    assert result["r5_config"] == runner.R5_CONFIG_SHA256
    assert result["r5_runner"] == runner.R5_RUNNER_SHA256
    assert result["r5_tests"] == runner.R5_TEST_SHA256
    assert result["r5_preregistration"] == runner.R5_PREREGISTRATION_SHA256
    assert result["r5_preseal"] == runner.R5_PRESEAL_SHA256
    assert result["r5_owner_no_go"] == runner.R5_OWNER_NO_GO_SHA256
    assert result["r5_tombstone"] == runner.R5_TOMBSTONE_SHA256


def test_r5_identity_lineage_binds_both_known_sets() -> None:
    paths = runner._paths(ROOT)
    config = _config()
    lineage = runner._verify_identity_lineage(paths, config)
    assert len(lineage["observations"]) == 2
    assert lineage["both_observations_match_r5_frozen_set"] is True
    assert lineage["reviewer_alternate_set_bound_by_r5_no_go"] is True
    assert lineage["accepted_identity_set_ids"] == ["r5_frozen", "reviewer_alternate"]
    assert lineage["observations"][0]["captured_at_kst"] < (
        lineage["observations"][1]["captured_at_kst"]
    )


def test_all_four_source_sha_and_bytes_are_exactly_equal_to_r5() -> None:
    paths = runner._paths(ROOT)
    equivalence = runner._verify_r5_source_content_equivalence(paths, _config())
    assert equivalence["all_four_sha256_and_bytes_equal"] is True
    assert set(equivalence["sources"]) == {
        "README.md",
        "train.csv",
        "test.csv",
        "sample_submission.csv",
    }
    for value in equivalence["sources"].values():
        assert value["r5_sha256"] == value["current_sha256"]
        assert value["r5_bytes"] == value["current_bytes"]


def test_identity_lineage_and_r5_content_poison_are_rejected() -> None:
    paths = runner._paths(ROOT)
    config = _config()
    poisoned_config = json.loads(json.dumps(config))
    poisoned_config["anchor_file_identity"]["source_readme"]["sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="source content differs from r5"):
        runner._verify_r5_source_content_equivalence(paths, poisoned_config)
    poisoned_config = json.loads(json.dumps(config))
    poisoned_config["identity_lineage"]["r5_owner_no_go"]["sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="configured identity lineage differs"):
        runner._verify_identity_lineage(paths, poisoned_config)


def _patch_live_data_identity(
    monkeypatch: pytest.MonkeyPatch,
    selected: dict[str, object],
) -> None:
    original_directory_identity = runner._directory_identity
    original_file_identity = runner._file_identity

    def fake_directory_identity(path: Path, *, role: str) -> dict[str, object]:
        if role == "data":
            return json.loads(json.dumps(selected["value"]))["directory"]
        return original_directory_identity(path, role=role)

    def fake_file_identity(
        path: Path,
        *,
        role: str,
        expected_sha256: str | None = None,
    ) -> dict[str, object]:
        if role in runner._DATA_ANCHOR_ROLES:
            return json.loads(json.dumps(selected["value"]))["files"][role]
        return original_file_identity(
            path,
            role=role,
            expected_sha256=expected_sha256,
        )

    monkeypatch.setattr(runner, "_directory_identity", fake_directory_identity)
    monkeypatch.setattr(runner, "_file_identity", fake_file_identity)


def test_both_known_identity_sets_are_accepted_and_normalize_identically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    allowed = runner._validate_allowed_data_identity_sets(config)
    selected: dict[str, object] = {"value": allowed["r5_frozen"]}
    _patch_live_data_identity(monkeypatch, selected)
    set_id, observed = runner._capture_live_data_identity(
        tmp_path / "virtual-data",
        allowed_sets=allowed,
    )
    assert set_id == "r5_frozen"
    assert observed == allowed["r5_frozen"]
    frozen_snapshot = runner._capture_anchor_snapshot(ROOT, tmp_path / "virtual-data", config)

    selected["value"] = allowed["reviewer_alternate"]
    set_id, observed = runner._capture_live_data_identity(
        tmp_path / "virtual-data",
        allowed_sets=allowed,
    )
    assert set_id == "reviewer_alternate"
    assert observed == allowed["reviewer_alternate"]
    alternate_snapshot = runner._capture_anchor_snapshot(
        ROOT,
        tmp_path / "virtual-data",
        config,
    )
    assert frozen_snapshot == alternate_snapshot
    assert frozen_snapshot["data_identity_policy"][
        "oscillation_between_known_sets_normalized"
    ] is True


@pytest.mark.parametrize("poison_kind", ["third_state", "mixed_state"])
def test_third_or_mixed_identity_state_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison_kind: str,
) -> None:
    config = _config()
    allowed = runner._validate_allowed_data_identity_sets(config)
    poison = json.loads(json.dumps(allowed["r5_frozen"]))
    if poison_kind == "third_state":
        poison["directory"]["inode"] += 1000
        for value in poison["files"].values():
            value["inode"] += 1000
    else:
        poison["files"]["train_source"] = json.loads(
            json.dumps(allowed["reviewer_alternate"]["files"]["train_source"])
        )
    selected: dict[str, object] = {"value": poison}
    _patch_live_data_identity(monkeypatch, selected)
    with pytest.raises(PermissionError, match="exactly one complete allowed set"):
        runner._capture_live_data_identity(
            tmp_path / "virtual-data",
            allowed_sets=allowed,
        )


def test_allowed_identity_set_contract_rejects_content_or_link_drift() -> None:
    config = _config()
    poisoned = json.loads(json.dumps(config))
    poisoned["allowed_data_identity_sets"]["reviewer_alternate"]["files"][
        "train_source"
    ]["sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="share exact source content"):
        runner._validate_allowed_data_identity_sets(poisoned)
    poisoned = json.loads(json.dumps(config))
    poisoned["allowed_data_identity_sets"]["reviewer_alternate"]["files"][
        "train_source"
    ]["nlink"] = 2
    with pytest.raises(PermissionError, match="link guard differs"):
        runner._validate_allowed_data_identity_sets(poisoned)


def test_entry_and_lock_apis_expose_only_opaque_capabilities() -> None:
    assert inspect.signature(runner.authorize_entry).return_annotation in {
        "_PreLockCapability",
        runner._PreLockCapability,
    }
    assert list(inspect.signature(runner._acquire_lock).parameters) == ["capability"]
    for name in ("_run_after_lock", "_run_curve", "_full_fit_models"):
        assert list(inspect.signature(getattr(runner, name)).parameters) == ["capability"]
    with pytest.raises(PermissionError, match="pre-lock capability mint"):
        runner._PreLockCapability(object(), object())
    with pytest.raises(PermissionError, match="post-lock capability mint"):
        runner._ExecutionCapability(object(), object())


def test_entrypoint_order_is_authorize_then_lock_then_execute_only() -> None:
    function = _functions()["run_experiment"]
    assert len(function.body) == 3
    authorize, acquire, execute = function.body
    assert isinstance(authorize, ast.Assign)
    assert isinstance(authorize.value, ast.Call)
    assert isinstance(authorize.value.func, ast.Name)
    assert authorize.value.func.id == "authorize_entry"
    assert isinstance(acquire, ast.Assign)
    assert isinstance(acquire.value, ast.Call)
    assert isinstance(acquire.value.func, ast.Name)
    assert acquire.value.func.id == "_acquire_lock"
    assert len(acquire.value.args) == 1
    assert not acquire.value.keywords
    assert isinstance(execute, ast.Return)
    assert isinstance(execute.value, ast.Call)
    assert isinstance(execute.value.func, ast.Name)
    assert execute.value.func.id == "_run_after_lock"
    assert [item.arg for item in execute.value.keywords] == ["capability"]


def test_execution_stages_are_internal_fixed_and_capability_first() -> None:
    functions = _functions()
    expected = {
        "_run_after_lock": ("LOCK_CONSUMED", "PREPARING_RUNTIME"),
        "_run_curve": ("BLIND_CURVE_READY", "CURVE_RUNNING"),
        "_full_fit_models": ("FULL_FIT_AUTHORIZED", "FULL_FIT_RUNNING"),
    }
    for name, (stage, transition) in expected.items():
        function = functions[name]
        assert [item.arg for item in function.args.kwonlyargs] == ["capability"]
        first = function.body[0]
        assert isinstance(first, ast.Assign)
        assert isinstance(first.value, ast.Call)
        assert isinstance(first.value.func, ast.Name)
        assert first.value.func.id == "_require_execution_capability"
        keywords = {
            item.arg: item.value for item in first.value.keywords if item.arg is not None
        }
        assert isinstance(keywords["expected_stage"], ast.Constant)
        assert keywords["expected_stage"].value == stage
        assert isinstance(keywords["transition_to"], ast.Constant)
        assert keywords["transition_to"].value == transition


def test_only_lock_acquisition_can_mint_postlock_capability() -> None:
    functions = _functions()
    callers: set[str] = set()
    for name, function in functions.items():
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_mint_execution_capability"
            ):
                callers.add(name)
    assert callers == {"_acquire_lock"}


def test_stale_or_forged_capabilities_and_arbitrary_stages_are_rejected() -> None:
    stale_prelock = runner._PreLockCapability(runner._PRELOCK_MINT, object())
    stale_execution = runner._ExecutionCapability(runner._EXECUTION_MINT, object())
    with pytest.raises(PermissionError, match="forged or stale"):
        runner._require_prelock_capability(
            stale_prelock,
            expected_stage="QA_AUTH_VERIFIED_PRELOCK",
        )
    with pytest.raises(PermissionError, match="forged, stale, or replayed"):
        runner._require_execution_capability(
            stale_execution,
            expected_stage="LOCK_CONSUMED",
        )
    for call in (
        lambda: runner._acquire_lock(object()),
        lambda: runner._run_after_lock(capability=object()),
        lambda: runner._run_curve(capability=object()),
        lambda: runner._full_fit_models(capability=object()),
    ):
        with pytest.raises(PermissionError, match="canonical"):
            call()
    assert not runner._LIVE_PRELOCK
    assert not runner._LIVE_EXECUTION


def test_forged_self_consistent_lock_cannot_replace_missing_qa(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "attempt.lock"
    forged_snapshot = {
        "schema_version": "p1_gen5r6_full_operational_snapshot.v1",
        "core": {"runner": {"sha256": "3" * 64}, "forged": True},
        "independent_qa_receipt": {"path": "forged", "sha256": "0" * 64},
        "execution_authorization": {"path": "forged", "sha256": "1" * 64},
    }
    forged_state = {
        "mint": runner._PRELOCK_STATE_MINT,
        "snapshot": forged_snapshot,
        "snapshot_sha256": runner._deep_sha(forged_snapshot),
        "qa_sha256": "0" * 64,
        "authorization_sha256": "1" * 64,
        "nonce_sha256": "2" * 64,
    }
    payload = runner._canonical_lock_payload(forged_state)
    lock.write_text(json.dumps(payload), encoding="utf-8")
    missing_qa = tmp_path / "missing-independent-qa.json"
    missing_auth = tmp_path / "missing-authorization.json"
    forged_context = SimpleNamespace(
        paths={
            "qa_receipt": missing_qa,
            "authorization": missing_auth,
            "lock": lock,
        }
    )
    with pytest.raises(PermissionError, match="independent-QA receipt is missing"):
        runner._verify_consumed_lock(forged_state, forged_context)


def test_missing_qa_and_missing_auth_fail_before_capability_or_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = runner._paths(ROOT)
    with pytest.raises(PermissionError, match="independent-QA receipt is missing"):
        runner.authorize_entry(environ=_environment())
    assert not runner._LIVE_PRELOCK
    assert not runner._LIVE_EXECUTION
    assert not paths["lock"].exists()
    assert not paths["artifact"].exists()

    monkeypatch.setattr(
        runner,
        "_verify_independent_qa",
        lambda context: ({}, "0" * 64),
    )
    with pytest.raises(PermissionError, match="execution authorization is missing"):
        runner.authorize_entry(environ=_environment())
    assert not runner._LIVE_PRELOCK
    assert not runner._LIVE_EXECUTION
    assert not paths["lock"].exists()
    assert not paths["artifact"].exists()


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


def test_canonical_check_only_is_target_free_and_side_effect_free() -> None:
    paths = runner._paths(ROOT)
    before = {
        name: path.exists()
        for name, path in paths.items()
        if name in {"artifact", "qa_receipt", "authorization", "lock"}
    }
    result = runner.check_only(environ=_environment())
    assert result["status"] == "CANONICAL_GEN5R6_CHECK_ONLY_PASS"
    assert result["split_cell_count"] == 15
    assert result["execution_closure_count"] == 29
    assert result["current_data_identity_set"] in {
        "r5_frozen",
        "reviewer_alternate",
    }
    assert result["allowed_data_identity_set_ids"] == [
        "r5_frozen",
        "reviewer_alternate",
    ]
    assert result["anchor_files_single_link_non_reparse"] is True
    assert result["runner_and_config_single_link_non_reparse"] is True
    assert result["opaque_target_index_decoded_scalars"] == 0
    assert result["frozen_oof_target_columns_decoded"] == 0
    assert result[
        "authorize_entry_missing_qa_auth_rejected_before_capability_mint"
    ] is True
    assert all(result["opaque_capability_direct_call_guards"].values())
    assert result["live_prelock_capability_count"] == 0
    assert result["live_postlock_capability_count"] == 0
    assert result["lock_api_caller_supplied_sha_parameters"] == 0
    assert result["model_fits"] == result["target_fold_scores"] == 0
    assert result["test_value_reads"] == result["candidate_files"] == 0
    assert result["uploads"] == 0
    after = {
        name: path.exists()
        for name, path in paths.items()
        if name in {"artifact", "qa_receipt", "authorization", "lock"}
    }
    assert after == before
