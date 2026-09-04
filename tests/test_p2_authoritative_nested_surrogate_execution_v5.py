from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from p2_restore import authoritative_nested_surrogate_execution_v5 as v5
from p2_restore.authoritative_nested_surrogate_execution_v5 import (
    ExecutionBindingV5,
    SemanticPreflightOutcomeV5,
    TERMINAL_STATUS,
    inspect_actual_namespace_read_only,
    run_resumable_execution_v5,
)
from scripts import run_p2_authoritative_nested_surrogate_45cell_v3 as v3_runner
from scripts import run_p2_authoritative_nested_surrogate_45cell_v5 as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/experiments/p2_authoritative_nested_surrogate_execution_20260825_v5.json"
RECIPE = PROJECT_ROOT / "configs/experiments/p2_authoritative_nested_surrogate_recipe_20260825_v5.json"


class SimulatedProcessDeath(BaseException):
    """Bypass graceful failure receipts like an abruptly killed process."""


def _binding(namespace: str) -> ExecutionBindingV5:
    return ExecutionBindingV5(
        namespace=namespace,
        execution_contract_sha256="a" * 64,
        parent_recipe_sha256="b" * 64,
        preexecution_seal_sha256="c" * 64,
        semantic_preflight_sha256="d" * 64,
        exact_command_sha256="e" * 64,
        authorization_sha256="f" * 64,
        module_sha256="1" * 64,
        runner_sha256="2" * 64,
    )


def _preflight() -> SemanticPreflightOutcomeV5:
    return SemanticPreflightOutcomeV5("d" * 64, None)


def _terminal_result() -> dict[str, object]:
    return {
        "status": TERMINAL_STATUS,
        "submission_files_generated": 0,
        "uploads": 0,
    }


def _authorization(command: str, schema: object) -> dict[str, object]:
    return {
        "schema_version": schema,
        "status": "APPROVED_EXACT_P2_45_CELL_COMMAND",
        "training_authorized": True,
        "preexecution_seal_sha256": "seal",
        "exact_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "official_test_access_authorized": False,
        "sample_submission_access_authorized": False,
        "submission_generation_authorized": False,
        "public_score_selection_authorized": False,
        "upload_authorized": False,
        "p3_process_mutation_authorized": False,
        "same_contract_crash_resume_authorized": True,
        "maximum_automatic_resume_attempts_after_initial": 2,
        "maximum_total_attempts": 3,
        "result_based_rerun_or_tuning_authorized": False,
        "v1_v2_v3_v4_job_or_cell_reuse_authorized": False,
        "single_scientific_contract_only": True,
        "single_execution_only": False,
    }


def test_v5_crash_resume_isolated_namespace_and_terminal_closes(tmp_path: Path) -> None:
    actual = tmp_path / "synthetic_actual_v5"
    binding = _binding(actual.name)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v5(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight,
            execute_curve=lambda _context, _contract: (_ for _ in ()).throw(
                SimulatedProcessDeath("hard stop")
            ),
        )
    interrupted = inspect_actual_namespace_read_only(actual, binding=binding)
    assert interrupted["total_attempts_started"] == 1
    terminal = run_resumable_execution_v5(
        actual_dir=actual,
        binding=binding,
        semantic_preflight=_preflight,
        execute_curve=lambda _context, _contract: _terminal_result(),
    )
    assert terminal["total_attempts_started"] == 2
    assert inspect_actual_namespace_read_only(actual, binding=binding)["status"] == "TERMINAL_COMPLETE_NO_RERUN"
    result = json.loads((actual / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == TERMINAL_STATUS
    assert result["official_test_sample_submission_reads"] == 0
    assert result["execution_binding_sha256"] == hashlib.sha256(
        json.dumps(binding.as_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    "schema",
    [None, "", "p2_authoritative_nested_surrogate_execution_authorization.v4", "p2_authoritative_nested_surrogate_execution_authorization.v5.extra"],
)
def test_authorization_rejects_every_nonexact_schema(tmp_path: Path, schema: object) -> None:
    command = "sealed v5 command"
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_authorization(command, schema)), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        runner._verify_authorization_v5(
            path,
            preexecution_seal_sha256="seal",
            exact_command=command,
        )


def test_authorization_accepts_exact_v5_schema_and_bounded_resume(tmp_path: Path) -> None:
    command = "sealed v5 command"
    path = tmp_path / "authorization.json"
    value = _authorization(command, runner.REQUIRED_AUTHORIZATION_SCHEMA_VERSION)
    path.write_text(json.dumps(value), encoding="utf-8")
    verified = runner._verify_authorization_v5(
        path,
        preexecution_seal_sha256="seal",
        exact_command=command,
    )
    assert verified["schema_version"] == runner.REQUIRED_AUTHORIZATION_SCHEMA_VERSION
    assert verified["maximum_automatic_resume_attempts_after_initial"] == 2
    assert verified["result_based_rerun_or_tuning_authorized"] is False
    assert verified["v1_v2_v3_v4_job_or_cell_reuse_authorized"] is False


def test_exact_command_literal_data_dir_and_v5_namespaces_only() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    command = raw["exact_command"]
    receipt = runner._command_namespace(command, config=raw)
    assert receipt["literal_resolved_data_directory"] == str(runner.APPROVED_DATA_DIRECTORY)
    assert receipt["environment_data_dir_dependency"] is False
    assert "$env:P2_DATA_DIR" not in command
    assert raw["required_authorization"]["schema_version"] == runner.REQUIRED_AUTHORIZATION_SCHEMA_VERSION


def test_resolved_data_dir_is_exact_and_observations_bytes_are_pinned(tmp_path: Path) -> None:
    config, _ = runner._load_config_v5(CONFIG)
    receipt = runner._verify_resolved_data_dir(runner.APPROVED_DATA_DIRECTORY, config)
    assert receipt["status"] == "PASS_LITERAL_APPROVED_DATA_DIR_OBSERVATIONS_ONLY"
    assert receipt["directory_entries_enumerated"] == 0
    assert receipt["observations_sha256"] == config["data_contract"]["sha256"]
    with pytest.raises(ValueError, match="sealed absolute path"):
        runner._verify_resolved_data_dir(tmp_path, config)


def test_actual_data_semantic_preflight_is_zero_fit_and_inherits_v4_digest() -> None:
    config, _ = runner._load_config_v5(CONFIG)
    recipe, _ = runner._load_recipe_v5(RECIPE)
    observations, access = v3_runner._load_observations(runner.APPROVED_DATA_DIRECTORY, config)
    plans, receipt = v5.semantic_preflight_actual_data_v5(observations, recipe=recipe, config=config)
    runner._verify_semantic_gates(config, receipt)
    assert access["files_opened"] == ["observations.csv"]
    assert len(plans) == 15
    assert receipt["inner_scopes_checked"] == 45
    assert receipt["outer_evaluation_rows_per_fraction"] == 78156
    assert (receipt["component_model_fits"], receipt["predictions_materialized"], receipt["scores_computed"]) == (0, 0, 0)
    assert receipt["orchestration_revision_v5"]["v4_resume_engine_byte_pinned"] is True


def test_v5_binding_refuses_foreign_namespace() -> None:
    with pytest.raises(ValueError, match="not isolated"):
        _binding("p2_authoritative_nested_surrogate_actual_20260825_v4").validate()


def test_static_contract_preserves_science_and_creates_no_actual_namespace() -> None:
    config, _recipe, static = runner._verify_static(CONFIG)
    assert static["status"] == "PASS_V5_STATIC_EXACT_AUTH_SCHEMA_LITERAL_DATA_DIR_AND_UNCHANGED_SCIENCE"
    assert static["only_launch_authorization_and_data_dir_binding_changed"] is True
    assert static["outer_evaluation_rows_per_fraction"] == 78156
    actual = PROJECT_ROOT / config["output"]["actual_directory"]
    assert not actual.exists()
