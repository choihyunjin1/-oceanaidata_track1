from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import verify_p2_authoritative_nested_surrogate_postexecution_v5 as verifier


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes(verifier._json_bytes(value))


def _self_hashed(value: dict, field: str) -> dict:
    result = dict(value)
    result[field] = verifier.canonical_sha256(result)
    return result


def _pin(path: Path) -> dict[str, object]:
    return {"sha256": verifier.sha256_file(path), "bytes": path.stat().st_size}


def _atomic_pin(path: Path) -> dict[str, object]:
    return {
        "status": "COMMITTED_BY_FSYNC_AND_ATOMIC_RENAME",
        **_pin(path),
        "partial_created": True,
        "partial_consumed_by_rename": True,
        "partial_policy": "SUCCESSFUL_CURRENT_PARTIAL_CONSUMED_OLD_PARTIALS_PRESERVED",
    }


def _store_entry(
    root: Path,
    *,
    identifier: str,
    contract_sha256: str,
    receipt: dict,
    payload_name: str,
) -> None:
    directory = root / identifier
    directory.mkdir(parents=True)
    (directory / "prediction.parquet").write_bytes(b"opaque-prediction-bytes")
    _write_json(directory / "receipt.json", receipt)
    (directory / payload_name).write_bytes(b"opaque-model-bytes")
    files = {
        name: _pin(directory / name)
        for name in ("prediction.parquet", "receipt.json", payload_name)
    }
    _write_json(
        directory / "manifest.json",
        {
            "schema_version": "p2_authoritative_nested_surrogate_job.v1",
            "job_id": identifier,
            "contract_sha256": contract_sha256,
            "files": files,
            "payload_files": [payload_name],
            "complete": True,
        },
    )


def _router_receipt(*, phase: str, seed: int) -> dict:
    return {
        "component": "router_400",
        "phase": phase,
        "seed": seed,
        "composite_lightgbm_estimators": 4,
        "rounds_per_estimator": 400,
        "cpu_threads_per_estimator": 4,
        "training_timestamp_count": 12,
        "prediction_timestamp_count": 3,
        "future_or_outer_labels_in_fit": False,
    }


def _contract(namespace: str) -> verifier.PostExecutionContract:
    scope_id = "outer_fixture__p100"
    seed = 20260823
    inner_job_id = f"{scope_id}__s{seed}__inner_1__router_400"
    full_job_id = f"{scope_id}__s{seed}__full__router_400"
    receipt_id = f"{scope_id}__s{seed}"
    directory_id = f"cell__{receipt_id}"
    binding = {
        "namespace": namespace,
        "execution_contract_sha256": "1" * 64,
        "parent_recipe_sha256": "2" * 64,
        "preexecution_seal_sha256": "3" * 64,
        "semantic_preflight_sha256": "4" * 64,
        "exact_command_sha256": "5" * 64,
        "authorization_sha256": "6" * 64,
        "module_sha256": "7" * 64,
        "runner_sha256": "8" * 64,
        "job_store_contract_sha256": "3" * 64,
        "expected_terminal_status": verifier.TERMINAL_STATUS,
        "maximum_resume_attempts": 2,
        "maximum_total_attempts": 3,
        "execution_contract_revision": "v5",
        "control_engine_schema_version": "v4",
    }
    fixture_outer_key_sha = verifier.canonical_sha256(
        [
            {
                "station": "SYNTHETIC",
                "layer": layer,
                "time": time,
            }
            for layer, time in zip(
                (2, 3, 4),
                (
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:10:00+00:00",
                    "2026-01-01T00:20:00+00:00",
                ),
                strict=True,
            )
        ]
    )
    return verifier.PostExecutionContract(
        namespace=namespace,
        binding=binding,
        scopes=(
            verifier.ScopeSpec(
                scope_id=scope_id,
                outer_fold="outer_fixture",
                fraction=1.0,
                cutoff_kst="2026-01-01T00:00:00+09:00",
                inner_folds=("inner_1",),
                prefix_time_count=12,
            ),
        ),
        jobs={
            inner_job_id: verifier.JobSpec(
                job_id=inner_job_id,
                scope_id=scope_id,
                outer_fold="outer_fixture",
                fraction=1.0,
                cutoff_kst="2026-01-01T00:00:00+09:00",
                pipeline_seed=seed,
                phase="inner_1",
                component="router_400",
                child_seed=11,
                training_supervised_time_count=12,
                validation_supervised_time_count=3,
                prefix_supervised_time_count=12,
            ),
            full_job_id: verifier.JobSpec(
                job_id=full_job_id,
                scope_id=scope_id,
                outer_fold="outer_fixture",
                fraction=1.0,
                cutoff_kst="2026-01-01T00:00:00+09:00",
                pipeline_seed=seed,
                phase="full",
                component="router_400",
                child_seed=12,
                prefix_supervised_time_count=12,
            ),
        },
        cells={
            directory_id: verifier.CellSpec(
                directory_id=directory_id,
                receipt_id=receipt_id,
                scope_id=scope_id,
                outer_fold="outer_fixture",
                fraction=1.0,
                cutoff_kst="2026-01-01T00:00:00+09:00",
                pipeline_seed=seed,
                inner_folds=("inner_1",),
                expected_outer_rows=3,
                expected_inner_oof_rows=3,
            )
        },
        seeds=(seed,),
        components=("router_400",),
        layers=(2, 3, 4),
        outer_folds=("outer_fixture",),
        tokens=("100",),
        family_settings=("INCUMBENT_NOOP",),
        expected_rows_per_fraction=3,
        expected_fold_rows={"outer_fixture": 3},
        expected_layer_rows={2: 1, 3: 1, 4: 1},
        expected_fold_layer_rows={"outer_fixture": {2: 1, 3: 1, 4: 1}},
        expected_outer_key_sha256={"outer_fixture": fixture_outer_key_sha},
        expected_base_fits=8,
        expected_deep_fits=0,
        expected_lightgbm_fits=8,
        expected_meta_optimizations=1,
        readiness_receipt={
            "readiness_manifest_sha256": "9" * 64,
            "preexecution_seal_sha256": "3" * 64,
            "authorization_sha256": "6" * 64,
        },
    )


def _rebind_terminal(actual: Path, contract: verifier.PostExecutionContract) -> None:
    result_path = actual / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resume_count = int(result["resume_attempts_started"])
    total_count = int(result["total_attempts_started"])
    result_sha = verifier.sha256_file(result_path)
    result_bytes = result_path.stat().st_size
    terminal = _self_hashed(
        {
            "schema_version": "p2_authoritative_terminal_receipt.v4",
            "status": "TERMINAL_COMPLETE_NO_RERUN",
            "completed_at_kst": "2026-08-25T12:00:00+09:00",
            "binding": contract.binding,
            "result_sha256": result_sha,
            "result_bytes": result_bytes,
            "result_atomic_publish": _atomic_pin(result_path),
            "initial_start_attempts": 1,
            "resume_attempts_started": resume_count,
            "total_attempts_started": total_count,
            "automatic_resume_budget_remaining": 3 - total_count,
            "terminal_rerun_allowed": False,
        },
        "terminal_receipt_sha256",
    )
    _write_json(actual / "terminal_receipt.json", terminal)


def _terminal_fixture(tmp_path: Path) -> tuple[Path, verifier.PostExecutionContract]:
    actual = tmp_path / "synthetic_actual_v5"
    actual.mkdir(parents=True)
    contract = _contract(actual.name)
    (actual / "execution.lock").write_text("fixture-lock", encoding="utf-8")
    (actual / "attempts").mkdir()
    (actual / "jobs").mkdir()
    (actual / "cells").mkdir()

    start = _self_hashed(
        {
            "schema_version": "p2_authoritative_execution_start.v4",
            "status": "STARTED_INCOMPLETE_RESUMABLE",
            "created_at_kst": "2026-08-25T10:00:00+09:00",
            "binding": contract.binding,
            "initial_start_attempt_count": 1,
            "resume_attempt_budget": 2,
            "total_attempt_budget": 3,
            "result_based_tuning_allowed": False,
            "cross_v1_v2_v3_job_reuse_allowed": False,
        },
        "execution_start_sha256",
    )
    _write_json(actual / "execution_start.json", start)
    attempt = _self_hashed(
        {
            "schema_version": "p2_authoritative_attempt_terminal.v4",
            "status": "COMPLETE_TERMINAL",
            "recorded_at_kst": "2026-08-25T12:00:01+09:00",
            "attempt_number": 1,
            "execution_start_sha256": start["execution_start_sha256"],
            "binding": contract.binding,
            "classification": "SUCCESS",
            "automatic_resume_permitted": False,
            "exception_type": None,
            "exception_message": None,
            "traceback_sha256": None,
            "traceback_text_recorded": False,
            "raw_observation_values_recorded": False,
        },
        "attempt_terminal_sha256",
    )
    _write_json(actual / "attempts" / "attempt_001_terminal.json", attempt)

    scope = contract.scopes[0]
    seed = contract.seeds[0]
    inner_id = f"{scope.scope_id}__s{seed}__inner_1__router_400"
    full_id = f"{scope.scope_id}__s{seed}__full__router_400"
    inner_receipt = _router_receipt(phase="inner_1", seed=11)
    full_receipt = _router_receipt(phase="full", seed=12)
    _store_entry(
        actual / "jobs",
        identifier=inner_id,
        contract_sha256=contract.binding["job_store_contract_sha256"],
        receipt=inner_receipt,
        payload_name="model.joblib",
    )
    _store_entry(
        actual / "jobs",
        identifier=full_id,
        contract_sha256=contract.binding["job_store_contract_sha256"],
        receipt=full_receipt,
        payload_name="model.joblib",
    )
    cell = next(iter(contract.cells.values()))
    cell_receipt = {
        "schema_version": "p2_authoritative_nested_surrogate_cell.v1",
        "cell_id": cell.receipt_id,
        "outer_fold": cell.outer_fold,
        "prefix_fraction": cell.fraction,
        "pipeline_seed": cell.pipeline_seed,
        "prefix_cutoff_kst": cell.cutoff_kst,
        "inner_oof_ledger": {
            "rows": 3,
            "ordered_key_sha256": "a" * 64,
            "ordered_key_truth_sha256": "b" * 64,
            "component_prediction_sha256": {"router_400": "c" * 64},
            "component_count": 1,
            "duplicate_keys": 0,
            "nonfinite_truth": 0,
            "nonfinite_predictions": 0,
            "same_ordered_key_and_truth_across_components": True,
        },
        "selected_inner_epochs": {},
        "full_prefix_epochs": {},
        "meta": {
            "scope_id": cell.receipt_id,
            "oof_rows": 3,
            "oof_key_truth_sha256": "d" * 64,
            "stack_method": "SCIPY_NNLS_THEN_SUM_NORMALIZE_UNIFORM_IF_ALL_ZERO",
            "stack_weights": {
                "2": {"router_400": 1.0},
                "3": {"router_400": 1.0},
                "4": {"router_400": 1.0},
            },
            "gate": {
                "feature_names": list(verifier.STATE_FEATURES),
                "prediction_columns": ["router_400"],
                "regularization": 10.0,
                "layers": {
                    str(layer): {
                        "prior": [1.0],
                        "coefficient_sha256": str(layer) * 64,
                        "optimizer_iterations": 1,
                        "objective_mse": 0.01,
                    }
                    for layer in (2, 3, 4)
                },
            },
            "parameter_source": "CURRENT_SCOPE_INNER_OOF_ONLY",
            "frozen_epoch_reused": False,
            "frozen_stack_reused": False,
            "frozen_gate_reused": False,
        },
        "postprocess": {
            "rows": 3,
            "deep_projection_active_rows": 0,
            "soft_route_rows": 2,
            "final_projection_active_rows": 0,
            "minimum": 10.0,
            "maximum": 12.2,
        },
        "component_receipts": [inner_receipt],
        "full_receipts": [full_receipt],
        "guards": {
            "joint_temp_psal_mask": True,
            "seven_day_embargo": True,
            "future_or_outer_labels_in_fit": False,
            "current_scope_meta_only": True,
            "frozen_epoch_stack_gate_reuse": False,
        },
    }
    _store_entry(
        actual / "cells",
        identifier=cell.directory_id,
        contract_sha256=contract.binding["job_store_contract_sha256"],
        receipt=cell_receipt,
        payload_name="meta.joblib",
    )

    frame = pd.DataFrame(
        {
            "fold": ["outer_fixture"] * 3,
            "station": ["SYNTHETIC"] * 3,
            "layer": [2, 3, 4],
            "time": [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:10:00+00:00",
                "2026-01-01T00:20:00+00:00",
            ],
            "truth": [10.0, 11.0, 12.0],
            f"seed_{seed}": [10.1, 10.9, 12.2],
            "INCUMBENT_NOOP": [10.1, 10.9, 12.2],
        }
    )
    aggregate_path = actual / "evaluated_oof_100.parquet"
    frame.to_parquet(aggregate_path, index=False)
    digest = verifier.canonical_sha256(
        frame.loc[:, [*verifier.OUTER_KEY_COLUMNS, "truth"]].astype(str).to_dict("records")
    )
    metric = verifier._recompute_metric(
        frame, frame["INCUMBENT_NOOP"].to_numpy(float), contract.layers
    )
    prefix_result = {
        "rows": 3,
        "seed_columns": [f"seed_{seed}"],
        "metrics_by_setting": {"INCUMBENT_NOOP": metric},
        "paired_day_bootstrap_vs_incumbent": {"INCUMBENT_NOOP": {"fixture": True}},
        "complementarity": {"INCUMBENT_NOOP": {"fixture": True}},
        "materialization_diagnostics": {},
        "per_seed_setting_count": {"INCUMBENT_NOOP": 1},
        "ordered_key_truth_sha256": digest,
        "evaluated_oof_publish": _atomic_pin(aggregate_path),
    }
    result = {
        "status": verifier.TERMINAL_STATUS,
        "outer_prefix_cells": 1,
        "seeded_cells": 1,
        "component_jobs_new_this_invocation": 2,
        "component_jobs_reused_this_invocation": 0,
        "cell_jobs_new_this_invocation": 1,
        "cell_jobs_reused_this_invocation": 0,
        "metrics_by_prefix": {"100": prefix_result},
        "top_level_component_jobs_total": 2,
        "underlying_base_estimator_fits_total": 8,
        "underlying_deep_fits_total": 0,
        "underlying_lightgbm_fits_total": 8,
        "meta_optimizations_total": 1,
        "same_population_digest_across_fractions": digest,
        "submission_files_generated": 0,
        "uploads": 0,
        "scientific_surface_inherited_byte_pinned_from_v3": True,
        "resume_or_result_based_tuning_performed": False,
        "execution_contract_revision": "v5",
        "v4_resume_engine_byte_pinned": True,
        "foreign_v1_v2_v3_v4_job_or_cell_reuse": 0,
        "execution_binding_sha256": verifier.canonical_sha256(contract.binding),
        "preexecution_seal_sha256": contract.binding["preexecution_seal_sha256"],
        "semantic_preflight_sha256": contract.binding["semantic_preflight_sha256"],
        "initial_start_attempt_count": 1,
        "resume_attempts_started": 0,
        "total_attempts_started": 1,
        "official_test_sample_submission_reads": 0,
    }
    _write_json(actual / "result.json", result)
    _rebind_terminal(actual, contract)
    return actual, contract


def test_synthetic_terminal_graph_and_score_pass(tmp_path: Path) -> None:
    actual, contract = _terminal_fixture(tmp_path)
    qa = verifier.verify_terminal_namespace(actual, contract=contract)
    assert qa["status"] == "PASS_TERMINAL_V5_RECURSIVE_GRAPH_AND_SCORE_QA"
    assert qa["job_store"]["completed_jobs"] == 2
    assert qa["cell_store"]["completed_cells"] == 1
    assert qa["aggregate_score_recomputation"]["prefixes"]["100"]["rows"] == 3


def test_one_transient_resume_then_terminal_success_passes(tmp_path: Path) -> None:
    actual, contract = _terminal_fixture(tmp_path)
    start = json.loads((actual / "execution_start.json").read_text(encoding="utf-8"))
    transient = _self_hashed(
        {
            "schema_version": "p2_authoritative_attempt_terminal.v4",
            "status": "FAILED_TRANSIENT_RESUMABLE",
            "recorded_at_kst": "2026-08-25T11:00:00+09:00",
            "attempt_number": 1,
            "execution_start_sha256": start["execution_start_sha256"],
            "binding": contract.binding,
            "classification": "TRANSIENT_RUNTIME_EXPLICIT",
            "automatic_resume_permitted": True,
            "exception_type": "builtins.TimeoutError",
            "exception_message": "synthetic transient",
            "traceback_sha256": "e" * 64,
            "traceback_text_recorded": False,
            "raw_observation_values_recorded": False,
        },
        "attempt_terminal_sha256",
    )
    _write_json(actual / "attempts" / "attempt_001_terminal.json", transient)
    resume = _self_hashed(
        {
            "schema_version": "p2_authoritative_resume_attempt.v4",
            "status": "RESUME_STARTED",
            "created_at_kst": "2026-08-25T11:01:00+09:00",
            "attempt_number": 2,
            "resume_attempt_number": 1,
            "remaining_resume_budget_after_start": 1,
            "execution_start_sha256": start["execution_start_sha256"],
            "binding": contract.binding,
            "read_only_namespace_audit": {
                "jobs_completed": 2,
                "cells_completed": 0,
                "job_manifest_ledger_sha256": "f" * 64,
                "cell_manifest_ledger_sha256": "0" * 64,
                "terminal_result_absent": True,
                "exclusive_lock_acquired_before_this_receipt": True,
            },
            "result_based_tuning_allowed": False,
        },
        "resume_attempt_sha256",
    )
    _write_json(actual / "attempts" / "resume_attempt_002.json", resume)
    success = _self_hashed(
        {
            "schema_version": "p2_authoritative_attempt_terminal.v4",
            "status": "COMPLETE_TERMINAL",
            "recorded_at_kst": "2026-08-25T12:00:01+09:00",
            "attempt_number": 2,
            "execution_start_sha256": start["execution_start_sha256"],
            "binding": contract.binding,
            "classification": "SUCCESS",
            "automatic_resume_permitted": False,
            "exception_type": None,
            "exception_message": None,
            "traceback_sha256": None,
            "traceback_text_recorded": False,
            "raw_observation_values_recorded": False,
        },
        "attempt_terminal_sha256",
    )
    _write_json(actual / "attempts" / "attempt_002_terminal.json", success)
    result_path = actual / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["component_jobs_new_this_invocation"] = 0
    result["component_jobs_reused_this_invocation"] = 2
    result["resume_attempts_started"] = 1
    result["total_attempts_started"] = 2
    _write_json(result_path, result)
    _rebind_terminal(actual, contract)
    qa = verifier.verify_terminal_namespace(actual, contract=contract)
    assert qa["control_plane"]["resume_attempts"] == 1
    assert qa["control_plane"]["prior_transient_receipts"] == 1


def test_finalization_only_recovered_result_receipt_passes(tmp_path: Path) -> None:
    actual, contract = _terminal_fixture(tmp_path)
    terminal_path = actual / "terminal_receipt.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal.pop("terminal_receipt_sha256")
    terminal["result_atomic_publish"] = {
        "status": "RECOVERED_VERIFIED_ATOMIC_FINAL",
        "sha256": verifier.sha256_file(actual / "result.json"),
        "bytes": (actual / "result.json").stat().st_size,
        "partial_created": False,
        "partial_policy": "STALE_PARTIALS_IGNORED_AND_PRESERVED_FOR_AUDIT",
    }
    _write_json(
        terminal_path,
        _self_hashed(terminal, "terminal_receipt_sha256"),
    )
    qa = verifier.verify_terminal_namespace(actual, contract=contract)
    assert qa["status"] == "PASS_TERMINAL_V5_RECURSIVE_GRAPH_AND_SCORE_QA"


def test_missing_terminal_fails_before_aggregate_values_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual, contract = _terminal_fixture(tmp_path)
    (actual / "terminal_receipt.json").unlink()
    monkeypatch.setattr(
        verifier.pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("aggregate was opened before terminal validation"),
    )
    with pytest.raises(ValueError, match="terminal_receipt.json"):
        verifier.verify_terminal_namespace(actual, contract=contract)


def test_recursive_payload_hash_tamper_fails_closed(tmp_path: Path) -> None:
    actual, contract = _terminal_fixture(tmp_path)
    payload = next((actual / "jobs").glob("*/model.joblib"))
    payload.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes changed|SHA-256 changed"):
        verifier.verify_terminal_namespace(actual, contract=contract)


def test_metric_tamper_fails_even_with_rebound_terminal_receipt(tmp_path: Path) -> None:
    actual, contract = _terminal_fixture(tmp_path)
    result_path = actual / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metrics_by_prefix"]["100"]["metrics_by_setting"]["INCUMBENT_NOOP"][
        "fold_equal_layer_equal_rmse_c"
    ] += 0.01
    _write_json(result_path, result)
    _rebind_terminal(actual, contract)
    with pytest.raises(ValueError, match="100.INCUMBENT_NOOP"):
        verifier.verify_terminal_namespace(actual, contract=contract)


def test_semantic_outer_key_tamper_fails_with_all_result_hashes_rebound(
    tmp_path: Path,
) -> None:
    actual, contract = _terminal_fixture(tmp_path)
    aggregate_path = actual / "evaluated_oof_100.parquet"
    frame = pd.read_parquet(aggregate_path)
    frame.loc[0, "station"] = "WRONG-STATION"
    frame.to_parquet(aggregate_path, index=False)
    result_path = actual / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    population_digest = verifier.canonical_sha256(
        frame.loc[:, [*verifier.OUTER_KEY_COLUMNS, "truth"]].astype(str).to_dict("records")
    )
    result["metrics_by_prefix"]["100"]["ordered_key_truth_sha256"] = population_digest
    result["metrics_by_prefix"]["100"]["evaluated_oof_publish"] = _atomic_pin(
        aggregate_path
    )
    result["same_population_digest_across_fractions"] = population_digest
    _write_json(result_path, result)
    _rebind_terminal(actual, contract)
    with pytest.raises(ValueError, match="sealed semantic outer key population"):
        verifier.verify_terminal_namespace(actual, contract=contract)


def test_cell_gate_contract_tamper_fails(tmp_path: Path) -> None:
    actual, contract = _terminal_fixture(tmp_path)
    spec = next(iter(contract.cells.values()))
    receipt = json.loads(
        (actual / "cells" / spec.directory_id / "receipt.json").read_text(
            encoding="utf-8"
        )
    )
    receipt["meta"]["gate"]["feature_names"][0] = "UNREGISTERED_FEATURE"
    job_receipts = {
        job_id: json.loads(
            (actual / "jobs" / job_id / "receipt.json").read_text(encoding="utf-8")
        )
        for job_id in contract.jobs
    }
    with pytest.raises(ValueError, match="gate features"):
        verifier._verify_cell_receipt(
            receipt,
            spec,
            job_receipts=job_receipts,
            components=contract.components,
        )


def test_partial_and_resume_counter_mismatch_each_fail_closed(tmp_path: Path) -> None:
    actual, contract = _terminal_fixture(tmp_path / "partial_case")
    (actual / ".result.json.partial.fixture").write_bytes(b"partial")
    with pytest.raises(ValueError, match="root entry set|partial"):
        verifier.verify_terminal_namespace(actual, contract=contract)

    actual, contract = _terminal_fixture(tmp_path / "counter_case")
    result_path = actual / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["component_jobs_new_this_invocation"] = 1
    _write_json(result_path, result)
    _rebind_terminal(actual, contract)
    with pytest.raises(ValueError, match="component .*count"):
        verifier.verify_terminal_namespace(actual, contract=contract)


def test_publish_is_atomic_and_refuses_nonpass(tmp_path: Path) -> None:
    actual, contract = _terminal_fixture(tmp_path / "fixture")
    qa = verifier.verify_terminal_namespace(actual, contract=contract)
    verifier_path = tmp_path / "verifier.py"
    test_path = tmp_path / "test_verifier.py"
    verifier_path.write_text("# fixture\n", encoding="utf-8")
    test_path.write_text("# fixture\n", encoding="utf-8")
    output = tmp_path / "qa_output"
    published = verifier._publish_verified_artifacts(
        qa,
        output_dir=output,
        verifier_path=verifier_path,
        test_path=test_path,
    )
    assert published["status"] == qa["status"]
    assert {path.name for path in output.iterdir()} == {
        "postexecution_qa.json",
        "REPORT_KO.md",
        "manifest.json",
    }
    with pytest.raises(ValueError, match="non-PASS"):
        verifier._publish_verified_artifacts(
            {**qa, "status": "FAIL"},
            output_dir=tmp_path / "must_not_exist",
            verifier_path=verifier_path,
            test_path=test_path,
        )
    assert not (tmp_path / "must_not_exist").exists()


def test_sealed_plan_statically_describes_exact_15_45_900_graph() -> None:
    project_root = Path(verifier.__file__).resolve().parents[1]
    plan_path = project_root / verifier.READY_RELATIVE / "execution_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["outer_prefix_cells"] == 15
    assert plan["seeded_cells"] == 45
    assert plan["top_level_component_jobs"] == 900
    assert tuple(plan["complete_pipeline_seeds"]) == verifier.EXPECTED_SEEDS
    assert tuple(plan["component_order"]) == verifier.COMPONENTS
    assert len(plan["prefix_plans"]) == 15
    assert sum(len(row["inner_folds"]) for row in plan["prefix_plans"]) == 45
    assert hashlib.sha256(
        (project_root / verifier.READY_RELATIVE / "manifest.json").read_bytes()
    ).hexdigest() == verifier.SEALED_SHA256["readiness_manifest"]


def test_v3_deep_adapter_receipts_enforce_registered_context_and_labels() -> None:
    inner_spec = verifier.JobSpec(
        job_id="scope__s1__inner_1__depth_query_bitcn",
        scope_id="scope",
        outer_fold="outer",
        fraction=1.0,
        cutoff_kst="2025-01-01T00:00:00+09:00",
        pipeline_seed=1,
        phase="inner_1",
        component="depth_query_bitcn",
        child_seed=22,
        train_last_kst="2025-01-01T00:00:00+09:00",
        validation_start_kst="2025-01-08T00:10:00+09:00",
        embargo_threshold_kst="2025-01-01T00:10:00+09:00",
        training_supervised_time_count=20,
        validation_supervised_time_count=10,
    )
    inner_receipt = {
        "component": "depth_query_bitcn",
        "phase": "inner_1",
        "seed": 22,
        "best_epoch": 1,
        "best_rmse_c": 0.2,
        "parameter_count": 100,
        "history": [
            {
                "epoch": 1,
                "train_mse_c": 0.04,
                "validation_rmse": 0.2,
                "learning_rate": 0.001,
            }
        ],
        "adapter": {
            "schema_version": "p2_authoritative_deep_inner_adapter.v3",
            "inner_fold": "inner_1",
            "panel_time_count": 100,
            "continuous_training_public_time_count": 70,
            "continuous_validation_public_time_count": 30,
            "training_supervised_time_count": 20,
            "validation_supervised_time_count": 10,
            "masked_nonregistered_target_values": 200,
            "training_context_last_kst": "2025-01-01T00:00:00+09:00",
            "validation_context_first_kst": "2025-01-08T00:10:00+09:00",
            "strict_embargo_pass": True,
            "continuous_public_covariates_preserved": True,
            "labels_restricted_to_registered_common_ledger": True,
        },
        "future_or_outer_labels_in_fit": False,
    }
    verifier._verify_job_receipt(inner_receipt, inner_spec, ["checkpoint.pt"])

    full_spec = verifier.JobSpec(
        job_id="scope__s1__full__depth_query_bitcn",
        scope_id="scope",
        outer_fold="outer",
        fraction=1.0,
        cutoff_kst="2025-01-01T00:00:00+09:00",
        pipeline_seed=1,
        phase="full",
        component="depth_query_bitcn",
        child_seed=23,
        prefix_supervised_time_count=20,
    )
    full_receipt = {
        "component": "depth_query_bitcn",
        "phase": "full",
        "seed": 23,
        "epochs": 1,
        "parameter_count": 100,
        "final_train_mse_c": 0.04,
        "adapter": {
            "schema_version": "p2_authoritative_deep_full_adapter.v3",
            "scope_id": "scope",
            "continuous_public_time_count": 100,
            "supervised_time_count": 20,
            "cutoff_kst": "2025-01-01T00:00:00+09:00",
            "later_public_time_count": 0,
            "continuous_public_covariates_preserved": True,
            "labels_restricted_to_registered_common_ledger": True,
        },
        "future_or_outer_labels_in_fit": False,
    }
    verifier._verify_job_receipt(full_receipt, full_spec, ["checkpoint.pt"])
    leaked = json.loads(json.dumps(inner_receipt))
    leaked["adapter"]["labels_restricted_to_registered_common_ledger"] = False
    with pytest.raises(ValueError, match="label ledger"):
        verifier._verify_job_receipt(leaked, inner_spec, ["checkpoint.pt"])
    wrong_best = json.loads(json.dumps(inner_receipt))
    wrong_best["best_epoch"] = 2
    with pytest.raises(ValueError, match="best epoch differs from history"):
        verifier._verify_job_receipt(wrong_best, inner_spec, ["checkpoint.pt"])
    wrong_cutoff = json.loads(json.dumps(full_receipt))
    wrong_cutoff["adapter"]["cutoff_kst"] = "2099-01-01T00:00:00+09:00"
    with pytest.raises(ValueError, match="adapter cutoff"):
        verifier._verify_job_receipt(wrong_cutoff, full_spec, ["checkpoint.pt"])


def test_verifier_has_no_mutable_p2_module_imports() -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert "from p2_restore" not in source
    assert "import p2_restore" not in source


def test_cli_requires_explicit_post_terminal_without_touching_actual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["postverify"])
    monkeypatch.setattr(
        verifier,
        "run_postterminal_verification",
        lambda **_kwargs: pytest.fail("live verifier was invoked without explicit gate"),
    )
    with pytest.raises(ValueError, match="explicit --post-terminal"):
        verifier.main()
