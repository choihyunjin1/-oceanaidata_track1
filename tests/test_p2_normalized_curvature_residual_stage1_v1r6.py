from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_restore.normalized_curvature_residual import (
    PUBLIC_LAYERS,
    build_normalized_curvature_design,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts/run_p2_normalized_curvature_residual_stage1_v1r6.py"
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/p2_normalized_curvature_residual_lgbm_stage1_v1r6.json"
)
V1R4_CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/p2_normalized_curvature_residual_lgbm_stage1_v1r4.json"
)
V1R5_CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/p2_normalized_curvature_residual_lgbm_stage1_v1r5.json"
)
V1R3_CLAIM_PATH = (
    PROJECT_ROOT
    / "artifacts/_ncr_stage1_claims/p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r3.claim.json"
)
V1R3_JOURNAL_PATH = (
    PROJECT_ROOT
    / "artifacts/_ncr_stage1_attempt_journals/p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r3.ndjson"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("ncr_stage1_v1r6_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature_frame() -> pd.DataFrame:
    rows = 4
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-08-01", periods=rows, freq="h", tz="UTC"),
            "layer": [2, 3, 4, 2],
            "target": [9.6, 9.1, 8.2, 9.4],
            "baseline": [9.5, 9.0, 8.0, 9.2],
            "target_depth": [5.0, 10.0, 15.0, 5.0],
            "public_temp_range": [2.0, 0.2, 3.0, 1.5],
            "public_temp_count": [5, 4, 5, 5],
            "doy_sin": [0.1, 0.2, 0.3, 0.4],
            "doy_cos": [0.9, 0.8, 0.7, 0.6],
            "hour_sin": [0.0, 0.2, 0.4, 0.6],
            "hour_cos": [1.0, 0.8, 0.6, 0.4],
            "m2_sin": [0.2, 0.3, 0.4, 0.5],
            "m2_cos": [0.8, 0.7, 0.6, 0.5],
        }
    )
    nominal = {1: 0.0, 5: 20.0, 6: 30.0, 7: 40.0, 8: 50.0}
    temperatures = {
        1: [10.5, np.nan, 10.0, 10.0],
        5: [8.5, 8.8, 7.0, 8.5],
        6: [8.0, 8.9, 6.9, 8.4],
        7: [7.8, 8.7, 6.8, 8.3],
        8: [7.5, 8.6, 6.7, 8.2],
    }
    for public_layer in PUBLIC_LAYERS:
        frame[f"temp_{public_layer}"] = temperatures[public_layer]
        frame[f"psal_{public_layer}"] = np.asarray(
            [34.0, 34.1, 34.2, 34.3], dtype=float
        ) + public_layer * 0.001
        frame[f"nominal_{public_layer}"] = nominal[public_layer]
        frame[f"depth_{public_layer}"] = nominal[public_layer] + np.asarray(
            [0.0, 0.1, -0.1, 0.2]
        )
    return frame


def _dummy_event_base() -> dict[str, str]:
    return {
        "contract_sha256": "c" * 64,
        "bundle_sha256": "b" * 64,
        "attempt_token_sha256": "t" * 64,
    }


def _valid_v1r6_prelaunch_events(runner):
    parent_pid = 4242
    started_epoch = 1000.0
    deadline_epoch = 2800.0
    readiness_sha256 = "r" * 64
    claimed = runner._journal_event(
        "ATTEMPT_CLAIMED",
        **_dummy_event_base(),
        journal_schema=runner.PRELAUNCH_JOURNAL_SCHEMA,
        parent_pid=parent_pid,
        started_epoch=started_epoch,
        deadline_epoch=deadline_epoch,
        literal_readiness_sha256=readiness_sha256,
        claim_directory_fsync_supported=False,
        physical_fit_slots=[
            {"slot": slot, "seed": seed, "status": "UNRESERVED"}
            for slot, seed in enumerate((20260823, 20260824, 20260825), start=1)
        ],
    )
    launching = runner._journal_event(
        "PARENT_LAUNCHING_SINGLE_WORKER",
        **_dummy_event_base(),
        journal_schema=runner.PRELAUNCH_JOURNAL_SCHEMA,
        parent_pid=parent_pid,
        worker_restart_count=0,
        hard_wall_seconds=1800.0,
        literal_readiness_sha256=readiness_sha256,
    )
    expected = {
        **_dummy_event_base(),
        "parent_pid": parent_pid,
        "deadline_epoch": deadline_epoch,
        "literal_readiness_sha256": readiness_sha256,
        "seeds": [20260823, 20260824, 20260825],
        "hard_wall_seconds": 1800.0,
    }
    return [claimed, launching], expected


def test_actual_v1r3_incident_is_pinned_and_proves_zero_fit() -> None:
    runner = _load_runner_module()
    assert V1R3_CLAIM_PATH.stat().st_size == 1908
    assert runner._sha256(V1R3_CLAIM_PATH) == (
        "377e43a54f22ee1c9232086f7e7acfbdb5babc5575d16ea62b11f22eace8e448"
    )
    assert V1R3_JOURNAL_PATH.stat().st_size == 1889
    assert runner._sha256(V1R3_JOURNAL_PATH) == (
        "eb95c80d4b3f429d49cf3c8d66181b920b3ff4b0938c07debb15a6a72c88839f"
    )
    events = runner._read_journal(V1R3_JOURNAL_PATH)
    assert [event["event"] for event in events] == [
        "ATTEMPT_CLAIMED",
        "PARENT_LAUNCHING_SINGLE_WORKER",
        "ATTEMPT_TERMINAL_FAILED",
    ]
    assert not any(event["event"] == "WORKER_STARTED" for event in events)
    assert not any(event["event"].startswith("FIT_SLOT_") for event in events)
    assert runner._fit_slot_states(events) == {
        1: "UNRESERVED",
        2: "UNRESERVED",
        3: "UNRESERVED",
    }
    terminal = events[-1]
    assert terminal["worker_stdout_sha256"] == hashlib.sha256(b"").hexdigest()
    assert terminal["worker_stderr_sha256"] == (
        "2af5a3248d326ef7973312db334bc231174a9b751de2e3e42d6ae8e6d2112d2b"
    )
    # The legacy v1r3 guard required len == 1 after parent had written two events.
    assert len(events[:2]) == 2
    assert not (len(events[:2]) == 1 and events[0]["event"] == "ATTEMPT_CLAIMED")


def test_v1r6_accepts_exact_two_event_prelaunch_contract() -> None:
    runner = _load_runner_module()
    events, expected = _valid_v1r6_prelaunch_events(runner)
    runner._validate_prelaunch_journal_events(events, **expected)


@pytest.mark.parametrize("mutation", ["extra", "missing", "reordered"])
def test_v1r6_rejects_any_nonexact_prelaunch_event_sequence(mutation: str) -> None:
    runner = _load_runner_module()
    events, expected = _valid_v1r6_prelaunch_events(runner)
    if mutation == "extra":
        events.append(runner._journal_event("UNEXPECTED", **_dummy_event_base()))
    elif mutation == "missing":
        events.pop()
    else:
        events.reverse()
    with pytest.raises(RuntimeError):
        runner._validate_prelaunch_journal_events(events, **expected)


def _worker_failure_fixture(runner, tmp_path: Path, scenario: str):
    journal = tmp_path / "worker_attempt.ndjson"
    claim = tmp_path / "worker.claim.json"
    final = tmp_path / "worker_final"
    claim.write_text("{}\n", encoding="utf-8")
    prelaunch, _expected = _valid_v1r6_prelaunch_events(runner)
    runner._exclusive_create_bytes(
        journal,
        b"".join(runner._journal_line(event) for event in prelaunch),
    )
    runner._append_journal_event(
        journal,
        runner._journal_event("WORKER_STARTED", **_dummy_event_base()),
    )
    if scenario in {"slot1_reserved", "slot1_completed", "all_completed"}:
        runner._reserve_fit_slot(
            journal,
            slot=1,
            seed=20260823,
            **_dummy_event_base(),
        )
    if scenario in {"slot1_completed", "all_completed"}:
        runner._complete_fit_slot(
            journal,
            slot=1,
            seed=20260823,
            elapsed_seconds=0.01,
            **_dummy_event_base(),
        )
    if scenario == "all_completed":
        for slot, seed in ((2, 20260824), (3, 20260825)):
            runner._reserve_fit_slot(
                journal,
                slot=slot,
                seed=seed,
                **_dummy_event_base(),
            )
            runner._complete_fit_slot(
                journal,
                slot=slot,
                seed=seed,
                elapsed_seconds=0.01,
                **_dummy_event_base(),
            )
    phase = {
        "after_worker_started": "WORKER_STARTED",
        "slot1_reserved": "FIT_SLOT_1_RESERVED_BEFORE_CALL",
        "slot1_completed": "FIT_SLOT_1_COMPLETED",
        "all_completed": "ALL_FITS_COMPLETED_BUILDING_RESULT",
    }[scenario]
    authorization = {
        "paths": {"journal": journal, "claim": claim, "final": final},
        "contract_sha256": "c" * 64,
        "bundle_sha256": "b" * 64,
        "token_sha256": "t" * 64,
        "phase": phase,
    }
    return authorization


@pytest.mark.parametrize(
    ("scenario", "reserved", "completed"),
    [
        ("after_worker_started", 0, 0),
        ("slot1_reserved", 1, 0),
        ("slot1_completed", 1, 1),
        ("all_completed", 3, 3),
    ],
)
def test_postauthorization_worker_failures_record_exactly_once_then_parent_terminal(
    tmp_path: Path, scenario: str, reserved: int, completed: int
) -> None:
    runner = _load_runner_module()
    authorization = _worker_failure_fixture(runner, tmp_path, scenario)
    assert runner._record_worker_failure_best_effort(
        authorization,
        RuntimeError("synthetic worker failure"),
        phase=authorization["phase"],
    )
    assert not runner._record_worker_failure_best_effort(
        authorization,
        RuntimeError("duplicate must not append"),
        phase=authorization["phase"],
    )
    journal = authorization["paths"]["journal"]
    events = runner._read_journal(journal)
    worker_failures = [event for event in events if event["event"] == "WORKER_FAILED"]
    assert len(worker_failures) == 1
    failure = worker_failures[0]
    assert failure["phase"] == authorization["phase"]
    assert failure["physical_reserved_count"] == reserved
    assert failure["physical_completed_count"] == completed
    runner._append_journal_event(
        journal,
        runner._journal_event(
            "ATTEMPT_TERMINAL_FAILED",
            **_dummy_event_base(),
            reason="WORKER_NONZERO_EXIT",
            automatic_rerun_allowed=False,
        ),
    )
    events = runner._read_journal(journal)
    assert [event["event"] for event in events[-2:]] == [
        "WORKER_FAILED",
        "ATTEMPT_TERMINAL_FAILED",
    ]
    state = runner._inspect_paths(authorization["paths"])
    assert state["eligible"] is False
    assert state["terminal_event_count"] == 1


def test_prelaunch_authorization_failure_never_emits_false_worker_failed(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    journal = tmp_path / "bad_prelaunch.ndjson"
    events, expected = _valid_v1r6_prelaunch_events(runner)
    events.append(runner._journal_event("UNEXPECTED", **_dummy_event_base()))
    runner._exclusive_create_bytes(
        journal,
        b"".join(runner._journal_line(event) for event in events),
    )
    with pytest.raises(RuntimeError, match="exactly two"):
        runner._validate_prelaunch_journal_events(
            runner._read_journal(journal),
            **expected,
        )
    assert not any(
        event["event"] == "WORKER_FAILED" for event in runner._read_journal(journal)
    )


def test_worker_failure_logging_failure_reports_stderr_and_preserves_original(
    tmp_path: Path, capsys
) -> None:
    runner = _load_runner_module()
    authorization = {
        "paths": {"journal": tmp_path / "missing.ndjson"},
        "contract_sha256": "c" * 64,
        "bundle_sha256": "b" * 64,
        "token_sha256": "t" * 64,
    }

    def operation():
        raise LookupError("original-worker-error")

    with pytest.raises(LookupError, match="original-worker-error"):
        runner._run_authorized_worker_guarded(
            authorization,
            {"phase": "SYNTHETIC_PHASE"},
            operation,
        )
    assert "WORKER_FAILURE_LOGGING_FAILED" in capsys.readouterr().err


@pytest.mark.parametrize(
    "failure_mode",
    [
        "malformed_stdout",
        "bad_json",
        "contract_mismatch",
        "static_drift",
        "slot_mismatch",
        "publish_failure",
    ],
)
def test_parent_postworker_failures_append_one_terminal_failure_and_forbid_rerun(
    tmp_path: Path, failure_mode: str
) -> None:
    runner = _load_runner_module()
    scenario = "after_worker_started" if failure_mode == "slot_mismatch" else "all_completed"
    authorization = _worker_failure_fixture(runner, tmp_path, scenario)
    paths = authorization["paths"]
    attempt = {
        "paths": paths,
        "contract_sha256": authorization["contract_sha256"],
        "bundle_sha256": authorization["bundle_sha256"],
        "token_sha256": authorization["token_sha256"],
    }
    required = {
        "schema_version": "synthetic.worker.result.v1",
        "experiment_id": "synthetic-experiment",
    }
    valid_payload = json.dumps(required, sort_keys=True) + "\n"
    completed_process = subprocess.CompletedProcess(
        ["synthetic-worker"],
        returncode=0,
        stdout=valid_payload,
        stderr="",
    )

    def implementation(_config_path, _config, _bundle, context):
        context["attempt"] = attempt
        context["completed"] = completed_process
        if failure_mode == "malformed_stdout":
            completed_process.stdout = "{}\n{}\n"
            runner._decode_and_validate_worker_stdout(
                completed_process, required, context
            )
        elif failure_mode == "bad_json":
            completed_process.stdout = "{bad-json}\n"
            runner._decode_and_validate_worker_stdout(
                completed_process, required, context
            )
        elif failure_mode == "contract_mismatch":
            completed_process.stdout = json.dumps(
                {"schema_version": "wrong", "experiment_id": "synthetic-experiment"}
            )
            runner._decode_and_validate_worker_stdout(
                completed_process, required, context
            )
        elif failure_mode == "slot_mismatch":
            runner._assert_three_completed_fit_slots(paths["journal"], context)
        elif failure_mode == "static_drift":
            context["phase"] = "POSTWORKER_STATIC_BUNDLE_REVERIFY"
            raise RuntimeError("synthetic static drift")
        else:
            context["phase"] = "PUBLISHING_AGGREGATE"
            (paths["final"].parent / f".{paths['final'].name}.staging-test").mkdir()
            raise OSError("synthetic staging/publish failure")
        raise AssertionError("failure injection did not raise")

    with pytest.raises((RuntimeError, json.JSONDecodeError, OSError)):
        runner._execute_parent(Path("unused"), {}, {}, implementation=implementation)

    events = runner._read_journal(paths["journal"])
    failures = [
        event for event in events if event["event"] == "ATTEMPT_TERMINAL_FAILED"
    ]
    assert len(failures) == 1
    failure = failures[0]
    assert failure["error_type"] in {"RuntimeError", "JSONDecodeError", "OSError"}
    assert failure["worker_returncode"] == 0
    assert len(failure["journal_prefix_sha256"]) == 64
    assert failure["journal_prefix_bytes"] > 0
    assert failure["final_exists"] is False
    assert failure["staging_count"] == (1 if failure_mode == "publish_failure" else 0)
    assert runner._inspect_paths(paths)["eligible"] is False
    assert runner._inspect_paths(paths)["terminal_event_count"] == 1


def test_parent_success_and_postsuccess_exception_never_append_terminal_failure(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()

    for late_exception in (False, True):
        case = tmp_path / ("late" if late_exception else "success")
        case.mkdir()
        authorization = _worker_failure_fixture(runner, case, "all_completed")
        paths = authorization["paths"]
        attempt = {
            "paths": paths,
            "contract_sha256": authorization["contract_sha256"],
            "bundle_sha256": authorization["bundle_sha256"],
            "token_sha256": authorization["token_sha256"],
        }

        def implementation(
            _config_path,
            _config,
            _bundle,
            context,
            *,
            bound_attempt=attempt,
            bound_paths=paths,
            raise_late=late_exception,
        ):
            context["attempt"] = bound_attempt
            context["phase"] = "APPENDING_TERMINAL_SUCCESS"
            bound_paths["final"].mkdir()
            runner._append_journal_event(
                bound_paths["journal"],
                runner._journal_event(
                    "ATTEMPT_TERMINAL_COMPLETE",
                    **_dummy_event_base(),
                    final_status="SYNTHETIC_COMPLETE",
                    automatic_rerun_allowed=False,
                ),
            )
            context["phase"] = "TERMINAL_SUCCESS"
            if raise_late:
                raise RuntimeError("synthetic exception after terminal success")
            return bound_paths["final"]

        if late_exception:
            with pytest.raises(RuntimeError, match="after terminal success"):
                runner._execute_parent(
                    Path("unused"), {}, {}, implementation=implementation
                )
        else:
            assert runner._execute_parent(
                Path("unused"), {}, {}, implementation=implementation
            ) == paths["final"]
        events = runner._read_journal(paths["journal"])
        assert sum(event["event"] == "ATTEMPT_TERMINAL_COMPLETE" for event in events) == 1
        assert not any(event["event"] == "ATTEMPT_TERMINAL_FAILED" for event in events)


def test_strict_preflight_pins_full_v1r6_bundle_runtime_and_literal_data() -> None:
    runner = _load_runner_module()
    config, bundle = runner._verify_static_bundle(CONFIG_PATH)
    assert config["experiment_id"].endswith("_v1r6")
    assert runner._sha256(CONFIG_PATH) == runner.EXPECTED_CONFIG_SHA256
    assert len(bundle["implementation_pins"]) == 5
    assert len(bundle["superseded_lineage_pins"]) == 22
    assert len(bundle["immutable_references"]) == 3
    packages = bundle["runtime_pins"]["packages"]
    assert packages["scipy"] == "1.18.0"
    assert packages["joblib"] == "1.5.3"
    assert packages["threadpoolctl"] == "3.6.0"
    assert len(bundle["runtime_pins"]["lightgbm_native_files"]) == 2
    readiness = runner._literal_source_readiness(
        config,
        bundle,
        require_environment=False,
        environ={},
    )
    assert readiness["bytes"] == 49058719
    assert readiness["sha256"] == (
        "cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a"
    )
    assert readiness["permanent_claim_created"] is False
    state = runner._inspect_control_state(config)
    assert state["claim_exists"] is False
    assert state["journal_exists"] is False
    assert state["journal_init_failure_exists"] is False
    assert state["postpublish_failure_exists"] is False
    assert state["final_exists"] is False
    assert state["orphan_or_active_staging_count"] == 0


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({}, "P2_DATA_DIR is required"),
        ({"P2_DATA_DIR": r"C:\wrong\p2\directory"}, "differs from the approved"),
    ],
)
def test_missing_or_wrong_execution_environment_fails_before_claim(
    environment: dict[str, str], message: str
) -> None:
    runner = _load_runner_module()
    config, bundle = runner._verify_static_bundle(CONFIG_PATH)
    claim_calls: list[object] = []

    def forbidden_claim(*args):
        claim_calls.append(args)
        raise AssertionError("claim must not be reached")

    with pytest.raises(RuntimeError, match=message):
        runner._prepare_readiness_then_acquire(
            config,
            bundle,
            environ=environment,
            claim_fn=forbidden_claim,
        )
    assert claim_calls == []


def test_wrong_literal_observations_path_fails_before_claim() -> None:
    runner = _load_runner_module()
    config, bundle = runner._verify_static_bundle(CONFIG_PATH)
    altered = json.loads(json.dumps(config))
    approved_parent = altered["data_contract"]["approved_literal_data_directory"]
    altered["data_contract"]["approved_literal_observations_path"] = str(
        Path(approved_parent) / "wrong_observations.csv"
    )
    claim_calls: list[object] = []

    def forbidden_claim(*args):
        claim_calls.append(args)
        raise AssertionError("claim must not be reached")

    with pytest.raises((FileNotFoundError, RuntimeError)):
        runner._prepare_readiness_then_acquire(
            altered,
            bundle,
            environ={"P2_DATA_DIR": approved_parent},
            claim_fn=forbidden_claim,
        )
    assert claim_calls == []


def test_runner_imports_no_numerical_module_before_preflight() -> None:
    code = f"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('ncr_v1r6_isolation', {str(RUNNER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
blocked = sorted(name for name in sys.modules if name in {{'numpy', 'pandas', 'lightgbm'}} or name.startswith('p2_restore'))
print(json.dumps(blocked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == []


def test_actual_design_columns_equal_sealed_allow_list_exactly() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    design = build_normalized_curvature_design(_feature_frame())
    expected = config["feature_contract"]["allowed_feature_columns"]
    assert list(design.features.columns) == expected
    assert {
        "log1p_profile_scale",
        "log1p_psal_scale",
        "log1p_depth_scale",
    }.issubset(expected)
    assert any("profile_scale^-2" in item for item in config["scientific_limitations"])


def test_v1r6_preserves_v1r5_scientific_model_feature_seed_and_gate_contract() -> None:
    old = json.loads(V1R4_CONFIG_PATH.read_text(encoding="utf-8"))
    v1r5 = json.loads(V1R5_CONFIG_PATH.read_text(encoding="utf-8"))
    new = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for name in (
        "stage1_split",
        "target_contract",
        "model",
        "metrics",
        "stage1_gate",
        "stage2_gate_contract_not_executed_by_this_runner",
    ):
        assert new[name] == old[name]
        assert new[name] == v1r5[name]
    assert new["selection_policy"]["stage1_model_fit_count"] == old["selection_policy"][
        "stage1_model_fit_count"
    ]
    for name in (
        "public_layers",
        "target_layers",
        "forbidden_features",
        "salinity_scale_floor",
        "depth_scale_floor_m",
    ):
        assert new["feature_contract"][name] == old["feature_contract"][name]
    assert new["implementation_pins"]["normalized_curvature_module"]["sha256"] == old[
        "implementation_pins"
    ]["normalized_curvature_module"]["sha256"]
    assert new["feature_contract"] == old["feature_contract"]
    assert new["superseded_execution_forbidden"] == {
        "enforced": True,
        "experiment_ids": [
            "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1",
            "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r2",
            "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r3",
            "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r4",
            "p2_normalized_curvature_residual_lgbm_stage1_20260826_v1r5",
        ],
        "policy": (
            "Only independently-QA-approved v1r6 may execute; v1, v1r2, terminal "
            "infrastructure-failed v1r3, and clean superseded v1r4/v1r5 must never "
            "execute."
        ),
    }


def test_exclusive_claim_has_exactly_one_concurrent_winner(tmp_path: Path) -> None:
    runner = _load_runner_module()
    claim = tmp_path / "stable.claim.json"
    barrier = threading.Barrier(2)

    def attempt(index: int) -> str:
        barrier.wait(timeout=5)
        try:
            runner._exclusive_create_json(claim, {"winner": index})
        except FileExistsError:
            return "LOST_FAIL_CLOSED"
        return "WON"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, (1, 2)))
    assert sorted(outcomes) == ["LOST_FAIL_CLOSED", "WON"]
    assert json.loads(claim.read_text(encoding="utf-8"))["winner"] in {1, 2}


def test_crash_after_durable_reservation_forbids_rerun_and_duplicate_slot(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    paths = {
        "final": tmp_path / "final",
        "claim": tmp_path / "stable.claim.json",
        "journal": tmp_path / "attempt.ndjson",
    }
    runner._exclusive_create_json(paths["claim"], {"permanent": True})
    initial = runner._journal_event(
        "ATTEMPT_CLAIMED",
        **_dummy_event_base(),
        physical_fit_slots=[
            {"slot": slot, "seed": 100 + slot, "status": "UNRESERVED"}
            for slot in (1, 2, 3)
        ],
    )
    runner._exclusive_create_bytes(paths["journal"], runner._journal_line(initial))
    runner._reserve_fit_slot(
        paths["journal"], slot=1, seed=101, **_dummy_event_base()
    )
    state = runner._inspect_paths(paths)
    assert state["eligible"] is False
    assert state["status"] == "INCOMPLETE_OR_CONCURRENT_ATTEMPT_NO_RERUN"
    assert runner._fit_slot_states(runner._read_journal(paths["journal"]))[1] == "RESERVED"
    with pytest.raises(RuntimeError, match="already consumed"):
        runner._reserve_fit_slot(
            paths["journal"], slot=1, seed=101, **_dummy_event_base()
        )


def test_torn_journal_and_orphan_staging_are_reported_fail_closed(tmp_path: Path) -> None:
    runner = _load_runner_module()
    final = tmp_path / "result_bundle"
    claim = tmp_path / "claim"
    journal = tmp_path / "journal.ndjson"
    claim.write_text("claim", encoding="utf-8")
    journal.write_bytes(b'{"event":"ATTEMPT_CLAIMED"')
    (tmp_path / f".{final.name}.staging-orphan").mkdir()
    state = runner._inspect_paths({"final": final, "claim": claim, "journal": journal})
    assert state["eligible"] is False
    assert state["status"] == "CORRUPT_OR_TORN_JOURNAL_NO_RERUN"
    assert state["journal_parse_error"] is not None
    assert state["orphan_or_active_staging_count"] == 1


def test_hash_read_binding_keeps_verified_bytes_after_path_mutation(tmp_path: Path) -> None:
    runner = _load_runner_module()
    source = tmp_path / "observations.csv"
    original = b"x,y\n1,2\n"
    source.write_bytes(original)
    expected_sha = hashlib.sha256(original).hexdigest()
    with runner._held_verified_bytes(
        source, expected_bytes=len(original), expected_sha256=expected_sha
    ) as (handle, captured, digest):
        source.write_bytes(b"mutated path bytes")
        assert handle.closed is False
        assert io.BytesIO(captured).read() == original
        assert digest == {"bytes": len(original), "sha256": expected_sha}
        runner._verify_captured_source(
            captured,
            expected_bytes=len(original),
            expected_sha256=expected_sha,
            label="synthetic",
        )
    assert source.read_bytes() != captured


def test_parent_timeout_uses_only_remaining_wall_and_fails_closed() -> None:
    runner = _load_runner_module()
    observed: dict[str, float] = {}

    def fake_run(command, **kwargs):
        observed["timeout"] = float(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(runner.HardWallTimeout):
        runner._run_subprocess_with_deadline(
            ["synthetic-worker"],
            environment={},
            deadline_epoch=105.0,
            run_fn=fake_run,
            now_fn=lambda: 100.0,
        )
    assert observed["timeout"] == pytest.approx(5.0)


def test_three_slots_can_each_be_reserved_and_completed_only_once(tmp_path: Path) -> None:
    runner = _load_runner_module()
    journal = tmp_path / "attempt.ndjson"
    prelaunch, expected = _valid_v1r6_prelaunch_events(runner)
    runner._exclusive_create_bytes(
        journal,
        b"".join(runner._journal_line(event) for event in prelaunch),
    )
    runner._validate_prelaunch_journal_events(
        runner._read_journal(journal),
        **expected,
    )
    runner._append_journal_event(
        journal,
        runner._journal_event("WORKER_STARTED", **_dummy_event_base()),
    )
    for slot in (1, 2, 3):
        runner._reserve_fit_slot(journal, slot=slot, seed=100 + slot, **_dummy_event_base())
        runner._complete_fit_slot(
            journal,
            slot=slot,
            seed=100 + slot,
            elapsed_seconds=0.01,
            **_dummy_event_base(),
        )
    assert runner._fit_slot_states(runner._read_journal(journal)) == {
        1: "COMPLETED",
        2: "COMPLETED",
        3: "COMPLETED",
    }
    with pytest.raises(RuntimeError):
        runner._reserve_fit_slot(journal, slot=3, seed=103, **_dummy_event_base())
    with pytest.raises(RuntimeError):
        runner._reserve_fit_slot(journal, slot=4, seed=104, **_dummy_event_base())


def test_atomic_publish_exposes_commit_marker_last_three_file_final(tmp_path: Path) -> None:
    runner = _load_runner_module()
    final = tmp_path / "final_artifact"
    durability = runner._publish_aggregate(
        final,
        {"status": "SYNTHETIC"},
        {"schema_version": "synthetic.manifest.v1"},
    )
    assert final.is_dir()
    assert sorted(path.name for path in final.iterdir()) == [
        "manifest.json",
        "result.json",
        "terminal_success.json",
    ]
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["result"]["sha256"] == runner._sha256(final / "result.json")
    assert not list(tmp_path.glob(".final_artifact.staging-*"))
    verified = runner._validate_published_success_inventory(final)
    assert len(verified["marker_sha256"]) == 64
    assert durability["terminal_success_inventory_verified"] is True
    assert durability["destination_overwrite_forbidden"] is True


def test_claim_survives_journal_init_fault_with_exact_once_out_of_band_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    experiment_id = "synthetic-v1r6-journal-init-fault"
    config = {
        "experiment_id": experiment_id,
        "execution": {"hard_wall_seconds": 1800},
        "model": {"seeds": [20260823, 20260824, 20260825]},
        "output": {"directory": f"artifacts/{experiment_id}"},
        "controls": {
            "claim_path": f"artifacts/_ncr_stage1_claims/{experiment_id}.claim.json",
            "journal_path": f"artifacts/_ncr_stage1_attempt_journals/{experiment_id}.ndjson",
            "journal_init_failure_path": (
                f"artifacts/_ncr_stage1_terminal_receipts/{experiment_id}.journal.json"
            ),
            "postpublish_failure_path": (
                f"artifacts/_ncr_stage1_terminal_receipts/{experiment_id}.publish.json"
            ),
        },
    }
    bundle = {"config": {"sha256": "c" * 64}, "bundle_sha256": "b" * 64}
    readiness = {
        "experiment_id": experiment_id,
        "contract_sha256": "c" * 64,
        "bundle_sha256": "b" * 64,
        "execution_environment": {
            "present": True,
            "matches_approved_literal_parent": True,
        },
    }
    readiness["readiness_sha256"] = runner._canonical_sha256(readiness)
    context: dict[str, object] = {}

    def fail_journal_create(_path: Path, _payload: bytes) -> bool:
        raise OSError("injected journal initialization fault")

    with pytest.raises(OSError, match="journal initialization fault") as caught:
        runner._acquire_attempt(
            config,
            bundle,
            readiness,
            parent_context=context,
            journal_create_fn=fail_journal_create,
        )
    attempt = context["attempt"]
    assert isinstance(attempt, dict)
    paths = attempt["paths"]
    assert paths["claim"].is_file()
    assert not paths["journal"].exists()
    assert paths["journal_init_failure"].is_file()
    receipt_before = paths["journal_init_failure"].read_bytes()
    receipt = json.loads(receipt_before)
    assert receipt["reason"] == "JOURNAL_INITIALIZATION_FAILED_AFTER_EXCLUSIVE_CLAIM"
    assert receipt["claim"]["sha256"] == hashlib.sha256(
        paths["claim"].read_bytes()
    ).hexdigest()
    assert receipt["physical_reserved_count"] == 0
    assert runner._record_parent_terminal_failure_best_effort(
        context, caught.value
    ) is False
    assert paths["journal_init_failure"].read_bytes() == receipt_before
    state = runner._inspect_paths(paths)
    assert state["status"] == "TERMINAL_JOURNAL_INIT_FAILURE_NO_RERUN"
    assert state["eligible"] is False


def test_commit_marker_makes_publish_before_journal_terminal_fault_unambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    final = tmp_path / "artifacts" / "synthetic-final"
    claim = tmp_path / "artifacts" / "_ncr_stage1_claims" / "synthetic.claim.json"
    journal = (
        tmp_path / "artifacts" / "_ncr_stage1_attempt_journals" / "synthetic.ndjson"
    )
    terminal_root = tmp_path / "artifacts" / "_ncr_stage1_terminal_receipts"
    for parent in (final.parent, claim.parent, journal.parent, terminal_root):
        parent.mkdir(parents=True, exist_ok=True)
    claim_payload = b'{"claim":"synthetic"}\n'
    claim.write_bytes(claim_payload)
    journal.write_bytes(
        runner._journal_line(runner._journal_event("ATTEMPT_CLAIMED", **_dummy_event_base()))
    )
    paths = {
        "final": final,
        "claim": claim,
        "journal": journal,
        "journal_init_failure": terminal_root / "synthetic.journal.json",
        "postpublish_failure": terminal_root / "synthetic.publish.json",
    }
    attempt = {
        "experiment_id": "synthetic-experiment",
        "paths": paths,
        "contract_sha256": "c" * 64,
        "bundle_sha256": "b" * 64,
        "token_sha256": "t" * 64,
        "claim_bytes": len(claim_payload),
        "claim_sha256": hashlib.sha256(claim_payload).hexdigest(),
    }

    def implementation(_config_path, _config, _bundle, context):
        context["attempt"] = attempt
        runner._publish_aggregate(
            final,
            {"experiment_id": "synthetic-experiment", "status": "COMPLETE"},
            {
                "experiment_id": "synthetic-experiment",
                "status": "COMPLETE",
                "config": {"sha256": "c" * 64},
                "bundle_sha256": "b" * 64,
            },
        )
        context["phase"] = "APPENDING_TERMINAL_SUCCESS"
        raise OSError("injected terminal journal append fault")

    assert runner._execute_parent(
        Path("unused"), {}, {}, implementation=implementation
    ) == final
    inventory = runner._validate_published_success_inventory(
        final,
        expected_experiment_id="synthetic-experiment",
        expected_contract_sha256="c" * 64,
        expected_bundle_sha256="b" * 64,
    )
    assert len(inventory["marker_sha256"]) == 64
    assert not paths["postpublish_failure"].exists()
    assert not any(
        event["event"].startswith("ATTEMPT_TERMINAL_")
        for event in runner._read_journal(journal)
    )


def test_final_directory_and_file_target_races_never_overwrite(tmp_path: Path) -> None:
    runner = _load_runner_module()
    final = tmp_path / "race-final"

    def race_reservation(path: Path) -> None:
        path.mkdir()
        (path / "racer.txt").write_bytes(b"racer-owned")
        runner._reserve_final_directory_create_only(path)

    with pytest.raises(FileExistsError):
        runner._publish_aggregate(
            final,
            {"status": "SYNTHETIC"},
            {"schema_version": "synthetic.manifest.v1"},
            reserve_final_fn=race_reservation,
        )
    assert (final / "racer.txt").read_bytes() == b"racer-owned"
    assert not (final / "terminal_success.json").exists()

    file_race_final = tmp_path / "file-race-final"

    def race_first_link(source: Path, destination: Path) -> None:
        if destination.name == "result.json":
            destination.write_bytes(b"concurrent-writer")
        runner._hardlink_create_only(source, destination)

    with pytest.raises(runner.ExistingAttemptError):
        runner._publish_aggregate(
            file_race_final,
            {"status": "SYNTHETIC"},
            {"schema_version": "synthetic.manifest.v1"},
            link_fn=race_first_link,
        )
    assert (file_race_final / "result.json").read_bytes() == b"concurrent-writer"
    assert not (file_race_final / "terminal_success.json").exists()


def test_hard_wall_timeout_kills_and_verifies_worker_descendant_tree() -> None:
    runner = _load_runner_module()
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        "time.sleep(60)"
    )
    with pytest.raises(runner.HardWallTimeout) as caught:
        runner._run_subprocess_with_deadline(
            [sys.executable, "-c", parent_code],
            environment=os.environ,
            deadline_epoch=time.time() + 2.0,
        )
    evidence = caught.value.termination_evidence
    assert evidence["root_terminated_verified"] is True
    assert evidence["all_captured_processes_terminated"] is True
    descendant_key = (
        "descendant_pids_before_taskkill"
        if os.name == "nt"
        else "descendant_pids_before_killpg"
    )
    assert evidence[descendant_key]
    assert all(evidence["descendant_termination_verified"].values())
    if os.name == "nt":
        assert evidence["taskkill_returncode"] == 0
        assert len(evidence["taskkill_stdout_sha256"]) == 64
