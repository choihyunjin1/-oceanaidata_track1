from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from p2_restore import authoritative_nested_surrogate_execution as v2
from p2_restore import authoritative_nested_surrogate_execution_v4 as v4
from p2_restore.authoritative_nested_surrogate_execution_v4 import (
    DeterministicExecutionClosed,
    ExecutionBindingV4,
    ResumeBudgetExhausted,
    SemanticPreflightOutcomeV4,
    TERMINAL_STATUS,
    TerminalExecutionClosed,
    TransientExecutionError,
    inspect_actual_namespace_read_only,
    run_resumable_execution_v4,
)


class SimulatedProcessDeath(BaseException):
    """Bypass graceful exception receipts like an abruptly killed process."""


def _binding(namespace: str, *, seal: str = "c" * 64) -> ExecutionBindingV4:
    return ExecutionBindingV4(
        namespace=namespace,
        execution_contract_sha256="a" * 64,
        parent_recipe_sha256="b" * 64,
        preexecution_seal_sha256=seal,
        semantic_preflight_sha256="d" * 64,
        exact_command_sha256="e" * 64,
        authorization_sha256="f" * 64,
        module_sha256="1" * 64,
        runner_sha256="2" * 64,
    )


def _preflight(context: object = None):
    def run() -> SemanticPreflightOutcomeV4:
        return SemanticPreflightOutcomeV4("d" * 64, context)

    return run


def _result() -> dict[str, object]:
    return {
        "status": TERMINAL_STATUS,
        "metrics_by_prefix": {"040": {"rmse": 1.0}},
        "submission_files_generated": 0,
        "uploads": 0,
    }


def _clock() -> str:
    return "2026-08-25T23:45:00+09:00"


def _job_product(token: int) -> v2.JobProduct:
    return v2.JobProduct(
        frame=pd.DataFrame(
            {"station": ["S-ORS"], "layer": [2], "time": ["2024-01-01T00:00:00+00:00"], "truth": [1.0], "prediction": [float(token)]}
        ),
        receipt={"token": token},
        artifacts={"checkpoint.bin": f"checkpoint-{token}".encode()},
    )


def test_crash_after_start_resumes_same_command_and_terminal_closes(tmp_path: Path) -> None:
    actual = tmp_path / "actual_v4"
    binding = _binding(actual.name)
    start_bytes: bytes | None = None

    def die(_context: object, _contract: str):
        nonlocal start_bytes
        start_bytes = (actual / "execution_start.json").read_bytes()
        raise SimulatedProcessDeath("hard stop after immutable start")

    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=die,
            now=_clock,
        )
    interrupted = inspect_actual_namespace_read_only(actual, binding=binding)
    assert interrupted["status"] == "INTERRUPTED_INCOMPLETE_RESUMABLE"
    assert interrupted["total_attempts_started"] == 1
    assert interrupted["automatic_resume_permitted"] is True

    completed = run_resumable_execution_v4(
        actual_dir=actual,
        binding=binding,
        semantic_preflight=_preflight(),
        execute_curve=lambda _context, _contract: _result(),
        now=_clock,
    )
    assert completed["status"] == "TERMINAL_COMPLETE_NO_RERUN"
    assert completed["total_attempts_started"] == 2
    assert (actual / "execution_start.json").read_bytes() == start_bytes
    with pytest.raises(TerminalExecutionClosed):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: _result(),
            now=_clock,
        )


def test_crash_after_n_jobs_verifies_and_reuses_only_v4_contract(tmp_path: Path) -> None:
    actual = tmp_path / "actual_v4_jobs"
    binding = _binding(actual.name)
    fits = {"count": 0}

    def factory(token: int):
        def make() -> v2.JobProduct:
            fits["count"] += 1
            return _job_product(token)

        return make

    def die_after_two(_context: object, contract: str):
        store = v2.JobStore(actual / "jobs", contract_sha256=contract)
        store.materialize("job_a", factory(1))
        store.materialize("job_b", factory(2))
        raise SimulatedProcessDeath("hard stop after two atomic jobs")

    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=die_after_two,
            now=_clock,
        )
    assert fits["count"] == 2

    def resume(_context: object, contract: str):
        store = v2.JobStore(actual / "jobs", contract_sha256=contract)
        store.materialize("job_a", factory(1))
        store.materialize("job_b", factory(2))
        store.materialize("job_c", factory(3))
        assert store.reused_jobs == 2
        assert store.new_jobs == 1
        return _result()

    run_resumable_execution_v4(
        actual_dir=actual,
        binding=binding,
        semantic_preflight=_preflight(),
        execute_curve=resume,
        now=_clock,
    )
    assert fits["count"] == 3
    terminal = json.loads((actual / "terminal_receipt.json").read_text(encoding="utf-8"))
    assert terminal["result_sha256"] == v2.sha256_file(actual / "result.json")


def test_crash_after_all_oof_before_terminal_ignores_stale_partial(tmp_path: Path) -> None:
    actual = tmp_path / "actual_v4_oof"
    binding = _binding(actual.name)
    payload = b"deterministic-oof-bytes"

    def execute(_context: object, _contract: str):
        v2.atomic_write_or_verify(actual / "evaluated_oof_040.parquet", payload)
        return _result()

    def hard_stop(path: Path, _result_value: object) -> None:
        (path / ".result.json.partial.777.deadbeef").write_bytes(b"stale")
        raise SimulatedProcessDeath("hard stop immediately before terminal publish")

    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=execute,
            before_terminal_publish=hard_stop,
            now=_clock,
        )
    audit = inspect_actual_namespace_read_only(actual, binding=binding)
    assert audit["stale_terminal_partials"] == 1
    assert audit["automatic_resume_permitted"] is True
    completed = run_resumable_execution_v4(
        actual_dir=actual,
        binding=binding,
        semantic_preflight=_preflight(),
        execute_curve=execute,
        now=_clock,
    )
    assert completed["result_sha256"] == v2.sha256_file(actual / "result.json")
    assert (actual / ".result.json.partial.777.deadbeef").read_bytes() == b"stale"


def test_concurrent_lock_rejected_before_resume_mutation(tmp_path: Path) -> None:
    actual = tmp_path / "actual_v4_lock"
    binding = _binding(actual.name)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: (_ for _ in ()).throw(
                SimulatedProcessDeath("interrupt")
            ),
            now=_clock,
        )
    with v2.process_lock(actual / "execution.lock"):
        with pytest.raises(RuntimeError, match="holds the lock"):
            run_resumable_execution_v4(
                actual_dir=actual,
                binding=binding,
                semantic_preflight=_preflight(),
                execute_curve=lambda _context, _contract: _result(),
                now=_clock,
            )
    assert not (actual / "attempts" / "resume_attempt_002.json").exists()


def test_graceful_deterministic_failure_closes_without_auto_resume(tmp_path: Path) -> None:
    actual = tmp_path / "actual_v4_deterministic"
    binding = _binding(actual.name)
    with pytest.raises(ValueError, match="deterministic shape mismatch"):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: (_ for _ in ()).throw(
                ValueError("deterministic shape mismatch")
            ),
            now=_clock,
        )
    audit = inspect_actual_namespace_read_only(actual, binding=binding)
    assert audit["status"] == "FAILED_DETERMINISTIC_CLOSED"
    assert audit["automatic_resume_permitted"] is False
    failure = json.loads(
        (actual / "attempts" / "attempt_001_terminal.json").read_text(encoding="utf-8")
    )
    assert failure["exception_type"] == "builtins.ValueError"
    assert len(failure["traceback_sha256"]) == 64
    with pytest.raises(DeterministicExecutionClosed):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: _result(),
            now=_clock,
        )


def test_explicit_transient_failure_may_resume_and_budget_is_two(tmp_path: Path) -> None:
    actual = tmp_path / "actual_v4_budget"
    binding = _binding(actual.name)

    def transient(_context: object, _contract: str):
        raise TransientExecutionError("explicit transient worker interruption")

    for expected_attempts in (1, 2, 3):
        with pytest.raises(TransientExecutionError):
            run_resumable_execution_v4(
                actual_dir=actual,
                binding=binding,
                semantic_preflight=_preflight(),
                execute_curve=transient,
                now=_clock,
            )
        audit = inspect_actual_namespace_read_only(actual, binding=binding)
        assert audit["total_attempts_started"] == expected_attempts
    with pytest.raises(ResumeBudgetExhausted):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: _result(),
            now=_clock,
        )


def test_resume_semantic_failure_records_gate_and_fails_closed(tmp_path: Path) -> None:
    actual = tmp_path / "actual_v4_gate"
    binding = _binding(actual.name)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: (_ for _ in ()).throw(
                SimulatedProcessDeath("interrupt")
            ),
            now=_clock,
        )

    def broken_preflight() -> SemanticPreflightOutcomeV4:
        raise ValueError("semantic contract mismatch")

    with pytest.raises(DeterministicExecutionClosed):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=broken_preflight,
            execute_curve=lambda _context, _contract: _result(),
            now=_clock,
        )
    audit = inspect_actual_namespace_read_only(actual, binding=binding)
    assert audit["status"] == "FAILED_DETERMINISTIC_CLOSED"
    assert audit["resume_attempts_started"] == 0
    assert (actual / "resume_gate_failure.json").is_file()


def test_changed_seal_and_foreign_job_contract_are_refused(tmp_path: Path) -> None:
    actual = tmp_path / "actual_v4_pin"
    binding = _binding(actual.name)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: (_ for _ in ()).throw(
                SimulatedProcessDeath("interrupt")
            ),
            now=_clock,
        )
    with pytest.raises(ValueError, match="binding changed"):
        inspect_actual_namespace_read_only(
            actual, binding=_binding(actual.name, seal="9" * 64)
        )

    foreign = tmp_path / "actual_v4_foreign"
    foreign_binding = _binding(foreign.name)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=foreign,
            binding=foreign_binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: (_ for _ in ()).throw(
                SimulatedProcessDeath("interrupt")
            ),
            now=_clock,
        )
    store = v2.JobStore(foreign / "jobs", contract_sha256="3" * 64)
    store.materialize("v3_job_must_not_reuse", lambda: _job_product(3))
    with pytest.raises(ValueError, match="job contract hash changed"):
        inspect_actual_namespace_read_only(foreign, binding=foreign_binding)


def test_control_receipt_partial_is_preserved_and_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = tmp_path / "actual_v4_control_partial"
    binding = _binding(actual.name)
    real_publish = v2.atomic_write_or_verify
    faulted = {"start": False, "resume": False}

    def fail_start(path: Path, payload: bytes):
        if path.name == "execution_start.json" and not faulted["start"]:
            faulted["start"] = True
            (path.parent / ".execution_start.json.partial.123.deadbeef").write_bytes(
                payload[:11]
            )
            raise SimulatedProcessDeath("crash during start receipt publication")
        return real_publish(path, payload)

    monkeypatch.setattr(v2, "atomic_write_or_verify", fail_start)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: _result(),
            now=_clock,
        )
    prestart = inspect_actual_namespace_read_only(actual, binding=binding)
    assert prestart["status"] == "EMPTY_PRESTART_NAMESPACE"
    assert prestart["stale_control_partials"] == 1

    monkeypatch.setattr(v2, "atomic_write_or_verify", real_publish)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: (_ for _ in ()).throw(
                SimulatedProcessDeath("hard stop after recovered start")
            ),
            now=_clock,
        )

    def fail_resume(path: Path, payload: bytes):
        if path.name == "resume_attempt_002.json" and not faulted["resume"]:
            faulted["resume"] = True
            (path.parent / ".resume_attempt_002.json.partial.124.deadbeef").write_bytes(
                payload[:13]
            )
            raise SimulatedProcessDeath("crash during resume receipt publication")
        return real_publish(path, payload)

    monkeypatch.setattr(v2, "atomic_write_or_verify", fail_resume)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: _result(),
            now=_clock,
        )
    interrupted = inspect_actual_namespace_read_only(actual, binding=binding)
    assert interrupted["resume_attempts_started"] == 0
    assert interrupted["stale_control_partials"] == 1

    monkeypatch.setattr(v2, "atomic_write_or_verify", real_publish)
    completed = run_resumable_execution_v4(
        actual_dir=actual,
        binding=binding,
        semantic_preflight=_preflight(),
        execute_curve=lambda _context, _contract: _result(),
        now=_clock,
    )
    assert completed["total_attempts_started"] == 2
    assert (actual / ".execution_start.json.partial.123.deadbeef").is_file()
    assert (
        actual / "attempts" / ".resume_attempt_002.json.partial.124.deadbeef"
    ).is_file()


def test_result_commit_without_terminal_receipt_uses_finalization_only_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = tmp_path / "actual_v4_finalize_only"
    binding = _binding(actual.name)
    real_publish = v2.atomic_write_or_verify
    execute_calls = {"count": 0}

    def execute(_context: object, _contract: str):
        execute_calls["count"] += 1
        return _result()

    def fail_after_result(path: Path, payload: bytes):
        if path.name == "terminal_receipt.json":
            (path.parent / ".terminal_receipt.json.partial.125.deadbeef").write_bytes(
                payload[:17]
            )
            raise SimulatedProcessDeath("crash after atomic result commit")
        return real_publish(path, payload)

    monkeypatch.setattr(v2, "atomic_write_or_verify", fail_after_result)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=execute,
            now=_clock,
        )
    assert execute_calls["count"] == 1
    pending = inspect_actual_namespace_read_only(actual, binding=binding)
    assert pending["status"] == "TERMINAL_RESULT_NEEDS_FINALIZATION"
    assert pending["total_attempts_started"] == 1

    monkeypatch.setattr(v2, "atomic_write_or_verify", real_publish)
    finalized = run_resumable_execution_v4(
        actual_dir=actual,
        binding=binding,
        semantic_preflight=lambda: (_ for _ in ()).throw(
            AssertionError("semantic preflight must not run during finalization-only recovery")
        ),
        execute_curve=lambda _context, _contract: (_ for _ in ()).throw(
            AssertionError("execute_curve must not run during finalization-only recovery")
        ),
        now=_clock,
    )
    assert finalized["finalization_only_recovery"] is True
    assert finalized["resume_budget_consumed"] is False
    assert finalized["total_attempts_started"] == 1
    assert execute_calls["count"] == 1
    assert (actual / ".terminal_receipt.json.partial.125.deadbeef").is_file()
    closed = inspect_actual_namespace_read_only(actual, binding=binding)
    assert closed["status"] == "TERMINAL_COMPLETE_NO_RERUN"


def test_prelock_enumerate_rename_race_retries_strictly_under_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = tmp_path / "actual_v4_audit_race"
    binding = _binding(actual.name)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: (_ for _ in ()).throw(
                SimulatedProcessDeath("interrupt")
            ),
            now=_clock,
        )
    real_inspect = v4.inspect_actual_namespace_read_only
    calls = {"count": 0}

    def racing_inspect(path: Path, *, binding: ExecutionBindingV4):
        calls["count"] += 1
        if calls["count"] == 1:
            raise FileNotFoundError("simulated atomic rename during advisory enumeration")
        return real_inspect(path, binding=binding)

    monkeypatch.setattr(v4, "inspect_actual_namespace_read_only", racing_inspect)
    completed = run_resumable_execution_v4(
        actual_dir=actual,
        binding=binding,
        semantic_preflight=_preflight(),
        execute_curve=lambda _context, _contract: _result(),
        now=_clock,
    )
    assert completed["status"] == "TERMINAL_COMPLETE_NO_RERUN"
    assert calls["count"] >= 2


def test_crash_after_atomic_start_before_attempt_directory_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = tmp_path / "actual_v4_start_gap"
    binding = _binding(actual.name)
    real_publish = v4._publish_control_json_atomic
    injected = {"done": False}

    def publish_then_die(path: Path, value: object):
        receipt = real_publish(path, value)
        if path.name == "execution_start.json" and not injected["done"]:
            injected["done"] = True
            raise SimulatedProcessDeath("crash after start rename before attempts mkdir")
        return receipt

    monkeypatch.setattr(v4, "_publish_control_json_atomic", publish_then_die)
    with pytest.raises(SimulatedProcessDeath):
        run_resumable_execution_v4(
            actual_dir=actual,
            binding=binding,
            semantic_preflight=_preflight(),
            execute_curve=lambda _context, _contract: _result(),
            now=_clock,
        )
    assert (actual / "execution_start.json").is_file()
    assert not (actual / "attempts").exists()
    audit = inspect_actual_namespace_read_only(actual, binding=binding)
    assert audit["status"] == "INTERRUPTED_INCOMPLETE_RESUMABLE"
    assert audit["stale_control_partials"] == 0

    monkeypatch.setattr(v4, "_publish_control_json_atomic", real_publish)
    completed = run_resumable_execution_v4(
        actual_dir=actual,
        binding=binding,
        semantic_preflight=_preflight(),
        execute_curve=lambda _context, _contract: _result(),
        now=_clock,
    )
    assert completed["total_attempts_started"] == 2
