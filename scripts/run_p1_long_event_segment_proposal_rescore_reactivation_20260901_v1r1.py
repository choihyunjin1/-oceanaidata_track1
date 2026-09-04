"""One-pass state-reuse repair for the frozen P1 segment-rescore screen."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "scripts/run_p1_long_event_segment_proposal_rescore_reactivation_20260901_v1.py"
CONFIG_PATH = ROOT / "configs/experiments/p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r1.json"
ARTIFACT_DIR = ROOT / "artifacts/p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r1"
EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r1"


def _load_base() -> ModuleType:
    specification = importlib.util.spec_from_file_location("p1_segment_reactivation_v1_base", BASE_RUNNER)
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load same-process base runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.CONFIG_PATH = CONFIG_PATH
    module.ARTIFACT_DIR = ARTIFACT_DIR
    module.__file__ = str(Path(__file__).resolve())
    return module


base = _load_base()


def _captured_readiness(v6: ModuleType, *, retain_snapshot: bool):
    captured: dict[str, Any] = {}
    original_load = v6._load_snapshot_numerical
    original_strict = v6._strict_target_free_snapshot_readiness

    def capture_load(*args, **kwargs):
        if "numerical" in captured:
            raise RuntimeError("numerical module loaded more than once in one readiness pass")
        numerical, execution, runtime = original_load(*args, **kwargs)
        captured.update(numerical=numerical, execution=execution, runtime=runtime)
        return numerical, execution, runtime

    def capture_strict(*args, **kwargs):
        if "state" in captured:
            raise RuntimeError("readiness state built more than once")
        readiness, state = original_strict(*args, **kwargs)
        captured.update(readiness=readiness, state=state)
        return readiness, state

    v6._load_snapshot_numerical = capture_load
    v6._strict_target_free_snapshot_readiness = capture_strict
    try:
        result, snapshot, records = v6._complete_readiness(retain_snapshot=retain_snapshot)
    finally:
        v6._load_snapshot_numerical = original_load
        v6._strict_target_free_snapshot_readiness = original_strict
    required = {"numerical", "execution", "runtime", "readiness", "state"}
    if set(captured) != required:
        raise RuntimeError(f"one-pass readiness capture incomplete: {sorted(captured)}")
    return result, snapshot, records, captured


def prepare(*, retain_snapshot: bool):
    if ARTIFACT_DIR.exists():
        raise FileExistsError("fresh v1r1 namespace is already consumed")
    config = base._validate_config()
    repair = config.get("same_process_state_reuse_repair", {})
    if repair != {
        "readiness_passes_per_execute": 1,
        "numerical_module_loads_per_execute": 1,
        "execution_module_loads_per_execute": 1,
        "state_builds_per_execute": 1,
        "load_worker_state_calls": 0,
        "reuse_exact_objects_from_readiness": ["numerical", "execution", "state", "readiness_receipt"],
        "scientific_change": False,
    }:
        raise RuntimeError("same-process state-reuse repair contract changed")
    source = base._validate_source(include_readme=True)
    v6 = base._load_v6()
    result, snapshot, records, captured = _captured_readiness(v6, retain_snapshot=retain_snapshot)
    if result["operation_counters"] != {
        "claims": 0,
        "physical_fits": 0,
        "scientific_materializations": 0,
        "outer_scores": 0,
        "candidate_files": 0,
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
        "uploads": 0,
    }:
        raise RuntimeError("one-pass readiness is not zero-operation")
    if v6._selected_readiness(captured["readiness"]) != result["readiness"]:
        raise RuntimeError("captured readiness receipt differs from the published pinned receipt")
    if retain_snapshot and (snapshot is None or records is None):
        raise RuntimeError("retained readiness snapshot is absent")
    public = base._public_preflight(result, source)
    public["same_process_state_reuse"] = {
        "readiness_passes": 1,
        "numerical_module_loads": 1,
        "execution_module_loads": 1,
        "state_builds": 1,
        "load_worker_state_calls": 0,
        "pinned_runtime_receipt_sha256": base._canonical_sha(captured["runtime"]),
    }
    public["verification_sha256"] = base._canonical_sha(
        {key: value for key, value in public.items() if key != "verification_sha256"}
    )
    return public, v6, snapshot, result, captured


def execute():
    started = base.time.time()
    public, v6, snapshot, _result, captured = prepare(retain_snapshot=True)
    assert snapshot is not None
    journal = None
    phase = "ONE_PASS_CAPTURED_STATE_CLAIM"
    try:
        deadline = started + base.HARD_WALL_SECONDS
        journal = base.AttemptJournal(deadline)
        journal.record_readiness(public)
        phase = "FIXED_72_FIT_NUMERICAL_SCREEN"
        closure = base._read_json(snapshot / v6._relative_literal(v6.CLOSURE_V3_PATH))
        execution = captured["execution"]
        previous_id = execution.EXPERIMENT_ID
        execution.EXPERIMENT_ID = EXPERIMENT_ID
        try:
            screen = execution.run_authorized_screen(
                captured["state"], captured["numerical"], closure, journal, deadline
            )
        finally:
            execution.EXPERIMENT_ID = previous_id
        phase = "AGGREGATE_ONLY_PUBLICATION"
        result_path = base._publish(screen, journal, started)
        return result_path, base._read_json(result_path)
    except BaseException as error:
        if journal is not None and journal.lock_path.exists():
            journal.fail(phase, error)
        raise
    finally:
        v6._remove_snapshot_modules(snapshot)
        v6._cleanup_snapshot(snapshot)


def qa() -> dict[str, Any]:
    public, _v6, _snapshot, _result, _captured = prepare(retain_snapshot=False)
    reuse = public["same_process_state_reuse"]
    checks = {
        "zero_operation": all(value == 0 for value in public["operation_counters"].values()),
        "fresh_namespace": public["namespace"]["fresh_artifact_namespace"] is True,
        "exact_single_load": reuse["readiness_passes"] == reuse["numerical_module_loads"] == 1,
        "state_built_once": reuse["state_builds"] == 1,
        "worker_state_reload_absent": reuse["load_worker_state_calls"] == 0,
        "forbidden_source_reads_zero": all(
            public["source"][key] == 0
            for key in ("official_test_reads", "sample_submission_reads", "submission_candidate_reads")
        ),
    }
    return {
        "schema_version": "p1_long_event_segment_proposal_rescore.reactivation_independent_qa.v1r1",
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "preflight_verification_sha256": public["verification_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--qa", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        public, *_ = prepare(retain_snapshot=False)
        print(base._canonical_bytes({"output": public}).decode("utf-8"), end="")
    elif args.qa:
        print(base._canonical_bytes(qa()).decode("utf-8"), end="")
    else:
        path, result = execute()
        print(
            base._canonical_bytes(
                {
                    "status": "complete",
                    "result_path": str(path),
                    "decision": result["decision"],
                    "f1_delta": result["metrics"]["pooled"]["f1_delta"],
                }
            ).decode("utf-8"),
            end="",
        )


if __name__ == "__main__":
    main()
