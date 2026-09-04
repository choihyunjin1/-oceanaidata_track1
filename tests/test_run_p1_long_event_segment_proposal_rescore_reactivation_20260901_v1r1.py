from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r1.py"
CONFIG = ROOT / "configs/experiments/p1_long_event_segment_proposal_rescore_reactivation_20260901_v1r1.json"


def _load():
    specification = importlib.util.spec_from_file_location("p1_segment_reactivation_v1r1", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_contract_repairs_only_state_reload_and_freezes_science() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    repair = payload["same_process_state_reuse_repair"]
    assert payload["experiment_id"].endswith("_v1r1")
    assert payload["scientific_experiment_id"] == "p1_long_event_segment_proposal_rescore_20260826_v1"
    assert repair["readiness_passes_per_execute"] == 1
    assert repair["load_worker_state_calls"] == 0
    assert repair["scientific_change"] is False
    assert payload["frozen_operation_graph"]["maximum_lifetime_physical_fits"] == 72
    assert payload["frozen_operation_graph"]["scientific_materializations"] == 21
    assert payload["unchanged_science"]["decision_gates"] is True


def test_launcher_never_calls_reload_or_process_auth_plumbing() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "._load_worker_state(" not in source
    assert "import subprocess" not in source
    assert "--worker" not in source
    assert "run_authorized_screen" in source


def test_one_pass_capture_reuses_nonempty_runtime_and_state() -> None:
    module = _load()
    calls = {"load": 0, "strict": 0, "worker_reload": 0}
    numerical = object()
    execution = object()
    state = {"identity": object()}
    runtime = {"loaded_runtime_file_count": 8, "loaded_runtime_file_aggregate_sha256": "a" * 64}
    readiness = {"status": "PASS", "runtime": runtime}

    def load(*args, **kwargs):
        calls["load"] += 1
        return numerical, execution, runtime

    def strict(*args, **kwargs):
        calls["strict"] += 1
        return readiness, state

    fake = SimpleNamespace(_load_snapshot_numerical=load, _strict_target_free_snapshot_readiness=strict)

    def complete(*, retain_snapshot):
        loaded_numerical, loaded_execution, loaded_runtime = fake._load_snapshot_numerical()
        observed_readiness, observed_state = fake._strict_target_free_snapshot_readiness()
        assert loaded_numerical is numerical
        assert loaded_execution is execution
        assert loaded_runtime is runtime
        assert observed_readiness is readiness
        assert observed_state is state
        return {"readiness": readiness}, Path("snapshot"), {"one": {}}

    fake._complete_readiness = complete
    fake._load_worker_state = lambda *args, **kwargs: calls.__setitem__(
        "worker_reload", calls["worker_reload"] + 1
    )
    result, snapshot, records, captured = module._captured_readiness(fake, retain_snapshot=True)
    assert result == {"readiness": readiness}
    assert snapshot == Path("snapshot")
    assert records == {"one": {}}
    assert captured["numerical"] is numerical
    assert captured["execution"] is execution
    assert captured["state"] is state
    assert captured["runtime"] is runtime
    assert calls == {"load": 1, "strict": 1, "worker_reload": 0}
