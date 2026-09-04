from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "run_p1_round_b_nonspike_long_event_residual_v1r2.py"
)
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1r2.json"
)
OLD_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1.json"
)


def _runner():
    name = f"p1_round_b_residual_v1r2_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _new_attempt(runner, artifact: Path):
    attempt = runner.AttemptJournal.begin(
        artifact,
        config_sha256="c" * 64,
        seal_sha256="s" * 64,
        maximum_fits=9,
    )
    return attempt


def test_runner_top_level_is_stdlib_only_before_trust() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden = {"lightgbm", "numpy", "p1_qc", "pandas", "pyarrow", "sklearn"}
    imported_roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
    assert imported_roots.isdisjoint(forbidden)


def test_numerical_loader_fails_before_trust() -> None:
    runner = _runner()
    assert runner._TRUST_VERIFIED is False
    with pytest.raises(RuntimeError, match="before trust verification"):
        runner._load_v1_numerical()


def test_superseding_config_preserves_every_numerical_section_exactly() -> None:
    runner = _runner()
    current = _load_json(CONFIG_PATH)
    old = _load_json(OLD_CONFIG_PATH)
    assert current["experiment_id"].endswith("v1r2")
    for section in runner.NUMERICAL_SECTIONS:
        assert current[section] == old[section], section
    assert current["rescue_decoder"]["probability_threshold"] == 0.8
    assert current["rescue_decoder"]["maximum_anchor_distance_rows"] == 18
    assert current["resource_budget"]["residual_model_fits"] == 9
    assert current["surface"]["seeds"] == [20260813, 20260829, 20260847]


def test_dependency_closure_and_runtime_versions_are_complete() -> None:
    runner = _runner()
    config = _load_json(CONFIG_PATH)
    paths = tuple(config["trust_contract"]["project_dependency_paths"])
    assert paths == runner.REQUIRED_DEPENDENCY_PATHS
    assert tuple(config["trust_contract"]["verification_paths"]) == (
        "tests/test_run_p1_round_b_nonspike_long_event_residual_v1r2.py",
    )
    discovered = runner._discover_project_dependency_closure(
        PROJECT_ROOT / config["trust_contract"]["numerical_entrypoint_path"]
    )
    assert discovered == set(paths)
    required = {
        "src/p1_qc/features.py",
        "src/p1_qc/metrics.py",
        "src/p1_qc/augment.py",
        "src/p1_qc/experiment.py",
        "src/p1_qc/models_tabular.py",
        "src/p1_qc/postprocess.py",
        "src/p1_qc/rules.py",
        "src/p1_qc/splits.py",
        "src/p1_qc/submission.py",
        "src/p1_qc/pipeline.py",
    }
    assert required.issubset(paths)
    expected = config["trust_contract"]["runtime_versions"]
    observed = {
        "python": runner.platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "lightgbm": importlib.metadata.version("lightgbm"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
    }
    assert observed == expected


def test_exclusive_lock_and_incomplete_journal_fail_closed(tmp_path: Path) -> None:
    runner = _runner()
    artifact = tmp_path / "artifact"
    first = _new_attempt(runner, artifact)
    with pytest.raises(FileExistsError):
        _new_attempt(runner, artifact)
    first.abandon()
    lock = artifact / "execution.lock"
    assert lock.exists()

    # Even deliberate stale-lock removal cannot authorize a second lifetime attempt.
    lock.unlink()
    with pytest.raises(FileExistsError, match="attempt journal already exists"):
        _new_attempt(runner, artifact)
    assert not lock.exists()
    assert (artifact / "attempt_journal" / "000_started.json").is_file()


def test_attempt_journal_is_hash_chained_and_tamper_evident(tmp_path: Path) -> None:
    runner = _runner()
    artifact = tmp_path / "artifact"
    attempt = _new_attempt(runner, artifact)
    attempt.record_left_censor_gate({fold: 0 for fold in runner.FOLD_ORDER})
    started_path = artifact / "attempt_journal" / "000_started.json"
    gate_path = artifact / "attempt_journal" / "005_left_censor_gate_passed.json"
    started = _load_json(started_path)
    gate = _load_json(gate_path)
    assert started["previous_entry_sha256"] is None
    assert gate["previous_entry_sha256"] == runner._sha256(started_path)
    attempt.verify_integrity()

    started_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="attempt journal entry changed"):
        attempt.before_fold(0, "2025_q2", 3)
    attempt.abandon()


def test_left_censored_positive_event_gate_counts_and_fails_before_fit() -> None:
    runner = _runner()
    import numpy as np
    import pandas as pd

    train = pd.DataFrame(
        {
            "station": ["A", "A", "B", "B"],
            "layer": [1, 1, 1, 1],
            "time": [
                "2024-01-01T00:10:00Z",
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:10:00Z",
            ],
            "label": [1, 0, 0, 1],
        }
    )
    folds = [
        {"name": fold, "train_idx": np.arange(len(train), dtype=np.int64)}
        for fold in runner.FOLD_ORDER
    ]
    numerical = SimpleNamespace(np=np, pd=pd)
    config = _load_json(CONFIG_PATH)
    counts = runner._enforce_left_censor_gate(train, folds, numerical, config)
    assert counts == {fold: 0 for fold in runner.FOLD_ORDER}

    train.loc[1, "label"] = 1
    with pytest.raises(RuntimeError, match="NO_GO_LEFT_CENSORED.*BEFORE_FIT"):
        runner._enforce_left_censor_gate(train, folds, numerical, config)


def test_atomic_writers_fsync_replace_and_leave_no_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    fsync_calls: list[int] = []
    real_fsync = runner.os.fsync

    def observing_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(runner.os, "fsync", observing_fsync)
    target = tmp_path / "nested" / "record.json"
    runner._atomic_json_new(target, {"value": 7})
    assert _load_json(target) == {"value": 7}
    assert fsync_calls
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))
    with pytest.raises(FileExistsError):
        runner._atomic_json_new(target, {"value": 8})

    class FakeFrame:
        def to_parquet(self, handle, *, index: bool, compression: str) -> None:
            assert index is False
            assert compression == "zstd"
            handle.write(b"PAR1-fake-atomic-payload")

    parquet = tmp_path / "nested" / "part.parquet"
    runner._atomic_parquet_new(parquet, FakeFrame())
    assert parquet.read_bytes() == b"PAR1-fake-atomic-payload"
    assert not list(parquet.parent.glob(f".{parquet.name}.*.tmp.parquet"))


@pytest.mark.parametrize(
    ("crash_after", "expect_manifest"),
    [("result", False), ("manifest", True)],
)
def test_terminal_crash_order_is_fail_closed(
    tmp_path: Path, crash_after: str, expect_manifest: bool
) -> None:
    runner = _runner()
    artifact = tmp_path / crash_after
    attempt = _new_attempt(runner, artifact)
    attempt.completed_fits = 9
    with pytest.raises(RuntimeError, match="injected crash"):
        runner._commit_terminal_artifacts(
            artifact,
            {"status": "complete"},
            {"artifacts": {}},
            attempt,
            crash_after=crash_after,
        )
    attempt.abandon()
    assert (artifact / "result.json").is_file()
    assert (artifact / "manifest.json").is_file() is expect_manifest
    assert not (artifact / "attempt_journal" / "999_completed.json").exists()
    assert (artifact / "execution.lock").is_file()
    assert not list(artifact.rglob("*.tmp"))


def test_terminal_success_completes_journal_then_releases_lock(tmp_path: Path) -> None:
    runner = _runner()
    artifact = tmp_path / "success"
    attempt = _new_attempt(runner, artifact)
    attempt.completed_fits = 9
    result = runner._commit_terminal_artifacts(
        artifact,
        {"status": "complete"},
        {"artifacts": {}},
        attempt,
    )
    assert result == artifact / "result.json"
    assert (artifact / "manifest.json").is_file()
    assert (artifact / "attempt_journal" / "999_completed.json").is_file()
    assert not (artifact / "execution.lock").exists()
    with pytest.raises(FileExistsError, match="attempt journal already exists"):
        _new_attempt(runner, artifact)
    assert not (artifact / "execution.lock").exists()


def test_source_contains_no_execute_side_effect_at_import() -> None:
    runner = _runner()
    assert runner._NUMERICAL_MODULE is None
    assert not (PROJECT_ROOT / "artifacts" / runner.DEFAULT_CONFIG.stem / "execution.lock").exists()
    assert os.environ.get("P1_R2_NUMERICAL_EXECUTED") is None
