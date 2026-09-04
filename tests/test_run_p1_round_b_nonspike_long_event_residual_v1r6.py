from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "run_p1_round_b_nonspike_long_event_residual_v1r6.py"
)
BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_p1_v1r3_feature_key_binding.py"
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1r6.json"
)
OLD_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1r5.json"
)
AUTH_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1r6_execution_authorization.json"
)


def _runner():
    name = f"p1_round_b_residual_v1r6_test_{uuid.uuid4().hex}"
    specification = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _attempt(runner, artifact: Path, *, deadline: float | None = None):
    return runner.AttemptJournal.begin(
        artifact,
        deadline_epoch=time.time() + 60 if deadline is None else deadline,
        snapshot_manifest_sha256="a" * 64,
    )


def _readiness_receipt(runner) -> dict:
    return {
        "status": "PASS_COMPLETE_READINESS_BEFORE_CLAIM",
        "feature_row_binding": {},
        "full_feature_cache_binding": {},
        "left_censored_positive_connected_event_count_by_fold": {
            fold: 0 for fold in runner.FOLD_ORDER
        },
        "exact_round_b_equivalence": {},
    }


def _initialization_failed_paths(runner, artifact: Path) -> list[Path]:
    candidates = [
        artifact / runner.INITIALIZATION_FAILED_FILENAME,
        artifact / "attempt_journal" / "997_failed.json",
    ]
    return [path for path in candidates if path.is_file()]


def test_runner_top_level_is_stdlib_only_before_external_authorization() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "dateutil",
        "joblib",
        "lightgbm",
        "narwhals",
        "numpy",
        "p1_qc",
        "pandas",
        "psutil",
        "pyarrow",
        "scipy",
        "six",
        "sklearn",
        "threadpoolctl",
    }
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", maxsplit=1)[0])
    assert roots.isdisjoint(forbidden)


def test_numerical_contract_is_exact_v1r5_and_still_nine_fits() -> None:
    runner = _runner()
    current = _json(CONFIG_PATH)
    prior = _json(OLD_CONFIG_PATH)
    assert current["experiment_id"].endswith("v1r6")
    for section in runner.NUMERICAL_SECTIONS:
        assert current[section] == prior[section], section
    assert current["surface"]["seeds"] == [20260813, 20260829, 20260847]
    assert current["rescue_decoder"]["probability_threshold"] == 0.8
    assert current["rescue_decoder"]["maximum_anchor_distance_rows"] == 18
    assert current["resource_budget"]["residual_model_fits"] == 9


def test_dependency_runtime_native_and_provenance_closure_are_pinned() -> None:
    runner = _runner()
    config = _json(CONFIG_PATH)
    trust = config["trust_contract"]
    assert tuple(trust["project_dependency_paths"]) == runner.REQUIRED_DEPENDENCY_PATHS
    assert tuple(trust["verification_paths"]) == runner.REQUIRED_VERIFICATION_PATHS
    assert tuple(trust["provenance_paths"]) == runner.REQUIRED_PROVENANCE_PATHS
    discovered = runner._discover_project_dependency_closure(
        PROJECT_ROOT / trust["numerical_entrypoint_path"]
    )
    assert discovered == set(runner.REQUIRED_DEPENDENCY_PATHS)
    observed = {"python": runner.platform.python_version()}
    for distribution in runner.EXPECTED_RUNTIME_VERSIONS:
        if distribution != "python":
            observed[distribution] = importlib.metadata.version(distribution)
    assert observed == trust["runtime_versions"] == runner.EXPECTED_RUNTIME_VERSIONS
    assert runner._native_lightgbm_record(config) == trust["lightgbm_native"]
    assert _sha256(BUILDER_PATH) == trust["provenance_sha256"]


def test_external_authorization_cycle_has_no_self_hash_and_is_exact() -> None:
    runner = _runner()
    config = _json(CONFIG_PATH)
    authorization = _json(AUTH_PATH)
    authorization_sha = _sha256(AUTH_PATH)
    assert "authorization_sha256" not in authorization
    assert runner.AUTHORIZATION_SHA256 == authorization_sha
    assert authorization["config_sha256"] == _sha256(CONFIG_PATH)
    assert authorization["runner_normalized_sha256"] == runner._normalised_runner_sha256()
    assert config["trust_contract"]["runner_normalized_sha256"] == (
        runner._normalised_runner_sha256()
    )
    assert authorization["supersedes"] == config["supersedes"]


def test_missing_or_wrong_external_hash_fails_before_any_json_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    monkeypatch.setattr(runner, "_assert_preauth_clean", lambda: None)
    reads: list[Path] = []

    def forbidden_read(path: Path, **_kwargs):
        reads.append(path)
        raise AssertionError("JSON read occurred before authorization")

    monkeypatch.setattr(runner, "_json_load_bound", forbidden_read)
    monkeypatch.delenv(runner.AUTHORIZATION_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="required before any config read"):
        runner._load_authorized(CONFIG_PATH, require_seal=False)
    assert reads == []
    monkeypatch.setenv(runner.AUTHORIZATION_ENV_VAR, "0" * 64)
    with pytest.raises(RuntimeError, match="differs from sealed runner anchor"):
        runner._load_authorized(CONFIG_PATH, require_seal=False)
    assert reads == []


def test_atomic_publication_is_true_create_only_and_directory_flushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    runner._fsync_directory(tmp_path)
    target = tmp_path / "value.json"
    runner._atomic_json_new(target, {"value": 1})
    original = target.read_bytes()

    def forbidden_replace(*_args, **_kwargs):
        raise AssertionError("overwrite-capable os.replace was called")

    monkeypatch.setattr(runner.os, "replace", forbidden_replace)
    with pytest.raises(FileExistsError):
        runner._atomic_json_new(target, {"value": 2})
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_concurrent_claim_has_one_winner_and_incomplete_attempt_is_permanent(
    tmp_path: Path,
) -> None:
    runner = _runner()
    artifact = tmp_path / "artifact"

    def claim() -> tuple[str, object]:
        try:
            return "ok", _attempt(runner, artifact)
        except BaseException as error:
            return "error", error

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _value: claim(), range(2)))
    winners = [value for status, value in outcomes if status == "ok"]
    errors = [value for status, value in outcomes if status == "error"]
    assert len(winners) == 1
    assert len(errors) == 1 and isinstance(errors[0], FileExistsError)
    winners[0].close_handle_keep_lock()
    lock = artifact / "execution.lock"
    assert lock.is_file()
    lock.unlink()
    with pytest.raises(FileExistsError, match="lifetime attempt journal already exists"):
        _attempt(runner, artifact)
    assert not lock.exists()


def test_preexisting_journal_is_rejected_before_claim_and_old_second_flush_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    artifact = tmp_path / "preexisting_journal"
    (artifact / "attempt_journal").mkdir(parents=True)
    original_flush = runner._fsync_directory
    artifact_flushes = 0

    def one_shot_second_artifact_flush(path: Path) -> None:
        nonlocal artifact_flushes
        if path == artifact:
            artifact_flushes += 1
            if artifact_flushes == 2:
                raise OSError("historical second artifact flush fault")
        original_flush(path)

    monkeypatch.setattr(runner, "_fsync_directory", one_shot_second_artifact_flush)
    with pytest.raises(FileExistsError, match="already exists before claim"):
        _attempt(runner, artifact)
    assert artifact_flushes == 0
    assert not (artifact / "execution.lock").exists()
    assert not _initialization_failed_paths(runner, artifact)


@pytest.mark.parametrize(
    ("failed_artifact_flush", "expected_phase", "terminal_location"),
    [
        (1, "LOCK_DIRECTORY_FLUSH", "artifact_root_fallback"),
        (2, "JOURNAL_DIRECTORY_FLUSH", "attempt_journal_hash_chain"),
    ],
)
def test_initialization_artifact_flush_fault_is_terminal_and_lock_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_artifact_flush: int,
    expected_phase: str,
    terminal_location: str,
) -> None:
    runner = _runner()
    artifact = tmp_path / f"flush_{failed_artifact_flush}"
    original_flush = runner._fsync_directory
    artifact_flushes = 0
    injected = False

    def one_shot_flush(path: Path) -> None:
        nonlocal artifact_flushes, injected
        if path == artifact:
            artifact_flushes += 1
            if artifact_flushes == failed_artifact_flush and not injected:
                injected = True
                raise OSError(f"injected artifact flush {failed_artifact_flush}")
        original_flush(path)

    monkeypatch.setattr(runner, "_fsync_directory", one_shot_flush)
    with pytest.raises(OSError, match=f"artifact flush {failed_artifact_flush}"):
        _attempt(runner, artifact)
    failed = _initialization_failed_paths(runner, artifact)
    assert injected is True
    assert len(failed) == 1
    payload = _json(failed[0])
    assert payload["status"] == runner.INITIALIZATION_FAILED_STATUS
    assert payload["phase"] == expected_phase
    assert payload["physical_fit_reservations"] == 0
    assert payload["physical_fits_completed"] == 0
    assert payload["terminal_location"] == terminal_location
    assert (artifact / "execution.lock").is_file()
    first = runner._record_parent_failure_if_claimed(
        artifact,
        "PARENT_AFTER_INIT_FAILURE",
        RuntimeError("parent observed worker failure"),
    )
    second = runner._record_parent_failure_if_claimed(
        artifact,
        "PARENT_AFTER_INIT_FAILURE_DUPLICATE",
        RuntimeError("parent observed worker failure again"),
    )
    assert first["failed_terminal_created"] is False
    assert second["failed_terminal_created"] is False
    assert _initialization_failed_paths(runner, artifact) == failed


def test_lock_payload_short_write_and_fsync_fault_keep_claim_with_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    original_write = runner.os.write
    artifact_short = tmp_path / "short_write"
    wrote = False

    def short_once(descriptor: int, payload: bytes) -> int:
        nonlocal wrote
        if not wrote:
            wrote = True
            return max(0, len(payload) - 1)
        return original_write(descriptor, payload)

    monkeypatch.setattr(runner.os, "write", short_once)
    with pytest.raises(OSError, match="short execution-lock write"):
        _attempt(runner, artifact_short)
    short_failed = _initialization_failed_paths(runner, artifact_short)
    assert len(short_failed) == 1
    assert _json(short_failed[0])["phase"] == "LOCK_PAYLOAD_WRITE"
    assert (artifact_short / "execution.lock").is_file()

    monkeypatch.setattr(runner.os, "write", original_write)
    original_fsync = runner.os.fsync
    artifact_fsync = tmp_path / "lock_fsync"
    fsync_failed = False

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal fsync_failed
        if not fsync_failed:
            fsync_failed = True
            raise OSError("injected lock fsync")
        original_fsync(descriptor)

    monkeypatch.setattr(runner.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="injected lock fsync"):
        _attempt(runner, artifact_fsync)
    fsync_terminals = _initialization_failed_paths(runner, artifact_fsync)
    assert len(fsync_terminals) == 1
    assert _json(fsync_terminals[0])["phase"] == "LOCK_PAYLOAD_FSYNC"
    assert (artifact_fsync / "execution.lock").is_file()


@pytest.mark.parametrize("terminal_edge", ["write", "fsync"])
def test_initialization_failure_terminal_write_and_fsync_edges_retry_exact_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_edge: str,
) -> None:
    runner = _runner()
    artifact = tmp_path / f"terminal_{terminal_edge}"
    original_write = runner.os.write
    original_atomic = runner._atomic_json_new
    original_fsync = runner.os.fsync
    lock_write_failed = False
    terminal_edge_failed = False

    def fail_lock_write_once(descriptor: int, payload: bytes) -> int:
        nonlocal lock_write_failed
        if not lock_write_failed:
            lock_write_failed = True
            return max(0, len(payload) - 1)
        return original_write(descriptor, payload)

    def fail_terminal_write_once(path: Path, value) -> None:
        nonlocal terminal_edge_failed
        if (
            terminal_edge == "write"
            and path.name == runner.INITIALIZATION_FAILED_FILENAME
            and not terminal_edge_failed
        ):
            terminal_edge_failed = True
            raise OSError("injected initialization terminal write")
        original_atomic(path, value)

    def fail_terminal_fsync_once(descriptor: int) -> None:
        nonlocal terminal_edge_failed
        if terminal_edge == "fsync" and not terminal_edge_failed:
            terminal_edge_failed = True
            raise OSError("injected initialization terminal fsync")
        original_fsync(descriptor)

    monkeypatch.setattr(runner.os, "write", fail_lock_write_once)
    monkeypatch.setattr(runner, "_atomic_json_new", fail_terminal_write_once)
    if terminal_edge == "fsync":
        monkeypatch.setattr(runner.os, "fsync", fail_terminal_fsync_once)
    with pytest.raises(OSError, match="short execution-lock write"):
        _attempt(runner, artifact)
    failed = _initialization_failed_paths(runner, artifact)
    assert lock_write_failed is True and terminal_edge_failed is True
    assert len(failed) == 1
    assert _json(failed[0])["status"] == runner.INITIALIZATION_FAILED_STATUS
    assert (artifact / "execution.lock").is_file()


def test_journal_mkdir_and_started_write_failures_are_exact_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    original_mkdir = runner.os.mkdir
    artifact_mkdir = tmp_path / "mkdir_failure"
    mkdir_failed = False

    def fail_journal_mkdir(path, *args, **kwargs):
        nonlocal mkdir_failed
        if Path(path) == artifact_mkdir / "attempt_journal" and not mkdir_failed:
            mkdir_failed = True
            raise OSError("injected journal mkdir")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(runner.os, "mkdir", fail_journal_mkdir)
    with pytest.raises(OSError, match="injected journal mkdir"):
        _attempt(runner, artifact_mkdir)
    mkdir_terminals = _initialization_failed_paths(runner, artifact_mkdir)
    assert len(mkdir_terminals) == 1
    assert mkdir_terminals[0].parent == artifact_mkdir
    assert (artifact_mkdir / "execution.lock").is_file()

    monkeypatch.setattr(runner.os, "mkdir", original_mkdir)
    original_atomic = runner._atomic_json_new
    artifact_started = tmp_path / "started_write_failure"
    started_failed = False

    def fail_started_once(path: Path, value) -> None:
        nonlocal started_failed
        if path.name == "000_started.json" and not started_failed:
            started_failed = True
            raise OSError("injected started write")
        original_atomic(path, value)

    monkeypatch.setattr(runner, "_atomic_json_new", fail_started_once)
    with pytest.raises(OSError, match="injected started write"):
        _attempt(runner, artifact_started)
    started_terminals = _initialization_failed_paths(runner, artifact_started)
    assert len(started_terminals) == 1
    assert started_terminals[0].parent.name == "attempt_journal"
    started_payload = _json(started_terminals[0])
    assert started_payload["journal_prefix"]["entry_count"] == 0
    assert started_payload["previous_entry_sha256"] is None
    assert (artifact_started / "execution.lock").is_file()


def test_started_terminal_postlink_fsync_fault_preserves_hash_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    artifact = tmp_path / "started_postlink_fsync"
    journal = artifact / "attempt_journal"
    original_flush = runner._fsync_directory
    journal_flush_failed = False

    def fail_first_journal_flush(path: Path) -> None:
        nonlocal journal_flush_failed
        if path == journal and not journal_flush_failed:
            journal_flush_failed = True
            raise OSError("injected started terminal directory fsync")
        original_flush(path)

    monkeypatch.setattr(runner, "_fsync_directory", fail_first_journal_flush)
    with pytest.raises(OSError, match="started terminal directory fsync"):
        _attempt(runner, artifact)
    assert journal_flush_failed is True
    failed = _initialization_failed_paths(runner, artifact)
    assert len(failed) == 1 and failed[0] == journal / "997_failed.json"
    previous, entries = runner._verify_journal_chain(
        journal,
        required_last_name="997_failed.json",
    )
    assert [entry["schema_version"] for entry in entries] == [
        "p1_round_b_residual.attempt.started.v1r6",
        "p1_round_b_residual.initialization_failed.v1r6",
    ]
    assert previous == _sha256(failed[0])
    assert (artifact / "execution.lock").is_file()


def test_failed_terminal_postlink_journal_fsync_is_retried_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    artifact = tmp_path / "failed_terminal_postlink_fsync"
    journal = artifact / "attempt_journal"
    original_atomic = runner._atomic_json_new
    original_flush = runner._fsync_directory
    started_failed = False
    failed_terminal_flush_failed = False

    def fail_started_before_publish(path: Path, value) -> None:
        nonlocal started_failed
        if path.name == "000_started.json" and not started_failed:
            started_failed = True
            raise OSError("injected started publication failure")
        original_atomic(path, value)

    def fail_failed_terminal_flush_once(path: Path) -> None:
        nonlocal failed_terminal_flush_failed
        if path == journal and not failed_terminal_flush_failed:
            failed_terminal_flush_failed = True
            raise OSError("injected failed-terminal postlink journal fsync")
        original_flush(path)

    monkeypatch.setattr(runner, "_atomic_json_new", fail_started_before_publish)
    monkeypatch.setattr(runner, "_fsync_directory", fail_failed_terminal_flush_once)
    with pytest.raises(OSError, match="started publication failure"):
        _attempt(runner, artifact)
    failed = _initialization_failed_paths(runner, artifact)
    assert started_failed is True and failed_terminal_flush_failed is True
    assert failed == [journal / "997_failed.json"]
    runner._verify_journal_chain(journal, required_last_name="997_failed.json")
    assert (artifact / "execution.lock").is_file()


def test_deadline_is_checked_before_fit_reservation(tmp_path: Path) -> None:
    runner = _runner()
    attempt = _attempt(runner, tmp_path / "deadline")
    attempt.record_readiness(_readiness_receipt(runner))
    attempt.begin_fold(0, runner.FOLD_ORDER[0])
    before = sorted(attempt.journal_dir.glob("*.json"))
    attempt.deadline_epoch = time.time() - 1
    with pytest.raises(TimeoutError, match="before physical fit reservation"):
        attempt.reserve_fit(runner.FOLD_ORDER[0], 0, 20260813)
    assert sorted(attempt.journal_dir.glob("*.json")) == before
    assert attempt.reserved_fits == 0
    attempt.close_handle_keep_lock()


def test_fit_guard_enforces_registered_order_and_lifetime_ceiling(tmp_path: Path) -> None:
    runner = _runner()

    class FakeClassifier:
        def __init__(self, *args, **kwargs) -> None:
            self.random_state = kwargs["random_state"]
            self.fit_calls = 0

        def fit(self, *_args, **_kwargs):
            self.fit_calls += 1
            return self

        def predict_proba(self, *_args, **_kwargs):
            return [[0.5, 0.5]]

    numerical = SimpleNamespace(lgb=SimpleNamespace(LGBMClassifier=FakeClassifier))
    attempt = _attempt(runner, tmp_path / "fits")
    attempt.record_readiness(_readiness_receipt(runner))
    context: dict[str, object] = {}
    runner._install_fit_guard(numerical, attempt, context)
    seeds = [20260813, 20260829, 20260847]
    for ordinal, fold in enumerate(runner.FOLD_ORDER):
        attempt.begin_fold(ordinal, fold)
        context.update(name=fold, ordinal=ordinal)
        for base_seed in seeds:
            model = numerical.lgb.LGBMClassifier(random_state=base_seed + ordinal)
            model.fit([[0.0]], [0])
    assert attempt.reserved_fits == attempt.completed_fits == 9
    with pytest.raises(RuntimeError, match="ceiling"):
        attempt.reserve_fit(runner.FOLD_ORDER[-1], 2, seeds[-1] + 2)
    attempt.close_handle_keep_lock()


def test_left_censor_gate_fails_closed_before_fit() -> None:
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
    assert runner._left_censored_positive_event_counts(train, folds, numerical) == {
        fold: 0 for fold in runner.FOLD_ORDER
    }
    train.loc[1, "label"] = 1
    with pytest.raises(RuntimeError, match="NO_GO_LEFT_CENSORED.*BEFORE_FIT"):
        runner._left_censored_positive_event_counts(train, folds, numerical)


def test_feature_cache_and_pinned_key_sidecar_reorder_fail_closed() -> None:
    runner = _runner()
    import numpy as np
    import pandas as pd

    train = pd.DataFrame(
        {
            "station": ["A", "A"],
            "year": [2024, 2024],
            "layer": [1, 1],
            "time": ["2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z"],
            "temp": [1.0, 2.0],
            "psal": [30.0, np.nan],
            "depth": [5.0, 5.0],
        }
    )
    features = pd.DataFrame(
        {
            "station": ["A", "A"],
            "layer_category": ["1", "1"],
            "temp_raw": np.array([1.0, 2.0], dtype=np.float32),
            "psal_raw": np.array([30.0, np.nan], dtype=np.float32),
            "depth_raw": np.array([5.0, 5.0], dtype=np.float32),
        }
    )
    numerical = SimpleNamespace(np=np, pd=pd)
    feature_binding = runner._validate_feature_row_binding(train, features, numerical)
    key_digest = feature_binding["source_train_key_digest"]
    sidecar = train.loc[:, list(runner.KEY_COLUMNS)].copy()
    sidecar.insert(0, "ordinal", np.arange(len(sidecar), dtype=np.int64))
    sidecar_spec = {
        "path": "binding.parquet",
        "sha256": "b" * 64,
        "bytes": 10,
        "rows": 2,
        "source_train_key_digest": key_digest,
    }
    cache_spec = {"source_sha256": "s" * 64, "sha256": "c" * 64}
    receipt_spec = {"sha256": "r" * 64}
    receipt = {
        "schema_version": "p1_round_b_residual.feature_cache_key_binding.v1r3",
        "status": "SEALED_ZERO_FIT_SOURCE_CACHE_POSITIONAL_BINDING",
        "rows": 2,
        "source_sha256": "s" * 64,
        "feature_cache_sha256": "c" * 64,
        "source_train_key_digest": key_digest,
        "feature_cache_key_digest": key_digest,
        "key_digests_equal": True,
        "sidecar_path": "binding.parquet",
        "sidecar_bytes": 10,
        "sidecar_sha256": "b" * 64,
        "operation_counters": {
            "candidate_files": 0,
            "model_fits": 0,
            "official_test_reads": 0,
            "predictions": 0,
            "sample_format_reads": 0,
            "scores": 0,
            "submission_candidate_reads": 0,
            "uploads": 0,
        },
    }
    validated = runner._validate_pinned_feature_key_binding(
        train,
        sidecar,
        feature_binding,
        cache_spec,
        sidecar_spec,
        receipt_spec,
        receipt,
        numerical,
    )
    assert validated["feature_cache_exact_key_values_equal_source"] is True
    reordered_features = features.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="raw row binding mismatch"):
        runner._validate_feature_row_binding(train, reordered_features, numerical)
    reordered_sidecar = sidecar.iloc[::-1].reset_index(drop=True)
    reordered_sidecar["ordinal"] = np.arange(len(sidecar), dtype=np.int64)
    with pytest.raises(RuntimeError, match="row-key digest"):
        runner._validate_pinned_feature_key_binding(
            train,
            reordered_sidecar,
            dict(feature_binding),
            cache_spec,
            sidecar_spec,
            receipt_spec,
            receipt,
            numerical,
        )


def test_full_feature_binding_detects_same_raw_derived_swap() -> None:
    runner = _runner()
    import numpy as np
    import pandas as pd

    rebuilt = pd.DataFrame(
        {
            "station": ["A", "A"],
            "layer_category": ["1", "1"],
            "temp_raw": np.array([1.0, 1.0], dtype=np.float32),
            "psal_raw": np.array([30.0, 30.0], dtype=np.float32),
            "depth_raw": np.array([5.0, 5.0], dtype=np.float32),
            "temp_lag_1": np.array([0.0, 2.0], dtype=np.float32),
        }
    )
    swapped = rebuilt.iloc[::-1].reset_index(drop=True)
    numerical = SimpleNamespace(np=np, pd=pd)
    raw_columns = ["station", "layer_category", "temp_raw", "psal_raw", "depth_raw"]
    raw_equal, _ = runner._feature_frames_exact(
        rebuilt.loc[:, raw_columns], swapped.loc[:, raw_columns], numerical
    )
    full_equal, unequal = runner._feature_frames_exact(rebuilt, swapped, numerical)
    assert raw_equal is True
    assert full_equal is False
    assert unequal == ["temp_lag_1"]
    config = _json(CONFIG_PATH)
    assert config["execution_safety"]["ambiguous_raw_swap_probe"] == {
        "expected_ambiguous_groups": 299,
        "expected_ambiguous_rows": 603,
        "expected_groups_with_multiple_derived_variants": 299,
        "synthetic_swap_must_fail": True,
    }


def test_sealed_score_uses_only_held_in_memory_truth() -> None:
    runner = _runner()
    import pandas as pd

    config = _json(CONFIG_PATH)
    config["immutable_inputs"]["frozen_truth_oof"] = dict(
        config["immutable_inputs"]["frozen_truth_oof"]
    )
    config["immutable_inputs"]["frozen_truth_oof"]["rows"] = 2
    config["immutable_inputs"]["frozen_truth_oof"]["sha256"] = "f" * 64
    truth = pd.DataFrame(
        {
            "station": ["A", "A"],
            "year": [2025, 2025],
            "layer": [1, 1],
            "time": ["t0", "t1"],
            "label": [0, 1],
            "anomaly_type": ["normal", "event"],
            "fold": ["2025_q2", "2025_q2"],
        }
    )
    predictions = pd.DataFrame({"row": [0, 1]})

    def original_resolver(_value):
        raise AssertionError("physical truth path was reopened")

    original_read = pd.read_parquet
    numerical = SimpleNamespace(pd=pd, _resolve_repo_path=original_resolver)

    def sealed_score(score_config, observed_predictions):
        observed = numerical.pd.read_parquet(
            numerical._resolve_repo_path(
                score_config["immutable_inputs"]["frozen_truth_oof"]["path"]
            ),
            columns=[*runner.KEY_COLUMNS, "label", "anomaly_type", "fold"],
        )
        assert observed.equals(truth)
        assert observed_predictions is predictions
        return {"rows": len(observed)}, {"gate": True}

    numerical._score = sealed_score
    receipt = {
        "actual_read_sha256_before_parse": "f" * 64,
        "actual_read_sha256_after_parse": "f" * 64,
        "path_reopens": 0,
    }
    metrics, checks, scoring = runner._score_with_verified_truth(
        numerical, config, predictions, truth, receipt
    )
    assert metrics == {"rows": 2}
    assert checks == {"gate": True}
    assert scoring["truth_path_reopens"] == 0
    assert scoring["held_in_memory_truth_injections"] == 1
    assert numerical._resolve_repo_path is original_resolver
    assert numerical.pd.read_parquet is original_read


def test_held_snapshot_reader_never_reopens_and_detects_tamper(tmp_path: Path) -> None:
    runner = _runner()
    import pandas as pd

    source = tmp_path / "input.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    expected = _sha256(source)
    reader = runner.HeldSnapshotInputs(tmp_path)
    frame = reader.read_csv("input.csv", expected, SimpleNamespace(pd=pd))
    assert frame["value"].tolist() == [1]
    assert reader.receipts["input.csv"]["parsed_from_same_held_handle"] is True
    assert reader.receipts["input.csv"]["path_reopens"] == 0
    with pytest.raises(RuntimeError, match="would be reopened"):
        reader.read_csv("input.csv", expected, SimpleNamespace(pd=pd))
    source.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        runner.HeldSnapshotInputs(tmp_path).read_csv(
            "input.csv", expected, SimpleNamespace(pd=pd)
        )


def test_verified_copy_is_independent_of_later_source_path_changes(tmp_path: Path) -> None:
    runner = _runner()
    source = tmp_path / "source.bin"
    snapshot = tmp_path / "private" / "snapshot.bin"
    source.write_bytes(b"immutable-original")
    expected = _sha256(source)
    receipt = runner._copy_verified_file(source, snapshot, expected)
    source.write_bytes(b"changed-after-copy")
    assert snapshot.read_bytes() == b"immutable-original"
    assert receipt["actual_read_sha256"] == expected == _sha256(snapshot)


def test_missing_readiness_fails_before_attempt_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    config = _json(CONFIG_PATH)
    monkeypatch.setattr(
        runner,
        "_load_authorized",
        lambda *_args, **_kwargs: (config, {"status": "ok"}, {"status": "ok"}),
    )
    monkeypatch.setattr(
        runner,
        "_verify_snapshot_manifest",
        lambda *_args, **_kwargs: (tmp_path, {"strict_readiness": {}}),
    )
    monkeypatch.setattr(runner, "_load_snapshot_numerical", lambda *_args: (object(), {}))

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("missing immutable input")

    monkeypatch.setattr(runner, "_strict_snapshot_readiness", missing)
    monkeypatch.setattr(
        runner.AttemptJournal,
        "begin",
        lambda *_args, **_kwargs: pytest.fail("claim created before readiness"),
    )
    with pytest.raises(FileNotFoundError, match="missing immutable input"):
        runner._worker_execute(CONFIG_PATH, tmp_path / "manifest.json", "a" * 64, time.time() + 60)


def test_existing_output_fails_before_claim(tmp_path: Path) -> None:
    runner = _runner()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="before claim"):
        runner._assert_output_namespace_ready(artifact)
    assert not (artifact / "execution.lock").exists()
    assert not (artifact / "attempt_journal").exists()


def test_parent_worker_command_uses_same_runner_and_absolute_deadline() -> None:
    runner = _runner()
    deadline = time.time() + 100
    command = runner._worker_command(CONFIG_PATH, Path("snapshot.json"), "a" * 64, deadline)
    assert Path(command[1]).resolve() == SCRIPT_PATH.resolve()
    assert "--worker" in command
    assert command[command.index("--deadline-epoch") + 1] == repr(deadline)


def test_parent_timeout_terminates_worker(tmp_path: Path) -> None:
    del tmp_path
    runner = _runner()
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(runner.WorkerTimeoutError, match="killed worker tree") as caught:
        runner._run_supervised(command, time.time() + 0.2)
    receipt = caught.value.termination_receipt
    if os.name == "nt":
        assert receipt["taskkill_returncode"] == 0
        assert receipt["target_and_descendants_confirmed_gone"] is True


def test_windows_taskkill_rc_and_descendant_confirmation_fail_closed() -> None:
    runner = _runner()

    class FakeProcess:
        pid = 100

        def __init__(self) -> None:
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            del timeout
            self.returncode = -9
            return self.returncode

    success_tables = iter([{100: 0, 101: 100, 102: 101}, {}])
    process = FakeProcess()
    success = runner._terminate_windows_process_tree(
        process,
        table_loader=lambda: next(success_tables),
        taskkill_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="SUCCESS", stderr=""
        ),
    )
    assert success["known_descendant_pids"] == [101, 102]
    assert success["taskkill_returncode"] == 0
    assert success["target_and_descendants_confirmed_gone"] is True

    failed_process = FakeProcess()
    with pytest.raises(runner.ProcessTreeTerminationError) as caught:
        runner._terminate_windows_process_tree(
            failed_process,
            table_loader=lambda: {100: 0, 101: 100},
            taskkill_runner=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=1, stdout="", stderr="denied"
            ),
            confirmation_timeout_seconds=0.01,
        )
    failed = caught.value.termination_receipt
    assert failed["taskkill_returncode"] == 1
    assert failed["target_and_descendants_confirmed_gone"] is False
    assert failed["pids_still_present"] == [100, 101]


@pytest.mark.parametrize(
    ("crash_after", "manifest_exists", "terminal_exists"),
    [
        ("result", False, False),
        ("manifest", True, False),
        ("worker_terminal", True, True),
    ],
)
def test_worker_terminal_crash_order_is_fail_closed(
    tmp_path: Path,
    crash_after: str,
    manifest_exists: bool,
    terminal_exists: bool,
) -> None:
    runner = _runner()
    artifact = tmp_path / crash_after
    attempt = _attempt(runner, artifact)
    attempt.reserved_fits = 9
    attempt.completed_fits = 9
    with pytest.raises(RuntimeError, match="injected crash"):
        runner._commit_worker_terminal_artifacts(
            artifact,
            {"status": "result"},
            {"status": "manifest"},
            attempt,
            crash_after=crash_after,
        )
    attempt.fail_terminal(crash_after, RuntimeError("injected crash"))
    assert (artifact / "result.json").is_file()
    assert (artifact / "manifest.json").is_file() is manifest_exists
    assert (
        artifact / "attempt_journal" / "998_worker_terminal.json"
    ).is_file() is terminal_exists
    failed_name = "999_failed.json" if terminal_exists else "997_failed.json"
    assert (artifact / "attempt_journal" / failed_name).is_file()
    assert not (artifact / "attempt_journal" / "999_completed.json").exists()
    assert (artifact / "execution.lock").is_file()


def test_worker_failure_terminal_is_hash_chained_exact_once_and_keeps_lock(
    tmp_path: Path,
) -> None:
    runner = _runner()
    artifact = tmp_path / "worker_failed"
    attempt = _attempt(runner, artifact)
    attempt.record_readiness(_readiness_receipt(runner))
    error = RuntimeError("injected fold failure")
    first = attempt.fail_terminal("FOLD_2025_q2_PRE_RESERVATION", error)
    second = attempt.fail_terminal("SHOULD_NOT_DUPLICATE", error)
    assert first == second
    paths = sorted((artifact / "attempt_journal").glob("*.json"))
    assert [path.name for path in paths].count("997_failed.json") == 1
    payload = _json(first)
    assert payload["status"] == "FAILED_FAIL_CLOSED_LOCK_RETAINED"
    assert payload["phase"] == "FOLD_2025_q2_PRE_RESERVATION"
    assert payload["physical_fit_reservations"] == 0
    assert payload["journal_prefix"]["entry_count"] == 2
    assert payload["previous_entry_sha256"] == payload["journal_prefix"][
        "last_entry_sha256"
    ]
    assert (artifact / "execution.lock").is_file()


def test_parent_failure_terminal_records_timeout_provenance_exact_once(
    tmp_path: Path,
) -> None:
    runner = _runner()
    artifact = tmp_path / "parent_failed"
    attempt = _attempt(runner, artifact)
    attempt.close_handle_keep_lock()
    termination = {
        "platform": "windows",
        "taskkill_returncode": 1,
        "target_and_descendants_confirmed_gone": False,
    }
    error = runner.WorkerTimeoutError("termination failed", termination)
    first = runner._record_parent_failure_if_claimed(
        artifact,
        "PARENT_WORKER_SUPERVISION",
        error,
        provenance=error.termination_receipt,
    )
    second = runner._record_parent_failure_if_claimed(
        artifact,
        "SHOULD_NOT_DUPLICATE",
        error,
        provenance=error.termination_receipt,
    )
    assert first["failed_terminal_created"] is True
    assert second["failed_terminal_created"] is False
    failed_paths = sorted((artifact / "attempt_journal").glob("*failed.json"))
    assert len(failed_paths) == 1
    payload = _json(failed_paths[0])
    assert payload["failure_actor"] == "parent"
    assert payload["failure_provenance"] == termination
    assert payload["journal_prefix"]["entry_count"] == 1
    assert (artifact / "execution.lock").is_file()


def test_parent_completion_is_durable_before_lock_release(tmp_path: Path) -> None:
    runner = _runner()
    artifact = tmp_path / "parent"
    (artifact / "attempt_journal").mkdir(parents=True)
    (artifact / "execution.lock").write_bytes(b"locked")
    with pytest.raises(RuntimeError, match="after parent completion") as caught:
        runner._publish_parent_completion(
            artifact,
            {
                "attempt_id": "injected",
                "status": "complete",
                "previous_entry_sha256": None,
            },
            crash_after_completion=True,
        )
    assert (artifact / "attempt_journal" / "999_completed.json").is_file()
    assert (artifact / "execution.lock").is_file()
    receipt = runner._record_parent_failure_if_claimed(
        artifact,
        "PARENT_COMPLETION_LOCK_RELEASE",
        caught.value,
    )
    assert receipt["failed_terminal_created"] is True
    failure = artifact / "attempt_journal" / "999_postcompletion_failed.json"
    assert failure.is_file()
    assert _json(failure)["fit_slot_state"]["parent_completion_record_present"] is True


def test_every_execution_lock_unlink_is_final_commit_with_zero_fallible_tail() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    completion = functions["_publish_parent_completion"]
    lock_unlinks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "unlink"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "lock_path"
    ]
    assert len(lock_unlinks) == 1
    assert source.count("lock_path.unlink()") == 1
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "unlink"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "lock_path"
        for node in ast.walk(functions["_record_initialization_failure"])
    )
    unlink_statement_index = next(
        index
        for index, statement in enumerate(completion.body)
        if any(node is lock_unlinks[0] for node in ast.walk(statement))
    )
    tail = completion.body[unlink_statement_index + 1 :]
    assert len(tail) == 1 and isinstance(tail[0], ast.Return)
    assert not any(isinstance(node, ast.Call) for node in ast.walk(tail[0]))
    config = _json(CONFIG_PATH)
    assert config["execution_safety"]["post_claim_rollback_lock_unlinks"] == 0
    assert config["execution_safety"]["post_lock_release_fallible_operations"] == 0


def test_exact_v1r5_postunlink_fsync_fault_is_structurally_impossible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    artifact = tmp_path / "final_commit"
    journal = artifact / "attempt_journal"
    journal.mkdir(parents=True)
    lock = artifact / "execution.lock"
    lock.write_bytes(b"locked")
    original_flush = runner._fsync_directory
    flush_observations: list[tuple[Path, bool]] = []

    def fail_if_flush_occurs_after_unlink(path: Path) -> None:
        lock_present = lock.is_file()
        flush_observations.append((path, lock_present))
        if not lock_present:
            raise RuntimeError("exact v1r5 post-unlink fsync fault")
        original_flush(path)

    monkeypatch.setattr(runner, "_fsync_directory", fail_if_flush_occurs_after_unlink)
    completed = runner._publish_parent_completion(
        artifact,
        {
            "attempt_id": "success",
            "status": "complete",
            "previous_entry_sha256": None,
        },
    )
    assert completed.is_file()
    assert not lock.exists()
    assert flush_observations
    assert all(lock_present for _path, lock_present in flush_observations)
    assert not list(journal.glob("*failed.json"))
    assert [path.name for path in journal.glob("*.json")] == ["999_completed.json"]


def test_precommit_flush_failure_has_unambiguous_failed_terminal_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    artifact = tmp_path / "precommit_flush_failure"
    journal = artifact / "attempt_journal"
    journal.mkdir(parents=True)
    lock = artifact / "execution.lock"
    lock.write_bytes(b"locked")
    original_flush = runner._fsync_directory

    def injected_flush(path: Path) -> None:
        if path == artifact:
            assert lock.is_file()
            raise RuntimeError("injected artifact precommit flush failure")
        original_flush(path)

    monkeypatch.setattr(runner, "_fsync_directory", injected_flush)
    with pytest.raises(RuntimeError, match="precommit flush failure") as caught:
        runner._publish_parent_completion(
            artifact,
            {
                "attempt_id": "failed",
                "status": "complete",
                "previous_entry_sha256": None,
            },
        )
    assert lock.is_file()
    assert (journal / "999_completed.json").is_file()
    monkeypatch.setattr(runner, "_fsync_directory", original_flush)
    receipt = runner._record_parent_failure_if_claimed(
        artifact,
        "PARENT_COMPLETION_PRECOMMIT_FLUSH",
        caught.value,
    )
    assert receipt["failed_terminal_created"] is True
    failure = journal / "999_postcompletion_failed.json"
    assert failure.is_file() and lock.is_file()
    _previous, entries = runner._verify_journal_chain(
        journal,
        required_last_name="999_postcompletion_failed.json",
    )
    assert [entry["status"] for entry in entries] == [
        "complete",
        "FAILED_FAIL_CLOSED_LOCK_RETAINED",
    ]


def test_lock_unlink_failure_stays_recoverable_and_records_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    artifact = tmp_path / "unlink_failure"
    journal = artifact / "attempt_journal"
    journal.mkdir(parents=True)
    lock = artifact / "execution.lock"
    lock.write_bytes(b"locked")
    original_unlink = Path.unlink

    def injected_unlink(path: Path, *args, **kwargs) -> None:
        if path == lock:
            raise PermissionError("injected lock unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", injected_unlink)
    with pytest.raises(PermissionError, match="lock unlink failure") as caught:
        runner._publish_parent_completion(
            artifact,
            {
                "attempt_id": "failed",
                "status": "complete",
                "previous_entry_sha256": None,
            },
        )
    assert lock.is_file()
    monkeypatch.setattr(Path, "unlink", original_unlink)
    receipt = runner._record_parent_failure_if_claimed(
        artifact,
        "PARENT_FINAL_LOCK_UNLINK",
        caught.value,
    )
    assert receipt["failed_terminal_created"] is True
    assert (journal / "999_postcompletion_failed.json").is_file()
    assert lock.is_file()


def test_success_path_snapshot_cleanup_precedes_finalization_and_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    config = _json(CONFIG_PATH)
    snapshot = tmp_path / "snapshot"
    manifest = snapshot / "snapshot_manifest.json"
    snapshot.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(
        runner,
        "_load_authorized",
        lambda *_args, **_kwargs: (config, {"status": "ok"}, {"status": "ok"}),
    )
    monkeypatch.setattr(
        runner,
        "_prepare_complete_snapshot",
        lambda *_args: (snapshot, manifest, {}),
    )
    monkeypatch.setattr(runner, "_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(runner, "_worker_command", lambda *_args: ["worker"])
    monkeypatch.setattr(
        runner,
        "_run_supervised",
        lambda *_args: (
            json.dumps({"status": "worker_ok", "result_path": str(tmp_path / "result")}),
            "",
        ),
    )

    def cleanup(path: Path) -> None:
        assert path == snapshot
        events.append("cleanup")

    def finalize(_config, _result_path):
        assert events == ["cleanup"]
        events.append("final_commit")
        return tmp_path / "result"

    monkeypatch.setattr(runner, "_cleanup_snapshot", cleanup)
    monkeypatch.setattr(runner, "_parent_finalize", finalize)
    output = runner.execute_parent(CONFIG_PATH)
    assert output == tmp_path / "result"
    assert events == ["cleanup", "final_commit"]


def test_key_binding_builder_receipt_is_zero_fit_and_hash_bound() -> None:
    config = _json(CONFIG_PATH)
    evidence = config["preexecution_evidence"]
    sidecar = PROJECT_ROOT / evidence["feature_cache_key_binding"]["path"]
    receipt_path = PROJECT_ROOT / evidence["feature_cache_key_binding_receipt"]["path"]
    receipt = _json(receipt_path)
    assert _sha256(sidecar) == evidence["feature_cache_key_binding"]["sha256"]
    assert _sha256(receipt_path) == evidence["feature_cache_key_binding_receipt"]["sha256"]
    assert receipt["source_train_key_digest"] == (
        evidence["feature_cache_key_binding"]["source_train_key_digest"]
    )
    assert all(value == 0 for value in receipt["operation_counters"].values())


def test_import_creates_no_execution_claim_or_fit() -> None:
    runner = _runner()
    artifact = PROJECT_ROOT / "artifacts" / runner.DEFAULT_CONFIG.stem
    assert not (artifact / "execution.lock").exists()
    assert not (artifact / "attempt_journal").exists()
    assert os.environ.get("P1_V1R6_NUMERICAL_EXECUTED") is None
