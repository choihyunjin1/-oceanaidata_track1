from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import scripts.run_p3_corrected_repeated_forward_catboost_v2 as runner
from p3_wave.one_shot_guard import (
    acquire_persistent_attempt_lock,
    authorize_canonical_contract,
    safe_new_stage_path,
)


def test_compiled_config_matches_canonical_path_sha_and_full_payload() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path, cache_path, output_path = runner._canonical_paths(root)
    config, paths = runner.authorize_entry(
        root=root,
        requested_config=config_path,
        requested_cache=cache_path,
        requested_output=output_path,
    )
    assert config == runner.EXPECTED_CONFIG
    assert runner.base.sha256_file(config_path) == runner.EXPECTED_CONFIG_SHA256
    assert paths["output"] == output_path
    assert config["validation"]["gap_hours"] == 78
    assert config["validation"]["footprint_hours"] == 72
    assert config["model"]["fold_seeds"] == [20260816, 20260817, 20260818]
    assert config["router"]["hyperparameter_search"] is False


def test_config_copy_arbitrary_cache_and_arbitrary_output_are_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path, cache_path, output_path = runner._canonical_paths(root)
    copied = tmp_path / "copied.json"
    copied.write_bytes(config_path.read_bytes())
    with pytest.raises(PermissionError, match="non-canonical config"):
        runner.authorize_entry(
            root=root,
            requested_config=copied,
            requested_cache=cache_path,
            requested_output=output_path,
        )
    arbitrary_cache = tmp_path / "cache"
    arbitrary_cache.mkdir()
    with pytest.raises(PermissionError, match="non-canonical cache"):
        runner.authorize_entry(
            root=root,
            requested_config=config_path,
            requested_cache=arbitrary_cache,
            requested_output=output_path,
        )
    with pytest.raises(PermissionError, match="non-canonical output"):
        runner.authorize_entry(
            root=root,
            requested_config=config_path,
            requested_cache=cache_path,
            requested_output=tmp_path / "other-output",
        )


def test_generic_authorizer_enforces_deep_equality_after_sha(tmp_path: Path) -> None:
    config = {"fixed": {"seed": 1, "gap": 78}}
    canonical = tmp_path / "config.json"
    canonical.write_text(json.dumps(config), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    with pytest.raises(PermissionError, match="deep equality"):
        authorize_canonical_contract(
            root=tmp_path,
            requested_config=canonical,
            requested_cache=cache,
            requested_output=tmp_path / "output",
            canonical_config_relative="config.json",
            canonical_cache_relative="cache",
            canonical_output_relative="output",
            expected_config_sha256=digest,
            expected_config={"fixed": {"seed": 2, "gap": 78}},
        )


def test_attempt_lock_is_persistent_o_excl_and_second_attempt_fails(tmp_path: Path) -> None:
    lock = tmp_path / "attempt.lock.json"
    first = acquire_persistent_attempt_lock(
        lock,
        experiment_id="v2",
        config_sha256="a" * 64,
        created_at="2026-08-22T00:00:00+09:00",
    )
    assert first["o_excl"] is True
    assert first["rerun_forbidden"] is True
    with pytest.raises(FileExistsError):
        acquire_persistent_attempt_lock(
            lock,
            experiment_id="v2",
            config_sha256="a" * 64,
            created_at="2026-08-22T00:00:01+09:00",
        )


def test_candidate_target_containment_traversal_protection_and_existing_refusal(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    protected = tmp_path / "protected"
    protected.mkdir()
    valid = safe_new_stage_path(stage, "candidate/submission.csv", protected_roots=(protected,))
    assert valid == (stage / "candidate/submission.csv").resolve()
    with pytest.raises(PermissionError, match="traversing"):
        safe_new_stage_path(stage, "../protected/submission.csv", protected_roots=(protected,))
    valid.parent.mkdir()
    valid.write_text("already exists", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        safe_new_stage_path(stage, "candidate/submission.csv", protected_roots=(protected,))


def test_run_api_has_no_config_cache_or_output_override_and_guard_precedes_lock() -> None:
    signature = inspect.signature(runner.run_experiment)
    assert set(signature.parameters) == {"root", "data_dir"}
    source = inspect.getsource(runner.run_experiment)
    assert source.index("authorize_entry") < source.index("acquire_persistent_attempt_lock")
    assert source.index("acquire_persistent_attempt_lock") < source.index("_run_after_lock")
    with pytest.raises(TypeError, match="unexpected keyword"):
        runner.run_experiment(  # type: ignore[call-arg]
            root=Path("."),
            data_dir=Path("."),
            config_path=Path("copied.json"),
        )


def test_candidate_writer_is_exclusive_and_guarded_before_full_fit() -> None:
    source = inspect.getsource(runner._run_after_lock)
    assert source.index("safe_new_stage_path") < source.index("_fit_full_and_infer")
    writer = inspect.getsource(runner._exclusive_candidate_writer)
    assert "safe_new_stage_path" in writer
    assert 'mode="x"' in writer
    assert "validate_submission" in writer
