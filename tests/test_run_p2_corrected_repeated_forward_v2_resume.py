from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _resume_module():
    path = Path("scripts/run_p2_corrected_repeated_forward_v2_resume.py").resolve()
    spec = importlib.util.spec_from_file_location("p2_v2_resume", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_order_binds_completed_models_and_only_missing_roles() -> None:
    module = _resume_module()
    assert [item["seed"] for item in module.FIT_ORDER] == [
        20260822,
        20260823,
        20260832,
        20260833,
        20260842,
        20260843,
        20270822,
    ]
    assert [item["role"] for item in module.FIT_ORDER[:3]] == [
        "outer_2024_sep_oct_inner",
        "outer_2024_sep_oct_outer",
        "outer_2025_may_jun_inner",
    ]
    assert all(item["sha256"] for item in module.FIT_ORDER[:3])
    assert all(item["sha256"] is None for item in module.FIT_ORDER[3:])


def test_interruption_disclosure_counts_fold1_outer_materialization_exactly() -> None:
    module = _resume_module()
    disclosure = module.INTERRUPTION_RESUME_DISCLOSURE
    fold1 = disclosure["fold1_outer"]
    assert disclosure["literal_inference_once_claimed"] is False
    assert fold1 == {
        "initial_ephemeral_unexposed_inference_invocations": 1,
        "resume_saved_model_materialization_inference_invocations": 1,
        "total_inference_invocations": 2,
        "blend_model_fits_initial_attempt": 1,
        "blend_model_refits_during_resume": 0,
        "outer_metric_exposures": 1,
        "metric_exposure_timing": "final resumed aggregate only",
        "persisted_oof_before_resume": False,
        "resume_role": "missing persisted materialization from the pinned saved model",
    }


def test_load_only_role_never_calls_fit_or_writes() -> None:
    module = _resume_module()
    sentinel = object()

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("completed partial model was refit")

    runner = SimpleNamespace(fit_fixed_blend=forbidden_fit)
    model, fitted = module._obtain_model(
        module.FIT_ORDER[0],
        {module.FIT_ORDER[0]["role"]: sentinel},
        runner,
        object(),
        object(),
        np.ones(3, dtype=bool),
    )
    assert model is sentinel
    assert fitted is False


def test_exclusive_materialize_refuses_overwrite_and_preserves_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _resume_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"frozen")
    before = module._sha256(target)
    with pytest.raises(FileExistsError, match="already exists"):
        module._exclusive_materialize(target, lambda path: path.write_bytes(b"changed"))
    assert module._sha256(target) == before
    assert target.read_bytes() == b"frozen"


def test_exclusive_materialize_creates_new_file_without_tmp_residue(tmp_path: Path) -> None:
    module = _resume_module()
    target = tmp_path / "new.bin"
    module._exclusive_materialize(target, lambda path: path.write_bytes(b"new"))
    assert target.read_bytes() == b"new"
    assert not target.with_name(target.name + ".resume.tmp").exists()


def test_resume_lock_is_concurrency_only_and_requires_reviewer_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _resume_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    lock = tmp_path / "resume.lock"
    with pytest.raises(PermissionError, match="reviewer authorization"):
        module._acquire_resume_lock("wrong", lock)
    assert not lock.exists()
    record = module._acquire_resume_lock(module.AUTHORIZATION_TOKEN, lock)
    assert record["sha256"]
    payload = lock.read_text(encoding="utf-8")
    assert "CONCURRENCY_GUARD_FOR_SAME_ATTEMPT_RESUME_ONLY" in payload
    assert '"new_experiment_or_attempt": false' in payload
    with pytest.raises(FileExistsError, match="already exists"):
        module._acquire_resume_lock(module.AUTHORIZATION_TOKEN, lock)


def test_pinned_core_files_match_static_resume_contract() -> None:
    module = _resume_module()
    paths = {
        "config": module.CONFIG_PATH,
        "helper": module.HELPER_PATH,
        "runner": module.RUNNER_PATH,
        "attempt_lock": module.ATTEMPT_LOCK,
        "interrupted_status": module.OUTPUT_DIR / "status.json",
    }
    for role, path in paths.items():
        assert module._sha256(path) == module.PINNED_CORE_SHA256[role]


def test_transitive_runtime_modules_are_pinned_before_runner_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _resume_module()
    verified: list[Path] = []

    def record(path: Path, expected: str, *, role: str) -> None:
        assert expected
        assert role
        verified.append(path.resolve())

    def fake_load_runner():
        for relative in module.PINNED_RUNTIME_MODULE_SHA256:
            assert (module.ROOT / relative).resolve() in verified
        return SimpleNamespace(
            _load_config=lambda path: {"canonical": True},
            _canonical_preflight=lambda config, path, output: config,
        )

    monkeypatch.setattr(module, "_verify_hash", record)
    monkeypatch.setattr(module, "_load_original_runner", fake_load_runner)
    _, canonical = module._verify_core_and_attempt()
    assert canonical == {"canonical": True}


def test_pinned_runtime_module_hashes_match_files() -> None:
    module = _resume_module()
    for relative, expected in module.PINNED_RUNTIME_MODULE_SHA256.items():
        assert module._sha256(module.ROOT / relative) == expected


def test_model_validation_checks_full_canonical_parameter_map() -> None:
    module = _resume_module()
    model = module.joblib.load(module._model_path(module.FIT_ORDER[0]["role"]))
    base = SimpleNamespace(feature_columns=model.base_model.feature_columns)
    lean = SimpleNamespace(feature_columns=model.lean_model.feature_columns)
    module._validate_model(model, base, lean, seed=20260822, role="pinned")
    model.base_model.estimator.set_params(n_jobs=7)
    with pytest.raises(ValueError, match="full estimator parameters changed"):
        module._validate_model(model, base, lean, seed=20260822, role="changed")


def test_resume_reapplies_fixed_per_fold_coverage_gate() -> None:
    module = _resume_module()
    config = {
        "validation": {
            "minimum_inner_target_coverage": 0.96,
            "minimum_outer_target_coverage": 0.96,
            "folds": [{"name": "f1", "inner": ["i0", "i1"], "outer": ["o0", "o1"]}],
        }
    }
    passing = SimpleNamespace(
        _coverage=lambda population, start, stop: {"target_coverage": 0.96}
    )
    assert module._validate_fold_coverage(passing, config, object())["f1"]
    values = iter([{"target_coverage": 0.959}, {"target_coverage": 1.0}])
    failing = SimpleNamespace(_coverage=lambda population, start, stop: next(values))
    with pytest.raises(ValueError, match="fails the fixed coverage floor"):
        module._validate_fold_coverage(failing, config, object())


def test_historical_status_is_immutable_and_completion_receipt_is_append_only() -> None:
    module = _resume_module()
    assert module._sha256(module.OUTPUT_DIR / "status.json") == (
        module.PINNED_CORE_SHA256["interrupted_status"]
    )
    assert module.RESUME_COMPLETION_RECEIPT == (
        module.OUTPUT_DIR / "resume_completion_status.json"
    )
    assert not module.RESUME_COMPLETION_RECEIPT.exists()
