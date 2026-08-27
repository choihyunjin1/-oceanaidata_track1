from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

import p3_wave.hierarchical_residual_basis_dense72_contract_r4 as guard
import p3_wave.hierarchical_residual_basis_dense72_execution_r4 as engine

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path.home() / "Downloads/p3/데이터셋_P3/P3_wave_forecast"
CONFIG = ROOT / guard.CONFIG_RELATIVE
RUNNER = ROOT / "scripts/run_p3_hierarchical_residual_basis_dense72_r4.py"
ENGINE = ROOT / "src/p3_wave/hierarchical_residual_basis_dense72_execution_r4.py"


@pytest.fixture(scope="module")
def canonical_preflight() -> tuple[dict[str, object], dict[str, object]]:
    return guard.prepare_execution_preflight(ROOT, DATA_DIR)


def test_config_identity_correction_contract_and_no_personal_root() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert guard.sha256_file(CONFIG) == guard.CONFIG_SHA256
    assert set(config["implementation_roles"]) == guard.EXPECTED_ROLES
    assert config["correction_contract"][
        "attempt_lock_created_and_deep_verified_before_capability_mint"
    ] is True
    assert config["correction_contract"]["capability_replay_allowed"] is False
    assert config["correction_contract"][
        "preflight_full_train_wave_hs_float_decode_allowed"
    ] is False
    assert config["correction_contract"][
        "full_predecessor_transitive_dependency_closure_pinned"
    ] is True
    assert config["correction_contract"][
        "raw_delta_fold_commitment_precedes_validation_current_hs_decode"
    ] is True
    assert config["correction_contract"][
        "canonical_raw_npy_readonly_memmap_identity_required"
    ] is True
    assert config["correction_contract"][
        "attempt_lock_full_payload_deep_equal_except_created_at_kst"
    ] is True
    assert config["target_free_split"]["historical_label_derived"] is True
    assert "C:\\Users" not in CONFIG.read_text(encoding="utf-8")


def test_full_predecessor_transitive_dependency_closure_is_live_pinned() -> None:
    config, _predecessor = guard.load_canonical_config(ROOT)
    pins = guard.implementation_pins(ROOT, config)
    r3_config, _r3_predecessor = guard.predecessor_r3_guard.load_canonical_config(
        ROOT
    )
    r3_pins = guard.predecessor_r3_guard.implementation_pins(ROOT, r3_config)
    expected = {f"R3_TRANSITIVE_{role}" for role in r3_pins}
    assert expected.issubset(pins)
    assert len(expected) == len(r3_pins)
    for role, expected_pin in r3_pins.items():
        assert pins[f"R3_TRANSITIVE_{role}"] == expected_pin


def test_check_state_has_no_qa_authorization_lock_or_output() -> None:
    config, _preflight = guard.load_canonical_config(ROOT)
    paths = guard.stage_paths(ROOT, config)
    assert all(not path.exists() for path in paths.values())


def test_operational_snapshot_deep_binds_in_memory_comparator_object(
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    _config, preflight = canonical_preflight
    metrics = preflight["gen1_metrics"]
    assert isinstance(metrics, dict)
    metrics["__r4_test_poison"] = True
    try:
        assert guard.operational_snapshot(preflight) != preflight[
            "operational_snapshot"
        ]
    finally:
        metrics.pop("__r4_test_poison")
    assert guard.operational_snapshot(preflight) == preflight["operational_snapshot"]


def test_raw_snapshot_binds_canonical_memmap_and_array_bytes_digest(
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    config, preflight = canonical_preflight
    raw = preflight["raw"]
    identity = guard.validate_canonical_raw_memmap(
        ROOT,
        raw,
        Path(preflight["input_paths"]["sequence_cache/train_values.npy"]),
        config,
    )
    assert identity == preflight["operational_snapshot"]["raw"]
    assert identity["exact_type"] == "numpy.memmap"
    assert identity["mode"] == "r"
    assert identity["offset_bytes"] == 128
    assert identity["array_bytes_sha256"] == config["raw_memmap_contract"][
        "canonical_array_bytes_sha256"
    ]


def test_live_preflight_rejects_same_shape_dtype_poisoned_raw_proxy(
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    config, preflight = canonical_preflight
    original = preflight["raw"]
    assert isinstance(original, np.memmap)
    poison = np.lib.stride_tricks.as_strided(
        np.zeros(1, dtype=original.dtype),
        shape=original.shape,
        strides=(0, 0, 0),
        writeable=False,
    )
    preflight["raw"] = poison
    try:
        with pytest.raises(PermissionError, match="exact numpy.memmap"):
            guard._validate_live_preflight(ROOT, config, preflight)
    finally:
        preflight["raw"] = original
    assert guard.operational_snapshot(preflight) == preflight["operational_snapshot"]


@pytest.mark.parametrize("variant", ["path", "mode", "offset", "shape", "dtype"])
def test_raw_memmap_path_mode_offset_shape_dtype_are_exact(
    variant: str,
    tmp_path: Path,
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    config, preflight = canonical_preflight
    canonical = Path(preflight["input_paths"]["sequence_cache/train_values.npy"])
    supplied = canonical
    if variant == "path":
        supplied = tmp_path / "raw-alias.npy"
        os.link(canonical, supplied)
        candidate = np.load(supplied, mmap_mode="r")
    elif variant == "mode":
        candidate = np.load(canonical, mmap_mode="r+")
    elif variant == "offset":
        candidate = np.memmap(
            canonical,
            dtype=np.dtype("<f4"),
            mode="r",
            offset=124,
            shape=(24_360, 289, 10),
        )
    elif variant == "shape":
        candidate = np.memmap(
            canonical,
            dtype=np.dtype("<f4"),
            mode="r",
            offset=128,
            shape=(24_360, 2_890),
        )
    else:
        candidate = np.memmap(
            canonical,
            dtype=np.dtype(">f4"),
            mode="r",
            offset=128,
            shape=(24_360, 289, 10),
        )
    try:
        with pytest.raises(PermissionError, match="raw (input path|memmap identity)"):
            guard.validate_canonical_raw_memmap(ROOT, candidate, supplied, config)
    finally:
        candidate._mmap.close()


@pytest.mark.parametrize(
    "forged_flag", ["candidate_or_test_prediction_allowed", "upload_allowed"]
)
def test_consumed_attempt_lock_rejects_prediction_or_upload_flag_forgery(
    forged_flag: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    config, preflight = canonical_preflight
    qa = tmp_path / "qa.json"
    authorization = tmp_path / "authorization.json"
    attempt_lock = tmp_path / "attempt.lock"
    qa.write_bytes(b"qa-fixture\n")
    authorization.write_bytes(b"authorization-fixture\n")
    payload = guard._lock_payload(
        ROOT,
        config,
        preflight,
        qa_sha256=guard.sha256_file(qa),
        authorization_sha256=guard.sha256_file(authorization),
        created_at_kst="2026-08-23T08:45:00+09:00",
    )
    payload[forged_flag] = True
    attempt_lock.write_bytes(guard.canonical_json_bytes(payload) + b"\n")
    fake_paths = {
        "attempt_lock": attempt_lock,
        "pre_execution_qa": qa,
        "authorization": authorization,
    }
    monkeypatch.setattr(guard, "stage_paths", lambda *_args, **_kwargs: fake_paths)
    _patch_lock_receipt_verifiers(monkeypatch, qa, authorization)
    with pytest.raises(PermissionError, match="complete payload differs"):
        guard.verify_consumed_attempt_lock(ROOT, config, preflight)


def _patch_lock_receipt_verifiers(
    monkeypatch: pytest.MonkeyPatch,
    qa: Path,
    authorization: Path,
) -> None:
    qa_sha = guard.sha256_file(qa)
    authorization_sha = guard.sha256_file(authorization)

    def verify_qa(
        _root: Path,
        _config: dict[str, object],
        _preflight: dict[str, object],
    ) -> tuple[dict[str, object], str]:
        return {}, qa_sha

    def verify_authorization(
        _root: Path,
        _config: dict[str, object],
        _preflight: dict[str, object],
        *,
        qa_sha256: str,
        allow_consumed_attempt_lock: bool = False,
    ) -> tuple[dict[str, object], str]:
        assert qa_sha256 == qa_sha
        assert allow_consumed_attempt_lock is True
        return {}, authorization_sha

    monkeypatch.setattr(guard, "verify_pre_execution_qa", verify_qa)
    monkeypatch.setattr(guard, "verify_execution_authorization", verify_authorization)


def _write_lock_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object],
    preflight: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    qa = tmp_path / "qa.json"
    authorization = tmp_path / "authorization.json"
    attempt_lock = tmp_path / "attempt.lock"
    qa.write_bytes(b"qa-fixture\n")
    authorization.write_bytes(b"authorization-fixture\n")
    payload = guard._lock_payload(
        ROOT,
        config,
        preflight,
        qa_sha256=guard.sha256_file(qa),
        authorization_sha256=guard.sha256_file(authorization),
        created_at_kst="2026-08-23T08:45:00+09:00",
    )
    fake_paths = {
        "attempt_lock": attempt_lock,
        "pre_execution_qa": qa,
        "authorization": authorization,
    }
    monkeypatch.setattr(guard, "stage_paths", lambda *_args, **_kwargs: fake_paths)
    _patch_lock_receipt_verifiers(monkeypatch, qa, authorization)
    return attempt_lock, payload


def test_consumed_attempt_lock_accepts_only_the_exact_canonical_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    config, preflight = canonical_preflight
    attempt_lock, payload = _write_lock_fixture(
        tmp_path, monkeypatch, config, preflight
    )
    attempt_lock.write_bytes(guard.canonical_json_bytes(payload) + b"\n")
    expected_sha = guard.sha256_file(attempt_lock)
    observed, lock_sha = guard.verify_consumed_attempt_lock(
        ROOT,
        config,
        preflight,
        expected_lock_sha256=expected_sha,
    )
    assert observed == payload
    assert lock_sha == expected_sha


def test_consumed_attempt_lock_rejects_nested_operational_snapshot_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    config, preflight = canonical_preflight
    attempt_lock, payload = _write_lock_fixture(
        tmp_path, monkeypatch, config, preflight
    )
    forged = copy.deepcopy(payload)
    forged["operational_snapshot"]["raw"]["array_bytes_sha256"] = "f" * 64
    attempt_lock.write_bytes(guard.canonical_json_bytes(forged) + b"\n")
    with pytest.raises(PermissionError, match="complete payload differs"):
        guard.verify_consumed_attempt_lock(ROOT, config, preflight)


def test_consumed_attempt_lock_revalidates_live_qa_and_authorization_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    config, preflight = canonical_preflight
    attempt_lock, payload = _write_lock_fixture(
        tmp_path, monkeypatch, config, preflight
    )
    attempt_lock.write_bytes(guard.canonical_json_bytes(payload) + b"\n")

    def reject_qa(*_args: object, **_kwargs: object) -> tuple[dict[str, object], str]:
        raise PermissionError("post-lock QA semantic rejection")

    monkeypatch.setattr(guard, "verify_pre_execution_qa", reject_qa)
    with pytest.raises(PermissionError, match="post-lock QA semantic rejection"):
        guard.verify_consumed_attempt_lock(ROOT, config, preflight)

    qa_sha = guard.sha256_file(tmp_path / "qa.json")
    monkeypatch.setattr(
        guard,
        "verify_pre_execution_qa",
        lambda *_args, **_kwargs: ({}, qa_sha),
    )

    def reject_authorization(
        *_args: object,
        allow_consumed_attempt_lock: bool = False,
        **_kwargs: object,
    ) -> tuple[dict[str, object], str]:
        assert allow_consumed_attempt_lock is True
        raise PermissionError("post-lock authorization semantic rejection")

    monkeypatch.setattr(guard, "verify_execution_authorization", reject_authorization)
    with pytest.raises(
        PermissionError, match="post-lock authorization semantic rejection"
    ):
        guard.verify_consumed_attempt_lock(ROOT, config, preflight)


def test_lock_creation_rechecks_qa_and_fails_before_lock(
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    config, preflight = canonical_preflight
    paths = guard.stage_paths(ROOT, config)
    with pytest.raises(PermissionError, match="QA receipt is missing"):
        guard.create_and_verify_attempt_lock(ROOT, config, preflight)
    assert not paths["attempt_lock"].exists()


def test_capability_cannot_be_minted_before_consumed_lock(
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    config, preflight = canonical_preflight
    with pytest.raises(PermissionError, match="attempt lock is missing"):
        guard.issue_execution_capability(
            ROOT,
            config,
            preflight,
            lock_sha256="0" * 64,
        )


def test_direct_engine_and_private_curve_reject_without_live_locked_phase(
    tmp_path: Path,
    canonical_preflight: tuple[dict[str, object], dict[str, object]],
) -> None:
    config, preflight = canonical_preflight
    with pytest.raises(PermissionError, match="capability"):
        engine.execute_curve_stage(
            capability=object(),
            root=ROOT,
            data_dir=DATA_DIR,
            config=config,
            preflight=preflight,
        )
    with pytest.raises(PermissionError, match="capability"):
        engine._run_curve(
            capability=object(),
            root=ROOT,
            data_dir=DATA_DIR,
            config=config,
            preflight=preflight,
            stage=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_live_capability_phase_is_single_use_even_for_identity_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = guard.ExecutionCapability(
        root_st_dev=int(ROOT.stat().st_dev),
        root_st_ino=int(ROOT.stat().st_ino),
        config_sha256=guard.CONFIG_SHA256,
        canonical_stage_relative="artifacts/example",
        attempt_lock_sha256="0" * 64,
        static_preflight_sha256="1" * 64,
        operational_snapshot_sha256="2" * 64,
        operational_snapshot_canonical_json="{}",
        qa_sha256="3" * 64,
        authorization_sha256="4" * 64,
        nonce="5" * 64,
    )
    monkeypatch.setattr(guard, "_LIVE_CAPABILITY", capability)
    monkeypatch.setattr(guard, "_LIVE_PHASE", "LOCK_VERIFIED_CAPABILITY_MINTED")
    monkeypatch.setattr(guard, "_LIVE_TEMP_STAGE", None)
    monkeypatch.setattr(guard, "_require_core", lambda *_args, **_kwargs: capability)
    with pytest.raises(PermissionError, match="single-use live.*curve phase"):
        guard.require_curve_capability(
            capability,
            root=ROOT,
            config={},
            preflight={},
            temporary_stage=tmp_path,
        )
    monkeypatch.setattr(guard, "_LIVE_PHASE", "CURVE_CALL_AUTHORIZED")
    monkeypatch.setattr(guard, "_LIVE_TEMP_STAGE", tmp_path.resolve())
    guard.require_curve_capability(
        capability,
        root=ROOT,
        config={},
        preflight={},
        temporary_stage=tmp_path,
    )
    with pytest.raises(PermissionError, match="single-use live.*curve phase"):
        guard.require_curve_capability(
            capability,
            root=ROOT,
            config={},
            preflight={},
            temporary_stage=tmp_path,
        )


def test_runner_and_engine_compile_lock_then_capability_then_single_use_curve() -> None:
    runner_source = RUNNER.read_text(encoding="utf-8")
    body = runner_source[runner_source.index("def run_once") :]
    lock = body.index("create_and_verify_attempt_lock")
    mint = body.index("issue_execution_capability")
    engine_import = body.index("importlib.import_module(ENGINE_MODULE)")
    assert lock < mint < engine_import
    assert 'default="check-only"' in runner_source

    tree = ast.parse(ENGINE.read_text(encoding="utf-8"), filename=str(ENGINE))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    private_first = functions["_run_curve"].body[0]
    public_first = functions["execute_curve_stage"].body[0]
    assert isinstance(private_first, ast.Expr)
    assert isinstance(private_first.value, ast.Call)
    assert isinstance(private_first.value.func, ast.Name)
    assert private_first.value.func.id == "require_curve_capability"
    assert isinstance(public_first, ast.Expr)
    assert isinstance(public_first.value, ast.Call)
    assert isinstance(public_first.value.func, ast.Name)
    assert public_first.value.func.id == "begin_execution_stage"
