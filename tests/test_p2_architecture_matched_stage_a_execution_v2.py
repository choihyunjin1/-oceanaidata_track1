from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from p2_restore.architecture_matched_stage_a_contract_v2 import (
    CONFIG_RELATIVE,
    CONFIG_SHA256,
    ENGINE_RELATIVE,
    IMPLEMENTATION_ROLES,
    StageAContractError,
    contained_path,
    exclusive_bytes,
    implementation_pins,
    load_canonical_config,
    sha256_file,
    validate_config,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_canonical_v2_config_sha_and_copy_rejection(tmp_path: Path) -> None:
    root = _root()
    config = load_canonical_config(root)
    assert sha256_file(root / CONFIG_RELATIVE) == CONFIG_SHA256
    assert config["problem"] == "P2"
    assert config["exact_official_incumbent_comparison"] is False
    assert config["official_promotion_allowed"] is False
    load_canonical_config(root, supplied_config=deepcopy(config))

    copy = tmp_path / "copied.json"
    copy.write_bytes((root / CONFIG_RELATIVE).read_bytes())
    with pytest.raises(StageAContractError, match="canonical v2 config path"):
        load_canonical_config(root, copy)


@pytest.mark.parametrize("problem", ["P1", "P3"])
def test_architecture_matched_mode_rejects_p1_and_p3(problem: str) -> None:
    config = deepcopy(load_canonical_config(_root()))
    config["problem"] = problem
    with pytest.raises(StageAContractError, match="P2-only"):
        validate_config(config)


def test_non_exact_and_no_upload_labels_fail_closed() -> None:
    for key, value in (
        ("exact_official_incumbent_comparison", True),
        ("explicitly_not_exact_official_incumbent", False),
        ("official_promotion_allowed", True),
        ("upload_allowed", True),
    ):
        config = deepcopy(load_canonical_config(_root()))
        config[key] = value
        with pytest.raises(StageAContractError):
            validate_config(config)


def test_implementation_pin_roles_are_exact_and_current() -> None:
    pins = implementation_pins(_root())
    assert list(pins) == list(IMPLEMENTATION_ROLES)
    assert set(pins) == {"CONFIG", "GUARD", "ENGINE", "RUNNER", "TESTS"}
    for role, relative in IMPLEMENTATION_ROLES.items():
        assert pins[role]["path"] == relative
        assert pins[role]["sha256"] == sha256_file(_root() / relative)
        assert pins[role]["bytes"] > 0


def test_output_containment_and_o_excl_rerun_guard(tmp_path: Path) -> None:
    output = tmp_path / "reference"
    assert contained_path(output, "seal.json") == (output / "seal.json").resolve()
    with pytest.raises(StageAContractError, match="unsafe"):
        contained_path(output, "../escape.json")
    with pytest.raises(StageAContractError, match="unsafe"):
        contained_path(output, tmp_path / "absolute.json")

    lock = tmp_path / "control" / "attempt.lock"
    exclusive_bytes(lock, b"first")
    before = lock.read_bytes()
    with pytest.raises(FileExistsError):
        exclusive_bytes(lock, b"second")
    assert lock.read_bytes() == before


def test_execution_plan_is_complete_and_not_scaffolding() -> None:
    engine = importlib.import_module("p2_restore.architecture_matched_stage_a_execution_v2")
    plan = engine.build_execution_plan(load_canonical_config(_root()))
    assert callable(engine.execute_stage_a)
    assert plan["outer_prefix_cells"] == 15
    assert plan["complete_pipeline_seeds"] == [20260823, 20260824, 20260825]
    assert plan["inner_splits_per_cell"] == 3
    assert plan["deep_training_jobs"] == 720
    assert plan["router_training_jobs"] == 180
    assert plan["challenger_jobs"] == 0
    assert plan["submission_predictions"] == 0
    source = (_root() / ENGINE_RELATIVE).read_text(encoding="utf-8")
    assert "NotImplementedError" not in source
    assert "scaffolding_only" not in source


def test_inner_split_recipe_is_strictly_embargoed() -> None:
    engine = importlib.import_module("p2_restore.architecture_matched_stage_a_execution_v2")
    times = pd.date_range("2023-01-01", periods=20_000, freq="10min", tz="UTC")
    splits = engine._inner_splits(  # noqa: SLF001 - contract-level white-box test
        times,
        edges=[0.55, 0.7, 0.85, 1.0],
        calibration_fraction=0.15,
        embargo_days=7,
    )
    assert len(splits) == 3
    for split in splits:
        assert split["inner_train"].max() < split["validation"].min() - pd.Timedelta(days=7)
        assert split["optimization"].max() < split["calibration"].min() - pd.Timedelta(
            days=7
        )


def test_check_only_never_imports_engine_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_p2_architecture_matched_reference_v2 as runner

    imported: list[str] = []
    monkeypatch.setattr(
        runner,
        "static_preflight",
        lambda *_args, **_kwargs: {
            "status": "PASS_STATIC_IMPLEMENTATION_ONLY",
            "files_written": 0,
            "attempt_locks_created": 0,
            "fits": 0,
            "predictions": 0,
            "uploads": 0,
        },
    )
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _name: SimpleNamespace())
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: imported.append(name),
    )
    result = runner.check_only(root=tmp_path, data_dir=tmp_path)
    assert imported == []
    assert result["execution_engine_imported"] is False
    assert result["attempt_lock_created"] is False
    assert result["reference_fit_started"] is False
    assert result["uploads"] == 0
    assert list(tmp_path.iterdir()) == []


def test_run_rejects_missing_qa_before_authorization_lock_or_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_p2_architecture_matched_reference_v2 as runner

    order: list[str] = []
    config = deepcopy(load_canonical_config(_root()))
    monkeypatch.setattr(runner, "load_canonical_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        runner,
        "static_preflight",
        lambda *_args, **_kwargs: order.append("preflight")
        or {"status": "PASS_STATIC_IMPLEMENTATION_ONLY", "implementation_pins": {}},
    )
    monkeypatch.setattr(
        runner,
        "verify_pre_execution_qa",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("QA missing")),
    )
    monkeypatch.setattr(
        runner,
        "verify_execution_authorization",
        lambda *_args, **_kwargs: order.append("authorization"),
    )
    monkeypatch.setattr(
        runner,
        "consume_attempt_lock",
        lambda *_args, **_kwargs: order.append("lock"),
    )
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: order.append(f"import:{name}"),
    )
    with pytest.raises(PermissionError, match="QA missing"):
        runner.run_authorized(root=tmp_path, data_dir=tmp_path)
    assert order == ["preflight"]
    assert not (tmp_path / config["canonical_paths"]["attempt_lock"]).exists()


def test_direct_engine_call_revalidates_qa_before_output(tmp_path: Path) -> None:
    engine = importlib.import_module("p2_restore.architecture_matched_stage_a_execution_v2")
    config_path = tmp_path / CONFIG_RELATIVE
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes((_root() / CONFIG_RELATIVE).read_bytes())
    config = load_canonical_config(tmp_path)
    lock = tmp_path / config["canonical_paths"]["attempt_lock"]
    lock.parent.mkdir(parents=True)
    lock.write_text("{}\n", encoding="utf-8")
    output = tmp_path / config["canonical_paths"]["output"]

    with pytest.raises(PermissionError, match="QA receipt is missing"):
        engine.execute_stage_a(
            root=tmp_path,
            data_dir=tmp_path,
            config=config,
            preflight={
                "status": "PASS_STATIC_IMPLEMENTATION_ONLY",
                "implementation_pins": {},
            },
            attempt_lock=lock,
        )
    assert not output.exists()


def test_authorized_order_is_qa_auth_lock_then_late_engine_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_p2_architecture_matched_reference_v2 as runner

    order: list[str] = []
    config = deepcopy(load_canonical_config(_root()))
    engine_path = tmp_path / ENGINE_RELATIVE
    engine_path.parent.mkdir(parents=True)
    engine_path.write_text("# test engine path\n", encoding="utf-8")
    lock = tmp_path / "attempt.lock"
    lock.write_text("locked\n", encoding="utf-8")
    pins = {"ENGINE": {"path": ENGINE_RELATIVE, "sha256": "a" * 64, "bytes": 1}}
    monkeypatch.setattr(runner, "load_canonical_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        runner,
        "static_preflight",
        lambda *_args, **_kwargs: order.append("preflight")
        or {"status": "PASS_STATIC_IMPLEMENTATION_ONLY", "implementation_pins": pins},
    )
    monkeypatch.setattr(
        runner,
        "verify_pre_execution_qa",
        lambda *_args, **_kwargs: order.append("qa") or ({"qa": True}, "b" * 64),
    )
    monkeypatch.setattr(
        runner,
        "verify_execution_authorization",
        lambda *_args, **_kwargs: order.append("authorization")
        or ({"authorized": True}, "c" * 64),
    )
    monkeypatch.setattr(
        runner,
        "implementation_pins",
        lambda *_args, **_kwargs: order.append("pins") or pins,
    )
    monkeypatch.setattr(
        runner,
        "consume_attempt_lock",
        lambda *_args, **_kwargs: order.append("lock") or lock,
    )
    engine = SimpleNamespace(
        __file__=str(engine_path),
        execute_stage_a=lambda **_kwargs: order.append("fit") or {"complete": True},
    )
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda _name: order.append("import") or engine,
    )
    monkeypatch.setattr(
        runner,
        "verify_stage_a_seal",
        lambda *_args, **_kwargs: order.append("seal") or {"verified": True},
    )
    result = runner.run_authorized(root=tmp_path, data_dir=tmp_path)
    assert order == ["preflight", "qa", "authorization", "pins", "lock", "import", "fit", "seal"]
    assert result["exact_official_incumbent_comparison"] is False
    assert result["official_promotion_allowed"] is False
    assert result["uploads"] == 0
