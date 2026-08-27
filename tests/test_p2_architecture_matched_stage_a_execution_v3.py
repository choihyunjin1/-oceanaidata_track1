from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from p2_restore import architecture_matched_stage_a_contract_v3 as guard


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_canonical_v3_config_sha_and_copy_rejection(tmp_path: Path) -> None:
    root = _root()
    config = guard.load_canonical_config(root)
    assert guard.sha256_file(root / guard.CONFIG_RELATIVE) == guard.CONFIG_SHA256
    assert config["schema_version"] == "p2_architecture_matched_stage_a_execution.v3"
    copy = tmp_path / "copy.json"
    copy.write_bytes((root / guard.CONFIG_RELATIVE).read_bytes())
    with pytest.raises(guard.StageAContractV3Error, match="canonical v3 config path"):
        guard.load_canonical_config(root, copy)


@pytest.mark.parametrize("problem", ["P1", "P3"])
def test_architecture_matched_mode_rejects_p1_and_p3(problem: str) -> None:
    overlay = deepcopy(guard.load_canonical_overlay(_root()))
    overlay["problem"] = problem
    with pytest.raises(guard.StageAContractV3Error, match="P2-only"):
        guard.validate_overlay(overlay)


def test_non_exact_and_no_promotion_labels_fail_closed() -> None:
    root = _root()
    for key, value in (
        ("exact_official_incumbent_comparison", True),
        ("official_promotion_allowed", True),
        ("upload_allowed", True),
        ("explicitly_not_exact_official_incumbent", False),
    ):
        overlay = deepcopy(guard.load_canonical_overlay(root))
        overlay[key] = value
        with pytest.raises(guard.StageAContractV3Error):
            guard.validate_overlay(overlay)


def test_exact_implementation_and_runtime_source_roles_are_current() -> None:
    root = _root()
    overlay = guard.load_canonical_overlay(root)
    assert list(overlay["implementation_roles"]) == [
        "CONFIG",
        "GUARD",
        "ENGINE",
        "RUNNER",
        "TESTS",
    ]
    assert set(guard.implementation_pins(root)) == set(guard.IMPLEMENTATION_ROLES)
    assert set(overlay["immutable_v2_implementation_pins"]) == guard.IMMUTABLE_V2_ROLES
    assert set(overlay["additional_runtime_source_pins"]) == {
        "MODEL_MODULE",
        "PACKAGE_INIT",
    }
    verified = guard._verify_exact_pin_map(
        root,
        overlay["additional_runtime_source_pins"],
        label="additional runtime source",
    )
    assert verified == overlay["additional_runtime_source_pins"]
    immutable = guard._verify_exact_pin_map(
        root,
        overlay["immutable_v2_implementation_pins"],
        label="immutable v2 implementation",
    )
    assert immutable == overlay["immutable_v2_implementation_pins"]


def test_model_or_package_source_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    pins = guard.load_canonical_overlay(root)["additional_runtime_source_pins"]
    real_sha = guard.sha256_file

    def drift(path: Path) -> str:
        if path.resolve() == (root / "src/p2_restore/model.py").resolve():
            return "0" * 64
        return real_sha(path)

    monkeypatch.setattr(guard, "sha256_file", drift)
    with pytest.raises(guard.StageAContractV3Error, match="MODEL_MODULE"):
        guard._verify_exact_pin_map(root, pins, label="additional runtime source")


def test_output_containment_and_o_excl_rerun_guard(tmp_path: Path) -> None:
    output = tmp_path / "reference"
    assert guard.contained_path(output, "seal.json") == (output / "seal.json").resolve()
    with pytest.raises(guard.StageAContractV3Error):
        guard.contained_path(output, "../escape.json")
    lock = tmp_path / "control" / "attempt.lock"
    guard.exclusive_bytes(lock, b"first")
    before = lock.read_bytes()
    with pytest.raises(FileExistsError):
        guard.exclusive_bytes(lock, b"second")
    assert lock.read_bytes() == before


def test_execution_plan_is_complete_and_not_scaffolding() -> None:
    engine = importlib.import_module("p2_restore.architecture_matched_stage_a_execution_v3")
    plan = engine.build_execution_plan(guard.load_canonical_config(_root()))
    assert callable(engine.execute_stage_a)
    assert plan["outer_prefix_cells"] == 15
    assert plan["deep_training_jobs"] == 720
    assert plan["router_training_jobs"] == 180
    assert plan["complete_pipeline_seeds"] == [20260823, 20260824, 20260825]
    assert plan["deep_components"] == [
        "depth_query_bitcn",
        "lsti_style",
        "timemixerpp_style",
        "moment_units_scratch",
    ]
    assert plan["implementation_lineage"].startswith("BYTE_PINNED_V2")


def test_direct_engine_call_ignores_forged_preflight_and_runs_fresh_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = importlib.import_module("p2_restore.architecture_matched_stage_a_execution_v3")
    config = guard.load_canonical_config(_root())
    order: list[str] = []

    def stop_at_fresh(*_args: object, **kwargs: object) -> dict[str, object]:
        order.append("fresh_preflight")
        assert kwargs["supplied_config"] is config
        raise guard.StageAContractV3Error("source drift")

    monkeypatch.setattr(engine, "static_preflight", stop_at_fresh)
    monkeypatch.setattr(
        engine,
        "_load_pinned_v2_engine",
        lambda *_args: order.append("late_import"),
    )
    output = _root() / config["canonical_paths"]["output"]
    assert not output.exists()
    with pytest.raises(guard.StageAContractV3Error, match="source drift"):
        engine.execute_stage_a(
            root=_root(),
            data_dir=tmp_path,
            config=config,
            preflight={"status": "FORGED_PASS", "implementation_pins": {}},
            attempt_lock=tmp_path / "forged.lock",
        )
    assert order == ["fresh_preflight"]
    assert not output.exists()


def test_engine_source_orders_all_guards_before_output_or_fit() -> None:
    source = (_root() / guard.ENGINE_RELATIVE).read_text(encoding="utf-8")
    start = source.index("def execute_stage_a(")
    body = source[start:]
    positions = [
        body.index("fresh_preflight = static_preflight("),
        body.index("engine_v2 = _load_pinned_v2_engine("),
        body.index("runtime = engine_v2._verify_runtime("),
        body.index("data_pins = engine_v2._verify_data_pins("),
        body.index("verify_consumed_attempt_lock("),
        body.index("os.mkdir(output)"),
        body.index("engine_v2._run_cell_seed("),
    ]
    assert positions == sorted(positions)
    assert "del preflight" in body[: positions[0]]


def test_check_only_parent_import_boundary_uses_isolated_child() -> None:
    root = _root()
    script = r'''
import json
import sys
from pathlib import Path
from scripts import run_p2_architecture_matched_reference_v3 as runner
before = sorted(sys.modules)
runner._isolated_preflight = lambda **kwargs: {
    "status": "PASS_STATIC_IMPLEMENTATION_ONLY",
    "preflight_process_loaded_numerical_modules": ["numpy", "pandas"],
    "runtime_probe": {
        "v2_execution_engine_imported_in_isolated_probe": True,
        "loaded_numerical_modules": ["torch"],
    },
}
result = runner.check_only(root=Path(sys.argv[1]), data_dir=Path(sys.argv[1]))
print(json.dumps({
    "result": result,
    "guard_loaded": runner.GUARD_MODULE in sys.modules,
    "engine_loaded": runner.ENGINE_MODULE in sys.modules,
    "new_modules": sorted(set(sys.modules) - set(before)),
}))
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "src")))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(completed.stdout)
    parent = observed["result"]["check_only_parent_process"]
    assert observed["guard_loaded"] is False
    assert observed["engine_loaded"] is False
    assert parent["new_numerical_modules"] == []
    assert parent["guard_imported_after"] is False
    assert parent["engine_imported_after"] is False
    assert observed["result"]["preflight_process_loaded_numerical_modules"] == [
        "numpy",
        "pandas",
    ]
    assert observed["result"]["runtime_probe"][
        "v2_execution_engine_imported_in_isolated_probe"
    ] is True


def test_run_rejects_missing_qa_before_lock_or_engine_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts import run_p2_architecture_matched_reference_v3 as runner

    config = guard.load_canonical_config(_root())
    order: list[str] = []
    fake_guard = type(
        "FakeGuard",
        (),
        {
            "__file__": str(_root() / runner.GUARD_RELATIVE),
            "load_canonical_config": staticmethod(lambda *_args, **_kwargs: config),
            "static_preflight": staticmethod(
                lambda *_args, **_kwargs: order.append("preflight")
                or {"implementation_pins": {}}
            ),
            "verify_pre_execution_qa": staticmethod(
                lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("missing QA"))
            ),
        },
    )

    def fake_import(name: str) -> object:
        order.append(f"import:{name}")
        if name == runner.GUARD_MODULE:
            return fake_guard
        raise AssertionError("engine import must not occur")

    monkeypatch.setattr(runner.importlib, "import_module", fake_import)
    with pytest.raises(PermissionError, match="missing QA"):
        runner.run_authorized(root=_root(), data_dir=tmp_path)
    assert order == [f"import:{runner.GUARD_MODULE}", "preflight"]


def test_static_state_has_no_v3_qa_auth_lock_or_output() -> None:
    root = _root()
    config = guard.load_canonical_config(root)
    paths = guard.stage_paths(root, config)
    assert {key: path.exists() for key, path in paths.items()} == {
        "output": False,
        "control": False,
        "pre_execution_qa": False,
        "authorization": False,
        "attempt_lock": False,
    }


def test_runtime_probe_is_explicitly_isolated_and_truthful_in_source() -> None:
    source = (_root() / guard.IMPLEMENTATION_ROLES["GUARD"]).read_text(encoding="utf-8")
    assert "subprocess.run(" in source
    assert '"loaded_numerical_modules": loaded' in source
    assert '"v2_execution_engine_imported_in_isolated_probe": True' in source
    assert '"runtime_probe_isolated": True' in source
