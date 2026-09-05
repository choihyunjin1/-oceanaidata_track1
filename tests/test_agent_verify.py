from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "agent_verify", Path(__file__).resolve().parents[1] / "scripts/agent_verify.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def successful(command):
    for argument in command:
        if argument.startswith("--junitxml="):
            Path(argument.split("=", 1)[1]).write_text(
                '<testsuites><testsuite tests="1" skipped="0"/></testsuites>', encoding="utf-8"
            )
    return SimpleNamespace(returncode=0, stdout="ok", stderr="")


@pytest.fixture
def repo(tmp_path):
    for directory in ("tests", "scripts", "src", "configs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "tests/test_unit.py").write_text("def test_ok(): pass", encoding="utf-8")
    (tmp_path / "scripts/unit.py").write_text("VALUE = 1", encoding="utf-8")
    return tmp_path


def test_unchanged_pass_reused_dependency_change_invalidates(repo, monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTEST_ADDOPTS"] == ""
        return successful(args[0])

    monkeypatch.setattr(VERIFY.subprocess, "run", run)
    assert (
        VERIFY.verify(repo, ["tests/test_unit.py"], ["scripts/unit.py"], reuse=True)["status"]
        == "PASS"
    )
    assert VERIFY.verify(repo, ["tests/test_unit.py"], ["scripts/unit.py"], reuse=True)["reused"]
    assert len(calls) == 2
    (repo / "src/dependency.py").write_text("NEW = True", encoding="utf-8")
    assert not VERIFY.verify(repo, ["tests/test_unit.py"], ["scripts/unit.py"], reuse=True)[
        "reused"
    ]
    assert len(calls) == 4


def test_failure_never_cached_and_stops_before_lint(repo, monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stdout="failed", stderr="")

    monkeypatch.setattr(VERIFY.subprocess, "run", run)
    for _ in range(2):
        assert (
            VERIFY.verify(repo, ["tests/test_unit.py"], ["scripts/unit.py"], reuse=True)["status"]
            == "FAIL"
        )
    assert len(calls) == 2
    assert not list((repo / "artifacts").rglob("*.json"))


def test_source_change_during_run_not_cached(repo, monkeypatch):
    def run(*args, **kwargs):
        (repo / "scripts/unit.py").write_text("VALUE = 2", encoding="utf-8")
        return successful(args[0])

    monkeypatch.setattr(VERIFY.subprocess, "run", run)
    assert (
        VERIFY.verify(repo, ["tests/test_unit.py"], ["scripts/unit.py"])["status"]
        == "PASS_SOURCE_CHANGED"
    )
    assert not list((repo / "artifacts").rglob("*.json"))


@pytest.mark.parametrize("value", ["tests", "--collect-only", "../outside.py", "scripts/unit.py"])
def test_invalid_selector_rejected(repo, value):
    with pytest.raises(ValueError):
        VERIFY.paths(repo, [value], tests=True)


def test_environment_and_command_change_invalidate(repo, monkeypatch):
    selected = ["tests/test_unit.py", "scripts/unit.py"]
    before = VERIFY.fingerprint(repo, selected, [["pytest"]])
    monkeypatch.setenv("PYTHONPATH", "changed-path")
    assert VERIFY.fingerprint(repo, selected, [["pytest"]]) != before
    before = VERIFY.fingerprint(repo, selected, [["pytest"]])
    assert VERIFY.fingerprint(repo, selected, [["ruff"]]) != before


def test_corrupt_cache_runs_again(repo, monkeypatch):
    monkeypatch.setattr(
        VERIFY.subprocess,
        "run",
        lambda *a, **k: successful(a[0]),
    )
    first = VERIFY.verify(repo, ["tests/test_unit.py"], ["scripts/unit.py"])
    (repo / "artifacts/agent_verification" / f"{first['fingerprint']}.json").write_text(
        "{", encoding="utf-8"
    )
    assert not VERIFY.verify(repo, ["tests/test_unit.py"], ["scripts/unit.py"], reuse=True)[
        "reused"
    ]


@pytest.mark.parametrize(
    "xml", [None, '<testsuite tests="0" skipped="0"/>', '<testsuite tests="3" skipped="3"/>']
)
def test_collect_only_or_all_skipped_is_not_pass(repo, monkeypatch, xml):
    def run(command, **kwargs):
        if xml is not None:
            target = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
            Path(target).write_text(xml, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="collected", stderr="")

    monkeypatch.setattr(VERIFY.subprocess, "run", run)
    assert VERIFY.verify(repo, ["tests/test_unit.py"], ["scripts/unit.py"])["status"] == "FAIL"


@pytest.mark.parametrize(
    "name",
    [
        "tests/helper.py",
        "tests/fixture.json",
        "pytest.ini",
        "setup.cfg",
        "tox.ini",
        ".ruff.toml",
        "conftest.py",
    ],
)
def test_support_and_configuration_changes_invalidate(repo, name):
    selected = ["tests/test_unit.py", "scripts/unit.py"]
    before = VERIFY.fingerprint(repo, selected, [])
    (repo / name).write_text("changed", encoding="utf-8")
    assert VERIFY.fingerprint(repo, selected, []) != before


def test_environment_collection_options_are_removed(monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    assert VERIFY.environment()["PYTEST_ADDOPTS"] == ""
