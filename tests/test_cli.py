from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from p1_qc import cli
from p1_qc.config import P1QCConfig
from scripts.build_final_package import (
    PackagePolicyError,
    build_package,
    collect_allowlisted_files,
)


def _complete_p1_directory(path: Path) -> Path:
    path.mkdir(parents=True)
    for name in cli.P1_REQUIRED_FILES:
        (path / name).write_text("placeholder\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("position", ["before", "after"])
def test_common_cli_arguments_work_on_both_sides_of_subcommand(
    tmp_path: Path, position: str
) -> None:
    config = tmp_path / "custom.toml"
    data_dir = tmp_path / "input"
    common = ["--config", str(config), "--data-dir", str(data_dir), "--mode", "causal"]
    argv = [*common, "audit"] if position == "before" else ["audit", *common]
    args = cli.build_parser().parse_args(argv)
    assert args.config == config
    assert args.data_dir == data_dir
    assert args.mode == "causal"


def test_default_cli_configuration_is_offline() -> None:
    args = cli.build_parser().parse_args(["audit"])
    assert cli._config(args).mode == "offline"
    assert cli._config(args).features.mode == "offline"


def test_cli_data_fallback_requires_one_complete_file_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _complete_p1_directory(tmp_path / "arbitrary_name")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    assert cli.resolve_data_dir(P1QCConfig()) == first.resolve()

    _complete_p1_directory(tmp_path / "another_complete_set")
    with pytest.raises(FileNotFoundError, match="found 2"):
        cli.resolve_data_dir(P1QCConfig())


def _minimal_package_project(root: Path) -> None:
    files = {
        "00_MUST_READ_FIRST.md": "# Required preflight\n",
        "AGENTS.md": "# Agent rules\n",
        "README.md": "# Portable package\nSet P1_DATA_DIR before running.\n",
        "pyproject.toml": '[project]\nname = "portable"\n',
        "requirements.txt": "numpy==2.3.5\n",
        "configs/p1.toml": '[project]\nmode = "offline"\n',
        "src/p1_qc/__init__.py": "\n",
        "src/p1_qc/__main__.py": "\n",
        "reports/ENVIRONMENT_2026-08-13.md": "# Environment\n",
        "scripts/analyze_oof_failures.py": "# Failure analysis\n",
        "scripts/benchmark_data_io.py": "# Data-loading benchmark\n",
        "reports/P1_FAILURE_RECON_2026-08-13.md": "# Failure reconnaissance\n",
        "reports/P1_DATA_LOADING_BENCHMARK_2026-08-13.md": "# Data-loading benchmark\n",
        "reports/P1_BREAKTHROUGH_RESEARCH_2026-08-13.md": "# Breakthrough research\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "train.csv").write_text("must,not,ship\n", encoding="utf-8")
    (root / "artifacts" / "cache").mkdir(parents=True)
    (root / "artifacts" / "cache" / "features.parquet").write_bytes(b"not parquet")
    (root / "artifacts" / "runs" / "candidate").mkdir(parents=True)
    (root / "artifacts" / "runs" / "candidate" / "oof.parquet").write_bytes(b"not oof")


def test_final_package_is_allowlisted_and_manifest_uses_logical_hashes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _minimal_package_project(project)
    output = tmp_path / "out" / "package.zip"
    receipt = build_package(project, output)
    assert receipt["uploaded"] is False
    assert len(str(receipt["sha256"])) == 64

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        required_new_members = {
            "scripts/analyze_oof_failures.py",
            "scripts/benchmark_data_io.py",
            "reports/P1_FAILURE_RECON_2026-08-13.md",
            "reports/P1_DATA_LOADING_BENCHMARK_2026-08-13.md",
            "reports/P1_BREAKTHROUGH_RESEARCH_2026-08-13.md",
        }
        assert required_new_members <= set(names)
        assert "train.csv" not in names
        assert not any("artifacts" in name for name in names)
        assert "artifacts/runs/candidate/oof.parquet" not in names
        assert not any(
            Path(name).suffix.casefold() in {".csv", ".parquet", ".zip"} for name in names
        )
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
    assert manifest["excludes_source_data"] is True
    assert all(not Path(item["logical_path"]).is_absolute() for item in manifest["files"])
    assert all(len(item["logical_sha256"]) == 64 for item in manifest["files"])


@pytest.mark.parametrize("path_style", ["windows", "posix"])
def test_final_package_rejects_local_absolute_path_references(
    tmp_path: Path, path_style: str
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    if path_style == "windows":
        local_path = "C:" + "\\" + "Users" + "\\" + "person" + "\\" + "private.csv"
    else:
        local_path = "/" + "home" + "/person/private.csv"
    (root / "README.md").write_text(f"do not embed {local_path}\n", encoding="utf-8")
    with pytest.raises(PackagePolicyError, match="absolute path"):
        collect_allowlisted_files(root, patterns=("README.md",), required=())
