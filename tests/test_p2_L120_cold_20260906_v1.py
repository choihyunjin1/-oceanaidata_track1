"""Source/CLI/path boundaries only; no fits, source observations or official reads."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts/portable_20260906/P2_L120_cold_v1"


def load(name):
    spec = importlib.util.spec_from_file_location("p2cold_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_arguments():
    stages = load("stages")
    assert stages.MODES["TRAIN"] == ("RUN_TRAINING", "TRAIN_REPLAY")
    assert stages.MODES["PREDICT"] == ("RUN_INFERENCE", "REPLAY", "FINAL_QA")
    assert "child.wait(timeout=remaining)" in (HERE / "stages.py").read_text()


def test_repo_boundary(tmp_path):
    boot = load("boot")
    repo, runtime = tmp_path / "repo", tmp_path / "repo/.venv"
    assert boot.denied_path(repo / "src/model.py", repo, runtime)
    assert boot.denied_path(repo / "artifacts/model.pt", repo, runtime)
    assert not boot.denied_path(runtime / "Lib/site-packages/torch/a.py", repo, runtime)
    assert not boot.denied_path(tmp_path / "outside/02_code/core.py", repo, runtime)


def test_actual_boot_repo_denial(tmp_path):
    import os
    import shutil

    repo, package = tmp_path / "research", tmp_path / "outside"
    repo.mkdir()
    code = package / "02_code"
    code.mkdir(parents=True)
    forbidden = repo / "existing.pt"
    forbidden.write_bytes(b"synthetic-not-a-model")
    shutil.copy2(HERE / "boot.py", code / "boot.py")
    (code / "run.py").write_text("from pathlib import Path\nimport os\nPath(os.environ['DENIED_FILE']).read_bytes()\n")
    env = {**os.environ, "P2_DENY_REPO": str(repo), "DENIED_FILE": str(forbidden)}
    result = subprocess.run([sys.executable, "-I", "-B", str(code / "boot.py")], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "original repository access denied" in result.stderr


def test_source_only_build(tmp_path):
    build = load("build_package")
    destination = build.package_source(tmp_path / "cold")
    manifest = json.loads((destination / "PACKAGE_MANIFEST.json").read_text())
    assert not list((destination / "03_model").iterdir())
    assert not list((destination / "05_answer").iterdir())
    assert not list(destination.rglob("*.csv"))
    assert not list(destination.rglob("*.pt"))
    assert manifest["files"]["02_code/core.py"] == "b18a8279a542f38f5ed550307772824ed8f597e088ad2ba7070ebfeff9020c84"
    assert manifest["fits"] == 0
    for role in ("TRAIN", "PREDICT"):
        notebook = json.loads((destination / (role + ".ipynb")).read_text(encoding="utf-8"))
        cells = "".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        assert "['02_code/stages.py', '" + role + "']" in cells
        assert role + ".ipynb" in manifest["files"]
    assert "C:\\Users\\" not in (destination / "02_code/run.py").read_text()


def test_archive_no_dataset(tmp_path):
    import pytest

    build = load("build_package")
    directory = tmp_path / "bad"
    directory.mkdir()
    (directory / "observations.csv").write_text("synthetic")
    with pytest.raises(ValueError, match="raw data"):
        build.archive(directory, tmp_path / "rejected.zip")


def test_archive_keeps_empty_stage_directories(tmp_path):
    import zipfile

    build = load("build_package")
    source = tmp_path / "source"
    for name in ("03_model", "04_logs", "05_answer"):
        (source / name).mkdir(parents=True)
    archive = tmp_path / "source.zip"
    build.archive(source, archive)
    target = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(target)
    assert all((target / name).is_dir() for name in ("03_model", "04_logs", "05_answer"))
    assert not list((target / "03_model").iterdir())
