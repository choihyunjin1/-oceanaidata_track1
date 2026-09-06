import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("release_builder", SCRIPTS / "build_final_release_20260907_v1.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_copy_pins_rejects_drift_and_traversal(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "run.py").write_text("x = 1\n")
    mod.copy_pins(a, b, {"run.py": mod.sha(a / "run.py")})
    assert (a / "run.py").read_bytes() == (b / "run.py").read_bytes()
    with pytest.raises(ValueError, match="mismatch"):
        mod.copy_pins(a, b, {"run.py": "0" * 64})
    with pytest.raises(ValueError, match="unsafe"):
        mod.owned(a, "../escape")


@pytest.mark.parametrize("name", ["observations.csv", "ATTEMPT_LOCK.json", "model.log", "cache.npz"])
def test_archive_rejects_forbidden(tmp_path, name):
    source = tmp_path / "source"
    source.mkdir()
    (source / name).write_bytes(b"x")
    with pytest.raises(ValueError):
        mod.archive(source, tmp_path / "out.zip")


def test_archive_preserves_empty_output_dirs_and_members(tmp_path):
    source = tmp_path / "source"
    (source / "03_model").mkdir(parents=True)
    (source / "run.py").write_text("pass\n")
    result = mod.archive(source, tmp_path / "out.zip")
    assert result["CRC_and_members"] == "PASS"
    with zipfile.ZipFile(tmp_path / "out.zip") as z:
        assert "03_model/" in z.namelist()
    with pytest.raises(FileExistsError):
        mod.archive(source, tmp_path / "out.zip")


def test_saved_notebook_only_infers(tmp_path):
    mod.saved_notebook("P2", tmp_path, [["-I", "02_code/boot.py", "RUN_INFERENCE"]])
    nb = mod.nbformat.read(tmp_path / "SAVED_PREDICT.ipynb", as_version=4)
    mod.nbformat.validate(nb)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "RUN_INFERENCE" in code and "RUN_TRAINING" not in code
    compile(code, "saved-notebook", "exec")


def test_exported_git_sources_restore_empty_directories(tmp_path):
    source = SCRIPTS.parent / "final_packages"
    pins = json.loads((source / "SOURCE_EXPORT_MANIFEST.json").read_text())["files"]
    for relative, expected in pins.items():
        assert mod.sha(source / relative) == expected
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    helper = tmp_path / "PREPARE_DIRECTORIES.py"
    shutil.copy2(source / helper.name, helper)
    subprocess.run([sys.executable, str(helper)], check=True, capture_output=True)
    for problem in ("P1", "P2", "P3"):
        assert (tmp_path / problem / "03_model").is_dir()
        assert not list((tmp_path / problem / "03_model").iterdir())
        assert not list((tmp_path / problem / "05_answer").iterdir())


def test_saved_archive_paths_and_publication_scope():
    from audit_final_release_git_20260907_v1 import permitted
    from execute_saved_release_20260907_v1 import validate_names

    validate_names(["03_model/", "03_model/model.pt", "02_code/run.py"])
    for names in (["../escape"], ["/absolute"], ["C:/absolute"], ["a.py", "A.py"]):
        with pytest.raises(ValueError):
            validate_names(names)
    assert permitted("final_packages/P1/02_code/run.py")
    for name in ("artifacts/result.json", "final_packages/P1/03_model/config.json",
                 "reports/example_20260907/answers.csv", "reports/example_20260907/ATTEMPT_LOCK.json"):
        assert not permitted(name)


def test_release_answer_checks_exact_hash_keys_and_values(tmp_path):
    from finalize_release_manifest_20260907_v1 import answer_check

    path = tmp_path / "answer.csv"
    path.write_text("key,label\na,1\nb,0\n")
    assert answer_check(path, 2, ["key", "label"], mod.sha(path))["rows"] == 2
    path.write_text("key,label\na,1\na,0\n")
    with pytest.raises(ValueError, match="duplicate"):
        answer_check(path, 2, ["key", "label"], mod.sha(path))
    path.write_text("key,value\na,nan\n")
    with pytest.raises(ValueError, match="nonfinite"):
        answer_check(path, 1, ["key", "value"], mod.sha(path))
