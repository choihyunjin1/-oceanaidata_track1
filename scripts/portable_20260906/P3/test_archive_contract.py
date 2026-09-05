"""Synthetic ZIP boundary and adapter no-fit contracts."""

import ast
import json
import zipfile
from pathlib import Path

import pytest
from build_archives import FOLDERS, read_sealed, sha, write_zip


def test_all_declared_empty_directories_are_in_archive(tmp_path):
    output = tmp_path / "synthetic.zip"
    receipt = write_zip(output, {"02_code/main.py": b"# synthetic\n"})
    assert receipt["all_entry_hashes_verified"]
    with zipfile.ZipFile(output) as archive:
        assert all(folder + "/" in archive.namelist() for folder in FOLDERS)
        archive.extractall(tmp_path / "new")
    assert all((tmp_path / "new" / folder).is_dir() for folder in FOLDERS)


def test_manifest_byte_change_is_blocked(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    content = root / "safe.py"
    content.write_bytes(b"# safe")
    (root / "PACKAGE_MANIFEST.json").write_text(json.dumps({"sha256": {"safe.py": sha(b"# safe")}}))
    assert read_sealed(root.resolve())["safe.py"] == b"# safe"
    content.write_bytes(b"# changed")
    with pytest.raises(ValueError, match="changed"):
        read_sealed(root.resolve())


def test_manifest_parent_escape_is_blocked(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "PACKAGE_MANIFEST.json").write_text(json.dumps({"sha256": {"../outside.py": "bad"}}))
    with pytest.raises(ValueError, match="escape"):
        read_sealed(root.resolve())


def test_saved_adapter_calls_frozen_frame_and_has_no_fit():
    tree = ast.parse(Path(__file__).with_name("archive_infer.py").read_text(encoding="utf-8"))
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "fit" not in calls
    assert calls.count("official_frame") == 1
    assert calls.count("load_models") == 1
    assert calls.count("csv_bytes") == 1


def test_archive_usage_does_not_promise_consumed_replay():
    source = Path(__file__).with_name("build_archives.py").read_text(encoding="utf-8")
    assert "Historical saved-model --replay remains available" not in source
    assert "Only archive_infer.py is supported for this archive." in source
