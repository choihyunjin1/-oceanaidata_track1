import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/build_candidate_upload_parts_20260906_v1.py"
SPEC = importlib.util.spec_from_file_location("parts", SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def build(tmp_path):
    source = tmp_path / "candidate.zip"
    with zipfile.ZipFile(source, "x") as archive:
        archive.writestr("README.md", "synthetic " * 100)
    parts = tmp_path / "parts"
    m.split(source, parts, chunk_size=100)
    return source, parts, parts / "REASSEMBLY_MANIFEST.json"


def test_byte_exact_and_no_overwrite(tmp_path):
    source, parts, manifest = build(tmp_path)
    output = tmp_path / "restored.zip"
    assert m.reassemble(manifest, output)["sha256"] == m.sha(source)
    assert source.read_bytes() == output.read_bytes()
    with pytest.raises(FileExistsError):
        m.reassemble(manifest, output)
    with pytest.raises(FileExistsError):
        m.split(source, parts)


@pytest.mark.parametrize("mode", ["bytes", "order", "missing", "whole_hash"])
def test_tamper_rejected_before_output(tmp_path, mode):
    _, parts, manifest = build(tmp_path)
    meta = json.loads(manifest.read_text())
    if mode == "bytes":
        with (parts / "part001.zip").open("ab") as stream:
            stream.write(b"x")
    elif mode == "order":
        meta["parts"][0]["name"] = "../outside.zip"
    elif mode == "missing":
        meta["parts"].pop()
    else:
        meta["original_sha256"] = "0" * 64
    manifest.write_text(json.dumps(meta))
    output = tmp_path / "restored.zip"
    with pytest.raises(ValueError):
        m.reassemble(manifest, output)
    assert not output.exists()
