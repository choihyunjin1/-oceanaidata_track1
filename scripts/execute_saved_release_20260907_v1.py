"""Extract a pinned saved archive and execute its inference notebook, without training."""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import nbformat
from build_final_release_20260907_v1 import read, save, sha
from jupyter_client import KernelManager
from nbclient import NotebookClient


def validate_names(names):
    seen = set()
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or name.casefold() in seen:
            raise ValueError("unsafe/duplicate archive member")
        seen.add(name.casefold())


def execute(release, destination):
    receipt = read(release / "BUILD_QA.json")
    archive = next(a for a in receipt["archives"] if "SAVED_MODELS" in a["file"])
    source = release / archive["file"]
    if sha(source) != archive["sha256"]:
        raise ValueError("archive hash changed")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(source) as z:
        validate_names(z.namelist())
        if z.testzip() is not None:
            raise ValueError("ZIP CRC mismatch")
        z.extractall(destination)
    for name, expected in archive["member_sha256"].items():
        if sha(destination / name) != expected:
            raise ValueError("extracted member hash mismatch")
    notebook_path = destination / "SAVED_PREDICT.ipynb"
    payload = notebook_path.read_bytes()
    notebook = nbformat.reads(payload.decode("utf-8"), as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    started = time.monotonic()
    status = "FAIL"
    try:
        NotebookClient(notebook, km=manager, timeout=1800,
                       resources={"metadata": {"path": str(destination)}}).execute(cleanup_kc=True)
        answer_name = Path(receipt["answer_file"]).name
        actual = sha(destination / "05_answer" / answer_name)
        if actual != receipt["answer_sha256"]:
            raise ValueError("saved archive answer differs from scored candidate")
        if notebook_path.read_bytes() != payload:
            raise ValueError("sealed notebook changed")
        status = "EXTRACTED_SAVED_NOTEBOOK_EXACT_ANSWER_PASS"
    finally:
        nbformat.write(notebook, destination / "06_docs/SAVED_PREDICT_executed.ipynb")
        result = {"status": status, "problem": receipt["problem"], "new_fits": 0,
                  "runtime_seconds": time.monotonic() - started, "archive_sha256": archive["sha256"],
                  "source_notebook_sha256": archive["member_sha256"]["SAVED_PREDICT.ipynb"],
                  "expected_answer_sha256": receipt["answer_sha256"], "uploads": 0}
        save(destination.parent / (destination.name + "-receipt.json"), result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    execute(args.release.resolve(), args.destination.resolve())
