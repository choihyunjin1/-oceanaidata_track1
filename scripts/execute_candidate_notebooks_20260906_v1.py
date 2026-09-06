"""Execute sealed thin notebooks in fresh kernels without rewriting their sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def execute(package, roles, timeout):
    package = Path(package).resolve()
    if any(role not in {"TRAIN", "PREDICT"} for role in roles):
        raise ValueError("explicit TRAIN/PREDICT roles required")
    output = package / "06_docs" / "executed_notebooks"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    receipts = []
    for role in roles:
        source = package / (role + ".ipynb")
        payload = source.read_bytes()
        notebook = nbformat.reads(payload.decode("utf-8"), as_version=4)
        manager = KernelManager(kernel_name="python3")
        # Override this manager's in-memory spec only, never the user's kernelspec.
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=timeout,
                                resources={"metadata": {"path": str(package)}})
        print(json.dumps({"stage": role, "status": "STARTING", "python": sys.executable}), flush=True)
        status = "FAIL"
        try:
            client.execute(cleanup_kc=True)
            status = "PASS"
        finally:
            with (output / (role + ".ipynb")).open("x", encoding="utf-8") as stream:
                nbformat.write(notebook, stream)
            if source.read_bytes() != payload:
                raise ValueError("sealed source notebook changed")
            receipts.append({"role": role, "status": status,
                             "source_sha256": hashlib.sha256(payload).hexdigest(),
                             "elapsed_seconds": time.monotonic() - started})
            with (output / (role + "-receipt.json")).open("x", encoding="utf-8") as stream:
                json.dump(receipts[-1], stream, indent=2)
        print(json.dumps(receipts[-1]), flush=True)
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--roles", nargs="+", choices=("TRAIN", "PREDICT"), default=["TRAIN", "PREDICT"])
    parser.add_argument("--timeout", type=int, default=21600)
    args = parser.parse_args()
    execute(args.package, args.roles, args.timeout)
