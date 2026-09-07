"""Actually execute both extracted P2 notebook paths and compare final bytes."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

import nbformat
import package_p2_smooth7_20260907_v2 as b
from jupyter_client import KernelManager
from nbclient import NotebookClient

os.environ["P2_DATA_DIR"] = "C:/Users/cedis/Downloads/p2/데이터셋_P2/P2_profile_restore"
os.environ["P2_DENY_REPO"] = str(b.REPO)
REPORT_V2 = b.REPO / "reports/p2_smooth7_portability_repair_20260907_v2"
REPORT_V2.mkdir(exist_ok=True)
started = time.monotonic()
receipts = []
for role, notebooks in [("EXTRACTED_SAVED", ["SAVED_PREDICT"]), ("EXTRACTED_COLD", ["TRAIN", "PREDICT", "SMOOTH_PROJECT"])]:
    root = b.OUT / role
    if role == "EXTRACTED_COLD":
        assert not list((root / "03_model").iterdir()) and not list((root / "05_answer").iterdir())
    for name in notebooks:
        path = root / (name + ".ipynb")
        before = b.sha(path)
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(nb, km=manager, timeout=None, resources={"metadata": {"path": str(root)}})
        t0 = time.monotonic()
        print(json.dumps({"role": role, "notebook": name, "status": "START"}), flush=True)
        client.execute(cleanup_kc=True)
        assert b.sha(path) == before
        receipt = {"role": role, "notebook": name, "status": "PASS", "seconds": time.monotonic()-t0, "sha256": before}
        receipts.append(receipt)
        b.save(root / "06_docs" / (name + "-executed.json"), receipt)
        print(json.dumps(receipt), flush=True)
    infer = b.read(root / "06_docs/smooth-infer.json")
    replay = b.read(root / "06_docs/smooth-replay.json")
    assert infer["sha256"] == replay["sha256"] and infer["pid"] != replay["pid"]
saved = b.read(b.OUT / "EXTRACTED_SAVED/06_docs/smooth-infer.json")
cold = b.read(b.OUT / "EXTRACTED_COLD/06_docs/smooth-infer.json")
assert saved["sha256"] == cold["sha256"]
answer = b.OUT / "ANSWER" / b.NAME
answer.parent.mkdir()
shutil.copy2(b.OUT / "EXTRACTED_COLD/05_answer" / b.NAME, answer)
result = {"status": "COLD_AND_SAVED_EXTRACTED_NOTEBOOK_EXACT_PASS", "answer": {"path": str(answer), "rows": saved["rows"], "bytes": answer.stat().st_size, "sha256": b.sha(answer)},
          "notebooks": receipts, "cold": b.read(b.OUT / "EXTRACTED_COLD/04_logs/cold-terminal.json"),
          "saved": saved, "postprocess_cold": cold, "seconds": time.monotonic()-started,
          "archives": b.read(b.OUT / "archives.json"), "new_candidate_training_fits": 3,
          "official_score": None, "uploads": 0, "final_designation": False}
b.save(b.OUT / "result.json", result)
b.save(REPORT_V2 / "candidate-ready.json", result)
print(json.dumps({"status": result["status"], "answer": result["answer"]}), flush=True)
