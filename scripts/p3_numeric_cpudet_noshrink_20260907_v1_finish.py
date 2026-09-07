"""Actual extracted notebook replay and compact artifact handoff; no fits/uploads."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

import nbformat
import p3_numeric_cpudet_noshrink_20260907_v1_package as b
from jupyter_client import KernelManager
from nbclient import NotebookClient

start = time.monotonic()
p = b.OUT / "EXTRACTED_REPLAY"
os.environ["P3_DENY_REPO"] = str(b.REPO)
os.environ["P3_DATA_DIR"] = "C:/Users/cedis/Downloads/p3/데이터셋_P3/P3_wave_forecast"
receipts = []
for role in ["RUN_INFERENCE", "REPLAY_INFERENCE"]:
    source = p / (role + ".ipynb")
    before = b.sha(source)
    nb = nbformat.read(source, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, km=manager, timeout=None, resources={"metadata": {"path": str(p)}})
    t = time.monotonic()
    client.execute(cleanup_kc=True)
    assert before == b.sha(source)
    receipt = {"role": role, "status": "PASS", "seconds": time.monotonic()-t, "source_sha256": before}
    receipts.append(receipt)
    b.save(p / "06_docs" / (role + "-executed.json"), receipt)
    print(json.dumps(receipt), flush=True)
local = json.loads((b.OUT / "LOCAL/06_docs/noshrink-answer.json").read_text())
extracted = json.loads((p / "06_docs/noshrink-replay.json").read_text())
assert local["candidate_sha256"] == extracted["candidate_sha256"]
assert local["pid"] != extracted["pid"]
answer = b.OUT / "ANSWER/submission_p3_numeric_cpudet_noshrink.csv"
answer.parent.mkdir()
shutil.copy2(p / "05_answer/submission_p3_numeric_cpudet_noshrink.csv", answer)
assert b.sha(answer) == extracted["candidate_sha256"]
result = {"status": "CANDIDATE_READY_BASE_FRESH_COLD_PENDING", "new_fits": 0,
          "answer": {"path": str(answer), "rows": 1200, "bytes": answer.stat().st_size,
                     "sha256": b.sha(answer)}, "internal_evaluation": json.loads(
                         (b.REPO / "reports" / b.ID / "internal-evaluation.json").read_text()),
          "independent_qa": json.loads((b.REPO / "reports" / b.ID / "independent-qa.json").read_text()),
          "archives": json.loads((b.OUT / "archives.json").read_text())["archives"],
          "local_inference": local, "extracted_replay": extracted,
          "notebook_receipts": receipts, "extracted_qa_seconds": time.monotonic()-start,
          "base_fresh_cold_2": "PENDING; not modified or interrupted",
          "base_completion_provenance": "29 verified prefix +7 new backbones +5 OOF routers, 3-seed CPU",
          "fresh_cold_claim": False, "source_training_included_but_not_reexecuted": True,
          "v1_packaging_failure": "Missing manifest-pinned root docs before inference; retained v1 without answer. v2 copies every base root pin unchanged; no model/parameter change.",
          "uploads": 0, "final_designation": False}
b.save(b.OUT / "result.json", result)
b.save(b.REPO / "reports" / b.ID / "result.json", result)
form = f"""# P3 no-shrink candidate — NOT final designated

Leaderboard answer only: `{answer}`
Rows: 1200. Bytes: {answer.stat().st_size}. SHA256: `{b.sha(answer)}`.
Title: `P3 numeric CPU s3 no-shrink 0907`
Summary: `배포 데이터 학습 CPU CatBoost 3-seed·학습 OOF router 유지, 12/18/24h 지속성 shrink만 제거. 내부 평균 RMSE 개선, 저파고 악화 별도 보고.`

Model attachments (only when the official model form requires them): SOURCE_ONLY.zip + SAVED_MODELS.zip, overlay at the same root. Start with START_HERE_NOSHRINK.md and RUN_TRAINING.ipynb / RUN_INFERENCE.ipynb. File count, size limits and deadline remain unverified. The historical README is retained byte-for-byte for source pin compatibility; the addon entrypoint is explicitly named.

Actual extracted saved-model inference and independent notebook replay: PASS, {result['extracted_qa_seconds']:.3f}s total this Windows PC, Python 3.12.10, Ryzen 7800X3D, CPU inference 2 threads. This is no new training and no OS-disconnected reenactment. Source full training recipe is included, but addon whole-cold execution is NOT newly performed. Base fresh_cold_2 remains pending; do not claim a two-cold result. No upload or final designation by this worker.
"""
b.write(b.OUT / "FORM.md", form)
print(json.dumps({"status": result["status"], "answer": result["answer"]}), flush=True)
