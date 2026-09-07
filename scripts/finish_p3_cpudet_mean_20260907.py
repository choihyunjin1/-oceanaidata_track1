"""Execute the extracted notebooks and publish an exact-hash candidate handoff."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

import nbformat
import package_p3_cpudet_mean_20260907 as b
from jupyter_client import KernelManager
from nbclient import NotebookClient

start = time.monotonic()
root = b.OUT / "EXTRACTED_REPLAY"
os.environ["P3_DENY_REPO"] = str(b.REPO)
os.environ["P3_DATA_DIR"] = "C:/Users/cedis/Downloads/p3/데이터셋_P3/P3_wave_forecast"
receipts = []
for role in ["RUN_INFERENCE", "REPLAY_INFERENCE"]:
    path = root / (role + ".ipynb")
    before = b.sha(path)
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, km=manager, timeout=None, resources={"metadata": {"path": str(root)}})
    stage_start = time.monotonic()
    client.execute(cleanup_kc=True)
    assert b.sha(path) == before
    receipt = {"role": role, "status": "PASS", "seconds": time.monotonic() - stage_start,
               "notebook_sha256": before}
    receipts.append(receipt)
    b.save(root / "06_docs" / (role + "-executed.json"), receipt)
    print(json.dumps(receipt), flush=True)
local = json.loads((b.OUT / "LOCAL/06_docs/mean-answer.json").read_text())
replay = json.loads((root / "06_docs/mean-replay.json").read_text())
assert local["sha256"] == replay["sha256"] and local["pid"] != replay["pid"]
prior = ["2015b38750d357630d5b2e9eee32807d2961ce752e35eb2b454ee16e579dda56",
         "70761affca4d3fc6f1d24ae53467e5b185b23465926ebb4b851f0300872cddbd",
         "ff42a6a08c76f0d58ed2f3a9ea31a08819ada5fa6e9007af942fe0b891937960"]
assert replay["sha256"] not in prior
answer = b.OUT / "ANSWER/submission_p3_cpudet_mean.csv"
answer.parent.mkdir()
shutil.copy2(root / "05_answer/submission_p3_cpudet_mean.csv", answer)
assert b.sha(answer) == replay["sha256"]
result = {"status": "CANDIDATE_READY_BASE_FRESH_COLD_PENDING", "answer": {
    "path": str(answer), "rows": 1200, "bytes": answer.stat().st_size, "sha256": b.sha(answer)},
    "notebooks": receipts, "local_inference": local, "extracted_replay": replay,
    "archives": json.loads((b.OUT / "archives.json").read_text()),
    "not_duplicate_of_three_current_comparators": True,
    "internal_result_sha256": b.sha(b.REPORT / "result.json"),
    "independent_qa_sha256": b.sha(b.REPORT / "independent-qa.json"),
    "seconds": time.monotonic() - start, "new_fits": 0, "uploads": 0,
    "whole_cold": "base fresh_cold_2 pending", "final_designation": False}
b.save(b.OUT / "result.json", result)
b.save(b.REPORT / "candidate-ready.json", result)
b.write(b.OUT / "FORM.md", f"""# P3 CPU equal-component candidate — not final designated

Leaderboard answer: `{answer}`
1200 rows; {answer.stat().st_size} bytes; SHA256 `{b.sha(answer)}`.
Title: `P3 numeric CPU s3 equal-component 0907`
Summary: `배포 데이터 CPU 3-seed CatBoost single/multi 단순 평균, 기존 장기리드 0.2 지속성 혼합 유지. 학습된 결합기만 제거한 독립 비교.`

Model attachments: SOURCE_ONLY.zip + SAVED_MODELS.zip. Overlay into the same root. START_HERE_MEAN.md / RUN_TRAINING.ipynb / RUN_INFERENCE.ipynb / REPLAY_INFERENCE.ipynb are the entry points. Both ZIPs must match archives.json. File-per limit is 50MB (live UI 09-07); attachment count and final deadline time remain unknown.

Extracted saved-model notebook inference/replay PASS, {result['seconds']:.3f}s total on Windows/Python 3.12.10/Ryzen 7800X3D, inference CPU2. Fresh training notebook is included but not rerun for this addon; ongoing base CPU4 fresh_cold_2 must supply full-cold evidence. No general six-hour compliance or official acceptance is claimed.
""")
print(json.dumps({"status": result["status"], "answer": result["answer"]}), flush=True)
