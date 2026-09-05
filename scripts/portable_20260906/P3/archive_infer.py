"""Inference-only replay adapter; no training or selection, frozen P3 functions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import run as r  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-approved", required=True, action="store_true")
    parser.parse_args()
    started = time.perf_counter()
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    r.guard(source, official=True)
    companion = json.loads((r.ROOT / "INFERENCE_ADAPTER_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in companion["sha256"].items():
        path = (r.ROOT / name).resolve()
        if r.ROOT not in path.parents or r.sha(path) != expected:
            raise ValueError("inference-only companion hash/boundary mismatch")
    config = json.loads(r.CONFIG.read_text(encoding="utf-8"))
    r.verify(config, source, prepared=True)
    training_path = r.REPORT / "training-result.json"
    train = json.loads(training_path.read_text(encoding="utf-8"))
    qa = json.loads((r.REPORT / "training-independent-qa.json").read_text(encoding="utf-8"))
    if qa["status"] != "PASS" or qa["training_result_sha256"] != r.sha(training_path):
        raise ValueError("saved training QA link differs")
    if train["pid"] == os.getpid():
        raise ValueError("inference adapter must be a new process")
    columns = json.loads((r.WORK / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    frame, context_rows = r.official_frame(source, columns, r.load_models(train))
    payload = r.csv_bytes(frame)
    digest = hashlib.sha256(payload).hexdigest()
    prior = json.loads((r.REPORT / "answer-qa.json").read_text(encoding="utf-8"))
    if digest != prior["sha256"] or len(frame) != 1200:
        raise ValueError("same saved model did not reproduce the sealed answer SHA")
    elapsed = time.perf_counter() - started
    if elapsed >= 21600:
        raise TimeoutError("inference-only process exceeded six hours")
    r.ANSWER.mkdir(parents=True, exist_ok=True)
    with (r.ANSWER / "replayed_submission.csv").open("xb") as stream:
        stream.write(payload)
    receipt = {
        "status": "SAVED_MODEL_INFERENCE_REPLAY_PASS",
        "pid": os.getpid(),
        "original_training_pid": train["pid"],
        "rows": 1200,
        "cases": 200,
        "sha256": digest,
        "output": "05_answer/replayed_submission.csv",
        "same_frozen_official_frame_and_predict_functions": True,
        "new_fits": 0,
        "new_selection": 0,
        "official_context_rows": context_rows,
        "official_index_rows": 1200,
        "sample_rows": 0,
        "hidden_rows": 0,
        "old_answer_rows_read": 0,
        "uploads": 0,
        "inference_only_seconds": elapsed,
        "uninterrupted_cold_start_claim": False,
        "archived_training_result_sha256": r.sha(training_path),
        "companion_sha256": r.sha(r.ROOT / "INFERENCE_ADAPTER_MANIFEST.json"),
        "os_network_isolation_or_new_venv_verified": False,
    }
    r.save(r.REPORT / "archive-inference-replay.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
