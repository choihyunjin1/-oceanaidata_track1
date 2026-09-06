"""Distinct notebook TRAIN and PREDICT cells, each using fresh child processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
LOGS = PACKAGE / "04_logs"
MODES = {"TRAIN": ("RUN_TRAINING", "TRAIN_REPLAY"), "PREDICT": ("RUN_INFERENCE", "REPLAY", "FINAL_QA")}


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify():
    manifest = json.loads((PACKAGE / "PACKAGE_MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        if hashlib.sha256((PACKAGE / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("portable source/provenance changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(MODES))
    args = parser.parse_args()
    verify()
    cfg = json.loads((PACKAGE / "config.json").read_text())
    cold_lock = LOGS / "COLD_ATTEMPT_LOCK.json"
    stage_lock = LOGS / (args.stage + "_STAGE_LOCK.json")
    if stage_lock.exists():
        raise RuntimeError("stage already consumed; no overwrite/restart")
    if args.stage == "TRAIN":
        if any((PACKAGE / "03_model").iterdir()) or any((PACKAGE / "05_answer").iterdir()):
            raise ValueError("cold TRAIN requires empty models and answers")
        save(cold_lock, {"pid": os.getpid(), "started_unix": time.time(), "maximum_new_fits": 3,
                        "wallcap_seconds": cfg["wallcap_seconds"], "empty_03_and_05": True})
    else:
        previous = json.loads((LOGS / "TRAIN_STAGE_PASS.json").read_text())
        if previous["status"] != "PASS":
            raise ValueError("completed TRAIN and training replay required")
    lock = json.loads(cold_lock.read_text())
    save(stage_lock, {"pid": os.getpid(), "stage": args.stage})
    records = []
    try:
        for mode in MODES[args.stage]:
            remaining = lock["wallcap_seconds"] - (time.time() - lock["started_unix"])
            if remaining <= 0:
                raise TimeoutError("cold process wallcap consumed, no automatic retry")
            command = [sys.executable, "-I", "-B", str(Path(__file__).with_name("boot.py")), mode]
            started = time.time()
            child = subprocess.Popen(command, cwd=PACKAGE)
            try:
                code = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False, capture_output=True)
                child.wait(timeout=30)
                raise TimeoutError("cold exact child tree exceeded wallcap") from None
            records.append({"mode": mode, "pid": child.pid, "returncode": code, "seconds": time.time() - started})
            if code:
                raise RuntimeError("terminal stage failure: " + mode)
        verify()
        save(LOGS / (args.stage + "_STAGE_PASS.json"), {"status": "PASS", "pid": os.getpid(),
             "stage": args.stage, "child_processes": records, "new_fits": 3 if args.stage == "TRAIN" else 0})
        if args.stage == "PREDICT":
            qa = json.loads((LOGS / "independent-qa.json").read_text())
            expected = json.loads((PACKAGE / "06_docs/expected-answer.json").read_text())
            save(LOGS / "cold-terminal.json", {"status": "COLD_TRAIN_INFER_REPLAY_COMPLETE", "new_fits": 3,
                 "runtime_seconds": time.time() - lock["started_unix"], "answer_sha256": qa["answer_sha256"],
                 "expected_answer_sha256": expected["sha256"], "exact_existing_candidate_sha_match": qa["answer_sha256"] == expected["sha256"],
                 "qa_sha256": hashlib.sha256((LOGS / "independent-qa.json").read_bytes()).hexdigest(), "upload": 0,
                 "original_repository_denial_requested": bool(os.environ.get("P2_DENY_REPO")),
                 "fresh_venv_tested": False, "network_guard": "Python audit hook, not OS isolation"})
        print(json.dumps({"stage": args.stage, "status": "PASS"}))
    except BaseException as error:
        save(LOGS / (args.stage + "_STAGE_FAILURE.json"), {"status": "TERMINAL_TECHNICAL_FAILURE",
             "stage": args.stage, "error_type": type(error).__name__, "message": str(error), "children": records,
             "automatic_restart": False})
        raise


if __name__ == "__main__":
    main()
