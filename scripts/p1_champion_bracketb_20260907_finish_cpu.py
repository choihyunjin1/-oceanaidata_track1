"""Wait for the sole P1 trainer, then independent CPU QA/build/infer-tree; never GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "p1_champion_bracketb_20260907_v1"
OUT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(pid, data, destination):
    start = time.monotonic()
    completed = []
    with (OUT / "CPU_FINISH_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump({"trainer_pid": pid, "gpu": False, "automatic_retraining": False}, handle)
    try:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-Command",
                f"Wait-Process -Id {int(pid)} -ErrorAction SilentlyContinue",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        terminal = read(OUT / "terminal_result.json")
        if terminal["status"] != "TRAIN_AND_INTERNAL_COMPLETE_QA_PENDING":
            raise ValueError("Trainer did not complete normally; no retry")
        commands = [
            (
                "independent_qa",
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts/p1_champion_bracketb_20260907_qa.py"),
                    "--data",
                    str(data),
                ],
            ),
            (
                "build",
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts/p1_champion_bracketb_20260907_pack.py"),
                    str(destination),
                ],
            ),
            (
                "infer_tree",
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(destination / "02_code/run.py"),
                    "infer-tree",
                    "--data",
                    str(data),
                ],
            ),
        ]
        for stage, command in commands:
            with (REPORT / f"{stage}.log").open("x", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            completed.append(stage)
        result = {
            "status": "CPU_QA_PACKAGE_TREE_READY_WAIT_GPU_OWNER_RELEASE",
            "completed": completed,
            "package": str(destination),
            "seconds": time.monotonic() - start,
            "next": "Wait for root/P2 explicit GPU release, then infer-ms and combine. No automatic GPU work.",
            "uploads": 0,
            "new_fits_by_supervisor": 0,
        }
    except BaseException as error:
        result = {
            "status": "CPU_FINISH_TECHNICAL_FAILURE",
            "error": str(error),
            "completed": completed,
            "seconds": time.monotonic() - start,
            "automatic_retry": False,
        }
        raise
    finally:
        with (REPORT / "cpu-finish.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    run(args.pid, args.data.resolve(), args.destination.resolve())
