"""New uncapped attempt: verified 29-fit completion, then independent fresh cold.

Never resumes a consumed directory. No model/seed/feature changes; no upload.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from build_final_release_20260907_v1 import archive, copy_pins, read, save, sha
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/p3_numeric_cpudet_unbounded_20260907_v2"
CONFIG = ROOT / "configs/experiments/p3_numeric_cpudet_unbounded_20260907_v2.json"


def build(parent, destination):
    terminal = read(parent / "terminal_result.json")
    assert terminal["status"] == "TERMINAL_TECHNICAL_FAILURE"
    previous = read(parent / "BUILD_QA.json")
    old = parent / "cold_1"
    fits = read(old / "fit-progress.json")
    assert len(fits) == 29
    previous_pins = {
        **previous["source_files"],
        "02_code/cpudet-manifest.json": sha(old / "02_code/cpudet-manifest.json"),
    }
    cfg = read(CONFIG)
    assert cfg["maximum_seconds_per_cold"] is None
    old_cfg = read(old / "02_code/cpudet-config.json")
    assert sha(old / "02_code/cpudet-config.json") == previous["config_sha256"]
    ignored = {"id", "maximum_seconds_per_cold", "execution_amendment"}
    assert {k: v for k, v in cfg.items() if k not in ignored} == {
        k: v for k, v in old_cfg.items() if k not in ignored
    }
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / "SOURCE_ONLY"
    for name in ["01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"]:
        (source / name).mkdir(parents=True)
    copy_pins(old, source, previous_pins)
    shutil.copy2(
        ROOT / "scripts/p3_cpudet_unbounded_20260907_v2.py", source / "02_code/unbounded.py"
    )
    shutil.copy2(CONFIG, source / "02_code/unbounded-config.json")
    for role, stages in [
        ("TRAIN", ["smoke", "prepare", "train", "qa"]),
        ("PREDICT", ["infer", "replay"]),
    ]:
        cells = [
            nbformat.v4.new_markdown_cell(
                f"# P3 uncapped CPU 3-seed — {role}\n\nOnly distributed data. Fixed 36 backbones +5 routers. CPU4/GPU0. No wall-clock kill; full iterations retained. A reuse-manifest marks explicitly authorized 29-fit prefix completion, NOT a fresh cold. Without it this is empty-model cold training. Elapsed time is reported and final submission time eligibility remains a separate check. No upload."
            )
        ]
        for stage in stages:
            cells += [
                nbformat.v4.new_markdown_cell("## " + stage),
                nbformat.v4.new_code_cell(
                    "import os, subprocess, sys\nassert os.environ.get('P3_DATA_DIR')\n"
                    f"subprocess.run([sys.executable, '-I', '-B', '02_code/unbounded.py', '{stage}'], check=True)"
                ),
            ]
        nb = nbformat.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
            },
        )
        nbformat.validate(nb)
        nbformat.write(nb, source / (role + "_UNBOUNDED.ipynb"))
    (source / "CURRENT_README.md").write_text(
        "# P3 CPU 3-seed uncapped execution\n\nUse TRAIN_UNBOUNDED.ipynb → PREDICT_UNBOUNDED.ipynb with P3_DATA_DIR set to distributed P3 folder. Do not use old TRAIN/PREDICT: those are retained immutable provenance for the old 4h workflow. The new unbounded.py disables only workflow timers, retaining every data-access/source guard and the fixed seeds, features, model parameters and shrink.\n\n"
        "SOURCE_ONLY has no weights/cache/answer/reuse manifest. Fresh training fits36 backbones+5routers. An explicitly prepared recovery copy carries29 verified completed prefix models: its7 new backbones+5routers are not a new whole cold. The independent fresh cold uses no inherited artifacts. Compare answer SHA; record runtime and official six-hour eligibility separately. No timed termination and no15:00 cutoff. No automatic retries after other technical failures.\n",
        encoding="utf-8",
    )
    pins = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob("*") if p.is_file()}
    save(source / "02_code/unbounded-manifest.json", pins)
    pins["02_code/unbounded-manifest.json"] = sha(source / "02_code/unbounded-manifest.json")
    sourcezip = archive(source, destination / "P3_unbounded_SOURCE_ONLY.zip")
    resumed = destination / "completion_1"
    for name in ["01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"]:
        (resumed / name).mkdir(parents=True)
    copy_pins(source, resumed, pins)
    prepared = read(old / "06_docs/prepare.json")
    imported = {x["model"]: x["sha256"] for x in fits}
    imported.update({"04_logs/" + n: h for n, h in prepared["files"].items()})
    imported["06_docs/prepare.json"] = sha(old / "06_docs/prepare.json")
    copy_pins(old, resumed, imported)
    shutil.copy2(old / "02_code/cpudet-config.json", resumed / "06_docs/parent-config.json")
    save(
        resumed / "06_docs/reuse-manifest.json",
        {
            "not_a_fresh_cold": True,
            "parent_terminal_sha256": sha(parent / "terminal_result.json"),
            "parent_config_sha256": sha(old / "02_code/cpudet-config.json"),
            "prepare_sha256": imported["06_docs/prepare.json"],
            "imported_files": imported,
            "parent_fit_receipt_sha256": sha(old / "fit-progress.json"),
            "fits": fits,
            "authorization": "User requested continuing without execution time limit; same-recipe complete fits only, new directory",
        },
    )
    save(
        destination / "BUILD_QA.json",
        {
            "source_pins": pins,
            "archive": sourcezip,
            "status": "BUILT_NOT_EXECUTED",
            "reused_complete_fits": 29,
            "new_backbone_fits_budget": 43,
            "new_router_fits_budget": 10,
            "new_production_fits_budget": 53,
            "parent_preserved": True,
            "source_archive_has_models": False,
            "hard_runtime_limit_seconds": None,
            "uploads": 0,
        },
    )
    print(json.dumps({"status": "BUILT", "reused_fits": 29, "new_production_fits_budget": 53}))


def notebook_run(package):
    output = package / "06_docs/executed_unbounded"
    output.mkdir(exist_ok=False)
    for role in ["TRAIN", "PREDICT"]:
        path = package / (role + "_UNBOUNDED.ipynb")
        payload = path.read_bytes()
        nb = nbformat.reads(payload.decode(), as_version=4)
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        started = time.monotonic()
        status = "FAIL"
        try:
            NotebookClient(
                nb, km=km, timeout=None, resources={"metadata": {"path": str(package)}}
            ).execute(cleanup_kc=True)
            assert path.read_bytes() == payload
            status = "PASS"
        finally:
            nbformat.write(nb, output / (role + ".ipynb"))
            save(
                output / (role + "-receipt.json"),
                {
                    "status": status,
                    "seconds": time.monotonic() - started,
                    "notebook_sha256": sha(path),
                    "wall_timeout": None,
                },
            )


def execute(destination):
    built = read(destination / "BUILD_QA.json")
    save(
        destination / "EXECUTION_LOCK.json",
        {
            "pid": os.getpid(),
            "started_utc": datetime.now(UTC).isoformat(),
            "hard_timeout": None,
            "run_sequence": ["completion_1", "fresh_cold_2"],
        },
    )
    start = time.monotonic()
    results = []
    try:
        for role in ["completion_1", "fresh_cold_2"]:
            target = destination / role
            if role == "fresh_cold_2":
                for name in ["01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"]:
                    (target / name).mkdir(parents=True, exist_ok=False)
                copy_pins(destination / "SOURCE_ONLY", target, built["source_pins"])
            for n, h in built["source_pins"].items():
                assert sha(target / n) == h
            run_start = time.monotonic()
            env = os.environ.copy()
            env["P3_DENY_REPO"] = str(ROOT)
            with (
                (destination / (role + ".stdout.log")).open("x") as out,
                (destination / (role + ".stderr.log")).open("x") as err,
            ):
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).resolve()),
                        "notebook",
                        "--destination",
                        str(target),
                    ],
                    stdout=out,
                    stderr=err,
                    env=env,
                    check=True,
                )
            train = read(target / "06_docs/cpudet-training.json")
            answer = read(target / "06_docs/cpudet-answer.json")
            replay = read(target / "06_docs/cpudet-replay.json")
            assert train["backbone_fits"] == 36 and train["router_fits"] == 5
            assert answer["sha256"] == replay["sha256"] == sha(target / "05_answer/submission.csv")
            expected_reuse = 29 if role == "completion_1" else 0
            assert train["reused_backbone_fits"] == expected_reuse
            result = {
                "run": role,
                "answer_sha256": answer["sha256"],
                "current_run_seconds": time.monotonic() - run_start,
                "run_kind": train["run_kind"],
                "reused_backbone_fits": expected_reuse,
                "new_production_fits": 36 - expected_reuse + 5,
                "answer_replay": replay["status"],
            }
            results.append(result)
            save(destination / (role + "-receipt.json"), result)
            print(json.dumps(result), flush=True)
        equal = results[0]["answer_sha256"] == results[1]["answer_sha256"]
        fresh_seconds = results[1]["current_run_seconds"]
        result = {
            "status": "COMPLETION_AND_FRESH_COLD_EXACT_PENDING_PACKAGE"
            if equal
            else "ANSWER_MISMATCH_NOT_READY",
            "runs": results,
            "answer_sha_equal": equal,
            "new_production_fits": 53,
            "reused_fits": 29,
            "completed_fresh_cold_runs": 1,
            "not_two_fresh_colds": True,
            "hard_timeout": None,
            "fresh_cold_duration_le_21600s": fresh_seconds <= 21600,
            "six_hour_rule_scope": "UNVERIFIED_FOR_GENERAL_P3; not an eligibility conclusion or kill switch",
            "total_current_seconds": time.monotonic() - start,
            "uploads": 0,
        }
        save(destination / "terminal_result.json", result)
        REPORT.mkdir(parents=True, exist_ok=True)
        save(REPORT / "result.json", result)
    except BaseException as exc:
        result = {
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "type": type(exc).__name__,
            "error": str(exc),
            "runs": results,
            "seconds": time.monotonic() - start,
            "automatic_restart": False,
            "uploads": 0,
        }
        save(destination / "terminal_result.json", result)
        REPORT.mkdir(parents=True, exist_ok=True)
        save(REPORT / "failure.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "execute", "notebook"])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    a = parser.parse_args()
    if a.action == "build":
        build(a.parent.resolve(), a.destination.resolve())
    elif a.action == "execute":
        execute(a.destination.resolve())
    else:
        notebook_run(a.destination.resolve())
