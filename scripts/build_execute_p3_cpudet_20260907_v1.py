"""Build isolated source package and execute two authorized cold replications."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p3_numeric_cpudet_s3_20260907_v1.json"
REPORT = ROOT / "reports/p3_numeric_cpudet_s3_20260907_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def build(destination):
    cfg = json.loads(CONFIG.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / "SOURCE_ONLY"
    shutil.copytree(
        ROOT / "final_packages/P3", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for folder in ["01_data", "03_model", "04_logs", "05_answer", "06_docs"]:
        (source / folder).mkdir(exist_ok=True)
    shutil.copyfile(
        ROOT / "scripts/p3_cpudet_final_day_20260907_v1.py", source / "02_code/cpudet.py"
    )
    shutil.copyfile(CONFIG, source / "02_code/cpudet-config.json")
    shutil.copyfile(
        ROOT / "scripts/execute_candidate_notebooks_20260906_v1.py",
        source / "02_code/execute_notebooks.py",
    )
    for role, stages in [("TRAIN", ["prepare", "train", "qa"]), ("PREDICT", ["infer", "replay"])]:
        nb = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    f"# P3 CPU deterministic 3-seed: {role}\n\nOnly distributed data. New empty directory; never rerun consumed training. No upload. TRAIN: 36 backbones + 5 routers; PREDICT: zero fit. CPU 4 threads, same environment."
                )
            ]
        )
        nb.cells.append(
            nbformat.v4.new_code_cell(
                "import os, subprocess, sys\nfrom pathlib import Path\nassert 'P3_DATA_DIR' in os.environ\nassert Path('02_code/cpudet.py').is_file()"
            )
        )
        for stage in stages:
            nb.cells.append(nbformat.v4.new_markdown_cell("## " + stage))
            nb.cells.append(
                nbformat.v4.new_code_cell(
                    f"subprocess.run([sys.executable, '-I', '-B', '02_code/cpudet.py', '{stage}'], check=True, timeout={cfg['maximum_seconds_per_cold']})"
                )
            )
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nbformat.validate(nb)
        nbformat.write(nb, source / (role + ".ipynb"))
    (source / "README.md").write_text(
        "# P3 numeric CPU deterministic 3-seed\n\nSet P3_DATA_DIR to organizer P3 folder. Use a new empty directory, install the pinned 02_code/requirements.txt plus Jupyter dependencies, and run TRAIN.ipynb then PREDICT.ipynb. Alternatively use python 02_code/execute_notebooks.py --package . --timeout 14400.\n\n41 fits per cold; CPU only, four CatBoost threads. 591 fixed features; 700/1200 iterations; seeds 20260817/18/19, clip each seed forecast before component averaging, fixed chronological TRAIN-OOF router and shrink .2. No external observations or pretrained weights. All fitted models/routers are generated from distributed data. No Public-derived coefficients.\n\nBefore upload require two independent cold answer SHA values to match and saved-model ZIP replay. This source build alone is NOT a completed candidate. Original run.py is an unchanged helper, not the active entry point: use cpudet.py only.\n",
        encoding="utf-8",
    )
    (source / "RELEASE_ROLE.md").write_text(
        "SOURCE_ONLY: CPU three-seed adapter, no old models/OOF/cache/answers. Computational parents are pinned, unchanged helper source; their original GPU entry point is not run. Status and eligibility require current cpudet receipts.\n",
        encoding="utf-8",
    )
    pins = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob("*") if p.is_file()}
    save(source / "02_code/cpudet-manifest.json", pins)
    archive = destination / "P3_numeric_cpudet_s3_SOURCE_ONLY.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in source.rglob("*"):
            z.write(p, p.relative_to(source).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    save(
        destination / "BUILD_QA.json",
        {
            "status": "SOURCE_BUILT_NOT_TRAINED",
            "source_archive_sha256": sha(archive),
            "config_sha256": sha(CONFIG),
            "source_files": pins,
            "production_fits": 0,
        },
    )
    print(json.dumps({"source": str(source), "archive_sha256": sha(archive)}))


def execute(destination):
    archive = destination / "P3_numeric_cpudet_s3_SOURCE_ONLY.zip"
    built = json.loads((destination / "BUILD_QA.json").read_text())
    assert sha(archive) == built["source_archive_sha256"]
    cfg = json.loads(CONFIG.read_text())
    assert sha(CONFIG) == built["config_sha256"]
    save(
        destination / "EXECUTION_LOCK.json",
        {
            "pid": os.getpid(),
            "started_utc": datetime.now(UTC).isoformat(),
            "replications": 2,
            "total_fit_budget": 82,
        },
    )
    env = os.environ.copy()
    env["P3_DENY_REPO"] = str(ROOT)
    results = []
    start = time.monotonic()
    try:
        for rep in [1, 2]:
            if rep == 2:
                deadline = datetime(2026, 9, 7, 15, tzinfo=timezone(timedelta(hours=9)))
                if (
                    datetime.now(UTC)
                    + timedelta(seconds=results[0]["whole_seconds"] * 1.2)
                    > deadline
                ):
                    raise TimeoutError(
                        "second complete cold no longer fits 15:00 KST cutoff; preserve first run, no candidate"
                    )
            target = destination / f"cold_{rep}"
            target.mkdir(exist_ok=False)
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    assert (target / name).resolve().is_relative_to(target)
                z.extractall(target)
            for name, expected in built["source_files"].items():
                assert sha(target / name) == expected, name
            run_start = time.monotonic()
            with (
                (destination / f"cold_{rep}.stdout.log").open("x") as out,
                (destination / f"cold_{rep}.stderr.log").open("x") as err,
            ):
                # A synthetic CPU compatibility/repeatability check precedes real fits.
                subprocess.run(
                    [sys.executable, "-I", "-B", str(target / "02_code/cpudet.py"), "smoke"],
                    cwd=target,
                    env=env,
                    stdout=out,
                    stderr=err,
                    check=True,
                    timeout=120,
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(target / "02_code/execute_notebooks.py"),
                        "--package",
                        str(target),
                        "--timeout",
                        str(cfg["maximum_seconds_per_cold"]),
                    ],
                    cwd=target,
                    env=env,
                    stdout=out,
                    stderr=err,
                    check=True,
                    timeout=cfg["maximum_seconds_per_cold"] + 180,
                )
            answer = json.loads((target / "06_docs/cpudet-answer.json").read_text())
            replay = json.loads((target / "06_docs/cpudet-replay.json").read_text())
            assert answer["sha256"] == replay["sha256"] == sha(target / "05_answer/submission.csv")
            results.append(
                {
                    "replication": rep,
                    "answer_sha256": answer["sha256"],
                    "whole_seconds": time.monotonic() - run_start,
                    "backbone_fits": 36,
                    "router_fits": 5,
                    "saved_process_replay": replay["status"],
                }
            )
            save(destination / f"cold_{rep}-receipt.json", results[-1])
            print(json.dumps(results[-1]), flush=True)
        equal = results[0]["answer_sha256"] == results[1]["answer_sha256"]
        result = {
            "status": "TWO_COLD_EXACT_PENDING_ARCHIVE_QA"
            if equal
            else "TWO_COLD_MISMATCH_NOT_READY",
            "cold_runs": results,
            "answer_sha_equal": equal,
            "runtime_seconds": time.monotonic() - start,
            "new_fits": 82,
            "uploads": 0,
            "hidden_rows": 0,
            "automatic_restart": False,
        }
        save(destination / "terminal_result.json", result)
        save(REPORT / "result.json", result)
    except BaseException as exc:
        result = {
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "type": type(exc).__name__,
            "error": str(exc),
            "completed_cold_runs": results,
            "elapsed_seconds": time.monotonic() - start,
            "automatic_restart": False,
            "uploads": 0,
        }
        save(destination / "terminal_result.json", result)
        save(REPORT / "failure.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "execute"])
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    (build if args.action == "build" else execute)(args.destination.resolve())
