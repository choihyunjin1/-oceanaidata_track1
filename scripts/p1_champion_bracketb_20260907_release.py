"""After explicit GPU-owner release: candidate and real ZIP notebook replay, no upload."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import nbformat
from build_candidate_upload_parts_20260906_v1 import split, validate_parts
from build_final_release_20260907_v1 import archive, copy_pins, read, save, sha
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
ID = "p1_champion_bracketb_20260907_v1"
REPORT = ROOT / "reports" / ID


def run(destination, data):
    import numpy as np
    import pandas as pd

    start = time.monotonic()
    package = destination / "PACKAGE"
    completed = []
    save(
        destination / "RELEASE_ATTEMPT_LOCK.json",
        {"pid": os.getpid(), "gpu_owner_release_confirmed": True},
    )
    try:
        for stage in ("infer-ms", "combine"):
            with (REPORT / f"{stage}.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(package / "02_code/run.py"),
                        stage,
                        "--data",
                        str(data),
                    ],
                    cwd=package,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            completed.append(stage)
        answer = package / "05_answer/P1_submission.csv"
        value = pd.read_csv(answer)
        keys = ["station", "year", "layer", "time"]
        reference = pd.read_csv(data / "test.csv", usecols=keys)
        checks = {
            "rows": len(value) == 169011,
            "columns": list(value) == keys + ["label"],
            "keys_order": value[keys].astype(str).equals(reference[keys].astype(str)),
            "unique": not value.duplicated(keys).any(),
            "binary_finite": bool(np.isin(value.label, [0, 1]).all()),
        }
        if not all(checks.values()):
            raise ValueError("Independent official answer schema/key/finite QA failed")
        (destination / "ANSWER").mkdir()
        shutil.copy2(answer, destination / "ANSWER/P1_champion_bracketB.csv")
        sourcepins = read(package / "source-manifest.json")["files"]
        sourcepins["source-manifest.json"] = sha(package / "source-manifest.json")
        for kind in ("SOURCE_ONLY", "SAVED_MODELS"):
            folder = destination / kind
            for name in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
                (folder / name).mkdir(parents=True)
            copy_pins(package, folder, sourcepins)
            for name in ("B_config.json", "B_internal_qa.json", "model-lineage.json"):
                shutil.copy2(package / "06_docs" / name, folder / "06_docs" / name)
            if kind == "SAVED_MODELS":
                modelpins = read(package / "06_docs/model-lineage.json")["files"]
                copy_pins(package, folder, modelpins)
        zipped = {
            kind: archive(destination / kind, destination / f"P1_bracketB_{kind}.zip")
            for kind in ("SOURCE_ONLY", "SAVED_MODELS")
        }
        save(destination / "ARCHIVE_QA.json", zipped)
        replay = destination / "SAVED_ZIP_REPLAY"
        replay.mkdir()
        with zipfile.ZipFile(destination / "P1_bracketB_SAVED_MODELS.zip") as z:
            for name in z.namelist():
                if not (replay / name).resolve().is_relative_to(replay.resolve()):
                    raise ValueError("Unsafe ZIP entry")
            z.extractall(replay)
        os.environ["P1_DATA_DIR"] = str(data)
        nbpath = replay / "SAVED_PREDICT.ipynb"
        original = nbpath.read_bytes()
        notebook = nbformat.reads(original.decode("utf-8"), as_version=4)
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        replay_start = time.monotonic()
        try:
            NotebookClient(
                notebook, km=km, timeout=None, resources={"metadata": {"path": str(replay)}}
            ).execute(cleanup_kc=True)
        finally:
            nbformat.write(notebook, destination / "executed_SAVED_PREDICT.ipynb")
        checks["notebook_source_unchanged"] = nbpath.read_bytes() == original
        checks["new_ZIP_notebook_answer_exact"] = sha(
            replay / "05_answer/P1_submission.csv"
        ) == sha(answer)
        if not all(checks.values()):
            raise ValueError("Extracted notebook answer mismatch")
        parts = split(
            destination / "P1_bracketB_SAVED_MODELS.zip", destination / "SAVED_MODEL_PARTS"
        )
        validate_parts(destination / "SAVED_MODEL_PARTS/REASSEMBLY_MANIFEST.json")
        result = {
            "status": "LOCAL_CANDIDATE_PACKAGE_REPLAY_PASS_NOT_UPLOADED",
            "checks": checks,
            "answer": str(destination / "ANSWER/P1_champion_bracketB.csv"),
            "rows": len(value),
            "positive_rows": int(value.label.sum()),
            "sha256": sha(answer),
            "bytes": answer.stat().st_size,
            "source_only": zipped["SOURCE_ONLY"],
            "saved_models": zipped["SAVED_MODELS"],
            "parts": len(parts["parts"]),
            "replay_seconds": time.monotonic() - replay_start,
            "runtime_seconds": time.monotonic() - start,
            "new_composed_whole_cold_executed": False,
            "reproduction_basis": "Existing exact O/MS cold evidence + new full B3 training and independent model replay + actual new ZIP saved notebook exact answer",
            "official_score": None,
            "uploads": 0,
            "hidden_rows": 0,
            "completed": completed,
        }
        (destination / "FORM.md").write_text(
            "# P1 candidate — not designated\n\nTitle: P1 Original O + Bracket-B3 + MS-TCN\n\n"
            "Summary: Distributed-data original composition; B alone adds 27 bracket features. "
            "Three original seeds, fixed thresholds, cell policy, GI and MS. No external or hidden inputs.\n\n"
            f"Answer: ANSWER/P1_champion_bracketB.csv; {len(value)} rows; SHA256 `{sha(answer)}`.\n\n"
            "Reproduction attachments: P1_bracketB_SOURCE_ONLY.zip and P1_bracketB_SAVED_MODELS.zip "
            "(or all SAVED_MODEL_PARTS files with reassembly manifest/script). Portal attachment limits unverified.\n\n"
            "Existing O/MS cold plus newly trained B3 and actual ZIP notebook replay are separate evidence; "
            "no new whole-composition cold claim. General official six-hour scope and submission deadline unverified. "
            "Historical six-cell selection program remains unrecovered. No prior score is assigned to this new SHA.\n",
            encoding="utf-8",
        )
    except BaseException as error:
        result = {
            "status": "RELEASE_TECHNICAL_FAILURE",
            "error": str(error),
            "completed": completed,
            "runtime_seconds": time.monotonic() - start,
            "automatic_retry": False,
        }
        raise
    finally:
        save(destination / "RELEASE_RESULT.json", result)
        save(REPORT / "release-result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "answer", "rows", "sha256")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gpu-owner-released", action="store_true", required=True)
    args = parser.parse_args()
    run(args.destination.resolve(), args.data.resolve())
