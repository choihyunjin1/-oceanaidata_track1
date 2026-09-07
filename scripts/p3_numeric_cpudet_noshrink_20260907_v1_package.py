"""Build disjoint source/saved addon packages; never change base or active cold files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ID = "p3_numeric_cpudet_noshrink_20260907_v1"
BASE = Path("C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_numeric_cpudet_unbounded_v2/completion_1")
OUT = Path("C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_numeric_cpudet_noshrink_v2")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def save(path, value):
    write(path, json.dumps(value, indent=2, allow_nan=False))


def notebook(path, stages):
    code = ("from pathlib import Path\nimport os, sys, subprocess\n"
            "root = Path.cwd()\nassert (root / '02_code').is_dir()\n"
            "assert os.environ.get('P3_DATA_DIR'), 'Set P3_DATA_DIR to the distributed data directory'\n")
    code += "\n".join(f"subprocess.run([sys.executable, '02_code/{runner}', '{stage}'], check=True, cwd=root)"
                       for runner, stage in stages)
    save(path, {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}},
        "cells": [{"cell_type": "code", "id": "entry", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": code.splitlines(keepends=True)}]})


README = """# P3 numeric CPU three-seed — no-shrink candidate

This is an optional, NOT final-designated, no-fit postprocessing variant. Use only organizer-distributed P3 data. No external data/pretrained weights/hidden truth/old answer inputs or Public-derived coefficients.

## Exact procedure

The base is CPU CatBoost numeric-lead single/multi, seeds 20260817/18/19, 36 backbone + 5 train-OOF routers. All model and router parameters are unchanged. The sole addition bypasses the fixed 0.2 persistence mixture on 12/18/24 h; 3/6/9 h remain byte-equivalent. All fitted values come from trained models/OOF router artifacts. Training thread_count remains 4; addon inference uses 2. No training runtime cutoff is added.

## Data / training / inference / output

1. Extract SOURCE_ONLY to a new folder; retain 01_data, 02_code, 03_model, 04_logs, 05_answer, 06_docs. Install requirements pinned in 02_code and use Python 3.12.10/CatBoost as pinned there.
2. Set P3_DATA_DIR to the organizer P3_wave_forecast folder containing train_wave.csv, train_atmos.csv, test_context.parquet, test_index.csv. Data is not redistributed in this archive.
3. To train from nothing, keep 03_model/04_logs/05_answer empty and run RUN_TRAINING.ipynb. This runs original source preparation, 41 CPU fits, original independent historical QA, then records the newly trained 7 deployment-model hashes. The selected recipe is not retuned. This notebook is included but NOT re-executed for this zero-fit addon.
4. Alternatively overlay SAVED_MODELS onto SOURCE_ONLY (same relative root); it contains only 7 deployment models, their hash/QA lineage, feature column names and approved addon internal-QA aggregates, not historical predictions or answers.
5. Run RUN_INFERENCE.ipynb once, then REPLAY_INFERENCE.ipynb from another Python PID. These use test keys/context, never test truth. Answer: 05_answer/submission_p3_numeric_cpudet_noshrink.csv, columns case_id,station,lead_h,hs_pred, 1200 rows, finite 0..30m, official key order.
6. Only the answer CSV is a leaderboard answer. SOURCE_ONLY.zip and SAVED_MODELS.zip are model-code assets; do not upload them as answer CSV. No automated upload/deletion/final designation occurs.

## Evidence and limits

Current CPU historical OOF 103602 rows: shrink 0.681321563 m -> no-shrink 0.677450331 m; retrospective paired station×episode bootstrap 951 blocks, 4000 resamples: P(improve)=0.914, CI90 [-0.008524918,+0.000733607] m. Current hs<1.7 worsens by +0.009921904 m. These are internal, not an expected Public score; worst-block risk is not an automatic veto. Training-prefix completion and same-model independent replay are distinct from a whole fresh cold run. Base fresh_cold_2 remains pending at package build; TWO fresh cold equivalence is NOT claimed. No-shrink addon new fits=0. Official six-hour limit applicability to this general model is unverified; all durations are this PC's measurements, not compliance proof.

Original unbounded.py source, its manifests, learned model bytes and contract stay unchanged. noshrink.py is the explicit additive inference policy. Old base notebooks/readmes are NOT the entrypoints of this addon; use RUN_TRAINING/RUN_INFERENCE above.
"""


def build():
    assert not OUT.exists(), "disjoint new output required"
    q = json.loads((REPO / "reports" / ID / "independent-qa.json").read_text())
    assert q["status"] == "PASS"
    src, saved, local = [OUT / n for n in ["SOURCE_ONLY", "SAVED_MODELS", "LOCAL"]]
    for name in ["01_data", "03_model", "04_logs", "05_answer", "06_docs"]:
        (src / name).mkdir(parents=True)
    shutil.copytree(BASE / "02_code", src / "02_code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(REPO / "scripts" / (ID + "_portable.py"), src / "02_code/noshrink.py")
    for name in ["shrink-provenance.json"]:
        shutil.copy2(BASE / "06_docs" / name, src / "06_docs" / name)
    for name in ["internal-evaluation.json", "independent-qa.json"]:
        shutil.copy2(REPO / "reports" / ID / name, src / "06_docs" / name)
    shutil.copy2(REPO / "configs/experiments" / (ID + ".json"), src / "02_code/noshrink-preregistration.json")
    # The original source manifest pins root notebooks and README. Preserve them verbatim;
    # place the explicit addon entrypoint alongside them, not inside the frozen documents.
    for name, expected in json.loads((BASE / "02_code/unbounded-manifest.json").read_text()).items():
        if "/" not in name:
            assert sha(BASE / name) == expected
            shutil.copy2(BASE / name, src / name)
    write(src / "START_HERE_NOSHRINK.md", README)
    write(OUT / "README.md", README)
    notebook(src / "RUN_TRAINING.ipynb", [("unbounded.py", x) for x in ["prepare", "train", "qa"]]
             + [("noshrink.py", "adopt-trained")])
    notebook(src / "RUN_INFERENCE.ipynb", [("noshrink.py", "infer")])
    notebook(src / "REPLAY_INFERENCE.ipynb", [("noshrink.py", "replay")])
    pinned = {p.relative_to(src).as_posix(): sha(p) for p in src.rglob("*") if p.is_file()}
    save(src / "02_code/noshrink-source-manifest.json", pinned)
    train = json.loads((BASE / "06_docs/cpudet-training.json").read_text())
    files = {n: h for n, h in train["models"].items()
             if n.startswith("03_model/full/") or n == "03_model/full_router.joblib"}
    assert len(files) == 7
    for name, h in files.items():
        assert sha(BASE / name) == h
        (saved / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, saved / name)
    (saved / "04_logs").mkdir(parents=True)
    shutil.copy2(BASE / "04_logs/columns.json", saved / "04_logs/columns.json")
    save(saved / "06_docs/noshrink-models.json", {"files": files,
        "columns_sha256": sha(saved / "04_logs/columns.json"),
        "training_receipt_sha256": sha(BASE / "06_docs/cpudet-training.json"),
        "training_qa_sha256": sha(BASE / "06_docs/cpudet-training-qa.json"),
        "expected_base_answer_sha256": "2015b38750d357630d5b2e9eee32807d2961ce752e35eb2b454ee16e579dda56",
        "cold_status": "VERIFIED_PREFIX_COMPLETION; independent BASE_FRESH_COLD_2_PENDING"})
    save(saved / "saved-model-manifest.json", {p.relative_to(saved).as_posix(): sha(p)
                                               for p in saved.rglob("*") if p.is_file()})
    shutil.copytree(src, local)
    shutil.copytree(saved, local, dirs_exist_ok=True)
    save(OUT / "build.json", {"status": "ADDON_BUILT", "new_fits": 0,
        "base": str(BASE), "base_training_sha256": sha(BASE / "06_docs/cpudet-training.json"),
        "source_files": len(pinned), "saved_deployment_models": 7,
        "base_active_run_modified": False, "source_train_not_reexecuted": True})


def package():
    replay = json.loads((OUT / "LOCAL/06_docs/noshrink-replay.json").read_text())
    assert replay["status"] == "EXACT_REPLAY_PASS"
    archives = {}
    for name in ["SOURCE_ONLY", "SAVED_MODELS"]:
        root, target = OUT / name, OUT / (name + ".zip")
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                if p.is_dir():
                    z.write(p, p.relative_to(root).as_posix() + "/")
                else:
                    assert p.suffix not in [".csv", ".parquet", ".npy", ".npz", ".log"]
                    z.write(p, p.relative_to(root).as_posix())
        archives[name] = {"path": str(target), "bytes": target.stat().st_size, "sha256": sha(target)}
    extracted = OUT / "EXTRACTED_REPLAY"
    extracted.mkdir()
    for name in ["SOURCE_ONLY", "SAVED_MODELS"]:
        with zipfile.ZipFile(OUT / (name + ".zip")) as z:
            for n in z.namelist():
                assert (extracted / n).resolve().is_relative_to(extracted.resolve())
            z.extractall(extracted)
    save(OUT / "archives.json", {"archives": archives, "extracted": str(extracted),
                                 "extracted_replay": "PENDING", "new_fits": 0})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "package"])
    build() if p.parse_args().stage == "build" else package()
