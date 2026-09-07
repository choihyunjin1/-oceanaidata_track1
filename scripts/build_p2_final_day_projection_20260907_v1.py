"""Build additive P2 projection packages; preserve original pinned L120 core."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from build_final_release_20260907_v1 import archive, copy_pins, read, save, saved_notebook, sha
from p2_final_day_materialize_20260907_v1 import BASE, NAME, endpoints
from p2_final_day_projection_20260907_v1 import project_profiles_vectorized

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p2_l120_s3_proj_20260907_v1.json"


def oof_check(data):
    path = ROOT / "artifacts/p2_c3_multiseed_completion_20260906_v1/evaluation.npz"
    assert sha(path) == "4ce3a80171e40cdb3ee760eec43e86b48441de05db9b2c587ef3e12ac2f695a5"
    prior = read(ROOT / "reports/final_day_evidence_check_20260907_v1/result.json")["P2"]
    ep = endpoints(data)
    with np.load(path, allow_pickle=False) as z:
        y,p,fold = z["truth"],z["natural_L120"],z["fold"]
        frame = pd.DataFrame({"time": z["time"], "layer": z["layer"]})
        projected = p.copy()
        for f in np.unique(fold):
            mask = fold == f
            projected[mask] = project_profiles_vectorized(frame.loc[mask], p[mask], ep).prediction
        metrics = {}
        for f in ["pooled", *np.unique(fold)]:
            mask = np.ones(len(y), bool) if f == "pooled" else fold == f
            before = float(np.sqrt(np.mean((y[mask]-p[mask])**2)))
            after = float(np.sqrt(np.mean((y[mask]-projected[mask])**2)))
            assert abs(after-prior["metrics"][f]["clip_then_pava_complete"]) < 1e-12
            metrics[str(f)] = {"baseline_rmse_C": before, "candidate_rmse_C": after, "delta_C": after-before}
    return {"status": "CANONICAL_SCALAR_AUDIT_MATCH", "oof_sha256": sha(path), "rows": len(y),
            "metrics": metrics, "new_fits": 0, "retrospective_not_fresh": True}


def build(parent, destination, data):
    destination.mkdir(parents=True, exist_ok=False)
    previous = read(parent / "BUILD_QA.json")
    assert previous["answer_sha256"] == BASE
    # Validate inherited exact cold provenance, not just a scored CSV.
    assert previous["original_full_cold_fits"] == 3
    provenance = {"parent_build_sha256": sha(parent / "BUILD_QA.json"),
                  "original_full_cold_fits": 3, "original_full_cold_seconds": previous["original_full_cold_seconds"],
                  "new_full_cold": False, "new_fits": 0}
    oof = oof_check(data)
    save(destination / "OOF_QA.json", oof)
    base = parent / previous["answer_file"]
    assert sha(base) == BASE
    answer = destination / "ANSWER" / NAME
    runner = ROOT / "scripts/p2_final_day_materialize_20260907_v1.py"
    for replay in [False, True]:
        command = [sys.executable, "-B", str(runner), "--base", str(base), "--output", str(answer),
                   "--data", str(data), "--receipt", str(destination / ("REPLAY_QA.json" if replay else "ANSWER_QA.json"))]
        if replay:
            command += ["--replay"]
        subprocess.run(command, check=True, timeout=1800)
    qa, replay = read(destination / "ANSWER_QA.json"), read(destination / "REPLAY_QA.json")
    assert qa["pid"] != replay["pid"] and qa["answer_sha256"] == replay["answer_sha256"]
    source_names = ["p2_final_day_projection_20260907_v1.py", "p2_final_day_materialize_20260907_v1.py"]
    commands = []
    for replay_mode in [False, True]:
        c = ["02_code/projection/p2_final_day_materialize_20260907_v1.py", "--base", "05_answer/submission_p2_L120_3seed.csv",
             "--output", "05_answer/"+NAME, "--receipt", "06_docs/"+("projection-replay.json" if replay_mode else "projection-answer.json")]
        if replay_mode:
            c += ["--replay"]
        commands.append(c)
    archives = []
    for role in ["SOURCE_ONLY", "SAVED_MODELS"]:
        target = destination / role
        for name in ["01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"]:
            (target / name).mkdir(parents=True)
        inherited = next(a for a in previous["archives"] if role in a["file"])
        assert sha(parent / inherited["file"]) == inherited["sha256"]
        copy_pins(parent / role, target, inherited["member_sha256"])
        (target / "02_code/projection").mkdir()
        for name in source_names:
            shutil.copy2(ROOT / "scripts" / name, target / "02_code/projection" / name)
        shutil.copy2(CONFIG, target / "06_docs/projection-config.json")
        save(target / "06_docs/projection-lineage.json", provenance)
        original = read(target / "PACKAGE_MANIFEST.json")["files"]
        assert "SAVED_PREDICT.ipynb" not in original
        for relative, expected in original.items():
            assert sha(target / relative) == expected
        if role == "SAVED_MODELS":
            fingerprint = {p.relative_to(target).as_posix(): sha(p) for p in
                           [target / "config.json", *sorted((target / "02_code").glob("*.py"))]}
            assert fingerprint == read(target / "03_model/MODEL_MANIFEST.json")["code_hashes"]
        body = "import os, subprocess, sys\nassert os.environ.get('P2_DATA_DIR')\n"
        for c in commands:
            body += f"subprocess.run([sys.executable, '-B', *{c!r}], check=True, timeout=1800)\n"
        notebook = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell("# P2_L120_s3_proj\nRun original TRAIN.ipynb then PREDICT.ipynb first. This notebook adds zero-fit projection and independent-process replay. Upload only the *_proj.csv answer if selected; the base CSV is intermediate."),
            nbformat.v4.new_code_cell(body)])
        nbformat.validate(notebook)
        nbformat.write(notebook, target / "PROJECT.ipynb")
        if role == "SAVED_MODELS":
            saved_notebook("P2", target, [["-I", "02_code/boot.py", stage] for stage in ["RUN_INFERENCE", "REPLAY", "FINAL_QA"]]+commands)
        (target / "CURRENT_README.md").write_text(
            "# P2_L120_s3_proj — current entry guide\n\n"
            "Original README and pinned L120 code remain unchanged for provenance. This current guide supersedes their candidate selection.\n\n"
            "Set P2_DATA_DIR to the organizer-distributed P2 directory. Use the original pinned environment; L120 core requires CUDA. No external observations, pretrained weights or leaderboard-derived coefficients.\n\n"
            "SOURCE_ONLY: empty 03_model, run TRAIN.ipynb → PREDICT.ipynb → PROJECT.ipynb. Training creates three L120 models from distributed data. SAVED_MODELS: fresh extraction, run SAVED_PREDICT.ipynb only (zero fit). Never rerun a consumed attempt.\n\n"
            "01_data: external distributed input location; 02_code: source; 03_model: learned weights; 05_answer: generated answers. Final candidate is 05_answer/"+NAME+"; submission_p2_L120_3seed.csv is intermediate. Do not upload both as one candidate.\n\n"
            "Projection: complete layers 2/3/4; finite T1 and first finite T5→T6→T7→T8; clip then exact endpoint-direction PAVA. Missing endpoints/incomplete profiles unchanged. No new fit.\n\n"
            "Evidence: inherited exact 3-fit L120 cold + separate-PID projection replay. This is not a newly executed complete cold of the combined package. Official score is UNKNOWN; exposed OOF is retrospective. See external BUILD_QA/OOF_QA and saved-extraction receipt. No automatic upload/final designation.\n",
            encoding="utf-8")
        archives.append(archive(target, destination / f"P2_L120_s3_proj_{role}.zip"))
    result = {"problem": "P2", "status": "BUILT_PENDING_EXTRACTED_SAVED_NOTEBOOK", "answer_file": "ANSWER/"+NAME,
              "answer_sha256": qa["answer_sha256"], "archives": archives, "provenance": provenance,
              "projection_separate_pid_replay": "PASS", "new_fits": 0, "uploads": 0, "official_score": None,
              "config_sha256": sha(CONFIG), "fresh_offline_OS_tested": False}
    save(destination / "BUILD_QA.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "archives"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    a = parser.parse_args()
    build(a.parent.resolve(), a.destination.resolve(), a.data.resolve())
