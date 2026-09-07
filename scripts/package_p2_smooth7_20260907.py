"""Build immutable additive notebooks around unchanged pinned L120 core."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import nbformat
from build_final_release_20260907_v1 import archive, copy_pins, read, save, saved_notebook, sha

REPO = Path(__file__).resolve().parents[1]
PARENT = Path("C:/Users/cedis/Documents/OceanFinalDay_20260907/P2_L120_s3_proj_v2")
OUT = Path("C:/Users/cedis/Documents/OceanFinalDay_20260907/P2_L120_s3_smooth7_proj_v1")
REPORT = REPO / "reports/p2_l120_s3_smooth7_projection_20260907_v1"
NAME = "submission_p2_L120_s3_smooth7_proj.csv"


def build():
    assert read(REPORT / "independent-qa.json")["status"] == "PASS"
    assert read(REPORT / "independent-qa.json")["result_sha256"] == sha(REPORT / "result.json")
    OUT.mkdir(exist_ok=False)
    previous = read(PARENT / "BUILD_QA.json")
    archives = []
    commands = [["02_code/projection/p2_smooth7_portable_20260907.py", mode] for mode in ["infer", "replay"]]
    for role in ["SOURCE_ONLY", "SAVED_MODELS"]:
        target = OUT / role
        for name in ["01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"]:
            (target / name).mkdir(parents=True)
        inherited = next(a for a in previous["archives"] if role in a["file"])
        assert sha(PARENT / inherited["file"]) == inherited["sha256"]
        copy_pins(PARENT / role, target, inherited["member_sha256"])
        shutil.copy2(REPO / "scripts/p2_smooth7_portable_20260907.py", target / "02_code/projection/p2_smooth7_portable_20260907.py")
        for name in ["result.json", "independent-qa.json", "replay-qa.json"]:
            shutil.copy2(REPORT / name, target / "06_docs" / ("smooth-" + name))
        shutil.copy2(REPO / "configs/experiments/p2_l120_s3_smooth7_projection_20260907_v1.json", target / "06_docs/smooth-config.json")
        body = "import os, subprocess, sys\nassert os.environ.get('P2_DATA_DIR')\n"
        for c in commands:
            body += f"subprocess.run([sys.executable, '-B', *{c!r}], check=True)\n"
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
            "# Frozen L120 s3 smooth7 + projection\nRun TRAIN then PREDICT first. This stage uses the generated unprojected intermediate; historical PROJECT is not required."), nbformat.v4.new_code_cell(body)])
        nbformat.validate(notebook)
        nbformat.write(notebook, target / "SMOOTH_PROJECT.ipynb")
        if role == "SAVED_MODELS":
            saved_notebook("P2", target, [["-I", "02_code/boot.py", stage] for stage in ["RUN_INFERENCE", "REPLAY", "FINAL_QA"]] + commands)
        (target / "CURRENT_README.md").write_text(
            "# P2 L120 s3 smooth7 projection — current entry\n\n"
            "Only organizer-distributed data. No external observations, pretrained weights, hidden truth or Public-fitted coefficients. Data is referenced by P2_DATA_DIR, not included.\n\n"
            "SOURCE_ONLY: fresh folder, empty 03_model/05_answer, run TRAIN.ipynb -> PREDICT.ipynb -> SMOOTH_PROJECT.ipynb. Three L120 models are fitted; TRAIN also includes the pinned training replay. SAVED_MODELS is a complete alternative package: fresh extraction, run SAVED_PREDICT.ipynb (zero fit). Do not overlay model directories into the cold test.\n\n"
            "01_data references data; 02_code contains source; 03_model contains learned models; 04_logs contains execution records; 05_answer contains regenerated CSV. Final file: 05_answer/" + NAME + ". Other CSVs are intermediates, not alternate answers to upload.\n\n"
            "New step: station/layer ten-minute grid centered seven-slot mean, missing predictions as NaN, min_periods=1, no interpolation; then complete-three-layer clip and endpoint-direction exact PAVA, T1 and first finite T5/T6/T7/T8. Missing endpoints/incomplete profiles preserve the smoothed values. Fixed window7 was selected by Fable on already exposed OOF, not a fresh confirmatory test. Fold-local OOF avoids cross-fold smoothing. B3 RMSE 0.463529468 -> 0.441853923 C; pooled 1.236732066 -> 1.204715055 C. Worst-fold risk is recorded, not hidden. No Public/Private improvement is promised.\n\n"
            "Original pinned README/code/manifests remain as provenance; this guide and new notebook supersede final-output instructions. CUDA required for original core, use pinned dependencies/Python 3.12.10. Runtime is measured on our PC, not a general official six-hour compliance claim. Actual extracted cold and saved notebook receipts accompany the outer package. No final designation is implied.\n",
            encoding="utf-8")
        for relative, expected in read(target / "PACKAGE_MANIFEST.json")["files"].items():
            assert sha(target / relative) == expected
        save(target / "SMOOTH_SOURCE_MANIFEST.json", {p.relative_to(target).as_posix(): sha(p) for p in target.rglob("*") if p.is_file() and "03_model" not in p.parts})
        archives.append(archive(target, OUT / ("P2_L120_s3_smooth7_proj_" + role + ".zip")))
    save(OUT / "archives.json", archives)
    for role, dest in [("SOURCE_ONLY", "EXTRACTED_COLD"), ("SAVED_MODELS", "EXTRACTED_SAVED")]:
        target = OUT / dest
        target.mkdir()
        a = next(a for a in archives if role in a["file"])
        with zipfile.ZipFile(OUT / a["file"]) as z:
            assert all((target / n).resolve().is_relative_to(target.resolve()) for n in z.namelist())
            z.extractall(target)
        for name in ["01_data", "03_model", "04_logs", "05_answer", "06_docs"]:
            (target / name).mkdir(exist_ok=True)
    print(json.dumps({"status": "BUILT", "root": str(OUT)}), flush=True)


if __name__ == "__main__":
    build()
