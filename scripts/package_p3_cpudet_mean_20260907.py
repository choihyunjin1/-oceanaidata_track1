"""Build disjoint mean-router candidate with source training and saved-model paths."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from p3_numeric_cpudet_noshrink_20260907_v1_package import notebook, save, sha, write

REPO = Path(__file__).resolve().parents[1]
BASE = Path("C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_numeric_cpudet_unbounded_v2/completion_1")
OUT = Path("C:/Users/cedis/Documents/OceanFinalDay_20260907/P3_cpudet_mean_router_ablation_v1")
REPORT = REPO / "reports/p3_cpudet_mean_router_ablation_20260907_v1"
README = """# P3 CPU three-seed equal-component candidate

NOT final-designated. Only organizer-distributed P3 data; no external observations, pretrained weights, hidden truth, old answer inputs or Public-fitted coefficients.

## Entry points

Extract SOURCE_ONLY.zip in a fresh folder. Set P3_DATA_DIR to the distributed P3_wave_forecast folder; install the pinned dependencies in 02_code using Python 3.12.10. Data is referenced, not redistributed.
For training from empty 03_model / 04_logs / 05_answer, run RUN_TRAINING.ipynb then RUN_INFERENCE.ipynb. The unchanged original CPU recipe runs 36 backbone + 5 diagnostic router fits (41 total); this variant does not USE the learned routers in inference. Training CPU4, inference CPU2. No added runtime cutoff.
For saved-model prediction, overlay SAVED_MODELS.zip on SOURCE_ONLY and run RUN_INFERENCE.ipynb, then REPLAY_INFERENCE.ipynb from another process. Answer: 05_answer/submission_p3_cpudet_mean.csv; schema case_id,station,lead_h,hs_pred; 1200 rows in provided key order.

## Exact change

Each single/multi component is the mean of three seed predictions after clipping each current+residual to [0,30]. Use equal single/multi weighting instead of the learned loss router. Preserve original persistence shrink: 0.2 at 12/18/24h only. This is not the separate no-shrink candidate. Equal weighting is a structural arithmetic mean, not a Public-fit parameter. Features, CPU models, seeds and splits are unchanged. Use these new RUN_* notebooks, not historical base entrypoints copied to preserve source manifests.

## Evidence and caveats

Retrospective 103602-row OOF: original CPU RMSE 0.681321563m -> 0.678334232m. Paired station x episode bootstrap 951 groups, 4000 draws: CI90 [-0.005066201,-0.000961097]m, P(improve)=0.99475. This is NOT a promised Public/Private score. Detailed fold/station/lead/low-wave risks accompany result.json.
New candidate fits=0. Existing base completion reused a verified 29-fit prefix and finished 12 fits; whole independent fresh_cold_2 remains pending at build. Saved-model replay and whole cold proof are different. No six-hour compliance claim: general-model applicability is unconfirmed. Final submission status is not implied by packaging.
"""


def build():
    assert not OUT.exists(), "Preserve all existing artifacts"
    qa = json.loads((REPORT / "independent-qa.json").read_text())
    assert qa["status"] == "PASS"
    assert qa["result_sha256"] == sha(REPORT / "result.json")
    src, saved = OUT / "SOURCE_ONLY", OUT / "SAVED_MODELS"
    for name in ["01_data", "03_model", "04_logs", "05_answer", "06_docs"]:
        (src / name).mkdir(parents=True)
    shutil.copytree(BASE / "02_code", src / "02_code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(REPO / "scripts/p3_cpudet_mean_portable_20260907.py", src / "02_code/mean_policy.py")
    for name, expected in json.loads((BASE / "02_code/unbounded-manifest.json").read_text()).items():
        if "/" not in name:
            assert sha(BASE / name) == expected
            shutil.copy2(BASE / name, src / name)
    for name in ["result.json", "independent-qa.json"]:
        shutil.copy2(REPORT / name, src / "06_docs" / name)
    shutil.copy2(BASE / "06_docs/shrink-provenance.json", src / "06_docs/shrink-provenance.json")
    shutil.copy2(REPO / "configs/experiments/p3_cpudet_mean_router_ablation_20260907_v1.json", src / "02_code/mean-preregistration.json")
    write(src / "START_HERE_MEAN.md", README)
    write(OUT / "README.md", README)
    notebook(src / "RUN_TRAINING.ipynb", [("unbounded.py", x) for x in ["prepare", "train", "qa"]] + [("mean_policy.py", "adopt-trained")])
    notebook(src / "RUN_INFERENCE.ipynb", [("mean_policy.py", "infer")])
    notebook(src / "REPLAY_INFERENCE.ipynb", [("mean_policy.py", "replay")])
    save(src / "02_code/mean-source-manifest.json", {p.relative_to(src).as_posix(): sha(p) for p in src.rglob("*") if p.is_file()})
    training = json.loads((BASE / "06_docs/cpudet-training.json").read_text())
    trainqa = json.loads((BASE / "06_docs/cpudet-training-qa.json").read_text())
    assert trainqa["status"] == "PASS" and trainqa["training_sha256"] == sha(BASE / "06_docs/cpudet-training.json")
    files = {n: h for n, h in training["models"].items() if n.startswith("03_model/full/")}
    assert len(files) == 6
    for n, h in files.items():
        assert sha(BASE / n) == h
        (saved / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / n, saved / n)
    (saved / "04_logs").mkdir(parents=True)
    shutil.copy2(BASE / "04_logs/columns.json", saved / "04_logs/columns.json")
    save(saved / "06_docs/mean-models.json", {"files": files,
        "columns_sha256": sha(saved / "04_logs/columns.json"),
        "training_sha256": sha(BASE / "06_docs/cpudet-training.json")})
    save(saved / "saved-model-manifest.json", {p.relative_to(saved).as_posix(): sha(p) for p in saved.rglob("*") if p.is_file()})
    shutil.copytree(src, OUT / "LOCAL")
    shutil.copytree(saved, OUT / "LOCAL", dirs_exist_ok=True)
    save(OUT / "build.json", {"new_fits": 0, "source_train_executed": False,
        "base_training_sha256": sha(BASE / "06_docs/cpudet-training.json"), "saved_models": 6})


def package():
    replay = json.loads((OUT / "LOCAL/06_docs/mean-replay.json").read_text())
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
            assert all((extracted / n).resolve().is_relative_to(extracted.resolve()) for n in z.namelist())
            z.extractall(extracted)
    save(OUT / "archives.json", archives)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["build", "package"])
    build() if parser.parse_args().stage == "build" else package()
