"""Package only two-cold-exact P3 CPU results; no new training or upload."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from build_candidate_upload_parts_20260906_v1 import split, validate_parts
from build_final_release_20260907_v1 import archive, copy_pins, read, save, saved_notebook, sha


def package(attempt, destination):
    terminal = read(attempt / "terminal_result.json")
    if terminal["status"] != "TWO_COLD_EXACT_PENDING_ARCHIVE_QA" or not terminal["answer_sha_equal"]:
        raise ValueError("two complete matching cold runs required")
    expected = terminal["cold_runs"][0]["answer_sha256"]
    assert len(terminal["cold_runs"]) == 2 and terminal["new_fits"] == 82
    independent = []
    for i in [1, 2]:
        cold = attempt / f"cold_{i}"
        train = read(cold / "06_docs/cpudet-training.json")
        qa = read(cold / "06_docs/cpudet-training-qa.json")
        infer = read(cold / "06_docs/cpudet-answer.json")
        replay = read(cold / "06_docs/cpudet-replay.json")
        assert train["backbone_fits"] == 36 and train["router_fits"] == 5
        assert len(train["fits"]) == 36 and train["official_rows"] == train["hidden_rows"] == 0
        assert qa["status"] == "PASS" and qa["oof_exact"]
        assert qa["training_sha256"] == sha(cold / "06_docs/cpudet-training.json")
        assert len({train["pid"], qa["pid"], infer["pid"], replay["pid"]}) == 4
        for relative, pin in train["models"].items():
            assert sha(cold / relative) == pin
        for fit in train["fits"]:
            assert fit["parameters"]["task_type"] == "CPU" and fit["parameters"]["thread_count"] == 4
            assert fit["iterations"] == fit["parameters"]["iterations"]
        assert infer["sha256"] == replay["sha256"] == expected == sha(cold / "05_answer/submission.csv")
        oof = pd.read_parquet(cold / "04_logs/candidate_oof.parquet")
        assert len(oof) == 103602 and sha(cold / "04_logs/candidate_oof.parquet") == train["oof_sha256"]
        metric = float(np.sqrt(np.mean((oof.target_hs-oof.final_prediction)**2)))
        assert abs(metric-train["internal_rmse"]) < 1e-12
        answer = pd.read_csv(cold / "05_answer/submission.csv")
        assert len(answer) == 1200 and np.isfinite(answer.hs_pred).all()
        assert infer["hidden_rows"] == infer["uploads"] == replay["hidden_rows"] == replay["uploads"] == 0
        independent.append({"cold": i, "answer_sha256": expected, "fits": 41,
                            "internal_rmse": metric, "training_seconds": train["seconds"],
                            "whole_seconds": terminal["cold_runs"][i-1]["whole_seconds"],
                            "oof_rows": len(oof), "qa": "PASS"})
    destination.mkdir(parents=True, exist_ok=False)
    source = attempt / "SOURCE_ONLY"
    pins = read(source / "02_code/cpudet-manifest.json")
    pins["02_code/cpudet-manifest.json"] = sha(source / "02_code/cpudet-manifest.json")
    archives = []
    for role in ["SOURCE_ONLY", "SAVED_MODELS"]:
        target = destination / role
        for name in ["01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"]:
            (target / name).mkdir(parents=True)
        copy_pins(source, target, pins)
        if role == "SAVED_MODELS":
            cold = attempt / "cold_1"
            modelpins = read(cold / "06_docs/cpudet-training.json")["models"]
            extra = ["04_logs/columns.json", "06_docs/prepare.json", "06_docs/cpudet-training.json", "06_docs/cpudet-training-qa.json"]
            modelpins = {**modelpins, **{n:sha(cold/n) for n in extra}}
            copy_pins(cold, target, modelpins)
            saved_notebook("P3", target, [["-I", "02_code/cpudet.py", stage] for stage in ["infer", "replay"]])
        (target / "CURRENT_README.md").write_text(
            "# P3 CPU numeric 3-seed\n\nSOURCE_ONLY: fresh empty model directory, TRAIN.ipynb then PREDICT.ipynb. SAVED_MODELS: fresh extraction, SAVED_PREDICT.ipynb only. Set P3_DATA_DIR to distributed P3 dataset. All fitted values originate in the local cold TRAIN artifacts; no leaderboard-derived fitted constants. No external/pretrained sources.\n\n"
            "36 CatBoost fits + 5 TRAIN-OOF router fits per cold; two independent cold answer hashes match. CPU 4 threads, fixed 3 seeds, fixed shrink 0.2 retained. Historical models are bundled because the unmodified training-lineage guard verifies every model, not because official inference needs historical predictions. No raw tables, prediction caches, answers or locks bundled.\n\n"
            "05_answer/submission.csv is the candidate. Expected SHA256: "+expected+". Official score remains unknown. Two-cold equality is same-environment evidence, not cross-platform determinism or an offline fresh-OS certification. See external extracted saved-notebook execution receipt before final designation.\n",
            encoding="utf-8")
        archives.append(archive(target, destination/f"P3_numeric_cpudet_s3_{role}.zip"))
    savedzip = destination / archives[1]["file"]
    if savedzip.stat().st_size > 45_000_000:
        split(savedzip, destination / "SAVED_MODEL_PARTS")
        validate_parts(destination / "SAVED_MODEL_PARTS/REASSEMBLY_MANIFEST.json")
    (destination / "ANSWER").mkdir()
    shutil.copy2(attempt / "cold_1/05_answer/submission.csv", destination / "ANSWER/submission.csv")
    save(destination / "BUILD_QA.json", {"problem": "P3", "status": "BUILT_PENDING_EXTRACTED_SAVED_NOTEBOOK",
          "answer_file": "ANSWER/submission.csv", "answer_sha256": expected, "archives": archives,
          "independent_qa": independent, "new_fits": 0, "prior_two_cold_fits": 82, "uploads": 0,
          "official_score": None, "fresh_offline_OS_tested": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    package(args.attempt.resolve(), args.destination.resolve())
