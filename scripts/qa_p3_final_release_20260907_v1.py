"""Aggregate QA linking scored saved replay and separate whole-cold numeric execution."""
from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from build_final_release_20260907_v1 import archive, read, save, sha


def prediction_ast(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    # Error-message labels differ from "cold" to "scored"; computational AST must not.
    for item in ast.walk(node):
        if isinstance(item, ast.Raise) and isinstance(item.exc, ast.Call):
            item.exc.args = [ast.Constant(value="diagnostic")]
    return ast.dump(node, include_attributes=False)


def verify(repo, root, output):
    cold = root / "P3_cold_validation/P3_numeric_cold_v1"
    training = read(cold / "06_docs/training-result.json")
    qa = read(cold / "06_docs/training-qa.json")
    replay = read(cold / "06_docs/answer-replay-qa.json")
    saved = read(root / "P3_saved_prediction/answer-qa.json")
    build = read(root / "P3/P3_SCORED_BUILD_QA.json")
    original = repo / "artifacts/p3_numeric_candidate_20260906_v1/05_answer/submission.csv"
    expected = "ff42a6a08c76f0d58ed2f3a9ea31a08819ada5fa6e9007af942fe0b891937960"
    checks = {
        "cold_12_backbone_5_router": (training["new_backbone_fits"], training["new_router_fits"]) == (12, 5),
        "cold_training_QA": qa["status"] == "PASS",
        "cold_replay": replay["status"] == "EXACT_ANSWER_REPLAY_PASS",
        "cold_6h": replay["elapsed_seconds"] < 21600,
        "cold_training_official_zero": training["official_rows"] == 0,
        "cold_hidden_zero": replay["hidden_rows"] == 0,
        "cold_train_notebook": read(cold / "06_docs/executed_notebooks/TRAIN-receipt.json")["status"] == "PASS",
        "cold_predict_notebook": read(cold / "06_docs/executed_notebooks/PREDICT-receipt.json")["status"] == "PASS",
        "scored_saved_exact": saved["status"] == "EXACT_SAVED_INFERENCE_PASS" and saved["sha256"] == expected,
        "scored_original_exact": sha(original) == expected,
        "scored_extracted_exact": sha(root / "P3_saved_prediction/submission.csv") == expected,
        "saved_hidden_zero": saved["hidden_rows"] == 0,
        "saved_fits_zero": saved["model_fits"] == 0,
        "saved_old_answers_zero": saved["old_answer_value_reads"] == 0,
        "saved_archive_hash": sha(root / "P3/P3_SAVED_MODELS.zip") == build["sha256"],
        "cold_own_answer_hash": sha(cold / "05_answer/submission.csv") == replay["sha256"],
    }
    old_code = (repo / "scripts/portable_20260906/P3_forward_saved_v1/infer.py").read_text()
    new_code = (root / "P3_saved_validation/02_code/infer.py").read_text()
    for name in ("load_policy", "make_answer"):
        checks[name + "_computation_AST_unchanged"] = prediction_ast(old_code, name) == prediction_ast(new_code, name)
    if not all(checks.values()):
        raise ValueError("final P3 checks failed: " + str([k for k, v in checks.items() if not v]))
    a = pd.read_csv(original)
    b = pd.read_csv(cold / "05_answer/submission.csv")
    if not a.iloc[:, :3].equals(b.iloc[:, :3]):
        raise ValueError("cold versus scored key order differs")
    delta = b.hs_pred.to_numpy() - a.hs_pred.to_numpy()
    difference = {"changed_rows": int(np.count_nonzero(delta)), "rows": len(a),
                  "max_abs_m": float(np.max(np.abs(delta))), "mean_abs_m": float(np.mean(np.abs(delta))),
                  "prediction_difference_rmse_m": float(np.sqrt(np.mean(delta * delta))),
                  "meaning": "difference between predictions, not RMSE against hidden truth"}
    source_zip = archive(repo / "final_packages/P3", root / "P3/P3_SOURCE_ONLY.zip")
    answer_dir = root / "P3/ANSWER"
    answer_dir.mkdir(exist_ok=False)
    shutil.copy2(original, answer_dir / "submission_scored_numeric.csv")
    shutil.copy2(cold / "05_answer/submission.csv", answer_dir / "NOT_SCORED_whole_cold.csv")
    evidence = {"saved_inference_status": "EXACT_SAVED_REPLAY_PASS", "answer_sha256": expected,
                "saved_inference_seconds": saved["seconds"],
                "full_cold_fits": {"backbone": 12, "router": 5},
                "full_cold_seconds": replay["elapsed_seconds"],
                "cold_answer_sha256": replay["sha256"], "cold_answer_equals_scored": replay["sha256"] == expected,
                "cold_vs_scored": difference, "checks": checks, "checks_passed": len(checks),
                "archives": [{k: source_zip[k] for k in ("file", "sha256", "bytes", "files")},
                             {k: build[k] for k in ("file", "sha256", "bytes", "files")}],
                "training_qa_sha256": sha(cold / "06_docs/training-qa.json"),
                "scored_saved_qa_sha256": sha(root / "P3_saved_prediction/answer-qa.json"),
                "split_parts": 0, "official_score_for_new_cold": None,
                "status": "SCORED_SAVED_EXACT_AND_SEPARATE_COLD_PASS_NOT_BYTE_IDENTICAL",
                "package_build_prefit_failure": "First metadata-only package copy stopped on overlapping provenance text replacement; preserved, fixed in new v2 folder. Fits and inference before that correction: 0."}
    save(output, evidence)
    print(evidence["status"], difference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.repo.resolve(), args.root.resolve(), args.output.resolve())
