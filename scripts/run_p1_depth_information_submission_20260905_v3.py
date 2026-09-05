"""Inference-only materialization of the already frozen P1-A complete policy.

Not a pure depth ablation: final-inner policy/threshold differs from the earlier
official control. No fitting, threshold search, sample labels or hidden truth.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
import run_p1_depth_contract_repair_20260905_v2 as depth

from p1_qc.submission import build_submission, validate_submission

RUN = "p1_depth_information_submission_20260905_v3"
CONFIG = ROOT / "configs/experiments" / (RUN + ".json")
OUT = ROOT / "artifacts" / RUN
REPORT = ROOT / "reports" / RUN
OLD = depth.old
TITLE = "P1 year-safe depth complete policy INFO_ONLY 20260905"
SUMMARY = (
    "배포 train-only balanced/0.2, year-safe 수심 및 final-inner 선택 고정; "
    "내부 F1 -0.002213, 수심 단독 인과검증이 아닌 완성정책 공식 비교."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_recipe(recipe):
    if (recipe["selection"] != "balanced"
            or recipe["calibrations"]["balanced"]["threshold"] != 0.2
            or recipe["decoder_on"] is not False
            or recipe["depth_policy"] != "current_observation_round_2m_explicit_missing"):
        raise ValueError("exact frozen balanced/0.2/depth/decoder contract required")


def checked_contract():
    cfg = read_json(CONFIG)
    if (cfg["experiment_id"] != RUN or cfg["new_fits"] != 0
            or cfg["new_calibrations"] != 0 or cfg["threads"] != 4
            or cfg["official_rows"] != 169011 or cfg["selection"] != "balanced"
            or cfg["threshold"] != 0.2 or cfg["decoder_on"] is not False
            or not cfg["official_test_raw_and_sample_keys_authorized"]
            or cfg["sample_values_and_hidden_authorized"] or cfg["upload_in_this_runner"]):
        raise ValueError("inference-only authorization/contract violation")
    source = ROOT / "artifacts" / cfg["source_run"]
    files = {
        "source_runner": ROOT / "scripts" / ("run_" + cfg["source_run"] + ".py"),
        "source_config": ROOT / "configs/experiments" / (cfg["source_run"] + ".json"),
        "source_result": ROOT / "reports" / cfg["source_run"] / "result.json",
        "source_qa": ROOT / "reports" / cfg["source_run"] / "independent-qa.json",
        "source_recipe": source / "04_models/frozen_recipe.json",
    }
    for name, path in files.items():
        if OLD.sha(path) != cfg[name + "_sha256"]:
            raise ValueError("source seal mismatch: " + name)
    result, qa, recipe = (read_json(files[k]) for k in
                          ("source_result", "source_qa", "source_recipe"))
    if (result["status"] != "TERMINAL_MODELS_FROZEN_OFFICIAL_UNREAD"
            or qa["status"] != "PASS" or not qa["all_16_model_hashes_match"]
            or (result["screen_fits"], result["final_inner_fits"], result["full_fits"])
            != (12, 2, 2)):
        raise ValueError("complete training and independent QA required")
    check_recipe(recipe)
    _, frozen, prior, _ = depth.load_contract(files["source_config"])
    if recipe["feature_contract"] != frozen:
        raise ValueError("feature contract changed")
    for name, expected in recipe["models"].items():
        if OLD.sha(source / "04_models" / (name + ".joblib")) != expected:
            raise ValueError("full model hash mismatch: " + name)
    for package, version in result["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("training/inference package version differs: " + package)
    return cfg, source, result, recipe, prior


def source_closure(prior):
    paths = {str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
             str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
             "scripts/run_p1_depth_contract_repair_20260905_v2.py",
             "scripts/run_p1_score_repair_20260905_v1.py",
             "configs/experiments/p1_depth_contract_repair_20260905_v2.json",
             "configs/experiments/p1_score_repair_20260905_v1.json",
             *prior["dependency_hashes"], *prior["recipe_hashes"]}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(ROOT):
            paths.add(str(path.relative_to(ROOT)).replace("\\", "/"))
    return {path: OLD.sha(ROOT / path) for path in sorted(paths)}


def preflight():
    cfg, _, result, recipe, prior = checked_contract()
    if OUT.exists():
        raise FileExistsError("new inference namespace already exists")
    hashes = source_closure(prior)
    for part in ("01_data", "02_code", "03_training", "04_models", "05_answer", "06_report"):
        (OUT / part).mkdir(parents=True, exist_ok=False)
    for relative in hashes:
        destination = OUT / "02_code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    OLD.write_json(OUT / "04_models/reference.json", {
        "source_run": cfg["source_run"], "models": recipe["models"],
        "recipe_sha256": cfg["source_recipe_sha256"], "model_copy_or_fit": 0,
    })
    receipt = {
        "experiment_id": RUN, "status": "PREFLIGHT_PASS_OFFICIAL_UNREAD",
        "purpose": cfg["purpose"], "source_hashes": hashes,
        "config_sha256": OLD.sha(CONFIG), "runner_sha256": OLD.sha(__file__),
        "source_result_sha256": cfg["source_result_sha256"],
        "source_recipe_sha256": cfg["source_recipe_sha256"],
        "source_model_hashes": recipe["models"], "package_versions": result["versions"],
        "training_input_sha256": result["train_sha256"],
        "source_complete_16fits_seconds": result["runtime_seconds"],
        "source_full2_fit_save_probe_seconds": sum(
            row["seconds"] for row in result["fits"] if row["fold"] == "full"),
        "official_rows": 0, "new_fits": 0, "new_calibrations": 0,
        "hidden_rows": 0, "sample_values": 0, "csv_written": 0, "upload": 0,
    }
    OLD.write_json(OUT / "seal.json", receipt)
    OLD.write_json(REPORT / "preflight.json", receipt)
    print(json.dumps({"status": receipt["status"], "source_files": len(hashes)}), flush=True)


def align_answer(evaluation, bits, sample_keys):
    if evaluation.duplicated(OLD.KEYS).any() or sample_keys.duplicated(OLD.KEYS).any():
        raise ValueError("duplicate official keys")
    answer = build_submission(evaluation, bits)
    aligned = sample_keys.merge(answer, on=OLD.KEYS, how="left", sort=False,
                                validate="one_to_one")
    if len(aligned) != len(evaluation) or aligned.label.isna().any():
        raise ValueError("official key sets differ")
    aligned["label"] = aligned.label.astype(np.int8)
    validate_submission(aligned, sample_keys)
    return aligned


def frozen_predict(evaluation, package, recipe):
    check_recipe(recipe)
    if package["current_depth"] is not True or package["feature_config"] != recipe["feature_contract"]:
        raise ValueError("model feature contract mismatch")
    bundle = depth.features(evaluation, package["train_stats"], recipe["feature_contract"],
                            current_depth=True)
    probability = package["model"].predict_proba(package["encoder"].transform(bundle))[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("nonfinite model probability")
    bits = OLD.decode(evaluation, probability, OLD.rule_masks(evaluation, package["train_stats"]),
                      recipe["feature_contract"], 0.2)
    return bits, depth.depth_audit(evaluation, bundle, package["train_stats"])


def predict(verify=False):
    cfg, source, result, recipe, _ = checked_contract()
    seal = read_json(OUT / "seal.json")
    for path, expected in seal["source_hashes"].items():
        if OLD.sha(ROOT / path) != expected:
            raise ValueError("inference source changed: " + path)
    if seal["config_sha256"] != OLD.sha(CONFIG):
        raise ValueError("inference config changed")
    answer_path = OUT / "05_answer/P1_submission.csv"
    if verify:
        prior_inference = read_json(OUT / "inference-qa.json")
        if prior_inference["status"] != "LOCAL_INFO_ONLY_CANDIDATE_READY":
            raise ValueError("successful candidate required before replay")
    else:
        with (OUT / "INFERENCE_ATTEMPT_LOCK.json").open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "config_sha256": OLD.sha(CONFIG)}, stream)
        if answer_path.exists():
            raise FileExistsError("candidate already exists")
    start = time.monotonic()
    data = Path(os.environ["P1_DATA_DIR"]).resolve()
    # Explicit usecols prevent loading sample label values or any hidden truth.
    evaluation = pd.read_csv(data / "test.csv", usecols=OLD.RAW)
    keys = pd.read_csv(data / "sample_submission.csv", usecols=OLD.KEYS)
    if len(evaluation) != 169011 or len(keys) != 169011:
        raise ValueError("169011 official rows required")
    if not evaluation[OLD.KEYS].equals(keys):
        raise ValueError("distributed test/sample key order differs")
    evaluation.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    evaluation.reset_index(drop=True, inplace=True)
    package = joblib.load(source / "04_models/balanced.joblib")
    bits, audit = frozen_predict(evaluation, package, recipe)
    answer = align_answer(evaluation, bits, keys)
    content = answer.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    official_receipt = read_json(ROOT / cfg["prior_official_receipt"])
    duplicates = [row["candidate"] for row in official_receipt["submissions"]
                  if row["problem"] == "OCN-01" and row["sha256"] == digest]
    if duplicates:
        raise ValueError("duplicate previously submitted P1 SHA")
    if verify:
        if content != answer_path.read_bytes() or digest != prior_inference["sha256"]:
            raise ValueError("fresh process full CSV byte replay differs")
        if prior_inference["pid"] == os.getpid():
            raise ValueError("fresh process required")
    else:
        with answer_path.open("xb") as stream:
            stream.write(content)
    qa = validate_submission(answer_path, keys)
    qa.pop("path", None)
    elapsed = time.monotonic() - start
    qa.update({
        "experiment_id": RUN, "status": "FRESH_PROCESS_FULL_CSV_REPLAY_PASS" if verify
        else "LOCAL_INFO_ONLY_CANDIDATE_READY", "pid": os.getpid(),
        "time_kst": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "purpose": cfg["purpose"], "title": TITLE, "summary": SUMMARY,
        "answer_path": str(answer_path.relative_to(ROOT)).replace("\\", "/"),
        "source_recipe_sha256": cfg["source_recipe_sha256"],
        "selection": "balanced", "threshold": 0.2, "decoder_on": False,
        "depth_audit": audit, "official_test_rows_this_process": len(evaluation),
        "sample_key_rows_this_process": len(keys), "sample_prediction_values_read": 0,
        "hidden_rows": 0, "new_fits": 0, "new_calibrations": 0, "upload": 0,
        "runtime_seconds": elapsed, "expected_official_score": None,
        "internal_delta_f1": result["delta_f1"],
        "prior_official_receipt_sha256": OLD.sha(ROOT / cfg["prior_official_receipt"]),
        "duplicate_prior_recorded_p1_sha": False, "current_browser_quota_checked": False,
        "training_plus_inference_seconds_evidence": result["runtime_seconds"] + elapsed,
        "six_hour_clean_machine_retraining_verified": False,
        "official_test_key_sha256": depth.keys_digest(keys),
        "sample_full_file_hash_not_read_to_avoid_sample_values": True,
        "sha256": digest, "final_model_lock_or_zip": False,
    })
    filename = "replay-qa.json" if verify else "inference-qa.json"
    OLD.write_json(OUT / filename, qa)
    OLD.write_json(REPORT / filename, qa)
    print(json.dumps(qa, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--predict", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        predict(args.verify)
