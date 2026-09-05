"""Two full distributed-train fits, then separately authorized frozen inference."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import joblib
import numpy as np
import pandas as pd
import run_p1_score_repair_20260905_v1 as screen
import run_p1_score_repair_decoder_20260905_v1 as decoder

from p1_qc.submission import build_submission, validate_submission, write_submission

RUN = "p1_clean_control_fulltrain_20260905_v1"
SOURCE_FILES = [
    "scripts/run_p1_clean_control_fulltrain_20260905_v1.py",
    "scripts/run_p1_score_repair_20260905_v1.py",
    "scripts/run_p1_score_repair_decoder_20260905_v1.py",
    "scripts/run_p1_meaningful_learning_curve_generation_v1.py",
    "configs/p1.toml",
    "configs/p1_meaningful_learning_curve_generation_v1.json",
    "configs/experiments/p1_score_repair_20260905_v1.json",
    "configs/experiments/p1_clean_control_fulltrain_20260905_v1.json",
    *[
        "src/p1_qc/" + name + ".py"
        for name in [
            "__init__", "augment", "config", "data", "experiment", "features", "metrics",
            "models_tabular", "pipeline", "postprocess", "rules", "splits", "submission",
            "validation",
        ]
    ],
    "src/ocean_goal/__init__.py",
    "src/ocean_goal/meaningful_score.py",
]


def checked_contract(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if (
        cfg["experiment_id"] != RUN or cfg["full_backbone_fits"] != 2
        or cfg["new_calibration_searches"] != 0 or cfg["threads"] != 4
        or cfg["selection"] != "balanced_union"
        or cfg["calibrations"]["original"]["threshold"] != 0.2
        or cfg["calibrations"]["balanced_union"]["threshold"] != 0.3
    ):
        raise ValueError("fixed fulltrain contract violation")
    for relative, expected in cfg["source_pins"].items():
        if screen.sha(ROOT / relative) != expected:
            raise ValueError("source pin mismatch: " + relative)
    prior = json.loads(
        (ROOT / "reports/p1_score_repair_20260905_v1/result.json").read_text(encoding="utf-8")
    )
    for relative, expected in prior["dependency_hashes"].items():
        if screen.sha(ROOT / relative) != expected:
            raise ValueError("screen dependency changed: " + relative)
    return cfg


def save_code(package, cfg):
    hashes = {}
    for relative in sorted(set(SOURCE_FILES) | set(cfg["source_pins"])):
        destination = package / "02_code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        hashes[relative] = screen.sha(destination)
    return hashes


def train(config_path, output):
    cfg = checked_contract(config_path)
    frozen = json.loads((ROOT / cfg["feature_contract"]).read_text(encoding="utf-8"))
    data = Path(os.environ["P1_DATA_DIR"])
    train_path = data / "train.csv"
    if screen.sha(train_path) != cfg["train_sha256"]:
        raise ValueError("distributed training hash mismatch")
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    receipt = {
        "experiment_id": RUN, "status": "RUNNING", "pid": os.getpid(),
        "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "full_backbone_fits": 0, "new_calibration_searches": 0,
        "transition_estimates": 0, "official_rows": 0, "csv_written": 0, "upload": 0,
        "config_sha256": screen.sha(config_path), "runner_sha256": screen.sha(__file__),
        "train_sha256": cfg["train_sha256"], "threads": 4, "cuda": False,
        "fits": [], "versions": {
            k: importlib.metadata.version(k)
            for k in ["numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "joblib",
                      "pyarrow", "psutil"]
        },
    }
    with (output / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as f:
        json.dump(receipt, f)
    screen.write_json(output / "contract.json", cfg)
    try:
        frame = pd.read_csv(train_path, usecols=screen.RAW + ["label", "anomaly_type"])
        if len(frame) != 776706 or frame.duplicated(screen.KEYS).any():
            raise ValueError("training keys/count invalid")
        if not frame.label.isin([0, 1]).all():
            raise ValueError("training labels invalid")
        frame.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        stats = screen.stats_fit(frame)
        bundle = screen.feature_pair(frame, stats, frozen)[0]
        encoder = screen.TabularEncoder().fit(bundle, np.arange(len(frame)))
        features = encoder.transform(bundle)
        target = frame.label.to_numpy(dtype=np.int8)
        receipt["training_rows"], receipt["feature_count"] = len(frame), features.shape[1]
        screen.write_json(output / "progress.json", receipt)
        models = output / "03_models"
        models.mkdir(exist_ok=True)
        base_cfg = screen.load_config(ROOT / frozen["base_config"], env={})
        lgb_cfg = json.loads((ROOT / frozen["lightgbm_recipe"]).read_text(encoding="utf-8"))
        for name in ["original", "balanced"]:
            fit_start = time.monotonic()
            if name == "original":
                model = screen._fit_model(
                    "xgboost", base_cfg.raw["models"]["xgboost"], frozen["seed"],
                    cfg["threads"], features, target,
                )
            else:
                params = screen._lgb_parameters(lgb_cfg, frozen["seed"], multiclass=False)
                params["n_jobs"] = cfg["threads"]
                model = screen.lgb.LGBMClassifier(**params)
                model.fit(features, target, sample_weight=screen._event_day_weight(frame, target))
            receipt["full_backbone_fits"] += 1
            model_path = models / (name + ".joblib")
            joblib.dump({"model": model, "encoder": encoder, "train_stats": stats}, model_path,
                        compress=3)
            probe = features[:4096]
            first = model.predict_proba(probe)[:, 1]
            replay = joblib.load(model_path)["model"].predict_proba(probe)[:, 1]
            if not np.isfinite(first).all() or not np.array_equal(first, replay):
                raise ValueError("saved model probe replay mismatch")
            receipt["fits"].append({"model": name, "runtime_seconds": time.monotonic()-fit_start,
                                    "sha256": screen.sha(model_path), "reload_probe_equal": True})
            receipt["runtime_seconds"] = time.monotonic() - start
            screen.write_json(output / "progress.json", receipt)
            print(json.dumps({"completed_full_fits": receipt["full_backbone_fits"],
                              "runtime_seconds": receipt["runtime_seconds"]}), flush=True)
            if receipt["runtime_seconds"] > cfg["wall_cap_seconds"]:
                raise RuntimeError("fulltrain wall cap exceeded; no auto restart")
            del model
            gc.collect()
        transition = None
        if cfg["decoder_enabled"]:
            transition = decoder.transition_fit(frame, 1.0)
            receipt["transition_estimates"] = 1
        recipe = {
            "config": cfg, "feature_contract": frozen, "transition": transition,
            "model_hashes": {r["model"]: r["sha256"] for r in receipt["fits"]},
            "official_rows_before_seal": 0,
        }
        screen.write_json(models / "frozen_recipe.json", recipe)
        receipt["frozen_recipe_sha256"] = screen.sha(models / "frozen_recipe.json")
        receipt["source_hashes"] = save_code(output, cfg)
        (output / "02_code" / "requirements.txt").write_text(
            "\n".join(k+"=="+v for k, v in receipt["versions"].items())+"\n", encoding="utf-8")
        receipt["status"] = "MODELS_AND_RECIPE_FROZEN_OFFICIAL_UNREAD"
    except Exception as exc:
        receipt.update(status="TERMINAL_TECHNICAL_FAILURE", error=str(exc))
        raise
    finally:
        receipt["runtime_seconds"] = time.monotonic() - start
        screen.write_json(output / "train_result.json", receipt)
        screen.write_json(ROOT / "reports" / RUN / "train-result.json", receipt)


def align_answer(evaluation, bits, sample_keys):
    if evaluation.duplicated(screen.KEYS).any() or sample_keys.duplicated(screen.KEYS).any():
        raise ValueError("duplicate official keys")
    answer = build_submission(evaluation, bits)
    merged = sample_keys.merge(answer, on=screen.KEYS, how="left", validate="one_to_one", sort=False)
    if len(merged) != len(evaluation) or merged.label.isna().any():
        raise ValueError("official key sets differ")
    merged["label"] = merged.label.astype(np.int8)
    validate_submission(merged, sample_keys)
    return merged


def predict(output, verification_only=False):
    receipt = json.loads((output / "train_result.json").read_text(encoding="utf-8"))
    if receipt["status"] != "MODELS_AND_RECIPE_FROZEN_OFFICIAL_UNREAD":
        raise ValueError("successful complete model seal required")
    models = output / "03_models"
    recipe_path = models / "frozen_recipe.json"
    if screen.sha(recipe_path) != receipt["frozen_recipe_sha256"]:
        raise ValueError("frozen recipe changed")
    for relative, expected in receipt["source_hashes"].items():
        if screen.sha(ROOT / relative) != expected:
            raise ValueError("inference source changed: " + relative)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    cfg, frozen = recipe["config"], recipe["feature_contract"]
    if not cfg["official_inference_after_model_seal_authorized"]:
        raise ValueError("official inference authorization absent")
    start = time.monotonic()
    data = Path(os.environ["P1_DATA_DIR"])
    # No hidden labels, sample values, baseline predictions, or old submission is read.
    evaluation = pd.read_csv(data / "test.csv", usecols=screen.RAW)
    sample_keys = pd.read_csv(data / "sample_submission.csv", usecols=screen.KEYS)
    if len(evaluation) != 169011 or len(sample_keys) != 169011:
        raise ValueError("official row count mismatch")
    test_keys = evaluation[screen.KEYS].copy()
    if not test_keys.equals(sample_keys):
        raise ValueError("distributed test/sample order mismatch")
    evaluation.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    evaluation.reset_index(drop=True, inplace=True)
    probabilities, stats, bundle = {}, None, None
    for name in ["original", "balanced"]:
        path = models / (name + ".joblib")
        if screen.sha(path) != recipe["model_hashes"][name]:
            raise ValueError("model hash mismatch")
        package = joblib.load(path)
        if bundle is None:
            stats = package["train_stats"]
            bundle = screen.feature_pair(evaluation, stats, frozen)[0]
        probability = package["model"].predict_proba(package["encoder"].transform(bundle))[:, 1]
        if not np.isfinite(probability).all():
            raise ValueError("nonfinite official inference")
        probabilities[name] = probability
    reference, unary, hard = decoder.control_components(
        evaluation, probabilities, screen.rule_masks(evaluation, stats), frozen,
        cfg["selection"], cfg["calibrations"], 1e-6,
    )
    prediction = (
        decoder.decode_viterbi(evaluation, unary, recipe["transition"], hard, 1.0)
        if cfg["decoder_enabled"] else reference
    )
    answer = align_answer(evaluation, prediction, sample_keys)
    answer_path = output / "05_answer" / "P1_submission.csv"
    if verification_only:
        actual_bytes = answer.to_csv(index=False, lineterminator="\n").encode("utf-8")
        if actual_bytes != answer_path.read_bytes():
            raise ValueError("fresh-process model replay answer mismatch")
    else:
        if answer_path.exists():
            raise FileExistsError("answer already exists; use --verify without overwriting")
        write_submission(answer, answer_path)
    qa = validate_submission(answer_path, sample_keys)
    qa.pop("path", None)
    qa.update({
        "status": "FRESH_PROCESS_REPLAY_PASS" if verification_only else "LOCAL_UNSCORED_CSV_READY",
        "official_test_rows": len(evaluation), "sample_key_rows": len(sample_keys),
        "sample_prediction_values_read": 0, "hidden_rows": 0, "upload": 0,
        "new_backbone_fits": 0, "new_calibration_searches": 0,
        "decoder_enabled": cfg["decoder_enabled"], "runtime_seconds": time.monotonic()-start,
        "test_sha256": screen.sha(data / "test.csv"),
        "sample_sha256": screen.sha(data / "sample_submission.csv"),
        "unknown_training_depth_key_rows": int(bundle.frame.nominal_depth_m.isna().sum()),
        "official_score": "UNMEASURED; historical official 28.909341 does not belong to this CSV",
        "final_zip_verified": False,
    })
    filename = "replay-qa.json" if verification_only else "inference-qa.json"
    screen.write_json(output / filename, qa)
    screen.write_json(ROOT / "reports" / RUN / filename, qa)
    print(json.dumps(qa), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/experiments" / (RUN + ".json"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / RUN)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true")
    action.add_argument("--predict", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.train:
        train(args.config, args.output)
    else:
        predict(args.output, args.verify)
