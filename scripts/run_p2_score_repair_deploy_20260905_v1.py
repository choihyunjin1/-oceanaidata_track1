"""Separate clean P2 fullfit and fresh-process prediction after internal QA.

Training needs only organizer observations, never OOF or old answer files.
Prediction reads official key columns only after the model recipe is frozen.
No score, baseline file, internet, upload or Git command is implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_score_repair_20260905_v1 as research  # noqa: E402

EXPERIMENT = "p2_score_repair_deploy_20260905_v1"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT
REPORT = ROOT / "reports" / EXPERIMENT
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT}.json"
KEYS = ["station", "layer", "time"]
OUTPUT_COLUMNS = [*KEYS, "temp"]


def canonical_keys(frame: pd.DataFrame) -> pd.Series:
    return frame["station"].astype(str) + "|" + frame["layer"].astype(int).astype(str) + "|" + pd.to_datetime(frame["time"], utc=True).astype(str)


def validate_output(frame: pd.DataFrame, sample: pd.DataFrame, index: pd.DataFrame, expected_rows: int) -> dict:
    keys, sample_keys, index_keys = canonical_keys(frame), canonical_keys(sample), canonical_keys(index)
    checks = {
        "schema_exact": list(frame.columns) == OUTPUT_COLUMNS,
        "row_count_exact": len(frame) == expected_rows == len(sample) == len(index),
        "keys_unique": not keys.duplicated().any() and not sample_keys.duplicated().any() and not index_keys.duplicated().any(),
        "sample_order_exact": keys.tolist() == sample_keys.tolist(),
        "index_keys_exact": set(keys) == set(index_keys),
        "finite_predictions": bool(np.isfinite(frame["temp"].to_numpy(float)).all()),
        "target_layers_exact": set(frame["layer"].astype(int)) == {2, 3, 4},
    }
    if not all(checks.values()):
        raise ValueError(f"P2 output contract failed: {checks}")
    return {key: bool(value) for key, value in checks.items()}


def load_contract() -> tuple[dict, dict, Path]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["experiment_id"] == EXPERIMENT and config["arm"] == "v23_blockmask"
    assert config["fullfit_count"] == 3 and config["expected_output_rows"] == 26061
    if research.file_hash(ROOT / "scripts/run_p2_score_repair_20260905_v1.py") != config["research_runner_sha256"]:
        raise RuntimeError("frozen research runner drift")
    if research.file_hash(research.CONFIG) != config["research_config_sha256"]:
        raise RuntimeError("frozen research config drift")
    data_root = os.environ.get("P2_DATA_DIR")
    if not data_root:
        raise RuntimeError("P2_DATA_DIR is required")
    source = Path(data_root).resolve() / "observations.csv"
    training = research.load_config()
    if research.file_hash(source) != training["source_sha256"]:
        raise RuntimeError("source drift")
    return config, training, source


def train(root_qa_pass: bool) -> dict:
    if not root_qa_pass:
        raise RuntimeError("root numerical QA approval must precede fullfit")
    config, training, source = load_contract()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training required by current resource contract")
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT / "TRAIN_ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "config_sha256": research.file_hash(CONFIG), "runner_sha256": research.file_hash(Path(__file__)), "root_numerical_qa_pass": True}, handle)
    started = time.monotonic()
    observations = pd.read_csv(source)
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    observations = observations.loc[(observations.time >= research.utc(training["train_start"])) & (observations.time < research.utc(training["train_stop"]))].copy()
    frame, truth = research.public_frame(observations)
    keep = np.isfinite(truth) & np.isfinite(frame.baseline) & frame.public_temp_count.ge(2).to_numpy()
    frame, truth = frame.loc[keep].reset_index(drop=True), truth[keep]
    data, training_receipt = research.training_arrays(frame, truth, config["arm"], training)
    fits = []
    for seed in training["seeds"]:
        def progress(epoch, elapsed, current_seed=seed):
            research.atomic_json(ARTIFACT / "progress.json", {"status": "TRAINING", "pid": os.getpid(), "seed": current_seed, "epoch": epoch, "fullfits_completed": len(fits), "fit_seconds": elapsed, "total_seconds": time.monotonic() - started})
        model, receipt = research.fit_model(data, config["arm"], seed, training, progress)
        path = ARTIFACT / f"model_seed{seed}.pt"
        torch.save(model.cpu().state_dict(), path)
        receipt.update({"file": path.name, "sha256": research.file_hash(path)})
        fits.append(receipt)
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"fullfit_complete": seed, "fit_seconds": receipt["runtime_seconds"]}), flush=True)
    result = {"experiment_id": EXPERIMENT, "status": "FULLFIT_COMPLETE", "arm": config["arm"], "training": training_receipt, "training_start": training["train_start"], "training_stop": training["train_stop"], "fullfit_count": len(fits), "calibration_fits": 0, "oas_fits": 0, "fits": fits, "official_key_rows_read": 0, "sample_value_rows_read": 0, "csv_written": 0, "upload": 0, "runtime_seconds": time.monotonic() - started, "source_sha256": training["source_sha256"], "config_sha256": research.file_hash(CONFIG), "runner_sha256": research.file_hash(Path(__file__)), "research_runner_sha256": config["research_runner_sha256"], "research_config_sha256": config["research_config_sha256"]}
    research.atomic_json(ARTIFACT / "train-result.json", result)
    research.atomic_json(REPORT / "train-result.json", result)
    return result


def predict() -> dict:
    config, _, source = load_contract()
    trained = json.loads((ARTIFACT / "train-result.json").read_text(encoding="utf-8"))
    if trained["config_sha256"] != research.file_hash(CONFIG) or trained["fullfit_count"] != 3:
        raise RuntimeError("train recipe mismatch")
    with (ARTIFACT / "PREDICT_ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "config_sha256": research.file_hash(CONFIG)}, handle)
    started = time.monotonic()
    # Read key columns only, never sample numerical placeholders or official baseline.
    index = pd.read_csv(source.parent / "test_index.csv", usecols=KEYS)
    sample = pd.read_csv(source.parent / "sample_submission.csv", usecols=KEYS)
    observations = pd.read_csv(source)
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    frame, _ = research.public_frame(observations)
    lookup = canonical_keys(frame)
    if lookup.duplicated().any():
        raise ValueError("feature keys are not unique")
    frame.index = lookup
    wanted = canonical_keys(sample)
    if not wanted.isin(frame.index).all():
        raise ValueError("official keys absent from released observations")
    query = frame.loc[wanted].reset_index(drop=True)
    query_arrays = research.arrays(query)
    per_seed = []
    replay_errors = []
    for receipt in trained["fits"]:
        path = ARTIFACT / receipt["file"]
        if research.file_hash(path) != receipt["sha256"]:
            raise RuntimeError("saved model hash drift")
        model = research.make_model(config["arm"], query_arrays[2].shape[1])
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        first = research.predict_model(model, *query_arrays)
        second = research.predict_model(model, *query_arrays)
        replay_errors.append(float(np.max(np.abs(first - second))))
        if not np.array_equal(first, second):
            raise RuntimeError("repeat inference mismatch")
        per_seed.append(query.baseline.to_numpy(float) + first * research.compute_profile_scale(query))
    prediction = np.mean(per_seed, axis=0)
    output = sample.loc[:, KEYS].copy()
    output["temp"] = prediction
    checks = validate_output(output, sample, index, config["expected_output_rows"])
    output_path = ARTIFACT / "submission_p2_v23_blockmask_3seed.csv"
    output.to_csv(output_path, index=False, float_format="%.12g")
    reread = pd.read_csv(output_path)
    checks.update({"serialized_" + key: value for key, value in validate_output(reread, sample, index, config["expected_output_rows"]).items()})
    serialization_error = float(np.max(np.abs(reread.temp.to_numpy(float) - prediction)))
    if serialization_error > 1e-8:
        raise RuntimeError("CSV serialization exceeds precision contract")
    np.savez_compressed(ARTIFACT / "official_predictions_private.npz", key=wanted.to_numpy(str), prediction=prediction, per_seed=np.stack(per_seed))
    result = {"experiment_id": EXPERIMENT, "status": "CANDIDATE_READY_NOT_UPLOADED", "candidate": str(output_path.relative_to(ROOT)).replace("\\", "/"), "candidate_sha256": research.file_hash(output_path), "rows": len(output), "columns": OUTPUT_COLUMNS, "official_key_rows_read": {"index": len(index), "sample": len(sample)}, "sample_value_rows_read": 0, "hidden_truth_rows_read": 0, "official_score_rows_read": 0, "old_answer_inputs": 0, "new_training_fits": 0, "checks": checks, "max_repeat_inference_abs_error": max(replay_errors), "max_serialization_abs_error": serialization_error, "runtime_seconds": time.monotonic() - started, "csv_written": 1, "upload": 0, "title": "P2 clean v23 blockmask 3seed", "one_line_summary": "배포 관측만으로 scratch 학습한 raw DeepSets 3seed 평균; T5/S5 연속결측 증강, 과거 답안·공식역산 계수 미사용."}
    research.atomic_json(ARTIFACT / "predict-result.json", result)
    research.atomic_json(REPORT / "predict-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "predict"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-qa-pass", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"stage": args.stage, "execute": False}))
        return
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            result = train(args.root_qa_pass) if args.stage == "train" else predict()
        print(json.dumps({"status": result["status"]}), flush=True)
    except Exception as exc:
        research.atomic_json(REPORT / f"{args.stage}-failure.json", {"status": "TERMINAL_TECHNICAL_FAILURE", "exception": type(exc).__name__, "error": str(exc), "restart_authorized": False})
        raise


if __name__ == "__main__":
    main()
