"""L120 exact frozen full3 -> training replay -> official answer -> full CSV replay."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import base
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

PACKAGE = Path(__file__).resolve().parents[1]
LOGS, MODELS, ANSWERS = (PACKAGE / p for p in ("04_logs", "03_model", "05_answer"))


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()


def verify():
    manifest = json.loads((PACKAGE / "PACKAGE_MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        if base.sha(PACKAGE / relative) != expected:
            raise ValueError("package source/provenance drift")
    cfg = json.loads((PACKAGE / "config.json").read_text())
    if cfg["recipe"] != "L120" or cfg["epochs"] != 120 or cfg["weight_decay"] != .0001 or cfg["maximum_new_fits"] != 3:
        raise ValueError("selected recipe changed")
    if cfg["seeds"] != [20260901, 20260902, 20260903] or cfg["cpu_threads"] != 2 or cfg["device"] != "cuda":
        raise ValueError("same-resource 3seed recipe required")
    historical = json.loads((PACKAGE / "06_docs/historical_result.json").read_text())
    qa = json.loads((PACKAGE / "06_docs/historical_qa.json").read_text())
    replay = json.loads((PACKAGE / "06_docs/historical_replay.json").read_text())
    root = json.loads((PACKAGE / "06_docs/root_qa.json").read_text())
    if not (historical["selected_challenger"] == "L120" and historical["retained"] and
            historical["completed_fits"] == 28 and qa["status"] == replay["status"] == root["status"] == "PASS"):
        raise ValueError("complete internally retained historical lineage required")
    if qa["terminal_result_sha256"] != cfg["historical_result_sha256"] or replay["terminal_result_sha256"] != cfg["historical_result_sha256"]:
        raise ValueError("historical result QA/replay link mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("same CUDA environment required; no silent CPU fallback")
    return cfg


def train(cfg, source):
    if any(MODELS.iterdir()):
        raise ValueError("empty03 required")
    original_fit = base.core.fit_model

    def tracked_fit(data, arm, seed, recipe, progress):
        model, receipt = original_fit(data, arm, seed, recipe, progress)
        normalized = base.core.predict_model(model, data[0][:166268], data[1][:166268], data[2][:166268])
        receipt["full_training_normalized_prediction_sha256"] = array_sha(normalized)
        receipt["training_arrays_sha256"] = [array_sha(a) for a in data]
        return model, receipt
    base.core.fit_model = tracked_fit
    base.train(cfg, source)


def train_replay(cfg, source):
    training = json.loads((MODELS / "MODEL_MANIFEST.json").read_text())
    if training["pid"] == os.getpid() or training["new_full_fits"] != 3 or training["code_hashes"] != base.fingerprint():
        raise ValueError("fresh-process full3 lineage required")
    base.save(LOGS / "TRAIN_REPLAY_ATTEMPT_LOCK.json", {"pid": os.getpid(), "training_pid": training["pid"]})
    base.install_guard("TRAIN_REPLAY", source)
    started = time.monotonic()
    frame, truth = base.read_observations(source, cfg, True)
    data, support = base.core.training_arrays(frame, truth, "v23_blockmask", cfg)
    checks = {"empty_before_training": training["empty_model_directory_before_training"],
              "same166268": len(frame) == 166268, "same_training_support": support == training["training"],
              "source_sha": base.sha(source) == cfg["source_sha256"], "new_full3": training["new_full_fits"] == 3,
              "training_official_zero": training["official_access_rows"] == 0 and training["csv_written"] == 0,
              "seeds": [f["seed"] for f in training["fits"]] == cfg["seeds"]}
    for fit in training["fits"]:
        path = MODELS / fit["file"]
        checks[str(fit["seed"]) + ":model_sha"] = base.sha(path) == fit["sha256"]
        checks[str(fit["seed"]) + ":training_arrays"] = [array_sha(a) for a in data] == fit["training_arrays_sha256"]
        checks[str(fit["seed"]) + ":recipe"] = fit["epochs"] == 120 and fit["device"] == "cuda" and fit["cpu_threads"] == 2
        model = base.core.make_model("v23_blockmask", 11)
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        normalized = base.core.predict_model(model, *base.core.arrays(frame))
        checks[str(fit["seed"]) + ":entire_training_replay"] = array_sha(normalized) == fit["full_training_normalized_prediction_sha256"]
        del model
        torch.cuda.empty_cache()
    if not all(checks.values()):
        raise ValueError("full training replay failed: " + str(checks))
    verify()
    base.save(LOGS / "training-independent-qa.json", {"status": "PASS", "pid": os.getpid(), "training_pid": training["pid"],
        "checks": checks, "checks_passed": len(checks), "rows_per_model": len(frame), "models": 3,
        "training_result_sha256": base.sha(LOGS / "training-result.json"), "model_manifest_sha256": base.sha(MODELS / "MODEL_MANIFEST.json"),
        "historical_qa_sha256": cfg["historical_qa_sha256"], "source_sha256": base.sha(source),
        "scope": "all source-training predictions exact replay; not a new validation score",
        "runtime_seconds": time.monotonic() - started, "new_fits": 0, "official_rows": 0})
    print(json.dumps({"status": "TRAIN_REPLAY_PASS", "checks": len(checks), "fits": 0}))


def infer(cfg, source, replay):
    qa = json.loads((LOGS / "training-independent-qa.json").read_text())
    training = json.loads((LOGS / "training-result.json").read_text())
    if (qa["status"] != "PASS" or qa["training_result_sha256"] != base.sha(LOGS / "training-result.json")
            or qa["model_manifest_sha256"] != base.sha(MODELS / "MODEL_MANIFEST.json")
            or qa["training_pid"] != training["pid"]):
        raise ValueError("independent training QA must precede official I/O")
    base.infer(cfg, source, replay)


def final_qa(cfg, source):
    base.install_guard("REPLAY", source)
    train = json.loads((LOGS / "training-result.json").read_text())
    qa = json.loads((LOGS / "training-independent-qa.json").read_text())
    infer = json.loads((LOGS / "inference-result.json").read_text())
    replay = json.loads((LOGS / "replay-result.json").read_text())
    path = PACKAGE / infer["answer"]
    frame = pd.read_csv(path)
    index = pd.read_csv(source.parent / "test_index.csv", usecols=base.KEYS)
    sample = pd.read_csv(source.parent / "sample_submission.csv", usecols=base.KEYS)
    checks = base.validate_output(frame, sample, index, cfg["expected_rows"])
    checks.update(training_pass=train["status"] == "SCRATCH_TRAINING_PASS", training_qa_pass=qa["status"] == "PASS",
        full_csv_replay=replay["status"] == "FRESH_PROCESS_REPLAY_PASS" and replay["exact_first_inference_sha_match"],
        separate_pids=len({train["pid"], qa["pid"], infer["pid"], replay["pid"]}) == 4,
        hash=base.sha(path) == infer["answer_sha256"] == replay["answer_sha256"] == base.sha(PACKAGE / replay["answer"]),
        LF=b"\r\n" not in path.read_bytes(),
        current_models_unchanged=all(base.sha(MODELS / fit["file"]) == fit["sha256"] for fit in train["fits"]),
        source_unchanged=base.sha(source) == cfg["source_sha256"],
        no_sample_values=all(x["access"]["sample_value_rows"] == 0 for x in (infer, replay)))
    verify()
    if not all(checks.values()):
        raise ValueError("candidate final QA failed: " + str(checks))
    base.save(LOGS / "independent-qa.json", {"status": "PASS", "pid": os.getpid(), "checks": checks, "checks_passed": len(checks),
        "answer": infer["answer"], "answer_sha256": base.sha(path), "rows": len(frame), "new_full_fits": 3,
        "training_sha256": base.sha(LOGS / "training-result.json"), "training_qa_sha256": base.sha(LOGS / "training-independent-qa.json"),
        "inference_sha256": base.sha(LOGS / "inference-result.json"), "replay_sha256": base.sha(LOGS / "replay-result.json"),
        "source_sha256": base.sha(source), "official_index_sha256": base.sha(source.parent / "test_index.csv"),
        "official_sample_sha256": base.sha(source.parent / "sample_submission.csv"), "hidden_rows": 0, "upload": 0})
    print(json.dumps({"status": "CANDIDATE_LOCAL_ANSWER_READY", "rows": len(frame), "sha256": base.sha(path)}))


def execute(cfg):
    started = time.time()
    base.save(LOGS / "FULL_CYCLE_ATTEMPT_LOCK.json", {"pid": os.getpid(), "started_unix": started,
        "maximum_new_fits": 3, "wallcap_seconds": cfg["wallcap_seconds"], "GPU0_authorized": True})
    for mode in ("RUN_TRAINING", "TRAIN_REPLAY", "RUN_INFERENCE", "REPLAY", "FINAL_QA"):
        remaining = cfg["wallcap_seconds"] - (time.time() - started)
        if remaining <= 0:
            raise RuntimeError("30minute full-cycle wallcap")
        child = subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()), mode])
        try:
            code = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False, capture_output=True)
            child.wait(timeout=30)
            raise RuntimeError("full-cycle cap, exact child tree stopped") from None
        if code:
            raise RuntimeError("terminal stage failure, no restart: " + mode)
    base.save(LOGS / "terminal_result.json", {"status": "COMPLETE_LOCAL_CANDIDATE_READY", "new_full_fits": 3,
        "runtime_seconds": time.time() - started, "pid": os.getpid(), "upload": 0,
        "qa_sha256": base.sha(LOGS / "independent-qa.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("execute", "RUN_TRAINING", "TRAIN_REPLAY", "RUN_INFERENCE", "REPLAY", "FINAL_QA"))
    args = parser.parse_args()
    cfg = verify()
    source = (Path(os.environ["P2_DATA_DIR"]) / "observations.csv").resolve()
    torch.set_num_threads(2)
    try:
        with threadpool_limits(2):
            if args.mode == "execute":
                execute(cfg)
            elif args.mode == "RUN_TRAINING":
                train(cfg, source)
            elif args.mode == "TRAIN_REPLAY":
                train_replay(cfg, source)
            elif args.mode in ("RUN_INFERENCE", "REPLAY"):
                infer(cfg, source, args.mode == "REPLAY")
            else:
                final_qa(cfg, source)
    except BaseException as error:
        failure = LOGS / (args.mode + "-failure.json")
        if not failure.exists():
            base.save(failure, {"status": "TERMINAL_TECHNICAL_FAILURE", "mode": args.mode, "type": type(error).__name__,
                "error": str(error), "automatic_restart": False})
        raise


if __name__ == "__main__":
    main()
