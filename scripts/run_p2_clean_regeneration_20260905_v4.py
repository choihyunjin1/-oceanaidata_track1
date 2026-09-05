"""Isolated empty-model C3 scratch regeneration, never import old answers."""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_score_repair_deploy_20260905_v1 as canonical  # noqa: E402

research = canonical.research
ID = "p2_clean_regeneration_20260905_v4"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
OUT, REPORT = ROOT / "artifacts" / ID, ROOT / "reports" / ID
MODELS, ANSWERS = OUT / "03_model", OUT / "05_answer"
SEAL = REPORT / "preregistration-seal.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["experiment_id"] == ID and cfg["maximum_new_full_fits"] == 3
    for kind in ("runner", "config"):
        assert research.file_hash(ROOT / cfg[f"canonical_deploy_{kind}"]) == cfg[f"canonical_deploy_{kind}_sha256"]
    deploy, training, source = canonical.load_contract()
    assert cfg["source_sha256"] == training["source_sha256"]
    assert training["seeds"] == cfg["seeds"] and training["epochs"] == cfg["epochs"]
    assert deploy["arm"] == "v23_blockmask"
    return cfg, training, source


def fingerprints():
    cfg, _, _ = config()
    paths = [cfg["canonical_deploy_runner"], cfg["canonical_deploy_config"], "scripts/run_p2_score_repair_20260905_v1.py", "configs/experiments/p2_score_repair_20260905_v1.json", *research.DEPENDENCIES]
    return {"runner": research.file_hash(Path(__file__)), "config": research.file_hash(CONFIG), "dependencies": {p: research.file_hash(ROOT / p) for p in paths}}


def path_allowed(path, mode, writing, source):
    """Pure synthetic-testable allowlist; old models, OOF and answers are denied."""
    path = Path(path).resolve()
    if "external_data" in path.parts or "hidden" in path.name.lower():
        return False
    if path.suffix.lower() == ".csv":
        if path == source:
            return not writing
        if mode == "RUN_INFERENCE" and path in {source.parent / "test_index.csv", source.parent / "sample_submission.csv"}:
            return not writing
        return mode == "RUN_INFERENCE" and ANSWERS in path.parents
    if path.suffix.lower() == ".pt":
        return MODELS in path.parents and (writing or mode == "RUN_INFERENCE")
    if path.suffix.lower() in {".npz", ".npy", ".parquet", ".pkl", ".pickle"}:
        return False
    return True


def install_guard(mode, source):
    counts = {"source_reads": 0, "official_index_key_rows": 0, "official_sample_key_rows": 0, "sample_value_rows": 0, "old_model_reads": 0, "old_answer_reads": 0}

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network prohibited")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        writing = isinstance(args[1], str) and any(c in args[1] for c in "wax+")
        if not path_allowed(args[0], mode, writing, source):
            raise PermissionError("regeneration data/model allowlist denied")
    sys.addaudithook(guard)
    reader = pd.read_csv

    def guarded_reader(path, *args, **kwargs):
        path = Path(path).resolve()
        if path == source:
            counts["source_reads"] += 1
        elif path in {source.parent / "test_index.csv", source.parent / "sample_submission.csv"}:
            if mode != "RUN_INFERENCE" or kwargs.get("usecols") != canonical.KEYS:
                raise PermissionError("official key columns only")
            value = reader(path, *args, **kwargs)
            counts["official_index_key_rows" if path.name == "test_index.csv" else "official_sample_key_rows"] += len(value)
            return value
        elif ANSWERS not in path.parents:
            raise PermissionError("unapproved CSV reader")
        return reader(path, *args, **kwargs)
    pd.read_csv = guarded_reader
    return counts


def forbidden_legacy(*args, **kwargs):
    raise PermissionError("historical OAS/alpha axis route not part of C regeneration")


def run_training():
    cfg, _, source = config()
    assert fingerprints() == json.loads(SEAL.read_text(encoding="utf-8"))["hashes"]
    if OUT.exists():
        raise RuntimeError("isolated regeneration exactly-once directory already exists")
    OUT.mkdir(parents=True)
    MODELS.mkdir()
    assert not list(MODELS.iterdir())
    save(OUT / "TRAIN_ATTEMPT_LOCK.json", {"pid": os.getpid(), "empty_model_directory_before_training": True, "time_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    counts = install_guard("RUN_TRAINING", source)
    research.predict_forward_seasonal_oas = forbidden_legacy
    canonical.ARTIFACT, canonical.REPORT = MODELS, OUT / "06_report"
    started = time.monotonic()
    result = canonical.train(True)
    assert result["fullfit_count"] == 3
    assert research.file_hash(source) == cfg["source_sha256"]
    receipt = {"experiment_id": ID, "status": "SCRATCH_TRAINING_PASS", "pid": os.getpid(), "source_sha256_before_after": cfg["source_sha256"], "empty_model_directory_before_training": True, "new_full_fits": 3, "model_files": result["fits"], "canonical_training_runtime_seconds": result["runtime_seconds"], "runtime_seconds": time.monotonic() - started, "canonical_train_result_sha256": research.file_hash(MODELS / "train-result.json"), "source_original_rows": result["training"]["original_rows"], "source_augmented_rows": result["training"]["augmented_rows"], "training_weight_sum": result["training"]["training_weight_sum"], "access": counts, "official_access_rows": 0, "csv_written": 0, "upload": 0, "legacy_route_blocked": True, "called_training_function": "canonical.train -> research.training_arrays -> research.fit_model -> newly_initialized_VerticalDeepSet"}
    save(REPORT / "training-result.json", receipt)
    print(json.dumps({"status": receipt["status"], "runtime_seconds": receipt["runtime_seconds"], "gpu_released_on_process_exit": True}), flush=True)


def run_inference():
    cfg, _, source = config()
    assert fingerprints() == json.loads(SEAL.read_text(encoding="utf-8"))["hashes"]
    training = json.loads((REPORT / "training-result.json").read_text(encoding="utf-8"))
    assert training["pid"] != os.getpid()
    if ANSWERS.exists():
        raise RuntimeError("exactly-once answer directory exists")
    ANSWERS.mkdir()
    save(OUT / "INFERENCE_ATTEMPT_LOCK.json", {"pid": os.getpid(), "training_pid": training["pid"]})
    access = install_guard("RUN_INFERENCE", source)
    started = time.monotonic()
    index = pd.read_csv(source.parent / "test_index.csv", usecols=canonical.KEYS)
    sample = pd.read_csv(source.parent / "sample_submission.csv", usecols=canonical.KEYS)
    observations = pd.read_csv(source)
    observations.time = pd.to_datetime(observations.time, utc=True)
    frame, _ = research.public_frame(observations)
    keys = canonical.canonical_keys(frame)
    assert not keys.duplicated().any()
    frame.index = keys
    wanted = canonical.canonical_keys(sample)
    assert wanted.isin(frame.index).all()
    query = frame.loc[wanted].reset_index(drop=True)
    data = research.arrays(query)
    per_seed = []
    for model in training["model_files"]:
        path = MODELS / model["file"]
        assert research.file_hash(path) == model["sha256"]
        network = research.make_model("v23_blockmask", data[2].shape[1])
        network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        first, second = research.predict_model(network, *data), research.predict_model(network, *data)
        assert np.array_equal(first, second)
        per_seed.append(query.baseline.to_numpy(float) + first * research.compute_profile_scale(query))
    prediction = np.mean(per_seed, axis=0)
    output = sample[canonical.KEYS].copy()
    output["temp"] = prediction
    checks = canonical.validate_output(output, sample, index, 26061)
    path = ANSWERS / "submission_p2_clean_C3.csv"
    output.to_csv(path, index=False, float_format="%.12g")
    reread = pd.read_csv(path)
    checks.update({"serialized_" + key: value for key, value in canonical.validate_output(reread, sample, index, 26061).items()})
    error = float(np.max(np.abs(reread.temp.to_numpy(float) - prediction)))
    assert error < 1e-8
    checksum = research.file_hash(path)
    assert research.file_hash(source) == cfg["source_sha256"]
    result = {"experiment_id": ID, "status": "EMPTY_MODEL_TRAIN_THEN_FRESH_INFERENCE_PASS", "baseline_empty_model_regeneration": "PASS", "pid": os.getpid(), "training_pid": training["pid"], "rows": 26061, "output": str(path.relative_to(ROOT)), "answer_sha256": checksum, "expected_C_sha256": cfg["expected_C_answer_sha256"], "exact_existing_C_answer_sha256_match": checksum == cfg["expected_C_answer_sha256"], "old_C_values_read": 0, "checks": checks, "serialization_max_abs_error": error, "access": access, "source_original_unchanged": True, "new_full_fits": 3, "runtime_inference_seconds": time.monotonic() - started, "runtime_total_seconds": training["runtime_seconds"] + time.monotonic() - started, "upload": 0, "score_used_for_coefficients": False, "training_recipe_hashes": fingerprints(), "final_submission_package_status": "LOCAL_REGENERATION_EVIDENCE_NOT_COMPLETE_FINAL_PACKAGE"}
    save(REPORT / "result.json", result)
    print(json.dumps({"status": result["status"], "exact_C_sha_match": result["exact_existing_C_answer_sha256_match"], "rows": result["rows"]}))


def lineage_source_contract():
    train_source = inspect.getsource(canonical.train)
    model_source = inspect.getsource(research.fit_model)
    assert "torch.load" not in train_source + model_source
    assert "np.load" not in train_source + model_source
    assert "model = make_model(" in model_source
    assert all(term not in train_source for term in ("router_anchor", "bin17_anchor", "gi_spike2_patch", "alpha_axis"))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "RUN_TRAINING", "RUN_INFERENCE"))
    args = parser.parse_args()
    lineage_source_contract()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.mode == "seal":
            save(SEAL, {"experiment_id": ID, "hashes": fingerprints(), "empty_model_from_scratch_required": True})
            print("SEALED")
        elif args.mode == "RUN_TRAINING":
            run_training()
        else:
            run_inference()
