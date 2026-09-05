"""One fixed full-data copula fit using regenerated C; local answers only."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_profile_copula_residual_20260905_v4 as profile  # noqa: E402
import run_p2_score_repair_deploy_20260905_v1 as canonical  # noqa: E402

np, pd, torch = profile.np, profile.pd, profile.torch
base, previous = profile.base, profile.previous
ID = "p2_profile_copula_deploy_20260905_v4"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
OUT, REPORT = ROOT / "artifacts" / ID, ROOT / "reports" / ID
MODELS, ANSWERS = OUT / "03_model", OUT / "05_answer"
REGEN = ROOT / "artifacts/p2_clean_regeneration_20260905_v6"
REGEN_REPORT = ROOT / "reports/p2_clean_regeneration_20260905_v6"
SEAL = REPORT / "preregistration-seal.json"
save = profile.save


def config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["experiment_id"] == ID
    assert cfg["correction_strength"] == 1.0 and cfg["new_full_copula_fits"] == 1
    assert cfg["new_backbone_fits"] == 0 and not cfg["gpu_used"]
    assert cfg["seeds"] == profile.read_config()["seeds"]
    return cfg


def prerequisites():
    cfg = config()
    regenerated = json.loads((REGEN_REPORT / "result.json").read_text(encoding="utf-8"))
    assert regenerated["status"] == "EMPTY_MODEL_TRAIN_THEN_FRESH_INFERENCE_PASS"
    assert regenerated["exact_existing_C_answer_sha256_match"]
    assert regenerated["answer_sha256"] == cfg["baseline_expected_sha256"]
    qa = json.loads((REGEN_REPORT / "independent-qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "PASS"
    history = json.loads((profile.REPORT / "result.json").read_text(encoding="utf-8"))
    assert history["selected_predefined_policy"] == "full"
    assert json.loads((profile.REPORT / "independent-qa.json").read_text(encoding="utf-8"))["status"] == "PASS"
    train = json.loads((REGEN_REPORT / "training-result.json").read_text(encoding="utf-8"))
    assert train["empty_model_directory_before_training"] and train["new_full_fits"] == 3
    for receipt in train["model_files"]:
        assert base.file_hash(REGEN / "03_model" / receipt["file"]) == receipt["sha256"]
    return train["model_files"]


def fingerprints():
    files = ["scripts/run_p2_profile_copula_residual_20260905_v4.py", "configs/experiments/p2_profile_copula_residual_20260905_v4.json", "scripts/run_p2_score_repair_deploy_20260905_v1.py", "scripts/run_p2_objective_alignment_20260905_v2.py", *previous.DEPENDENCIES, "reports/p2_clean_regeneration_20260905_v6/result.json", "reports/p2_clean_regeneration_20260905_v6/independent-qa.json", "reports/p2_profile_copula_residual_20260905_v4/result.json", "reports/p2_profile_copula_residual_20260905_v4/independent-qa.json"]
    return {"runner": base.file_hash(Path(__file__)), "config": base.file_hash(CONFIG), "dependencies": {p: base.file_hash(ROOT / p) for p in files}, "regenerated_C_models": prerequisites()}


def allowed_path(path, stage, writing, source):
    path = Path(path).resolve()
    if "external_data" in path.parts or "hidden" in path.name.lower():
        return False
    if path.suffix.lower() == ".csv":
        if path == source:
            return not writing
        if stage != "RUN_TRAINING" and path in {source.parent / "test_index.csv", source.parent / "sample_submission.csv"}:
            return not writing
        return stage != "RUN_TRAINING" and ANSWERS in path.parents
    if path.suffix.lower() == ".pt":
        return MODELS in path.parents or (not writing and REGEN / "03_model" in path.parents)
    if path.suffix.lower() == ".npz":
        return MODELS in path.parents
    if path.suffix.lower() in {".npy", ".parquet", ".pkl", ".pickle"}:
        return False
    return True


def guard(stage):
    source = Path(os.environ["P2_DATA_DIR"]).resolve() / "observations.csv"
    counts = {"released_source_reads": 0, "official_index_key_rows": 0, "official_sample_key_rows": 0, "sample_value_rows": 0, "old_answer_values": 0, "old_OOF_reads": 0}

    def audit(event, args):
        if event == "socket.connect":
            raise PermissionError("network forbidden")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        writing = isinstance(args[1], str) and any(c in args[1] for c in "wax+")
        if not allowed_path(args[0], stage, writing, source):
            raise PermissionError("copula deployment input allowlist denied")
    sys.addaudithook(audit)
    reader = pd.read_csv

    def read_csv(path, *args, **kwargs):
        path = Path(path).resolve()
        if path == source:
            counts["released_source_reads"] += 1
        elif path in {source.parent / "test_index.csv", source.parent / "sample_submission.csv"}:
            assert stage != "RUN_TRAINING" and kwargs.get("usecols") == canonical.KEYS
            result = reader(path, *args, **kwargs)
            counts["official_index_key_rows" if path.name == "test_index.csv" else "official_sample_key_rows"] += len(result)
            return result
        elif ANSWERS not in path.parents:
            raise PermissionError("CSV outside released source/keys/new answers")
        return reader(path, *args, **kwargs)
    pd.read_csv = read_csv
    return source, counts


def cmean(frame, receipts):
    values = []
    for receipt in receipts:
        path = MODELS / receipt["file"]
        assert base.file_hash(path) == receipt["sha256"]
        network = base.make_model("v23_blockmask", 11)
        network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        values.append(previous.predict_absolute(network.eval(), frame))
    assert len(values) == 3
    return np.mean(values, axis=0)


def train():
    cfg = config()
    hashes = fingerprints()
    assert hashes == json.loads(SEAL.read_text(encoding="utf-8"))["hashes"]
    if OUT.exists():
        raise RuntimeError("exactly-once output exists")
    MODELS.mkdir(parents=True)
    save(OUT / "TRAIN_ATTEMPT_LOCK.json", {"pid": os.getpid(), "new_copula_fit_cap": 1, "new_backbone_fit_cap": 0})
    source, counts = guard("RUN_TRAINING")
    started = time.monotonic()
    receipts = hashes["regenerated_C_models"]
    for receipt in receipts:
        shutil.copyfile(REGEN / "03_model" / receipt["file"], MODELS / receipt["file"])
    frame, truth = previous.load_data(previous.load_config())
    control = cmean(frame, receipts)
    x = profile.physical_features(frame, control)
    fit_started = time.monotonic()
    model = profile.fit_copula(x, truth - control)
    fit_seconds = time.monotonic() - fit_started
    np.savez_compressed(MODELS / "copula_full.npz", **model)
    assert np.array_equal(profile.predict_copula(model, x[:128]), profile.predict_copula(dict(np.load(MODELS / "copula_full.npz", allow_pickle=False)), x[:128]))
    assert base.file_hash(source) == cfg["source_sha256"]
    result = {"experiment_id": ID, "status": "FULL_COPULA_TRAINING_PASS", "pid": os.getpid(), "new_full_copula_fits": 1, "new_backbone_fits": 0, "regenerated_C_models": receipts, "train_rows": len(truth), "copula_sha256": base.file_hash(MODELS / "copula_full.npz"), "source_sha256_before_after": cfg["source_sha256"], "access": counts, "covariance_shrinkage_train_only": float(model["shrinkage"]), "fit_seconds": fit_seconds, "runtime_seconds": time.monotonic() - started, "official_access_rows": 0, "csv_written": 0, "upload": 0, "coefficient": 1.0, "model_reload_128_train_rows": "PASS_not_full_official_replay"}
    save(REPORT / "training-result.json", result)
    print(json.dumps({"status": result["status"], "train_rows": len(truth), "new_fits": 1, "runtime_seconds": result["runtime_seconds"]}))


def infer(replay=False):
    cfg = config()
    assert fingerprints() == json.loads(SEAL.read_text(encoding="utf-8"))["hashes"]
    trained = json.loads((REPORT / "training-result.json").read_text(encoding="utf-8"))
    assert trained["pid"] != os.getpid()
    name = "REPLAY" if replay else "INFERENCE"
    save(OUT / f"{name}_ATTEMPT_LOCK.json", {"pid": os.getpid(), "training_pid": trained["pid"]})
    source, counts = guard(name)
    started = time.monotonic()
    index = pd.read_csv(source.parent / "test_index.csv", usecols=canonical.KEYS)
    sample = pd.read_csv(source.parent / "sample_submission.csv", usecols=canonical.KEYS)
    observations = pd.read_csv(source)
    observations.time = pd.to_datetime(observations.time, utc=True)
    frame, _ = base.public_frame(observations)
    frame.index = canonical.canonical_keys(frame)
    assert not frame.index.duplicated().any()
    wanted = canonical.canonical_keys(sample)
    assert wanted.isin(frame.index).all()
    query = frame.loc[wanted].reset_index(drop=True)
    assert np.isfinite(query.baseline).all() and query.public_temp_count.ge(2).all()
    control = cmean(query, trained["regenerated_C_models"])
    model_path = MODELS / "copula_full.npz"
    assert base.file_hash(model_path) == trained["copula_sha256"]
    correction = profile.predict_copula(dict(np.load(model_path, allow_pickle=False)), profile.physical_features(query, control))
    prediction = control + correction
    ANSWERS.mkdir(exist_ok=True)
    output = sample[canonical.KEYS].copy()
    output["temp"] = prediction
    checks = canonical.validate_output(output, sample, index, 26061)
    output_path = ANSWERS / ("replay_p2_profile_copula.csv" if replay else "submission_p2_profile_copula.csv")
    output.to_csv(output_path, index=False, float_format="%.12g")
    serialized = pd.read_csv(output_path)
    checks.update({"serialized_" + key: value for key, value in canonical.validate_output(serialized, sample, index, 26061).items()})
    serialization_error = float(np.max(np.abs(serialized.temp.to_numpy(float) - prediction)))
    assert serialization_error < 1e-8
    control_output = sample[canonical.KEYS].copy()
    control_output["temp"] = control
    control_path = ANSWERS / ("replay_C3.csv" if replay else "control_C3.csv")
    control_output.to_csv(control_path, index=False, float_format="%.12g")
    assert base.file_hash(control_path) == cfg["baseline_expected_sha256"]
    checksum = base.file_hash(output_path)
    assert checksum != cfg["baseline_expected_sha256"]
    assert base.file_hash(source) == cfg["source_sha256"]
    if replay:
        original = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
        assert os.getpid() != original["pid"] and checksum == original["candidate_sha256"]
    result = {"experiment_id": ID, "status": "FRESH_PROCESS_FULL_CSV_REPLAY_PASS" if replay else "INFORMATION_VALUE_CANDIDATE_LOCAL_READY_NOT_UPLOADED", "pid": os.getpid(), "training_pid": trained["pid"], "candidate": str(output_path.relative_to(ROOT)), "candidate_sha256": checksum, "control_sha256": base.file_hash(control_path), "control_exact_C_match": True, "rows": 26061, "checks": checks, "changed_rows_vs_regenerated_C": int(np.count_nonzero(prediction != control)), "correction_max_absolute_C": float(np.max(np.abs(correction))), "correction_mean_C": float(np.mean(correction)), "serialization_max_abs_error": serialization_error, "access": counts, "source_original_unchanged": True, "runtime_seconds": time.monotonic() - started, "new_fits": 0, "csv_written": 2, "upload": 0, "policy": "fixed_full_1.0_no_new_tuning", "baseline_empty_model_training": "PASS_from_separate_v6_regeneration", "historical_risk": "autumn_primary_improved_but_artificial_autumn_7d_and_14d_worse", "expected_official_score": None, "final_package_status": "LOCAL_CANDIDATE_REPRODUCTION_ONLY"}
    save(REPORT / ("replay.json" if replay else "result.json"), result)
    print(json.dumps({"status": result["status"], "candidate_sha256": checksum, "changed_rows": result["changed_rows_vs_regenerated_C"], "rows": 26061}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "RUN_TRAINING", "RUN_INFERENCE", "RUN_REPLAY"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    with profile.threadpool_limits(limits=2):
        if args.mode == "seal":
            save(SEAL, {"experiment_id": ID, "hashes": fingerprints(), "new_copula_full_fit_cap": 1, "frozen_correction_strength": 1.0})
            print("SEALED")
        elif args.mode == "RUN_TRAINING":
            train()
        else:
            infer(args.mode == "RUN_REPLAY")
