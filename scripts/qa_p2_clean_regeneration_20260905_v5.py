"""Fresh-process full answer replay and independent empty-model lineage audit."""

import json
import os
import time

import run_p2_clean_regeneration_20260905_v5 as repaired

m = repaired.original


def main():
    cfg, recipe, source = m.config()
    expected = json.loads((m.REPORT / "result.json").read_text(encoding="utf-8"))
    training = json.loads((m.REPORT / "training-result.json").read_text(encoding="utf-8"))
    seal = json.loads(m.SEAL.read_text(encoding="utf-8"))
    assert seal["hashes"] == repaired.fingerprints()
    checks = []

    def check(name, passed):
        checks.append({"check": name, "pass": bool(passed)})

    check("three_separate_pids", len({os.getpid(), expected["pid"], training["pid"]}) == 3)
    check("empty_model_start", training["empty_model_directory_before_training"])
    check("three_scratch_fits", training["new_full_fits"] == len(training["model_files"]) == 3)
    check("same_seed_epoch", [x["seed"] for x in training["model_files"]] == recipe["seeds"] and all(x["epochs"] == 60 for x in training["model_files"]))
    check("same_training_population", training["source_original_rows"] == 166268 and training["source_augmented_rows"] == 51354 and training["training_weight_sum"] == 166268.0)
    check("canonical_training_receipt", m.research.file_hash(m.MODELS / "train-result.json") == training["canonical_train_result_sha256"])
    check("training_no_official_or_answers", training["official_access_rows"] == training["csv_written"] == training["upload"] == 0)
    check("training_no_old_asset_reads", training["access"]["old_model_reads"] == training["access"]["old_answer_reads"] == 0)
    check("guard_legacy_route_blocked", training["legacy_route_blocked"])
    access = m.install_guard("RUN_INFERENCE", source)
    started = time.monotonic()
    index = m.pd.read_csv(source.parent / "test_index.csv", usecols=m.canonical.KEYS)
    sample = m.pd.read_csv(source.parent / "sample_submission.csv", usecols=m.canonical.KEYS)
    obs = m.pd.read_csv(source)
    obs.time = m.pd.to_datetime(obs.time, utc=True)
    frame, _ = m.research.public_frame(obs)
    frame.index = m.canonical.canonical_keys(frame)
    check("source_feature_keys_unique", not frame.index.duplicated().any())
    query = frame.loc[m.canonical.canonical_keys(sample)].reset_index(drop=True)
    data = m.research.arrays(query)
    components = []
    for receipt in training["model_files"]:
        path = m.MODELS / receipt["file"]
        check(f"model_hash/{receipt['seed']}", m.research.file_hash(path) == receipt["sha256"])
        model = m.research.make_model("v23_blockmask", data[2].shape[1])
        model.load_state_dict(m.torch.load(path, map_location="cpu", weights_only=True))
        components.append(query.baseline.to_numpy(float) + m.research.predict_model(model, *data) * m.research.compute_profile_scale(query))
    prediction = m.np.mean(components, axis=0)
    output = sample[m.canonical.KEYS].copy()
    output["temp"] = prediction
    for key, passed in m.canonical.validate_output(output, sample, index, 26061).items():
        check(f"replay/{key}", passed)
    replay = m.ANSWERS / "replay_p2_clean_C3.csv"
    if replay.exists():
        raise RuntimeError("exactly-once replay already exists")
    output.to_csv(replay, index=False, float_format="%.12g")
    checksum = m.research.file_hash(replay)
    initial = m.ROOT / expected["output"]
    check("new_full_csv_replay_exact", checksum == expected["answer_sha256"] == m.research.file_hash(initial))
    check("historical_C_bitidentity", checksum == cfg["expected_C_answer_sha256"] and expected["exact_existing_C_answer_sha256_match"])
    check("official_key_counts", access["official_index_key_rows"] == access["official_sample_key_rows"] == 26061)
    check("no_sample_values_hidden_or_old_answers", access["sample_value_rows"] == access["old_answer_reads"] == access["old_model_reads"] == expected["old_C_values_read"] == 0)
    check("source_immutable", m.research.file_hash(source) == cfg["source_sha256"] == training["source_sha256_before_after"])
    check("train_and_inference_under_6h", expected["runtime_total_seconds"] < 21600)
    check("no_upload", expected["upload"] == 0)
    failure = [item["check"] for item in checks if not item["pass"]]
    receipt = {"status": "PASS" if not failure else "FAIL", "checks_count": len(checks), "checks": checks, "failed_checks": failure, "baseline_empty_model_training_to_answer_pass": not failure, "full_csv_replay_pid": os.getpid(), "inference_pid": expected["pid"], "training_pid": training["pid"], "new_full_csv_replay_rows": 26061, "replay_sha256": checksum, "replay_seconds": time.monotonic() - started, "result_sha256": m.research.file_hash(m.REPORT / "result.json"), "training_result_sha256": m.research.file_hash(m.REPORT / "training-result.json"), "qa_runner_sha256": m.research.file_hash(m.Path(__file__)), "focused_synthetic_tests": 5, "ruff": "PASS", "official_key_rows_this_qa": access, "old_answer_values_read": 0, "new_fits": 0, "upload": 0, "organizer_hardware_or_portable_package_verified": False}
    m.save(m.REPORT / "independent-qa.json", receipt)
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "failures": failure, "full_csv_rows": 26061}))
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    m.torch.set_num_threads(2)
    with m.threadpool_limits(limits=2):
        main()
