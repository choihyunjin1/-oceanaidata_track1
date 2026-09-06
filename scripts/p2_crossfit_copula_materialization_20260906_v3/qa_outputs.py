"""Independent source-hash/key/SHA/receipt verification; never loads a model."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parents[2] / "reports/p2_crossfit_copula_materialization_20260906_v3"
KEYS = ["station", "layer", "time"]
FALLBACK_SHA = "46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name, data):
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / name).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def input_hashes():
    source = Path(os.environ["P2_DATA_DIR"]).resolve()
    return {name: sha(source / name) for name in ("observations.csv", "test_index.csv", "sample_submission.csv")}


def ordered_keys(frame):
    return list(zip(frame.station.astype(str), frame.layer.astype(int),
                    pd.to_datetime(frame.time, utc=True).astype(str), strict=True))


def before(package):
    assert read(package / "04_logs/training-qa.json")["status"] == "PASS"
    save("official-input-hash-before.json", {"status": "SEALED_AFTER_TRAINING_QA_BEFORE_OFFICIAL_INFERENCE",
         "input_sha256": input_hashes(), "sample_prediction_columns_loaded": 0, "hidden_rows": 0,
         "new_fits": 0, "uploads": 0})


def verify(package, extracted, archive, original):
    cfg = read(package / "config.json")
    manifest = read(package / "06_docs/BUILD_MANIFEST.json")
    train, trainqa, infer, replay = [read(package / "04_logs" / name) for name in
        ("training-result.json", "training-qa.json", "inference-qa.json", "replay-qa.json")]
    zipreplay = read(extracted / "04_logs/zip-extraction-replay.json")
    archive_manifest = read(archive.with_suffix(".manifest.json"))
    checks = {}

    def check(name, result):
        checks[name] = bool(result)

    check("original_execution_outside_repo", not original.is_relative_to(REPORT.parents[1]) and original != package)
    for path in original.rglob("*"):
        if path.is_file():
            name = path.relative_to(original)
            check(f"preservation_copy/{name.as_posix()}", (package / name).is_file() and sha(path) == sha(package / name))
    check("source_inputs_unchanged", input_hashes() == read(REPORT / "official-input-hash-before.json")["input_sha256"])
    check("source_training_sha", cfg["source_sha256"] == input_hashes()["observations.csv"])
    check("empty_scratch_training", train["status"] == "TRAINING_COMPLETE_FROM_EMPTY_MODEL" and train["empty_model_verified"])
    check("no_old_model_answer", train["old_model_reads"] == train["old_answer_reads"] == 0)
    check("no_train_official", train["official_rows"] == train["csv_rows"] == train["uploads"] == 0)
    check("fixed_three_plus_one_fits", len(train["fits"]) == cfg["backbone_fits"] == 3 and len(train["copula_fits"]) == cfg["copula_fits"] == 1)
    check("fixed_selected_arm", cfg["selected_arms"] == ["insample_full"] and cfg["correction_strength"] == 1)
    check("three_seeds_cpu60", [fit["seed"] for fit in train["fits"]] == cfg["seeds"] == [20260901, 20260902, 20260903]
          and all(fit["device"] == "cpu" and fit["epochs"] == 60 for fit in train["fits"]))
    check("runtime_cap", train["runtime_seconds"] <= cfg["training_cap_seconds"] == 3600)
    check("166268_training", train["training_rows"] == trainqa["rows"] == 166268)
    check("training_qa_link", trainqa["status"] == "PASS" and trainqa["training_result_sha256"] == sha(package / "04_logs/training-result.json"))
    check("new_processes_distinct", len({r["pid"] for r in (train, trainqa, infer, replay, zipreplay)}) == 5)
    check("inference_replay_pass", infer["status"] == replay["status"] == zipreplay["status"] == "PASS")
    check("no_hidden_values_or_upload", all(r["hidden_rows"] == r["sample_prediction_reads"] == r["uploads"] == 0 for r in (infer, replay, zipreplay)))
    check("archive_sha", sha(archive) == archive_manifest["archive_sha256"])
    check("archive_exclusion", archive_manifest["source_data_files"] == archive_manifest["attempt_locks"] == archive_manifest["calibration_or_replay_arrays"] == 0)
    for name, digest in manifest["files_sha256"].items():
        check(f"seal/{name}", sha(package / name) == sha(extracted / name) == digest == train["fingerprint"][name])
    for name, digest in train["model_hashes"].items():
        check(f"model/{name}", sha(package / "03_model" / name) == sha(extracted / "03_model" / name) == digest)
    for name, digest in archive_manifest["files_sha256"].items():
        check(f"archive/{name}", sha(package / name) == sha(extracted / name) == digest)
    source = Path(os.environ["P2_DATA_DIR"])
    sample = pd.read_csv(source / "sample_submission.csv", usecols=KEYS)
    index = pd.read_csv(source / "test_index.csv", usecols=KEYS)
    sk, ik = ordered_keys(sample), ordered_keys(index)
    check("26061_public_key_contract", len(sk) == len(ik) == len(set(sk)) == len(set(ik)) == 26061 and set(sk) == set(ik))
    answers, outputs = {}, {}
    for arm, item in infer["outputs"].items():
        path = package / item["file"]
        frame = pd.read_csv(path)
        check(f"{arm}/schema_keys_order", list(frame.columns) == [*KEYS, "temp"] and ordered_keys(frame) == sk)
        check(f"{arm}/finite", np.isfinite(frame.temp.to_numpy(float)).all())
        check(f"{arm}/sha_replay", sha(path) == item["sha256"] == replay["outputs"][arm]["sha256"] == zipreplay["outputs"][arm]["sha256"])
        answers[arm] = frame.temp.to_numpy(float)
        outputs[arm] = {**item, "matches_preserved_cuda_fallback_sha": item["sha256"] == FALLBACK_SHA}
    change = answers["insample_full"] - answers["C3_control"]
    failed = [name for name, value in checks.items() if not value]
    payload = {"status": "FAIL" if failed else "PASS", "checks_count": len(checks), "checks": checks,
               "failures": failed, "outputs": outputs, "input_sha256": input_hashes(),
               "candidate_minus_same_run_control": {"changed_rows": int(np.count_nonzero(change)),
                    "prediction_delta_rmse_celsius": float(np.sqrt(np.mean(change ** 2))),
                    "meaning": "prediction difference only, not true target RMSE or estimated official improvement"},
               "process_ids": {name: result["pid"] for name, result in zip(
                   ("training", "training_qa", "inference", "replay", "zip_extraction_replay"),
                   (train, trainqa, infer, replay, zipreplay), strict=True)},
               "training_seconds": train["runtime_seconds"], "new_fits": 0, "uploads": 0,
               "qa_runner_sha256": sha(Path(__file__)), "full_training_runs": 1,
               "scope": "full-source scratch and independent saved-model replay; ZIP extraction is not a second full retraining"}
    save("independent-qa.json", payload)
    print(json.dumps({"status": payload["status"], "checks": len(checks), "failures": failed, "outputs": outputs}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("before", "verify"))
    parser.add_argument("package", type=Path)
    parser.add_argument("--extracted", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--original", type=Path)
    args = parser.parse_args()
    if args.mode == "before":
        before(args.package.resolve())
    else:
        verify(args.package.resolve(), args.extracted.resolve(), args.archive.resolve(), args.original.resolve())
