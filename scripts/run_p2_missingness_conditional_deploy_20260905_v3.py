"""Frozen C/R information-only candidate adapter. Explicit root authorization required."""

import argparse
import hashlib
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
import run_p2_missingness_conditional_3seed_20260905_v3 as train  # noqa: E402
import run_p2_score_repair_deploy_20260905_v1 as control  # noqa: E402

ID = "p2_missingness_conditional_deploy_20260905_v3"
REPORT, OUT = ROOT / "reports" / ID, ROOT / "artifacts" / ID
SEAL = REPORT / "preregistration-seal.json"
CANDIDATE = OUT / "submission_p2_missingness_conditional_3seed.csv"


def validate_route(c, r, trigger, prediction):
    if not (c.shape == r.shape == trigger.shape == prediction.shape):
        raise ValueError("row alignment")
    if not np.isfinite(np.stack((c, r, prediction))).all():
        raise ValueError("finite model output required")
    if not np.array_equal(prediction, np.where(trigger, r, c)):
        raise ValueError("frozen conditional route violated")
    return {"route_exact": True, "nontrigger_exact_C": bool(np.array_equal(prediction[~trigger], c[~trigger])), "trigger_rows": int(trigger.sum()), "changed_rows": int((prediction != c).sum())}


def fingerprints():
    files = [train.REPORT / name for name in ("result.json", "independent-qa.json", "fresh-process-replay.json", "preregistration-seal.json")]
    files += [train.OUT / "fit-receipts.json", control.ARTIFACT / "predict-result.json"]
    return {"runner": train.base.file_hash(Path(__file__)), "training": train.fingerprints(), "files": {str(p.relative_to(ROOT)): train.base.file_hash(p) for p in files}}


def manifest():
    seal = json.loads(train.SEAL.read_text(encoding="utf-8"))
    fits = json.loads((train.OUT / "fit-receipts.json").read_text(encoding="utf-8"))
    return [r for r in seal["hashes"]["models"] if r["stage"] == "full"] + [r for r in fits if r["stage"] == "full"]


def install_guard(models):
    source = Path(os.environ["P2_DATA_DIR"]).resolve()
    allow_csv = {source / "observations.csv", source / "test_index.csv", source / "sample_submission.csv", CANDIDATE}
    allow_models = {(ROOT / r["path"]).resolve() for r in models}
    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network forbidden")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("external/hidden forbidden")
        if path.suffix.lower() == ".csv" and path not in allow_csv:
            raise PermissionError("unapproved CSV")
        if source in path.parents and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
            raise PermissionError("source immutable")
        if path.suffix.lower() == ".pt" and path not in allow_models:
            raise PermissionError("unapproved model")
        if path.suffix.lower() == ".npz" and OUT not in path.parents:
            raise PermissionError("unapproved prediction input")
    sys.addaudithook(guard)


def infer(models):
    source = Path(os.environ["P2_DATA_DIR"]).resolve()
    assert train.base.file_hash(source / "observations.csv") == train.previous.load_config()["source_sha256"]
    sample = pd.read_csv(source / "sample_submission.csv", usecols=control.KEYS)
    index = pd.read_csv(source / "test_index.csv", usecols=control.KEYS)
    observations = pd.read_csv(source / "observations.csv")
    observations.time = pd.to_datetime(observations.time, utc=True)
    frame, _ = train.base.public_frame(observations)
    frame.index = control.canonical_keys(frame)
    wanted = control.canonical_keys(sample)
    assert not frame.index.duplicated().any() and wanted.isin(frame.index).all()
    query = frame.loc[wanted].reset_index(drop=True)
    assert np.isfinite(query.baseline).all() and query.public_temp_count.ge(2).all()
    payload = {}
    for item in models:
        assert train.base.file_hash(ROOT / item["path"]) == item["sha256"]
        payload[f"{item['arm']}_{item['seed']}"] = train.previous.predict_absolute(train.load_model(ROOT / item["path"]), query)
    seeds = train.read_config()["seeds"]
    c = train.component_mean([payload[f"C_{s}"] for s in seeds])
    r = train.component_mean([payload[f"R_{s}"] for s in seeds])
    trigger = train.prior.route_trigger(query)
    prediction = train.prior.conditional(c, r, trigger)
    checks = validate_route(c, r, trigger, prediction)
    baseline = sample.loc[:, control.KEYS].copy()
    baseline["temp"] = c
    regenerated_control_sha = hashlib.sha256(baseline.to_csv(index=False, float_format="%.12g").encode("utf-8")).hexdigest()
    old_receipt = json.loads((control.ARTIFACT / "predict-result.json").read_text(encoding="utf-8"))
    assert regenerated_control_sha == old_receipt["candidate_sha256"]
    output = sample.loc[:, control.KEYS].copy()
    output["temp"] = prediction
    checks.update(control.validate_output(output, sample, index, 26061))
    payload.update(C=c, R=r, trigger=trigger, prediction=prediction, key=wanted.to_numpy(str))
    return output, sample, index, payload, checks, regenerated_control_sha


def execute(replay=False):
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    assert sealed["hashes"] == fingerprints()
    models = manifest()
    assert len(models) == 6
    install_guard(models)
    torch.set_num_threads(1)
    started = time.monotonic()
    if not replay:
        if OUT.exists():
            raise RuntimeError("materialization already attempted")
        OUT.mkdir(parents=True)
        train.save(OUT / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "root_authorized": True, "seal": sealed})
    else:
        first = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
        assert first["pid"] != os.getpid()
    output, sample, index, payload, checks, control_sha = infer(models)
    encoded = output.to_csv(index=False, float_format="%.12g").encode("utf-8")
    expected_hash = hashlib.sha256(encoded).hexdigest()
    assert expected_hash != control_sha
    if replay:
        assert expected_hash == first["candidate_sha256"] == train.base.file_hash(CANDIDATE)
        saved = np.load(OUT / "official_predictions_private.npz", allow_pickle=False)
        assert all(np.array_equal(saved[key], value) for key, value in payload.items())
        receipt = {"status": "PASS", "pid": os.getpid(), "materialization_pid": first["pid"], "full_models": 6, "rows": 26061, "all_component_arrays_exact": True, "whole_csv_bytes_sha_exact": True, "candidate_sha256": expected_hash, "checks": checks, "new_fits": 0, "csv_written": 0, "upload": 0, "sample_value_rows_read": 0, "hidden_truth_rows_read": 0, "runtime_seconds": time.monotonic()-started}
        train.save(REPORT / "fresh-process-replay.json", receipt)
    else:
        output.to_csv(CANDIDATE, index=False, float_format="%.12g")
        assert train.base.file_hash(CANDIDATE) == expected_hash
        reread = pd.read_csv(CANDIDATE)
        checks.update({"serialized_" + key: value for key, value in control.validate_output(reread, sample, index, 26061).items()})
        assert np.max(np.abs(reread.temp.to_numpy() - payload["prediction"])) < 1e-8
        np.savez_compressed(OUT / "official_predictions_private.npz", **payload)
        receipt = {"experiment_id": ID, "status": "INFORMATION_ONLY_CANDIDATE_PENDING_REPLAY", "pid": os.getpid(), "candidate": str(CANDIDATE.relative_to(ROOT)).replace("\\", "/"), "candidate_sha256": expected_hash, "control_csv_regenerated_exact_sha256": control_sha, "control_values_not_used_for_fit_or_rule": True, "rows": 26061, "columns": control.OUTPUT_COLUMNS, "checks": checks, "internal_result_sha256": train.base.file_hash(train.REPORT / "result.json"), "internal_qa_sha256": train.base.file_hash(train.REPORT / "independent-qa.json"), "models": models, "runtime_seconds": time.monotonic()-started, "official_key_rows_read": {"sample": 26061, "index": 26061}, "sample_value_rows_read": 0, "hidden_truth_rows_read": 0, "official_score_rows_read": 0, "old_answer_inputs": 0, "new_fits": 0, "csv_written": 1, "upload": 0, "title": "P2 clean missingness conditional 3seed", "one_line_summary": "배포 관측 scratch DeepSets C/R 각 3seed 평균; 공개 T5/S5 결측일 때만 R, 나머지는 오늘 clean C와 동일. 내부 가을 근거 및 계절 위험을 분리한 정보가치 후보."}
        train.save(REPORT / "result.json", receipt)
    print(json.dumps({"status": receipt["status"], "rows": 26061, "candidate_sha256": expected_hash, "trigger_rows": checks["trigger_rows"], "changed_rows": checks["changed_rows"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seal", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--replay", action="store_true")
    parser.add_argument("--root-authorized", action="store_true")
    args = parser.parse_args()
    if not args.root_authorized:
        raise RuntimeError("root must separately authorize official inputs and materialization")
    with threadpool_limits(limits=1):
        if args.seal:
            result = json.loads((train.REPORT / "result.json").read_text(encoding="utf-8"))
            qa = json.loads((train.REPORT / "independent-qa.json").read_text(encoding="utf-8"))
            replay = json.loads((train.REPORT / "fresh-process-replay.json").read_text(encoding="utf-8"))
            assert result["decision"] == "INFO_ONLY_CANDIDATE" and qa["status"] == replay["status"] == "PASS"
            train.save(SEAL, {"experiment_id": ID, "root_authorized": True, "hashes": fingerprints(), "new_fits": 0, "maximum_csv": 1, "postprocess": "none", "rule": train.read_config()["rule"], "models": manifest()})
            print(json.dumps({"status": "SEALED", "runner_sha256": train.base.file_hash(Path(__file__))}))
        else:
            execute(args.replay)
