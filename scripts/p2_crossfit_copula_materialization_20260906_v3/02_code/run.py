"""CPU-only source-to-model candidate package, no research-repository imports."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import copula  # noqa: E402
import core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL, LOG, ANSWER = (ROOT / name for name in ("03_model", "04_logs", "05_answer"))
KEYS = ["station", "layer", "time"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def fingerprint():
    paths = [ROOT / "config.json", *sorted((ROOT / "02_code").glob("*.py"))]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def keys(frame):
    return (frame.station.astype(str) + "|" + frame.layer.astype(int).astype(str) + "|"
            + pd.to_datetime(frame.time, utc=True).astype(str)).to_numpy(str)


def keysha(frame):
    return hashlib.sha256("\n".join(keys(frame)).encode()).hexdigest()


def guard(mode, source):
    def hook(event, args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise PermissionError("network prohibited")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        writing = (isinstance(args[1], str) and any(c in args[1] for c in "wax+")) or (
            isinstance(args[2], int) and bool(args[2] & (os.O_WRONLY | os.O_RDWR)))
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("forbidden ancestry")
        if path.is_relative_to(source.parent):
            allowed = {source}
            if mode in {"RUN_INFERENCE", "REPLAY"}:
                allowed |= {source.parent / "test_index.csv", source.parent / "sample_submission.csv"}
            if writing or path not in allowed:
                raise PermissionError("source immutable/phase access")
        elif path.suffix.lower() in {".csv", ".pt", ".pth", ".npz", ".npy", ".parquet", ".cbm", ".joblib"}:
            if not path.is_relative_to(ROOT):
                raise PermissionError("prior model/array/answer prohibited")
            if path.suffix.lower() == ".csv" and (mode not in {"RUN_INFERENCE", "REPLAY"} or not path.is_relative_to(ANSWER)):
                raise PermissionError("CSV not authorized in this phase")
    sys.addaudithook(hook)
    if mode == "RUN_TRAINING":
        def deny_load(*_args, **_kwargs):
            raise PermissionError("scratch training cannot load any model")
        torch.load = deny_load


def public_population(source, cfg, training):
    if sha(source) != cfg["source_sha256"]:
        raise ValueError("distributed source SHA mismatch")
    observations = pd.read_csv(source)
    if observations.duplicated(KEYS).any():
        raise ValueError("duplicate source keys")
    observations.time = pd.to_datetime(observations.time, utc=True)
    target = observations.layer.isin([2, 3, 4])
    truth = observations.loc[target, "temp"].to_numpy(float).copy()
    observations.loc[target, ["temp", "psal"]] = np.nan
    frame, empty = core.public_frame(observations)
    assert np.isnan(empty).all()
    if training:
        selected = np.isfinite(truth) & np.isfinite(frame.baseline) & (frame.public_temp_count >= 2)
        frame, truth = frame.loc[selected].reset_index(drop=True), truth[selected]
        assert len(frame) == 166268 and not frame.duplicated(KEYS).any()
    return frame, truth


def inner_plan(frame):
    """Same source-only calendar-position rule with full data as the outer pool."""
    months = pd.date_range("2024-05-01", "2026-01-01", freq="MS", tz="Asia/Seoul")
    stamp = pd.to_datetime(frame.time, utc=True)
    options = []
    for start, stop in zip(months[:-1], months[1:], strict=True):
        valid = np.asarray((stamp >= start) & (stamp < stop))
        train = np.asarray((stamp < start - pd.Timedelta(days=7)) | (stamp >= stop + pd.Timedelta(days=7)))
        if valid.any():
            options.append({"start": start.isoformat(), "end": stop.isoformat(),
                            "validation_rows": int(valid.sum()), "train_rows": int(train.sum())})
    positions = [(len(options) - 1) // 3, 2 * (len(options) - 1) // 3]
    assert len(set(positions)) == 2
    return [{"id": f"I{i+1}", **options[j]} for i, j in enumerate(positions)]


def split(frame, spec):
    stamp = pd.to_datetime(frame.time, utc=True)
    start, stop = pd.Timestamp(spec["start"]), pd.Timestamp(spec["end"])
    return (np.asarray((stamp < start - pd.Timedelta(days=7)) | (stamp >= stop + pd.Timedelta(days=7))),
            np.asarray((stamp >= start) & (stamp < stop)))


def prediction(models, frame):
    values = core.arrays(frame)
    return np.mean(np.stack([frame.baseline.to_numpy(float) + core.predict_model(model, *values)
                            * core.compute_profile_scale(frame) for model in models]), axis=0)


def load_models(stage, cfg, receipt):
    result = []
    for seed in cfg["seeds"]:
        name = f"{stage}_{seed}.pt"
        path = MODEL / name
        assert sha(path) == receipt["model_hashes"][name]
        model = core.make_model("v23_blockmask", 11)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        result.append(model.eval())
    return result


def train(source, cfg):
    if any(MODEL.iterdir()) or any(ANSWER.iterdir()):
        raise FileExistsError("empty 03_model/05_answer required; never delete old files")
    save(ROOT / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "fingerprint": fingerprint()})
    before = fingerprint()
    started, fits, learned = time.monotonic(), [], {}
    deadline = started + cfg["training_cap_seconds"]

    def fit_set(frame, truth, stage):
        arrays, weighting = core.training_arrays(frame, truth, "v23_blockmask", cfg)
        models = []
        for seed in cfg["seeds"]:
            if time.monotonic() >= deadline or len(fits) >= cfg["backbone_fits"]:
                raise RuntimeError("predeclared CPU fit/time budget reached")
            def progress(epoch, seconds, current_seed=seed):
                if time.monotonic() >= deadline:
                    raise RuntimeError("CPU fit budget reached at epoch boundary")
                print(json.dumps({"stage": stage, "seed": current_seed, "epoch": epoch,
                    "new_backbones": len(fits), "fit_seconds": seconds}), flush=True)
            model, measured = core.fit_model(arrays, "v23_blockmask", seed, cfg, progress)
            name = f"{stage}_{seed}.pt"
            torch.save(model.state_dict(), MODEL / name)
            learned[name] = sha(MODEL / name)
            fits.append({"stage": stage, "seed": seed, "model": name, "model_sha256": learned[name],
                         "training_key_sha256": keysha(frame), "weighting": weighting, **measured})
            models.append(model)
        return models

    try:
        frame, truth = public_population(source, cfg, True)
        assert keysha(frame) == cfg["training_key_sha256"]
        assert inner_plan(frame) == cfg["inner_plan"]
        full = fit_set(frame, truth, "full")
        c = prediction(full, frame)
        calibration = {}
        if "insample_full" in cfg["selected_arms"]:
            calibration["insample_full"] = (np.arange(len(frame)), copula.physical_features(frame, c), truth - c)
        if "crossfit_full" in cfg["selected_arms"]:
            rows, features, residual = [], [], []
            for spec in cfg["inner_plan"]:
                selected, valid = split(frame, spec)
                models = fit_set(frame.loc[selected].reset_index(drop=True), truth[selected], spec["id"])
                query = frame.loc[valid].reset_index(drop=True)
                pc = prediction(models, query)
                rows.extend(np.flatnonzero(valid).tolist())
                features.append(copula.physical_features(query, pc))
                residual.append(truth[valid] - pc)
            calibration["crossfit_full"] = (np.asarray(rows), np.concatenate(features), np.concatenate(residual))
        replay = {"key": keys(frame), "C3": c}
        correction_receipts = []
        for arm in cfg["selected_arms"]:
            ids, x, residual = calibration[arm]
            t0 = time.monotonic()
            model = copula.fit_copula(x, residual)
            path = MODEL / f"{arm}.npz"
            np.savez_compressed(path, **model)
            learned[path.name] = sha(path)
            data_path = LOG / f"calibration_{arm}.npz"
            np.savez_compressed(data_path, row_indices=ids, physical_x=x, residual=residual)
            replay[arm] = c + copula.predict_copula(model, copula.physical_features(frame, c))
            correction_receipts.append({"arm": arm, "rows": len(ids), "model_sha256": sha(path),
                "calibration_sha256": sha(data_path), "runtime_seconds": time.monotonic() - t0,
                "training_key_sha256": keysha(frame.iloc[ids]), "correction_strength": 1.0})
        assert len(fits) == cfg["backbone_fits"] and len(correction_receipts) == cfg["copula_fits"]
        np.savez_compressed(LOG / "training_replay.npz", **replay)
        assert before == fingerprint() and sha(source) == cfg["source_sha256"]
        receipt = {"status": "TRAINING_COMPLETE_FROM_EMPTY_MODEL", "pid": os.getpid(),
            "fingerprint": before, "model_hashes": learned, "fits": fits, "copula_fits": correction_receipts,
            "training_replay_sha256": sha(LOG / "training_replay.npz"),
            "empty_model_verified": True, "source_sha256": sha(source), "training_rows": len(frame),
            "runtime_seconds": time.monotonic() - started, "gpu_used": False, "threads": 2,
            "versions": cfg["versions"], "module_origins": {"core": str(Path(core.__file__).relative_to(ROOT)), "copula": str(Path(copula.__file__).relative_to(ROOT))},
            "old_model_reads": 0, "old_answer_reads": 0, "official_rows": 0, "csv_rows": 0, "uploads": 0}
        assert receipt["runtime_seconds"] <= cfg["training_cap_seconds"]
        save(LOG / "training-result.json", receipt)
        print(json.dumps({"status": receipt["status"], "new_fits": len(fits) + len(correction_receipts), "seconds": receipt["runtime_seconds"]}), flush=True)
    except Exception as exc:
        save(LOG / "terminal-failure.json", {"status": "TERMINAL_TECHNICAL_OR_RESOURCE_FAILURE", "type": type(exc).__name__,
             "message": str(exc), "finished_backbones": len(fits), "new_models": learned, "automatic_restart": False})
        raise


def verify_training(source, cfg):
    started = time.monotonic()
    result = json.loads((LOG / "training-result.json").read_text())
    assert os.getpid() != result["pid"] and fingerprint() == result["fingerprint"]
    raw = np.load(LOG / "training_replay.npz", allow_pickle=False)
    assert sha(LOG / "training_replay.npz") == result["training_replay_sha256"]
    frame, truth = public_population(source, cfg, True)
    assert np.array_equal(keys(frame), raw["key"])
    full = load_models("full", cfg, result)
    c = prediction(full, frame)
    assert np.array_equal(c, raw["C3"])
    checks = {"full_C3_entire_training_replay_exact": True}
    for arm in cfg["selected_arms"]:
        receipt = next(r for r in result["copula_fits"] if r["arm"] == arm)
        path = MODEL / f"{arm}.npz"
        assert sha(path) == receipt["model_sha256"]
        data_path = LOG / f"calibration_{arm}.npz"
        assert sha(data_path) == receipt["calibration_sha256"]
        data, model = np.load(data_path, allow_pickle=False), dict(np.load(path, allow_pickle=False))
        if arm == "insample_full":
            indices, x, residual = np.arange(len(frame)), copula.physical_features(frame, c), truth - c
        else:
            rows, features, errors = [], [], []
            for spec in cfg["inner_plan"]:
                _, valid = split(frame, spec)
                query = frame.loc[valid].reset_index(drop=True)
                pc = prediction(load_models(spec["id"], cfg, result), query)
                rows.extend(np.flatnonzero(valid).tolist())
                features.append(copula.physical_features(query, pc))
                errors.append(truth[valid] - pc)
            indices, x, residual = np.asarray(rows), np.concatenate(features), np.concatenate(errors)
        assert np.array_equal(indices, data["row_indices"])
        assert np.array_equal(x, data["physical_x"], equal_nan=True) and np.array_equal(residual, data["residual"])
        checks.update({f"{arm}/{name}": value for name, value in copula.covariance_checks(model, x, residual).items()})
        p = c + copula.predict_copula(model, copula.physical_features(frame, c))
        assert np.array_equal(p, raw[arm])
        checks[f"{arm}/all_training_rows_exact"] = True
    for fit in result["fits"]:
        selected = np.ones(len(frame), bool) if fit["stage"] == "full" else split(frame, next(s for s in cfg["inner_plan"] if s["id"] == fit["stage"]))[0]
        assert keysha(frame.loc[selected]) == fit["training_key_sha256"]
        assert fit["seed"] in cfg["seeds"] and fit["epochs"] == 60 and fit["device"] == "cpu"
        assert np.isclose(fit["weighting"]["training_weight_sum"], int(selected.sum()), rtol=1e-5)
    assert sha(source) == cfg["source_sha256"] and fingerprint() == result["fingerprint"]
    save(LOG / "training-qa.json", {"status": "PASS", "pid": os.getpid(), "training_pid": result["pid"],
        "checks": checks, "rows": len(frame), "runtime_seconds": time.monotonic() - started,
        "training_result_sha256": sha(LOG / "training-result.json"), "new_fits": 0, "official_rows": 0,
        "scope": "source-and-inner-key/analytic-covariance/full-row-saved-model-replay; not fresh holdout performance"})
    print(json.dumps({"status": "TRAINING_QA_PASS", "checks": len(checks), "rows": len(frame)}), flush=True)


def inference(source, cfg, replay=False, replay_receipt="replay-qa.json"):
    started = time.monotonic()
    training = json.loads((LOG / "training-result.json").read_text())
    qa = json.loads((LOG / "training-qa.json").read_text())
    assert qa["status"] == "PASS" and qa["training_result_sha256"] == sha(LOG / "training-result.json")
    assert os.getpid() not in {training["pid"], qa["pid"]} and fingerprint() == training["fingerprint"]
    frame, _ = public_population(source, cfg, False)
    sample = pd.read_csv(source.parent / "sample_submission.csv", usecols=KEYS)
    index = pd.read_csv(source.parent / "test_index.csv", usecols=KEYS)
    sk, ik = keys(sample), keys(index)
    assert len(sample) == len(index) == 26061 and len(set(sk)) == len(set(ik)) == 26061 and set(sk) == set(ik)
    positions = pd.Series(np.arange(len(frame)), index=keys(frame)).reindex(sk)
    assert positions.notna().all()
    query = frame.iloc[positions.to_numpy(int)].reset_index(drop=True)
    assert np.array_equal(keys(query), sk) and np.isfinite(query.baseline).all() and (query.public_temp_count >= 2).all()
    c = prediction(load_models("full", cfg, training), query)
    predictions = {"C3_control": c}
    for arm in cfg["selected_arms"]:
        path = MODEL / f"{arm}.npz"
        assert sha(path) == training["model_hashes"][path.name]
        fitted = dict(np.load(path, allow_pickle=False))
        predictions[arm] = c + copula.predict_copula(fitted, copula.physical_features(query, c))
    if replay:
        prior = json.loads((LOG / "inference-qa.json").read_text())
        assert os.getpid() != prior["pid"]
    outputs = {}
    for arm, values in predictions.items():
        assert np.isfinite(values).all()
        answer = sample.copy()
        answer["temp"] = values
        payload = answer.to_csv(index=False, lineterminator="\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        path = ANSWER / f"P2_{arm}.csv"
        if replay:
            assert path.read_bytes() == payload and prior["outputs"][arm]["sha256"] == digest
        else:
            with path.open("xb") as stream:
                stream.write(payload)
        outputs[arm] = {"file": path.relative_to(ROOT).as_posix(), "sha256": digest, "rows": len(answer),
                        "schema_key_order_finite_unique": True}
    assert sha(source) == cfg["source_sha256"] and fingerprint() == training["fingerprint"]
    save(LOG / (replay_receipt if replay else "inference-qa.json"), {"status": "PASS", "pid": os.getpid(),
        "training_pid": training["pid"], "training_qa_pid": qa["pid"], "outputs": outputs,
        "runtime_seconds": time.monotonic() - started, "new_fits": 0, "hidden_rows": 0, "sample_prediction_reads": 0,
        "official_key_rows": 26061, "source_unchanged": sha(source) == cfg["source_sha256"], "uploads": 0})
    print(json.dumps({"status": "REPLAY_PASS" if replay else "INFERENCE_PASS", "rows": 26061, "outputs": outputs}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("RUN_TRAINING", "VERIFY_TRAINING", "RUN_INFERENCE", "REPLAY"))
    parser.add_argument("--replay-receipt", default="replay-qa.json")
    args = parser.parse_args()
    assert Path(args.replay_receipt).name == args.replay_receipt and args.replay_receipt.endswith(".json")
    assert args.replay_receipt not in {"training-result.json", "training-qa.json", "inference-qa.json"}
    cfg = json.loads((ROOT / "config.json").read_text())
    manifest = json.loads((ROOT / "06_docs/BUILD_MANIFEST.json").read_text())
    assert fingerprint() == manifest["files_sha256"], "pre-training code/config seal changed"
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "torch", "scipy", "scikit-learn", "threadpoolctl")}
    versions["python"] = platform.python_version()
    assert versions == cfg["versions"], "frozen dependency environment changed"
    assert cfg["cpu_threads"] == 2 and cfg["device"] == "cpu" and cfg["epochs"] == 60
    assert cfg["seeds"] == [20260901, 20260902, 20260903] and cfg["correction_strength"] == 1.0
    assert cfg["selected_arms"] and set(cfg["selected_arms"]) <= {"insample_full", "crossfit_full"}
    source = (Path(os.environ["P2_DATA_DIR"]) / "observations.csv").resolve()
    assert not source.is_relative_to(ROOT) and not ROOT.is_relative_to(source.parent)
    torch.set_num_threads(2)
    guard(args.mode, source)
    with threadpool_limits(limits=2):
        if args.mode == "RUN_TRAINING":
            train(source, cfg)
        elif args.mode == "VERIFY_TRAINING":
            verify_training(source, cfg)
        else:
            inference(source, cfg, args.mode == "REPLAY", args.replay_receipt)


if __name__ == "__main__":
    main()
