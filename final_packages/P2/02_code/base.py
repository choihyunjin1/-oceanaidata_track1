"""Portable source -> scratch model -> answer, with consumed stage receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import core  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
MODELS, LOGS, ANSWERS = (PACKAGE / name for name in ("03_model", "04_logs", "05_answer"))
KEYS = ["station", "layer", "time"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def fingerprint():
    paths = [PACKAGE / "config.json", *sorted((PACKAGE / "02_code").glob("*.py"))]
    return {p.relative_to(PACKAGE).as_posix(): sha(p) for p in paths}


def canonical_keys(frame):
    return (
        frame.station.astype(str)
        + "|"
        + frame.layer.astype(int).astype(str)
        + "|"
        + pd.to_datetime(frame.time, utc=True).astype(str)
    )


def validate_output(frame, sample, index, expected_rows):
    keys, sk, ik = (canonical_keys(data) for data in (frame, sample, index))
    checks = {
        "schema": list(frame.columns) == [*KEYS, "temp"],
        "rows": len(frame) == len(sample) == len(index) == expected_rows,
        "unique": not keys.duplicated().any()
        and not sk.duplicated().any()
        and not ik.duplicated().any(),
        "sample_order": keys.tolist() == sk.tolist(),
        "index_keyset": set(keys) == set(ik),
        "finite": bool(np.isfinite(frame.temp.to_numpy(float)).all()),
        "layers": set(frame.layer.astype(int)) == {2, 3, 4},
    }
    if not all(checks.values()):
        raise ValueError(f"answer schema/key contract: {checks}")
    return {k: bool(v) for k, v in checks.items()}


def path_allowed(path, mode, writing, source):
    path = Path(path).resolve()
    if "external_data" in path.parts or "hidden" in path.name.lower():
        return False
    if path == source:
        return not writing
    if path.suffix.lower() == ".csv":
        if mode in ("RUN_INFERENCE", "REPLAY") and path in {
            source.parent / "test_index.csv",
            source.parent / "sample_submission.csv",
        }:
            return not writing
        return mode in ("RUN_INFERENCE", "REPLAY") and ANSWERS in path.parents
    if path.suffix.lower() in {".pt", ".pth"}:
        return MODELS in path.parents
    if path.suffix.lower() in {".npz", ".npy", ".parquet", ".pkl", ".pickle", ".cbm"}:
        return False
    return True


def install_guard(mode, source):
    counts = {
        "source_reads": 0,
        "official_index_key_rows": 0,
        "official_sample_key_rows": 0,
        "sample_value_rows": 0,
        "old_model_reads": 0,
        "old_answer_reads": 0,
        "external_data_reads": 0,
    }

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network forbidden")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        writing = (isinstance(args[1], str) and any(c in args[1] for c in "wax+")) or (
            isinstance(args[2], int) and bool(args[2] & (os.O_WRONLY | os.O_RDWR))
        )
        if not path_allowed(args[0], mode, writing, source):
            raise PermissionError("portable input/model allowlist denied")

    sys.addaudithook(guard)
    reader = pd.read_csv

    def guarded_reader(path, *args, **kwargs):
        path = Path(path).resolve()
        if path == source:
            counts["source_reads"] += 1
        elif path in {source.parent / "test_index.csv", source.parent / "sample_submission.csv"}:
            if mode == "RUN_TRAINING" or kwargs.get("usecols") != KEYS:
                raise PermissionError("only official key columns permitted after training")
            result = reader(path, *args, **kwargs)
            counts[
                "official_index_key_rows"
                if path.name == "test_index.csv"
                else "official_sample_key_rows"
            ] += len(result)
            return result
        elif ANSWERS not in path.parents:
            raise PermissionError("unapproved CSV")
        return reader(path, *args, **kwargs)

    pd.read_csv = guarded_reader
    if mode == "RUN_TRAINING":

        def denied(*args, **kwargs):
            raise PermissionError("torch.load forbidden during scratch training")

        torch.load = denied
    return counts


def read_observations(source, cfg, training):
    if sha(source) != cfg["source_sha256"]:
        raise ValueError("released source SHA mismatch")
    obs = pd.read_csv(source)
    if (
        set(obs.columns)
        != {"station", "year", "time", "layer", "depth", "nominal_depth", "temp", "psal"}
        or obs.duplicated(KEYS).any()
        or set(obs.station) != {"S-ORS"}
    ):
        raise ValueError("source schema or keys")
    obs.time = pd.to_datetime(obs.time, utc=True)
    if training:
        obs = obs.loc[
            obs.time.ge(pd.Timestamp(cfg["train_start"]))
            & obs.time.lt(pd.Timestamp(cfg["train_stop"]))
        ].copy()
    target = obs.layer.isin(core.TARGET_LAYERS)
    truth = obs.loc[target, "temp"].to_numpy(float).copy()
    obs.loc[target, ["temp", "psal"]] = np.nan
    frame, placeholder = core.public_frame(obs)
    if not np.isnan(placeholder).all():
        raise ValueError("target fields not masked")
    if training:
        keep = (
            np.isfinite(truth)
            & np.isfinite(frame.baseline)
            & frame.public_temp_count.ge(2).to_numpy()
        )
        frame, truth = frame.loc[keep].reset_index(drop=True), truth[keep]
    return frame, truth


def train(cfg, source):
    if any(MODELS.iterdir()) or (LOGS / "TRAIN_ATTEMPT_LOCK.json").exists():
        raise RuntimeError("scratch requires empty 03_model and unconsumed training lock")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("frozen CUDA recipe unavailable; no silent CPU fallback")
    started = time.monotonic()
    save(
        LOGS / "TRAIN_ATTEMPT_LOCK.json",
        {
            "pid": os.getpid(),
            "empty_model_directory": True,
            "code_hashes": fingerprint(),
            "utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )
    counts = install_guard("RUN_TRAINING", source)
    frame, truth = read_observations(source, cfg, True)
    if len(frame) != 166268:
        raise ValueError("frozen training population changed")
    data, support = core.training_arrays(frame, truth, "v23_blockmask", cfg)
    fits = []
    for seed in cfg["seeds"]:

        def progress(epoch, elapsed, current_seed=seed):
            payload = {
                "status": "TRAINING",
                "pid": os.getpid(),
                "seed": current_seed,
                "epoch": epoch,
                "fits_completed": len(fits),
                "fit_seconds": elapsed,
                "runtime_seconds": time.monotonic() - started,
            }
            with (LOGS / "progress.json").open("w", encoding="utf-8") as stream:
                json.dump(payload, stream)

        model, receipt = core.fit_model(data, "v23_blockmask", seed, cfg, progress)
        path = MODELS / f"model_seed{seed}.pt"
        torch.save(model.cpu().state_dict(), path)
        fits.append({**receipt, "file": path.name, "sha256": sha(path)})
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"fit_complete": seed, "seconds": receipt["runtime_seconds"]}), flush=True)
    result = {
        "status": "SCRATCH_TRAINING_PASS",
        "pid": os.getpid(),
        "empty_model_directory_before_training": True,
        "source_sha256": sha(source),
        "code_hashes": fingerprint(),
        "new_full_fits": len(fits),
        "fits": fits,
        "training": support,
        "training_key_sha256": hashlib.sha256(
            "\n".join(canonical_keys(frame)).encode()
        ).hexdigest(),
        "runtime_seconds": time.monotonic() - started,
        "device": cfg["device"],
        "cpu_threads": cfg["cpu_threads"],
        "module_origins": {
            "core": Path(core.__file__).resolve().relative_to(PACKAGE).as_posix(),
            "entry": Path(__file__).resolve().relative_to(PACKAGE).as_posix(),
        },
        "access": counts,
        "official_access_rows": 0,
        "csv_written": 0,
        "upload": 0,
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "torch", "threadpoolctl")
        },
    }
    if result["source_sha256"] != cfg["source_sha256"]:
        raise ValueError("source mutation")
    save(MODELS / "MODEL_MANIFEST.json", result)
    save(LOGS / "training-result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "new_full_fits": len(fits),
                "runtime_seconds": result["runtime_seconds"],
            }
        ),
        flush=True,
    )


def infer(cfg, source, replay):
    training = json.loads((MODELS / "MODEL_MANIFEST.json").read_text())
    if (
        training["status"] != "SCRATCH_TRAINING_PASS"
        or training["pid"] == os.getpid()
        or training["new_full_fits"] != 3
        or training["code_hashes"] != fingerprint()
    ):
        raise ValueError("new-process training lineage contract")
    mode = "REPLAY" if replay else "RUN_INFERENCE"
    path = ANSWERS / ("replay_p2_L120_3seed.csv" if replay else "submission_p2_L120_3seed.csv")
    if path.exists():
        raise RuntimeError("answer already exists; never overwrite")
    if replay:
        previous = json.loads((LOGS / "inference-result.json").read_text())
        if os.getpid() == previous["pid"]:
            raise ValueError("replay requires a third process")
    save(LOGS / f"{mode}_ATTEMPT_LOCK.json", {"pid": os.getpid(), "training_pid": training["pid"]})
    access = install_guard(mode, source)
    started = time.monotonic()
    index = pd.read_csv(source.parent / "test_index.csv", usecols=KEYS)
    sample = pd.read_csv(source.parent / "sample_submission.csv", usecols=KEYS)
    frame, _ = read_observations(source, cfg, False)
    frame.index = canonical_keys(frame)
    wanted = canonical_keys(sample)
    if frame.index.duplicated().any() or not wanted.isin(frame.index).all():
        raise ValueError("query feature coverage")
    query = frame.loc[wanted].reset_index(drop=True)
    data = core.arrays(query)
    per_seed = []
    for model in training["fits"]:
        weight = MODELS / model["file"]
        if sha(weight) != model["sha256"]:
            raise ValueError("new model hash mismatch")
        network = core.make_model("v23_blockmask", data[2].shape[1])
        network.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
        normalized = core.predict_model(network, *data)
        per_seed.append(
            query.baseline.to_numpy(float) + normalized * core.compute_profile_scale(query)
        )
    prediction = np.mean(per_seed, axis=0)
    output = sample[KEYS].copy()
    output["temp"] = prediction
    checks = validate_output(output, sample, index, cfg["expected_rows"])
    output.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    roundtrip = pd.read_csv(path)
    checks.update(
        {
            "serialized_" + k: v
            for k, v in validate_output(roundtrip, sample, index, cfg["expected_rows"]).items()
        }
    )
    error = float(np.max(np.abs(roundtrip.temp.to_numpy(float) - prediction)))
    if error >= 1e-8 or sha(source) != cfg["source_sha256"]:
        raise ValueError("serialization or source integrity")
    result = {
        "status": "FRESH_PROCESS_REPLAY_PASS" if replay else "FRESH_PROCESS_INFERENCE_PASS",
        "pid": os.getpid(),
        "training_pid": training["pid"],
        "rows": len(output),
        "answer": path.relative_to(PACKAGE).as_posix(),
        "answer_sha256": sha(path),
        "checks": checks,
        "serialization_max_abs_error": error,
        "code_hashes": fingerprint(),
        "source_sha256_unchanged": True,
        "runtime_seconds": time.monotonic() - started,
        "access": access,
        "new_fits": 0,
        "upload": 0,
    }
    if replay:
        result["first_inference_pid"] = previous["pid"]
        result["exact_first_inference_sha_match"] = (
            result["answer_sha256"] == previous["answer_sha256"]
        )
        if not result["exact_first_inference_sha_match"]:
            result["status"] = "REPLAY_MISMATCH"
    save(LOGS / ("replay-result.json" if replay else "inference-result.json"), result)
    print(
        json.dumps({k: result[k] for k in ("status", "rows", "answer_sha256", "runtime_seconds")})
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("RUN_TRAINING", "RUN_INFERENCE", "REPLAY"))
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2), default=None)
    args = parser.parse_args()
    cfg = json.loads((PACKAGE / "config.json").read_text())
    if args.cpu_threads is not None:
        cfg["cpu_threads"] = args.cpu_threads
    assert cfg["seeds"] == [20260901, 20260902, 20260903] and cfg["epochs"] == 60
    for path in (MODELS, LOGS, ANSWERS):
        path.mkdir(exist_ok=True)
    source = (Path(os.environ["P2_DATA_DIR"]) / "observations.csv").resolve()
    torch.set_num_threads(cfg["cpu_threads"])
    try:
        with threadpool_limits(cfg["cpu_threads"]):
            train(cfg, source) if args.mode == "RUN_TRAINING" else infer(
                cfg, source, args.mode == "REPLAY"
            )
    except Exception as exc:
        failure = LOGS / f"{args.mode}-failure.json"
        if not failure.exists():
            save(
                failure,
                {
                    "status": "TERMINAL_TECHNICAL_FAILURE",
                    "mode": args.mode,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "automatic_restart": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
