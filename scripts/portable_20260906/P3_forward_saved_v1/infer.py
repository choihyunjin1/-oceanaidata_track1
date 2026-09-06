"""Long-term saved-model inference, not a cold-training time-limit workaround."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
MODEL_NAMES = ("full_single.cbm", "full_multi.cbm", "full_router.joblib")
PUBLIC_NAMES = ("test_context.parquet", "test_index.csv")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def postprocessing_contract(recipe):
    expected = {
        "alpha": 10.0,
        "temperature_multiplier": 2.0,
        "strength": 0.5,
        "name": "smooth_medium",
        "active_leads": [12, 18, 24],
    }
    if (
        {key: recipe["router"][key] for key in expected} != expected
        or recipe["shrink"] != {"active_leads": [12, 18, 24], "persistence_weight": 0.2}
        or recipe["model"]["single_weight"] != 0.5
        or recipe["model"]["multi_weight"] != 0.5
    ):
        raise ValueError("fixed postprocessing provenance differs")


def preflight(root=ROOT):
    root = Path(root).resolve()
    manifest = read(root / "02_code/manifest.json")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("saved package manifest mismatch")
    c = read(root / "02_code/frozen.json")
    if c["purpose"] != "LONG_TERM_SAVED_INFERENCE_NOT_COLD_TRAINING":
        raise ValueError("wrong package purpose")
    if c["variant"] not in ("numeric", "hmax") or c["worker_timeout_seconds"] != 600:
        raise ValueError("fixed variant/resource contract mismatch")
    postprocessing_contract(c["recipe"])
    columns = c["selected_columns"]
    if len(columns) != len(set(columns)) or len(columns) != (
        527 if c["variant"] == "hmax" else 591
    ):
        raise ValueError("selected feature count mismatch")
    if c["variant"] == "hmax" and any(name.startswith("hmax_") for name in columns):
        raise ValueError("hmax feature survived removal")
    if set(c["full_models_sha256"]) != set(MODEL_NAMES):
        raise ValueError("exactly three full models required")
    for name, expected in c["full_models_sha256"].items():
        if sha(root / "03_model" / name) != expected:
            raise ValueError("saved model bytes changed")
    if c["cold_training_qa_status"] != "PASS" or len(c["expected_answer_sha256"]) != 64:
        raise ValueError("accepted cold training/answer provenance missing")
    return c


def access_guard(source, output):
    allowed_data = {source / name for name in PUBLIC_NAMES}
    allowed_models = {ROOT / "03_model" / name for name in MODEL_NAMES}
    deny = os.environ.get("P3_DENY_REPO")

    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Python network guard; not OS isolation")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(k in mode for k in "wax+")) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        )
        if any(
            k in path.name.lower()
            for k in ("hidden", "credentials", "refined_alpha", "axis_contract")
        ):
            raise PermissionError("forbidden ancestry")
        if path.is_relative_to(source) and (path not in allowed_data or writing):
            raise PermissionError("public input allowlist only; immutable source")
        if (
            deny
            and path.is_relative_to(Path(deny).resolve())
            and not path.is_relative_to(Path(sys.prefix).resolve())
        ):
            raise PermissionError("original repository denied")
        if (path.is_relative_to(CODE) or path in allowed_models) and writing:
            raise PermissionError("frozen source/models immutable")
        if path.suffix.lower() in {
            ".csv",
            ".parquet",
            ".npz",
            ".npy",
            ".cbm",
            ".joblib",
            ".pt",
            ".ckpt",
        }:
            own_answer = path == output / "submission.csv"
            if path not in allowed_data and path not in allowed_models and not own_answer:
                raise PermissionError(
                    "no historical cache, OOF, probe, old answer or other model input"
                )

    sys.addaudithook(audit)


def load_policy(c):
    sys.path[:0] = [str(CODE / "scripts"), str(CODE / "src")]
    import joblib
    from catboost import CatBoostRegressor

    materializer = importlib.import_module("p3_forward_candidate_materialize_20260906_v1")
    name = (
        "run_p3_hmax_removed_forward_20260906_v1"
        if c["variant"] == "hmax"
        else "run_p3_numeric_lead_forward_gpu_20260906_v2"
    )
    e = importlib.import_module(name)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    models = (
        CatBoostRegressor().load_model(ROOT / "03_model/full_single.cbm"),
        CatBoostRegressor().load_model(ROOT / "03_model/full_multi.cbm"),
        joblib.load(ROOT / "03_model/full_router.joblib"),
    )
    single, multi, router = models
    cats = [0, 1] if c["variant"] == "hmax" else [0]
    if (
        single.tree_count_ != 700
        or multi.tree_count_ != 1200
        or single.get_cat_feature_indices() != cats
        or multi.get_cat_feature_indices() != [0]
    ):
        raise ValueError("native model schema/iterations mismatch")
    if single.feature_names_ != [
        "station",
        "lead_h",
        "current_hs_for_residual",
        *c["selected_columns"],
    ] or multi.feature_names_ != ["station", *c["selected_columns"]]:
        raise ValueError("native model feature order mismatch")
    if (single.random_seed_, multi.random_seed_) != (20260817, 20260817):
        raise ValueError("full model seed mismatch")
    if (
        router.config.alpha,
        router.config.temperature_multiplier,
        router.config.strength,
        router.config.name,
    ) != (10.0, 2.0, 0.5, "smooth_medium"):
        raise ValueError("frozen router configuration mismatch")
    if {"target_hs", "truth", "label", "anomaly_type"}.intersection(router.columns) or (
        c["variant"] == "hmax" and any(k.startswith("hmax_") for k in router.columns)
    ):
        raise ValueError("router feature boundary mismatch")
    cfg = {"removal": {"base_prefix": "hmax_"}} if c["variant"] == "hmax" else {}
    return e, materializer, cfg, models


def make_answer(c, source, e, materializer, cfg, models):
    import numpy as np
    import pandas as pd

    hashes = {name: sha(source / name) for name in PUBLIC_NAMES}
    if hashes != c["public_input_sha256"]:
        raise ValueError("public input bytes differ from accepted cold inference")
    context = pd.read_parquet(
        source / "test_context.parquet",
        columns=["case_id", "station", "step_minute", *e.BASE_COLUMNS, *e.DIRECTION_COLUMNS],
    )
    index = pd.read_csv(source / "test_index.csv", usecols=materializer.KEYS)
    if len(context) != 57800 or len(index) != 1200 or index.duplicated(materializer.KEYS).any():
        raise ValueError("public count/unique keys mismatch")
    records = []
    for case_id, group in context.groupby("case_id", sort=False):
        group = group.sort_values("step_minute")
        if (
            len(group) != 289
            or group.station.nunique() != 1
            or not np.array_equal(group.step_minute, np.arange(-2880, 1, 10))
        ):
            raise ValueError("public anonymous 48h grid mismatch")
        records.append(
            {
                "case_id": case_id,
                "station": str(group.station.iloc[0]),
                **e.summarize_context(group),
            }
        )
    cases = (
        index[["case_id", "station"]]
        .drop_duplicates()
        .merge(
            pd.DataFrame(records),
            on=["case_id", "station"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
    )
    if len(cases) != 200 or not cases._merge.eq("both").all():
        raise ValueError("public case key mismatch")
    keys, prediction = materializer.predict_cases(
        e, cfg, cases.drop(columns="_merge"), c["selected_columns"], models
    )
    keys["hs_pred"] = prediction
    answer = index.merge(keys, on=materializer.KEYS, how="left", validate="one_to_one")
    if (
        not answer[materializer.KEYS].equals(index)
        or not np.isfinite(answer.hs_pred).all()
        or not answer.hs_pred.between(0, 30).all()
    ):
        raise ValueError("answer key/order/finite/range mismatch")
    if hashes != {name: sha(source / name) for name in PUBLIC_NAMES}:
        raise ValueError("public input changed during inference")
    payload = answer.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != c["expected_answer_sha256"]:
        raise ValueError("saved-model answer bytes differ from accepted cold answer")
    return payload, hashes


def worker(output):
    started = time.perf_counter()
    timer = threading.Timer(600, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    access_guard(source, output)
    try:
        c = preflight()
        payload, hashes = make_answer(c, source, *load_policy(c))
        with (output / "submission.csv").open("xb") as stream:
            stream.write(payload)
        receipt = {
            "status": "EXACT_SAVED_INFERENCE_PASS",
            "pid": os.getpid(),
            "rows": 1200,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "expected_cold_answer_sha256": c["expected_answer_sha256"],
            "public_input_sha256": hashes,
            "seconds": time.perf_counter() - started,
            "manifest_sha256": sha(CODE / "manifest.json"),
            "cold_provenance_sha256": sha(ROOT / "06_docs/cold-provenance.json"),
            "model_fits": 0,
            "historical_cache_oof_probe_reads": 0,
            "old_answer_value_reads": 0,
            "sample_rows": 0,
            "hidden_rows": 0,
            "uploads": 0,
            "final_model_locked": False,
            "cold_training_reexecuted": False,
            "python_network_guard_only": True,
        }
        save(output / "answer-qa.json", receipt)
        print(json.dumps(receipt), flush=True)
        timer.cancel()
    except Exception as exc:
        save(
            output / "failure.json",
            {
                "status": "FAIL",
                "type": type(exc).__name__,
                "raw_error_omitted": True,
                "pid": os.getpid(),
                "fits": 0,
            },
        )
        raise SystemExit(1) from None


def terminate_worker(process):
    if os.name == "nt":
        # The Windows venv launcher may own the actual Python child process.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            # The child also has its own 600-second exit timer.
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=15)


def supervise(output, timeout=600):
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    output = output.resolve()
    if (
        output.exists()
        or output.is_relative_to(source)
        or output.is_relative_to(CODE)
        or output.is_relative_to(ROOT / "03_model")
    ):
        raise ValueError("new output folder outside data/code/models required")
    output.mkdir(parents=True)
    started = time.perf_counter()
    command = [
        sys.executable,
        "-I",
        str(CODE / "infer.py"),
        "--worker",
        "--official-approved",
        "--output-dir",
        str(output),
    ]
    with (
        (output / "worker.stdout.log").open("xb") as stdout,
        (output / "worker.stderr.log").open("xb") as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            code = process.wait(timeout=timeout)
            status = "COMPLETE" if code == 0 else "WORKER_FAILURE"
        except subprocess.TimeoutExpired:
            terminate_worker(process)
            code, status = 124, "600_SECOND_WORKER_LIMIT_TERMINATED"
    receipt = {
        "status": status,
        "supervisor_pid": os.getpid(),
        "worker_pid": process.pid,
        "elapsed_seconds": time.perf_counter() - started,
        "timeout_seconds": timeout,
        "exit_code": code,
        "new_fits": 0,
        "cold_training_clock_not_reused": True,
    }
    save(output / "process-receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--infer", action="store_true")
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--official-approved", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.preflight:
        c = preflight()
        print(
            json.dumps(
                {
                    "status": "STATIC_SAVED_PACKAGE_PREFLIGHT_PASS",
                    "variant": c["variant"],
                    "fits": 0,
                    "official_rows": 0,
                }
            )
        )
        return
    if not args.official_approved or args.output_dir is None:
        parser.error("official-input authorization and a new output directory are required")
    if args.worker:
        worker(args.output_dir.resolve())
    else:
        raise SystemExit(supervise(args.output_dir, 600))


if __name__ == "__main__":
    main()
