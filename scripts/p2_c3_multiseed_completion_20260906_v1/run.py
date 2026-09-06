"""Complete C60/L120 eight-fold three-seed evidence without replaying old fits."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ID = "p2_c3_multiseed_completion_20260906_v1"
OUT, REPORT = ROOT / "artifacts" / ID, ROOT / "reports" / ID
CONFIG = ROOT / "configs/experiments" / (ID + ".json")
OLD_SCRIPT = ROOT / "scripts/p2_c3_training_comparison_20260906_v1/run.py"
SPEC = importlib.util.spec_from_file_location("p2_completion_fixed_helper", OLD_SCRIPT)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
import numpy as np
import torch
from threadpoolctl import threadpool_limits


def cfg():
    c = r.read(CONFIG)
    if c["arms"] != ["C60", "L120"] or c["new_fits"] != 28 or c["combined_models"] != 48:
        raise ValueError("fixed two-arm completion budget")
    if c["new_folds"] != ["B1", "B2", "B4", "B5", "B6", "B7", "B8"] or c["new_seeds"] != [20260902, 20260903]:
        raise ValueError("only missing combinations may be trained")
    if c["cpu_threads"] != 2 or c["device"] != "cuda" or c["wallcap_seconds"] != 3600:
        raise ValueError("fixed CPU2/CUDA/60min")
    return c


def parent():
    c = cfg()
    paths = [(r.OUT / "terminal_result.json", c["parent_result_sha256"]),
             (r.REPORT / "independent-qa.json", c["parent_qa_sha256"]),
             (r.OUT / "fresh-replay.json", c["parent_replay_sha256"])]
    for path, digest in paths:
        if r.sha(path) != digest:
            raise ValueError("parent immutable result/QA/replay drift")
    value = r.read(paths[0][0])
    if value["status"] != "COMPLETE_24_SCREEN_PLUS_4_PRIMARY_SEED_FITS" or value["completed_fits"] != 28:
        raise ValueError("complete parent required")
    return value


def reused():
    fits = [f for f in parent()["fits"] if f["recipe"] in cfg()["arms"]]
    expected = {(arm, f"B{k}", 20260901) for arm in cfg()["arms"] for k in range(1, 9)}
    expected |= {(arm, "B3", seed) for arm in cfg()["arms"] for seed in cfg()["new_seeds"]}
    if len(fits) != 20 or {(f["recipe"], f["fold"], f["seed"]) for f in fits} != expected:
        raise ValueError("exact twenty-model reuse product")
    for f in fits:
        if r.sha(r.OUT / f["model_file"]) != f["model_sha256"] or r.sha(r.OUT / f["predictions_file"]) != f["predictions_sha256"]:
            raise ValueError("reused model/prediction SHA mismatch")
    return fits


def pins():
    paths = [Path(__file__), Path(__file__).with_name("qa.py"), CONFIG,
        ROOT / "tests/test_p2_c3_multiseed_completion_20260906_v1.py",
        OLD_SCRIPT, r.CONFIG, r.CORE, r.SEAL, r.OUT / "terminal_result.json",
        r.REPORT / "independent-qa.json", r.OUT / "fresh-replay.json", r.REPORT / "source-support.json",
        ROOT / "configs/evaluation/ocean_forward_v5.json"]
    return {p.relative_to(ROOT).as_posix(): r.sha(p) for p in paths}


def seal():
    r.verify_seal()
    selected = reused()
    r.save(REPORT / "preregistration-seal.json", {"status": "BEFORE_NEW_FITS", "pins": pins(),
        "new_fits": 0, "reuse_count": len(selected), "reused_fits": selected, "created_unix": time.time()})


def verify():
    r.verify_seal()
    if r.read(REPORT / "preregistration-seal.json")["pins"] != pins():
        raise ValueError("new completion seal drift")
    return cfg()


def guard(training):
    allowed = {r.source_path()}
    for f in r.read(REPORT / "preregistration-seal.json")["reused_fits"]:
        allowed |= {r.OUT / f["model_file"], r.OUT / f["predictions_file"]}
    allowed.add(r.OUT / "evaluation.npz")

    def audit(event, args):
        if event == "socket.connect":
            raise PermissionError("no network")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(args[0]).resolve()
        writing = (isinstance(args[1], str) and any(ch in args[1] for ch in "wax+")) or (
            isinstance(args[2], int) and bool(args[2] & (os.O_WRONLY | os.O_RDWR)))
        if path in allowed:
            if writing:
                raise PermissionError("reused/source artifacts immutable")
            return
        if not r.path_allowed(path, writing, r.source_path(), OUT):
            raise PermissionError("source/new/exact-reuse allowlist")
    sys.addaudithook(audit)
    if training:
        def denied(*_args, **_kwargs):
            raise PermissionError("no warm start torch.load during new fits")
        torch.load = denied


def data():
    frame, truth = r.load_source(r.cfg())
    if r.support_payload(frame, truth) != r.read(r.REPORT / "source-support.json"):
        raise ValueError("same support/keys/features required")
    return frame, truth


def worker():
    c = verify()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    guard(True)
    started = time.monotonic()
    result = {"status": "RUNNING", "pid": os.getpid(), "new_fits": 0, "attempted_fits": 0, "reused_models": 20,
              "fits": [], "official_rows": 0, "csv_written": 0, "upload": 0}
    try:
        frame, truth = data()
        (OUT / "03_model").mkdir()
        (OUT / "04_evaluation").mkdir()
        for fold in r.contract()["folds"]:
            if fold["id"] not in c["new_folds"]:
                continue
            train, valid = r.split_masks(frame, fold)
            training, query = frame.loc[train].reset_index(drop=True), frame.loc[valid].reset_index(drop=True)
            altered, _, supported = r.make_outage(query, fold)
            for arm in c["arms"]:
                recipe = next(rec for rec in r.cfg()["recipes"] if rec["id"] == arm)
                recipe_cfg = r.recipe_config(r.cfg(), recipe)
                arrays, mass = r.core.training_arrays(training, truth[train], "v23_blockmask", recipe_cfg)
                prior_fit = next(f for f in reused() if f["recipe"] == arm and f["fold"] == fold["id"])
                if [r.array_sha(a) for a in arrays] != prior_fit["training_arrays_sha256"]:
                    raise ValueError("new fit input differs from exact first-seed comparator")
                for seed in c["new_seeds"]:
                    name = f"{arm}_{fold['id']}_seed{seed}"

                    def progress(epoch, elapsed, current_name=name):
                        r.save(OUT / "progress.json", {"stage": current_name, "pid": os.getpid(), "epoch": epoch,
                            "completed_fits": result["new_fits"], "maximum_fits": 28,
                            "runtime_seconds": time.monotonic() - started, "fit_seconds": elapsed}, progress=True)
                    result["attempted_fits"] += 1
                    if result["attempted_fits"] > 28:
                        raise ValueError("new fit budget exceeded")
                    model, fit = r.core.fit_model(arrays, "v23_blockmask", seed, recipe_cfg, progress)
                    model_path = OUT / "03_model" / (name + ".pt")
                    with model_path.open("xb") as stream:
                        torch.save(model.state_dict(), stream)
                    predictions = {"natural": r.predict(model, query)}
                    if supported.all():
                        predictions["outage"] = r.predict(model, altered)
                    prediction_path = OUT / "04_evaluation" / (name + ".npz")
                    np.savez(prediction_path, **predictions)
                    fit.update(fit_id=name, recipe=arm, fold=fold["id"], model_file=model_path.relative_to(OUT).as_posix(),
                        model_sha256=r.sha(model_path), predictions_file=prediction_path.relative_to(OUT).as_posix(),
                        predictions_sha256=r.sha(prediction_path), recipe_config=recipe_cfg, training=mass,
                        train_keys_sha256=r.array_sha(r.key_values(training)), train_truth_sha256=r.array_sha(truth[train]),
                        valid_keys_sha256=r.array_sha(r.key_values(query)), training_arrays_sha256=[r.array_sha(a) for a in arrays])
                    result["fits"].append(fit)
                    result["new_fits"] += 1
                    r.save(OUT / "fit-receipts.json", result["fits"], progress=True)
                    print(json.dumps({"completed_fits": result["new_fits"], "maximum_fits": 28}), flush=True)
                    del model
                    torch.cuda.empty_cache()
        if result["new_fits"] != result["attempted_fits"] or result["new_fits"] != 28:
            raise ValueError("new28 fit accounting")
        all_fits = [(r.OUT, f) for f in reused()] + [(OUT, f) for f in result["fits"]]
        buffer = {"key": r.key_values(frame), "truth": truth, "fold": np.full(len(frame), "", dtype="U2"),
                  "time": frame.time.to_numpy(str), "layer": frame.layer.to_numpy(),
                  "natural_T5_missing": ~np.isfinite(frame.temp_5.to_numpy())}
        for arm in c["arms"]:
            for surface in ("natural", "outage"):
                buffer[surface + "_" + arm] = np.full(len(frame), np.nan)
        for fold in r.contract()["folds"]:
            _, valid = r.split_masks(frame, fold)
            buffer["fold"][valid] = fold["id"]
            for arm in c["arms"]:
                items = sorted(((root, f) for root, f in all_fits if f["recipe"] == arm and f["fold"] == fold["id"]), key=lambda item: item[1]["seed"])
                if [f["seed"] for _, f in items] != c["all_seeds"]:
                    raise ValueError("three seed ensemble product")
                for surface in ("natural", "outage"):
                    predictions = []
                    for root, fit in items:
                        with np.load(root / fit["predictions_file"], allow_pickle=False) as values:
                            if surface not in values:
                                predictions = []
                                break
                            predictions.append(values[surface])
                    if predictions:
                        buffer[surface + "_" + arm][valid] = np.mean(predictions, axis=0)
        with np.load(r.OUT / "evaluation.npz", allow_pickle=False) as previous:
            b3 = buffer["fold"] == "B3"
            for arm in c["arms"]:
                for surface in ("natural", "outage"):
                    if not np.array_equal(buffer[surface + "_" + arm][b3], previous[f"B3_{arm}_{surface}_all3"]):
                        raise ValueError("B3 primary must remain exact")
        np.savez(OUT / "evaluation.npz", **buffer)
        summary = r.build_summary(truth, {a: buffer["natural_" + a] for a in c["arms"]},
            {a: buffer["outage_" + a] for a in c["arms"]}, parent()["support"], buffer["fold"], buffer["layer"], buffer["time"])
        verify()
        result.update(status="COMPLETE_28_NEW_PLUS_20_REUSED_MODELS", combined_models=48, summary=summary,
            runtime_seconds=time.monotonic() - started, evaluation_sha256=r.sha(OUT / "evaluation.npz"),
            primary_B3_unchanged=True, selection_changed=False, source_sha256=r.sha(r.source_path()),
            parent_result_sha256=c["parent_result_sha256"], device="cuda", cpu_threads=2)
        r.save(OUT / "terminal_result.json", result)
        print(json.dumps({"status": result["status"], "new_fits": 28, "seconds": result["runtime_seconds"]}), flush=True)
    except BaseException as error:
        result.update(status="TERMINAL_TECHNICAL_FAILURE", error_type=type(error).__name__, error=str(error),
                      automatic_restart=False, runtime_seconds=time.monotonic() - started)
        if not (OUT / "terminal_result.json").exists():
            r.save(OUT / "terminal_result.json", result)
        raise


def replay():
    verify()
    result = r.read(OUT / "terminal_result.json")
    if result["status"] != "COMPLETE_28_NEW_PLUS_20_REUSED_MODELS" or os.getpid() == result["pid"]:
        raise ValueError("complete/newPID required")
    if not torch.cuda.is_available():
        raise RuntimeError("same CUDA replay required")
    r.save(OUT / "REPLAY_ATTEMPT_LOCK.json", {"pid": os.getpid(), "training_pid": result["pid"]})
    guard(False)
    started = time.monotonic()
    frame, truth = data()
    fits = [(r.OUT, f) for f in reused()] + [(OUT, f) for f in result["fits"]]
    checks = []
    for fold in r.contract()["folds"]:
        train, valid = r.split_masks(frame, fold)
        query = frame.loc[valid].reset_index(drop=True)
        altered, _, supported = r.make_outage(query, fold)
        for root, f in ((root, fit) for root, fit in fits if fit["fold"] == fold["id"]):
            if r.sha(root / f["model_file"]) != f["model_sha256"] or r.sha(root / f["predictions_file"]) != f["predictions_sha256"]:
                raise ValueError("replay artifact SHA")
            if r.array_sha(truth[train]) != f["train_truth_sha256"] or r.array_sha(r.key_values(frame.loc[train])) != f["train_keys_sha256"]:
                raise ValueError("training source lineage")
            model = r.core.make_model("v23_blockmask", 11)
            model.load_state_dict(torch.load(root / f["model_file"], map_location="cpu", weights_only=True))
            with np.load(root / f["predictions_file"], allow_pickle=False) as pred:
                if not np.array_equal(r.predict(model, query), pred["natural"]):
                    raise ValueError("entire natural replay mismatch")
                if supported.all() and not np.array_equal(r.predict(model, altered), pred["outage"]):
                    raise ValueError("entire outage replay mismatch")
            checks.append(f["fit_id"])
            del model
            torch.cuda.empty_cache()
    verify()
    r.save(OUT / "fresh-replay.json", {"status": "PASS", "models": len(checks), "checks": checks,
        "pid": os.getpid(), "training_pid": result["pid"], "scope": "entire validation rows natural+supported outage for48 models",
        "new_fits": 0, "max_absolute_error": 0, "runtime_seconds": time.monotonic() - started,
        "terminal_result_sha256": r.sha(OUT / "terminal_result.json"), "official_rows": 0})
    print(json.dumps({"status": "PASS", "models_replayed": len(checks), "fits": 0}))


def execute():
    c = verify()
    if OUT.exists():
        raise ValueError("fresh output only")
    OUT.mkdir(parents=True)
    r.save(OUT / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "started_unix": time.time(), "new_fits": 28,
        "reused_models": 20, "wallcap_seconds": 3600, "GPU0_handoff": "root explicit authorization",
        "seal_sha256": r.sha(REPORT / "preregistration-seal.json")})
    child = subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "worker"])
    try:
        code = child.wait(timeout=c["wallcap_seconds"])
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False, capture_output=True)
        child.wait(timeout=30)
        if not (OUT / "terminal_result.json").exists():
            r.save(OUT / "terminal_result.json", {"status": "BUDGET_WALLCAP_TERMINAL", "automatic_restart": False})
        raise SystemExit(124) from None
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "execute", "worker", "replay"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        {"seal": seal, "execute": execute, "worker": worker, "replay": replay}[args.mode]()
