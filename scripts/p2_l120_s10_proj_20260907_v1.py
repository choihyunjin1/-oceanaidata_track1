"""Fixed L120 ten-seed extension; no timeout, no change to inherited attempts."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "p2_l120_s10_proj_20260907_v1"
OUT, REPORT = ROOT / "artifacts" / ID, ROOT / "reports" / ID
CONFIG = ROOT / "configs/experiments" / (ID + ".json")
spec = importlib.util.spec_from_file_location("s10_historical", ROOT / "scripts/p2_c3_training_comparison_20260906_v1/run.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from threadpoolctl import threadpool_limits

FULL = Path("C:/Users/cedis/Documents/OceanFinalRelease_20260907/P2/SAVED_MODELS")
COMPLETION = ROOT / "artifacts/p2_c3_multiseed_completion_20260906_v1"


def inherited():
    items = []
    for folder in (r.OUT, COMPLETION):
        for f in r.read(folder / "terminal_result.json")["fits"]:
            if f["recipe"] == "L120":
                for name, digest in (("model_file", "model_sha256"), ("predictions_file", "predictions_sha256")):
                    assert r.sha(folder / f[name]) == f[digest]
                items.append((folder, f))
    assert len(items) == 24
    assert {(f["fold"], f["seed"]) for _, f in items} == {(f"B{k}", seed) for k in range(1, 9) for seed in range(20260901, 20260904)}
    return items


def seal():
    c = r.read(CONFIG)
    assert c["maximum_new_fits"] == 63 and c["hard_wallcap_seconds"] is None
    assert c["seeds"] == list(range(20260901, 20260911))
    r.cfg()
    r.contract()
    items = inherited()
    full = r.read(FULL / "03_model/MODEL_MANIFEST.json")
    assert full["new_full_fits"] == 3
    for f in full["fits"]:
        assert r.sha(FULL / "03_model" / f["file"]) == f["sha256"]
    paths = [CONFIG, Path(__file__), r.CORE, r.CONFIG,
             ROOT / "configs/evaluation/ocean_forward_v5.json",
             Path(r.__file__), ROOT / "scripts/p2_final_day_projection_20260907_v1.py",
             FULL / "03_model/MODEL_MANIFEST.json", FULL / "config.json",
             r.OUT / "terminal_result.json", COMPLETION / "terminal_result.json", COMPLETION / "evaluation.npz"]
    r.save(REPORT / "seal.json", {"status":"PREREGISTERED", "pins":{str(p):r.sha(p) for p in paths},
        "internal_reuse":24,"full_reuse":3,"new_fits":63,"created_unix":time.time(),
        "inherited_models":[f["model_sha256"] for _,f in items]})


def verify():
    for path, digest in r.read(REPORT / "seal.json")["pins"].items():
        assert r.sha(path) == digest, "sealed source drift"
    return r.read(CONFIG)


def guard():
    allowed = {r.source_path()}
    for folder, fit in inherited():
        allowed.update((folder / fit["model_file"], folder / fit["predictions_file"]))
    allowed.add(COMPLETION / "evaluation.npz")
    allowed.update(FULL / "03_model" / f["file"] for f in r.read(FULL / "03_model/MODEL_MANIFEST.json")["fits"])
    def audit(event, args):
        if event == "socket.connect":
            raise PermissionError("no network")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(args[0]).resolve()
        writing = (isinstance(args[1], str) and any(s in args[1] for s in "wax+")) or (isinstance(args[2], int) and bool(args[2] & (os.O_WRONLY | os.O_RDWR)))
        if path in allowed:
            if writing:
                raise PermissionError("immutable parent/source")
            return
        if not r.path_allowed(path, writing, r.source_path(), OUT):
            raise PermissionError("training-only allowlist")
    sys.addaudithook(audit)


def worker():
    c = verify()
    OUT.mkdir(parents=True, exist_ok=False)
    r.save(OUT / "ATTEMPT_LOCK.json", {"pid":os.getpid(),"started_unix":time.time(),"hard_wallcap":None,"new_fits":63})
    start = time.monotonic()
    fits = []
    try:
        guard()
        frame, truth = r.load_source(r.cfg())
        support = r.support_payload(frame, truth)
        assert support == r.read(r.REPORT / "source-support.json")
        old = inherited()
        recipe = r.recipe_config(r.cfg(), {"epochs":120,"weight_decay":0.0001})
        (OUT / "03_model").mkdir()
        (OUT / "04_evaluation").mkdir()
        for fid in c["order"]:
            full = fid == "FULL"
            if full:
                train = np.ones(len(frame), bool)
                training, query = frame, frame
                altered, supported = None, None
                previous = r.read(FULL / "03_model/MODEL_MANIFEST.json")["fits"]
            else:
                fold = next(f for f in r.contract()["folds"] if f["id"] == fid)
                train, valid = r.split_masks(frame, fold)
                training, query = frame.loc[train].reset_index(drop=True), frame.loc[valid].reset_index(drop=True)
                altered, _, supported = r.make_outage(query, fold)
                previous = [f for _, f in old if f["fold"] == fid]
            arrays, mass = r.core.training_arrays(training, truth[train], "v23_blockmask", recipe)
            hashes = [r.array_sha(a) for a in arrays]
            assert all(f["training_arrays_sha256"] == hashes for f in previous)
            for seed in c["new_seeds"]:
                name = f"L120_{fid}_seed{seed}"
                def progress(epoch, elapsed):
                    r.save(OUT / "progress.json", {"stage":name,"pid":os.getpid(),"epoch":epoch,"completed_fits":len(fits),"maximum_fits":63,"fit_seconds":elapsed,"runtime_seconds":time.monotonic()-start}, progress=True)
                progress(0, 0)
                model, fit = r.core.fit_model(arrays, "v23_blockmask", seed, recipe, progress)
                mp = OUT / "03_model" / (name + ".pt")
                with mp.open("xb") as stream:
                    torch.save(model.state_dict(), stream)
                predictions = {"natural": r.predict(model, query)}
                if not full and supported.all():
                    predictions["outage"] = r.predict(model, altered)
                pp = OUT / "04_evaluation" / (name + ".npz")
                np.savez(pp, **predictions)
                fit.update(fold=fid, recipe="L120", model_file=mp.relative_to(OUT).as_posix(), model_sha256=r.sha(mp), predictions_file=pp.relative_to(OUT).as_posix(), predictions_sha256=r.sha(pp), training_arrays_sha256=hashes, training=mass, recipe_config=recipe)
                fits.append(fit)
                r.save(OUT / "fit-receipts.json", fits, progress=True)
                print(json.dumps({"fit":name,"completed":len(fits),"seconds":fit["runtime_seconds"]}), flush=True)
                del model
                torch.cuda.empty_cache()
            del arrays
        assert len(fits) == 63
        verify()
        r.save(OUT / "terminal_result.json", {"status":"TRAINED_PENDING_INDEPENDENT_QA", "pid":os.getpid(),"new_fits":63,"reused_fits":27,"fits":fits,"source_sha256":r.sha(r.source_path()),"runtime_seconds":time.monotonic()-start,"official_rows":0,"hidden_rows":0,"csv_written":0,"uploads":0})
    except BaseException as error:
        r.save(OUT / "terminal_result.json", {"status":"TERMINAL_TECHNICAL_FAILURE","pid":os.getpid(),"error":str(error),"new_fits":len(fits),"automatic_restart":False,"runtime_seconds":time.monotonic()-start})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["seal","worker"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        {"seal":seal,"worker":worker}[args.mode]()
