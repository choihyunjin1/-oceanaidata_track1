"""No-fit same-row learning-stage blend with matched natural/outage projection."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from p2_final_day_projection_20260907_v1 import project_profiles_vectorized
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_historical_equal", ROOT / "scripts/p2_c3_training_comparison_20260906_v1/run.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
ID = "p2_c60_l120_equal_projection_20260907_v1"


def endpoints(frame):
    ep = frame[["time", "temp_1", "temp_5", "temp_6", "temp_7", "temp_8"]].drop_duplicates("time").copy()
    deep = ep[["temp_5", "temp_6", "temp_7", "temp_8"]]
    ep["temp_5"] = deep.where(np.isfinite(deep)).bfill(axis=1)["temp_5"]
    return ep[["time", "temp_1", "temp_5"]]


def compare(y, a, b):
    sa, sb = float(np.sum((a-y)**2)), float(np.sum((b-y)**2))
    ra, rb = float(np.sqrt(sa/len(y))), float(np.sqrt(sb/len(y)))
    return {"rows": len(y), "base_sse": sa, "candidate_sse": sb,
            "base_rmse_C": ra, "candidate_rmse_C": rb, "delta_C": rb-ra}


def bootstrap(y, a, b, times, cfg):
    epoch = pd.Timestamp("2024-01-01", tz="Asia/Seoul")
    groups = ((pd.to_datetime(times, utc=True)-epoch).total_seconds() // (cfg["days"]*86400)).astype(int)
    _, ids = np.unique(groups, return_inverse=True)
    n = np.bincount(ids)
    sa, sb = np.bincount(ids, weights=(a-y)**2), np.bincount(ids, weights=(b-y)**2)
    ix = np.random.default_rng(cfg["seed"]).integers(0, len(n), size=(cfg["resamples"], len(n)))
    delta = np.sqrt(sb[ix].sum(axis=1)/n[ix].sum(axis=1))-np.sqrt(sa[ix].sum(axis=1)/n[ix].sum(axis=1))
    return {"CI90": np.quantile(delta,[.05,.95]).tolist(), "P_improvement": float(np.mean(delta<0)),
            "blocks": len(n), "resamples": cfg["resamples"], "retrospective": True}


def main(candidate_id=ID, transform=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["evaluate", "qa"])
    args = parser.parse_args()
    start = time.monotonic()
    config = ROOT / "configs/experiments" / (candidate_id + ".json")
    cfg = r.read(config)
    path = ROOT / "artifacts/p2_c3_multiseed_completion_20260906_v1/evaluation.npz"
    assert r.sha(path) == cfg["evaluation_sha256"]
    if transform is None:
        assert cfg["weights"] == [.5,.5]
    frame, truth = r.load_source(r.cfg())
    buffers = {k: np.full(len(frame), np.nan) for k in ["base", "candidate", "outage_base", "outage_candidate"]}
    folds = np.full(len(frame), "", dtype="U2")
    metrics = {}
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z["key"], r.key_values(frame))
        assert np.array_equal(z["truth"], truth)
        for fold in r.contract()["folds"]:
            _, valid = r.split_masks(frame, fold)
            q = frame.loc[valid].reset_index(drop=True)
            altered, outage_rows, supported = r.make_outage(q, fold)
            folds[valid] = fold["id"]
            item = {}
            for surface, query in [("natural", q), ("outage", altered)]:
                if surface == "outage" and not supported.all():
                    item[surface] = {"status": "SUPPORT_BLOCKED"}
                    continue
                a, c = z[surface + "_L120"][valid], z[surface + "_C60"][valid]
                assert np.isfinite(a).all() and np.isfinite(c).all()
                ep = endpoints(query)
                candidate = (a+c)*.5 if transform is None else transform(query, a)
                ap, bp = [project_profiles_vectorized(query, x, ep).prediction for x in (a, candidate)]
                prefix = "" if surface == "natural" else "outage_"
                buffers[prefix+"base"][valid], buffers[prefix+"candidate"][valid] = ap, bp
                item[surface] = compare(truth[valid], ap, bp)
                if surface == "outage":
                    item["outage_interval"] = compare(truth[valid][outage_rows], ap[outage_rows], bp[outage_rows]) if outage_rows.any() else {"status":"NOT_ESTIMABLE_NO_ROWS"}
                else:
                    item["layers"] = {str(layer): compare(truth[valid][q.layer.to_numpy()==layer], ap[q.layer.to_numpy()==layer], bp[q.layer.to_numpy()==layer]) for layer in [2,3,4]}
            metrics[fold["id"]] = item
    metrics["pooled"] = compare(truth, buffers["base"], buffers["candidate"])
    assert abs(metrics["pooled"]["base_rmse_C"] - 1.236732066) < 1e-9
    missing = ~np.isfinite(frame.temp_5.to_numpy())
    metrics["natural_T5_missing"] = compare(truth[missing], buffers["base"][missing], buffers["candidate"][missing])
    boot = {name: bootstrap(truth[m], buffers["base"][m], buffers["candidate"][m], pd.DatetimeIndex(frame.time[m]), cfg["bootstrap"])
            for name,m in [("B3",folds=="B3"),("pooled",np.ones(len(frame),bool))]}
    report = ROOT / "reports" / candidate_id
    if args.mode == "evaluate":
        r.save(report / "result.json", {"status":"RETROSPECTIVE_INTERNAL_COMPLETE", "pid":os.getpid(),
            "config_sha256":r.sha(config), "evaluation_sha256":r.sha(path), "metrics":metrics, "bootstrap":boot,
            "new_fits":0, "official_rows":0, "hidden_rows":0, "seconds":time.monotonic()-start})
    else:
        old=r.read(report / "result.json")
        assert old["pid"] != os.getpid() and old["config_sha256"] == r.sha(config)
        assert old["metrics"] == metrics and old["bootstrap"] == boot
        r.save(report / "replay-qa.json", {"status":"EXACT_AGGREGATE_REPLAY_PASS", "pid":os.getpid(),
            "result_sha256":r.sha(report / "result.json"), "new_fits":0, "seconds":time.monotonic()-start})
    print(json.dumps({"B3":metrics["B3"],"pooled":metrics["pooled"],"bootstrap":boot}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
