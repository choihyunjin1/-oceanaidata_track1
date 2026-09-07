"""Independent full OOF replay and same-row ten-versus-three-seed projection QA."""
# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s10_job", Path(__file__).with_name("p2_l120_s10_proj_20260907_v1.py"))
j = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j)
r = j.r
from p2_final_day_projection_20260907_v1 import project_profiles_vectorized


def endpoint_frame(query):
    ep = query[["time", "temp_1", "temp_5", "temp_6", "temp_7", "temp_8"]].drop_duplicates("time").copy()
    ep["temp_5"] = ep[["temp_5", "temp_6", "temp_7", "temp_8"]].where(np.isfinite(ep[["temp_5", "temp_6", "temp_7", "temp_8"]])).bfill(axis=1)["temp_5"]
    return ep[["time", "temp_1", "temp_5"]]


def compare(y, a, b):
    return {"rows":len(y), "s3_sse_C2":float(np.sum((a-y)**2)), "s10_sse_C2":float(np.sum((b-y)**2)), "s3_rmse_C":r.metric(y,a)["rmse_C"], "s10_rmse_C":r.metric(y,b)["rmse_C"], "delta_C":r.metric(y,b)["rmse_C"]-r.metric(y,a)["rmse_C"]}


def bootstrap(y, a, b, times):
    epoch = pd.Timestamp("2024-01-01", tz="Asia/Seoul")
    groups = ((pd.to_datetime(times, utc=True)-epoch).total_seconds() // (7*86400)).astype(int)
    _, ids = np.unique(groups, return_inverse=True)
    n = np.bincount(ids)
    sa, sb = np.bincount(ids, weights=(a-y)**2), np.bincount(ids, weights=(b-y)**2)
    idx = np.random.default_rng(20260907).integers(0, len(n), size=(2000, len(n)))
    delta = np.sqrt(sb[idx].sum(axis=1)/n[idx].sum(axis=1))-np.sqrt(sa[idx].sum(axis=1)/n[idx].sum(axis=1))
    return {"CI90":np.quantile(delta,[.05,.95]).tolist(), "P_improvement":float(np.mean(delta<0)), "blocks":len(n), "resamples":2000, "retrospective":True}


def run():
    started = time.monotonic()
    j.verify()
    terminal = r.read(j.OUT / "terminal_result.json")
    assert terminal["status"] == "TRAINED_PENDING_INDEPENDENT_QA" and terminal["pid"] != os.getpid()
    r.save(j.OUT / "QA_ATTEMPT_LOCK.json", {"pid":os.getpid(),"training_pid":terminal["pid"]})
    j.guard()
    frame, truth = r.load_source(r.cfg())
    old = j.inherited()
    result = {"status":"PASS","pid":os.getpid(),"training_pid":terminal["pid"],"new_fits":0,"official_rows":0,"hidden_rows":0,"uploads":0,"metrics":{},"model_replay_checks":0}
    bufs = {s: np.full(len(frame),np.nan) for s in ("s3","s10","s3_proj","s10_proj")}
    foldids = np.full(len(frame),"",dtype="U2")
    with np.load(j.COMPLETION / "evaluation.npz",allow_pickle=False) as z:
        canonical = z["natural_L120"].copy()
        assert np.array_equal(z["truth"],truth)
    for fold in r.contract()["folds"]:
        train, valid = r.split_masks(frame,fold)
        q = frame.loc[valid].reset_index(drop=True)
        altered, outage_rows, supported = r.make_outage(q,fold)
        items = sorted([(root,f) for root,f in old if f["fold"]==fold["id"]]+[(j.OUT,f) for f in terminal["fits"] if f["fold"]==fold["id"]],key=lambda v:v[1]["seed"])
        assert [f["seed"] for _,f in items] == list(range(20260901,20260911))
        natural, stress = [], []
        for root, f in items:
            assert r.sha(root/f["model_file"]) == f["model_sha256"]
            assert r.sha(root/f["predictions_file"]) == f["predictions_sha256"]
            model = r.core.make_model("v23_blockmask",11)
            model.load_state_dict(torch.load(root/f["model_file"],map_location="cpu",weights_only=True))
            with np.load(root/f["predictions_file"],allow_pickle=False) as z:
                p=r.predict(model,q)
                np.testing.assert_array_equal(p,z["natural"])
                natural.append(p)
                if supported.all():
                    p=r.predict(model,altered)
                    np.testing.assert_array_equal(p,z["outage"])
                    stress.append(p)
            result["model_replay_checks"] += 1
        a,b=np.mean(natural[:3],axis=0),np.mean(natural,axis=0)
        np.testing.assert_array_equal(a,canonical[valid])
        ap,bp=[project_profiles_vectorized(q,p,endpoint_frame(q)).prediction for p in (a,b)]
        for name,values in zip(bufs,(a,b,ap,bp),strict=True):
            bufs[name][valid]=values
        foldids[valid]=fold["id"]
        item={"natural_unprojected":compare(truth[valid],a,b),"natural_projected":compare(truth[valid],ap,bp),"layers":{}}
        for layer in (2,3,4):
            mask=q.layer.to_numpy()==layer
            item["layers"][str(layer)]=compare(truth[valid][mask],ap[mask],bp[mask])
        if supported.all():
            sa,sb=np.mean(stress[:3],axis=0),np.mean(stress,axis=0)
            sap,sbp=[project_profiles_vectorized(altered,p,endpoint_frame(altered)).prediction for p in (sa,sb)]
            item["outage_projected_wholefold"]=compare(truth[valid],sap,sbp)
            item["outage_projected_interval"]=compare(truth[valid][outage_rows],sap[outage_rows],sbp[outage_rows]) if outage_rows.any() else {"status":"NO_EVALUATION_ROWS"}
        else:
            item["outage"]={"status":"SUPPORT_BLOCKED","unsupported_rows":int((~supported).sum())}
        result["metrics"][fold["id"]]=item
    pooled=compare(truth,bufs["s3_proj"],bufs["s10_proj"])
    assert abs(pooled["s3_rmse_C"]-1.236732066)<1e-9
    result["metrics"]["pooled_projected"]=pooled
    result["bootstrap"]={}
    for name,mask in (("B3",foldids=="B3"),("pooled",np.ones(len(frame),bool))):
        result["bootstrap"][name]=bootstrap(truth[mask],bufs["s3_proj"][mask],bufs["s10_proj"][mask],pd.DatetimeIndex(frame.time[mask]))
    result["mean_improved_B3"]=result["metrics"]["B3"]["natural_projected"]["delta_C"]<0
    result["promotion_surface_pass"]=result["mean_improved_B3"] and result["bootstrap"]["B3"]["P_improvement"]>=.8
    result["automatic_worstblock_veto"]=False
    # Full-data models: all seven new predictions must replay; inherited three have their own linked full-cold receipt.
    fullframe,fulltruth=r.load_source(r.cfg())
    for fit in [f for f in terminal["fits"] if f["fold"]=="FULL"]:
        model=r.core.make_model("v23_blockmask",11)
        model.load_state_dict(torch.load(j.OUT/fit["model_file"],map_location="cpu",weights_only=True))
        assert r.sha(j.OUT/fit["model_file"])==fit["model_sha256"]
        with np.load(j.OUT/fit["predictions_file"],allow_pickle=False) as z:
            np.testing.assert_array_equal(r.predict(model,fullframe),z["natural"])
    np.savez(j.OUT/"evaluation.npz",truth=truth,fold=foldids,time=frame.time.to_numpy(str),layer=frame.layer.to_numpy(),**bufs)
    result.update(full_new_model_replay_checks=7,runtime_seconds=time.monotonic()-started,terminal_sha256=r.sha(j.OUT/"terminal_result.json"),evaluation_sha256=r.sha(j.OUT/"evaluation.npz"))
    r.save(j.REPORT/"independent-qa.json",result)
    print({"status":"PASS","internal_replayed":80,"full_new_replayed":7,"seconds":result["runtime_seconds"]})


if __name__=="__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
