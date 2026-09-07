"""Portable ten-seed L120 train, full-source replay, infer, exact answer replay."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/"02_code"))
import base as b
import numpy as np
import pandas as pd
import torch
from projection import project_profiles_vectorized, public_endpoint_frame
from threadpoolctl import threadpool_limits

M,L,A=P/"03_model",P/"04_logs",P/"05_answer"


def ah(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def settings():
    for name,digest in read(P/"SOURCE_MANIFEST.json")["files"].items():
        assert b.sha(P/name)==digest,"source drift"
    c=read(P/"config.json")
    assert c["seeds"]==list(range(20260901,20260911)) and c["epochs"]==120 and c["device"]=="cuda"
    assert c["wallcap_seconds"] is None and torch.cuda.is_available()
    return c


def guard(source,official=False):
    def hook(event,args):
        if event=="socket.connect":
            raise PermissionError("offline only")
        if event!="open" or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(args[0]).resolve()
        writing=(isinstance(args[1],str) and any(s in args[1] for s in "wax+")) or (isinstance(args[2],int) and bool(args[2] & (os.O_WRONLY|os.O_RDWR)))
        if "hidden" in path.name.lower() or "external_data" in path.parts:
            raise PermissionError("forbidden ancestry")
        if path==source or (official and path in {source.parent/"test_index.csv",source.parent/"sample_submission.csv"}):
            if writing:
                raise PermissionError("data immutable")
            return
        if path.suffix.lower() in {".csv",".pt",".npz",".npy",".parquet",".cbm",".pkl"} and P not in path.parents:
            raise PermissionError("package/source allowlist")
    sys.addaudithook(hook)


def train(c,source):
    assert not any(M.iterdir()),"cold requires empty model folder"
    save(L/"TRAIN_LOCK.json",{"pid":os.getpid(),"started":time.time(),"new_fits":10})
    guard(source)
    start=time.monotonic()
    frame,y=b.read_observations(source,c,True)
    data,mass=b.core.training_arrays(frame,y,"v23_blockmask",c)
    fits=[]
    for seed in c["seeds"]:
        def progress(epoch,elapsed,current=seed):
            (L/"progress.json").write_text(json.dumps({"seed":current,"epoch":epoch,"completed_fits":len(fits),"maximum_fits":10,"pid":os.getpid(),"seconds":time.monotonic()-start}),encoding="utf-8")
        model,f=b.core.fit_model(data,"v23_blockmask",seed,c,progress)
        path=M/f"model_seed{seed}.pt"
        with path.open("xb") as stream:
            torch.save(model.state_dict(),stream)
        normalized=b.core.predict_model(model,*b.core.arrays(frame))
        f.update(file=path.name,sha256=b.sha(path),normalized_sha256=ah(normalized),training_arrays_sha256=[ah(x) for x in data])
        fits.append(f)
        del model
        torch.cuda.empty_cache()
    save(M/"MODEL_MANIFEST.json",{"status":"TRAINED","run_kind":"FRESH_COLD","pid":os.getpid(),"fits":fits,"new_fits":10,"reused_fits":0,"seconds":time.monotonic()-start,"source_sha256":b.sha(source),"training_rows":len(frame),"mass":mass,"official_rows":0,"hidden_rows":0})


def trainqa(c,source):
    guard(source)
    start=time.monotonic()
    manifest=read(M/"MODEL_MANIFEST.json")
    assert manifest["pid"]!=os.getpid()
    assert [f["seed"] for f in manifest["fits"]]==c["seeds"]
    frame,y=b.read_observations(source,c,True)
    arrays,mass=b.core.training_arrays(frame,y,"v23_blockmask",c)
    checks=[]
    for f in manifest["fits"]:
        assert b.sha(M/f["file"])==f["sha256"]
        assert [ah(x) for x in arrays]==f["training_arrays_sha256"]
        model=b.core.make_model("v23_blockmask",11)
        model.load_state_dict(torch.load(M/f["file"],map_location="cpu",weights_only=True))
        norm=b.core.predict_model(model,*b.core.arrays(frame))
        if "normalized_sha256" in f:
            assert ah(norm)==f["normalized_sha256"]
        else:
            prediction=frame.baseline.to_numpy(float)+b.core.compute_profile_scale(frame)*norm
            assert ah(prediction)==f["absolute_prediction_sha256"]
        checks.append(f["seed"])
    save(L/"TRAIN_QA.json",{"status":"PASS","pid":os.getpid(),"training_pid":manifest["pid"],"model_manifest_sha256":b.sha(M/"MODEL_MANIFEST.json"),"models":len(checks),"rows_per_model":len(frame),"seconds":time.monotonic()-start,"official_rows":0,"hidden_rows":0})


def infer(c,source,replay):
    qa=read(L/"TRAIN_QA.json")
    assert qa["status"]=="PASS" and qa["model_manifest_sha256"]==b.sha(M/"MODEL_MANIFEST.json")
    assert read(P/"06_docs/internal-qa.json")["status"]=="PASS"
    guard(source,True)
    start=time.monotonic()
    sample=pd.read_csv(source.parent/"sample_submission.csv",usecols=b.KEYS)
    index=pd.read_csv(source.parent/"test_index.csv",usecols=b.KEYS)
    frame,_=b.read_observations(source,c,False)
    frame.index=b.canonical_keys(frame)
    q=frame.loc[b.canonical_keys(sample)].reset_index(drop=True)
    fits=read(M/"MODEL_MANIFEST.json")["fits"]
    predictions=[]
    for f in fits:
        assert b.sha(M/f["file"])==f["sha256"]
        model=b.core.make_model("v23_blockmask",11)
        model.load_state_dict(torch.load(M/f["file"],map_location="cpu",weights_only=True))
        norm=b.core.predict_model(model,*b.core.arrays(q))
        predictions.append(q.baseline.to_numpy(float)+b.core.compute_profile_scale(q)*norm)
    output=sample[b.KEYS].copy()
    output["temp"]=np.mean(predictions,axis=0)
    # Preserve historical L120 base serialization before the identical projection.
    output=pd.read_csv(io.StringIO(output.to_csv(index=False,float_format="%.12g",lineterminator="\n")),float_precision="round_trip")
    public=[]
    for chunk in pd.read_csv(source,usecols=["station","time","layer","temp"],chunksize=100000):
        public.append(chunk.loc[chunk.layer.isin([1,5,6,7,8])])
    ep=public_endpoint_frame(pd.concat(public,ignore_index=True))
    proj=project_profiles_vectorized(output,output.temp.to_numpy(float),ep)
    output["temp"]=proj.prediction
    checks=b.validate_output(output,sample,index,26061)
    payload=output.to_csv(index=False,lineterminator="\n").encode()
    digest=hashlib.sha256(payload).hexdigest()
    path=A/"submission_p2_L120_s10_proj.csv"
    if replay:
        assert b.sha(path)==digest
        assert read(L/"ANSWER_QA.json")["pid"]!=os.getpid()
    else:
        with path.open("xb") as stream:
            stream.write(payload)
    save(L/("REPLAY_QA.json" if replay else "ANSWER_QA.json"),{"status":"PASS","pid":os.getpid(),"rows":26061,"answer_sha256":digest,"answer_bytes":len(payload),"checks":checks,"projection":proj.diagnostics(),"seconds":time.monotonic()-start,"new_fits":0,"official_key_rows":26061,"sample_value_rows":0,"hidden_truth_rows":0,"uploads":0})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["train","trainqa","infer","replay"])
    args=parser.parse_args()
    c=settings()
    source=Path(os.environ["P2_DATA_DIR"]).resolve()/"observations.csv"
    torch.set_num_threads(2)
    with threadpool_limits(2):
        if args.mode=="train":
            train(c,source)
        elif args.mode=="trainqa":
            trainqa(c,source)
        else:
            infer(c,source,args.mode=="replay")
