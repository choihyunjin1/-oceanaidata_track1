"""Independent prefix-sum time-window and scalar PAVA check of smoothing claims."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from p2_c60_l120_equal_projection_20260907_v1 import ROOT, compare, endpoints, r
from p2_final_day_projection_20260907_v1 import project_profiles, project_profiles_vectorized
from p2_l120_s3_smooth7_projection_20260907_v1 import smooth7
from threadpoolctl import threadpool_limits


def prefix_reference(frame, values):
    out = np.empty(len(frame))
    f=frame[["station","layer","time"]].copy()
    f["pos"]=np.arange(len(f))
    for _,part in f.groupby(["station","layer"],sort=False):
        part=part.sort_values("time")
        ix=part.pos.to_numpy()
        stamps=pd.DatetimeIndex(part.time).as_unit("ns").asi8
        lo=np.searchsorted(stamps,stamps-30*60*10**9,side="left")
        hi=np.searchsorted(stamps,stamps+30*60*10**9,side="right")
        sums=np.r_[0.,np.cumsum(values[ix],dtype=np.longdouble)]
        out[ix]=(sums[hi]-sums[lo])/(hi-lo)
    return out


def main():
    start=time.monotonic()
    report=ROOT / "reports/p2_l120_s3_smooth7_projection_20260907_v1"
    old=r.read(report / "result.json")
    assert old["pid"]!=os.getpid()
    frame,y=r.load_source(r.cfg())
    with np.load(ROOT / "artifacts/p2_c3_multiseed_completion_20260906_v1/evaluation.npz",allow_pickle=False) as z:
        assert np.array_equal(z["key"],r.key_values(frame)) and np.array_equal(z["truth"],y)
        p=z["natural_L120"].copy()
        folds=z["fold"].copy()
    local=np.empty(len(y))
    max_smoothing_error=0.
    max_projection_error=0.
    for fold in np.unique(folds):
        mask=folds==fold
        q=frame.loc[mask].reset_index(drop=True)
        sm=smooth7(q,p[mask])
        ref=prefix_reference(q,p[mask])
        max_smoothing_error=max(max_smoothing_error,float(np.max(np.abs(sm-ref))))
        assert np.allclose(sm,ref,atol=1e-10,rtol=0)
        ep=endpoints(q)
        scalar=project_profiles(q,sm,ep)
        vector=project_profiles_vectorized(q,sm,ep)
        max_projection_error=max(max_projection_error,float(np.max(np.abs(scalar.prediction-vector.prediction))))
        assert np.array_equal(scalar.eligible_mask,vector.eligible_mask)
        assert np.allclose(scalar.prediction,vector.prediction,atol=1e-12,rtol=0)
        base=project_profiles_vectorized(q,p[mask],ep).prediction
        assert compare(y[mask],base,vector.prediction)==old["metrics"][str(fold)]["natural"]
        local[mask]=vector.prediction
    # A pooled smoothing diagnostic (NOT deployment/evaluation candidate) tests cross-fold boundary effects.
    global_sm=smooth7(frame,p)
    global_projected=project_profiles_vectorized(frame,global_sm,endpoints(frame)).prediction
    b3=folds=="B3"
    diag={"cross_fold_global_B3_rmse_C":float(np.sqrt(np.mean((global_projected[b3]-y[b3])**2))),
          "fold_local_B3_rmse_C":float(np.sqrt(np.mean((local[b3]-y[b3])**2))),
          "cross_fold_global_pooled_rmse_C":float(np.sqrt(np.mean((global_projected-y)**2))),
          "fold_local_pooled_rmse_C":float(np.sqrt(np.mean((local-y)**2))),
          "rows_different":int(np.sum(np.abs(global_projected-local)>1e-12)),
          "note":"Global diagnostic permits neighboring predictions across outer fold boundaries; final assessment uses fold-local smoothing only. Fable exact implementation was not supplied, so similarity alone does not prove the source of its difference."}
    out={"status":"PASS","pid":os.getpid(),"result_sha256":r.sha(report / "result.json"),
         "prefix_vs_rolling_max_abs_C":max_smoothing_error,"scalar_vs_vector_PAVA_max_abs_C":max_projection_error,
         "scope": "all 166268 natural OOF rows; outage covered by separate full aggregate replay",
         "boundary_diagnostic":diag,"new_fits":0,"official_rows":0,"hidden_rows":0,"seconds":time.monotonic()-start}
    r.save(report / "independent-qa.json",out)
    print(json.dumps(out),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
