import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_sealed_seed_budget_and_no_timeout():
    cfg = json.loads((ROOT / "configs/experiments/p2_l120_s10_proj_20260907_v1.json").read_text())
    assert cfg["seeds"] == list(range(20260901, 20260911))
    assert cfg["maximum_new_fits"] == len(cfg["new_seeds"]) * 9 == 63
    assert cfg["hard_wallcap_seconds"] is None
    assert cfg["order"][0] == "B3"


def test_frozen_core_synthetic_forward_and_gradient():
    spec = importlib.util.spec_from_file_location("synthetic_c3", ROOT / "scripts/portable_20260906/P2/02_code/core.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    model = core.make_model("v23_blockmask", 11)
    # Same public token/context tensor widths as core arrays; no source data.
    tokens = torch.zeros((8, 5, 8), requires_grad=True)
    mask = torch.ones((8, 5))
    context = torch.zeros((8, 11))
    result = model(tokens, mask, context)
    assert tuple(result.shape) == (8,)
    assert np.isfinite(result.detach().numpy()).all()
    result.sum().backward()


def test_endpoint_fallback_and_missing_t1_noop():
    spec = importlib.util.spec_from_file_location("s10_qa_test", ROOT / "scripts/p2_l120_s10_proj_20260907_v1_qa.py")
    qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa)
    frame = pd.DataFrame({"time":["2024-01-01"]*3,"layer":[2,3,4],"temp_1":[np.nan]*3,"temp_5":[np.nan]*3,"temp_6":[5.0]*3,"temp_7":[7.0]*3,"temp_8":[8.0]*3})
    ep=qa.endpoint_frame(frame)
    assert ep.temp_5.iloc[0]==5.0 and np.isnan(ep.temp_1.iloc[0])
    p=np.array([99.0,2.0,-20.0])
    result=qa.project_profiles_vectorized(frame,p,ep)
    np.testing.assert_array_equal(result.prediction,p)
    assert result.eligible_mask.sum()==0


def test_metric_is_pooled_sse_and_bootstrap_deterministic():
    spec = importlib.util.spec_from_file_location("s10_qa_metrics", ROOT / "scripts/p2_l120_s10_proj_20260907_v1_qa.py")
    qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa)
    y=np.zeros(20)
    a=np.ones(20)
    b=np.full(20,.5)
    metric=qa.compare(y,a,b)
    assert metric["s3_sse_C2"]==20 and metric["s10_sse_C2"]==5
    t=pd.date_range("2024-01-01",periods=20,tz="UTC")
    bs=qa.bootstrap(y,a,b,t)
    assert bs["P_improvement"]==1 and bs["CI90"]==[-.5,-.5]
