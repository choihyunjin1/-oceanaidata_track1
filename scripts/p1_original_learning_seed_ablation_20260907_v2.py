"""Nine new fits: original O-slow and B+2seed changes, disjoint and immutable."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import p1_champion_bracketb_20260907_v1 as m

ID = "p1_original_learning_seed_ablation_20260907_v2"
OUT = m.ROOT / "artifacts" / ID
REPORT = m.ROOT / "reports" / ID
CONFIG = m.ROOT / "configs/experiments" / (ID + ".json")


def progress(fits, stage, start):
    value = {"pid": os.getpid(), "stage": stage, "completed_fits": len(fits),
             "maximum_fits": 9, "seconds": time.monotonic()-start}
    (OUT / "progress.json").write_text(json.dumps(value), encoding="utf-8")
    print(json.dumps(value), flush=True)


def fit(r, train, valid, arm, seed, cfg, fits, fold, start):
    progress(fits, f"{fold}/{arm}/{seed}", start)
    begin = time.monotonic()
    features = m.bundle(r, train, False, cfg)
    encoder = r.TabularEncoder().fit(features, r.np.arange(len(train)))
    x, vx = encoder.transform(features), encoder.transform(m.bundle(r, valid, False, cfg))
    y = train.label.to_numpy(r.np.int8)
    if arm == "O_slow":
        params = m.read(m.ROOT / "scripts/p1_champion_reconstruction_20260906_v1/tree-contract.json")["xgboost_parameters"]
        params.update(learning_rate=.02, n_estimators=1400)
        model = r._fit_model("xgboost", params, seed, 4, x, y)
        assert model.model.get_booster().num_boosted_rounds() == 1400
    else:
        params = m.read(m.SOURCE / "tree_recipe.json")["B_parameters"]
        model = r.lgb.LGBMClassifier(**params, objective="binary", random_state=seed, n_jobs=4,
            verbosity=-1, deterministic=True, force_row_wise=True, feature_fraction_seed=seed,
            bagging_seed=seed, data_random_seed=seed, extra_seed=seed)
        model.fit(x, y, sample_weight=r.weight._event_day_weight(train[r.composition.KEYS], y))
        assert model.booster_.num_trees() == 700
    prediction = model.predict_proba(vx)[:, 1]
    package = {"encoder": encoder, "packages": [{"seed": seed, "global": model}], "seeds": [seed],
               "postprocess": cfg["postprocess"], "bracket": False}
    path = OUT / "models" / f"{fold}_{arm}_{seed}.joblib"
    r.joblib.dump(package, path, compress=3)
    replay = r.joblib.load(path)["packages"][0]["global"].predict_proba(vx)[:, 1]
    assert r.np.array_equal(prediction, replay)
    fits.append({"fold":fold,"arm":arm,"seed":seed,"model":path.name,"sha256":m.sha(path),
                 "training_rows":len(train),"features":80,"parameters":params,
                 "seconds":time.monotonic()-begin})
    del x, vx, features, model, package
    gc.collect()
    return prediction


def saved_predict(r, valid, fold, arm, cfg):
    package = r.joblib.load(m.OUT / "models" / f"{fold}_{arm}.joblib")
    x = package["encoder"].transform(m.bundle(r, valid, False, cfg))
    return r.np.mean([p["global"].predict_proba(x)[:,1] for p in package["packages"]], axis=0)


def synthetic():
    r = m.runtime()
    cfg = m.read(CONFIG)
    assert cfg["changes"] == {"O_slow":{"learning_rate":.02,"n_estimators":1400},"B5":{"additional_seeds":[20260861,20260875]}}
    x = r.np.random.default_rng(7).normal(size=(64, 4))
    y = (x[:,0]>0).astype(int)
    params = m.read(m.ROOT / "scripts/p1_champion_reconstruction_20260906_v1/tree-contract.json")["xgboost_parameters"]
    params.update(learning_rate=.02,n_estimators=2)
    model=r._fit_model("xgboost",params,20260813,4,x,y)
    assert model.model.get_booster().num_boosted_rounds() == 2
    assert model.model.get_params()["learning_rate"] == .02
    assert r.np.isfinite(model.predict_proba(x)).all()
    print("SYNTHETIC_PASS: 1 tiny XGBoost compatibility fit; production fits 0")


def run(data):
    r = m.runtime()
    cfg = m.read(m.ROOT / "configs/experiments/p1_champion_bracketb_20260907_v1.json")
    c = m.read(CONFIG)
    assert c["new_fits"] == 9
    assert m.sha(data / "train.csv") == cfg["train_sha256"]
    qa = m.read(m.OUT / "independent-qa.json")
    assert qa["status"] == "PASS" and all(qa["checks"].values())
    old = m.read(m.OUT / "terminal_result.json")
    for item in old["fits"]:
        assert m.sha(m.OUT / "models" / item["model_file"]) == item["model_sha256"]
    inherited_pins = m.read(m.OUT / "source-seal.json")
    assert all(m.sha(m.ROOT / p)==h for p,h in inherited_pins.items())
    # Combined wrappers used for inference are sealed here as well as their individual model receipts.
    pins={str(p):m.sha(p) for p in [CONFIG,Path(__file__),*(m.OUT / "models").glob("*.joblib")]}
    OUT.mkdir(exist_ok=False)
    (OUT / "models").mkdir()
    m.write(OUT / "ATTEMPT_LOCK.json", {"pid":os.getpid(),"new_fits":9,"started_unix":time.time()})
    m.write(OUT / "source-seal.json", pins)
    start=time.monotonic()
    result={"id":ID,"pid":os.getpid(),"fits":[],"new_fits":0,"official_rows":0,"hidden_rows":0}
    try:
        frame=r.pd.read_csv(data / "train.csv",usecols=r.scope["old"].RAW+["label"])
        frame.time=r.pd.to_datetime(frame.time,utc=True)
        frame=frame.sort_values(["station","layer","time"],kind="stable").reset_index(drop=True)
        assert len(frame)==776706 and not frame.duplicated(r.composition.KEYS).any()
        path=m.ROOT / "artifacts/p1_champion_reconstruction_20260906_v1_historical_proposals/proposals.parquet"
        assert m.sha(path)==cfg["proposal_sha256"]
        proposal=r.pd.read_parquet(path)
        proposal.time=r.pd.to_datetime(proposal.time,utc=True)
        contract=m.read(m.ROOT / "reports/p1_champion_reconstruction_20260906_v1/union-contract-v2.json")
        index=r.pd.MultiIndex.from_frame(frame[r.composition.KEYS])
        parts=[]
        offset=0
        for fold in contract["folds"]:
            source=proposal.iloc[offset:offset+fold["source_rows"]]
            offset+=fold["source_rows"]
            ids=index.get_indexer(r.pd.MultiIndex.from_frame(source[r.composition.KEYS]))
            assert (ids>=0).all() and len(set(ids))==len(ids)
            valid=frame.iloc[ids].reset_index(drop=True)
            order=valid.sort_values(["station","layer","time"],kind="stable").index.to_numpy()
            valid=valid.iloc[order].reset_index(drop=True)
            train=frame.loc[frame.time.le(r.pd.Timestamp(fold["training_max_utc"]))].reset_index(drop=True)
            assert train.time.max() < valid.time.min()-r.pd.Timedelta(hours=337)
            op,bp=[saved_predict(r,valid,fold["id"],arm,cfg) for arm in ["O","control"]]
            slow=fit(r,train,valid,"O_slow",20260813,cfg,result["fits"],fold["id"],start)
            extra=[fit(r,train,valid,"B_extra",seed,cfg,result["fits"],fold["id"],start) for seed in [20260861,20260875]]
            b5=(bp*3+extra[0]+extra[1])/5
            ms=source.proposal.to_numpy()[order]
            part=valid[r.composition.KEYS+["label"]].copy()
            part["fold"]=fold["id"]
            for arm,a,b in [("control",op,bp),("O_slow",slow,bp),("B5",op,b5)]:
                part[arm]=m.compose(r,valid,a,b,ms,cfg)
            parts.append(part)
        paired=r.pd.concat(parts,ignore_index=True)
        control=r.pd.read_parquet(m.OUT / "paired.parquet")
        assert len(paired)==287862 and paired[r.composition.KEYS].equals(control[r.composition.KEYS])
        assert r.np.array_equal(paired.control,control.control)
        paired.to_parquet(OUT / "paired.parquet",index=False)
        result["metrics"]={a:r.composition.metrics(paired.label,paired[a]) for a in ["control","O_slow","B5"]}
        result["by_fold"]={str(f):{a:r.composition.metrics(p.label,p[a]) for a in ["control","O_slow","B5"]} for f,p in paired.groupby("fold")}
        result["paired_sha256"]=m.sha(OUT / "paired.parquet")
        # Full fits use original unsorted input order, matching original deployment training semantics.
        full=r.pd.read_csv(data / "train.csv",usecols=r.scope["old"].RAW+["label"])
        fit(r,full,full.iloc[:256].copy(),"O_slow",20260813,cfg,result["fits"],"full",start)
        for seed in [20260861,20260875]:
            fit(r,full,full.iloc[:256].copy(),"B_extra",seed,cfg,result["fits"],"full",start)
        assert len(result["fits"])==9 and all(m.sha(Path(p))==h for p,h in pins.items())
        result["status"]="TRAIN_INTERNAL_COMPLETE_QA_PENDING"
    except BaseException as error:
        result.update(status="TERMINAL_TECHNICAL_FAILURE",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["new_fits"]=len(result["fits"])
        result["seconds"]=time.monotonic()-start
        m.write(OUT / "terminal_result.json",result)
        m.write(REPORT / "result.json",result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["synthetic","run"])
    parser.add_argument("--data",type=Path)
    args=parser.parse_args()
    synthetic() if args.mode=="synthetic" else run(args.data.resolve())
