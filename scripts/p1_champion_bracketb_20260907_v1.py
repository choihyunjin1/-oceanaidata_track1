"""Paired original-feature B80/B107 composition, new immutable CPU4 attempt."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"
ROOT = Path(__file__).resolve().parents[1]
ID = "p1_champion_bracketb_20260907_v1"
OUT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID
SOURCE = ROOT / "final_packages/P1/02_code"
sys.path.insert(0, str(SOURCE / "source/src"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for piece in iter(lambda: handle.read(1048576), b""):
            h.update(piece)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def runtime():
    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    from p1_qc.config import load_config
    from p1_qc.features import FeatureBundle, build_features
    from p1_qc.pipeline import TabularEncoder, _fit_model, apply_postprocess
    from p1_qc.rules import detect_plateaus, detect_singleton_spikes

    composition = module("bracketb_composition", SOURCE / "composition.py")
    weight = module("bracketb_weight", SOURCE / "event_weight.py")
    scope = {"np": np, "pd": pd}

    def segments(frame):
        t = pd.to_datetime(frame.time, utc=True)
        return (
            frame.station.ne(frame.station.shift())
            | frame.layer.ne(frame.layer.shift())
            | t.diff().ne(pd.Timedelta(minutes=10))
        ).cumsum()

    scope["old"] = SimpleNamespace(
        RAW=["station", "year", "layer", "time", "temp", "psal", "depth"],
        KEYS=composition.KEYS,
        segments=segments,
    )
    path = ROOT / "scripts/run_p1_bracket_forward_20260906_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "bracket_features"
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(
        np=np,
        pd=pd,
        joblib=joblib,
        lgb=lgb,
        load_config=load_config,
        FeatureBundle=FeatureBundle,
        build_features=build_features,
        TabularEncoder=TabularEncoder,
        apply_postprocess=apply_postprocess,
        _fit_model=_fit_model,
        detect_plateaus=detect_plateaus,
        detect_singleton_spikes=detect_singleton_spikes,
        composition=composition,
        weight=weight,
        scope=scope,
        bracket_features=scope["bracket_features"],
    )


def bundle(r, frame, extra, cfg):
    raw = frame[r.scope["old"].RAW].reset_index(drop=True)
    base = r.build_features(raw, config=r.load_config(SOURCE / "tree_config.toml", env={}))
    if len(base.feature_columns) != 80:
        raise ValueError("Original feature count drift")
    if not extra:
        return base
    order = raw.sort_values(["station", "layer", "time"], kind="stable").index.to_numpy()
    extra_frame = r.bracket_features(raw.iloc[order].reset_index(drop=True), cfg)
    extra_frame.index = order
    features = r.pd.concat([base.frame, extra_frame.reindex(raw.index)], axis=1)
    if features.shape[1] != 107:
        raise ValueError("Bracket count drift")
    return r.FeatureBundle(features, tuple(features), base.categorical_columns)


def progress(result, stage, start):
    value = {
        "status": "RUNNING",
        "stage": stage,
        "pid": os.getpid(),
        "completed_fits": len(result["fits"]),
        "maximum_fits": 17,
        "seconds": time.monotonic() - start,
    }
    (OUT / "progress.json").write_text(json.dumps(value), encoding="utf-8")
    print(json.dumps(value), flush=True)


def fit_arm(r, train, valid, arm, cfg, result, fold, start):
    features = bundle(r, train, arm == "bracket", cfg)
    encoder = r.TabularEncoder().fit(features, r.np.arange(len(train)))
    x = encoder.transform(features)
    xp = encoder.transform(bundle(r, valid, arm == "bracket", cfg))
    recipe = read(SOURCE / "tree_recipe.json")
    y = train.label.to_numpy(dtype=r.np.int8)
    weights = r.weight._event_day_weight(train[r.composition.KEYS], y)
    models, probabilities = [], []
    seeds = cfg["seeds"][:1] if arm == "O" else cfg["seeds"]
    for seed in seeds:
        progress(result, f"{fold}/{arm}/{seed}/FIT_STARTED", start)
        clock = time.monotonic()
        if arm == "O":
            params = read(
                ROOT / "scripts/p1_champion_reconstruction_20260906_v1/tree-contract.json"
            )["xgboost_parameters"]
            model = r._fit_model("xgboost", params, seed, 4, x, y)
        else:
            model = r.lgb.LGBMClassifier(
                **recipe["B_parameters"],
                objective="binary",
                random_state=seed,
                n_jobs=4,
                verbosity=-1,
                deterministic=True,
                force_row_wise=True,
                feature_fraction_seed=seed,
                bagging_seed=seed,
                data_random_seed=seed,
                extra_seed=seed,
            )
            model.fit(x, y, sample_weight=weights)
        p = model.predict_proba(xp)[:, 1]
        if not r.np.isfinite(p).all():
            raise ValueError("Nonfinite probabilities")
        model_file = OUT / "models" / f"{fold}_{arm}_{seed}.joblib"
        r.joblib.dump(model, model_file, compress=3)
        reloaded = r.joblib.load(model_file)
        if not r.np.array_equal(reloaded.predict_proba(xp)[:, 1], p):
            raise ValueError("Saved probability replay mismatch")
        result["fits"].append(
            {
                "fold": fold,
                "arm": arm,
                "seed": seed,
                "train_rows": len(train),
                "features": len(features.feature_columns),
                "trees": 700,
                "model_file": model_file.name,
                "model_sha256": sha(model_file),
                "seconds": time.monotonic() - clock,
            }
        )
        models.append({"seed": seed, "global": model})
        probabilities.append(p)
    saved = {
        "encoder": encoder,
        "packages": models,
        "seeds": seeds,
        "postprocess": cfg["postprocess"],
        "bracket": arm == "bracket",
    }
    r.joblib.dump(saved, OUT / "models" / f"{fold}_{arm}.joblib", compress=3)
    del features, x, xp, models
    gc.collect()
    return r.np.mean(r.np.vstack(probabilities), axis=0)


def compose(r, frame, op, bp, proposal, cfg):
    plateau = r.detect_plateaus(frame).to_numpy()
    spike = r.detect_singleton_spikes(frame).to_numpy()
    o = r.apply_postprocess(frame, op, plateau, spike, cfg["postprocess"])
    b = r.apply_postprocess(frame, bp, plateau, spike, cfg["postprocess"])
    router, gi = r.composition.compose_tree(
        frame.station,
        frame.layer,
        o,
        b,
        add_cells=r.composition.ADD_CELLS,
        remove_cells=r.composition.REMOVE_CELLS,
    )
    types = r.np.full(len(frame), "", dtype=object)
    types[plateau & b.astype(bool)] = "flatline"
    types[spike & b.astype(bool)] = "spike"
    added = (gi == 1) & (b == 0)
    types[added & plateau] = "flatline"
    types[added & ~plateau & spike & (op >= cfg["postprocess"]["high_threshold"])] = "spike"
    return r.composition.compose_mstcn_spike(router, gi, proposal, types)


def run(data):
    cfg = read(ROOT / "configs/experiments" / f"{ID}.json")
    r = runtime()
    OUT.mkdir(exist_ok=False)
    (OUT / "models").mkdir()
    result = {
        "id": ID,
        "status": "RUNNING",
        "pid": os.getpid(),
        "fits": [],
        "official_rows": 0,
        "hidden_rows": 0,
        "uploads": 0,
        "config_sha256": sha(ROOT / "configs/experiments" / f"{ID}.json"),
        "runner_sha256": sha(__file__),
        "evaluation_scope": cfg["primary"],
    }
    write(OUT / "ATTEMPT_LOCK.json", result)
    owned = [
        Path(__file__),
        ROOT / "configs/experiments" / f"{ID}.json",
        ROOT / "scripts/run_p1_bracket_forward_20260906_v1.py",
        ROOT / "scripts/p1_champion_reconstruction_20260906_v1/tree-contract.json",
    ]
    owned += [p for p in SOURCE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    pins = {str(p.relative_to(ROOT)): sha(p) for p in owned}
    write(OUT / "source-seal.json", pins)
    start = time.monotonic()
    try:
        progress(result, "input_and_lineage_QA", start)
        if sha(data / "train.csv") != cfg["train_sha256"]:
            raise ValueError("Training input drift")
        proposal_root = (
            ROOT / "artifacts/p1_champion_reconstruction_20260906_v1_historical_proposals"
        )
        for file, pin in [
            ("proposals.parquet", "proposal_sha256"),
            ("qa.json", "proposal_qa_sha256"),
        ]:
            if sha(proposal_root / file) != cfg[pin]:
                raise ValueError("MS historical source drift")
        frame = r.pd.read_csv(data / "train.csv", usecols=r.scope["old"].RAW + ["label"])
        frame.time = r.pd.to_datetime(frame.time, utc=True)
        frame = frame.sort_values(["station", "layer", "time"], kind="stable").reset_index(
            drop=True
        )
        if len(frame) != 776706 or frame.duplicated(r.composition.KEYS).any():
            raise ValueError("Training key population mismatch")
        proposal = r.pd.read_parquet(proposal_root / "proposals.parquet")
        proposal.time = r.pd.to_datetime(proposal.time, utc=True)
        contract = read(
            ROOT / "reports/p1_champion_reconstruction_20260906_v1/union-contract-v2.json"
        )
        parts, offset = [], 0
        index = r.pd.MultiIndex.from_frame(frame[r.composition.KEYS])
        for fold in contract["folds"]:
            source = proposal.iloc[offset : offset + fold["source_rows"]].copy()
            offset += fold["source_rows"]
            ids = index.get_indexer(r.pd.MultiIndex.from_frame(source[r.composition.KEYS]))
            if (ids < 0).any() or len(set(ids)) != len(ids):
                raise ValueError("MS key coverage failure")
            valid = frame.iloc[ids].reset_index(drop=True)
            # Features are partition-local and chronologically ordered; restore original MS order after fit.
            order = valid.sort_values(["station", "layer", "time"], kind="stable").index.to_numpy()
            valid = valid.iloc[order].reset_index(drop=True)
            train = frame.loc[frame.time.le(r.pd.Timestamp(fold["training_max_utc"]))].reset_index(
                drop=True
            )
            if train.time.max() >= valid.time.min() - r.pd.Timedelta(hours=337):
                raise ValueError("Dependency purge insufficient")
            op = fit_arm(r, train, valid, "O", cfg, result, fold["id"], start)
            bp = fit_arm(r, train, valid, "control", cfg, result, fold["id"], start)
            cp = fit_arm(r, train, valid, "bracket", cfg, result, fold["id"], start)
            part = valid[r.composition.KEYS + ["label"]].copy()
            part["fold"] = fold["id"]
            ms = source.proposal.to_numpy()[order]
            part["control"] = compose(r, valid, op, bp, ms, cfg)
            part["candidate"] = compose(r, valid, op, cp, ms, cfg)
            parts.append(part)
        paired = r.pd.concat(parts, ignore_index=True)
        if len(paired) != 287862 or paired.duplicated(r.composition.KEYS).any():
            raise ValueError("Entire historical MS key set required")
        paired.to_parquet(OUT / "paired.parquet", index=False)
        result["metrics"] = {
            arm: r.composition.metrics(paired.label, paired[arm])
            for arm in ("control", "candidate")
        }
        result["delta_f1"] = (
            result["metrics"]["candidate"]["f1"] - result["metrics"]["control"]["f1"]
        )
        result["by_fold"] = {
            str(fold): {
                arm: r.composition.metrics(p.label, p[arm]) for arm in ("control", "candidate")
            }
            for fold, p in paired.groupby("fold")
        }
        result["paired_sha256"] = sha(OUT / "paired.parquet")
        write(OUT / "internal-evaluation.json", result)
        progress(result, "full_B107_three_seed", start)
        full = r.pd.read_csv(data / "train.csv", usecols=r.scope["old"].RAW + ["label"])
        fit_arm(r, full, full.iloc[:256].copy(), "bracket", cfg, result, "full", start)
        result["status"] = "TRAIN_AND_INTERNAL_COMPLETE_QA_PENDING"
        if any(sha(ROOT / p) != value for p, value in pins.items()):
            raise ValueError("Source changed during attempt")
    except BaseException as error:
        result.update(status="TERMINAL_TECHNICAL_FAILURE", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        write(OUT / "terminal_result.json", result)
        write(REPORT / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    run(parser.parse_args().data.resolve())
