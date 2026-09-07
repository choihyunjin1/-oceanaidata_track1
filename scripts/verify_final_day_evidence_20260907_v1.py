"""Read-only source/OOF reconciliation; writes only aggregate audit evidence."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/final_day_evidence_check_20260907_v1.json"
OUT = ROOT / "reports/final_day_evidence_check_20260907_v1/result.json"


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or not len(y) or not np.isfinite([y, p]).all():
        raise ValueError("finite aligned nonempty metric arrays required")
    return float(np.sqrt(np.mean(np.square(y - p))))


def pava(values, increasing=True):
    """Independent arbitrary-length unit-weight implementation."""
    sign = 1 if increasing else -1
    blocks = []
    for value in values:
        blocks.append([float(value) * sign, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop()
            blocks[-1][0] += right[0]
            blocks[-1][1] += right[1]
    return np.concatenate([np.repeat(sign * total / count, count) for total, count in blocks])


def p2_check(cfg):
    path = ROOT / "artifacts/p2_c3_multiseed_completion_20260906_v1/evaluation.npz"
    receipt = ROOT / "artifacts/p2_c3_multiseed_completion_20260906_v1/terminal_result.json"
    assert sha(path) == json.loads(receipt.read_text())["evaluation_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        d = {key: archive[key] for key in archive.files}
    n = len(d["truth"])
    assert n == cfg["rows"] and len(np.unique(d["key"])) == n
    obs_path = Path(os.environ["P2_DATA_DIR"]) / "observations.csv"
    obs = pd.read_csv(obs_path, usecols=["station", "time", "layer", "temp"])
    assert obs.station.nunique() == 1
    # Only public layers supply endpoints. No hidden target values enter projection.
    public = obs.loc[obs.layer.isin([1, 5, 6, 7, 8])].copy()
    public["time"] = pd.to_datetime(public.time, utc=True)
    assert not public.duplicated(["time", "layer"]).any()
    wide = public.pivot(index="time", columns="layer", values="temp").reindex(columns=[1, 5, 6, 7, 8])
    deep = wide[[5, 6, 7, 8]].bfill(axis=1)[5]
    endpoint = pd.DataFrame({"top": wide[1], "deep": deep})
    frame = pd.DataFrame({"time": pd.to_datetime(d["time"], utc=True), "layer": d["layer"], "fold": d["fold"]})
    assert not frame.duplicated(["fold", "time", "layer"]).any()
    frame = frame.join(endpoint, on="time", validate="many_to_one")
    base = d["natural_L120"]
    variants = {key: base.copy() for key in ["clip_then_pava_complete", "clip_then_pava_available", "pava_then_clip_complete"]}
    counts = {"complete_profiles": 0, "incomplete_profiles": 0, "missing_endpoint_profiles": 0, "fallback_endpoint_rows": 0}
    for _, group in frame.groupby(["fold", "time"], sort=False):
        group = group.sort_values("layer")
        rows = group.index.to_numpy()
        top, bottom = group.iloc[0][["top", "deep"]]
        complete = len(group) == 3 and set(group.layer) == {2, 3, 4}
        counts["complete_profiles" if complete else "incomplete_profiles"] += 1
        if not np.isfinite([top, bottom]).all():
            counts["missing_endpoint_profiles"] += 1
            continue
        clipped = np.clip(base[rows], min(top, bottom), max(top, bottom))
        projected = pava(clipped, increasing=top <= bottom)
        variants["clip_then_pava_available"][rows] = projected
        if complete:
            variants["clip_then_pava_complete"][rows] = projected
            variants["pava_then_clip_complete"][rows] = np.clip(pava(base[rows], top <= bottom), min(top, bottom), max(top, bottom))
    metrics = {}
    for label in ["pooled", *sorted(np.unique(d["fold"]))]:
        mask = np.ones(n, bool) if label == "pooled" else d["fold"] == label
        metrics[label] = {"rows": int(mask.sum()), "baseline": rmse(d["truth"][mask], base[mask])}
        for key, pred in variants.items():
            metrics[label][key] = rmse(d["truth"][mask], pred[mask])
    matches = {key: all(abs(metrics[k]["baseline"] - pair[0]) <= cfg["rounded_metric_tolerance"] and abs(metrics[k][key] - pair[1]) <= cfg["rounded_metric_tolerance"] for k, pair in cfg["expected_rmse"].items()) for key in variants}
    return {"oof_sha256": sha(path), "observations_sha256": sha(obs_path), "counts": counts, "metrics": metrics, "numeric_matches_by_explicit_variant": matches, "historical_exposed_oof_not_fresh": True, "fits": 0}


def cluster_bootstrap(frame, base, pred, settings):
    y = frame.target_hs.to_numpy()
    stats = pd.DataFrame({"station": frame.station, "episode": frame.episode_id, "b": (base-y)**2, "c": (pred-y)**2, "n": 1}).groupby(["station", "episode"], sort=True)[["b", "c", "n"]].sum().to_numpy()
    rng = np.random.default_rng(settings["seed"])
    results = []
    for _ in range(settings["resamples"]):
        s = stats[rng.integers(0, len(stats), size=len(stats))].sum(axis=0)
        results.append(np.sqrt(s[1]/s[2])-np.sqrt(s[0]/s[2]))
    return {"clusters": len(stats), "resamples": settings["resamples"], "seed": settings["seed"], "ci90": np.quantile(results, settings["ci_quantiles"]).tolist(), "p_improve": float(np.mean(np.array(results)<0))}


def p3_check(cfg):
    path = ROOT / "artifacts/p3_numeric_lead_forward_gpu_20260906_v2/candidate_oof.parquet"
    expected = json.loads((ROOT / "reports/p3_numeric_lead_forward_gpu_20260906_v2/result.json").read_text())["artifacts"][path.name]
    assert sha(path) == expected
    frame = pd.read_parquet(path)
    assert len(frame) == cfg["rows"] and not frame.duplicated(["station", "anchor_id", "lead_h"]).any()
    y, base, persistence = (frame[k].to_numpy(float) for k in ["target_hs", "final_prediction", "persistence"])
    assert np.isfinite([y, base, persistence]).all()
    long = frame.lead_h.isin(cfg["undo_shrink_leads"]).to_numpy()
    raw = base.copy()
    raw[long] = (base[long] - .2*persistence[long])/.8
    metrics = {}
    for w in cfg["weights"]:
        pred = raw.copy()
        pred[long] = (1-w)*raw[long]+w*persistence[long]
        metrics[str(w)] = {"pooled": rmse(y, pred), "long_leads": rmse(y[long], pred[long]), "hs0_lt_1p7": rmse(y[persistence<1.7], pred[persistence<1.7]), "bootstrap_vs_w02": cluster_bootstrap(frame, base, pred, cfg["bootstrap"])}
    ols = {}
    for lead in sorted(frame.lead_h.unique()):
        mask = frame.lead_h.eq(lead).to_numpy()
        target, pred, per = y[mask], raw[mask], persistence[mask]
        ols[str(lead)] = {"target_on_prediction_through_origin": float(pred@target/(pred@pred)), "change_from_persistence_through_origin": float((pred-per)@(target-per)/((pred-per)@(pred-per)))}
    return {"oof_sha256": sha(path), "rows": len(frame), "anchors": int(frame.anchor_id.nunique()), "metrics": metrics, "ols_definitions": ols, "pooled_numeric_match": abs(metrics['0.0']['pooled']-.6798)<=cfg['rounded_metric_tolerance'] and abs(metrics['0.2']['pooled']-.6835)<=cfg['rounded_metric_tolerance'], "ols_is_in_sample_not_promotion_evidence": True, "fits": 0}


def p1_check(cfg):
    root = Path(os.environ["P1_DATA_DIR"])
    train = pd.read_csv(root / "train.csv", usecols=["temp", "label"])
    test = pd.read_csv(root / "test.csv", usecols=["station", "year", "layer", "time", "temp"])
    answer_path = Path(os.environ["FINAL_RELEASE_DIR"]) / "P1/ANSWER/P1_submission.csv"
    assert sha(answer_path) == "57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687"
    answer = pd.read_csv(answer_path)
    keys = ["station", "year", "layer", "time"]
    assert test[keys].equals(answer[keys]) and not test.duplicated(keys).any()
    natural = train.loc[train.label.eq(0), "temp"]
    assert np.isfinite(natural).all()
    minimum, maximum = float(natural.min()), float(natural.max())
    results = {}
    for name, (lo, hi) in {"train_fitted_minmax": (minimum, maximum), "old_report_literal_0_35_diagnostic_only": (0, 35)}.items():
        outside_train = (train.temp<lo)|(train.temp>hi)
        outside_test = (test.temp<lo)|(test.temp>hi)
        results[name] = {"minimum": lo, "maximum": hi, "train_label0_outside": int((outside_train&train.label.eq(0)).sum()), "train_label1_outside": int((outside_train&train.label.eq(1)).sum()), "test_outside": int(outside_test.sum()), "test_outside_already_positive": int((outside_test&answer.label.eq(1)).sum()), "new_positive_predictions_not_proven_TP": int((outside_test&answer.label.eq(0)).sum())}
    return {"train_sha256": sha(root/'train.csv'), "test_sha256": sha(root/'test.csv'), "answer_sha256": sha(answer_path), "rules": results, "requested_fitted_rule_matches": all(results['train_fitted_minmax'][k]==v for k,v in cfg['expected_counts'].items()), "test_truth_not_available_FP_zero_not_proven": True, "train_label0_zero_outside_is_tautological_for_fitted_extrema": True, "fits": 0}


def main():
    if OUT.exists():
        raise FileExistsError("preserve existing audit")
    started = time.monotonic()
    cfg = json.loads(CONFIG.read_text())
    result = {"config_sha256": sha(CONFIG), "script_sha256": sha(__file__), "pid": os.getpid(), "scope": "aggregate_recalculation_only", "new_model_fits": 0, "candidate_csv_written": 0, "uploads": 0, "hidden_truth_access": 0}
    for name, function in [('P2', p2_check), ('P3', p3_check), ('P1', p1_check)]:
        result[name] = function(cfg[name.lower()])
        print(json.dumps({"completed": name, "seconds": time.monotonic()-started}), flush=True)
    result['runtime_seconds'] = time.monotonic()-started
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
