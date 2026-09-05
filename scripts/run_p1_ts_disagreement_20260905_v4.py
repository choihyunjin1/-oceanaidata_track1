"""One clean T/S feature ablation, six historical fits; no official inputs."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import joblib
import numpy as np
import pandas as pd
import run_p1_depth_contract_repair_20260905_v2 as depth
import run_p1_score_repair_20260905_v1 as old

ID = "p1_ts_disagreement_20260905_v4"
CONFIG = ROOT / f"configs/experiments/{ID}.json"
OUT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID


def fit_epsilon(training):
    seg = old.segments(training)
    delta = training.psal.groupby(seg, sort=False).diff().abs()
    positive = delta[np.isfinite(delta) & (delta > 0)]
    return max(1e-6, float(positive.median())) if len(positive) else 1e-6


def ts_features(observed, epsilon, windows=(6, 36, 144)):
    """No labels or learned evaluation statistics; missing salinity is unknown."""
    if set(observed) != set(old.RAW):
        raise ValueError("raw observation columns only")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive training epsilon required")
    frame = observed.reset_index(drop=True)
    seg = old.segments(frame)
    output = pd.DataFrame(index=frame.index)
    for positions in frame.groupby(seg, sort=False).indices.values():
        part = frame.iloc[positions].reset_index(drop=True)
        t, s = part.temp.astype(float), part.psal.astype(float)
        dt, ds = t.diff(), s.diff()
        local = pd.DataFrame(index=part.index)
        local["ts_jump_prev"] = dt.abs() / (ds.abs() + epsilon)
        local["ts_jump_next"] = dt.shift(-1).abs() / (ds.shift(-1).abs() + epsilon)
        local["ts_salinity_step_prev"] = ds.abs()
        local["ts_salinity_step_next"] = ds.shift(-1).abs()
        for window in windows:
            minimum = max(3, int(np.ceil(window / 4)))
            rt = t.rolling(window, center=True, min_periods=minimum)
            rs = s.rolling(window, center=True, min_periods=minimum)
            dts = dt.rolling(window, center=True, min_periods=minimum).std()
            dss = ds.rolling(window, center=True, min_periods=minimum).std()
            local[f"ts_rough_ratio_{window}"] = dts / (dss + epsilon)
            local[f"ts_salinity_std_{window}"] = rs.std()
            local[f"ts_corr_{window}"] = rt.corr(s)
            # Use identical nonmissing pairs for all regression moments.
            tp, sp = t.where(s.notna()), s.where(t.notna())
            rp = tp.rolling(window, center=True, min_periods=minimum)
            rq = sp.rolling(window, center=True, min_periods=minimum)
            slope = rp.cov(sp) / (rq.var() + epsilon**2)
            local[f"ts_residual_{window}"] = t - (rp.mean() + slope * (s - rq.mean()))
            local[f"ts_diff_ac1_{window}"] = dt.rolling(
                window, center=True, min_periods=minimum
            ).corr(dt.shift())
        for name in local:
            if name not in output:
                output[name] = np.nan
            output.loc[positions, name] = local[name].to_numpy()
    return output.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def bundle(frame, stats, frozen, epsilon=None):
    base = depth.features(frame, stats, frozen, current_depth=False)
    if epsilon is None:
        return base
    extra = ts_features(frame[old.RAW], epsilon)
    data = pd.concat([base.frame, extra], axis=1)
    return old.FeatureBundle(data, tuple(data), base.categorical_columns)


def counts(target, prediction):
    y, p = np.asarray(target), np.asarray(prediction)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("binary aligned counts required")
    tp = int(np.count_nonzero((y == 1) & (p == 1)))
    fp = int(np.count_nonzero((y == 0) & (p == 1)))
    fn = int(np.count_nonzero((y == 1) & (p == 0)))
    return {"rows": len(y), "tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / max(1, 2 * tp + fp + fn)}


def load_contract():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (
        cfg["experiment_id"] != ID
        or cfg["official_access"]
        or cfg["max_historical_fits"] != 6
        or cfg["windows_rows"] != [6, 36, 144]
    ):
        raise ValueError("experiment boundary")
    for name, expected in cfg["pinned_inputs"].items():
        if old.sha(ROOT / name) != expected:
            raise ValueError("pinned dependency changed: " + name)
    source = ROOT / cfg["source_screen"]
    prior = json.loads((source / "terminal_result.json").read_text(encoding="utf-8"))
    frozen = json.loads((source / "contract.json").read_text(encoding="utf-8"))
    if old.sha(old.__file__) != prior["runner_sha256"]:
        raise ValueError("base runner changed")
    for name, expected in {**prior["dependency_hashes"], **prior["recipe_hashes"]}.items():
        if old.sha(ROOT / name) != expected:
            raise ValueError("base dependency changed: " + name)
    for fit in prior["fit_receipts"]:
        if fit["model"] in depth.MODELS:
            name = f"{fit['fold']}_{fit['stage']}_{fit['model']}.joblib"
            if old.sha(source / name) != fit["model_sha256"]:
                raise ValueError("base model changed: " + name)
    return cfg, source, prior, frozen


def load_training(frozen):
    path = Path(os.environ["P1_DATA_DIR"]).resolve() / "train.csv"
    if old.sha(path) != frozen["expected_training_sha256"]:
        raise ValueError("training hash mismatch")
    frame = pd.read_csv(path, usecols=old.RAW + ["label", "anomaly_type"])
    frame["row_id"] = np.arange(len(frame))
    if frame.duplicated(old.KEYS).any() or not frame.label.isin([0, 1]).all():
        raise ValueError("training schema invalid")
    return frame.sort_values(["station", "layer", "time"], kind="stable").reset_index(drop=True)


def evaluate(frame, probabilities, rules, frozen, selection):
    name, calibrations = selection
    return depth.policies(frame, probabilities, rules, frozen, calibrations)[name]


def execute():
    cfg, source, prior, frozen = load_contract()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": ID,
        "status": "RUNNING",
        "pid": os.getpid(),
        "fits": [],
        "folds": [],
        "official_rows": 0,
        "csv_written": 0,
        "uploads": 0,
        "full_fits": 0,
        "expected_official_score": None,
        "baseline_regeneration_check": "LINEAGE_VERIFIED_NEW_EMPTY_MODEL_DIRECTORY_CHECK_PENDING",
        "config_sha256": old.sha(CONFIG),
        "runner_sha256": old.sha(__file__),
        "surface": "repeated_historical_development_not_fresh_confirmation",
    }
    with (OUT / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle)
    started = time.monotonic()

    def progress(stage):
        elapsed = time.monotonic() - started
        obj = {
            "stage": stage,
            "completed_fits": len(result["fits"]),
            "max_fits": 6,
            "elapsed_seconds": elapsed,
            "pid": os.getpid(),
        }
        old.write_json(OUT / "progress.json", obj)
        print(json.dumps(obj), flush=True)
        if elapsed > cfg["wall_cap_seconds"]:
            raise RuntimeError("resource budget exceeded, preserve attempt")

    try:
        data = load_training(frozen)
        times = pd.to_datetime(data.time, utc=True)
        recipe = json.loads((ROOT / frozen["lightgbm_recipe"]).read_text(encoding="utf-8"))
        all_rows = []
        for fold, previous in zip(frozen["folds"], prior["folds"], strict=True):
            name, begin, end = fold["name"], pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
            if name != previous["fold"]:
                raise ValueError("fold alignment")
            cutoff = begin - pd.Timedelta(days=frozen["purge_days"])
            inner = cutoff - pd.Timedelta(days=frozen["inner_days"])
            selections = None
            fold_result = {"fold": name, "surfaces": {}}
            for stage, start, stop, train_stop in [
                ("inner", inner, cutoff, inner - pd.Timedelta(days=frozen["purge_days"])),
                ("outer", begin, end, cutoff),
            ]:
                training = old.train_slice(data, train_stop)
                ev = data.loc[(times >= start) & (times < stop)].reset_index(drop=True)
                if set(training.row_id) & set(ev.row_id):
                    raise ValueError("training evaluation overlap")
                progress(f"{name}/{stage}/FEATURES")
                package = joblib.load(source / f"{name}_{stage}_balanced.joblib")
                stats, eps = package["train_stats"], fit_epsilon(training)
                base_ev = bundle(ev, stats, frozen)
                controls = {}
                for model_name in depth.MODELS:
                    loaded = joblib.load(source / f"{name}_{stage}_{model_name}.joblib")
                    controls[model_name] = loaded["model"].predict_proba(
                        loaded["encoder"].transform(base_ev)
                    )[:, 1]
                x_bundle = bundle(training, stats, frozen, eps)
                encoder = old.TabularEncoder().fit(x_bundle, np.arange(len(training)))
                x = encoder.transform(x_bundle)
                xe = encoder.transform(bundle(ev, stats, frozen, eps))
                params = old._lgb_parameters(recipe, cfg["seed"], multiclass=False)
                params["n_jobs"] = cfg["threads"]
                model = old.lgb.LGBMClassifier(**params)
                fit_start = time.monotonic()
                progress(f"{name}/{stage}/FIT")
                model.fit(
                    x,
                    training.label,
                    sample_weight=old._event_day_weight(training, training.label.to_numpy()),
                )
                pred = model.predict_proba(xe)[:, 1]
                if not np.isfinite(pred).all():
                    raise ValueError("nonfinite candidate probability")
                model_path = OUT / f"{name}_{stage}.joblib"
                joblib.dump(
                    {
                        "model": model,
                        "encoder": encoder,
                        "stats": stats,
                        "epsilon": eps,
                        "frozen": frozen,
                    },
                    model_path,
                    compress=3,
                )
                result["fits"].append(
                    {
                        "fold": name,
                        "stage": stage,
                        "seconds": time.monotonic() - fit_start,
                        "training_rows": len(training),
                        "training_max": pd.to_datetime(training.time, utc=True).max().isoformat(),
                        "evaluation_min": start.isoformat(),
                        "features": x.shape[1],
                        "epsilon": eps,
                        "model_sha256": old.sha(model_path),
                    }
                )
                rules = old.rule_masks(ev, stats)
                candidate = {"original": controls["original"], "balanced": pred}
                if stage == "inner":
                    ctrl_sel = depth.select_inner(ev, controls, rules, frozen)
                    cand_sel = depth.select_inner(ev, candidate, rules, frozen)
                    if ctrl_sel[0] != previous["selected_control"]:
                        raise ValueError("control selection did not reproduce")
                    selections = (ctrl_sel, cand_sel)
                    fold_result["selections"] = {"control": ctrl_sel, "candidate": cand_sel}
                else:
                    for surface in ("intact", "fragmented"):
                        view, keep = (
                            (ev, np.ones(len(ev), dtype=bool))
                            if surface == "intact"
                            else old.fragmented(ev, frozen)
                        )
                        if surface == "intact":
                            c, p = controls, candidate
                        else:
                            basic = bundle(view, stats, frozen)
                            c = {}
                            for model_name in depth.MODELS:
                                loaded = joblib.load(source / f"{name}_outer_{model_name}.joblib")
                                c[model_name] = loaded["model"].predict_proba(
                                    loaded["encoder"].transform(basic)
                                )[:, 1]
                            p = {
                                "original": c["original"],
                                "balanced": model.predict_proba(
                                    encoder.transform(bundle(view, stats, frozen, eps))
                                )[:, 1],
                            }
                        r = old.rule_masks(view, stats)
                        cb, pb = [
                            evaluate(view, probs, r, frozen, sel)
                            for probs, sel in zip((c, p), selections, strict=True)
                        ]
                        reference = pd.read_parquet(source / f"{name}_{surface}_oof.parquet")
                        if not np.array_equal(reference.row_id, view.row_id) or not np.array_equal(
                            reference.selected_control, cb
                        ):
                            raise ValueError("exact control/key mismatch")
                        if not np.array_equal(reference.label, view.label):
                            raise ValueError("target alignment mismatch")
                        part = view[old.KEYS + ["row_id", "label", "anomaly_type"]].copy()
                        part["fold"], part["surface"] = name, surface
                        part["control"], part["candidate"] = cb, pb
                        part["candidate_probability"] = p["balanced"]
                        part["balanced_probability"] = c["balanced"]
                        part.to_parquet(OUT / f"{name}_{surface}.parquet", index=False)
                        all_rows.append(part)
                        fold_result["surfaces"][surface] = {
                            "control": counts(view.label, cb),
                            "candidate": counts(view.label, pb),
                        }
                        if surface == "intact":
                            intact_c, intact_p = cb, pb
                        else:
                            fold_result["intact_same_retained"] = {
                                "control": counts(ev.label.to_numpy()[keep], intact_c[keep]),
                                "candidate": counts(ev.label.to_numpy()[keep], intact_p[keep]),
                            }
                del training, x_bundle, x, xe, model, encoder, base_ev
                gc.collect()
                progress(f"{name}/{stage}/DONE")
            result["folds"].append(fold_result)
        complete = pd.concat(all_rows, ignore_index=True)
        result["metrics"] = {}
        for surface, part in complete.groupby("surface"):
            result["metrics"][surface] = {
                arm: counts(part.label, part[arm]) for arm in ("control", "candidate")
            }
        primary = result["metrics"]["intact"]
        result["delta_f1"] = primary["candidate"]["f1"] - primary["control"]["f1"]
        result["decision"] = "INTERNAL_CANDIDATE" if result["delta_f1"] > 0 else "NO_INTERNAL_GAIN"
        result["status"] = "COMPLETE"
        result["artifact_hashes"] = {p.name: old.sha(p) for p in OUT.glob("*.parquet")}
        result["runtime_seconds"] = time.monotonic() - started
        old.write_json(OUT / "terminal_result.json", result)
        old.write_json(REPORT / "result.json", result)
    except Exception as exc:
        result.update(
            status="TECHNICAL_FAILURE",
            error=f"{type(exc).__name__}: {exc}",
            runtime_seconds=time.monotonic() - started,
        )
        old.write_json(OUT / "terminal_result.json", result)
        raise


def qa_replay():
    cfg, _, _, frozen = load_contract()
    result = json.loads((OUT / "terminal_result.json").read_text(encoding="utf-8"))
    if result["status"] != "COMPLETE" or len(result["fits"]) != 6:
        raise ValueError("incomplete attempt")
    if old.sha(__file__) != result["runner_sha256"] or old.sha(CONFIG) != result["config_sha256"]:
        raise ValueError("candidate contract drift")
    data = load_training(frozen).set_index("row_id", drop=False)
    checks, replay = [], []
    parts = []
    for name, expected in result["artifact_hashes"].items():
        path = OUT / name
        if old.sha(path) != expected:
            raise ValueError("OOF hash changed")
        part = pd.read_parquet(path)
        checks.append(not part.duplicated(old.KEYS).any())
        parts.append(part)
        fold = part.fold.iloc[0]
        fit = next(x for x in result["fits"] if x["fold"] == fold and x["stage"] == "outer")
        model_path = OUT / f"{fold}_outer.joblib"
        if old.sha(model_path) != fit["model_sha256"]:
            raise ValueError("model hash changed")
        pack = joblib.load(model_path)
        ev = data.loc[part.row_id].reset_index(drop=True)
        checks.append(np.array_equal(ev.label, part.label))
        pred = pack["model"].predict_proba(
            pack["encoder"].transform(bundle(ev, pack["stats"], frozen, pack["epsilon"]))
        )[:, 1]
        exact = np.array_equal(pred, part.candidate_probability)
        checks.append(exact)
        replay.append({"file": name, "rows": len(part), "probability_exact": bool(exact)})
    all_rows = pd.concat(parts, ignore_index=True)
    slices = []
    for surface, part in all_rows.groupby("surface"):
        for arm in ("control", "candidate"):
            own = counts(part.label, part[arm])
            checks.append(own == result["metrics"][surface][arm])
        for keys, group in part.groupby(["fold", "station"]):
            slices.append(
                {
                    "surface": surface,
                    "fold": keys[0],
                    "station": keys[1],
                    **{arm: counts(group.label, group[arm]) for arm in ("control", "candidate")},
                }
            )
    for fit in result["fits"]:
        checks.append(
            pd.Timestamp(fit["training_max"])
            < pd.Timestamp(fit["evaluation_min"]) - pd.Timedelta(days=frozen["purge_days"])
        )
    qa = {
        "status": "PASS" if all(checks) else "FAIL",
        "checks": len(checks),
        "passed": sum(bool(x) for x in checks),
        "replay_pid": os.getpid(),
        "training_pid": result["pid"],
        "fresh_process": os.getpid() != result["pid"],
        "replay": replay,
        "slices": slices,
        "result_sha256": old.sha(OUT / "terminal_result.json"),
        "new_fits": 0,
        "official_rows": 0,
        "csv_written": 0,
        "uploads": 0,
    }
    old.write_json(REPORT / "independent-qa.json", qa)
    if not all(checks) or not qa["fresh_process"]:
        raise ValueError("independent replay QA failed")
    print(json.dumps({"status": qa["status"], "checks": len(checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--qa-replay", action="store_true")
    args = parser.parse_args()
    if args.execute and args.qa_replay:
        raise SystemExit("choose one mode")
    if args.execute:
        execute()
    elif args.qa_replay:
        qa_replay()
    else:
        load_contract()
        print(json.dumps({"status": "CONTRACT_ONLY", "max_fits": 6, "official_rows": 0}))
