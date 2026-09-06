"""Bounded train-only O1/B3 reconstruction. No historical CLI or official I/O."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "4"
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import ocean_evaluation_contract_v5 as evaluation
import pandas as pd
import run_p1_score_repair_20260905_v1 as core

ID = "p1_champion_reconstruction_20260906_v1_tree"
CONTRACT = HERE / "tree-contract.json"
ARMS = ("O", "B", "union", "router")
CELL_ORDER = ("B", "O", "AND", "OR")
RAW = core.RAW
KEYS = core.KEYS


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def key_digest(frame):
    return digest(pd.util.hash_pandas_object(frame[KEYS], index=False).to_numpy())


def metric(y, bits):
    evaluation.pooled_metric(y, bits, "f1")  # fail-closed binary/finite/length guard
    return core.metric(y, bits)


def config():
    cfg = read_json(CONTRACT)
    if (cfg["id"], cfg["threads"], cfg["historical_fits"], cfg["full_fits"]) != (ID, 4, 24, 4):
        raise ValueError("resource contract changed")
    if cfg["B_seeds"] != [20260813, 20260829, 20260847] or cfg["O_seed"] != 20260813:
        raise ValueError("seed recipe changed")
    if cfg["deployment_selector"] != "2025_q4_inner":
        raise ValueError("additional selector fits not authorized")
    return cfg


def source_hashes():
    paths = list((ROOT / "src/p1_qc").glob("*.py")) + [
        Path(__file__), CONTRACT, ROOT / "configs/p1.toml",
        ROOT / "configs/p1_meaningful_learning_curve_generation_v1.json",
        ROOT / "scripts/run_p1_meaningful_learning_curve_generation_v1.py",
        ROOT / "scripts/run_p1_score_repair_20260905_v1.py",
        ROOT / "scripts/ocean_evaluation_contract_v5.py",
    ]
    return {p.relative_to(ROOT).as_posix(): core.sha(p) for p in sorted(paths)}


def make_seal(path):
    if Path(path).exists():
        raise FileExistsError("seal is immutable; use a new path")
    cfg = config()
    if core.sha(ROOT / cfg["weight_source"]) != cfg["weight_source_sha256"]:
        raise ValueError("audited weight/parameter source changed")
    core.write_json(path, {"id": ID, "status": "CODE_SEALED_FITS_NOT_STARTED",
                          "config": cfg, "sources": source_hashes(), "fits": 0,
                          "thread_environment_after_imports": thread_environment(),
                          "packages": {p: importlib.metadata.version(p) for p in
                                       ("numpy", "pandas", "xgboost", "lightgbm", "joblib")}})


def thread_environment():
    return {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}


def check_seal(path):
    seal = read_json(path)
    if seal["sources"] != source_hashes() or seal["config"] != config():
        raise ValueError("sealed source or config changed")
    if seal["packages"] != {p: importlib.metadata.version(p) for p in seal["packages"]}:
        raise ValueError("sealed environment changed")
    if seal["thread_environment_after_imports"] != thread_environment():
        raise ValueError("thread environment changed")
    return seal


def training_source(cfg):
    path = Path(os.environ["P1_DATA_DIR"]).resolve() / "train.csv"
    if core.sha(path) != cfg["train_sha256"]:
        raise ValueError("distributed training source hash mismatch")
    frame = pd.read_csv(path, usecols=RAW + ["label"])
    if len(frame) != 776706 or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("training population or keys mismatch")
    metric(frame.label, frame.label)
    frame["row_id"] = np.arange(len(frame))
    times = pd.to_datetime(frame.time, utc=True)
    if not np.array_equal(times.dt.tz_convert("Asia/Seoul").dt.year, frame.year):
        raise ValueError("KST year mismatch")
    return frame.sort_values(["station", "layer", "time"], kind="stable").reset_index(drop=True)


def split_masks(frame, cfg):
    """Same run-start ownership and whole-run train exclusion as active all_split_masks."""
    ec = {"P1": {"positive_runs": {"cadence_minutes": 10}, "folds": cfg["folds"]}}
    meta = evaluation.p1_run_metadata(frame, ec)
    times = pd.to_datetime(frame.time, utc=True)
    starts = pd.Series([v.value for v in times]).groupby(meta["run_id"]).transform("min").to_numpy()
    y = frame.label.to_numpy()
    masks, support = {}, []
    for fold in cfg["folds"]:
        start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
        stop = start - pd.Timedelta(days=cfg["purge_days"])
        inner = stop - pd.Timedelta(days=cfg["inner_days"])
        for stage, begin, finish in (("inner", inner, stop), ("outer", start, end)):
            cutoff = begin - pd.Timedelta(days=cfg["purge_days"])
            train = (times < cutoff).to_numpy() & ((y == 0) | (meta["end_ns"] < cutoff.value))
            valid = ((y == 0) & (times >= begin).to_numpy() & (times < finish).to_numpy()) | (
                (y == 1) & (starts >= begin.value) & (starts < finish.value))
            if stage == "outer" and not np.array_equal(valid & (y == 1), meta["owner"] == fold["id"]):
                raise ValueError("run ownership mismatch")
            if not train.any() or not valid.any() or len(np.unique(y[train])) != 2 or (train & valid).any():
                raise ValueError("insufficient split support; dates may not be changed")
            masks[(fold["id"], stage)] = (train, valid)
            known = set(zip(frame.loc[train, "station"], frame.loc[train, "layer"], strict=True))
            unseen = np.array([k not in known for k in zip(frame.loc[valid, "station"],
                                                          frame.loc[valid, "layer"], strict=True)])
            support.append({"fold": fold["id"], "stage": stage, "train_rows": int(train.sum()),
                            "validation_rows": int(valid.sum()), "train_positive": int(y[train].sum()),
                            "validation_positive": int(y[valid].sum()),
                            "unseen_station_layer_rows": int(unseen.sum()),
                            "train_cutoff_exclusive": cutoff.isoformat(),
                            "validation_start": begin.isoformat(), "validation_end": finish.isoformat(),
                            "validation_keys_sha256": key_digest(frame.loc[valid])})
    return masks, support


def base_features(frame, stats, cfg):
    """Exact active 80-column clean arm; do not compute unused flank features."""
    raw = frame[RAW].reset_index(drop=True)
    bundle = core.build_features(raw, config=core.load_config(ROOT / cfg["base_config"], env={}))
    base = bundle.frame
    nominal = np.array([stats["depth"].get(k, np.nan)
                        for k in zip(raw.station, raw.year, raw.layer, strict=True)])
    nominal = np.round(nominal / 2.0) * 2.0
    base["nominal_depth_m"] = nominal.astype(np.float32)
    base["depth_regime"] = pd.Series([
        f"{s}|d{d:06.1f}" if np.isfinite(d) else f"{s}|unknown|l{layer}"
        for s, layer, d in zip(raw.station, raw.layer, nominal, strict=True)], dtype="string")
    for col in ("plateau_full_length", "plateau_count", "plateau_elapsed"):
        base[col] = base[col].clip(upper=168 * 6)
    if len(base.columns) != 80 or {"label", "row_id"} & set(base):
        raise ValueError("80-column observation-only contract mismatch")
    return core.FeatureBundle(base, tuple(base), bundle.categorical_columns)


def parameters(cfg, arm, seed):
    if arm == "O":
        return dict(cfg["xgboost_parameters"])
    params = core._lgb_parameters({"lightgbm_parameters": cfg["lightgbm_parameters"]}, seed,
                                  multiclass=False)
    params["n_jobs"] = cfg["threads"]
    return params


def select_policy(inner, probabilities, rules, cfg):
    thresholds, bits = {}, {}
    for arm in ("O", "B"):
        selected, bits[arm] = core.calibrate(inner, probabilities[arm], rules, cfg)
        thresholds[arm] = selected["threshold"]
    choices = {"B": bits["B"], "O": bits["O"], "AND": bits["B"] & bits["O"],
               "OR": bits["B"] | bits["O"]}
    cells = []
    for (station, layer), positions in inner.groupby(["station", "layer"], sort=True).indices.items():
        scores = {name: metric(inner.label.iloc[positions], choices[name][positions])["f1"]
                  for name in CELL_ORDER}
        name = max(CELL_ORDER, key=lambda k: scores[k])  # stable B-first tie
        cells.append({"station": str(station), "layer": int(layer), "policy": name,
                      "inner_rows": len(positions), "inner_f1": scores[name]})
    return {"thresholds": thresholds, "cells": cells, "unseen_cell_fallback": "B",
            "selection_scope": "earlier_inner_only", "cell_tie_order": list(CELL_ORDER)}


def predict_policy(frame, probabilities, rules, cfg, selector):
    bits = {arm: core.decode(frame, probabilities[arm], rules, cfg, selector["thresholds"][arm])
            for arm in ("O", "B")}
    bits["union"] = bits["O"] | bits["B"]
    choices = {"B": bits["B"], "O": bits["O"], "AND": bits["B"] & bits["O"], "OR": bits["union"]}
    mapping = {(r["station"], r["layer"]): r["policy"] for r in selector["cells"]}
    router = bits["B"].copy()
    for key, positions in frame.groupby(["station", "layer"], sort=False).indices.items():
        router[positions] = choices[mapping.get(key, "B")][positions]
    bits["router"] = router
    return bits


def fit_four(training, valid, cfg, directory, receipt, progress, *, in_sample=False):
    directory.mkdir(parents=True, exist_ok=False)
    stats = core.stats_fit(training)
    bundle = base_features(training, stats, cfg)
    encoder = core.TabularEncoder().fit(bundle, np.arange(len(training)))
    x = encoder.transform(bundle)
    if in_sample:
        positions = pd.Index(training.row_id).get_indexer(valid.row_id)
        if (positions < 0).any():
            raise ValueError("in-sample probe key absent")
        xp = x[positions]
    else:
        xp = encoder.transform(base_features(valid, stats, cfg))
    target = training.label.to_numpy(dtype=np.int8)
    weights = core._event_day_weight(training, target)
    probabilities = {"O": [], "B": []}
    model_files = []
    for arm, seed in [("O", cfg["O_seed"])] + [("B", s) for s in cfg["B_seeds"]]:
        if receipt["attempted_fits"] >= receipt["fit_cap"]:
            raise ValueError("fit budget exceeded")
        receipt["attempted_fits"] += 1
        progress(f"{directory.name}/{arm}/{seed}/FIT_STARTED")
        start = time.monotonic()
        params = parameters(cfg, arm, seed)
        if arm == "O":
            model = core._fit_model("xgboost", params, seed, cfg["threads"], x, target)
        else:
            model = core.lgb.LGBMClassifier(**params)
            model.fit(x, target, sample_weight=weights)
        probability = np.asarray(model.predict_proba(xp)[:, 1], dtype=np.float64)
        if probability.shape != (len(valid),) or not np.isfinite(probability).all():
            raise ValueError("prediction validity failure")
        path = directory / f"{arm}_{seed}.joblib"
        core.joblib.dump(model, path)
        model_files.append(path.name)
        probabilities[arm].append(probability)
        receipt["completed_fits"] += 1
        receipt["fit_receipts"].append({"path": str(path.relative_to(directory.parent.parent)),
            "arm": arm, "seed": seed, "model_sha256": core.sha(path),
            "train_rows": len(training), "train_positive": int(target.sum()),
            "train_keys_sha256": key_digest(training), "target_sha256": digest(target),
            "features": 80, "trees": 700, "threads": cfg["threads"],
            "probability_sha256": digest(probability), "runtime_seconds": time.monotonic() - start})
        del model
        gc.collect()
        progress(f"{directory.name}/{arm}/{seed}/FIT_COMPLETE")
    core.joblib.dump({"stats": stats, "encoder": encoder, "model_files": model_files,
                     "feature_columns": list(bundle.feature_columns)}, directory / "preprocess.joblib")
    result = {arm: np.mean(values, axis=0) for arm, values in probabilities.items()}
    return result, core.rule_masks(valid, stats)


def summarize(oof, cfg):
    regions = {"primary_Q3_Q4": oof.fold.isin(["2025_q3", "2025_q4"]).to_numpy(),
               "all_three_folds": np.ones(len(oof), dtype=bool),
               "H1_2025_partial_Q2_not_full_H1": oof.fold.eq("2025_q2").to_numpy()}
    ec = {"common": {"bootstrap": {"minimum_clusters": 2, "seed": 20260906,
                                    "resamples": 2000, "ci_quantiles": [0.05, 0.95]}}}
    report = {}
    for scope, mask in regions.items():
        part = oof.loc[mask]
        days = pd.to_datetime(part.time, utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
        report[scope] = {"rows": len(part), "keys_sha256": key_digest(part),
                        "metrics": {arm: metric(part.label, part[arm]) for arm in ARMS},
                        "vs_B": {arm: evaluation.paired_bootstrap(part.label, part.B, part[arm],
                                    days, "f1", ec) for arm in ("O", "union", "router")}}
    slices = []
    for columns in (["fold"], ["station", "layer"]):
        for key, part in oof.groupby(columns, sort=True):
            slices.append({"by": list(columns), "key": str(key), "rows": len(part),
                           "metrics": {arm: metric(part.label, part[arm]) for arm in ARMS}})
    report["slices"] = slices
    report["worst_router_minus_B"] = min(s["metrics"]["router"]["f1"] - s["metrics"]["B"]["f1"]
                                          for s in slices)
    return report


def worker(phase, output, seal_path, historical=None):
    seal = check_seal(seal_path)
    cfg = seal["config"]
    receipt = read_json(output / "ATTEMPT_LOCK.json")
    receipt.update({"worker_pid": os.getpid(), "attempted_fits": 0, "completed_fits": 0,
                    "fit_receipts": [], "official_rows": 0, "hidden_rows": 0,
                    "csv_written": 0, "upload": 0, "thread_environment": thread_environment()})

    def expired():
        core.write_json(output / "terminal_result.json", {"status": "WALL_CAP_TERMINAL",
            "worker_pid": os.getpid(), "attempted_fits": receipt["attempted_fits"],
            "completed_fits": receipt["completed_fits"], "retry_authorized": False})
        os._exit(124)

    remaining = receipt["wall_cap_seconds"] - (time.time() - receipt["started_unix"])
    timer = threading.Timer(max(0, remaining), expired)
    timer.daemon = True
    timer.start()

    def progress(stage):
        receipt.update(stage=stage, runtime_seconds=time.time() - receipt["started_unix"])
        core.write_json(output / "progress.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("stage", "runtime_seconds", "completed_fits")}), flush=True)

    try:
        data = training_source(cfg)
        progress("TRAIN_SOURCE_VALIDATED")
        if phase == "historical":
            masks, support = split_masks(data, cfg)
            core.write_json(output / "support.json", support)
            parts = []
            for fold in cfg["folds"]:
                name = fold["id"]
                selector = None
                for stage in ("inner", "outer"):
                    check_seal(seal_path)
                    tm, vm = masks[(name, stage)]
                    train = data.loc[tm].reset_index(drop=True)
                    valid = data.loc[vm].reset_index(drop=True)
                    directory = output / name / stage
                    probabilities, rules = fit_four(train, valid, cfg, directory, receipt, progress)
                    if stage == "inner":
                        selector = select_policy(valid, probabilities, rules, cfg)
                        core.write_json(output / name / "selector.json", selector)
                    bits = predict_policy(valid, probabilities, rules, cfg, selector)
                    result = valid[KEYS + ["row_id", "label"]].copy()
                    result["fold"] = name
                    for arm in ("O", "B"):
                        result[f"probability_{arm}"] = probabilities[arm]
                    for arm in ARMS:
                        result[arm] = bits[arm]
                    result.to_parquet(directory / "predictions.parquet", index=False)
                    if stage == "outer":
                        parts.append(result)
                    del train, valid, probabilities, result
                    gc.collect()
            oof = pd.concat(parts, ignore_index=True)
            if oof.duplicated(KEYS).any():
                raise ValueError("outer key duplication")
            oof.to_parquet(output / "oof.parquet", index=False)
            receipt["evaluation"] = summarize(oof, cfg)
            receipt["deployment_selector"] = "2025_q4/selector.json"
            receipt["oof_sha256"] = core.sha(output / "oof.parquet")
        else:
            old = read_json(historical / "terminal_result.json")
            if old["status"] != "COMPLETE" or old["completed_fits"] != 24 or old["seal_sha256"] != core.sha(seal_path):
                raise ValueError("completed historical lineage required")
            verify_artifacts(historical, old)
            qa = read_json(historical / "independent-qa.json")
            if qa.get("status") != "PASS" or qa.get("terminal_result_sha256") != core.sha(historical / "terminal_result.json"):
                raise ValueError("root independent QA PASS linked to historical result required")
            selector = read_json(historical / "2025_q4/selector.json")
            core.write_json(output / "selector.json", selector)
            # Only a deterministic in-sample probe, never a quality/held-out claim.
            probe = data.iloc[np.linspace(0, len(data) - 1, 2048, dtype=int)].reset_index(drop=True)
            probabilities, _ = fit_four(data, probe, cfg, output / "full" / "models", receipt,
                                        progress, in_sample=True)
            np.savez(output / "full_probe.npz", row_id=probe.row_id.to_numpy(), **probabilities)
            receipt["historical_result_sha256"] = core.sha(historical / "terminal_result.json")
        check_seal(seal_path)
        if core.sha(Path(os.environ["P1_DATA_DIR"]) / "train.csv") != cfg["train_sha256"]:
            raise ValueError("training source changed during execution")
        if receipt["completed_fits"] != receipt["fit_cap"]:
            raise ValueError("fit count mismatch")
        receipt["files"] = {p.relative_to(output).as_posix(): core.sha(p)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name not in
            {"ATTEMPT_LOCK.json", "progress.json", "terminal_result.json"}}
        receipt["status"] = "COMPLETE"
        progress("TERMINAL")
        core.write_json(output / "terminal_result.json", receipt)
    except Exception as exc:
        receipt.update(status="TERMINAL_TECHNICAL_FAILURE", error_type=type(exc).__name__, error=str(exc))
        progress("FAILED_NO_RESTART")
        core.write_json(output / "terminal_result.json", receipt)
        raise
    finally:
        timer.cancel()


def verify_artifacts(output, result):
    for relative, expected in result["files"].items():
        path = (output / relative).resolve()
        if output.resolve() not in path.parents or core.sha(path) != expected:
            raise ValueError("artifact hash/path mismatch: " + relative)


def replay(output, seal_path):
    """New process, raw training source only; no fitting or old-answer inputs."""
    if (output / "fresh-replay-qa.json").exists():
        raise FileExistsError("replay receipt already exists; do not overwrite")
    seal = check_seal(seal_path)
    result = read_json(output / "terminal_result.json")
    if result["status"] != "COMPLETE" or result["seal_sha256"] != core.sha(seal_path):
        raise ValueError("completed exact sealed run required")
    if result["worker_pid"] == os.getpid():
        raise ValueError("fresh replay must use another process")
    verify_artifacts(output, result)
    started = time.monotonic()
    data, cfg = training_source(seal["config"]), seal["config"]
    checks = []
    if result["phase"] == "historical":
        masks, _ = split_masks(data, cfg)
        jobs = [(fold["id"], stage) for fold in cfg["folds"] for stage in ("inner", "outer")]
    else:
        masks, jobs = {}, [("full", "models")]
    for fold, stage in jobs:
        folder = output / fold / stage
        stored = core.joblib.load(folder / "preprocess.joblib")
        if result["phase"] == "historical":
            tm, vm = masks[(fold, stage)]
            train, valid = data.loc[tm].reset_index(drop=True), data.loc[vm].reset_index(drop=True)
            saved = pd.read_parquet(folder / "predictions.parquet")
            if key_digest(valid) != key_digest(saved) or not np.array_equal(valid.label, saved.label):
                raise ValueError("replay target/key mismatch")
        else:
            train = data
            probe = np.load(output / "full_probe.npz", allow_pickle=False)
            valid = data.iloc[pd.Index(data.row_id).get_indexer(probe["row_id"])].reset_index(drop=True)
            saved = {"probability_O": probe["O"], "probability_B": probe["B"]}
        stats = core.stats_fit(train)
        if core.joblib.hash(stats) != core.joblib.hash(stored["stats"]):
            raise ValueError("recomputed train statistics mismatch")
        tb = base_features(train, stats, cfg)
        encoder = core.TabularEncoder().fit(tb, np.arange(len(train)))
        if core.joblib.hash(encoder) != core.joblib.hash(stored["encoder"]):
            raise ValueError("recomputed encoder mismatch")
        if result["phase"] == "historical":
            xp = encoder.transform(base_features(valid, stats, cfg))
        else:
            xp = encoder.transform(tb)[pd.Index(train.row_id).get_indexer(valid.row_id)]
        probabilities = {"O": [], "B": []}
        for filename in stored["model_files"]:
            model = core.joblib.load(folder / filename)
            probabilities[filename[0]].append(model.predict_proba(xp)[:, 1])
            del model
        probabilities = {arm: np.mean(values, axis=0) for arm, values in probabilities.items()}
        for arm in ("O", "B"):
            if not np.array_equal(probabilities[arm], saved[f"probability_{arm}"]):
                raise ValueError("fresh probability replay mismatch")
            checks.append(f"{fold}/{stage}/{arm}_probability_exact")
        if result["phase"] == "historical":
            rules = core.rule_masks(valid, stats)
            selector = read_json(output / fold / "selector.json")
            if stage == "inner" and select_policy(valid, probabilities, rules, cfg) != selector:
                raise ValueError("inner selector replay mismatch")
            for arm, values in predict_policy(valid, probabilities, rules, cfg, selector).items():
                if not np.array_equal(values, saved[arm]):
                    raise ValueError("frozen policy replay mismatch")
                checks.append(f"{fold}/{stage}/{arm}_bits_exact")
        del xp, train, valid, stored, tb
        gc.collect()
    verify_artifacts(output, result)
    check_seal(seal_path)
    core.write_json(output / "fresh-replay-qa.json", {"status": "PASS", "pid": os.getpid(),
        "training_pid": result["worker_pid"], "checks": checks, "check_count": len(checks),
        "fits": 0, "official_rows": 0, "runtime_seconds": time.monotonic() - started,
        "terminal_result_sha256": core.sha(output / "terminal_result.json")})


def launch(args):
    check_seal(args.seal)
    if args.output.exists():
        raise FileExistsError("new empty output path required; attempts cannot be restarted")
    cfg = config()
    if args.phase == "full" and args.historical is None:
        raise ValueError("full needs completed historical receipt and separate approval")
    cap = 5400 if args.phase == "historical" else 21600
    if args.phase == "full":
        cap -= time.time() - read_json(args.historical / "ATTEMPT_LOCK.json")["started_unix"]
        if cap <= 0:
            raise ValueError("six-hour combined workflow cap exceeded")
    args.output.mkdir(parents=True)
    core.write_json(args.output / "ATTEMPT_LOCK.json", {
        "id": ID, "phase": args.phase, "launcher_pid": os.getpid(), "started_unix": time.time(),
        "seal_sha256": core.sha(args.seal), "fit_cap": cfg[f"{args.phase}_fits"],
        "wall_cap_seconds": cap, "status": "ONE_SHOT_STARTED"})
    command = [sys.executable, "-I", str(Path(__file__).resolve()), args.phase,
               "--seal", str(args.seal.resolve()), "--output", str(args.output.resolve()), "--worker"]
    if args.historical:
        command += ["--historical", str(args.historical.resolve())]
    child = subprocess.Popen(command)
    try:
        returncode = child.wait(timeout=cap)
        if returncode and not (args.output / "terminal_result.json").exists():
            core.write_json(args.output / "terminal_result.json", {"status": "TERMINAL_TECHNICAL_FAILURE",
                "returncode": returncode, "retry_authorized": False})
        if returncode:
            raise SystemExit(returncode)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:
            child.kill()
        child.wait(timeout=30)
        core.write_json(args.output / "terminal_result.json", {"status": "WALL_CAP_TERMINAL",
            "wall_cap_seconds": cap, "retry_authorized": False})
        raise SystemExit(124) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["preflight", "historical", "full", "replay"])
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase == "preflight":
        make_seal(args.seal)
    elif args.output is None:
        parser.error("--output is required")
    elif args.phase == "replay":
        replay(args.output.resolve(), args.seal)
    elif args.worker:
        worker(args.phase, args.output.resolve(), args.seal, args.historical)
    else:
        launch(args)


if __name__ == "__main__":
    main()
