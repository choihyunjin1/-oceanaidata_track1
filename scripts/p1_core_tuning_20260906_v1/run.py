"""Sealed CPU2 inner-selected O1/B3 tuning; historical training input only."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("champion_tree", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/tree.py")
t = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t)
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
import numpy as np
import pandas as pd
import xgboost  # noqa: F401 -- load native pool before applying the resource limit
from threadpoolctl import threadpool_info, threadpool_limits

ID = "p1_core_tuning_20260906_v1"
CONFIG = ROOT / "configs/experiments/p1_core_tuning_20260906_v1.json"
core = t.core
KEYS = t.KEYS
BASE_COMPONENTS = ("O_control", "B_control")


def config():
    cfg = t.read_json(CONFIG)
    if (cfg["id"], cfg["threads"], cfg["fit_cap"], cfg["wall_seconds"], cfg["gpu"]) != (ID, 2, 48, 5400, False):
        raise ValueError("resource contract changed")
    if list(cfg["policies"]) != cfg["policy_tie_order"] or len(cfg["policies"]) != 4:
        raise ValueError("four fixed policies required")
    return cfg


def base_config():
    cfg = t.config()
    return {**cfg, "threads": 2}


def sources():
    paths = {**t.source_hashes()}
    for path in (Path(__file__), CONFIG, ROOT / "tests/test_p1_core_tuning_20260906_v1.py"):
        paths[path.relative_to(ROOT).as_posix()] = core.sha(path)
    return paths


def packages():
    return {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "xgboost", "lightgbm", "joblib", "threadpoolctl")}


def check_threads():
    pools = threadpool_info()
    if not pools or any(p["num_threads"] > 2 for p in pools):
        raise ValueError("native thread budget exceeded")
    return [{k: p[k] for k in ("internal_api", "num_threads", "prefix")} for p in pools]


def seal(path):
    if path.exists():
        raise FileExistsError("immutable seal already exists")
    core.write_json(path, {"status": "SEALED_FITS_0", "config": config(), "base": base_config(),
                          "sources": sources(), "packages": packages(), "native_pools": check_threads(),
                          "baseline_fits": 24, "alternative_inner_fits": 15,
                          "selected_outer_max_fits": 9, "full_fits_authorized": 0})


def check_seal(path):
    saved = t.read_json(path)
    if saved["config"] != config() or saved["base"] != base_config() or saved["sources"] != sources() or saved["packages"] != packages():
        raise ValueError("sealed code/config/package mismatch")
    check_threads()
    return saved


def seeds(component, cfg):
    return [cfg["O_seed"]] if component.startswith("O_") else cfg["B_seeds"]


def parameters(component, seed, cfg, contract):
    spec = contract["components"][component]
    params = t.parameters(cfg, spec["arm"], seed)
    params.update(spec["overrides"])
    if spec["arm"] == "B":
        params["n_jobs"] = 2
    return params


def outer_components(selected, contract):
    return list(dict.fromkeys([*BASE_COMPONENTS, *contract["policies"][selected]]))


def fit_count(selected, contract, cfg):
    return 24 + 15 + sum(sum(len(seeds(c, cfg)) for c in outer_components(s, contract)
                              if c not in BASE_COMPONENTS) for s in selected)


def select_inner(frame, probabilities, rules, cfg, contract):
    thresholds, decoded = {}, {}
    for name, probability in probabilities.items():
        choice, decoded[name] = core.calibrate(frame, probability, rules, cfg)
        thresholds[name] = choice["threshold"]
    scores = {policy: t.metric(frame.label, decoded[pair[0]] | decoded[pair[1]])
              for policy, pair in contract["policies"].items()}
    selected = max(contract["policy_tie_order"], key=lambda p: scores[p]["f1"])
    return {"selected": selected, "thresholds": thresholds, "inner_policy_metrics": scores,
            "scope": "earlier_inner_only", "tie_order": contract["policy_tie_order"]}


def policy_bits(frame, probabilities, rules, cfg, contract, selector):
    decoded = {c: core.decode(frame, p, rules, cfg, selector["thresholds"][c])
               for c, p in probabilities.items()}
    def union(policy):
        left, right = contract["policies"][policy]
        return decoded[left] | decoded[right]
    return {"control": union("control"), "candidate": union(selector["selected"])}


def fit_stage(training, valid, directory, components, cfg, contract, receipt, progress):
    directory.mkdir(parents=True, exist_ok=False)
    stats = core.stats_fit(training)
    bundle = t.base_features(training, stats, cfg)
    encoder = core.TabularEncoder().fit(bundle, np.arange(len(training)))
    x = encoder.transform(bundle)
    xp = encoder.transform(t.base_features(valid, stats, cfg))
    y = training.label.to_numpy(dtype=np.int8)
    weights = core._event_day_weight(training, y)
    probabilities, files = {}, {}
    for component in components:
        values, files[component] = [], []
        for seed in seeds(component, cfg):
            if receipt["attempted_fits"] >= contract["fit_cap"]:
                raise ValueError("actual model fit cap exceeded")
            receipt["attempted_fits"] += 1
            progress(f"{directory.parent.name}/{directory.name}/{component}/{seed}/STARTED")
            started = time.monotonic()
            params = parameters(component, seed, cfg, contract)
            if component.startswith("O_"):
                model = core._fit_model("xgboost", params, seed, 2, x, y)
                native = model.model
            else:
                model = core.lgb.LGBMClassifier(**params)
                model.fit(x, y, sample_weight=weights)
                native = model
            if native.get_params()["n_jobs"] != 2:
                raise ValueError("model thread budget mismatch")
            check_threads()
            p = np.asarray(model.predict_proba(xp)[:, 1], dtype=np.float64)
            if p.shape != (len(valid),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError("invalid probabilities")
            path = directory / f"{component}_{seed}.joblib"
            core.joblib.dump(model, path)
            files[component].append(path.name)
            values.append(p)
            receipt["fit_receipts"].append({"path": path.relative_to(directory.parent.parent).as_posix(),
                "component": component, "seed": seed, "parameters": params, "threads": 2,
                "model_sha256": core.sha(path), "probability_sha256": t.digest(p),
                "train_rows": len(training), "train_positive": int(y.sum()),
                "train_keys_sha256": t.key_digest(training), "target_sha256": t.digest(y),
                "features": 80, "trees": params["n_estimators"], "runtime_seconds": time.monotonic() - started})
            receipt["completed_fits"] += 1
            del model, native
            gc.collect()
            progress(f"{directory.parent.name}/{directory.name}/{component}/{seed}/COMPLETE")
        probabilities[component] = np.mean(values, axis=0)
    core.joblib.dump({"stats": stats, "encoder": encoder, "files": files,
                     "feature_columns": list(bundle.feature_columns)}, directory / "preprocess.joblib")
    return probabilities, core.rule_masks(valid, stats)


def summarize(oof, contract):
    ec = {"common": {"bootstrap": contract["bootstrap"]}}
    regions = {"primary_all_Q2_Q3_Q4": np.ones(len(oof), dtype=bool),
               "secondary_old_Q3_Q4": oof.fold.isin(["2025_q3", "2025_q4"]).to_numpy()}
    result = {}
    for name, mask in regions.items():
        part = oof.loc[mask]
        days = pd.to_datetime(part.time, utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
        result[name] = {"rows": len(part), "positive": int(part.label.sum()),
            "keys_sha256": t.key_digest(part),
            "metrics": {arm: t.metric(part.label, part[arm]) for arm in ("control", "candidate")},
            "paired_day_bootstrap": t.evaluation.paired_bootstrap(part.label, part.control, part.candidate, days, "f1", ec)}
    rows = []
    oof = oof.copy()
    oof["day"] = pd.to_datetime(oof.time, utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    for fields in (["fold"], ["station", "layer"], ["fold", "supported"], ["day"]):
        for key, part in oof.groupby(fields, sort=True):
            m = {arm: t.metric(part.label, part[arm]) for arm in ("control", "candidate")}
            rows.append({"by": fields, "key": str(key), "rows": len(part), "positive": int(part.label.sum()),
                         "metrics": m, "delta_f1": m["candidate"]["f1"] - m["control"]["f1"]})
    result["slices"] = rows
    result["worst_station_layer"] = min((r for r in rows if r["by"] == ["station", "layer"]), key=lambda r: r["delta_f1"])
    result["worst_day"] = min((r for r in rows if r["by"] == ["day"]), key=lambda r: r["delta_f1"])
    return result


def worker(output, seal_path):
    saved = check_seal(seal_path)
    cfg, contract = saved["base"], saved["config"]
    receipt = t.read_json(output / "ATTEMPT_LOCK.json")
    receipt.update(worker_pid=os.getpid(), attempted_fits=0, completed_fits=0, fit_receipts=[],
                   official_rows=0, hidden_rows=0, csv_written=0, upload=0, native_pools=check_threads())
    def progress(stage):
        receipt.update(stage=stage, runtime_seconds=time.time() - receipt["started_unix"])
        core.write_json(output / "progress.json", receipt)
        print({k: receipt[k] for k in ("stage", "runtime_seconds", "completed_fits")}, flush=True)
    def expired():
        receipt.update(status="WALL_CAP_TERMINAL", retry_authorized=False)
        core.write_json(output / "terminal_result.json", receipt)
        os._exit(124)
    timer = threading.Timer(max(0, contract["wall_seconds"] - (time.time() - receipt["started_unix"])), expired)
    timer.daemon = True
    timer.start()
    try:
        data = t.training_source(cfg)
        masks, support = t.split_masks(data, cfg)
        for row in support:
            row["supported_station_layer_rows"] = row["validation_rows"] - row["unseen_station_layer_rows"]
            if row["stage"] == "outer" and row["validation_rows"] != contract["support_expected"][row["fold"]]:
                raise ValueError("fixed evaluation support mismatch")
        core.write_json(output / "support.json", support)
        parts, selections = [], []
        for fold in cfg["folds"]:
            name = fold["id"]
            selector = None
            for stage in ("inner", "outer"):
                check_seal(seal_path)
                tm, vm = masks[(name, stage)]
                training, valid = data.loc[tm].reset_index(drop=True), data.loc[vm].reset_index(drop=True)
                if stage == "outer" and selector is None:
                    raise ValueError("inner selector must be frozen before outer")
                components = list(contract["components"]) if stage == "inner" else outer_components(selector["selected"], contract)
                directory = output / name / stage
                probabilities, rules = fit_stage(training, valid, directory, components, cfg, contract, receipt, progress)
                if stage == "inner":
                    selector = select_inner(valid, probabilities, rules, cfg, contract)
                    core.write_json(output / name / "selector.json", selector)
                    selections.append(selector["selected"])
                bits = policy_bits(valid, probabilities, rules, cfg, contract, selector)
                result = valid[KEYS + ["row_id", "label"]].copy()
                result["fold"] = name
                known = set(zip(training.station, training.layer, strict=True))
                result["supported"] = [k in known for k in zip(valid.station, valid.layer, strict=True)]
                for component, p in probabilities.items():
                    result["probability_" + component] = p
                for arm, values in bits.items():
                    result[arm] = values
                result.to_parquet(directory / "predictions.parquet", index=False)
                if stage == "outer":
                    parts.append(result[KEYS + ["row_id", "label", "fold", "supported", "control", "candidate"]])
                del training, valid, probabilities, result
                gc.collect()
        if receipt["attempted_fits"] != receipt["completed_fits"] or receipt["completed_fits"] != fit_count(selections, contract, cfg):
            raise ValueError("actual fit ledger mismatch")
        oof = pd.concat(parts, ignore_index=True)
        if len(oof) != 421032 or oof.duplicated(KEYS).any() or oof.row_id.duplicated().any():
            raise ValueError("outer keys/population mismatch")
        oof.to_parquet(output / "oof.parquet", index=False)
        receipt["evaluation"] = summarize(oof, contract)
        receipt["selections"] = selections
        receipt["oof_sha256"] = core.sha(output / "oof.parquet")
        if core.sha(Path(os.environ["P1_DATA_DIR"]) / "train.csv") != cfg["train_sha256"]:
            raise ValueError("source changed during execution")
        check_seal(seal_path)
        receipt["files"] = {p.relative_to(output).as_posix(): core.sha(p) for p in sorted(output.rglob("*"))
                            if p.is_file() and p.name not in {"ATTEMPT_LOCK.json", "progress.json", "terminal_result.json"}}
        receipt["status"] = "COMPLETE"
        progress("TERMINAL")
        core.write_json(output / "terminal_result.json", receipt)
    except Exception as exc:
        receipt.update(status="TERMINAL_TECHNICAL_FAILURE", error_type=type(exc).__name__, error=str(exc), retry_authorized=False)
        progress("FAILED_NO_RESTART")
        core.write_json(output / "terminal_result.json", receipt)
        raise
    finally:
        timer.cancel()


def replay(output, seal_path):
    path = output / "fresh-replay-qa.json"
    if path.exists():
        raise FileExistsError("immutable replay receipt exists")
    saved, result = check_seal(seal_path), t.read_json(output / "terminal_result.json")
    if result["status"] != "COMPLETE" or result["worker_pid"] == os.getpid():
        raise ValueError("completed run and fresh PID required")
    t.verify_artifacts(output, result)
    started = time.monotonic()
    cfg, contract = saved["base"], saved["config"]
    data = t.training_source(cfg)
    masks, _ = t.split_masks(data, cfg)
    checks = []
    for fold in cfg["folds"]:
        name = fold["id"]
        selector = t.read_json(output / name / "selector.json")
        for stage in ("inner", "outer"):
            tm, vm = masks[(name, stage)]
            train, valid = data.loc[tm].reset_index(drop=True), data.loc[vm].reset_index(drop=True)
            directory = output / name / stage
            stored = core.joblib.load(directory / "preprocess.joblib")
            rows = pd.read_parquet(directory / "predictions.parquet")
            if t.key_digest(valid) != t.key_digest(rows) or not np.array_equal(valid.label, rows.label):
                raise ValueError("replay keys/labels mismatch")
            stats = core.stats_fit(train)
            bundle = t.base_features(train, stats, cfg)
            encoder = core.TabularEncoder().fit(bundle, np.arange(len(train)))
            if core.joblib.hash(stats) != core.joblib.hash(stored["stats"]) or core.joblib.hash(encoder) != core.joblib.hash(stored["encoder"]):
                raise ValueError("train-only statistics/encoder replay mismatch")
            xp = encoder.transform(t.base_features(valid, stats, cfg))
            probabilities = {}
            for component, filenames in stored["files"].items():
                probabilities[component] = np.mean([core.joblib.load(directory / f).predict_proba(xp)[:, 1] for f in filenames], axis=0)
                if not np.array_equal(probabilities[component], rows["probability_" + component]):
                    raise ValueError("fresh probability mismatch")
                checks.append(f"{name}/{stage}/{component}/exact_probability")
            rules = core.rule_masks(valid, stats)
            if stage == "inner" and select_inner(valid, probabilities, rules, cfg, contract) != selector:
                raise ValueError("inner selection replay mismatch")
            for arm, bits in policy_bits(valid, probabilities, rules, cfg, contract, selector).items():
                if not np.array_equal(bits, rows[arm]):
                    raise ValueError("fixed policy replay mismatch")
                checks.append(f"{name}/{stage}/{arm}/exact_bits")
            del xp, train, valid, bundle, encoder, stored, rows
            gc.collect()
    t.verify_artifacts(output, result)
    check_seal(seal_path)
    core.write_json(path, {"status": "PASS", "pid": os.getpid(), "training_pid": result["worker_pid"],
        "checks": checks, "check_count": len(checks), "fits": 0, "official_rows": 0,
        "runtime_seconds": time.monotonic() - started,
        "terminal_result_sha256": core.sha(output / "terminal_result.json")})


def launch(args):
    saved = check_seal(args.seal)
    if args.output.exists():
        raise FileExistsError("new output path required; no restart")
    args.output.mkdir(parents=True)
    cap = saved["config"]["wall_seconds"]
    core.write_json(args.output / "ATTEMPT_LOCK.json", {"id": ID, "started_unix": time.time(),
        "launcher_pid": os.getpid(), "seal_sha256": core.sha(args.seal), "fit_cap": 48, "wall_seconds": cap})
    child = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "train", "--worker",
        "--seal", str(args.seal.resolve()), "--output", str(args.output.resolve())])
    try:
        code = child.wait(timeout=cap)
        if code:
            raise SystemExit(code)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True, check=False)
        else:
            child.kill()
        child.wait(timeout=30)
        raise SystemExit(124) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["preflight", "train", "replay"])
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.phase == "preflight":
            seal(args.seal)
        elif args.output is None:
            parser.error("--output required")
        elif args.phase == "replay":
            replay(args.output.resolve(), args.seal)
        elif args.worker:
            worker(args.output.resolve(), args.seal)
        else:
            launch(args)


if __name__ == "__main__":
    main()
