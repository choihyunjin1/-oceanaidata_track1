"""Separate nested two-sided P1 comparison and inner-only iteration diagnostics."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

for _p1_thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_p1_thread_variable] = "2"

import audit_p1_twosided_support_20260906_v1 as support
import numpy as np
import pandas as pd
import run_p1_bracket_forward_20260906_v1 as base

ROOT = Path(__file__).resolve().parents[1]
ID = "p1_tuning_twosided_20260906_v1"
CONFIG = ROOT / "configs/experiments" / (ID + ".json")
OUT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID
old, policy, evaluation = base.old, base.policy, base.evaluation
FOLDS = ("H2_2024", "H1_2025", "H2_2025")
ARMS = ("original", "balanced", "bracket")


def contract():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    prior_cfg, frozen, ev = base.load_contract()
    checks = {
        "source_v1_config_sha256": base.CONFIG,
        "source_v1_runner_sha256": Path(base.__file__),
        "source_v1_result_sha256": base.OUT / "terminal_result.json",
        "source_twosided_runner_sha256": Path(support.__file__),
        "source_twosided_result_sha256": support.REPORT / "result.json",
    }
    for key, path in checks.items():
        if old.sha(path) != cfg[key]:
            raise ValueError("source pin changed: " + key)
    if (cfg["experiment_id"], cfg["threads"], cfg["gpu"], cfg["max_new_fits"]) != (ID, 2, False, 18):
        raise ValueError("resource contract changed")
    if any(cfg[key] for key in ("official_access", "csv_generation", "upload", "diagnostic_changes_outer")):
        raise ValueError("scope violation")
    if cfg["diagnostic_iterations"] != [100, 200, 400, 700, 1000, 1400]:
        raise ValueError("iteration bank changed")
    return cfg, prior_cfg, frozen, ev


def nested_masks(data, ev, cfg):
    meta = evaluation.p1_run_metadata(data, ev)
    times = pd.to_datetime(data.time, utc=True)
    labels = data.label.to_numpy()
    starts = pd.Series([x.value for x in times]).groupby(meta["run_id"]).transform("min").to_numpy()
    masks, counts = {}, []
    for fold in ev["P1"]["folds"]:
        if fold["warmup"]:
            continue
        outer_train, outer_val, _, _, _ = support.split_masks(data, fold, ev, meta)
        end = pd.Timestamp(fold["start"]) - pd.Timedelta(days=cfg["purge_days"])
        begin = end - pd.Timedelta(days=cfg["inner_days"])
        left, right = begin - pd.Timedelta(days=21), end + pd.Timedelta(days=21)
        inner_train = outer_train & ((times < left) | (times >= right)).to_numpy() & (
            (labels == 0) | (meta["end_ns"] < left.value) | (starts >= right.value)
        )
        inner_val = outer_train & (
            ((labels == 0) & (times >= begin).to_numpy() & (times < end).to_numpy())
            | ((labels == 1) & (starts >= begin.value) & (starts < end.value))
        )
        for stage, train, val in (("inner", inner_train, inner_val), ("outer", outer_train, outer_val)):
            if not train.any() or not val.any() or np.any(train & val) or len(np.unique(labels[train])) != 2:
                raise ValueError(f"{fold['id']}/{stage}: invalid support; no date amendment")
            truns = set(meta["run_id"][train & (labels == 1)])
            vruns = set(meta["run_id"][val & (labels == 1)])
            if truns & vruns:
                raise ValueError("shared positive run")
            if stage == "inner" and (np.any(train & outer_val) or np.any(val & outer_val)):
                raise ValueError("outer entered inner")
            for run in truns | vruns:
                part = train if run in truns else val
                if not part[meta["run_id"] == run].all():
                    raise ValueError("partial positive run")
            masks[(fold["id"], stage)] = (train, val)
            known = set(zip(data.loc[train, "station"], data.loc[train, "layer"], strict=True))
            unseen = sum(k not in known for k in zip(data.loc[val, "station"], data.loc[val, "layer"], strict=True))
            counts.append({"fold": fold["id"], "stage": stage, "train_rows": int(train.sum()),
                           "validation_rows": int(val.sum()), "train_positive": int(labels[train].sum()),
                           "validation_positive": int(labels[val].sum()), "unseen_rows": int(unseen),
                           "training_keys_sha256": base.key_digest(data.loc[train]),
                           "evaluation_keys_sha256": base.key_digest(data.loc[val]),
                           "inner_start": begin.isoformat(), "inner_end": end.isoformat()})
    return masks, counts


def fit(arm, x, training, frozen, cfg, iterations):
    if arm == "original":
        config = old.load_config(ROOT / frozen["base_config"], env={})
        parameters = dict(config.raw["models"]["xgboost"])
        parameters["n_estimators"] = iterations
        return old._fit_model("xgboost", parameters, cfg["seed"], cfg["threads"], x,
                              training.label.to_numpy(dtype=np.int8))
    recipe = json.loads((ROOT / frozen["lightgbm_recipe"]).read_text(encoding="utf-8"))
    parameters = old._lgb_parameters({"lightgbm_parameters": recipe["lightgbm_parameters"]},
                                     cfg["seed"], multiclass=False)
    parameters.update(n_jobs=cfg["threads"], n_estimators=iterations)
    model = old.lgb.LGBMClassifier(**parameters)
    y = training.label.to_numpy(dtype=np.int8)
    model.fit(x, y, sample_weight=old._event_day_weight(training, y))
    return model


def probability(model, x, arm, iteration):
    if arm == "original":
        p = model.model.predict_proba(x, iteration_range=(0, iteration))[:, 1].astype(float)
    else:
        p = model.predict_proba(x, num_iteration=iteration)[:, 1]
    if len(p) != len(x) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid probability")
    return p


def diagnostic_row(frame, p, rules, frozen, iteration):
    bits = old.decode(frame, p, rules, frozen, 0.5)
    y = frame.label.to_numpy()
    q = np.clip(p, 1e-15, 1 - 1e-15)
    loss = -(y * np.log(q) + (1 - y) * np.log1p(-q))
    return {"iteration": iteration, **old.metric(y, bits), "logloss_sum": float(loss.sum()),
            "logloss": float(loss.mean()), "fixed_decoder_threshold": 0.5}


def diagnostic_summary(records):
    summary = {}
    for arm in ARMS:
        rows = []
        for iteration in (100, 200, 400, 700, 1000, 1400):
            items = [x for x in records if x["arm"] == arm and x["iteration"] == iteration]
            if len(items) != 3:
                raise ValueError("three inner folds required")
            totals = {k: sum(x[k] for x in items) for k in ("rows", "tp", "fp", "fn", "logloss_sum")}
            rows.append({"iteration": iteration, **totals,
                         "f1": 2 * totals["tp"] / max(1, 2 * totals["tp"] + totals["fp"] + totals["fn"]),
                         "logloss": totals["logloss_sum"] / totals["rows"]})
        best = max(rows, key=lambda x: (x["f1"], -x["iteration"]))
        summary[arm] = {"curve": rows, "best_fixed_decoder_iteration": best["iteration"],
                        "best_is_ceiling": best["iteration"] == 1400,
                        "outer_iteration_unchanged": 700, "not_actual_policy_selected_decoder": True}
    return summary


def progress(result, stage, started, cfg):
    elapsed = time.monotonic() - started
    old.write_json(OUT / "progress.json", {"stage": stage, "seconds": elapsed,
                                           "new_fits": result["new_fits"], "pid": os.getpid()})
    print(json.dumps({"stage": stage, "new_fits": result["new_fits"], "seconds": round(elapsed, 2)}), flush=True)
    if elapsed >= cfg["wall_cap_seconds"]:
        raise TimeoutError("60-minute cap; preserve partial attempt and do not restart")


def stage_run(data, masks, fold, stage, arm, cfg, prior_cfg, frozen, result, started):
    progress(result, f"{fold}/{stage}/{arm}/prepare", started, cfg)
    train_mask, val_mask = masks[(fold, stage)]
    prepared = base.partition_inputs(data, train_mask, val_mask, frozen, prior_cfg, arm == "bracket")
    stem = f"{fold}_{stage}_{arm}"
    record = {"fold": fold, "stage": stage, "arm": arm,
              "training_keys_sha256": base.key_digest(prepared["train"]),
              "evaluation_keys_sha256": base.key_digest(prepared["validation"]),
              "train_rows": len(prepared["train"]), "validation_rows": len(prepared["validation"])}
    prior = json.loads((base.OUT / "terminal_result.json").read_text(encoding="utf-8"))
    exact = next((r for r in prior["fits"] if r.get("fold") == fold and r.get("stage") == stage
                  and r.get("arm") == arm and r.get("training_keys_sha256") == record["training_keys_sha256"]
                  and r.get("evaluation_keys_sha256") == record["evaluation_keys_sha256"]), None)
    if stage == "outer" and exact is not None:
        path = base.OUT / exact["model_file"]
        if old.sha(path) != exact["model_sha256"] or old.sha(base.OUT / exact["probability_file"]) != exact["probability_sha256"]:
            raise ValueError("reused artifact integrity failure")
        package = old.joblib.load(path)
        if package["train_stats"] != prepared["stats"] or package["encoder"].category_maps != prepared["encoder"].category_maps:
            raise ValueError("reused preprocessing mismatch")
        model = package["model"]
        pred = probability(model, prepared["xe"], arm, 700)
        if not np.array_equal(pred, np.load(base.OUT / exact["probability_file"], allow_pickle=False)):
            raise ValueError("reused prediction not exact")
        record.update(reused=True, model_path=str(path.relative_to(ROOT)), model_sha256=old.sha(path),
                      source_result_sha256=cfg["source_v1_result_sha256"])
        result["reused_fits"] += 1
    else:
        if result["new_fits"] >= cfg["max_new_fits"]:
            raise RuntimeError("fit cap")
        begin = time.monotonic()
        model = fit(arm, prepared["x"], prepared["train"], frozen, cfg,
                    cfg["inner_max_iterations"] if stage == "inner" else cfg["outer_iterations"])
        result["new_fits"] += 1
        pred = probability(model, prepared["xe"], arm, 700)
        path = OUT / (stem + ".joblib")
        old.joblib.dump({"model": model, "train_stats": prepared["stats"], "encoder": prepared["encoder"],
                         "bracket": arm == "bracket"}, path, compress=3)
        record.update(reused=False, model_path=str(path.relative_to(ROOT)), model_sha256=old.sha(path),
                      seconds=time.monotonic() - begin)
    np.save(OUT / (stem + ".npy"), pred, allow_pickle=False)
    record["probability_sha256"] = old.sha(OUT / (stem + ".npy"))
    if stage == "inner":
        for iteration in cfg["diagnostic_iterations"]:
            p = probability(model, prepared["xe"], arm, iteration)
            result["inner_diagnostics"].append({"fold": fold, "arm": arm, **diagnostic_row(
                prepared["validation"], p, prepared["rules"], frozen, iteration)})
    result["fits"].append(record)
    del model, prepared
    gc.collect()
    progress(result, f"{fold}/{stage}/{arm}/complete", started, cfg)


def build_oof(data, masks, frozen):
    parts, selections = [], {}
    for fold in FOLDS:
        itr, iva = masks[(fold, "inner")]
        inner = data.loc[iva].reset_index(drop=True)
        irules = old.rule_masks(inner, old.stats_fit(data.loc[itr].reset_index(drop=True)))
        probabilities = {a: np.load(OUT / f"{fold}_inner_{a}.npy", allow_pickle=False) for a in ARMS}
        choices = {name: policy.select_inner(inner, {"original": probabilities["original"],
                                                     "balanced": probabilities[b]}, irules, frozen)
                   for name, b in (("control", "balanced"), ("candidate", "bracket"))}
        selections[fold] = choices
        tr, va = masks[(fold, "outer")]
        val = data.loc[va].reset_index(drop=True)
        rules = old.rule_masks(val, old.stats_fit(data.loc[tr].reset_index(drop=True)))
        probs = {a: np.load(OUT / f"{fold}_outer_{a}.npy", allow_pickle=False) for a in ARMS}
        part = val[old.KEYS + ["row_id", "label", "anomaly_type"]].copy()
        part["fold"] = fold
        known = set(zip(data.loc[tr, "station"], data.loc[tr, "layer"], strict=True))
        part["known_station_layer"] = [k in known for k in zip(val.station, val.layer, strict=True)]
        for name, b in (("control", "balanced"), ("candidate", "bracket")):
            chosen, calibration = choices[name]
            part[name] = policy.policies(val, {"original": probs["original"], "balanced": probs[b]},
                                         rules, frozen, calibration)[chosen]
        parts.append(part)
    return pd.concat(parts, ignore_index=True), selections


def summarize(oof, ev):
    result = base.summarize_oof(oof, ev)
    result["metrics"]["all_twosided"] = result["metrics"].pop("all_forward")
    return result


def execute():
    cfg, prior_cfg, frozen, ev = contract()
    OUT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "runner_sha256": old.sha(__file__), "config_sha256": old.sha(CONFIG)}, handle)
    started = time.monotonic()
    result = {"experiment_id": ID, "pid": os.getpid(), "status": "RUNNING", "new_fits": 0,
              "reused_fits": 0, "fits": [], "inner_diagnostics": [], "official_rows": 0, "hidden_rows": 0,
              "csv_written": 0, "uploads": 0, "runner_sha256": old.sha(__file__),
              "config_sha256": old.sha(CONFIG), "surface": cfg["surface"], "threads": 2,
              "score_inversion_used": False, "diagnostic_changes_outer": False}
    try:
        data = base.training_source(prior_cfg)
        masks, counts = nested_masks(data, ev, cfg)
        expected = json.loads((support.REPORT / "result.json").read_text(encoding="utf-8"))
        for row in counts:
            if row["stage"] == "outer":
                ref = next(x for x in expected["folds"] if x["fold"] == row["fold"])
                if (row["training_keys_sha256"], row["evaluation_keys_sha256"]) != (ref["train_keys_sha256"], ref["validation_keys_sha256"]):
                    raise ValueError("support keys changed")
        result["support"] = counts
        old.write_json(REPORT / "support.json", {"status": "COUNT_SUPPORT_PASS", "folds": counts})
        for arms in (("original", "balanced"), ("bracket",)):
            for fold in FOLDS:
                for stage in ("inner", "outer"):
                    for arm in arms:
                        stage_run(data, masks, fold, stage, arm, cfg, prior_cfg, frozen, result, started)
            if len(arms) == 2:
                result["baseline_oof_sealed_before_challenger"] = True
        oof, choices = build_oof(data, masks, frozen)
        oof.to_parquet(OUT / "oof.parquet", index=False)
        result.update(summarize(oof, ev))
        result["earlier_inner_selections"] = choices
        result["learning_curve_diagnostic"] = diagnostic_summary(result["inner_diagnostics"])
        result.update(status="COMPLETE_QA_PENDING", oof_sha256=old.sha(OUT / "oof.parquet"))
    except Exception as exc:
        result.update(status="TERMINAL_TECHNICAL_OR_RESOURCE_FAILURE", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result["runtime_seconds"] = time.monotonic() - started
        old.write_json(OUT / "terminal_result.json", result)
        old.write_json(REPORT / "result.json", result)


def qa():
    cfg, prior_cfg, frozen, ev = contract()
    result = json.loads((OUT / "terminal_result.json").read_text(encoding="utf-8"))
    if result["status"] != "COMPLETE_QA_PENDING" or result["pid"] == os.getpid():
        raise ValueError("completed run and fresh process required")
    data = base.training_source(prior_cfg)
    masks, counts = nested_masks(data, ev, cfg)
    checks = {"support_exact": counts == result["support"], "runner_hash": old.sha(__file__) == result["runner_sha256"],
              "config_hash": old.sha(CONFIG) == result["config_sha256"], "oof_hash": old.sha(OUT / "oof.parquet") == result["oof_sha256"]}
    diagnostics = []
    for fit_record in result["fits"]:
        fold, stage, arm = (fit_record[k] for k in ("fold", "stage", "arm"))
        path = ROOT / fit_record["model_path"]
        key = f"{fold}_{stage}_{arm}"
        checks[key + "_model_hash"] = old.sha(path) == fit_record["model_sha256"]
        checks[key + "_prob_hash"] = old.sha(OUT / (key + ".npy")) == fit_record["probability_sha256"]
        package = old.joblib.load(path)
        val = data.loc[masks[(fold, stage)][1]].reset_index(drop=True)
        features = base.bundle(val, package["train_stats"], frozen, prior_cfg, arm == "bracket")
        x = package["encoder"].transform(features)
        p = probability(package["model"], x, arm, 700)
        checks[key + "_prob_exact"] = bool(np.array_equal(p, np.load(OUT / (key + ".npy"), allow_pickle=False)))
        if stage == "inner":
            rules = old.rule_masks(val, package["train_stats"])
            for iteration in cfg["diagnostic_iterations"]:
                diagnostics.append({"fold": fold, "arm": arm, **diagnostic_row(
                    val, probability(package["model"], x, arm, iteration), rules, frozen, iteration)})
        del package, features, x
        gc.collect()
    oof, choices = build_oof(data, masks, frozen)
    checks["oof_keys_bits"] = oof.equals(pd.read_parquet(OUT / "oof.parquet"))
    checks["selection_exact"] = json.loads(json.dumps(choices)) == result["earlier_inner_selections"]
    checks["metrics_ci_slices_exact"] = all(v == result[k] for k, v in summarize(oof, ev).items())
    checks["inner_diagnostics_exact"] = diagnostics == result["inner_diagnostics"]
    checks["diagnostic_summary_exact"] = diagnostic_summary(diagnostics) == result["learning_curve_diagnostic"]
    receipt = {"status": "PASS" if all(checks.values()) else "FAIL", "passed": sum(checks.values()),
               "total": len(checks), "checks": checks, "pid": os.getpid(), "training_pid": result["pid"],
               "new_fits": 0, "official_rows": 0, "csv_written": 0, "uploads": 0,
               "result_sha256": old.sha(OUT / "terminal_result.json")}
    old.write_json(REPORT / "independent-qa.json", receipt)
    print(json.dumps({"status": receipt["status"], "checks": len(checks)}))
    if not all(checks.values()):
        raise ValueError("QA failure; no automatic retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute()
    elif args.qa:
        qa()
    else:
        contract()
        print("CONTRACT_ONLY")
