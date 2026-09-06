"""Independent arithmetic, seed-ensemble and saved-state QA; no fits/official I/O."""
# ruff: noqa: E402
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("p2_c3_audit_runner", Path(__file__).with_name("run.py"))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def metric(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("finite same-row values required")
    e = p - y
    sse = float(np.sum(e * e, dtype=np.float64))
    return {"n": len(y), "sse": sse, "rmse_C": float(np.sqrt(sse / len(y))) if len(y) else None,
            "bias_C": float(e.mean()) if len(y) else None}


def clusters(times):
    delta = pd.DatetimeIndex(pd.to_datetime(times, utc=True)) - pd.Timestamp("2024-01-01T00:00:00+09:00")
    return np.asarray(delta // pd.Timedelta(days=7), dtype=np.int64)


def bootstrap(y, control, candidate, block, resamples=2000, seed=20260906):
    y, control, candidate, block = map(np.asarray, (y, control, candidate, block))
    unique = np.unique(block)
    if len(unique) < 2:
        return {"status": "NOT_ESTIMABLE", "blocks": len(unique), "rows": len(y)}
    n = np.array([(block == k).sum() for k in unique])
    c = np.array([np.sum((control[block == k] - y[block == k]) ** 2) for k in unique])
    p = np.array([np.sum((candidate[block == k] - y[block == k]) ** 2) for k in unique])
    draws = np.random.default_rng(seed).integers(len(unique), size=(resamples, len(unique)))
    den = n[draws].sum(axis=1)
    difference = np.sqrt(p[draws].sum(axis=1) / den) - np.sqrt(c[draws].sum(axis=1) / den)
    group_delta = np.sqrt(p / n) - np.sqrt(c / n)
    return {"status": "ESTIMABLE_RETROSPECTIVE", "blocks": len(unique), "rows": len(y),
        "delta_rmse_C": float(np.sqrt(p.sum() / n.sum()) - np.sqrt(c.sum() / n.sum())),
        "ci90_C": np.quantile(difference, [.05, .95]).tolist(),
        "empirical_improvement_fraction": float(np.mean(difference < 0)),
        "worst_7day_delta_rmse_C": float(group_delta.max()), "worsened_blocks": int((group_delta > 0).sum()),
        "resamples": resamples, "seed": seed, "interpretation": "not official improvement probability or fresh holdout"}


def risks(y, c, p, times, layers, missing, interval):
    masks = {"whole_scope": np.ones(len(y), bool), "natural_T5_missing": missing, "outage_interval": interval}
    masks.update({f"layer{k}": layers == k for k in (2, 3, 4)})
    output = {}
    for label, take in masks.items():
        a, b = metric(y[take], c[take]), metric(y[take], p[take])
        output[label] = {"control": a, "candidate": b,
            "delta_rmse_C": b["rmse_C"] - a["rmse_C"] if a["n"] else None,
            "paired_7day": bootstrap(y[take], c[take], p[take], clusters(times[take]))}
    return output


def run():
    config = r.verify_seal()
    result = r.read(r.OUT / "terminal_result.json")
    replay = r.read(r.OUT / "fresh-replay.json")
    checks = []

    def check(name, condition):
        checks.append({"name": name, "pass": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def compare(name, actual, expected):
        check(name + ":n", actual["n"] == expected["n"])
        for field in ("sse", "rmse_C", "bias_C"):
            a, b = actual[field], expected[field]
            check(name + ":" + field, a is None and b is None or a is not None and b is not None and
                  np.isclose(a, b, rtol=1e-11, atol=1e-11))

    check("complete", result["status"] == "COMPLETE_24_SCREEN_PLUS_4_PRIMARY_SEED_FITS")
    check("28fits", result["completed_fits"] == result["attempted_fits"] == len(result["fits"]) == 28)
    check("same_resource", result["device"] == "cuda" and result["cpu_threads"] == 2)
    check("source", result["source_sha256"] == config["source_sha256"])
    check("zero_official", all(result[k] == 0 for k in ("official_rows", "hidden_rows", "csv_written", "upload")))
    check("replay_link", replay["status"] == "PASS" and replay["terminal_result_sha256"] == r.sha(r.OUT / "terminal_result.json"))
    check("fresh_pid", replay["pid"] != replay["training_pid"] == result["worker_pid"])
    check("28_replays", replay["full_models_replayed"] == 28 and replay["maximum_absolute_error"] == 0)
    check("evaluation_sha", r.sha(r.OUT / "evaluation.npz") == result["evaluation_sha256"])
    values = np.load(r.OUT / "evaluation.npz", allow_pickle=False)
    y, folds, layer, times = (values[k] for k in ("truth", "fold", "layer", "time"))
    primary = folds == "B3"
    check("exact_population", len(y) == config["expected_eligible_rows"] == len(np.unique(values["key"])))
    check("8fold", set(folds) == {f"B{k}" for k in range(1, 9)})
    check("caveat", result["confirmation_is_new_holdout"] is False and result["all8fold3seed_evaluated"] is False)
    interval = np.zeros(len(y), bool)
    parsed = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    for fold in r.contract()["folds"]:
        take = folds == fold["id"]
        support = next(s for s in result["support"]["folds"] if s["fold"] == fold["id"])
        check(fold["id"] + ":key", r.array_sha(values["key"][take]) == support["valid_keys_sha256"])
        check(fold["id"] + ":truth", r.array_sha(y[take]) == support["truth_sha256"])
        check(fold["id"] + ":n", int(take.sum()) == support["valid_rows"])
        interval |= take & (parsed >= pd.Timestamp(fold["end"]) - pd.Timedelta(days=17)) & (parsed < pd.Timestamp(fold["end"]))
    for arm in r.RECIPE_IDS:
        natural, outage = values["natural_" + arm], values["outage_" + arm]
        scopes = [("all8_pooled", np.ones(len(y), bool)), ("B3_primary", primary)]
        scopes += [(f"B{k}", folds == f"B{k}") for k in range(1, 9)]
        for name, take in scopes:
            compare(arm + name, metric(y[take], natural[take]), result["screen"]["natural"][arm][name])
        for fold in r.contract()["folds"]:
            take = folds == fold["id"]
            ref = result["screen"]["outage"][arm][fold["id"]]
            if ref["status"] == "SUPPORT_BLOCKED":
                check(arm + fold["id"] + ":unscored", ref["metrics"] is None)
            else:
                compare(arm + fold["id"] + ":outage", metric(y[take], outage[take]), ref["whole_fold"])
                compare(arm + fold["id"] + ":interval", metric(y[take & interval], outage[take & interval]), ref["interval"])
                if fold["id"] in ("B4", "B8"):
                    check(arm + fold["id"] + ":empty", ref["status"] == "NOT_ESTIMABLE_NO_ROWS" and ref["interval"]["n"] == 0)
        for sl in result["screen"]["slices"][arm]:
            take = (folds == sl["fold"]) & (layer == sl["layer"])
            compare(arm + sl["fold"] + str(sl["layer"]), metric(y[take], natural[take]), sl)
    winner = result["selected_challenger"]
    selected = min(("L120", "D60"), key=lambda arm: (metric(y[primary], values["natural_" + arm][primary])["rmse_C"],
                    metric(y, values["natural_" + arm])["rmse_C"], ("L120", "D60").index(arm)))
    check("fixed_selection", selected == winner)
    expected = {(arm, f"B{k}", config["seeds"][0]) for arm in r.RECIPE_IDS for k in range(1, 9)}
    expected |= {(arm, "B3", seed) for arm in ("C60", winner) for seed in config["seeds"][1:]}
    check("fit_product", {(f["recipe"], f["fold"], f["seed"]) for f in result["fits"]} == expected)
    for fit in result["fits"]:
        name = fit["fit_id"]
        check(name + ":hash", r.sha(r.OUT / fit["model_file"]) == fit["model_sha256"] and r.sha(r.OUT / fit["predictions_file"]) == fit["predictions_sha256"])
        recipe = next(p for p in config["recipes"] if p["id"] == fit["recipe"])
        check(name + ":recipe", fit["recipe_config"] == r.recipe_config(config, recipe) and fit["epochs"] == recipe["epochs"])
        check(name + ":resource", fit["device"] == "cuda" and fit["cpu_threads"] == 2)
        check(name + ":mass", np.isclose(fit["training"]["original_rows"], fit["training"]["training_weight_sum"], rtol=1e-5))
        if fit["seed"] == config["seeds"][0]:
            with np.load(r.OUT / fit["predictions_file"], allow_pickle=False) as predictions:
                check(name + ":connect", np.array_equal(predictions["natural"], values["natural_" + fit["recipe"]][folds == fit["fold"]]))
    for arm in ("C60", winner):
        fits = sorted((f for f in result["fits"] if f["recipe"] == arm and f["fold"] == "B3"), key=lambda f: f["seed"])
        for surface in ("natural", "outage"):
            predictions = []
            for fit in fits:
                with np.load(r.OUT / fit["predictions_file"], allow_pickle=False) as prediction:
                    predictions.append(prediction[surface])
            for label, subset in (("all3", predictions), ("heldback2", predictions[1:])):
                mean = np.mean(subset, axis=0)
                check(arm + surface + label + ":mean", np.array_equal(mean, values[f"B3_{arm}_{surface}_{label}"]))
                compare(arm + surface + label, metric(y[primary], mean), result["confirmation"][arm][surface + "_" + label])
    delta = result["confirmation"][winner]["natural_all3"]["rmse_C"] - result["confirmation"]["C60"]["natural_all3"]["rmse_C"]
    check("retention", result["retained"] == (delta < 0) and np.isclose(delta, result["primary_three_seed_delta_rmse_C"], rtol=0, atol=1e-14))
    risk = {}
    for arm in ("L120", "D60"):
        for scope, take in (("all8", np.ones(len(y), bool)), ("B3", primary)):
            for surface in ("natural", "outage"):
                c, p = values[surface + "_C60"], values[surface + "_" + arm]
                if np.isfinite(c[take]).all() and np.isfinite(p[take]).all():
                    risk["screen_" + arm + "_" + scope + "_" + surface] = risks(y[take], c[take], p[take], times[take], layer[take], values["natural_T5_missing"][take], interval[take])
    for label in ("all3", "heldback2"):
        for surface in ("natural", "outage"):
            c, p = (values[f"B3_{arm}_{surface}_{label}"] for arm in ("C60", winner))
            risk["B3_" + label + "_" + surface] = risks(y[primary], c, p, times[primary], layer[primary], values["natural_T5_missing"][primary], interval[primary])
    r.verify_seal()
    r.save(r.REPORT / "independent-qa.json", {"status": "PASS", "checks_passed": len(checks), "checks": checks,
        "terminal_result_sha256": r.sha(r.OUT / "terminal_result.json"), "evaluation_sha256": r.sha(r.OUT / "evaluation.npz"),
        "replay_sha256": r.sha(r.OUT / "fresh-replay.json"), "source_support_sha256": r.sha(r.REPORT / "source-support.json"),
        "risk": risk, "new_fits": 0, "official_rows": 0, "arithmetic_rtol": 1e-11, "arithmetic_atol": 1e-11,
        "selection_caveat": "B3 used for firstseed selection; heldback seeds measure sensitivity on exposed B3, not independent holdout"})
    print({"status": "PASS", "checks": len(checks), "fits": 0})


if __name__ == "__main__":
    run()
