"""Independent 48-model ensemble arithmetic; new fits0, official access0."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("p2_multiseed_audit", Path(__file__).with_name("run.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
OLD = importlib.util.spec_from_file_location("p2_prior_independent_arithmetic", m.OLD_SCRIPT.with_name("qa.py"))
q = importlib.util.module_from_spec(OLD)
OLD.loader.exec_module(q)


def run():
    c = m.verify()
    result = m.r.read(m.OUT / "terminal_result.json")
    replay = m.r.read(m.OUT / "fresh-replay.json")
    checks = []

    def check(name, condition):
        checks.append({"name": name, "pass": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def same_metric(name, actual, expected):
        check(name + ":n", actual["n"] == expected["n"])
        for key in ("sse", "rmse_C", "bias_C"):
            a, b = actual[key], expected[key]
            check(name + key, a is None and b is None or a is not None and b is not None and np.isclose(a, b, rtol=1e-11, atol=1e-11))

    check("complete", result["status"] == "COMPLETE_28_NEW_PLUS_20_REUSED_MODELS")
    check("28new", result["new_fits"] == result["attempted_fits"] == len(result["fits"]) == 28)
    check("48combined", result["combined_models"] == 48 and result["reused_models"] == 20)
    check("scope", result["official_rows"] == result["csv_written"] == result["upload"] == 0)
    check("replay", replay["status"] == "PASS" and replay["models"] == 48 and replay["max_absolute_error"] == 0)
    check("replay_link", replay["terminal_result_sha256"] == m.r.sha(m.OUT / "terminal_result.json"))
    check("freshPID", replay["pid"] != replay["training_pid"] == result["pid"])
    check("evaluationsha", result["evaluation_sha256"] == m.r.sha(m.OUT / "evaluation.npz"))
    expected_new = {(a, f, s) for a in c["arms"] for f in c["new_folds"] for s in c["new_seeds"]}
    check("newproduct", {(f["recipe"], f["fold"], f["seed"]) for f in result["fits"]} == expected_new)
    fits = [(m.r.OUT, f) for f in m.reused()] + [(m.OUT, f) for f in result["fits"]]
    check("combinedproduct", {(f["recipe"], f["fold"], f["seed"]) for _, f in fits} ==
          {(a, f"B{k}", s) for a in c["arms"] for k in range(1, 9) for s in c["all_seeds"]})
    values = np.load(m.OUT / "evaluation.npz", allow_pickle=False)
    y, folds, times, layers = (values[k] for k in ("truth", "fold", "time", "layer"))
    prior = np.load(m.r.OUT / "evaluation.npz", allow_pickle=False)
    for key in ("key", "truth", "fold", "time", "layer", "natural_T5_missing"):
        check(key + ":prior_exact", np.array_equal(values[key], prior[key]))
    check("all_rows", len(y) == len(np.unique(values["key"])) == 166268)
    interval = np.zeros(len(y), bool)
    parsed = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    for fold in m.r.contract()["folds"]:
        take = folds == fold["id"]
        interval |= take & (parsed >= pd.Timestamp(fold["end"]) - pd.Timedelta(days=17)) & (parsed < pd.Timestamp(fold["end"]))
        for arm in c["arms"]:
            items = sorted(((root, f) for root, f in fits if f["fold"] == fold["id"] and f["recipe"] == arm), key=lambda item: item[1]["seed"])
            check(arm + fold["id"] + ":input_parity", len({tuple(f["training_arrays_sha256"]) for _, f in items}) == 1 and
                  len({f["train_keys_sha256"] for _, f in items}) == len({f["train_truth_sha256"] for _, f in items}) == 1)
            for root, fit in items:
                check(fit["fit_id"] + ":files", m.r.sha(root / fit["model_file"]) == fit["model_sha256"] and
                      m.r.sha(root / fit["predictions_file"]) == fit["predictions_sha256"])
                check(fit["fit_id"] + ":resource", fit["cpu_threads"] == 2 and fit["device"] == "cuda")
                recipe = next(rec for rec in m.r.cfg()["recipes"] if rec["id"] == arm)
                check(fit["fit_id"] + ":recipe", fit["recipe_config"] == m.r.recipe_config(m.r.cfg(), recipe))
            for surface in ("natural", "outage"):
                predictions = []
                for root, fit in items:
                    with np.load(root / fit["predictions_file"], allow_pickle=False) as stored:
                        if surface not in stored:
                            predictions = []
                            break
                        predictions.append(stored[surface])
                if predictions:
                    check(arm + fold["id"] + surface + ":mean3", np.array_equal(np.mean(predictions, axis=0), values[surface + "_" + arm][take]))
            same_metric(arm + fold["id"] + ":natural", q.metric(y[take], values["natural_" + arm][take]), result["summary"]["natural"][arm][fold["id"]])
            stress = result["summary"]["outage"][arm][fold["id"]]
            if stress["status"] != "SUPPORT_BLOCKED":
                same_metric(arm + fold["id"] + ":stress", q.metric(y[take], values["outage_" + arm][take]), stress["whole_fold"])
                same_metric(arm + fold["id"] + ":interval", q.metric(y[take & interval], values["outage_" + arm][take & interval]), stress["interval"])
                if fold["id"] in ("B4", "B8"):
                    check(arm + fold["id"] + ":not_estimable", stress["status"] == "NOT_ESTIMABLE_NO_ROWS" and stress["interval"]["n"] == 0)
    primary = folds == "B3"
    for arm in c["arms"]:
        for surface in ("natural", "outage"):
            check(arm + surface + ":B3exact", np.array_equal(values[surface + "_" + arm][primary], prior[f"B3_{arm}_{surface}_all3"]))
        same_metric(arm + ":all8", q.metric(y, values["natural_" + arm]), result["summary"]["natural"][arm]["all8_pooled"])
        same_metric(arm + ":B3", q.metric(y[primary], values["natural_" + arm][primary]), result["summary"]["natural"][arm]["B3_primary"])
    risk = {}
    for scope, take in (("B3_primary", primary), ("all8_secondary", np.ones(len(y), bool))):
        for surface in ("natural", "outage"):
            a, b = values[surface + "_C60"], values[surface + "_L120"]
            if np.isfinite(a[take]).all() and np.isfinite(b[take]).all():
                risk[scope + "_" + surface] = q.risks(y[take], a[take], b[take], times[take], layers[take], values["natural_T5_missing"][take], interval[take])
    m.verify()
    m.r.save(m.REPORT / "independent-qa.json", {"status": "PASS", "checks_passed": len(checks), "checks": checks,
        "risk": risk, "new_fits": 0, "official_rows": 0, "primary_B3_unchanged": True,
        "terminal_result_sha256": m.r.sha(m.OUT / "terminal_result.json"), "evaluation_sha256": m.r.sha(m.OUT / "evaluation.npz"),
        "replay_sha256": m.r.sha(m.OUT / "fresh-replay.json"), "retrospective_not_independent_holdout": True})
    print({"status": "PASS", "checks": len(checks), "fits": 0})


if __name__ == "__main__":
    run()
