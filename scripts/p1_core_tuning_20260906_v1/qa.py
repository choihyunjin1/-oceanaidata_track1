"""Post-terminal independent arithmetic/selection/model-lineage QA; zero fits."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("p1_tuning_run", HERE / "run.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
SPEC_Q = importlib.util.spec_from_file_location("champion_independent_qa", r.ROOT / "scripts/p1_champion_reconstruction_20260906_v1/qa_tree.py")
q = importlib.util.module_from_spec(SPEC_Q)
SPEC_Q.loader.exec_module(q)
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def independent_selector(rows, rules, cfg, contract):
    thresholds, decoded = {}, {}
    for component in contract["components"]:
        best_f1, best_threshold, best_bits = -1., None, None
        for threshold in cfg["threshold_grid"]:
            bits = r.core.decode(rows, rows["probability_" + component].to_numpy(), rules, cfg, threshold)
            f1 = q.counts(rows.label, bits)["f1"]
            if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
                best_f1, best_threshold, best_bits = f1, threshold, bits
        thresholds[component], decoded[component] = best_threshold, best_bits
    scores = {}
    winner, winner_f1 = None, -1.
    for policy in contract["policy_tie_order"]:
        left, right = contract["policies"][policy]
        scores[policy] = q.counts(rows.label, decoded[left] | decoded[right])
        if scores[policy]["f1"] > winner_f1:
            winner, winner_f1 = policy, scores[policy]["f1"]
    return {"selected": winner, "thresholds": thresholds, "inner_policy_metrics": scores,
            "scope": "earlier_inner_only", "tie_order": contract["policy_tie_order"]}


def schema(rows, components, *, oof=False):
    expected = r.KEYS + ["row_id", "label", "fold", "supported"]
    if not oof:
        expected += ["probability_" + c for c in components]
    expected += ["control", "candidate"]
    if list(rows) != expected or rows.empty or rows[r.KEYS].isna().any().any():
        raise ValueError("invalid schema/keys/empty")
    if rows.duplicated(r.KEYS).any() or rows.row_id.duplicated().any():
        raise ValueError("duplicate evaluation keys")
    if not rows.supported.isin([True, False]).all():
        raise ValueError("support flag missing")
    for arm in ("control", "candidate"):
        q.counts(rows.label, rows[arm])
    for c in components:
        p = rows["probability_" + c].to_numpy()
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("probability invalid")


def verify(output, seal_path):
    started = time.monotonic()
    destination = output / "independent-qa.json"
    if destination.exists():
        raise FileExistsError("immutable QA receipt exists")
    result = r.t.read_json(output / "terminal_result.json")
    if result.get("status") != "COMPLETE":
        raise ValueError("terminal COMPLETE required")
    sealed = r.check_seal(seal_path)
    r.t.verify_artifacts(output, result)
    cfg, contract = sealed["base"], sealed["config"]
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    check("exact_terminal_seal", result["seal_sha256"] == r.core.sha(seal_path))
    selections = [r.t.read_json(output / f["id"] / "selector.json")["selected"] for f in cfg["folds"]]
    check("attempted_completed_actual_model_count", result["attempted_fits"] == result["completed_fits"] == len(result["fit_receipts"]) ==
          r.fit_count(selections, contract, cfg) and 39 <= result["completed_fits"] <= 48)
    check("worker90min_and_no_official", 0 < result["runtime_seconds"] <= 5400 and all(result[k] == 0 for k in
          ("official_rows", "hidden_rows", "csv_written", "upload")))
    check("native_pool_CPU2", all(p["num_threads"] <= 2 for p in result["native_pools"]))
    replay = r.t.read_json(output / "fresh-replay-qa.json")
    check("fresh_PID_exact_replay", replay["status"] == "PASS" and replay["pid"] != result["worker_pid"] and
          replay["terminal_result_sha256"] == r.core.sha(output / "terminal_result.json") and replay["fits"] == 0)
    data = r.t.training_source(cfg)
    masks = q.independent_splits(data, cfg)
    support = {(s["fold"], s["stage"]): s for s in r.t.read_json(output / "support.json")}
    receipts = {q.normalized_fit_path(f["path"]): f for f in result["fit_receipts"]}
    check("unique_actual_fit_receipts", len(receipts) == result["completed_fits"])
    all_outer, categories = [], {"baseline": 0, "alternative_inner": 0, "alternative_outer": 0}
    replay_expected = 0
    for fold in cfg["folds"]:
        name = fold["id"]
        selector = r.t.read_json(output / name / "selector.json")
        for stage in ("inner", "outer"):
            components = list(contract["components"]) if stage == "inner" else r.outer_components(selector["selected"], contract)
            replay_expected += len(components) + 2
            tm, vm = masks[(name, stage)]
            train, valid = data.loc[tm].reset_index(drop=True), data.loc[vm].reset_index(drop=True)
            prefix = f"{name}/{stage}"
            folder = output / name / stage
            rec = support[(name, stage)]
            check(prefix + "/independent_split_support", len(train) == rec["train_rows"] and len(valid) == rec["validation_rows"] and
                  int(train.label.sum()) == rec["train_positive"] and int(valid.label.sum()) == rec["validation_positive"])
            check(prefix + "/purge_disjoint", not (tm & vm).any() and
                  (pd.to_datetime(train.time, utc=True) < pd.Timestamp(rec["train_cutoff_exclusive"])).all())
            rows = pd.read_parquet(folder / "predictions.parquet")
            schema(rows, components)
            check(prefix + "/exact_keys_labels_order", r.t.key_digest(rows) == r.t.key_digest(valid) == rec["validation_keys_sha256"] and
                  np.array_equal(rows.label, valid.label) and np.array_equal(rows.row_id, valid.row_id) and rows.fold.eq(name).all())
            known = set(zip(train.station, train.layer, strict=True))
            expected_support = np.array([key in known for key in zip(valid.station, valid.layer, strict=True)])
            check(prefix + "/all_supported_unseen_rows_retained", np.array_equal(rows.supported, expected_support) and
                  int(expected_support.sum()) == rec["supported_station_layer_rows"] and
                  int((~expected_support).sum()) == rec["unseen_station_layer_rows"])
            if stage == "outer":
                check(prefix + "/fixed_outer_denominator", len(rows) == contract["support_expected"][name])
            stored = r.core.joblib.load(folder / "preprocess.joblib")
            expected_files = {c: [f"{c}_{s}.joblib" for s in r.seeds(c, cfg)] for c in components}
            check(prefix + "/only_expected_components", stored["files"] == expected_files and
                  sorted(p.name for p in folder.glob("*.joblib") if p.name != "preprocess.joblib") ==
                  sorted(f for files in expected_files.values() for f in files))
            check(prefix + "/80_feature_encoder", len(stored["feature_columns"]) == len(stored["encoder"].feature_columns) == 80)
            for component, files in expected_files.items():
                for seed, filename in zip(r.seeds(component, cfg), files, strict=True):
                    path = folder / filename
                    rec_fit = receipts[path.relative_to(output).as_posix()]
                    model = r.core.joblib.load(path)
                    native = model.model if component.startswith("O_") else model
                    params = native.get_params()
                    expected = r.parameters(component, seed, cfg, contract)
                    trees = native.get_booster().num_boosted_rounds() if component.startswith("O_") else native.booster_.num_trees()
                    check(prefix + "/" + filename + "/recipe", trees == expected["n_estimators"] and native.n_features_in_ == 80 and
                          params["n_jobs"] == 2 and params["random_state"] == seed and all(params[k] == v for k, v in expected.items()))
                    check(prefix + "/" + filename + "/lineage", rec_fit["model_sha256"] == r.core.sha(path) and
                          rec_fit["seed"] == seed and rec_fit["component"] == component and rec_fit["parameters"] == expected and
                          rec_fit["train_rows"] == len(train) and rec_fit["train_positive"] == int(train.label.sum()) and
                          rec_fit["train_keys_sha256"] == r.t.key_digest(train) and rec_fit["target_sha256"] == r.t.digest(train.label.to_numpy(dtype=np.int8)))
                    category = "baseline" if component in r.BASE_COMPONENTS else "alternative_" + stage
                    categories[category] += 1
                    del model, native
            rules = r.core.rule_masks(valid, stored["stats"])
            if stage == "inner":
                check(prefix + "/independent_inner_only_selection", independent_selector(rows, rules, cfg, contract) == selector)
            decoded = {c: r.core.decode(rows, rows["probability_" + c].to_numpy(), rules, cfg, selector["thresholds"][c]) for c in components}
            for column, policy in (("control", "control"), ("candidate", selector["selected"])):
                left, right = contract["policies"][policy]
                check(prefix + "/" + column + "/fixed_union", np.array_equal(rows[column], decoded[left] | decoded[right]))
            if stage == "outer":
                all_outer.append(rows[r.KEYS + ["row_id", "label", "fold", "supported", "control", "candidate"]])
            del train, valid, rows, stored
            gc.collect()
    check("24_baseline15_alt_inner_exact", categories["baseline"] == 24 and categories["alternative_inner"] == 15 and categories["alternative_outer"] <= 9)
    check("every_probability_and_bits_replayed", replay["check_count"] == len(replay["checks"]) == replay_expected)
    oof = pd.read_parquet(output / "oof.parquet")
    schema(oof, [], oof=True)
    check("OOF_concat_421032_exact", len(oof) == 421032 and oof.equals(pd.concat(all_outer, ignore_index=True)))
    check("OOF_SHA_exact", r.core.sha(output / "oof.parquet") == result["oof_sha256"])
    verified = {}
    for scope, mask in {"primary_all_Q2_Q3_Q4": np.ones(len(oof), dtype=bool),
                        "secondary_old_Q3_Q4": oof.fold.isin(["2025_q3", "2025_q4"])}.items():
        part = oof.loc[mask]
        report = result["evaluation"][scope]
        verified[scope] = {arm: q.counts(part.label, part[arm]) for arm in ("control", "candidate")}
        check(scope + "/pure_counts_and_denominator", verified[scope] == report["metrics"] and len(part) == report["rows"] and
              int(part.label.sum()) == report["positive"] and r.t.key_digest(part) == report["keys_sha256"])
        days = pd.to_datetime(part.time, utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
        boot = q.independent_bootstrap(part.label, part.control, part.candidate, days)
        reported = report["paired_day_bootstrap"]
        check(scope + "/independent_bootstrap", np.allclose(boot["ci90"], reported["ci90"], rtol=0, atol=1e-15) and
              boot["p_improve"] == reported["p_improve"] and boot["delta"] == reported["delta_candidate_minus_control"] and
              boot["clusters"] == reported["n_clusters"])
        check(scope + "/uncertainty_not_veto", reported["probability_hard_gate"] is None and reported["automatic_promotion"] is False)
    expected_slices = []
    oof["day"] = pd.to_datetime(oof.time, utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    for fields in (["fold"], ["station", "layer"], ["fold", "supported"], ["day"]):
        for key, part in oof.groupby(fields, sort=True):
            m = {arm: q.counts(part.label, part[arm]) for arm in ("control", "candidate")}
            expected_slices.append({"by": fields, "key": str(key), "rows": len(part), "positive": int(part.label.sum()),
                "metrics": m, "delta_f1": m["candidate"]["f1"] - m["control"]["f1"]})
    check("all_risk_slices_exact", expected_slices == result["evaluation"]["slices"])
    for name, fields in (("worst_station_layer", ["station", "layer"]), ("worst_day", ["day"])):
        check(name + "/exact", min((s for s in expected_slices if s["by"] == fields), key=lambda s: s["delta_f1"]) == result["evaluation"][name])
    check("distributed_source_immutable", r.core.sha(Path(os.environ["P1_DATA_DIR"]) / "train.csv") == cfg["train_sha256"])
    r.check_seal(seal_path)
    r.t.verify_artifacts(output, result)
    qa = {"status": "PASS", "pid": os.getpid(), "check_count": len(checks), "checks": checks,
          "terminal_result_sha256": r.core.sha(output / "terminal_result.json"), "seal_sha256": r.core.sha(seal_path),
          "qa_source_sha256": r.core.sha(__file__), "independent_helper_sha256": r.core.sha(q.__file__),
          "fresh_replay_sha256": r.core.sha(output / "fresh-replay-qa.json"), "model_fit_categories": categories,
          "metrics_independently_recomputed": verified, "runtime_seconds": time.monotonic() - started,
          "fits": 0, "official_rows": 0, "hidden_rows": 0, "csv_written": 0, "upload": 0,
          "caveats": ["retrospective exposed validation", "new overall primary with old Q3Q4 separately reported",
                      "CPU2 fresh baseline; no stored CPU4 model/OOF reuse", "unselected altered policies not outer evaluated",
                      "saved replay not repeated full retraining or official score", "full train/inference still requires root review"]}
    r.core.write_json(destination, qa)
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = verify(args.output.resolve(), args.seal.resolve())
    print({"status": result["status"], "checks": result["check_count"], "fits": 0})


if __name__ == "__main__":
    main()
