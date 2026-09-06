"""Post-terminal, train-only independent arithmetic/lineage QA; zero model fits."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import time
from pathlib import Path, PurePosixPath

SPEC = importlib.util.spec_from_file_location("champion_tree_qa_source", Path(__file__).with_name("tree.py"))
t = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t)
import numpy as np
import pandas as pd


def counts(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("empty or mismatched metric inputs")
    if not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("nonbinary/nonfinite values")
    tp = int(np.count_nonzero((y == 1) & (p == 1)))
    fp = int(np.count_nonzero((y == 0) & (p == 1)))
    fn = int(np.count_nonzero((y == 1) & (p == 0)))
    return {"rows": len(y), "tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.}


def normalized_fit_path(value):
    """Producer records native Windows separators; comparisons use relative POSIX."""
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in str(path):
        raise ValueError("fit receipt path must stay relative")
    return path.as_posix()


def independent_splits(frame, cfg):
    """Vectorized run boundaries, independent of production p1_run_metadata."""
    y = frame.label.to_numpy()
    times = pd.to_datetime(frame.time, utc=True)
    group = frame.station.ne(frame.station.shift()) | frame.layer.ne(frame.layer.shift())
    boundary = group | times.diff().ne(pd.Timedelta(minutes=10)) | frame.label.ne(frame.label.shift())
    run = boundary.cumsum()
    nanoseconds = pd.Series([v.value for v in times])
    starts = nanoseconds.groupby(run).transform("min").to_numpy()
    ends = nanoseconds.groupby(run).transform("max").to_numpy()
    output = {}
    for fold in cfg["folds"]:
        start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
        stop = start - pd.Timedelta(days=cfg["purge_days"])
        begin = stop - pd.Timedelta(days=cfg["inner_days"])
        for stage, left, right in [("inner", begin, stop), ("outer", start, end)]:
            cutoff = left - pd.Timedelta(days=cfg["purge_days"])
            train = (times < cutoff).to_numpy() & ((y == 0) | (ends < cutoff.value))
            valid = ((y == 0) & (times >= left).to_numpy() & (times < right).to_numpy()) | (
                (y == 1) & (starts >= left.value) & (starts < right.value))
            if (train & valid).any() or not train.any() or not valid.any():
                raise ValueError("split overlap/empty")
            output[(fold["id"], stage)] = (train, valid)
    return output


def schema(frame):
    expected = t.KEYS + ["row_id", "label", "fold", "probability_O", "probability_B", *t.ARMS]
    if list(frame) != expected or frame.empty or frame[t.KEYS].isna().any().any():
        raise ValueError("OOF schema/empty/keys mismatch")
    if frame.duplicated(t.KEYS).any() or frame.row_id.duplicated().any():
        raise ValueError("OOF duplicate keys")
    for arm in t.ARMS:
        counts(frame.label, frame[arm])
    for name in ("probability_O", "probability_B"):
        p = frame[name].to_numpy()
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("invalid probability")


def independent_bootstrap(y, baseline, candidate, groups):
    """Preaggregate confusion counts per day; resample blocks, never rows."""
    y, b, c = np.asarray(y), np.asarray(baseline), np.asarray(candidate)
    labels = list(dict.fromkeys(list(groups)))
    mapping = {label: i for i, label in enumerate(labels)}
    ids = np.array([mapping[g] for g in groups])

    def stats(p):
        return np.stack([np.bincount(ids, weights=v.astype(np.int64), minlength=len(labels))
            for v in [(y == 1) & (p == 1), (y == 0) & (p == 1), (y == 1) & (p == 0)]], axis=1)

    base, cand = stats(b), stats(c)
    rng = np.random.default_rng(20260906)
    values = []
    for _ in range(2000):
        sample = rng.integers(0, len(labels), len(labels))
        totals = np.stack([base[sample].sum(axis=0), cand[sample].sum(axis=0)])
        denom = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        f1 = np.divide(2 * totals[:, 0], denom, out=np.zeros(2), where=denom != 0)
        values.append(f1[1] - f1[0])
    return {"ci90": np.quantile(values, [.05, .95]).tolist(),
            "p_improve": float((np.asarray(values) > 0).mean()), "clusters": len(labels),
            "delta": counts(y, c)["f1"] - counts(y, b)["f1"]}


def independent_selector(inner, rules, cfg):
    chosen, bit = {}, {}
    for arm in ("O", "B"):
        ranked = []
        for threshold in cfg["threshold_grid"]:
            bits = t.core.decode(inner, inner[f"probability_{arm}"].to_numpy(), rules, cfg, threshold)
            ranked.append((counts(inner.label, bits)["f1"], threshold, bits))
        winner = sorted(ranked, key=lambda row: (row[0], row[1]))[-1]
        chosen[arm], bit[arm] = winner[1], winner[2]
    candidates = {"B": bit["B"], "O": bit["O"], "AND": bit["B"] & bit["O"], "OR": bit["B"] | bit["O"]}
    cells = []
    for (station, layer), index in inner.groupby(["station", "layer"], sort=True).indices.items():
        winner, best = "B", -1.
        for arm in ("B", "O", "AND", "OR"):
            f1 = counts(inner.label.iloc[index], candidates[arm][index])["f1"]
            if f1 > best:
                winner, best = arm, f1
        cells.append({"station": str(station), "layer": int(layer), "policy": winner,
                      "inner_rows": len(index), "inner_f1": best})
    return {"thresholds": chosen, "cells": cells, "unseen_cell_fallback": "B",
            "selection_scope": "earlier_inner_only", "cell_tie_order": ["B", "O", "AND", "OR"]}


def verify(output, seal_path):
    started = time.monotonic()
    result = t.read_json(output / "terminal_result.json")
    if result.get("status") != "COMPLETE" or result.get("phase") != "historical":
        raise ValueError("complete historical terminal required before reads")
    if (output / "independent-qa.json").exists():
        raise FileExistsError("do not overwrite QA receipt")
    seal = t.check_seal(seal_path)
    t.verify_artifacts(output, result)
    cfg, checks = seal["config"], []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    check("terminal_exact_seal", result["seal_sha256"] == t.core.sha(seal_path))
    check("24_attempted_completed_receipts", result["attempted_fits"] == result["completed_fits"] ==
          result["fit_cap"] == len(result["fit_receipts"]) == 24)
    check("runtime_within_90_minutes", 0 < result["runtime_seconds"] <= 5400)
    check("official_hidden_csv_upload_zero", all(result[k] == 0 for k in
                                               ("official_rows", "hidden_rows", "csv_written", "upload")))
    replay = t.read_json(output / "fresh-replay-qa.json")
    check("fresh_process_replay_PASS", replay["status"] == "PASS" and replay["fits"] == 0 and
          replay["pid"] != result["worker_pid"] and replay["check_count"] == 36 and
          replay["terminal_result_sha256"] == t.core.sha(output / "terminal_result.json"))
    source = Path(os.environ["P1_DATA_DIR"]).resolve() / "train.csv"
    check("training_source_SHA", t.core.sha(source) == cfg["train_sha256"])
    data = pd.read_csv(source, usecols=t.RAW + ["label"])
    check("source_population_keys", len(data) == 776706 and not data.duplicated(t.KEYS).any())
    data["row_id"] = np.arange(len(data))
    data.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    data.reset_index(drop=True, inplace=True)
    masks = independent_splits(data, cfg)
    support = {(r["fold"], r["stage"]): r for r in t.read_json(output / "support.json")}
    fit_map = {normalized_fit_path(r["path"]): r for r in result["fit_receipts"]}
    check("model_paths_unique", len(fit_map) == 24)
    all_outer, model_counts = [], {"O": 0, "B": 0}
    for fold in cfg["folds"]:
        name = fold["id"]
        for stage in ("inner", "outer"):
            folder = output / name / stage
            train_mask, valid_mask = masks[(name, stage)]
            training, valid = data.loc[train_mask], data.loc[valid_mask]
            record = support[(name, stage)]
            prefix = f"{name}/{stage}"
            check(prefix + "/split_counts", len(training) == record["train_rows"] and
                  len(valid) == record["validation_rows"] and int(training.label.sum()) == record["train_positive"] and
                  int(valid.label.sum()) == record["validation_positive"])
            check(prefix + "/purge_and_disjoint", not (train_mask & valid_mask).any() and
                  (pd.to_datetime(training.time, utc=True) < pd.Timestamp(record["train_cutoff_exclusive"])).all())
            pred = pd.read_parquet(folder / "predictions.parquet")
            schema(pred)
            check(prefix + "/exact_keys_labels_order", t.key_digest(pred) == t.key_digest(valid) ==
                  record["validation_keys_sha256"] and np.array_equal(pred.row_id, valid.row_id) and
                  np.array_equal(pred.label, valid.label))
            check(prefix + "/fold_names", pred.fold.eq(name).all())
            preprocess = t.core.joblib.load(folder / "preprocess.joblib")
            expected_models = [f"O_{cfg['O_seed']}.joblib"] + [f"B_{s}.joblib" for s in cfg["B_seeds"]]
            check(prefix + "/exact_O1_B3_files", preprocess["model_files"] == expected_models and
                  sorted(p.name for p in folder.glob("[OB]_*.joblib")) == sorted(expected_models))
            check(prefix + "/features80", len(preprocess["feature_columns"]) ==
                  len(preprocess["encoder"].feature_columns) == 80)
            for model_name in expected_models:
                path = folder / model_name
                arm, seed = model_name[0], int(model_name[2:-7])
                record_fit = fit_map[path.relative_to(output).as_posix()]
                model = t.core.joblib.load(path)
                actual = model.model if arm == "O" else model
                params = actual.get_params()
                ntrees = actual.get_booster().num_boosted_rounds() if arm == "O" else actual.booster_.num_trees()
                check(prefix + "/" + model_name + "/recipe", ntrees == 700 and
                      actual.n_features_in_ == 80 and params["n_jobs"] == 4 and params["random_state"] == seed)
                expected_params = cfg["xgboost_parameters"] if arm == "O" else cfg["lightgbm_parameters"]
                check(prefix + "/" + model_name + "/fixed_parameters", all(params[k] == v for k, v in expected_params.items()))
                check(prefix + "/" + model_name + "/fit_receipt", record_fit["model_sha256"] == t.core.sha(path) and
                      record_fit["seed"] == seed and record_fit["train_rows"] == len(training) and
                      record_fit["train_keys_sha256"] == t.key_digest(training) and
                      record_fit["target_sha256"] == t.digest(training.label.to_numpy(dtype=np.int8)))
                model_counts[arm] += 1
                del model, actual
            rules = t.core.rule_masks(valid.reset_index(drop=True), preprocess["stats"])
            selector = t.read_json(output / name / "selector.json")
            if stage == "inner":
                check(prefix + "/independent_threshold_cell_selection", independent_selector(pred, rules, cfg) == selector)
            check(prefix + "/union_exact", np.array_equal(pred.union, pred.O | pred.B))
            mapping = {(r["station"], r["layer"]): r["policy"] for r in selector["cells"]}
            expected_router = pred.B.to_numpy().copy()
            alternatives = {"B": pred.B.to_numpy(), "O": pred.O.to_numpy(),
                            "AND": (pred.B & pred.O).to_numpy(), "OR": (pred.B | pred.O).to_numpy()}
            for cell, rows in pred.groupby(["station", "layer"], sort=False).indices.items():
                expected_router[rows] = alternatives[mapping.get(cell, "B")][rows]
            check(prefix + "/router_exact_frozen_cells", np.array_equal(expected_router, pred.router))
            if stage == "outer":
                all_outer.append(pred)
            del training, valid, preprocess, pred
            gc.collect()
    check("model_count_O6_B18", model_counts == {"O": 6, "B": 18})
    oof = pd.read_parquet(output / "oof.parquet")
    schema(oof)
    check("OOF_outer_concat_exact", oof.equals(pd.concat(all_outer, ignore_index=True)))
    check("OOF_SHA", t.core.sha(output / "oof.parquet") == result["oof_sha256"])
    selections = {"primary_Q3_Q4": oof.fold.isin(["2025_q3", "2025_q4"]),
                  "all_three_folds": np.ones(len(oof), dtype=bool),
                  "H1_2025_partial_Q2_not_full_H1": oof.fold.eq("2025_q2")}
    verified_metrics = {}
    for scope, mask in selections.items():
        part = oof.loc[mask]
        report = result["evaluation"][scope]
        check(scope + "/denominator", len(part) == report["rows"] and t.key_digest(part) == report["keys_sha256"])
        verified_metrics[scope] = {arm: counts(part.label, part[arm]) for arm in t.ARMS}
        for arm in t.ARMS:
            check(scope + "/" + arm + "/pure_counts", verified_metrics[scope][arm] == report["metrics"][arm])
        days = pd.to_datetime(part.time, utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
        for arm in ("O", "union", "router"):
            boot = independent_bootstrap(part.label, part.B, part[arm], days)
            reported = report["vs_B"][arm]
            check(scope + "/" + arm + "/CI90_p_delta", np.allclose(boot["ci90"], reported["ci90"], rtol=0, atol=1e-15) and
                  boot["p_improve"] == reported["p_improve"] and boot["delta"] == reported["delta_candidate_minus_control"] and
                  boot["clusters"] == reported["n_clusters"])
            check(scope + "/" + arm + "/risk_not_hardgate", reported["probability_hard_gate"] is None and
                  reported["automatic_promotion"] is False and reported["resamples"] == 2000 and
                  reported["seed"] == 20260906)
    expected_slices = []
    for columns in (["fold"], ["station", "layer"]):
        for key, part in oof.groupby(columns, sort=True):
            expected_slices.append({"by": list(columns), "key": str(key), "rows": len(part),
                                    "metrics": {arm: counts(part.label, part[arm]) for arm in t.ARMS}})
    check("risk_slices_exact", expected_slices == result["evaluation"]["slices"])
    worst = min(r["metrics"]["router"]["f1"] - r["metrics"]["B"]["f1"] for r in expected_slices)
    check("worst_slice_exact", worst == result["evaluation"]["worst_router_minus_B"])
    check("Q4_inner_deployment_no_additional_fit", result["deployment_selector"] == "2025_q4/selector.json")
    check("source_immutable_after_QA", t.core.sha(source) == cfg["train_sha256"])
    t.check_seal(seal_path)
    t.verify_artifacts(output, result)
    qa = {"status": "PASS", "pid": os.getpid(), "terminal_result_sha256": t.core.sha(output / "terminal_result.json"),
          "seal_sha256": t.core.sha(seal_path), "qa_source_sha256": t.core.sha(__file__),
          "fresh_replay_sha256": t.core.sha(output / "fresh-replay-qa.json"),
          "check_count": len(checks), "checks": checks, "model_counts": model_counts,
          "metrics_independently_recomputed": verified_metrics, "runtime_seconds": time.monotonic() - started,
          "fits": 0, "official_rows": 0, "hidden_rows": 0, "csv_written": 0, "upload": 0,
          "caveats": ["retrospective exposed validation, not fresh confirmation",
                      "pooled F1 primary Q3+Q4 does not replace active H1 contract",
                      "QA and replay PASS do not imply official improvement or exact historical answer restoration",
                      "full all-train fits and official materialization remain separately authorized"]}
    t.core.write_json(output / "independent-qa.json", qa)
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    qa = verify(args.output.resolve(), args.seal.resolve())
    print({"status": qa["status"], "check_count": qa["check_count"], "fits": 0})


if __name__ == "__main__":
    main()
