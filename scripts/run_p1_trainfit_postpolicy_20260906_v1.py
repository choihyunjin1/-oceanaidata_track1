"""Zero-backbone-fit, train-only independent range/cell policy comparison."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_var] = "2"

import numpy as np
import pandas as pd
import run_p1_tuning_twosided_20260906_v1 as prior

ROOT = Path(__file__).resolve().parents[1]
ID = "p1_trainfit_postpolicy_20260906_v1"
CONFIG = ROOT / "configs/experiments" / (ID + ".json")
OUT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID
ORDER = ("B", "O", "AND", "OR")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def fit_bounds(frame):
    values = frame.loc[frame.label.eq(0), "temp"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no finite normal training support")
    return {"minimum": float(values.min()), "maximum": float(values.max()),
            "normal_training_rows": len(values)}


def range_bits(frame, bounds):
    x = frame.temp.to_numpy(dtype=float)
    return (np.isfinite(x) & ((x < bounds["minimum"]) | (x > bounds["maximum"]))).astype(np.int8)


def arm_bits(options):
    original, balanced = options["original"], options["balanced"]
    return {"O": original, "B": balanced, "AND": np.minimum(original, balanced),
            "OR": np.maximum(original, balanced)}


def cell_keys(frame):
    return frame.station.astype(str) + "/" + frame.layer.astype(int).astype(str)


def fit_cells(inner, options, global_name):
    choices = arm_bits(options)
    keys = cell_keys(inner)
    cells = {}
    for key in sorted(keys.unique()):
        mask = keys.eq(key).to_numpy()
        y = inner.label.to_numpy()[mask]
        if len(np.unique(y)) < 2:
            cells[key] = {"choice": "GLOBAL", "rows": int(mask.sum()),
                          "reason": "single_class_inner_support"}
            continue
        metrics = {name: prior.old.metric(y, choices[name][mask]) for name in ORDER}
        best = max(ORDER, key=lambda name: metrics[name]["f1"])
        cells[key] = {"choice": best, "rows": int(mask.sum()), "metrics": metrics}
    return {"global_name": global_name, "cells": cells}


def apply_cells(frame, options, policy):
    choices = arm_bits(options)
    keys = cell_keys(frame)
    out = np.array(options[policy["global_name"]], dtype=np.int8, copy=True)
    for key, item in policy["cells"].items():
        if item["choice"] != "GLOBAL":
            mask = keys.eq(key).to_numpy()
            out[mask] = choices[item["choice"]][mask]
    return out


def verify_sources(cfg):
    checks = {}
    for path, expected in cfg["source_pins"].items():
        checks[path] = sha(ROOT / path) == expected
    source = json.loads((prior.OUT / "terminal_result.json").read_text())
    qa = json.loads((prior.REPORT / "independent-qa.json").read_text())
    checks["source_qa_pass"] = qa["status"] == "PASS" and qa["passed"] == qa["total"]
    checks["source_qa_result_link"] = qa["result_sha256"] == sha(prior.OUT / "terminal_result.json")
    for fit in source["fits"]:
        stem = "_".join(fit[k] for k in ("fold", "stage", "arm"))
        checks[stem + "_model"] = sha(ROOT / fit["model_path"]) == fit["model_sha256"]
        checks[stem + "_probability"] = sha(prior.OUT / (stem + ".npy")) == fit["probability_sha256"]
    if not all(checks.values()):
        raise ValueError("source provenance check failed")
    return checks


def compute(cfg):
    started = time.monotonic()
    checks = verify_sources(cfg)
    source_cfg, base_cfg, frozen, evaluation = prior.contract()
    data = prior.base.training_source(base_cfg)
    masks, support = prior.nested_masks(data, evaluation, source_cfg)
    reference, reference_choices = prior.build_oof(data, masks, frozen)
    stored = pd.read_parquet(prior.OUT / "oof.parquet")
    if not reference.equals(stored):
        raise ValueError("reference keys/predictions did not reproduce")
    parts, fitted = [], {}
    for fold in prior.FOLDS:
        itr, iva = masks[(fold, "inner")]
        otr, ova = masks[(fold, "outer")]
        inner = data.loc[iva].reset_index(drop=True)
        outer = data.loc[ova].reset_index(drop=True)
        chosen, calibration = reference_choices[fold]["candidate"]
        options = {}
        for stage, frame, trainmask in (("inner", inner, itr), ("outer", outer, otr)):
            probs = {"original": np.load(prior.OUT / f"{fold}_{stage}_original.npy", allow_pickle=False),
                     "balanced": np.load(prior.OUT / f"{fold}_{stage}_bracket.npy", allow_pickle=False)}
            rules = prior.old.rule_masks(frame, prior.old.stats_fit(data.loc[trainmask].reset_index(drop=True)))
            options[stage] = prior.policy.policies(frame, probs, rules, frozen, calibration)
        cells = fit_cells(inner, options["inner"], chosen)
        bounds = fit_bounds(data.loc[otr])
        baseline = options["outer"][chosen]
        ref = reference.loc[reference.fold.eq(fold)].reset_index(drop=True)
        if not np.array_equal(baseline, ref.candidate) or not prior.base.key_digest(outer) == prior.base.key_digest(ref):
            raise ValueError("bracket baseline alignment mismatch")
        part = ref.drop(columns=["control", "candidate"]).copy()
        rule = range_bits(outer, bounds)
        cell = apply_cells(outer, options["outer"], cells)
        part["control"] = baseline
        part["range"] = np.maximum(baseline, rule)
        part["cell"] = cell
        part["combined"] = np.maximum(cell, rule)
        fitted[fold] = {"bounds": bounds, "cell_policy": cells,
                        "training_keys_sha256": prior.base.key_digest(data.loc[otr]),
                        "selection_keys_sha256": prior.base.key_digest(inner),
                        "outer_keys_sha256": prior.base.key_digest(outer),
                        "range_outer_trigger_rows": int(rule.sum()),
                        "range_added_tp": int(((rule == 1) & (baseline == 0) & outer.label.eq(1)).sum()),
                        "range_added_fp": int(((rule == 1) & (baseline == 0) & outer.label.eq(0)).sum())}
        parts.append(part)
        if time.monotonic() - started > cfg["wall_cap_seconds"]:
            raise TimeoutError("fixed no-fit runtime cap")
        print(json.dumps({"fold_complete": fold, "new_backbone_fits": 0}), flush=True)
    oof = pd.concat(parts, ignore_index=True)
    results = {}
    for name in ("range", "cell", "combined"):
        probe = oof.drop(columns=[n for n in ("range", "cell", "combined") if n != name]).rename(columns={name: "candidate"})
        results[name] = prior.summarize(probe, evaluation)
    both = all(results[n]["decision"] == "MEAN_GAIN_CANDIDATE_RETAINED" for n in ("range", "cell"))
    results["combined"]["eligible_after_separate_checks"] = both
    return oof, {"comparisons": results, "fitted": fitted, "support": support,
                 "source_checks": checks, "source_checks_passed": sum(checks.values()),
                 "new_backbone_fits": 0, "range_statistic_fits": 3, "cell_policy_fits": 3,
                 "baseline_reproduced_exact": True, "normal_rows_removed": 0,
                 "normal_weights_changed": False, "official_rows": 0, "hidden_rows": 0,
                 "csv_written": 0, "uploads": 0, "runtime_seconds": time.monotonic() - started}


def load_contract():
    cfg = json.loads(CONFIG.read_text())
    if cfg["experiment_id"] != ID or cfg["new_backbone_fits"] != 0 or cfg["gpu"]:
        raise ValueError("resource contract changed")
    if cfg["cell_order"] != list(ORDER) or any(cfg[k] for k in ("official_input_access", "csv_written", "upload")):
        raise ValueError("scope or policy changed")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    cfg = load_contract()
    if args.qa:
        result = json.loads((REPORT / "result.json").read_text())
        assert result["status"] == "COMPLETE" and result["pid"] != os.getpid()
        assert result["runner_sha256"] == sha(__file__) and result["config_sha256"] == sha(CONFIG)
        oof, replay = compute(cfg)
        saved = pd.read_parquet(OUT / "oof.parquet")
        checks = {"oof_exact": oof.equals(saved), "oof_hash": sha(OUT / "oof.parquet") == result["oof_sha256"]}
        for name in ("comparisons", "fitted", "support", "source_checks"):
            checks[name + "_exact"] = replay[name] == result[name]
        assert all(checks.values()), checks
        write_new(REPORT / "replay-qa.json", {"status": "PASS", "checks": checks,
                  "passed": len(checks), "pid": os.getpid(), "new_backbone_fits": 0,
                  "result_sha256": sha(REPORT / "result.json"),
                  "note": "same-code fresh-process replay; root arithmetic audit is separate"})
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        write_new(OUT / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "config_sha256": sha(CONFIG), "runner_sha256": sha(__file__)})
        try:
            oof, result = compute(cfg)
            oof.to_parquet(OUT / "oof.parquet", index=False)
            result.update(status="COMPLETE", experiment_id=ID, pid=os.getpid(),
                          config_sha256=sha(CONFIG), runner_sha256=sha(__file__), oof_sha256=sha(OUT / "oof.parquet"),
                          caveats=["Retrospective repeated historical evaluation, not fresh confirmation.",
                                   "Training-normal range FP0 is tautological; outer/official FP0 is not guaranteed.",
                                   "Cell policies fit on globally calibrated inner predictions; risk reported on outer.",
                                   "No official scores or official observations enter this computation."])
            write_new(REPORT / "result.json", result)
        except Exception as exc:
            write_new(REPORT / "terminal-failure.json", {"status": "TECHNICAL_FAILURE", "type": type(exc).__name__, "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
