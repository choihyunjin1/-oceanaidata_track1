"""New-ID continuation of interrupted zero-fit stress; prior attempt immutable."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

for _thread in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread] = "2"

import numpy as np
import pandas as pd
import run_p1_tuning_depth_stress_20260906_v1 as legacy

ROOT = Path(__file__).resolve().parents[1]
ID = "p1_tuning_depth_stress_20260906_v2"
OUT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID
CONFIG = ROOT / "configs/experiments" / (ID + ".json")
MANIFEST = REPORT / "partial-manifest.json"
source, old, base = legacy.source, legacy.old, legacy.base
REUSED = ("H2_2024", "H1_2025")


def valid_probability(values, rows):
    return (values.ndim == 1 and len(values) == rows and np.isfinite(values).all()
            and (values >= 0).all() and (values <= 1).all())


def preflight():
    legacy.contract()
    lock_path = legacy.OUT / "ATTEMPT_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if old.sha(legacy.__file__) != lock["runner_sha256"] or old.sha(legacy.CONFIG) != lock["config_sha256"]:
        raise ValueError("interrupted source differs from attempt lock")
    if (legacy.OUT / "terminal_result.json").exists():
        raise ValueError("unexpected prior terminal; re-audit required")
    reference = json.loads((source.OUT / "terminal_result.json").read_text(encoding="utf-8"))
    entries = []
    for fold in REUSED:
        for arm in source.ARMS:
            record = next(x for x in reference["fits"] if (x["fold"], x["stage"], x["arm"]) == (fold, "outer", arm))
            if old.sha(ROOT / record["model_path"]) != record["model_sha256"]:
                raise ValueError("model pin changed")
            for condition in legacy.CONDITIONS:
                path = legacy.OUT / f"{fold}_{arm}_{condition}.npy"
                values = np.load(path, allow_pickle=False)
                if not valid_probability(values, record["validation_rows"]):
                    raise ValueError("invalid partial probability")
                if condition == legacy.CONDITIONS[0]:
                    standard = source.OUT / f"{fold}_outer_{arm}.npy"
                    if old.sha(standard) != record["probability_sha256"] or not np.array_equal(values, np.load(standard, allow_pickle=False)):
                        raise ValueError("standard partial probability not exact")
                entries.append({"fold": fold, "arm": arm, "condition": condition,
                    "path": str(path.relative_to(ROOT)), "sha256": old.sha(path), "rows": len(values),
                    "model_sha256": record["model_sha256"], "validation_keys_sha256": record["evaluation_keys_sha256"]})
        path = legacy.OUT / (fold + "_rules.npy")
        rules = np.load(path, allow_pickle=False)
        if rules.shape != (2, record["validation_rows"]) or not np.isin(rules, [0, 1]).all():
            raise ValueError("partial rules invalid")
        entries.append({"fold": fold, "arm": "rules", "path": str(path.relative_to(ROOT)), "sha256": old.sha(path)})
    receipt = {"status": "PARTIAL_ACCEPTED_PENDING_FRESH_REPLAY", "experiment_id": ID,
        "cause": "abrupt interruption; process absent, exit1, no terminal/finally or traceback; exact OS cause unrecorded",
        "not_established": "no evidence of permission failure or 600s timeout; last logged stage275s and files around311s",
        "prior_lock_sha256": old.sha(lock_path), "prior_runner_sha256": lock["runner_sha256"],
        "prior_config_sha256": lock["config_sha256"], "entries": entries,
        "hash_caveat": "Partial per-file hashes were not persisted before interruption; these are adoption-time hashes, to be independently replay-verified.",
        "completed_probability_arrays": 18, "completed_rules_arrays": 2,
        "remaining_folds": ["H2_2025"], "model_fits": 0, "official_rows": 0, "csv_written": 0}
    REPORT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"status": receipt["status"], "manifest_sha256": old.sha(MANIFEST), "arrays": 18}))


def contract():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (cfg["experiment_id"], cfg["threads"], cfg["max_fits"], cfg["wall_cap_seconds"]) != (ID, 2, 0, 900):
        raise ValueError("continuation resource contract changed")
    for path, value in cfg["source_pins"].items():
        if old.sha(ROOT / path) != value:
            raise ValueError("continuation source pin changed: " + path)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest["entries"]:
        if old.sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("adopted partial artifact changed")
    return cfg, manifest, legacy.contract()[1]


def execute():
    cfg, manifest, (source_cfg, prior, frozen, ev) = contract()
    synth = legacy.synthetic_audit()
    if synth["status"] != "PASS":
        raise ValueError("synthetic preflight failed")
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"experiment_id": ID, "status": "RUNNING", "pid": os.getpid(),
        "runner_sha256": old.sha(__file__), "config_sha256": old.sha(CONFIG), "source_pins": cfg["source_pins"],
        "model_fits": 0, "threads": 2, "gpu": False, "official_rows": 0, "hidden_rows": 0,
        "csv_written": 0, "uploads": 0, "rules_refit": False, "model_selection": False,
        "threshold_selection": False, "official_score_estimate": None, "synthetic": synth,
        "feature_audit": [], "checks": {}, "rules_sha256": {}, "reused_probabilities": 0,
        "new_probabilities": 0, "partial_manifest_sha256": old.sha(MANIFEST)}
    with (OUT / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "runner_sha256": result["runner_sha256"], "config_sha256": result["config_sha256"]}, handle)

    def tick(stage):
        elapsed = time.monotonic() - started
        old.write_json(OUT / "progress.json", {"stage": stage, "seconds": elapsed,
            "reused_probabilities": result["reused_probabilities"], "new_probabilities": result["new_probabilities"],
            "pid": os.getpid()})
        print(json.dumps({"stage": stage, "seconds": round(elapsed, 2)}), flush=True)
        if elapsed >= 900:
            raise TimeoutError("15-minute cap; preserve attempt; no automatic retry")

    try:
        reference = json.loads((source.OUT / "terminal_result.json").read_text(encoding="utf-8"))
        sealed = pd.read_parquet(source.OUT / "oof.parquet")
        if old.sha(source.OUT / "oof.parquet") != reference["oof_sha256"]:
            raise ValueError("source OOF hash changed")
        data = base.training_source(prior)
        masks, _ = source.nested_masks(data, ev, source_cfg)
        parts = {condition: [] for condition in legacy.CONDITIONS}
        for fold in source.FOLDS:
            val = data.loc[masks[(fold, "outer")][1]].reset_index(drop=True)
            original = sealed.loc[sealed.fold.eq(fold)].reset_index(drop=True)
            columns = old.KEYS + ["row_id", "label", "anomaly_type"]
            if not val[columns].equals(original[columns]):
                raise ValueError("keys/order/labels mismatch")
            probabilities = {condition: {} for condition in legacy.CONDITIONS}
            rules = None
            for arm in source.ARMS:
                tick(fold + "/" + arm)
                record = next(x for x in reference["fits"] if (x["fold"], x["stage"], x["arm"]) == (fold, "outer", arm))
                model_path = ROOT / record["model_path"]
                if old.sha(model_path) != record["model_sha256"]:
                    raise ValueError("model pin changed")
                package = old.joblib.load(model_path)
                features = base.bundle(val, package["train_stats"], frozen, prior, arm == "bracket")
                arm_rules = old.rule_masks(val, package["train_stats"])
                if rules is not None and any(not np.array_equal(a, b) for a, b in zip(rules, arm_rules, strict=True)):
                    raise ValueError("rules mismatch between frozen arm packages")
                rules = arm_rules
                for condition in legacy.CONDITIONS:
                    altered = legacy.replace_metadata(features, val, package["train_stats"], condition)
                    x = legacy.matrix_guard(package["encoder"], features, altered)
                    if fold in REUSED:
                        adopted = next(item for item in manifest["entries"] if (item["fold"], item["arm"], item.get("condition")) == (fold, arm, condition))
                        path = ROOT / adopted["path"]
                        prob = np.load(path, allow_pickle=False)
                        result["reused_probabilities"] += 1
                    else:
                        prob = source.probability(package["model"], x, arm, 700)
                        path = OUT / f"{fold}_{arm}_{condition}.npy"
                        np.save(path, prob, allow_pickle=False)
                        result["new_probabilities"] += 1
                    if not valid_probability(prob, len(val)):
                        raise ValueError("invalid probability")
                    if condition == legacy.CONDITIONS[0] and not np.array_equal(prob, np.load(source.OUT / f"{fold}_outer_{arm}.npy", allow_pickle=False)):
                        raise ValueError("standard replay not exact")
                    probabilities[condition][arm] = prob
                    result["feature_audit"].append({"fold": fold, "arm": arm, "condition": condition,
                        "rows": len(val), "reused": fold in REUSED, "model_path": record["model_path"],
                        "model_sha256": record["model_sha256"], "probability_path": str(path.relative_to(ROOT)),
                        "probability_sha256": old.sha(path), "nominal_missing": int(altered.frame[legacy.META[0]].isna().sum()),
                        "category_unseen": int((x[:, package["encoder"].feature_columns.index(legacy.META[1])] == -1).sum()),
                        "other_feature_columns_exact": len(features.feature_columns) - 2})
                result["checks"][fold + "/" + arm + "/two_columns_only_standard_exact"] = True
                del package, features, x
                gc.collect()
                old.write_json(OUT / "partial-receipt.json", result)
            if fold in REUSED:
                old_rules = np.load(legacy.OUT / (fold + "_rules.npy"), allow_pickle=False)
                if not np.array_equal(np.stack(rules), old_rules):
                    raise ValueError("adopted rules not exact")
            path = OUT / (fold + "_rules.npy")
            np.save(path, np.stack(rules), allow_pickle=False)
            result["rules_sha256"][fold] = old.sha(path)
            for condition in legacy.CONDITIONS:
                part = original.copy()
                for name, balanced in (("control", "balanced"), ("candidate", "bracket")):
                    chosen, calibration = reference["earlier_inner_selections"][fold][name]
                    part[name] = source.policy.policies(val, {"original": probabilities[condition]["original"],
                        "balanced": probabilities[condition][balanced]}, rules, frozen, calibration)[chosen]
                if condition == legacy.CONDITIONS[0] and not part.equals(original):
                    raise ValueError("standard completed policy not exact")
                parts[condition].append(part)
            result["checks"][fold + "/keys_labels_calendar_rules_policy_fixed"] = True
            tick(fold + "/complete")
        result["conditions"] = {}
        for condition in legacy.CONDITIONS:
            tick(condition + "/metrics")
            oof = pd.concat(parts[condition], ignore_index=True)
            values = source.summarize(oof, ev)
            if condition == legacy.CONDITIONS[0] and any(value != reference[key] for key, value in values.items()):
                raise ValueError("standard metrics mismatch")
            values["decision"] = "DIAGNOSTIC_ONLY_NO_SELECTION"
            path = OUT / (condition + ".parquet")
            oof.to_parquet(path, index=False)
            result["conditions"][condition] = {**values, "oof_sha256": old.sha(path)}
        result.update(status="COMPLETE_FRESH_QA_PENDING", decision="DIAGNOSTIC_ONLY_NO_SELECTION")
    except Exception as exc:
        result.update(status="TERMINAL_TECHNICAL_OR_RESOURCE_FAILURE", error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        result["runtime_seconds"] = time.monotonic() - started
        old.write_json(OUT / "terminal_result.json", result)
        old.write_json(REPORT / "result.json", result)


def qa():
    """Fresh source-to-probability replay also verifies adopted interrupted arrays."""
    _, _, (source_cfg, prior, frozen, ev) = contract()
    result = json.loads((OUT / "terminal_result.json").read_text(encoding="utf-8"))
    if result["status"] != "COMPLETE_FRESH_QA_PENDING" or result["pid"] == os.getpid():
        raise ValueError("completed execution and fresh process required")
    started = time.monotonic()
    data = base.training_source(prior)
    masks, _ = source.nested_masks(data, ev, source_cfg)
    reference = json.loads((source.OUT / "terminal_result.json").read_text(encoding="utf-8"))
    sealed = pd.read_parquet(source.OUT / "oof.parquet")
    checks = {"runner_hash": old.sha(__file__) == result["runner_sha256"],
              "config_hash": old.sha(CONFIG) == result["config_sha256"],
              "source_oof_hash": old.sha(source.OUT / "oof.parquet") == reference["oof_sha256"]}
    for fold in source.FOLDS:
        val = data.loc[masks[(fold, "outer")][1]].reset_index(drop=True)
        probs = {condition: {} for condition in legacy.CONDITIONS}
        for arm in source.ARMS:
            if time.monotonic() - started + result["runtime_seconds"] >= 900:
                raise TimeoutError("continuation plus QA 15-minute bound reached")
            records = [r for r in result["feature_audit"] if r["fold"] == fold and r["arm"] == arm]
            package = old.joblib.load(ROOT / records[0]["model_path"])
            checks[fold + "/" + arm + "/model_hash"] = old.sha(ROOT / records[0]["model_path"]) == records[0]["model_sha256"]
            features = base.bundle(val, package["train_stats"], frozen, prior, arm == "bracket")
            for item in records:
                condition = item["condition"]
                altered = legacy.replace_metadata(features, val, package["train_stats"], condition)
                x = legacy.matrix_guard(package["encoder"], features, altered)
                fresh = source.probability(package["model"], x, arm, 700)
                path = ROOT / item["probability_path"]
                saved = np.load(path, allow_pickle=False)
                checks[f"{fold}/{arm}/{condition}/hash_and_fresh_probability_exact"] = (
                    old.sha(path) == item["probability_sha256"] and np.array_equal(fresh, saved))
                probs[condition][arm] = fresh
            del package, features, x
            gc.collect()
            print(json.dumps({"qa_stage": fold + "/" + arm, "seconds": round(time.monotonic() - started, 2)}), flush=True)
        rules_path = OUT / (fold + "_rules.npy")
        rules = np.load(rules_path, allow_pickle=False)
        checks[fold + "/rules_hash"] = old.sha(rules_path) == result["rules_sha256"][fold]
        for condition in legacy.CONDITIONS:
            oof = pd.read_parquet(OUT / (condition + ".parquet"))
            part = oof.loc[oof.fold.eq(fold)].reset_index(drop=True)
            for name, balanced in (("control", "balanced"), ("candidate", "bracket")):
                chosen, calibration = reference["earlier_inner_selections"][fold][name]
                bits = source.policy.policies(val, {"original": probs[condition]["original"],
                    "balanced": probs[condition][balanced]}, rules, frozen, calibration)[chosen]
                checks[f"{fold}/{condition}/{name}/decoder_exact"] = np.array_equal(bits, part[name])
    for condition in legacy.CONDITIONS:
        path = OUT / (condition + ".parquet")
        oof = pd.read_parquet(path)
        keep = [c for c in sealed if c not in ("control", "candidate")]
        checks[condition + "/keys_labels_calendar_hash"] = (oof[keep].equals(sealed[keep])
            and old.sha(path) == result["conditions"][condition]["oof_sha256"])
        recomputed = source.summarize(oof, ev)
        recomputed["decision"] = "DIAGNOSTIC_ONLY_NO_SELECTION"
        checks[condition + "/metrics_ci_slices"] = all(v == result["conditions"][condition][k] for k, v in recomputed.items())
        if condition == legacy.CONDITIONS[0]:
            checks["standard_original_bits_exact"] = oof.equals(sealed)
    checks = {key: bool(value) for key, value in checks.items()}
    receipt = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "pid": os.getpid(), "execution_pid": result["pid"],
        "runtime_seconds": time.monotonic() - started, "model_fits": 0, "official_rows": 0,
        "csv_written": 0, "uploads": 0, "partial_adoption_hash_caveat_resolved_by_fresh_probability_replay": all(checks.values()),
        "result_sha256": old.sha(OUT / "terminal_result.json")}
    old.write_json(REPORT / "independent-qa.json", receipt)
    print(json.dumps({"status": receipt["status"], "passed": receipt["passed"], "total": receipt["total"]}))
    if receipt["status"] != "PASS":
        raise ValueError("QA failure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.execute:
        execute()
    else:
        qa()
