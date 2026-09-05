"""Zero-fit, two-feature counterfactual stress; never an official score estimate."""

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
import run_p1_tuning_twosided_20260906_v1 as source

ROOT = Path(__file__).resolve().parents[1]
ID = "p1_tuning_depth_stress_20260906_v1"
OUT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID
CONFIG = ROOT / "configs/experiments" / (ID + ".json")
META = ("nominal_depth_m", "depth_regime")
CONDITIONS = ("standard_twosided", "unknown_year", "training_year_median_fallback")
base, old = source.base, source.old


def replace_metadata(bundle, frame, stats, condition):
    """No raw/calendar/key/label change; finite year medians have equal weight."""
    if condition not in CONDITIONS:
        raise ValueError("unregistered stress condition")
    result = bundle.frame.copy()
    if condition != CONDITIONS[0]:
        by_cell = {}
        if condition == CONDITIONS[2]:
            for (station, _year, layer), value in stats["depth"].items():
                if np.isfinite(value):
                    by_cell.setdefault((station, layer), []).append(float(value))
        values = {key: float(np.median(value)) for key, value in by_cell.items()}
        depth = np.array([values.get((s, layer), np.nan)
                          for s, layer in zip(frame.station, frame.layer, strict=True)])
        nominal = np.round(depth / 2.0) * 2.0
        result[META[0]] = nominal.astype(np.float32)
        result[META[1]] = pd.Series([
            f"{station}|d{d:06.1f}" if np.isfinite(d) else f"{station}|unknown|l{layer}"
            for station, layer, d in zip(frame.station, frame.layer, nominal, strict=True)
        ], dtype="string")
    untouched = [name for name in bundle.feature_columns if name not in META]
    if not result[untouched].equals(bundle.frame[untouched]):
        raise ValueError("a nonmetadata feature changed")
    return old.FeatureBundle(result, bundle.feature_columns, bundle.categorical_columns)


def matrix_guard(encoder, original, altered):
    a, b = encoder.transform(original), encoder.transform(altered)
    keep = [i for i, name in enumerate(encoder.feature_columns) if name not in META]
    if not np.array_equal(a[:, keep], b[:, keep], equal_nan=True):
        raise ValueError("encoded nonmetadata feature changed")
    return b


def toy(start="2025-01-01", n=400):
    dates = pd.date_range(start, periods=n, freq="10min", tz="Asia/Seoul")
    t = np.arange(n)
    return pd.DataFrame({"station": "S-ORS", "year": dates.year, "layer": 1,
                         "time": dates.astype(str), "temp": 10 + np.sin(t / 20),
                         "psal": 32 + np.cos(t / 20) / 10, "depth": 12.0,
                         "label": (t % 150 < 20).astype(np.int8), "anomaly_type": "offset"})


def synthetic_audit():
    _, prior, frozen, _ = source.contract()
    train, future = toy(), toy("2026-01-01")
    stats = old.stats_fit(train)
    known = base.bundle(train, stats, frozen, prior)
    unknown = base.bundle(future, stats, frozen, prior)
    encoder = old.TabularEncoder().fit(known, np.arange(len(train)))
    x = encoder.transform(unknown)
    untouched = [name for name in known.feature_columns if name not in META]
    ni, ci = [encoder.feature_columns.index(name) for name in META]
    corrected = source.policy.features(future, stats, frozen, current_depth=True)
    shifted = future.copy()
    shifted["depth"] = 14.0
    shifted_current = source.policy.features(shifted, stats, frozen, current_depth=True)
    missing = future.copy()
    missing["depth"] = np.nan
    actual_missing = base.bundle(missing, stats, frozen, prior)
    forced = replace_metadata(known, train, stats, CONDITIONS[1])
    fallback = replace_metadata(known, train, stats, CONDITIONS[2])
    matrix_guard(encoder, known, forced)
    matrix_guard(encoder, known, fallback)
    checks = {
        "year_is_stats_key": set(stats["depth"]) == {("S-ORS", 2025, 1)},
        "known_nominal_finite_12": bool(known.frame.nominal_depth_m.eq(12).all()),
        "unknown_nominal_nan_not_zero_or_median": bool(np.isnan(x[:, ni]).all()),
        "unseen_category_encoded_minus_one": bool((x[:, ci] == -1).all()),
        "raw_depth_survives_lookup_miss": bool(unknown.frame.depth_raw.eq(12).all()),
        "raw_missing_flag_not_lookup_missing": bool(unknown.frame.depth_missing.eq(0).all()),
        "other_78_features_exact_on_january_year_shift": known.frame[untouched].equals(unknown.frame[untouched]),
        "current_depth_policy_year_invariant_when_finite": corrected.frame.equals(known.frame),
        "current_depth_changes_physical_value_not_just_key": bool(shifted_current.frame.nominal_depth_m.eq(14).all()),
        "actual_raw_missing_remains_nan": bool(actual_missing.frame.depth_raw.isna().all()),
        "actual_raw_missing_flag_one": bool(actual_missing.frame.depth_missing.eq(1).all()),
        "forced_unknown_matches_lookup_miss": forced.frame.equals(unknown.frame),
        "single_supported_year_fallback_restores_metadata": fallback.frame.equals(known.frame),
        "matrix_nonmetadata_exact": True,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "passed": sum(checks.values()), "total": len(checks), "model_fits": 0,
            "source_observation_rows": 0, "official_rows": 0, "hidden_rows": 0,
            "csv_written": 0, "uploads": 0,
            "scope": "synthetic feature/encoder behavior only; no score or causal performance claim"}


def contract():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (cfg["experiment_id"], cfg["threads"], cfg["max_fits"], cfg["wall_cap_seconds"]) != (ID, 2, 0, 600):
        raise ValueError("resource contract changed")
    if cfg["conditions"] != list(CONDITIONS) or cfg["changed_columns"] != list(META):
        raise ValueError("condition bank changed")
    for path, sha in cfg["source_pins"].items():
        if old.sha(ROOT / path) != sha:
            raise ValueError("source changed: " + path)
    receipt = json.loads((source.REPORT / "independent-qa.json").read_text(encoding="utf-8"))
    if receipt["status"] != "PASS" or receipt["passed"] != receipt["total"]:
        raise ValueError("source replay has not passed")
    return cfg, source.contract()


def execute():
    cfg, (source_cfg, prior, frozen, ev) = contract()
    synth = synthetic_audit()
    if synth["status"] != "PASS":
        raise ValueError("synthetic audit failure")
    OUT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {"experiment_id": ID, "status": "RUNNING", "pid": os.getpid(),
              "runner_sha256": old.sha(__file__), "config_sha256": old.sha(CONFIG),
              "source_pins": cfg["source_pins"], "model_fits": 0, "threads": 2, "gpu": False,
              "official_rows": 0, "hidden_rows": 0, "csv_written": 0, "uploads": 0,
              "model_selection": False, "threshold_selection": False, "rules_refit": False,
              "official_score_estimate": None, "synthetic": synth, "checks": {}, "feature_audit": [],
              "rules_sha256": {}}
    with (OUT / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "runner_sha256": result["runner_sha256"],
                   "config_sha256": result["config_sha256"]}, handle)

    def tick(stage):
        elapsed = time.monotonic() - started
        print(json.dumps({"stage": stage, "seconds": round(elapsed, 2)}), flush=True)
        if elapsed >= cfg["wall_cap_seconds"]:
            raise TimeoutError("10-minute cap; preserve partial result, no automatic retry")

    try:
        reference = json.loads((source.OUT / "terminal_result.json").read_text(encoding="utf-8"))
        sealed = pd.read_parquet(source.OUT / "oof.parquet")
        if old.sha(source.OUT / "oof.parquet") != reference["oof_sha256"]:
            raise ValueError("OOF hash mismatch")
        data = base.training_source(prior)
        masks, _ = source.nested_masks(data, ev, source_cfg)
        parts = {condition: [] for condition in CONDITIONS}
        for fold in source.FOLDS:
            val = data.loc[masks[(fold, "outer")][1]].reset_index(drop=True)
            original_part = sealed.loc[sealed.fold.eq(fold)].reset_index(drop=True)
            columns = old.KEYS + ["row_id", "label", "anomaly_type"]
            if not val[columns].equals(original_part[columns]):
                raise ValueError("original keys/order/labels mismatch")
            predictions = {condition: {} for condition in CONDITIONS}
            rules = None
            for arm in source.ARMS:
                tick(f"{fold}/{arm}/start")
                record = next(x for x in reference["fits"] if (x["fold"], x["stage"], x["arm"]) == (fold, "outer", arm))
                path = ROOT / record["model_path"]
                if old.sha(path) != record["model_sha256"]:
                    raise ValueError("model hash mismatch")
                package = old.joblib.load(path)
                features = base.bundle(val, package["train_stats"], frozen, prior, arm == "bracket")
                arm_rules = old.rule_masks(val, package["train_stats"])
                if rules is not None and any(not np.array_equal(a, b) for a, b in zip(rules, arm_rules, strict=True)):
                    raise ValueError("rules differ across same fold model packages")
                rules = arm_rules
                for condition in CONDITIONS:
                    altered = replace_metadata(features, val, package["train_stats"], condition)
                    x = matrix_guard(package["encoder"], features, altered)
                    prob = source.probability(package["model"], x, arm, 700)
                    predictions[condition][arm] = prob
                    name = f"{fold}_{arm}_{condition}.npy"
                    np.save(OUT / name, prob, allow_pickle=False)
                    encoded_category = x[:, package["encoder"].feature_columns.index(META[1])]
                    result["feature_audit"].append({"fold": fold, "arm": arm, "condition": condition,
                        "rows": len(val), "nominal_missing": int(altered.frame[META[0]].isna().sum()),
                        "category_unseen": int((encoded_category == -1).sum()),
                        "other_feature_columns_exact": len(features.feature_columns) - 2,
                        "model_sha256": old.sha(path), "probability_file": name,
                        "probability_sha256": old.sha(OUT / name)})
                    if condition == CONDITIONS[0]:
                        saved = np.load(source.OUT / f"{fold}_outer_{arm}.npy", allow_pickle=False)
                        if not np.array_equal(prob, saved):
                            raise ValueError("standard probability replay not exact")
                result["checks"][f"{fold}_{arm}_same_model_standard_probability_and_only_two_columns"] = True
                del package, features, x
                gc.collect()
            np.save(OUT / (fold + "_rules.npy"), np.stack(rules), allow_pickle=False)
            result["rules_sha256"][fold] = old.sha(OUT / (fold + "_rules.npy"))
            for condition in CONDITIONS:
                part = original_part.copy()
                for name, balanced in (("control", "balanced"), ("candidate", "bracket")):
                    chosen, calibration = reference["earlier_inner_selections"][fold][name]
                    part[name] = source.policy.policies(val, {"original": predictions[condition]["original"],
                        "balanced": predictions[condition][balanced]}, rules, frozen, calibration)[chosen]
                if condition == CONDITIONS[0] and not part.equals(original_part):
                    raise ValueError("standard completed policy replay not exact")
                parts[condition].append(part)
            result["checks"][fold + "_keys_labels_calendar_rules_policy_fixed"] = True
            tick(fold + "/complete")
        result["conditions"] = {}
        for condition in CONDITIONS:
            tick(condition + "/metrics")
            oof = pd.concat(parts[condition], ignore_index=True)
            metrics = source.summarize(oof, ev)
            if condition == CONDITIONS[0] and any(value != reference[key] for key, value in metrics.items()):
                raise ValueError("standard metrics/CI/slices mismatch")
            metrics["decision"] = "DIAGNOSTIC_ONLY_NO_SELECTION"
            oof.to_parquet(OUT / (condition + ".parquet"), index=False)
            result["conditions"][condition] = {**metrics, "oof_sha256": old.sha(OUT / (condition + ".parquet"))}
        result["checks"]["standard_metrics_ci_slices_exact"] = True
        result["status"] = "COMPLETE_DIAGNOSTIC_ONLY"
        result["limitations"] = [
            "Already-exposed retrospective OOF, not new/future/official evidence.",
            "Frozen models were trained with year-specific statistics, not with the fallback policy.",
            "Only two metadata columns change; raw depth and actual missingness remain.",
            "Unknown-year is forced for all rows; real deployment could differ in more than two columns.",
            "No fitted-policy, threshold, iteration or fallback rule is selected from these outcomes."]
    except Exception as exc:
        result.update(status="TERMINAL_TECHNICAL_OR_RESOURCE_FAILURE", error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        result["runtime_seconds"] = time.monotonic() - started
        old.write_json(OUT / "terminal_result.json", result)
        old.write_json(REPORT / "result.json", result)


def qa():
    """Separate-process artifact arithmetic/policy QA, not full model re-inference."""
    _, (_, _, frozen, ev) = contract()
    result = json.loads((OUT / "terminal_result.json").read_text(encoding="utf-8"))
    if result["status"] != "COMPLETE_DIAGNOSTIC_ONLY" or result["pid"] == os.getpid():
        raise ValueError("completed result and fresh process required")
    reference = json.loads((source.OUT / "terminal_result.json").read_text(encoding="utf-8"))
    sealed = pd.read_parquet(source.OUT / "oof.parquet")
    checks = {"source_oof_hash": old.sha(source.OUT / "oof.parquet") == reference["oof_sha256"],
              "runner_hash": old.sha(__file__) == result["runner_sha256"],
              "config_hash": old.sha(CONFIG) == result["config_sha256"]}
    probabilities = {}
    for item in result["feature_audit"]:
        key = (item["fold"], item["arm"], item["condition"])
        path = OUT / item["probability_file"]
        probabilities[key] = np.load(path, allow_pickle=False)
        checks["/".join(key) + "/prob_hash_finite"] = (
            old.sha(path) == item["probability_sha256"] and np.isfinite(probabilities[key]).all()
            and len(probabilities[key]) == item["rows"] and np.isin(probabilities[key] >= 0, [True]).all()
            and bool((probabilities[key] <= 1).all()))
    for condition in CONDITIONS:
        path = OUT / (condition + ".parquet")
        frame = pd.read_parquet(path)
        keep = [c for c in sealed.columns if c not in ("control", "candidate")]
        checks[condition + "/oof_hash_keys_labels"] = (
            old.sha(path) == result["conditions"][condition]["oof_sha256"]
            and frame[keep].equals(sealed[keep]) and not frame.duplicated(old.KEYS).any())
        for fold in source.FOLDS:
            part = frame.loc[frame.fold.eq(fold)].reset_index(drop=True)
            rules_path = OUT / (fold + "_rules.npy")
            rules = np.load(rules_path, allow_pickle=False)
            checks[fold + "/rules_hash"] = old.sha(rules_path) == result["rules_sha256"][fold]
            for name, balanced in (("control", "balanced"), ("candidate", "bracket")):
                chosen, calibration = reference["earlier_inner_selections"][fold][name]
                bits = source.policy.policies(part, {
                    "original": probabilities[(fold, "original", condition)],
                    "balanced": probabilities[(fold, balanced, condition)]}, rules, frozen, calibration)[chosen]
                checks[f"{condition}/{fold}/{name}/decoder_exact"] = np.array_equal(bits, part[name])
        computed = source.summarize(frame, ev)
        computed["decision"] = "DIAGNOSTIC_ONLY_NO_SELECTION"
        checks[condition + "/metrics_ci_slices_exact"] = all(
            value == result["conditions"][condition][key] for key, value in computed.items())
        if condition == CONDITIONS[0]:
            checks["standard_original_bits_exact"] = frame.equals(sealed)
    checks = {key: bool(value) for key, value in checks.items()}
    receipt = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "passed": sum(checks.values()), "total": len(checks), "pid": os.getpid(),
               "execution_pid": result["pid"], "model_fits": 0, "official_rows": 0,
               "csv_written": 0, "uploads": 0, "result_sha256": old.sha(OUT / "terminal_result.json"),
               "scope": "fresh-process saved-probability/decoder/OOF arithmetic QA; not repeated model inference"}
    old.write_json(REPORT / "independent-qa.json", receipt)
    print(json.dumps({"status": receipt["status"], "passed": receipt["passed"], "total": receipt["total"]}))
    if receipt["status"] != "PASS":
        raise ValueError("QA failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute()
    elif args.qa:
        qa()
    elif args.synthetic:
        receipt = synthetic_audit()
        old.write_json(REPORT / "synthetic-qa.json", receipt)
        print(json.dumps(receipt))
        if receipt["status"] != "PASS":
            raise ValueError("synthetic audit failed")
