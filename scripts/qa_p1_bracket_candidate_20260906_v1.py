"""Independent train-only review of a sealed bracket candidate; zero fits/official I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    started = time.monotonic()
    script_sha = sha(Path(__file__))
    os.environ["P1_DATA_DIR"] = str(args.data.resolve())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    sys.path.insert(0, str(package / "02_code"))
    import run as candidate

    np, pd, screen = candidate.np, candidate.pd, candidate.screen
    candidate.install_boundary_guard("model-qa")
    checks = []

    def check(name, condition):
        checks.append({"check": name, "pass": bool(condition)})
        if not condition:
            raise ValueError("independent QA failed: " + name)

    train_path = package / "train_result.json"
    recipe_path = package / "03_model/frozen_recipe.json"
    replay_path = package / "06_docs/model-replay-qa.json"
    training = json.loads(train_path.read_text(encoding="utf-8"))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    cfg, frozen, source_hashes = candidate.source_contract()
    check("source_snapshot", source_hashes == training["source_hashes"] == recipe["source_hashes"])
    check("four_fits_terminal", training["status"] == "FOUR_FITS_COMPLETE_OFFICIAL_UNREAD"
          and len(training["fits"]) == 4)
    check("recipe_hash", sha(recipe_path) == training["recipe_sha256"])
    check("sealed_candidate_contract", recipe["candidate_contract"] == cfg)
    check("resource_contract", training["threads"] == cfg["threads"] == 4
          and not training["gpu"] and cfg["trees"] == 700 and cfg["max_fits"] == 4)
    check("fit_order", [f["stage"] + "_" + f["model"] for f in training["fits"]]
          == ["inner_balanced", "inner_original", "full_original", "full_balanced"])
    check("fresh_model_qa", replay["status"] == "PASS" and replay["fresh_process"]
          and len({training["pid"], replay["pid"], os.getpid()}) == 3)
    check("model_qa_hash_links", replay["train_result_sha256"] == sha(train_path)
          and replay["recipe_sha256"] == sha(recipe_path))
    check("model_qa_four_exact", len(replay["model_checks"]) == 4
          and all(c["exact_probability_replay"] and c["training_stats_recomputed"]
                  for c in replay["model_checks"]))
    check("no_official_before_review", all(training[k] == 0 for k in (
        "prior_models_read", "prior_answer_values_read", "official_rows", "sample_values_read",
        "hidden_rows", "external_observation_rows", "csv_written", "uploads"))
        and not training["score_inversion_used"] and not any((package / "05_answer").iterdir()))
    check("recipe_official_zero", recipe["official_rows_before_seal"] == 0
          and not recipe["extra_decoder_enabled"])
    path = args.data / "train.csv"
    check("distributed_train_hash", sha(path) == training["train_sha256"]
          == frozen["expected_training_sha256"])
    frame = pd.read_csv(path, usecols=screen.RAW + ["label", "anomaly_type"])
    frame = frame.sort_values(["station", "layer", "time"], kind="stable").reset_index(drop=True)
    check("train_count_keys_labels", len(frame) == 776706 and not frame.duplicated(screen.KEYS).any()
          and frame.label.isin([0, 1]).all())
    start, end, stop = (pd.Timestamp(v) for v in (
        "2025-07-12T00:00:00+09:00", "2025-09-10T00:00:00+09:00", "2025-06-21T00:00:00+09:00"))
    times = pd.to_datetime(frame.time, utc=True)
    evaluation = frame.loc[(times >= start) & (times < end)].reset_index(drop=True)
    inner_train = screen.train_slice(frame, stop)
    check("fixed_dates", tuple(pd.Timestamp(cfg[k]) for k in
          ("inner_start", "inner_end", "training_stop")) == (start, end, stop))
    check("population", len(evaluation) == 123372 and len(inner_train) == 467282
          and int(evaluation.label.sum()) == 4445)
    check("purge", start - stop == pd.Timedelta(days=21)
          and pd.to_datetime(inner_train.time, utc=True).max() < stop)
    check("evaluation_key_digest", candidate.key_sha(evaluation)
          == training["inner_split"]["evaluation_keys_sha256"])
    check("evaluation_target_digest", candidate.array_sha(evaluation.label.to_numpy(dtype=np.int8))
          == training["inner_split"]["evaluation_target_sha256"])
    probabilities = {}
    for fit in training["fits"]:
        stage, name = fit["stage"], fit["model"]
        prefix = stage + "_" + name
        check(prefix + "_model_hash", sha(package / "03_model" / fit["model_file"]) == fit["sha256"])
        check(prefix + "_features", fit["features"] == (80 if name == "original" else 107))
        model_package = screen.joblib.load(package / "03_model" / fit["model_file"])
        model = model_package["model"]
        backend = model.model if name == "original" else model
        parameters = backend.get_params()
        actual_trees = (backend.get_booster().num_boosted_rounds() if name == "original"
                        else backend.booster_.num_trees())
        check(prefix + "_actual_fitted_resource", actual_trees == 700
              and parameters["n_estimators"] == 700 and parameters["n_jobs"] == 4
              and model.n_features_in_ == fit["features"])
        del model_package, model, backend
        subset = inner_train if stage == "inner" else frame
        check(prefix + "_fit_population", fit["training_rows"] == len(subset)
              and fit["training_keys_sha256"] == candidate.key_sha(subset))
        if name == "original":
            check(prefix + "_unchanged_baseline_sha", fit["sha256"]
                  == cfg["baseline_original_model_sha_metadata_only"][stage])
        probe_path = package / "04_logs" / fit["probability_file"]
        check(prefix + "_probability_file_hash", sha(probe_path) == fit["probability_sha256"])
        probability = np.load(probe_path, allow_pickle=False)
        check(prefix + "_probabilities", probability.shape == (fit["probe_rows"],)
              and np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
              and candidate.array_sha(probability) == fit["probability_values_sha256"])
        if stage == "inner":
            probabilities[name] = probability
    stats = screen.stats_fit(inner_train)
    rules = screen.rule_masks(evaluation, stats)
    truth = evaluation.label.to_numpy(dtype=bool)

    def metric(bits):
        bits = np.asarray(bits, dtype=bool)
        tp, fp, fn = int((truth & bits).sum()), int((~truth & bits).sum()), int((truth & ~bits).sum())
        return {"rows": len(truth), "tp": tp, "fp": fp, "fn": fn,
                "f1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn)}

    # Independently reconstruct the grid search, tie-break, union and F1 arithmetic.
    bits, calibrations = {}, {}
    for name in ("original", "balanced", "balanced_union"):
        arm = "balanced" if name == "balanced_union" else name
        choices = []
        for threshold in frozen["threshold_grid"]:
            value = screen.decode(evaluation, probabilities[arm], rules, frozen, threshold)
            if name == "balanced_union":
                value = np.maximum(bits["original"], value)
            choices.append((metric(value)["f1"], threshold, value))
        score, threshold, bits[name] = max(choices, key=lambda c: (c[0], c[1]))
        calibrations[name] = {"threshold": threshold, "inner_f1": score}
    bits["original_balanced_union"] = np.maximum(bits["original"], bits["balanced"])
    metrics = {name: metric(bits[name]) for name in candidate.POLICIES}
    selected = max(candidate.POLICIES, key=lambda name: metrics[name]["f1"])
    check("independent_calibration", calibrations == recipe["calibrations"]
          == training["rederived_inner"]["calibrations"])
    check("independent_selection", selected == recipe["selection"] == training["rederived_inner"]["selected"])
    check("independent_confusion_metrics", metrics == training["rederived_inner"]["metrics"])
    check("workflow_budget", time.time() - training["started_unix"] < 3600)
    check("answer_still_absent", not any((package / "05_answer").iterdir()))
    receipt = {"status": "PASS", "pid": os.getpid(), "script_sha256": script_sha,
               "train_result_sha256": sha(train_path), "recipe_sha256": sha(recipe_path),
               "model_replay_qa_sha256": sha(replay_path), "checks": checks,
               "passed": len(checks), "failed": 0, "independent_inner_metrics": metrics,
               "selected": selected, "calibrations": calibrations,
               "official_rows": 0, "hidden_rows": 0, "csv_written": 0, "uploads": 0, "new_fits": 0,
               "runtime_seconds": time.monotonic() - started,
               "scope": "Independent train population and saved-probability selection arithmetic; linked fresh model replay.",
               "quality_limit": "Final-inner was used for selection; not fresh holdout or official score."}
    destination = package / "06_docs/independent-training-qa.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
    print(json.dumps({"status": "PASS", "checks": len(checks), "runtime_seconds": receipt["runtime_seconds"]}))


if __name__ == "__main__":
    main()
