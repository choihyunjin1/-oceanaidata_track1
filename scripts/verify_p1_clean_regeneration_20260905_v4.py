"""Regenerate clean P1 from distributed train, then infer using only new models.

Q4 earlier-inner O/B (2 fits) rederive the policy; full O/B (2 fits) follow.
Existing model/answer contents are never inputs. This is not a new search family.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p1_clean_control_fulltrain_20260905_v1 as canonical
import run_p1_score_repair_20260905_v1 as screen

joblib, np, pd = screen.joblib, screen.np, screen.pd
RUN = "p1_clean_regeneration_20260905_v4"
OUT = ROOT / "artifacts" / RUN
REPORT = ROOT / "reports" / RUN
THREADS = 2
MAX_FITS = 4
WALL_SECONDS = 3600
EXPECTED_ANSWER_SHA = "064ef022faf2a3e8bc7c70633210847aa494060858374aa43f28f4eced84ec43"
POLICIES = ("original", "balanced", "original_balanced_union", "balanced_union")


def write(path, value):
    screen.write_json(path, value)


def policy_bits(frame, probabilities, rules, frozen, calibrations):
    bits = {
        name: screen.decode(frame, probabilities[name], rules, frozen,
                            calibrations[name]["threshold"])
        for name in ("original", "balanced")
    }
    bits["original_balanced_union"] = np.maximum(bits["original"], bits["balanced"])
    bits["balanced_union"] = np.maximum(
        bits["original"], screen.decode(frame, probabilities["balanced"], rules, frozen,
                                         calibrations["balanced_union"]["threshold"])
    )
    return bits


def select_policy(frame, probabilities, rules, frozen):
    calibrations, bits = {}, {}
    for name in ("original", "balanced"):
        calibrations[name], bits[name] = screen.calibrate(
            frame, probabilities[name], rules, frozen
        )
    calibrations["balanced_union"], _ = screen.calibrate(
        frame, probabilities["balanced"], rules, frozen, bits["original"]
    )
    options = policy_bits(frame, probabilities, rules, frozen, calibrations)
    selected = max(POLICIES, key=lambda name: screen.metric(frame.label, options[name])["f1"])
    return selected, calibrations, {
        name: screen.metric(frame.label, value) for name, value in options.items()
    }


def ensure_empty_models(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("03_model must be empty; never delete or reuse existing models")


def source_contract():
    cfg_path = ROOT / "configs/experiments/p1_clean_control_fulltrain_20260905_v1.json"
    cfg = canonical.checked_contract(cfg_path)
    frozen = json.loads((ROOT / cfg["feature_contract"]).read_text(encoding="utf-8"))
    paths = set(canonical.SOURCE_FILES) | set(cfg["source_pins"]) | {
        "scripts/verify_p1_clean_regeneration_20260905_v4.py"
    }
    hashes = {p: screen.sha(ROOT / p) for p in sorted(paths)}
    return cfg, frozen, hashes


def assert_sources(hashes):
    for relative, expected in hashes.items():
        if screen.sha(ROOT / relative) != expected:
            raise ValueError("source changed: " + relative)


def fit_pair(training, evaluation, stage, frozen, receipt, started):
    stats = screen.stats_fit(training)
    bundle = screen.feature_pair(training, stats, frozen)[0]
    encoder = screen.TabularEncoder().fit(bundle, np.arange(len(training)))
    features = encoder.transform(bundle)
    target = training.label.to_numpy(dtype=np.int8)
    eval_features = None
    if evaluation is not None:
        eval_features = encoder.transform(screen.feature_pair(evaluation, stats, frozen)[0])
    base = screen.load_config(ROOT / frozen["base_config"], env={})
    lgb = json.loads((ROOT / frozen["lightgbm_recipe"]).read_text(encoding="utf-8"))
    probabilities = {}
    for name in ("original", "balanced"):
        if len(receipt["fits"]) >= MAX_FITS:
            raise RuntimeError("fit budget exceeded")
        if time.monotonic() - started > WALL_SECONDS:
            raise RuntimeError("wall budget exceeded; no restart")
        begin = time.monotonic()
        if name == "original":
            model = screen._fit_model(
                "xgboost", base.raw["models"]["xgboost"], frozen["seed"],
                THREADS, features, target
            )
        else:
            params = screen._lgb_parameters(lgb, frozen["seed"], multiclass=False)
            params["n_jobs"] = THREADS
            model = screen.lgb.LGBMClassifier(**params)
            model.fit(features, target, sample_weight=screen._event_day_weight(training, target))
        path = OUT / "03_model" / (stage + "_" + name + ".joblib")
        if path.exists():
            raise FileExistsError("new model path already exists")
        joblib.dump({"model": model, "encoder": encoder, "train_stats": stats}, path, compress=3)
        if evaluation is not None:
            probability = model.predict_proba(eval_features)[:, 1]
            if not np.isfinite(probability).all():
                raise ValueError("nonfinite inner probability")
            probabilities[name] = probability
        receipt["fits"].append({
            "stage": stage, "model": name, "model_file": path.name,
            "sha256": screen.sha(path), "training_rows": len(training),
            "features": features.shape[1], "seconds": time.monotonic() - begin,
            "fresh_initialization": True,
            "training_max": pd.to_datetime(training.time, utc=True).max().isoformat(),
        })
        receipt["runtime_seconds"] = time.monotonic() - started
        write(OUT / "progress.json", receipt)
        print(json.dumps({"stage": stage, "completed_fits": len(receipt["fits"]),
                          "runtime_seconds": receipt["runtime_seconds"]}), flush=True)
        del model
        gc.collect()
    return probabilities, stats


def train():
    cfg, frozen, hashes = source_contract()
    ensure_empty_models(OUT / "03_model")
    if (OUT / "05_answer").exists() and any((OUT / "05_answer").iterdir()):
        raise FileExistsError("05_answer must be empty before regeneration")
    started = time.monotonic()
    receipt = {
        "experiment_id": RUN, "status": "RUNNING", "pid": os.getpid(),
        "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat(), "fits": [],
        "empty_03_model_verified_before_training": True, "prior_models_read": 0,
        "prior_answer_values_read": 0, "official_rows": 0, "sample_values_read": 0,
        "hidden_rows": 0, "external_observation_rows": 0, "csv_written": 0, "uploads": 0,
        "score_inversion_used": False, "cuda": False, "threads": THREADS,
        "resource_amendment": "CPU threads 4 to 2; same model and selection recipe",
        "source_hashes": hashes, "max_fits": MAX_FITS, "wall_cap_seconds": WALL_SECONDS,
        "baseline_regeneration": "PENDING",
        "expected_answer_sha_metadata_only": EXPECTED_ANSWER_SHA,
    }
    with (OUT / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle)
    try:
        train_path = Path(os.environ["P1_DATA_DIR"]) / "train.csv"
        if screen.sha(train_path) != frozen["expected_training_sha256"]:
            raise ValueError("training data hash mismatch")
        receipt["train_sha256"] = screen.sha(train_path)
        frame = pd.read_csv(train_path, usecols=screen.RAW + ["label", "anomaly_type"])
        if len(frame) != 776706 or frame.duplicated(screen.KEYS).any():
            raise ValueError("training keys/count invalid")
        if not frame.label.isin([0, 1]).all():
            raise ValueError("training labels invalid")
        frame.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        fold = next(f for f in frozen["folds"] if f["name"] == "2025_q4")
        cutoff = pd.Timestamp(fold["start"]) - pd.Timedelta(days=frozen["purge_days"])
        inner_start = cutoff - pd.Timedelta(days=frozen["inner_days"])
        train_stop = inner_start - pd.Timedelta(days=frozen["purge_days"])
        times = pd.to_datetime(frame.time, utc=True)
        evaluation = frame.loc[(times >= inner_start) & (times < cutoff)].reset_index(drop=True)
        training = screen.train_slice(frame, train_stop)
        if not pd.to_datetime(training.time, utc=True).max() < train_stop:
            raise ValueError("inner training boundary violated")
        receipt["inner_split"] = {"start": inner_start.isoformat(), "end": cutoff.isoformat(),
                                  "training_stop": train_stop.isoformat(),
                                  "rows": len(evaluation), "purge_days": frozen["purge_days"]}
        probabilities, stats = fit_pair(training, evaluation, "inner", frozen, receipt, started)
        selected, calibrations, metrics = select_policy(
            evaluation, probabilities, screen.rule_masks(evaluation, stats), frozen
        )
        receipt["rederived_inner"] = {"selected": selected, "calibrations": calibrations,
                                      "metrics": metrics, "outer_labels_used": 0}
        equivalent = (selected == cfg["selection"] and all(
            calibrations[k]["threshold"] == v["threshold"] for k, v in cfg["calibrations"].items()
        ))
        receipt["inner_policy_equivalent_to_canonical"] = equivalent
        if not equivalent:
            raise ValueError("inner policy rederivation mismatch; no threshold override")
        del training, evaluation, probabilities
        gc.collect()
        fit_pair(frame, None, "full", frozen, receipt, started)
        assert_sources(hashes)
        recipe = {"feature_contract": frozen, "selection": selected,
                  "calibrations": calibrations, "decoder_enabled": False,
                  "model_hashes": {f["model"]: f["sha256"] for f in receipt["fits"]
                                   if f["stage"] == "full"},
                  "source_hashes": hashes, "inner_training_pid": os.getpid(),
                  "official_rows_before_seal": 0}
        write(OUT / "03_model/frozen_recipe.json", recipe)
        receipt["recipe_sha256"] = screen.sha(OUT / "03_model/frozen_recipe.json")
        for relative, expected in hashes.items():
            destination = OUT / "02_code" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
            if screen.sha(destination) != expected:
                raise ValueError("source copy hash mismatch")
        receipt["status"] = "FOUR_FITS_COMPLETE_OFFICIAL_UNREAD"
        receipt["baseline_regeneration"] = "TRAINING_PASS_INFERENCE_PENDING"
    except Exception as exc:
        receipt.update(status="TERMINAL_TECHNICAL_FAILURE", error=str(exc))
        raise
    finally:
        receipt["runtime_seconds"] = time.monotonic() - started
        write(OUT / "train_result.json", receipt)
        write(REPORT / "train-result.json", receipt)


def infer(verify=False):
    started = time.monotonic()
    training = json.loads((OUT / "train_result.json").read_text(encoding="utf-8"))
    if training["status"] != "FOUR_FITS_COMPLETE_OFFICIAL_UNREAD" or len(training["fits"]) != 4:
        raise ValueError("complete four-fit seal required")
    if training["pid"] == os.getpid():
        raise ValueError("inference must run in a separate process")
    assert_sources(training["source_hashes"])
    recipe_path = OUT / "03_model/frozen_recipe.json"
    if screen.sha(recipe_path) != training["recipe_sha256"]:
        raise ValueError("recipe hash mismatch")
    for fit in training["fits"]:
        if screen.sha(OUT / "03_model" / fit["model_file"]) != fit["sha256"]:
            raise ValueError("regenerated model hash mismatch")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    root_data = Path(os.environ["P1_DATA_DIR"])
    evaluation = pd.read_csv(root_data / "test.csv", usecols=screen.RAW)
    keys = pd.read_csv(root_data / "sample_submission.csv", usecols=screen.KEYS)
    if len(evaluation) != 169011 or len(keys) != 169011 or not evaluation[screen.KEYS].equals(keys):
        raise ValueError("official rows/key/order mismatch")
    evaluation.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    evaluation.reset_index(drop=True, inplace=True)
    probabilities, stats, bundle = {}, None, None
    for name in ("original", "balanced"):
        package = joblib.load(OUT / "03_model" / ("full_" + name + ".joblib"))
        if bundle is None:
            stats = package["train_stats"]
            bundle = screen.feature_pair(evaluation, stats, recipe["feature_contract"])[0]
        probabilities[name] = package["model"].predict_proba(
            package["encoder"].transform(bundle)
        )[:, 1]
        if not np.isfinite(probabilities[name]).all():
            raise ValueError("nonfinite regenerated inference")
    bits = policy_bits(evaluation, probabilities, screen.rule_masks(evaluation, stats),
                       recipe["feature_contract"], recipe["calibrations"])[recipe["selection"]]
    answer = canonical.align_answer(evaluation, bits, keys)
    data = answer.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    path = OUT / "05_answer/P1_submission.csv"
    if verify:
        first = json.loads((REPORT / "inference-qa.json").read_text(encoding="utf-8"))
        if first["pid"] == os.getpid() or digest != first["sha256"] or data != path.read_bytes():
            raise ValueError("separate-process regenerated answer mismatch")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
    qa = {"experiment_id": RUN, "status": "FRESH_PROCESS_REPLAY_PASS" if verify
          else "TRAIN_TO_ANSWER_REGENERATION_COMPLETE", "pid": os.getpid(),
          "training_pid": training["pid"], "fresh_process": True,
          "sha256": digest, "rows": len(answer), "positive_rows": int(answer.label.sum()),
          "schema_key_order_binary_finite_unique_pass": True,
          "same_as_canonical_answer_sha": digest == EXPECTED_ANSWER_SHA,
          "prior_answer_values_read": 0, "prior_models_read": 0,
          "new_models_loaded": 2, "official_test_rows": len(evaluation),
          "sample_key_rows": len(keys), "sample_prediction_values_read": 0,
          "hidden_rows": 0, "uploads": 0, "new_fits_in_inference": 0,
          "runtime_seconds": time.monotonic() - started,
          "training_plus_inference_seconds": training["runtime_seconds"] + time.monotonic() - started,
          "full_rebuild_executed": True, "isolated_clean_machine_verified": False,
          "code_snapshot_execution_verified": False, "final_package_ready": False}
    write(REPORT / ("replay-qa.json" if verify else "inference-qa.json"), qa)
    print(json.dumps(qa), flush=True)


def self_test():
    import pytest

    def test_empty_guard():
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "03_model"
            ensure_empty_models(path)
            (path / "sentinel").touch()
            with pytest.raises(FileExistsError):
                ensure_empty_models(path)

    def test_align():
        frame = pd.DataFrame({"station": ["S", "S"], "year": [2025, 2025],
                              "layer": [1, 1], "time": ["b", "a"]})
        keys = frame.iloc[::-1].reset_index(drop=True)
        answer = canonical.align_answer(frame, np.array([1, 0]), keys)
        assert answer.label.tolist() == [0, 1]
        with pytest.raises(ValueError):
            canonical.align_answer(pd.concat([frame, frame]), np.array([1, 0, 1, 0]), keys)

    def test_selection_is_inner_fitted():
        frame = pd.DataFrame({"station": "S", "year": 2025, "layer": 1,
                              "time": pd.date_range("2025-01-01", periods=24, freq="10min"),
                              "label": [0] * 12 + [1] * 12})
        probabilities = {"original": np.r_[np.zeros(12), np.full(12, .7)],
                         "balanced": np.r_[np.zeros(12), np.full(12, .6)]}
        rules = (np.zeros(24, bool), np.zeros(24, bool))
        frozen = {"threshold_grid": [.2, .3], "low_ratio": .5,
                  "close_gap_rows": 0, "minimum_positive_run": 1}
        selected, cals, metrics = select_policy(frame, probabilities, rules, frozen)
        assert selected == "original"
        assert cals["original"]["threshold"] == .3
        assert metrics["original"]["f1"] == 1

    checks = (test_empty_guard, test_align, test_selection_is_inner_fitted)
    for check in checks:
        check()
    write(REPORT / "synthetic-qa.json", {"status": "PASS", "checks": len(checks),
          "runner_sha256": screen.sha(__file__), "training_data_rows": 0,
          "official_rows": 0, "model_fits": 0})
    print("3 synthetic contract checks PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true")
    action.add_argument("--infer", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    elif args.self_test:
        self_test()
    else:
        infer(args.verify)
