"""Year-safe P1 depth contract: 12 historical + 2 final-inner + 2 full fits.

All observations/labels come from distributed train.csv. Official inputs and CSV
materialization are deliberately absent. Old code/config/locks remain immutable.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import run_p1_score_repair_20260905_v1 as old  # noqa: E402

RUN = "p1_depth_contract_repair_20260905_v2"
MODELS = ("original", "balanced")
POLICIES = ("original", "balanced", "original_balanced_union", "balanced_union")


def features(frame, statistics, frozen, *, current_depth):
    """Only two columns change; offline temporal/peer/rule behavior is unchanged."""
    frame = frame.reset_index(drop=True)
    bundle = old.build_features(
        frame[old.RAW], config=old.load_config(ROOT / frozen["base_config"], env={})
    )
    output = bundle.frame
    if current_depth:
        depth = pd.to_numeric(frame.depth, errors="raise").to_numpy(dtype=float)
    else:
        depth = np.asarray(
            [
                statistics["depth"].get(k, np.nan)
                for k in zip(frame.station, frame.year, frame.layer, strict=True)
            ]
        )
    nominal = np.round(depth / 2.0) * 2.0
    output["nominal_depth_m"] = nominal.astype(np.float32)
    output["depth_regime"] = pd.Series(
        [
            f"{s}|d{d:06.1f}" if np.isfinite(d) else f"{s}|unknown|l{layer}"
            for s, layer, d in zip(frame.station, frame.layer, nominal, strict=True)
        ],
        dtype="string",
    )
    for column in ("plateau_full_length", "plateau_count", "plateau_elapsed"):
        output[column] = output[column].clip(upper=frozen["flank_outer_hours"] * 6)
    return old.FeatureBundle(output, tuple(output), bundle.categorical_columns)


def policies(frame, probabilities, rules, frozen, calibrations):
    result = {
        m: old.decode(frame, probabilities[m], rules, frozen, calibrations[m]["threshold"])
        for m in MODELS
    }
    result["original_balanced_union"] = np.maximum(result["original"], result["balanced"])
    result["balanced_union"] = np.maximum(
        result["original"],
        old.decode(
            frame,
            probabilities["balanced"],
            rules,
            frozen,
            calibrations["balanced_union"]["threshold"],
        ),
    )
    return result


def select_inner(frame, probabilities, rules, frozen):
    calibrations, bits = {}, {}
    for model in MODELS:
        calibrations[model], bits[model] = old.calibrate(frame, probabilities[model], rules, frozen)
    calibrations["balanced_union"], _ = old.calibrate(
        frame, probabilities["balanced"], rules, frozen, bits["original"]
    )
    choices = policies(frame, probabilities, rules, frozen, calibrations)
    chosen = max(POLICIES, key=lambda k: old.metric(frame.label, choices[k])["f1"])
    return chosen, calibrations


def depth_audit(frame, bundle, stats):
    observed = np.isfinite(frame.depth.to_numpy(dtype=float))
    nominal = np.isfinite(bundle.frame.nominal_depth_m.to_numpy())
    unseen = np.asarray(
        [k not in stats["depth"] for k in zip(frame.station, frame.year, frame.layer, strict=True)]
    )
    return {
        "rows": len(frame),
        "raw_depth_missing": int((~observed).sum()),
        "nominal_depth_missing": int((~nominal).sum()),
        "unseen_year_key_rows": int(unseen.sum()),
        "observed_depth_but_nominal_missing": int((observed & ~nominal).sum()),
    }


def load_contract(config_path):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        cfg["experiment_id"] != RUN
        or cfg["official_input_authorized"]
        or cfg["upload_authorized"]
        or cfg["gpu_training_authorized"]
        or [cfg[k] for k in ("screen_fits", "final_inner_fits", "full_fits")] != [12, 2, 2]
        or cfg["threads"] != 4
        or cfg["seed"] != 20260813
    ):
        raise ValueError("frozen experiment boundary violated")
    source = ROOT / "artifacts" / cfg["source_screen"]
    for path, expected in [
        (source / "terminal_result.json", cfg["source_result_sha256"]),
        (source / "contract.json", cfg["source_artifact_contract_sha256"]),
        (
            ROOT / "configs/experiments/p1_score_repair_20260905_v1.json",
            cfg["source_config_sha256"],
        ),
        (Path(old.__file__), cfg["source_runner_sha256"]),
    ]:
        if old.sha(path) != expected:
            raise ValueError("control integrity mismatch: " + path.name)
    prior = json.loads((source / "terminal_result.json").read_text(encoding="utf-8"))
    frozen = json.loads((source / "contract.json").read_text(encoding="utf-8"))
    for path, expected in prior["dependency_hashes"].items():
        if old.sha(ROOT / path) != expected:
            raise ValueError("control dependency mismatch: " + path)
    for path, expected in prior["recipe_hashes"].items():
        if old.sha(ROOT / path) != expected:
            raise ValueError("control recipe mismatch: " + path)
    for record in prior["fit_receipts"]:
        if record["model"] not in MODELS:
            continue
        path = source / f"{record['fold']}_{record['stage']}_{record['model']}.joblib"
        if old.sha(path) != record["model_sha256"]:
            raise ValueError("control model mismatch: " + path.name)
    return cfg, frozen, prior, source


def fit_save(training, evaluation, frozen, model_name, path, *, current_depth=True):
    stats = old.stats_fit(training)
    train_bundle = features(training, stats, frozen, current_depth=current_depth)
    encoder = old.TabularEncoder().fit(train_bundle, np.arange(len(training)))
    matrix = encoder.transform(train_bundle)
    target = training.label.to_numpy(dtype=np.int8)
    started = time.monotonic()
    if model_name == "original":
        base = old.load_config(ROOT / frozen["base_config"], env={})
        model = old._fit_model(
            "xgboost", base.raw["models"]["xgboost"], frozen["seed"], 4, matrix, target
        )
    else:
        recipe = json.loads((ROOT / frozen["lightgbm_recipe"]).read_text(encoding="utf-8"))
        params = old._lgb_parameters(recipe, frozen["seed"], multiclass=False)
        params["n_jobs"] = 4
        model = old.lgb.LGBMClassifier(**params)
        model.fit(matrix, target, sample_weight=old._event_day_weight(training, target))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "train_stats": stats,
            "current_depth": current_depth,
            "feature_config": frozen,
        },
        path,
        compress=3,
    )
    eval_bundle = features(evaluation, stats, frozen, current_depth=current_depth)
    eval_matrix = encoder.transform(eval_bundle)
    prediction = model.predict_proba(eval_matrix)[:, 1]
    loaded = joblib.load(path)
    replay = loaded["model"].predict_proba(loaded["encoder"].transform(eval_bundle))[:, 1]
    if not np.isfinite(prediction).all() or not np.array_equal(prediction, replay):
        raise ValueError("saved model prediction replay failed")
    record = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "model": model_name,
        "rows": len(training),
        "features": matrix.shape[1],
        "training_keys_sha256": keys_digest(training),
        "training_end": pd.to_datetime(training.time, utc=True).max().isoformat(),
        "sha256": old.sha(path),
        "seconds": time.monotonic() - started,
        "reload_exact": True,
    }
    return prediction, record, stats


def keys_digest(frame):
    import hashlib

    return hashlib.sha256(
        pd.util.hash_pandas_object(frame[old.KEYS], index=False).to_numpy(dtype="<u8").tobytes()
    ).hexdigest()


def summarize(part, prediction, reference):
    y = part.label.to_numpy(dtype=bool)
    p, r = part[prediction].to_numpy(dtype=bool), part[reference].to_numpy(dtype=bool)
    result = old.metric(y, p)
    result.update(
        added_tp=int((p & ~r & y).sum()),
        added_fp=int((p & ~r & ~y).sum()),
        removed_tp=int((~p & r & y).sum()),
        removed_fp=int((~p & r & ~y).sum()),
    )
    result["station_layer"] = {
        f"{s}/L{layer}": old.metric(rows.label, rows[prediction])
        for (s, layer), rows in part.groupby(["station", "layer"])
    }
    return result


def execute(config_path):
    cfg, frozen, prior, source = load_contract(config_path)
    data_path = Path(os.environ["P1_DATA_DIR"]).resolve() / "train.csv"
    if old.sha(data_path) != prior["train_sha256"]:
        raise ValueError("distributed train changed")
    artifact, report = ROOT / "artifacts" / RUN, ROOT / "reports" / RUN
    artifact.mkdir(exist_ok=False)
    report.mkdir(parents=True, exist_ok=True)
    for directory in ("01_data", "02_code", "03_training", "04_models", "05_answer", "06_report"):
        (artifact / directory).mkdir()
    started = time.monotonic()
    receipt = {
        "experiment_id": RUN,
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "runner_sha256": old.sha(__file__),
        "config_sha256": old.sha(config_path),
        "source_result_sha256": cfg["source_result_sha256"],
        "train_sha256": old.sha(data_path),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "screen_fits": 0,
        "final_inner_fits": 0,
        "full_fits": 0,
        "reused_control_models": 0,
        "calibration_searches": 0,
        "transition_estimates": 0,
        "official_rows": 0,
        "hidden_rows": 0,
        "csv_written": 0,
        "upload": 0,
        "threads": 4,
        "cuda": False,
        "fits": [],
        "folds": [],
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("numpy", "pandas", "xgboost", "lightgbm", "scikit-learn", "joblib")
        },
    }
    old.write_json(artifact / "ATTEMPT_LOCK.json", receipt)
    old.write_json(artifact / "contract.json", cfg)
    old.write_json(
        artifact / "01_data" / "manifest.json",
        {
            "input": "P1_DATA_DIR/train.csv",
            "sha256": receipt["train_sha256"],
            "source_rows_copied": 0,
            "official_input": False,
        },
    )
    for relative in [
        str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "scripts/run_p1_score_repair_20260905_v1.py",
        "configs/experiments/" + RUN + ".json",
        *prior["dependency_hashes"],
        *prior["recipe_hashes"],
    ]:
        dest = artifact / "02_code" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)

    def progress(stage):
        receipt["stage"] = stage
        receipt["runtime_seconds"] = time.monotonic() - started
        receipt["rss_gib"] = psutil.Process().memory_info().rss / 2**30
        old.write_json(artifact / "progress.json", receipt)
        print(
            json.dumps(
                {
                    k: receipt[k]
                    for k in (
                        "stage",
                        "runtime_seconds",
                        "screen_fits",
                        "final_inner_fits",
                        "full_fits",
                    )
                }
            ),
            flush=True,
        )
        if receipt["runtime_seconds"] > cfg["wall_cap_seconds"] or receipt["rss_gib"] > 12:
            raise RuntimeError("resource cap reached; no restart authorized")

    try:
        frame = pd.read_csv(data_path, usecols=old.RAW + ["label", "anomaly_type"])
        frame["row_id"] = np.arange(len(frame))
        if len(frame) != 776706 or frame.duplicated(old.KEYS).any():
            raise ValueError("train identity failure")
        frame.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        times = pd.to_datetime(frame.time, utc=True)
        parts = []
        for fold, prior_fold in zip(frozen["folds"], prior["folds"], strict=True):
            name = fold["name"]
            begin, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
            cutoff = begin - pd.Timedelta(days=frozen["purge_days"])
            inner_begin = cutoff - pd.Timedelta(days=frozen["inner_days"])
            inner = frame.loc[(times >= inner_begin) & (times < cutoff)].reset_index(drop=True)
            outer = frame.loc[(times >= begin) & (times < end)].reset_index(drop=True)
            selections = {}
            fold_receipt = {
                "fold": name,
                "inner_start": inner_begin.isoformat(),
                "inner_end": cutoff.isoformat(),
                "outer_start": begin.isoformat(),
                "outer_end": end.isoformat(),
                "purge_days": frozen["purge_days"],
            }
            for stage, evaluation, boundary in [
                ("inner", inner, inner_begin - pd.Timedelta(days=frozen["purge_days"])),
                ("outer", outer, cutoff),
            ]:
                training = old.train_slice(frame, boundary)
                if np.intersect1d(training.row_id, evaluation.row_id).size:
                    raise ValueError("training evaluation overlap")
                stats = old.stats_fit(training)
                rules = old.rule_masks(evaluation, stats)
                control_bundle = features(evaluation, stats, frozen, current_depth=False)
                candidate_bundle = features(evaluation, stats, frozen, current_depth=True)
                fold_receipt[stage + "_depth"] = {
                    "control": depth_audit(evaluation, control_bundle, stats),
                    "candidate": depth_audit(evaluation, candidate_bundle, stats),
                }
                predictions = {"control": {}, "candidate": {}}
                for model_name in MODELS:
                    path = source / f"{name}_{stage}_{model_name}.joblib"
                    loaded = joblib.load(path)
                    predictions["control"][model_name] = loaded["model"].predict_proba(
                        loaded["encoder"].transform(control_bundle)
                    )[:, 1]
                    receipt["reused_control_models"] += 1
                    progress(name + "/" + stage + "/" + model_name + "/fit_start")
                    prediction, record, _ = fit_save(
                        training,
                        evaluation,
                        frozen,
                        model_name,
                        artifact / "03_training" / f"{name}_{stage}_{model_name}.joblib",
                    )
                    receipt["fits"].append({**record, "fold": name, "stage": stage})
                    receipt["screen_fits"] += 1
                    predictions["candidate"][model_name] = prediction
                    progress(name + "/" + stage + "/" + model_name + "/fit_complete")
                for arm in ("control", "candidate"):
                    if stage == "inner":
                        selection, calibrations = select_inner(
                            evaluation, predictions[arm], rules, frozen
                        )
                        selections[arm] = (selection, calibrations)
                        receipt["calibration_searches"] += 3
                        if arm == "control":
                            expected = {
                                k: prior_fold["calibrations"][k]
                                for k in ("original", "balanced", "balanced_union")
                            }
                            if (
                                selection != prior_fold["selected_control"]
                                or calibrations != expected
                            ):
                                raise ValueError("control inner reconstruction mismatch")
                    else:
                        selection, calibrations = selections[arm]
                        predictions[arm]["bits"] = policies(
                            evaluation, predictions[arm], rules, frozen, calibrations
                        )[selection]
                if stage == "outer":
                    old_oof = pd.read_parquet(source / f"{name}_intact_oof.parquet")
                    if not old_oof[old.KEYS].equals(evaluation[old.KEYS]):
                        raise ValueError("control OOF key mismatch")
                    for model_name in MODELS:
                        if not np.array_equal(
                            old_oof[model_name + "_probability"], predictions["control"][model_name]
                        ):
                            raise ValueError("control probability replay mismatch")
                    if not np.array_equal(old_oof.selected_control, predictions["control"]["bits"]):
                        raise ValueError("control binary replay mismatch")
                    part = evaluation[old.KEYS + ["row_id", "label", "anomaly_type"]].copy()
                    part["fold"] = name
                    for arm in ("control", "candidate"):
                        part[arm] = predictions[arm]["bits"]
                        for model_name in MODELS:
                            part[arm + "_" + model_name + "_probability"] = predictions[arm][
                                model_name
                            ]
                    sel, cal = selections["control"]
                    part["candidate_fixed_control_threshold"] = policies(
                        evaluation, predictions["candidate"], rules, frozen, cal
                    )[sel]
                    part.to_parquet(artifact / "03_training" / f"{name}_oof.parquet", index=False)
                    parts.append(part)
                    fold_receipt.update(
                        {
                            "selections": {
                                a: {"policy": s, "calibrations": c}
                                for a, (s, c) in selections.items()
                            },
                            "keys_sha256": keys_digest(evaluation),
                            "control_exact_replay": True,
                            "metrics": {
                                c: summarize(part, c, "control")
                                for c in (
                                    "control",
                                    "candidate",
                                    "candidate_fixed_control_threshold",
                                )
                            },
                        }
                    )
                del training
                gc.collect()
            receipt["folds"].append(fold_receipt)
            old.write_json(report / (name + ".json"), fold_receipt)
            progress(name + "/complete")
        pooled = pd.concat(parts, ignore_index=True)
        pooled.to_parquet(artifact / "03_training" / "oof.parquet", index=False)
        receipt["pooled"] = {
            c: summarize(pooled, c, "control")
            for c in ("control", "candidate", "candidate_fixed_control_threshold")
        }
        receipt["delta_f1"] = (
            receipt["pooled"]["candidate"]["f1"] - receipt["pooled"]["control"]["f1"]
        )
        receipt["screen_status"] = "POSITIVE_DEVELOPMENT" if receipt["delta_f1"] > 0 else "NO_GO_A"
        receipt["decoder_status"] = "OFF_AWAIT_SEPARATE_P1_C_AFTER_A_AND_B"
        receipt["oof_sha256"] = old.sha(artifact / "03_training" / "oof.parquet")
        old.write_json(report / "screen-result.json", receipt)
        # Final-inner is defined from the available training end, never from test.
        final_end = times.max() + pd.Timedelta(minutes=10)
        final_begin = final_end - pd.Timedelta(days=frozen["inner_days"])
        training = old.train_slice(frame, final_begin - pd.Timedelta(days=frozen["purge_days"]))
        evaluation = frame.loc[(times >= final_begin) & (times < final_end)].reset_index(drop=True)
        final_prob = {}
        for model_name in MODELS:
            final_prob[model_name], record, stats = fit_save(
                training,
                evaluation,
                frozen,
                model_name,
                artifact / "03_training" / f"final_inner_{model_name}.joblib",
            )
            receipt["fits"].append({**record, "fold": "final", "stage": "inner"})
            receipt["final_inner_fits"] += 1
            progress("final_inner/" + model_name)
        selection, calibration = select_inner(
            evaluation, final_prob, old.rule_masks(evaluation, stats), frozen
        )
        receipt["calibration_searches"] += 3
        final_recipe = {
            "selection": selection,
            "calibrations": calibration,
            "final_inner_start": final_begin.isoformat(),
            "final_inner_end": final_end.isoformat(),
            "depth_policy": cfg["depth_contract"],
            "decoder_on": False,
            "feature_contract": frozen,
            "models": {},
        }
        for model_name in MODELS:
            _, record, _ = fit_save(
                frame,
                frame.iloc[:4096].reset_index(drop=True),
                frozen,
                model_name,
                artifact / "04_models" / (model_name + ".joblib"),
            )
            receipt["fits"].append({**record, "fold": "full", "stage": "train"})
            receipt["full_fits"] += 1
            final_recipe["models"][model_name] = record["sha256"]
            progress("full/" + model_name)
        old.write_json(artifact / "04_models" / "frozen_recipe.json", final_recipe)
        receipt["frozen_recipe_sha256"] = old.sha(artifact / "04_models" / "frozen_recipe.json")
        receipt["final_recipe"] = final_recipe
        if (receipt["screen_fits"], receipt["final_inner_fits"], receipt["full_fits"]) != (
            12,
            2,
            2,
        ):
            raise ValueError("fit count mismatch")
        receipt["status"] = "TERMINAL_MODELS_FROZEN_OFFICIAL_UNREAD"
        receipt["expected_official_score"] = None
        progress("terminal")
    except Exception as exc:
        receipt.update(
            status="TERMINAL_TECHNICAL_FAILURE",
            error=str(exc),
            runtime_seconds=time.monotonic() - started,
        )
        raise
    finally:
        old.write_json(artifact / "terminal_result.json", receipt)
        old.write_json(report / "result.json", receipt)


def verify(config_path):
    """Fresh-process train-only OOF replay plus independent confusion arithmetic."""
    from sklearn.metrics import confusion_matrix, f1_score

    _cfg, frozen, _prior, _source = load_contract(config_path)
    artifact, report = ROOT / "artifacts" / RUN, ROOT / "reports" / RUN
    result = json.loads((report / "result.json").read_text(encoding="utf-8"))
    if (
        old.sha(__file__) != result["runner_sha256"]
        or old.sha(config_path) != result["config_sha256"]
    ):
        raise ValueError("sealed execution code/config changed")
    train_path = Path(os.environ["P1_DATA_DIR"]) / "train.csv"
    if old.sha(train_path) != result["train_sha256"]:
        raise ValueError("training input drift")
    frame = pd.read_csv(train_path, usecols=old.RAW + ["label", "anomaly_type"])
    frame.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    times = pd.to_datetime(frame.time, utc=True)
    started = time.monotonic()
    checks = []
    for fold in frozen["folds"]:
        ev = frame.loc[
            (times >= pd.Timestamp(fold["start"])) & (times < pd.Timestamp(fold["end"]))
        ].reset_index(drop=True)
        stored = pd.read_parquet(artifact / "03_training" / f"{fold['name']}_oof.parquet")
        if not ev[old.KEYS].equals(stored[old.KEYS]) or not np.array_equal(ev.label, stored.label):
            raise ValueError("OOF key/truth alignment failed")
        for model in MODELS:
            path = artifact / "03_training" / f"{fold['name']}_outer_{model}.joblib"
            expected = next(
                r["sha256"]
                for r in result["fits"]
                if r["fold"] == fold["name"] and r["stage"] == "outer" and r["model"] == model
            )
            if old.sha(path) != expected:
                raise ValueError("saved model hash drift")
            load = joblib.load(path)
            bundle = features(ev, load["train_stats"], frozen, current_depth=True)
            probability = load["model"].predict_proba(load["encoder"].transform(bundle))[:, 1]
            difference = float(
                np.max(np.abs(probability - stored["candidate_" + model + "_probability"]))
            )
            if difference != 0:
                raise ValueError("fresh process model replay not exact")
            checks.append({"fold": fold["name"], "model": model, "max_abs_diff": difference})
    final_recipe_path = artifact / "04_models" / "frozen_recipe.json"
    if old.sha(final_recipe_path) != result["frozen_recipe_sha256"]:
        raise ValueError("frozen recipe hash drift")
    for record in result["fits"]:
        if old.sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("model artifact hash drift")
    final_replay = []
    for model in MODELS:
        load = joblib.load(artifact / "04_models" / (model + ".joblib"))
        probe = frame.iloc[:4096].reset_index(drop=True)
        bundle = features(probe, load["train_stats"], frozen, current_depth=True)
        probability = load["model"].predict_proba(load["encoder"].transform(bundle))[:, 1]
        if not np.isfinite(probability).all():
            raise ValueError("full model probe not finite")
        final_replay.append({"model": model, "probe_rows": len(probe), "finite": True})
    pooled = pd.read_parquet(artifact / "03_training" / "oof.parquet")
    metrics = {}
    for name in ("control", "candidate", "candidate_fixed_control_threshold"):
        tn, fp, fn, tp = confusion_matrix(pooled.label, pooled[name], labels=[0, 1]).ravel()
        metric = {
            "f1": float(f1_score(pooled.label, pooled[name])),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
        if any(metric[k] != result["pooled"][name][k] for k in metric):
            raise ValueError("independent metrics mismatch")
        metrics[name] = metric
    qa = {
        "status": "PASS",
        "keys_unique": not pooled.duplicated(old.KEYS).any(),
        "rows": len(pooled),
        "metrics": metrics,
        "replay": checks,
        "all_16_model_hashes_match": True,
        "full_model_probe": final_replay,
        "result_sha256": old.sha(report / "result.json"),
        "oof_sha256": old.sha(artifact / "03_training" / "oof.parquet"),
        "screen_fits": result["screen_fits"],
        "final_inner_fits": result["final_inner_fits"],
        "full_fits": result["full_fits"],
        "official_rows": 0,
        "hidden_rows": 0,
        "csv_written": 0,
        "upload": 0,
        "verify_seconds": time.monotonic() - started,
        "scope": "independent sklearn/confusion arithmetic and saved-model fresh-process replay; not retraining",
    }
    if len(pooled) != 421032 or not qa["keys_unique"]:
        raise ValueError("pooled row contract failed")
    old.write_json(report / "independent-qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiments" / (RUN + ".json")
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    execute(args.config) if args.execute else verify(args.config)
