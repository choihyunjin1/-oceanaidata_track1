"""Train-only, sealed P1 flank representation/decoder screening; no official I/O."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from run_p1_meaningful_learning_curve_generation_v1 import _event_day_weight, _lgb_parameters

from p1_qc.config import load_config
from p1_qc.features import FeatureBundle, build_features
from p1_qc.pipeline import TabularEncoder, _fit_model, apply_postprocess
from p1_qc.rules import detect_plateaus

RUN = "p1_score_repair_20260905_v1"
KEYS = ["station", "year", "layer", "time"]
RAW = KEYS + ["temp", "psal", "depth"]
DEPENDENCIES = [
    "src/p1_qc/features.py",
    "src/p1_qc/pipeline.py",
    "src/p1_qc/models_tabular.py",
    "src/p1_qc/rules.py",
    "src/p1_qc/postprocess.py",
    "src/p1_qc/data.py",
    "scripts/run_p1_meaningful_learning_curve_generation_v1.py",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def segments(frame):
    """Input must be station/layer/time sorted; segment codes never bridge a gap."""
    t = pd.to_datetime(frame.time, utc=True)
    group_change = frame.station.ne(frame.station.shift()) | frame.layer.ne(frame.layer.shift())
    return (group_change | t.diff().ne(pd.Timedelta(minutes=10))).cumsum()


def stats_fit(frame):
    """Only training observations may fit station/deployment statistics."""
    depth = frame.groupby(["station", "year", "layer"], observed=True).depth.median().to_dict()
    delta = frame.temp.diff().abs().where(segments(frame).eq(segments(frame).shift()))
    scale = (
        frame.assign(__delta=delta)
        .groupby(["station", "layer"], observed=True)
        .__delta.median()
        .to_dict()
    )
    return {"depth": depth, "spike_scale": scale}


def feature_pair(frame, train_stats, cfg):
    """Common clean existing features plus one distinct excluded-center flank arm."""
    frame = frame.reset_index(drop=True)
    bundle = build_features(frame[RAW], config=load_config(ROOT / cfg["base_config"], env={}))
    base = bundle.frame
    nominal = np.array(
        [
            train_stats["depth"].get(k, np.nan)
            for k in zip(frame.station, frame.year, frame.layer, strict=True)
        ]
    )
    nominal = np.round(nominal / 2.0) * 2.0
    base["nominal_depth_m"] = nominal.astype(np.float32)
    base["depth_regime"] = pd.Series(
        [
            f"{s}|d{d:06.1f}" if np.isfinite(d) else f"{s}|unknown|l{layer}"
            for s, layer, d in zip(frame.station, frame.layer, nominal, strict=True)
        ],
        dtype="string",
    )
    # Unlimited plateau totals are replaced by capped local evidence in BOTH arms.
    for column in ("plateau_full_length", "plateau_count", "plateau_elapsed"):
        base[column] = base[column].clip(upper=cfg["flank_outer_hours"] * 6)
    extra = flank_features(frame, cfg)
    full = pd.concat([base, extra], axis=1)
    return (
        FeatureBundle(base, tuple(base), bundle.categorical_columns),
        FeatureBundle(full, tuple(full), bundle.categorical_columns),
    )


def flank_features(frame, cfg):
    """Current-minus-flank uses observations 24--168h away, never across gaps."""
    seg = segments(frame)
    shift = cfg["flank_inner_hours"] * 6 + 1
    width = (cfg["flank_outer_hours"] - cfg["flank_inner_hours"]) * 6
    minimum = max(2, int(np.ceil(width * cfg["flank_min_fraction"])))
    peers = frame.groupby(["station", "time"], observed=True).temp
    count = peers.transform("count") - 1
    residual = frame.temp - (peers.transform("sum") - frame.temp) / count.replace(0, np.nan)
    source = {"temp": frame.temp, "psal": frame.psal, "peer": residual}
    out = pd.DataFrame(index=frame.index)
    signed = {}
    for name, values in source.items():
        left = pd.Series(np.nan, index=frame.index)
        right = left.copy()
        support_left = left.copy()
        support_right = left.copy()
        for positions in frame.groupby(seg, sort=False).indices.values():
            part = values.iloc[positions].reset_index(drop=True)
            past = part.shift(shift).rolling(width, min_periods=minimum)
            future = (
                part.iloc[::-1]
                .reset_index(drop=True)
                .shift(shift)
                .rolling(width, min_periods=minimum)
            )
            left.iloc[positions] = past.median().to_numpy()
            right.iloc[positions] = future.median().to_numpy()[::-1]
            support_left.iloc[positions] = (
                part.shift(shift).rolling(width, min_periods=1).count().to_numpy() / width
            )
            support_right.iloc[positions] = (
                part.iloc[::-1]
                .reset_index(drop=True)
                .shift(shift)
                .rolling(width, min_periods=1)
                .count()
                .to_numpy()[::-1]
                / width
            )
        dl, dr = values - left, values - right
        for tag, array in {
            "left_delta": dl,
            "right_delta": dr,
            "flank_change": right - left,
            "min_abs_delta": np.minimum(dl.abs(), dr.abs()),
            "same_sign": np.sign(dl) * np.sign(dr),
            "left_support": support_left,
            "right_support": support_right,
        }.items():
            out[f"flank24_168_{name}_{tag}"] = array
        signed[name] = (dl + dr) / 2
    # Dimensionless disagreement: scale from the same flank differences, not targets.
    for name in ("psal", "peer"):
        out[f"flank24_168_temp_{name}_sign_disagreement"] = np.sign(signed["temp"]) != np.sign(
            signed[name]
        )
        out[f"flank24_168_temp_{name}_both_available"] = (
            signed["temp"].notna() & signed[name].notna()
        )
    return out.astype(np.float32)


def rule_masks(frame, train_stats):
    plateau = detect_plateaus(frame).to_numpy()
    seg = segments(frame)
    left, right = frame.temp.shift(), frame.temp.shift(-1)
    valid = seg.eq(seg.shift()) & seg.eq(seg.shift(-1))
    excursion = np.minimum((frame.temp - left).abs(), (frame.temp - right).abs())
    threshold = np.array(
        [
            max(0.5, 8 * train_stats["spike_scale"].get(k, 0.0))
            for k in zip(frame.station, frame.layer, strict=True)
        ]
    )
    spike = valid & (excursion >= threshold) & ((right - left).abs() <= 0.35 * excursion)
    return plateau, spike.to_numpy()


def metric(y, prediction):
    y, p = np.asarray(y, dtype=bool), np.asarray(prediction, dtype=bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    return {
        "rows": len(y),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def decode(frame, probability, rules, cfg, threshold):
    return apply_postprocess(
        frame,
        probability,
        *rules,
        {
            "high_threshold": threshold,
            "low_threshold": threshold * cfg["low_ratio"],
            "close_gap_rows": cfg["close_gap_rows"],
            "minimum_positive_run": cfg["minimum_positive_run"],
        },
    )


def calibrate(frame, probability, rules, cfg, anchor=None):
    choices = []
    for threshold in cfg["threshold_grid"]:
        bits = decode(frame, probability, rules, cfg, threshold)
        if anchor is not None:
            bits = np.maximum(bits, anchor)
        choices.append((metric(frame.label, bits)["f1"], threshold, bits))
    best = max(choices, key=lambda x: (x[0], x[1]))
    return {"threshold": best[1], "inner_f1": best[0]}, best[2]


def train_slice(frame, end):
    """Retreat trailing positive runs using only labels BEFORE the cutoff."""
    part = frame.loc[pd.to_datetime(frame.time, utc=True) < end].copy()
    keep = np.ones(len(part), dtype=bool)
    part.reset_index(drop=True, inplace=True)
    for positions in part.groupby(["station", "layer"], sort=False).indices.values():
        cursor = len(positions) - 1
        while cursor >= 0 and int(part.label.iloc[positions[cursor]]) == 1:
            keep[positions[cursor]] = False
            if cursor == 0 or pd.Timestamp(part.time.iloc[positions[cursor]]) - pd.Timestamp(
                part.time.iloc[positions[cursor - 1]]
            ) != pd.Timedelta(minutes=10):
                break
            cursor -= 1
    return part.loc[keep].reset_index(drop=True)


def fragmented(frame, cfg):
    """Missing observation stress independent of label/type; retain original keys."""
    rng = np.random.default_rng(cfg["fragmentation"]["seed"])
    keep = np.ones(len(frame), dtype=bool)
    for positions in frame.groupby(segments(frame), sort=False).indices.values():
        starts = np.flatnonzero(
            rng.random(len(positions)) < cfg["fragmentation"]["start_probability"]
        )
        for start in starts:
            length = int(rng.choice(cfg["fragmentation"]["length_rows"]))
            keep[positions[start : start + length]] = False
    return frame.loc[keep].reset_index(drop=True), keep


def diagnostic(frame, prediction, reference, probability):
    y, p, r = frame.label.to_numpy().astype(bool), prediction.astype(bool), reference.astype(bool)
    seg = segments(frame)
    events = (seg.ne(seg.shift()) | frame.label.ne(frame.label.shift())).cumsum()
    length = frame.groupby(events).label.transform("size").to_numpy()
    long = y & (length >= 48 * 6)
    event_hits = (
        frame.assign(__event=events, __p=p).loc[y].groupby("__event").__p.agg(["sum", "size"])
    )
    partial = (event_hits["sum"] > 0) & (event_hits["sum"] < event_hits["size"])
    return {
        **metric(y, p),
        "long_positive_rows": int(long.sum()),
        "long_recall": float(p[long].mean()) if long.any() else None,
        "long_probability_mean": float(probability[long].mean()) if long.any() else None,
        "long_below_probability_005": int((long & (probability < 0.05)).sum()),
        "normal_fp_rate": float(p[~y].mean()),
        "added_tp": int((p & ~r & y).sum()),
        "added_fp": int((p & ~r & ~y).sum()),
        "removed_tp": int((~p & r & y).sum()),
        "removed_fp": int((~p & r & ~y).sum()),
        "partial_event_missing_rows": int(
            (event_hits.loc[partial, "size"] - event_hits.loc[partial, "sum"]).sum()
        ),
        "station": {
            str(s): metric(frame.loc[frame.station == s, "label"], p[frame.station == s])
            for s in frame.station.unique()
        },
    }


def run(cfg_path):
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if (
        cfg["experiment_id"] != RUN
        or cfg["training_input"] != "train.csv"
        or cfg["official_access_authorized"]
    ):
        raise ValueError("contract boundary violation")
    root_data = Path(os.environ["P1_DATA_DIR"]).resolve()
    train_path = root_data / "train.csv"
    if sha(train_path) != cfg["expected_training_sha256"]:
        raise ValueError("distributed training hash mismatch")
    artifact, report = ROOT / "artifacts" / RUN, ROOT / "reports" / RUN
    artifact.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    lock = artifact / "ATTEMPT_LOCK.json"
    started = time.monotonic()
    receipt = {
        "experiment_id": RUN,
        "pid": os.getpid(),
        "started": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "config_sha256": sha(cfg_path),
        "runner_sha256": sha(__file__),
        "dependency_hashes": {p: sha(ROOT / p) for p in DEPENDENCIES},
        "train_sha256": sha(train_path),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "threads": cfg["threads"],
        "cuda": False,
        "lgbm_fits": 0,
        "xgboost_fits": 0,
        "calibration_searches": 0,
        "official_rows": 0,
        "hidden_rows": 0,
        "csv_written": 0,
        "upload": 0,
        "max_rss_gib": 0.0,
        "fit_receipts": [],
        "folds": [],
    }
    with lock.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    write_json(artifact / "contract.json", cfg)

    def progress(stage):
        receipt["stage"] = stage
        receipt["runtime_seconds"] = time.monotonic() - started
        receipt["max_rss_gib"] = max(
            receipt["max_rss_gib"], psutil.Process().memory_info().rss / 2**30
        )
        write_json(artifact / "progress.json", receipt)
        print(
            json.dumps(
                {
                    k: receipt[k]
                    for k in [
                        "stage",
                        "runtime_seconds",
                        "lgbm_fits",
                        "xgboost_fits",
                        "max_rss_gib",
                    ]
                }
            ),
            flush=True,
        )
        if (
            receipt["runtime_seconds"] > cfg["wall_cap_seconds"]
            or receipt["max_rss_gib"] > cfg["memory_limit_gib"]
        ):
            raise RuntimeError("sealed resource budget exceeded; no auto restart")

    try:
        frame = pd.read_csv(train_path, usecols=RAW + ["label", "anomaly_type"])
        if frame.duplicated(KEYS).any() or not frame.label.isin([0, 1]).all():
            raise ValueError("training keys/labels invalid")
        frame["row_id"] = np.arange(len(frame))
        frame.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        base_cfg = load_config(ROOT / cfg["base_config"], env={})
        recipe = json.loads((ROOT / cfg["lightgbm_recipe"]).read_text(encoding="utf-8"))
        receipt["recipe_hashes"] = {
            p: sha(ROOT / p) for p in [cfg["base_config"], cfg["lightgbm_recipe"]]
        }
        progress("TRAIN_READ_VALIDATED")
        all_parts = []
        for fold in cfg["folds"]:
            name, begin, end = fold["name"], pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
            cutoff = begin - pd.Timedelta(days=cfg["purge_days"])
            inner_begin = cutoff - pd.Timedelta(days=cfg["inner_days"])
            times = pd.to_datetime(frame.time, utc=True)
            inner_eval = frame.loc[(times >= inner_begin) & (times < cutoff)].reset_index(drop=True)
            outer_eval = frame.loc[(times >= begin) & (times < end)].reset_index(drop=True)
            stage_predictions = {}
            fold_receipt = {
                "fold": name,
                "outer_rows": len(outer_eval),
                "inner_rows": len(inner_eval),
                "outer_start": begin.isoformat(),
                "outer_end": end.isoformat(),
                "inner_start": inner_begin.isoformat(),
                "inner_end": cutoff.isoformat(),
                "purge_days": cfg["purge_days"],
                "previously_exposed_development_replay": True,
            }
            for stage, boundary, evaluation in [
                ("inner", inner_begin - pd.Timedelta(days=cfg["purge_days"]), inner_eval),
                ("outer", cutoff, outer_eval),
            ]:
                training = train_slice(frame, boundary)
                if set(training.row_id) & set(evaluation.row_id):
                    raise ValueError("train evaluation overlap")
                stats = stats_fit(training)
                progress(f"{name}/{stage}/FEATURES")
                train_bundles = feature_pair(training, stats, cfg)
                eval_bundles = feature_pair(evaluation, stats, cfg)
                stress, keep = fragmented(evaluation, cfg) if stage == "outer" else (None, None)
                stress_bundles = feature_pair(stress, stats, cfg) if stress is not None else None
                rules = rule_masks(evaluation, stats)
                stress_rules = rule_masks(stress, stats) if stress is not None else None
                prediction = {
                    "frame": evaluation,
                    "rules": rules,
                    "stress": stress,
                    "keep": keep,
                    "stress_rules": stress_rules,
                }
                for model_name, which in [("balanced", 0), ("flank", 1), ("original", 0)]:
                    encoder = TabularEncoder().fit(train_bundles[which], np.arange(len(training)))
                    x = encoder.transform(train_bundles[which])
                    xp = encoder.transform(eval_bundles[which])
                    target = training.label.to_numpy(dtype=np.int8)
                    progress(f"{name}/{stage}/{model_name}/FIT_STARTED")
                    fit_started = time.monotonic()
                    if model_name == "original":
                        model = _fit_model(
                            "xgboost",
                            base_cfg.raw["models"]["xgboost"],
                            cfg["seed"],
                            cfg["threads"],
                            x,
                            target,
                        )
                        receipt["xgboost_fits"] += 1
                    else:
                        parameters = _lgb_parameters(recipe, cfg["seed"], multiclass=False)
                        parameters["n_jobs"] = cfg["threads"]
                        model = lgb.LGBMClassifier(**parameters)
                        model.fit(x, target, sample_weight=_event_day_weight(training, target))
                        receipt["lgbm_fits"] += 1
                    elapsed = time.monotonic() - fit_started
                    model_path = artifact / f"{name}_{stage}_{model_name}.joblib"
                    joblib.dump(
                        {"model": model, "encoder": encoder, "train_stats": stats, "config": cfg},
                        model_path,
                        compress=3,
                    )
                    probability = model.predict_proba(xp)[:, 1]
                    if not np.isfinite(probability).all():
                        raise ValueError("nonfinite probabilities")
                    prediction[model_name] = probability
                    if stress is not None:
                        prediction[model_name + "_stress"] = model.predict_proba(
                            encoder.transform(stress_bundles[which])
                        )[:, 1]
                    receipt["fit_receipts"].append(
                        {
                            "fold": name,
                            "stage": stage,
                            "model": model_name,
                            "seconds": elapsed,
                            "training_rows": len(training),
                            "features": x.shape[1],
                            "model_sha256": sha(model_path),
                        }
                    )
                    if (
                        len(receipt["fit_receipts"]) == 1
                        and elapsed > cfg["first_fit_warning_seconds"]
                    ):
                        print("FIRST_FIT_RUNTIME_WARNING", flush=True)
                    progress(f"{name}/{stage}/{model_name}/FIT_COMPLETE")
                    del model, x, xp
                    gc.collect()
                stage_predictions[stage] = prediction
                fold_receipt[stage + "_train_rows"] = len(training)
                fold_receipt[stage + "_train_max"] = (
                    pd.to_datetime(training.time, utc=True).max().isoformat()
                )
                del training, train_bundles, eval_bundles, stress_bundles
                gc.collect()
            inner = stage_predictions["inner"]
            calibrations, inner_bits = {}, {}
            for model_name in ["original", "balanced", "flank"]:
                calibrations[model_name], inner_bits[model_name] = calibrate(
                    inner_eval, inner[model_name], inner["rules"], cfg
                )
                receipt["calibration_searches"] += 1
            inner_bits["original_balanced_union"] = np.maximum(
                inner_bits["original"], inner_bits["balanced"]
            )
            for model_name in ["balanced", "flank"]:
                key = model_name + "_union"
                calibrations[key], inner_bits[key] = calibrate(
                    inner_eval, inner[model_name], inner["rules"], cfg, inner_bits["original"]
                )
                receipt["calibration_searches"] += 1
            controls = ["original", "balanced", "original_balanced_union", "balanced_union"]
            best_control = max(
                controls, key=lambda k: metric(inner_eval.label, inner_bits[k])["f1"]
            )
            best_candidate = max(
                ["flank", "flank_union"],
                key=lambda k: metric(inner_eval.label, inner_bits[k])["f1"],
            )
            fold_receipt.update(
                {
                    "calibrations": calibrations,
                    "selected_control": best_control,
                    "selected_candidate": best_candidate,
                }
            )
            outer = stage_predictions["outer"]
            intact_bits = {}
            for surface in ["intact", "fragmented"]:
                ev = outer_eval if surface == "intact" else outer["stress"]
                rules = outer["rules"] if surface == "intact" else outer["stress_rules"]
                suffix = "" if surface == "intact" else "_stress"
                bits = {
                    k: decode(ev, outer[k + suffix], rules, cfg, calibrations[k]["threshold"])
                    for k in ["original", "balanced", "flank"]
                }
                bits["original_balanced_union"] = np.maximum(bits["original"], bits["balanced"])
                for k in ["balanced", "flank"]:
                    bits[k + "_union"] = np.maximum(
                        bits["original"],
                        decode(
                            ev,
                            outer[k + suffix],
                            rules,
                            cfg,
                            calibrations[k + "_union"]["threshold"],
                        ),
                    )
                bits["selected_control"], bits["selected_candidate"] = (
                    bits[best_control],
                    bits[best_candidate],
                )
                part = ev[KEYS + ["row_id", "label", "anomaly_type"]].copy()
                part["fold"], part["surface"] = name, surface
                surface_metrics = {}
                for k, value in bits.items():
                    part[k] = value
                    probability_name = (
                        "flank"
                        if k in ["flank", "flank_union", "selected_candidate"]
                        else "balanced"
                    )
                    surface_metrics[k] = diagnostic(
                        ev, value, bits["selected_control"], outer[probability_name + suffix]
                    )
                for k in ["original", "balanced", "flank"]:
                    part[k + "_probability"] = outer[k + suffix]
                part.to_parquet(artifact / f"{name}_{surface}_oof.parquet", index=False)
                all_parts.append(part)
                fold_receipt[surface] = surface_metrics
                if surface == "fragmented":
                    fold_receipt["intact_same_retained_rows"] = {
                        k: metric(outer_eval.label.to_numpy()[outer["keep"]], v[outer["keep"]])
                        for k, v in intact_bits.items()
                    }
                else:
                    intact_bits = bits
            receipt["folds"].append(fold_receipt)
            write_json(report / f"{name}.json", fold_receipt)
            progress(f"{name}/COMPLETE")
            del stage_predictions
            gc.collect()
        pooled = pd.concat(all_parts, ignore_index=True)
        qa = pooled.loc[pooled.surface == "intact"]
        np.savez_compressed(
            artifact / "qa_oof.npz",
            key=qa[KEYS].astype(str).agg("|".join, axis=1).to_numpy(dtype=str),
            fold=qa.fold.to_numpy(dtype=str),
            truth=qa.label.to_numpy(dtype=np.int8),
            reference=qa.selected_control.to_numpy(dtype=np.int8),
            prediction=qa.selected_candidate.to_numpy(dtype=np.int8),
        )
        if (
            receipt["lgbm_fits"] != cfg["screen_max_lgbm_fits"]
            or receipt["xgboost_fits"] != cfg["screen_max_xgboost_fits"]
        ):
            raise RuntimeError("fit count mismatch")
        receipt["pooled"] = {}
        for surface in ["intact", "fragmented"]:
            part = pooled.loc[pooled.surface == surface]
            receipt["pooled"][surface] = {k: metric(part.label, part[k]) for k in list(intact_bits)}
        delta = (
            receipt["pooled"]["intact"]["selected_candidate"]["f1"]
            - receipt["pooled"]["intact"]["selected_control"]["f1"]
        )
        receipt["delta_f1"] = delta
        receipt["conditional_public_points_if_transferred"] = delta * 26.6
        receipt["status"] = (
            "SCREEN_POSITIVE_CONFIRM_SEEDS_PENDING" if delta > 0 else "SCREEN_NO_GO_RETAIN_CONTROL"
        )
        receipt["official_score"] = (
            "UNMEASURED; current official 28.909341 is not this retrained control"
        )
        progress("TERMINAL")
        write_json(report / "result.json", receipt)
        write_json(artifact / "terminal_result.json", receipt)
    except Exception as exc:
        receipt.update(
            {
                "status": "TERMINAL_TECHNICAL_FAILURE",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.monotonic() - started,
            }
        )
        write_json(report / "result.json", receipt)
        write_json(artifact / "terminal_result.json", receipt)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiments" / (RUN + ".json")
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute required; one attempt only")
    run(args.config)
