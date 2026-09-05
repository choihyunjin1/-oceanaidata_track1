"""P1 frozen forward bracket comparison, CPU2, no official I/O or answer generation."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import ocean_evaluation_contract_v5 as evaluation
import pandas as pd
import run_p1_depth_contract_repair_20260905_v2 as policy
import run_p1_score_repair_20260905_v1 as old

ID = "p1_bracket_forward_20260906_v1"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
OUT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID
joblib = old.joblib


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def key_digest(frame):
    return digest(pd.util.hash_pandas_object(frame[old.KEYS], index=False).to_numpy())


def load_contract():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (cfg["experiment_id"], cfg["threads"], cfg["gpu"], cfg["max_fits"]) != (ID, 2, False, 20):
        raise ValueError("experiment resource contract changed")
    if any(cfg[k] for k in ("official_access", "csv_generation", "upload")):
        raise ValueError("official operations are not authorized")
    if cfg["windows_hours"] != [6, 24, 72] or cfg["flank_hours"] != 1:
        raise ValueError("bracket bank changed")
    for path, expected in cfg["source_pins"].items():
        if old.sha(ROOT / path) != expected:
            raise ValueError("source pin changed: " + path)
    frozen = json.loads((ROOT / cfg["source_feature_contract"]).read_text(encoding="utf-8"))
    return cfg, frozen, evaluation.load_contract(ROOT / cfg["evaluation_contract"])


def training_source(cfg, *, row_id=True):
    path = Path(os.environ["P1_DATA_DIR"]).resolve() / "train.csv"
    if old.sha(path) != cfg["expected_training_sha256"]:
        raise ValueError("distributed training hash mismatch")
    frame = pd.read_csv(path, usecols=old.RAW + ["label", "anomaly_type"])
    if (
        len(frame) != 776706
        or frame.duplicated(old.KEYS).any()
        or not frame.label.isin([0, 1]).all()
    ):
        raise ValueError("training count/keys/labels invalid")
    if row_id:
        frame["row_id"] = np.arange(len(frame))
    return frame.sort_values(["station", "layer", "time"], kind="stable").reset_index(drop=True)


def bracket_features(raw, cfg):
    """Centered duration W with two exterior one-hour flanks; maximal radius 37h.

    Exact-cadence segments make six rows exactly one hour. Missing rows split the
    segment; they never compress a physical gap. Flanks exclude the interior.
    No label-dependent window selection, station parameters or fitted constants.
    """
    if set(raw) != set(old.RAW) or raw.duplicated(old.KEYS).any():
        raise ValueError("bracket requires raw observation columns only, with unique keys")
    if not raw.index.equals(pd.RangeIndex(len(raw))):
        raise ValueError("bracket requires reset row index")
    result = pd.DataFrame(index=raw.index)
    for positions in raw.groupby(old.segments(raw), sort=False).indices.values():
        group = raw.iloc[positions]
        stamps = pd.to_datetime(group.time, utc=True)
        if len(group) > 1 and not stamps.diff().iloc[1:].eq(pd.Timedelta(minutes=10)).all():
            raise ValueError("noncontiguous segment")
        values = pd.Series(group.temp.to_numpy(dtype=float))
        local = {}
        for hours in cfg["windows_hours"]:
            half = int(hours * 3)
            flank = int(cfg["flank_hours"] * 6)
            # Interior [t-W/2,t+W/2], flanks strictly exterior to it.
            center = values.rolling(
                2 * half + 1, center=True, min_periods=max(3, int((2 * half + 1) * 0.25))
            ).median()
            left = values.shift(half + 1).rolling(flank, min_periods=2).median()
            right = (
                values.iloc[::-1]
                .reset_index(drop=True)
                .shift(half + 1)
                .rolling(flank, min_periods=2)
                .median()
                .iloc[::-1]
                .reset_index(drop=True)
            )
            # Boundary-adjacent medians use interior one-hour bands, not extra outer context.
            left_inside = values.shift(half - flank + 1).rolling(flank, min_periods=2).median()
            right_inside = (
                values.iloc[::-1]
                .reset_index(drop=True)
                .shift(half - flank + 1)
                .rolling(flank, min_periods=2)
                .median()
                .iloc[::-1]
                .reset_index(drop=True)
            )
            incoming, outgoing = left_inside - left, right - right_inside
            expected = 2 * (half + flank) + 1
            coverage = values.rolling(expected, center=True, min_periods=1).count() / expected
            for name, array in {
                "entry_jump": incoming,
                "exit_jump": outgoing,
                "jump_sign_product": np.sign(incoming) * np.sign(outgoing),
                "jump_cancellation": incoming + outgoing,
                "outside_difference": right - left,
                "interior_minus_left": center - left,
                "interior_minus_right": center - right,
                "outside_return_abs": (right - left).abs(),
                "coverage": coverage,
            }.items():
                local[f"bracket_{hours}h_{name}"] = array.to_numpy()
        for name, array in local.items():
            if name not in result:
                result[name] = np.nan
            result.loc[positions, name] = array
    return result.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def base_bundle(frame, stats, frozen):
    # Exactly the clean 80-column base. Avoid computing the rejected extra flank arm.
    return policy.features(frame, stats, frozen, current_depth=False)


def bundle(frame, stats, frozen, cfg, bracket=False):
    base = base_bundle(frame, stats, frozen)
    if base.frame.shape[1] != 80:
        raise ValueError("original clean feature count changed")
    if not bracket:
        return base
    extra = bracket_features(frame[old.RAW], cfg)
    if set(extra) & set(base.frame):
        raise ValueError("duplicate bracket/base names")
    combined = pd.concat([base.frame, extra], axis=1)
    return old.FeatureBundle(combined, tuple(combined), base.categorical_columns)


def partition_inputs(data, train_mask, validation_mask, frozen, cfg, bracket=False):
    """Select partitions BEFORE any stats/features/encoder/rule/decoder computation."""
    if np.any(train_mask & validation_mask):
        raise ValueError("overlapping partitions")
    train = data.loc[train_mask].reset_index(drop=True)
    validation = data.loc[validation_mask].reset_index(drop=True)
    if train.empty or validation.empty or train.label.nunique() != 2:
        raise ValueError("inner/outer support insufficient; dates must not be changed")
    stats = old.stats_fit(train)
    train_bundle = bundle(train, stats, frozen, cfg, bracket)
    validation_bundle = bundle(validation, stats, frozen, cfg, bracket)
    encoder = old.TabularEncoder().fit(train_bundle, np.arange(len(train)))
    return {
        "train": train,
        "validation": validation,
        "stats": stats,
        "encoder": encoder,
        "x": encoder.transform(train_bundle),
        "xe": encoder.transform(validation_bundle),
        "rules": old.rule_masks(validation, stats),
    }


def all_split_masks(data, eval_contract, cfg):
    # Same run convention as frozen v5, computed once over historical metadata.
    meta = evaluation.p1_run_metadata(data, eval_contract)
    times = pd.to_datetime(data.time, utc=True)
    # pandas asi8 may use non-nanosecond resolution; use Timestamp.value explicitly.
    run_starts = (
        pd.Series([stamp.value for stamp in times])
        .groupby(meta["run_id"])
        .transform("min")
        .to_numpy()
    )
    labels = data.label.to_numpy()
    output, support = {}, []
    for fold in eval_contract["P1"]["folds"]:
        if fold["warmup"]:
            continue
        start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
        cutoff = start - pd.Timedelta(days=cfg["purge_days"])
        inner_start = cutoff - pd.Timedelta(days=cfg["inner_days"])
        for stage, begin, stop in (("inner", inner_start, cutoff), ("outer", start, end)):
            train_end = begin - pd.Timedelta(days=cfg["purge_days"])
            train = (times < train_end).to_numpy() & (
                (labels == 0) | (meta["end_ns"] < train_end.value)
            )
            validation = (
                (labels == 0) & (times >= begin).to_numpy() & (times < stop).to_numpy()
            ) | ((labels == 1) & (run_starts >= begin.value) & (run_starts < stop.value))
            if not train.any() or not validation.any() or len(np.unique(labels[train])) != 2:
                raise ValueError(f"{fold['id']}/{stage}: insufficient support, no date amendment")
            if stage == "outer":
                expected = (
                    (labels == 0) & (times >= start).to_numpy() & (times < end).to_numpy()
                ) | ((labels == 1) & (meta["owner"] == fold["id"]))
                if not np.array_equal(expected, validation):
                    raise ValueError("v5 run ownership mismatch")
            output[(fold["id"], stage)] = (train, validation)
            known = set(zip(data.loc[train, "station"], data.loc[train, "layer"], strict=True))
            unseen = [
                key not in known
                for key in zip(
                    data.loc[validation, "station"], data.loc[validation, "layer"], strict=True
                )
            ]
            support.append(
                {
                    "fold": fold["id"],
                    "stage": stage,
                    "train_rows": int(train.sum()),
                    "validation_rows": int(validation.sum()),
                    "train_positive": int(labels[train].sum()),
                    "validation_positive": int(labels[validation].sum()),
                    "unseen_station_layer_rows": int(sum(unseen)),
                    "train_cutoff_exclusive": train_end.isoformat(),
                    "validation_start": begin.isoformat(),
                    "validation_end": stop.isoformat(),
                    "train_keys_sha256": key_digest(data.loc[train]),
                    "validation_keys_sha256": key_digest(data.loc[validation]),
                }
            )
    return output, support


def fit_model(name, x, train, frozen, cfg):
    target = train.label.to_numpy(dtype=np.int8)
    if name == "original":
        base = old.load_config(ROOT / frozen["base_config"], env={})
        return old._fit_model(
            "xgboost", base.raw["models"]["xgboost"], cfg["seed"], cfg["threads"], x, target
        )
    lgb_recipe = json.loads((ROOT / frozen["lightgbm_recipe"]).read_text(encoding="utf-8"))
    # Only the recipe's parameter dictionary is consumed; old OOF/answer/threshold fields are not.
    params = old._lgb_parameters(
        {"lightgbm_parameters": lgb_recipe["lightgbm_parameters"]}, cfg["seed"], multiclass=False
    )
    params["n_jobs"] = cfg["threads"]
    model = old.lgb.LGBMClassifier(**params)
    model.fit(x, target, sample_weight=old._event_day_weight(train, target))
    return model


def deterministic_worker(ordinal):
    cfg, frozen, _ = load_contract()
    root_seal = json.loads((OUT / "ATTEMPT_LOCK.json").read_text(encoding="utf-8"))
    if root_seal["runner_sha256"] != old.sha(__file__) or root_seal["config_sha256"] != old.sha(
        CONFIG
    ):
        raise ValueError("worker differs from parent seal")
    destination = OUT / f"determinism_{ordinal}"
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # Canonical all-train feature/encoder/order path, but scheduled CPU2 not historical CPU4.
    data = training_source(cfg, row_id=False)
    stats = old.stats_fit(data)
    source_bundle = old.feature_pair(data, stats, frozen)[0]
    encoder = old.TabularEncoder().fit(source_bundle, np.arange(len(data)))
    x = encoder.transform(source_bundle)
    model = fit_model("original", x, data, frozen, cfg)
    path = destination / "original.joblib"
    joblib.dump({"model": model, "encoder": encoder, "train_stats": stats}, path, compress=3)
    prediction = model.predict_proba(x[:4096])[:, 1]
    reload_prediction = joblib.load(path)["model"].predict_proba(x[:4096])[:, 1]
    if not np.isfinite(prediction).all() or not np.array_equal(prediction, reload_prediction):
        raise ValueError("determinism saved-model replay failure")
    np.save(destination / "probe.npy", prediction, allow_pickle=False)
    result = {
        "status": "COMPLETE",
        "pid": os.getpid(),
        "ordinal": ordinal,
        "fits": 1,
        "model_sha256": old.sha(path),
        "probe_array_sha256": digest(prediction),
        "probe_file_sha256": old.sha(destination / "probe.npy"),
        "matrix_sha256": digest(x),
        "training_keys_sha256": key_digest(data),
        "probe_scope": "in_sample_train_feature_determinism_only_not_quality",
        "quality_metrics_computed": False,
        "used_in_oof": False,
        "rows": len(data),
        "features": x.shape[1],
        "threads": 2,
        "gpu": False,
        "seconds": time.monotonic() - started,
    }
    old.write_json(destination / "receipt.json", result)
    print(
        json.dumps(
            {"determinism_worker": ordinal, "status": "COMPLETE", "seconds": result["seconds"]}
        ),
        flush=True,
    )


def progress(result, stage, started):
    elapsed = time.monotonic() - started
    old.write_json(
        OUT / "progress.json",
        {
            "status": "RUNNING",
            "stage": stage,
            "completed_fits": len(result["fits"]),
            "max_fits": 20,
            "seconds": elapsed,
            "pid": os.getpid(),
            "candidate_metrics_exposed": False,
        },
    )
    print(
        json.dumps({"stage": stage, "completed_fits": len(result["fits"]), "seconds": elapsed}),
        flush=True,
    )
    if elapsed >= 5400:
        raise RuntimeError("90-minute resource cap; no automatic retry")


def stage_fit(data, masks, arm, fold, stage, frozen, cfg, result, started):
    progress(result, f"{arm}/{fold}/{stage}/features", started)
    prepared = partition_inputs(data, *masks, frozen, cfg, bracket=arm == "bracket")
    if len(result["fits"]) >= cfg["max_fits"]:
        raise RuntimeError("fit cap")
    begin = time.monotonic()
    model = fit_model(
        "original" if arm == "original" else "balanced",
        prepared["x"],
        prepared["train"],
        frozen,
        cfg,
    )
    probabilities = model.predict_proba(prepared["xe"])[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("nonfinite predictions")
    path = OUT / f"{fold}_{stage}_{arm}.joblib"
    if path.exists():
        raise FileExistsError("model path exists")
    joblib.dump(
        {
            "model": model,
            "encoder": prepared["encoder"],
            "train_stats": prepared["stats"],
            "bracket": arm == "bracket",
        },
        path,
        compress=3,
    )
    pred_path = OUT / f"{fold}_{stage}_{arm}.npy"
    np.save(pred_path, probabilities, allow_pickle=False)
    result["fits"].append(
        {
            "scope": "historical",
            "fold": fold,
            "stage": stage,
            "arm": arm,
            "rows": len(prepared["train"]),
            "evaluation_rows": len(probabilities),
            "features": prepared["x"].shape[1],
            "seconds": time.monotonic() - begin,
            "model_file": path.name,
            "model_sha256": old.sha(path),
            "probability_file": pred_path.name,
            "probability_sha256": old.sha(pred_path),
            "training_keys_sha256": key_digest(prepared["train"]),
            "evaluation_keys_sha256": key_digest(prepared["validation"]),
        }
    )
    del model, prepared
    gc.collect()
    progress(result, f"{arm}/{fold}/{stage}/complete", started)


def build_oof(data, masks, cfg, frozen, result):
    parts, choices = [], {}
    for fold in ("H2_2024", "H1_2025", "H2_2025"):
        inner_train, inner_val = masks[(fold, "inner")]
        inner = data.loc[inner_val].reset_index(drop=True)
        stats = old.stats_fit(data.loc[inner_train].reset_index(drop=True))
        rule = old.rule_masks(inner, stats)
        inner_prob = {
            name: np.load(OUT / f"{fold}_inner_{name}.npy", allow_pickle=False)
            for name in ("original", "balanced", "bracket")
        }
        ctrl = {"original": inner_prob["original"], "balanced": inner_prob["balanced"]}
        cand = {"original": inner_prob["original"], "balanced": inner_prob["bracket"]}
        selection = {
            "control": policy.select_inner(inner, ctrl, rule, frozen),
            "candidate": policy.select_inner(inner, cand, rule, frozen),
        }
        choices[fold] = selection
        train_mask, val_mask = masks[(fold, "outer")]
        val = data.loc[val_mask].reset_index(drop=True)
        stats = old.stats_fit(data.loc[train_mask].reset_index(drop=True))
        rules = old.rule_masks(val, stats)
        probabilities = {
            name: np.load(OUT / f"{fold}_outer_{name}.npy", allow_pickle=False)
            for name in ("original", "balanced", "bracket")
        }
        part = val[old.KEYS + ["row_id", "label", "anomaly_type"]].copy()
        part["fold"] = fold
        known = set(
            zip(data.loc[train_mask, "station"], data.loc[train_mask, "layer"], strict=True)
        )
        part["known_station_layer"] = [
            key in known for key in zip(val.station, val.layer, strict=True)
        ]
        for arm, balanced in (("control", "balanced"), ("candidate", "bracket")):
            name, calibrations = selection[arm]
            part[arm] = policy.policies(
                val,
                {"original": probabilities["original"], "balanced": probabilities[balanced]},
                rules,
                frozen,
                calibrations,
            )[name]
        parts.append(part)
    result["earlier_inner_selections"] = choices
    return pd.concat(parts, ignore_index=True)


def summarize_oof(oof, eval_contract):
    if oof.duplicated(old.KEYS).any() or oof.duplicated("row_id").any():
        raise ValueError("duplicated OOF keys")
    primary = evaluation.primary_mask(oof, "P1", eval_contract)
    rows = oof.loc[primary]
    grouped = evaluation.bootstrap_groups(rows, "P1", eval_contract)
    bootstrap = evaluation.paired_bootstrap(
        rows.label, rows.control, rows.candidate, grouped, "f1", eval_contract
    )
    metrics = {
        surface: {arm: old.metric(part.label, part[arm]) for arm in ("control", "candidate")}
        for surface, part in (("primary_H1_2025", rows), ("all_forward", oof))
    }
    slices = []
    for columns in (["fold"], ["fold", "station", "layer"], ["fold", "known_station_layer"]):
        for key, part in oof.groupby(columns, sort=True, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            item = {
                name: (value.item() if isinstance(value, np.generic) else value)
                for name, value in zip(columns, key, strict=True)
            }
            item.update(
                {arm: old.metric(part.label, part[arm]) for arm in ("control", "candidate")}
            )
            item["delta_f1"] = item["candidate"]["f1"] - item["control"]["f1"]
            slices.append(item)
    type_rows = []
    for anomaly in ("spike", "noise", "flatline", "offset", "drift"):
        mask = oof.anomaly_type.fillna("").str.split("+").map(lambda tags, name=anomaly: name in tags)
        part = oof.loc[mask & oof.label.eq(1)]
        type_rows.append(
            {
                "type": anomaly,
                "positive_rows": len(part),
                **{
                    arm: {"tp": int(part[arm].sum()), "fn": int(len(part) - part[arm].sum())}
                    for arm in ("control", "candidate")
                },
            }
        )
    ordered = oof.sort_values(["station", "layer", "time"], kind="stable").reset_index(drop=True)
    segments = old.segments(ordered)
    y = ordered.label.eq(1)
    boundary = y & (
        ~y.shift(fill_value=False)
        | ~y.shift(-1, fill_value=False)
        | segments.ne(segments.shift())
        | segments.ne(segments.shift(-1))
    )
    run_rows = {}
    for name, mask in (
        ("positive_boundary", boundary),
        ("positive_interior", y & ~boundary),
        ("normal", ~y),
    ):
        part = ordered.loc[mask]
        run_rows[name] = {
            "rows": len(part),
            **{
                arm: {
                    "predicted_positive": int(part[arm].sum()),
                    "missed_positive": int((part.label.eq(1) & part[arm].eq(0)).sum()),
                    "false_positive": int((part.label.eq(0) & part[arm].eq(1)).sum()),
                }
                for arm in ("control", "candidate")
            },
        }
    return {
        "metrics": metrics,
        "paired_day_bootstrap": bootstrap,
        "slices": slices,
        "anomaly_type_diagnostics": type_rows,
        "run_boundary_and_fp_diagnostics": run_rows,
        "decision": "MEAN_GAIN_CANDIDATE_RETAINED"
        if bootstrap["candidate_retained"]
        else "NO_MEAN_GAIN",
        "official_score_estimate": None,
        "official_score_estimate_reason": "no valid score transfer",
    }


def execute():
    cfg, frozen, eval_contract = load_contract()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": ID,
        "status": "RUNNING",
        "pid": os.getpid(),
        "fits": [],
        "config_sha256": old.sha(CONFIG),
        "runner_sha256": old.sha(__file__),
        "source_pins": cfg["source_pins"],
        "threads": 2,
        "gpu": False,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib")
        },
        "official_rows": 0,
        "hidden_rows": 0,
        "csv_written": 0,
        "uploads": 0,
        "prior_models_or_answers_read": 0,
        "score_inversion_used": False,
        "surface": cfg["surface"],
        "maximum_fits": 20,
        "wall_cap_seconds": 5400,
    }
    with (OUT / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        progress(result, "pre_fit_inner_outer_support", started)
        data = training_source(cfg)
        masks, support = all_split_masks(data, eval_contract, cfg)
        result["support"] = support
        old.write_json(REPORT / "support.json", {"status": "COUNT_SUPPORT_PASS", "folds": support})
        for ordinal in (1, 2):
            progress(result, f"determinism_{ordinal}", started)
            child = subprocess.run(
                [sys.executable, __file__, "--determinism-worker", str(ordinal)],
                cwd=ROOT,
                check=False,
                timeout=max(1, 5400 - (time.monotonic() - started)),
            )
            if child.returncode:
                raise RuntimeError("determinism worker failed; no restart")
            rec = json.loads(
                (OUT / f"determinism_{ordinal}/receipt.json").read_text(encoding="utf-8")
            )
            result["fits"].append({"scope": "full_original_determinism", **rec})
        first, second = result["fits"]
        result["determinism"] = {
            key: first[key] == second[key]
            for key in (
                "model_sha256",
                "probe_array_sha256",
                "matrix_sha256",
                "training_keys_sha256",
            )
        }
        result["determinism"]["distinct_child_pids"] = (
            len({first["pid"], second["pid"], os.getpid()}) == 3
        )
        if not all(result["determinism"].values()):
            raise RuntimeError("same-command full-O determinism failed; no OOF launch")
        progress(result, "support", started)
        source_support = json.loads(
            (ROOT / "reports/ocean_forward_support_20260906_v1/result.json").read_text(
                encoding="utf-8"
            )
        )["P1"]["folds"]
        for expected, actual in zip(
            source_support, [row for row in support if row["stage"] == "outer"], strict=True
        ):
            if (
                expected["validation"],
                expected["train"],
                expected["unseen_station_layer_validation_rows"],
            ) != (
                actual["validation_rows"],
                actual["train_rows"],
                actual["unseen_station_layer_rows"],
            ):
                raise ValueError("source-support parity changed")
        old.write_json(REPORT / "support.json", {"status": "COUNT_SUPPORT_PASS", "folds": support})
        # Seal every comparator OOF before fitting the challenger; no performance-based branch.
        for fold in ("H2_2024", "H1_2025", "H2_2025"):
            for stage in ("inner", "outer"):
                for arm in ("original", "balanced"):
                    stage_fit(
                        data, masks[(fold, stage)], arm, fold, stage, frozen, cfg, result, started
                    )
        result["baseline_all_oof_sealed_before_challenger"] = True
        for fold in ("H2_2024", "H1_2025", "H2_2025"):
            for stage in ("inner", "outer"):
                stage_fit(
                    data, masks[(fold, stage)], "bracket", fold, stage, frozen, cfg, result, started
                )
        oof = build_oof(data, masks, cfg, frozen, result)
        oof.to_parquet(OUT / "oof.parquet", index=False)
        result.update(summarize_oof(oof, eval_contract))
        result["oof_sha256"] = old.sha(OUT / "oof.parquet")
        result["status"] = "COMPLETE_INTERNAL_EVALUATION_QA_PENDING"
    except Exception as exc:
        result.update(status="TERMINAL_TECHNICAL_FAILURE", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["runtime_seconds"] = time.monotonic() - started
        old.write_json(OUT / "terminal_result.json", result)
        old.write_json(REPORT / "result.json", result)


def replay_qa():
    cfg, frozen, eval_contract = load_contract()
    result = json.loads((OUT / "terminal_result.json").read_text(encoding="utf-8"))
    if result["status"] != "COMPLETE_INTERNAL_EVALUATION_QA_PENDING" or len(result["fits"]) != 20:
        raise ValueError("complete 20-fit result required")
    if result["pid"] == os.getpid() or result["runner_sha256"] != old.sha(__file__):
        raise ValueError("fresh process or frozen runner required")
    data = training_source(cfg)
    masks, support = all_split_masks(data, eval_contract, cfg)
    checks = {
        "fresh_process": True,
        "config_hash": result["config_sha256"] == old.sha(CONFIG),
        "support_exact": support == result["support"],
        "oof_hash": old.sha(OUT / "oof.parquet") == result["oof_sha256"],
    }
    for fit in result["fits"]:
        if fit["scope"] != "historical":
            continue
        path = OUT / fit["model_file"]
        key = f"{fit['fold']}_{fit['stage']}_{fit['arm']}"
        checks[key + "_model_hash"] = old.sha(path) == fit["model_sha256"]
        checks[key + "_prediction_hash"] = (
            old.sha(OUT / fit["probability_file"]) == fit["probability_sha256"]
        )
        package = joblib.load(path)
        _, mask = masks[(fit["fold"], fit["stage"])]
        val = data.loc[mask].reset_index(drop=True)
        features = bundle(val, package["train_stats"], frozen, cfg, package["bracket"])
        prediction = package["model"].predict_proba(package["encoder"].transform(features))[:, 1]
        reference = np.load(OUT / fit["probability_file"], allow_pickle=False)
        checks[key + "_probability_exact"] = bool(np.array_equal(prediction, reference))
        del package, features
        gc.collect()
    rebuilt = build_oof(data, masks, cfg, frozen, {})
    saved = pd.read_parquet(OUT / "oof.parquet")
    checks["oof_keys_bits_exact"] = rebuilt.equals(saved)
    own = summarize_oof(rebuilt, eval_contract)
    checks["aggregates_exact"] = all(own[key] == result[key] for key in own)
    qa = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "pid": os.getpid(),
        "training_pid": result["pid"],
        "result_sha256": old.sha(OUT / "terminal_result.json"),
        "new_fits": 0,
        "official_rows": 0,
        "csv_written": 0,
        "uploads": 0,
    }
    old.write_json(REPORT / "independent-qa.json", qa)
    if not all(checks.values()):
        raise ValueError("independent replay failed")
    print(json.dumps({"status": qa["status"], "checks": len(checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--qa", action="store_true")
    actions.add_argument("--determinism-worker", type=int, choices=[1, 2])
    arguments = parser.parse_args()
    if arguments.execute:
        execute()
    elif arguments.qa:
        replay_qa()
    elif arguments.determinism_worker:
        deterministic_worker(arguments.determinism_worker)
    else:
        load_contract()
        print(json.dumps({"status": "CONTRACT_ONLY", "max_fits": 20, "official_rows": 0}))
