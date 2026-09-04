"""Exactly-once causal path cross-moment add-only P1 falsification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v7_causal_path_crossmoment_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SCORER_PATH = ROOT / "scripts/run_p1_clean_state_capa_falsification_20260831_v1.py"
PART_COLUMNS = ("station", "year", "layer", "time", "row_position", "baseline_prediction")
KEY_COLUMNS = ("station", "year", "layer", "time")
POINTS_PER_F1 = 0.6778 / 0.0255
TRANSPORT_FACTOR = 0.30


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module load failed")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


scorer = _module(SCORER_PATH, "p1_v7_score_helpers")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode() + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _time_ns(values: pd.Series) -> np.ndarray:
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.hasnans:
        raise RuntimeError("time must be complete and timezone-aware")
    result = index.as_unit("ns").asi8
    if len(result) and (result.min() < pd.Timestamp("2020-01-01T00:00:00Z").value or result.max() >= pd.Timestamp("2030-01-01T00:00:00Z").value):
        raise RuntimeError("time unit is not authorized epoch nanoseconds")
    return result


def preflight(train_csv: Path) -> dict[str, Any]:
    if ARTIFACT.exists() or LOCK.exists():
        raise FileExistsError("namespace consumed")
    config = _read(CONFIG)
    train = train_csv.resolve(strict=True)
    if train.name != "train.csv" or _sha(train) != config["source"]["train_sha256"]:
        raise RuntimeError("train binding invalid")
    for relative, expected in config["semantic_audit"]["evidence"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"semantic evidence drifted: {relative}")
    if config["semantic_audit"]["decision"] != "NOVEL_EXPLORATORY_PROCEED_ONCE":
        raise RuntimeError("semantic gate closed")
    parsed = pd.to_datetime(pd.read_csv(train, usecols=["time"])["time"], utc=True, errors="raise", format="mixed")
    all_ns = _time_ns(parsed)
    parts, boundaries = {}, []
    for fold, item in config["parts"].items():
        path, audit = ROOT / item["path"], _read(ROOT / item["audit"])
        if _sha(path) != item["sha256"] or audit["target_fold_validation_labels_read_before_prediction"] != 0:
            raise RuntimeError("champion part binding invalid")
        cutoff_ns = pd.Timestamp(audit["adjusted_cutoff_utc"]).value
        prefix = np.sort(np.unique(all_ns[all_ns <= cutoff_ns]))
        boundary = int(prefix[max(0, int(len(prefix) * config["selection"]["inner_train_fraction"]) - 1)])
        boundaries.append(boundary)
        parts[fold] = {"cutoff": audit["adjusted_cutoff_utc"], "boundary": pd.Timestamp(boundary, tz="UTC").isoformat()}
    if len(set(boundaries)) != 3:
        raise RuntimeError("cutoff-specific boundaries failed")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_ZERO_OPERATION",
        "surface": config["surface"],
        "train": str(train),
        "config_sha256": _sha(CONFIG),
        "runner_sha256": _sha(Path(__file__)),
        "parts": parts,
        "semantic_audit": config["semantic_audit"],
        "counters": {"fits": 0, "targets": 0, "official": 0, "csv": 0, "uploads": 0},
    }


def path_features(frame: pd.DataFrame, windows: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    output: list[pd.DataFrame] = []
    names: list[str] | None = None
    for _key, group in frame.groupby(["station", "layer"], sort=False, observed=True):
        ordered = group.sort_values("_time", kind="stable")
        temp = ordered["temp"].astype(float)
        psal_missing = ordered["psal"].isna().astype(float)
        depth_missing = ordered["depth"].isna().astype(float)
        psal = ordered["psal"].astype(float).ffill().fillna(0.0)
        depth = ordered["depth"].astype(float).ffill().fillna(0.0)
        channels = {"temp": temp, "psal": psal, "depth": depth}
        feature = pd.DataFrame(index=ordered.index)
        for window in windows:
            for channel, values in channels.items():
                difference = values.diff().fillna(0.0)
                feature[f"{channel}_increment_{window}"] = values - values.shift(window)
                feature[f"{channel}_variation_{window}"] = difference.abs().rolling(window, min_periods=1).sum()
                feature[f"{channel}_quadratic_{window}"] = difference.pow(2).rolling(window, min_periods=1).sum()
            dtemp, dpsal = temp.diff().fillna(0.0), psal.diff().fillna(0.0)
            area_step = temp.shift(1).fillna(temp.iloc[0]) * dpsal - psal.shift(1).fillna(psal.iloc[0]) * dtemp
            feature[f"temp_psal_signed_area_{window}"] = area_step.rolling(window, min_periods=1).sum()
            feature[f"psal_support_{window}"] = (1.0 - psal_missing).rolling(window, min_periods=1).mean()
            feature[f"depth_support_{window}"] = (1.0 - depth_missing).rolling(window, min_periods=1).mean()
        feature = feature.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if names is None:
            names = list(feature.columns)
        feature["_position"] = ordered.index
        output.append(feature)
    combined = pd.concat(output).sort_values("_position", kind="stable")
    return combined.drop(columns="_position").to_numpy(np.float32), names or []


def _wilson_lower(successes: int, count: int, z: float = 1.6448536269514722) -> float:
    if count == 0:
        return 0.0
    rate = successes / count
    denominator = 1 + z * z / count
    center = rate + z * z / (2 * count)
    margin = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
    return (center - margin) / denominator


def _select(scores: np.ndarray, labels: np.ndarray, selection: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for quantile in selection["threshold_quantiles"]:
        threshold = float(np.quantile(scores, quantile))
        use = scores >= threshold
        count, true = int(use.sum()), int(labels[use].sum())
        candidates.append({"quantile": quantile, "threshold": threshold, "count": count, "share": count / len(scores), "precision": true / count if count else 0.0, "precision_lcb": _wilson_lower(true, count)})
    eligible = [item for item in candidates if item["count"] >= selection["minimum_additions"] and item["share"] <= selection["maximum_addition_share"] and item["precision_lcb"] >= selection["wilson90_lcb_minimum"]]
    return {"chosen": eligible[0] if eligible else None, "candidates": candidates}


def _model_hash(model: SGDClassifier, scaler: StandardScaler) -> str:
    digest = hashlib.sha256()
    for value in (model.coef_, model.intercept_, scaler.mean_, scaler.scale_):
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _additions(scores: np.ndarray, incumbent: np.ndarray, chosen: dict[str, Any] | None, share: float) -> np.ndarray:
    result = np.zeros(len(scores), dtype=bool)
    if chosen is None:
        return result
    eligible = np.flatnonzero((incumbent == 0) & (scores >= chosen["threshold"]))
    maximum = int(math.floor(len(scores) * share))
    if len(eligible) > maximum:
        eligible = eligible[np.lexsort((eligible, -scores[eligible]))[:maximum]]
    result[eligible] = True
    return result


def execute(train_csv: Path) -> dict[str, Any]:
    started = time.monotonic()
    ready, config = preflight(train_csv), _read(CONFIG)
    _write(LOCK, {"experiment_id": EXPERIMENT_ID, "status": "CONSUMED_EXACTLY_ONCE", "config_sha256": ready["config_sha256"], "runner_sha256": ready["runner_sha256"]})
    ARTIFACT.mkdir(exist_ok=False)
    _write(ARTIFACT / "preflight.json", ready)
    frame = pd.read_csv(ready["train"], usecols=["station", "year", "layer", "time", "temp", "psal", "depth", "label", "anomaly_type"])
    frame["_time"] = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    times_ns = _time_ns(frame["_time"])
    labels = frame["label"].to_numpy(np.int8)
    features, feature_names = path_features(frame, tuple(config["representation"]["windows_rows"]))
    seals, fit_count = [], 0
    for fold, part_config in config["parts"].items():
        cutoff_ns = pd.Timestamp(ready["parts"][fold]["cutoff"]).value
        boundary_ns = pd.Timestamp(ready["parts"][fold]["boundary"]).value
        train_mask = times_ns <= boundary_ns
        inner_mask = (times_ns > boundary_ns) & (times_ns <= cutoff_ns)
        probabilities, model_hashes = [], []
        for seed in config["model"]["seeds"]:
            scaler = StandardScaler().fit(features[train_mask])
            model = SGDClassifier(loss="log_loss", penalty="l2", alpha=config["model"]["alpha"], max_iter=config["model"]["epochs"], tol=None, class_weight={0: 1.0, 1: config["model"]["positive_class_weight"]}, shuffle=True, random_state=seed)
            model.fit(scaler.transform(features[train_mask]), labels[train_mask])
            probabilities.append(model.predict_proba(scaler.transform(features))[:, 1].astype(np.float32))
            model_hashes.append(_model_hash(model, scaler))
            fit_count += 1
        scores = np.mean(np.stack(probabilities), axis=0)
        selection = _select(scores[inner_mask], labels[inner_mask], config["selection"])
        part = pd.read_parquet(ROOT / part_config["path"], columns=list(PART_COLUMNS))
        positions = part["row_position"].to_numpy(np.int64)
        incumbent = part["baseline_prediction"].to_numpy(np.int8)
        outer_scores = scores[positions]
        additions = _additions(outer_scores, incumbent, selection["chosen"], config["selection"]["maximum_addition_share"])
        candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
        path = ARTIFACT / f"{fold}_sealed.npz"
        np.savez_compressed(path, positions=positions, incumbent=incumbent, additions=additions, candidate=candidate, scores=outer_scores)
        seal = {"fold": fold, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha(path), "selection": selection, "model_hashes": model_hashes, "fits": 3, "outer_target_reads_before_seal": 0}
        _write(ARTIFACT / f"{fold}_seal.json", seal)
        seals.append(seal)
    _write(ARTIFACT / "predictions_complete.json", {"experiment_id": EXPERIMENT_ID, "fits": fit_count, "feature_names": feature_names, "seals": seals, "outer_target_reads_before_all_seals": 0})
    fold_scores, pool = [], {key: [] for key in ("truth", "incumbent", "candidate", "additions", "types", "metadata")}
    for seal in seals:
        with np.load(ROOT / seal["path"], allow_pickle=False) as values:
            positions, incumbent, additions, candidate = values["positions"], values["incumbent"], values["additions"], values["candidate"]
        truth = labels[positions]
        metadata = frame.iloc[positions].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
        types = frame.iloc[positions]["anomaly_type"].reset_index(drop=True)
        fold_scores.append({"fold": seal["fold"], **scorer._score_surface(truth, incumbent, candidate, additions, types, metadata)})
        for key, value in (("truth", truth), ("incumbent", incumbent), ("candidate", candidate), ("additions", additions), ("types", types), ("metadata", metadata)):
            pool[key].append(value)
    truth, incumbent, candidate, additions = (np.concatenate(pool[key]) for key in ("truth", "incumbent", "candidate", "additions"))
    types, metadata = pd.concat(pool["types"], ignore_index=True), pd.concat(pool["metadata"], ignore_index=True)
    pooled = scorer._score_surface(truth, incumbent, candidate, additions, types, metadata)
    bootstrap = scorer._paired_cluster_bootstrap(truth, incumbent, candidate, metadata, replicates=config["decision"]["bootstrap_replicates"], seed=config["decision"]["seed"])
    passed = pooled["delta_f1"] > 0 and bootstrap["ci90"][0] >= 0 and all(item["delta_f1"] >= 0 for item in fold_scores)
    result = {"experiment_id": EXPERIMENT_ID, "surface": config["surface"], "decision": config["decision"]["pass"] if passed else config["decision"]["fail"], "pooled": pooled, "fold_scores": fold_scores, "block_bootstrap": bootstrap, "worst_slices": sorted(pooled["station_layer_diagnostics"], key=lambda item: item["delta_f1"])[:5], "points": {"nominal": pooled["delta_f1"] * POINTS_PER_F1, "transport_adjusted": pooled["delta_f1"] * POINTS_PER_F1 * TRANSPORT_FACTOR}, "counters": {"fits": fit_count, "anchor_removals": pooled["incumbent_positive_removals"], "official": 0, "csv": 0, "uploads": 0}, "runtime_seconds": time.monotonic() - started, "hashes": {"config": ready["config_sha256"], "runner": ready["runner_sha256"], "completion": _sha(ARTIFACT / "predictions_complete.json"), "lock": _sha(LOCK)}}
    _write(ARTIFACT / "result.json", result)
    return result


def qa(train_csv: Path) -> dict[str, Any]:
    ready, config = preflight(train_csv), _read(CONFIG)
    checks = {"zero": all(value == 0 for value in ready["counters"].values()), "novel": config["semantic_audit"]["decision"] == "NOVEL_EXPLORATORY_PROCEED_ONCE", "past_only": config["representation"]["past_only"], "max9": config["model"]["fits"] == 9, "no_sweep": config["model"]["sweep"] == 0, "outer_tuning_zero": config["selection"]["outer_tuning"] == 0, "add_only": config["anchor"]["removals"] == 0, "access0": config["source"]["official_test_sample_submission_hidden"] == 0}
    return {"experiment_id": EXPERIMENT_ID, "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = preflight(args.train_csv) if args.preflight else qa(args.train_csv) if args.qa else execute(args.train_csv)
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False), end="")


if __name__ == "__main__":
    main()
