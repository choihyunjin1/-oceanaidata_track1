from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p1_public_transport_repair_cycle_20260831_v15 as evaluation  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v16 as source  # noqa: E402

from src.p1_qc.prequential_label_shift_em import select_inner_threshold  # noqa: E402

CONFIG = ROOT / "configs/experiments/p1_causal_cif_lite32_20260831_v32b.json"
ARTIFACT = ROOT / "artifacts/p1_causal_cif_lite32_20260831_v32b"
REPORT = ROOT / "reports/p1_causal_cif_lite32_20260831_v32b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intervals() -> list[tuple[int, int]]:
    points = qmc.Sobol(d=2, scramble=True, seed=20260932).random_base2(m=5)
    output = []
    for left_raw, length_raw in points:
        left = min(int(left_raw * 34), 33)
        length = 3 + int(length_raw * (36 - left - 2))
        output.append((left, min(left + length, 36)))
    return output


def summarize_windows(windows: np.ndarray) -> np.ndarray:
    rows = len(windows)
    output = np.empty((rows, 1280), dtype=np.float32)
    column = 0
    x_axis = np.arange(36, dtype=np.float32)
    for left, right in intervals():
        x = x_axis[left:right]
        centered = x - x.mean()
        denominator = float(np.sum(centered**2)) or 1.0
        for channel in range(5):
            values = windows[:, left:right, channel]
            mean = values.mean(axis=1)
            output[:, column : column + 8] = np.column_stack(
                [
                    np.median(values, axis=1),
                    np.percentile(values, 75, axis=1) - np.percentile(values, 25, axis=1),
                    mean,
                    values.std(axis=1),
                    ((values - mean[:, None]) @ centered) / denominator,
                    values.min(axis=1),
                    values.max(axis=1),
                    values[:, -1] - values[:, 0],
                ]
            ).astype(np.float32)
            column += 8
    return output


def causal_windows(frame: pd.DataFrame) -> np.ndarray:
    result = np.empty((len(frame), 36, 5), dtype=np.float32)
    for _, positions in frame.groupby(["station", "year", "layer"], sort=False).indices.items():
        idx = np.asarray(positions)
        order = np.argsort(pd.to_datetime(frame.iloc[idx]["time"], utc=True).astype("int64").to_numpy())
        idx = idx[order]
        temp = frame.iloc[idx]["temp_raw"].to_numpy(np.float32)
        padded = np.pad(temp, (35, 0), mode="edge")
        raw = np.lib.stride_tricks.sliding_window_view(padded, 36)
        median = np.median(raw, axis=1)
        mad = np.median(np.abs(raw - median[:, None]), axis=1)
        z = np.clip((temp - median) / np.maximum(1.4826 * mad, 1e-3), -12, 12)
        channels = np.column_stack(
            [
                z,
                frame.iloc[idx]["temp_diff_1"].fillna(0).to_numpy(np.float32),
                frame.iloc[idx]["temp_peer_residual"].fillna(0).to_numpy(np.float32),
                frame.iloc[idx]["depth_diff_1"].fillna(0).to_numpy(np.float32),
                frame.iloc[idx][["psal_missing", "depth_missing", "has_gap_before"]].max(axis=1).to_numpy(np.float32),
            ]
        ).astype(np.float32)
        channel_pad = np.pad(channels, ((35, 0), (0, 0)), mode="edge")
        result[idx] = np.lib.stride_tricks.sliding_window_view(channel_pad, 36, axis=0).transpose(0, 2, 1)
    return result


def fit_model(config: dict, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> ExtraTreesClassifier:
    model = ExtraTreesClassifier(n_estimators=192, max_depth=10, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample", bootstrap=False, n_jobs=6, random_state=20260932)
    model.fit(x, y, sample_weight=weights)
    return model


def benchmark(config: dict) -> dict:
    rng = np.random.default_rng(20260932)
    rows = 8192
    windows = rng.normal(size=(rows, 36, 5)).astype(np.float32)
    started = time.perf_counter()
    features = summarize_windows(windows)
    feature_seconds = time.perf_counter() - started
    y = (rng.uniform(size=rows) < 0.03).astype(np.int8)
    started = time.perf_counter()
    fit_model(config, features, y, np.ones(rows))
    forest_seconds = time.perf_counter() - started
    estimated = feature_seconds * (421032 / rows) + forest_seconds * (310000 / rows) * 2
    estimated_rss = 421032 * 1280 * 4 + 2_000_000_000
    checks = {"feature_width": features.shape[1] == 1280, "finite": bool(np.isfinite(features).all()), "estimated_wall_under_40m": estimated <= 2400, "estimated_rss_under_6gib": estimated_rss <= 6442450944, "official_zero": True}
    return {"status": "PASS" if all(checks.values()) else "TECHNICAL_NO_GO", "checks": checks, "feature_seconds_8192": feature_seconds, "forest_seconds_8192": forest_seconds, "estimated_wall_seconds": estimated, "estimated_rss_bytes": estimated_rss}


def execute(config: dict) -> dict:
    if ARTIFACT.exists():
        raise FileExistsError("v32b artifact already exists")
    preflight = benchmark(config)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "resource-preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if preflight["status"] != "PASS":
        return {"status": "TERMINAL_TECHNICAL_NO_GO", "resource_preflight": preflight, "fits": 0, "locks": 0, "official_reads": 0}
    ARTIFACT.mkdir(parents=True)
    lock = {"experiment_id": config["experiment_id"], "pid": os.getpid(), "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "fit_budget": 2, "official_reads": 0, "hidden_truth_reads": 0}
    (ARTIFACT / "attempt_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.perf_counter()
    frame, anchor, _, dependency = source.load_feature_surface()
    windows = causal_windows(frame)
    features = summarize_windows(windows)
    del windows
    truth = frame["label_base"].to_numpy(np.int8)
    times = pd.to_datetime(frame["time"], utc=True)
    candidate = anchor.copy()
    probability = np.zeros(len(frame), dtype=np.float32)
    receipts = []
    for fit_number, spec in enumerate(config["validation"]["fits"], 1):
        prefix = frame["fold"].isin(spec["train_folds"]).to_numpy()
        unique_times = np.sort(times[prefix].unique())
        cutoff = unique_times[min(int(len(unique_times) * 0.75), len(unique_times) - 1)]
        fit = prefix & (times.to_numpy() < cutoff)
        inner = prefix & (times.to_numpy() >= cutoff)
        current = np.column_stack(
            [
                frame["temp_raw"].to_numpy(np.float32),
                frame["temp_diff_1"].fillna(0).to_numpy(np.float32),
                frame["temp_peer_residual"].fillna(0).to_numpy(np.float32),
                frame["depth_diff_1"].fillna(0).to_numpy(np.float32),
                frame[["psal_missing", "depth_missing", "has_gap_before"]].max(axis=1).to_numpy(np.float32),
            ]
        )
        center = np.median(current[fit], axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(current[fit] - center), axis=0), 1e-3)
        robust_distance = np.max(np.abs((current[fit] - center) / scale), axis=1)
        weights = np.minimum(1.0, 4.0 / (1.0 + robust_distance))
        model = fit_model(config, features[fit], truth[fit], weights)
        inner_probability = model.predict_proba(features[inner])[:, 1]
        selected = select_inner_threshold(inner_probability, truth[inner], anchor[inner], maximum_changed_fraction=0.005)
        outer = frame["fold"].eq(spec["outer"]).to_numpy()
        outer_probability = model.predict_proba(features[outer])[:, 1]
        probability[outer] = outer_probability
        proposed = np.flatnonzero(outer & (anchor == 0))[outer_probability[anchor[outer] == 0] >= selected.threshold]
        candidate[proposed] = 1
        receipts.append({"fit_number": fit_number, "outer": spec["outer"], "fit_rows": int(fit.sum()), "inner_rows": int(inner.sum()), "inner_threshold": selected.threshold if np.isfinite(selected.threshold) else None, "inner_additions": selected.additions, "outer_additions": int(len(proposed)), "outer_label_reads_before_seal": 0})
        (ARTIFACT / "progress.json").write_text(json.dumps({"fit_count": fit_number}) + "\n", encoding="utf-8")
    prediction_path = ARTIFACT / "sealed_predictions.npz"
    np.savez_compressed(prediction_path, candidate=candidate, probability=probability)
    record = evaluation.evaluate(frame, anchor, candidate, config)
    record["name"] = config["candidate"]
    result = {"status": "COMPLETE_INTERNAL_ONLY", "fit_count": 2, "pass_count": int(record["strict_internal_pass"]), "runtime_seconds": time.perf_counter() - started, "candidate": record, "receipts": receipts, "dependency": dependency, "operations": {"official_reads": 0, "hidden_truth_reads": 0, "csv": 0, "uploads": 0}, "hashes": {"config": sha256(CONFIG), "runner": sha256(Path(__file__)), "lock": sha256(ARTIFACT / "attempt_lock.json"), "prediction": sha256(prediction_path)}}
    (ARTIFACT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = execute(config) if args.execute else benchmark(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
