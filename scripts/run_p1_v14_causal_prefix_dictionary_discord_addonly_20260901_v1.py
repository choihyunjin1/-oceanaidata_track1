"""Exactly-once causal non-normalized prefix-dictionary discord falsification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v14_causal_prefix_dictionary_discord_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V13_RUNNER = ROOT / "scripts/run_p1_v13_causal_endpoint_visibility_topology_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000
_DATA_DIR: Path | None = None


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v14_shared_execution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V13_RUNNER)
core, base = shared.core, shared.base
_ORIGINAL_WRITE = shared._ORIGINAL_WRITE


def _set_transport_context(frame: pd.DataFrame, train_boundary_ns: int) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    all_ns = core._time_ns(frame["_time"])
    shared._CURRENT_ENVIRONMENTS = None
    for item in config["parts"].values():
        audit = json.loads((ROOT / item["audit"]).read_text(encoding="utf-8"))
        cutoff = pd.Timestamp(audit["adjusted_cutoff_utc"]).value
        prefix_times = np.sort(np.unique(all_ns[all_ns <= cutoff]))
        boundary = int(prefix_times[max(0, int(len(prefix_times) * config["selection"]["inner_train_fraction"]) - 1)])
        if boundary != train_boundary_ns:
            continue
        inner = (all_ns > boundary) & (all_ns <= cutoff)
        inner_times = all_ns[inner]
        ordered_times = np.sort(np.unique(inner_times))
        half_cutoff = ordered_times[max(0, len(ordered_times) // 2 - 1)]
        shared._CURRENT_ENVIRONMENTS = {"station": frame.loc[inner, "station"].astype(str).to_numpy(), "layer": frame.loc[inner, "layer"].astype(str).to_numpy(), "half": (inner_times > half_cutoff).astype(np.int8)}
        return


def _profile_segment(values: np.ndarray, window: int, lag_blocks: tuple[int, ...]) -> np.ndarray:
    feature = np.zeros((len(values), 5), dtype=np.float64)
    if len(values) < window:
        return feature
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    distance = np.full((len(windows), len(lag_blocks)), np.inf, dtype=np.float64)
    derivative = np.full_like(distance, np.inf)
    window_diff = np.diff(windows, axis=1)
    for column, block in enumerate(lag_blocks):
        lag = block * window
        if lag >= len(windows):
            continue
        delta = windows[lag:] - windows[:-lag]
        distance[lag:, column] = np.sqrt(np.mean(delta * delta, axis=1))
        delta_diff = window_diff[lag:] - window_diff[:-lag]
        derivative[lag:, column] = np.sqrt(np.mean(delta_diff * delta_diff, axis=1))
    supported = np.isfinite(distance).sum(axis=1) >= 2
    if not supported.any():
        return feature
    valid = distance[supported]
    nearest = np.min(valid, axis=1)
    second = np.partition(valid, 1, axis=1)[:, 1]
    finite = np.where(np.isfinite(valid), valid, np.nan)
    median = np.nanmedian(finite, axis=1)
    nearest_derivative = np.min(derivative[supported], axis=1)
    row = np.flatnonzero(supported) + window - 1
    feature[row] = np.column_stack([nearest, second, median, nearest_derivative, nearest / (median + 1e-9)])
    return feature


def discord_features(frame: pd.DataFrame, train_boundary_ns: int, representation: dict[str, Any]) -> np.ndarray:
    """Causal past-dictionary subsequence distances with prefix-global scale."""

    _set_transport_context(frame, train_boundary_ns)
    windows = tuple(int(value) for value in representation["window_rows"])
    lag_blocks = tuple(int(value) for value in representation["past_dictionary_lag_blocks"])
    minimum_prefix = int(representation["minimum_prefix_rows"])
    output = np.zeros((len(frame), len(windows) * 5), dtype=np.float32)
    for _key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = core._time_ns(ordered["_time"])
        raw = ordered["temp"].to_numpy(np.float64)
        prefix = raw[(times <= train_boundary_ns) & np.isfinite(raw)]
        if len(prefix) < minimum_prefix:
            continue
        center = float(np.median(prefix))
        scale = float(1.4826 * np.median(np.abs(prefix - center)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(prefix))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        values = np.clip(np.nan_to_num((raw - center) / scale, nan=0.0), -12.0, 12.0)
        starts = np.flatnonzero(np.r_[True, np.diff(times) != CADENCE_NS])
        ends = np.r_[starts[1:], len(ordered)]
        group_feature = np.zeros((len(ordered), len(windows) * 5), dtype=np.float64)
        for start, end in zip(starts, ends, strict=True):
            for scale_index, window in enumerate(windows):
                group_feature[start:end, scale_index * 5 : (scale_index + 1) * 5] = _profile_segment(values[start:end], window, lag_blocks)
        if not np.isfinite(group_feature).all():
            raise RuntimeError("discord features are nonfinite")
        output[positions] = group_feature.astype(np.float32)
    return output


def _boundary_recall(truth: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    work = metadata.reset_index(drop=True).copy()
    work["truth"], work["incumbent"], work["candidate"] = truth, incumbent, candidate
    boundary_indices: list[int] = []
    runs = 0
    for _key, group in work.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("time", kind="stable")
        times = core._time_ns(pd.to_datetime(ordered["time"], utc=True))
        labels = ordered["truth"].to_numpy(bool)
        indices = ordered.index.to_numpy(np.int64)
        start = 0
        while start < len(ordered):
            if not labels[start]:
                start += 1
                continue
            stop = start + 1
            while stop < len(ordered) and labels[stop] and times[stop] - times[stop - 1] == CADENCE_NS:
                stop += 1
            if stop - start >= 18:
                runs += 1
                boundary_indices.extend(indices[start : min(stop, start + 6)])
                boundary_indices.extend(indices[max(start, stop - 6) : stop])
            start = stop
    selected = np.unique(boundary_indices)
    return {"definition": "positive 10-minute runs >=18 rows; first and last 6 rows", "runs": runs, "boundary_rows": int(len(selected)), "anchor_recall": float(incumbent[selected].mean()) if len(selected) else 0.0, "candidate_recall": float(candidate[selected].mean()) if len(selected) else 0.0, "delta_recall": float(candidate[selected].mean() - incumbent[selected].mean()) if len(selected) else 0.0}


def boundary_recall_from_artifacts(data_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(data_dir / "train.csv", usecols=["station", "layer", "time", "label"])
    truth_parts, incumbent_parts, candidate_parts, metadata_parts = [], [], [], []
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for fold in config["parts"]:
        seal = json.loads((ARTIFACT / f"{fold}_seal.json").read_text(encoding="utf-8"))
        with np.load(ROOT / seal["path"], allow_pickle=False) as values:
            positions = values["positions"]
            incumbent_parts.append(values["incumbent"])
            candidate_parts.append(values["candidate"])
        truth_parts.append(frame.iloc[positions]["label"].to_numpy(np.int8))
        metadata_parts.append(frame.iloc[positions][["station", "layer", "time"]].reset_index(drop=True))
    return _boundary_recall(np.concatenate(truth_parts), np.concatenate(incumbent_parts), np.concatenate(candidate_parts), pd.concat(metadata_parts, ignore_index=True))


def _write_v14(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        if _DATA_DIR is None:
            raise RuntimeError("data directory unavailable for boundary receipt")
        payload = dict(value)
        payload["schema_version"] = json.loads(CONFIG.read_text(encoding="utf-8"))["result_schema_version"]
        payload["long_event_boundary"] = boundary_recall_from_artifacts(_DATA_DIR)
    _ORIGINAL_WRITE(path, payload)


def _configure() -> None:
    core.EXPERIMENT_ID, core.CONFIG, core.ARTIFACT, core.LOCK = EXPERIMENT_ID, CONFIG, ARTIFACT, LOCK
    core.__file__ = str(Path(__file__).resolve())
    core.recurrence_laminar_features = discord_features
    core._write = _write_v14
    base._select = shared._select_transport


def preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.preflight(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.qa(data_dir)


def execute(data_dir: Path) -> dict[str, Any]:
    global _DATA_DIR
    _DATA_DIR = data_dir.resolve(strict=True)
    _configure()
    return core.execute(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qa", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = preflight(args.data_dir) if args.preflight else qa(args.data_dir) if args.qa else execute(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
