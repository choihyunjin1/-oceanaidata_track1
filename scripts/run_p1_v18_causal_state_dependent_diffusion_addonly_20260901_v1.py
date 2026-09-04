"""Exactly-once causal state-dependent drift/diffusion P1 falsification."""

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
EXPERIMENT_ID = "p1_v18_causal_state_dependent_diffusion_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V16_RUNNER = ROOT / "scripts/run_p1_v16_causal_delay_embedding_persistence_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000
_DATA_DIR: Path | None = None


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v18_shared_execution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V16_RUNNER)
core, base = shared.core, shared.base
_ORIGINAL_WRITE = shared._ORIGINAL_WRITE


def _causal_mean(values: np.ndarray, rows: int) -> np.ndarray:
    cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    index = np.arange(len(values), dtype=np.int64)
    start = np.maximum(0, index + 1 - rows)
    return (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)


def conditional_moment_segment(
    values: np.ndarray,
    prefix_rows: int,
    representation: dict[str, Any],
) -> np.ndarray:
    bins = int(representation["state_quantile_bins"])
    minimum = int(representation["minimum_prefix_transitions_per_bin"])
    floor = float(representation["variance_floor"])
    output = np.zeros((len(values), 8), dtype=np.float64)
    if prefix_rows < 2 or len(values) < 2:
        return output
    prefix_rows = min(prefix_rows, len(values))
    state = values[:-1]
    increment = np.diff(values)
    prefix_state = state[: prefix_rows - 1]
    prefix_increment = increment[: prefix_rows - 1]
    edges = np.unique(np.quantile(prefix_state, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    state_bin = np.searchsorted(edges, state, side="right")
    prefix_bin = state_bin[: prefix_rows - 1]
    global_drift = float(np.mean(prefix_increment))
    global_variance = max(float(np.var(prefix_increment)), floor)
    drift = np.full(bins, global_drift, dtype=np.float64)
    variance = np.full(bins, global_variance, dtype=np.float64)
    for index in range(bins):
        selected = prefix_increment[prefix_bin == index]
        if len(selected) < minimum:
            continue
        drift[index] = float(np.mean(selected))
        variance[index] = max(float(np.var(selected)), floor)
    expected = drift[state_bin]
    diffusion_sd = np.sqrt(variance[state_bin])
    standardized = np.clip((increment - expected) / diffusion_sd, -12.0, 12.0)
    energy = standardized * standardized
    row = np.arange(1, len(values))
    output[row, 0] = expected
    output[row, 1] = diffusion_sd
    output[row, 2] = standardized
    output[row, 3] = np.abs(standardized)
    output[row, 4] = energy
    for column, window in enumerate(representation["rolling_rows"], start=5):
        output[row, column] = _causal_mean(energy, int(window))
    return output


def diffusion_features(frame: pd.DataFrame, train_boundary_ns: int, representation: dict[str, Any]) -> np.ndarray:
    """Prefix-fitted state-conditional increment moments with causal gap resets."""

    shared._set_transport_context(frame, train_boundary_ns)
    minimum_prefix = int(representation["minimum_prefix_rows"])
    output = np.zeros((len(frame), 8), dtype=np.float32)
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
        feature = np.zeros((len(ordered), 8), dtype=np.float64)
        for start, end in zip(starts, ends, strict=True):
            prefix_rows = int(np.searchsorted(times[start:end], train_boundary_ns, side="right"))
            if prefix_rows >= minimum_prefix:
                feature[start:end] = conditional_moment_segment(values[start:end], prefix_rows, representation)
        if not np.isfinite(feature).all():
            raise RuntimeError("conditional moment features are nonfinite")
        output[positions] = feature.astype(np.float32)
    return output


def _write_v18(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        if _DATA_DIR is None:
            raise RuntimeError("data directory unavailable for boundary receipt")
        payload = dict(value)
        payload["schema_version"] = json.loads(CONFIG.read_text(encoding="utf-8"))["result_schema_version"]
        payload["long_event_boundary"] = shared.boundary_recall_from_artifacts(_DATA_DIR)
    _ORIGINAL_WRITE(path, payload)


def _configure() -> None:
    shared.CONFIG, shared.ARTIFACT, shared.LOCK = CONFIG, ARTIFACT, LOCK
    core.EXPERIMENT_ID, core.CONFIG, core.ARTIFACT, core.LOCK = EXPERIMENT_ID, CONFIG, ARTIFACT, LOCK
    core.__file__ = str(Path(__file__).resolve())
    core.recurrence_laminar_features = diffusion_features
    core._write = _write_v18
    base._select = shared.shared._select_transport


def preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.preflight(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.qa(data_dir)


def execute(data_dir: Path) -> dict[str, Any]:
    global _DATA_DIR
    _DATA_DIR = data_dir.resolve(strict=True)
    shared._DATA_DIR = _DATA_DIR
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
