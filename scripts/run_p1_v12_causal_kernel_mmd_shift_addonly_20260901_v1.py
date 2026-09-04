"""Exactly-once causal recent-vs-prefix kernel-MMD P1 falsification."""

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
EXPERIMENT_ID = "p1_v12_causal_kernel_mmd_shift_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V10_RUNNER = ROOT / "scripts/run_p1_v10_causal_recurrence_laminar_state_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v12_shared_execution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


core = _module(V10_RUNNER)
base = core.base
_ORIGINAL_WRITE = core._write


def _causal_matrix_mean(values: np.ndarray, rows: int) -> np.ndarray:
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1])), np.cumsum(values, axis=0, dtype=np.float64)]
    )
    index = np.arange(len(values), dtype=np.int64)
    start = np.maximum(0, index + 1 - rows)
    count = (index + 1 - start)[:, None]
    return (cumulative[index + 1] - cumulative[start]) / count


def kernel_mmd_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    representation: dict[str, Any],
) -> np.ndarray:
    """Causal RBF landmark mean-embedding displacement by station-layer."""

    quantiles = representation["prefix_landmark_quantiles"]
    bandwidths = tuple(float(value) for value in representation["rbf_bandwidths_prefix_scale"])
    recent_rows = int(representation["recent_window_rows"])
    minimum_prefix = int(representation["minimum_prefix_rows"])
    output = np.zeros((len(frame), len(bandwidths) * 3), dtype=np.float32)
    for _key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = core._time_ns(ordered["_time"])
        raw = ordered["temp"].to_numpy(np.float64)
        prefix_mask = (times <= train_boundary_ns) & np.isfinite(raw)
        prefix = raw[prefix_mask]
        if len(prefix) < minimum_prefix:
            continue
        center = float(np.median(prefix))
        scale = float(1.4826 * np.median(np.abs(prefix - center)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(prefix))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        normalized = np.clip(
            np.nan_to_num((raw - center) / scale, nan=0.0, posinf=12.0, neginf=-12.0),
            -12.0,
            12.0,
        )
        landmarks = np.quantile(normalized[prefix_mask], quantiles)
        gap = np.r_[True, np.diff(times) != CADENCE_NS]
        starts = np.flatnonzero(gap)
        ends = np.r_[starts[1:], len(ordered)]
        feature = np.zeros((len(ordered), len(bandwidths) * 3), dtype=np.float64)
        for bandwidth_index, bandwidth in enumerate(bandwidths):
            kernels = np.exp(
                -0.5
                * ((normalized[:, None] - landmarks[None, :]) / bandwidth) ** 2
            )
            reference = kernels[prefix_mask].mean(axis=0)
            for start, end in zip(starts, ends, strict=True):
                recent = _causal_matrix_mean(kernels[start:end], recent_rows)
                displacement = recent - reference[None, :]
                column = bandwidth_index * 3
                feature[start:end, column] = np.sqrt(np.mean(displacement**2, axis=1))
                feature[start:end, column + 1] = np.mean(np.abs(displacement), axis=1)
                feature[start:end, column + 2] = np.max(np.abs(displacement), axis=1)
        if not np.isfinite(feature).all():
            raise RuntimeError("kernel MMD features are nonfinite")
        output[positions] = feature.astype(np.float32)
    return output


def _write_v12(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        payload = dict(value)
        payload["schema_version"] = json.loads(CONFIG.read_text(encoding="utf-8"))[
            "result_schema_version"
        ]
    _ORIGINAL_WRITE(path, payload)


def _configure() -> None:
    core.EXPERIMENT_ID = EXPERIMENT_ID
    core.CONFIG = CONFIG
    core.ARTIFACT = ARTIFACT
    core.LOCK = LOCK
    core.__file__ = str(Path(__file__).resolve())
    core.recurrence_laminar_features = kernel_mmd_features
    core._write = _write_v12


def preflight(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.preflight(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _configure()
    return core.qa(data_dir)


def execute(data_dir: Path) -> dict[str, Any]:
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
