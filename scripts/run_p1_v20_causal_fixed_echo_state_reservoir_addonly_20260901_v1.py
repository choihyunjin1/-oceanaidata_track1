"""Exactly-once fixed echo-state reservoir P1 falsification."""

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
EXPERIMENT_ID = "p1_v20_causal_fixed_echo_state_reservoir_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V16 = ROOT / "scripts/run_p1_v16_causal_delay_embedding_persistence_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000
_DATA_DIR: Path | None = None


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v20_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


shared = _module(V16)
core, base = shared.core, shared.base
_ORIGINAL_WRITE = shared._ORIGINAL_WRITE


def reservoir_matrices(rep: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    units = int(rep["reservoir_units"])
    rng = np.random.default_rng(int(rep["weight_seed"]))
    raw = rng.normal(size=(units, units))
    norm = float(np.linalg.norm(raw, ord=2))
    recurrent = raw * (float(rep["spectral_norm_bound"]) / norm)
    inputs = rng.normal(scale=float(rep["input_scale"]), size=(units, 2))
    bias = rng.normal(scale=float(rep["bias_scale"]), size=units)
    return recurrent, inputs, bias


def reservoir_segment(values: np.ndarray, rep: dict[str, Any]) -> np.ndarray:
    recurrent, inputs, bias = reservoir_matrices(rep)
    leak = float(rep["leak_rate"])
    state = np.zeros(len(recurrent))
    output = np.zeros((len(values), len(recurrent)), dtype=np.float64)
    previous = values[0] if len(values) else 0.0
    for row, value in enumerate(values):
        vector = np.array([value, value - previous])
        proposal = np.tanh(recurrent @ state + inputs @ vector + bias)
        state = (1.0 - leak) * state + leak * proposal
        output[row] = state
        previous = value
    return output


def reservoir_features(
    frame: pd.DataFrame, train_boundary_ns: int, representation: dict[str, Any]
) -> np.ndarray:
    shared._set_transport_context(frame, train_boundary_ns)
    units = int(representation["reservoir_units"])
    output = np.zeros((len(frame), units), dtype=np.float32)
    for _key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = core._time_ns(ordered["_time"])
        raw = ordered["temp"].to_numpy(np.float64)
        prefix = raw[(times <= train_boundary_ns) & np.isfinite(raw)]
        if len(prefix) < int(representation["minimum_prefix_rows"]):
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
        feature = np.zeros((len(ordered), units))
        for start, end in zip(starts, ends, strict=True):
            feature[start:end] = reservoir_segment(values[start:end], representation)
        if not np.isfinite(feature).all():
            raise RuntimeError("reservoir features nonfinite")
        output[positions] = feature.astype(np.float32)
    return output


def _write_v20(path: Path, value: dict[str, Any]) -> None:
    payload = value
    if path.name == "result.json":
        if _DATA_DIR is None:
            raise RuntimeError("data directory unavailable")
        payload = dict(value)
        payload["schema_version"] = json.loads(CONFIG.read_text(encoding="utf-8"))[
            "result_schema_version"
        ]
        payload["long_event_boundary"] = shared.boundary_recall_from_artifacts(_DATA_DIR)
    _ORIGINAL_WRITE(path, payload)


def _configure() -> None:
    shared.CONFIG, shared.ARTIFACT, shared.LOCK = CONFIG, ARTIFACT, LOCK
    core.EXPERIMENT_ID, core.CONFIG, core.ARTIFACT, core.LOCK = (
        EXPERIMENT_ID,
        CONFIG,
        ARTIFACT,
        LOCK,
    )
    core.__file__ = str(Path(__file__).resolve())
    core.recurrence_laminar_features = reservoir_features
    core._write = _write_v20
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
    value = (
        preflight(args.data_dir)
        if args.preflight
        else qa(args.data_dir)
        if args.qa
        else execute(args.data_dir)
    )
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
