"""Exactly-once causal rolling 0-1-chaos-coordinate P1 audit."""

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
EXPERIMENT_ID = "p1_v40_causal_zero_one_chaos_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V34 = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v40_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


engine = _module(V34)
base = engine.base
LinearProbeClassifier = engine.shared.LinearProbeClassifier


def zero_one_coordinates(values: np.ndarray, representation: dict[str, Any]) -> np.ndarray:
    """Rolling translation-variable displacement growth for one contiguous segment."""

    values = np.asarray(values, dtype=np.float64)
    window = int(representation["rolling_rows"])
    lags = np.asarray(representation["displacement_lags"], dtype=np.int64)
    frequencies = [float(value) for value in representation["angular_frequencies"]]
    output = np.zeros((len(values), 2 * len(frequencies) + 1), dtype=np.float64)
    if len(values) < window:
        return output
    centered_lags = lags.astype(np.float64) - float(lags.mean())
    lag_norm = float(np.sqrt(np.square(centered_lags).sum()))
    index = np.arange(len(values), dtype=np.float64)
    all_supported = np.ones(len(values), dtype=bool)
    for frequency_index, frequency in enumerate(frequencies):
        translation = np.cumsum(values * np.exp(1j * frequency * index))
        mean_displacements = []
        for lag in lags:
            displacement = np.full(len(values), np.nan, dtype=np.float64)
            displacement[lag:] = np.square(np.abs(translation[lag:] - translation[:-lag]))
            mean = pd.Series(displacement).rolling(window - int(lag), min_periods=window - int(lag)).mean()
            mean_displacements.append(mean.to_numpy(np.float64))
        displacement_matrix = np.column_stack(mean_displacements)
        supported = np.isfinite(displacement_matrix).all(axis=1) & (displacement_matrix > 1e-12).all(axis=1)
        all_supported &= supported
        finite_count = np.isfinite(displacement_matrix).sum(axis=1, keepdims=True)
        row_sum = np.nansum(displacement_matrix, axis=1, keepdims=True)
        row_mean = np.divide(
            row_sum,
            finite_count,
            out=np.zeros_like(row_sum),
            where=finite_count > 0,
        )
        centered = displacement_matrix - row_mean
        denominator = np.sqrt(np.nansum(np.square(centered), axis=1)) * lag_norm
        correlation = np.divide(
            np.nansum(centered * centered_lags[None, :], axis=1),
            denominator,
            out=np.zeros(len(values), dtype=np.float64),
            where=supported & (denominator > 1e-12),
        )
        growth = np.zeros(len(values), dtype=np.float64)
        growth[supported] = np.log(
            displacement_matrix[supported, -1] / displacement_matrix[supported, 0]
        ) / np.log(float(lags[-1]) / float(lags[0]))
        output[:, 2 * frequency_index] = np.clip(correlation, -1.0, 1.0)
        output[:, 2 * frequency_index + 1] = np.clip(growth, -12.0, 12.0)
    output[:, -1] = all_supported.astype(np.float64)
    output[~all_supported, :-1] = 0.0
    if not np.isfinite(output).all():
        raise RuntimeError("0-1 chaos coordinates are nonfinite")
    return output


def causal_zero_one_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    representation: dict[str, Any],
) -> np.ndarray:
    output = np.zeros((len(frame), 9), dtype=np.float32)
    for _key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = base._time_ns(ordered["_time"])
        raw = ordered["temp"].to_numpy(np.float64)
        prefix = raw[(times <= train_boundary_ns) & np.isfinite(raw)]
        if not len(prefix):
            continue
        center = float(np.median(prefix))
        scale = float(1.4826 * np.median(np.abs(prefix - center)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(prefix))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        values = np.clip(np.nan_to_num((raw - center) / scale, nan=0.0), -12.0, 12.0)
        coordinates = np.zeros((len(values), 7), dtype=np.float64)
        gaps = np.r_[True, np.diff(times) != CADENCE_NS]
        starts = np.flatnonzero(gaps)
        ends = np.r_[starts[1:], len(values)]
        for start, end in zip(starts, ends, strict=True):
            coordinates[start:end] = zero_one_coordinates(values[start:end], representation)
        difference = np.r_[0.0, np.diff(values)]
        difference[gaps] = 0.0
        features = np.column_stack([values, difference, coordinates])
        if features.shape[1] != 9 or not np.isfinite(features).all():
            raise RuntimeError("causal 0-1 feature contract failed")
        output[positions] = features.astype(np.float32)
    return output


def _synthetic_guards(representation: dict[str, Any]) -> dict[str, bool]:
    rows = 320
    periodic = np.sin(2.0 * np.pi * np.arange(rows, dtype=np.float64) / 17.0)
    chaotic = np.empty(rows, dtype=np.float64)
    chaotic[0] = 0.211
    for index in range(1, rows):
        chaotic[index] = 4.0 * chaotic[index - 1] * (1.0 - chaotic[index - 1])
    periodic_coordinates = zero_one_coordinates(periodic, representation)
    chaotic_coordinates = zero_one_coordinates(chaotic, representation)
    periodic_k = np.median(periodic_coordinates[160:, [0, 2, 4]])
    chaotic_k = np.median(chaotic_coordinates[160:, [0, 2, 4]])
    times = pd.date_range("2024-01-01", periods=160, freq="10min", tz="UTC")
    frame = pd.DataFrame(
        {
            "station": np.repeat(["S-A", "S-B"], 160),
            "layer": np.repeat(["L1", "L2"], 160),
            "_time": np.tile(times, 2),
            "temp": np.tile(np.sin(np.arange(160, dtype=np.float64) / 7.0), 2),
        }
    )
    boundary = int(times[79].value)
    original = causal_zero_one_features(frame, boundary, representation)
    changed = frame.copy()
    future = base._time_ns(changed["_time"]) > boundary
    changed.loc[future, "temp"] += np.linspace(100.0, 500.0, int(future.sum()))
    perturbed = causal_zero_one_features(changed, boundary, representation)
    return {
        "chaotic_translation_growth_exceeds_periodic": bool(chaotic_k > periodic_k + 0.3),
        "periodic_and_chaotic_finite": bool(np.isfinite(periodic_coordinates).all() and np.isfinite(chaotic_coordinates).all()),
        "station_layer_group_reset": bool(
            np.all(original[:95, -1] == 0.0)
            and np.all(original[160:255, -1] == 0.0)
            and original[95, -1] == original[255, -1] == 1.0
        ),
        "prefix_future_invariant": bool(np.array_equal(original[~future], perturbed[~future])),
        "ns_cutoff_distinct": bool(
            base._time_ns(times).dtype == np.dtype("int64")
            and int(times[78].value) < boundary < int(times[80].value)
        ),
        "shape_finite": bool(original.shape == (320, 9) and np.isfinite(original).all()),
    }


def _configure() -> None:
    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.CONFIG = CONFIG
    engine.ARTIFACT = ARTIFACT
    engine.LOCK = LOCK
    engine.dfa_features = causal_zero_one_features
    engine._synthetic_guards = _synthetic_guards
    engine.shared.LinearProbeClassifier = LinearProbeClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = causal_zero_one_features
    base.VIBClassifier = LinearProbeClassifier


def _install_hooks() -> None:
    engine._configure = _configure
    _configure()


def preflight(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return engine.preflight(data_dir)


def execute(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return engine.execute(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return engine.qa(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    value = preflight(args.data_dir) if args.preflight else execute(args.data_dir) if args.execute else qa(args.data_dir)
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
