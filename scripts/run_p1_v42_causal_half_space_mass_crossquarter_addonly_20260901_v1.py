"""Exactly-once causal prefix Half-Space mass P1 audit."""

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
EXPERIMENT_ID = "p1_v42_causal_half_space_mass_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V34 = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v42_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


engine = _module(V34)
base = engine.base
LinearProbeClassifier = engine.shared.LinearProbeClassifier


def causal_state_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    representation: dict[str, Any],
) -> np.ndarray:
    """Fixed causal temperature state with prefix-only robust normalization."""

    lags = [int(value) for value in representation["lag_rows"]]
    output = np.zeros((len(frame), 8), dtype=np.float32)
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
        lagged = np.zeros((len(values), len(lags)), dtype=np.float64)
        supported = np.zeros((len(values), len(lags)), dtype=bool)
        for column, lag in enumerate(lags):
            valid = np.arange(lag, len(values))
            exact = times[valid] - times[valid - lag] == lag * CADENCE_NS
            rows = valid[exact]
            lagged[rows, column] = values[rows - lag]
            supported[rows, column] = True
        difference = values - lagged[:, 0]
        difference[~supported[:, 0]] = 0.0
        previous_difference = np.r_[0.0, difference[:-1]]
        acceleration = difference - previous_difference
        acceleration[~supported[:, 0]] = 0.0
        gap = (~supported.all(axis=1)).astype(np.float64)
        features = np.column_stack(
            [values, lagged, difference, np.abs(difference), acceleration, gap]
        )
        if features.shape[1] != 8 or not np.isfinite(features).all():
            raise RuntimeError("causal state feature contract failed")
        output[positions] = features.astype(np.float32)
    return output


def fixed_half_space_tree(
    representation: dict[str, Any],
    tree_index: int,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a complete random half-space tree without inspecting data."""

    depth = int(representation["maximum_depth"])
    nodes = 2**depth - 1
    low_value, high_value = [float(value) for value in representation["normalized_bounds"]]
    gap_low, gap_high = [float(value) for value in representation["gap_bounds"]]
    lower = np.full((nodes, feature_count), low_value, dtype=np.float64)
    upper = np.full((nodes, feature_count), high_value, dtype=np.float64)
    lower[:, -1] = gap_low
    upper[:, -1] = gap_high
    dimensions = np.zeros(nodes, dtype=np.int16)
    thresholds = np.zeros(nodes, dtype=np.float64)
    rng = np.random.default_rng(int(representation["tree_seed"]) + 1009 * int(tree_index))
    for node in range(nodes):
        dimension = int(rng.integers(0, feature_count))
        threshold = 0.5 * (lower[node, dimension] + upper[node, dimension])
        dimensions[node] = dimension
        thresholds[node] = threshold
        left, right = 2 * node + 1, 2 * node + 2
        if left < nodes:
            lower[left] = lower[node]
            upper[left] = upper[node]
            upper[left, dimension] = threshold
            lower[right] = lower[node]
            upper[right] = upper[node]
            lower[right, dimension] = threshold
    return dimensions, thresholds


def half_space_rarity(
    states: np.ndarray,
    prefix_mask: np.ndarray,
    representation: dict[str, Any],
) -> np.ndarray:
    """Return mean, maximum, and standard deviation of prefix leaf-mass rarity."""

    values = np.asarray(states, dtype=np.float64)
    prefix_mask = np.asarray(prefix_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 8 or len(prefix_mask) != len(values):
        raise ValueError("half-space input shape invalid")
    prefix_count = int(prefix_mask.sum())
    if prefix_count == 0:
        return np.zeros((len(values), 3), dtype=np.float32)
    row_indices = np.arange(len(values), dtype=np.int64)
    depth = int(representation["maximum_depth"])
    leaf_offset = 2**depth - 1
    leaf_count = 2**depth
    tree_count = int(representation["tree_count"])
    total = np.zeros(len(values), dtype=np.float64)
    total_square = np.zeros(len(values), dtype=np.float64)
    maximum = np.zeros(len(values), dtype=np.float64)
    normalization = np.log1p(float(prefix_count))
    for tree_index in range(tree_count):
        dimensions, thresholds = fixed_half_space_tree(representation, tree_index, values.shape[1])
        nodes = np.zeros(len(values), dtype=np.int64)
        for _level in range(depth):
            node_dimensions = dimensions[nodes]
            right = values[row_indices, node_dimensions] >= thresholds[nodes]
            nodes = 2 * nodes + 1 + right.astype(np.int64)
        leaves = nodes - leaf_offset
        masses = np.bincount(leaves[prefix_mask], minlength=leaf_count)
        rarity = 1.0 - np.log1p(masses[leaves].astype(np.float64)) / normalization
        total += rarity
        total_square += np.square(rarity)
        maximum = np.maximum(maximum, rarity)
    mean = total / tree_count
    deviation = np.sqrt(np.maximum(0.0, total_square / tree_count - np.square(mean)))
    output = np.column_stack([mean, maximum, deviation]).astype(np.float32)
    if not np.isfinite(output).all() or np.any(output < -1e-7) or np.any(output > 1.0 + 1e-7):
        raise RuntimeError("half-space rarity feature contract failed")
    return output


def causal_half_space_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    representation: dict[str, Any],
) -> np.ndarray:
    states = causal_state_features(frame, train_boundary_ns, representation)
    prefix_mask = base._time_ns(frame["_time"]) <= int(train_boundary_ns)
    rarity = half_space_rarity(states, prefix_mask, representation)
    output = np.column_stack([states, rarity]).astype(np.float32)
    if output.shape != (len(frame), 11) or not np.isfinite(output).all():
        raise RuntimeError("causal Half-Space feature contract failed")
    return output


def _synthetic_guards(representation: dict[str, Any]) -> dict[str, bool]:
    rng = np.random.default_rng(17)
    reference = rng.normal(0.0, 0.25, size=(256, 8))
    reference[:, -1] = 0.0
    outliers = np.full((32, 8), 9.0, dtype=np.float64)
    outliers[:, -1] = 0.0
    states = np.vstack([reference, outliers])
    rarity = half_space_rarity(states, np.arange(len(states)) < len(reference), representation)

    rows = 160
    times = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    frame = pd.DataFrame(
        {
            "station": np.repeat(["S-A", "S-B"], rows),
            "layer": np.repeat(["L1", "L2"], rows),
            "_time": np.tile(times, 2),
            "temp": np.tile(np.sin(np.arange(rows, dtype=np.float64) / 7.0), 2),
        }
    )
    boundary = int(times[79].value)
    original = causal_half_space_features(frame, boundary, representation)
    changed = frame.copy()
    future = base._time_ns(changed["_time"]) > boundary
    changed.loc[future, "temp"] += np.linspace(100.0, 500.0, int(future.sum()))
    perturbed = causal_half_space_features(changed, boundary, representation)
    dimensions_a, thresholds_a = fixed_half_space_tree(representation, 0, 8)
    dimensions_b, thresholds_b = fixed_half_space_tree(representation, 0, 8)
    return {
        "outliers_have_higher_mass_rarity": bool(
            float(rarity[len(reference) :, 0].mean()) > float(rarity[: len(reference), 0].mean()) + 0.1
        ),
        "tree_structure_data_independent_deterministic": bool(
            np.array_equal(dimensions_a, dimensions_b) and np.array_equal(thresholds_a, thresholds_b)
        ),
        "station_layer_group_reset": bool(np.array_equal(original[:rows], original[rows:])),
        "prefix_future_invariant": bool(np.array_equal(original[~future], perturbed[~future])),
        "ns_cutoff_distinct": bool(
            base._time_ns(times).dtype == np.dtype("int64")
            and int(times[78].value) < boundary < int(times[80].value)
        ),
        "shape_finite": bool(original.shape == (2 * rows, 11) and np.isfinite(original).all()),
    }


def _configure() -> None:
    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.CONFIG = CONFIG
    engine.ARTIFACT = ARTIFACT
    engine.LOCK = LOCK
    engine.dfa_features = causal_half_space_features
    engine._synthetic_guards = _synthetic_guards
    engine.shared.LinearProbeClassifier = LinearProbeClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = causal_half_space_features
    base.VIBClassifier = LinearProbeClassifier


def _install_hooks() -> None:
    engine._configure = _configure
    _configure()


def preflight(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    ready = engine.preflight(data_dir)
    config = base._read(CONFIG)
    rarity_variances = ready["representation_support"]["feature_variances"][-3:]
    minimum = float(config["representation_support_gate"]["minimum_rarity_variance"])
    if max(rarity_variances) < minimum:
        raise RuntimeError(config["representation_support_gate"]["failure"])
    ready["representation_support"]["rarity_variances"] = rarity_variances
    ready["representation_support"]["target_free_prefix_mass_gate"] = "PASS"
    return ready


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
