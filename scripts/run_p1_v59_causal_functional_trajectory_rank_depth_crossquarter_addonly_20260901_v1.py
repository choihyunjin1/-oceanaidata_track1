"""Exactly-once causal functional trajectory rank-depth P1 audit."""

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
EXPERIMENT_ID = "p1_v59_causal_functional_trajectory_rank_depth_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V34 = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000
LAST_SUPPORT_SUMMARY: dict[str, Any] = {}


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v59_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared execution module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


engine = _module(V34)
base = engine.base
LinearProbeClassifier = engine.shared.LinearProbeClassifier
ORIGINAL_ENGINE_PREFLIGHT = engine.preflight
ORIGINAL_ENGINE_QA = engine.qa


def trajectory_depth_statistics(
    candidates: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Return modified band depth and directional outlyingness coordinates."""

    curves = np.asarray(candidates, dtype=np.float64)
    library = np.asarray(reference, dtype=np.float64)
    if curves.ndim != 2 or library.ndim != 2 or curves.shape[1] != library.shape[1]:
        raise ValueError("trajectory matrices are misaligned")
    if len(library) < 3 or not np.isfinite(curves).all() or not np.isfinite(library).all():
        raise ValueError("trajectory reference contract failed")
    below = np.sum(library[None, :, :] < curves[:, None, :], axis=1, dtype=np.int64)
    above = np.sum(library[None, :, :] > curves[:, None, :], axis=1, dtype=np.int64)
    total_pairs = float(len(library) * (len(library) - 1) / 2)
    excluded_below = below * (below - 1) / 2.0
    excluded_above = above * (above - 1) / 2.0
    pointwise_depth = np.clip(
        (total_pairs - excluded_below - excluded_above) / total_pairs,
        0.0,
        1.0,
    )
    modified_band_depth = pointwise_depth.mean(axis=1)
    median = np.median(library, axis=0)
    scale = 1.4826 * np.median(np.abs(library - median[None, :]), axis=0)
    fallback = np.std(library, axis=0)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-6), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-6), scale, 1.0)
    directional = np.clip((curves - median[None, :]) / scale[None, :], -30.0, 30.0)
    mean_directional = directional.mean(axis=1)
    magnitude = np.abs(mean_directional)
    shape = np.sqrt(np.mean(np.square(directional - mean_directional[:, None]), axis=1))
    maximum = np.max(np.abs(directional), axis=1)
    endpoint = np.abs(directional[:, -1])
    output = np.column_stack(
        [
            modified_band_depth,
            1.0 - modified_band_depth,
            mean_directional,
            magnitude,
            shape,
            maximum,
            endpoint,
        ]
    )
    if output.shape != (len(curves), 7) or not np.isfinite(output).all():
        raise RuntimeError("functional trajectory statistics are invalid")
    return output.astype(np.float32)


def _valid_window_ends(times_ns: np.ndarray, window: int) -> np.ndarray:
    times = np.asarray(times_ns, dtype=np.int64)
    if window < 2:
        raise ValueError("trajectory window must be at least two rows")
    if not len(times):
        return np.empty(0, dtype=np.int64)
    gaps = np.r_[True, np.diff(times) != CADENCE_NS]
    starts = np.maximum.accumulate(np.where(gaps, np.arange(len(times)), 0))
    age = np.arange(len(times)) - starts + 1
    return np.flatnonzero(age >= window).astype(np.int64)


def causal_functional_depth_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    representation: dict[str, Any],
) -> np.ndarray:
    """Build causal trailing curves against a prefix-only reference library."""

    window = int(representation["trajectory_rows"])
    reference_count = int(representation["reference_curves"])
    chunk_rows = int(representation["chunk_rows"])
    output = np.zeros((len(frame), 10), dtype=np.float32)
    supported_positions = np.zeros(len(frame), dtype=bool)
    supported_identities: set[str] = set()
    supported_stations: set[str] = set()
    reference_counts: list[int] = []
    offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
    for key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
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
        valid_ends = _valid_window_ends(times, window)
        prefix_ends = valid_ends[times[valid_ends] <= int(train_boundary_ns)]
        if len(prefix_ends) < reference_count:
            continue
        chosen = np.linspace(0, len(prefix_ends) - 1, reference_count, dtype=np.int64)
        reference_ends = prefix_ends[chosen]
        reference = values[reference_ends[:, None] - offsets[None, :]]
        reference_counts.append(int(len(reference)))
        statistics = np.zeros((len(valid_ends), 7), dtype=np.float32)
        for start in range(0, len(valid_ends), chunk_rows):
            end = min(len(valid_ends), start + chunk_rows)
            curve_ends = valid_ends[start:end]
            curves = values[curve_ends[:, None] - offsets[None, :]]
            statistics[start:end] = trajectory_depth_statistics(curves, reference)
        exact = np.r_[False, np.diff(times) == CADENCE_NS]
        difference = np.r_[0.0, np.diff(values)]
        difference[~exact] = 0.0
        features = np.column_stack(
            [
                values[valid_ends],
                difference[valid_ends],
                statistics,
                np.ones(len(valid_ends), dtype=np.float64),
            ]
        )
        if features.shape != (len(valid_ends), 10) or not np.isfinite(features).all():
            raise RuntimeError("causal functional-depth feature contract failed")
        output[positions[valid_ends]] = features.astype(np.float32)
        supported_positions[positions[valid_ends]] = True
        station, layer = str(key[0]), str(key[1])
        supported_identities.add(f"{station}|{layer}")
        supported_stations.add(station)
    LAST_SUPPORT_SUMMARY.clear()
    LAST_SUPPORT_SUMMARY.update(
        {
            "rows": int(len(frame)),
            "supported_rows": int(supported_positions.sum()),
            "supported_row_share": float(supported_positions.mean()) if len(frame) else 0.0,
            "supported_station_layers": len(supported_identities),
            "distinct_stations": len(supported_stations),
            "minimum_reference_curves": min(reference_counts) if reference_counts else 0,
        }
    )
    return output


def _synthetic_guards(representation: dict[str, Any]) -> dict[str, bool]:
    prior_summary = dict(LAST_SUPPORT_SUMMARY)
    try:
        window = int(representation["trajectory_rows"])
        phase = np.linspace(0.0, 2.0 * np.pi, window, endpoint=False)
        reference = np.stack(
            [
                np.sin(phase + shift) + 0.02 * index
                for index, shift in enumerate(np.linspace(-0.25, 0.25, 32))
            ]
        )
        center = np.sin(phase)[None, :]
        magnitude_outlier = center + 5.0
        shape_outlier = (np.sin(phase) + 3.0 * np.sin(3.0 * phase))[None, :]
        center_stats = trajectory_depth_statistics(center, reference)[0]
        magnitude_stats = trajectory_depth_statistics(magnitude_outlier, reference)[0]
        shape_stats = trajectory_depth_statistics(shape_outlier, reference)[0]

        rows = 360
        times = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
        signal = np.sin(np.arange(rows, dtype=np.float64) / 11.0)
        frame = pd.DataFrame(
            {
                "station": np.repeat(["S-A", "S-B"], rows),
                "layer": np.repeat(["L1", "L2"], rows),
                "_time": np.tile(times, 2),
                "temp": np.tile(signal, 2),
            }
        )
        boundary = int(times[179].value)
        original = causal_functional_depth_features(frame, boundary, representation)
        changed = frame.copy()
        future = base._time_ns(changed["_time"]) > boundary
        changed.loc[future, "temp"] += 1000.0
        perturbed = causal_functional_depth_features(changed, boundary, representation)

        gap_frame = frame.iloc[:rows].copy()
        gap_frame.loc[200:, "_time"] += pd.Timedelta(minutes=10)
        gap_features = causal_functional_depth_features(gap_frame, boundary, representation)
        return {
            "central_curve_has_greater_depth": bool(
                center_stats[0] > magnitude_stats[0] and center_stats[0] > shape_stats[0]
            ),
            "magnitude_component_identifies_offset": bool(magnitude_stats[3] > center_stats[3] + 2.0),
            "shape_component_identifies_distortion": bool(shape_stats[4] > center_stats[4] + 1.0),
            "strict_prefix_future_invariant": bool(np.array_equal(original[~future], perturbed[~future])),
            "station_layer_group_reset": bool(np.array_equal(original[:rows], original[rows:])),
            "cadence_gap_resets_window": bool(np.all(gap_features[200 : 200 + window - 1, -1] == 0.0)),
            "shape_finite": bool(original.shape == (2 * rows, 10) and np.isfinite(original).all()),
        }
    finally:
        LAST_SUPPORT_SUMMARY.clear()
        LAST_SUPPORT_SUMMARY.update(prior_summary)


def _configure() -> None:
    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.CONFIG = CONFIG
    engine.ARTIFACT = ARTIFACT
    engine.LOCK = LOCK
    engine.dfa_features = causal_functional_depth_features
    engine._synthetic_guards = _synthetic_guards
    engine.shared.LinearProbeClassifier = LinearProbeClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = causal_functional_depth_features
    base.VIBClassifier = LinearProbeClassifier


def _install_hooks() -> None:
    engine._configure = _configure
    engine.preflight = preflight
    _configure()


def preflight(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    ready = ORIGINAL_ENGINE_PREFLIGHT(data_dir)
    config = base._read(CONFIG)
    policy = config["policy_binding"]
    if base._sha(ROOT / policy["path"]) != policy["sha256"]:
        raise RuntimeError("organizer policy binding drifted")
    if not policy["distributed_data_only"] or policy["pretrained_weights"] != 0:
        raise RuntimeError("distributed-data scratch provenance gate failed")
    support = dict(LAST_SUPPORT_SUMMARY)
    gate = config["representation_support_gate"]
    feature_variances = ready["representation_support"]["feature_variances"]
    functional_variance = float(max(feature_variances[2:9]))
    checks = {
        "supported_row_share": support["supported_row_share"] >= gate["minimum_supported_row_share"],
        "supported_station_layers": support["supported_station_layers"] >= gate["minimum_supported_station_layers"],
        "distinct_stations": support["distinct_stations"] >= gate["minimum_distinct_stations"],
        "reference_curves": support["minimum_reference_curves"] >= gate["minimum_reference_curves_per_supported_identity"],
        "functional_feature_variance": functional_variance >= gate["minimum_functional_feature_variance"],
        "target_columns_read_zero": gate["target_columns_read"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"{gate['failure']}: {checks}")
    ready["status"] = "READY_ZERO_OPERATION"
    ready["ready"] = True
    ready["policy_binding"] = {
        "sha256": policy["sha256"],
        "distributed_data_only": True,
        "pretrained_weights": 0,
        "external_lineages": 0,
    }
    ready["support_qualification"] = {
        **support,
        "functional_feature_variance": functional_variance,
        "checks": checks,
        "gate": "PASS",
    }
    return ready


def execute(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    return engine.execute(data_dir)


def qa(data_dir: Path) -> dict[str, Any]:
    _install_hooks()
    value = ORIGINAL_ENGINE_QA(data_dir)
    config = base._read(CONFIG)
    checks = value["checks"]
    checks["organizer_policy_binding"] = (
        base._sha(ROOT / config["policy_binding"]["path"])
        == config["policy_binding"]["sha256"]
    )
    checks["distributed_data_scratch_only"] = bool(
        config["policy_binding"]["distributed_data_only"]
        and config["policy_binding"]["pretrained_weights"] == 0
        and config["model"]["pretrained_weights"] == 0
    )
    stored_preflight = ARTIFACT / "preflight.json"
    if stored_preflight.exists():
        receipt = base._read(stored_preflight)
        checks["support_qualified"] = bool(
            receipt["ready"] and receipt["support_qualification"]["gate"] == "PASS"
        )
    value["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    value = (
        preflight(args.data_dir)
        if args.preflight
        else execute(args.data_dir)
        if args.execute
        else qa(args.data_dir)
    )
    print(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False), end="")


if __name__ == "__main__":
    main()
