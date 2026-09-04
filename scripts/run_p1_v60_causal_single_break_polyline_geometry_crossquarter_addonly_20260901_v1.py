"""Exactly-once causal single-break polyline-geometry P1 audit."""

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
EXPERIMENT_ID = "p1_v60_causal_single_break_polyline_geometry_crossquarter_addonly_20260901_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V34 = ROOT / "scripts/run_p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1.py"
CADENCE_NS = 600_000_000_000
LAST_SUPPORT_SUMMARY: dict[str, Any] = {}


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("p1_v60_shared", path)
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


def single_break_polyline_statistics(
    curves: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Return a deterministic one-split maximum-deviation polyline geometry."""

    paths = np.asarray(curves, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] < 3 or not np.isfinite(paths).all():
        raise ValueError("polyline paths must be finite with at least three rows")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("polyline tolerance must be positive")
    rows, width = paths.shape
    x = np.arange(width, dtype=np.float64)
    chord_slope = (paths[:, -1] - paths[:, 0]) / float(width - 1)
    chord = paths[:, :1] + chord_slope[:, None] * x[None, :]
    residual = paths - chord
    interior = np.abs(residual[:, 1:-1])
    split = np.argmax(interior, axis=1).astype(np.int64) + 1
    index = np.arange(rows, dtype=np.int64)
    signed_deviation = residual[index, split]
    maximum_deviation = np.abs(signed_deviation)
    split_value = paths[index, split]
    split_float = split.astype(np.float64)
    left_slope = (split_value - paths[:, 0]) / split_float
    right_slope = (paths[:, -1] - split_value) / (float(width - 1) - split_float)
    left = paths[:, :1] + left_slope[:, None] * x[None, :]
    right = split_value[:, None] + right_slope[:, None] * (x[None, :] - split_float[:, None])
    piecewise = np.where(x[None, :] <= split_float[:, None], left, right)
    piecewise_rms = np.sqrt(np.mean(np.square(paths - piecewise), axis=1))
    left_length = np.hypot(split_float, split_value - paths[:, 0])
    right_length = np.hypot(
        float(width - 1) - split_float,
        paths[:, -1] - split_value,
    )
    chord_length = np.hypot(float(width - 1), paths[:, -1] - paths[:, 0])
    path_excess = np.maximum(0.0, (left_length + right_length) / chord_length - 1.0)
    turning_angle = np.arctan(right_slope) - np.arctan(left_slope)
    active = (maximum_deviation > tolerance).astype(np.float64)
    output = np.column_stack(
        [
            chord_slope,
            maximum_deviation,
            signed_deviation,
            split_float / float(width - 1),
            piecewise_rms,
            path_excess,
            turning_angle,
            active,
        ]
    )
    if output.shape != (rows, 8) or not np.isfinite(output).all():
        raise RuntimeError("single-break polyline statistics are invalid")
    return output.astype(np.float32)


def _valid_window_ends(times_ns: np.ndarray, window: int) -> np.ndarray:
    times = np.asarray(times_ns, dtype=np.int64)
    if window < 3:
        raise ValueError("polyline window must contain at least three rows")
    if not len(times):
        return np.empty(0, dtype=np.int64)
    gaps = np.r_[True, np.diff(times) != CADENCE_NS]
    starts = np.maximum.accumulate(np.where(gaps, np.arange(len(times)), 0))
    age = np.arange(len(times)) - starts + 1
    return np.flatnonzero(age >= window).astype(np.int64)


def causal_polyline_features(
    frame: pd.DataFrame,
    train_boundary_ns: int,
    representation: dict[str, Any],
) -> np.ndarray:
    """Build past-only adaptive polyline features with prefix-only normalization."""

    window = int(representation["trajectory_rows"])
    tolerance = float(representation["split_tolerance_prefix_sigma"])
    chunk_rows = int(representation["chunk_rows"])
    output = np.zeros((len(frame), 11), dtype=np.float32)
    supported_positions = np.zeros(len(frame), dtype=bool)
    supported_identities: set[str] = set()
    supported_stations: set[str] = set()
    active_count = 0
    offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
    for key, group in frame.groupby(["station", "layer"], sort=True, observed=True):
        ordered = group.sort_values("_time", kind="stable")
        positions = ordered.index.to_numpy(np.int64)
        times = base._time_ns(ordered["_time"])
        raw = ordered["temp"].to_numpy(np.float64)
        prefix = raw[(times <= train_boundary_ns) & np.isfinite(raw)]
        if len(prefix) < window:
            continue
        center = float(np.median(prefix))
        scale = float(1.4826 * np.median(np.abs(prefix - center)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(prefix))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        values = np.clip(np.nan_to_num((raw - center) / scale, nan=0.0), -12.0, 12.0)
        valid_ends = _valid_window_ends(times, window)
        if not len(valid_ends):
            continue
        statistics = np.zeros((len(valid_ends), 8), dtype=np.float32)
        for start in range(0, len(valid_ends), chunk_rows):
            stop = min(len(valid_ends), start + chunk_rows)
            ends = valid_ends[start:stop]
            curves = values[ends[:, None] - offsets[None, :]]
            statistics[start:stop] = single_break_polyline_statistics(curves, tolerance)
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
        if features.shape != (len(valid_ends), 11) or not np.isfinite(features).all():
            raise RuntimeError("causal polyline feature contract failed")
        output[positions[valid_ends]] = features.astype(np.float32)
        supported_positions[positions[valid_ends]] = True
        active_count += int(statistics[:, -1].sum())
        station, layer = str(key[0]), str(key[1])
        supported_identities.add(f"{station}|{layer}")
        supported_stations.add(station)
    supported_rows = int(supported_positions.sum())
    LAST_SUPPORT_SUMMARY.clear()
    LAST_SUPPORT_SUMMARY.update(
        {
            "rows": int(len(frame)),
            "supported_rows": supported_rows,
            "supported_row_share": float(supported_positions.mean()) if len(frame) else 0.0,
            "supported_station_layers": len(supported_identities),
            "distinct_stations": len(supported_stations),
            "minimum_trajectory_rows": window if supported_rows else 0,
            "split_active_rows": active_count,
            "split_active_share": float(active_count / supported_rows) if supported_rows else 0.0,
        }
    )
    return output


def _source_lineage_audit(data_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    policy = config["policy_binding"]
    lineage = config["execution_lineage"]
    readme = data_dir.resolve() / "README.md"
    train = data_dir.resolve() / "train.csv"
    dependencies = lineage["dependencies"]
    dependency_checks = {
        path: bool((ROOT / path).is_file() and base._sha(ROOT / path) == expected)
        for path, expected in dependencies.items()
    }
    lowered_paths = [path.replace("\\", "/").lower() for path in dependencies]
    forbidden_tokens = [str(value).lower() for value in lineage["forbidden_path_tokens"]]
    checks = {
        "allowed_files_exact": source["allowed_files"] == ["README.md", "train.csv"],
        "model_inputs_exact": source["model_input_columns"] == ["station", "layer", "time", "temp"],
        "readme_hash": readme.is_file() and base._sha(readme) == source["readme_sha256"],
        "train_hash": train.is_file() and base._sha(train) == source["train_sha256"],
        "external_lineage_inputs_zero": source["external_lineage_inputs"] == [],
        "non_distributed_iors_zero": policy["non_distributed_iors_lineages"] == 0,
        "external_observation_zero": policy["external_observation_reanalysis_forecast"] == 0,
        "pretrained_weights_zero": policy["pretrained_weights"] == 0,
        "dependencies_hash_bound": all(dependency_checks.values()),
        "dependency_paths_internal": all(not Path(path).is_absolute() for path in dependencies),
        "forbidden_dependency_tokens_absent": not any(
            token in path for token in forbidden_tokens for path in lowered_paths
        ),
        "protected_values_zero": source["official_test_sample_submission_hidden_reads"] == 0,
    }
    return {
        "gate": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "dependency_checks": dependency_checks,
        "allowed_source_hashes": {
            "README.md": source["readme_sha256"],
            "train.csv": source["train_sha256"],
        },
        "external_lineages": 0,
        "non_distributed_iors_lineages": 0,
        "pretrained_weights": 0,
        "official_test_sample_submission_hidden_reads": 0,
    }


def _synthetic_guards(representation: dict[str, Any]) -> dict[str, bool]:
    prior_summary = dict(LAST_SUPPORT_SUMMARY)
    try:
        window = int(representation["trajectory_rows"])
        tolerance = float(representation["split_tolerance_prefix_sigma"])
        x = np.arange(window, dtype=np.float64)
        straight = (0.05 * x)[None, :]
        kink = np.where(x < window // 2, 0.03 * x, 0.03 * (window // 2) + 0.24 * (x - window // 2))[None, :]
        step = np.where(x < window // 2, 0.0, 2.5)[None, :]
        straight_stats = single_break_polyline_statistics(straight, tolerance)[0]
        kink_stats = single_break_polyline_statistics(kink, tolerance)[0]
        step_stats = single_break_polyline_statistics(step, tolerance)[0]

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
        original = causal_polyline_features(frame, boundary, representation)
        changed = frame.copy()
        future = base._time_ns(changed["_time"]) > boundary
        changed.loc[future, "temp"] += 1000.0
        perturbed = causal_polyline_features(changed, boundary, representation)

        gap_frame = frame.iloc[:rows].copy()
        gap_frame.loc[200:, "_time"] += pd.Timedelta(minutes=10)
        gap_features = causal_polyline_features(gap_frame, boundary, representation)
        return {
            "straight_path_inactive": bool(straight_stats[1] < 1e-10 and straight_stats[-1] == 0.0),
            "kink_path_detected": bool(kink_stats[1] > tolerance and abs(kink_stats[6]) > 0.05),
            "step_path_detected": bool(step_stats[1] > tolerance and step_stats[-1] == 1.0),
            "adaptive_break_is_internal": bool(0.0 < kink_stats[3] < 1.0 and 0.0 < step_stats[3] < 1.0),
            "strict_prefix_future_invariant": bool(np.array_equal(original[~future], perturbed[~future])),
            "station_layer_group_reset": bool(np.array_equal(original[:rows], original[rows:])),
            "cadence_gap_resets_window": bool(np.all(gap_features[200 : 200 + window - 1, -1] == 0.0)),
            "shape_finite": bool(original.shape == (2 * rows, 11) and np.isfinite(original).all()),
        }
    finally:
        LAST_SUPPORT_SUMMARY.clear()
        LAST_SUPPORT_SUMMARY.update(prior_summary)


def _configure() -> None:
    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.CONFIG = CONFIG
    engine.ARTIFACT = ARTIFACT
    engine.LOCK = LOCK
    engine.dfa_features = causal_polyline_features
    engine._synthetic_guards = _synthetic_guards
    engine.shared.LinearProbeClassifier = LinearProbeClassifier
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.LOCK = LOCK
    base.__file__ = str(Path(__file__).resolve())
    base.shared.causal_evidential_features = causal_polyline_features
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
    lineage = _source_lineage_audit(data_dir, config)
    if lineage["gate"] != "PASS":
        raise RuntimeError(f"source lineage gate failed: {lineage['checks']}")
    support = dict(LAST_SUPPORT_SUMMARY)
    gate = config["representation_support_gate"]
    feature_variances = ready["representation_support"]["feature_variances"]
    polyline_variance = float(max(feature_variances[2:10]))
    checks = {
        "supported_row_share": support["supported_row_share"] >= gate["minimum_supported_row_share"],
        "supported_station_layers": support["supported_station_layers"] >= gate["minimum_supported_station_layers"],
        "distinct_stations": support["distinct_stations"] >= gate["minimum_distinct_stations"],
        "trajectory_rows": support["minimum_trajectory_rows"] >= gate["minimum_trajectory_rows"],
        "polyline_feature_variance": polyline_variance >= gate["minimum_polyline_feature_variance"],
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
        "non_distributed_iors_lineages": 0,
    }
    ready["source_lineage_audit"] = lineage
    ready["support_qualification"] = {
        **support,
        "polyline_feature_variance": polyline_variance,
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
    lineage = _source_lineage_audit(data_dir, config)
    checks["source_lineage_audit"] = lineage["gate"] == "PASS"
    stored_preflight = ARTIFACT / "preflight.json"
    if stored_preflight.exists():
        receipt = base._read(stored_preflight)
        checks["support_qualified"] = bool(
            receipt["ready"] and receipt["support_qualification"]["gate"] == "PASS"
        )
        checks["stored_source_lineage_pass"] = (
            receipt["source_lineage_audit"]["gate"] == "PASS"
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
