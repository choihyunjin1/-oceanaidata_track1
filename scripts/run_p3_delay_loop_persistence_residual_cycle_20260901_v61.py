"""Sealed P3 v61 delay-cloud H1 loop-persistence residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

import run_p3_extreme_order_increment_tail_residual_cycle_20260901_v60 as v60  # noqa: E402

EXPERIMENT_ID = "p3_delay_loop_persistence_residual_cycle_20260901_v61"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS, DELAY, DIMENSION, CLOUD_POINTS = ((0, 145), (72, 145)), 2, 2, 16
FEATURE_COUNT = 32
BASE = v60.BASE
SPEC_CLASS = v60.SPECS[0].__class__
SPECS = (SPEC_CLASS("P3_1_H1LOOP32_RIDGE512_ADD10", 512.0), SPEC_CLASS("P3_2_H1LOOP32_RIDGE2048_ADD10", 2048.0))
BLEND, EPSILON = 0.10, 1e-12
sha256, canonical, write_new = v60.sha256, v60.canonical, v60.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v61 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.delay_loop_persistence_residual.config.v61",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_DELAY_LOOP_H1_PERSISTENCE_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "embedding": encoder["delay_rows"] == DELAY and encoder["embedding_dimension"] == DIMENSION,
        "cloud": encoder["cloud_points"] == CLOUD_POINTS,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_v42_excluded": "excluded" in config["duplication_audit"]["official_v42_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v61 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def normalized_cloud(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    median = float(np.median(path))
    q25, q75 = np.quantile(path, (0.25, 0.75))
    path = (path - median) / max(float(q75 - q25), EPSILON)
    count = len(path) - DELAY * (DIMENSION - 1)
    if count < CLOUD_POINTS:
        raise ContractError("delay-cloud support below fixed point count")
    cloud = np.column_stack([path[offset * DELAY : offset * DELAY + count] for offset in range(DIMENSION)])
    indices = np.linspace(0, count - 1, CLOUD_POINTS, dtype=np.int64)
    cloud = cloud[indices]
    cloud -= np.mean(cloud, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(np.square(cloud), axis=1))))
    cloud /= max(scale, EPSILON)
    if cloud.shape != (CLOUD_POINTS, DIMENSION) or not np.isfinite(cloud).all():
        raise ContractError("normalized delay cloud differs")
    return cloud


def h1_lifetimes(cloud: np.ndarray) -> np.ndarray:
    points = np.asarray(cloud, dtype=np.float64)
    if points.ndim != 2 or len(points) != CLOUD_POINTS or not np.isfinite(points).all():
        raise ContractError("H1 cloud contract differs")
    distance = np.sqrt(np.sum(np.square(points[:, None, :] - points[None, :, :]), axis=2))
    simplices: list[tuple[float, int, tuple[int, ...]]] = []
    simplices.extend((0.0, 0, (index,)) for index in range(CLOUD_POINTS))
    for i, j in itertools.combinations(range(CLOUD_POINTS), 2):
        simplices.append((float(distance[i, j]), 1, (i, j)))
    for i, j, k in itertools.combinations(range(CLOUD_POINTS), 3):
        simplices.append((float(max(distance[i, j], distance[i, k], distance[j, k])), 2, (i, j, k)))
    simplices.sort(key=lambda item: (item[0], item[1], item[2]))
    simplex_index = {simplex: index for index, (_, _, simplex) in enumerate(simplices)}
    reduced: list[set[int]] = []
    pivot_column: dict[int, int] = {}
    lifetimes: list[float] = []
    for column_index, (filtration, dimension, simplex) in enumerate(simplices):
        if dimension == 0:
            column: set[int] = set()
        else:
            column = {simplex_index[simplex[:offset] + simplex[offset + 1 :]] for offset in range(len(simplex))}
        while column and max(column) in pivot_column:
            column ^= reduced[pivot_column[max(column)]]
        reduced.append(column)
        if column:
            pivot = max(column)
            pivot_column[pivot] = column_index
            if dimension == 2 and simplices[pivot][1] == 1:
                lifetime = float(filtration - simplices[pivot][0])
                if lifetime > EPSILON:
                    lifetimes.append(lifetime)
    result = np.sort(np.asarray(lifetimes, dtype=np.float64))
    if not np.isfinite(result).all():
        raise ContractError("H1 lifetimes are nonfinite")
    return result


def loop_statistics_from_cloud(cloud: np.ndarray) -> np.ndarray:
    points = np.asarray(cloud, dtype=np.float64)
    points = points - np.mean(points, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(np.square(points), axis=1))))
    points = points / max(scale, EPSILON)
    lifetimes = h1_lifetimes(points)
    if len(lifetimes) == 0:
        return np.zeros(4, dtype=np.float64)
    total = float(np.sum(lifetimes))
    probability = lifetimes / max(total, EPSILON)
    entropy = 0.0 if len(lifetimes) == 1 else float(-np.sum(probability * np.log(np.maximum(probability, EPSILON))) / np.log(len(lifetimes)))
    result = np.asarray([float(np.max(lifetimes)), total, float(np.linalg.norm(lifetimes)), entropy], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ContractError("H1 loop statistics differ")
    return result


def loop_persistence_features(sequence: np.ndarray) -> np.ndarray:
    path = v60.v59.v58.v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        for start, stop in WINDOWS:
            output.extend(loop_statistics_from_cloud(normalized_cloud(path[start:stop, channel])))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("H1 loop feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    angle = np.linspace(0.0, 2.0 * np.pi, CLOUD_POINTS, endpoint=False)
    circle = np.column_stack([np.cos(angle), np.sin(angle)])
    line = np.column_stack([np.linspace(-1.0, 1.0, CLOUD_POINTS), np.zeros(CLOUD_POINTS)])
    circle_stats = loop_statistics_from_cloud(circle)
    line_stats = loop_statistics_from_cloud(line)
    if not circle_stats[0] > line_stats[0] + 0.10:
        raise ContractError("synthetic line-circle H1 direction guard failed")
    if not np.allclose(circle_stats, loop_statistics_from_cloud(7.0 * circle + 3.0), rtol=1e-10, atol=1e-10):
        raise ContractError("scale-translation invariance guard failed")
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = loop_persistence_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(feature, loop_persistence_features(extended[:289])):
        raise ContractError("future isolation guard failed")
    return {"feature_count": len(feature), "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(feature).all()), "circle_max_h1_lifetime": float(circle_stats[0]), "line_max_h1_lifetime": float(line_stats[0]), "scale_translation_invariant": True, "future_isolated": True, "simplex_order": "filtration_dimension_lexicographic"}


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(BASE.SEQUENCES, mmap_mode="r")
    station_codes = np.load(BASE.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = loop_persistence_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def case_surface() -> tuple[pd.DataFrame, Any, Any, Any]:
    return v60.v59.v58.v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(CLOUD_POINTS >= int(gate["minimum_cloud_points"]) and nonzero_share >= float(gate["minimum_nonzero_share"]) and positive_variance >= int(gate["minimum_positive_variance_features"]))
    return {**metadata, "cloud_points": CLOUD_POINTS, "nonzero_share": nonzero_share, "positive_variance_features": positive_variance, "target_used": False, "passed": passed}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v61 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {"schema_version": "p3.delay_loop_persistence_residual.preflight.v61", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12 if support["passed"] else 0, "synthetic": synthetic_receipt(), "historical_support": support, "v36_features_predictions_used": False, "official_v42_used_for_features_gates_selection": False, "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = BASE.surface_features, BASE.SPECS
    BASE.surface_features, BASE.SPECS = surface_features, SPECS
    try:
        result, arrays = BASE.execute(config)
    finally:
        BASE.surface_features, BASE.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.delay_loop_persistence_residual.result.v61", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_H1_LOOP_PERSISTENCE_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 delay-cloud H1 loop-persistence residual cycle v61", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- v61 computes H1 loop lifetimes; v36/v36r1 H0 MST features and all prior predictions are unused.", "- Official v42 feedback is excluded from features, gates, fitting and selection; the repeated 182-case surface is EXPLORATORY_ONLY."]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; raw {points['raw_gain']:+.6f} points; adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; lead {item['worst_lead_delta_m']:+.9f}m; station-lead {item['worst_station_lead_delta_m']:+.9f}m; tail {item['worst_reference_tail_block_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.")
    lines.append("Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT.exists() or REPORT.exists() or LOCK.exists():
        raise ContractError("v61 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "official_access": 0}))
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    if preflight["status"] == "STOP_SUPPORT_GATE":
        result = {"schema_version": "p3.delay_loop_persistence_residual.result.v61", "experiment_id": EXPERIMENT_ID, "status": "COMPLETE", "decision": "STOP_SUPPORT_GATE_ZERO_FIT", "fit_count": 0, "support_receipt": preflight["historical_support"], "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"], "data_access": {"historical_target_rows": 0, "official_test_rows": 0, "official_sample_rows": 0, "official_submission_rows": 0, "hidden_truth_rows": 0, "csv_materializations": 0, "uploads": 0}, "provenance": {"runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "input_sha256": config["inputs"]}}
        result_path = ARTIFACT / "result.json"
        write_new(result_path, canonical(result))
        write_new(REPORT / "result.json", canonical(result))
        report_path = REPORT / "report-source.md"
        write_new(report_path, b"# P3 delay-cloud H1 loop-persistence residual cycle v61\n\n## Conclusion\n\n- **STOP_SUPPORT_GATE_ZERO_FIT**.\n- Sealed target-free support failed; no target, outer score, official input, CSV, or upload was used.\n")
        write_new(REPORT / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "result_sha256": sha256(result_path), "report_sha256": sha256(report_path), "fit_count": 0, "official_access": 0, "csv_materializations": 0, "uploads": 0}))
        write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Sliding-window loop persistence can represent periodic signal structure | Perea and Harer 2015, DOI:10.1007/s10208-014-9206-z | mechanism only |\n| Sealed target-free support failed before scoring | v61 preflight receipt | zero-fit decision |\n")
        print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 0, "official_access": 0}, ensure_ascii=False))
        return 0
    result, arrays = execute(config)
    array_path = ARTIFACT / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {"runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "evaluation_arrays_sha256": sha256(array_path), "preflight_receipt_sha256": preflight["receipt_sha256"], "input_sha256": config["inputs"]}
    result_path = ARTIFACT / "result.json"
    write_new(result_path, canonical(result))
    report_path = REPORT / "report-source.md"
    write_new(report_path, render_report(result).encode())
    write_new(REPORT / "result.json", canonical(result))
    write_new(REPORT / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "result_sha256": sha256(result_path), "arrays_sha256": sha256(array_path), "report_sha256": sha256(report_path), "fit_count": 12, "official_access": 0, "csv_materializations": 0, "uploads": 0}))
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Sliding-window persistent homology can represent periodic signal structure | Perea and Harer 2015, DOI:10.1007/s10208-014-9206-z | mechanism only |\n| Birth-death persistence pairs follow filtration reduction | Edelsbrunner, Letscher and Zomorodian 2002, DOI:10.1007/s00454-002-2885-2 | algorithm motivation only |\n| v36/v36r1 compute only H0 MST lifetimes, whereas v61 computes H1 edge-triangle loop lifetimes | repository semantic audit | novelty boundary |\n| Official v42 feedback and all prior topology features/predictions were excluded | sealed v61 contract | transport and reuse boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
