"""Sealed P3 v47 discrete 3D trajectory-geometry residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

import run_p3_triplet_oinformation_residual_cycle_20260901_v46 as v46  # noqa: E402

EXPERIMENT_ID = "p3_discrete_trajectory_geometry_residual_cycle_20260901_v47"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
TRAJECTORIES = ((0, 1, 2), (0, 5, 6), (0, 1, 5), (0, 10, 11))
WINDOWS = (145, 73)
CURVATURE_CLIP = 100.0
FEATURE_COUNT = 72
SPECS = (
    v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_1_GEOM72_RIDGE512_ADD10", 512.0),
    v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_2_GEOM72_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v46.sha256, v46.canonical, v46.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v47 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.discrete_trajectory_geometry_residual.config.v47",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_DISCRETE_3D_TRAJECTORY_GEOMETRY_AXIS",
        "trajectories": tuple(tuple(item["channels"]) for item in encoder["trajectories"]) == TRAJECTORIES,
        "windows": tuple(encoder["windows_rows"]) == WINDOWS,
        "curvature_clip": float(encoder["curvature_clip"]) == CURVATURE_CLIP,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
        "target_free_support": not encoder["support_gate"]["target_used"],
    }
    if not all(checks.values()):
        raise ContractError(f"v47 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def trajectory_statistics(points: np.ndarray) -> np.ndarray:
    path = np.asarray(points, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 8 or not np.isfinite(path).all():
        raise ContractError("trajectory input contract differs")
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    safe_lengths = np.maximum(lengths, 1e-12)
    tangents = segments / safe_lengths[:, None]
    valid_turn = (lengths[:-1] > 1e-12) & (lengths[1:] > 1e-12)
    turns = np.zeros(len(lengths) - 1, dtype=np.float64)
    turns[valid_turn] = np.arccos(np.clip(np.sum(tangents[:-1][valid_turn] * tangents[1:][valid_turn], axis=1), -1.0, 1.0))
    crosses = np.cross(segments[:-1], segments[1:])
    chords = np.linalg.norm(path[2:] - path[:-2], axis=1)
    curvature_denominator = lengths[:-1] * lengths[1:] * chords
    curvature = np.zeros_like(chords)
    valid_curvature = curvature_denominator > 1e-12
    curvature[valid_curvature] = 2.0 * np.linalg.norm(crosses[valid_curvature], axis=1) / curvature_denominator[valid_curvature]
    curvature = np.clip(curvature, 0.0, CURVATURE_CLIP)
    first_normals, second_normals = crosses[:-1], crosses[1:]
    normal_denominator = np.linalg.norm(first_normals, axis=1) * np.linalg.norm(second_normals, axis=1)
    torsion = np.zeros(len(first_normals), dtype=np.float64)
    valid_torsion = normal_denominator > 1e-12
    if valid_torsion.any():
        cosine = np.sum(first_normals[valid_torsion] * second_normals[valid_torsion], axis=1) / normal_denominator[valid_torsion]
        sine = np.sum(np.cross(first_normals[valid_torsion], second_normals[valid_torsion]) * tangents[1:-1][valid_torsion], axis=1) / normal_denominator[valid_torsion]
        torsion[valid_torsion] = np.arctan2(sine, np.clip(cosine, -1.0, 1.0))
    total_length = float(lengths.sum())
    straightness = float(np.linalg.norm(path[-1] - path[0]) / total_length) if total_length > 1e-12 else 0.0
    statistics = np.asarray([
        np.log1p(np.mean(lengths)),
        np.log1p(np.std(lengths)),
        np.mean(turns),
        np.quantile(turns, 0.90),
        np.log1p(np.mean(curvature)),
        np.log1p(np.quantile(curvature, 0.90)),
        np.mean(np.abs(torsion)),
        np.std(torsion),
        straightness,
    ], dtype=np.float64)
    if statistics.shape != (9,) or not np.isfinite(statistics).all() or not 0.0 <= straightness <= 1.0 + 1e-9:
        raise ContractError("trajectory statistic contract differs")
    return statistics


def trajectory_geometry_features(sequence: np.ndarray) -> np.ndarray:
    path = v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for window in WINDOWS:
        current = path[-window:]
        for channels in TRAJECTORIES:
            output.extend(trajectory_statistics(current[:, channels]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("trajectory geometry feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    parameter = np.linspace(0.0, 4.0 * np.pi, 201)
    line = np.column_stack([parameter, np.zeros_like(parameter), np.zeros_like(parameter)])
    circle = np.column_stack([np.cos(parameter), np.sin(parameter), np.zeros_like(parameter)])
    helix = np.column_stack([np.cos(parameter), np.sin(parameter), parameter / (4.0 * np.pi)])
    line_stats, circle_stats, helix_stats = (trajectory_statistics(value) for value in (line, circle, helix))
    if line_stats[4] > 1e-10 or circle_stats[4] < 0.1 or circle_stats[6] > 1e-10 or helix_stats[6] < 1e-4:
        raise ContractError("synthetic line/circle/helix geometry guard failed")
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = trajectory_geometry_features(sequence)
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "line_mean_curvature_log1p": float(line_stats[4]),
        "circle_mean_curvature_log1p": float(circle_stats[4]),
        "circle_mean_abs_torsion": float(circle_stats[6]),
        "helix_mean_abs_torsion": float(helix_stats[6]),
    }


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(v46.v44.v43.v42.v41.v40.v39.SEQUENCES, mmap_mode="r")
    station_codes = np.load(v46.v44.v43.v42.v41.v40.v39.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = trajectory_geometry_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(nonzero_share >= float(gate["minimum_nonzero_share"]) and positive_variance >= int(gate["minimum_positive_variance_features"]))
    if not passed:
        raise ContractError("historical target-free feature support gate failed")
    return {**metadata, "nonzero_share": nonzero_share, "positive_variance_features": positive_variance, "target_used": False, "passed": passed}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v47 exactly-once namespace is consumed")
    payload = {"schema_version": "p3.discrete_trajectory_geometry_residual.preflight.v47", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12, "synthetic": synthetic_receipt(), "historical_support": support_receipt(config), "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = v46.v44.v43.v42.v41.v40.v39.surface_features, v46.v44.v43.v42.v41.v40.v39.SPECS
    v46.v44.v43.v42.v41.v40.v39.surface_features, v46.v44.v43.v42.v41.v40.v39.SPECS = surface_features, SPECS
    try:
        result, arrays = v46.v44.v43.v42.v41.v40.v39.execute(config)
    finally:
        v46.v44.v43.v42.v41.v40.v39.surface_features, v46.v44.v43.v42.v41.v40.v39.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.discrete_trajectory_geometry_residual.result.v47", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_DISCRETE_TRAJECTORY_GEOMETRY_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 discrete trajectory-geometry residual cycle v47", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- v47 measures fixed local bending and twisting of four 3D historical physical-state trajectories; it reuses no v42-v46 output.", "- Muller and Vaxman (2021) motivates the mechanism only; the 182-case surface remains EXPLORATORY_ONLY."]
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
        raise ContractError("v47 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "official_access": 0}))
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Curvature and torsion characterize local bending and twisting of discrete space curves | Muller and Vaxman, Annali di Matematica 200, 2021, DOI:10.1007/s10231-021-01065-x | mechanism only |\n| Trajectories, windows, summaries and all v42-v46 reuse prohibitions were sealed before scoring | sealed v47 contract | novelty boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
