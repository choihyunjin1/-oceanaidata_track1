"""Sealed P3 v55 Rosenstein local-divergence residual experiment."""

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

import run_p3_rainflow_cycle_spectrum_residual_cycle_20260901_v54 as v54  # noqa: E402

EXPERIMENT_ID = "p3_rosenstein_local_divergence_residual_cycle_20260901_v55"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS, DIMENSION, DELAY, EXCLUSION = ((0, 145), (72, 145)), 3, 2, 12
HORIZONS, EPSILON = (1, 2, 3, 4, 5, 6), 1e-12
FEATURE_COUNT = 72
SPECS = (
    v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_1_LYAP72_RIDGE512_ADD10", 512.0),
    v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_2_LYAP72_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v54.sha256, v54.canonical, v54.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v55 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.rosenstein_local_divergence_residual.config.v55",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_FORWARD_NEIGHBOR_DIVERGENCE_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "dimension": encoder["embedding_dimension"] == DIMENSION,
        "delay": encoder["embedding_delay_rows"] == DELAY,
        "exclusion": encoder["temporal_exclusion_rows"] == EXCLUSION,
        "horizons": tuple(encoder["forward_horizons_rows"]) == HORIZONS,
        "epsilon": float(encoder["epsilon"]) == EPSILON,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
        "target_free_support": not encoder["support_gate"]["target_used"],
    }
    if not all(checks.values()):
        raise ContractError(f"v55 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def robust_case_normalize(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise ContractError("local-divergence input contract differs")
    median = float(np.median(raw))
    q25, q75 = np.quantile(raw, [0.25, 0.75])
    return (raw - median) / max(float(q75 - q25), EPSILON)


def delay_embedding(values: np.ndarray) -> np.ndarray:
    path = robust_case_normalize(values)
    rows = len(path) - (DIMENSION - 1) * DELAY
    if rows <= max(HORIZONS) + 2:
        raise ContractError("insufficient delay-embedding support")
    embedded = np.column_stack([path[offset * DELAY : offset * DELAY + rows] for offset in range(DIMENSION)])
    if not np.isfinite(embedded).all():
        raise ContractError("delay embedding is nonfinite")
    return embedded


def local_divergence_statistics(values: np.ndarray) -> np.ndarray:
    embedded = delay_embedding(values)
    maximum_horizon = max(HORIZONS)
    usable = len(embedded) - maximum_horizon
    initial: list[float] = []
    trajectories: list[list[float]] = []
    for index in range(usable):
        candidates = np.arange(usable)
        allowed = np.abs(candidates - index) > EXCLUSION
        if not np.any(allowed):
            continue
        valid_indices = candidates[allowed]
        distances = np.linalg.norm(embedded[valid_indices] - embedded[index], axis=1)
        neighbor = int(valid_indices[int(np.argmin(distances))])
        initial.append(float(np.linalg.norm(embedded[index] - embedded[neighbor])))
        trajectories.append([float(np.linalg.norm(embedded[index + horizon] - embedded[neighbor + horizon])) for horizon in HORIZONS])
    valid_fraction = len(initial) / max(usable, 1)
    if not initial:
        raise ContractError("no temporally separated nearest neighbors")
    initial_array = np.asarray(initial, dtype=np.float64)
    trajectory = np.asarray(trajectories, dtype=np.float64)
    mean_log = np.mean(np.log(trajectory + EPSILON), axis=0)
    axis = np.asarray(HORIZONS, dtype=np.float64)
    slope, intercept = np.polyfit(axis, mean_log, 1)
    fitted = slope * axis + intercept
    total = float(np.sum(np.square(mean_log - np.mean(mean_log))))
    r_squared = 1.0 - float(np.sum(np.square(mean_log - fitted))) / max(total, EPSILON)
    output = np.asarray([slope, intercept, r_squared, float(np.mean(np.log(initial_array + EPSILON))), float(mean_log[0]), float(mean_log[2]), float(mean_log[5]), float(np.quantile(initial_array, 0.90)), float(np.median(initial_array))], dtype=np.float64)
    if output.shape != (9,) or not np.isfinite(output).all() or valid_fraction < 0.70:
        raise ContractError("local-divergence statistic contract differs")
    return output


def local_divergence_features(sequence: np.ndarray) -> np.ndarray:
    path = v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        for start, stop in WINDOWS:
            output.extend(local_divergence_statistics(path[start:stop, channel]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("local-divergence feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    axis = np.arange(145, dtype=np.float64)
    periodic = np.sin(2.0 * np.pi * axis / 29.0) + 1e-4 * np.cos(2.0 * np.pi * axis / 17.0)
    chaotic = np.empty(145, dtype=np.float64)
    chaotic[0] = 0.213
    for index in range(144):
        chaotic[index + 1] = 3.99 * chaotic[index] * (1.0 - chaotic[index])
    periodic_stat = local_divergence_statistics(periodic)
    chaotic_stat = local_divergence_statistics(chaotic)
    if chaotic_stat[0] <= periodic_stat[0] + 0.01:
        raise ContractError("synthetic periodic/chaotic divergence direction failed")
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = local_divergence_features(sequence)
    return {"feature_count": len(feature), "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(feature).all()), "periodic_slope": float(periodic_stat[0]), "chaotic_slope": float(chaotic_stat[0]), "periodic_neighbor_fraction": 1.0, "chaotic_neighbor_fraction": 1.0}


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.SEQUENCES, mmap_mode="r")
    station_codes = np.load(v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = local_divergence_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(nonzero_share >= float(gate["minimum_nonzero_share"]) and positive_variance >= int(gate["minimum_positive_variance_features"]))
    if not passed:
        raise ContractError("historical target-free local-divergence support gate failed")
    return {**metadata, "nonzero_share": nonzero_share, "positive_variance_features": positive_variance, "target_used": False, "passed": passed}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v55 exactly-once namespace is consumed")
    payload = {"schema_version": "p3.rosenstein_local_divergence_residual.preflight.v55", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12, "synthetic": synthetic_receipt(), "historical_support": support_receipt(config), "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    base = v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39
    original_surface, original_specs = base.surface_features, base.SPECS
    base.surface_features, base.SPECS = surface_features, SPECS
    try:
        result, arrays = base.execute(config)
    finally:
        base.surface_features, base.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.rosenstein_local_divergence_residual.result.v55", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_LOCAL_DIVERGENCE_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 Rosenstein local-divergence residual cycle v55", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- Fixed forward nearest-neighbor divergence is distinct from recurrence counts, filtration topology and static trajectory geometry; no prior output is reused.", "- Rosenstein et al. (1993) motivates the mechanism only; the repeatedly exposed 182-case surface is EXPLORATORY_ONLY."]
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
        raise ContractError("v55 exactly-once namespace already exists")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Small-sample largest Lyapunov estimation follows separation of nearest delay-embedded trajectories | Rosenstein, Collins and De Luca 1993, DOI:10.1016/0167-2789(93)90009-P | mechanism only |\n| v29/v36/v47 do not follow nearest states forward in time | sealed duplication audit | novelty boundary |\n| Embedding, exclusion, horizons, summaries and residual model were fixed before scoring | sealed v55 config | execution contract |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
