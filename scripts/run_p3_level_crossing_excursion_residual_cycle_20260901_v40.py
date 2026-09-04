"""Sealed P3 v40 level-crossing/excursion residual experiment."""

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

import run_p3_multiscale_sample_entropy_residual_cycle_20260901_v39 as v39  # noqa: E402

EXPERIMENT_ID = "p3_level_crossing_excursion_residual_cycle_20260901_v40"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS = (0, 1, 2, 5)
CHANNEL_NAMES = ("hs", "tp", "hmax", "wspd")
LEVELS = (-1.0, 0.0, 1.0)
FEATURE_COUNT = 72
SPECS = (
    v39.v38.v36.v26.Spec("P3_1_LX72_RIDGE512_ADD10", 512.0),
    v39.v38.v36.v26.Spec("P3_2_LX72_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v39.sha256, v39.canonical, v39.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v40 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"]
        == "p3.level_crossing_excursion_residual.config.v40",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_LEVEL_CROSSING_EXCURSION_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "levels": tuple(encoder["levels_iqr_units"]) == LEVELS,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple(
            (item["name"], float(item["ridge_alpha"]))
            for item in config["model"]["candidates"]
        )
        == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(
            float(item["additive_residual_weight"]) == BLEND
            for item in config["model"]["candidates"]
        ),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v40 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def run_lengths(mask: np.ndarray) -> np.ndarray:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return (stops - starts).astype(np.float64)


def level_statistics(values: np.ndarray, level: float) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    previous, current = path[:-1], path[1:]
    up = (previous < level) & (current >= level)
    down = (previous >= level) & (current < level)
    crossing = up | down
    above = path >= level
    lengths = run_lengths(above)
    denominator = float(len(path) - 1)
    mean_run = float(np.mean(lengths) / len(path)) if len(lengths) else 0.0
    maximum_run = float(np.max(lengths) / len(path)) if len(lengths) else 0.0
    crossing_slope = (
        float(np.mean(np.abs(current[crossing] - previous[crossing])))
        if crossing.any()
        else 0.0
    )
    output = np.asarray(
        [
            np.count_nonzero(up) / denominator,
            np.count_nonzero(down) / denominator,
            np.mean(above),
            mean_run,
            maximum_run,
            crossing_slope,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(output).all():
        raise ContractError("level-crossing statistic is nonfinite")
    return output


def level_crossing_features(sequence: np.ndarray) -> np.ndarray:
    path = v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        values = path[:, channel]
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = max(float(q75 - q25), 1e-12)
        standardized = (values - median) / scale
        for level in LEVELS:
            output.extend(level_statistics(standardized, level))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("level-crossing feature contract differs")
    return features


def directional_receipt() -> dict[str, Any]:
    monotone = np.linspace(-2.0, 2.0, 145)
    increasing = level_statistics(monotone, 0.0)
    decreasing = level_statistics(monotone[::-1], 0.0)
    periodic = level_statistics(
        np.sin(2.0 * np.pi * np.arange(145, dtype=np.float64) / 24.0), 0.0
    )
    if not (increasing[0] > 0 and increasing[1] == 0):
        raise ContractError("synthetic upcross recovery failed")
    if not (decreasing[1] > 0 and decreasing[0] == 0):
        raise ContractError("synthetic downcross recovery failed")
    if not (periodic[0] > 0 and periodic[1] > 0):
        raise ContractError("synthetic periodic crossing recovery failed")
    return {
        "monotone_increasing": increasing.tolist(),
        "monotone_decreasing": decreasing.tolist(),
        "periodic": periodic.tolist(),
    }


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = level_crossing_features(sequence)
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "directional": directional_receipt(),
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if np.load(v39.SEQUENCES, mmap_mode="r").shape != (24360, 289, 10):
        raise ContractError("sequence cache shape differs")
    if np.load(v39.STATIONS, mmap_mode="r").shape != (24360,):
        raise ContractError("station cache shape differs")
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v40 exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.level_crossing_excursion_residual.preflight.v40",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2,
        "maximum_model_fits": 12,
        "synthetic": synthetic_receipt(),
        "official_access": 0,
        "csv_materializations": 0,
        "uploads": 0,
        "config_status": config["status"],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(v39.SEQUENCES, mmap_mode="r")
    station_codes = np.load(v39.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = level_crossing_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = v39.surface_features, v39.SPECS
    v39.surface_features, v39.SPECS = surface_features, SPECS
    try:
        result, arrays = v39.execute(config)
    finally:
        v39.surface_features, v39.SPECS = original_surface, original_specs
    result.update(
        {
            "schema_version": "p3.level_crossing_excursion_residual.result.v40",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE"
            if any(item["decision"] != "NO_GO" for item in result["candidates"])
            else "NO_GO_ALL_LEVEL_CROSSING_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 level-crossing/excursion residual cycle v40",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v40 encodes directional threshold crossings and excursion duration/slope geometry; it contains no posterior run-length model, graph, entropy, or prior-cycle prediction.",
        "- Rice (1944) motivates level-crossing analysis only; the 182-case surface remains EXPLORATORY_ONLY.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(
            f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; "
            f"delta {metric['delta_candidate_minus_uniform']:+.9f}m; raw {points['raw_gain']:+.6f} points; "
            f"transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; "
            f"worst block {item['worst_block_delta_m']:+.9f}m; worst lead {item['worst_lead_delta_m']:+.9f}m; "
            f"worst station-lead {item['worst_station_lead_delta_m']:+.9f}m; tail {item['worst_reference_tail_block_delta_m']:+.9f}m; "
            f"episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}."
        )
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
        raise ContractError("v40 exactly-once namespace already exists")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Level-crossing counts characterize random-process threshold passages | Rice, Bell System Technical Journal 23, 1944, DOI:10.1002/j.1538-7305.1944.tb00874.x | mechanism only |\n| v40 has no prior-cycle prediction, router, or posthoc threshold | sealed v40 contract | anti-posthoc boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
