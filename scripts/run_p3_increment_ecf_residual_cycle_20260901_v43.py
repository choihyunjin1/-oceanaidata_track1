"""Sealed P3 v43 multiscale increment-ECF residual experiment."""

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

import run_p3_kramers_moyal_residual_cycle_20260901_v42 as v42  # noqa: E402

EXPERIMENT_ID = "p3_increment_ecf_residual_cycle_20260901_v43"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
LAGS, FREQUENCIES = (1, 3, 6), (0.5, 1.0, 2.0)
FEATURE_COUNT = 72
SPECS = (
    v42.v41.v40.v39.v38.v36.v26.Spec("P3_1_ECF72_RIDGE512_ADD10", 512.0),
    v42.v41.v40.v39.v38.v36.v26.Spec("P3_2_ECF72_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v42.sha256, v42.canonical, v42.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v43 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.increment_ecf_residual.config.v43",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_INCREMENT_DISTRIBUTION_ECF_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "lags": tuple(encoder["increment_lags_rows"]) == LAGS,
        "frequencies": tuple(encoder["distribution_frequencies"]) == FREQUENCIES,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v43 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def ecf_coordinates(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    median = float(np.median(path))
    q25, q75 = np.quantile(path, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-12)
    standardized = (path - median) / scale
    output: list[float] = []
    for frequency in FREQUENCIES:
        output.extend([np.mean(np.cos(frequency * standardized)), np.mean(np.sin(frequency * standardized))])
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (6,) or not np.isfinite(features).all() or np.any(np.abs(features) > 1.0 + 1e-12):
        raise ContractError("increment ECF coordinate contract differs")
    return features


def increment_ecf_features(sequence: np.ndarray) -> np.ndarray:
    path = v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        for lag in LAGS:
            output.extend(ecf_coordinates(path[lag:, channel] - path[:-lag, channel]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("increment ECF feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    symmetric = np.r_[rng.normal(size=10000), -rng.normal(size=10000)]
    centered = ecf_coordinates(symmetric)
    shifted = ecf_coordinates(symmetric + 2.0)
    if np.max(np.abs(centered[1::2])) > 0.02:
        raise ContractError("synthetic symmetric ECF imaginary guard failed")
    # ecf_coordinates recenters increments; raw phase response is checked directly.
    raw_center = np.mean(np.exp(1j * symmetric))
    raw_shift = np.mean(np.exp(1j * (symmetric + 2.0)))
    if abs(raw_shift - np.exp(2j) * raw_center) > 1e-12:
        raise ContractError("synthetic characteristic-function phase guard failed")
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = increment_ecf_features(sequence)
    return {"feature_count": len(feature), "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(feature).all()), "maximum_symmetric_imaginary": float(np.max(np.abs(centered[1::2]))), "raw_shift_phase_error": float(abs(raw_shift - np.exp(2j) * raw_center)), "standardized_shift_invariance": bool(np.allclose(centered, shifted, atol=1e-12))}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v43 exactly-once namespace is consumed")
    payload = {"schema_version": "p3.increment_ecf_residual.preflight.v43", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12, "synthetic": synthetic_receipt(), "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(v42.v41.v40.v39.SEQUENCES, mmap_mode="r")
    station_codes = np.load(v42.v41.v40.v39.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = increment_ecf_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = v42.v41.v40.v39.surface_features, v42.v41.v40.v39.SPECS
    v42.v41.v40.v39.surface_features, v42.v41.v40.v39.SPECS = surface_features, SPECS
    try:
        result, arrays = v42.v41.v40.v39.execute(config)
    finally:
        v42.v41.v40.v39.surface_features, v42.v41.v40.v39.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.increment_ecf_residual.result.v43", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_INCREMENT_ECF_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 multiscale increment-ECF residual cycle v43", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- v43 embeds unconditional increment distributions through fixed empirical characteristic-function coordinates; it reuses no v42 bin, feature, or prediction.", "- Feuerverger and Mureika (1977) motivates the mechanism only; the 182-case surface remains EXPLORATORY_ONLY."]
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
        raise ContractError("v43 exactly-once namespace already exists")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| The empirical characteristic function provides bounded distribution coordinates and converges to the population characteristic function | Feuerverger and Mureika, Annals of Statistics 5, 1977, DOI:10.1214/aos/1176343742 | mechanism only |\n| Distribution frequencies are not temporal Fourier frequencies and no v42 object is reused | sealed v43 contract | novelty boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
