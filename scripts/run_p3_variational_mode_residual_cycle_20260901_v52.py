"""Sealed P3 v52 fixed variational-mode residual experiment."""

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

import run_p3_multiscale_teager_energy_residual_cycle_20260901_v50 as v50  # noqa: E402

EXPERIMENT_ID = "p3_variational_mode_residual_cycle_20260901_v52"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
MODE_COUNT, BANDWIDTH_PENALTY, ITERATIONS = 3, 800.0, 40
INITIAL_CENTERS = np.asarray([0.03, 0.10, 0.22], dtype=np.float64)
FEATURE_COUNT = 72
SPECS = (
    v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_1_VMD72_RIDGE512_ADD10", 512.0),
    v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_2_VMD72_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v50.sha256, v50.canonical, v50.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v52 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.variational_mode_residual.config.v52",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_JOINT_VARIATIONAL_NARROWBAND_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "modes": encoder["mode_count"] == MODE_COUNT,
        "penalty": float(encoder["bandwidth_penalty"]) == BANDWIDTH_PENALTY,
        "iterations": encoder["iterations"] == ITERATIONS,
        "initial": np.array_equal(np.asarray(encoder["initial_center_frequencies"]), INITIAL_CENTERS),
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
        "target_free_support": not encoder["support_gate"]["target_used"],
    }
    if not all(checks.values()):
        raise ContractError(f"v52 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def robust_case_normalize(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or len(raw) != 145 or not np.isfinite(raw).all():
        raise ContractError("VMD input contract differs")
    median = float(np.median(raw))
    q25, q75 = np.quantile(raw, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-12)
    return (raw - median) / scale


def fixed_variational_modes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = robust_case_normalize(values)
    spectrum = np.fft.rfft(normalized)
    frequencies = np.fft.rfftfreq(len(normalized))
    centers = INITIAL_CENTERS.copy()
    mode_spectra = np.zeros((MODE_COUNT, len(frequencies)), dtype=np.complex128)
    for _ in range(ITERATIONS):
        for mode in range(MODE_COUNT):
            residual_spectrum = spectrum - (np.sum(mode_spectra, axis=0) - mode_spectra[mode])
            mode_spectra[mode] = residual_spectrum / (1.0 + BANDWIDTH_PENALTY * np.square(frequencies - centers[mode]))
            weighted_power = np.square(np.abs(mode_spectra[mode]))
            denominator = float(np.sum(weighted_power))
            if denominator > 1e-12:
                centers[mode] = float(np.sum(frequencies * weighted_power) / denominator)
    order = np.argsort(centers)
    centers = centers[order]
    mode_spectra = mode_spectra[order]
    modes = np.asarray([np.fft.irfft(mode_spectra[index], n=len(normalized)) for index in range(MODE_COUNT)])
    residual = normalized - np.sum(modes, axis=0)
    if modes.shape != (MODE_COUNT, 145) or not np.isfinite(modes).all() or not np.isfinite(centers).all() or not np.isfinite(residual).all():
        raise ContractError("fixed variational decomposition failed")
    return modes, centers, residual


def mode_features(values: np.ndarray) -> np.ndarray:
    modes, centers, residual = fixed_variational_modes(values)
    total_energy = max(float(np.sum(np.square(modes)) + np.sum(np.square(residual))), 1e-12)
    frequencies = np.fft.rfftfreq(modes.shape[1])
    midpoint = modes.shape[1] // 2
    output: list[float] = []
    for index, mode in enumerate(modes):
        spectrum_power = np.square(np.abs(np.fft.rfft(mode)))
        denominator = max(float(np.sum(spectrum_power)), 1e-12)
        bandwidth = float(np.sqrt(np.sum(np.square(frequencies - centers[index]) * spectrum_power) / denominator))
        early = float(np.mean(np.square(mode[:midpoint])))
        recent = float(np.mean(np.square(mode[midpoint:])))
        output.extend([float(np.sum(np.square(mode)) / total_energy), float(centers[index]), bandwidth, float(np.mean(np.abs(mode))), float(np.quantile(np.abs(mode), 0.90)), float(np.log((recent + 1e-8) / (early + 1e-8)))])
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (18,) or not np.isfinite(features).all():
        raise ContractError("VMD feature contract differs")
    return features


def variational_mode_features(sequence: np.ndarray) -> np.ndarray:
    path = v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    features = np.concatenate([mode_features(path[:, channel]) for channel in CHANNELS])
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("variational-mode feature contract differs")
    return features


def two_tone_receipt() -> dict[str, Any]:
    axis = np.arange(145, dtype=np.float64)
    signal = np.sin(2.0 * np.pi * 0.08 * axis) + 0.55 * np.sin(2.0 * np.pi * 0.22 * axis + 0.3)
    modes, centers, residual = fixed_variational_modes(signal)
    reconstructed = np.sum(modes, axis=0) + residual
    distances = [float(np.min(np.abs(centers - target))) for target in (0.08, 0.22)]
    if max(distances) > 0.035 or np.max(np.abs(reconstructed - robust_case_normalize(signal))) > 1e-12:
        raise ContractError("synthetic two-tone VMD guard failed")
    return {"centers": centers.tolist(), "target_frequency_distances": distances, "reconstruction_max_abs_error": float(np.max(np.abs(reconstructed - robust_case_normalize(signal)))), "finite": bool(np.isfinite(modes).all())}


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = variational_mode_features(sequence)
    return {"feature_count": len(feature), "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(feature).all()), "two_tone": two_tone_receipt()}


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.SEQUENCES, mmap_mode="r")
    station_codes = np.load(v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = variational_mode_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(nonzero_share >= float(gate["minimum_nonzero_share"]) and positive_variance >= int(gate["minimum_positive_variance_features"]))
    if not passed:
        raise ContractError("historical target-free VMD support gate failed")
    return {**metadata, "nonzero_share": nonzero_share, "positive_variance_features": positive_variance, "target_used": False, "passed": passed}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v52 exactly-once namespace is consumed")
    payload = {"schema_version": "p3.variational_mode_residual.preflight.v52", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12, "synthetic": synthetic_receipt(), "historical_support": support_receipt(config), "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.surface_features, v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.SPECS
    v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.surface_features, v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.SPECS = surface_features, SPECS
    try:
        result, arrays = v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.execute(config)
    finally:
        v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.surface_features, v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.variational_mode_residual.result.v52", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_VARIATIONAL_MODE_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 variational-mode residual cycle v52", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- Fixed simultaneous variational narrow-band modes are distinct from recursive EMD/Hilbert and fixed wavelet banks; no prior output is reused.", "- Dragomiretskiy and Zosso (2014) motivates the mechanism only; the repeatedly exposed 182-case surface is EXPLORATORY_ONLY."]
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
        raise ContractError("v52 exactly-once namespace already exists")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| VMD jointly estimates constrained narrow-band modes | Dragomiretskiy and Zosso 2014, DOI:10.1109/TSP.2013.2288675 | mechanism only |\n| P3 v35 uses recursive extrema-envelope EMD and no output is reused | sealed duplication audit | novelty boundary |\n| K, penalty, iterations, initialization, summaries and residual model were fixed before scoring | sealed v52 config | execution contract |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
