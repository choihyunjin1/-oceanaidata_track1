"""Sealed P3 v41 fixed echo-state residual experiment."""

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

import run_p3_level_crossing_excursion_residual_cycle_20260901_v40 as v40  # noqa: E402

EXPERIMENT_ID = "p3_fixed_echo_state_residual_cycle_20260901_v41"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
STATE_COUNT, INPUT_COUNT, SEED = 32, 12, 20260901
SPECTRAL_RADIUS, LEAK, INPUT_SCALE = 0.90, 0.30, 0.25
FEATURE_COUNT = 96
SPECS = (
    v40.v39.v38.v36.v26.Spec("P3_1_ESN96_RIDGE512_ADD10", 512.0),
    v40.v39.v38.v36.v26.Spec("P3_2_ESN96_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v40.sha256, v40.canonical, v40.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v41 contract differs."""


def reservoir_matrices() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    recurrent = rng.normal(size=(STATE_COUNT, STATE_COUNT))
    radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
    recurrent *= SPECTRAL_RADIUS / radius
    inputs = INPUT_SCALE * rng.normal(size=(STATE_COUNT, INPUT_COUNT)) / np.sqrt(INPUT_COUNT)
    return recurrent.astype(np.float64), inputs.astype(np.float64)


RECURRENT, INPUTS = reservoir_matrices()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.fixed_echo_state_residual.config.v41",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_FIXED_NONLINEAR_RECURRENT_STATE_AXIS",
        "shape": (encoder["input_channels"], encoder["reservoir_states"])
        == (INPUT_COUNT, STATE_COUNT),
        "seed": encoder["seed"] == SEED,
        "radius": encoder["spectral_radius"] == SPECTRAL_RADIUS,
        "leak": encoder["leak_rate"] == LEAK,
        "input_scale": encoder["input_scale"] == INPUT_SCALE,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"])
        == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v41 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def standardized_path(sequence: np.ndarray) -> np.ndarray:
    path = v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, INPUT_COUNT):
        raise ContractError("fixed 20-minute path differs")
    median = np.median(path, axis=0)
    q25, q75 = np.quantile(path, [0.25, 0.75], axis=0)
    scale = np.where(q75 - q25 > 1e-12, q75 - q25, 1.0)
    return np.clip((path - median) / scale, -8.0, 8.0)


def reservoir_states(path: np.ndarray) -> np.ndarray:
    state = np.zeros(STATE_COUNT, dtype=np.float64)
    states = np.empty((len(path), STATE_COUNT), dtype=np.float64)
    for index, row in enumerate(path):
        proposal = np.tanh(RECURRENT @ state + INPUTS @ row)
        state = (1.0 - LEAK) * state + LEAK * proposal
        states[index] = state
    if not np.isfinite(states).all():
        raise ContractError("echo states are nonfinite")
    return states


def echo_state_features(sequence: np.ndarray) -> np.ndarray:
    states = reservoir_states(standardized_path(sequence))
    features = np.concatenate([states[-1], states.mean(axis=0), states.std(axis=0)])
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("echo-state feature contract differs")
    return features


def fading_memory_receipt() -> dict[str, Any]:
    base = np.zeros((145, INPUT_COUNT), dtype=np.float64)
    perturbed = base.copy()
    perturbed[0, 0] = 1.0
    difference = np.linalg.norm(reservoir_states(perturbed) - reservoir_states(base), axis=1)
    radius = float(np.max(np.abs(np.linalg.eigvals(RECURRENT))))
    if abs(radius - SPECTRAL_RADIUS) > 1e-12:
        raise ContractError("reservoir spectral radius differs")
    if not (difference[0] > 0 and difference[-1] < difference[0] * 1e-3):
        raise ContractError("synthetic fading-memory guard failed")
    return {
        "operator_spectral_radius": radius,
        "initial_impulse_difference": float(difference[0]),
        "terminal_impulse_difference": float(difference[-1]),
        "terminal_to_initial_ratio": float(difference[-1] / difference[0]),
    }


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = echo_state_features(sequence)
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "fading_memory": fading_memory_receipt(),
        "recurrent_sha256": hashlib.sha256(RECURRENT.astype("<f8").tobytes()).hexdigest(),
        "input_sha256": hashlib.sha256(INPUTS.astype("<f8").tobytes()).hexdigest(),
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if np.load(v40.v39.SEQUENCES, mmap_mode="r").shape != (24360, 289, 10):
        raise ContractError("sequence cache shape differs")
    if np.load(v40.v39.STATIONS, mmap_mode="r").shape != (24360,):
        raise ContractError("station cache shape differs")
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v41 exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.fixed_echo_state_residual.preflight.v41",
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
    sequences = np.load(v40.v39.SEQUENCES, mmap_mode="r")
    station_codes = np.load(v40.v39.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = echo_state_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = v40.v39.surface_features, v40.v39.SPECS
    v40.v39.surface_features, v40.v39.SPECS = surface_features, SPECS
    try:
        result, arrays = v40.v39.execute(config)
    finally:
        v40.v39.surface_features, v40.v39.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.fixed_echo_state_residual.result.v41", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_FIXED_ESN_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 fixed echo-state residual cycle v41", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- v41 uses one fixed nonlinear recurrent reservoir and fits only the residual readout; it reuses no v40 prediction or feature.", "- Jaeger (2001) motivates the mechanism only; the 182-case surface remains EXPLORATORY_ONLY."]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; raw {points['raw_gain']:+.6f} points; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; worst lead {item['worst_lead_delta_m']:+.9f}m; station-lead {item['worst_station_lead_delta_m']:+.9f}m; tail {item['worst_reference_tail_block_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.")
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
        raise ContractError("v41 exactly-once namespace already exists")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Fixed-weight reservoirs provide nonlinear recurrent state while fitting only a readout | Jaeger, GMD Report 148, 2001, DOI:10.24406/publica-fhg-291111 | mechanism only |\n| P3 has no prior executed ESN and v41 imports no P1 result or hyperparameter | repository semantic audit and sealed v41 contract | novelty boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
