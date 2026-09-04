"""Exactly-once P3 station-graph Laplacian residual cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

import run_p3_kma_wind_work_residual_axis_cycle_20260901_v20 as v20  # noqa: E402
import run_p3_path_signature_residual_cycle_20260901_v23 as v23  # noqa: E402
from run_p3_parallel_candidate_cycle_20260831_v4 import rmse  # noqa: E402
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    POINTS_PER_RMSE_M,
    bootstrap,
)

EXPERIMENT_ID = "p3_station_graph_laplacian_residual_cycle_20260901_v24"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = ROOT / "artifacts/p3/sequences_all20_v1/train_values.npy"
STATIONS = ROOT / "artifacts/p3/sequences_all20_v1/train_station.npy"
STATION_NAMES = ("G-ORS", "I-ORS", "S-ORS")
BLOCKS = v23.BLOCKS
LEADS = v23.LEADS
RAW_NON_DIRECTION = (0, 1, 2, 4, 5, 7, 8, 9)
FEATURE_COUNT = 104
WINSOR = (0.025, 0.975)
RIDGE_ALPHA = 128.0
BLEND = 0.10
TRANSPORT_PENALTY_POINTS = 0.04958605409228893
OFFICIAL_CHAMPION_POINTS = 24.203599


class ContractError(RuntimeError):
    """Raised when the sealed v24 contract differs."""


@dataclass(frozen=True)
class Spec:
    name: str
    graph_strength: float


SPECS = (
    Spec("P3_1_STATION_GRAPH_LAP16_ADD10", 16.0),
    Spec("P3_2_STATION_GRAPH_LAP64_ADD10", 64.0),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "schema": config["schema_version"] == "p3.station_graph_laplacian_residual.config.v24",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_ARCHITECTURE_AXIS",
        "concurrent_stop": config["duplication_audit"]["concurrent_cross_station_propagation"] == "STOP_NO_DEPLOYMENT_CONTRACT",
        "directional_stop": config["duplication_audit"]["directional_wind_sea_wave_age"] == "STOP_SEMANTIC_DUPLICATE",
        "features": config["features"]["feature_count"] == FEATURE_COUNT,
        "ridge": float(config["model"]["ridge_alpha"]) == RIDGE_ALPHA,
        "blend": float(config["model"]["additive_residual_weight"]) == BLEND,
        "specs": tuple((item["name"], float(item["graph_strength"])) for item in config["model"]["candidates"])
        == tuple((item.name, item.graph_strength) for item in SPECS),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
    }
    if not all(checks.values()):
        raise ContractError(f"v24 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def _fill(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(raw)
    result = np.empty_like(raw)
    index = np.arange(len(raw), dtype=np.float64)
    for column in range(raw.shape[1]):
        finite = observed[:, column]
        result[:, column] = np.interp(index, index[finite], raw[finite, column]) if finite.any() else 0.0
    return result, observed


def _slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    return float(np.dot(centered, values - values.mean()) / denominator) if denominator else 0.0


def trajectory_features(sequence: np.ndarray) -> np.ndarray:
    raw = np.asarray(sequence, dtype=np.float64)
    if raw.shape != (289, 10):
        raise ContractError("sequence shape differs")
    direct, direct_observed = _fill(raw[:, RAW_NON_DIRECTION])
    transformed: list[np.ndarray] = [direct[:, 0], direct[:, 1], direct[:, 2]]
    for direction_index in (3,):
        direction = raw[:, direction_index]
        radians = np.deg2rad(direction)
        trig, _ = _fill(np.column_stack([np.sin(radians), np.cos(radians)]))
        transformed.extend([trig[:, 0], trig[:, 1]])
    transformed.extend([direct[:, 3], direct[:, 4]])
    direction = raw[:, 6]
    radians = np.deg2rad(direction)
    trig, _ = _fill(np.column_stack([np.sin(radians), np.cos(radians)]))
    transformed.extend([trig[:, 0], trig[:, 1]])
    transformed.extend([direct[:, 5], direct[:, 6], direct[:, 7]])
    matrix = np.column_stack(transformed)
    if matrix.shape != (289, 12) or not np.isfinite(matrix).all():
        raise ContractError("transformed trajectory differs")
    features: list[float] = []
    for values in matrix.T:
        six = values[-37:]
        day = values[-145:]
        features.extend((values[-1], six.mean(), day.mean(), six.std(), day.std(), _slope(six), _slope(day)))
    raw_observed = np.isfinite(raw)
    for observed in raw_observed.T:
        features.extend((float(1.0 - observed[-37:].mean()), float(1.0 - observed[-145:].mean())))
    result = np.asarray(features, dtype=np.float64)
    if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all():
        raise ContractError("trajectory feature contract differs")
    return result


def haversine_km(one: tuple[float, float], two: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, one)
    lat2, lon2 = map(math.radians, two)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def station_laplacian(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    coordinates = {key: tuple(value) for key, value in config["station_graph"]["coordinates_degrees"].items()}
    weights = np.zeros((3, 3), dtype=np.float64)
    distances = np.zeros((3, 3), dtype=np.float64)
    for i, left in enumerate(STATION_NAMES):
        for j, right in enumerate(STATION_NAMES):
            if i < j:
                distance = haversine_km(coordinates[left], coordinates[right])
                distances[i, j] = distances[j, i] = distance
                weights[i, j] = weights[j, i] = math.exp(-distance / 300.0)
    laplacian = np.diag(weights.sum(axis=1)) - weights
    eigenvalues = np.linalg.eigvalsh(laplacian)
    if not np.allclose(laplacian, laplacian.T) or eigenvalues.min() < -1e-10:
        raise ContractError("station graph Laplacian is not PSD")
    return laplacian, {
        "distances_km": distances.tolist(),
        "edge_weights": weights.tolist(),
        "laplacian_eigenvalues": eigenvalues.tolist(),
        "sha256": hashlib.sha256(laplacian.astype("<f8").tobytes()).hexdigest(),
    }


def synthetic_receipt(config: dict[str, Any]) -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::5, (0, 3, 6)] = np.nan
    feature = trajectory_features(sequence)
    laplacian, graph = station_laplacian(config)
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "graph_sha256": graph["sha256"],
        "graph_trace": float(np.trace(laplacian)),
        "finite": bool(np.isfinite(feature).all()),
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    sequences = np.load(SEQUENCES, mmap_mode="r")
    stations = np.load(STATIONS, mmap_mode="r")
    if sequences.shape != (24360, 289, 10) or stations.shape != (24360,):
        raise ContractError("sequence cache shape differs")
    if ARTIFACT_DIR.exists() or LOCK.exists():
        raise ContractError("exactly-once namespace is already consumed")
    payload = {
        "schema_version": "p3.station_graph_laplacian_residual.preflight.v24",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "sequence_shape": list(sequences.shape),
        "candidate_count": len(SPECS),
        "maximum_model_fits": 12,
        "synthetic": synthetic_receipt(config),
        "official_access": 0,
        "csv_materializations": 0,
        "uploads": 0,
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sequences = np.load(SEQUENCES, mmap_mode="r")
    station_codes = np.load(STATIONS, mmap_mode="r")
    station_map = {name: index for index, name in enumerate(STATION_NAMES)}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    codes = np.empty(len(cases), dtype=np.int8)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        expected = station_map[str(row.station)]
        if int(station_codes[anchor_id]) != expected:
            raise ContractError("sequence station key differs")
        codes[position] = expected
        features[position] = trajectory_features(sequences[anchor_id])
    return features, codes, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "station_counts": {name: int((codes == index).sum()) for index, name in enumerate(STATION_NAMES)},
    }


def fit_predict(features: np.ndarray, station: np.ndarray, residual: np.ndarray, train: np.ndarray, valid: np.ndarray, spec: Spec, laplacian: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = features[train]
    center = np.median(x_train, axis=0)
    q25, q75 = np.quantile(x_train, (0.25, 0.75), axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    train_z = np.clip((x_train - center) / scale, -8.0, 8.0)
    valid_z = np.clip((features[valid] - center) / scale, -8.0, 8.0)
    train_z = np.column_stack([np.ones(len(train_z)), train_z])
    valid_z = np.column_stack([np.ones(len(valid_z)), valid_z])
    p = train_z.shape[1]
    design = np.zeros((len(train), 3 * p), dtype=np.float64)
    for row, code in enumerate(station[train]):
        design[row, int(code) * p : (int(code) + 1) * p] = train_z[row]
    low, high = np.quantile(residual[train], WINSOR, axis=0)
    target = np.clip(residual[train], low, high)
    ridge = np.tile(np.r_[0.0, np.full(p - 1, RIDGE_ALPHA)], 3)
    penalty = np.diag(ridge) + spec.graph_strength * np.kron(laplacian, np.eye(p))
    coefficients = np.linalg.solve(design.T @ design + penalty + 1e-9 * np.eye(3 * p), design.T @ target)
    prediction = np.empty((int(valid.sum()), len(LEADS)), dtype=np.float64)
    for row, code in enumerate(station[valid]):
        block = coefficients[int(code) * p : (int(code) + 1) * p]
        prediction[row] = valid_z[row] @ block
    if not np.isfinite(prediction).all():
        raise ContractError("graph residual prediction is non-finite")
    return prediction, {
        "candidate": spec.name,
        "graph_strength": spec.graph_strength,
        "ridge_alpha": RIDGE_ALPHA,
        "train_cases": len(train),
        "valid_cases": int(valid.sum()),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "row_deletion": 0,
        "fit_count": 1,
    }


def crossfit(cases: pd.DataFrame, features: np.ndarray, station: np.ndarray, targets: np.ndarray, reference: np.ndarray, laplacian: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    residual = targets - reference
    outputs = {spec.name: reference.copy() for spec in SPECS}
    receipts: list[dict[str, Any]] = []
    for block in BLOCKS:
        valid = cases["block"].eq(block).to_numpy()
        train = v23.purged_train_indices(cases, valid)
        for spec in SPECS:
            predicted, receipt = fit_predict(features, station, residual, train, valid, spec, laplacian)
            outputs[spec.name][valid] = np.clip(reference[valid] + BLEND * predicted, 0.0, 30.0)
            receipt.update({"block": block, "additive_residual_weight": BLEND})
            receipts.append(receipt)
    if len(receipts) != 12:
        raise ContractError("fit budget differs")
    return outputs, receipts


def score(frame: pd.DataFrame, prediction: np.ndarray, spec: Spec) -> dict[str, Any]:
    flat = prediction.reshape(-1)
    truth = frame["target_hs"].to_numpy(float)
    reference = frame["reference"].to_numpy(float)
    before, after = rmse(truth, reference), rmse(truth, flat)
    delta = after - before
    by_block = v20.group_deltas(frame, flat, reference, ["block"])
    by_station = v20.group_deltas(frame, flat, reference, ["station"])
    by_lead = v20.group_deltas(frame, flat, reference, ["lead_h"])
    station_lead = v20.group_deltas(frame, flat, reference, ["station", "lead_h"])
    improved = sum(item["delta_rmse_m"] < 0 for item in by_block.values())
    worst_block = max(item["delta_rmse_m"] for item in by_block.values())
    worst_slice = max(item["delta_rmse_m"] for item in station_lead.values())
    offset = SPECS.index(spec) * 100
    episode_ci = bootstrap(frame, flat, ("episode_id",), 20261001 + offset)
    group_ci = bootstrap(frame, flat, ("block", "station"), 20261002 + offset)
    stable_checks = {
        "delta_rmse_negative": delta < 0,
        "minimum_four_improved_blocks": improved >= 4,
        "episode_ci90_upper_below_zero": episode_ci["ci90_m"][1] < 0,
        "block_station_ci90_upper_below_zero": group_ci["ci90_m"][1] < 0,
        "worst_station_lead_at_most_0p01m": worst_slice <= 0.01,
        "finite_predictions": bool(np.isfinite(flat).all()),
    }
    high_risk_checks = {
        "delta_rmse_at_most_minus_0p005m": delta <= -0.005,
        "worst_station_lead_at_most_0p02m": worst_slice <= 0.02,
        "finite_predictions": stable_checks["finite_predictions"],
    }
    stable = all(stable_checks.values())
    high_risk = not stable and all(high_risk_checks.values())
    points = -delta * POINTS_PER_RMSE_M
    return {
        "name": spec.name,
        "decision": "PASS_STABLE" if stable else "PRESERVE_HIGH_RISK" if high_risk else "NO_GO",
        "graph_strength": spec.graph_strength,
        "rmse_m": {"uniform_0p425": before, "candidate": after, "delta_candidate_minus_uniform": delta},
        "expected_points": {
            "raw_gain": points,
            "transport_penalty": TRANSPORT_PENALTY_POINTS,
            "transport_adjusted_gain": points - TRANSPORT_PENALTY_POINTS,
            "nominal_official_score": OFFICIAL_CHAMPION_POINTS + points,
        },
        "improved_blocks": int(improved),
        "by_block": by_block,
        "station": by_station,
        "lead": by_lead,
        "station_lead": station_lead,
        "worst_block_delta_m": worst_block,
        "worst_station_lead_delta_m": worst_slice,
        "episode_bootstrap": episode_ci,
        "block_station_bootstrap": group_ci,
        "stable_checks": stable_checks,
        "high_risk_checks": high_risk_checks,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 station-graph Laplacian residual cycle v24",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- Concurrent cross-station propagation was stopped at 0 fits because anonymous cases lack deployment-time UTC/simultaneity linkage.",
        "- Directional wind-sea/wave-age was stopped at 0 fits as a semantic duplicate of existing sea-state and forcing-conditioned implementations.",
        "- The executed axis shares station-specific residual coefficients through a fixed geographic Laplacian and never reads another station at query time.",
        "- This is EXPLORATORY_ONLY on the repeatedly exposed 182-case surface, not a Public transport guarantee.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; raw {points['raw_gain']:+.6f} points; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; worst station-lead {item['worst_station_lead_delta_m']:+.9f}m.")
        lines.append(f"  - episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.")
    lines.extend(["", "No official test/sample/submission/hidden value was read. No CSV was materialized and no upload occurred. Target winsorization was fit on each outer-training fold, with zero row deletion."])
    return "\n".join(lines) + "\n"


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, profile = v23.case_surface()
    features, station, feature_receipt = surface_features(cases)
    laplacian, graph_receipt = station_laplacian(config)
    predictions, fit_receipts = crossfit(cases, features, station, targets, reference, laplacian)
    frame = v23.long_frame(cases, targets, reference)
    scored = [score(frame, predictions[spec.name], spec) for spec in SPECS]
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {
        "schema_version": "p3.station_graph_laplacian_residual.result.v24",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE" if passing else "NO_GO_ALL_STATION_GRAPH_CANDIDATES",
        "surface_claim": config["validation"]["surface"],
        "reference": config["reference"],
        "duplication_audit": config["duplication_audit"],
        "candidates": scored,
        "fit_receipts": fit_receipts,
        "fit_count": 12,
        "feature_receipt": feature_receipt,
        "graph_receipt": graph_receipt,
        "data_profile": profile,
        "data_access": {"historical_target_rows": len(frame), "official_test_rows": 0, "official_sample_rows": 0, "official_submission_rows": 0, "hidden_truth_rows": 0, "csv_materializations": 0, "uploads": 0},
        "execution": {"python": platform.python_version(), "elapsed_seconds": time.perf_counter() - started, "candidate_count": 2, "result_based_tuning": False, "outer_result_parameter_changes": 0, "row_deletion": 0},
    }
    arrays = {
        "truth": targets,
        "uniform": reference,
        "candidate_1": predictions[SPECS[0].name],
        "candidate_2": predictions[SPECS[1].name],
        "anchor_id": cases["anchor_id"].to_numpy(np.int32),
        "lead_h": np.asarray(LEADS, dtype=np.int16),
        "block": cases["block"].to_numpy(dtype="U5"),
        "station": cases["station"].to_numpy(dtype="U5"),
        "episode": cases["episode_id"].to_numpy(dtype="U32"),
    }
    return result, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT_DIR.exists() or REPORT_DIR.exists() or LOCK.exists():
        raise ContractError("v24 exactly-once namespace already exists")
    config = load_config()
    preflight = preflight_payload()
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "official_access": 0}))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    result, arrays = execute(config)
    array_path = ARTIFACT_DIR / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {"runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "evaluation_arrays_sha256": sha256(array_path), "preflight_receipt_sha256": preflight["receipt_sha256"], "input_sha256": config["inputs"]}
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, render_report(result).encode("utf-8"))
    write_new(REPORT_DIR / "result.json", canonical(result))
    write_new(REPORT_DIR / "gap-matrix.md", (b"# Gap matrix\n\n| Axis | Verdict | Reason |\n|---|---|---|\n| Concurrent cross-station propagation | STOP_NO_DEPLOYMENT_CONTRACT | No case-to-UTC/cross-station simultaneity manifest |\n| Directional wind-sea/wave-age | STOP_SEMANTIC_DUPLICATE | Existing sea-state and forcing-conditioned implementations use the same physical quantities |\n| Station graph coefficient sharing | EXECUTED | Station ID and fixed coordinates are deployable without remote query observations |\n"))
    write_new(REPORT_DIR / "claim-source-ledger.md", (b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Case-to-UTC linkage is absent | `reports/next_action_meta_deep_research_20260831_v1/report-source.md` | Stops concurrent propagation |\n| Inverse wave age already exists | `src/p3_wave/sea_state.py` | Duplicate stop |\n| Directional wind-wave forcing already exists | `configs/experiments/p3_causal_forcing_conditioned_episode_analog_v2.json` | Duplicate stop |\n| Three fixed coordinates exist | `src/p3_wave/era5_pretrain_data.py` | Builds deployment-safe graph |\n"))
    write_new(REPORT_DIR / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "result_sha256": sha256(result_path), "arrays_sha256": sha256(array_path), "report_sha256": sha256(report_path), "fit_count": 12, "official_access": 0, "csv_materializations": 0, "uploads": 0}))
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
