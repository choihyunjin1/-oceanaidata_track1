"""Exactly-once P3 deterministic multichannel ROCKET residual cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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

EXPERIMENT_ID = "p3_causal_multichannel_rocket_residual_cycle_20260901_v26"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = ROOT / "artifacts/p3/sequences_all20_v1/train_values.npy"
STATIONS = ROOT / "artifacts/p3/sequences_all20_v1/train_station.npy"
BLOCKS = v23.BLOCKS
LEADS = v23.LEADS
KERNEL_COUNT = 256
FEATURE_COUNT = 512
KERNEL_SEED = 20260901
LENGTHS = (7, 9, 11)
DILATIONS = (1, 2, 4, 8, 16, 32)
CENTER = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 70, 1013), dtype=np.float64)
SCALE = np.asarray((5, 20, 10, 1, 1, 30, 40, 1, 1, 30, 50, 50), dtype=np.float64)
WINSOR = (0.025, 0.975)
BLEND = 0.10
TRANSPORT_PENALTY_POINTS = 0.04958605409228893
OFFICIAL_CHAMPION_POINTS = 24.203599


class ContractError(RuntimeError):
    """Raised when the sealed v26 contract differs."""


@dataclass(frozen=True)
class Spec:
    name: str
    alpha: float


@dataclass(frozen=True)
class Kernel:
    channels: tuple[int, int, int]
    dilation: int
    weights: np.ndarray


SPECS = (
    Spec("P3_1_ROCKET256_RIDGE512_ADD10", 512.0),
    Spec("P3_2_ROCKET256_RIDGE2048_ADD10", 2048.0),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "schema": config["schema_version"] == "p3.causal_multichannel_rocket_residual.config.v26",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_REPRESENTATION_AXIS",
        "seed": config["rocket"]["kernel_seed"] == KERNEL_SEED,
        "kernels": config["rocket"]["kernel_count"] == KERNEL_COUNT,
        "features": config["rocket"]["feature_count"] == FEATURE_COUNT,
        "lengths": tuple(config["rocket"]["kernel_lengths"]) == LENGTHS,
        "dilations": tuple(config["rocket"]["dilations"]) == DILATIONS,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
    }
    if not all(checks.values()):
        raise ContractError(f"v26 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def kernel_bank() -> tuple[Kernel, ...]:
    rng = np.random.default_rng(KERNEL_SEED)
    kernels: list[Kernel] = []
    for _ in range(KERNEL_COUNT):
        length = int(rng.choice(LENGTHS))
        dilation = int(rng.choice([value for value in DILATIONS if (length - 1) * value < 289]))
        channels = tuple(sorted(int(value) for value in rng.choice(12, size=3, replace=False)))
        weights = rng.normal(size=(3, length)).astype(np.float64)
        weights -= weights.mean()
        norm = float(np.linalg.norm(weights))
        if norm < 1e-12:
            raise ContractError("degenerate fixed kernel")
        weights /= norm
        kernels.append(Kernel(channels, dilation, weights))
    return tuple(kernels)


def kernel_receipt(kernels: tuple[Kernel, ...]) -> dict[str, Any]:
    payload = bytearray()
    for kernel in kernels:
        payload.extend(np.asarray(kernel.channels, dtype="<i2").tobytes())
        payload.extend(np.asarray([kernel.dilation], dtype="<i2").tobytes())
        payload.extend(kernel.weights.astype("<f8").tobytes())
    return {"count": len(kernels), "sha256": hashlib.sha256(payload).hexdigest(), "length_counts": {str(length): sum(kernel.weights.shape[1] == length for kernel in kernels) for length in LENGTHS}, "dilation_counts": {str(dilation): sum(kernel.dilation == dilation for kernel in kernels) for dilation in DILATIONS}}


def _fill(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    result = np.empty_like(raw)
    index = np.arange(len(raw), dtype=np.float64)
    for column in range(raw.shape[1]):
        finite = np.isfinite(raw[:, column])
        result[:, column] = np.interp(index, index[finite], raw[finite, column]) if finite.any() else 0.0
    return result


def transformed_path(sequence: np.ndarray) -> np.ndarray:
    raw = np.asarray(sequence, dtype=np.float64)
    if raw.shape != (289, 10):
        raise ContractError("raw context shape differs")
    direct = _fill(raw[:, (0, 1, 2, 4, 5, 7, 8, 9)])
    wave_angle = np.deg2rad(raw[:, 3])
    wind_angle = np.deg2rad(raw[:, 6])
    wave = _fill(np.column_stack([np.sin(wave_angle), np.cos(wave_angle)]))
    wind = _fill(np.column_stack([np.sin(wind_angle), np.cos(wind_angle)]))
    matrix = np.column_stack([direct[:, 0], direct[:, 1], direct[:, 2], wave, direct[:, 3], direct[:, 4], wind, direct[:, 5], direct[:, 6], direct[:, 7]])
    matrix = (matrix - CENTER) / SCALE
    if matrix.shape != (289, 12) or not np.isfinite(matrix).all():
        raise ContractError("transformed path differs")
    return matrix


def rocket_features(sequence: np.ndarray, kernels: tuple[Kernel, ...]) -> np.ndarray:
    path = transformed_path(sequence)
    features = np.empty(FEATURE_COUNT, dtype=np.float64)
    for index, kernel in enumerate(kernels):
        length = kernel.weights.shape[1]
        stop = len(path) - (length - 1) * kernel.dilation
        response = np.zeros(stop, dtype=np.float64)
        for offset in range(length):
            rows = path[offset * kernel.dilation : offset * kernel.dilation + stop, kernel.channels]
            response += rows @ kernel.weights[:, offset]
        features[2 * index] = np.mean(response > 0.0)
        features[2 * index + 1] = np.max(response)
    if not np.isfinite(features).all():
        raise ContractError("ROCKET features are non-finite")
    return features


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    kernels = kernel_bank()
    feature = rocket_features(sequence, kernels)
    return {"kernel_bank": kernel_receipt(kernels), "feature_count": len(feature), "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(feature).all())}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    sequences = np.load(SEQUENCES, mmap_mode="r")
    stations = np.load(STATIONS, mmap_mode="r")
    if sequences.shape != (24360, 289, 10) or stations.shape != (24360,):
        raise ContractError("sequence cache shape differs")
    if ARTIFACT_DIR.exists() or LOCK.exists():
        raise ContractError("exactly-once namespace is already consumed")
    payload = {"schema_version": "p3.causal_multichannel_rocket_residual.preflight.v26", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12, "synthetic": synthetic_receipt(), "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def surface_features(cases: pd.DataFrame, kernels: tuple[Kernel, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(SEQUENCES, mmap_mode="r")
    station_codes = np.load(STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = rocket_features(sequences[anchor_id], kernels)
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def fit_predict(features: np.ndarray, residual: np.ndarray, train: np.ndarray, valid: np.ndarray, spec: Spec) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = features[train]
    center = np.median(x_train, axis=0)
    q25, q75 = np.quantile(x_train, (0.25, 0.75), axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    train_z = np.clip((x_train - center) / scale, -8.0, 8.0)
    valid_z = np.clip((features[valid] - center) / scale, -8.0, 8.0)
    low, high = np.quantile(residual[train], WINSOR, axis=0)
    target = np.clip(residual[train], low, high)
    model = Ridge(alpha=spec.alpha, fit_intercept=True, solver="cholesky")
    model.fit(train_z, target)
    prediction = np.asarray(model.predict(valid_z), dtype=np.float64)
    if prediction.shape != (int(valid.sum()), len(LEADS)) or not np.isfinite(prediction).all():
        raise ContractError("joint residual prediction differs")
    return prediction, {"candidate": spec.name, "ridge_alpha": spec.alpha, "train_cases": len(train), "valid_cases": int(valid.sum()), "target_winsor": list(WINSOR), "coefficient_l2": float(np.linalg.norm(model.coef_)), "row_deletion": 0, "fit_count": 1}


def crossfit(cases: pd.DataFrame, features: np.ndarray, targets: np.ndarray, reference: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    residual = targets - reference
    outputs = {spec.name: reference.copy() for spec in SPECS}
    receipts: list[dict[str, Any]] = []
    for block in BLOCKS:
        valid = cases["block"].eq(block).to_numpy()
        train = v23.purged_train_indices(cases, valid)
        for spec in SPECS:
            predicted, receipt = fit_predict(features, residual, train, valid, spec)
            outputs[spec.name][valid] = np.clip(reference[valid] + BLEND * predicted, 0.0, 30.0)
            receipt.update({"block": block, "additive_residual_weight": BLEND})
            receipts.append(receipt)
    if len(receipts) != 12:
        raise ContractError("fit budget differs")
    return outputs, receipts


def score(frame: pd.DataFrame, prediction: np.ndarray, spec: Spec) -> dict[str, Any]:
    flat, truth, reference = prediction.reshape(-1), frame["target_hs"].to_numpy(float), frame["reference"].to_numpy(float)
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
    episode_ci = bootstrap(frame, flat, ("episode_id",), 20261021 + offset)
    group_ci = bootstrap(frame, flat, ("block", "station"), 20261022 + offset)
    stable_checks = {"delta_rmse_negative": delta < 0, "minimum_four_improved_blocks": improved >= 4, "episode_ci90_upper_below_zero": episode_ci["ci90_m"][1] < 0, "block_station_ci90_upper_below_zero": group_ci["ci90_m"][1] < 0, "worst_station_lead_at_most_0p01m": worst_slice <= 0.01, "finite_predictions": bool(np.isfinite(flat).all())}
    high_risk_checks = {"delta_rmse_at_most_minus_0p005m": delta <= -0.005, "worst_station_lead_at_most_0p02m": worst_slice <= 0.02, "finite_predictions": stable_checks["finite_predictions"]}
    stable = all(stable_checks.values())
    high_risk = not stable and all(high_risk_checks.values())
    points = -delta * POINTS_PER_RMSE_M
    return {"name": spec.name, "decision": "PASS_STABLE" if stable else "PRESERVE_HIGH_RISK" if high_risk else "NO_GO", "ridge_alpha": spec.alpha, "additive_residual_weight": BLEND, "rmse_m": {"uniform_0p425": before, "candidate": after, "delta_candidate_minus_uniform": delta}, "expected_points": {"raw_gain": points, "transport_penalty": TRANSPORT_PENALTY_POINTS, "transport_adjusted_gain": points - TRANSPORT_PENALTY_POINTS, "nominal_official_score": OFFICIAL_CHAMPION_POINTS + points}, "improved_blocks": int(improved), "by_block": by_block, "station": by_station, "lead": by_lead, "station_lead": station_lead, "worst_block_delta_m": worst_block, "worst_station_lead_delta_m": worst_slice, "episode_bootstrap": episode_ci, "block_station_bootstrap": group_ci, "stable_checks": stable_checks, "high_risk_checks": high_risk_checks}


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 deterministic multichannel ROCKET residual cycle v26", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- The representation is novel within the P3 ledger: fixed, untrained dilated multichannel kernels summarized only by PPV and maximum response.", "- This is EXPLORATORY_ONLY on the repeatedly exposed 182-case surface, not a Public transport guarantee."]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; raw {points['raw_gain']:+.6f} points; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; worst station-lead {item['worst_station_lead_delta_m']:+.9f}m.")
        lines.append(f"  - episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.")
    lines.extend(["", "No official test/sample/submission/hidden value was read. No CSV was materialized and no upload occurred. No row was deleted and no outer result changed the kernel bank, Ridge strengths, or residual blend."])
    return "\n".join(lines) + "\n"


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, profile = v23.case_surface()
    kernels = kernel_bank()
    features, feature_receipt = surface_features(cases, kernels)
    predictions, receipts = crossfit(cases, features, targets, reference)
    frame = v23.long_frame(cases, targets, reference)
    scored = [score(frame, predictions[spec.name], spec) for spec in SPECS]
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {"schema_version": "p3.causal_multichannel_rocket_residual.result.v26", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "PASS_CANDIDATE_AVAILABLE" if passing else "NO_GO_ALL_ROCKET_CANDIDATES", "surface_claim": config["validation"]["surface"], "reference": config["reference"], "duplication_audit": config["duplication_audit"], "kernel_receipt": kernel_receipt(kernels), "feature_receipt": feature_receipt, "candidates": scored, "fit_receipts": receipts, "fit_count": 12, "data_profile": profile, "data_access": {"historical_target_rows": 1092, "official_test_rows": 0, "official_sample_rows": 0, "official_submission_rows": 0, "hidden_truth_rows": 0, "csv_materializations": 0, "uploads": 0}, "execution": {"python": platform.python_version(), "elapsed_seconds": time.perf_counter() - started, "candidate_count": 2, "result_based_tuning": False, "outer_result_parameter_changes": 0, "row_deletion": 0}}
    arrays = {"truth": targets, "uniform": reference, "candidate_1": predictions[SPECS[0].name], "candidate_2": predictions[SPECS[1].name], "anchor_id": cases["anchor_id"].to_numpy(np.int32), "lead_h": np.asarray(LEADS, dtype=np.int16), "block": cases["block"].to_numpy(dtype="U5"), "station": cases["station"].to_numpy(dtype="U5"), "episode": cases["episode_id"].to_numpy(dtype="U32")}
    return result, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT_DIR.exists() or REPORT_DIR.exists() or LOCK.exists():
        raise ContractError("v26 exactly-once namespace already exists")
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
    write_new(report_path, render_report(result).encode())
    write_new(REPORT_DIR / "result.json", canonical(result))
    write_new(REPORT_DIR / "gap-matrix.md", b"# Gap matrix\n\n| Family | Difference from v26 | Verdict |\n|---|---|---|\n| TCN/PatchTST/TimeXer | Learned encoder weights | Distinct |\n| Causal spectral | Global Fourier/random-frequency amplitudes | Distinct |\n| Path signature | Global ordered iterated integrals | Distinct |\n| v26 ROCKET | Fixed local dilated kernels with PPV/max | Executed |\n")
    write_new(REPORT_DIR / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Prior learned sequence families exist | `reports/historical_model_reaudit_20260831_v1/model-cards/p3-neural-sequence-revin-ssl.md` | Semantic distinction |\n| Causal spectral exact family exists | `reports/historical_model_reaudit_20260831_v1/candidate-ledger.json` | Semantic distinction |\n| No P3 ROCKET/PPV convolution implementation was found | repository code audit before sealing | Novelty gate |\n")
    write_new(REPORT_DIR / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "result_sha256": sha256(result_path), "arrays_sha256": sha256(array_path), "report_sha256": sha256(report_path), "fit_count": 12, "official_access": 0, "csv_materializations": 0, "uploads": 0}))
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
