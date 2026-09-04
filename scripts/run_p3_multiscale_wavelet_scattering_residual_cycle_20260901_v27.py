"""Exactly-once P3 fixed multiscale wavelet-scattering residual cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

import run_p3_causal_multichannel_rocket_residual_cycle_20260901_v26 as v26  # noqa: E402
import run_p3_kma_wind_work_residual_axis_cycle_20260901_v20 as v20  # noqa: E402
import run_p3_path_signature_residual_cycle_20260901_v23 as v23  # noqa: E402
from run_p3_parallel_candidate_cycle_20260831_v4 import rmse  # noqa: E402
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    POINTS_PER_RMSE_M,
    bootstrap,
)

EXPERIMENT_ID = "p3_multiscale_wavelet_scattering_residual_cycle_20260901_v27"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = ROOT / "artifacts/p3/sequences_all20_v1/train_values.npy"
STATIONS = ROOT / "artifacts/p3/sequences_all20_v1/train_station.npy"
SCALES = (1, 2, 4, 8, 16, 32)
FEATURE_COUNT = 336
SPECS = (
    v26.Spec("P3_1_SCATTER336_RIDGE256_ADD10", 256.0),
    v26.Spec("P3_2_SCATTER336_RIDGE1024_ADD10", 1024.0),
)
BLEND = 0.10
TRANSPORT_PENALTY_POINTS = 0.04958605409228893
OFFICIAL_CHAMPION_POINTS = 24.203599


class ContractError(RuntimeError):
    """Raised when the sealed v27 contract differs."""


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
        "schema": config["schema_version"] == "p3.multiscale_wavelet_scattering_residual.config.v27",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_FIXED_SCATTERING_AXIS",
        "scales": tuple(config["scattering"]["scales"]) == SCALES,
        "features": config["scattering"]["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
    }
    if not all(checks.values()):
        raise ContractError(f"v27 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def scattering_features(sequence: np.ndarray) -> np.ndarray:
    path = v26.transformed_path(sequence)
    output: list[float] = []
    modulus: dict[int, np.ndarray] = {}
    for scale in SCALES:
        values = np.abs(path[scale:] - path[:-scale])
        modulus[scale] = values
        output.extend(np.mean(values, axis=0))
        output.extend(np.std(values, axis=0))
        output.extend(np.max(values, axis=0))
    for first, second in zip(SCALES[:-1], SCALES[1:], strict=True):
        values = modulus[first]
        cascade = np.abs(values[second:] - values[:-second])
        output.extend(np.mean(cascade, axis=0))
        output.extend(np.std(cascade, axis=0))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("scattering feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = scattering_features(sequence)
    return {"feature_count": len(feature), "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(feature).all())}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if np.load(SEQUENCES, mmap_mode="r").shape != (24360, 289, 10) or np.load(STATIONS, mmap_mode="r").shape != (24360,):
        raise ContractError("sequence cache shape differs")
    if ARTIFACT_DIR.exists() or LOCK.exists():
        raise ContractError("exactly-once namespace is already consumed")
    payload = {"schema_version": "p3.multiscale_wavelet_scattering_residual.preflight.v27", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12, "synthetic": synthetic_receipt(), "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(SEQUENCES, mmap_mode="r")
    station_codes = np.load(STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        if int(station_codes[int(row.anchor_id)]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = scattering_features(sequences[int(row.anchor_id)])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def crossfit(cases: pd.DataFrame, features: np.ndarray, targets: np.ndarray, reference: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    residual = targets - reference
    outputs = {spec.name: reference.copy() for spec in SPECS}
    receipts: list[dict[str, Any]] = []
    for block in v23.BLOCKS:
        valid = cases["block"].eq(block).to_numpy()
        train = v23.purged_train_indices(cases, valid)
        for spec in SPECS:
            predicted, receipt = v26.fit_predict(features, residual, train, valid, spec)
            outputs[spec.name][valid] = np.clip(reference[valid] + BLEND * predicted, 0.0, 30.0)
            receipt.update({"block": block, "additive_residual_weight": BLEND})
            receipts.append(receipt)
    if len(receipts) != 12:
        raise ContractError("fit budget differs")
    return outputs, receipts


def score(frame: pd.DataFrame, prediction: np.ndarray, spec: v26.Spec) -> dict[str, Any]:
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
    episode_ci = bootstrap(frame, flat, ("episode_id",), 20261201 + offset)
    group_ci = bootstrap(frame, flat, ("block", "station"), 20261202 + offset)
    stable_checks = {"delta_rmse_negative": delta < 0, "minimum_four_improved_blocks": improved >= 4, "episode_ci90_upper_below_zero": episode_ci["ci90_m"][1] < 0, "block_station_ci90_upper_below_zero": group_ci["ci90_m"][1] < 0, "worst_station_lead_at_most_0p01m": worst_slice <= 0.01, "finite_predictions": bool(np.isfinite(flat).all())}
    high_risk_checks = {"delta_rmse_at_most_minus_0p005m": delta <= -0.005, "worst_station_lead_at_most_0p02m": worst_slice <= 0.02, "finite_predictions": stable_checks["finite_predictions"]}
    stable = all(stable_checks.values())
    high_risk = not stable and all(high_risk_checks.values())
    points = -delta * POINTS_PER_RMSE_M
    return {"name": spec.name, "decision": "PASS_STABLE" if stable else "PRESERVE_HIGH_RISK" if high_risk else "NO_GO", "ridge_alpha": spec.alpha, "additive_residual_weight": BLEND, "rmse_m": {"uniform_0p425": before, "candidate": after, "delta_candidate_minus_uniform": delta}, "expected_points": {"raw_gain": points, "transport_penalty": TRANSPORT_PENALTY_POINTS, "transport_adjusted_gain": points - TRANSPORT_PENALTY_POINTS, "nominal_official_score": OFFICIAL_CHAMPION_POINTS + points}, "improved_blocks": int(improved), "by_block": by_block, "station": by_station, "lead": by_lead, "station_lead": station_lead, "worst_block_delta_m": worst_block, "worst_station_lead_delta_m": worst_slice, "episode_bootstrap": episode_ci, "block_station_bootstrap": group_ci, "stable_checks": stable_checks, "high_risk_checks": high_risk_checks}


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, profile = v23.case_surface()
    features, feature_receipt = surface_features(cases)
    predictions, receipts = crossfit(cases, features, targets, reference)
    frame = v23.long_frame(cases, targets, reference)
    scored = [score(frame, predictions[spec.name], spec) for spec in SPECS]
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {"schema_version": "p3.multiscale_wavelet_scattering_residual.result.v27", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "PASS_CANDIDATE_AVAILABLE" if passing else "NO_GO_ALL_SCATTERING_CANDIDATES", "surface_claim": config["validation"]["surface"], "reference": config["reference"], "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"], "feature_receipt": feature_receipt, "candidates": scored, "fit_receipts": receipts, "fit_count": 12, "data_profile": profile, "data_access": {"historical_target_rows": 1092, "official_test_rows": 0, "official_sample_rows": 0, "official_submission_rows": 0, "hidden_truth_rows": 0, "csv_materializations": 0, "uploads": 0}, "execution": {"python": platform.python_version(), "elapsed_seconds": time.perf_counter() - started, "candidate_count": 2, "result_based_tuning": False, "outer_result_parameter_changes": 0, "row_deletion": 0}}
    arrays = {"truth": targets, "uniform": reference, "candidate_1": predictions[SPECS[0].name], "candidate_2": predictions[SPECS[1].name], "anchor_id": cases["anchor_id"].to_numpy(np.int32), "lead_h": np.asarray(v23.LEADS, dtype=np.int16), "block": cases["block"].to_numpy(dtype="U5"), "station": cases["station"].to_numpy(dtype="U5"), "episode": cases["episode_id"].to_numpy(dtype="U32")}
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 fixed multiscale wavelet-scattering residual cycle v27", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- The feature axis is nonduplicate in this repository: deterministic dyadic wavelet-modulus cascades and invariant moments, with no learned encoder.", "- The repeatedly exposed 182-case surface is EXPLORATORY_ONLY, not a Public transport guarantee."]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; raw {points['raw_gain']:+.6f} points; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; worst station-lead {item['worst_station_lead_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.")
    lines.extend(["", "Official test/sample/submission/hidden access, CSV materialization, and upload were all zero. No row was deleted and no outer result changed features, Ridge strengths, or blend."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT_DIR.exists() or REPORT_DIR.exists() or LOCK.exists():
        raise ContractError("v27 exactly-once namespace already exists")
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
    write_new(REPORT_DIR / "gap-matrix.md", b"# Gap matrix\n\n| Family | Difference from v27 | Verdict |\n|---|---|---|\n| Spectral | Global Fourier/random-frequency amplitude | Distinct |\n| Path signature | Ordered iterated integrals | Distinct |\n| ROCKET | Random local kernels and PPV/max | Distinct |\n| v27 scattering | Fixed wavelet modulus cascades and invariant moments | Executed |\n")
    write_new(REPORT_DIR / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Scattering uses fixed wavelet modulus cascades for stable invariant representations | Bruna and Mallat, arXiv:1112.1120 | Architecture basis |\n| Joint multiscale modulus captures interactions beyond first-order spectra | Anden, Lostanlen, and Mallat, arXiv:1807.08869 | Multiscale interaction basis |\n| No P3 scattering implementation exists locally | repository audit before sealing | Novelty gate |\n")
    write_new(REPORT_DIR / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "result_sha256": sha256(result_path), "arrays_sha256": sha256(array_path), "report_sha256": sha256(report_path), "fit_count": 12, "official_access": 0, "csv_materializations": 0, "uploads": 0}))
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
