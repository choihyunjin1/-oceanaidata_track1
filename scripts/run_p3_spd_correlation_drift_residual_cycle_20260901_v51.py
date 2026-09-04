"""Sealed P3 v51 SPD correlation-drift residual experiment."""

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

EXPERIMENT_ID = "p3_spd_correlation_drift_residual_cycle_20260901_v51"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
GROUPS = ((0, 1, 2), (5, 6, 7, 8), (9, 10, 11), (0, 1, 2, 5, 6))
SHRINKAGE = 0.05
FEATURE_COUNT = 24
SPECS = (
    v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_1_SPD24_RIDGE512_ADD10", 512.0),
    v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_2_SPD24_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v50.sha256, v50.canonical, v50.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v51 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.spd_correlation_drift_residual.config.v51",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_SPD_DEPENDENCE_DRIFT_AXIS",
        "cross_problem": "P2 v20" in config["duplication_audit"]["cross_problem_note"],
        "groups": tuple(tuple(item["channels"]) for item in encoder["groups"]) == GROUPS,
        "shrinkage": float(encoder["fixed_identity_shrinkage"]) == SHRINKAGE,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
        "target_free_support": not encoder["support_gate"]["target_used"],
    }
    if not all(checks.values()):
        raise ContractError(f"v51 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def shrunk_correlation(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != 72 or matrix.shape[1] < 3 or not np.isfinite(matrix).all():
        raise ContractError("correlation input contract differs")
    centered = matrix - matrix.mean(axis=0)
    scale = np.std(centered, axis=0)
    standardized = centered / np.maximum(scale, 1e-12)
    covariance = standardized.T @ standardized / len(standardized)
    result = (1.0 - SHRINKAGE) * covariance + SHRINKAGE * np.eye(matrix.shape[1])
    eigenvalues = np.linalg.eigvalsh(result)
    if not np.isfinite(result).all() or eigenvalues.min() <= 0.0:
        raise ContractError("shrunk correlation is not SPD")
    return result


def matrix_log(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.min() <= 0.0:
        raise ContractError("matrix log requires SPD")
    return (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T


def spd_drift_statistics(early: np.ndarray, late: np.ndarray) -> np.ndarray:
    first, second = shrunk_correlation(early), shrunk_correlation(late)
    first_values, first_vectors = np.linalg.eigh(first)
    second_values, second_vectors = np.linalg.eigh(second)
    offdiag = ~np.eye(len(first), dtype=bool)
    statistics = np.asarray([
        np.linalg.norm(matrix_log(second) - matrix_log(first), ord="fro"),
        np.linalg.norm(second - first, ord="fro"),
        np.linalg.slogdet(second)[1] - np.linalg.slogdet(first)[1],
        second_values[-1] - first_values[-1],
        abs(float(np.dot(first_vectors[:, -1], second_vectors[:, -1]))),
        np.mean((second - first)[offdiag]),
    ], dtype=np.float64)
    if statistics.shape != (6,) or not np.isfinite(statistics).all() or np.any(statistics[:2] < 0.0) or not 0.0 <= statistics[4] <= 1.0 + 1e-9:
        raise ContractError("SPD drift statistic contract differs")
    return statistics


def spd_correlation_drift_features(sequence: np.ndarray) -> np.ndarray:
    path = v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    early, late = path[:72], path[73:145]
    output: list[float] = []
    for channels in GROUPS:
        output.extend(spd_drift_statistics(early[:, channels], late[:, channels]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("SPD correlation drift feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    base = rng.normal(size=(72, 3))
    same = spd_drift_statistics(base, base.copy())
    mixed = base.copy()
    mixed[:, 1] = 0.8 * base[:, 0] + 0.2 * base[:, 1]
    rotated = spd_drift_statistics(base, mixed)
    permutation = np.asarray([2, 0, 1])
    permuted = spd_drift_statistics(base[:, permutation], mixed[:, permutation])
    if np.max(np.abs(same[[0, 1, 2, 3, 5]])) > 1e-12 or rotated[0] <= 0.1 or not np.allclose(rotated, permuted, atol=1e-12):
        raise ContractError("synthetic SPD invariant/rotation guard failed")
    time = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * time) + 0.1 * index * time for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = spd_correlation_drift_features(sequence)
    return {"feature_count": len(feature), "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(feature).all()), "same_structure_zero_max": float(np.max(np.abs(same[[0, 1, 2, 3, 5]]))), "rotated_log_euclidean_distance": float(rotated[0]), "permutation_max_abs_error": float(np.max(np.abs(rotated - permuted)))}


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.SEQUENCES, mmap_mode="r")
    station_codes = np.load(v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = spd_correlation_drift_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
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
        raise ContractError("v51 exactly-once namespace is consumed")
    payload = {"schema_version": "p3.spd_correlation_drift_residual.preflight.v51", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12, "synthetic": synthetic_receipt(), "historical_support": support_receipt(config), "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.surface_features, v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.SPECS
    v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.surface_features, v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.SPECS = surface_features, SPECS
    try:
        result, arrays = v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.execute(config)
    finally:
        v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.surface_features, v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.spd_correlation_drift_residual.result.v51", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_SPD_CORRELATION_DRIFT_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 SPD correlation-drift residual cycle v51", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- v51 measures fixed early-to-late multichannel dependence-geometry drift; it imports no P2 output and reuses no v42-v50 output.", "- Arsigny et al. (2006) motivates the SPD geometry only; the 182-case surface remains EXPLORATORY_ONLY."]
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
        raise ContractError("v51 exactly-once namespace already exists")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Log-Euclidean coordinates provide a geometry for SPD matrices | Arsigny et al., Magnetic Resonance in Medicine 56, 2006, DOI:10.1002/mrm.20965 | mechanism only |\n| P2 covariance alignment is a different problem and no P2 object is reused | sealed cross-problem receipt | scope boundary |\n| Groups, shrinkage, splits and all v42-v50 reuse prohibitions were sealed before scoring | sealed v51 contract | novelty boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
