"""Sealed P3 v71 equilibrium-error and mean-reversion experiment."""

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

import run_p3_burg_reflection_residual_cycle_20260901_v70 as v70  # noqa: E402

EXPERIMENT_ID = "p3_equilibrium_error_residual_cycle_20260901_v71"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
PAIRS, PAIR_NAMES = ((0, 2), (0, 1), (0, 5), (1, 5)), ("hs~hmax", "hs~tp", "hs~wspd", "tp~wspd")
WINDOWS = ((0, 145), (72, 145))
FEATURE_COUNT = 32
BASE = v70.BASE
SPEC_CLASS = v70.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_EQERR32_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_EQERR32_RIDGE2048_ADD10", 2048.0),
)
BLEND, MAD_SCALE, EPSILON = 0.10, 1.4826, 1e-12
sha256, canonical, write_new = v70.sha256, v70.canonical, v70.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v71 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.equilibrium_error_residual.config.v71",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_EQUILIBRIUM_ERROR_MEAN_REVERSION_AXIS",
        "pairs": tuple(encoder["ordered_pairs"]) == PAIR_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_excluded": "excluded" in config["duplication_audit"]["official_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v71 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def robust_normalize(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < 16 or not np.isfinite(path).all():
        raise ContractError("equilibrium path support differs")
    center = float(np.median(path))
    scale = MAD_SCALE * float(np.median(np.abs(path - center)))
    if scale <= EPSILON:
        return np.zeros_like(path)
    return (path - center) / scale


def equilibrium_geometry(response: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    y, x = robust_normalize(response), robust_normalize(regressor)
    denominator = float(np.dot(x, x))
    slope = 0.0 if denominator <= EPSILON else float(np.dot(x, y) / denominator)
    residual = y - slope * x
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    lagged = residual[:-1]
    current = residual[1:]
    lag_centered = lagged - float(np.mean(lagged))
    current_centered = current - float(np.mean(current))
    corr_denominator = float(np.sqrt(np.dot(lag_centered, lag_centered) * np.dot(current_centered, current_centered)))
    lag_corr = 0.0 if corr_denominator <= EPSILON else float(np.dot(lag_centered, current_centered) / corr_denominator)
    ec_denominator = float(np.dot(lagged, lagged))
    error_correction = 0.0 if ec_denominator <= EPSILON else float(np.dot(lagged, np.diff(residual)) / ec_denominator)
    features = np.asarray([slope, residual_rms, lag_corr, error_correction], dtype=np.float64)
    if features.shape != (4,) or not np.isfinite(features).all():
        raise ContractError("equilibrium feature contract differs")
    return features


def transformed_path(sequence: np.ndarray) -> np.ndarray:
    return v70.transformed_path(np.asarray(sequence)[:289])


def equilibrium_features(sequence: np.ndarray) -> np.ndarray:
    path = transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[np.ndarray] = []
    for response, regressor in PAIRS:
        for start, stop in WINDOWS:
            output.append(equilibrium_geometry(path[start:stop, response], path[start:stop, regressor]))
    features = np.concatenate(output)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("equilibrium surface feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    x = np.cumsum(rng.normal(size=512))
    disturbance = np.zeros(512, dtype=np.float64)
    innovations = rng.normal(scale=0.25, size=512)
    for index in range(1, len(disturbance)):
        disturbance[index] = 0.20 * disturbance[index - 1] + innovations[index]
    cointegrated = 1.7 * x + disturbance
    independent = np.cumsum(rng.normal(size=512))
    coin_feature = equilibrium_geometry(cointegrated, x)
    independent_feature = equilibrium_geometry(independent, x)
    if not abs(coin_feature[0]) > 0.50:
        raise ContractError("equilibrium slope recovery failed")
    if not coin_feature[1] < 0.50 * independent_feature[1]:
        raise ContractError("cointegrated residual-spread guard failed")
    if not coin_feature[2] < independent_feature[2] - 0.30:
        raise ContractError("cointegrated residual-persistence guard failed")
    if not coin_feature[3] < independent_feature[3] - 0.30:
        raise ContractError("error-correction direction guard failed")
    affine = equilibrium_geometry(5.0 * cointegrated + 2.0, 7.0 * x - 3.0)
    if not np.allclose(coin_feature, affine, rtol=1e-12, atol=1e-12):
        raise ContractError("positive affine invariance guard failed")
    reversed_pair = equilibrium_geometry(x, cointegrated)
    if np.array_equal(coin_feature, reversed_pair):
        raise ContractError("ordered-pair guard failed")
    constant = equilibrium_geometry(np.full(80, 4.0), np.full(80, -2.0))
    if not np.array_equal(constant, np.zeros(4)):
        raise ContractError("constant path bound guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * axis) + 0.1 * index * axis for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = equilibrium_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, equilibrium_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "cointegrated_slope": float(coin_feature[0]),
        "cointegrated_residual_rms": float(coin_feature[1]),
        "independent_residual_rms": float(independent_feature[1]),
        "cointegrated_lag1": float(coin_feature[2]),
        "independent_lag1": float(independent_feature[2]),
        "cointegrated_error_correction": float(coin_feature[3]),
        "independent_error_correction": float(independent_feature[3]),
        "positive_affine_invariant": True,
        "ordered_pair": True,
        "constant_zero": True,
        "future_isolated": True,
    }


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(BASE.SEQUENCES, mmap_mode="r")
    station_codes = np.load(BASE.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = equilibrium_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v70.v69.v68.v67.v66.v65.v64.v63.v62.case_surface()
    features, metadata = surface_features(cases)
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    finite_pair_windows = int(np.sum(np.all(np.isfinite(features.reshape(len(features), 8, 4)), axis=(0, 2))))
    gate = config["encoder"]["support_gate"]
    passed = bool(
        len(features) >= int(gate["minimum_cases"])
        and positive_variance >= int(gate["minimum_positive_variance_features"])
        and finite_pair_windows >= int(gate["minimum_finite_pair_windows"])
    )
    return {
        **metadata,
        "positive_variance_features": positive_variance,
        "finite_pair_windows": finite_pair_windows,
        "target_used": False,
        "passed": passed,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v71 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.equilibrium_error_residual.preflight.v71",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2,
        "maximum_model_fits": 12 if support["passed"] else 0,
        "synthetic": synthetic_receipt(),
        "historical_support": support,
        "prior_outputs_used": False,
        "official_used_for_features_gates_selection": False,
        "official_access": 0,
        "csv_materializations": 0,
        "uploads": 0,
        "config_status": config["status"],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = BASE.surface_features, BASE.SPECS
    BASE.surface_features, BASE.SPECS = surface_features, SPECS
    try:
        result, arrays = BASE.execute(config)
    finally:
        BASE.surface_features, BASE.SPECS = original_surface, original_specs
    result.update(
        {
            "schema_version": "p3.equilibrium_error_residual.result.v71",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_EQUILIBRIUM_ERROR_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 equilibrium-error residual cycle v71",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v71 uses fixed ordered-pair equilibrium residual and error-correction geometry, not prior candidate outputs.",
        "- DCCA, SPD covariance and transfer-entropy boundaries are recorded; the surface is EXPLORATORY_ONLY.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; nominal score {points['nominal_official_score']:.6f}; planning {points['raw_gain']:+.6f}; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; lead {item['worst_lead_delta_m']:+.9f}m; station-lead {item['worst_station_lead_delta_m']:+.9f}m; tail {item['worst_reference_tail_block_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.")
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
        raise ContractError("v71 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v71 support gate failed; zero-fit closure required")
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "official_access": 0}))
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    result, arrays = execute(config)
    array_path = ARTIFACT / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {"runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "evaluation_arrays_sha256": sha256(array_path), "preflight_receipt_sha256": preflight["receipt_sha256"], "input_sha256": config["inputs"]}
    result_path = ARTIFACT / "result.json"
    write_new(result_path, canonical(result))
    write_new(REPORT / "result.json", canonical(result))
    report_path = REPORT / "report-source.md"
    write_new(report_path, render_report(result).encode())
    write_new(REPORT / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "result_sha256": sha256(result_path), "arrays_sha256": sha256(array_path), "report_sha256": sha256(report_path), "fit_count": 12, "official_access": 0, "csv_materializations": 0, "uploads": 0}))
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Equilibrium errors motivate an error-correction representation | Engle and Granger 1987, DOI:10.2307/1913236 | mechanism only |\n| No executed P3 equilibrium-error residual axis exists | repository semantic audit | novelty boundary |\n| Prior/official outputs were excluded | sealed v71 contract | reuse boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
