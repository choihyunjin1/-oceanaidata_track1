"""Sealed P3 v81 historical extreme-wave ratio representation."""

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

import run_p3_gust_factor_intermittency_residual_cycle_20260901_v80 as v80  # noqa: E402

EXPERIMENT_ID = "p3_extreme_wave_ratio_residual_cycle_20260901_v81"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
HS_COLUMN, HMAX_COLUMN = 0, 2
WINDOWS, HS_FLOOR, FEATURE_COUNT = ((0, 145), (72, 145)), 0.1, 16
BASE = v80.BASE
SPEC_CLASS = v80.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_HRATIO16_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_HRATIO16_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v80.sha256, v80.canonical, v80.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v81 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder, audit = config["encoder"], config["duplication_audit"]
    checks = {
        "schema": config["schema_version"] == "p3.extreme_wave_ratio_residual.config.v81",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": audit["semantic_verdict"]
        == "NON_DUPLICATE_HISTORICAL_EXTREME_WAVE_RATIO_GEOMETRY",
        "columns": tuple(encoder["raw_column_indices"]) == (HS_COLUMN, HMAX_COLUMN),
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "floor": float(encoder["hs_floor_m"]) == HS_FLOOR,
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"])
        == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not audit["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v81 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def ratio_statistics(hs: np.ndarray, hmax: np.ndarray) -> np.ndarray:
    significant = np.maximum(np.asarray(hs, dtype=np.float64), 0.0)
    maximum = np.maximum(np.asarray(hmax, dtype=np.float64), 0.0)
    if significant.shape != maximum.shape or len(significant) < 24:
        raise ContractError("extreme-wave ratio support differs")
    ratio = maximum / np.maximum(significant, HS_FLOOR)
    q10, q25, median, q75, q90 = np.quantile(ratio, (0.10, 0.25, 0.50, 0.75, 0.90))
    time = np.linspace(-0.5, 0.5, len(ratio), dtype=np.float64)
    slope = float(np.dot(time, ratio - np.mean(ratio)) / np.dot(time, time))
    result = np.asarray(
        [np.mean(ratio), np.std(ratio), median, q75 - q25, q10, q90, ratio[-1] - median, slope],
        dtype=np.float64,
    )
    if result.shape != (8,) or not np.isfinite(result).all():
        raise ContractError("extreme-wave ratio statistics differ")
    return result


def ratio_features(sequence: np.ndarray) -> np.ndarray:
    raw = v80.v79.v78.fill_prefix(np.asarray(sequence)[:289])[::2]
    features = np.concatenate(
        [ratio_statistics(raw[start:stop, HS_COLUMN], raw[start:stop, HMAX_COLUMN]) for start, stop in WINDOWS]
    )
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("extreme-wave ratio feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    hs = 2.0 + 0.25 * np.sin(np.linspace(0.0, 6.0, 145))
    stable = 1.6 * hs
    elevated = stable.copy()
    elevated[-36:] = 2.0 * hs[-36:]
    baseline = ratio_statistics(hs, stable)
    changed = ratio_statistics(hs, elevated)
    if not changed[1] > baseline[1] or not changed[5] > baseline[5] or not changed[6] > baseline[6] or not changed[7] > baseline[7]:
        raise ContractError("extreme-wave regime sensitivity guard failed")
    scaled = ratio_statistics(3.0 * hs, 3.0 * elevated)
    if not np.allclose(changed, scaled, rtol=0.0, atol=1e-12):
        raise ContractError("common-scale invariance guard failed")
    calm = ratio_statistics(np.zeros(145), np.zeros(145))
    if not np.array_equal(calm, np.zeros(8)):
        raise ContractError("calm-path guard failed")
    sequence = np.zeros((289, 10), dtype=np.float64)
    axis = np.linspace(0.0, 8.0, 289)
    sequence[:, HS_COLUMN] = 2.0 + 0.25 * np.sin(axis)
    sequence[:, HMAX_COLUMN] = 1.6 * sequence[:, HS_COLUMN] + 0.05 * np.cos(2.0 * axis)
    sequence[1::7, (HS_COLUMN, HMAX_COLUMN)] = np.nan
    direct = ratio_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, ratio_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "late_extreme_ratio_variance_q90_endpoint_slope_increased": True,
        "common_scale_invariant": True,
        "calm_path_zero": True,
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
        features[position] = ratio_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v80.v79.case_surface()
    features, metadata = surface_features(cases)
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(len(features) >= int(gate["minimum_cases"]) and positive_variance >= int(gate["minimum_positive_variance_features"]) and nonzero_share >= float(gate["minimum_nonzero_feature_share"]))
    return {**metadata, "positive_variance_features": positive_variance, "nonzero_feature_share": nonzero_share, "target_used": False, "passed": passed}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v81 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {"schema_version": "p3.extreme_wave_ratio_residual.preflight.v81", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE_ZERO_FIT", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12 if support["passed"] else 0, "synthetic": synthetic_receipt(), "historical_support": support, "prior_outputs_used": False, "official_used_for_features_gates_selection": False, "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = BASE.surface_features, BASE.SPECS
    BASE.surface_features, BASE.SPECS = surface_features, SPECS
    try:
        result, arrays = BASE.execute(config)
    finally:
        BASE.surface_features, BASE.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.extreme_wave_ratio_residual.result.v81", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_EXTREME_WAVE_RATIO_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 historical extreme-wave ratio residual cycle v81", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- v81 encodes the past distribution and temporal displacement of Hmax/Hs; it does not reuse current-only v21 routing, v71 outputs, v64, or official feedback.", "- Forristall (1978) motivates normalization of high-wave behavior by significant height only; the repeatedly exposed 182-case surface is EXPLORATORY_ONLY."]
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
        raise ContractError("v81 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v81 support gate failed; zero-fit closure required")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| High-wave distributions are normalized by a consistent significant-wave-height definition | Forristall 1978, DOI:10.1029/JC083iC05p02353 | physical ratio motivation only |\n| No prior P3 output stores the historical Hmax/Hs distribution, displacement and trend | repository semantic audit | novelty boundary |\n| v64/prior/official outputs were excluded | sealed v81 contract | reuse boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
