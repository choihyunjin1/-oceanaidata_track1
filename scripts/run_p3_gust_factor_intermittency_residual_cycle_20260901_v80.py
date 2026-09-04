"""Sealed P3 v80 gust-factor intermittency representation."""

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

import run_p3_bds_embedding_independence_residual_cycle_20260901_v79 as v79  # noqa: E402

EXPERIMENT_ID = "p3_gust_factor_intermittency_residual_cycle_20260901_v80"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
WIND_COLUMN, GUST_COLUMN = 4, 5
WINDOWS = ((0, 145), (72, 145))
WIND_FLOOR, GUST_MULTIPLIER = 1.0, 1.25
FEATURE_COUNT = 16
BASE = v79.BASE
SPEC_CLASS = v79.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_GUST16_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_GUST16_RIDGE2048_ADD10", 2048.0),
)
BLEND, EPSILON = 0.10, 1e-12
sha256, canonical, write_new = v79.sha256, v79.canonical, v79.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v80 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder, audit = config["encoder"], config["duplication_audit"]
    checks = {
        "schema": config["schema_version"] == "p3.gust_factor_intermittency_residual.config.v80",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": audit["semantic_verdict"]
        == "NON_DUPLICATE_HISTORICAL_GUST_FACTOR_INTERMITTENCY_AXIS",
        "columns": tuple(encoder["raw_column_indices"]) == (WIND_COLUMN, GUST_COLUMN),
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "floor": float(encoder["wind_floor_mps"]) == WIND_FLOOR,
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
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
        "no_posthoc": not audit["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v80 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def lag1_correlation(values: np.ndarray) -> float:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < 24 or not np.isfinite(path).all():
        raise ContractError("gust-excess correlation support differs")
    left, right = path[:-1], path[1:]
    if np.std(left) <= EPSILON or np.std(right) <= EPSILON:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def gust_window_statistics(wind: np.ndarray, gust: np.ndarray) -> np.ndarray:
    sustained = np.maximum(np.asarray(wind, dtype=np.float64), 0.0)
    peak = np.maximum(np.asarray(gust, dtype=np.float64), 0.0)
    if sustained.shape != peak.shape or len(sustained) < 24:
        raise ContractError("gust-factor support differs")
    gust_factor = peak / np.maximum(sustained, WIND_FLOOR)
    scale = max(float(np.median(sustained)), WIND_FLOOR)
    gust_excess = np.maximum(peak - sustained, 0.0) / scale
    quantiles = np.quantile(gust_factor, (0.50, 0.75, 0.90))
    features = np.asarray(
        [
            *quantiles,
            np.max(gust_factor),
            np.mean(gust_excess),
            np.quantile(gust_excess, 0.90),
            np.mean(peak > GUST_MULTIPLIER * np.maximum(sustained, WIND_FLOOR)),
            lag1_correlation(gust_excess),
        ],
        dtype=np.float64,
    )
    if features.shape != (8,) or not np.isfinite(features).all():
        raise ContractError("gust-factor statistics differ")
    return features


def gust_features(sequence: np.ndarray) -> np.ndarray:
    raw = v79.v78.fill_prefix(np.asarray(sequence)[:289])[::2]
    features = np.concatenate(
        [
            gust_window_statistics(
                raw[start:stop, WIND_COLUMN], raw[start:stop, GUST_COLUMN]
            )
            for start, stop in WINDOWS
        ]
    )
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("gust-factor feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    axis = np.linspace(0.0, 4.0 * np.pi, 145)
    wind = 6.0 + 0.5 * np.sin(axis)
    steady_gust = 1.10 * wind
    burst_gust = steady_gust.copy()
    burst_gust[::12] += 8.0
    steady = gust_window_statistics(wind, steady_gust)
    burst = gust_window_statistics(wind, burst_gust)
    if not burst[3] > steady[3] or not burst[4] > steady[4] or not burst[6] > steady[6]:
        raise ContractError("gust-burst sensitivity guard failed")
    scaled = gust_window_statistics(3.0 * wind, 3.0 * burst_gust)
    if not np.allclose(burst, scaled, rtol=0.0, atol=1e-12):
        raise ContractError("common-scale invariance guard failed")
    calm = gust_window_statistics(np.zeros(145), np.zeros(145))
    if not np.array_equal(calm, np.zeros(8)):
        raise ContractError("calm-path bound guard failed")
    sequence = np.column_stack(
        [np.sin((column + 1) * np.linspace(-1.0, 1.0, 289)) for column in range(10)]
    )
    sequence[:, WIND_COLUMN] = 6.0 + 0.5 * np.sin(np.linspace(0.0, 8.0, 289))
    sequence[:, GUST_COLUMN] = sequence[:, WIND_COLUMN] + 1.0 + 0.2 * np.cos(
        np.linspace(0.0, 12.0, 289)
    )
    sequence[1::7, (WIND_COLUMN, GUST_COLUMN)] = np.nan
    direct = gust_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, gust_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "burst_max_increased": True,
        "burst_mean_excess_increased": True,
        "burst_exceedance_share_increased": True,
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
        features[position] = gust_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v79.case_surface()
    features, metadata = surface_features(cases)
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(
        len(features) >= int(gate["minimum_cases"])
        and positive_variance >= int(gate["minimum_positive_variance_features"])
        and nonzero_share >= float(gate["minimum_nonzero_feature_share"])
    )
    return {
        **metadata,
        "positive_variance_features": positive_variance,
        "nonzero_feature_share": nonzero_share,
        "target_used": False,
        "passed": passed,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v80 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.gust_factor_intermittency_residual.preflight.v80",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE_ZERO_FIT",
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
            "schema_version": "p3.gust_factor_intermittency_residual.result.v80",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE"
            if any(item["decision"] != "NO_GO" for item in result["candidates"])
            else "NO_GO_ALL_GUST_FACTOR_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 gust-factor intermittency residual cycle v80",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v80 represents the historical distribution and persistence of a physically explicit gust factor and gust excess. It does not reuse v20 coefficients, generic v26 outputs, v64, or official feedback.",
        "- Wieringa (1973) motivates gust factor over open water only; it is not P3 performance evidence. The repeatedly exposed 182-case surface is EXPLORATORY_ONLY.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(
            f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; nominal score {points['nominal_official_score']:.6f}; planning {points['raw_gain']:+.6f}; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; lead {item['worst_lead_delta_m']:+.9f}m; station-lead {item['worst_station_lead_delta_m']:+.9f}m; tail {item['worst_reference_tail_block_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}."
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
        raise ContractError("v80 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v80 support gate failed; zero-fit closure required")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Gust factor characterizes peak relative to sustained wind over open water | Wieringa 1973, DOI:10.1007/BF01034986 | physical operator motivation only |\n| No prior P3 contract combines historical gust-factor distribution with excess persistence | repository semantic audit | novelty boundary |\n| v64/prior/official outputs were excluded | sealed v80 contract | reuse boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
