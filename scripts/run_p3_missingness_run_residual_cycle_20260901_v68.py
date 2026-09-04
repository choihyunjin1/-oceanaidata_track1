"""Sealed P3 v68 raw missingness-run geometry experiment."""

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

import run_p3_pettitt_rank_change_residual_cycle_20260901_v67 as v67  # noqa: E402

EXPERIMENT_ID = "p3_missingness_run_residual_cycle_20260901_v68"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNEL_NAMES = ("hs", "tp", "hmax", "wvdir", "wspd", "gust", "wdir", "airt", "relh", "caph")
WINDOWS = ((0, 289), (144, 289))
FEATURE_COUNT = 80
BASE = v67.BASE
SPEC_CLASS = v67.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_MASKRUN80_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_MASKRUN80_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v67.sha256, v67.canonical, v67.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v68 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.missingness_run_residual.config.v68",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_MISSINGNESS_RUN_GEOMETRY_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "values_unused": not encoder["observed_values_used"],
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "risk_only": "only" in config["duplication_audit"]["known_transport_risk"],
        "official_excluded": "excluded" in config["duplication_audit"]["official_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v68 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def missing_runs(mask: np.ndarray) -> np.ndarray:
    missing = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], missing, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return stops - starts


def mask_statistics(finite: np.ndarray) -> np.ndarray:
    observed = np.asarray(finite, dtype=bool)
    n = len(observed)
    if n < 2:
        raise ContractError("mask path support differs")
    runs = missing_runs(~observed)
    terminal_age = 0
    for value in (~observed)[::-1]:
        if not value:
            break
        terminal_age += 1
    result = np.asarray(
        [
            float(np.mean(observed)),
            len(runs) / n,
            (float(np.max(runs)) if len(runs) else 0.0) / n,
            terminal_age / n,
        ],
        dtype=np.float64,
    )
    if result.shape != (4,) or not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise ContractError("mask statistic contract differs")
    return result


def mask_features(sequence: np.ndarray) -> np.ndarray:
    raw = np.asarray(sequence)
    if raw.ndim != 2 or raw.shape[0] < 289 or raw.shape[1] != 10:
        raise ContractError("native ten-channel context differs")
    finite = np.isfinite(raw[:289])
    output: list[float] = []
    for channel in range(10):
        for start, stop in WINDOWS:
            output.extend(mask_statistics(finite[start:stop, channel]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("missingness-run feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    isolated = np.ones(96, dtype=bool)
    isolated[np.arange(8, 88, 8)] = False
    burst = np.ones(96, dtype=bool)
    burst[40:50] = False
    isolated_stats = mask_statistics(isolated)
    burst_stats = mask_statistics(burst)
    if not isolated_stats[0] == burst_stats[0] or not isolated_stats[1] > burst_stats[1] or not burst_stats[2] > isolated_stats[2]:
        raise ContractError("isolated-versus-burst missingness guard failed")
    all_observed = mask_statistics(np.ones(96, dtype=bool))
    all_missing = mask_statistics(np.zeros(96, dtype=bool))
    if not np.array_equal(all_observed, [1.0, 0.0, 0.0, 0.0]):
        raise ContractError("all-observed bound guard failed")
    if not np.array_equal(all_missing, [0.0, 1.0 / 96.0, 1.0, 1.0]):
        raise ContractError("all-missing bound guard failed")
    sequence = np.arange(2890, dtype=np.float64).reshape(289, 10)
    for channel in range(10):
        sequence[(np.arange(289) + channel) % (7 + channel) == 0, channel] = np.nan
    direct = mask_features(sequence).reshape(10, 2, 4)
    permutation = np.asarray([9, 7, 5, 3, 1, 8, 6, 4, 2, 0])
    permuted = mask_features(sequence[:, permutation]).reshape(10, 2, 4)
    if not np.array_equal(permuted, direct[permutation]):
        raise ContractError("channel permutation guard failed")
    extended = np.vstack([sequence, np.full((12, 10), np.nan)])
    if not np.array_equal(direct.ravel(), mask_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": FEATURE_COUNT,
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": True,
        "equal_missing_fraction": float(isolated_stats[0]),
        "isolated_run_count": float(isolated_stats[1]),
        "burst_run_count": float(burst_stats[1]),
        "isolated_longest_run": float(isolated_stats[2]),
        "burst_longest_run": float(burst_stats[2]),
        "all_observed": all_observed.tolist(),
        "all_missing": all_missing.tolist(),
        "channel_permutation": True,
        "future_isolated": True,
    }


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(BASE.SEQUENCES, mmap_mode="r")
    station_codes = np.load(BASE.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    channel_observed = np.zeros(10, dtype=bool)
    channel_missing = np.zeros(10, dtype=bool)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        raw = sequences[anchor_id]
        features[position] = mask_features(raw)
        finite = np.isfinite(raw[:289])
        channel_observed |= np.any(finite, axis=0)
        channel_missing |= np.any(~finite, axis=0)
    mixed_channels = int(np.sum(channel_observed & channel_missing))
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
        "channels_with_observed_and_missing_examples": mixed_channels,
        "observed_numeric_values_used": False,
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v67.v66.v65.v64.v63.v62.case_surface()
    features, metadata = surface_features(cases)
    nonconstant = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(
        len(features) >= int(gate["minimum_cases"])
        and nonconstant >= int(gate["minimum_nonconstant_features"])
        and metadata["channels_with_observed_and_missing_examples"] >= int(gate["minimum_channels_with_observed_and_missing_examples"])
    )
    return {**metadata, "nonconstant_features": nonconstant, "target_used": False, "passed": passed}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v68 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.missingness_run_residual.preflight.v68",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2,
        "maximum_model_fits": 12 if support["passed"] else 0,
        "synthetic": synthetic_receipt(),
        "historical_support": support,
        "known_missingness_shift_used_for_selection": False,
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
            "schema_version": "p3.missingness_run_residual.result.v68",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_MISSINGNESS_RUN_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 missingness-run geometry residual cycle v68",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v68 uses only raw finite/missing masks and their run geometry; observed values are not features.",
        "- Known missingness shift is a fixed transport warning only. Prior outputs and official feedback are excluded; the surface is EXPLORATORY_ONLY.",
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
        raise ContractError("v68 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v68 support gate failed; zero-fit closure required")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Observation masks and elapsed missingness can carry predictive information | Che et al. 2018, DOI:10.1038/s41598-018-24271-9 | mechanism only; no GRU-D claim |\n| No dedicated executed P3 mask-run residual axis exists | repository semantic audit | novelty boundary |\n| Known missingness shift, prior outputs and official feedback were excluded from selection | sealed v68 contract | transport/reuse boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
