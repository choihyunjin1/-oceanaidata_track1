"""Sealed P3 v46 ternary-rank triplet O-information experiment."""

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

import run_p3_lagged_distance_correlation_residual_cycle_20260901_v44 as v44  # noqa: E402

EXPERIMENT_ID = "p3_triplet_oinformation_residual_cycle_20260901_v46"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
TRIPLETS = ((0, 1, 2), (0, 5, 6), (0, 3, 4), (0, 7, 8), (0, 9, 10), (0, 10, 11), (0, 1, 5), (0, 2, 6))
WINDOWS = (145, 73)
FEATURE_COUNT = 16
SPECS = (
    v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_1_OINFO16_RIDGE512_ADD10", 512.0),
    v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_2_OINFO16_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v44.sha256, v44.canonical, v44.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v46 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.triplet_oinformation_residual.config.v46",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_HIGH_ORDER_TRIPLET_INFORMATION_AXIS",
        "triplets": tuple(tuple(item["channels"]) for item in encoder["triplets"]) == TRIPLETS,
        "windows": tuple(encoder["windows_rows"]) == WINDOWS,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
        "target_free_support": not encoder["support_gate"]["target_used"],
    }
    if not all(checks.values()):
        raise ContractError(f"v46 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def ternary_symbols(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if path.ndim != 1 or len(path) < 6 or not np.isfinite(path).all():
        raise ContractError("symbol input contract differs")
    lower, upper = np.quantile(path, [1.0 / 3.0, 2.0 / 3.0])
    symbols = np.digitize(path, [lower, upper], right=True).astype(np.int8)
    if symbols.shape != path.shape or np.any((symbols < 0) | (symbols > 2)):
        raise ContractError("ternary symbol contract differs")
    return symbols


def entropy_from_codes(codes: np.ndarray) -> float:
    values = np.asarray(codes, dtype=np.int64)
    counts = np.bincount(values)
    probabilities = counts[counts > 0].astype(np.float64) / len(values)
    return float(-np.sum(probabilities * np.log(probabilities)))


def triplet_oinformation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    a, b, c = (np.asarray(value, dtype=np.int64) for value in (first, second, third))
    if not (a.shape == b.shape == c.shape) or a.ndim != 1 or len(a) < 6:
        raise ContractError("O-information symbol shape differs")
    if any(np.any((value < 0) | (value > 2)) for value in (a, b, c)):
        raise ContractError("O-information symbols differ")
    value = (
        entropy_from_codes(a)
        + entropy_from_codes(b)
        + entropy_from_codes(c)
        + entropy_from_codes(a * 9 + b * 3 + c)
        - entropy_from_codes(a * 3 + b)
        - entropy_from_codes(a * 3 + c)
        - entropy_from_codes(b * 3 + c)
    )
    if not np.isfinite(value):
        raise ContractError("O-information is nonfinite")
    return float(value)


def triplet_oinformation_features(sequence: np.ndarray) -> np.ndarray:
    path = v44.v43.v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for window in WINDOWS:
        current = path[-window:]
        symbols = np.column_stack([ternary_symbols(current[:, channel]) for channel in range(current.shape[1])])
        for first, second, third in TRIPLETS:
            output.append(triplet_oinformation(symbols[:, first], symbols[:, second], symbols[:, third]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("triplet O-information feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    first = np.tile(np.array([0, 1], dtype=np.int8), 128)
    second = np.repeat(np.array([0, 1], dtype=np.int8), 128)
    xor = np.bitwise_xor(first, second)
    redundant = triplet_oinformation(first, first, first)
    synergistic = triplet_oinformation(first, second, xor)
    if redundant <= 0.5 or synergistic >= -0.5:
        raise ContractError("synthetic O-information sign guard failed")
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = triplet_oinformation_features(sequence)
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "redundant_copy_oinformation": redundant,
        "xor_oinformation": synergistic,
        "sign_guard": bool(redundant > 0.0 and synergistic < 0.0),
    }


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(v44.v43.v42.v41.v40.v39.SEQUENCES, mmap_mode="r")
    station_codes = np.load(v44.v43.v42.v41.v40.v39.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = triplet_oinformation_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
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
        raise ContractError("v46 exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.triplet_oinformation_residual.preflight.v46",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2,
        "maximum_model_fits": 12,
        "synthetic": synthetic_receipt(),
        "historical_support": support_receipt(config),
        "official_access": 0,
        "csv_materializations": 0,
        "uploads": 0,
        "config_status": config["status"],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = v44.v43.v42.v41.v40.v39.surface_features, v44.v43.v42.v41.v40.v39.SPECS
    v44.v43.v42.v41.v40.v39.surface_features, v44.v43.v42.v41.v40.v39.SPECS = surface_features, SPECS
    try:
        result, arrays = v44.v43.v42.v41.v40.v39.execute(config)
    finally:
        v44.v43.v42.v41.v40.v39.surface_features, v44.v43.v42.v41.v40.v39.SPECS = original_surface, original_specs
    result.update({
        "schema_version": "p3.triplet_oinformation_residual.result.v46",
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_TRIPLET_OINFORMATION_CANDIDATES",
        "duplication_audit": config["duplication_audit"],
        "primary_sources": config["primary_sources"],
    })
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 triplet O-information residual cycle v46",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v46 measures sealed contemporaneous three-channel redundancy versus synergy and reuses no v42-v44 output.",
        "- Rosas et al. (2019) motivates the mechanism only; the 182-case surface remains EXPLORATORY_ONLY.",
    ]
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
        raise ContractError("v46 exactly-once namespace already exists")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| O-information distinguishes redundancy-dominated from synergy-dominated high-order dependence | Rosas et al., Physical Review E 100, 2019, DOI:10.1103/PhysRevE.100.032305 | mechanism only |\n| Triplets, windows, symbols and all v42-v44 reuse prohibitions were sealed before scoring | sealed v46 contract | novelty boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
