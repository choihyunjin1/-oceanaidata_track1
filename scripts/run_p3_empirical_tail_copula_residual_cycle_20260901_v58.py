"""Sealed P3 v58 contemporaneous empirical tail-copula residual experiment."""

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

import run_p3_lz_rank_complexity_residual_cycle_20260901_v57 as v57  # noqa: E402

EXPERIMENT_ID = "p3_empirical_tail_copula_residual_cycle_20260901_v58"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
WINDOWS, Q = ((0, 145), (72, 145)), 0.10
FEATURE_COUNT = 60
BASE = v57.BASE
SPECS = (
    v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_1_TAILCOP60_RIDGE512_ADD10", 512.0),
    v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_2_TAILCOP60_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v57.sha256, v57.canonical, v57.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v58 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.empirical_tail_copula_residual.config.v58",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_CONTEMPORANEOUS_EMPIRICAL_TAIL_COPULA_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "q": float(encoder["tail_quantiles"]["lower"]) == Q and float(encoder["tail_quantiles"]["upper"]) == 1.0 - Q,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
        "target_free_support": not encoder["support_gate"]["target_used"],
    }
    if not all(checks.values()):
        raise ContractError(f"v58 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def empirical_mid_ranks(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if path.ndim != 1 or len(path) < 8 or not np.isfinite(path).all():
        raise ContractError("empirical-rank input contract differs")
    order = np.argsort(path, kind="stable")
    ranks = np.empty(len(path), dtype=np.float64)
    ranks[order] = np.arange(len(path), dtype=np.float64)
    return (ranks + 0.5) / len(path)


def pair_tail_statistics(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    u, v = empirical_mid_ranks(left), empirical_mid_ranks(right)
    lower_u, lower_v = u <= Q, v <= Q
    upper_u, upper_v = u >= 1.0 - Q, v >= 1.0 - Q
    ll = float(np.mean(lower_u & lower_v) / Q)
    uu = float(np.mean(upper_u & upper_v) / Q)
    lu = float(np.mean(lower_u & upper_v) / Q)
    ul = float(np.mean(upper_u & lower_v) / Q)
    return np.asarray([ll, uu, lu, ul, uu - ll], dtype=np.float64)


def tail_copula_features(sequence: np.ndarray) -> np.ndarray:
    path = v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for start, stop in WINDOWS:
        window = path[start:stop, CHANNELS]
        for left, right in PAIRS:
            output.extend(pair_tail_statistics(window[:, left], window[:, right]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("tail-copula feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    left = rng.normal(size=5000)
    comonotonic = pair_tail_statistics(left, left)
    independent = pair_tail_statistics(left, rng.normal(size=len(left)))
    if not (comonotonic[0] > independent[0] + 0.5 and comonotonic[1] > independent[1] + 0.5):
        raise ContractError("comonotonic-versus-independent tail guard failed")
    if not np.allclose(pair_tail_statistics(left, left), pair_tail_statistics(np.exp(left), 3.0 * left + 7.0), atol=0.0, rtol=0.0):
        raise ContractError("marginal monotone invariance guard failed")
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = tail_copula_features(sequence)
    return {"feature_count": len(feature), "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(feature).all()), "comonotonic_lower_tail": float(comonotonic[0]), "independent_lower_tail": float(independent[0]), "comonotonic_upper_tail": float(comonotonic[1]), "independent_upper_tail": float(independent[1]), "monotone_invariant": True}


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(BASE.SEQUENCES, mmap_mode="r")
    station_codes = np.load(BASE.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = tail_copula_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(nonzero_share >= float(gate["minimum_nonzero_share"]) and positive_variance >= int(gate["minimum_positive_variance_features"]))
    return {**metadata, "nonzero_share": nonzero_share, "positive_variance_features": positive_variance, "target_used": False, "passed": passed}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v58 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {"schema_version": "p3.empirical_tail_copula_residual.preflight.v58", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12 if support["passed"] else 0, "synthetic": synthetic_receipt(), "historical_support": support, "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = BASE.surface_features, BASE.SPECS
    BASE.surface_features, BASE.SPECS = surface_features, SPECS
    try:
        result, arrays = BASE.execute(config)
    finally:
        BASE.surface_features, BASE.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.empirical_tail_copula_residual.result.v58", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_EMPIRICAL_TAIL_COPULA_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 empirical tail-copula residual cycle v58", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- Fixed contemporaneous empirical tail-copula geometry is distinct from lagged cross-quantilograms, triplet O-information and SPD correlations; no prior output or official v42 result is used.", "- Schmidt and Stadtmuller (2006) motivates the mechanism only; the repeatedly exposed 182-case surface is EXPLORATORY_ONLY."]
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
        raise ContractError("v58 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "official_access": 0}))
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    if preflight["status"] == "STOP_SUPPORT_GATE":
        result = {"schema_version": "p3.empirical_tail_copula_residual.result.v58", "experiment_id": EXPERIMENT_ID, "status": "COMPLETE", "decision": "STOP_SUPPORT_GATE_ZERO_FIT", "fit_count": 0, "support_receipt": preflight["historical_support"], "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"], "data_access": {"historical_target_rows": 0, "official_test_rows": 0, "official_sample_rows": 0, "official_submission_rows": 0, "hidden_truth_rows": 0, "csv_materializations": 0, "uploads": 0}, "provenance": {"runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "input_sha256": config["inputs"]}}
        result_path = ARTIFACT / "result.json"
        write_new(result_path, canonical(result))
        write_new(REPORT / "result.json", canonical(result))
        report_path = REPORT / "report-source.md"
        write_new(report_path, b"# P3 empirical tail-copula residual cycle v58\n\n## Conclusion\n\n- **STOP_SUPPORT_GATE_ZERO_FIT**.\n- Target-free nonzero feature share was below the sealed gate; no target, outer score, official input, CSV, or upload was used.\n")
        write_new(REPORT / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "result_sha256": sha256(result_path), "report_sha256": sha256(report_path), "fit_count": 0, "official_access": 0, "csv_materializations": 0, "uploads": 0}))
        write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Empirical tail copulas describe extremal dependence | Schmidt and Stadtmuller 2006, DOI:10.1111/j.1467-9469.2005.00483.x | mechanism only |\n| Sealed target-free support failed before scoring | v58 preflight receipt | zero-fit decision |\n")
        print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 0, "official_access": 0}, ensure_ascii=False))
        return 0
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Empirical tail copulas describe lower and upper extremal dependence using marginal ranks | Schmidt and Stadtmuller 2006, DOI:10.1111/j.1467-9469.2005.00483.x | mechanism only |\n| Prior P3 lagged quantile, O-information and SPD axes do not estimate same-time empirical joint tail probabilities | sealed duplication audit | novelty boundary |\n| Pairs, ranks, q, windows, statistics, residual model and validation were fixed before scoring | sealed v58 config | execution contract |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
