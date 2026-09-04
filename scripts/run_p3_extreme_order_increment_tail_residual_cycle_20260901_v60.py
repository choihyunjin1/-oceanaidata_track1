"""Sealed P3 v60 extreme-order increment-tail residual experiment."""

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

import run_p3_intrinsic_dimension_residual_cycle_20260901_v59 as v59  # noqa: E402

EXPERIMENT_ID = "p3_extreme_order_increment_tail_residual_cycle_20260901_v60"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS, LAGS, ORDER_K = ((0, 145), (72, 145)), (1, 3, 6), 4
FEATURE_COUNT = 72
BASE = v59.BASE
SPEC_CLASS = v59.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_EOTAIL72_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_EOTAIL72_RIDGE2048_ADD10", 2048.0),
)
BLEND, EPSILON = 0.10, 1e-12
sha256, canonical, write_new = v59.sha256, v59.canonical, v59.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v60 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.extreme_order_increment_tail_residual.config.v60",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_EXTREME_ORDER_INCREMENT_TAIL_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "lags": tuple(encoder["increment_lags_rows"]) == LAGS,
        "order_k": encoder["extreme_order_k"] == ORDER_K,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_v42_excluded": "excluded" in config["duplication_audit"]["official_v42_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v60 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def extreme_order_statistics(values: np.ndarray) -> np.ndarray:
    increments = np.abs(np.asarray(values, dtype=np.float64))
    increments = increments[np.isfinite(increments)]
    if len(increments) < 4 * ORDER_K:
        raise ContractError("extreme-order support below 4k")
    ordered = np.sort(increments)
    x_k, x_2k, x_4k = ordered[-ORDER_K], ordered[-2 * ORDER_K], ordered[-4 * ORDER_K]
    numerator = max(float(x_k - x_2k), EPSILON)
    denominator = max(float(x_2k - x_4k), EPSILON)
    pickands = float(np.clip(np.log2(numerator / denominator), -5.0, 5.0))
    top = ordered[-ORDER_K:]
    median = max(float(np.median(ordered)), EPSILON)
    top_mean_ratio = float(np.log1p(float(np.mean(top)) / median))
    top_mass_share = float(np.sum(top) / max(float(np.sum(ordered)), EPSILON))
    result = np.asarray([pickands, top_mean_ratio, top_mass_share], dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ContractError("extreme-order statistic contract differs")
    return result


def extreme_order_features(sequence: np.ndarray) -> np.ndarray:
    path = v59.v58.v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        for start, stop in WINDOWS:
            values = path[start:stop, channel]
            median = float(np.median(values))
            q25, q75 = np.quantile(values, (0.25, 0.75))
            normalized = (values - median) / max(float(q75 - q25), EPSILON)
            for lag in LAGS:
                increments = normalized[lag:] - normalized[:-lag]
                output.extend(extreme_order_statistics(increments))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("extreme-order feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    gaussian = np.abs(rng.normal(size=512))
    heavy = rng.pareto(1.6, size=512) + 1.0
    gaussian_stats = extreme_order_statistics(gaussian)
    heavy_stats = extreme_order_statistics(heavy)
    if not heavy_stats[2] > gaussian_stats[2] + 0.05:
        raise ContractError("synthetic heavy-tail concentration direction guard failed")
    affine = extreme_order_statistics(7.0 * gaussian)
    if not np.allclose(gaussian_stats, affine, rtol=1e-10, atol=1e-10):
        raise ContractError("affine-scale invariance guard failed")
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = extreme_order_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(feature, extreme_order_features(extended[:289])):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "gaussian_top_mass_share": float(gaussian_stats[2]),
        "heavy_tail_top_mass_share": float(heavy_stats[2]),
        "affine_scale_invariant": True,
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
        features[position] = extreme_order_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v59.v58.v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    minimum_count = min((stop - start - max(LAGS)) for start, stop in WINDOWS)
    passed = bool(
        minimum_count >= int(gate["minimum_increment_count_per_cell"])
        and nonzero_share >= float(gate["minimum_nonzero_share"])
        and positive_variance >= int(gate["minimum_positive_variance_features"])
    )
    return {
        **metadata,
        "minimum_increment_count_per_cell": minimum_count,
        "nonzero_share": nonzero_share,
        "positive_variance_features": positive_variance,
        "target_used": False,
        "passed": passed,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v60 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.extreme_order_increment_tail_residual.preflight.v60",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2,
        "maximum_model_fits": 12 if support["passed"] else 0,
        "synthetic": synthetic_receipt(),
        "historical_support": support,
        "official_v42_used_for_features_gates_selection": False,
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
    result.update({
        "schema_version": "p3.extreme_order_increment_tail_residual.result.v60",
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_EXTREME_ORDER_TAIL_CANDIDATES",
        "duplication_audit": config["duplication_audit"],
        "primary_sources": config["primary_sources"],
    })
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 extreme-order increment-tail residual cycle v60",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- Fixed increment extreme-order spacing/concentration is distinct from aggregate jump variation, full increment ECF, level excursions and cross-channel tail copulas.",
        "- The official v42 sign reversal was excluded from features, gates, fitting and selection; only the pre-existing fixed aggregate transport penalty remains.",
        "- Pickands (1975) motivates the mechanism only; the repeatedly exposed 182-case surface is EXPLORATORY_ONLY.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(
            f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; "
            f"raw {points['raw_gain']:+.6f} points; adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; "
            f"worst block {item['worst_block_delta_m']:+.9f}m; lead {item['worst_lead_delta_m']:+.9f}m; station-lead {item['worst_station_lead_delta_m']:+.9f}m; "
            f"tail {item['worst_reference_tail_block_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}."
        )
    lines.append("Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.")
    return "\n".join(lines) + "\n"


def write_support_stop(config: dict[str, Any], preflight: dict[str, Any]) -> None:
    result = {
        "schema_version": "p3.extreme_order_increment_tail_residual.result.v60",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "decision": "STOP_SUPPORT_GATE_ZERO_FIT",
        "fit_count": 0,
        "support_receipt": preflight["historical_support"],
        "duplication_audit": config["duplication_audit"],
        "primary_sources": config["primary_sources"],
        "data_access": {"historical_target_rows": 0, "official_test_rows": 0, "official_sample_rows": 0, "official_submission_rows": 0, "hidden_truth_rows": 0, "csv_materializations": 0, "uploads": 0},
        "provenance": {"runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "input_sha256": config["inputs"]},
    }
    result_path = ARTIFACT / "result.json"
    write_new(result_path, canonical(result))
    write_new(REPORT / "result.json", canonical(result))
    report_path = REPORT / "report-source.md"
    write_new(report_path, b"# P3 extreme-order increment-tail residual cycle v60\n\n## Conclusion\n\n- **STOP_SUPPORT_GATE_ZERO_FIT**.\n- Sealed target-free support failed; no target, outer score, official input, CSV, or upload was used.\n")
    write_new(REPORT / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "result_sha256": sha256(result_path), "report_sha256": sha256(report_path), "fit_count": 0, "official_access": 0, "csv_materializations": 0, "uploads": 0}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT.exists() or REPORT.exists() or LOCK.exists():
        raise ContractError("v60 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "official_access": 0}))
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    if preflight["status"] == "STOP_SUPPORT_GATE":
        write_support_stop(config, preflight)
        write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Extreme-order spacings characterize tail behavior | Pickands 1975, DOI:10.1214/aos/1176343003 | mechanism only |\n| Sealed target-free support failed before scoring | v60 preflight receipt | zero-fit decision |\n")
        print(json.dumps({"status": "COMPLETE", "decision": "STOP_SUPPORT_GATE_ZERO_FIT", "fit_count": 0, "official_access": 0}, ensure_ascii=False))
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Fixed extreme-order spacings characterize tail behavior | Pickands, Annals of Statistics 3, 1975, DOI:10.1214/aos/1176343003 | mechanism only |\n| Prior P3 jump-variation, ECF, excursion and tail-copula axes do not encode marginal increment extreme-order spacing/concentration | sealed duplication audit | novelty boundary |\n| Channels, windows, lags, k, statistics, residual model and validation were fixed before scoring | sealed v60 config | execution contract |\n| Official v42 aggregate reversal was excluded from features, gates, fitting and selection | sealed v60 contract | transport-risk boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
