"""Sealed P3 v59 delay-cloud intrinsic-dimension residual experiment."""

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

import run_p3_empirical_tail_copula_residual_cycle_20260901_v58 as v58  # noqa: E402

EXPERIMENT_ID = "p3_intrinsic_dimension_residual_cycle_20260901_v59"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS, DELAY, DIMENSION, KS = ((0, 145), (72, 145)), 2, 4, (5, 10)
FEATURE_COUNT = 32
BASE = v58.BASE
SPECS = (
    v58.v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_1_IDIM32_RIDGE512_ADD10", 512.0),
    v58.v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.Spec("P3_2_IDIM32_RIDGE2048_ADD10", 2048.0),
)
BLEND, EPSILON = 0.10, 1e-12
sha256, canonical, write_new = v58.sha256, v58.canonical, v58.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v59 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.intrinsic_dimension_residual.config.v59",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_DELAY_CLOUD_INTRINSIC_DIMENSION_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "embedding": encoder["delay_rows"] == DELAY and encoder["embedding_dimension"] == DIMENSION,
        "ks": tuple(encoder["neighbor_k"]) == KS,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v59 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def delay_cloud(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    median = float(np.median(path))
    q25, q75 = np.quantile(path, (0.25, 0.75))
    path = (path - median) / max(float(q75 - q25), EPSILON)
    count = len(path) - DELAY * (DIMENSION - 1)
    if count < 60:
        raise ContractError("delay-cloud neighbor support failed")
    cloud = np.column_stack([path[offset * DELAY : offset * DELAY + count] for offset in range(DIMENSION)])
    if not np.isfinite(cloud).all():
        raise ContractError("delay cloud is nonfinite")
    return cloud


def intrinsic_statistics_from_cloud(cloud: np.ndarray) -> np.ndarray:
    points = np.asarray(cloud, dtype=np.float64)
    distance = np.sqrt(np.sum(np.square(points[:, None, :] - points[None, :, :]), axis=2))
    np.fill_diagonal(distance, np.inf)
    nearest = np.sort(distance, axis=1)[:, : max(KS)]
    output: list[float] = []
    for k in KS:
        selected = np.maximum(nearest[:, :k], EPSILON)
        radius = selected[:, k - 1]
        denominator = np.sum(np.log(radius[:, None] / selected[:, : k - 1]), axis=1)
        local = (k - 1) / np.maximum(denominator, EPSILON)
        local = np.clip(local[np.isfinite(local)], 0.0, 50.0)
        if len(local) < 60:
            raise ContractError("valid local-dimension support failed")
        q25, median, q75 = np.quantile(local, (0.25, 0.5, 0.75))
        output.extend([float(median), float(q75 - q25)])
    result = np.asarray(output, dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ContractError("intrinsic-dimension statistic contract differs")
    return result


def intrinsic_dimension_features(sequence: np.ndarray) -> np.ndarray:
    path = v58.v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        for start, stop in WINDOWS:
            output.extend(intrinsic_statistics_from_cloud(delay_cloud(path[start:stop, channel])))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("intrinsic-dimension feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    coordinate = np.linspace(-3.0, 3.0, 240)
    line = np.column_stack([coordinate, 2.0 * coordinate, -coordinate, 0.5 * coordinate])
    volume = rng.normal(size=(240, 4))
    line_dim = intrinsic_statistics_from_cloud(line)[0]
    volume_dim = intrinsic_statistics_from_cloud(volume)[0]
    if not volume_dim > line_dim + 0.5:
        raise ContractError("synthetic intrinsic-dimension direction guard failed")
    if not np.allclose(intrinsic_statistics_from_cloud(line), intrinsic_statistics_from_cloud(7.0 * line + 3.0), rtol=1e-10, atol=1e-10):
        raise ContractError("affine-scale invariance guard failed")
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = intrinsic_dimension_features(sequence)
    return {"feature_count": len(feature), "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(feature).all()), "line_dimension": float(line_dim), "volume_dimension": float(volume_dim), "affine_scale_invariant": True}


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(BASE.SEQUENCES, mmap_mode="r")
    station_codes = np.load(BASE.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = intrinsic_dimension_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v58.v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v32.v23.case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(nonzero_share >= float(gate["minimum_nonzero_share"]) and positive_variance >= int(gate["minimum_positive_variance_features"]))
    return {**metadata, "minimum_delay_cloud_points": 67, "nonzero_share": nonzero_share, "positive_variance_features": positive_variance, "target_used": False, "passed": passed}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v59 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {"schema_version": "p3.intrinsic_dimension_residual.preflight.v59", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12 if support["passed"] else 0, "synthetic": synthetic_receipt(), "historical_support": support, "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = BASE.surface_features, BASE.SPECS
    BASE.surface_features, BASE.SPECS = surface_features, SPECS
    try:
        result, arrays = BASE.execute(config)
    finally:
        BASE.surface_features, BASE.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.intrinsic_dimension_residual.result.v59", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_INTRINSIC_DIMENSION_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 delay-cloud intrinsic-dimension residual cycle v59", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- Fixed neighbor-distance-ratio dimension differs from recurrence, persistence, trajectory geometry and forward local divergence; no prior output or official v42 result is used.", "- Levina and Bickel (2004) motivates the mechanism only; the repeatedly exposed 182-case surface is EXPLORATORY_ONLY."]
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
        raise ContractError("v59 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v59 support gate failed; zero-fit closure required")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Local neighbor-distance ratios estimate intrinsic dimension | Levina and Bickel, NIPS 2004 official proceedings | mechanism only |\n| Prior P3 recurrence, persistence, trajectory and divergence axes do not estimate local intrinsic dimension | sealed duplication audit | novelty boundary |\n| Delay, embedding, k, windows, summaries, residual model and validation were fixed before scoring | sealed v59 config | execution contract |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
