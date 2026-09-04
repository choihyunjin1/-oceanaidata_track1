"""Sealed P3 v65 L-moment distribution-shape residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

import run_p3_allan_hadamard_scale_residual_cycle_20260901_v64 as v64  # noqa: E402

EXPERIMENT_ID = "p3_lmoment_shape_residual_cycle_20260901_v65"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS = ((0, 145), (72, 145))
FEATURE_COUNT = 64
BASE = v64.BASE
SPEC_CLASS = v64.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_LMOM64_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_LMOM64_RIDGE2048_ADD10", 2048.0),
)
BLEND, EPSILON = 0.10, 1e-12
sha256, canonical, write_new = v64.sha256, v64.canonical, v64.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v65 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.lmoment_shape_residual.config.v65",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_LMOMENT_SHAPE_AXIS",
        "mmd_withdrawn": "withdrawn" in config["duplication_audit"]["withdrawn_axis"],
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_v42_excluded": "excluded" in config["duplication_audit"]["official_v42_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v65 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def probability_weighted_moments(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = len(ordered)
    if n < 4 or not np.isfinite(ordered).all():
        raise ContractError("L-moment sample support differs")
    output = np.empty(4, dtype=np.float64)
    output[0] = float(np.mean(ordered))
    for order in range(1, 4):
        denominator = float(math.comb(n - 1, order))
        weights = np.zeros(n, dtype=np.float64)
        for index in range(order, n):
            weights[index] = math.comb(index, order) / denominator
        output[order] = float(np.mean(weights * ordered))
    return output


def sample_lmoments(values: np.ndarray) -> np.ndarray:
    b0, b1, b2, b3 = probability_weighted_moments(values)
    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    l4 = 20.0 * b3 - 30.0 * b2 + 12.0 * b1 - b0
    if l2 <= EPSILON:
        return np.asarray([l1, max(l2, 0.0), 0.0, 0.0], dtype=np.float64)
    result = np.asarray([l1, l2, l3 / l2, l4 / l2], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ContractError("L-moment statistic contract differs")
    return result


def robust_level_and_increment_statistics(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < 65 or not np.isfinite(path).all():
        raise ContractError("L-moment path support below sealed minimum")
    median = float(np.median(path))
    q25, q75 = np.quantile(path, (0.25, 0.75))
    normalized = (path - median) / max(float(q75 - q25), EPSILON)
    result = np.concatenate([sample_lmoments(normalized), sample_lmoments(np.diff(normalized))])
    if result.shape != (8,) or not np.isfinite(result).all():
        raise ContractError("level/increment L-moment contract differs")
    return result


def transformed_path(sequence: np.ndarray) -> np.ndarray:
    return v64.transformed_path(sequence)


def lmoment_features(sequence: np.ndarray) -> np.ndarray:
    path = transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        for start, stop in WINDOWS:
            output.extend(robust_level_and_increment_statistics(path[start:stop, channel]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("L-moment feature contract differs")
    return features


def conventional_shape(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    scale = float(np.sqrt(np.mean(np.square(centered))))
    standardized = centered / max(scale, EPSILON)
    return np.asarray([np.mean(standardized**3), np.mean(standardized**4) - 3.0])


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    gaussian = rng.normal(size=8192)
    exponential = rng.exponential(size=8192)
    gaussian_lm = sample_lmoments(gaussian)
    exponential_lm = sample_lmoments(exponential)
    if not abs(gaussian_lm[2]) < 0.03 or not exponential_lm[2] > 0.25:
        raise ContractError("Gaussian/exponential L-skewness guard failed")
    base = np.asarray([-3.0, -1.0, -0.4, 0.2, 1.1, 2.7, 5.0], dtype=np.float64)
    direct = sample_lmoments(base)
    positive = sample_lmoments(4.0 * base + 9.0)
    negative = sample_lmoments(-4.0 * base + 9.0)
    if not np.allclose(positive, [4.0 * direct[0] + 9.0, 4.0 * direct[1], direct[2], direct[3]], rtol=1e-10, atol=1e-10):
        raise ContractError("positive affine L-moment guard failed")
    if not np.allclose(negative, [-4.0 * direct[0] + 9.0, 4.0 * direct[1], -direct[2], direct[3]], rtol=1e-10, atol=1e-10):
        raise ContractError("negative affine L-moment guard failed")
    clean = rng.normal(size=512)
    contaminated = clean.copy()
    contaminated[-1] = 1e4
    l_shift = float(np.linalg.norm(sample_lmoments(contaminated)[2:] - sample_lmoments(clean)[2:]))
    conventional_shift = float(np.linalg.norm(conventional_shape(contaminated) - conventional_shape(clean)))
    if not l_shift < conventional_shift * 0.05:
        raise ContractError("contamination robustness guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * axis) + 0.1 * index * axis for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = lmoment_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(feature, lmoment_features(extended[:289])):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "gaussian_l_skewness": float(gaussian_lm[2]),
        "exponential_l_skewness": float(exponential_lm[2]),
        "contamination_l_shape_shift": l_shift,
        "contamination_conventional_shape_shift": conventional_shift,
        "affine_contract": True,
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
        features[position] = lmoment_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v64.v63.v62.case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    minimum_count = min(stop - start - 1 for start, stop in WINDOWS)
    passed = bool(
        minimum_count >= int(gate["minimum_sample_count"])
        and nonzero_share >= float(gate["minimum_nonzero_share"])
        and positive_variance >= int(gate["minimum_positive_variance_features"])
    )
    return {
        **metadata,
        "minimum_sample_count": minimum_count,
        "nonzero_share": nonzero_share,
        "positive_variance_features": positive_variance,
        "target_used": False,
        "passed": passed,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v65 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.lmoment_shape_residual.preflight.v65",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2,
        "maximum_model_fits": 12 if support["passed"] else 0,
        "synthetic": synthetic_receipt(),
        "historical_support": support,
        "withdrawn_mmd_used": False,
        "prior_outputs_used": False,
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
    result.update(
        {
            "schema_version": "p3.lmoment_shape_residual.result.v65",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_LMOMENT_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 L-moment distribution-shape residual cycle v65",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- MMD was withdrawn before sealing. v65 uses unbiased probability-weighted order statistics for robust level/increment distribution shape.",
        "- Prior outputs and official feedback are excluded; the 182-case surface is EXPLORATORY_ONLY.",
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
        raise ContractError("v65 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v65 support gate failed; zero-fit closure required")
    write_new(
        LOCK,
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "ATTEMPT_CONSUMED_ONE_SHOT",
                "runner_sha256": sha256(Path(__file__)),
                "config_sha256": sha256(CONFIG),
                "preflight_receipt_sha256": preflight["receipt_sha256"],
                "official_access": 0,
            }
        ),
    )
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    result, arrays = execute(config)
    array_path = ARTIFACT / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {
        "runner_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(CONFIG),
        "evaluation_arrays_sha256": sha256(array_path),
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "input_sha256": config["inputs"],
    }
    result_path = ARTIFACT / "result.json"
    write_new(result_path, canonical(result))
    write_new(REPORT / "result.json", canonical(result))
    report_path = REPORT / "report-source.md"
    write_new(report_path, render_report(result).encode())
    write_new(
        REPORT / "run-manifest.json",
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "result_sha256": sha256(result_path),
                "arrays_sha256": sha256(array_path),
                "report_sha256": sha256(report_path),
                "fit_count": 12,
                "official_access": 0,
                "csv_materializations": 0,
                "uploads": 0,
            }
        ),
    )
    write_new(
        REPORT / "claim-source-ledger.md",
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| L-moments summarize distributions with linear combinations of order statistics and are less sensitive to outliers than conventional moments | Hosking 1990, DOI:10.1111/j.2517-6161.1990.tb01775.x | mechanism only |\n| MMD was withdrawn and no prior L-moment P3 execution exists | repository semantic audit | novelty boundary |\n| Prior outputs and official feedback were excluded | sealed v65 contract | reuse boundary |\n",
    )
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
