"""Sealed P3 v64 Allan/Hadamard scale-spectrum residual experiment."""

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

import run_p3_volatility_clustering_residual_cycle_20260901_v63 as v63  # noqa: E402

EXPERIMENT_ID = "p3_allan_hadamard_scale_residual_cycle_20260901_v64"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS, SCALES = ((0, 145), (72, 145)), (1, 2, 4, 8)
FEATURE_COUNT = 64
BASE = v63.BASE
SPEC_CLASS = v63.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_AHVAR64_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_AHVAR64_RIDGE2048_ADD10", 2048.0),
)
BLEND, EPSILON = 0.10, 1e-12
sha256, canonical, write_new = v63.sha256, v63.canonical, v63.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v64 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.allan_hadamard_scale_residual.config.v64",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_ALLAN_HADAMARD_SCALE_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "scales": tuple(encoder["averaging_scales_rows"]) == SCALES,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_v42_excluded": "excluded" in config["duplication_audit"]["official_v42_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v64 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def overlapping_block_means(values: np.ndarray, scale: int) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if scale < 1 or len(path) < 3 * scale or not np.isfinite(path).all():
        raise ContractError("Allan/Hadamard block support differs")
    return np.convolve(path, np.ones(scale, dtype=np.float64) / scale, mode="valid")


def allan_hadamard_statistics(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    median = float(np.median(path))
    q25, q75 = np.quantile(path, (0.25, 0.75))
    normalized = (path - median) / max(float(q75 - q25), EPSILON)
    output: list[float] = []
    for scale in SCALES:
        means = overlapping_block_means(normalized, scale)
        first = means[scale:] - means[:-scale]
        second = means[2 * scale :] - 2.0 * means[scale:-scale] + means[: -2 * scale]
        if len(second) < 40:
            raise ContractError("largest-scale Hadamard support below sealed minimum")
        output.extend([0.5 * float(np.mean(np.square(first))), float(np.mean(np.square(second))) / 6.0])
    result = np.asarray(output, dtype=np.float64)
    if result.shape != (8,) or not np.isfinite(result).all():
        raise ContractError("Allan/Hadamard statistic contract differs")
    return result


def transformed_path(sequence: np.ndarray) -> np.ndarray:
    return v63.v62.v61.v60.v59.v58.v57.v56.v55.v54.v53.v52.v50.v49.v48.v47.v46.v44.v43.v42.v41.v40.v39.v38.v36.v26.transformed_path(sequence)


def allan_hadamard_features(sequence: np.ndarray) -> np.ndarray:
    path = transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        for start, stop in WINDOWS:
            output.extend(allan_hadamard_statistics(path[start:stop, channel]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("Allan/Hadamard feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    white = rng.normal(size=1024)
    random_walk = np.cumsum(rng.normal(size=1024))
    white_stats = allan_hadamard_statistics(white)
    walk_stats = allan_hadamard_statistics(random_walk)
    white_ratio = float(white_stats[-2] / max(white_stats[0], EPSILON))
    walk_ratio = float(walk_stats[-2] / max(walk_stats[0], EPSILON))
    if not white_ratio < 0.40 or not walk_ratio > 2.0:
        raise ContractError("white-noise/random-walk scale-profile guard failed")
    linear = np.linspace(-3.0, 7.0, 256)
    linear_stats = allan_hadamard_statistics(linear)
    if not linear_stats[-1] <= max(linear_stats[-2] * 1e-12, 1e-24):
        raise ContractError("linear-drift Hadamard suppression guard failed")
    if not np.allclose(walk_stats, allan_hadamard_statistics(-7.0 * random_walk + 3.0), rtol=1e-10, atol=1e-10):
        raise ContractError("sign-scale invariance guard failed")
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * base) + 0.1 * index * base for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = allan_hadamard_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(feature, allan_hadamard_features(extended[:289])):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "white_allan_scale8_over_scale1": white_ratio,
        "random_walk_allan_scale8_over_scale1": walk_ratio,
        "linear_hadamard_scale8": float(linear_stats[-1]),
        "linear_allan_scale8": float(linear_stats[-2]),
        "sign_scale_invariant": True,
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
        features[position] = allan_hadamard_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v63.v62.case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    effective_terms = min((stop - start) - 3 * max(SCALES) + 1 for start, stop in WINDOWS)
    passed = bool(
        effective_terms >= int(gate["minimum_effective_hadamard_terms_at_largest_scale"])
        and nonzero_share >= float(gate["minimum_nonzero_share"])
        and positive_variance >= int(gate["minimum_positive_variance_features"])
    )
    return {
        **metadata,
        "minimum_effective_hadamard_terms_at_largest_scale": effective_terms,
        "nonzero_share": nonzero_share,
        "positive_variance_features": positive_variance,
        "target_used": False,
        "passed": passed,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v64 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.allan_hadamard_scale_residual.preflight.v64",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2,
        "maximum_model_fits": 12 if support["passed"] else 0,
        "synthetic": synthetic_receipt(),
        "historical_support": support,
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
            "schema_version": "p3.allan_hadamard_scale_residual.result.v64",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_ALLAN_HADAMARD_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 Allan/Hadamard scale-spectrum residual cycle v64",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- Fixed Allan/Hadamard profiles encode scale-dependent block-average stability and drift suppression, not jump mass, volatility ACF, or a frequency decomposition.",
        "- Prior outputs and official v42 feedback are excluded; the 182-case surface is EXPLORATORY_ONLY.",
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
        raise ContractError("v64 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v64 support gate failed; zero-fit closure required")
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
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Two-sample variance measures scale-dependent stability | Allan 1966, DOI:10.1109/PROC.1966.4631506 | mechanism only |\n| Prior P3 jump, volatility-ACF and spectral axes do not compute overlapping block-average first/second-difference variances | repository semantic audit | novelty boundary |\n| Prior outputs and official v42 feedback were excluded | sealed v64 contract | reuse boundary |\n",
    )
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
