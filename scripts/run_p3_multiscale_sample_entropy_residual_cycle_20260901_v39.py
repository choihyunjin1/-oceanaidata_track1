"""Sealed P3 v39 multiscale sample-entropy residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

import run_p3_cross_quantilogram_residual_cycle_20260901_v38 as v38  # noqa: E402

EXPERIMENT_ID = "p3_multiscale_sample_entropy_residual_cycle_20260901_v39"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = v38.SEQUENCES
STATIONS = v38.STATIONS
CHANNELS = (0, 1, 2, 5)
CHANNEL_NAMES = ("hs", "tp", "hmax", "wspd")
SCALES = (1, 2, 4, 8)
M = 2
R_FRACTION = 0.2
FEATURE_COUNT = 48
SPECS = (
    v38.v36.v26.Spec("P3_1_MSE48_RIDGE512_ADD10", 512.0),
    v38.v36.v26.Spec("P3_2_MSE48_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10


class ContractError(RuntimeError):
    """Raised when the sealed v39 contract differs."""


sha256 = v38.sha256
canonical = v38.canonical
write_new = v38.write_new


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"]
        == "p3.multiscale_sample_entropy_residual.config.v39",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_MULTISCALE_COMPLEXITY_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "scales": tuple(encoder["coarse_grain_scales"]) == SCALES,
        "dimension": encoder["template_dimension"] == M,
        "tolerance": encoder["tolerance_fraction_of_original_sd"] == R_FRACTION,
        "features": encoder["feature_count"] == FEATURE_COUNT,
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
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v39 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def coarse_grain(values: np.ndarray, scale: int) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    count = len(path) // scale
    if count < M + 2:
        raise ContractError("coarse path is too short")
    return path[: count * scale].reshape(count, scale).mean(axis=1)


def match_probability(values: np.ndarray, dimension: int, tolerance: float) -> float:
    templates = np.lib.stride_tricks.sliding_window_view(values, dimension)
    count = len(templates)
    pair_count = count * (count - 1) // 2
    if pair_count <= 0:
        raise ContractError("template pair count is zero")
    distance = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
    matches = int(np.count_nonzero(np.triu(distance <= tolerance, k=1)))
    return float((matches + 0.5) / (pair_count + 1.0))


def sample_entropy_triplet(values: np.ndarray, scale: int) -> tuple[float, float, float]:
    path = np.asarray(values, dtype=np.float64)
    tolerance = R_FRACTION * float(np.std(path, ddof=0))
    if tolerance <= 1e-15:
        tolerance = 1e-15
    coarse = coarse_grain(path, scale)
    probability_m = match_probability(coarse, M, tolerance)
    probability_m1 = match_probability(coarse, M + 1, tolerance)
    entropy = -float(np.log(probability_m1 / probability_m))
    output = (entropy, probability_m, probability_m1)
    if not np.isfinite(output).all():
        raise ContractError("sample-entropy triplet is nonfinite")
    return output


def multiscale_entropy_features(sequence: np.ndarray) -> np.ndarray:
    path = v38.v36.v26.transformed_path(sequence)
    native = path[::2]
    if native.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        for scale in SCALES:
            output.extend(sample_entropy_triplet(native[:, channel], scale))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("multiscale entropy feature contract differs")
    return features


def complexity_receipt() -> dict[str, Any]:
    time_axis = np.arange(145, dtype=np.float64)
    periodic = np.sin(2.0 * np.pi * time_axis / 12.0)
    noise = np.random.default_rng(20260901).normal(size=145)
    periodic_entropy = sample_entropy_triplet(periodic, 1)[0]
    noise_entropy = sample_entropy_triplet(noise, 1)[0]
    if noise_entropy <= periodic_entropy + 0.25:
        raise ContractError("synthetic complexity ordering failed")
    constant = sample_entropy_triplet(np.ones(145), 8)
    return {
        "periodic_entropy_scale1": periodic_entropy,
        "noise_entropy_scale1": noise_entropy,
        "noise_minus_periodic": noise_entropy - periodic_entropy,
        "constant_scale8": list(constant),
    }


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = multiscale_entropy_features(sequence)
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "complexity": complexity_receipt(),
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if np.load(SEQUENCES, mmap_mode="r").shape != (24360, 289, 10):
        raise ContractError("sequence cache shape differs")
    if np.load(STATIONS, mmap_mode="r").shape != (24360,):
        raise ContractError("station cache shape differs")
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v39 exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.multiscale_sample_entropy_residual.preflight.v39",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2,
        "maximum_model_fits": 12,
        "synthetic": synthetic_receipt(),
        "official_access": 0,
        "csv_materializations": 0,
        "uploads": 0,
        "config_status": config["status"],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(SEQUENCES, mmap_mode="r")
    station_codes = np.load(STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = multiscale_entropy_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, profile = v38.v36.v32.v23.case_surface()
    features, feature_receipt = surface_features(cases)
    original_specs = v38.v36.v32.v28.SPECS
    v38.v36.v32.v28.SPECS = SPECS
    try:
        predictions, receipts = v38.v36.v32.v28.crossfit(
            cases, features, targets, reference
        )
        frame = v38.v36.v32.v23.long_frame(cases, targets, reference)
        scored = [
            v38.v36.v32.v28.score(frame, predictions[spec.name], spec)
            for spec in SPECS
        ]
    finally:
        v38.v36.v32.v28.SPECS = original_specs
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {
        "schema_version": "p3.multiscale_sample_entropy_residual.result.v39",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE"
        if passing
        else "NO_GO_ALL_MULTISCALE_ENTROPY_CANDIDATES",
        "surface_claim": config["validation"]["surface"],
        "reference": config["reference"],
        "duplication_audit": config["duplication_audit"],
        "primary_sources": config["primary_sources"],
        "feature_receipt": feature_receipt,
        "candidates": scored,
        "fit_receipts": receipts,
        "fit_count": 12,
        "data_profile": profile,
        "data_access": {
            "historical_target_rows": 1092,
            "official_test_rows": 0,
            "official_sample_rows": 0,
            "official_submission_rows": 0,
            "hidden_truth_rows": 0,
            "csv_materializations": 0,
            "uploads": 0,
        },
        "execution": {
            "python": platform.python_version(),
            "elapsed_seconds": time.perf_counter() - started,
            "candidate_count": 2,
            "result_based_tuning": False,
            "outer_result_parameter_changes": 0,
            "row_deletion": 0,
        },
    }
    arrays = {
        "truth": targets,
        "uniform": reference,
        "candidate_1": predictions[SPECS[0].name],
        "candidate_2": predictions[SPECS[1].name],
        "anchor_id": cases["anchor_id"].to_numpy(np.int32),
        "lead_h": np.asarray(v38.v36.v32.v23.LEADS, dtype=np.int16),
        "block": cases["block"].to_numpy(dtype="U5"),
        "station": cases["station"].to_numpy(dtype="U5"),
        "episode": cases["episode_id"].to_numpy(dtype="U32"),
    }
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 multiscale sample-entropy residual cycle v39",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v39 measures amplitude-sensitive template complexity across fixed coarse-grained scales; it is not v29 ordinal/recurrence geometry, v33 covariance, or v38 tail-hit directionality.",
        "- Richman/Moorman and Costa et al. motivate the statistic only; the 182-case surface remains EXPLORATORY_ONLY.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(
            f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; "
            f"delta {metric['delta_candidate_minus_uniform']:+.9f}m; raw {points['raw_gain']:+.6f} points; "
            f"transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; "
            f"worst block {item['worst_block_delta_m']:+.9f}m; worst lead {item['worst_lead_delta_m']:+.9f}m; "
            f"worst station-lead {item['worst_station_lead_delta_m']:+.9f}m; "
            f"worst reference-tail block {item['worst_reference_tail_block_delta_m']:+.9f}m; "
            f"episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}."
        )
    lines.append(
        "Official test/sample/submission/hidden access, CSV materialization, and upload were all zero."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT.exists() or REPORT.exists() or LOCK.exists():
        raise ContractError("v39 exactly-once namespace already exists")
    config = load_config()
    preflight = preflight_payload()
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
    report_path = REPORT / "report-source.md"
    write_new(report_path, render_report(result).encode())
    write_new(REPORT / "result.json", canonical(result))
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
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Sample entropy measures conditional template-match complexity and was proposed for short noisy series | Richman and Moorman, AJP Heart 278, 2000, DOI:10.1152/ajpheart.2000.278.6.H2039 | mechanism only |\n| Multiscale entropy applies entropy after fixed coarse-graining at multiple scales | Costa et al., Physical Review Letters 89, 2002, DOI:10.1103/PhysRevLett.89.068102 | representation rationale only |\n| v39 does not reuse prior-cycle predictions, routers, or posthoc thresholds | sealed v39 contract | anti-posthoc boundary |\n",
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "decision": result["decision"],
                "fit_count": 12,
                "official_access": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
