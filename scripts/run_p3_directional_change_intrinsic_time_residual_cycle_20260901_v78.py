"""Sealed P3 v78 directional-change intrinsic-time representation."""

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

import run_p3_natural_time_increment_energy_residual_cycle_20260901_v77 as v77  # noqa: E402

EXPERIMENT_ID = "p3_directional_change_intrinsic_time_residual_cycle_20260901_v78"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS = ((0, 145), (72, 145))
THRESHOLDS = (0.5, 1.0, 2.0)
FEATURE_COUNT = 144
BASE = v77.BASE
SPEC_CLASS = v77.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_DIRCHANGE144_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_DIRCHANGE144_RIDGE2048_ADD10", 2048.0),
)
BLEND, MAD_SCALE, EPSILON = 0.10, 1.4826, 1e-12
sha256, canonical, write_new = v77.sha256, v77.canonical, v77.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v78 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"]
        == "p3.directional_change_intrinsic_time_residual.config.v78",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_P3_DIRECTIONAL_CHANGE_INTRINSIC_TIME_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "thresholds": tuple(float(value) for value in encoder["thresholds_mad_units"])
        == THRESHOLDS,
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
        "specs": tuple(
            (item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]
        )
        == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(
            float(item["additive_residual_weight"]) == BLEND
            for item in config["model"]["candidates"]
        ),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_excluded": "excluded" in config["duplication_audit"]["official_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v78 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def fill_prefix(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (289, 10):
        raise ContractError("raw context shape differs")
    result = np.empty_like(raw)
    index = np.arange(len(raw), dtype=np.float64)
    for column in range(raw.shape[1]):
        finite = np.isfinite(raw[:, column])
        result[:, column] = (
            np.interp(index, index[finite], raw[finite, column]) if finite.any() else 0.0
        )
    return result


def robust_normalize(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < 24 or not np.isfinite(path).all():
        raise ContractError("directional-change path support differs")
    center = float(np.median(path))
    scale = MAD_SCALE * float(np.median(np.abs(path - center)))
    return np.zeros_like(path) if scale <= EPSILON else (path - center) / scale


def directional_change_events(
    values: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < 3 or threshold <= 0.0 or not np.isfinite(path).all():
        raise ContractError("directional-change event input differs")
    direction = 0
    high = low = float(path[0])
    high_index = low_index = 0
    last_extreme = float(path[0])
    last_extreme_index = 0
    directions: list[int] = []
    overshoots: list[float] = []
    durations: list[int] = []
    for index, value_raw in enumerate(path[1:], start=1):
        value = float(value_raw)
        if direction == 0:
            if value > high:
                high, high_index = value, index
            if value < low:
                low, low_index = value, index
            if value - low >= threshold:
                direction = 1
                last_extreme, last_extreme_index = low, low_index
                high, high_index = value, index
            elif high - value >= threshold:
                direction = -1
                last_extreme, last_extreme_index = high, high_index
                low, low_index = value, index
            continue
        if direction == 1:
            if value > high:
                high, high_index = value, index
            if high - value >= threshold:
                directions.append(1)
                overshoots.append(max(0.0, high - last_extreme - threshold))
                durations.append(max(1, high_index - last_extreme_index))
                direction = -1
                last_extreme, last_extreme_index = high, high_index
                low, low_index = value, index
        else:
            if value < low:
                low, low_index = value, index
            if value - low >= threshold:
                directions.append(-1)
                overshoots.append(max(0.0, last_extreme - low - threshold))
                durations.append(max(1, low_index - last_extreme_index))
                direction = 1
                last_extreme, last_extreme_index = low, low_index
                high, high_index = value, index
    return (
        np.asarray(directions, dtype=np.int8),
        np.asarray(overshoots, dtype=np.float64),
        np.asarray(durations, dtype=np.int64),
    )


def threshold_statistics(values: np.ndarray, threshold: float) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    directions, overshoots, durations = directional_change_events(path, threshold)
    count = len(directions)
    if count == 0:
        return np.zeros(6, dtype=np.float64)
    denominator = max(1, len(path) - 1)
    features = np.asarray(
        [
            count / denominator,
            np.mean(directions > 0),
            np.mean(overshoots) / threshold,
            np.max(overshoots) / threshold,
            np.mean(durations) / denominator,
            np.max(durations) / denominator,
        ],
        dtype=np.float64,
    )
    if features.shape != (6,) or not np.isfinite(features).all():
        raise ContractError("directional-change statistics differ")
    return features


def directional_change_features(sequence: np.ndarray) -> np.ndarray:
    raw = fill_prefix(np.asarray(sequence)[:289])[::2]
    features = np.concatenate(
        [
            threshold_statistics(robust_normalize(raw[start:stop, channel]), threshold)
            for channel in CHANNELS
            for start, stop in WINDOWS
            for threshold in THRESHOLDS
        ]
    )
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("directional-change feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    trend = np.linspace(-3.0, 3.0, 73)
    triangle = np.tile(np.r_[np.linspace(-2.0, 2.0, 9), np.linspace(1.5, -2.0, 8)], 5)[:73]
    trend_counts = [len(directional_change_events(trend, value)[0]) for value in THRESHOLDS]
    triangle_counts = [len(directional_change_events(triangle, value)[0]) for value in THRESHOLDS]
    if (
        trend_counts != [0, 0, 0]
        or not triangle_counts[0] >= triangle_counts[1] >= triangle_counts[2] > 0
    ):
        raise ContractError("trend or threshold monotonicity guard failed")
    directions, overshoots, durations = directional_change_events(triangle, 1.0)
    if len(directions) < 4 or np.any(directions[1:] == directions[:-1]):
        raise ContractError("alternating-event guard failed")
    inverted = directional_change_events(-triangle, 1.0)
    if not np.array_equal(directions, -inverted[0]):
        raise ContractError("direction reversal guard failed")
    if not np.allclose(overshoots, inverted[1], rtol=0.0, atol=1e-12) or not np.array_equal(
        durations, inverted[2]
    ):
        raise ContractError("overshoot reversal guard failed")
    normalized = robust_normalize(triangle)
    affine = robust_normalize(7.0 + 3.0 * triangle)
    if not np.allclose(normalized, affine, rtol=0.0, atol=1e-12):
        raise ContractError("positive-affine invariance guard failed")
    if any(len(directional_change_events(np.ones(73), value)[0]) for value in THRESHOLDS):
        raise ContractError("constant-path guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack(
        [np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)]
    )
    sequence[:, 5] = 6.0 + np.sin(3.0 * axis)
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = directional_change_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, directional_change_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "trend_event_counts": trend_counts,
        "triangle_event_counts": triangle_counts,
        "threshold_event_count_monotone": True,
        "alternating_events": True,
        "direction_reversal": True,
        "overshoot_duration_reversal_invariant": True,
        "positive_affine_invariant": True,
        "constant_path_zero_events": True,
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
        features[position] = directional_change_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v77.v76.v75.v74.v73.v72.v71.v70.v69.v68.v67.v66.v65.v64.v63.v62.case_surface()
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
        raise ContractError("v78 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.directional_change_intrinsic_time_residual.preflight.v78",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE",
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
            "schema_version": "p3.directional_change_intrinsic_time_residual.result.v78",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE"
            if any(item["decision"] != "NO_GO" for item in result["candidates"])
            else "NO_GO_ALL_DIRECTIONAL_CHANGE_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 directional-change intrinsic-time residual cycle v78",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v78 encodes alternating reversal confirmations, overshoots and durations in past-only intrinsic event time. It uses no prior prediction or official feedback.",
        "- Finance sources motivate the event operator only; they are not ocean performance evidence. The repeatedly exposed 182-case surface is EXPLORATORY_ONLY.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(
            f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; nominal score {points['nominal_official_score']:.6f}; planning {points['raw_gain']:+.6f}; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; lead {item['worst_lead_delta_m']:+.9f}m; station-lead {item['worst_station_lead_delta_m']:+.9f}m; tail {item['worst_reference_tail_block_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}."
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
        raise ContractError("v78 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v78 support gate failed; zero-fit closure required")
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
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n"
        b"| Fixed-amplitude reversals define directional-change intrinsic events | Guillaume et al. 1997, DOI:10.1007/s007800050018 | event-operator motivation only |\n"
        b"| Overshoots extend a trend between directional-change confirmations | Petrov et al. 2020, DOI:10.2139/ssrn.3240456 | overshoot definition motivation only |\n"
        b"| No executed P3 directional-change contract exists | repository semantic audit | novelty boundary |\n"
        b"| Prior/official outputs were excluded | sealed v78 contract | reuse boundary |\n",
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
