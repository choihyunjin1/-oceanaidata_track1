"""Sealed P3 v77 natural-time increment-energy representation."""

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

import run_p3_moist_air_momentum_residual_cycle_20260901_v76 as v76  # noqa: E402

EXPERIMENT_ID = "p3_natural_time_increment_energy_residual_cycle_20260901_v77"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS = ((0, 145), (72, 145))
FEATURE_COUNT = 32
BASE = v76.BASE
SPEC_CLASS = v76.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_NATTIME32_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_NATTIME32_RIDGE2048_ADD10", 2048.0),
)
BLEND, EPSILON = 0.10, 1e-12
sha256, canonical, write_new = v76.sha256, v76.canonical, v76.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v77 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"]
        == "p3.natural_time_increment_energy_residual.config.v77",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_P3_NATURAL_TIME_INCREMENT_ENERGY_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
        "stats": tuple(encoder["statistics"])
        == ("kappa1", "natural_time_entropy_S", "reversed_entropy_S_rev", "S_minus_S_rev"),
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
        raise ContractError(f"v77 config contract failed: {checks}")
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


def natural_time_statistics(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < 24 or not np.isfinite(path).all():
        raise ContractError("natural-time path support differs")
    energy = np.square(np.diff(path))
    event_count = len(energy)
    total = float(np.sum(energy))
    weights = (
        np.full(event_count, 1.0 / event_count, dtype=np.float64)
        if total <= EPSILON
        else energy / total
    )
    natural_time = np.arange(1, event_count + 1, dtype=np.float64) / event_count

    def entropy(probability: np.ndarray) -> float:
        mean_time = float(np.dot(probability, natural_time))
        return float(
            np.dot(probability, natural_time * np.log(natural_time)) - mean_time * np.log(mean_time)
        )

    mean_time = float(np.dot(weights, natural_time))
    kappa1 = float(np.dot(weights, np.square(natural_time)) - mean_time**2)
    entropy_forward = entropy(weights)
    entropy_reverse = entropy(weights[::-1])
    features = np.asarray(
        [kappa1, entropy_forward, entropy_reverse, entropy_forward - entropy_reverse],
        dtype=np.float64,
    )
    if features.shape != (4,) or not np.isfinite(features).all():
        raise ContractError("natural-time statistics differ")
    return features


def natural_time_features(sequence: np.ndarray) -> np.ndarray:
    raw = fill_prefix(np.asarray(sequence)[:289])[::2]
    features = np.concatenate(
        [
            natural_time_statistics(raw[start:stop, channel])
            for channel in CHANNELS
            for start, stop in WINDOWS
        ]
    )
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("natural-time feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    axis = np.linspace(0.0, 1.0, 289)
    sequence = np.column_stack(
        [np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)]
    )
    sequence[:, 5] = 6.0 + np.sin(3.0 * axis)
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = natural_time_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, natural_time_features(extended)):
        raise ContractError("future isolation guard failed")

    path = np.zeros(73, dtype=np.float64)
    path[12:] += 1.0
    path[50:] += 2.0
    forward = natural_time_statistics(path)
    reverse = natural_time_statistics(path[::-1])
    if not np.isclose(forward[0], reverse[0], rtol=0.0, atol=1e-12):
        raise ContractError("time-reversal kappa guard failed")
    if not np.isclose(forward[1], reverse[2], rtol=0.0, atol=1e-12):
        raise ContractError("time-reversal entropy guard failed")
    if not np.isclose(forward[3], -reverse[3], rtol=0.0, atol=1e-12):
        raise ContractError("time-reversal asymmetry guard failed")
    affine = natural_time_statistics(7.0 + 3.0 * path)
    if not np.allclose(forward, affine, rtol=0.0, atol=1e-12):
        raise ContractError("positive-affine invariance guard failed")
    constant = natural_time_statistics(np.ones(73, dtype=np.float64))
    if not np.isfinite(constant).all() or not 0.0 <= constant[0] <= 1.0 / 12.0:
        raise ContractError("constant-path guard failed")
    pulse_a = np.zeros(73, dtype=np.float64)
    pulse_b = np.zeros(73, dtype=np.float64)
    pulse_a[[12, 13, 50, 51]] = (1.0, 0.0, 3.0, 0.0)
    pulse_b[[20, 21, 44, 45]] = (1.0, 0.0, 3.0, 0.0)
    pulse_order_distinct = not np.allclose(
        natural_time_statistics(pulse_a),
        natural_time_statistics(pulse_b),
        rtol=0.0,
        atol=1e-12,
    )
    if not pulse_order_distinct:
        raise ContractError("pulse-order guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "future_isolated": True,
        "time_reversal_kappa_invariant": True,
        "time_reversal_entropy_swapped": True,
        "time_reversal_asymmetry_sign_flipped": True,
        "positive_affine_invariant": True,
        "constant_path_bounded": True,
        "pulse_order_distinct": True,
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
        features[position] = natural_time_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v76.v75.v74.v73.v72.v71.v70.v69.v68.v67.v66.v65.v64.v63.v62.case_surface()
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
        raise ContractError("v77 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.natural_time_increment_energy_residual.preflight.v77",
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
            "schema_version": "p3.natural_time_increment_energy_residual.result.v77",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE"
            if any(item["decision"] != "NO_GO" for item in result["candidates"])
            else "NO_GO_ALL_NATURAL_TIME_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 natural-time increment-energy residual cycle v77",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v77 measures the ordering of past increment energy in natural time; it uses no prior prediction, official feedback, absolute time, future covariate, spectrum, pole, codifference, or causality score.",
        "- Primary sources motivate the representation only. The repeatedly exposed 182-case surface is EXPLORATORY_ONLY.",
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
        raise ContractError("v77 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v77 support gate failed; zero-fit closure required")
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
        b"| Natural-time entropy weights normalized event order by event energy | Varotsos et al. 2004, DOI:10.1103/PhysRevE.70.011106 | representation motivation only |\n"
        b"| Reversing event-energy order defines a distinct natural-time entropy | Varotsos et al. 2006, DOI:10.1103/PhysRevE.73.031114 | representation motivation only |\n"
        b"| No executed P3 natural-time contract exists | repository semantic audit | novelty boundary |\n"
        b"| Prior/official outputs were excluded | sealed v77 contract | reuse boundary |\n",
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
