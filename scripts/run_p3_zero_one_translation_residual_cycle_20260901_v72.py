"""Sealed P3 v72 fixed zero-one translation-diffusion experiment."""

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

import run_p3_equilibrium_error_residual_cycle_20260901_v71 as v71  # noqa: E402

EXPERIMENT_ID = "p3_zero_one_translation_residual_cycle_20260901_v72"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS = ((0, 145), (72, 145))
C_VALUES = np.asarray([0.83, 1.01, 1.25, 1.57, 1.75, 1.99, 2.31, 2.49], dtype=np.float64)
FEATURE_COUNT = 24
BASE = v71.BASE
SPEC_CLASS = v71.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_ZEROONE24_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_ZEROONE24_RIDGE2048_ADD10", 2048.0),
)
BLEND, MAD_SCALE, EPSILON = 0.10, 1.4826, 1e-12
sha256, canonical, write_new = v71.sha256, v71.canonical, v71.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v72 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.zero_one_translation_residual.config.v72",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_ZERO_ONE_TRANSLATION_DIFFUSION_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "c": np.array_equal(np.asarray(encoder["c_values"], dtype=np.float64), C_VALUES),
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_excluded": "excluded" in config["duplication_audit"]["official_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v72 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def robust_normalize(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < 16 or not np.isfinite(path).all():
        raise ContractError("zero-one path support differs")
    center = float(np.median(path))
    scale = MAD_SCALE * float(np.median(np.abs(path - center)))
    if scale <= EPSILON:
        return np.zeros_like(path)
    return (path - center) / scale


def zero_one_k(values: np.ndarray, c_value: float) -> float:
    path = robust_normalize(values)
    if not np.any(path):
        return 0.0
    indices = np.arange(1, len(path) + 1, dtype=np.float64)
    p_path = np.cumsum(path * np.cos(indices * c_value))
    q_path = np.cumsum(path * np.sin(indices * c_value))
    max_lag = len(path) // 10
    lags = np.arange(1, max_lag + 1, dtype=np.float64)
    displacement = np.asarray([
        np.mean((p_path[lag:] - p_path[:-lag]) ** 2 + (q_path[lag:] - q_path[:-lag]) ** 2)
        for lag in range(1, max_lag + 1)
    ])
    if float(np.std(displacement)) <= EPSILON:
        return 0.0
    value = float(np.corrcoef(lags, displacement)[0, 1])
    if not np.isfinite(value):
        raise ContractError("zero-one correlation is nonfinite")
    return float(np.clip(value, -1.0, 1.0))


def translation_summary(values: np.ndarray) -> np.ndarray:
    k_values = np.asarray([zero_one_k(values, c_value) for c_value in C_VALUES])
    features = np.asarray([
        np.median(k_values),
        np.quantile(k_values, 0.75) - np.quantile(k_values, 0.25),
        np.mean(k_values > 0.5),
    ], dtype=np.float64)
    if features.shape != (3,) or not np.isfinite(features).all():
        raise ContractError("zero-one summary differs")
    return features


def transformed_path(sequence: np.ndarray) -> np.ndarray:
    return v71.transformed_path(np.asarray(sequence)[:289])


def zero_one_features(sequence: np.ndarray) -> np.ndarray:
    path = transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output = [translation_summary(path[start:stop, channel]) for channel in CHANNELS for start, stop in WINDOWS]
    features = np.concatenate(output)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("zero-one surface feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    length = 145
    index = np.arange(length, dtype=np.float64)
    periodic = np.sin(2.0 * np.pi * index / 17.0)
    state = 0.123456
    logistic_values: list[float] = []
    for step in range(length + 500):
        state = 4.0 * state * (1.0 - state)
        if step >= 500:
            logistic_values.append(state)
    logistic = np.asarray(logistic_values)
    periodic_summary = translation_summary(periodic)
    logistic_summary = translation_summary(logistic)
    if not periodic_summary[0] < 0.20:
        raise ContractError("periodic bounded-translation guard failed")
    if not logistic_summary[0] > 0.80:
        raise ContractError("logistic diffusive-translation guard failed")
    if not logistic_summary[0] - periodic_summary[0] > 0.60:
        raise ContractError("periodic-chaotic separation guard failed")
    omega = 2.0 * np.pi / 17.0
    harmonics = np.arange(1, 7, dtype=np.float64) * omega
    minimum_harmonic_distance = float(np.min(np.abs(C_VALUES[:, None] - harmonics[None, :])))
    if not minimum_harmonic_distance > 0.06 or not float(np.min(np.diff(C_VALUES))) > 0.15:
        raise ContractError("fixed nonresonance guard failed")
    affine = translation_summary(5.0 * logistic + 2.0)
    signed = translation_summary(-logistic)
    if not np.allclose(logistic_summary, affine, rtol=1e-12, atol=1e-12):
        raise ContractError("positive affine invariance guard failed")
    if not np.allclose(logistic_summary, signed, rtol=1e-12, atol=1e-12):
        raise ContractError("sign invariance guard failed")
    if not np.array_equal(translation_summary(np.ones(80)), np.zeros(3)):
        raise ContractError("constant path bound guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = zero_one_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, zero_one_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "periodic_median_k": float(periodic_summary[0]),
        "logistic_median_k": float(logistic_summary[0]),
        "separation": float(logistic_summary[0] - periodic_summary[0]),
        "minimum_period17_harmonic_distance_rad": minimum_harmonic_distance,
        "minimum_c_spacing_rad": float(np.min(np.diff(C_VALUES))),
        "positive_affine_invariant": True,
        "sign_invariant": True,
        "constant_zero": True,
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
        features[position] = zero_one_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v71.v70.v69.v68.v67.v66.v65.v64.v63.v62.case_surface()
    features, metadata = surface_features(cases)
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(len(features) >= int(gate["minimum_cases"]) and positive_variance >= int(gate["minimum_positive_variance_features"]))
    return {**metadata, "positive_variance_features": positive_variance, "target_used": False, "passed": passed}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v72 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.zero_one_translation_residual.preflight.v72",
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
    result.update({
        "schema_version": "p3.zero_one_translation_residual.result.v72",
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_ZERO_ONE_CANDIDATES",
        "duplication_audit": config["duplication_audit"],
        "primary_sources": config["primary_sources"],
    })
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 fixed zero-one translation-diffusion residual cycle v72",
        "", "## 결론", "", f"- overall decision: **{result['decision']}**.",
        "- P1 v40 adjacency is disclosed; P3 v72 is a distinct regression/data/action cycle and uses no prior outputs.",
        "- Fixed nonresonant translation diffusion differs from local divergence, recurrence geometry, and learned reservoir states; the 182-case surface is EXPLORATORY_ONLY.",
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
        raise ContractError("v72 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v72 support gate failed; zero-fit closure required")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Translation variables motivate a bounded-versus-diffusive chaos diagnostic | Gottwald and Melbourne 2004, DOI:10.1098/rspa.2003.1183 | mechanism only |\n| P1 v40 adjacency and no P3 executed equivalent | repository semantic audit | novelty boundary |\n| Prior/official outputs were excluded | sealed v72 contract | reuse boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
