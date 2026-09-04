"""Sealed P3 v73 fixed pairwise predictive-causality experiment."""

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

import run_p3_zero_one_translation_residual_cycle_20260901_v72 as v72  # noqa: E402

EXPERIMENT_ID = "p3_pairwise_predictive_causality_residual_cycle_20260901_v73"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
PAIRS = tuple((source, target) for source in CHANNELS for target in CHANNELS if source != target)
PAIR_NAMES = tuple(f"{CHANNEL_NAMES[CHANNELS.index(source)]}->{CHANNEL_NAMES[CHANNELS.index(target)]}" for source, target in PAIRS)
WINDOWS = ((0, 145), (72, 145))
LAG_ORDER, FEATURE_COUNT = 2, 24
BASE = v72.BASE
SPEC_CLASS = v72.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_GRANGER24_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_GRANGER24_RIDGE2048_ADD10", 2048.0),
)
BLEND, MAD_SCALE, EPSILON = 0.10, 1.4826, 1e-12
sha256, canonical, write_new = v72.sha256, v72.canonical, v72.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v73 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.pairwise_predictive_causality_residual.config.v73",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_PAIRWISE_LINEAR_PREDICTIVE_CAUSALITY_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "pairs": tuple(encoder["ordered_pairs"]) == PAIR_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "lag": int(encoder["lag_order"]) == LAG_ORDER,
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_excluded": "excluded" in config["duplication_audit"]["official_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v73 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def robust_normalize(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < 16 or not np.isfinite(path).all():
        raise ContractError("predictive-causality path support differs")
    center = float(np.median(path))
    scale = MAD_SCALE * float(np.median(np.abs(path - center)))
    if scale <= EPSILON:
        return np.zeros_like(path)
    return (path - center) / scale


def predictive_variance_ratio(source: np.ndarray, target: np.ndarray) -> float:
    x, y = robust_normalize(source), robust_normalize(target)
    if float(np.var(y)) <= EPSILON:
        return 0.0
    response = y[LAG_ORDER:]
    target_lags = np.column_stack([y[LAG_ORDER - lag : -lag] for lag in range(1, LAG_ORDER + 1)])
    source_lags = np.column_stack([x[LAG_ORDER - lag : -lag] for lag in range(1, LAG_ORDER + 1)])
    unrestricted = np.column_stack([target_lags, source_lags])
    restricted_coef = np.linalg.lstsq(target_lags, response, rcond=None)[0]
    unrestricted_coef = np.linalg.lstsq(unrestricted, response, rcond=None)[0]
    restricted_variance = float(np.mean((response - target_lags @ restricted_coef) ** 2))
    unrestricted_variance = float(np.mean((response - unrestricted @ unrestricted_coef) ** 2))
    value = max(0.0, float(np.log((restricted_variance + EPSILON) / (unrestricted_variance + EPSILON))))
    if not np.isfinite(value):
        raise ContractError("predictive variance ratio is nonfinite")
    return value


def transformed_path(sequence: np.ndarray) -> np.ndarray:
    return v72.transformed_path(np.asarray(sequence)[:289])


def predictive_causality_features(sequence: np.ndarray) -> np.ndarray:
    path = transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    features = np.asarray([
        predictive_variance_ratio(path[start:stop, source], path[start:stop, target])
        for source, target in PAIRS for start, stop in WINDOWS
    ], dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("predictive-causality surface feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    length = 512
    source = np.zeros(length, dtype=np.float64)
    target = np.zeros(length, dtype=np.float64)
    for index in range(2, length):
        source[index] = 0.65 * source[index - 1] + rng.normal(scale=0.60)
        target[index] = 0.55 * target[index - 1] + 0.75 * source[index - 1] + rng.normal(scale=0.35)
    forward = predictive_variance_ratio(source, target)
    reverse = predictive_variance_ratio(target, source)
    if not forward > 0.50 or not forward > reverse + 0.50:
        raise ContractError("directed coupling recovery guard failed")
    affine = predictive_variance_ratio(5.0 * source + 2.0, 7.0 * target - 3.0)
    signed = predictive_variance_ratio(-source, -target)
    if not np.isclose(forward, affine, rtol=1e-12, atol=1e-12):
        raise ContractError("positive affine invariance guard failed")
    if not np.isclose(forward, signed, rtol=1e-12, atol=1e-12):
        raise ContractError("sign invariance guard failed")
    if predictive_variance_ratio(source, np.ones(length)) != 0.0:
        raise ContractError("constant-target bound guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = predictive_causality_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, predictive_causality_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "forward_variance_ratio": forward,
        "reverse_variance_ratio": reverse,
        "directed_margin": forward - reverse,
        "positive_affine_invariant": True,
        "sign_invariant": True,
        "constant_target_zero": True,
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
        features[position] = predictive_causality_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v72.v71.v70.v69.v68.v67.v66.v65.v64.v63.v62.case_surface()
    features, metadata = surface_features(cases)
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(len(features) >= int(gate["minimum_cases"]) and positive_variance >= int(gate["minimum_positive_variance_features"]))
    return {**metadata, "positive_variance_features": positive_variance, "target_used": False, "passed": passed}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v73 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.pairwise_predictive_causality_residual.preflight.v73",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE",
        "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2, "maximum_model_fits": 12 if support["passed"] else 0,
        "synthetic": synthetic_receipt(), "historical_support": support,
        "prior_outputs_used": False, "official_used_for_features_gates_selection": False,
        "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"],
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
        "schema_version": "p3.pairwise_predictive_causality_residual.result.v73",
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_PAIRWISE_PREDICTIVE_CAUSALITY_CANDIDATES",
        "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"],
    })
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 pairwise predictive-causality residual cycle v73", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- Fixed lag-2 incremental linear predictability is separate from discretized transfer information, static equilibrium error, and univariate Burg memory.", "- Prior and official outputs were excluded; the repeatedly exposed 182-case surface is EXPLORATORY_ONLY."]
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
        raise ContractError("v73 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v73 support gate failed; zero-fit closure required")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Past-source incremental predictability motivates a testable directional relation | Granger 1969, DOI:10.2307/1912791 | mechanism only; no structural-causality or performance claim |\n| No executed P3 restricted/unrestricted pairwise VAR residual-ratio axis exists | repository semantic audit | novelty boundary |\n| Prior/official outputs were excluded | sealed v73 contract | reuse boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
