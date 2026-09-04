"""Sealed P3 v70 Burg reflection-coefficient memory experiment."""

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

import run_p3_sax_word_histogram_residual_cycle_20260901_v69 as v69  # noqa: E402

EXPERIMENT_ID = "p3_burg_reflection_residual_cycle_20260901_v70"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS, ORDER = ((0, 145), (72, 145)), 12
FEATURE_COUNT = 192
BASE = v69.BASE
SPEC_CLASS = v69.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_BURG192_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_BURG192_RIDGE2048_ADD10", 2048.0),
)
BLEND, MAD_SCALE, EPSILON = 0.10, 1.4826, 1e-12
sha256, canonical, write_new = v69.sha256, v69.canonical, v69.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v70 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.burg_reflection_residual.config.v70",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_BURG_REFLECTION_MEMORY_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "order": int(encoder["order"]) == ORDER,
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_excluded": "excluded" in config["duplication_audit"]["official_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v70 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def robust_normalize(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if len(path) <= ORDER + 1 or not np.isfinite(path).all():
        raise ContractError("Burg path support differs")
    center = float(np.median(path))
    scale = MAD_SCALE * float(np.median(np.abs(path - center)))
    if scale <= EPSILON:
        return np.zeros_like(path)
    return (path - center) / scale


def burg_memory(values: np.ndarray) -> np.ndarray:
    path = robust_normalize(values)
    forward = path[1:].copy()
    backward = path[:-1].copy()
    base_variance = max(float(np.mean(path * path)), EPSILON)
    innovation = base_variance
    reflections: list[float] = []
    log_ratios: list[float] = []
    for level in range(ORDER):
        denominator = float(np.dot(forward, forward) + np.dot(backward, backward))
        reflection = 0.0 if denominator <= EPSILON else -2.0 * float(np.dot(backward, forward)) / denominator
        reflection = float(np.clip(reflection, -1.0 + 1e-12, 1.0 - 1e-12))
        reflections.append(reflection)
        innovation *= max(1.0 - reflection * reflection, EPSILON)
        log_ratios.append(float(np.log(max(innovation / base_variance, EPSILON))))
        if level < ORDER - 1:
            old_forward, old_backward = forward, backward
            forward = old_forward[1:] + reflection * old_backward[1:]
            backward = old_backward[:-1] + reflection * old_forward[:-1]
    features = np.asarray(reflections + log_ratios, dtype=np.float64)
    if features.shape != (2 * ORDER,) or not np.isfinite(features).all():
        raise ContractError("Burg feature contract differs")
    if np.any(np.abs(features[:ORDER]) >= 1.0) or np.any(np.diff(features[ORDER:]) > 1e-12):
        raise ContractError("Burg stability contract differs")
    return features


def transformed_path(sequence: np.ndarray) -> np.ndarray:
    return v69.transformed_path(np.asarray(sequence)[:289])


def burg_features(sequence: np.ndarray) -> np.ndarray:
    path = transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[np.ndarray] = []
    for channel in CHANNELS:
        for start, stop in WINDOWS:
            output.append(burg_memory(path[start:stop, channel]))
    features = np.concatenate(output)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("Burg surface feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    noise = rng.normal(size=512)
    ar1 = np.zeros(512, dtype=np.float64)
    for index in range(1, len(ar1)):
        ar1[index] = 0.82 * ar1[index - 1] + noise[index]
    white_feature, ar_feature = burg_memory(noise), burg_memory(ar1)
    if not abs(ar_feature[0]) > abs(white_feature[0]) + 0.50:
        raise ContractError("white-versus-AR1 reflection recovery failed")
    if not np.array_equal(ar_feature, burg_memory(-ar1)):
        raise ContractError("sign invariance guard failed")
    if not np.allclose(ar_feature, burg_memory(7.0 * ar1 + 3.0), rtol=1e-12, atol=1e-12):
        raise ContractError("positive affine invariance guard failed")
    constant = burg_memory(np.full(145, 4.0))
    if not np.array_equal(constant, np.zeros(2 * ORDER)):
        raise ContractError("constant path bound guard failed")
    if np.any(np.abs(ar_feature[:ORDER]) >= 1.0) or np.any(np.diff(ar_feature[ORDER:]) > 1e-12):
        raise ContractError("reflection/innovation stability guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * axis) + 0.1 * index * axis for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = burg_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, burg_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "white_first_abs_reflection": float(abs(white_feature[0])),
        "ar1_first_abs_reflection": float(abs(ar_feature[0])),
        "reflection_strictly_bounded": True,
        "innovation_monotone": True,
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
        features[position] = burg_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v69.v68.v67.v66.v65.v64.v63.v62.case_surface()
    features, metadata = surface_features(cases)
    reflection_columns = np.concatenate([np.arange(base, base + ORDER) for base in range(0, FEATURE_COUNT, 2 * ORDER)])
    maximum_abs_reflection = float(np.max(np.abs(features[:, reflection_columns])))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(
        len(features) >= int(gate["minimum_cases"])
        and positive_variance >= int(gate["minimum_positive_variance_features"])
        and maximum_abs_reflection < float(gate["maximum_abs_reflection_strictly_below"])
    )
    return {
        **metadata,
        "positive_variance_features": positive_variance,
        "maximum_abs_reflection": maximum_abs_reflection,
        "target_used": False,
        "passed": passed,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v70 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.burg_reflection_residual.preflight.v70",
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
            "schema_version": "p3.burg_reflection_residual.result.v70",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_BURG_REFLECTION_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 Burg reflection-memory residual cycle v70",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v70 uses fixed Burg reflection coefficients and innovation decay, not a learned AR forecaster or prior candidate output.",
        "- Ordinary spectrum, volatility ACF, NLinear and state-space boundaries are recorded; the surface is EXPLORATORY_ONLY.",
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
        raise ContractError("v70 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v70 support gate failed; zero-fit closure required")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Reflection coefficients encode sequential AR prediction-error structure | Ulrych and Bishop 1975, DOI:10.1029/RG013i001p00183 | mechanism only |\n| No dedicated executed P3 Burg/reflection residual axis exists | repository semantic audit | novelty boundary |\n| Prior/official outputs were excluded | sealed v70 contract | reuse boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
