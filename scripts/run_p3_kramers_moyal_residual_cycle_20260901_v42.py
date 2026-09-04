"""Sealed P3 v42 Kramers-Moyal conditional-moment residual experiment."""

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

import run_p3_fixed_echo_state_residual_cycle_20260901_v41 as v41  # noqa: E402

EXPERIMENT_ID = "p3_kramers_moyal_residual_cycle_20260901_v42"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
BIN_EDGES = np.asarray([-np.inf, -1.0, -0.25, 0.25, 1.0, np.inf])
FEATURE_COUNT = 80
SPECS = (
    v41.v40.v39.v38.v36.v26.Spec("P3_1_KM80_RIDGE512_ADD10", 512.0),
    v41.v40.v39.v38.v36.v26.Spec("P3_2_KM80_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
sha256, canonical, write_new = v41.sha256, v41.canonical, v41.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v42 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.kramers_moyal_residual.config.v42",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_STATE_CONDITIONAL_INCREMENT_MOMENT_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "bins": encoder["state_bin_edges_iqr_units"] == ["-inf", -1.0, -0.25, 0.25, 1.0, "+inf"],
        "lag": encoder["increment_lag_rows"] == 1,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "support": encoder["real_support_gate"] == {"nonzero_feature_share_gte": 0.5, "positive_variance_features_gte": 60},
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"])
        == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v42 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def conditional_moments(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    state, increment = path[:-1], np.diff(path)
    bins = np.digitize(state, BIN_EDGES[1:-1], right=False)
    output: list[float] = []
    for index in range(5):
        selected = increment[bins == index]
        occupancy = len(selected) / len(increment)
        if len(selected):
            output.extend([occupancy, np.mean(selected), np.mean(selected**2), np.mean(selected**3)])
        else:
            output.extend([0.0, 0.0, 0.0, 0.0])
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (20,) or not np.isfinite(features).all():
        raise ContractError("conditional moment contract differs")
    return features


def kramers_moyal_features(sequence: np.ndarray) -> np.ndarray:
    path = v41.v40.v39.v38.v36.v26.transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        values = path[:, channel]
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = max(float(q75 - q25), 1e-12)
        output.extend(conditional_moments((values - median) / scale))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("Kramers-Moyal feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    path = np.zeros(4000, dtype=np.float64)
    for index in range(1, len(path)):
        path[index] = 0.80 * path[index - 1] + rng.normal(scale=0.35)
    moments = conditional_moments(path)
    low_drift = float(moments[1])
    high_drift = float(moments[-3])
    if not (low_drift > 0 and high_drift < 0):
        raise ContractError("synthetic mean-reverting drift guard failed")
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = kramers_moyal_features(sequence)
    return {"feature_count": len(feature), "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(feature).all()), "mean_reverting_low_bin_drift": low_drift, "mean_reverting_high_bin_drift": high_drift}


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(v41.v40.v39.SEQUENCES, mmap_mode="r")
    station_codes = np.load(v41.v40.v39.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = kramers_moyal_features(sequences[anchor_id])
    return features, {"rows": len(features), "columns": features.shape[1], "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(), "finite": bool(np.isfinite(features).all())}


def real_support_receipt() -> dict[str, Any]:
    cases, _, _, _ = v41.v40.v39.v38.v36.v32.v23.case_surface()
    features, receipt = surface_features(cases)
    nonzero_share = float(np.count_nonzero(np.abs(features) > 1e-15) / features.size)
    positive_variance = int(np.count_nonzero(np.var(features, axis=0) > 1e-15))
    passed = nonzero_share >= 0.5 and positive_variance >= 60
    return {**receipt, "nonzero_feature_share": nonzero_share, "positive_variance_features": positive_variance, "gate_pass": passed, "target_used_for_feature_gate": False}


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v42 exactly-once namespace is consumed")
    support = real_support_receipt()
    payload = {"schema_version": "p3.kramers_moyal_residual.preflight.v42", "experiment_id": EXPERIMENT_ID, "status": "READY_EXACTLY_ONCE" if support["gate_pass"] else "STOP_SUPPORT_GATE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "candidate_count": 2, "maximum_model_fits": 12 if support["gate_pass"] else 0, "synthetic": synthetic_receipt(), "real_support": support, "official_access": 0, "csv_materializations": 0, "uploads": 0, "config_status": config["status"]}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = v41.v40.v39.surface_features, v41.v40.v39.SPECS
    v41.v40.v39.surface_features, v41.v40.v39.SPECS = surface_features, SPECS
    try:
        result, arrays = v41.v40.v39.execute(config)
    finally:
        v41.v40.v39.surface_features, v41.v40.v39.SPECS = original_surface, original_specs
    result.update({"schema_version": "p3.kramers_moyal_residual.result.v42", "experiment_id": EXPERIMENT_ID, "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_KRAMERS_MOYAL_CANDIDATES", "duplication_audit": config["duplication_audit"], "primary_sources": config["primary_sources"]})
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = ["# P3 Kramers-Moyal residual cycle v42", "", "## 결론", "", f"- overall decision: **{result['decision']}**.", "- v42 estimates fixed-bin state-conditional increment moments; it contains no transfer entropy, threshold-crossing count, recurrent state, or prior-cycle prediction.", "- Siegert et al. (1998) motivates the mechanism only; the 182-case surface remains EXPLORATORY_ONLY."]
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
        raise ContractError("v42 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v42 real support gate failed; zero-fit closure required")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Conditional moments estimate state-dependent stochastic drift and diffusion | Siegert, Friedrich, and Peinke, Physics Letters A 243, 1998, DOI:10.1016/S0375-9601(98)00283-7 | mechanism only |\n| P3 support is independently checked and no P1 result is reused | sealed v42 preflight | novelty and support boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
