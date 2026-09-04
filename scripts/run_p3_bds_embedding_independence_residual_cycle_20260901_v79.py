"""Sealed P3 v79 BDS-motivated embedding-independence representation."""

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

import run_p3_directional_change_intrinsic_time_residual_cycle_20260901_v78 as v78  # noqa: E402

EXPERIMENT_ID = "p3_bds_embedding_independence_residual_cycle_20260901_v79"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS = ((0, 145), (72, 145))
EMBEDDING_DIMENSION, EMBEDDING_DELAY, RADIUS = 3, 1, 0.5
FEATURE_COUNT = 16
BASE = v78.BASE
SPEC_CLASS = v78.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_BDS16_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_BDS16_RIDGE2048_ADD10", 2048.0),
)
BLEND, EPSILON = 0.10, 1e-12
sha256, canonical, write_new = v78.sha256, v78.canonical, v78.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v79 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    audit = config["duplication_audit"]
    checks = {
        "schema": config["schema_version"]
        == "p3.bds_embedding_independence_residual.config.v79",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": audit["semantic_verdict"]
        == "NON_DUPLICATE_EMBEDDING_INDEPENDENCE_CONTRAST_WITH_RQA_ADJACENCY",
        "not_reconstructible": audit["strict_reconstructibility_verdict"]
        == "PROCEED_NOT_RECONSTRUCTIBLE_FROM_PRIOR_OUTPUTS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "dimension": int(encoder["embedding_dimension"]) == EMBEDDING_DIMENSION,
        "delay": int(encoder["embedding_delay"]) == EMBEDDING_DELAY,
        "radius": float(encoder["epsilon_mad_units"]) == RADIUS,
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
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
        "no_posthoc": not audit["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v79 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def correlation_integral(values: np.ndarray, dimension: int) -> float:
    path = v78.robust_normalize(np.asarray(values, dtype=np.float64))
    if dimension < 1 or dimension > EMBEDDING_DIMENSION:
        raise ContractError("embedding dimension differs")
    rows = len(path) - (dimension - 1) * EMBEDDING_DELAY
    if rows < 24:
        raise ContractError("embedding support differs")
    embedded = np.column_stack(
        [path[offset : offset + rows] for offset in range(0, dimension * EMBEDDING_DELAY, EMBEDDING_DELAY)]
    )
    distances = np.max(np.abs(embedded[:, None, :] - embedded[None, :, :]), axis=2)
    recurrent = int(np.count_nonzero(distances <= RADIUS) - rows)
    return recurrent / float(rows * (rows - 1))


def embedding_independence_statistics(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    rows = len(path) - (EMBEDDING_DIMENSION - 1) * EMBEDDING_DELAY
    # Use the same scalar support as the embedded vectors so the factorization
    # contrast cannot be driven by an endpoint-count difference.
    scalar = path[:rows]
    c1 = correlation_integral(scalar, 1)
    c3 = correlation_integral(path, EMBEDDING_DIMENSION)
    factorized = c1**EMBEDDING_DIMENSION
    contrast = c3 - factorized
    log_ratio = float(np.log((c3 + EPSILON) / (factorized + EPSILON)))
    result = np.asarray([contrast, log_ratio], dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ContractError("embedding-independence statistics differ")
    return result


def bds_features(sequence: np.ndarray) -> np.ndarray:
    raw = v78.fill_prefix(np.asarray(sequence)[:289])[::2]
    features = np.concatenate(
        [
            embedding_independence_statistics(raw[start:stop, channel])
            for channel in CHANNELS
            for start, stop in WINDOWS
        ]
    )
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("BDS feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(1179)
    innovations = rng.normal(size=145)
    dependent = np.zeros(145, dtype=np.float64)
    for index in range(1, len(dependent)):
        dependent[index] = 0.85 * dependent[index - 1] + innovations[index]
    shuffled = dependent[rng.permutation(len(dependent))]
    dep = embedding_independence_statistics(dependent)
    shuffled_value = embedding_independence_statistics(shuffled)
    if not dep[0] > shuffled_value[0] + 0.01 or not dep[1] > shuffled_value[1] + 0.25:
        raise ContractError("serial-dependence contrast guard failed")
    affine = embedding_independence_statistics(7.0 + 3.0 * dependent)
    if not np.allclose(dep, affine, rtol=0.0, atol=1e-12):
        raise ContractError("positive-affine invariance guard failed")
    constant = embedding_independence_statistics(np.ones(145))
    if not np.array_equal(constant, np.zeros(2)):
        raise ContractError("constant factorization guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack(
        [np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)]
    )
    sequence[:, 5] = 6.0 + np.sin(3.0 * axis)
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = bds_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, bds_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "dependent_contrast": dep.tolist(),
        "shuffled_contrast": shuffled_value.tolist(),
        "serial_dependence_recovered": True,
        "positive_affine_invariant": True,
        "constant_factorization_zero": True,
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
        features[position] = bds_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def case_surface() -> tuple[Any, ...]:
    return v78.v77.v76.v75.v74.v73.v72.v71.v70.v69.v68.v67.v66.v65.v64.v63.v62.case_surface()


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = case_surface()
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
        raise ContractError("v79 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.bds_embedding_independence_residual.preflight.v79",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE_ZERO_FIT",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2,
        "maximum_model_fits": 12 if support["passed"] else 0,
        "synthetic": synthetic_receipt(),
        "historical_support": support,
        "strict_reconstructibility_verdict": config["duplication_audit"]["strict_reconstructibility_verdict"],
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
            "schema_version": "p3.bds_embedding_independence_residual.result.v79",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE"
            if any(item["decision"] != "NO_GO" for item in result["candidates"])
            else "NO_GO_ALL_BDS_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 BDS embedding-independence residual cycle v79",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- Strict audit: v29 stored C1 and RQA line/topology summaries but no delay-embedded C3; therefore C3-C1^3 cannot be reconstructed from prior outputs. v79 remains recurrence-adjacent and that boundary is explicit.",
        "- Broock et al. motivates the independence-factorization operator only; it is not ocean-performance evidence. The repeatedly exposed 182-case surface is EXPLORATORY_ONLY.",
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
        raise ContractError("v79 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v79 support gate failed; zero-fit closure required")
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
        b"| Correlation-integral factorization supplies an IID specification diagnostic | Broock et al. 1996, DOI:10.1080/07474939608800353 | operator motivation only |\n"
        b"| v29 cannot reconstruct C3-C1^3 because it stores no delay-embedded C3 | repository semantic and code audit | novelty boundary |\n"
        b"| Prior/official outputs were excluded | sealed v79 contract | reuse boundary |\n",
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
