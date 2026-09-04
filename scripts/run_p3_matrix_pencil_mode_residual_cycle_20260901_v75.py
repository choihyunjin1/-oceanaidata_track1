"""Sealed P3 v75 fixed matrix-pencil damped-mode experiment."""

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

import run_p3_lagged_codifference_residual_cycle_20260901_v74 as v74  # noqa: E402

EXPERIMENT_ID = "p3_matrix_pencil_mode_residual_cycle_20260901_v75"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS = ((0, 145), (72, 145))
PENCIL_COLUMNS, TRUNCATED_RANK, FEATURE_COUNT = 16, 4, 72
BASE = v74.BASE
SPEC_CLASS = v74.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_MPENCIL72_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_MPENCIL72_RIDGE2048_ADD10", 2048.0),
)
BLEND, MAD_SCALE, EPSILON = 0.10, 1.4826, 1e-12
sha256, canonical, write_new = v74.sha256, v74.canonical, v74.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v75 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.matrix_pencil_mode_residual.config.v75",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_P3_MATRIX_PENCIL_DAMPED_MODE_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "columns": int(encoder["hankel_pencil_columns"]) == PENCIL_COLUMNS,
        "rank": int(encoder["truncated_rank"]) == TRUNCATED_RANK,
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
        raise ContractError(f"v75 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def robust_normalize(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < 48 or not np.isfinite(path).all():
        raise ContractError("matrix-pencil path support differs")
    center = float(np.median(path))
    scale = MAD_SCALE * float(np.median(np.abs(path - center)))
    if scale <= EPSILON:
        return np.zeros_like(path)
    return (path - center) / scale


def matrix_pencil_features(values: np.ndarray) -> np.ndarray:
    path = robust_normalize(values)
    if not np.any(path):
        return np.zeros(2 * TRUNCATED_RANK + 1, dtype=np.float64)
    hankel = np.lib.stride_tricks.sliding_window_view(path, PENCIL_COLUMNS + 1)
    left, right = hankel[:, :-1], hankel[:, 1:]
    u_matrix, singular, vh_matrix = np.linalg.svd(left, full_matrices=False)
    if singular[0] <= EPSILON:
        return np.zeros(2 * TRUNCATED_RANK + 1, dtype=np.float64)
    effective_rank = min(TRUNCATED_RANK, int(np.sum(singular > singular[0] * 1e-10)))
    reduced_shift = (
        u_matrix[:, :effective_rank].T
        @ right
        @ vh_matrix[:effective_rank].T
        @ np.diag(1.0 / singular[:effective_rank])
    )
    poles = np.linalg.eigvals(reduced_shift)
    poles = np.pad(poles, (0, TRUNCATED_RANK - len(poles)))
    poles = poles[np.lexsort((-np.abs(poles), np.abs(np.angle(poles))))]
    pole_geometry = np.column_stack([np.abs(poles), np.abs(np.angle(poles)) / np.pi]).ravel()
    residual_energy = max(
        0.0, 1.0 - float(np.sum(singular[:effective_rank] ** 2) / np.sum(singular**2))
    )
    features = np.concatenate([pole_geometry, [residual_energy]]).astype(np.float64)
    if features.shape != (9,) or not np.isfinite(features).all():
        raise ContractError("matrix-pencil feature contract differs")
    return features


def transformed_path(sequence: np.ndarray) -> np.ndarray:
    return v74.transformed_path(np.asarray(sequence)[:289])


def mode_features(sequence: np.ndarray) -> np.ndarray:
    path = transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    features = np.concatenate(
        [
            matrix_pencil_features(path[start:stop, channel])
            for channel in CHANNELS
            for start, stop in WINDOWS
        ]
    )
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("matrix-pencil surface feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    index = np.arange(145, dtype=np.float64)
    path = 0.995**index * np.cos(0.31 * index) + 0.5 * 0.98**index * np.cos(0.77 * index + 0.2)
    features = matrix_pencil_features(path)
    radii, angles = features[:8:2], features[1:8:2] * np.pi
    recovered_radii = np.sort(radii)[::-1]
    recovered_angles = np.sort(angles)
    expected_radii = np.asarray([0.995, 0.995, 0.98, 0.98])
    expected_angles = np.asarray([0.31, 0.31, 0.77, 0.77])
    if not np.max(np.abs(recovered_radii - expected_radii)) < 0.02:
        raise ContractError("damped-radius recovery guard failed")
    if not np.max(np.abs(recovered_angles - expected_angles)) < 0.02:
        raise ContractError("mode-frequency recovery guard failed")
    if not features[-1] < 0.02:
        raise ContractError("rank-four reconstruction guard failed")
    affine = matrix_pencil_features(5.0 * path + 2.0)
    signed = matrix_pencil_features(-path)
    if not np.allclose(features, affine, rtol=1e-10, atol=1e-10):
        raise ContractError("positive affine invariance guard failed")
    if not np.allclose(features, signed, rtol=1e-10, atol=1e-10):
        raise ContractError("sign invariance guard failed")
    if not np.array_equal(matrix_pencil_features(np.ones(80)), np.zeros(9)):
        raise ContractError("constant path bound guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack(
        [np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)]
    )
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = mode_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, mode_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "maximum_radius_error": float(np.max(np.abs(recovered_radii - expected_radii))),
        "maximum_frequency_error_rad": float(np.max(np.abs(recovered_angles - expected_angles))),
        "rank_four_residual_energy": float(features[-1]),
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
        features[position] = mode_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v74.v73.v72.v71.v70.v69.v68.v67.v66.v65.v64.v63.v62.case_surface()
    features, metadata = surface_features(cases)
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(
        len(features) >= int(gate["minimum_cases"])
        and positive_variance >= int(gate["minimum_positive_variance_features"])
    )
    return {
        **metadata,
        "positive_variance_features": positive_variance,
        "target_used": False,
        "passed": passed,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v75 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.matrix_pencil_mode_residual.preflight.v75",
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
            "schema_version": "p3.matrix_pencil_mode_residual.result.v75",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE"
            if any(item["decision"] != "NO_GO" for item in result["candidates"])
            else "NO_GO_ALL_MATRIX_PENCIL_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 fixed matrix-pencil damped-mode residual cycle v75",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v75 extracts fixed complex poles and residual rank energy; it does not reuse stationary spectral magnitudes, adaptive EMD/VMD modes, predictive state-space outputs, or any prior candidate.",
        "- Prior and official outputs were excluded; the repeatedly exposed 182-case surface is EXPLORATORY_ONLY.",
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
        raise ContractError("v75 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v75 support gate failed; zero-fit closure required")
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
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Matrix pencils estimate complex poles of damped and undamped sinusoids | Hua and Sarkar 1990, DOI:10.1109/29.56027 | mechanism only |\n| No executed P3 matrix-pencil or complex-pole geometry exists | repository semantic audit | novelty boundary |\n| Prior/official outputs were excluded | sealed v75 contract | reuse boundary |\n",
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
