"""Sealed P3 v35 empirical-mode/Hilbert residual experiment."""

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
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

import run_p3_bocpd_regime_age_residual_cycle_20260901_v32 as v32  # noqa: E402
import run_p3_causal_multichannel_rocket_residual_cycle_20260901_v26 as v26  # noqa: E402

EXPERIMENT_ID = "p3_empirical_mode_hilbert_residual_cycle_20260901_v35"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = ROOT / "artifacts/p3/sequences_all20_v1/train_values.npy"
STATIONS = ROOT / "artifacts/p3/sequences_all20_v1/train_station.npy"
CHANNELS = (0, 1, 2, 5)
CHANNEL_NAMES = ("hs", "tp", "hmax", "wspd")
MODE_COUNT = 3
SIFTS_PER_MODE = 8
FEATURE_COUNT = 80
SPECS = (
    v26.Spec("P3_1_EMDH80_RIDGE512_ADD10", 512.0),
    v26.Spec("P3_2_EMDH80_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10


class ContractError(RuntimeError):
    """Raised when the sealed v35 contract differs."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"]
        == "p3.empirical_mode_hilbert_residual.config.v35",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_ADAPTIVE_INTRINSIC_MODE_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "rows": encoder["context_rows"] == 145,
        "modes": encoder["mode_count"] == MODE_COUNT,
        "sifts": encoder["sifts_per_mode"] == SIFTS_PER_MODE,
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
        raise ContractError(f"v35 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def robust_case_normalize(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    median = float(np.median(raw))
    q25, q75 = np.quantile(raw, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        scale = 1.0
    output = (raw - median) / scale
    if not np.isfinite(output).all():
        raise ContractError("case normalization produced nonfinite values")
    return output


def extrema_indices(values: np.ndarray, maximum: bool) -> np.ndarray:
    middle = (
        (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
        if maximum
        else (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    )
    interior = np.flatnonzero(middle) + 1
    return np.unique(np.concatenate(([0], interior, [len(values) - 1]))).astype(np.int64)


def extrema_envelope(values: np.ndarray, maximum: bool) -> np.ndarray:
    knots = extrema_indices(values, maximum)
    axis = np.arange(len(values), dtype=np.float64)
    if len(knots) < 3:
        return np.interp(axis, knots.astype(np.float64), values[knots])
    return CubicSpline(
        knots.astype(np.float64), values[knots], bc_type="natural", extrapolate=False
    )(axis)


def fixed_emd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = robust_case_normalize(values)
    modes = np.empty((MODE_COUNT, len(residual)), dtype=np.float64)
    for mode_index in range(MODE_COUNT):
        candidate = residual.copy()
        for _ in range(SIFTS_PER_MODE):
            upper = extrema_envelope(candidate, True)
            lower = extrema_envelope(candidate, False)
            candidate = candidate - 0.5 * (upper + lower)
            if not np.isfinite(candidate).all():
                raise ContractError("fixed EMD sifting produced nonfinite values")
        modes[mode_index] = candidate
        residual = residual - candidate
    if not np.isfinite(modes).all() or not np.isfinite(residual).all():
        raise ContractError("fixed EMD output is nonfinite")
    return modes, residual


def mode_features(values: np.ndarray) -> np.ndarray:
    modes, residual = fixed_emd(values)
    total_energy = float(np.sum(np.square(modes)) + np.sum(np.square(residual)))
    total_energy = max(total_energy, 1e-12)
    output: list[float] = []
    midpoint = len(values) // 2
    for mode in modes:
        analytic = hilbert(mode)
        amplitude = np.abs(analytic)
        frequency = np.abs(np.diff(np.unwrap(np.angle(analytic))) / (2.0 * np.pi))
        early = float(np.mean(np.square(mode[:midpoint])))
        recent = float(np.mean(np.square(mode[midpoint:])))
        output.extend(
            [
                float(np.sum(np.square(mode)) / total_energy),
                float(np.mean(amplitude)),
                float(np.quantile(amplitude, 0.90)),
                float(np.mean(frequency)),
                float(np.quantile(frequency, 0.90)),
                float(np.log((recent + 1e-8) / (early + 1e-8))),
            ]
        )
    output.extend(
        [
            float(np.sum(np.square(residual)) / total_energy),
            float((residual[-1] - residual[0]) / max(len(residual) - 1, 1)),
        ]
    )
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (20,) or not np.isfinite(features).all():
        raise ContractError("per-path EMD/Hilbert feature contract differs")
    return features


def empirical_mode_hilbert_features(sequence: np.ndarray) -> np.ndarray:
    path = v26.transformed_path(sequence)
    native = path[::2]
    if native.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    features = np.concatenate([mode_features(native[:, channel]) for channel in CHANNELS])
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("EMD/Hilbert feature contract differs")
    return features


def am_fm_receipt() -> dict[str, Any]:
    axis = np.arange(145, dtype=np.float64)
    envelope = 1.0 + 0.30 * np.sin(2.0 * np.pi * axis / 80.0)
    signal = envelope * np.sin(2.0 * np.pi * axis / 10.0) + 0.25 * np.sin(
        2.0 * np.pi * axis / 90.0
    )
    modes, residual = fixed_emd(signal)
    analytic = hilbert(modes[0])
    amplitude_correlation = float(np.corrcoef(np.abs(analytic), envelope)[0, 1])
    instantaneous = np.abs(np.diff(np.unwrap(np.angle(analytic))) / (2.0 * np.pi))
    reconstructed = modes.sum(axis=0) + residual
    normalized = robust_case_normalize(signal)
    return {
        "amplitude_envelope_correlation": amplitude_correlation,
        "median_instantaneous_frequency": float(np.median(instantaneous)),
        "reconstruction_max_abs_error": float(np.max(np.abs(reconstructed - normalized))),
        "finite": bool(
            np.isfinite(modes).all()
            and np.isfinite(residual).all()
            and np.isfinite(instantaneous).all()
        ),
    }


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = empirical_mode_hilbert_features(sequence)
    am_fm = am_fm_receipt()
    if (
        not am_fm["finite"]
        or am_fm["amplitude_envelope_correlation"] <= 0.50
        or not 0.07 <= am_fm["median_instantaneous_frequency"] <= 0.13
        or am_fm["reconstruction_max_abs_error"] > 1e-10
    ):
        raise ContractError(f"AM-FM recovery contract failed: {am_fm}")
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "am_fm": am_fm,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if np.load(SEQUENCES, mmap_mode="r").shape != (24360, 289, 10):
        raise ContractError("sequence cache shape differs")
    if np.load(STATIONS, mmap_mode="r").shape != (24360,):
        raise ContractError("station cache shape differs")
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v35 exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.empirical_mode_hilbert_residual.preflight.v35",
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
        features[position] = empirical_mode_hilbert_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, profile = v32.v23.case_surface()
    features, feature_receipt = surface_features(cases)
    original_specs = v32.v28.SPECS
    v32.v28.SPECS = SPECS
    try:
        predictions, receipts = v32.v28.crossfit(cases, features, targets, reference)
        frame = v32.v23.long_frame(cases, targets, reference)
        scored = [v32.v28.score(frame, predictions[spec.name], spec) for spec in SPECS]
    finally:
        v32.v28.SPECS = original_specs
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {
        "schema_version": "p3.empirical_mode_hilbert_residual.result.v35",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE"
        if passing
        else "NO_GO_ALL_EMD_HILBERT_CANDIDATES",
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
        "lead_h": np.asarray(v32.v23.LEADS, dtype=np.int16),
        "block": cases["block"].to_numpy(dtype="U5"),
        "station": cases["station"].to_numpy(dtype="U5"),
        "episode": cases["episode_id"].to_numpy(dtype="U32"),
    }
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 empirical-mode/Hilbert residual cycle v35",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v35 is a fixed extrema-envelope adaptive intrinsic-mode representation, not a fixed Fourier/wavelet basis or a linear state-space/SSA factorization.",
        "- Huang et al. (1998) motivates the representation only; it is not performance evidence. The 182-case surface remains EXPLORATORY_ONLY.",
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
    lines.extend(
        [
            "",
            "Official test/sample/submission/hidden access, CSV materialization, and upload were all zero. No row was deleted and no outer result changed modes, sifts, edge treatment, features, Ridge strengths, or blend.",
        ]
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
        raise ContractError("v35 exactly-once namespace already exists")
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
        REPORT / "gap-matrix.md",
        b"# Gap matrix\n\n| Audited axis | Verdict |\n|---|---|\n| fixed Fourier/wavelet representations | closed basis lineages |\n| SSA/state-space | linear low-rank/transition structure; distinct |\n| extrema-envelope EMD plus Hilbert analytic signal | nonduplicate adaptive intrinsic-mode axis; executed as v35 |\n",
    )
    write_new(
        REPORT / "claim-source-ledger.md",
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| EMD defines data-adaptive intrinsic modes using local extrema/envelopes and Hilbert analysis supplies instantaneous amplitude/frequency | Huang et al., Proceedings of the Royal Society A 454, 1998, DOI:10.1098/rspa.1998.0193 | representation motivation only, not performance evidence |\n| No corresponding P3 implementation exists and fixed spectral/wavelet/SSA-state-space families encode different structure | repository exact and semantic audit before sealing | novelty gate |\n",
    )
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
