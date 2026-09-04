"""Sealed P3 v34 normalized bispectral residual experiment."""

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

import run_p3_bocpd_regime_age_residual_cycle_20260901_v32 as v32  # noqa: E402
import run_p3_causal_multichannel_rocket_residual_cycle_20260901_v26 as v26  # noqa: E402

EXPERIMENT_ID = "p3_bicoherence_residual_cycle_20260901_v34"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = ROOT / "artifacts/p3/sequences_all20_v1/train_values.npy"
STATIONS = ROOT / "artifacts/p3/sequences_all20_v1/train_station.npy"
SEGMENT_LENGTHS = (32, 64)
TRIADS = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
)
FEATURE_COUNT = 72
SPECS = (
    v26.Spec("P3_1_BIC72_RIDGE512_ADD10", 512.0),
    v26.Spec("P3_2_BIC72_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10


class ContractError(RuntimeError):
    """Raised when the sealed v34 contract differs."""


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
        "schema": config["schema_version"] == "p3.bicoherence_residual.config.v34",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_QUADRATIC_PHASE_COUPLING_AXIS",
        "segments": tuple(encoder["segment_lengths"]) == SEGMENT_LENGTHS,
        "triads": tuple(tuple(item) for item in encoder["fixed_triads"]) == TRIADS,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "detrend": encoder["detrend_order"] == 1,
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
        raise ContractError(f"v34 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def detrended_spectra(values: np.ndarray, length: int) -> np.ndarray:
    hop = length // 2
    starts = range(0, len(values) - length + 1, hop)
    axis = np.arange(length, dtype=np.float64)
    design = np.column_stack([axis, np.ones(length, dtype=np.float64)])
    window = np.hanning(length)
    spectra = []
    for start in starts:
        segment = values[start : start + length]
        residual = segment - design @ np.linalg.lstsq(design, segment, rcond=None)[0]
        spectra.append(np.fft.rfft(residual * window))
    output = np.asarray(spectra, dtype=np.complex128)
    if output.ndim != 2 or len(output) < 3 or not np.isfinite(output).all():
        raise ContractError("bispectral segment contract differs")
    return output


def normalized_bispectrum(
    spectra: np.ndarray, first_frequency: int, second_frequency: int
) -> complex:
    sum_frequency = first_frequency + second_frequency
    first_product = spectra[:, first_frequency] * spectra[:, second_frequency]
    third = spectra[:, sum_frequency]
    numerator = np.mean(first_product * np.conjugate(third))
    denominator = np.sqrt(
        np.mean(np.square(np.abs(first_product))) * np.mean(np.square(np.abs(third)))
    )
    if denominator <= 1e-15:
        return 0.0 + 0.0j
    value = numerator / denominator
    magnitude = abs(value)
    if magnitude > 1.0:
        value /= magnitude
    return complex(value)


def bicoherence_features(sequence: np.ndarray) -> np.ndarray:
    path = v26.transformed_path(sequence)
    native_hs = path[::2, 0]
    if native_hs.shape != (145,):
        raise ContractError("native 20-minute hs path differs")
    output: list[float] = []
    for length in SEGMENT_LENGTHS:
        spectra = detrended_spectra(native_hs, length)
        for first, second in TRIADS:
            value = normalized_bispectrum(spectra, first, second)
            output.extend([abs(value), value.real, value.imag])
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("bicoherence feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = bicoherence_features(sequence)
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if np.load(SEQUENCES, mmap_mode="r").shape != (24360, 289, 10):
        raise ContractError("sequence cache shape differs")
    if np.load(STATIONS, mmap_mode="r").shape != (24360,):
        raise ContractError("station cache shape differs")
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v34 exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.bicoherence_residual.preflight.v34",
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
        features[position] = bicoherence_features(sequences[anchor_id])
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
        predictions, receipts = v32.v28.crossfit(
            cases, features, targets, reference
        )
        frame = v32.v23.long_frame(cases, targets, reference)
        scored = [
            v32.v28.score(frame, predictions[spec.name], spec) for spec in SPECS
        ]
    finally:
        v32.v28.SPECS = original_specs
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {
        "schema_version": "p3.bicoherence_residual.result.v34",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE"
        if passing
        else "NO_GO_ALL_BICOHERENCE_CANDIDATES",
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
        "# P3 normalized bispectral residual cycle v34",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v34 represents fixed third-order quadratic phase coupling, not ordinary power, pairwise cross-wavelet phase, or DCCA covariance.",
        "- The 182-case surface remains EXPLORATORY_ONLY.",
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
            "Official test/sample/submission/hidden access, CSV materialization, and upload were all zero. No row was deleted and no outer result changed triads, segments, Ridge strengths, or blend.",
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
        raise ContractError("v34 exactly-once namespace already exists")
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
        b"# Gap matrix\n\n| Audited axis | Verdict |\n|---|---|\n| ordinary power spectrum | closed second-order spectral lineage |\n| cross-wavelet/DCCA | pairwise phase or second-order covariance; distinct |\n| normalized bispectrum | nonduplicate third-order quadratic phase-coupling axis; executed as v34 |\n",
    )
    write_new(
        REPORT / "claim-source-ledger.md",
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Bispectra contain nonlinearity and quadratic phase-coupling information absent from ordinary power spectra | Raghuveer and Nikias, IEEE TASSP 33(5), 1985, DOI:10.1109/TASSP.1985.1164679 | fixed normalized third-order spectrum |\n| Bicoherence has been used to characterize nonlinear wave-wave coupling | Ma et al., Coastal Engineering 55, 2008, DOI:10.1016/j.coastaleng.2008.02.015 | wave-process rationale only |\n| No corresponding P3 implementation exists | repository semantic audit before sealing | novelty gate |\n",
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
