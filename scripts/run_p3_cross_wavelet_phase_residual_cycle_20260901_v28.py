"""Exactly-once P3 fixed cross-wavelet relative-phase residual cycle."""

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
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

import run_p3_causal_multichannel_rocket_residual_cycle_20260901_v26 as v26  # noqa: E402
import run_p3_kma_wind_work_residual_axis_cycle_20260901_v20 as v20  # noqa: E402
import run_p3_path_signature_residual_cycle_20260901_v23 as v23  # noqa: E402
from run_p3_parallel_candidate_cycle_20260831_v4 import rmse  # noqa: E402
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    POINTS_PER_RMSE_M,
    bootstrap,
)

EXPERIMENT_ID = "p3_cross_wavelet_phase_residual_cycle_20260901_v28"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = ROOT / "artifacts/p3/sequences_all20_v1/train_values.npy"
STATIONS = ROOT / "artifacts/p3/sequences_all20_v1/train_station.npy"
SCALES = (2, 4, 8, 16, 32)
FEATURE_COUNT = 330
SPECS = (
    v26.Spec("P3_1_XWPHASE330_RIDGE512_ADD10", 512.0),
    v26.Spec("P3_2_XWPHASE330_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10
TRANSPORT_PENALTY_POINTS = 0.04958605409228893
OFFICIAL_CHAMPION_POINTS = 24.203599


class ContractError(RuntimeError):
    """Raised when the sealed v28 contract differs."""


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
    checks = {
        "schema": config["schema_version"] == "p3.cross_wavelet_phase_residual.config.v28",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_CROSS_WAVELET_PHASE_AXIS",
        "scales": tuple(config["phase_encoder"]["scales"]) == SCALES,
        "features": config["phase_encoder"]["feature_count"] == FEATURE_COUNT,
        "phase_only": config["phase_encoder"]["amplitude_features"] == 0,
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
        "tail_guards": config["validation"]["tail_guards"]
        == {
            "worst_block_delta_m_lte": 0.015,
            "worst_lead_delta_m_lte": 0.010,
            "worst_station_lead_delta_m_lte": 0.010,
            "reference_tail_quantile_within_block": 0.80,
            "worst_reference_tail_block_delta_m_lte": 0.015,
        },
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_v27_posthoc": not config["duplication_audit"]["posthoc_v27_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v28 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def complex_wavelet(scale: int) -> np.ndarray:
    steps = np.arange(-2 * scale, 2 * scale + 1, dtype=np.float64)
    window = np.exp(-0.5 * np.square(steps / scale))
    carrier = np.exp(1j * np.pi * steps / scale)
    kernel = window * carrier
    kernel -= kernel.mean()
    kernel /= np.linalg.norm(kernel)
    return kernel


def circular_moments(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    cross = reference * np.conjugate(other)
    magnitude = np.abs(cross)
    unit = np.divide(
        cross,
        magnitude,
        out=np.zeros_like(cross),
        where=magnitude > 1e-12,
    )
    first = np.mean(unit)
    second = np.mean(np.square(unit))
    return np.asarray(
        [first.real, first.imag, abs(first), second.real, second.imag, abs(second)],
        dtype=np.float64,
    )


def phase_features(sequence: np.ndarray) -> np.ndarray:
    path = v26.transformed_path(sequence)
    output: list[float] = []
    for scale in SCALES:
        kernel = complex_wavelet(scale)
        coefficients = np.column_stack(
            [np.convolve(path[:, column], kernel, mode="valid") for column in range(12)]
        )
        reference = coefficients[:, 0]
        for column in range(1, 12):
            output.extend(circular_moments(reference, coefficients[:, column]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("cross-wavelet phase feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = phase_features(sequence)
    kernels = b"".join(complex_wavelet(scale).astype("<c16").tobytes() for scale in SCALES)
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(kernels).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if np.load(SEQUENCES, mmap_mode="r").shape != (24360, 289, 10):
        raise ContractError("sequence cache shape differs")
    if np.load(STATIONS, mmap_mode="r").shape != (24360,):
        raise ContractError("station cache shape differs")
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.cross_wavelet_phase_residual.preflight.v28",
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
        features[position] = phase_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def crossfit(
    cases: pd.DataFrame,
    features: np.ndarray,
    targets: np.ndarray,
    reference: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    residual = targets - reference
    outputs = {spec.name: reference.copy() for spec in SPECS}
    receipts: list[dict[str, Any]] = []
    for block in v23.BLOCKS:
        valid = cases["block"].eq(block).to_numpy()
        train = v23.purged_train_indices(cases, valid)
        for spec in SPECS:
            predicted, receipt = v26.fit_predict(features, residual, train, valid, spec)
            outputs[spec.name][valid] = np.clip(
                reference[valid] + BLEND * predicted, 0.0, 30.0
            )
            receipt.update({"block": block, "additive_residual_weight": BLEND})
            receipts.append(receipt)
    if len(receipts) != 12:
        raise ContractError("fit budget differs")
    return outputs, receipts


def score(frame: pd.DataFrame, prediction: np.ndarray, spec: v26.Spec) -> dict[str, Any]:
    flat = prediction.reshape(-1)
    truth = frame["target_hs"].to_numpy(float)
    reference = frame["reference"].to_numpy(float)
    before, after = rmse(truth, reference), rmse(truth, flat)
    delta = after - before
    by_block = v20.group_deltas(frame, flat, reference, ["block"])
    by_station = v20.group_deltas(frame, flat, reference, ["station"])
    by_lead = v20.group_deltas(frame, flat, reference, ["lead_h"])
    station_lead = v20.group_deltas(frame, flat, reference, ["station", "lead_h"])
    improved = sum(item["delta_rmse_m"] < 0 for item in by_block.values())
    worst_block = max(item["delta_rmse_m"] for item in by_block.values())
    worst_slice = max(item["delta_rmse_m"] for item in station_lead.values())
    worst_lead = max(item["delta_rmse_m"] for item in by_lead.values())
    reference_tail_by_block: dict[str, dict[str, float | int]] = {}
    for block, group in frame.assign(candidate=flat).groupby("block", observed=True):
        threshold = float(np.quantile(group["reference"].to_numpy(float), 0.80))
        tail = group["reference"].to_numpy(float) >= threshold
        tail_truth = group.loc[tail, "target_hs"].to_numpy(float)
        tail_reference = group.loc[tail, "reference"].to_numpy(float)
        tail_candidate = group.loc[tail, "candidate"].to_numpy(float)
        reference_tail_by_block[str(block)] = {
            "rows": int(tail.sum()),
            "uniform_threshold_m": threshold,
            "delta_rmse_m": rmse(tail_truth, tail_candidate)
            - rmse(tail_truth, tail_reference),
        }
    worst_reference_tail = max(
        float(item["delta_rmse_m"]) for item in reference_tail_by_block.values()
    )
    offset = SPECS.index(spec) * 100
    episode_ci = bootstrap(frame, flat, ("episode_id",), 20261301 + offset)
    group_ci = bootstrap(frame, flat, ("block", "station"), 20261302 + offset)
    stable_checks = {
        "delta_rmse_negative": delta < 0,
        "minimum_four_improved_blocks": improved >= 4,
        "episode_ci90_upper_below_zero": episode_ci["ci90_m"][1] < 0,
        "block_station_ci90_upper_below_zero": group_ci["ci90_m"][1] < 0,
        "worst_block_at_most_0p015m": worst_block <= 0.015,
        "worst_lead_at_most_0p010m": worst_lead <= 0.010,
        "worst_station_lead_at_most_0p01m": worst_slice <= 0.01,
        "worst_reference_tail_block_at_most_0p015m": worst_reference_tail <= 0.015,
        "finite_predictions": bool(np.isfinite(flat).all()),
    }
    high_risk_checks = {
        "delta_rmse_at_most_minus_0p005m": delta <= -0.005,
        "worst_station_lead_at_most_0p02m": worst_slice <= 0.02,
        "finite_predictions": stable_checks["finite_predictions"],
    }
    stable = all(stable_checks.values())
    high_risk = not stable and all(high_risk_checks.values())
    points = -delta * POINTS_PER_RMSE_M
    return {
        "name": spec.name,
        "decision": "PASS_STABLE" if stable else "PRESERVE_HIGH_RISK" if high_risk else "NO_GO",
        "ridge_alpha": spec.alpha,
        "additive_residual_weight": BLEND,
        "rmse_m": {
            "uniform_0p425": before,
            "candidate": after,
            "delta_candidate_minus_uniform": delta,
        },
        "expected_points": {
            "raw_gain": points,
            "transport_penalty": TRANSPORT_PENALTY_POINTS,
            "transport_adjusted_gain": points - TRANSPORT_PENALTY_POINTS,
            "nominal_official_score": OFFICIAL_CHAMPION_POINTS + points,
        },
        "improved_blocks": int(improved),
        "by_block": by_block,
        "station": by_station,
        "lead": by_lead,
        "station_lead": station_lead,
        "worst_block_delta_m": worst_block,
        "worst_lead_delta_m": worst_lead,
        "worst_station_lead_delta_m": worst_slice,
        "reference_tail_by_block": reference_tail_by_block,
        "worst_reference_tail_block_delta_m": worst_reference_tail,
        "episode_bootstrap": episode_ci,
        "block_station_bootstrap": group_ci,
        "stable_checks": stable_checks,
        "high_risk_checks": high_risk_checks,
    }


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, profile = v23.case_surface()
    features, feature_receipt = surface_features(cases)
    predictions, receipts = crossfit(cases, features, targets, reference)
    frame = v23.long_frame(cases, targets, reference)
    scored = [score(frame, predictions[spec.name], spec) for spec in SPECS]
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {
        "schema_version": "p3.cross_wavelet_phase_residual.result.v28",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE" if passing else "NO_GO_ALL_PHASE_CANDIDATES",
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
        "lead_h": np.asarray(v23.LEADS, dtype=np.int16),
        "block": cases["block"].to_numpy(dtype="U5"),
        "station": cases["station"].to_numpy(dtype="U5"),
        "episode": cases["episode_id"].to_numpy(dtype="U32"),
    }
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 fixed cross-wavelet relative-phase residual cycle v28",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v28 encodes fixed local relative phase and phase concentration only; it is not a v27 amplitude/scattering retune.",
        "- This repeatedly exposed 182-case surface is EXPLORATORY_ONLY, not a Public transport guarantee.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(
            f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; "
            f"delta {metric['delta_candidate_minus_uniform']:+.9f}m; raw "
            f"{points['raw_gain']:+.6f} points; transport-adjusted "
            f"{points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; "
            f"worst block {item['worst_block_delta_m']:+.9f}m; worst station-lead "
            f"{item['worst_station_lead_delta_m']:+.9f}m; episode CI90 "
            f"{item['episode_bootstrap']['ci90_m']}; block-station CI90 "
            f"{item['block_station_bootstrap']['ci90_m']}."
        )
    lines.extend(
        [
            "",
            "Official test/sample/submission/hidden access, CSV materialization, and upload were all zero. No row was deleted and no outer result changed features, Ridge strengths, or blend.",
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
        raise ContractError("v28 exactly-once namespace already exists")
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
        b"# Gap matrix\n\n| Family | Information | Verdict |\n|---|---|---|\n| v27 scattering | wavelet modulus magnitudes | Distinct |\n| global spectral | Fourier/random-frequency amplitude | Distinct |\n| path signature | ordered iterated integrals | Distinct |\n| v28 cross-wavelet phase | local signed relative phase and circular concentration | Executed |\n",
    )
    write_new(
        REPORT / "claim-source-ledger.md",
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Complex wavelets localize scale and phase | Torrence and Compo, 1998, DOI:10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2 | fixed phase representation |\n| Cross-wavelet phase/coherence describes localized relation between two geophysical series | Grinsted, Moore, and Jevrejeva, 2004, DOI:10.5194/npg-11-561-2004 | cross-channel phase basis |\n| No P3 cross-wavelet phase implementation exists locally | repository semantic audit before sealing | novelty gate |\n",
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
