"""Sealed P3 v36 delay-embedding H0-persistence residual experiment."""

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
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

import run_p3_bocpd_regime_age_residual_cycle_20260901_v32 as v32  # noqa: E402
import run_p3_causal_multichannel_rocket_residual_cycle_20260901_v26 as v26  # noqa: E402

EXPERIMENT_ID = "p3_delay_persistence_residual_cycle_20260901_v36"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = ROOT / "artifacts/p3/sequences_all20_v1/train_values.npy"
STATIONS = ROOT / "artifacts/p3/sequences_all20_v1/train_station.npy"
CHANNELS = (0, 1, 2, 5)
CHANNEL_NAMES = ("hs", "tp", "hmax", "wspd")
DELAYS = (2, 6, 12)
EMBEDDING_DIMENSION = 3
FEATURE_COUNT = 72
SPECS = (
    v26.Spec("P3_1_DPH72_RIDGE512_ADD10", 512.0),
    v26.Spec("P3_2_DPH72_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10


class ContractError(RuntimeError):
    """Raised when the sealed v36 contract differs."""


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
        "schema": config["schema_version"] == "p3.delay_persistence_residual.config.v36",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_DELAY_PERSISTENCE_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "rows": encoder["context_rows"] == 145,
        "dimension": encoder["embedding_dimension"] == EMBEDDING_DIMENSION,
        "delays": tuple(encoder["fixed_delays_rows"]) == DELAYS,
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
        raise ContractError(f"v36 config contract failed: {checks}")
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


def delay_embedding(values: np.ndarray, delay: int) -> np.ndarray:
    normalized = robust_case_normalize(values)
    rows = len(normalized) - (EMBEDDING_DIMENSION - 1) * delay
    if rows <= 3:
        raise ContractError("delay embedding has insufficient rows")
    points = np.column_stack(
        [normalized[offset * delay : offset * delay + rows] for offset in range(3)]
    )
    if points.shape != (rows, 3) or not np.isfinite(points).all():
        raise ContractError("delay embedding contract differs")
    return points


def persistence_lifetimes(values: np.ndarray, delay: int) -> np.ndarray:
    points = delay_embedding(values, delay)
    distances = squareform(pdist(points, metric="euclidean"))
    off_diagonal_zero = (distances == 0.0) & ~np.eye(len(points), dtype=bool)
    distances[off_diagonal_zero] = 1e-15
    tree = minimum_spanning_tree(distances)
    lifetimes = np.sort(np.asarray(tree.data, dtype=np.float64))
    if lifetimes.shape != (len(points) - 1,) or not np.isfinite(lifetimes).all():
        raise ContractError("H0 persistence/MST lifetime contract differs")
    if np.any(lifetimes < 0.0):
        raise ContractError("negative persistence lifetime")
    return lifetimes


def lifetime_features(values: np.ndarray, delay: int) -> np.ndarray:
    lifetimes = persistence_lifetimes(values, delay)
    top_count = max(1, int(np.ceil(0.10 * len(lifetimes))))
    total = float(np.sum(lifetimes))
    features = np.asarray(
        [
            np.mean(lifetimes),
            np.std(lifetimes),
            np.median(lifetimes),
            np.quantile(lifetimes, 0.90),
            np.max(lifetimes),
            np.sum(lifetimes[-top_count:]) / max(total, 1e-12),
        ],
        dtype=np.float64,
    )
    if features.shape != (6,) or not np.isfinite(features).all():
        raise ContractError("persistence summary contract differs")
    return features


def delay_persistence_features(sequence: np.ndarray) -> np.ndarray:
    path = v26.transformed_path(sequence)
    native = path[::2]
    if native.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    features = np.concatenate(
        [
            lifetime_features(native[:, channel], delay)
            for channel in CHANNELS
            for delay in DELAYS
        ]
    )
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("delay-persistence feature contract differs")
    return features


def topology_sensitivity_receipt() -> dict[str, Any]:
    axis = np.arange(145, dtype=np.float64)
    periodic = np.sin(2.0 * np.pi * axis / 18.0)
    regime = periodic.copy()
    regime[73:] += 3.0
    first = np.concatenate([lifetime_features(periodic, delay) for delay in DELAYS])
    second = np.concatenate([lifetime_features(regime, delay) for delay in DELAYS])
    difference = float(np.linalg.norm(first - second))
    if difference <= 0.05:
        raise ContractError("synthetic topology sensitivity is too small")
    return {
        "periodic_feature_sha256": hashlib.sha256(first.astype("<f8").tobytes()).hexdigest(),
        "regime_feature_sha256": hashlib.sha256(second.astype("<f8").tobytes()).hexdigest(),
        "l2_difference": difference,
        "finite": bool(np.isfinite(first).all() and np.isfinite(second).all()),
    }


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = delay_persistence_features(sequence)
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "topology_sensitivity": topology_sensitivity_receipt(),
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if np.load(SEQUENCES, mmap_mode="r").shape != (24360, 289, 10):
        raise ContractError("sequence cache shape differs")
    if np.load(STATIONS, mmap_mode="r").shape != (24360,):
        raise ContractError("station cache shape differs")
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v36 exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.delay_persistence_residual.preflight.v36",
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
        features[position] = delay_persistence_features(sequences[anchor_id])
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
        "schema_version": "p3.delay_persistence_residual.result.v36",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE"
        if passing
        else "NO_GO_ALL_DELAY_PERSISTENCE_CANDIDATES",
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
        "# P3 delay-embedding H0-persistence residual cycle v36",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v36 summarizes all-threshold H0 merge lifetimes of fixed delay-coordinate point clouds; it is not v29's single-threshold ordinal/RQA representation and does not reuse v35 predictions.",
        "- Perea and Harer (2015) motivates sliding-window persistent topology only. v36 makes the narrower deterministic H0/MST claim, and the 182-case surface remains EXPLORATORY_ONLY.",
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
            "Official test/sample/submission/hidden access, CSV materialization, and upload were all zero. No row was deleted and no outer result changed channels, delays, embedding dimension, lifetime summaries, Ridge strengths, or blend.",
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
        raise ContractError("v36 exactly-once namespace already exists")
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
        b"# Gap matrix\n\n| Audited axis | Verdict |\n|---|---|\n| v29 ordinal/RQA | ordinal words plus single-threshold recurrence topology |\n| fixed spectral/wavelet or v35 adaptive modes | different frequency/mode representations |\n| v36 delay-coordinate H0 persistence | nonduplicate all-threshold component-merger filtration; executed |\n",
    )
    write_new(
        REPORT / "claim-source-ledger.md",
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Sliding-window embeddings can be studied through persistent homology | Perea and Harer, Foundations of Computational Mathematics 15, 2015, DOI:10.1007/s10208-014-9206-z | representation motivation only |\n| v36 computes only deterministic H0 finite lifetimes, equivalent to MST edge lengths, and not the paper's full 1D method | sealed local implementation contract | scope boundary |\n| No corresponding P3 persistence/Vietoris/Betti implementation exists; v29 is fixed-threshold RQA | repository exact and semantic audit before sealing | novelty gate |\n",
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
