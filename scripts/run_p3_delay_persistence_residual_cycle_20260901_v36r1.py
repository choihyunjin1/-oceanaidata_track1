"""Science-neutral numerical recovery for sealed P3 v36 persistence cycle."""

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

import run_p3_delay_persistence_residual_cycle_20260901_v36 as v36  # noqa: E402

EXPERIMENT_ID = "p3_delay_persistence_residual_cycle_20260901_v36r1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SOURCE_ID = "p3_delay_persistence_residual_cycle_20260901_v36"
SOURCE_CONFIG = ROOT / "configs/experiments" / f"{SOURCE_ID}.json"
SOURCE_RUNNER = ROOT / "scripts" / f"run_{SOURCE_ID}.py"
SOURCE_LOCK = ROOT / "artifacts" / f"{SOURCE_ID}.ATTEMPT_LOCK.json"
SOURCE_FAILURE = ROOT / "reports" / SOURCE_ID / "failure-receipt.json"
SOURCE_FAILURE_REPORT = ROOT / "reports" / SOURCE_ID / "failure-report.md"
SEQUENCES = v36.SEQUENCES
STATIONS = v36.STATIONS
CHANNELS = v36.CHANNELS
CHANNEL_NAMES = v36.CHANNEL_NAMES
DELAYS = v36.DELAYS
EMBEDDING_DIMENSION = v36.EMBEDDING_DIMENSION
FEATURE_COUNT = v36.FEATURE_COUNT
SPECS = v36.SPECS
BLEND = v36.BLEND


class ContractError(RuntimeError):
    """Raised when the sealed v36r1 recovery contract differs."""


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
    recovery = config["recovery"]
    checks = {
        "schema": config["schema_version"]
        == "p3.delay_persistence_residual.config.v36r1",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "fresh_recovery": recovery["source_experiment_id"] == SOURCE_ID
        and not recovery["same_id_restart"],
        "science_zero_adapter_one": recovery["science_changes"] == 0
        and recovery["numerical_adapter_changes"] == 1,
        "source_unscored": recovery["source_fit_count"] == 0
        and recovery["source_outer_scores_exposed"] == 0,
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
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
        raise ContractError(f"v36r1 config contract failed: {checks}")
    source_paths = {
        SOURCE_CONFIG: recovery["source_hashes"]["config_sha256"],
        SOURCE_RUNNER: recovery["source_hashes"]["runner_sha256"],
        SOURCE_LOCK: recovery["source_hashes"]["lock_sha256"],
        SOURCE_FAILURE: recovery["source_hashes"]["failure_receipt_sha256"],
        SOURCE_FAILURE_REPORT: recovery["source_hashes"]["failure_report_sha256"],
    }
    for path, expected in source_paths.items():
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"v36 source evidence differs: {path}")
    if (ROOT / "artifacts" / SOURCE_ID / "result.json").exists():
        raise ContractError("v36 unexpectedly has a terminal result")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def kruskal_lifetimes(points: np.ndarray) -> np.ndarray:
    count = len(points)
    lower, upper = np.triu_indices(count, k=1)
    weights = np.linalg.norm(points[lower] - points[upper], axis=1)
    order = np.lexsort((upper, lower, weights))
    parent = np.arange(count, dtype=np.int64)
    rank = np.zeros(count, dtype=np.int8)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    selected: list[float] = []
    for index in order:
        first = find(int(lower[index]))
        second = find(int(upper[index]))
        if first == second:
            continue
        if rank[first] < rank[second]:
            first, second = second, first
        parent[second] = first
        if rank[first] == rank[second]:
            rank[first] += 1
        selected.append(float(weights[index]))
        if len(selected) == count - 1:
            break
    lifetimes = np.asarray(selected, dtype=np.float64)
    if lifetimes.shape != (count - 1,) or not np.isfinite(lifetimes).all():
        raise ContractError("deterministic Kruskal lifetime contract differs")
    if np.any(np.diff(lifetimes) < 0.0) or np.any(lifetimes < 0.0):
        raise ContractError("Kruskal lifetime ordering differs")
    return lifetimes


def persistence_lifetimes(values: np.ndarray, delay: int) -> np.ndarray:
    return kruskal_lifetimes(v36.delay_embedding(values, delay))


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
        raise ContractError("recovered persistence summary contract differs")
    return features


def delay_persistence_features(sequence: np.ndarray) -> np.ndarray:
    path = v36.v26.transformed_path(sequence)
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
        raise ContractError("recovered delay-persistence feature contract differs")
    return features


def recovery_receipt() -> dict[str, Any]:
    constant = np.ones(145, dtype=np.float64)
    constant_lifetimes = persistence_lifetimes(constant, 2)
    rng = np.random.default_rng(20260901)
    benign = rng.normal(size=145)
    recovered = persistence_lifetimes(benign, 6)
    original = v36.persistence_lifetimes(benign, 6)
    agreement = float(np.max(np.abs(recovered - original)))
    if (
        len(constant_lifetimes) != 140
        or np.count_nonzero(constant_lifetimes) != 0
        or agreement > 1e-12
    ):
        raise ContractError("science-neutral Kruskal recovery preflight failed")
    return {
        "science_changes": 0,
        "numerical_adapter_changes": 1,
        "constant_cloud_mst_edges": len(constant_lifetimes),
        "constant_cloud_nonzero_edges": int(np.count_nonzero(constant_lifetimes)),
        "benign_original_adapter_max_abs_difference": agreement,
        "source_failure_receipt_sha256": sha256(SOURCE_FAILURE),
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
        "recovery": recovery_receipt(),
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if np.load(SEQUENCES, mmap_mode="r").shape != (24360, 289, 10):
        raise ContractError("sequence cache shape differs")
    if np.load(STATIONS, mmap_mode="r").shape != (24360,):
        raise ContractError("station cache shape differs")
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v36r1 exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.delay_persistence_residual.preflight.v36r1",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE_SCIENCE_NEUTRAL_RECOVERY",
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
    cases, targets, reference, profile = v36.v32.v23.case_surface()
    features, feature_receipt = surface_features(cases)
    original_specs = v36.v32.v28.SPECS
    v36.v32.v28.SPECS = SPECS
    try:
        predictions, receipts = v36.v32.v28.crossfit(cases, features, targets, reference)
        frame = v36.v32.v23.long_frame(cases, targets, reference)
        scored = [v36.v32.v28.score(frame, predictions[spec.name], spec) for spec in SPECS]
    finally:
        v36.v32.v28.SPECS = original_specs
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {
        "schema_version": "p3.delay_persistence_residual.result.v36r1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE"
        if passing
        else "NO_GO_ALL_DELAY_PERSISTENCE_CANDIDATES",
        "surface_claim": config["validation"]["surface"],
        "reference": config["reference"],
        "recovery": config["recovery"],
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
        "lead_h": np.asarray(v36.v32.v23.LEADS, dtype=np.int16),
        "block": cases["block"].to_numpy(dtype="U5"),
        "station": cases["station"].to_numpy(dtype="U5"),
        "episode": cases["episode_id"].to_numpy(dtype="U32"),
    }
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 delay-persistence science-neutral recovery v36r1",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v36r1 changes only the numerical MST adapter so exact-zero edges remain valid. Science changes are zero; the failed v36 exposed zero fits and zero outer scores.",
        "- The 182-case surface remains EXPLORATORY_ONLY and no v35 prediction, ensemble, or router was reused.",
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
        raise ContractError("v36r1 exactly-once namespace already exists")
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
        REPORT / "run-manifest.json",
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "result_sha256": sha256(result_path),
                "arrays_sha256": sha256(array_path),
                "report_sha256": sha256(report_path),
                "fit_count": 12,
                "science_changes": 0,
                "numerical_adapter_changes": 1,
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
