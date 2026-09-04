"""Sealed P3 v37 full horizontal-visibility graph residual experiment."""

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

EXPERIMENT_ID = "p3_horizontal_visibility_residual_cycle_20260901_v37"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = v36.SEQUENCES
STATIONS = v36.STATIONS
CHANNELS = v36.CHANNELS
CHANNEL_NAMES = v36.CHANNEL_NAMES
WINDOWS = (145, 73)
FEATURE_COUNT = 64
SPECS = (
    v36.v26.Spec("P3_1_HVG64_RIDGE512_ADD10", 512.0),
    v36.v26.Spec("P3_2_HVG64_RIDGE2048_ADD10", 2048.0),
)
BLEND = 0.10


class ContractError(RuntimeError):
    """Raised when the sealed v37 contract differs."""


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
        == "p3.horizontal_visibility_residual.config.v37",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_FULL_HORIZONTAL_VISIBILITY_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(encoder["fixed_windows_rows"]) == WINDOWS,
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
        raise ContractError(f"v37 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def horizontal_visibility_adjacency(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if path.ndim != 1 or len(path) < 3 or not np.isfinite(path).all():
        raise ContractError("HVG input contract differs")
    adjacency = np.zeros((len(path), len(path)), dtype=np.uint8)
    for left in range(len(path) - 1):
        maximum_between = -np.inf
        for right in range(left + 1, len(path)):
            if right > left + 1:
                maximum_between = max(maximum_between, path[right - 1])
            if maximum_between < min(path[left], path[right]):
                adjacency[left, right] = adjacency[right, left] = 1
    if not np.array_equal(adjacency, adjacency.T) or np.any(np.diag(adjacency)):
        raise ContractError("HVG adjacency contract differs")
    if not np.all(adjacency[np.arange(len(path) - 1), np.arange(1, len(path))] == 1):
        raise ContractError("HVG adjacent-edge contract differs")
    return adjacency


def graph_features(values: np.ndarray) -> np.ndarray:
    adjacency = horizontal_visibility_adjacency(values)
    degree = adjacency.sum(axis=1).astype(np.float64)
    edge_left, edge_right = np.where(np.triu(adjacency, k=1) == 1)
    spans = (edge_right - edge_left).astype(np.float64)
    counts = np.bincount(degree.astype(np.int64))
    probability = counts[counts > 0].astype(np.float64) / len(degree)
    entropy = float(-np.sum(probability * np.log(probability)))
    triangles = float(np.trace(adjacency.astype(np.float64) @ adjacency @ adjacency) / 6.0)
    triples = float(np.sum(degree * (degree - 1.0) / 2.0))
    left_degree = np.tril(adjacency, k=-1).sum(axis=1).astype(np.float64)
    right_degree = np.triu(adjacency, k=1).sum(axis=1).astype(np.float64)
    features = np.asarray(
        [
            np.mean(degree),
            np.std(degree),
            np.max(degree),
            entropy,
            np.mean(spans),
            np.quantile(spans, 0.90),
            3.0 * triangles / max(triples, 1.0),
            np.mean(np.abs(left_degree - right_degree)),
        ],
        dtype=np.float64,
    )
    if features.shape != (8,) or not np.isfinite(features).all():
        raise ContractError("HVG summary contract differs")
    return features


def horizontal_visibility_features(sequence: np.ndarray) -> np.ndarray:
    path = v36.v26.transformed_path(sequence)
    native = path[::2]
    if native.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    features = np.concatenate(
        [
            graph_features(native[-window:, channel])
            for channel in CHANNELS
            for window in WINDOWS
        ]
    )
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("horizontal-visibility feature contract differs")
    return features


def graph_preflight_receipt() -> dict[str, Any]:
    values = np.asarray([0.2, 1.0, 0.1, 0.8, 0.3, 1.2, 0.4], dtype=np.float64)
    transformed = np.exp(values)
    adjacency = horizontal_visibility_adjacency(values)
    transformed_adjacency = horizontal_visibility_adjacency(transformed)
    feature = graph_features(values)
    reversed_feature = graph_features(values[::-1])
    if not np.array_equal(adjacency, transformed_adjacency):
        raise ContractError("strictly monotone order-obstruction invariance failed")
    if feature[-1] <= 0.0 or reversed_feature[-1] <= 0.0:
        raise ContractError("synthetic temporal asymmetry signal absent")
    return {
        "node_count": len(values),
        "edge_count": int(adjacency.sum() // 2),
        "monotone_transform_adjacency_identical": True,
        "mean_absolute_left_right_asymmetry": float(feature[-1]),
        "reverse_mean_absolute_left_right_asymmetry": float(reversed_feature[-1]),
        "adjacency_sha256": hashlib.sha256(adjacency.tobytes()).hexdigest(),
    }


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = horizontal_visibility_features(sequence)
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "graph": graph_preflight_receipt(),
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if np.load(SEQUENCES, mmap_mode="r").shape != (24360, 289, 10):
        raise ContractError("sequence cache shape differs")
    if np.load(STATIONS, mmap_mode="r").shape != (24360,):
        raise ContractError("station cache shape differs")
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v37 exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.horizontal_visibility_residual.preflight.v37",
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
        features[position] = horizontal_visibility_features(sequences[anchor_id])
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
        "schema_version": "p3.horizontal_visibility_residual.result.v37",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE"
        if passing
        else "NO_GO_ALL_HORIZONTAL_VISIBILITY_CANDIDATES",
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
        "lead_h": np.asarray(v36.v32.v23.LEADS, dtype=np.int16),
        "block": cases["block"].to_numpy(dtype="U5"),
        "station": cases["station"].to_numpy(dtype="U5"),
        "episode": cases["episode_id"].to_numpy(dtype="U32"),
    }
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 full horizontal-visibility graph residual cycle v37",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v37 uses full P3 visibility graphs, unlike P1's endpoint-only detector, v29 metric recurrence, or v36 distance filtration. No earlier prediction or feature is reused.",
        "- Luque et al. (2009) motivates the graph mechanism only; the 182-case surface remains EXPLORATORY_ONLY.",
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
        raise ContractError("v37 exactly-once namespace already exists")
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
                "official_access": 0,
                "csv_materializations": 0,
                "uploads": 0,
            }
        ),
    )
    write_new(
        REPORT / "claim-source-ledger.md",
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Horizontal visibility maps time-series order obstruction to a graph | Luque et al., Physical Review E 80, 2009, DOI:10.1103/PhysRevE.80.046103 | graph mechanism only |\n| P1 v13 is endpoint-only; P3 v37 is a full graph regression representation | repository semantic audit | cross-problem distinction |\n| P3 v29 recurrence and v36 persistence use metric state-space distances, unlike HVG order obstruction | local sealed contracts | P3 novelty boundary |\n",
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
