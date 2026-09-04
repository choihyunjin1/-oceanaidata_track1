"""Exactly-once P3 Bayesian online changepoint posterior residual cycle."""

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
import run_p3_cross_wavelet_phase_residual_cycle_20260901_v28 as v28  # noqa: E402
import run_p3_path_signature_residual_cycle_20260901_v23 as v23  # noqa: E402

EXPERIMENT_ID = "p3_bocpd_regime_age_residual_cycle_20260901_v32"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = ROOT / "artifacts/p3/sequences_all20_v1/train_values.npy"
STATIONS = ROOT / "artifacts/p3/sequences_all20_v1/train_station.npy"
HAZARD = 1.0 / 72.0
RUN_CAP = 96
FEATURE_COUNT = 96
SPECS = (
    v26.Spec("P3_1_BOCPD96_RIDGE512_ADD10", 512.0),
    v26.Spec("P3_2_BOCPD96_RIDGE2048_ADD10", 2048.0),
)


class ContractError(RuntimeError):
    """Raised when the sealed v32 contract differs."""


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
        "schema": config["schema_version"] == "p3.bocpd_regime_age_residual.config.v32",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_CAUSAL_CHANGEPOINT_POSTERIOR_AXIS",
        "hazard": float(encoder["constant_hazard"]) == HAZARD,
        "cap": encoder["maximum_run_length"] == RUN_CAP,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple(
            (item["name"], float(item["ridge_alpha"]))
            for item in config["model"]["candidates"]
        )
        == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(
            float(item["additive_residual_weight"]) == 0.10
            for item in config["model"]["candidates"]
        ),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v32 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def normal_density(value: float, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(value - mean) / variance) / np.sqrt(
        2.0 * np.pi * variance
    )


def bocpd_summary(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    q25, q75 = np.quantile(values, (0.25, 0.75))
    normalized = (values - median) / max(float(q75 - q25), 1e-8)
    probabilities = np.asarray([1.0], dtype=np.float64)
    means = np.asarray([0.0], dtype=np.float64)
    precisions = np.asarray([1.0], dtype=np.float64)
    changepoint_history: list[float] = []
    for value in normalized:
        predictive = normal_density(value, means, 1.0 + 1.0 / precisions)
        prior_predictive = float(normal_density(value, np.asarray([0.0]), np.asarray([2.0]))[0])
        size = min(len(probabilities) + 1, RUN_CAP + 1)
        updated = np.zeros(size, dtype=np.float64)
        updated[0] = HAZARD * prior_predictive * probabilities.sum()
        growth_count = min(len(probabilities), RUN_CAP)
        updated[1 : growth_count + 1] = (
            probabilities[:growth_count] * (1.0 - HAZARD) * predictive[:growth_count]
        )
        total = float(updated.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ContractError("BOCPD posterior normalization failed")
        probabilities = updated / total
        new_means = np.empty(size, dtype=np.float64)
        new_precisions = np.empty(size, dtype=np.float64)
        new_means[0] = value / 2.0
        new_precisions[0] = 2.0
        if growth_count:
            new_means[1 : growth_count + 1] = (
                precisions[:growth_count] * means[:growth_count] + value
            ) / (precisions[:growth_count] + 1.0)
            new_precisions[1 : growth_count + 1] = precisions[:growth_count] + 1.0
        means, precisions = new_means, new_precisions
        changepoint_history.append(float(probabilities[0]))
    run_length = np.arange(len(probabilities), dtype=np.float64)
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive))) / np.log(RUN_CAP + 1.0)
    output = np.asarray(
        [
            probabilities[0],
            float(probabilities @ run_length) / RUN_CAP,
            entropy,
            float(probabilities.max()),
            float(probabilities[: min(7, len(probabilities))].sum()),
            float(probabilities[48:].sum()) if len(probabilities) > 48 else 0.0,
            float(np.mean(changepoint_history[-12:])),
            float(np.max(changepoint_history[-24:])),
        ],
        dtype=np.float64,
    )
    if output.shape != (8,) or not np.isfinite(output).all():
        raise ContractError("BOCPD summary contract differs")
    return output


def bocpd_features(sequence: np.ndarray) -> np.ndarray:
    path = v26.transformed_path(sequence)
    features = np.concatenate([bocpd_summary(path[:, column]) for column in range(12)])
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("BOCPD feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = bocpd_features(sequence)
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
        raise ContractError("exactly-once namespace is consumed")
    payload = {
        "schema_version": "p3.bocpd_regime_age_residual.preflight.v32",
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
        features[position] = bocpd_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, profile = v23.case_surface()
    features, feature_receipt = surface_features(cases)
    original_specs = v28.SPECS
    v28.SPECS = SPECS
    try:
        predictions, receipts = v28.crossfit(cases, features, targets, reference)
        frame = v23.long_frame(cases, targets, reference)
        scored = [v28.score(frame, predictions[spec.name], spec) for spec in SPECS]
    finally:
        v28.SPECS = original_specs
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {
        "schema_version": "p3.bocpd_regime_age_residual.result.v32",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE"
        if passing
        else "NO_GO_ALL_BOCPD_CANDIDATES",
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
        "# P3 BOCPD regime-age residual cycle v32",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- Lead-increment and shape-projection ideas were rejected before fit as closed basis/smoothing semantics.",
        "- v32 uses fixed causal run-length posterior summaries; the 182-case surface remains EXPLORATORY_ONLY.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(
            f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; "
            f"raw {points['raw_gain']:+.6f} points; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; "
            f"worst block {item['worst_block_delta_m']:+.9f}m; worst lead {item['worst_lead_delta_m']:+.9f}m; "
            f"worst station-lead {item['worst_station_lead_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; "
            f"block-station CI90 {item['block_station_bootstrap']['ci90_m']}."
        )
    lines.extend(
        [
            "",
            "Official test/sample/submission/hidden access, CSV materialization, and upload were all zero. No row was deleted and no outer result changed the encoder, Ridge strengths, or blend.",
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
        raise ContractError("v32 exactly-once namespace already exists")
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
        b"# Gap matrix\n\n| Audited axis | Verdict |\n|---|---|\n| lead-increment target | semantic duplicate of hierarchical/joint basis |\n| lead-shape projection | no new signal; smoothing overlap |\n| BOCPD run-length posterior | nonduplicate; executed as v32 |\n",
    )
    write_new(
        REPORT / "claim-source-ledger.md",
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| BOCPD maintains a causal posterior over run length | Adams and MacKay, 2007, arXiv:0710.3742 | fixed regime-age representation |\n| No corresponding P3 implementation exists | repository semantic audit before sealing | novelty gate |\n",
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
