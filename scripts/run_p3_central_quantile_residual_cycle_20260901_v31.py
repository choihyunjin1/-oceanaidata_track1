"""Exactly-once P3 central conditional-quantile residual cycle."""

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
from sklearn.linear_model import QuantileRegressor

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

import run_p3_causal_multichannel_rocket_residual_cycle_20260901_v26 as v26  # noqa: E402
import run_p3_cross_wavelet_phase_residual_cycle_20260901_v28 as v28  # noqa: E402
import run_p3_path_signature_residual_cycle_20260901_v23 as v23  # noqa: E402

EXPERIMENT_ID = "p3_central_quantile_residual_cycle_20260901_v31"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = ROOT / "artifacts/p3/sequences_all20_v1/train_values.npy"
STATIONS = ROOT / "artifacts/p3/sequences_all20_v1/train_station.npy"
QUANTILES = (0.25, 0.75)
L1_ALPHA = 0.10
BLEND = 0.10
CASE_FEATURE_COUNT = 108
ROW_FEATURE_COUNT = 117
SPEC = v26.Spec("P3_1_Q25_Q75_MID_ALPHA010_ADD10", L1_ALPHA)


class ContractError(RuntimeError):
    """Raised when the sealed v31 contract differs."""


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
        "schema": config["schema_version"]
        == "p3.central_quantile_residual.config.v31",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_DISTRIBUTIONAL_TARGET_AXIS",
        "quantiles": tuple(config["model"]["quantiles"]) == QUANTILES,
        "alpha": float(config["model"]["l1_alpha"]) == L1_ALPHA,
        "blend": float(config["model"]["additive_residual_weight"]) == BLEND,
        "candidate": config["model"]["candidate_name"] == SPEC.name,
        "case_features": config["features"]["case_feature_count"]
        == CASE_FEATURE_COUNT,
        "row_features": config["features"]["row_feature_count"]
        == ROW_FEATURE_COUNT,
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "no_posthoc": not config["duplication_audit"][
            "posthoc_v27_v28_v29_v30_adjustment"
        ],
    }
    if not all(checks.values()):
        raise ContractError(f"v31 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def _slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=np.float64)
    x -= x.mean()
    denominator = float(x @ x)
    return float(x @ (values - values.mean()) / denominator)


def case_statistics(sequence: np.ndarray) -> np.ndarray:
    path = v26.transformed_path(sequence)
    output: list[float] = []
    for column in range(12):
        values = path[:, column]
        last24 = values[-24:]
        last72 = values[-72:]
        output.extend(
            (
                values[-1],
                float(last24.mean()),
                float(last72.mean()),
                float(last72.std()),
                _slope(last24),
                _slope(last72),
                *np.quantile(values, (0.10, 0.50, 0.90)),
            )
        )
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (CASE_FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("case-statistic feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([base * (index + 1) + index for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = case_statistics(sequence)
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
        "schema_version": "p3.central_quantile_residual.preflight.v31",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 1,
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
    features = np.empty((len(cases), CASE_FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = case_statistics(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def row_features(cases: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    stations = pd.get_dummies(cases["station"], dtype=float).reindex(
        columns=["G-ORS", "I-ORS", "S-ORS"], fill_value=0.0
    )
    lead_eye = np.eye(6, dtype=np.float64)
    matrix = np.column_stack(
        [
            np.repeat(features, 6, axis=0),
            np.repeat(stations.to_numpy(float), 6, axis=0),
            np.tile(lead_eye, (len(cases), 1)),
        ]
    )
    if matrix.shape != (len(cases) * 6, ROW_FEATURE_COUNT):
        raise ContractError("row feature shape differs")
    return matrix


def crossfit(
    cases: pd.DataFrame,
    features: np.ndarray,
    targets: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix = row_features(cases, features)
    residual = (targets - reference).reshape(-1)
    prediction = np.empty_like(targets)
    receipts: list[dict[str, Any]] = []
    for block in v23.BLOCKS:
        valid_case = cases["block"].eq(block).to_numpy()
        valid_cases = np.flatnonzero(valid_case)
        train_cases = v23.purged_train_indices(cases, valid_case)
        train_rows = np.concatenate(
            [np.arange(index * 6, index * 6 + 6) for index in train_cases]
        )
        valid_rows = np.concatenate(
            [np.arange(index * 6, index * 6 + 6) for index in valid_cases]
        )
        center = np.median(matrix[train_rows], axis=0)
        q25, q75 = np.quantile(matrix[train_rows], (0.25, 0.75), axis=0)
        scale = np.where(q75 - q25 > 1e-8, q75 - q25, 1.0)
        train_x = np.clip((matrix[train_rows] - center) / scale, -8.0, 8.0)
        valid_x = np.clip((matrix[valid_rows] - center) / scale, -8.0, 8.0)
        quantile_predictions: list[np.ndarray] = []
        for quantile in QUANTILES:
            model = QuantileRegressor(
                quantile=quantile,
                alpha=L1_ALPHA,
                fit_intercept=True,
                solver="highs",
            )
            model.fit(train_x, residual[train_rows])
            current = model.predict(valid_x)
            if not np.isfinite(current).all():
                raise ContractError("quantile prediction is non-finite")
            quantile_predictions.append(current)
            receipts.append(
                {
                    "candidate": SPEC.name,
                    "block": block,
                    "quantile": quantile,
                    "train_rows": len(train_rows),
                    "valid_rows": len(valid_rows),
                    "row_deletion": 0,
                }
            )
        midpoint = 0.5 * (quantile_predictions[0] + quantile_predictions[1])
        prediction[valid_case] = (
            reference[valid_case] + BLEND * midpoint.reshape(len(valid_cases), 6)
        )
    if not np.isfinite(prediction).all():
        raise ContractError("crossfit prediction is incomplete")
    return prediction, receipts


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, profile = v23.case_surface()
    features, feature_receipt = surface_features(cases)
    prediction, receipts = crossfit(cases, features, targets, reference)
    frame = v23.long_frame(cases, targets, reference)
    scored = v28.score(frame, prediction, SPEC)
    result = {
        "schema_version": "p3.central_quantile_residual.result.v31",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE"
        if scored["decision"] != "NO_GO"
        else "NO_GO_CENTRAL_QUANTILE_CANDIDATE",
        "surface_claim": config["validation"]["surface"],
        "reference": config["reference"],
        "duplication_audit": config["duplication_audit"],
        "primary_sources": config["primary_sources"],
        "feature_receipt": feature_receipt,
        "candidate": scored,
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
            "candidate_count": 1,
            "result_based_tuning": False,
            "outer_result_parameter_changes": 0,
            "row_deletion": 0,
        },
    }
    arrays = {
        "truth": targets,
        "uniform": reference,
        "candidate_1": prediction,
        "anchor_id": cases["anchor_id"].to_numpy(np.int32),
        "lead_h": np.asarray(v23.LEADS, dtype=np.int16),
        "block": cases["block"].to_numpy(dtype="U5"),
        "station": cases["station"].to_numpy(dtype="U5"),
        "episode": cases["episode_id"].to_numpy(dtype="U32"),
    }
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    item = result["candidate"]
    metric, points = item["rmse_m"], item["expected_points"]
    return (
        "# P3 central conditional-quantile residual cycle v31\n\n## 결론\n\n"
        f"- overall decision: **{result['decision']}**.\n"
        "- Fixed shapelets and half-expectiles were rejected as semantic duplicates before fit.\n"
        f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; "
        f"raw {points['raw_gain']:+.6f} points; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; "
        f"episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.\n"
        "- The 182-case surface is EXPLORATORY_ONLY. Official/hidden/CSV/upload access is zero.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT.exists() or REPORT.exists() or LOCK.exists():
        raise ContractError("v31 exactly-once namespace already exists")
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
        b"# Gap matrix\n\n| Audited axis | Verdict |\n|---|---|\n| fixed shapelets | semantic duplicate of ROCKET/analog |\n| half-expectile | direct squared-location duplicate |\n| central conditional quantiles | nonduplicate target formulation; executed as v31 |\n",
    )
    write_new(
        REPORT / "claim-source-ledger.md",
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Pinball loss estimates conditional quantiles | Koenker and Bassett, 1978, DOI:10.2307/1913643 | sealed distributional target |\n| No corresponding P3 implementation exists | repository semantic audit before sealing | novelty gate |\n",
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
