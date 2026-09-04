"""Exactly-once P3 deterministic level-2 path-signature residual cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

import run_p3_kma_wind_work_residual_axis_cycle_20260901_v20 as v20  # noqa: E402
from run_p3_parallel_candidate_cycle_20260831_v4 import load_historical, rmse  # noqa: E402
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    POINTS_PER_RMSE_M,
    bootstrap,
)

EXPERIMENT_ID = "p3_path_signature_residual_cycle_20260901_v23"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SEQUENCES = ROOT / "artifacts/p3/sequences_all20_v1/train_values.npy"
STATIONS = ROOT / "artifacts/p3/sequences_all20_v1/train_station.npy"
ANCHORS = ROOT / "artifacts/p3/features_all20_v1/train_anchors.parquet"
BLOCKS = ("01_02", "03_04", "05_06", "07_08", "09_10", "11_12")
LEADS = (3, 6, 9, 12, 18, 24)
CHANNELS = (0, 1, 2, 4, 9)
CHANNEL_NAMES = ("hs", "tp", "hmax", "wspd", "caph")
PHYSICAL_CENTER = np.asarray((0.0, 0.0, 0.0, 0.0, 1013.0), dtype=np.float64)
PHYSICAL_SCALE = np.asarray((5.0, 20.0, 10.0, 30.0, 50.0), dtype=np.float64)
FEATURE_COUNT = 140
PURGE_HOURS = 78.0
WINSOR = (0.025, 0.975)
BLEND = 0.15
TRANSPORT_PENALTY_POINTS = 0.04958605409228893
OFFICIAL_CHAMPION_POINTS = 24.203599
BOOTSTRAP_REPLICATES = 5000


class ContractError(RuntimeError):
    """Raised when the sealed v23 contract differs."""


@dataclass(frozen=True)
class Spec:
    name: str
    alpha: float


SPECS = (
    Spec("P3_1_PATHSIG_L2_RIDGE256_ADD15", 256.0),
    Spec("P3_2_PATHSIG_L2_RIDGE1024_ADD15", 1024.0),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "schema": config["schema_version"] == "p3.path_signature_residual.config.v23",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "nonduplicate": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_REPRESENTATION_AXIS",
        "level": config["path"]["signature_level"] == 2,
        "features": config["path"]["feature_count"] == FEATURE_COUNT,
        "channels": tuple(config["path"]["selected_channels"]) == CHANNEL_NAMES,
        "candidates": tuple(item["name"] for item in config["model"]["candidates"])
        == tuple(item.name for item in SPECS),
        "alphas": tuple(float(item["ridge_alpha"]) for item in config["model"]["candidates"])
        == tuple(item.alpha for item in SPECS),
        "blend": all(
            float(item["additive_residual_weight"]) == BLEND
            for item in config["model"]["candidates"]
        ),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
    }
    if not all(checks.values()):
        raise ContractError(f"v23 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def fill_past_path(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(sequence[:, CHANNELS], dtype=np.float64)
    if raw.shape != (289, len(CHANNELS)):
        raise ContractError("path shape differs")
    observed = np.isfinite(raw)
    filled = np.empty_like(raw)
    time_index = np.arange(len(raw), dtype=np.float64)
    for channel in range(raw.shape[1]):
        finite = observed[:, channel]
        if finite.any():
            filled[:, channel] = np.interp(
                time_index, time_index[finite], raw[finite, channel]
            )
        else:
            filled[:, channel] = 0.0
    scaled = (filled - PHYSICAL_CENTER) / PHYSICAL_SCALE
    time_path = np.linspace(0.0, 1.0, num=len(raw), dtype=np.float64)[:, None]
    path = np.concatenate([time_path, scaled, observed.astype(np.float64)], axis=1)
    if not np.isfinite(path).all():
        raise ContractError("filled path contains non-finite values")
    return path, scaled[-1]


def level2_signature(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ContractError("signature input contract differs")
    level1 = np.zeros(values.shape[1], dtype=np.float64)
    level2 = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    for increment in np.diff(values, axis=0):
        level2 += np.outer(level1, increment) + 0.5 * np.outer(increment, increment)
        level1 += increment
    return level1, level2


def path_signature_features(sequence: np.ndarray, station_code: int) -> np.ndarray:
    if station_code not in (0, 1, 2):
        raise ContractError("station code differs")
    path, endpoint = fill_past_path(sequence)
    level1, level2 = level2_signature(path)
    station = np.eye(3, dtype=np.float64)[station_code]
    feature = np.concatenate([endpoint, level1, level2.reshape(-1), station])
    if feature.shape != (FEATURE_COUNT,) or not np.isfinite(feature).all():
        raise ContractError("path-signature feature contract differs")
    return feature


def synthetic_signature_receipt() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, num=289, dtype=np.float64)
    sequence = np.empty((289, 10), dtype=np.float64)
    for channel in range(10):
        sequence[:, channel] = base * (channel + 1) + channel
    sequence[1::2, :4] = np.nan
    feature = path_signature_features(sequence, 2).astype("<f8", copy=False)
    return {
        "feature_count": int(len(feature)),
        "feature_sha256": hashlib.sha256(feature.tobytes(order="C")).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    sequence_meta = np.load(SEQUENCES, mmap_mode="r")
    station_meta = np.load(STATIONS, mmap_mode="r")
    if sequence_meta.shape != (24360, 289, 10) or station_meta.shape != (24360,):
        raise ContractError("sequence cache shape differs")
    if ARTIFACT_DIR.exists() or LOCK.exists():
        raise ContractError("exactly-once namespace is already consumed")
    payload = {
        "schema_version": "p3.path_signature_residual.preflight.v23",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "sequence_shape": list(sequence_meta.shape),
        "station_shape": list(station_meta.shape),
        "candidate_count": len(SPECS),
        "maximum_model_fits": len(SPECS) * len(BLOCKS),
        "signature": synthetic_signature_receipt(),
        "official_access": 0,
        "csv_materializations": 0,
        "uploads": 0,
        "config_status": config["status"],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def case_surface() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    frame, profile = load_historical()
    records: list[dict[str, Any]] = []
    truth: list[np.ndarray] = []
    reference: list[np.ndarray] = []
    for key, group in frame.groupby(["fold", "anchor_id", "station"], observed=True, sort=True):
        ordered = group.sort_values("lead_h", kind="stable")
        leads = tuple(ordered["lead_h"].astype(int))
        if leads != LEADS:
            raise ContractError("case lacks exact six-lead order")
        records.append(
            {
                "fold": str(key[0]),
                "anchor_id": int(key[1]),
                "station": str(key[2]),
                "anchor_time": ordered["anchor_time"].iloc[0],
                "block": str(ordered["block"].iloc[0]),
                "episode_id": str(ordered["episode_id"].iloc[0]),
            }
        )
        truth.append(ordered["target_hs"].to_numpy(dtype=np.float64))
        reference.append(ordered["reference"].to_numpy(dtype=np.float64))
    cases = pd.DataFrame(records)
    targets = np.vstack(truth)
    baseline = np.vstack(reference)
    if len(cases) != 182 or targets.shape != (182, 6) or baseline.shape != (182, 6):
        raise ContractError("182-by-six historical contract differs")
    if set(cases["block"]) != set(BLOCKS):
        raise ContractError("bimonth block contract differs")
    return cases, targets, baseline, profile


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(SEQUENCES, mmap_mode="r")
    station_codes = np.load(STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        expected = station_map[str(row.station)]
        actual = int(station_codes[anchor_id])
        if actual != expected:
            raise ContractError("sequence station key differs")
        features[position] = path_signature_features(sequences[anchor_id], actual)
    receipt = {
        "rows": int(len(features)),
        "columns": int(features.shape[1]),
        "finite": bool(np.isfinite(features).all()),
        "matrix_sha256": hashlib.sha256(
            features.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
    }
    return features, receipt


def purged_train_indices(cases: pd.DataFrame, valid: np.ndarray) -> np.ndarray:
    train_positions = np.flatnonzero(~valid)
    keep = np.ones(len(train_positions), dtype=bool)
    valid_cases = cases.loc[valid]
    for station, group in valid_cases.groupby("station", observed=True):
        local_positions = np.flatnonzero(
            cases.iloc[train_positions]["station"].eq(station).to_numpy()
        )
        if not len(local_positions):
            continue
        candidate_index = train_positions[local_positions]
        train_ns = pd.DatetimeIndex(cases.iloc[candidate_index]["anchor_time"]).as_unit("ns").asi8
        valid_ns = pd.DatetimeIndex(group["anchor_time"]).as_unit("ns").asi8
        distance = np.min(np.abs(train_ns[:, None] - valid_ns[None, :]), axis=1) / 3.6e12
        keep[local_positions] = distance > PURGE_HOURS
    result = train_positions[keep]
    if len(result) < 50:
        raise ContractError("purged outer train has too few cases")
    return result


def fit_predict(
    features: np.ndarray,
    target_residual: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    spec: Spec,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = features[train]
    center = np.median(x_train, axis=0)
    q25, q75 = np.quantile(x_train, (0.25, 0.75), axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    train_z = np.clip((x_train - center) / scale, -8.0, 8.0)
    valid_z = np.clip((features[valid] - center) / scale, -8.0, 8.0)
    low, high = np.quantile(target_residual[train], WINSOR, axis=0)
    y_train = np.clip(target_residual[train], low, high)
    model = Ridge(alpha=spec.alpha, fit_intercept=True, solver="cholesky")
    model.fit(train_z, y_train)
    prediction = np.asarray(model.predict(valid_z), dtype=np.float64)
    if prediction.shape != (int(valid.sum()), len(LEADS)) or not np.isfinite(prediction).all():
        raise ContractError("multi-output residual prediction differs")
    return prediction, {
        "candidate": spec.name,
        "ridge_alpha": spec.alpha,
        "train_cases": int(len(train)),
        "valid_cases": int(valid.sum()),
        "target_winsor": list(WINSOR),
        "row_deletion": 0,
        "fit_count": 1,
        "coefficient_l2": float(np.linalg.norm(model.coef_)),
    }


def crossfit(
    cases: pd.DataFrame,
    features: np.ndarray,
    targets: np.ndarray,
    reference: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    target_residual = targets - reference
    outputs = {spec.name: reference.copy() for spec in SPECS}
    receipts: list[dict[str, Any]] = []
    for block in BLOCKS:
        valid = cases["block"].eq(block).to_numpy()
        train = purged_train_indices(cases, valid)
        for spec in SPECS:
            residual, receipt = fit_predict(features, target_residual, train, valid, spec)
            outputs[spec.name][valid] = np.clip(
                reference[valid] + BLEND * residual, 0.0, 30.0
            )
            receipt["block"] = block
            receipt["additive_residual_weight"] = BLEND
            receipts.append(receipt)
    if len(receipts) != 12 or sum(item["fit_count"] for item in receipts) != 12:
        raise ContractError("fit budget differs")
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise ContractError("crossfit output is non-finite")
    return outputs, receipts


def long_frame(cases: pd.DataFrame, targets: np.ndarray, reference: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for case_position, case in enumerate(cases.itertuples(index=False)):
        for lead_position, lead in enumerate(LEADS):
            records.append(
                {
                    "case_position": case_position,
                    "lead_h": lead,
                    "target_hs": targets[case_position, lead_position],
                    "reference": reference[case_position, lead_position],
                    "block": case.block,
                    "station": case.station,
                    "episode_id": case.episode_id,
                }
            )
    return pd.DataFrame(records)


def score_candidate(frame: pd.DataFrame, prediction: np.ndarray, spec: Spec) -> dict[str, Any]:
    flat = prediction.reshape(-1)
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    before = rmse(truth, reference)
    after = rmse(truth, flat)
    delta = after - before
    by_block = v20.group_deltas(frame, flat, reference, ["block"])
    station = v20.group_deltas(frame, flat, reference, ["station"])
    lead = v20.group_deltas(frame, flat, reference, ["lead_h"])
    station_lead = v20.group_deltas(frame, flat, reference, ["station", "lead_h"])
    improved_blocks = sum(item["delta_rmse_m"] < 0 for item in by_block.values())
    worst_station_lead = max(item["delta_rmse_m"] for item in station_lead.values())
    worst_block = max(item["delta_rmse_m"] for item in by_block.values())
    offset = SPECS.index(spec) * 100
    episode_ci = bootstrap(frame, flat, ("episode_id",), 20260931 + offset)
    group_ci = bootstrap(frame, flat, ("block", "station"), 20260932 + offset)
    stable_checks = {
        "delta_rmse_negative": delta < 0,
        "minimum_four_improved_blocks": improved_blocks >= 4,
        "episode_ci90_upper_below_zero": episode_ci["ci90_m"][1] < 0,
        "block_station_ci90_upper_below_zero": group_ci["ci90_m"][1] < 0,
        "worst_station_lead_at_most_0p01m": worst_station_lead <= 0.01,
        "finite_predictions": bool(np.isfinite(flat).all()),
    }
    high_risk_checks = {
        "delta_rmse_at_most_minus_0p005m": delta <= -0.005,
        "worst_station_lead_at_most_0p02m": worst_station_lead <= 0.02,
        "finite_predictions": stable_checks["finite_predictions"],
    }
    stable = all(stable_checks.values())
    high_risk = (not stable) and all(high_risk_checks.values())
    raw_points = -delta * POINTS_PER_RMSE_M
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
            "raw_gain": raw_points,
            "transport_penalty": TRANSPORT_PENALTY_POINTS,
            "transport_adjusted_gain": raw_points - TRANSPORT_PENALTY_POINTS,
            "nominal_official_score": OFFICIAL_CHAMPION_POINTS + raw_points,
        },
        "improved_blocks": int(improved_blocks),
        "by_block": by_block,
        "station": station,
        "lead": lead,
        "station_lead": station_lead,
        "worst_block_delta_m": worst_block,
        "worst_station_lead_delta_m": worst_station_lead,
        "episode_bootstrap": episode_ci,
        "block_station_bootstrap": group_ci,
        "stable_checks": stable_checks,
        "high_risk_checks": high_risk_checks,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 deterministic path-signature residual cycle v23",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- This is EXPLORATORY_ONLY on the repeatedly exposed 182-case surface; it is not a Public transport guarantee.",
    ]
    for item in result["candidates"]:
        metric = item["rmse_m"]
        points = item["expected_points"]
        lines.append(
            f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; raw {points['raw_gain']:+.6f} points; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m."
        )
        lines.append(
            f"  - episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}; worst station-lead {item['worst_station_lead_delta_m']:+.9f}m."
        )
    lines.extend(
        [
            "",
            "## Method and research basis",
            "",
            "The new representation is a deterministic level-2 path signature: ordered iterated integrals over time, five past-only physical channels, and their observation masks. Kiraly and Oberhauser describe signature features as ordered sample cross-moments for sequential data ([JMLR 2019](https://www.jmlr.org/papers/v20/16-314.html)). Neural CDE work confirms the broader continuous-path view for partially observed multivariate series, but this bounded cycle deliberately uses no deep network ([Kidger et al., NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html)).",
            "",
            "No official test/sample/submission/hidden value was read. No CSV was materialized and no upload occurred. Target winsorization was fixed on each outer-training fold, and no row was deleted.",
        ]
    )
    return "\n".join(lines) + "\n"


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    cases, targets, reference, profile = case_surface()
    features, feature_receipt = surface_features(cases)
    predictions, fit_receipts = crossfit(cases, features, targets, reference)
    frame = long_frame(cases, targets, reference)
    scored = [score_candidate(frame, predictions[spec.name], spec) for spec in SPECS]
    passing = [item for item in scored if item["decision"] != "NO_GO"]
    result = {
        "schema_version": "p3.path_signature_residual.result.v23",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE" if passing else "NO_GO_ALL_PATH_SIGNATURES",
        "surface_claim": config["validation"]["surface"],
        "reference": config["reference"],
        "duplication_audit": config["duplication_audit"],
        "path_contract": config["path"],
        "candidates": scored,
        "fit_receipts": fit_receipts,
        "fit_count": 12,
        "feature_receipt": feature_receipt,
        "data_profile": profile,
        "data_access": {
            "historical_target_rows": int(len(frame)),
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
            "candidate_count": len(SPECS),
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
        "anchor_id": cases["anchor_id"].to_numpy(dtype=np.int32),
        "lead_h": np.asarray(LEADS, dtype=np.int16),
        "block": cases["block"].to_numpy(dtype="U5"),
        "station": cases["station"].to_numpy(dtype="U5"),
        "episode": cases["episode_id"].to_numpy(dtype="U32"),
    }
    return result, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode("utf-8"), end="")
        return 0
    if ARTIFACT_DIR.exists() or REPORT_DIR.exists() or LOCK.exists():
        raise ContractError("v23 exactly-once namespace already exists")
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
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    result, arrays = execute(config)
    array_path = ARTIFACT_DIR / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {
        "runner_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(CONFIG),
        "evaluation_arrays_sha256": sha256(array_path),
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "input_sha256": config["inputs"],
    }
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, render_report(result).encode("utf-8"))
    write_new(
        REPORT_DIR / "result.json",
        canonical(result),
    )
    write_new(
        REPORT_DIR / "run-manifest.json",
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "result_sha256": sha256(result_path),
                "arrays_sha256": sha256(array_path),
                "report_sha256": sha256(report_path),
                "fit_count": result["fit_count"],
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
                "fit_count": result["fit_count"],
                "official_access": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
