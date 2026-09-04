"""Evaluate two sealed target-free selective robust fallbacks over uniform KMA 0.425."""

from __future__ import annotations

import argparse
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

import run_p3_kma_wind_work_residual_axis_cycle_20260901_v20 as v20  # noqa: E402
from run_p3_kma_continuous_energy_affine_cycle_20260831_v15b import block_order  # noqa: E402
from run_p3_parallel_candidate_cycle_20260831_v4 import purge_training_cases, rmse  # noqa: E402
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    POINTS_PER_RMSE_M,
    bootstrap,
    canonical,
    sha256,
    write_new,
)

EXPERIMENT_ID = "p3_uniform_selective_robust_fallback_cycle_20260901_v21"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
FEATURES = ("hs_std_24h", "hmax_hs_ratio_current")
ACTIVE_LEADS = (18, 24)
PURGE_HOURS = 78
SPREAD_QUANTILE = 0.90
EXTREME_QUANTILE = 0.90
WINSOR_LOW = 0.01
WINSOR_HIGH = 0.99
TRANSPORT_PENALTY_POINTS = 0.04958605409228893
OFFICIAL_CHAMPION_POINTS = 24.203599
CANDIDATES = (
    "P3_1_EXTREME_DISAGREEMENT_PERSISTENCE_QUARTER_FALLBACK",
    "P3_2_EXTREME_DISAGREEMENT_THREE_COMPONENT_MEDIAN",
)


class ContractError(RuntimeError):
    """Raised when the sealed v21 contract is violated."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "schema": config["schema_version"] == "p3.uniform_selective_robust_fallback.config.v21",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "candidates": tuple(item["name"] for item in config["candidates"]) == CANDIDATES,
        "thresholds": "0.90" in config["gate"]["fixed_rule"],
        "target_free": config["gate"]["target_rows_used_to_fit_gate"] == 0,
        "no_search": config["gate"]["threshold_search"] is False,
        "no_retry": config["decision"]["no_result_based_retry"] is True,
        "non_duplicate": config["duplication_audit"]["verdict"] == "NON_DUPLICATE",
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
    }
    if not all(checks.values()):
        raise ContractError(f"v21 config contract failed: {checks}")
    return config


def load_features() -> pd.DataFrame:
    frame = pd.read_parquet(v20.TRAIN_FEATURES, columns=["anchor_id", "station", *FEATURES])
    if frame.duplicated(["anchor_id", "station"]).any():
        raise ContractError("feature keys are duplicated")
    return frame


def attach_features(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    output = frame.merge(features, on=["anchor_id", "station"], how="left", validate="many_to_one")
    if len(output) != len(frame):
        raise ContractError("feature join changed row count")
    return output


def component_spread(frame: pd.DataFrame) -> np.ndarray:
    values = np.column_stack(
        [
            frame["base"].to_numpy(dtype=np.float64),
            frame["reference"].to_numpy(dtype=np.float64),
            frame["current_hs"].to_numpy(dtype=np.float64),
        ]
    )
    return np.max(values, axis=1) - np.min(values, axis=1)


def finite_quantile(values: np.ndarray, quantile: float, fallback: float = np.inf) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, quantile)) if len(finite) else fallback


def winsor_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return -np.inf, np.inf
    return float(np.quantile(finite, WINSOR_LOW)), float(np.quantile(finite, WINSOR_HIGH))


def ecdf_rank(prefix: np.ndarray, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rank = np.zeros(len(values), dtype=np.float64)
    finite_prefix = np.sort(prefix[np.isfinite(prefix)])
    if len(finite_prefix):
        rank[valid] = np.searchsorted(finite_prefix, values[valid], side="right") / len(
            finite_prefix
        )
    return np.clip(rank, 0.0, 1.0)


def fixed_sparse_gate(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[np.ndarray, dict[str, Any]]:
    train_active = train.loc[train["lead_h"].isin(ACTIVE_LEADS)].copy()
    spread_train = component_spread(train_active)
    spread_valid = component_spread(valid)
    spread_threshold = finite_quantile(spread_train, SPREAD_QUANTILE)

    train_hs = train_active["current_hs"].to_numpy(dtype=np.float64)
    hs_median = float(np.nanmedian(train_hs))
    physical_train = [np.abs(train_hs - hs_median)]
    physical_valid = [np.abs(valid["current_hs"].to_numpy(dtype=np.float64) - hs_median)]
    winsor_receipts: dict[str, list[float]] = {}
    for feature in FEATURES:
        train_values = train_active[feature].to_numpy(dtype=np.float64)
        valid_values = valid[feature].to_numpy(dtype=np.float64)
        low, high = winsor_limits(train_values)
        physical_train.append(np.clip(train_values, low, high))
        physical_valid.append(np.clip(valid_values, low, high))
        winsor_receipts[feature] = [low, high]

    ranks = []
    for train_values, valid_values in zip(physical_train, physical_valid, strict=True):
        valid_mask = np.isfinite(valid_values)
        ranks.append(ecdf_rank(train_values, valid_values, valid_mask))
    extreme_rank = np.max(np.column_stack(ranks), axis=1)
    active = valid["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    gate = active & (spread_valid >= spread_threshold) & (extreme_rank >= EXTREME_QUANTILE)
    receipt = {
        "train_active_rows": int(len(train_active)),
        "validation_rows": int(len(valid)),
        "spread_threshold_q90": spread_threshold,
        "winsor_q01_q99": winsor_receipts,
        "gated_rows": int(gate.sum()),
        "target_rows_read_before_gate_fixed": 0,
        "feature_only_quantile_fits": 1,
    }
    return gate, receipt


def candidate_predictions(
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    outputs = {name: frame["reference"].to_numpy(dtype=np.float64).copy() for name in CANDIDATES}
    receipts: list[dict[str, Any]] = []
    for block in block_order(frame):
        outer = frame["block"].eq(block).to_numpy()
        valid = frame.loc[outer]
        train = purge_training_cases(frame.loc[~outer], valid)
        gate, receipt = fixed_sparse_gate(train, valid)
        uniform = valid["reference"].to_numpy(dtype=np.float64)
        persistence = valid["current_hs"].to_numpy(dtype=np.float64)
        base = valid["base"].to_numpy(dtype=np.float64)
        first = uniform.copy()
        first[gate] = 0.75 * uniform[gate] + 0.25 * persistence[gate]
        second = uniform.copy()
        second[gate] = np.median(
            np.column_stack([base[gate], uniform[gate], persistence[gate]]), axis=1
        )
        outputs[CANDIDATES[0]][outer] = np.clip(first, 0.0, 30.0)
        outputs[CANDIDATES[1]][outer] = np.clip(second, 0.0, 30.0)
        receipt["block"] = str(block)
        receipts.append(receipt)
    if not all(np.isfinite(prediction).all() for prediction in outputs.values()):
        raise ContractError("candidate predictions are non-finite")
    return outputs, receipts


def score_candidate(frame: pd.DataFrame, prediction: np.ndarray, name: str) -> dict[str, Any]:
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    uniform = frame["reference"].to_numpy(dtype=np.float64)
    candidate_rmse = rmse(truth, prediction)
    uniform_rmse = rmse(truth, uniform)
    delta = candidate_rmse - uniform_rmse
    by_block = v20.group_deltas(frame, prediction, uniform, ["block"])
    station = v20.group_deltas(frame, prediction, uniform, ["station"])
    lead = v20.group_deltas(frame, prediction, uniform, ["lead_h"])
    station_lead = v20.group_deltas(frame, prediction, uniform, ["station", "lead_h"])
    improved = sum(item["delta_rmse_m"] < 0 for item in by_block.values())
    worst = max(item["delta_rmse_m"] for item in station_lead.values())
    episode_ci = bootstrap(
        frame, prediction, ("episode_id",), 20260921 + CANDIDATES.index(name) * 10
    )
    group_ci = bootstrap(
        frame, prediction, ("block", "station"), 20260922 + CANDIDATES.index(name) * 10
    )
    changed = np.abs(prediction - uniform) > 1e-12
    stable_checks = {
        "delta_rmse_negative": delta < 0,
        "minimum_four_improved_blocks": improved >= 4,
        "episode_ci90_upper_below_zero": episode_ci["ci90_m"][1] < 0,
        "block_station_ci90_upper_below_zero": group_ci["ci90_m"][1] < 0,
        "worst_station_lead_at_most_0p01m": worst <= 0.01,
        "short_leads_bit_exact_uniform": bool(
            np.array_equal(
                prediction[frame["lead_h"].isin([3, 6, 9, 12])],
                uniform[frame["lead_h"].isin([3, 6, 9, 12])],
            )
        ),
        "finite_predictions": bool(np.isfinite(prediction).all()),
    }
    risk_checks = {
        "delta_rmse_at_most_minus_0p005m": delta <= -0.005,
        "worst_station_lead_at_most_0p02m": worst <= 0.02,
        "finite_predictions": stable_checks["finite_predictions"],
    }
    stable = all(stable_checks.values())
    high_risk = (not stable) and all(risk_checks.values())
    raw_points = -delta * POINTS_PER_RMSE_M
    return {
        "name": name,
        "decision": "PASS_STABLE" if stable else "PRESERVE_HIGH_RISK" if high_risk else "NO_GO",
        "rmse_m": {
            "uniform_0p425": uniform_rmse,
            "candidate": candidate_rmse,
            "delta_candidate_minus_uniform": delta,
        },
        "expected_points": {
            "raw_gain": raw_points,
            "transport_penalty": TRANSPORT_PENALTY_POINTS,
            "calibrated_gain": raw_points - TRANSPORT_PENALTY_POINTS,
            "nominal_official_score": OFFICIAL_CHAMPION_POINTS + raw_points,
        },
        "changed_rows": int(changed.sum()),
        "changed_share": float(changed.mean()),
        "by_block": by_block,
        "improved_blocks": int(improved),
        "station": station,
        "lead": lead,
        "station_lead": station_lead,
        "worst_station_lead_delta_m": worst,
        "episode_bootstrap": episode_ci,
        "block_station_bootstrap": group_ci,
        "stable_checks": stable_checks,
        "high_risk_checks": risk_checks,
    }


def evaluate(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frame, profile = v20.v19.load_historical()
    frame = attach_features(frame, load_features())
    predictions, receipts = candidate_predictions(frame)
    scored = [score_candidate(frame, predictions[name], name) for name in CANDIDATES]
    passes = [item for item in scored if item["decision"] != "NO_GO"]
    result = {
        "schema_version": "p3.uniform_selective_robust_fallback.result.v21",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_CANDIDATE_AVAILABLE" if passes else "NO_GO_ALL_SELECTIVE_FALLBACKS",
        "reference": config["reference"],
        "surface_claim": config["validation"]["surface_claim"],
        "candidates": scored,
        "gate_receipts": receipts,
        "fit_count": {"target_fits": 0, "shared_feature_only_quantile_fits": 6},
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
            "result_based_tuning": False,
            "threshold_searches": 0,
            "candidate_count": 2,
            "python": platform.python_version(),
        },
    }
    arrays = {
        "truth": frame["target_hs"].to_numpy(dtype=np.float64),
        "uniform": frame["reference"].to_numpy(dtype=np.float64),
        "candidate_1": predictions[CANDIDATES[0]],
        "candidate_2": predictions[CANDIDATES[1]],
        "lead_h": frame["lead_h"].to_numpy(dtype=np.int16),
        "block": frame["block"].astype(str).to_numpy(dtype="U5"),
        "station": frame["station"].astype(str).to_numpy(dtype="U5"),
        "episode": frame["episode_id"].astype(str).to_numpy(dtype="U32"),
    }
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 uniform selective robust fallback v21",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- current official champion/reference: uniform KMA alpha=.425, RMSE 0.575233m, 24.203599 points.",
    ]
    for item in result["candidates"]:
        metrics = item["rmse_m"]
        points = item["expected_points"]
        lines.append(
            f"- {item['name']}: {item['decision']}; RMSE {metrics['candidate']:.9f}m; delta {metrics['delta_candidate_minus_uniform']:+.9f}m; raw {points['raw_gain']:+.9f} points; calibrated {points['calibrated_gain']:+.9f}; blocks {item['improved_blocks']}/6; changed {item['changed_rows']} rows."
        )
        lines.append(
            f"  - episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}; worst station-lead {item['worst_station_lead_delta_m']:+.9f}m."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "EXPLORATORY_ONLY on a repeatedly exposed 182-case historical surface; no Public transport guarantee. The uniform champion is the exact no-op default. Gate calibration reads training-only inputs, removes no rows, and uses no target labels. Official test/sample/submission/hidden/CSV/upload access is all zero.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "LAUNCH_READY"}))
        return 0
    if ARTIFACT_DIR.exists() or (REPORT_DIR / "result.json").exists() or LOCK.exists():
        raise ContractError("v21 exactly-once namespace already exists")
    config = load_config()
    runner_hash = sha256(Path(__file__))
    write_new(
        LOCK,
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "ATTEMPT_CONSUMED_ONE_SHOT",
                "runner_sha256": runner_hash,
                "config_sha256": sha256(CONFIG),
                "official_access": 0,
            }
        ),
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result, arrays = evaluate(config)
    result["execution"]["elapsed_seconds"] = time.perf_counter() - started
    array_path = ARTIFACT_DIR / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {
        "runner_sha256": runner_hash,
        "config_sha256": sha256(CONFIG),
        "train_features_sha256": sha256(v20.TRAIN_FEATURES),
        "official_ledger_sha256": sha256(
            ROOT / "reports/p3_official_candidate_ledger_20260901_v1/official-lineage-audit.json"
        ),
        "evaluation_arrays_sha256": sha256(array_path),
    }
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, render_report(result).encode("utf-8"))
    write_new(
        REPORT_DIR / "run-manifest.json",
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "result_sha256": sha256(result_path),
                "arrays_sha256": sha256(array_path),
                "report_sha256": sha256(report_path),
                "official_access": 0,
                "csv_materializations": 0,
                "uploads": 0,
            }
        ),
    )
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "official_access": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
