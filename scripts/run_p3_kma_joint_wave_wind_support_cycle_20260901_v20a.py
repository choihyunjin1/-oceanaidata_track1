"""Run the sealed target-free joint wave-power and wind-support KMA policy."""

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

EXPERIMENT_ID = "p3_kma_joint_wave_wind_support_cycle_20260901_v20a"
CANDIDATE = "P3_1_V19_JOINT_WAVE_WIND_SUPPORT"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
PURGE_HOURS = 78
OFFICIAL_CHAMPION_POINTS = 24.203599
TRANSPORT_PENALTY_POINTS = 0.04958605409228893


class ContractError(RuntimeError):
    """Raised when the sealed v20a contract is violated."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "schema": config["schema_version"] == "p3.kma_joint_wave_wind_support.config.v20a",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "candidate": config["candidate"]["name"] == CANDIDATE,
        "target_free": config["candidate"]["target_fits"] == 0,
        "six_feature_fits": config["candidate"]["feature_only_fits"] == 6,
        "no_search": config["candidate"]["grid_or_search"] is False,
        "final_adjustment": config["duplication_audit"]["verdict"]
        == "NON_DUPLICATE_FINAL_ADJUSTMENT",
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
    }
    if not all(checks.values()):
        raise ContractError(f"v20a config contract failed: {checks}")
    return config


def wind_support(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    wind = frame["wind_input_proxy_current"].to_numpy(dtype=np.float64)
    alignment = frame["wind_wave_alignment_current"].to_numpy(dtype=np.float64)
    valid = np.isfinite(wind) & np.isfinite(alignment)
    score = np.full(len(frame), np.nan, dtype=np.float64)
    score[valid] = np.log1p(np.maximum(wind[valid], 0.0)) * np.maximum(alignment[valid], 0.0)
    return score, valid


def ranks_from_prefix(prefix: np.ndarray, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rank = np.full(len(values), 0.5, dtype=np.float64)
    if len(prefix):
        rank[valid] = np.searchsorted(prefix, values[valid], side="right") / len(prefix)
    return np.clip(rank, 0.0, 1.0)


def evaluate(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frame, profile = v20.v19.load_historical()
    history = v20.v19.load_wave_power_history()
    frame = v20.v19.attach_wave_power(frame, history)
    frame = v20.attach_wind_features(frame, v20.load_wind_features())
    v19_prediction, v19_alpha, v19_receipts = v20.build_v19_crossfit(frame, history)
    frame = frame.assign(v19_prediction=v19_prediction, v19_alpha=v19_alpha)
    candidate = v19_prediction.copy()
    candidate_alpha = v19_alpha.copy()
    receipts: list[dict[str, Any]] = []

    for block in block_order(frame):
        outer = frame["block"].eq(block).to_numpy()
        train = purge_training_cases(frame.loc[~outer], frame.loc[outer])
        train_unique = train.drop_duplicates(["anchor_id", "station"])
        train_score, train_valid = wind_support(train_unique)
        prefix = np.sort(train_score[train_valid])
        valid_score, valid = wind_support(frame.loc[outer])
        support_rank = ranks_from_prefix(prefix, valid_score, valid)
        fold_alpha = v19_alpha[outer].copy()
        lead24 = frame.loc[outer, "lead_h"].eq(24).to_numpy()
        wave_rank = np.clip((fold_alpha[lead24] - 0.20) / 0.40, 0.0, 1.0)
        joint_rank = np.sqrt(wave_rank * support_rank[lead24])
        fold_alpha[lead24] = 0.20 + 0.40 * joint_rank
        fold_prediction = v19_prediction[outer].copy()
        fold_prediction[lead24] = frame.loc[outer].loc[lead24, "base"].to_numpy(
            dtype=np.float64
        ) + fold_alpha[lead24] * frame.loc[outer].loc[lead24, "delta"].to_numpy(dtype=np.float64)
        candidate[outer] = np.clip(fold_prediction, 0.0, 30.0)
        candidate_alpha[outer] = fold_alpha
        receipts.append(
            {
                "block": str(block),
                "purged_unique_train_cases": int(len(train_unique)),
                "wind_support_prefix_cases": int(len(prefix)),
                "feature_only_fits": 1,
                "target_rows_read_before_policy_fixed": 0,
            }
        )

    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    uniform = frame["reference"].to_numpy(dtype=np.float64)
    persistence = frame["current_hs"].to_numpy(dtype=np.float64)
    candidate_rmse = rmse(truth, candidate)
    v19_rmse = rmse(truth, v19_prediction)
    uniform_rmse = rmse(truth, uniform)
    delta_v19 = candidate_rmse - v19_rmse
    delta_uniform = candidate_rmse - uniform_rmse
    by_block = v20.group_deltas(frame, candidate, v19_prediction, ["block"])
    station = v20.group_deltas(frame, candidate, v19_prediction, ["station"])
    lead = v20.group_deltas(frame, candidate, v19_prediction, ["lead_h"])
    station_lead = v20.group_deltas(frame, candidate, v19_prediction, ["station", "lead_h"])
    improved = sum(item["delta_rmse_m"] < 0 for item in by_block.values())
    worst = max(item["delta_rmse_m"] for item in station_lead.values())
    compare_frame = frame.assign(reference=v19_prediction)
    episode_ci = bootstrap(compare_frame, candidate, ("episode_id",), 20260911)
    block_station_ci = bootstrap(compare_frame, candidate, ("block", "station"), 20260912)
    short_and_18 = frame["lead_h"].isin([3, 6, 9, 12, 18]).to_numpy()
    stable_checks = {
        "delta_rmse_vs_v19_negative": delta_v19 < 0,
        "minimum_four_improved_blocks_vs_v19": improved >= 4,
        "episode_ci90_upper_vs_v19_below_zero": episode_ci["ci90_m"][1] < 0,
        "block_station_ci90_upper_vs_v19_below_zero": block_station_ci["ci90_m"][1] < 0,
        "worst_station_lead_vs_v19_at_most_0p01m": worst <= 0.01,
        "non24_leads_bit_exact_v19": bool(
            np.array_equal(candidate[short_and_18], v19_prediction[short_and_18])
        ),
        "six_feature_only_fits": sum(item["feature_only_fits"] for item in receipts) == 6,
        "finite_predictions": bool(np.isfinite(candidate).all()),
    }
    risk_checks = {
        "delta_rmse_vs_v19_at_most_minus_0p002m": delta_v19 <= -0.002,
        "delta_rmse_vs_uniform_at_most_minus_0p005m": delta_uniform <= -0.005,
        "worst_station_lead_vs_v19_at_most_0p02m": worst <= 0.02,
        "finite_predictions": bool(np.isfinite(candidate).all()),
    }
    stable = all(stable_checks.values())
    high_risk = (not stable) and all(risk_checks.values())
    decision = (
        "PASS_STABLE_CANDIDATE"
        if stable
        else "PRESERVE_HIGH_RISK_SCORE_CANDIDATE"
        if high_risk
        else "NO_GO_JOINT_WAVE_WIND_SUPPORT"
    )
    raw_gain = -delta_uniform * POINTS_PER_RMSE_M
    result = {
        "schema_version": "p3.kma_joint_wave_wind_support.result.v20a",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": decision,
        "candidate": {
            "name": CANDIDATE,
            "surface_claim": config["validation"]["surface_claim"],
            "rmse_m": {
                "persistence": rmse(truth, persistence),
                "uniform_kma_0p425": uniform_rmse,
                "v19_wave_power": v19_rmse,
                "candidate": candidate_rmse,
                "delta_candidate_minus_persistence": candidate_rmse - rmse(truth, persistence),
                "delta_candidate_minus_uniform": delta_uniform,
                "delta_candidate_minus_v19": delta_v19,
            },
            "expected_points": {
                "raw_gain_vs_uniform": raw_gain,
                "incremental_gain_vs_v19": -delta_v19 * POINTS_PER_RMSE_M,
                "transport_penalty_points": TRANSPORT_PENALTY_POINTS,
                "calibrated_gain_vs_uniform": raw_gain - TRANSPORT_PENALTY_POINTS,
                "expected_official_score_from_current_champion": OFFICIAL_CHAMPION_POINTS
                + raw_gain,
            },
            "by_block_vs_v19": by_block,
            "improved_blocks_vs_v19": int(improved),
            "station_vs_v19": station,
            "lead_vs_v19": lead,
            "station_lead_vs_v19": station_lead,
            "worst_station_lead_delta_vs_v19_m": worst,
            "episode_bootstrap_vs_v19": episode_ci,
            "block_station_bootstrap_vs_v19": block_station_ci,
            "feature_only_receipts": receipts,
            "v19_feature_only_receipts": v19_receipts,
            "stable_gate_checks": stable_checks,
            "high_risk_preservation_checks": risk_checks,
            "stable_pass": stable,
            "high_risk_preserved": high_risk,
        },
        "fit_count": {"target_fits": 0, "wind_support_ecdf_feature_only_fits": 6},
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
            "grid_searches": 0,
            "python": platform.python_version(),
        },
    }
    arrays = {
        "truth": truth,
        "persistence": persistence,
        "uniform": uniform,
        "v19": v19_prediction,
        "candidate": candidate,
        "candidate_alpha": candidate_alpha,
        "lead_h": frame["lead_h"].to_numpy(dtype=np.int16),
        "block": frame["block"].astype(str).to_numpy(dtype="U5"),
        "station": frame["station"].astype(str).to_numpy(dtype="U5"),
        "episode": frame["episode_id"].astype(str).to_numpy(dtype="U32"),
    }
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    candidate = result["candidate"]
    metrics = candidate["rmse_m"]
    points = candidate["expected_points"]
    return (
        "# P3 joint wave-wind support v20a\n\n## 결론\n\n"
        f"- decision: **{result['decision']}**.\n"
        f"- candidate RMSE `{metrics['candidate']:.9f}m`; vs uniform `{metrics['delta_candidate_minus_uniform']:+.9f}m`; vs v19 `{metrics['delta_candidate_minus_v19']:+.9f}m`.\n"
        f"- raw expected gain `{points['raw_gain_vs_uniform']:+.9f}` points; calibrated `{points['calibrated_gain_vs_uniform']:+.9f}`; expected official score `{points['expected_official_score_from_current_champion']:.9f}`.\n"
        f"- blocks `{candidate['improved_blocks_vs_v19']}/6`; worst station×lead `{candidate['worst_station_lead_delta_vs_v19_m']:+.9f}m`; episode CI90 `{candidate['episode_bootstrap_vs_v19']['ci90_m']}`; block×station CI90 `{candidate['block_station_bootstrap_vs_v19']['ci90_m']}`.\n"
        "- official test/sample/submission/hidden/CSV/upload access: all 0.\n\n"
        "This is exploratory on an exposed historical development surface and does not guarantee Public transport. No anonymous-case absolute time or future KMA/ERA5 alignment is used.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "LAUNCH_READY"}))
        return 0
    if ARTIFACT_DIR.exists() or REPORT_DIR.exists() or LOCK.exists():
        raise ContractError("v20a exactly-once namespace already exists")
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
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result, arrays = evaluate(config)
    result["execution"]["elapsed_seconds"] = time.perf_counter() - started
    array_path = ARTIFACT_DIR / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {
        "runner_sha256": runner_hash,
        "config_sha256": sha256(CONFIG),
        "train_features_sha256": sha256(v20.TRAIN_FEATURES),
        "v19_result_sha256": sha256(v20.v19.ARTIFACT_DIR / "result.json"),
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
