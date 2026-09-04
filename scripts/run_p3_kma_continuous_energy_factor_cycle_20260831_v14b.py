"""Run the sealed continuous-energy KMA factor P3 validation cycle."""

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

from run_p3_parallel_candidate_cycle_20260831_v4 import (  # noqa: E402
    KEYS,
    load_historical,
    official_frame,
    rmse,
)
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    POINTS_PER_RMSE_M,
    bootstrap,
    canonical,
    sha256,
    write_new,
)

EXPERIMENT_ID = "p3_kma_continuous_energy_factor_cycle_20260831_v14b"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CONFIG = ROOT / "configs/experiments/p3_kma_continuous_energy_factor_cycle_20260831_v14b.json"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v2/calibration.json"
TRAIN_FEATURES = ROOT / "artifacts/p3/features_all20_v1/train_features.parquet"
TRAIN_ANCHORS = ROOT / "artifacts/p3/features_all20_v1/train_anchors.parquet"
TEST_FEATURES = ROOT / "artifacts/p3/features_all20_v1/test_features.parquet"
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_KMA_CONTINUOUS_ENERGY_V14B"
)

CANDIDATE = "P3_1_KMA_CONTINUOUS_ENERGY_FACTOR"
FAMILY_ID = "P3_FIXED_KMA_LONGLEAD_FACTOR_CONTINUOUS_ENERGY"
TIER_ID = "LOW_DOF_FIXED"
CALIBRATION_SHA = "1a1d2c96cbe6c2c69b753fb4a130eb092922cc46524286cabcc0f9fce150441f"
REFERENCE_ALPHA = 0.425
ALPHA_18 = 0.20
ALPHA_24_MIN = 0.20
ALPHA_24_SPAN = 0.40
PURGE_HOURS = 78
PENALTY_POINTS = 0.04958605409228893
RAW_THRESHOLD_POINTS = 0.05958605409228893
MIN_CALIBRATED_POINTS = 0.01
MIN_IMPROVED_BLOCKS = 4
MAX_WORST_STATION_LEAD_M = 0.01
MAX_CHANGED_SHARE = 1.0 / 3.0


class ContractError(RuntimeError):
    """Raised when the sealed validation contract is violated."""


def load_energy_history() -> pd.DataFrame:
    features = pd.read_parquet(
        TRAIN_FEATURES,
        columns=["anchor_id", "station", "wave_energy_current"],
    )
    anchors = pd.read_parquet(
        TRAIN_ANCHORS,
        columns=["anchor_id", "station", "anchor_time"],
    )
    history = features.merge(
        anchors,
        on=["anchor_id", "station"],
        how="inner",
        validate="one_to_one",
    )
    history["anchor_time"] = pd.to_datetime(history["anchor_time"], utc=True)
    if (
        len(history) != len(features)
        or history.duplicated(["anchor_id", "station"]).any()
        or history["wave_energy_current"].isna().any()
        or not np.isfinite(history["wave_energy_current"]).all()
    ):
        raise ContractError("train energy history contract failed")
    return history


def attach_energy(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    energy = history[["anchor_id", "station", "wave_energy_current"]]
    merged = frame.merge(
        energy,
        on=["anchor_id", "station"],
        how="left",
        validate="many_to_one",
    )
    if merged["wave_energy_current"].isna().any():
        raise ContractError("historical OOF energy is missing")
    return merged


def ecdf(sorted_prefix: np.ndarray, values: np.ndarray) -> np.ndarray:
    prefix = np.asarray(sorted_prefix, dtype=np.float64)
    query = np.asarray(values, dtype=np.float64)
    if prefix.size == 0 or not np.isfinite(prefix).all() or not np.isfinite(query).all():
        raise ContractError("ECDF input is empty or non-finite")
    return np.searchsorted(prefix, query, side="right").astype(np.float64) / prefix.size


def predict_policy(frame: pd.DataFrame, rank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(rank) != len(frame):
        raise ContractError("energy rank length mismatch")
    prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
    alpha = np.zeros(len(frame), dtype=np.float64)
    lead18 = frame["lead_h"].eq(18).to_numpy()
    lead24 = frame["lead_h"].eq(24).to_numpy()
    alpha[lead18] = ALPHA_18
    alpha[lead24] = ALPHA_24_MIN + ALPHA_24_SPAN * rank[lead24]
    prediction[lead18] = (
        frame.loc[lead18, "base"].to_numpy(dtype=np.float64)
        + alpha[lead18] * frame.loc[lead18, "delta"].to_numpy(dtype=np.float64)
    )
    prediction[lead24] = (
        frame.loc[lead24, "base"].to_numpy(dtype=np.float64)
        + alpha[lead24] * frame.loc[lead24, "delta"].to_numpy(dtype=np.float64)
    )
    return np.clip(prediction, 0.0, 30.0), alpha


def evaluate(frame: pd.DataFrame, history: pd.DataFrame) -> dict[str, Any]:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    alpha = np.zeros(len(frame), dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    block_order = (
        frame.groupby("block", observed=True)["anchor_time"]
        .min()
        .sort_values()
        .index.tolist()
    )
    for block in block_order:
        valid_mask = frame["block"].eq(block).to_numpy()
        valid = frame.loc[valid_mask]
        block_start = pd.Timestamp(valid["anchor_time"].min())
        boundary = block_start - pd.Timedelta(hours=PURGE_HOURS)
        prefix = history.loc[history["anchor_time"].le(boundary)]
        prefix_values = np.sort(prefix["wave_energy_current"].to_numpy(dtype=np.float64))
        ranks = ecdf(
            prefix_values,
            valid["wave_energy_current"].to_numpy(dtype=np.float64),
        )
        fold_prediction, fold_alpha = predict_policy(valid, ranks)
        prediction[valid_mask] = fold_prediction
        alpha[valid_mask] = fold_alpha
        receipts.append(
            {
                "block": str(block),
                "block_start_utc": block_start.isoformat(),
                "prefix_end_boundary_utc": boundary.isoformat(),
                "prefix_max_anchor_time_utc": pd.Timestamp(prefix["anchor_time"].max()).isoformat(),
                "prefix_cases": int(len(prefix)),
                "validation_rows": int(len(valid)),
                "calibration_fits": 1,
                "outer_target_rows_read_before_policy_fixed": 0,
            }
        )
    if not np.isfinite(prediction).all():
        raise ContractError("OOF policy prediction incomplete")

    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    delta = rmse(truth, prediction) - rmse(truth, reference)
    scored = frame.assign(candidate=prediction, candidate_alpha=alpha)
    by_block: dict[str, Any] = {}
    for block, part in scored.groupby("block", observed=True, sort=True):
        by_block[str(block)] = {
            "rows": int(len(part)),
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy()),
        }
    station_lead: dict[str, Any] = {}
    for (station, lead), part in scored.groupby(
        ["station", "lead_h"], observed=True, sort=True
    ):
        station_lead[f"{station}|{int(lead)}"] = {
            "rows": int(len(part)),
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy()),
        }
    episode = bootstrap(frame, prediction, ("episode_id",), 20260831)
    block_station = bootstrap(frame, prediction, ("block", "station"), 20260832)
    bootstrap_upper = max(episode["ci90_m"][1], block_station["ci90_m"][1])
    raw_lcb = max(0.0, -bootstrap_upper * POINTS_PER_RMSE_M)
    comparator_alpha = np.where(frame["lead_h"].isin([18, 24]), REFERENCE_ALPHA, 0.0)
    changed = np.abs(alpha - comparator_alpha) > 1e-12
    short = frame["lead_h"].isin([3, 6, 9, 12]).to_numpy()
    short_exact = bool(np.array_equal(prediction[short], reference[short]))
    changed_share = float(changed.mean())
    improved_blocks = sum(value["delta_rmse_m"] < 0 for value in by_block.values())
    worst_station_lead = max(value["delta_rmse_m"] for value in station_lead.values())
    calibrated_lcb = raw_lcb - PENALTY_POINTS
    checks = {
        "pooled_rmse_improves": delta < 0,
        "minimum_four_improved_bimonth_blocks": improved_blocks >= MIN_IMPROVED_BLOCKS,
        "episode_ci90_upper_below_zero": episode["ci90_m"][1] < 0,
        "block_station_ci90_upper_below_zero": block_station["ci90_m"][1] < 0,
        "worst_station_lead_within_0p01m": worst_station_lead <= MAX_WORST_STATION_LEAD_M,
        "changed_rows_share_at_most_one_third": changed_share <= MAX_CHANGED_SHARE + 1e-12,
        "local_lcb_raw_points_meets_family_threshold": raw_lcb >= RAW_THRESHOLD_POINTS,
        "calibrated_lcb_points_at_least_0p01_inclusive": calibrated_lcb >= MIN_CALIBRATED_POINTS,
        "short_leads_bit_exact_comparator": short_exact,
        "finite_predictions": bool(np.isfinite(prediction).all()),
        "six_train_prefix_ecdf_fits": sum(x["calibration_fits"] for x in receipts) == 6,
    }
    return {
        "candidate": {
            "name": CANDIDATE,
            "family_id": FAMILY_ID,
            "tier_id": TIER_ID,
            "representation_changed": False,
            "routing_discontinuous": False,
            "active_share_rule": "changed_rows_share_le_1_over_3",
            "exact_comparator": "uniform_kma_alpha_0.425",
            "local_lcb_raw_points": "negative_max_episode_and_block_station_ci90_upper_times_metric_slope",
            "selected_penalty_provenance_sha256": sha256(CALIBRATION),
            "alpha_18": ALPHA_18,
            "alpha_24_formula": "0.20 + 0.40 * train_prefix_ecdf(wave_energy_current)",
        },
        "reference_rmse_m": rmse(truth, reference),
        "candidate_rmse_m": rmse(truth, prediction),
        "delta_candidate_minus_reference_rmse_m": delta,
        "by_block": by_block,
        "improved_bimonth_blocks": int(improved_blocks),
        "station_lead": station_lead,
        "worst_station_lead_delta_rmse_m": worst_station_lead,
        "episode_bootstrap": episode,
        "block_station_bootstrap": block_station,
        "changed_rows": int(changed.sum()),
        "changed_rows_share": changed_share,
        "short_leads_bit_exact": short_exact,
        "ecdf_calibration_receipts": receipts,
        "fit_count": {"ecdf_calibration_fits": 6, "model_fits": 0},
        "expected_points": {
            "raw_central": -delta * POINTS_PER_RMSE_M,
            "local_lcb_raw_points": raw_lcb,
            "selected_transport_penalty_points": PENALTY_POINTS,
            "calibrated_conservative": calibrated_lcb,
        },
        "gate_checks": checks,
        "passed": all(checks.values()),
    }


def materialize(
    passed: bool,
    history: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    access = {
        "official_test_index_rows_read": 0,
        "official_test_context_rows_read": 0,
        "official_feature_rows_read": 0,
        "official_component_prediction_rows_read": 0,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }
    if not passed:
        return [], access
    frame, champion = official_frame()
    test_energy = pd.read_parquet(
        TEST_FEATURES,
        columns=["case_id", "station", "wave_energy_current"],
    )
    frame = frame.merge(
        test_energy,
        on=["case_id", "station"],
        how="left",
        validate="many_to_one",
    )
    if not champion[KEYS].equals(frame[KEYS]) or frame["wave_energy_current"].isna().any():
        raise ContractError("official order or energy contract failed")
    prefix = np.sort(history["wave_energy_current"].to_numpy(dtype=np.float64))
    ranks = ecdf(prefix, frame["wave_energy_current"].to_numpy(dtype=np.float64))
    prediction, _ = predict_policy(frame, ranks)
    submission = frame[KEYS].copy()
    submission["hs_pred"] = prediction
    if (
        len(submission) != 1200
        or submission.duplicated(KEYS).any()
        or not np.isfinite(submission["hs_pred"]).all()
    ):
        raise ContractError("official submission structural QA failed")
    DELIVERY.mkdir(parents=True, exist_ok=False)
    path = DELIVERY / "P3_submission.csv"
    payload = submission.to_csv(index=False, lineterminator="\n").encode()
    write_new(path, payload)
    output = {
        "candidate": CANDIDATE,
        "path": str(path),
        "rows": int(len(submission)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "uploads": 0,
    }
    write_new(
        DELIVERY / "SET_MANIFEST.json",
        canonical({"experiment_id": EXPERIMENT_ID, "outputs": [output], "uploads": 0}),
    )
    access.update(
        {
            "official_test_index_rows_read": 1200,
            "official_test_context_rows_read": 200,
            "official_feature_rows_read": 200,
            "official_component_prediction_rows_read": 3600,
        }
    )
    return [output], access


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "candidate": CANDIDATE}))
        return 0
    if ARTIFACT_DIR.exists() or LOCK.exists():
        raise ContractError("exactly-once artifact or lock already exists")
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if sha256(CALIBRATION) != CALIBRATION_SHA or calibration["status"] != "FAMILY_AWARE_GUARDRAIL_READY":
        raise ContractError("family-aware calibration seal changed")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runner_hash = sha256(Path(__file__))
    write_new(
        LOCK,
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "ATTEMPT_CONSUMED_ONE_SHOT",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "runner_sha256": runner_hash,
                "config_sha256": sha256(CONFIG),
                "calibration_sha256": sha256(CALIBRATION),
                "candidate_registration": {
                    "name": CANDIDATE,
                    "family_id": FAMILY_ID,
                    "tier_id": TIER_ID,
                    "representation_changed": False,
                    "routing_discontinuous": False,
                    "active_share_rule": "changed_rows_share_le_1_over_3",
                    "exact_comparator": "uniform_kma_alpha_0.425",
                    "local_lcb_raw_points": "negative_max_episode_and_block_station_ci90_upper_times_metric_slope",
                    "selected_penalty_provenance_sha256": sha256(CALIBRATION),
                },
            }
        ),
    )
    frame, profile = load_historical()
    history = load_energy_history()
    frame = attach_energy(frame, history)
    candidate = evaluate(frame, history)
    outputs, access = materialize(candidate["passed"], history)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "p3.kma_continuous_energy_factor.result.v14b",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_MATERIALIZED_NOT_UPLOADED" if candidate["passed"] else "NO_GO_ENERGY_KMA_DIRECTION_CLOSED",
        "candidate_count": 1,
        "passing_candidate_count": int(candidate["passed"]),
        "candidate": candidate,
        "fit_budget": {"ecdf_calibration_fits": 6, "model_fits": 0, "full_deployment_ecdf_fits": int(candidate["passed"])},
        "data_profile": profile,
        "outputs": outputs,
        "data_access": access,
        "provenance": {
            "runner_sha256": runner_hash,
            "config_sha256": sha256(CONFIG),
            "calibration_sha256": sha256(CALIBRATION),
            "train_features_sha256": sha256(TRAIN_FEATURES),
            "train_anchors_sha256": sha256(TRAIN_ANCHORS),
        },
        "execution": {
            "elapsed_seconds": elapsed,
            "python": platform.python_version(),
            "result_based_tuning_or_retry": False,
            "hidden_truth_rows_read": 0,
            "upload_attempt_count": 0,
        },
    }
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report = (
        "# P3 continuous-energy KMA factor v14b\n\n"
        "## 결론\n\n"
        f"- 판정: **{result['decision']}**; PASS {result['passing_candidate_count']}/1; CSV {len(outputs)}; upload 0.\n"
        f"- pooled delta RMSE: {candidate['delta_candidate_minus_reference_rmse_m']:+.9f}m.\n"
        f"- episode CI90: {candidate['episode_bootstrap']['ci90_m']}; block×station CI90: {candidate['block_station_bootstrap']['ci90_m']}.\n"
        f"- raw LCB {candidate['expected_points']['local_lcb_raw_points']:.9f}점, family penalty {PENALTY_POINTS:.9f}점, calibrated {candidate['expected_points']['calibrated_conservative']:.9f}점.\n"
        f"- improved blocks {candidate['improved_bimonth_blocks']}/6; worst station×lead {candidate['worst_station_lead_delta_rmse_m']:+.9f}m; changed share {candidate['changed_rows_share']:.6f}.\n\n"
        "## 설계\n\n"
        "- α18=.20 고정, α24=.20+.40·train-prefix ECDF(wave_energy_current); hard threshold와 결과 기반 검색은 없다.\n"
        "- 각 outer block 시작 78시간 전까지의 feature-only history로 ECDF를 고정한 뒤 outer truth를 채점했다.\n"
        "- 중복된 90,601-point α-grid v14는 실행하지 않고 취소 receipt로 보존했다.\n"
    )
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, report.encode("utf-8"))
    write_new(
        REPORT_DIR / "run-manifest.json",
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "runner_sha256": runner_hash,
                "result_sha256": sha256(result_path),
                "report_sha256": sha256(report_path),
                "outputs": outputs,
            }
        ),
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "passing": result["passing_candidate_count"],
                "outputs": len(outputs),
                "calibration_fits": 6,
                "model_fits": 0,
                "elapsed_seconds": elapsed,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
