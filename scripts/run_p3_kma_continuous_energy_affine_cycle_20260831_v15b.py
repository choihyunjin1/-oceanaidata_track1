"""Run sealed prefix-ridge continuous-energy affine KMA calibration for P3."""

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

from run_p3_kma_continuous_energy_factor_cycle_20260831_v14b import (  # noqa: E402
    CALIBRATION,
    CALIBRATION_SHA,
    KEYS,
    POINTS_PER_RMSE_M,
    attach_energy,
    bootstrap,
    canonical,
    ecdf,
    load_energy_history,
    load_historical,
    official_frame,
    rmse,
    sha256,
    write_new,
)

EXPERIMENT_ID = "p3_kma_continuous_energy_affine_cycle_20260831_v15b"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CONFIG = ROOT / "configs/experiments/p3_kma_continuous_energy_affine_cycle_20260831_v15b.json"
TRAIN_FEATURES = ROOT / "artifacts/p3/features_all20_v1/train_features.parquet"
TEST_FEATURES = ROOT / "artifacts/p3/features_all20_v1/test_features.parquet"
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_KMA_CONTINUOUS_ENERGY_AFFINE_V15B"
)

CANDIDATE = "P3_1_PREFIX_RIDGE_CONTINUOUS_ENERGY_AFFINE"
FAMILY_ID = "LOW_DOF_FIXED_CONTINUOUS_KMA"
TIER_ID = "LOW_DOF_FIXED"
REFERENCE_ALPHA = 0.425
ALPHA_18 = 0.20
THETA_PRIOR = np.asarray([0.20, 0.40], dtype=np.float64)
THETA_PRIOR_SCALES = np.asarray([0.20, 0.40], dtype=np.float64)
PURGE_HOURS = 78
PENALTY_POINTS = 0.04958605409228893
RAW_THRESHOLD_POINTS = 0.05958605409228893
MIN_CALIBRATED_POINTS = 0.01
MIN_IMPROVED_BLOCKS = 4
MAX_WORST_STATION_LEAD_M = 0.01
MAX_CHANGED_SHARE = 1.0 / 3.0


class ContractError(RuntimeError):
    """Raised when a sealed affine-cycle invariant is violated."""


def block_order(frame: pd.DataFrame) -> list[str]:
    return [
        str(value)
        for value in (
            frame.groupby("block", observed=True)["anchor_time"]
            .min()
            .sort_values()
            .index.tolist()
        )
    ]


def sequential_energy_rank(frame: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
    cases = frame[
        ["anchor_id", "station", "anchor_time", "wave_energy_current"]
    ].drop_duplicates(["anchor_id", "station"])
    ranks: dict[tuple[str, str], float] = {}
    for row in cases.itertuples(index=False):
        boundary = pd.Timestamp(row.anchor_time) - pd.Timedelta(hours=PURGE_HOURS)
        prefix = np.sort(
            history.loc[
                history["anchor_time"].le(boundary), "wave_energy_current"
            ].to_numpy(dtype=np.float64)
        )
        ranks[(str(row.anchor_id), str(row.station))] = float(
            ecdf(prefix, np.asarray([row.wave_energy_current], dtype=np.float64))[0]
        )
    return np.asarray(
        [ranks[(str(a), str(s))] for a, s in zip(frame["anchor_id"], frame["station"], strict=True)],
        dtype=np.float64,
    )


def design(frame: pd.DataFrame, rank: np.ndarray) -> np.ndarray:
    delta = frame["delta"].to_numpy(dtype=np.float64)
    return np.column_stack([delta, delta * (rank - 0.5)])


def freeze_ridge_prior(
    frame: pd.DataFrame,
    history: pd.DataFrame,
) -> dict[str, Any]:
    first = block_order(frame)[0]
    valid = frame.loc[frame["block"].eq(first) & frame["lead_h"].eq(24)].copy()
    boundary = pd.Timestamp(valid["anchor_time"].min()) - pd.Timedelta(hours=PURGE_HOURS)
    prefix = np.sort(
        history.loc[
            history["anchor_time"].le(boundary), "wave_energy_current"
        ].to_numpy(dtype=np.float64)
    )
    rank = ecdf(prefix, valid["wave_energy_current"].to_numpy(dtype=np.float64))
    matrix = design(valid, rank)
    moments = np.sqrt(np.mean(np.square(matrix), axis=0))
    sigma0_sq = float(
        np.mean(
            np.square(
                valid["reference"].to_numpy(dtype=np.float64)
                - valid["current_hs"].to_numpy(dtype=np.float64)
            )
        )
    )
    if sigma0_sq <= 0 or np.any(moments <= 0) or not np.isfinite(moments).all():
        raise ContractError("feature-only ridge prior scale is invalid")
    prior_beta = THETA_PRIOR * moments
    prior_beta_sd = THETA_PRIOR_SCALES * moments
    ridge_diag = sigma0_sq / np.square(prior_beta_sd)
    return {
        "first_outer_block": first,
        "sigma0_sq_feature_only": sigma0_sq,
        "feature_rms": moments.tolist(),
        "prior_beta": prior_beta.tolist(),
        "prior_beta_sd": prior_beta_sd.tolist(),
        "ridge_diag": ridge_diag.tolist(),
        "outer_truth_rows_read": 0,
    }


def fit_theta(
    train: pd.DataFrame,
    rank: np.ndarray,
    prior: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    moments = np.asarray(prior["feature_rms"], dtype=np.float64)
    prior_beta = np.asarray(prior["prior_beta"], dtype=np.float64)
    ridge = np.diag(np.asarray(prior["ridge_diag"], dtype=np.float64))
    if train.empty:
        beta = prior_beta.copy()
        action = "prior_only_empty_prefix"
    else:
        matrix = design(train, rank) / moments
        # y-reference + reference_alpha*delta equals y-base, so theta is alpha-axis.
        response = (
            train["target_hs"].to_numpy(dtype=np.float64)
            - train["reference"].to_numpy(dtype=np.float64)
            + REFERENCE_ALPHA * train["delta"].to_numpy(dtype=np.float64)
        )
        beta = np.linalg.solve(
            matrix.T @ matrix + ridge,
            matrix.T @ response + ridge @ prior_beta,
        )
        action = "closed_form_prefix_bayesian_ridge"
    theta = beta / moments
    if not np.isfinite(theta).all():
        raise ContractError("affine theta non-finite")
    return theta, {
        "action": action,
        "train_rows": int(len(train)),
        "theta_a": float(theta[0]),
        "theta_b": float(theta[1]),
        "coefficient_calibration_fits": 2,
    }


def predict(frame: pd.DataFrame, rank: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
    alpha = np.zeros(len(frame), dtype=np.float64)
    lead18 = frame["lead_h"].eq(18).to_numpy()
    lead24 = frame["lead_h"].eq(24).to_numpy()
    alpha[lead18] = ALPHA_18
    alpha[lead24] = theta[0] + theta[1] * (rank[lead24] - 0.5)
    for mask in (lead18, lead24):
        prediction[mask] = (
            frame.loc[mask, "base"].to_numpy(dtype=np.float64)
            + alpha[mask] * frame.loc[mask, "delta"].to_numpy(dtype=np.float64)
        )
    return np.clip(prediction, 0.0, 30.0), alpha


def evaluate(frame: pd.DataFrame, history: pd.DataFrame) -> dict[str, Any]:
    prior = freeze_ridge_prior(frame, history)
    row_rank = sequential_energy_rank(frame, history)
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    alpha = np.zeros(len(frame), dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    for block in block_order(frame):
        valid_mask = frame["block"].eq(block).to_numpy()
        valid = frame.loc[valid_mask]
        boundary = pd.Timestamp(valid["anchor_time"].min()) - pd.Timedelta(hours=PURGE_HOURS)
        train_mask = frame["anchor_time"].le(boundary) & frame["lead_h"].eq(24)
        train = frame.loc[train_mask]
        theta, receipt = fit_theta(train, row_rank[train_mask.to_numpy()], prior)
        prefix = np.sort(
            history.loc[
                history["anchor_time"].le(boundary), "wave_energy_current"
            ].to_numpy(dtype=np.float64)
        )
        valid_rank = ecdf(prefix, valid["wave_energy_current"].to_numpy(dtype=np.float64))
        fold_prediction, fold_alpha = predict(valid, valid_rank, theta)
        prediction[valid_mask] = fold_prediction
        alpha[valid_mask] = fold_alpha
        receipt.update(
            {
                "block": block,
                "boundary_utc": boundary.isoformat(),
                "energy_prefix_cases": int(len(prefix)),
                "outer_target_rows_read_before_theta_fixed": 0,
            }
        )
        receipts.append(receipt)
    if not np.isfinite(prediction).all():
        raise ContractError("affine OOF prediction incomplete")
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    delta_rmse = rmse(truth, prediction) - rmse(truth, reference)
    scored = frame.assign(candidate=prediction, candidate_alpha=alpha)
    by_block = {
        str(block): {
            "rows": int(len(part)),
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy()),
        }
        for block, part in scored.groupby("block", observed=True, sort=True)
    }
    station_lead = {
        f"{station}|{int(lead)}": {
            "rows": int(len(part)),
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy()),
        }
        for (station, lead), part in scored.groupby(
            ["station", "lead_h"], observed=True, sort=True
        )
    }
    episode = bootstrap(frame, prediction, ("episode_id",), 20260831)
    group = bootstrap(frame, prediction, ("block", "station"), 20260832)
    upper = max(episode["ci90_m"][1], group["ci90_m"][1])
    raw_lcb = max(0.0, -upper * POINTS_PER_RMSE_M)
    comparator_alpha = np.where(frame["lead_h"].isin([18, 24]), REFERENCE_ALPHA, 0.0)
    changed = np.abs(alpha - comparator_alpha) > 1e-12
    short = frame["lead_h"].isin([3, 6, 9, 12]).to_numpy()
    improved_blocks = sum(value["delta_rmse_m"] < 0 for value in by_block.values())
    worst = max(value["delta_rmse_m"] for value in station_lead.values())
    changed_share = float(changed.mean())
    calibrated = raw_lcb - PENALTY_POINTS
    checks = {
        "pooled_rmse_improves": delta_rmse < 0,
        "minimum_four_improved_bimonth_blocks": improved_blocks >= 4,
        "episode_ci90_upper_below_zero": episode["ci90_m"][1] < 0,
        "block_station_ci90_upper_below_zero": group["ci90_m"][1] < 0,
        "worst_station_lead_within_0p01m": worst <= MAX_WORST_STATION_LEAD_M,
        "changed_rows_share_at_most_one_third": changed_share <= MAX_CHANGED_SHARE + 1e-12,
        "raw_lcb_points_meets_family_threshold": raw_lcb >= RAW_THRESHOLD_POINTS,
        "calibrated_lcb_at_least_0p01": calibrated >= MIN_CALIBRATED_POINTS,
        "short_leads_bit_exact": bool(np.array_equal(prediction[short], reference[short])),
        "twelve_coefficient_calibration_fits": sum(x["coefficient_calibration_fits"] for x in receipts) == 12,
        "finite_predictions": bool(np.isfinite(prediction).all()),
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
        },
        "ridge_prior": prior,
        "calibration_receipts": receipts,
        "reference_rmse_m": rmse(truth, reference),
        "candidate_rmse_m": rmse(truth, prediction),
        "delta_candidate_minus_reference_rmse_m": delta_rmse,
        "by_block": by_block,
        "improved_bimonth_blocks": int(improved_blocks),
        "station_lead": station_lead,
        "worst_station_lead_delta_rmse_m": worst,
        "episode_bootstrap": episode,
        "block_station_bootstrap": group,
        "changed_rows": int(changed.sum()),
        "changed_rows_share": changed_share,
        "fit_count": {"coefficient_calibration_fits": 12, "model_fits": 0},
        "expected_points": {
            "raw_central": -delta_rmse * POINTS_PER_RMSE_M,
            "local_lcb_raw_points": raw_lcb,
            "transport_penalty_points": PENALTY_POINTS,
            "calibrated_conservative": calibrated,
        },
        "gate_checks": checks,
        "passed": all(checks.values()),
    }


def materialize(passed: bool, frame: pd.DataFrame, history: pd.DataFrame, prior: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    access = {"official_test_index_rows_read": 0, "official_test_context_rows_read": 0, "official_feature_rows_read": 0, "official_component_prediction_rows_read": 0, "hidden_truth_rows_read": 0, "uploads": 0}
    if not passed:
        return [], access
    training_rank = sequential_energy_rank(frame, history)
    train24 = frame["lead_h"].eq(24)
    theta, _ = fit_theta(frame.loc[train24], training_rank[train24.to_numpy()], prior)
    official, champion = official_frame()
    energy = pd.read_parquet(TEST_FEATURES, columns=["case_id", "station", "wave_energy_current"])
    official = official.merge(energy, on=["case_id", "station"], how="left", validate="many_to_one")
    if not champion[KEYS].equals(official[KEYS]) or official["wave_energy_current"].isna().any():
        raise ContractError("official key/order/energy contract failed")
    prefix = np.sort(history["wave_energy_current"].to_numpy(dtype=np.float64))
    rank = ecdf(prefix, official["wave_energy_current"].to_numpy(dtype=np.float64))
    prediction, _ = predict(official, rank, theta)
    submission = official[KEYS].copy()
    submission["hs_pred"] = prediction
    if len(submission) != 1200 or submission.duplicated(KEYS).any() or not np.isfinite(prediction).all():
        raise ContractError("official structural QA failed")
    DELIVERY.mkdir(parents=True, exist_ok=False)
    path = DELIVERY / "P3_submission.csv"
    payload = submission.to_csv(index=False, lineterminator="\n").encode()
    write_new(path, payload)
    output = {"candidate": CANDIDATE, "path": str(path), "rows": 1200, "sha256": hashlib.sha256(payload).hexdigest(), "uploads": 0}
    write_new(DELIVERY / "SET_MANIFEST.json", canonical({"experiment_id": EXPERIMENT_ID, "outputs": [output], "uploads": 0}))
    access.update({"official_test_index_rows_read": 1200, "official_test_context_rows_read": 200, "official_feature_rows_read": 200, "official_component_prediction_rows_read": 3600})
    return [output], access


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "candidate": CANDIDATE}))
        return 0
    if ARTIFACT_DIR.exists() or LOCK.exists():
        raise ContractError("exactly-once artifact or lock exists")
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if sha256(CALIBRATION) != CALIBRATION_SHA or calibration["status"] != "FAMILY_AWARE_GUARDRAIL_READY":
        raise ContractError("calibration seal changed")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runner_hash = sha256(Path(__file__))
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "created_at_utc": datetime.now(UTC).isoformat(), "runner_sha256": runner_hash, "config_sha256": sha256(CONFIG), "calibration_sha256": sha256(CALIBRATION), "candidate_registration": {"name": CANDIDATE, "family_id": FAMILY_ID, "tier_id": TIER_ID, "representation_changed": False, "routing_discontinuous": False, "active_share_rule": "changed_rows_share_le_1_over_3", "exact_comparator": "uniform_kma_alpha_0.425", "local_lcb_raw_points": "negative_max_episode_and_block_station_ci90_upper_times_metric_slope", "selected_penalty_provenance_sha256": sha256(CALIBRATION)}}))
    frame, profile = load_historical()
    history = load_energy_history()
    frame = attach_energy(frame, history)
    candidate = evaluate(frame, history)
    outputs, access = materialize(candidate["passed"], frame, history, candidate["ridge_prior"])
    elapsed = time.perf_counter() - started
    result = {"schema_version": "p3.kma_continuous_energy_affine.result.v15b", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "PASS_MATERIALIZED_NOT_UPLOADED" if candidate["passed"] else "NO_GO_PREFIX_RIDGE_AFFINE_GATE", "candidate_count": 1, "passing_candidate_count": int(candidate["passed"]), "candidate": candidate, "fit_budget": {"coefficient_calibration_fits": 12, "model_fits": 0, "full_deployment_fits": 2 * int(candidate["passed"])}, "data_profile": profile, "outputs": outputs, "data_access": access, "provenance": {"runner_sha256": runner_hash, "config_sha256": sha256(CONFIG), "calibration_sha256": sha256(CALIBRATION), "train_features_sha256": sha256(TRAIN_FEATURES)}, "execution": {"elapsed_seconds": elapsed, "python": platform.python_version(), "result_based_tuning_or_retry": False, "hidden_truth_rows_read": 0, "upload_attempt_count": 0}}
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report = "# P3 prefix-ridge continuous-energy affine v15b\n\n## 결론\n\n" + f"- {result['decision']}; PASS {result['passing_candidate_count']}/1; CSV {len(outputs)}; upload 0.\n" + f"- pooled delta {candidate['delta_candidate_minus_reference_rmse_m']:+.9f}m; blocks {candidate['improved_bimonth_blocks']}/6; episode CI {candidate['episode_bootstrap']['ci90_m']}; group CI {candidate['block_station_bootstrap']['ci90_m']}.\n" + f"- raw LCB {candidate['expected_points']['local_lcb_raw_points']:.9f} points; calibrated {candidate['expected_points']['calibrated_conservative']:.9f} points.\n"
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, report.encode("utf-8"))
    write_new(REPORT_DIR / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "runner_sha256": runner_hash, "result_sha256": sha256(result_path), "report_sha256": sha256(report_path), "outputs": outputs}))
    print(json.dumps({"status": "COMPLETE", "passing": result["passing_candidate_count"], "outputs": len(outputs), "calibration_fits": 12, "model_fits": 0, "elapsed_seconds": elapsed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
