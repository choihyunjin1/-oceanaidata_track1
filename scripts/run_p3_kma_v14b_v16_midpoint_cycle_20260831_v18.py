"""Run the sealed fixed midpoint between the v14b and v16 P3 policies."""

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

from run_p3_kma_annual_harmonic_shrink_cycle_20260831_v16 import (  # noqa: E402
    CALIBRATION,
    CALIBRATION_SHA,
    KEYS,
    POINTS_PER_RMSE_M,
    attach_energy,
    block_order,
    bootstrap,
    canonical,
    ecdf,
    fit_theta,
    load_energy_history,
    load_historical,
    official_frame,
    rmse,
    sequential_energy_rank,
    sha256,
    v14b_correction,
    write_new,
)
from run_p3_kma_annual_harmonic_shrink_cycle_20260831_v16 import (  # noqa: E402
    predict as predict_v16,
)

EXPERIMENT_ID = "p3_kma_v14b_v16_midpoint_cycle_20260831_v18"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CONFIG = ROOT / "configs/experiments/p3_kma_v14b_v16_midpoint_cycle_20260831_v18.json"
DUPLICATION_AUDIT = REPORT_DIR / "duplication-audit.json"
TEST_FEATURES = ROOT / "artifacts/p3/features_all20_v1/test_features.parquet"
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_KMA_V14B_V16_MIDPOINT_V18"
)

CANDIDATE = "P3_1_FIXED_V14B_V16_MIDPOINT"
FAMILY_ID = "P3_FIXED_KMA_LONGLEAD_FACTOR"
TIER_ID = "LOW_DOF_FIXED"
FIXED_WEIGHT = 0.5
PURGE_HOURS = 78
PENALTY_POINTS = 0.04958605409228893
RAW_THRESHOLD_POINTS = 0.05958605409228893
MIN_CALIBRATED_POINTS = 0.01
MAX_WORST_STATION_LEAD_M = 0.01
MAX_CHANGED_SHARE = 1.0 / 3.0


class ContractError(RuntimeError):
    """Raised when a sealed midpoint-cycle invariant is violated."""


def midpoint_prediction(
    frame: pd.DataFrame,
    correction: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, multiplier16 = predict_v16(frame, correction, theta)
    multiplier18 = FIXED_WEIGHT * (1.0 + multiplier16)
    prediction = frame["reference"].to_numpy(dtype=np.float64) + multiplier18 * correction
    return np.clip(prediction, 0.0, 30.0), multiplier18


def evaluate(frame: pd.DataFrame, history: pd.DataFrame) -> dict[str, Any]:
    row_rank = sequential_energy_rank(frame, history)
    all_correction = v14b_correction(frame, row_rank)
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    multiplier = np.ones(len(frame), dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    for block in block_order(frame):
        valid_mask = frame["block"].eq(block).to_numpy()
        valid = frame.loc[valid_mask]
        boundary = pd.Timestamp(valid["anchor_time"].min()) - pd.Timedelta(hours=PURGE_HOURS)
        train_mask = frame["anchor_time"].le(boundary) & frame["lead_h"].isin([18, 24])
        theta, receipt = fit_theta(
            frame.loc[train_mask],
            all_correction[train_mask.to_numpy()],
        )
        prefix = np.sort(
            history.loc[
                history["anchor_time"].le(boundary), "wave_energy_current"
            ].to_numpy(dtype=np.float64)
        )
        rank = ecdf(prefix, valid["wave_energy_current"].to_numpy(dtype=np.float64))
        correction = v14b_correction(valid, rank)
        fold_prediction, fold_multiplier = midpoint_prediction(valid, correction, theta)
        prediction[valid_mask] = fold_prediction
        multiplier[valid_mask] = fold_multiplier
        receipt.update(
            {
                "block": block,
                "boundary_utc": boundary.isoformat(),
                "energy_prefix_cases": int(len(prefix)),
                "midpoint_weight": FIXED_WEIGHT,
                "outer_target_rows_read_before_theta_fixed": 0,
            }
        )
        receipts.append(receipt)
    if not np.isfinite(prediction).all():
        raise ContractError("midpoint OOF prediction incomplete")
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    delta_rmse = rmse(truth, prediction) - rmse(truth, reference)
    scored = frame.assign(candidate=prediction, multiplier=multiplier)
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
    raw_central = -delta_rmse * POINTS_PER_RMSE_M
    calibrated_central = raw_central - PENALTY_POINTS
    changed = np.abs(prediction - reference) > 1e-12
    short = frame["lead_h"].isin([3, 6, 9, 12]).to_numpy()
    improved_blocks = sum(value["delta_rmse_m"] < 0 for value in by_block.values())
    worst = max(value["delta_rmse_m"] for value in station_lead.values())
    changed_share = float(changed.mean())
    checks = {
        "pooled_rmse_improves": delta_rmse < 0,
        "minimum_four_improved_bimonth_blocks": improved_blocks >= 4,
        "episode_ci90_upper_below_zero": episode["ci90_m"][1] < 0,
        "block_station_ci90_upper_below_zero": group["ci90_m"][1] < 0,
        "worst_station_lead_within_0p01m": worst <= MAX_WORST_STATION_LEAD_M,
        "changed_rows_share_at_most_one_third": changed_share <= MAX_CHANGED_SHARE + 1e-12,
        "central_raw_points_meets_family_threshold_inclusive": raw_central >= RAW_THRESHOLD_POINTS,
        "central_calibrated_points_at_least_0p01_inclusive": calibrated_central >= MIN_CALIBRATED_POINTS,
        "short_leads_bit_exact": bool(np.array_equal(prediction[short], reference[short])),
        "six_prefix_ridge_solves": sum(x["ridge_solve_count"] for x in receipts) == 6,
        "fixed_midpoint_weight_unchanged": all(x["midpoint_weight"] == FIXED_WEIGHT for x in receipts),
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
            "local_lcb_raw_points": "descriptive_only_not_transport_gate",
            "selected_penalty_provenance_sha256": sha256(CALIBRATION),
            "fixed_midpoint_weight": FIXED_WEIGHT,
        },
        "surface_claim": "adaptive_182_case_development_surface_with_multiplicity_not_independent_confirmation",
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
        "fit_count": {"prefix_ridge_solves": 6, "model_fits": 0},
        "expected_points": {
            "central_raw": raw_central,
            "transport_penalty_points": PENALTY_POINTS,
            "central_calibrated": calibrated_central,
            "episode_lcb_raw_descriptive": max(0.0, -episode["ci90_m"][1] * POINTS_PER_RMSE_M),
            "block_station_lcb_raw_descriptive": max(0.0, -group["ci90_m"][1] * POINTS_PER_RMSE_M),
        },
        "gate_checks": checks,
        "passed": all(checks.values()),
    }


def materialize(passed: bool, frame: pd.DataFrame, history: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    access = {"official_test_index_rows_read": 0, "official_test_context_rows_read": 0, "official_feature_rows_read": 0, "official_component_prediction_rows_read": 0, "hidden_truth_rows_read": 0, "uploads": 0}
    if not passed:
        return [], access
    training_rank = sequential_energy_rank(frame, history)
    active = frame["lead_h"].isin([18, 24])
    correction = v14b_correction(frame.loc[active], training_rank[active.to_numpy()])
    theta, _ = fit_theta(frame.loc[active], correction)
    official, champion = official_frame()
    energy = pd.read_parquet(TEST_FEATURES, columns=["case_id", "station", "wave_energy_current"])
    official = official.merge(energy, on=["case_id", "station"], how="left", validate="many_to_one")
    if not champion[KEYS].equals(official[KEYS]) or official["wave_energy_current"].isna().any():
        raise ContractError("official key/order/energy contract failed")
    prefix = np.sort(history["wave_energy_current"].to_numpy(dtype=np.float64))
    rank = ecdf(prefix, official["wave_energy_current"].to_numpy(dtype=np.float64))
    official_correction = v14b_correction(official, rank)
    prediction, _ = midpoint_prediction(official, official_correction, theta)
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
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "LAUNCH_READY", "candidate": CANDIDATE}))
        return 0
    if ARTIFACT_DIR.exists() or LOCK.exists():
        raise ContractError("exactly-once artifact or lock exists")
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    duplicate = json.loads(DUPLICATION_AUDIT.read_text(encoding="utf-8"))
    if sha256(CALIBRATION) != CALIBRATION_SHA or calibration["status"] != "FAMILY_AWARE_GUARDRAIL_READY":
        raise ContractError("calibration seal changed")
    if duplicate["status"] != "NON_DUPLICATE_LAUNCH_ELIGIBLE":
        raise ContractError("duplication audit not launch eligible")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runner_hash = sha256(Path(__file__))
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "created_at_utc": datetime.now(UTC).isoformat(), "runner_sha256": runner_hash, "config_sha256": sha256(CONFIG), "calibration_sha256": sha256(CALIBRATION), "duplication_audit_sha256": sha256(DUPLICATION_AUDIT), "fixed_midpoint_weight": FIXED_WEIGHT, "candidate_registration": {"name": CANDIDATE, "family_id": FAMILY_ID, "tier_id": TIER_ID, "representation_changed": False, "routing_discontinuous": False, "active_share_rule": "changed_rows_share_le_1_over_3", "exact_comparator": "uniform_kma_alpha_0.425", "local_lcb_raw_points": "descriptive_only_not_transport_gate", "selected_penalty_provenance_sha256": sha256(CALIBRATION)}}))
    frame, profile = load_historical()
    history = load_energy_history()
    frame = attach_energy(frame, history)
    candidate = evaluate(frame, history)
    outputs, access = materialize(candidate["passed"], frame, history)
    elapsed = time.perf_counter() - started
    result = {"schema_version": "p3.kma_v14b_v16_midpoint.result.v18", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "PASS_MATERIALIZED_NOT_UPLOADED" if candidate["passed"] else "NO_GO_FIXED_MIDPOINT_GATE", "candidate_count": 1, "passing_candidate_count": int(candidate["passed"]), "candidate": candidate, "fit_budget": {"prefix_ridge_solves": 6, "model_fits": 0, "full_deployment_solves": int(candidate["passed"])}, "data_profile": profile, "outputs": outputs, "data_access": access, "provenance": {"runner_sha256": runner_hash, "config_sha256": sha256(CONFIG), "calibration_sha256": sha256(CALIBRATION), "duplication_audit_sha256": sha256(DUPLICATION_AUDIT)}, "execution": {"elapsed_seconds": elapsed, "python": platform.python_version(), "result_based_tuning_or_retry": False, "midpoint_weight_retuned": False, "hidden_truth_rows_read": 0, "upload_attempt_count": 0}}
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report = "# P3 fixed v14b/v16 midpoint v18\n\n## 결론\n\n" + f"- {result['decision']}; PASS {result['passing_candidate_count']}/1; CSV {len(outputs)}; upload 0.\n" + f"- pooled delta {candidate['delta_candidate_minus_reference_rmse_m']:+.9f}m; blocks {candidate['improved_bimonth_blocks']}/6; episode CI {candidate['episode_bootstrap']['ci90_m']}; group CI {candidate['block_station_bootstrap']['ci90_m']}.\n" + f"- central raw {candidate['expected_points']['central_raw']:.9f}; calibrated {candidate['expected_points']['central_calibrated']:.9f} points.\n\n" + "The 0.5 weight was sealed before this run and cannot be retuned. This is adaptive 182-case development evidence with multiplicity, not independent confirmation.\n"
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, report.encode("utf-8"))
    write_new(REPORT_DIR / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "runner_sha256": runner_hash, "result_sha256": sha256(result_path), "report_sha256": sha256(report_path), "outputs": outputs}))
    print(json.dumps({"status": "COMPLETE", "passing": result["passing_candidate_count"], "outputs": len(outputs), "prefix_ridge_solves": 6, "model_fits": 0, "elapsed_seconds": elapsed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
