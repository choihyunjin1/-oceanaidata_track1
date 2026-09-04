"""Run sealed train-prefix steepness ECDF KMA factor validation for P3."""

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

from run_p3_kma_continuous_energy_affine_cycle_20260831_v15b import (  # noqa: E402
    block_order,
)
from run_p3_kma_continuous_energy_factor_cycle_20260831_v14b import (  # noqa: E402
    CALIBRATION,
    CALIBRATION_SHA,
    KEYS,
    POINTS_PER_RMSE_M,
    bootstrap,
    canonical,
    ecdf,
    load_historical,
    official_frame,
    rmse,
    sha256,
    write_new,
)

EXPERIMENT_ID = "p3_kma_continuous_steepness_factor_cycle_20260831_v17"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CONFIG = ROOT / "configs/experiments/p3_kma_continuous_steepness_factor_cycle_20260831_v17.json"
DUPLICATION_AUDIT = REPORT_DIR / "duplication-audit.json"
TRAIN_FEATURES = ROOT / "artifacts/p3/features_all20_v1/train_features.parquet"
TRAIN_ANCHORS = ROOT / "artifacts/p3/features_all20_v1/train_anchors.parquet"
TEST_FEATURES = ROOT / "artifacts/p3/features_all20_v1/test_features.parquet"
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_KMA_CONTINUOUS_STEEPNESS_V17"
)

CANDIDATE = "P3_1_KMA_CONTINUOUS_STEEPNESS_FACTOR"
FAMILY_ID = "P3_FIXED_KMA_LONGLEAD_FACTOR_CONTINUOUS_STEEPNESS"
TIER_ID = "LOW_DOF_FIXED"
REFERENCE_ALPHA = 0.425
ALPHA_18 = 0.20
TP_VALID_MIN = 0.10
DENOMINATOR_EPS = 0.01
NEUTRAL_ECDF = (REFERENCE_ALPHA - 0.20) / 0.40
PURGE_HOURS = 78
PENALTY_POINTS = 0.04958605409228893
RAW_THRESHOLD_POINTS = 0.05958605409228893
MIN_CALIBRATED_POINTS = 0.01
MAX_WORST_STATION_LEAD_M = 0.01
MAX_CHANGED_SHARE = 1.0 / 3.0


class ContractError(RuntimeError):
    """Raised when a sealed steepness-cycle invariant is violated."""


def compute_steepness(hs: np.ndarray, tp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hs_array = np.asarray(hs, dtype=np.float64)
    tp_array = np.asarray(tp, dtype=np.float64)
    valid = np.isfinite(hs_array) & np.isfinite(tp_array) & (hs_array >= 0) & (tp_array > TP_VALID_MIN)
    z = np.full(len(hs_array), np.nan, dtype=np.float64)
    z[valid] = np.log1p(hs_array[valid] / np.maximum(np.square(tp_array[valid]), DENOMINATOR_EPS))
    return z, valid


def load_steepness_history() -> pd.DataFrame:
    features = pd.read_parquet(
        TRAIN_FEATURES,
        columns=["anchor_id", "station", "hs_current", "tp_current"],
    )
    anchors = pd.read_parquet(
        TRAIN_ANCHORS,
        columns=["anchor_id", "station", "anchor_time"],
    )
    history = features.merge(anchors, on=["anchor_id", "station"], validate="one_to_one")
    history["anchor_time"] = pd.to_datetime(history["anchor_time"], utc=True)
    history["steepness_z"], history["steepness_valid"] = compute_steepness(
        history["hs_current"].to_numpy(), history["tp_current"].to_numpy()
    )
    if len(history) != len(features) or history.duplicated(["anchor_id", "station"]).any():
        raise ContractError("steepness history key contract failed")
    if not history.loc[history["steepness_valid"], "steepness_z"].map(np.isfinite).all():
        raise ContractError("valid steepness contains non-finite values")
    return history


def attach_steepness(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    columns = ["anchor_id", "station", "steepness_z", "steepness_valid"]
    merged = frame.merge(history[columns], on=["anchor_id", "station"], how="left", validate="many_to_one")
    if merged["steepness_valid"].isna().any():
        raise ContractError("historical steepness mapping missing")
    return merged


def ranks_from_prefix(prefix: np.ndarray, z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rank = np.full(len(z), NEUTRAL_ECDF, dtype=np.float64)
    if np.any(valid):
        rank[valid] = ecdf(prefix, z[valid])
    return rank


def predict_policy(frame: pd.DataFrame, rank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
    alpha = np.zeros(len(frame), dtype=np.float64)
    lead18 = frame["lead_h"].eq(18).to_numpy()
    lead24 = frame["lead_h"].eq(24).to_numpy()
    alpha[lead18] = ALPHA_18
    alpha[lead24] = 0.20 + 0.40 * rank[lead24]
    for mask in (lead18, lead24):
        prediction[mask] = (
            frame.loc[mask, "base"].to_numpy(dtype=np.float64)
            + alpha[mask] * frame.loc[mask, "delta"].to_numpy(dtype=np.float64)
        )
    return np.clip(prediction, 0.0, 30.0), alpha


def evaluate(frame: pd.DataFrame, history: pd.DataFrame) -> dict[str, Any]:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    alpha = np.zeros(len(frame), dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    for block in block_order(frame):
        valid_mask = frame["block"].eq(block).to_numpy()
        valid_frame = frame.loc[valid_mask]
        boundary = pd.Timestamp(valid_frame["anchor_time"].min()) - pd.Timedelta(hours=PURGE_HOURS)
        prefix_frame = history.loc[history["anchor_time"].le(boundary) & history["steepness_valid"]]
        prefix = np.sort(prefix_frame["steepness_z"].to_numpy(dtype=np.float64))
        valid = valid_frame["steepness_valid"].to_numpy(dtype=bool)
        rank = ranks_from_prefix(
            prefix,
            valid_frame["steepness_z"].to_numpy(dtype=np.float64),
            valid,
        )
        fold_prediction, fold_alpha = predict_policy(valid_frame, rank)
        prediction[valid_mask] = fold_prediction
        alpha[valid_mask] = fold_alpha
        receipts.append(
            {
                "block": block,
                "boundary_utc": boundary.isoformat(),
                "prefix_valid_cases": int(len(prefix)),
                "validation_invalid_rows": int((~valid).sum()),
                "ecdf_calibration_fits": 1,
                "outer_target_rows_read_before_policy_fixed": 0,
            }
        )
    if not np.isfinite(prediction).all():
        raise ContractError("steepness OOF prediction incomplete")
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
    raw_central = -delta_rmse * POINTS_PER_RMSE_M
    calibrated_central = raw_central - PENALTY_POINTS
    comparator_alpha = np.where(frame["lead_h"].isin([18, 24]), REFERENCE_ALPHA, 0.0)
    changed = np.abs(alpha - comparator_alpha) > 1e-12
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
        "six_prefix_ecdf_calibrations": sum(x["ecdf_calibration_fits"] for x in receipts) == 6,
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
        },
        "surface_claim": "adaptive_182_case_development_surface_not_independent_confirmation",
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
        "fit_count": {"ecdf_calibration_fits": 6, "model_fits": 0},
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


def materialize(passed: bool, history: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    access = {"official_test_index_rows_read": 0, "official_test_context_rows_read": 0, "official_feature_rows_read": 0, "official_component_prediction_rows_read": 0, "hidden_truth_rows_read": 0, "uploads": 0}
    if not passed:
        return [], access
    official, champion = official_frame()
    features = pd.read_parquet(TEST_FEATURES, columns=["case_id", "station", "hs_current", "tp_current"])
    z, valid = compute_steepness(features["hs_current"].to_numpy(), features["tp_current"].to_numpy())
    features = features.assign(steepness_z=z, steepness_valid=valid)
    official = official.merge(features[["case_id", "station", "steepness_z", "steepness_valid"]], on=["case_id", "station"], how="left", validate="many_to_one")
    if not champion[KEYS].equals(official[KEYS]) or official["steepness_valid"].isna().any():
        raise ContractError("official key/order/steepness contract failed")
    prefix = np.sort(history.loc[history["steepness_valid"], "steepness_z"].to_numpy(dtype=np.float64))
    rank = ranks_from_prefix(prefix, official["steepness_z"].to_numpy(dtype=np.float64), official["steepness_valid"].to_numpy(dtype=bool))
    prediction, _ = predict_policy(official, rank)
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
        raise ContractError("duplication audit is not launch eligible")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runner_hash = sha256(Path(__file__))
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "created_at_utc": datetime.now(UTC).isoformat(), "runner_sha256": runner_hash, "config_sha256": sha256(CONFIG), "calibration_sha256": sha256(CALIBRATION), "duplication_audit_sha256": sha256(DUPLICATION_AUDIT), "candidate_registration": {"name": CANDIDATE, "family_id": FAMILY_ID, "tier_id": TIER_ID, "representation_changed": False, "routing_discontinuous": False, "active_share_rule": "changed_rows_share_le_1_over_3", "exact_comparator": "uniform_kma_alpha_0.425", "local_lcb_raw_points": "descriptive_only_not_transport_gate", "selected_penalty_provenance_sha256": sha256(CALIBRATION)}}))
    frame, profile = load_historical()
    history = load_steepness_history()
    frame = attach_steepness(frame, history)
    candidate = evaluate(frame, history)
    outputs, access = materialize(candidate["passed"], history)
    elapsed = time.perf_counter() - started
    result = {"schema_version": "p3.kma_continuous_steepness_factor.result.v17", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "PASS_MATERIALIZED_NOT_UPLOADED" if candidate["passed"] else "NO_GO_CONTINUOUS_STEEPNESS_GATE", "candidate_count": 1, "passing_candidate_count": int(candidate["passed"]), "candidate": candidate, "fit_budget": {"ecdf_calibration_fits": 6, "model_fits": 0, "full_deployment_ecdf_fits": int(candidate["passed"])}, "data_profile": profile, "outputs": outputs, "data_access": access, "provenance": {"runner_sha256": runner_hash, "config_sha256": sha256(CONFIG), "calibration_sha256": sha256(CALIBRATION), "duplication_audit_sha256": sha256(DUPLICATION_AUDIT)}, "execution": {"elapsed_seconds": elapsed, "python": platform.python_version(), "result_based_tuning_or_retry": False, "hidden_truth_rows_read": 0, "upload_attempt_count": 0}}
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report = "# P3 continuous-steepness KMA factor v17\n\n## 결론\n\n" + f"- {result['decision']}; PASS {result['passing_candidate_count']}/1; CSV {len(outputs)}; upload 0.\n" + f"- pooled delta {candidate['delta_candidate_minus_reference_rmse_m']:+.9f}m; blocks {candidate['improved_bimonth_blocks']}/6; episode CI {candidate['episode_bootstrap']['ci90_m']}; group CI {candidate['block_station_bootstrap']['ci90_m']}.\n" + f"- central raw {candidate['expected_points']['central_raw']:.9f}; calibrated {candidate['expected_points']['central_calibrated']:.9f} points. CI LCB is descriptive, not subtracted from the transport gate.\n\n" + "This is an adaptive 182-case development surface, not an independent confirmation.\n"
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, report.encode("utf-8"))
    write_new(REPORT_DIR / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "runner_sha256": runner_hash, "result_sha256": sha256(result_path), "report_sha256": sha256(report_path), "outputs": outputs}))
    print(json.dumps({"status": "COMPLETE", "passing": result["passing_candidate_count"], "outputs": len(outputs), "ecdf_calibration_fits": 6, "model_fits": 0, "elapsed_seconds": elapsed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
