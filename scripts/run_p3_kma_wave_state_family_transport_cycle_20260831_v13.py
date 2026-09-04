"""Run the sealed family-aware P3 fixed KMA factor and wave-state policies."""

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

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

from run_p3_parallel_candidate_cycle_20260831_v4 import (  # noqa: E402
    ACTIVE_LEADS,
    KEYS,
    official_frame,
    purge_training_cases,
    rmse,
)
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    POINTS_PER_RMSE_M,
    bootstrap,
    canonical,
    load_historical,
    sha256,
    write_new,
)

EXPERIMENT_ID = "p3_kma_wave_state_family_transport_cycle_20260831_v13"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CONFIG = ROOT / "configs/experiments/p3_kma_wave_state_family_transport_cycle_20260831_v13.json"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v2/calibration.json"
TRAIN_FEATURES = ROOT / "artifacts/p3/features_all20_v1/train_features.parquet"
TEST_FEATURES = ROOT / "artifacts/p3/features_all20_v1/test_features.parquet"
DELIVERY = Path(r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_KMA_WAVE_STATE_FAMILY_V13")
REFERENCE_ALPHA = 0.425
MIN_CALIBRATED_POINTS = 0.01
MIN_IMPROVED_BLOCKS = 4
MAX_WORST_STATION_LEAD_M = 0.01
MAX_ACTIVE_SHARE = 1.0 / 3.0
MIN_CONDITIONAL_ACTIVE = 12


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Spec:
    name: str
    alpha_24: float
    support: str
    quantile: float | None
    family_id: str
    tier_id: str
    penalty: float
    raw_threshold: float
    routing_discontinuous: bool


SPECS = (
    Spec("P3_1_FIXED_KMA_A18_0425_A24_0600", 0.6, "all", None, "P3_FIXED_KMA_LONGLEAD_FACTOR", "LOW_DOF_FIXED", 0.04958605409228893, 0.05958605409228893, False),
    Spec("P3_2_HIGH_ENERGY_KMA_A24_0600", 0.6, "high", 0.67, "P3_KMA_HIGH_ENERGY_HARD_ROUTER", "HARD_CONDITIONAL_ROUTER", 0.3219056897594759, 0.33190568975947593, True),
    Spec("P3_3_LOW_ENERGY_KMA_A24_0200", 0.2, "low", 0.33, "P3_KMA_LOW_ENERGY_HARD_ROUTER", "HARD_CONDITIONAL_ROUTER", 0.3219056897594759, 0.33190568975947593, True),
)


def attach_energy(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    key = "anchor_id" if "anchor_id" in frame.columns else "case_id"
    columns = [key, "station", "wave_energy_current"]
    features = pd.read_parquet(path, columns=columns)
    if features.duplicated([key, "station"]).any():
        raise ContractError("energy feature key duplicated")
    return frame.merge(features, on=[key, "station"], how="left", validate="many_to_one")


def threshold_from_train(train: pd.DataFrame, quantile: float) -> float:
    cases = train[["anchor_id", "station", "wave_energy_current"]].drop_duplicates()
    if cases["wave_energy_current"].isna().any():
        raise ContractError("train energy support missing")
    return float(cases["wave_energy_current"].quantile(quantile))


def policy_prediction(frame: pd.DataFrame, spec: Spec, threshold: float | None) -> tuple[np.ndarray, np.ndarray]:
    prediction = frame["reference"].to_numpy(float).copy()
    alpha = np.where(frame["lead_h"].isin(ACTIVE_LEADS), REFERENCE_ALPHA, 0.0).astype(float)
    lead24 = frame["lead_h"].eq(24).to_numpy()
    if spec.support == "all":
        selected = lead24
    elif spec.support == "high":
        selected = lead24 & frame["wave_energy_current"].ge(float(threshold)).to_numpy()
    elif spec.support == "low":
        selected = lead24 & frame["wave_energy_current"].le(float(threshold)).to_numpy()
    else:
        raise ContractError("unknown support")
    alpha[selected] = spec.alpha_24
    prediction[selected] = frame.loc[selected, "base"].to_numpy(float) + spec.alpha_24 * frame.loc[selected, "delta"].to_numpy(float)
    prediction = np.clip(prediction, 0.0, 30.0)
    return prediction, alpha


def evaluate(frame: pd.DataFrame, spec: Spec) -> tuple[dict[str, Any], float | None]:
    prediction = np.full(len(frame), np.nan)
    alpha = np.full(len(frame), np.nan)
    receipts: list[dict[str, Any]] = []
    blocks = sorted(frame["block"].unique())
    for block in blocks:
        valid_mask = frame["block"].eq(block).to_numpy()
        valid = frame.loc[valid_mask]
        threshold = None
        if spec.quantile is not None:
            train = purge_training_cases(frame.loc[~valid_mask], valid)
            threshold = threshold_from_train(train, spec.quantile)
            receipts.append({"block": block, "action": "train_only_energy_quantile", "train_rows_after_78h_purge": len(train), "threshold": threshold, "calibration_fits": 1})
        else:
            receipts.append({"block": block, "action": "fixed_factor_no_fit", "calibration_fits": 0})
        fold_prediction, fold_alpha = policy_prediction(valid, spec, threshold)
        prediction[valid_mask] = fold_prediction
        alpha[valid_mask] = fold_alpha
    if not np.isfinite(prediction).all():
        raise ContractError("OOF policy incomplete")
    truth = frame["target_hs"].to_numpy(float)
    reference = frame["reference"].to_numpy(float)
    delta = rmse(truth, prediction) - rmse(truth, reference)
    by_block: dict[str, Any] = {}
    for block, part in frame.assign(candidate=prediction).groupby("block", observed=True, sort=True):
        by_block[str(block)] = {"rows": len(part), "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy()) - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy())}
    station_lead: dict[str, Any] = {}
    for (station, lead), part in frame.assign(candidate=prediction).groupby(["station", "lead_h"], observed=True, sort=True):
        station_lead[f"{station}|{int(lead)}"] = {"rows": len(part), "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy()) - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy())}
    episode = bootstrap(frame, prediction, ("episode_id",), 20260831)
    group = bootstrap(frame, prediction, ("block", "station"), 20260832)
    raw = max(0.0, -episode["ci90_m"][1] * POINTS_PER_RMSE_M)
    changed = np.abs(alpha - np.where(frame["lead_h"].isin(ACTIVE_LEADS), REFERENCE_ALPHA, 0.0)) > 1e-12
    active_share = float(changed.mean())
    improved_blocks = sum(item["delta_rmse_m"] < 0 for item in by_block.values())
    worst = max(item["delta_rmse_m"] for item in station_lead.values())
    checks = {
        "pooled_rmse_improves": delta < 0,
        "episode_ci90_upper_below_zero": episode["ci90_m"][1] < 0,
        "group_ci90_upper_below_zero": group["ci90_m"][1] < 0,
        "minimum_four_improved_bimonth_blocks": improved_blocks >= MIN_IMPROVED_BLOCKS,
        "worst_station_lead_within_0p01m": worst <= MAX_WORST_STATION_LEAD_M,
        "active_share_at_most_one_third": active_share <= MAX_ACTIVE_SHARE,
        "conditional_active_at_least_12": spec.support == "all" or int(changed.sum()) >= MIN_CONDITIONAL_ACTIVE,
        "local_lcb_raw_points_meets_family_threshold": raw >= spec.raw_threshold,
        "calibrated_expected_points_at_least_0p01": raw - spec.penalty >= MIN_CALIBRATED_POINTS,
        "finite_predictions": bool(np.isfinite(prediction).all()),
    }
    full_threshold = None if spec.quantile is None else threshold_from_train(frame, spec.quantile)
    return {"spec": {**spec.__dict__, "representation_changed": False, "exact_comparator": "uniform_kma_alpha_0.425", "active_share_rule": "changed_rows_share_le_1_over_3", "selected_penalty_provenance_sha256": sha256(CALIBRATION)}, "calibration_receipts": receipts, "train_only_calibration_fit_count": sum(x["calibration_fits"] for x in receipts), "reference_rmse_m": rmse(truth, reference), "candidate_rmse_m": rmse(truth, prediction), "delta_candidate_minus_reference_rmse_m": delta, "changed_rows": int(changed.sum()), "active_share": active_share, "improved_bimonth_blocks": improved_blocks, "by_block": by_block, "station_lead": station_lead, "worst_station_lead_delta_rmse_m": worst, "episode_bootstrap": episode, "group_bootstrap": group, "expected_points": {"raw_central": -delta * POINTS_PER_RMSE_M, "local_lcb_raw_points": raw, "selected_transport_penalty_points": spec.penalty, "calibrated_conservative": raw - spec.penalty}, "gate_checks": checks, "passed": all(checks.values())}, full_threshold


def materialize(passing: list[tuple[dict[str, Any], float | None]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not passing:
        return [], {"official_test_index_rows_read": 0, "official_test_context_rows_read": 0, "official_feature_rows_read": 0, "official_component_prediction_rows_read": 0, "hidden_truth_rows_read": 0, "uploads": 0}
    frame, champion = official_frame()
    frame = frame.merge(pd.read_parquet(TEST_FEATURES, columns=["case_id", "station", "wave_energy_current"]), on=["case_id", "station"], validate="many_to_one")
    if not champion[KEYS].equals(frame[KEYS]):
        raise ContractError("official order changed")
    DELIVERY.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    for record, threshold in passing[:3]:
        spec = next(item for item in SPECS if item.name == record["spec"]["name"])
        prediction, _ = policy_prediction(frame, spec, threshold)
        submission = frame[KEYS].copy()
        submission["hs_pred"] = prediction
        payload = submission.to_csv(index=False, lineterminator="\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        directory = DELIVERY / spec.name
        directory.mkdir(parents=True)
        path = directory / "P3_submission.csv"
        write_new(path, payload)
        outputs.append({"candidate": spec.name, "path": str(path), "rows": len(submission), "sha256": digest, "uploads": 0})
    write_new(DELIVERY / "SET_MANIFEST.json", canonical({"experiment_id": EXPERIMENT_ID, "outputs": outputs, "uploads": 0}))
    return outputs, {"official_test_index_rows_read": 1200, "official_test_context_rows_read": 200, "official_feature_rows_read": 200, "official_component_prediction_rows_read": 3600, "hidden_truth_rows_read": 0, "uploads": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "candidates": [x.name for x in SPECS]}))
        return 0
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if sha256(CALIBRATION) != "1a1d2c96cbe6c2c69b753fb4a130eb092922cc46524286cabcc0f9fce150441f" or calibration["status"] != "FAMILY_AWARE_GUARDRAIL_READY":
        raise ContractError("family calibration seal changed")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runner_hash = sha256(Path(__file__))
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "created_at_utc": datetime.now(UTC).isoformat(), "runner_sha256": runner_hash, "config_sha256": sha256(CONFIG), "calibration_sha256": sha256(CALIBRATION), "candidate_registration": [{"name": x.name, "family_id": x.family_id, "tier_id": x.tier_id, "representation_changed": False, "routing_discontinuous": x.routing_discontinuous, "active_share_rule": "changed_rows_share_le_1_over_3", "exact_comparator": "uniform_kma_alpha_0.425", "local_lcb_raw_points": "episode_bootstrap_ci90_upper", "selected_penalty_provenance_sha256": sha256(CALIBRATION)} for x in SPECS]}))
    frame, profile = load_historical()
    frame = attach_energy(frame, TRAIN_FEATURES)
    evaluated = [evaluate(frame, spec) for spec in SPECS]
    candidates = [item[0] for item in evaluated]
    passing = [(item[0], item[1]) for item in evaluated if item[0]["passed"]]
    outputs, access = materialize(passing)
    result = {"schema_version": "p3.kma_wave_state_family_transport.result.v13", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "PASS_MATERIALIZED_NOT_UPLOADED" if passing else "NO_GO_FAMILY_AWARE_TRANSPORT_GATE", "candidate_count": 3, "passing_candidate_count": len(passing), "candidates": candidates, "fit_budget": {"train_only_calibration_fits": sum(x["train_only_calibration_fit_count"] for x in candidates), "full_threshold_calibrations": len([x for x in passing if x[0]["spec"]["support"] != "all"])}, "data_profile": profile, "outputs": outputs, "data_access": access, "provenance": {"runner_sha256": runner_hash, "config_sha256": sha256(CONFIG), "calibration_sha256": sha256(CALIBRATION), "train_features_sha256": sha256(TRAIN_FEATURES)}, "execution": {"elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(), "result_based_tuning_or_retry": False, "hidden_truth_rows_read": 0, "upload_attempt_count": 0}}
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report = "# P3 family-aware KMA factor v13\n\n## 결론\n\n" + f"- PASS: **{len(passing)}/3**, CSV {len(outputs)}, upload 0\n\n" + "\n".join(f"- {x['spec']['name']}: delta={x['delta_candidate_minus_reference_rmse_m']:.6f}m, LCB raw={x['expected_points']['local_lcb_raw_points']:.6f}, calibrated={x['expected_points']['calibrated_conservative']:.6f}, PASS={x['passed']}" for x in candidates) + "\n"
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, report.encode())
    write_new(REPORT_DIR / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "runner_sha256": runner_hash, "result_sha256": sha256(result_path), "report_sha256": sha256(report_path), "outputs": outputs}))
    print(json.dumps({"status": "COMPLETE", "passing": len(passing), "outputs": len(outputs), "fits": result["fit_budget"]["train_only_calibration_fits"], "elapsed_seconds": result["execution"]["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
