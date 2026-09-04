"""Sealed P3 v12 prequential S-ORS high-energy ERA5 orthogonal residual cycle."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    CALIBRATION,
    MAX_WORST_STATION_LEAD_M,
    MIN_CALIBRATED_POINTS,
    POINTS_PER_RMSE_M,
    bootstrap,
    canonical,
    load_historical,
    rmse,
    sha256,
    write_new,
)

EXPERIMENT_ID = "p3_sors_era5_orthogonal_prequential_cycle_20260831_v12"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
ERA5 = ROOT / "artifacts/p3_era5_context_transfer_dependency_recovery_20260828_v2/sealed_historical_blind_predictions.parquet"
KMA = ROOT / "artifacts/p3_kma_calibrated_longlead_blend_v2/one_shot/blind_predictions.parquet"
FEATURES = ROOT / "artifacts/p3/features_all20_v1/train_features.parquet"
FOLD_ORDER = ("2024_h2_storm", "winter_transition", "2025_h1")
ENERGY_COLUMNS = ("wave_energy_current", "wave_energy_delta_6h", "wave_energy_std_12h")
CI_TARGET_M = -0.020913058224751535
ENERGY_QUANTILE = 0.67
CORRECTION_CAP_M = 0.25
MIN_TRAIN_SUPPORT = 12
MIN_VALID_SUPPORT = 4
MAX_ACTIVE_SHARE = 1.0 / 3.0


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    summary: str


SPECS = (
    Spec("P3_1_ANALYTIC_HUBER_SLOPE_ORTHOGONAL", "analytic_huber", "IRLS Huber slope on train-only KMA-orthogonal ERA5 innovation."),
    Spec("P3_2_RIDGE100_ENERGY_ORTHOGONAL", "ridge", "Ridge alpha=100 on orthogonal ERA5 and fixed energy interactions."),
    Spec("P3_3_HUBER1P35_ENERGY_ORTHOGONAL", "huber", "Huber epsilon=1.35 alpha=0.001 on the same fixed basis."),
)


def load_table() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, profile = load_historical()
    era5 = pd.read_parquet(ERA5)
    kma = pd.read_parquet(KMA)
    energy = pd.read_parquet(FEATURES, columns=["anchor_id", "station", *ENERGY_COLUMNS])
    keys = ["fold", "anchor_id", "station", "lead_h"]
    frame = frame.merge(
        era5[keys + ["incumbent_prediction", "transfer_prediction"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        kma[keys + ["candidate_final"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(energy, on=["anchor_id", "station"], how="left", validate="many_to_one")
    frame["d_era5"] = frame["transfer_prediction"] - frame["incumbent_prediction"]
    frame["d_kma"] = frame["candidate_final"] - frame["incumbent_final"]
    frame["is_lead24"] = frame["lead_h"].eq(24).astype(float)
    return frame, {**profile, "era5_rows": len(era5), "kma_rows": len(kma), "energy_rows": len(energy)}


def purge(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    keep = np.ones(len(train), dtype=bool)
    train_time = pd.to_datetime(train["anchor_time"], utc=True)
    for timestamp in pd.to_datetime(valid["anchor_time"], utc=True):
        keep &= np.abs(train_time - timestamp) > pd.Timedelta(hours=78)
    return train.loc[keep].copy()


def orthogonalize(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    denom = float(np.square(train["d_kma"]).sum())
    beta = 0.0 if denom <= 1e-12 else float(np.dot(train["d_era5"], train["d_kma"]) / denom)
    train = train.copy()
    valid = valid.copy()
    train["z"] = train["d_era5"] - beta * train["d_kma"]
    valid["z"] = valid["d_era5"] - beta * valid["d_kma"]
    return train, valid, beta


def basis(table: pd.DataFrame) -> np.ndarray:
    z = table["z"].to_numpy(float)
    energy = table["wave_energy_current"].to_numpy(float)
    delta = table["wave_energy_delta_6h"].to_numpy(float)
    std = table["wave_energy_std_12h"].to_numpy(float)
    lead = table["is_lead24"].to_numpy(float)
    return np.column_stack([z, energy, delta, std, lead, z * energy, z * lead])


def analytic_huber_slope(z: np.ndarray, y: np.ndarray) -> float:
    slope = float(np.dot(z, y) / max(np.dot(z, z), 1e-12))
    for _ in range(20):
        residual = y - slope * z
        scale = max(float(np.median(np.abs(residual - np.median(residual)))) * 1.4826, 1e-6)
        weight = np.minimum(1.0, 1.35 * scale / np.maximum(np.abs(residual), 1e-12))
        slope = float(np.dot(weight * z, y) / max(np.dot(weight * z, z), 1e-12))
    return float(np.clip(slope, -2.0, 2.0))


def fit_predict(spec: Spec, train: pd.DataFrame, valid: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
    target = train["target_hs"].to_numpy(float) - train["reference"].to_numpy(float)
    if spec.family == "analytic_huber":
        slope = analytic_huber_slope(train["z"].to_numpy(float), target)
        return slope * valid["z"].to_numpy(float), {"slope": slope}, slope
    if spec.family == "ridge":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
    elif spec.family == "huber":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=500))
    else:
        raise ContractError("unknown family")
    model.fit(basis(train), target)
    return np.asarray(model.predict(basis(valid)), float), {}, model


def evaluate(frame: pd.DataFrame, spec: Spec, penalty: float) -> tuple[dict[str, Any], Any]:
    prediction = frame["reference"].to_numpy(float).copy()
    active = np.zeros(len(frame), dtype=bool)
    receipts: list[dict[str, Any]] = []
    fitted = None
    for fold_index, fold in enumerate(FOLD_ORDER):
        valid_fold = frame["fold"].eq(fold)
        if fold_index == 0:
            receipts.append({"fold": fold, "action": "exact_noop_no_prior_support", "fits": 0})
            continue
        train_fold = frame["fold"].isin(FOLD_ORDER[:fold_index])
        valid_all = frame.loc[valid_fold]
        train_all = purge(frame.loc[train_fold], valid_all)
        finite_train = train_all[["d_era5", "d_kma", *ENERGY_COLUMNS]].notna().all(axis=1)
        finite_valid = valid_all[["d_era5", "d_kma", *ENERGY_COLUMNS]].notna().all(axis=1)
        train_base = train_all.loc[finite_train & train_all["station"].eq("S-ORS") & train_all["lead_h"].isin([18, 24])].copy()
        valid_base = valid_all.loc[finite_valid & valid_all["station"].eq("S-ORS") & valid_all["lead_h"].isin([18, 24])].copy()
        threshold = float(train_base["wave_energy_current"].quantile(ENERGY_QUANTILE))
        train = train_base.loc[train_base["wave_energy_current"] >= threshold].copy()
        valid = valid_base.loc[valid_base["wave_energy_current"] >= threshold].copy()
        if len(train) < MIN_TRAIN_SUPPORT or len(valid) < MIN_VALID_SUPPORT:
            receipts.append({"fold": fold, "action": "exact_noop_insufficient_support", "train_support": len(train), "valid_support": len(valid), "fits": 0})
            continue
        train, valid, beta = orthogonalize(train, valid)
        correction, meta, fitted = fit_predict(spec, train, valid)
        cap = np.minimum(np.abs(valid["z"].to_numpy(float)), CORRECTION_CAP_M)
        correction = np.clip(correction, -cap, cap)
        prediction[valid.index] = valid["reference"].to_numpy(float) + correction
        active[valid.index] = np.abs(correction) > 0
        receipts.append({"fold": fold, "action": "prequential_fit", "train_support": len(train), "valid_support": len(valid), "energy_q67": threshold, "orthogonal_beta": beta, "active_rows": int(np.sum(np.abs(correction) > 0)), "fits": 1, **meta})
    truth = frame["target_hs"].to_numpy(float)
    reference = frame["reference"].to_numpy(float)
    delta = rmse(truth, prediction) - rmse(truth, reference)
    by_fold: dict[str, Any] = {}
    for fold, part in frame.assign(candidate=prediction).groupby("fold", observed=True, sort=True):
        by_fold[str(fold)] = {"rows": len(part), "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy()) - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy())}
    station_lead: dict[str, Any] = {}
    for (station, lead), part in frame.assign(candidate=prediction).groupby(["station", "lead_h"], observed=True, sort=True):
        station_lead[f"{station}|{int(lead)}"] = {"rows": len(part), "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy()) - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy())}
    episode = bootstrap(frame, prediction, ("episode_id",), 20260831)
    group = bootstrap(frame, prediction, ("fold", "station"), 20260832)
    raw = max(0.0, -episode["ci90_m"][1] * POINTS_PER_RMSE_M)
    evaluable = [receipt["fold"] for receipt in receipts if receipt["action"] == "prequential_fit"]
    lead_nonworse = all(station_lead[f"S-ORS|{lead}"]["delta_rmse_m"] <= 0 for lead in (18, 24))
    active_share = float(active.mean())
    worst = max(item["delta_rmse_m"] for item in station_lead.values())
    checks = {
        "pooled_rmse_improves": delta < 0,
        "episode_ci90_upper_at_most_minus_0p020913058": episode["ci90_m"][1] <= CI_TARGET_M,
        "group_ci90_upper_below_zero": group["ci90_m"][1] < 0,
        "both_evaluable_folds_improve": len(evaluable) == 2 and all(by_fold[fold]["delta_rmse_m"] < 0 for fold in evaluable),
        "sors_18_24_nonworse": lead_nonworse,
        "worst_station_lead_within_0p01m": worst <= MAX_WORST_STATION_LEAD_M,
        "active_at_least_12": int(active.sum()) >= 12,
        "active_share_at_most_one_third": active_share <= MAX_ACTIVE_SHARE,
        "raw_conservative_points_at_least_0p331905690": raw >= penalty + MIN_CALIBRATED_POINTS,
        "calibrated_conservative_points_at_least_0p01": raw - penalty >= MIN_CALIBRATED_POINTS,
        "inactive_exact_noop": bool(np.array_equal(prediction[~active], reference[~active])),
        "finite_predictions": bool(np.isfinite(prediction).all()),
    }
    return {"spec": asdict(spec), "fit_receipts": receipts, "historical_fit_count": sum(x["fits"] for x in receipts), "reference_rmse_m": rmse(truth, reference), "candidate_rmse_m": rmse(truth, prediction), "delta_candidate_minus_reference_rmse_m": delta, "active_rows": int(active.sum()), "active_share": active_share, "by_fold": by_fold, "station_lead": station_lead, "worst_station_lead_delta_rmse_m": worst, "episode_bootstrap": episode, "group_bootstrap": group, "expected_points": {"raw_central": -delta * POINTS_PER_RMSE_M, "raw_conservative": raw, "public_reversal_penalty": penalty, "calibrated_conservative": raw - penalty}, "gate_checks": checks, "passed": all(checks.values())}, fitted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--technical-recovery", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "specs": [asdict(x) for x in SPECS]}))
        return 0
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=args.technical_recovery)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runner_hash = sha256(Path(__file__))
    if args.technical_recovery:
        if not LOCK.exists() or (ARTIFACT_DIR / "result.json").exists():
            raise ContractError("technical recovery guard failed")
        original = json.loads(LOCK.read_text(encoding="utf-8"))
        write_new(ARTIFACT_DIR / "technical_recovery.json", canonical({"experiment_id": EXPERIMENT_ID, "initial_runner_sha256": original["runner_sha256"], "corrected_runner_sha256": runner_hash, "root_cause": "historical frame already contains incumbent_final; KMA merge produced suffixes", "repair": "merge only candidate_final and retain the authoritative historical incumbent_final", "fits_before_failure": 0, "predictions_before_failure": 0, "official_rows_read_before_failure": 0, "candidate_or_gate_changes": 0}))
    else:
        write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "created_at_utc": datetime.now(UTC).isoformat(), "runner_sha256": runner_hash, "maximum_fits": 9, "sealed_support": {"station": "S-ORS", "leads": [18, 24], "energy_quantile": ENERGY_QUANTILE, "purge_hours": 78, "fold_order": FOLD_ORDER}}))
    frame, profile = load_table()
    penalty = float(json.loads(CALIBRATION.read_text(encoding="utf-8"))["gates"]["P3"]["transport_penalty_points"])
    evaluated = [evaluate(frame, spec, penalty) for spec in SPECS]
    candidates = [item[0] for item in evaluated]
    passing = [item for item in candidates if item["passed"]]
    if passing:
        raise ContractError("unexpected PASS: official materialization requires separately frozen component transport")
    result = {"schema_version": "p3.sors_era5_orthogonal_prequential.result.v12", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "NO_GO_PUBLIC_TRANSPORT_GATE", "candidate_count": 3, "passing_candidate_count": 0, "candidates": candidates, "fit_budget": {"maximum": 9, "actual_historical": sum(x["historical_fit_count"] for x in candidates), "actual_full": 0}, "data_profile": profile, "transport": {"penalty_points": penalty, "minimum_raw_points": penalty + MIN_CALIBRATED_POINTS, "minimum_calibrated_points": MIN_CALIBRATED_POINTS, "equivalent_ci_upper_m": CI_TARGET_M}, "outputs": [], "data_access": {"official_test_index_rows_read": 0, "official_feature_rows_read": 0, "hidden_truth_rows_read": 0, "uploads": 0}, "provenance": {"runner_sha256": runner_hash, "era5_sha256": sha256(ERA5), "kma_sha256": sha256(KMA), "feature_sha256": sha256(FEATURES), "calibration_sha256": sha256(CALIBRATION)}, "execution": {"elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(), "result_based_tuning_or_retry": False, "upload_attempt_count": 0}}
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report = "# P3 S-ORS ERA5 orthogonal prequential v12\n\n## 결론\n\n- calibrated PASS: **0/3**\n- CSV 0, upload 0\n\n" + "\n".join(f"- {x['spec']['name']}: delta={x['delta_candidate_minus_reference_rmse_m']:.6f}m, active={x['active_rows']}, CI90 upper={x['episode_bootstrap']['ci90_m'][1]:.6f}, PASS={x['passed']}" for x in candidates) + "\n"
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, report.encode())
    write_new(REPORT_DIR / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "runner_sha256": runner_hash, "result_sha256": sha256(result_path), "report_sha256": sha256(report_path), "outputs": []}))
    print(json.dumps({"status": "COMPLETE", "passing": 0, "fits": result["fit_budget"]["actual_historical"], "elapsed_seconds": result["execution"]["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
