"""Run the sealed P3 v19 plus train-only wind-work residual-axis experiment."""

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
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

import run_p3_kma_continuous_wave_power_factor_cycle_20260831_v19 as v19  # noqa: E402
from run_p3_kma_continuous_energy_affine_cycle_20260831_v15b import (  # noqa: E402
    block_order,
)
from run_p3_parallel_candidate_cycle_20260831_v4 import (  # noqa: E402
    purge_training_cases,
    rmse,
)
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    POINTS_PER_RMSE_M,
    bootstrap,
    canonical,
    sha256,
    write_new,
)

EXPERIMENT_ID = "p3_kma_wind_work_residual_axis_cycle_20260901_v20"
CANDIDATE = "P3_1_V19_WIND_WORK_RESIDUAL_AXIS"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
TRAIN_FEATURES = ROOT / "artifacts/p3/features_all20_v1/train_features.parquet"

ACTIVE_LEADS = (18, 24)
RAW_FEATURES = (
    "wind_input_proxy_current",
    "wind_wave_alignment_current",
    "wspd_change_6h",
    "caph_change_6h",
    "gust_excess_current",
)
RIDGE_ALPHA = 64.0
TRUST = 0.50
RAW_ADJUSTMENT_CAP = 0.20
ALPHA_MIN = 0.0
ALPHA_MAX = 0.65
PURGE_HOURS = 78
BOOTSTRAP_REPLICATES = 5_000
OFFICIAL_CHAMPION_POINTS = 24.203599
TRANSPORT_PENALTY_POINTS = 0.04958605409228893


class ContractError(RuntimeError):
    """Raised when the sealed v20 contract is violated."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "schema": config["schema_version"] == "p3.kma_wind_work_residual_axis.config.v20",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "candidate": config["candidate"]["name"] == CANDIDATE,
        "features": tuple(config["candidate"]["features"])
        == (
            "log1p_nonnegative_wind_input_proxy_current",
            "wind_wave_alignment_current",
            "wspd_change_6h",
            "caph_change_6h",
            "gust_excess_current",
        ),
        "ridge": "Ridge(alpha=64" in config["candidate"]["model"],
        "no_search": config["candidate"]["grid_or_search"] is False,
        "fits": config["validation"]["outer_model_fits"] == 6,
        "purge": config["validation"]["purge_hours"] == PURGE_HOURS,
        "non_duplicate": config["duplication_audit"]["verdict"] == "NON_DUPLICATE",
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
    }
    if not all(checks.values()):
        raise ContractError(f"v20 config contract failed: {checks}")
    return config


def transformed_raw_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [name for name in RAW_FEATURES if name not in frame.columns]
    if missing:
        raise ContractError(f"wind-work features missing: {missing}")
    raw = frame.loc[:, RAW_FEATURES].to_numpy(dtype=np.float64, copy=True)
    raw[:, 0] = np.where(
        np.isfinite(raw[:, 0]),
        np.log1p(np.maximum(raw[:, 0], 0.0)),
        np.nan,
    )
    raw[:, 1] = np.where(np.isfinite(raw[:, 1]), np.clip(raw[:, 1], -1.0, 1.0), np.nan)
    eligible = np.isfinite(raw).sum(axis=1) >= 2
    return raw, eligible


def fit_transformer(raw: np.ndarray) -> dict[str, np.ndarray]:
    median = np.nanmedian(raw, axis=0)
    q25 = np.nanpercentile(raw, 25, axis=0)
    q75 = np.nanpercentile(raw, 75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    median = np.where(np.isfinite(median), median, 0.0)
    missing = ~np.isfinite(raw)
    imputed = np.where(missing, median, raw)
    scaled = np.clip((imputed - median) / scale, -4.0, 4.0)
    combined = np.column_stack([scaled, missing.astype(np.float64)])
    center = combined.mean(axis=0)
    return {"median": median, "scale": scale, "center": center}


def apply_transformer(raw: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    missing = ~np.isfinite(raw)
    imputed = np.where(missing, state["median"], raw)
    scaled = np.clip((imputed - state["median"]) / state["scale"], -4.0, 4.0)
    combined = np.column_stack([scaled, missing.astype(np.float64)])
    output = combined - state["center"]
    if not np.isfinite(output).all():
        raise ContractError("wind-work transformed basis is non-finite")
    return output


def load_wind_features() -> pd.DataFrame:
    columns = ["anchor_id", "station", *RAW_FEATURES]
    features = pd.read_parquet(TRAIN_FEATURES, columns=columns)
    if features.duplicated(["anchor_id", "station"]).any():
        raise ContractError("wind feature key is duplicated")
    return features


def attach_wind_features(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    output = frame.merge(
        features,
        on=["anchor_id", "station"],
        how="left",
        validate="many_to_one",
    )
    if len(output) != len(frame):
        raise ContractError("wind feature join changed row count")
    return output


def build_v19_crossfit(
    frame: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    alpha = np.full(len(frame), np.nan, dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    for block in block_order(frame):
        outer = frame["block"].eq(block).to_numpy()
        valid_frame = frame.loc[outer]
        boundary = pd.Timestamp(valid_frame["anchor_time"].min()) - pd.Timedelta(hours=PURGE_HOURS)
        prefix = np.sort(
            history.loc[
                history["anchor_time"].le(boundary) & history["wave_power_valid"],
                "wave_power",
            ].to_numpy(dtype=np.float64)
        )
        valid = valid_frame["wave_power_valid"].to_numpy(dtype=bool)
        rank = v19.ranks_from_prefix(
            prefix,
            valid_frame["wave_power"].to_numpy(dtype=np.float64),
            valid,
        )
        fold_prediction, fold_alpha = v19.predict_policy(valid_frame, rank)
        prediction[outer] = fold_prediction
        alpha[outer] = fold_alpha
        receipts.append(
            {
                "block": str(block),
                "boundary_utc": boundary.isoformat(),
                "prefix_valid_cases": int(len(prefix)),
                "feature_only_ecdf_fits": 1,
                "outer_target_rows_read_before_v19_policy_fixed": 0,
            }
        )
    if not np.isfinite(prediction).all() or not np.isfinite(alpha).all():
        raise ContractError("v19 crossfit reconstruction is incomplete")
    return prediction, alpha, receipts


def fit_residual_axis(train: pd.DataFrame) -> tuple[Ridge, dict[str, np.ndarray], dict[str, Any]]:
    active = train["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    work = train.loc[active].reset_index(drop=True)
    raw, eligible = transformed_raw_features(work)
    if int(eligible.sum()) < 100:
        raise ContractError("insufficient eligible wind-work training rows")
    state = fit_transformer(raw[eligible])
    z = apply_transformer(raw[eligible], state)
    delta = work.loc[eligible, "delta"].to_numpy(dtype=np.float64)
    design = delta[:, None] * z
    target = work.loc[eligible, "target_hs"].to_numpy(dtype=np.float64) - work.loc[
        eligible, "v19_prediction"
    ].to_numpy(dtype=np.float64)
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    model.fit(design, target)
    if not np.isfinite(model.coef_).all():
        raise ContractError("ridge coefficients are non-finite")
    receipt = {
        "train_active_rows": int(len(work)),
        "train_eligible_rows": int(eligible.sum()),
        "ridge_alpha": RIDGE_ALPHA,
        "coefficient_l2": float(np.linalg.norm(model.coef_)),
        "feature_medians": state["median"].tolist(),
        "feature_scales": state["scale"].tolist(),
        "outer_target_rows_read_before_fit": 0,
    }
    return model, state, receipt


def predict_residual_axis(
    frame: pd.DataFrame,
    model: Ridge,
    state: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw, eligible = transformed_raw_features(frame)
    z = apply_transformer(raw, state)
    raw_adjustment = z @ np.asarray(model.coef_, dtype=np.float64)
    adjustment = TRUST * np.clip(raw_adjustment, -RAW_ADJUSTMENT_CAP, RAW_ADJUSTMENT_CAP)
    adjustment[~eligible] = 0.0
    active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    adjustment[~active] = 0.0
    alpha = frame["v19_alpha"].to_numpy(dtype=np.float64).copy()
    alpha[active] = np.clip(alpha[active] + adjustment[active], ALPHA_MIN, ALPHA_MAX)
    prediction = frame["v19_prediction"].to_numpy(dtype=np.float64).copy()
    prediction[active] = frame.loc[active, "base"].to_numpy(dtype=np.float64) + alpha[
        active
    ] * frame.loc[active, "delta"].to_numpy(dtype=np.float64)
    prediction = np.clip(prediction, 0.0, 30.0)
    if not np.isfinite(prediction).all() or not np.isfinite(alpha).all():
        raise ContractError("wind-work candidate is non-finite")
    return prediction, alpha, eligible


def group_deltas(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    comparator: np.ndarray,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    work = frame.assign(candidate=prediction, comparator=comparator)
    output: dict[str, dict[str, Any]] = {}
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, part in work.groupby(grouper, observed=True, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        label = "|".join(
            str(int(value)) if isinstance(value, (int, np.integer)) else str(value)
            for value in values
        )
        output[label] = {
            "rows": int(len(part)),
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["comparator"].to_numpy()),
        }
    return output


def evaluate(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frame, profile = v19.load_historical()
    history = v19.load_wave_power_history()
    frame = v19.attach_wave_power(frame, history)
    frame = attach_wind_features(frame, load_wind_features())
    v19_prediction, v19_alpha, v19_receipts = build_v19_crossfit(frame, history)
    frame = frame.assign(v19_prediction=v19_prediction, v19_alpha=v19_alpha)

    candidate_prediction = np.full(len(frame), np.nan, dtype=np.float64)
    candidate_alpha = np.full(len(frame), np.nan, dtype=np.float64)
    fit_receipts: list[dict[str, Any]] = []
    for block in block_order(frame):
        outer = frame["block"].eq(block).to_numpy()
        train = purge_training_cases(frame.loc[~outer], frame.loc[outer])
        model, state, receipt = fit_residual_axis(train)
        fold_prediction, fold_alpha, eligible = predict_residual_axis(
            frame.loc[outer], model, state
        )
        candidate_prediction[outer] = fold_prediction
        candidate_alpha[outer] = fold_alpha
        receipt.update(
            {
                "block": str(block),
                "validation_rows": int(outer.sum()),
                "validation_eligible_rows": int(eligible.sum()),
                "model_fits": 1,
            }
        )
        fit_receipts.append(receipt)
    if not np.isfinite(candidate_prediction).all() or not np.isfinite(candidate_alpha).all():
        raise ContractError("outer prediction assembly is incomplete")

    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    uniform = frame["reference"].to_numpy(dtype=np.float64)
    persistence = frame["current_hs"].to_numpy(dtype=np.float64)
    candidate_rmse = rmse(truth, candidate_prediction)
    uniform_rmse = rmse(truth, uniform)
    v19_rmse = rmse(truth, v19_prediction)
    persistence_rmse = rmse(truth, persistence)
    delta_vs_uniform = candidate_rmse - uniform_rmse
    delta_vs_v19 = candidate_rmse - v19_rmse

    by_block_vs_v19 = group_deltas(frame, candidate_prediction, v19_prediction, ["block"])
    by_block_vs_uniform = group_deltas(frame, candidate_prediction, uniform, ["block"])
    station_lead_vs_v19 = group_deltas(
        frame, candidate_prediction, v19_prediction, ["station", "lead_h"]
    )
    station_vs_v19 = group_deltas(frame, candidate_prediction, v19_prediction, ["station"])
    lead_vs_v19 = group_deltas(frame, candidate_prediction, v19_prediction, ["lead_h"])
    improved_blocks = sum(item["delta_rmse_m"] < 0 for item in by_block_vs_v19.values())
    worst_station_lead = max(item["delta_rmse_m"] for item in station_lead_vs_v19.values())

    versus_v19_frame = frame.assign(reference=v19_prediction)
    episode_vs_v19 = bootstrap(versus_v19_frame, candidate_prediction, ("episode_id",), 20260901)
    block_station_vs_v19 = bootstrap(
        versus_v19_frame, candidate_prediction, ("block", "station"), 20260902
    )
    episode_vs_uniform = bootstrap(frame, candidate_prediction, ("episode_id",), 20260903)

    short = frame["lead_h"].isin([3, 6, 9, 12]).to_numpy()
    active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    changed = np.abs(candidate_prediction - v19_prediction) > 1e-12
    stable_checks = {
        "delta_rmse_vs_v19_negative": delta_vs_v19 < 0,
        "minimum_four_improved_blocks_vs_v19": improved_blocks >= 4,
        "episode_ci90_upper_vs_v19_below_zero": episode_vs_v19["ci90_m"][1] < 0,
        "block_station_ci90_upper_vs_v19_below_zero": block_station_vs_v19["ci90_m"][1] < 0,
        "worst_station_lead_vs_v19_at_most_0p01m": worst_station_lead <= 0.01,
        "short_leads_bit_exact_uniform": bool(
            np.array_equal(candidate_prediction[short], uniform[short])
        ),
        "six_outer_model_fits": sum(item["model_fits"] for item in fit_receipts) == 6,
        "finite_predictions": bool(np.isfinite(candidate_prediction).all()),
    }
    high_risk_checks = {
        "delta_rmse_vs_v19_at_most_minus_0p002m": delta_vs_v19 <= -0.002,
        "delta_rmse_vs_uniform_at_most_minus_0p005m": delta_vs_uniform <= -0.005,
        "worst_station_lead_vs_v19_at_most_0p02m": worst_station_lead <= 0.02,
        "short_leads_bit_exact_uniform": stable_checks["short_leads_bit_exact_uniform"],
        "finite_predictions": stable_checks["finite_predictions"],
    }
    stable = all(stable_checks.values())
    high_risk = (not stable) and all(high_risk_checks.values())
    decision = (
        "PASS_STABLE_CANDIDATE"
        if stable
        else "PRESERVE_HIGH_RISK_SCORE_CANDIDATE"
        if high_risk
        else "NO_GO_WIND_WORK_RESIDUAL_AXIS"
    )
    raw_points_vs_uniform = -delta_vs_uniform * POINTS_PER_RMSE_M
    raw_points_vs_v19 = -delta_vs_v19 * POINTS_PER_RMSE_M
    result = {
        "schema_version": "p3.kma_wind_work_residual_axis.result.v20",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": decision,
        "candidate": {
            "name": CANDIDATE,
            "reference": "frozen_v19_wave_power_schedule",
            "surface_claim": config["validation"]["surface_claim"],
            "rmse_m": {
                "persistence": persistence_rmse,
                "uniform_kma_0p425": uniform_rmse,
                "v19_wave_power": v19_rmse,
                "candidate": candidate_rmse,
                "delta_candidate_minus_persistence": candidate_rmse - persistence_rmse,
                "delta_candidate_minus_uniform": delta_vs_uniform,
                "delta_candidate_minus_v19": delta_vs_v19,
            },
            "expected_points": {
                "raw_gain_vs_uniform": raw_points_vs_uniform,
                "incremental_gain_vs_v19": raw_points_vs_v19,
                "transport_penalty_points": TRANSPORT_PENALTY_POINTS,
                "calibrated_gain_vs_uniform": raw_points_vs_uniform - TRANSPORT_PENALTY_POINTS,
                "expected_official_score_from_current_champion": OFFICIAL_CHAMPION_POINTS
                + raw_points_vs_uniform,
            },
            "by_block_vs_v19": by_block_vs_v19,
            "by_block_vs_uniform": by_block_vs_uniform,
            "improved_blocks_vs_v19": int(improved_blocks),
            "station_vs_v19": station_vs_v19,
            "lead_vs_v19": lead_vs_v19,
            "station_lead_vs_v19": station_lead_vs_v19,
            "worst_station_lead_delta_vs_v19_m": worst_station_lead,
            "episode_bootstrap_vs_v19": episode_vs_v19,
            "block_station_bootstrap_vs_v19": block_station_vs_v19,
            "episode_bootstrap_vs_uniform": episode_vs_uniform,
            "changed_rows_vs_v19": int(changed.sum()),
            "changed_rows_share_all": float(changed.mean()),
            "changed_rows_share_active": float(changed[active].mean()),
            "fit_receipts": fit_receipts,
            "v19_feature_only_receipts": v19_receipts,
            "stable_gate_checks": stable_checks,
            "high_risk_preservation_checks": high_risk_checks,
            "stable_pass": stable,
            "high_risk_preserved": high_risk,
        },
        "fit_count": {"ridge_outer_fits": 6, "v19_ecdf_feature_only_fits": 6},
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
        "candidate": candidate_prediction,
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
    risk = (
        "stable"
        if candidate["stable_pass"]
        else "high-risk preserved despite stability failures"
        if candidate["high_risk_preserved"]
        else "not preserved"
    )
    return (
        "# P3 v19 wind-work residual-axis v20\n\n"
        "## 결론\n\n"
        f"- decision: **{result['decision']}** ({risk}).\n"
        f"- candidate RMSE `{metrics['candidate']:.9f}m`; vs uniform `{metrics['delta_candidate_minus_uniform']:+.9f}m`; vs v19 `{metrics['delta_candidate_minus_v19']:+.9f}m`.\n"
        f"- expected raw score gain vs current official champion `{points['raw_gain_vs_uniform']:+.9f}`; transport-calibrated `{points['calibrated_gain_vs_uniform']:+.9f}`; expected score `{points['expected_official_score_from_current_champion']:.9f}`.\n"
        f"- improved bimonth blocks vs v19 `{candidate['improved_blocks_vs_v19']}/6`; worst station×lead `{candidate['worst_station_lead_delta_vs_v19_m']:+.9f}m`.\n"
        f"- episode CI90 vs v19 `{candidate['episode_bootstrap_vs_v19']['ci90_m']}`; block×station CI90 `{candidate['block_station_bootstrap_vs_v19']['ci90_m']}`.\n"
        "- official test/sample/submission/hidden/CSV/upload access: all 0.\n\n"
        "## 해석 경계\n\n"
        "This is a repeatedly exposed 182-case development surface. It is an internal comparative result, not an independent official-score guarantee. The wind-work residual axis was sealed before this candidate's outer scores were computed and no result-based retry occurred.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "LAUNCH_READY"}))
        return 0
    if ARTIFACT_DIR.exists() or REPORT_DIR.exists() or LOCK.exists():
        raise ContractError("v20 exactly-once namespace already exists")
    config = load_config()
    runner_hash = sha256(Path(__file__))
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ATTEMPT_CONSUMED_ONE_SHOT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runner_sha256": runner_hash,
        "config_sha256": sha256(CONFIG),
        "candidate": CANDIDATE,
        "result_based_tuning": False,
        "official_access": 0,
    }
    write_new(LOCK, canonical(lock))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result, arrays = evaluate(config)
    result["execution"]["elapsed_seconds"] = time.perf_counter() - started
    result["provenance"] = {
        "runner_sha256": runner_hash,
        "config_sha256": sha256(CONFIG),
        "train_features_sha256": sha256(TRAIN_FEATURES),
        "v19_result_sha256": sha256(v19.ARTIFACT_DIR / "result.json"),
    }
    array_path = ARTIFACT_DIR / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"]["evaluation_arrays_sha256"] = sha256(array_path)
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, render_report(result).encode("utf-8"))
    write_new(
        REPORT_DIR / "run-manifest.json",
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "runner_sha256": runner_hash,
                "config_sha256": sha256(CONFIG),
                "result_sha256": sha256(result_path),
                "arrays_sha256": sha256(array_path),
                "report_sha256": sha256(report_path),
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
                "ridge_outer_fits": 6,
                "official_access": 0,
                "csv_materializations": 0,
                "uploads": 0,
                "elapsed_seconds": result["execution"]["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
