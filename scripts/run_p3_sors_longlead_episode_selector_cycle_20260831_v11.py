"""Sealed P3 v11 S-ORS long-lead two-stage episode selector.

The first-stage expert is the already sealed v6 CatBoost-Huber lead-gain OOF
policy.  The second stage learns, using only outer-training episodes and
78-hour-purged inner cross-fits, whether that policy should replace the exact
current champion for an S-ORS episode.  May-June is an unconditional no-op and
the intervention budget is capped at 20% before outer labels are inspected.
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

from run_p3_parallel_candidate_cycle_20260831_v4 import load_historical, rmse  # noqa: E402

EXPERIMENT_ID = "p3_sors_longlead_episode_selector_cycle_20260831_v11"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V6_OOF = ROOT / "artifacts/p3_inner_lcb_router_cycle_20260831_v6/internal_oof_active_leads.parquet"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v1/calibration.json"
V6_PRED_COL = "prediction__P3_2_CATBOOST_HUBER_LEAD_GAIN_LCB"
POINTS_PER_RMSE_M = 15.870739046986959
MIN_CALIBRATED_POINTS = 0.01
MAX_WORST_STATION_LEAD_M = 0.01
INTERVENTION_BUDGET = 0.20
LCB_QUANTILE = 0.80
BOOTSTRAP_REPLICATES = 5000
SEED = 20260831


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Spec:
    name: str
    policy: str
    summary: str


SPECS = (
    Spec("P3_1_ET_TWO_STAGE_SORS_EPISODE_LCB", "extra_trees", "ExtraTrees sign+gain LCB selector."),
    Spec("P3_2_HGB_TWO_STAGE_SORS_EPISODE_LCB", "hist_gbdt", "Shallow HGB sign+absolute-gain LCB selector."),
    Spec("P3_3_CONSENSUS_TWO_STAGE_SORS_EPISODE_LCB", "consensus", "Both sealed selectors must agree."),
)

FEATURES = [
    "current_hs",
    "base_18",
    "base_24",
    "base_rise_18_24",
    "wave_energy",
    "delta_18",
    "delta_24",
    "kma_disagreement_abs",
    "v6_lcb_mean",
    "v6_lcb_min",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def build_episode_table(frame: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    active = frame.loc[(frame["station"] == "S-ORS") & frame["lead_h"].isin([18, 24])].copy()
    frozen = oof.loc[(oof["station"] == "S-ORS") & oof["lead_h"].isin([18, 24])].copy()
    columns = ["anchor_id", "station", "lead_h", "catboost_lcb", V6_PRED_COL]
    active = active.merge(frozen[columns], on=["anchor_id", "station", "lead_h"], validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for episode_id, part in active.groupby("episode_id", observed=True, sort=True):
        by_lead = part.groupby("lead_h", observed=True)[["base", "delta"]].mean()
        reference_sse = float(np.square(part["target_hs"] - part["reference"]).sum())
        expert_sse = float(np.square(part["target_hs"] - part[V6_PRED_COL]).sum())
        row = {
            "episode_id": str(episode_id),
            "anchor_id": str(part["anchor_id"].iloc[0]),
            "station": "S-ORS",
            "anchor_time": pd.Timestamp(part["anchor_time"].min()),
            "block": str(part["block"].iloc[0]),
            "episode_gain": reference_sse - expert_sse,
            "current_hs": float(part["current_hs"].iloc[0]),
            "base_18": float(by_lead.loc[18, "base"]),
            "base_24": float(by_lead.loc[24, "base"]),
            "base_rise_18_24": float(by_lead.loc[24, "base"] - by_lead.loc[18, "base"]),
            "wave_energy": float(np.square(part["base"]).mean()),
            "delta_18": float(by_lead.loc[18, "delta"]),
            "delta_24": float(by_lead.loc[24, "delta"]),
            "kma_disagreement_abs": float(np.abs(part["delta"]).sum()),
            "v6_lcb_mean": float(part["catboost_lcb"].mean()),
            "v6_lcb_min": float(part["catboost_lcb"].min()),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty or table.duplicated("episode_id").any() or not np.isfinite(table[FEATURES]).all().all():
        raise ContractError("episode selector table contract failed")
    return table


def purged(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    keep = np.ones(len(train), dtype=bool)
    for timestamp in pd.to_datetime(valid["anchor_time"], utc=True):
        distance = np.abs(pd.to_datetime(train["anchor_time"], utc=True) - timestamp)
        keep &= distance > pd.Timedelta(hours=78)
    return train.loc[keep].copy()


def models(family: str, seed: int) -> tuple[Any, Any]:
    if family == "extra_trees":
        common = dict(n_estimators=350, max_depth=5, min_samples_leaf=4, max_features=0.7, random_state=seed, n_jobs=4)
        return (
            make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(class_weight="balanced", **common)),
            make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(**common)),
        )
    if family == "hist_gbdt":
        return (
            make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7, l2_regularization=10.0, random_state=seed)),
            make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="absolute_error", max_iter=120, max_leaf_nodes=7, l2_regularization=10.0, random_state=seed)),
        )
    raise ContractError(f"unknown family {family}")


def fit_pair(family: str, train: pd.DataFrame, seed: int) -> tuple[Any, Any]:
    classifier, regressor = models(family, seed)
    labels = train["episode_gain"].gt(0).astype(int)
    if labels.nunique() < 2:
        raise ContractError("purged selector train has one class")
    classifier.fit(train[FEATURES], labels)
    regressor.fit(train[FEATURES], train["episode_gain"])
    return classifier, regressor


def family_outer_scores(family: str, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    probabilities = np.full(len(table), np.nan)
    lcbs = np.full(len(table), np.nan)
    receipts: list[dict[str, Any]] = []
    blocks = sorted(table["block"].unique())
    for outer_index, block in enumerate(blocks):
        test_mask = table["block"].eq(block).to_numpy()
        outer_test = table.loc[test_mask]
        outer_train = purged(table.loc[~test_mask], outer_test)
        inner_probability = np.full(len(outer_train), np.nan)
        inner_gain = np.full(len(outer_train), np.nan)
        for inner_index, inner_block in enumerate(sorted(outer_train["block"].unique())):
            valid_mask = outer_train["block"].eq(inner_block).to_numpy()
            inner_valid = outer_train.loc[valid_mask]
            inner_train = purged(outer_train.loc[~valid_mask], inner_valid)
            classifier, regressor = fit_pair(family, inner_train, SEED + outer_index * 100 + inner_index)
            inner_probability[valid_mask] = classifier.predict_proba(inner_valid[FEATURES])[:, 1]
            inner_gain[valid_mask] = regressor.predict(inner_valid[FEATURES])
        if not np.isfinite(inner_probability).all() or not np.isfinite(inner_gain).all():
            raise ContractError("inner crossfit incomplete")
        overprediction = inner_gain - outer_train["episode_gain"].to_numpy(float)
        residual = float(np.quantile(overprediction, LCB_QUANTILE, method="higher"))
        classifier, regressor = fit_pair(family, outer_train, SEED + outer_index * 100 + 99)
        probabilities[test_mask] = classifier.predict_proba(outer_test[FEATURES])[:, 1]
        lcbs[test_mask] = regressor.predict(outer_test[FEATURES]) - residual
        receipts.append({"block": block, "train_episodes_after_78h_purge": len(outer_train), "test_episodes": len(outer_test), "residual_quantile": residual, "fits": 2 * (len(outer_train["block"].unique()) + 1)})
    return probabilities, lcbs, receipts


def bootstrap(frame: pd.DataFrame, prediction: np.ndarray, columns: tuple[str, ...], seed: int) -> dict[str, Any]:
    work = frame.assign(candidate=prediction)
    grouped = list(work.groupby(list(columns), observed=True, sort=True))
    ref_sse = np.asarray([np.square(part["reference"] - part["target_hs"]).sum() for _, part in grouped])
    cand_sse = np.asarray([np.square(part["candidate"] - part["target_hs"]).sum() for _, part in grouped])
    counts = np.asarray([len(part) for _, part in grouped])
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(grouped), len(grouped))
        denom = counts[draw].sum()
        values[index] = np.sqrt(cand_sse[draw].sum() / denom) - np.sqrt(ref_sse[draw].sum() / denom)
    return {"unit": "|".join(columns), "units": len(grouped), "replicates": BOOTSTRAP_REPLICATES, "ci90_m": [float(x) for x in np.quantile(values, [0.05, 0.95])], "median_m": float(np.median(values)), "probability_improve": float(np.mean(values < 0))}


def evaluate(frame: pd.DataFrame, oof: pd.DataFrame, table: pd.DataFrame, scores: dict[str, tuple[np.ndarray, np.ndarray]], spec: Spec, penalty: float) -> dict[str, Any]:
    if spec.policy == "consensus":
        qualified = (scores["extra_trees"][0] >= 0.5) & (scores["hist_gbdt"][0] >= 0.5) & (scores["extra_trees"][1] > 0) & (scores["hist_gbdt"][1] > 0)
        rank_score = np.minimum(scores["extra_trees"][1], scores["hist_gbdt"][1])
    else:
        probability, lcb = scores[spec.policy]
        qualified = (probability >= 0.5) & (lcb > 0)
        rank_score = lcb
    qualified &= ~table["block"].eq("05_06").to_numpy()
    eligible = int(np.sum(~table["block"].eq("05_06")))
    budget = int(np.floor(INTERVENTION_BUDGET * eligible))
    selected = np.zeros(len(table), dtype=bool)
    indices = np.flatnonzero(qualified)
    if budget and len(indices):
        chosen = indices[np.argsort(rank_score[indices])[-min(budget, len(indices)):]]
        selected[chosen] = True
    selected_episodes = set(table.loc[selected, "episode_id"].astype(str))
    prediction = frame["reference"].to_numpy(float).copy()
    frozen = oof[["anchor_id", "station", "lead_h", V6_PRED_COL]].copy()
    merged = frame[["anchor_id", "station", "lead_h", "episode_id"]].merge(frozen, on=["anchor_id", "station", "lead_h"], how="left", validate="one_to_one")
    change = frame["station"].eq("S-ORS") & frame["lead_h"].isin([18, 24]) & merged["episode_id"].astype(str).isin(selected_episodes)
    prediction[change] = merged.loc[change, V6_PRED_COL].to_numpy(float)
    truth = frame["target_hs"].to_numpy(float)
    reference = frame["reference"].to_numpy(float)
    delta = rmse(truth, prediction) - rmse(truth, reference)
    station_lead: dict[str, Any] = {}
    for (station, lead), part in frame.assign(candidate=prediction).groupby(["station", "lead_h"], observed=True, sort=True):
        station_lead[f"{station}|{int(lead)}"] = {"rows": len(part), "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy()) - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy())}
    episode = bootstrap(frame, prediction, ("episode_id",), SEED)
    group = bootstrap(frame, prediction, ("block", "station"), SEED + 1)
    raw = max(0.0, -episode["ci90_m"][1] * POINTS_PER_RMSE_M)
    worst = max(item["delta_rmse_m"] for item in station_lead.values())
    checks = {
        "pooled_rmse_improves": delta < 0,
        "episode_ci90_upper_below_zero": episode["ci90_m"][1] < 0,
        "group_ci90_upper_below_zero": group["ci90_m"][1] < 0,
        "worst_station_lead_within_0p01m": worst <= MAX_WORST_STATION_LEAD_M,
        "raw_conservative_points_at_least_0p331905690": raw >= penalty + MIN_CALIBRATED_POINTS,
        "calibrated_conservative_points_at_least_0p01": raw - penalty >= MIN_CALIBRATED_POINTS,
        "may_june_exact_noop": not bool(change[frame["block"].eq("05_06")].any()),
        "intervention_budget_at_most_20pct": float(selected.mean()) <= INTERVENTION_BUDGET,
        "finite_predictions": bool(np.isfinite(prediction).all()),
    }
    return {"spec": asdict(spec), "reference_rmse_m": rmse(truth, reference), "candidate_rmse_m": rmse(truth, prediction), "delta_candidate_minus_reference_rmse_m": delta, "selected_episode_count": len(selected_episodes), "eligible_episode_count": eligible, "intervention_episode_share": float(selected.mean()), "changed_rows": int(change.sum()), "station_lead": station_lead, "worst_station_lead_delta_rmse_m": worst, "episode_bootstrap": episode, "group_bootstrap": group, "expected_points": {"raw_central": -delta * POINTS_PER_RMSE_M, "raw_conservative": raw, "public_reversal_penalty": penalty, "calibrated_conservative": raw - penalty}, "gate_checks": checks, "passed": all(checks.values())}


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
        write_new(
            ARTIFACT_DIR / "technical_recovery.json",
            canonical(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "initial_runner_sha256": original["runner_sha256"],
                    "corrected_runner_sha256": runner_hash,
                    "root_cause": "episode_id contains multiple anchors; lead lookup returned a Series instead of a scalar",
                    "repair": "aggregate base and KMA delta to episode-by-lead means before scalar feature extraction",
                    "fits_before_failure": 0,
                    "predictions_before_failure": 0,
                    "official_rows_read_before_failure": 0,
                    "candidate_or_gate_changes": 0,
                }
            ),
        )
    else:
        write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "created_at_utc": datetime.now(UTC).isoformat(), "runner_sha256": runner_hash, "maximum_unique_historical_fits": 144, "sealed_policy": {"station": "S-ORS", "leads": [18, 24], "lcb_quantile": LCB_QUANTILE, "may_june_exact_noop": True, "intervention_budget": INTERVENTION_BUDGET}}))
    frame, profile = load_historical()
    oof = pd.read_parquet(V6_OOF)
    table = build_episode_table(frame, oof)
    et_probability, et_lcb, et_receipts = family_outer_scores("extra_trees", table)
    hgb_probability, hgb_lcb, hgb_receipts = family_outer_scores("hist_gbdt", table)
    scores = {"extra_trees": (et_probability, et_lcb), "hist_gbdt": (hgb_probability, hgb_lcb)}
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    penalty = float(calibration["gates"]["P3"]["transport_penalty_points"])
    candidates = [evaluate(frame, oof, table, scores, spec, penalty) for spec in SPECS]
    passing = [item for item in candidates if item["passed"]]
    if passing:
        raise ContractError("unexpected PASS: full frozen-v6 official materializer is intentionally separate")
    result = {"schema_version": "p3.sors_longlead_episode_selector.result.v11", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "COMPLETE", "decision": "NO_GO_PUBLIC_TRANSPORT_GATE", "candidate_count": len(candidates), "passing_candidate_count": 0, "candidates": candidates, "fit_budget": {"maximum_unique_historical": 144, "actual_unique_historical": int(sum(x["fits"] for x in et_receipts + hgb_receipts)), "actual_full": 0}, "selector_calibration": {"extra_trees": et_receipts, "hist_gbdt": hgb_receipts}, "data_profile": {**profile, "selector_episodes": len(table), "sors_known_positive_expert_episodes": int(table["episode_gain"].gt(0).sum())}, "transport": {"penalty_points": penalty, "minimum_raw_points": penalty + MIN_CALIBRATED_POINTS, "minimum_calibrated_points": MIN_CALIBRATED_POINTS}, "outputs": [], "data_access": {"official_test_index_rows_read": 0, "official_feature_rows_read": 0, "hidden_truth_rows_read": 0, "uploads": 0}, "provenance": {"runner_sha256": runner_hash, "v6_oof_sha256": sha256(V6_OOF), "calibration_sha256": sha256(CALIBRATION)}, "execution": {"elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(), "result_based_tuning_or_retry": False, "outer_label_threshold_tuning": False, "upload_attempt_count": 0}}
    result_path = ARTIFACT_DIR / "result.json"
    write_new(result_path, canonical(result))
    report = "# P3 S-ORS long-lead episode selector v11\n\n## 결론\n\n- calibrated PASS: **0/3**\n- CSV 0, upload 0\n\n" + "\n".join(f"- {x['spec']['name']}: delta={x['delta_candidate_minus_reference_rmse_m']:.6f}m, selected={x['selected_episode_count']}, calibrated={x['expected_points']['calibrated_conservative']:.6f}, PASS={x['passed']}" for x in candidates) + "\n"
    report_path = REPORT_DIR / "report-source.md"
    write_new(report_path, report.encode())
    write_new(REPORT_DIR / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "runner_sha256": runner_hash, "result_sha256": sha256(result_path), "report_sha256": sha256(report_path), "outputs": []}))
    print(json.dumps({"status": "COMPLETE", "passing": 0, "fits": result["fit_budget"]["actual_unique_historical"], "elapsed_seconds": result["execution"]["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
