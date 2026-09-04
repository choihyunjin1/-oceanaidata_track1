"""Run the frozen P3 lead-level inner-crossfit LCB router cycle.

The two base learners predict train-only squared-error gain for selecting the
existing alpha=0.425 KMA physical expert over the unadjusted base at 18 h and
24 h.  One-sided residual lower confidence bounds are calibrated exclusively
from inner historical block cross-fits.  Official inputs remain unopened unless
at least one candidate passes the governing historical gates.
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
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for entry in (SCRIPTS, SRC):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

from run_p3_parallel_candidate_cycle_20260831_v4 import (  # noqa: E402
    ACTIVE_LEADS,
    BOOTSTRAP_SEED,
    KEYS,
    episode_bootstrap,
    episode_diagnostics,
    purge_training_cases,
    rmse,
)
from run_p3_physical_expert_router_cycle_20260831_v5 import (  # noqa: E402
    ALPHA_MAX,
    FEATURE_COLUMNS,
    REFERENCE_ALPHA,
    TEST_FEATURES,
    TRAIN_FEATURES,
    build_historical_cases,
    conditional_score_translation,
    json_bytes,
    load_official_cases,
    sha256,
    write_new,
)

EXPERIMENT_ID = "p3_inner_lcb_router_cycle_20260831_v6"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
DELIVERY_DIR = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_INNER_LCB_ROUTER_V6"
)
ATTEMPT_LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V5_RUNNER = ROOT / "scripts/run_p3_physical_expert_router_cycle_20260831_v5.py"
V5_RESULT = ROOT / "artifacts/p3_physical_expert_router_cycle_20260831_v5/result.json"
LCB_RESIDUAL_QUANTILE = 0.80
INNER_FAMILY_FITS_PER_OUTER = 6
INTERNAL_UNIQUE_FITS = 72
MAX_UNIQUE_FITS = 86


class ContractError(RuntimeError):
    """Raised when the immutable v6 data or evaluation contract is violated."""


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    policy: str
    summary: str


SPECS = (
    CandidateSpec(
        "P3_1_EXTRATREES_LEAD_GAIN_LCB",
        "extra_trees_lcb",
        "Lead-level ExtraTrees gain router with train-only one-sided inner-crossfit LCB abstention.",
    ),
    CandidateSpec(
        "P3_2_CATBOOST_HUBER_LEAD_GAIN_LCB",
        "catboost_lcb",
        "Lead-level CatBoost Huber gain router with train-only one-sided inner-crossfit LCB abstention.",
    ),
    CandidateSpec(
        "P3_3_CONSENSUS_LEAD_GAIN_LCB",
        "consensus_lcb",
        "Physical expert is selected only when both independent lead-level gain LCBs are positive.",
    ),
)


def build_route_rows(
    frame: pd.DataFrame, cases: pd.DataFrame, case_features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Expand 182 case rows to the two active physical-routing decisions."""

    active = frame.loc[
        frame["lead_h"].isin(ACTIVE_LEADS),
        [
            "anchor_id",
            "station",
            "anchor_time",
            "block",
            "episode_id",
            "lead_h",
            "base",
            "delta",
            "reference",
            "target_hs",
        ],
    ].copy()
    route = active.merge(
        cases[["anchor_id", "station", *case_features]],
        on=["anchor_id", "station"],
        how="left",
        validate="many_to_one",
    )
    route["selected_base"] = route["base"]
    route["selected_delta"] = route["delta"]
    route["selected_source"] = route["base"] + route["delta"]
    route["is_lead_24h"] = route["lead_h"].eq(24).astype(np.float64)
    route["lead_gain"] = np.square(route["target_hs"] - route["base"]) - np.square(
        route["target_hs"] - route["reference"]
    )
    features = [
        *case_features,
        "selected_base",
        "selected_delta",
        "selected_source",
        "is_lead_24h",
    ]
    if len(route) != 364 or route.duplicated(["anchor_id", "station", "lead_h"]).any():
        raise ContractError("historical lead router grain changed")
    matrix = route[features].apply(pd.to_numeric, errors="coerce")
    if matrix.notna().sum(axis=0).eq(0).any():
        raise ContractError("lead router feature has no finite historical support")
    return route, features


def build_estimator(family: str, seed: int) -> Any:
    if family == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=7,
                min_samples_leaf=6,
                max_features=0.30,
                bootstrap=True,
                random_state=int(seed),
                n_jobs=6,
            ),
        )
    if family == "catboost":
        return CatBoostRegressor(
            iterations=400,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=25.0,
            random_strength=0.5,
            loss_function="Huber:delta=1.0",
            random_seed=int(seed),
            thread_count=6,
            verbose=False,
            allow_writing_files=False,
        )
    raise ContractError(f"unknown family: {family}")


def _fit(
    family: str,
    train: pd.DataFrame,
    features: list[str],
    seed: int,
) -> Any:
    model = build_estimator(family, seed)
    target = train["lead_gain"].to_numpy(dtype=np.float64)
    if family == "extra_trees":
        low, high = np.quantile(target, [0.025, 0.975])
        target = np.clip(target, low, high)
    model.fit(train[features], target)
    return model


def _purged_by_cases(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return purge_training_cases(train, test)


def fit_outer_lcb(
    family: str,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict an outer fold and calibrate a signed one-sided LCB on inner OOF residuals."""

    inner_prediction = np.full(len(outer_train), np.nan, dtype=np.float64)
    inner_blocks = sorted(outer_train["block"].astype(str).unique())
    if len(inner_blocks) != 5:
        raise ContractError("outer training set must contain five inner blocks")
    for inner_index, inner_block in enumerate(inner_blocks):
        valid_mask = outer_train["block"].astype(str).eq(inner_block).to_numpy()
        inner_valid = outer_train.loc[valid_mask].copy()
        inner_train = _purged_by_cases(outer_train.loc[~valid_mask].copy(), inner_valid)
        model = _fit(family, inner_train, features, seed + inner_index)
        inner_prediction[valid_mask] = model.predict(inner_valid[features])
    if not np.isfinite(inner_prediction).all():
        raise ContractError("inner cross-fit predictions are incomplete")
    signed_overprediction = inner_prediction - outer_train["lead_gain"].to_numpy(float)
    residual_quantile = float(
        np.quantile(signed_overprediction, LCB_RESIDUAL_QUANTILE, method="higher")
    )
    final_model = _fit(family, outer_train, features, seed + 99)
    point = np.asarray(final_model.predict(outer_test[features]), dtype=np.float64)
    lcb = point - residual_quantile
    if not np.isfinite(lcb).all():
        raise ContractError("outer LCB is nonfinite")
    return lcb, {
        "inner_blocks": inner_blocks,
        "inner_oof_rows": int(len(outer_train)),
        "one_sided_residual_quantile": residual_quantile,
        "lcb_residual_quantile_level": LCB_RESIDUAL_QUANTILE,
        "fits": INNER_FAMILY_FITS_PER_OUTER,
    }


def candidate_route(spec: CandidateSpec, et_lcb: np.ndarray, cb_lcb: np.ndarray) -> np.ndarray:
    if spec.policy == "extra_trees_lcb":
        route = et_lcb > 0.0
    elif spec.policy == "catboost_lcb":
        route = cb_lcb > 0.0
    elif spec.policy == "consensus_lcb":
        route = (et_lcb > 0.0) & (cb_lcb > 0.0)
    else:
        raise ContractError(f"unknown policy: {spec.policy}")
    return route.astype(np.float64) * REFERENCE_ALPHA


def expand_route_prediction(
    frame: pd.DataFrame, route: pd.DataFrame, alpha_column: str
) -> np.ndarray:
    mapping = frame[["anchor_id", "station", "lead_h"]].merge(
        route[["anchor_id", "station", "lead_h", alpha_column]],
        on=["anchor_id", "station", "lead_h"],
        how="left",
        validate="one_to_one",
    )
    active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    if mapping.loc[active, alpha_column].isna().any():
        raise ContractError("active lead route is missing")
    prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
    alpha = mapping.loc[active, alpha_column].to_numpy(dtype=np.float64)
    if alpha.min() < 0.0 or alpha.max() > ALPHA_MAX:
        raise ContractError("physical alpha bound violated")
    prediction[active] = (
        frame.loc[active, "base"].to_numpy(float)
        + alpha * frame.loc[active, "delta"].to_numpy(float)
    )
    prediction = np.clip(prediction, 0.0, 30.0)
    if not np.array_equal(
        prediction[~active], frame.loc[~active, "reference"].to_numpy(dtype=np.float64)
    ):
        raise ContractError("short lead changed")
    return prediction


def evaluate_candidates(
    frame: pd.DataFrame, route: pd.DataFrame, features: list[str]
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    et_lcb = np.full(len(route), np.nan, dtype=np.float64)
    cb_lcb = np.full(len(route), np.nan, dtype=np.float64)
    calibration: dict[str, Any] = {}
    blocks = sorted(route["block"].astype(str).unique())
    for outer_index, block in enumerate(blocks):
        test_mask = route["block"].astype(str).eq(block).to_numpy()
        outer_test = route.loc[test_mask].copy()
        outer_train = _purged_by_cases(route.loc[~test_mask].copy(), outer_test)
        et_lcb[test_mask], et_meta = fit_outer_lcb(
            "extra_trees",
            outer_train,
            outer_test,
            features,
            BOOTSTRAP_SEED + 1000 * outer_index,
        )
        cb_lcb[test_mask], cb_meta = fit_outer_lcb(
            "catboost",
            outer_train,
            outer_test,
            features,
            BOOTSTRAP_SEED + 1000 * outer_index + 500,
        )
        calibration[block] = {
            "train_route_rows_after_78h_purge": int(len(outer_train)),
            "test_route_rows": int(len(outer_test)),
            "extra_trees": et_meta,
            "catboost": cb_meta,
        }
    if not np.isfinite(et_lcb).all() or not np.isfinite(cb_lcb).all():
        raise ContractError("outer cross-fit LCB is incomplete")
    oof = route[
        [
            "anchor_id",
            "station",
            "anchor_time",
            "lead_h",
            "block",
            "episode_id",
            "target_hs",
            "base",
            "reference",
            "lead_gain",
        ]
    ].copy()
    oof["extra_trees_lcb"] = et_lcb
    oof["catboost_lcb"] = cb_lcb
    y = frame["target_hs"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    before = rmse(y, reference)
    records: list[dict[str, Any]] = []
    for spec in SPECS:
        alpha = candidate_route(spec, et_lcb, cb_lcb)
        alpha_column = f"alpha__{spec.name}"
        prediction_column = f"prediction__{spec.name}"
        route[alpha_column] = alpha
        prediction = expand_route_prediction(frame, route, alpha_column)
        oof[alpha_column] = alpha
        # Store the active-lead subset for independent recomputation; inactive leads are exact reference.
        active_prediction = frame.loc[frame["lead_h"].isin(ACTIVE_LEADS), [
            "anchor_id", "station", "lead_h"
        ]].copy()
        active_prediction[prediction_column] = prediction[frame["lead_h"].isin(ACTIVE_LEADS)]
        oof = oof.merge(
            active_prediction,
            on=["anchor_id", "station", "lead_h"],
            how="left",
            validate="one_to_one",
        )
        after = rmse(y, prediction)
        by_block: dict[str, Any] = {}
        for block in blocks:
            mask = frame["block"].astype(str).eq(block).to_numpy()
            block_before = rmse(y[mask], reference[mask])
            block_after = rmse(y[mask], prediction[mask])
            by_block[block] = {
                "rows": int(mask.sum()),
                "reference_rmse": block_before,
                "candidate_rmse": block_after,
                "delta_rmse": block_after - block_before,
                "calibration": calibration[block],
            }
        episodes = episode_diagnostics(frame, prediction)
        bootstrap = episode_bootstrap(frame, prediction)
        improved_blocks = int(sum(v["delta_rmse"] < 0.0 for v in by_block.values()))
        worst = float(max(v["delta_rmse"] for v in by_block.values()))
        validity = {
            "all_predictions_finite": bool(np.isfinite(prediction).all()),
            "short_leads_exact_noop": bool(
                np.array_equal(
                    prediction[~frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()],
                    reference[~frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()],
                )
            ),
            "physical_alpha_bounds": bool(alpha.min() >= 0.0 and alpha.max() <= ALPHA_MAX),
            "complete_active_routes": bool(len(alpha) == 364),
        }
        governing = {
            "validity_hard_pass": bool(all(validity.values())),
            "pooled_rmse_improves": after < before,
            "episode_bootstrap_ci90_upper_below_zero": bootstrap["ci90_high"] < 0.0,
        }
        records.append(
            {
                "spec": asdict(spec),
                "attributed_fit_count": 36 if spec.policy != "consensus_lcb" else 72,
                "reference_rmse": before,
                "candidate_rmse": after,
                "delta_rmse": after - before,
                "by_block": by_block,
                "diagnostics": {
                    "improved_block_count": improved_blocks,
                    "worst_block_delta_rmse": worst,
                    "episodes": episodes,
                    "route_share": float(np.mean(alpha > 0.0)),
                },
                "bootstrap": bootstrap,
                "conditional_leaderboard_translation": conditional_score_translation(
                    after - before, bootstrap["ci90_low"], bootstrap["ci90_high"]
                ),
                "validity": validity,
                "governing_gates": governing,
                "diagnostic_only_not_promotion_gates": [
                    "improved_block_count",
                    "strict_majority_episodes_improve",
                    "worst_block_delta_rmse",
                ],
                "strict_internal_pass": bool(all(governing.values())),
            }
        )
    return records, oof


def fit_full_family(
    family: str, route: pd.DataFrame, features: list[str], seed: int
) -> tuple[Any, float, dict[str, Any]]:
    inner_prediction = np.full(len(route), np.nan, dtype=np.float64)
    blocks = sorted(route["block"].astype(str).unique())
    for index, block in enumerate(blocks):
        valid_mask = route["block"].astype(str).eq(block).to_numpy()
        valid = route.loc[valid_mask].copy()
        train = _purged_by_cases(route.loc[~valid_mask].copy(), valid)
        model = _fit(family, train, features, seed + index)
        inner_prediction[valid_mask] = model.predict(valid[features])
    residual = inner_prediction - route["lead_gain"].to_numpy(float)
    quantile = float(np.quantile(residual, LCB_RESIDUAL_QUANTILE, method="higher"))
    model = _fit(family, route, features, seed + 99)
    return model, quantile, {"inner_oof_rows": int(len(route)), "fits": 7}


def build_official_route_rows(
    official: pd.DataFrame, cases: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    active = official.loc[
        official["lead_h"].isin(ACTIVE_LEADS),
        ["case_id", "station", "lead_h", "base", "delta", "reference"],
    ].copy()
    case_features = [
        column
        for column in features
        if column not in {"selected_base", "selected_delta", "selected_source", "is_lead_24h"}
    ]
    route = active.merge(
        cases[["case_id", "station", *case_features]],
        on=["case_id", "station"],
        how="left",
        validate="many_to_one",
    )
    route["selected_base"] = route["base"]
    route["selected_delta"] = route["delta"]
    route["selected_source"] = route["base"] + route["delta"]
    route["is_lead_24h"] = route["lead_h"].eq(24).astype(np.float64)
    if len(route) != 400 or route.duplicated(["case_id", "station", "lead_h"]).any():
        raise ContractError("official lead route grain changed")
    return route


def materialize(
    passing: list[dict[str, Any]],
    historical_route: pd.DataFrame,
    features: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    if not passing:
        return [], {
            "test_index_rows_read": 0,
            "official_test_feature_rows_read": 0,
            "official_prediction_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "uploads": 0,
        }, 0
    et_model, et_q, et_meta = fit_full_family(
        "extra_trees", historical_route, features, BOOTSTRAP_SEED + 20000
    )
    cb_model, cb_q, cb_meta = fit_full_family(
        "catboost", historical_route, features, BOOTSTRAP_SEED + 30000
    )
    official, cases, champion = load_official_cases(
        [
            column
            for column in features
            if column not in {"selected_base", "selected_delta", "selected_source", "is_lead_24h"}
        ]
    )
    route = build_official_route_rows(official, cases, features)
    et_lcb = np.asarray(et_model.predict(route[features]), dtype=np.float64) - et_q
    cb_lcb = np.asarray(cb_model.predict(route[features]), dtype=np.float64) - cb_q
    DELIVERY_DIR.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    active_mask = official["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    for record in passing[:3]:
        spec = next(item for item in SPECS if item.name == record["spec"]["name"])
        alpha = candidate_route(spec, et_lcb, cb_lcb)
        route_alpha = route[["case_id", "station", "lead_h"]].copy()
        route_alpha["router_alpha"] = alpha
        mapping = official[["case_id", "station", "lead_h"]].merge(
            route_alpha,
            on=["case_id", "station", "lead_h"],
            how="left",
            validate="one_to_one",
        )
        prediction = official["reference"].to_numpy(float).copy()
        prediction[active_mask] = (
            official.loc[active_mask, "base"].to_numpy(float)
            + mapping.loc[active_mask, "router_alpha"].to_numpy(float)
            * official.loc[active_mask, "delta"].to_numpy(float)
        )
        prediction = np.clip(prediction, 0.0, 30.0)
        if not np.array_equal(
            prediction[~active_mask], champion.loc[~active_mask, "hs_pred"].to_numpy(float)
        ):
            raise ContractError("official short lead changed")
        submission = official[KEYS].copy()
        submission["hs_pred"] = prediction
        if len(submission) != 1200 or submission.duplicated(KEYS).any() or not np.isfinite(prediction).all():
            raise ContractError("official submission structure failed")
        payload = submission.to_csv(index=False, lineterminator="\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            raise ContractError("duplicate v6 submission")
        seen.add(digest)
        directory = DELIVERY_DIR / spec.name
        path = directory / "P3_submission.csv"
        write_new(path, payload)
        output = {
            "name": spec.name,
            "path": str(path),
            "rows": 1200,
            "sha256": digest,
            "bytes": len(payload),
            "minimum_m": float(prediction.min()),
            "maximum_m": float(prediction.max()),
            "short_lead_exact_noop": True,
            "changed_rows_vs_champion": int(
                np.sum(np.abs(prediction - champion["hs_pred"].to_numpy(float)) > 1e-12)
            ),
            "calibration": {"extra_trees": et_meta, "catboost": cb_meta},
            "internal": record,
        }
        outputs.append(output)
        write_new(
            directory / "submission-info.txt",
            (
                f"title: {spec.name}\n"
                f"summary: {spec.summary}\n"
                f"sha256: {digest}\n"
                "status: INTERNAL_PASS_MATERIALIZED_NOT_UPLOADED\n"
            ).encode(),
        )
    write_new(
        DELIVERY_DIR / "SET_MANIFEST.json",
        json_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "INTERNAL_PASS_MATERIALIZED_NOT_UPLOADED",
                "outputs": outputs,
                "hidden_truth_rows_read": 0,
                "uploads": 0,
            }
        ),
    )
    return outputs, {
        "test_index_rows_read": 1200,
        "official_test_feature_rows_read": 200,
        "official_prediction_rows_read": 1200 * len(outputs),
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }, 14


def make_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 inner-crossfit LCB router cycle v6",
        "",
        "## 결론",
        "",
        f"- governing internal PASS: **{result['passing_candidate_count']}/{result['candidate_count']}**",
        f"- 제출 CSV: **{len(result['outputs'])}개**, upload 0",
        "- v5 결과를 이용한 threshold 재조정 없이, lead-level expected MSE gain과 train-only inner-crossfit one-sided LCB로 구조를 변경했다.",
        "",
        "| candidate | delta RMSE(m) | CI90 high | P(improve) | blocks | episodes | worst block | C/M/O points | PASS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in result["candidates"]:
        score = record["conditional_leaderboard_translation"]["scenarios"]
        diag = record["diagnostics"]
        lines.append(
            "| {name} | {delta:+.6f} | {hi:+.6f} | {prob:.3f} | {blocks}/6 | {episodes:.3f} | {worst:+.6f} | {c:.6f}/{m:.6f}/{o:.6f} | {passed} |".format(
                name=record["spec"]["name"],
                delta=record["delta_rmse"],
                hi=record["bootstrap"]["ci90_high"],
                prob=record["bootstrap"]["probability_improved"],
                blocks=diag["improved_block_count"],
                episodes=diag["episodes"]["improved_episode_share"],
                worst=diag["worst_block_delta_rmse"],
                c=score["conservative"]["projected_points"],
                m=score["central"]["projected_points"],
                o=score["optimistic"]["projected_points"],
                passed="PASS" if record["strict_internal_pass"] else "FAIL",
            )
        )
    lines += [
        "",
        "## 고정 평가 계약",
        "",
        "- outer 6 bimonth episode-blocked folds, station-local ±78h purge; every outer fold uses five inner block cross-fits.",
        "- LCB is point gain minus the fixed 80th percentile of signed inner-OOF overprediction residuals; selection boundary is metric-aligned zero gain.",
        "- governing gates: structure/finite/physical validity, pooled RMSE improvement, 133-episode bootstrap CI90 upper < 0.",
        "- block win count, episode win share, and worst block are retained as diagnostics, not promotion gates; no official mixture identifies justified hard thresholds for them.",
        "- 18/24h may select only base or frozen alpha=.425 physical expert; 3/6/9/12h are exact champion no-op.",
        "- official inputs are opened only after internal PASS; hidden truth and upload remain zero.",
        "- C/M/O score ranges conditionally map episode CI and internal delta to the current 0.575233m / 24.203599 champion with slope -15.870739 points/m.",
        "- This 1:1 mapping is a planning range only. Historical P3 local-to-official sign reversals make transport uncertain.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("--execute required")
    if ARTIFACT_DIR.exists() or REPORT_DIR.exists() or DELIVERY_DIR.exists() or ATTEMPT_LOCK.exists():
        raise FileExistsError("v6 output or attempt lock already exists")
    for path in (TRAIN_FEATURES, TEST_FEATURES, FEATURE_COLUMNS, V5_RUNNER, V5_RESULT):
        if not path.is_file():
            raise FileNotFoundError(path)
    runner_hash = sha256(Path(__file__))
    lock = {
        "schema_version": "p3.inner_lcb_router.attempt.20260831.v6",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "ATTEMPT_CONSUMED",
        "runner_sha256": runner_hash,
        "v5_runner_sha256": sha256(V5_RUNNER),
        "v5_result_sha256": sha256(V5_RESULT),
        "candidate_specs": [asdict(spec) for spec in SPECS],
        "candidate_count": 3,
        "planned_internal_unique_fits": INTERNAL_UNIQUE_FITS,
        "planned_max_unique_fits": MAX_UNIQUE_FITS,
        "lcb_residual_quantile": LCB_RESIDUAL_QUANTILE,
        "governing_gates": [
            "validity_hard_pass",
            "pooled_rmse_improves",
            "episode_bootstrap_ci90_upper_below_zero",
        ],
        "result_based_retry_or_threshold_tuning": False,
        "official_rows_before_gate": 0,
    }
    write_new(ATTEMPT_LOCK, json_bytes(lock))
    started = time.perf_counter()
    frame, cases, profile, case_features = build_historical_cases()
    route, features = build_route_rows(frame, cases, case_features)
    records, oof = evaluate_candidates(frame, route, features)
    passing = [record for record in records if record["strict_internal_pass"]]
    outputs, access, official_fits = materialize(passing, route, features)
    result = {
        "schema_version": "p3.inner_lcb_router.result.20260831.v6",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE_INTERNAL_PASS_READY_NOT_UPLOADED" if outputs else "COMPLETE_NO_INTERNAL_PASS",
        "runtime_seconds": float(time.perf_counter() - started),
        "runner_sha256": runner_hash,
        "attempt_lock_sha256": sha256(ATTEMPT_LOCK),
        "v5_runner_sha256": sha256(V5_RUNNER),
        "v5_result_sha256": sha256(V5_RESULT),
        "source_hashes": {
            "train_features": sha256(TRAIN_FEATURES),
            "feature_columns": sha256(FEATURE_COLUMNS),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data_profile": {
            **profile,
            "lead_router_rows": int(len(route)),
            "active_leads": list(ACTIVE_LEADS),
            "lcb_residual_quantile": LCB_RESIDUAL_QUANTILE,
        },
        "internal_comparator": "OFFICIAL_CHAMPION_LINEAGE_UNIFORM_KMA_ALPHA_0P425_OOF_PROXY",
        "candidate_count": len(records),
        "passing_candidate_count": len(passing),
        "internal_unique_fit_count": INTERNAL_UNIQUE_FITS,
        "official_full_fit_count": official_fits,
        "total_unique_fit_count": INTERNAL_UNIQUE_FITS + official_fits,
        "candidates": records,
        "outputs": outputs,
        "official_access": access,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
        "result_based_retry_or_threshold_tuning": False,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    oof_path = ARTIFACT_DIR / "internal_oof_active_leads.parquet"
    oof.to_parquet(oof_path, index=False)
    result["internal_oof_active_leads"] = {
        "path": str(oof_path),
        "rows": int(len(oof)),
        "sha256": sha256(oof_path),
    }
    write_new(ARTIFACT_DIR / "result.json", json_bytes(result))
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    write_new(REPORT_DIR / "report-source.md", make_report(result).encode())
    write_new(
        REPORT_DIR / "run-manifest.json",
        json_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "result_sha256": sha256(ARTIFACT_DIR / "result.json"),
                "oof_sha256": sha256(oof_path),
                "runner_sha256": runner_hash,
                "attempt_lock_sha256": sha256(ATTEMPT_LOCK),
                "total_unique_fit_count": INTERNAL_UNIQUE_FITS + official_fits,
                "passing_candidate_count": len(passing),
                "hidden_truth_rows_read": 0,
                "uploads": 0,
            }
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
