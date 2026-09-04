"""Train and score frozen case-level physical expert routers for P3.

Each model predicts whether the existing alpha=0.425 KMA expert beats the
unadjusted base for a historical case.  The router may only interpolate between
those two physical experts; it cannot emit an unconstrained residual.  Official
test features are opened only if a candidate passes every historical gate.
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
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for entry in (SCRIPTS, SRC):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

from run_p3_parallel_candidate_cycle_20260831_v4 import (  # noqa: E402
    ACTIVE_LEADS,
    ALPHA_MAX,
    BASE_PATH,
    BOOTSTRAP_SEED,
    CHAMPION_PATH,
    KEYS,
    P3_DATA,
    REFERENCE_ALPHA,
    SOURCE_PATH,
    episode_bootstrap,
    episode_diagnostics,
    load_historical,
    purge_training_cases,
    rmse,
)

EXPERIMENT_ID = "p3_physical_expert_router_cycle_20260831_v5"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
DELIVERY_DIR = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_PHYSICAL_EXPERT_ROUTER_V5"
)
ATTEMPT_LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
TRAIN_FEATURES = ROOT / "artifacts/p3/features_all20_v1/train_features.parquet"
TEST_FEATURES = ROOT / "artifacts/p3/features_all20_v1/test_features.parquet"
FEATURE_COLUMNS = ROOT / "submissions/p3_frozen_catboost/feature_columns.json"
V4_RUNNER = ROOT / "scripts/run_p3_parallel_candidate_cycle_20260831_v4.py"
OFFICIAL_CHAMPION_RMSE_M = 0.575233
OFFICIAL_CHAMPION_POINTS = 24.203599
POINT_SLOPE_PER_RMSE_M = -15.870739046986959


class ContractError(RuntimeError):
    """Raised when a frozen v5 contract is violated."""


@dataclass(frozen=True)
class RouterSpec:
    name: str
    family: str
    policy: str
    summary: str


SPECS = (
    RouterSpec(
        "P3_1_CATBOOST_SOFT_PHYSICAL_ROUTER",
        "catboost",
        "soft_probability",
        "CatBoost case-level advantage probability blends only base and the frozen KMA champion.",
    ),
    RouterSpec(
        "P3_2_EXTRATREES_HARD_PHYSICAL_ROUTER",
        "extra_trees",
        "hard_0p50",
        "ExtraTrees case-level advantage classifier selects only base or the frozen KMA champion.",
    ),
    RouterSpec(
        "P3_3_LOGISTIC_ABSTAIN_PHYSICAL_ROUTER",
        "logistic",
        "abstain_0p40",
        "Regularized logistic router falls back to base only when champion advantage probability is below 0.40.",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _trajectory_features(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    required = {id_column, "station", "lead_h", "base", "delta", "current_hs"}
    if not required.issubset(frame.columns):
        raise ContractError("trajectory frame schema changed")
    rows: list[dict[str, Any]] = []
    for (identifier, station), group in frame.groupby([id_column, "station"], observed=True, sort=False):
        by_lead = group.set_index("lead_h")
        if set(by_lead.index.astype(int)) != {3, 6, 9, 12, 18, 24}:
            raise ContractError("trajectory case does not have six leads")
        row: dict[str, Any] = {
            id_column: identifier,
            "station": station,
            "route_current_hs": float(group["current_hs"].iloc[0]),
        }
        for lead in (3, 6, 9, 12, 18, 24):
            row[f"route_base_{lead}"] = float(by_lead.loc[lead, "base"])
        for lead in ACTIVE_LEADS:
            row[f"route_delta_{lead}"] = float(by_lead.loc[lead, "delta"])
            row[f"route_source_{lead}"] = float(by_lead.loc[lead, "base"] + by_lead.loc[lead, "delta"])
        row["route_base_slope_3_24"] = row["route_base_24"] - row["route_base_3"]
        row["route_base_curvature"] = row["route_base_24"] - 2.0 * row["route_base_12"] + row["route_base_3"]
        row["route_delta_abs_sum"] = abs(row["route_delta_18"]) + abs(row["route_delta_24"])
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.duplicated([id_column, "station"]).any():
        raise ContractError("trajectory feature key duplicated")
    return result


def build_historical_cases() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    frame, base_profile = load_historical()
    configured = json.loads(FEATURE_COLUMNS.read_text(encoding="utf-8"))
    if not isinstance(configured, list) or len(configured) != 591 or len(set(configured)) != 591:
        raise ContractError("frozen 591 feature contract changed")
    raw = pd.read_parquet(TRAIN_FEATURES, columns=["anchor_id", "station", *configured])
    if raw.duplicated(["anchor_id", "station"]).any():
        raise ContractError("historical feature key duplicated")
    trajectory = _trajectory_features(frame, "anchor_id")
    case_meta = (
        frame[["anchor_id", "station", "anchor_time", "block", "episode_id"]]
        .drop_duplicates()
        .copy()
    )
    if case_meta.duplicated(["anchor_id", "station"]).any():
        raise ContractError("historical case metadata key duplicated")
    gain = (
        frame.assign(
            base_sse=np.square(frame["target_hs"] - frame["base"]),
            reference_sse=np.square(frame["target_hs"] - frame["reference"]),
        )
        .groupby(["anchor_id", "station"], observed=True)[["base_sse", "reference_sse"]]
        .sum()
        .reset_index()
    )
    gain["advantage_mse_gain"] = gain["base_sse"] - gain["reference_sse"]
    gain["champion_better"] = gain["advantage_mse_gain"].gt(0.0).astype(np.int8)
    cases = case_meta.merge(gain, on=["anchor_id", "station"], validate="one_to_one")
    cases = cases.merge(trajectory, on=["anchor_id", "station"], validate="one_to_one")
    cases = cases.merge(raw, on=["anchor_id", "station"], validate="one_to_one")
    feature_names = [
        *configured,
        *[column for column in trajectory.columns if column not in {"anchor_id", "station"}],
        "route_station_g",
        "route_station_i",
        "route_station_s",
    ]
    cases["route_station_g"] = cases["station"].eq("G-ORS").astype(np.float64)
    cases["route_station_i"] = cases["station"].eq("I-ORS").astype(np.float64)
    cases["route_station_s"] = cases["station"].eq("S-ORS").astype(np.float64)
    if len(cases) != 182 or cases.duplicated(["anchor_id", "station"]).any():
        raise ContractError("historical router case grain changed")
    matrix = cases[feature_names].apply(pd.to_numeric, errors="coerce")
    if matrix.notna().sum(axis=0).eq(0).any():
        raise ContractError("router feature has no historical finite support")
    profile = {
        **base_profile,
        "router_cases": int(len(cases)),
        "router_features": int(len(feature_names)),
        "champion_better_cases": int(cases["champion_better"].sum()),
        "champion_better_share": float(cases["champion_better"].mean()),
        "feature_missing_fraction": float(matrix.isna().to_numpy().mean()),
        "official_feature_rows_read_before_internal_gate": 0,
    }
    return frame, cases, profile, feature_names


def build_estimator(spec: RouterSpec, seed: int) -> Any:
    if spec.family == "catboost":
        return CatBoostClassifier(
            iterations=350,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=20.0,
            random_strength=0.5,
            loss_function="Logloss",
            random_seed=int(seed),
            thread_count=6,
            verbose=False,
            allow_writing_files=False,
        )
    if spec.family == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=8,
                max_features=0.25,
                class_weight="balanced",
                random_state=int(seed),
                n_jobs=6,
            ),
        )
    if spec.family == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(
                C=0.05,
                penalty="l2",
                class_weight="balanced",
                solver="liblinear",
                max_iter=2_000,
                random_state=int(seed),
            ),
        )
    raise ContractError(f"unknown router family: {spec.family}")


def route_alpha(probability: np.ndarray, policy: str) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    if policy == "soft_probability":
        alpha = REFERENCE_ALPHA * p
    elif policy == "hard_0p50":
        alpha = np.where(p >= 0.50, REFERENCE_ALPHA, 0.0)
    elif policy == "abstain_0p40":
        alpha = np.where(p < 0.40, 0.0, REFERENCE_ALPHA)
    else:
        raise ContractError(f"unknown router policy: {policy}")
    if not np.isfinite(alpha).all() or alpha.min() < 0.0 or alpha.max() > ALPHA_MAX:
        raise ContractError("router alpha violates the physical bounds")
    return alpha


def conditional_score_translation(
    delta_rmse: float, ci90_low: float, ci90_high: float
) -> dict[str, Any]:
    """Translate internal deltas under an explicitly conditional 1:1 transport assumption."""

    scenarios = {
        "conservative": float(ci90_high),
        "central": float(delta_rmse),
        "optimistic": float(ci90_low),
    }
    projected: dict[str, Any] = {}
    for name, delta in scenarios.items():
        projected[name] = {
            "assumed_official_delta_rmse_m": delta,
            "projected_official_rmse_m": OFFICIAL_CHAMPION_RMSE_M + delta,
            "projected_point_delta": POINT_SLOPE_PER_RMSE_M * delta,
            "projected_points": OFFICIAL_CHAMPION_POINTS + POINT_SLOPE_PER_RMSE_M * delta,
        }
    return {
        "status": "CONDITIONAL_PLANNING_ESTIMATE_NOT_A_PROMOTION_GATE",
        "official_anchor": {
            "rmse_m": OFFICIAL_CHAMPION_RMSE_M,
            "points": OFFICIAL_CHAMPION_POINTS,
        },
        "linear_score_slope_points_per_rmse_m": POINT_SLOPE_PER_RMSE_M,
        "scenarios": projected,
        "transport_caveat": (
            "Historical internal delta and episode CI are mapped 1:1 only for planning. "
            "Past P3 local-to-Public direction reversals show that actual official transport "
            "can differ in sign and magnitude."
        ),
    }


def _sample_weight(gain: pd.Series) -> np.ndarray:
    magnitude = np.abs(gain.to_numpy(dtype=np.float64))
    positive = magnitude[magnitude > 0.0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return np.clip(magnitude / max(scale, 1e-8), 0.25, 4.0)


def expand_case_alpha(frame: pd.DataFrame, case_alpha: pd.DataFrame) -> np.ndarray:
    mapping = frame[["anchor_id", "station"]].merge(
        case_alpha, on=["anchor_id", "station"], how="left", validate="many_to_one"
    )
    if mapping["router_alpha"].isna().any():
        raise ContractError("case alpha expansion left missing rows")
    alpha = mapping["router_alpha"].to_numpy(dtype=np.float64)
    prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
    active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    prediction[active] = (
        frame.loc[active, "base"].to_numpy(float)
        + alpha[active] * frame.loc[active, "delta"].to_numpy(float)
    )
    prediction = np.clip(prediction, 0.0, 30.0)
    if not np.array_equal(
        prediction[~active], frame.loc[~active, "reference"].to_numpy(dtype=np.float64)
    ):
        raise ContractError("router changed a short lead")
    return prediction


def evaluate(
    frame: pd.DataFrame,
    cases: pd.DataFrame,
    feature_names: list[str],
    spec: RouterSpec,
    spec_index: int,
) -> tuple[dict[str, Any], Any]:
    probability = np.full(len(cases), np.nan, dtype=np.float64)
    blocks = sorted(cases["block"].astype(str).unique())
    fit_records: dict[str, Any] = {}
    for block_index, block in enumerate(blocks):
        test_mask = cases["block"].astype(str).eq(block).to_numpy()
        test = cases.loc[test_mask].copy()
        train = purge_training_cases(cases.loc[~test_mask].copy(), test)
        estimator = build_estimator(spec, BOOTSTRAP_SEED + 100 * spec_index + block_index)
        estimator.fit(
            train[feature_names],
            train["champion_better"].to_numpy(dtype=np.int8),
            **(
                {"sample_weight": _sample_weight(train["advantage_mse_gain"])}
                if spec.family == "catboost"
                else {
                    "extratreesclassifier__sample_weight": _sample_weight(train["advantage_mse_gain"])
                }
                if spec.family == "extra_trees"
                else {"logisticregression__sample_weight": _sample_weight(train["advantage_mse_gain"])}
            ),
        )
        probability[test_mask] = estimator.predict_proba(test[feature_names])[:, 1]
        fit_records[block] = {
            "train_cases_after_78h_purge": int(len(train)),
            "test_cases": int(len(test)),
        }
    if not np.isfinite(probability).all():
        raise ContractError("router cross-fit probability is incomplete")
    alpha = route_alpha(probability, spec.policy)
    case_alpha = cases[["anchor_id", "station"]].copy()
    case_alpha["router_alpha"] = alpha
    prediction = expand_case_alpha(frame, case_alpha)
    y = frame["target_hs"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    before = rmse(y, reference)
    after = rmse(y, prediction)
    by_block: dict[str, Any] = {}
    for block in blocks:
        mask = frame["block"].astype(str).eq(block).to_numpy()
        block_before = rmse(y[mask], reference[mask])
        block_after = rmse(y[mask], prediction[mask])
        by_block[block] = {
            **fit_records[block],
            "rows": int(mask.sum()),
            "reference_rmse": block_before,
            "candidate_rmse": block_after,
            "delta_rmse": block_after - block_before,
        }
    episodes = episode_diagnostics(frame, prediction)
    bootstrap = episode_bootstrap(frame, prediction)
    improved_blocks = int(sum(value["delta_rmse"] < 0.0 for value in by_block.values()))
    worst = float(max(value["delta_rmse"] for value in by_block.values()))
    gates = {
        "pooled_rmse_improves": after < before,
        "strict_majority_episodes_improve": episodes["improved_episode_share"] > 0.50,
        "at_least_four_of_six_blocks_improve": improved_blocks >= 4,
        "bootstrap_probability_at_least_0p80": bootstrap["probability_improved"] >= 0.80,
        "bootstrap_ci90_upper_below_zero": bootstrap["ci90_high"] < 0.0,
        "catastrophic_block_guard_max_degradation_le_0p01m": worst <= 0.01,
    }
    record = {
        "spec": asdict(spec),
        "fit_count": 6,
        "reference_rmse": before,
        "candidate_rmse": after,
        "delta_rmse": after - before,
        "by_block": by_block,
        "improved_block_count": improved_blocks,
        "worst_block_delta_rmse": worst,
        "episodes": episodes,
        "bootstrap": bootstrap,
        "conditional_leaderboard_translation": conditional_score_translation(
            after - before,
            bootstrap["ci90_low"],
            bootstrap["ci90_high"],
        ),
        "routing": {
            "probability_min": float(probability.min()),
            "probability_median": float(np.median(probability)),
            "probability_max": float(probability.max()),
            "base_route_share": float(np.mean(alpha == 0.0)),
            "champion_route_share": float(np.mean(alpha == REFERENCE_ALPHA)),
            "soft_route_share": float(np.mean((alpha > 0.0) & (alpha < REFERENCE_ALPHA))),
        },
        "gates": gates,
        "strict_internal_pass": bool(all(gates.values())),
    }
    full = build_estimator(spec, BOOTSTRAP_SEED + 1_000 + spec_index)
    full.fit(
        cases[feature_names],
        cases["champion_better"].to_numpy(dtype=np.int8),
        **(
            {"sample_weight": _sample_weight(cases["advantage_mse_gain"])}
            if spec.family == "catboost"
            else {
                "extratreesclassifier__sample_weight": _sample_weight(cases["advantage_mse_gain"])
            }
            if spec.family == "extra_trees"
            else {"logisticregression__sample_weight": _sample_weight(cases["advantage_mse_gain"])}
        ),
    )
    return record, full


def load_official_cases(feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_index = pd.read_csv(P3_DATA / "test_index.csv", dtype={"case_id": "string", "station": "string"})
    base = pd.read_csv(BASE_PATH, dtype={"case_id": "string", "station": "string"})
    source = pd.read_csv(SOURCE_PATH, dtype={"case_id": "string", "station": "string"})
    champion = pd.read_csv(CHAMPION_PATH, dtype={"case_id": "string", "station": "string"})
    official = test_index.copy()
    official["base"] = base["hs_pred"].to_numpy(float)
    official["delta"] = source["hs_pred"].to_numpy(float) - official["base"]
    official["reference"] = champion["hs_pred"].to_numpy(float)
    raw = pd.read_parquet(TEST_FEATURES)
    if raw.duplicated(["case_id", "station"]).any() or len(raw) != 200:
        raise ContractError("official test feature key/grain changed")
    current = raw[["case_id", "station", "hs_current"]].rename(columns={"hs_current": "current_hs"})
    official = official.merge(current, on=["case_id", "station"], how="left", validate="many_to_one")
    trajectory = _trajectory_features(official, "case_id")
    cases = trajectory.merge(raw, on=["case_id", "station"], validate="one_to_one")
    cases["route_station_g"] = cases["station"].eq("G-ORS").astype(np.float64)
    cases["route_station_i"] = cases["station"].eq("I-ORS").astype(np.float64)
    cases["route_station_s"] = cases["station"].eq("S-ORS").astype(np.float64)
    missing = set(feature_names) - set(cases.columns)
    if missing:
        raise ContractError(f"official router features missing: {sorted(missing)[:5]}")
    if not test_index[KEYS].equals(base[KEYS]) or not test_index[KEYS].equals(source[KEYS]) or not test_index[KEYS].equals(champion[KEYS]):
        raise ContractError("official key order changed")
    return official, cases, champion


def materialize(
    passing: list[tuple[dict[str, Any], Any]], feature_names: list[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not passing:
        return [], {
            "test_index_rows_read": 0,
            "official_test_feature_rows_read": 0,
            "official_prediction_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "uploads": 0,
        }
    official, cases, champion = load_official_cases(feature_names)
    DELIVERY_DIR.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record, model in passing[:3]:
        probability = model.predict_proba(cases[feature_names])[:, 1]
        alpha = route_alpha(probability, record["spec"]["policy"])
        case_alpha = cases[["case_id", "station"]].copy()
        case_alpha["router_alpha"] = alpha
        mapping = official[["case_id", "station"]].merge(
            case_alpha, on=["case_id", "station"], how="left", validate="many_to_one"
        )
        active = official["lead_h"].isin(ACTIVE_LEADS).to_numpy()
        prediction = official["reference"].to_numpy(float).copy()
        prediction[active] = (
            official.loc[active, "base"].to_numpy(float)
            + mapping.loc[active, "router_alpha"].to_numpy(float)
            * official.loc[active, "delta"].to_numpy(float)
        )
        prediction = np.clip(prediction, 0.0, 30.0)
        inactive = ~active
        if not np.array_equal(prediction[inactive], champion.loc[inactive, "hs_pred"].to_numpy(float)):
            raise ContractError("official short lead changed")
        submission = official[KEYS].copy()
        submission["hs_pred"] = prediction
        if len(submission) != 1_200 or submission.duplicated(KEYS).any() or not np.isfinite(prediction).all():
            raise ContractError("official submission structure failed")
        payload = submission.to_csv(index=False, lineterminator="\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            raise ContractError("duplicate v5 submission")
        seen.add(digest)
        directory = DELIVERY_DIR / record["spec"]["name"]
        path = directory / "P3_submission.csv"
        write_new(path, payload)
        info = {
            "name": record["spec"]["name"],
            "path": str(path),
            "rows": 1_200,
            "sha256": digest,
            "bytes": len(payload),
            "minimum_m": float(prediction.min()),
            "maximum_m": float(prediction.max()),
            "short_lead_exact_noop": True,
            "changed_rows_vs_champion": int(
                np.sum(np.abs(prediction - champion["hs_pred"].to_numpy(float)) > 1e-12)
            ),
            "rms_change_vs_champion_m": float(
                np.sqrt(np.mean(np.square(prediction - champion["hs_pred"].to_numpy(float))))
            ),
            "internal": record,
        }
        outputs.append(info)
        write_new(
            directory / "submission-info.txt",
            (
                f"title: {record['spec']['name']}\n"
                f"summary: {record['spec']['summary']}\n"
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
        "test_index_rows_read": 1_200,
        "official_test_feature_rows_read": 200,
        "official_prediction_rows_read": 3_600,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }


def make_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 physical expert router cycle v5",
        "",
        "## 결론",
        "",
        f"- 엄격 내부 PASS: **{result['passing_candidate_count']}/{result['candidate_count']}**",
        f"- 제출 CSV: **{len(result['outputs'])}개**, upload 0",
        "- v4의 연속 alpha 회귀를 중단하고, 과거 591 past-only feature와 six-lead 궤적으로 champion-vs-base advantage를 분류했다.",
        "",
        "| candidate | delta RMSE(m) | blocks | episodes | P(improve) | conditional points C/M/O | PASS |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in result["candidates"]:
        lines.append(
            "| {name} | {delta:+.6f} | {blocks}/6 | {episodes:.3f} | {prob:.3f} | {conservative:.6f}/{central:.6f}/{optimistic:.6f} | {passed} |".format(
                name=record["spec"]["name"],
                delta=record["delta_rmse"],
                blocks=record["improved_block_count"],
                episodes=record["episodes"]["improved_episode_share"],
                prob=record["bootstrap"]["probability_improved"],
                conservative=record["conditional_leaderboard_translation"]["scenarios"]["conservative"]["projected_points"],
                central=record["conditional_leaderboard_translation"]["scenarios"]["central"]["projected_points"],
                optimistic=record["conditional_leaderboard_translation"]["scenarios"]["optimistic"]["projected_points"],
                passed="PASS" if record["strict_internal_pass"] else "FAIL",
            )
        )
    lines += [
        "",
        "## 계약과 QA",
        "",
        "- 6 bimonth holdouts, station-local ±78h purge; output target never enters features.",
        "- 133 contiguous historical episodes, 5,000-replicate episode bootstrap.",
        "- Router output is base↔alpha0.425 champion only; 3/6/9/12h exact no-op.",
        "- PASS requires pooled improvement, episode majority, 4/6 blocks, P>=0.8, CI90 upper<0, worst block<=+0.01m.",
        "- official inputs remain unopened unless a strict PASS exists; hidden truth and upload are always zero.",
        "- Conditional score ranges use current Public 0.575233m / 24.203599 points and slope -15.870739 points/m.",
        "- The range is a 1:1 internal-to-official planning translation, not evidence of transport; prior P3 direction reversals remain a required caveat.",
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
        raise FileExistsError("v5 output or attempt lock already exists")
    for path in (TRAIN_FEATURES, TEST_FEATURES, FEATURE_COLUMNS, V4_RUNNER):
        if not path.is_file():
            raise FileNotFoundError(path)
    runner_hash = sha256(Path(__file__))
    lock = {
        "schema_version": "p3.physical_expert_router.attempt.20260831.v5",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "ATTEMPT_CONSUMED",
        "runner_sha256": runner_hash,
        "v4_dependency_sha256": sha256(V4_RUNNER),
        "candidate_specs": [asdict(spec) for spec in SPECS],
        "candidate_count": 3,
        "planned_fit_count": 21,
        "gate_or_threshold_adaptation_after_scoring": False,
        "official_rows_before_gate": 0,
    }
    write_new(ATTEMPT_LOCK, json_bytes(lock))
    started = time.perf_counter()
    frame, cases, profile, feature_names = build_historical_cases()
    records: list[dict[str, Any]] = []
    passing: list[tuple[dict[str, Any], Any]] = []
    for index, spec in enumerate(SPECS):
        record, model = evaluate(frame, cases, feature_names, spec, index)
        records.append(record)
        if record["strict_internal_pass"]:
            passing.append((record, model))
    outputs, access = materialize(passing, feature_names)
    result = {
        "schema_version": "p3.physical_expert_router.result.20260831.v5",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE_INTERNAL_PASS_READY_NOT_UPLOADED" if outputs else "COMPLETE_NO_INTERNAL_PASS",
        "runtime_seconds": float(time.perf_counter() - started),
        "runner_sha256": runner_hash,
        "attempt_lock_sha256": sha256(ATTEMPT_LOCK),
        "v4_dependency_sha256": sha256(V4_RUNNER),
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
        "data_profile": profile,
        "internal_comparator": "OFFICIAL_CHAMPION_LINEAGE_UNIFORM_KMA_ALPHA_0P425_OOF_PROXY",
        "candidate_count": len(records),
        "passing_candidate_count": len(passing),
        "fit_count": 21,
        "candidates": records,
        "outputs": outputs,
        "official_access": access,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
        "result_based_retry_or_threshold_tuning": False,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    write_new(ARTIFACT_DIR / "result.json", json_bytes(result))
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    write_new(REPORT_DIR / "report-source.md", make_report(result).encode())
    write_new(
        REPORT_DIR / "run-manifest.json",
        json_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "result_sha256": sha256(ARTIFACT_DIR / "result.json"),
                "runner_sha256": runner_hash,
                "attempt_lock_sha256": sha256(ATTEMPT_LOCK),
                "fit_count": 21,
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
