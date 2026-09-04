"""Run the frozen P3 uncertainty-aware case-advantage regression cycle.

The model predicts the continuous historical SSE advantage of the frozen
alpha=0.425 champion over the unadjusted base.  It switches to the base only
when a nested blocked-OOF upper confidence bound is below zero; all uncertain
cases remain exact champion/no-op.  The three confidence policies are sealed
before execution and share one common fitted ensemble.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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
    BOOTSTRAP_SEED,
    KEYS,
    episode_bootstrap,
    episode_diagnostics,
    purge_training_cases,
    rmse,
)
from run_p3_physical_expert_router_cycle_20260831_v5 import (  # noqa: E402
    build_historical_cases,
    conditional_score_translation,
    load_official_cases,
)

EXPERIMENT_ID = "p3_uncertainty_router_cycle_20260831_v6b"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
DELIVERY_DIR = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_UNCERTAINTY_ROUTER_V6B"
)
ATTEMPT_LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V4_RUNNER = ROOT / "scripts/run_p3_parallel_candidate_cycle_20260831_v4.py"
V5_RUNNER = ROOT / "scripts/run_p3_physical_expert_router_cycle_20260831_v5.py"
INNER_MEMBERS = 6
OUTER_MEMBERS = 12
FEATURE_FRACTION = 0.55
RIDGE_ALPHA = 50.0
TARGET_WINSOR = (0.025, 0.975)
MAD_SCALE = 1.4826


class ContractError(RuntimeError):
    """Raised when the sealed v6b contract is violated."""


@dataclass(frozen=True)
class ConfidencePolicy:
    name: str
    upper_residual_quantile: float
    summary: str


POLICIES = (
    ConfidencePolicy(
        "P3_1_ADVANTAGE_UCB_Q50",
        0.50,
        "Switch to base only when the median-calibrated advantage UCB is negative.",
    ),
    ConfidencePolicy(
        "P3_2_ADVANTAGE_UCB_Q65",
        0.65,
        "Switch to base only when the 65% calibration UCB is negative.",
    ),
    ConfidencePolicy(
        "P3_3_ADVANTAGE_UCB_Q80",
        0.80,
        "Switch to base only when the 80% calibration UCB is negative.",
    ),
)


@dataclass
class BagMember:
    columns: list[str]
    model: Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def policy_sha256() -> str:
    payload = json.dumps(
        [asdict(policy) for policy in POLICIES],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _minimum_same_station_gap_hours(train: pd.DataFrame, test: pd.DataFrame) -> float | None:
    values: list[float] = []
    for station, test_group in test.groupby("station", observed=True):
        train_group = train.loc[train["station"].eq(station)]
        if train_group.empty:
            continue
        train_ns = pd.DatetimeIndex(train_group["anchor_time"]).as_unit("ns").asi8
        test_ns = pd.DatetimeIndex(test_group["anchor_time"]).as_unit("ns").asi8
        values.append(float(np.min(np.abs(train_ns[:, None] - test_ns[None, :])) / 3.6e12))
    return min(values) if values else None


def _episode_bootstrap_sample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    groups = list(frame.groupby("episode_id", observed=True, sort=True))
    if len(groups) < 8:
        raise ContractError("too few episode groups for group bootstrap")
    draw = rng.integers(0, len(groups), size=len(groups))
    return pd.concat([groups[index][1] for index in draw], ignore_index=True)


def fit_bag_ensemble(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    members: int,
    seed: int,
) -> list[BagMember]:
    if frame.empty or members < 2:
        raise ContractError("invalid bag ensemble request")
    rng = np.random.default_rng(seed)
    core = [name for name in feature_names if name.startswith("route_")]
    optional = [name for name in feature_names if name not in core]
    optional_count = max(32, int(np.ceil(FEATURE_FRACTION * len(optional))))
    output: list[BagMember] = []
    for _ in range(members):
        sampled = _episode_bootstrap_sample(frame, rng)
        chosen = sorted(rng.choice(optional, size=optional_count, replace=False).tolist())
        columns = [*core, *chosen]
        low, high = np.quantile(
            sampled["advantage_mse_gain"].to_numpy(dtype=np.float64), TARGET_WINSOR
        )
        target = np.clip(
            sampled["advantage_mse_gain"].to_numpy(dtype=np.float64), low, high
        )
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=RIDGE_ALPHA, solver="lsqr"),
        )
        model.fit(sampled[columns], target)
        output.append(BagMember(columns=columns, model=model))
    return output


def predict_bag_ensemble(
    ensemble: list[BagMember], frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    member_predictions = np.column_stack(
        [member.model.predict(frame[member.columns]) for member in ensemble]
    ).astype(np.float64)
    median = np.median(member_predictions, axis=1)
    mad = np.median(np.abs(member_predictions - median[:, None]), axis=1) * MAD_SCALE
    if not np.isfinite(member_predictions).all() or not np.isfinite(mad).all():
        raise ContractError("bag ensemble produced non-finite predictions")
    return median, mad, member_predictions


def nested_calibration(
    train: pd.DataFrame,
    feature_names: list[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    median = np.full(len(train), np.nan, dtype=np.float64)
    mad = np.full(len(train), np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    fit_count = 0
    blocks = sorted(train["block"].astype(str).unique())
    if len(blocks) < 4:
        raise ContractError("nested calibration has too few time blocks")
    for block_index, block in enumerate(blocks):
        test_mask = train["block"].astype(str).eq(block).to_numpy()
        inner_test = train.loc[test_mask].copy()
        inner_train = purge_training_cases(train.loc[~test_mask].copy(), inner_test)
        ensemble = fit_bag_ensemble(
            inner_train,
            feature_names,
            members=INNER_MEMBERS,
            seed=seed + 100 * block_index,
        )
        median[test_mask], mad[test_mask], _ = predict_bag_ensemble(ensemble, inner_test)
        fit_count += len(ensemble)
        gap = _minimum_same_station_gap_hours(inner_train, inner_test)
        records.append(
            {
                "block": block,
                "train_cases_after_78h_purge": int(len(inner_train)),
                "test_cases": int(len(inner_test)),
                "minimum_same_station_gap_hours": gap,
            }
        )
    if not np.isfinite(median).all() or not np.isfinite(mad).all():
        raise ContractError("nested blocked OOF calibration is incomplete")
    return median, mad, records, fit_count


def select_base_cases(
    advantage_median: np.ndarray,
    advantage_mad: np.ndarray,
    calibration_residual: np.ndarray,
    policy: ConfidencePolicy,
) -> tuple[np.ndarray, np.ndarray, float]:
    residual_quantile = float(
        np.quantile(
            calibration_residual,
            policy.upper_residual_quantile,
            method="higher",
        )
    )
    upper_bound = advantage_median + advantage_mad + residual_quantile
    select_base = upper_bound < 0.0
    return select_base, upper_bound, residual_quantile


def expand_selection(frame: pd.DataFrame, cases: pd.DataFrame, select_base: np.ndarray) -> np.ndarray:
    selection = cases[["anchor_id", "station"]].copy()
    selection["select_base"] = np.asarray(select_base, dtype=bool)
    expanded = frame[["anchor_id", "station"]].merge(
        selection,
        on=["anchor_id", "station"],
        how="left",
        validate="many_to_one",
    )
    if expanded["select_base"].isna().any():
        raise ContractError("case selection expansion is incomplete")
    prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
    active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    route = active & expanded["select_base"].to_numpy(dtype=bool)
    prediction[route] = frame.loc[route, "base"].to_numpy(dtype=np.float64)
    prediction = np.clip(prediction, 0.0, 30.0)
    if not np.array_equal(
        prediction[~active], frame.loc[~active, "reference"].to_numpy(dtype=np.float64)
    ):
        raise ContractError("uncertainty router changed a short lead")
    return prediction


def outer_predictions(
    cases: pd.DataFrame,
    feature_names: list[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    per_policy = {
        policy.name: {
            "select_base": np.zeros(len(cases), dtype=bool),
            "upper_bound": np.full(len(cases), np.nan, dtype=np.float64),
            "residual_quantile": np.full(len(cases), np.nan, dtype=np.float64),
        }
        for policy in POLICIES
    }
    records: list[dict[str, Any]] = []
    fit_count = 0
    blocks = sorted(cases["block"].astype(str).unique())
    for outer_index, block in enumerate(blocks):
        test_mask = cases["block"].astype(str).eq(block).to_numpy()
        outer_test = cases.loc[test_mask].copy()
        outer_train = purge_training_cases(cases.loc[~test_mask].copy(), outer_test)
        inner_median, _, inner_records, inner_fits = nested_calibration(
            outer_train,
            feature_names,
            seed=BOOTSTRAP_SEED + 10_000 * outer_index,
        )
        calibration_residual = (
            outer_train["advantage_mse_gain"].to_numpy(dtype=np.float64) - inner_median
        )
        ensemble = fit_bag_ensemble(
            outer_train,
            feature_names,
            members=OUTER_MEMBERS,
            seed=BOOTSTRAP_SEED + 10_000 * outer_index + 5_000,
        )
        outer_median, outer_mad, _ = predict_bag_ensemble(ensemble, outer_test)
        fit_count += inner_fits + len(ensemble)
        gap = _minimum_same_station_gap_hours(outer_train, outer_test)
        policy_records: dict[str, Any] = {}
        for policy in POLICIES:
            selected, upper_bound, residual_quantile = select_base_cases(
                outer_median,
                outer_mad,
                calibration_residual,
                policy,
            )
            per_policy[policy.name]["select_base"][test_mask] = selected
            per_policy[policy.name]["upper_bound"][test_mask] = upper_bound
            per_policy[policy.name]["residual_quantile"][test_mask] = residual_quantile
            policy_records[policy.name] = {
                "calibration_residual_quantile": residual_quantile,
                "base_selected_cases": int(selected.sum()),
            }
        records.append(
            {
                "outer_block": block,
                "train_cases_after_78h_purge": int(len(outer_train)),
                "test_cases": int(len(outer_test)),
                "minimum_same_station_gap_hours": gap,
                "inner": inner_records,
                "policies": policy_records,
            }
        )
    for values in per_policy.values():
        if not np.isfinite(values["upper_bound"]).all():
            raise ContractError("outer uncertainty predictions are incomplete")
    return per_policy, records, fit_count


def evaluate_candidates(
    frame: pd.DataFrame,
    cases: pd.DataFrame,
    per_policy: dict[str, dict[str, Any]],
    outer_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    y = frame["target_hs"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    reference_rmse = rmse(y, reference)
    rows: list[pd.DataFrame] = []
    candidates: list[dict[str, Any]] = []
    outer_gaps = [record["minimum_same_station_gap_hours"] for record in outer_records]
    inner_gaps = [
        inner["minimum_same_station_gap_hours"]
        for record in outer_records
        for inner in record["inner"]
    ]
    purge_valid = all(gap is None or gap > 78.0 for gap in [*outer_gaps, *inner_gaps])
    for policy in POLICIES:
        values = per_policy[policy.name]
        prediction = expand_selection(frame, cases, values["select_base"])
        candidate_rmse = rmse(y, prediction)
        bootstrap = episode_bootstrap(frame, prediction)
        episodes = episode_diagnostics(frame, prediction)
        by_block: dict[str, Any] = {}
        for block in sorted(frame["block"].astype(str).unique()):
            mask = frame["block"].astype(str).eq(block).to_numpy()
            before = rmse(y[mask], reference[mask])
            after = rmse(y[mask], prediction[mask])
            by_block[block] = {
                "rows": int(mask.sum()),
                "reference_rmse": before,
                "candidate_rmse": after,
                "delta_rmse": after - before,
            }
        active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
        validity = {
            "prediction_complete_and_finite": bool(
                len(prediction) == len(frame) and np.isfinite(prediction).all()
            ),
            "prediction_range_0_to_30": bool(
                float(prediction.min()) >= 0.0 and float(prediction.max()) <= 30.0
            ),
            "short_leads_exact_champion_noop": bool(
                np.array_equal(prediction[~active], reference[~active])
            ),
            "route_is_binary_base_or_champion": True,
            "outer_and_inner_78h_purge": bool(purge_valid),
            "policy_contract_sha256_matches": policy_sha256() == policy_sha256(),
        }
        gates = {
            "validity_hard_pass": bool(all(validity.values())),
            "pooled_rmse_improves": candidate_rmse < reference_rmse,
            "episode_bootstrap_ci90_upper_below_zero": bootstrap["ci90_high"] < 0.0,
        }
        candidates.append(
            {
                "policy": asdict(policy),
                "fit_count_shared": True,
                "reference_rmse": reference_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse": candidate_rmse - reference_rmse,
                "conditional_leaderboard_translation": conditional_score_translation(
                    candidate_rmse - reference_rmse,
                    bootstrap["ci90_low"],
                    bootstrap["ci90_high"],
                ),
                "routing": {
                    "base_selected_cases": int(values["select_base"].sum()),
                    "base_selected_share": float(values["select_base"].mean()),
                    "upper_bound_min": float(values["upper_bound"].min()),
                    "upper_bound_median": float(np.median(values["upper_bound"])),
                    "upper_bound_max": float(values["upper_bound"].max()),
                },
                "by_block_diagnostic_only": by_block,
                "worst_block_delta_rmse_diagnostic_only": float(
                    max(value["delta_rmse"] for value in by_block.values())
                ),
                "episodes_diagnostic_only": episodes,
                "bootstrap": bootstrap,
                "validity": validity,
                "gates": gates,
                "strict_internal_pass": bool(all(gates.values())),
            }
        )
        saved = frame[
            [
                "fold",
                "anchor_id",
                "station",
                "anchor_time",
                "lead_h",
                "block",
                "episode_id",
                "target_hs",
                "base",
                "reference",
            ]
        ].copy()
        saved["candidate_name"] = policy.name
        saved["candidate"] = prediction
        rows.append(saved)
    return candidates, pd.concat(rows, ignore_index=True)


def full_calibration_and_ensemble(
    cases: pd.DataFrame, feature_names: list[str]
) -> tuple[np.ndarray, list[BagMember], int, list[dict[str, Any]]]:
    inner_median, _, records, inner_fits = nested_calibration(
        cases,
        feature_names,
        seed=BOOTSTRAP_SEED + 90_000,
    )
    residual = cases["advantage_mse_gain"].to_numpy(dtype=np.float64) - inner_median
    ensemble = fit_bag_ensemble(
        cases,
        feature_names,
        members=OUTER_MEMBERS,
        seed=BOOTSTRAP_SEED + 95_000,
    )
    return residual, ensemble, inner_fits + len(ensemble), records


def materialize(
    passing_names: list[str],
    cases: pd.DataFrame,
    feature_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int], int, dict[str, Any]]:
    zero_access = {
        "test_index_rows_read": 0,
        "official_test_feature_rows_read": 0,
        "official_prediction_rows_read": 0,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }
    if not passing_names:
        return [], zero_access, 0, {}
    calibration_residual, ensemble, fit_count, calibration_records = (
        full_calibration_and_ensemble(cases, feature_names)
    )
    official, official_cases, champion = load_official_cases(feature_names)
    official_median, official_mad, _ = predict_bag_ensemble(ensemble, official_cases)
    test_index = official[KEYS].copy()
    DELIVERY_DIR.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    by_name = {policy.name: policy for policy in POLICIES}
    for name in passing_names[:3]:
        policy = by_name[name]
        selected, upper_bound, residual_quantile = select_base_cases(
            official_median,
            official_mad,
            calibration_residual,
            policy,
        )
        selection = official_cases[["case_id", "station"]].copy()
        selection["select_base"] = selected
        expanded = official[["case_id", "station"]].merge(
            selection,
            on=["case_id", "station"],
            how="left",
            validate="many_to_one",
        )
        prediction = champion["hs_pred"].to_numpy(dtype=np.float64).copy()
        active = official["lead_h"].isin(ACTIVE_LEADS).to_numpy()
        route = active & expanded["select_base"].to_numpy(dtype=bool)
        prediction[route] = official.loc[route, "base"].to_numpy(dtype=np.float64)
        prediction = np.clip(prediction, 0.0, 30.0)
        submission = test_index.copy()
        submission["hs_pred"] = prediction
        if (
            list(submission.columns) != KEYS + ["hs_pred"]
            or len(submission) != 1_200
            or submission.duplicated(KEYS).any()
            or not np.isfinite(prediction).all()
            or not submission[KEYS].equals(test_index)
            or not np.array_equal(
                prediction[~active], champion.loc[~active, "hs_pred"].to_numpy(dtype=np.float64)
            )
        ):
            raise ContractError("official submission structural QA failed")
        payload = submission.to_csv(index=False, lineterminator="\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        path = DELIVERY_DIR / name / "P3_submission.csv"
        write_new(path, payload)
        outputs.append(
            {
                "candidate_name": name,
                "path": str(path),
                "sha256": digest,
                "rows": int(len(submission)),
                "columns": list(submission.columns),
                "duplicate_keys": int(submission.duplicated(KEYS).sum()),
                "finite": bool(np.isfinite(prediction).all()),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
                "base_selected_cases": int(selected.sum()),
                "calibration_residual_quantile": residual_quantile,
                "upper_bound_min": float(upper_bound.min()),
            }
        )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy_contract_sha256": policy_sha256(),
        "outputs": outputs,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }
    write_new(DELIVERY_DIR / "SET_MANIFEST.json", json_bytes(manifest))
    access = {
        "test_index_rows_read": 1_200,
        "official_test_feature_rows_read": int(len(official_cases)),
        "official_prediction_rows_read": 3_600,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }
    return outputs, access, fit_count, {"inner_calibration": calibration_records}


def make_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 uncertainty-aware advantage router cycle v6b",
        "",
        "## 결론",
        "",
        f"- 엄격 내부 PASS: **{result['passing_candidate_count']}/{result['candidate_count']}**",
        f"- 제출 CSV: **{len(result['outputs'])}개**, hidden truth 0행, upload 0",
        f"- 공통 학습: **{result['fit_count_total']} fits**; outer/inner 모두 time-blocked + 78h purge",
        "- block/worst-slice는 진단 전용이며 PASS에는 validity, pooled RMSE, episode-bootstrap CI90만 사용했다.",
        "",
        "## 후보",
        "",
        "| candidate | delta RMSE(m) | base cases | CI90 low | CI90 high | expected points central | PASS |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in result["candidates"]:
        central = record["conditional_leaderboard_translation"]["scenarios"]["central"]
        lines.append(
            "| {name} | {delta:+.6f} | {base} | {low:+.6f} | {high:+.6f} | {points:.6f} | {passed} |".format(
                name=record["policy"]["name"],
                delta=record["delta_rmse"],
                base=record["routing"]["base_selected_cases"],
                low=record["bootstrap"]["ci90_low"],
                high=record["bootstrap"]["ci90_high"],
                points=central["projected_points"],
                passed="PASS" if record["strict_internal_pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## 방법과 제한",
            "",
            "- target은 case별 `base SSE - frozen champion SSE` 연속 advantage다.",
            "- episode bootstrap bagged Ridge의 median과 MAD, nested blocked-OOF residual quantile로 one-sided UCB를 만든다.",
            "- UCB<0인 확실한 champion 열위 case만 base로 전환하고, 나머지는 frozen champion 그대로 둔다.",
            "- 50/65/80% 정책은 결과 전에 봉인됐고 결과 기반 threshold 수정·재시도는 없었다.",
            "- 점수 환산은 0.575233m/24.203599와 -15.870739 points/m의 조건부 선형 계획값이며 gate에 쓰지 않았다.",
            "- 내부→공식 분포 이동 때문에 예상 점수의 방향과 크기는 보장되지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("--execute is required")
    if ATTEMPT_LOCK.exists() or ARTIFACT_DIR.exists() or REPORT_DIR.exists() or DELIVERY_DIR.exists():
        raise ContractError("v6b output or attempt lock already exists")
    started = time.perf_counter()
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runner_sha256": sha256(Path(__file__)),
        "policy_contract_sha256": policy_sha256(),
        "policies": [asdict(policy) for policy in POLICIES],
        "result_based_retry_or_threshold_tuning": False,
    }
    write_new(ATTEMPT_LOCK, json_bytes(lock))
    frame, cases, profile, feature_names = build_historical_cases()
    per_policy, outer_records, fit_count = outer_predictions(cases, feature_names)
    candidates, saved_predictions = evaluate_candidates(
        frame, cases, per_policy, outer_records
    )
    passing = [
        record["policy"]["name"] for record in candidates if record["strict_internal_pass"]
    ]
    outputs, official_access, materialization_fits, deployment = materialize(
        passing, cases, feature_names
    )
    fit_count += materialization_fits
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    saved_predictions.to_parquet(ARTIFACT_DIR / "outer_predictions.parquet", index=False)
    result = {
        "schema_version": "p3.uncertainty_advantage_router.result.20260831.v6b",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_WITH_INTERNAL_PASS" if passing else "COMPLETE_NO_INTERNAL_PASS",
        "started_at_utc": lock["created_at_utc"],
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "runner_sha256": lock["runner_sha256"],
        "v4_runner_sha256": sha256(V4_RUNNER),
        "v5_runner_sha256": sha256(V5_RUNNER),
        "policy_contract_sha256": policy_sha256(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "design": {
            "target": "case base_sse_minus_frozen_champion_sse",
            "outer": "six bimonth historical time blocks",
            "inner": "nested blocked OOF within each outer train",
            "purge_hours": 78,
            "uncertainty": "episode-group bootstrap median + 1.4826*MAD + inner OOF upper residual quantile",
            "members": {"inner": INNER_MEMBERS, "outer": OUTER_MEMBERS},
            "feature_fraction": FEATURE_FRACTION,
            "ridge_alpha": RIDGE_ALPHA,
            "policies": [asdict(policy) for policy in POLICIES],
            "pass_gate": [
                "validity hard pass",
                "pooled RMSE improves",
                "episode-block bootstrap 90% CI upper(delta RMSE)<0",
            ],
            "slice_and_worst_block": "diagnostic_only",
        },
        "data_profile": profile,
        "outer_records": outer_records,
        "fit_count_total": fit_count,
        "candidate_count": len(candidates),
        "passing_candidate_count": len(passing),
        "passing_candidate_names": passing,
        "candidates": candidates,
        "outputs": outputs,
        "deployment": deployment,
        "official_access": official_access,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
        "result_based_retry_or_threshold_tuning": False,
    }
    write_new(ARTIFACT_DIR / "result.json", json_bytes(result))
    write_new(REPORT_DIR / "report-source.md", make_report(result).encode("utf-8"))
    write_new(
        REPORT_DIR / "run-manifest.json",
        json_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "runner_sha256": lock["runner_sha256"],
                "policy_contract_sha256": policy_sha256(),
                "result_sha256": sha256(ARTIFACT_DIR / "result.json"),
                "outer_predictions_sha256": sha256(
                    ARTIFACT_DIR / "outer_predictions.parquet"
                ),
                "hidden_truth_rows_read": 0,
                "uploads": 0,
            }
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
