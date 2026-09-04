"""Run a sealed past-only physical-expert selector for P3.

Unlike v8, this runner never regresses a new wave-height residual.  It selects
among a frozen bank of physically interpretable KMA-axis experts using only
completed prior-fold labels.  Unsupported or weak-benefit regimes are exact
champion no-ops.  Official inputs are opened only after every gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_p3_parallel_candidate_cycle_20260831_v4 import (  # noqa: E402
    BASE_PATH,
    CHAMPION_PATH,
    KEYS,
    P3_DATA,
    SOURCE_PATH,
    load_historical,
    rmse,
)

EXPERIMENT_ID = "p3_public_transport_expert_selector_cycle_20260831_v9"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
DELIVERY_DIR = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_PUBLIC_TRANSPORT_SELECTOR_V9"
)
ATTEMPT_LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
TRAIN_FEATURES = ROOT / "artifacts/p3/features_all20_v1/train_features.parquet"
TEST_FEATURES = ROOT / "artifacts/p3/features_all20_v1/test_features.parquet"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v1/calibration.json"
OFFICIAL_LEDGER = ROOT / "reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json"

FOLD_ORDER = ("2024_h2_storm", "winter_transition", "2025_h1")
ACTIVE_LEADS = (18, 24)
REFERENCE_ALPHA = 0.425
POINTS_PER_RMSE_M = 15.870739046986959
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260832
MIN_CALIBRATED_POINTS = 0.01
MAX_STATION_LEAD_REGRESSION_M = 0.01


class ContractError(RuntimeError):
    """Raised when the sealed v9 contract changes."""


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    regime_columns: tuple[str, ...]
    alpha_bank: tuple[float, ...]
    minimum_support: int
    minimum_prior_rmse_benefit_m: float
    summary: str


SPECS = (
    SelectorSpec(
        "P3_1_FINE_REGIME_PHYSICAL_EXPERT_SELECTOR",
        ("station", "lead_h", "current_bin", "rise_bin"),
        (0.0, 0.2, 0.425, 0.65, 1.0),
        4,
        0.03,
        "Fine station/lead/current/rise selector with exact no-op outside strong prior support.",
    ),
    SelectorSpec(
        "P3_2_COARSE_REGIME_PHYSICAL_EXPERT_SELECTOR",
        ("station", "lead_h", "rise_bin"),
        (0.0, 0.2, 0.425, 0.65, 1.0),
        6,
        0.025,
        "Coarser station/lead/rise selector to reduce sparse-regime variance.",
    ),
    SelectorSpec(
        "P3_3_ADD_ONLY_PHYSICAL_EXPERT_SELECTOR",
        ("station", "lead_h", "current_bin"),
        (0.425, 0.55, 0.65, 1.0),
        6,
        0.02,
        "Add-only KMA intervention; never reduces the official champion alpha.",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def attach_regime(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["current_bin"] = pd.cut(
        output["hs_current"], [1.5, 1.75, 2.0, 2.2], right=False, labels=False
    ).fillna(-1).astype(int)
    output["rise_bin"] = pd.cut(
        output["hs_delta_12h"], [0.2, 0.4, 0.7, np.inf], right=False, labels=False
    ).fillna(-1).astype(int)
    return output


def selection_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["hs_current"].between(1.5, 2.2, inclusive="left").to_numpy()
        & frame["hs_delta_12h"].gt(0.2).to_numpy()
    )


def expert_prediction(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
    active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    prediction[active] = (
        frame.loc[active, "base"].to_numpy(dtype=np.float64)
        + float(alpha) * frame.loc[active, "delta"].to_numpy(dtype=np.float64)
    )
    return np.clip(prediction, 0.0, 30.0)


def fit_selector(train: pd.DataFrame, spec: SelectorSpec) -> tuple[dict[tuple[Any, ...], float], dict[str, Any]]:
    active = train[train["lead_h"].isin(ACTIVE_LEADS)].copy()
    table: dict[tuple[Any, ...], float] = {}
    receipts: list[dict[str, Any]] = []
    for key, group in active.groupby(list(spec.regime_columns), observed=True, sort=True):
        normalized_key = key if isinstance(key, tuple) else (key,)
        if len(group) < spec.minimum_support:
            continue
        truth = group["target_hs"].to_numpy(dtype=np.float64)
        reference = group["reference"].to_numpy(dtype=np.float64)
        reference_rmse = rmse(truth, reference)
        scores = {
            float(alpha): rmse(truth, expert_prediction(group, alpha))
            for alpha in spec.alpha_bank
        }
        best_alpha = min(scores, key=lambda value: (scores[value], abs(value - REFERENCE_ALPHA)))
        benefit = reference_rmse - scores[best_alpha]
        if benefit >= spec.minimum_prior_rmse_benefit_m and best_alpha != REFERENCE_ALPHA:
            table[normalized_key] = float(best_alpha)
            receipts.append(
                {
                    "regime": [str(value) for value in normalized_key],
                    "rows": int(len(group)),
                    "selected_alpha": float(best_alpha),
                    "prior_rmse_benefit_m": float(benefit),
                }
            )
    return table, {
        "fit_count": 1,
        "train_rows": int(len(train)),
        "eligible_regimes": int(len(receipts)),
        "selected_regimes": receipts,
    }


def apply_selector(
    frame: pd.DataFrame,
    spec: SelectorSpec,
    table: dict[tuple[Any, ...], float],
) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.full(len(frame), REFERENCE_ALPHA, dtype=np.float64)
    active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy() & selection_mask(frame)
    for position in np.flatnonzero(active):
        row = frame.iloc[position]
        key = tuple(row[column] for column in spec.regime_columns)
        alpha[position] = table.get(key, REFERENCE_ALPHA)
    prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
    changed = active & (np.abs(alpha - REFERENCE_ALPHA) > 1e-12)
    prediction[changed] = (
        frame.loc[changed, "base"].to_numpy(dtype=np.float64)
        + alpha[changed] * frame.loc[changed, "delta"].to_numpy(dtype=np.float64)
    )
    return np.clip(prediction, 0.0, 30.0), alpha


def load_selection_matched_history() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, profile = load_historical()
    features = pd.read_parquet(
        TRAIN_FEATURES,
        columns=["anchor_id", "station", "hs_current", "hs_delta_12h"],
    )
    frame = frame.merge(features, on=["anchor_id", "station"], validate="many_to_one")
    frame = attach_regime(frame.loc[selection_mask(frame)].copy())
    frame["fold"] = pd.Categorical(frame["fold"], categories=FOLD_ORDER, ordered=True)
    frame = frame.sort_values(["fold", "anchor_id", "lead_h"]).reset_index(drop=True)
    if len(frame) != 810 or frame["anchor_id"].nunique() != 135:
        raise ContractError("selection-matched OOF bank contract changed")
    return frame, {
        **profile,
        "selection_matched_rows": int(len(frame)),
        "selection_matched_cases": int(frame["anchor_id"].nunique()),
        "by_fold_cases": {
            str(key): int(part["anchor_id"].nunique())
            for key, part in frame.groupby("fold", observed=True, sort=True)
        },
    }


def bootstrap(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    unit_columns: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    work = frame.assign(candidate=prediction).reset_index(drop=True)
    grouped = list(work.groupby(list(unit_columns), observed=True, sort=True))
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    ref_sse = np.asarray(
        [np.square(part["reference"] - part["target_hs"]).sum() for _, part in grouped]
    )
    cand_sse = np.asarray(
        [np.square(part["candidate"] - part["target_hs"]).sum() for _, part in grouped]
    )
    counts = np.asarray([len(part) for _, part in grouped], dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(grouped), size=len(grouped))
        denominator = float(counts[draw].sum())
        values[index] = np.sqrt(cand_sse[draw].sum() / denominator) - np.sqrt(
            ref_sse[draw].sum() / denominator
        )
    return {
        "unit": "|".join(unit_columns),
        "units": len(grouped),
        "replicates": BOOTSTRAP_REPLICATES,
        "ci90_m": [float(value) for value in np.quantile(values, [0.05, 0.95])],
        "median_m": float(np.median(values)),
        "probability_improve": float(np.mean(values < 0.0)),
    }


def evaluate(
    frame: pd.DataFrame, spec: SelectorSpec, transport_penalty: float
) -> tuple[dict[str, Any], dict[tuple[Any, ...], float] | None]:
    prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
    alpha = np.full(len(frame), REFERENCE_ALPHA, dtype=np.float64)
    fit_receipts: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(FOLD_ORDER):
        valid = frame["fold"].astype(str).eq(fold).to_numpy()
        train = frame["fold"].astype(str).isin(FOLD_ORDER[:fold_number]).to_numpy()
        if fold_number == 0:
            fit_receipts.append({"fold": fold, "action": "exact_champion_no_op", "fits": 0})
            continue
        table, receipt = fit_selector(frame.loc[train], spec)
        fold_prediction, fold_alpha = apply_selector(frame.loc[valid], spec, table)
        prediction[valid] = fold_prediction
        alpha[valid] = fold_alpha
        fit_receipts.append({"fold": fold, "action": "prior_completed_folds_only", "fits": 1, **receipt})
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    before = rmse(truth, reference)
    after = rmse(truth, prediction)
    delta = after - before
    by_fold: dict[str, Any] = {}
    for key, part in frame.assign(candidate=prediction).groupby("fold", observed=True, sort=True):
        by_fold[str(key)] = {
            "rows": int(len(part)),
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy()),
        }
    station_lead: dict[str, Any] = {}
    for (station, lead), part in frame.assign(candidate=prediction).groupby(
        ["station", "lead_h"], observed=True, sort=True
    ):
        station_lead[f"{station}|{int(lead)}"] = {
            "rows": int(len(part)),
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy()),
        }
    episode = bootstrap(frame, prediction, unit_columns=("episode_id",), seed=BOOTSTRAP_SEED)
    group = bootstrap(
        frame,
        prediction,
        unit_columns=("fold", "station"),
        seed=BOOTSTRAP_SEED + 1,
    )
    raw_conservative = max(0.0, -float(episode["ci90_m"][1]) * POINTS_PER_RMSE_M)
    calibrated = raw_conservative - transport_penalty
    worst = max(item["delta_rmse_m"] for item in station_lead.values())
    checks = {
        "pooled_rmse_improves": delta < 0.0,
        "episode_ci90_upper_below_zero": float(episode["ci90_m"][1]) < 0.0,
        "group_ci90_upper_below_zero": float(group["ci90_m"][1]) < 0.0,
        "worst_station_lead_within_0p01m": worst <= MAX_STATION_LEAD_REGRESSION_M,
        "raw_conservative_points_at_least_0p331905690": raw_conservative
        >= transport_penalty + MIN_CALIBRATED_POINTS,
        "calibrated_conservative_points_at_least_0p01": calibrated >= MIN_CALIBRATED_POINTS,
        "finite_predictions": bool(np.isfinite(prediction).all()),
    }
    passed = all(checks.values())
    full_table = None
    if passed:
        full_table, _ = fit_selector(frame, spec)
    return {
        "spec": asdict(spec),
        "fit_receipts": fit_receipts,
        "historical_fit_count": int(sum(item["fits"] for item in fit_receipts)),
        "reference_rmse_m": before,
        "candidate_rmse_m": after,
        "delta_candidate_minus_reference_rmse_m": delta,
        "changed_rows": int(np.sum(np.abs(alpha - REFERENCE_ALPHA) > 1e-12)),
        "alpha_distribution": {
            "minimum": float(alpha.min()),
            "median": float(np.median(alpha)),
            "maximum": float(alpha.max()),
        },
        "by_fold": by_fold,
        "station_lead": station_lead,
        "worst_station_lead_delta_rmse_m": worst,
        "episode_bootstrap": episode,
        "group_bootstrap": group,
        "expected_points": {
            "raw_central": -delta * POINTS_PER_RMSE_M,
            "raw_conservative": raw_conservative,
            "public_reversal_penalty": transport_penalty,
            "calibrated_conservative": calibrated,
        },
        "gate_checks": checks,
        "passed": passed,
    }, full_table


def load_penalty() -> float:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    penalty = float(calibration["gates"]["P3"]["transport_penalty_points"])
    if abs(penalty - 0.3219056897594759) > 1e-12:
        raise ContractError("Public transport penalty changed")
    return penalty


def materialize(
    passing: list[tuple[dict[str, Any], dict[tuple[Any, ...], float]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not passing:
        return [], {
            "official_test_index_rows_read": 0,
            "official_case_feature_rows_read": 0,
            "official_component_prediction_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "uploads": 0,
        }
    index = pd.read_csv(P3_DATA / "test_index.csv", dtype={"case_id": "string", "station": "string"})
    base = pd.read_csv(BASE_PATH, dtype={"case_id": "string", "station": "string"})
    source = pd.read_csv(SOURCE_PATH, dtype={"case_id": "string", "station": "string"})
    champion = pd.read_csv(CHAMPION_PATH, dtype={"case_id": "string", "station": "string"})
    if not index[KEYS].equals(base[KEYS]) or not index[KEYS].equals(source[KEYS]) or not index[KEYS].equals(champion[KEYS]):
        raise ContractError("official key order changed")
    features = pd.read_parquet(
        TEST_FEATURES, columns=["case_id", "station", "hs_current", "hs_delta_12h"]
    )
    official = index.merge(features, on=["case_id", "station"], validate="many_to_one")
    official["base"] = base["hs_pred"].to_numpy(dtype=np.float64)
    official["delta"] = source["hs_pred"].to_numpy(dtype=np.float64) - official["base"]
    official["reference"] = champion["hs_pred"].to_numpy(dtype=np.float64)
    official = attach_regime(official)
    DELIVERY_DIR.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record, table in passing[:3]:
        prediction, alpha = apply_selector(official, SelectorSpec(**record["spec"]), table)
        submission = official[KEYS].copy()
        submission["hs_pred"] = prediction
        payload = submission.to_csv(index=False, lineterminator="\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        if len(submission) != 1200 or submission.duplicated(KEYS).any() or digest in seen:
            raise ContractError("submission structure or uniqueness failed")
        seen.add(digest)
        directory = DELIVERY_DIR / record["spec"]["name"]
        path = directory / "P3_submission.csv"
        write_new(path, payload)
        output = {
            "candidate": record["spec"]["name"],
            "path": str(path),
            "rows": 1200,
            "sha256": digest,
            "changed_rows_vs_champion": int(np.sum(np.abs(alpha - REFERENCE_ALPHA) > 1e-12)),
            "minimum_m": float(prediction.min()),
            "maximum_m": float(prediction.max()),
        }
        outputs.append(output)
        write_new(directory / "submission-info.json", canonical_bytes(output))
    write_new(
        DELIVERY_DIR / "SET_MANIFEST.json",
        canonical_bytes({"experiment_id": EXPERIMENT_ID, "outputs": outputs, "uploads": 0}),
    )
    return outputs, {
        "official_test_index_rows_read": 1200,
        "official_case_feature_rows_read": 200,
        "official_component_prediction_rows_read": 3600,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }


def make_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 Public-transport physical expert selector v9",
        "",
        "## 결론",
        "",
        f"- calibrated PASS: **{result['passing_candidate_count']}/{result['candidate_count']}**",
        f"- CSV: **{len(result['outputs'])}개**, upload 0",
        "- v8 residual regression을 폐기하고, prior-only regime expert selection과 exact no-op fallback만 평가했다.",
        "",
        "| candidate | delta RMSE | changed rows | episode CI90 upper | group CI90 upper | worst station-lead | calibrated pts | PASS |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result["candidates"]:
        lines.append(
            "| {name} | {delta:.6f} | {changed} | {episode:.6f} | {group:.6f} | {worst:.6f} | {points:.6f} | {passed} |".format(
                name=item["spec"]["name"],
                delta=item["delta_candidate_minus_reference_rmse_m"],
                changed=item["changed_rows"],
                episode=item["episode_bootstrap"]["ci90_m"][1],
                group=item["group_bootstrap"]["ci90_m"][1],
                worst=item["worst_station_lead_delta_rmse_m"],
                points=item["expected_points"]["calibrated_conservative"],
                passed=item["passed"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "specs": [asdict(x) for x in SPECS]}))
        return 0
    for path in (TRAIN_FEATURES, CALIBRATION, OFFICIAL_LEDGER):
        if not path.exists():
            raise ContractError(f"dependency missing: {path}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runner_hash = sha256(Path(__file__))
    write_new(
        ATTEMPT_LOCK,
        canonical_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "status": "ATTEMPT_CONSUMED_ONE_SHOT",
                "runner_sha256": runner_hash,
                "specs": [asdict(item) for item in SPECS],
                "maximum_fits": 9,
            }
        ),
    )
    penalty = load_penalty()
    frame, profile = load_selection_matched_history()
    candidates: list[dict[str, Any]] = []
    passing: list[tuple[dict[str, Any], dict[tuple[Any, ...], float]]] = []
    for spec in SPECS:
        record, table = evaluate(frame, spec, penalty)
        candidates.append(record)
        if record["passed"] and table is not None:
            passing.append((record, table))
        print(
            json.dumps(
                {
                    "candidate": spec.name,
                    "delta_rmse_m": record["delta_candidate_minus_reference_rmse_m"],
                    "changed_rows": record["changed_rows"],
                    "calibrated_points": record["expected_points"]["calibrated_conservative"],
                    "passed": record["passed"],
                }
            ),
            flush=True,
        )
    outputs, access = materialize(passing)
    result = {
        "schema_version": "p3.public_transport_expert_selector.result.v9",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_MATERIALIZED_NOT_UPLOADED" if passing else "NO_GO_PUBLIC_TRANSPORT_GATE",
        "candidate_count": len(candidates),
        "passing_candidate_count": len(passing),
        "candidates": candidates,
        "fit_budget": {
            "maximum": 9,
            "actual_historical": int(sum(item["historical_fit_count"] for item in candidates)),
            "actual_full": len(passing),
            "actual_total": int(sum(item["historical_fit_count"] for item in candidates) + len(passing)),
        },
        "data_profile": profile,
        "transport": {
            "penalty_points": penalty,
            "minimum_raw_points": penalty + MIN_CALIBRATED_POINTS,
            "minimum_calibrated_points": MIN_CALIBRATED_POINTS,
        },
        "outputs": outputs,
        "data_access": access,
        "provenance": {
            "runner_sha256": runner_hash,
            "train_features_sha256": sha256(TRAIN_FEATURES),
            "calibration_sha256": sha256(CALIBRATION),
            "official_ledger_sha256": sha256(OFFICIAL_LEDGER),
        },
        "execution": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "result_based_tuning_or_retry": False,
            "hidden_truth_rows_read": 0,
            "upload_attempt_count": 0,
        },
    }
    write_new(ARTIFACT_DIR / "result.json", canonical_bytes(result))
    write_new(REPORT_DIR / "report-source.md", make_report(result).encode())
    write_new(
        REPORT_DIR / "run-manifest.json",
        canonical_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "runner_sha256": runner_hash,
                "result_sha256": sha256(ARTIFACT_DIR / "result.json"),
                "report_sha256": sha256(REPORT_DIR / "report-source.md"),
                "outputs": outputs,
            }
        ),
    )
    print(json.dumps({"decision": result["decision"], "passing": len(passing), "outputs": len(outputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
