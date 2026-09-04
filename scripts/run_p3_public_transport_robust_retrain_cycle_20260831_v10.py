"""Run a sealed robust base-retrain and safe-blend cycle for P3."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_p3_parallel_candidate_cycle_20260831_v4 import (  # noqa: E402
    BASE_PATH,
    CHAMPION_PATH,
    KEYS,
    P3_DATA,
    load_historical,
    rmse,
)
from run_p3_public_transport_expert_selector_cycle_20260831_v9 import (  # noqa: E402
    FOLD_ORDER,
    bootstrap,
    selection_mask,
)

EXPERIMENT_ID = "p3_public_transport_robust_retrain_cycle_20260831_v10"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
DELIVERY_DIR = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_PUBLIC_TRANSPORT_ROBUST_V10"
)
ATTEMPT_LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
TRAIN_FEATURES = ROOT / "artifacts/p3/features_all20_v1/train_features.parquet"
TEST_FEATURES = ROOT / "artifacts/p3/features_all20_v1/test_features.parquet"
FEATURE_COLUMNS = ROOT / "submissions/p3_frozen_catboost/feature_columns.json"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v1/calibration.json"
OFFICIAL_LEDGER = ROOT / "reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json"

POINTS_PER_RMSE_M = 15.870739046986959
BOOTSTRAP_SEED = 20260833
MIN_CALIBRATED_POINTS = 0.01
MAX_STATION_LEAD_REGRESSION_M = 0.01


class ContractError(RuntimeError):
    """Raised when a v10 invariant changes."""


@dataclass(frozen=True)
class RobustSpec:
    name: str
    family: str
    trust: float
    correction_cap_m: float
    summary: str


SPECS = (
    RobustSpec(
        "P3_1_WINSOR_WEIGHTED_HUBER_BASE_BLEND",
        "huber",
        0.25,
        0.35,
        "Huber direct base residual with train-only winsorized episode weights and champion safe blend.",
    ),
    RobustSpec(
        "P3_2_WINSOR_WEIGHTED_ABSOLUTE_HGB_BASE_BLEND",
        "hist_gbdt",
        0.25,
        0.35,
        "Absolute-loss shallow HGB direct base residual with train-only winsor weights.",
    ),
    RobustSpec(
        "P3_3_WINSOR_WEIGHTED_STRONG_RIDGE_BASE_BLEND",
        "ridge",
        0.15,
        0.25,
        "Strong Ridge direct base residual with minimal champion blend and no validation trimming.",
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


def iter_numeric(payload: Any, prefix: str = "") -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    if isinstance(payload, dict):
        for key, item in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            values.extend(iter_numeric(item, child))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            values.extend(iter_numeric(item, f"{prefix}[{index}]"))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        if re.search(r"delta.*rmse|rmse.*delta", prefix, flags=re.IGNORECASE):
            value = float(payload)
            if np.isfinite(value):
                values.append((prefix, value))
    return values


def audit_prior_results() -> dict[str, Any]:
    paths = sorted(
        {
            *ROOT.glob("artifacts/**/*result*.json"),
            *ROOT.glob("artifacts/**/*metrics*.json"),
            *ROOT.glob("reports/**/*result*.json"),
            *ROOT.glob("reports/**/*metrics*.json"),
        }
    )
    records: list[dict[str, Any]] = []
    files_read = 0
    excluded: list[str] = []
    for path in paths:
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        lower = relative.lower()
        if "p3" not in lower:
            continue
        if any(token in lower for token in ("official", "hidden", "submission")):
            excluded.append(relative)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        files_read += 1
        for key, value in iter_numeric(payload):
            records.append({"path": relative, "key": key, "delta_rmse_m": value})
    ordered = sorted(records, key=lambda item: item["delta_rmse_m"])
    return {
        "files_read": files_read,
        "numeric_delta_rmse_claims": len(records),
        "excluded_official_hidden_submission_paths": excluded,
        "largest_internal_improvements": ordered[:25],
        "largest_internal_regressions": list(reversed(ordered[-25:])),
    }


def load_features() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    frame, profile = load_historical()
    columns = json.loads(FEATURE_COLUMNS.read_text(encoding="utf-8"))
    if not isinstance(columns, list) or len(columns) != 591:
        raise ContractError("591 feature contract changed")
    features = pd.read_parquet(TRAIN_FEATURES, columns=["anchor_id", "station", *columns])
    frame = frame.merge(features, on=["anchor_id", "station"], validate="many_to_one")
    frame = frame.loc[selection_mask(frame)].copy()
    frame["fold"] = pd.Categorical(frame["fold"], categories=FOLD_ORDER, ordered=True)
    frame = frame.sort_values(["fold", "anchor_id", "lead_h"]).reset_index(drop=True)
    if len(frame) != 810 or frame["anchor_id"].nunique() != 135:
        raise ContractError("selection-matched robust surface changed")
    return frame, columns, profile


def design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    lead = frame["lead_h"].to_numpy(dtype=np.float64) / 24.0
    station = frame["station"].astype(str)
    extra = np.column_stack(
        [
            lead,
            station.eq("G-ORS").to_numpy(dtype=np.float64),
            station.eq("I-ORS").to_numpy(dtype=np.float64),
            station.eq("S-ORS").to_numpy(dtype=np.float64),
            frame["base"].to_numpy(dtype=np.float64)
            - frame["current_hs"].to_numpy(dtype=np.float64),
        ]
    )
    return np.column_stack([frame[columns].to_numpy(dtype=np.float64), extra])


def train_only_weights(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    residual = np.abs(
        frame["target_hs"].to_numpy(dtype=np.float64)
        - frame["reference"].to_numpy(dtype=np.float64)
    )
    cap = float(np.quantile(residual, 0.90))
    weight = np.ones(len(frame), dtype=np.float64)
    extreme = residual > cap
    weight[extreme] = np.maximum(0.2, cap / np.maximum(residual[extreme], 1e-12))
    return weight, {
        "train_only_abs_reference_residual_q90_m": cap,
        "downweighted_rows": int(extreme.sum()),
        "validation_rows_deleted": 0,
        "minimum_weight": float(weight.min()),
    }


def build_model(spec: RobustSpec, seed: int) -> Any:
    if spec.family == "huber":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=700, tol=1e-5),
        )
    if spec.family == "hist_gbdt":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.04,
                max_iter=120,
                max_leaf_nodes=7,
                min_samples_leaf=20,
                l2_regularization=20.0,
                random_state=seed,
            ),
        )
    if spec.family == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=500.0),
        )
    raise ContractError(f"unknown robust family {spec.family}")


def fit_model(model: Any, spec: RobustSpec, x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> Any:
    parameter = {
        "huber": "huberregressor__sample_weight",
        "hist_gbdt": "histgradientboostingregressor__sample_weight",
        "ridge": "ridge__sample_weight",
    }[spec.family]
    model.fit(x, y, **{parameter: weight})
    return model


def predict(
    model: Any,
    spec: RobustSpec,
    frame: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    raw = frame["base"].to_numpy(dtype=np.float64) + np.asarray(
        model.predict(design(frame, columns)), dtype=np.float64
    )
    reference = frame["reference"].to_numpy(dtype=np.float64)
    correction = np.clip(
        spec.trust * (raw - reference), -spec.correction_cap_m, spec.correction_cap_m
    )
    correction[~selection_mask(frame)] = 0.0
    return np.clip(reference + correction, 0.0, 30.0)


def evaluate(
    frame: pd.DataFrame,
    columns: list[str],
    spec: RobustSpec,
    penalty: float,
) -> tuple[dict[str, Any], Any | None]:
    prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
    receipts: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(FOLD_ORDER):
        valid = frame["fold"].astype(str).eq(fold).to_numpy()
        train = frame["fold"].astype(str).isin(FOLD_ORDER[:fold_number]).to_numpy()
        if fold_number == 0:
            receipts.append({"fold": fold, "action": "exact_champion_no_op", "fits": 0})
            continue
        weight, weight_receipt = train_only_weights(frame.loc[train])
        model = build_model(spec, BOOTSTRAP_SEED + fold_number)
        fit_model(
            model,
            spec,
            design(frame.loc[train], columns),
            (
                frame.loc[train, "target_hs"].to_numpy(dtype=np.float64)
                - frame.loc[train, "base"].to_numpy(dtype=np.float64)
            ),
            weight,
        )
        prediction[valid] = predict(model, spec, frame.loc[valid], columns)
        receipts.append(
            {
                "fold": fold,
                "action": "prior_fold_robust_base_fit",
                "fits": 1,
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                **weight_receipt,
            }
        )
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    before = rmse(truth, reference)
    after = rmse(truth, prediction)
    delta = after - before
    episode = bootstrap(frame, prediction, unit_columns=("episode_id",), seed=BOOTSTRAP_SEED)
    group = bootstrap(
        frame,
        prediction,
        unit_columns=("fold", "station"),
        seed=BOOTSTRAP_SEED + 1,
    )
    station_lead: dict[str, Any] = {}
    work = frame.assign(candidate=prediction)
    for (station, lead), part in work.groupby(["station", "lead_h"], observed=True, sort=True):
        station_lead[f"{station}|{int(lead)}"] = {
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy()),
            "rows": int(len(part)),
        }
    by_fold = {
        str(key): {
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["reference"].to_numpy()),
            "rows": int(len(part)),
        }
        for key, part in work.groupby("fold", observed=True, sort=True)
    }
    worst = max(item["delta_rmse_m"] for item in station_lead.values())
    raw_conservative = max(0.0, -float(episode["ci90_m"][1]) * POINTS_PER_RMSE_M)
    calibrated = raw_conservative - penalty
    checks = {
        "pooled_rmse_improves": delta < 0.0,
        "episode_ci90_upper_below_zero": float(episode["ci90_m"][1]) < 0.0,
        "group_ci90_upper_below_zero": float(group["ci90_m"][1]) < 0.0,
        "worst_station_lead_within_0p01m": worst <= MAX_STATION_LEAD_REGRESSION_M,
        "raw_conservative_points_at_least_0p331905690": raw_conservative
        >= penalty + MIN_CALIBRATED_POINTS,
        "calibrated_conservative_points_at_least_0p01": calibrated >= MIN_CALIBRATED_POINTS,
        "validation_rows_deleted_zero": all(
            item.get("validation_rows_deleted", 0) == 0 for item in receipts
        ),
        "finite_predictions": bool(np.isfinite(prediction).all()),
    }
    passed = all(checks.values())
    full_model = None
    full_weight_receipt = None
    if passed:
        weight, full_weight_receipt = train_only_weights(frame)
        full_model = fit_model(
            build_model(spec, BOOTSTRAP_SEED + 99),
            spec,
            design(frame, columns),
            truth - frame["base"].to_numpy(dtype=np.float64),
            weight,
        )
    return {
        "spec": asdict(spec),
        "fit_receipts": receipts,
        "historical_fit_count": int(sum(item["fits"] for item in receipts)),
        "full_fit_weight_receipt": full_weight_receipt,
        "reference_rmse_m": before,
        "candidate_rmse_m": after,
        "delta_candidate_minus_reference_rmse_m": delta,
        "by_fold": by_fold,
        "station_lead": station_lead,
        "worst_station_lead_delta_rmse_m": worst,
        "episode_bootstrap": episode,
        "group_bootstrap": group,
        "expected_points": {
            "raw_central": -delta * POINTS_PER_RMSE_M,
            "raw_conservative": raw_conservative,
            "public_reversal_penalty": penalty,
            "calibrated_conservative": calibrated,
        },
        "gate_checks": checks,
        "passed": passed,
    }, full_model


def load_penalty() -> float:
    value = float(json.loads(CALIBRATION.read_text(encoding="utf-8"))["gates"]["P3"]["transport_penalty_points"])
    if abs(value - 0.3219056897594759) > 1e-12:
        raise ContractError("transport penalty changed")
    return value


def materialize(
    passing: list[tuple[dict[str, Any], Any]], columns: list[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not passing:
        return [], {
            "official_test_index_rows_read": 0,
            "official_case_feature_rows_read": 0,
            "official_prediction_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "uploads": 0,
        }
    index = pd.read_csv(P3_DATA / "test_index.csv", dtype={"case_id": "string", "station": "string"})
    base = pd.read_csv(BASE_PATH, dtype={"case_id": "string", "station": "string"})
    champion = pd.read_csv(CHAMPION_PATH, dtype={"case_id": "string", "station": "string"})
    features = pd.read_parquet(TEST_FEATURES, columns=["case_id", "station", *columns])
    official = index.merge(features, on=["case_id", "station"], validate="many_to_one")
    official["base"] = base["hs_pred"].to_numpy(dtype=np.float64)
    official["reference"] = champion["hs_pred"].to_numpy(dtype=np.float64)
    official["current_hs"] = official["hs_current"]
    official["delta"] = 0.0
    DELIVERY_DIR.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record, model in passing[:3]:
        prediction = predict(model, RobustSpec(**record["spec"]), official, columns)
        submission = official[KEYS].copy()
        submission["hs_pred"] = prediction
        payload = submission.to_csv(index=False, lineterminator="\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        if len(submission) != 1200 or submission.duplicated(KEYS).any() or digest in seen:
            raise ContractError("official output contract failed")
        seen.add(digest)
        directory = DELIVERY_DIR / record["spec"]["name"]
        path = directory / "P3_submission.csv"
        write_new(path, payload)
        output = {
            "candidate": record["spec"]["name"],
            "path": str(path),
            "rows": 1200,
            "sha256": digest,
            "changed_rows_vs_champion": int(
                np.sum(np.abs(prediction - champion["hs_pred"].to_numpy(dtype=float)) > 1e-12)
            ),
            "minimum_m": float(prediction.min()),
            "maximum_m": float(prediction.max()),
        }
        outputs.append(output)
        write_new(directory / "submission-info.json", canonical_bytes(output))
    write_new(DELIVERY_DIR / "SET_MANIFEST.json", canonical_bytes({"outputs": outputs, "uploads": 0}))
    return outputs, {
        "official_test_index_rows_read": 1200,
        "official_case_feature_rows_read": 200,
        "official_prediction_rows_read": 2400,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }


def audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Prior P3 result and metric JSON audit",
        "",
        f"- files read: {audit['files_read']}",
        f"- numeric delta-RMSE claims: {audit['numeric_delta_rmse_claims']}",
        f"- excluded official/hidden/submission paths: {len(audit['excluded_official_hidden_submission_paths'])}",
        "",
        "## Largest reported internal improvements",
        "",
        "| delta RMSE(m) | path | key |",
        "|---:|---|---|",
    ]
    for item in audit["largest_internal_improvements"]:
        lines.append(f"| {item['delta_rmse_m']:.9f} | `{item['path']}` | `{item['key']}` |")
    lines.extend(["", "## Largest reported regressions", "", "| delta RMSE(m) | path | key |", "|---:|---|---|"])
    for item in audit["largest_internal_regressions"]:
        lines.append(f"| {item['delta_rmse_m']:.9f} | `{item['path']}` | `{item['key']}` |")
    return "\n".join(lines) + "\n"


def make_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 Public-transport robust retrain v10",
        "",
        "## 결론",
        "",
        f"- calibrated PASS: **{result['passing_candidate_count']}/{result['candidate_count']}**",
        f"- CSV: **{len(result['outputs'])}개**, upload 0",
        "- training outliers were downweighted from train-only residuals; no validation row was deleted.",
        "",
        "| candidate | delta RMSE | episode CI90 upper | group CI90 upper | worst station-lead | calibrated pts | PASS |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in result["candidates"]:
        lines.append(
            "| {name} | {delta:.6f} | {episode:.6f} | {group:.6f} | {worst:.6f} | {points:.6f} | {passed} |".format(
                name=item["spec"]["name"],
                delta=item["delta_candidate_minus_reference_rmse_m"],
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
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "specs": [asdict(item) for item in SPECS]}))
        return 0
    for path in (TRAIN_FEATURES, FEATURE_COLUMNS, CALIBRATION, OFFICIAL_LEDGER):
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
                "maximum_fits": 9,
                "specs": [asdict(item) for item in SPECS],
            }
        ),
    )
    audit = audit_prior_results()
    write_new(REPORT_DIR / "prior-results-audit.json", canonical_bytes(audit))
    write_new(REPORT_DIR / "prior-results-audit.md", audit_markdown(audit).encode())
    frame, columns, profile = load_features()
    penalty = load_penalty()
    candidates: list[dict[str, Any]] = []
    passing: list[tuple[dict[str, Any], Any]] = []
    for spec in SPECS:
        record, model = evaluate(frame, columns, spec, penalty)
        candidates.append(record)
        if record["passed"] and model is not None:
            passing.append((record, model))
        print(
            json.dumps(
                {
                    "candidate": spec.name,
                    "delta_rmse_m": record["delta_candidate_minus_reference_rmse_m"],
                    "calibrated_points": record["expected_points"]["calibrated_conservative"],
                    "passed": record["passed"],
                }
            ),
            flush=True,
        )
    outputs, access = materialize(passing, columns)
    result = {
        "schema_version": "p3.public_transport_robust_retrain.result.v10",
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
        "data_profile": {
            **profile,
            "selection_matched_rows": int(len(frame)),
            "selection_matched_cases": int(frame["anchor_id"].nunique()),
            "validation_rows_deleted": 0,
        },
        "prior_results_audit": {
            "files_read": audit["files_read"],
            "numeric_delta_rmse_claims": audit["numeric_delta_rmse_claims"],
            "json_sha256": sha256(REPORT_DIR / "prior-results-audit.json"),
            "markdown_sha256": sha256(REPORT_DIR / "prior-results-audit.md"),
        },
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
            "feature_columns_sha256": sha256(FEATURE_COLUMNS),
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
