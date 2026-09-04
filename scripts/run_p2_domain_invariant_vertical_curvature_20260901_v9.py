"""Exactly-once exploratory P2 domain-invariant vertical-curvature cycle.

The only data inputs are ``observations.csv`` and a truth-free historical
prediction frame.  Official query/sample/baseline/score/submission files and
hidden truth have no code path in this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p2_group_balanced_raw_residual_20260901_v8 as prior_engine  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)

EXPERIMENT_ID = "p2_domain_invariant_vertical_curvature_20260901_v9"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
TARGET_LAYERS = (2, 3, 4)
FOLD_ORDER = ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")
SCHEMA_VERSION = "p2.domain_invariant_vertical_curvature.result.20260901.v9"


class ContractError(RuntimeError):
    """Raised when the sealed experiment contract is violated."""


@dataclass(frozen=True)
class CandidateSpec:
    """A jointly sealed candidate; both candidates always execute."""

    name: str
    normalization: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_observations(config: dict[str, Any]) -> Path:
    raw = os.environ.get(config["source_contract"]["environment_variable"])
    if not raw:
        raise ContractError("P2_DATA_DIR is required")
    path = Path(raw).resolve() / config["source_contract"]["only_source_filename"]
    if path.name != "observations.csv" or not path.is_file():
        raise ContractError("only P2_DATA_DIR/observations.csv may be opened")
    if sha256_file(path) != config["source_contract"]["observations_sha256"]:
        raise ContractError("observations.csv hash drift")
    return path


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ContractError("experiment id drift")
    if config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED":
        raise ContractError("experiment is not preregistered")
    if config["postprocess"] != "NONE" or config["result_adaptive_tuning"]:
        raise ContractError("postprocess or tuning contract drift")
    if len(config["candidates"]) != 2:
        raise ContractError("exactly two jointly sealed candidates are required")
    if config["training"]["row_deletion"]:
        raise ContractError("row deletion is forbidden")
    if int(config["operation_limits"]["maximum_fit_count"]) != 6:
        raise ContractError("fit budget drift")
    return config


def preflight() -> dict[str, Any]:
    config = load_config()
    names = [item["name"] for item in config["candidates"]]
    if names != [
        "P2_V9_GLOBAL_ROBUST_NCR_RIDGE_BLEND020",
        "P2_V9_LAYER_MONTH_ANOMALY_NCR_RIDGE_BLEND020",
    ]:
        raise ContractError("candidate order drift")
    canonical = prior_engine.canonical_time_ns(
        pd.DatetimeIndex(["2024-09-01T00:00:00Z"]).as_unit("us")
    )
    expected = prior_engine.canonical_time_ns(
        pd.DatetimeIndex(["2024-09-01T00:00:00Z"]).as_unit("ns")
    )
    if canonical.tolist() != expected.tolist():
        raise ContractError("datetime-unit canonicalization failed")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "candidate_names": names,
        "candidate_count": 2,
        "maximum_fit_count": 6,
        "config_sha256": sha256_file(CONFIG),
        "runner_sha256": sha256_file(RUNNER),
        "normalized_curvature_module_sha256": sha256_file(
            ROOT / "src" / "p2_restore" / "normalized_curvature_residual.py"
        ),
        "datetime_unit_contract": "PASS",
        "data_rows_read": 0,
        "model_fits": 0,
        "artifacts_written": 0,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    payload["preflight_sha256"] = sha256_json(payload)
    return payload


def robust_winsorize(
    values: np.ndarray, multiplier: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed, no-deletion robust handling for the dimensionless target."""

    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ContractError("target contains non-finite values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = max(1.4826 * mad, 1e-6)
    lower = median - multiplier * robust_sigma
    upper = median + multiplier * robust_sigma
    clipped = np.clip(values, lower, upper)
    return clipped, {
        "median": median,
        "mad": mad,
        "robust_sigma": robust_sigma,
        "lower": lower,
        "upper": upper,
        "rows": int(len(values)),
        "rows_deleted": 0,
        "rows_clipped": int(np.count_nonzero(clipped != values)),
    }


def _finite_column_median(frame: pd.DataFrame) -> pd.Series:
    cleaned = frame.replace([np.inf, -np.inf], np.nan)
    median = cleaned.median(axis=0, numeric_only=True)
    if median.isna().any():
        raise ContractError("feature column has no finite training value")
    return median


def month_center_features(
    train: pd.DataFrame,
    query: pd.DataFrame,
    train_month: np.ndarray,
    query_month: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Subtract training-only month medians with a global fallback."""

    train = train.replace([np.inf, -np.inf], np.nan).copy()
    query = query.replace([np.inf, -np.inf], np.nan).copy()
    global_center = _finite_column_median(train)
    train_output = train.copy()
    query_output = query.copy()
    month_receipt: dict[str, int] = {}
    available = sorted(int(value) for value in np.unique(train_month))
    for month in available:
        selected = train_month == month
        center = train.loc[selected].median(axis=0, numeric_only=True).fillna(global_center)
        train_output.loc[selected] = train.loc[selected] - center
        month_receipt[str(month)] = int(selected.sum())
    fallback_rows = 0
    for month in sorted(int(value) for value in np.unique(query_month)):
        selected = query_month == month
        if month in available:
            center = train.loc[train_month == month].median(
                axis=0, numeric_only=True
            ).fillna(global_center)
        else:
            center = global_center
            fallback_rows += int(selected.sum())
        query_output.loc[selected] = query.loc[selected] - center
    return train_output, query_output, {
        "available_training_months": available,
        "training_rows_by_month": month_receipt,
        "query_rows_using_layer_global_fallback": fallback_rows,
    }


def robust_scale_features(
    train: pd.DataFrame,
    query: pd.DataFrame,
    *,
    lower_quantile: float,
    upper_quantile: float,
    clip: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = train.replace([np.inf, -np.inf], np.nan)
    query = query.replace([np.inf, -np.inf], np.nan)
    median = _finite_column_median(train)
    filled_train = train.fillna(median)
    filled_query = query.fillna(median)
    lower = filled_train.quantile(lower_quantile)
    upper = filled_train.quantile(upper_quantile)
    scale = (upper - lower).clip(lower=1e-6)
    train_scaled = ((filled_train - median) / scale).clip(-clip, clip)
    query_scaled = ((filled_query - median) / scale).clip(-clip, clip)
    if not (np.isfinite(train_scaled.to_numpy()).all() and np.isfinite(query_scaled.to_numpy()).all()):
        raise ContractError("scaled feature contains non-finite values")
    return train_scaled.to_numpy(float), query_scaled.to_numpy(float), {
        "feature_count": int(train.shape[1]),
        "training_missing_cells_imputed": int(train.isna().sum().sum()),
        "query_missing_cells_imputed": int(query.isna().sum().sum()),
        "zero_or_tiny_scale_columns": int(((upper - lower) < 1e-6).sum()),
        "scaled_clip": float(clip),
    }


def climatology_center_target(
    train_target: np.ndarray,
    train_month: np.ndarray,
    query_month: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Center target by training month; unseen months use training global median."""

    global_center = float(np.median(train_target))
    train_center = np.full(len(train_target), global_center, dtype=float)
    query_center = np.full(len(query_month), global_center, dtype=float)
    medians: dict[str, float] = {}
    available = sorted(int(value) for value in np.unique(train_month))
    for month in available:
        value = float(np.median(train_target[train_month == month]))
        medians[str(month)] = value
        train_center[train_month == month] = value
        query_center[query_month == month] = value
    fallback = int(np.count_nonzero(~np.isin(query_month, available)))
    return train_target - train_center, query_center, {
        "training_month_medians": medians,
        "layer_global_fallback": global_center,
        "query_rows_using_layer_global_fallback": fallback,
    }


def fit_candidate(
    spec: CandidateSpec,
    config: dict[str, Any],
    design: Any,
    train_mask: np.ndarray,
    query_positions: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    training = config["training"]
    all_local = design.keys["time"].dt.tz_convert("Asia/Seoul")
    all_month = all_local.dt.month.to_numpy(int)
    query_month = all_month[query_positions]
    candidate = reference.copy()
    receipts: list[dict[str, Any]] = []
    for layer in TARGET_LAYERS:
        layer_train = train_mask & design.keys["layer"].eq(layer).to_numpy()
        layer_query = design.keys.iloc[query_positions]["layer"].eq(layer).to_numpy()
        train_x = design.features.loc[layer_train].reset_index(drop=True)
        query_x = design.features.iloc[query_positions].loc[layer_query].reset_index(drop=True)
        train_month = all_month[layer_train]
        layer_query_month = query_month[layer_query]
        target = design.normalized_target[layer_train].copy()
        climatology_receipt: dict[str, Any] = {
            "mode": "none",
            "query_rows_using_layer_global_fallback": 0,
        }
        feature_center_receipt: dict[str, Any] = {
            "mode": "none",
            "query_rows_using_layer_global_fallback": 0,
        }
        query_target_center = np.zeros(int(layer_query.sum()), dtype=float)
        if spec.normalization == "training_only_layer_month_climatology_with_layer_fallback":
            train_x, query_x, feature_center_receipt = month_center_features(
                train_x, query_x, train_month, layer_query_month
            )
            target, query_target_center, climatology_receipt = climatology_center_target(
                target, train_month, layer_query_month
            )
            climatology_receipt["mode"] = "training_only_layer_month"
            feature_center_receipt["mode"] = "training_only_layer_month"
        elif spec.normalization != "global_robust_per_target_layer":
            raise ContractError(f"unknown normalization: {spec.normalization}")

        target, winsor_receipt = robust_winsorize(
            target, float(training["target_winsor_mad_multiplier"])
        )
        train_scaled, query_scaled, scale_receipt = robust_scale_features(
            train_x,
            query_x,
            lower_quantile=float(training["feature_scale_quantiles"][0]),
            upper_quantile=float(training["feature_scale_quantiles"][1]),
            clip=float(training["feature_scaled_clip"]),
        )
        model = Ridge(
            alpha=float(training["ridge_alpha"]),
            fit_intercept=bool(training["fit_intercept"]),
        )
        model.fit(train_scaled, target)
        predicted_z = np.asarray(model.predict(query_scaled), dtype=float) + query_target_center
        positions = query_positions[layer_query]
        model_absolute = design.baseline[positions] + predicted_z * design.profile_scale[positions]
        raw_delta = model_absolute - reference[layer_query]
        cap = np.minimum(
            float(training["model_minus_champion_clip_absolute_C"]),
            float(training["model_minus_champion_clip_scale_fraction"])
            * design.profile_scale[positions],
        )
        bounded_delta = np.clip(raw_delta, -cap, cap)
        model_weight = float(training["model_weight"])
        candidate[layer_query] = reference[layer_query] + model_weight * bounded_delta
        receipts.append(
            {
                "layer": layer,
                "rows_train": int(layer_train.sum()),
                "rows_query": int(layer_query.sum()),
                "ridge_alpha": float(training["ridge_alpha"]),
                "coefficient_l2": float(np.linalg.norm(model.coef_)),
                "intercept": float(model.intercept_),
                "winsor": winsor_receipt,
                "feature_scaling": scale_receipt,
                "feature_month_center": feature_center_receipt,
                "target_climatology": climatology_receipt,
                "raw_model_minus_champion_rms_C": float(
                    np.sqrt(np.mean(np.square(raw_delta)))
                ),
                "bounded_rows": int(np.count_nonzero(bounded_delta != raw_delta)),
                "final_change_rms_C": float(
                    np.sqrt(np.mean(np.square(model_weight * bounded_delta)))
                ),
            }
        )
    if not np.isfinite(candidate).all():
        raise ContractError("candidate contains non-finite predictions")
    return candidate, receipts


def audit_v7_report_only(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source_contract"]
    result = json.loads((ROOT / source["v7_report_result"]).read_text(encoding="utf-8"))
    qa = json.loads((ROOT / source["v7_report_root_qa"]).read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / source["v7_official_receipt"]).read_text(encoding="utf-8"))
    extra_result = json.loads(
        (ROOT / "reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json").read_text(
            encoding="utf-8"
        )
    )
    passing = next(item for item in result["candidates"] if item["pass"])
    p2_receipt = receipt["problems"]["P2"]
    attested_hash = qa["submission_recomputed"]["sha256"]
    expected_hash = "c6f2a7e02ff3e5064ec653af0a52b117cbf8ae49d80e651a2a96276190f4f620"
    pack_path = Path(qa["submission_recomputed"]["path"])
    calibrated_points = float(passing["calibrated_expected_points_delta"])
    points_per_rmse = float(config["evaluation"]["points_per_rmse_C"])
    champion_rmse = 0.430194
    champion_points = 27.935464
    exact_family_submitted = any(
        str(item.get("candidate")) == "P2_V7_EXTRATREES_PUBLIC_BENEFIT_GATE"
        for item in extra_result["submissions"]
    )
    checks = {
        "v7_result_pass": result["status"] == "COMPLETE_WITH_PASS",
        "root_ready_qa_pass": qa["status"] == "PASS" and all(qa["checks"].values()),
        "attested_rows_schema_key_order_finite": all(
            qa["submission_recomputed"][key]
            for key in (
                "rows_26061",
                "schema_exact",
                "key_order_exact",
                "duplicates_zero",
                "finite",
                "sha256_matches",
            )
        ),
        "attested_hash_exact": attested_hash == expected_hash,
        "pack_path_still_exists_metadata_only": pack_path.is_file(),
        "upload_was_not_attempted": p2_receipt["upload_attempted"] is False,
        "blocked_by_daily_limit": p2_receipt["verdict"] == "UPLOAD_BLOCKED_DAILY_LIMIT",
        "no_later_exact_family_submission_in_report_ledger": not exact_family_submitted,
    }
    return {
        "audit_scope": "REPORT_VALUES_PLUS_PACK_FILE_METADATA_ONLY_NO_CSV_VALUE_OR_HASH_REREAD",
        "status": "REPORT_ATTESTED_READY_UNSUBMITTED_HIGH_TRANSPORT_RISK"
        if all(checks.values())
        else "REPORT_ATTESTATION_INCOMPLETE",
        "checks": checks,
        "candidate": "P2_V7_EXTRATREES_PUBLIC_BENEFIT_GATE",
        "attested_csv_sha256": attested_hash,
        "pack_metadata": {
            "path": str(pack_path),
            "exists": pack_path.is_file(),
            "bytes": pack_path.stat().st_size if pack_path.is_file() else None,
            "last_write_time_ns": pack_path.stat().st_mtime_ns if pack_path.is_file() else None,
        },
        "internal_october_delta_rmse_C": float(passing["october_delta_rmse"]),
        "internal_raw_expected_points_delta": float(passing["raw_expected_points_delta"]),
        "transport_calibrated_expected_points_delta": calibrated_points,
        "current_champion": {"rmse_C": champion_rmse, "points": champion_points},
        "nominal_if_calibrated_delta_transported": {
            "rmse_C": champion_rmse - calibrated_points / points_per_rmse,
            "points": champion_points + calibrated_points,
        },
        "exact_family_refuted_by_later_official_result": False,
        "why_not_refuted": "Later P2 official regressions were HGB absolute-profile and shallow residual families, not the exact ExtraTrees benefit-gate pack.",
        "priority": "SECONDARY_INFORMATION_PROBE_NOT_PRIMARY",
        "priority_reason": "The report-attested edge is only +0.013765 points after the then-known penalty, all validation is exposed, and later non-identical P2 families showed severe sign reversal on Public.",
    }


def write_report(result: dict[str, Any]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P2 domain-invariant vertical-curvature exploratory cycle 20260901 v9",
        "",
        "## 결론",
        "",
        f"상태: `{result['status']}`. 세 historical fold는 모두 이미 노출된 exploratory surface이며 fresh confirmation이 아니다.",
        "",
        "| 후보 | pooled ΔRMSE | Sep-Oct | Jul-Aug | Nov-Dec | raw 예상점수 | transport 보정점수 | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result["candidates"]:
        folds = item["by_fold"]
        lines.append(
            "| {name} | {pooled:+.9f} | {sep:+.9f} | {jul:+.9f} | {nov:+.9f} | {raw:+.6f} | {cal:+.6f} | {gate} |".format(
                name=item["name"],
                pooled=item["delta_rmse"],
                sep=folds["2024_sep_oct"]["delta_rmse"],
                jul=folds["2025_jul_aug"]["delta_rmse"],
                nov=folds["2025_nov_dec"]["delta_rmse"],
                raw=item["raw_expected_points_delta"],
                cal=item["transport_calibrated_expected_points_delta"],
                gate="PASS" if item["strict_exploratory_pass"] else "NO_GO",
            )
        )
    v7 = result["v7_readiness_audit"]
    lines.extend(
        [
            "",
            "## v7 기존 제출팩 보고서 감사",
            "",
            f"`{v7['candidate']}`는 보고서 원장상 `{v7['status']}`이다. 당시 daily limit로 upload attempted=false였고 exact family 후속 공식 제출 기록은 없다.",
            f"보정 기대치는 +{v7['transport_calibrated_expected_points_delta']:.6f}점, 현 챔피언 기준 명목 RMSE {v7['nominal_if_calibrated_delta_transported']['rmse_C']:.6f}°C / {v7['nominal_if_calibrated_delta_transported']['points']:.6f}점이다.",
            "CSV 값이나 bytes를 다시 읽지 않았고, 기존 independent root-ready QA의 schema/key/order/finite/hash attestation과 현재 파일 metadata 존재만 대조했다.",
            "",
            "## 중복 배제와 안전장치",
            "",
            "v8r1과 달리 raw-Celsius L1/L2 잔차를 학습하지 않는다. 공개층 기반의 무차원 normalized-curvature target을 layer별 Ridge로 학습하고, 결과 전 고정한 0.2 blend와 bounded delta로 챔피언 proxy를 0.8 보존한다.",
            "두 번째 후보는 training-only layer-month median을 제거하며 unseen month에는 layer-global median만 사용한다. 행 삭제는 0이고 target에만 4-MAD fixed winsorization을 적용했다.",
            "후보 postprocess, DINEOF/GP/CatBoost/soft-gate/PAVA/rank-season-bin search는 사용하지 않았다. comparator의 기존 projection lineage는 변경하지 않았다.",
            "",
            "## 검증 경계",
            "",
            "점수 환산은 기존 소규모 delta calibration의 명목값이며 공식 기대값이 아니다. 세 fold와 bootstrap은 후보 폐쇄와 우선순위 판단만 지원한다.",
            f"fits={result['fit_count']}; official/test/sample/baseline/score/query/hidden rows=0; submission CSV=0; upload=0.",
        ]
    )
    (REPORT / "report-source.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    started = time.perf_counter()
    if ARTIFACT.exists():
        raise FileExistsError(f"exactly-once artifact already exists: {ARTIFACT}")
    config = load_config()
    ARTIFACT.mkdir(parents=True)
    atomic_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(CONFIG),
            "runner_sha256": sha256_file(RUNNER),
            "candidate_order": [item["name"] for item in config["candidates"]],
            "candidate_count": 2,
            "maximum_fit_count": 6,
            "result_adaptive_tuning": False,
            "row_deletion": False,
            "official_access_before_lock": 0,
        },
    )

    observations_path = resolve_observations(config)
    scoring_path = ROOT / config["source_contract"]["scoring_frame"]
    if sha256_file(scoring_path) != config["source_contract"]["scoring_frame_sha256"]:
        raise ContractError("truth-free scoring frame hash drift")
    observations = pd.read_csv(
        observations_path, dtype={"station": "string", "time": "string"}
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    if observations.duplicated(["time", "layer"]).any():
        raise ContractError("observation keys are duplicated")
    scored = pd.read_parquet(scoring_path)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    if tuple(sorted(scored["fold"].unique())) != tuple(sorted(FOLD_ORDER)):
        raise ContractError("historical fold set drift")
    blind, reference = prior_engine.make_reference(observations, scored)

    feature_table = build_training_features(observations)
    design = build_normalized_curvature_design(feature_table.frame)
    local = design.keys["time"].dt.tz_convert("Asia/Seoul")
    window_ids = prior_engine.registered_window_ids(
        local, config["training"]["registered_windows_kst"]
    )
    train_mask = window_ids != ""
    if int(train_mask.sum()) != int(config["training"]["expected_rows"]):
        raise ContractError("registered training row count drift")
    design_index = pd.MultiIndex.from_arrays(
        [prior_engine.canonical_time_ns(design.keys["time"]), design.keys["layer"]],
        names=("time", "layer"),
    )
    query_index = pd.MultiIndex.from_arrays(
        [prior_engine.canonical_time_ns(blind["time"]), blind["layer"]],
        names=("time", "layer"),
    )
    if design_index.has_duplicates:
        raise ContractError("feature design keys are duplicated")
    query_positions = design_index.get_indexer(query_index)
    if np.any(query_positions < 0):
        raise ContractError("historical feature alignment failed")

    specs = [
        CandidateSpec(item["name"], item["normalization"])
        for item in config["candidates"]
    ]
    pending: list[tuple[CandidateSpec, np.ndarray, dict[str, Any]]] = []
    fit_count = 0
    for spec in specs:
        candidate, receipts = fit_candidate(
            spec, config, design, train_mask, query_positions, reference
        )
        fit_count += len(receipts)
        output = ARTIFACT / f"{spec.name}.npz"
        np.savez_compressed(
            output,
            time_ns=prior_engine.canonical_time_ns(blind["time"]),
            layer=blind["layer"].to_numpy(np.int16),
            fold=blind["fold"].to_numpy(str),
            reference=reference,
            candidate=candidate,
        )
        commitment = {
            "name": spec.name,
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": int(len(candidate)),
            "fit_receipts": receipts,
            "metric_computed_at_commitment": False,
            "truth_used_to_choose_action": False,
        }
        atomic_json(ARTIFACT / f"{spec.name}.commitment.json", commitment)
        pending.append((spec, candidate, commitment))
    if fit_count != 6:
        raise ContractError(f"fit count drift: {fit_count}")

    # Truth is consulted only after both prediction vectors and hashes are sealed.
    truth = design.truth[query_positions]
    candidates: list[dict[str, Any]] = []
    for spec, candidate, commitment in pending:
        metric_spec = prior_engine.CandidateSpec(
            name=spec.name,
            objective="ridge_dimensionless_vertical_curvature",
            conditional=False,
        )
        record = prior_engine.evaluate_candidate(
            metric_spec, blind, truth, reference, candidate, config
        )
        record["normalization"] = spec.normalization
        record["prediction_commitment"] = commitment
        candidates.append(record)

    v7_audit = audit_v7_report_only(config)
    status = (
        "EXPLORATORY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if any(item["strict_exploratory_pass"] for item in candidates)
        else "EXPLORATORY_NO_GO_BOTH_SEALED_CANDIDATES"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "claim_level": "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": fit_count,
        "training_rows": int(train_mask.sum()),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "v7_readiness_audit": v7_audit,
        "operation_counters": {
            "observations_rows_read": int(len(observations)),
            "historical_truth_free_scoring_rows_read": int(len(scored)),
            "report_files_read_for_v7_audit": 4,
            "official_test_index_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_file_rows_read": 0,
            "score_file_rows_read": 0,
            "query_support_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "existing_v7_submission_csv_value_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config": sha256_file(CONFIG),
            "runner": sha256_file(RUNNER),
            "observations": sha256_file(observations_path),
            "scoring_frame": sha256_file(scoring_path),
            "normalized_curvature_module": sha256_file(
                ROOT / "src" / "p2_restore" / "normalized_curvature_residual.py"
            ),
            "prior_metric_engine": sha256_file(
                ROOT / "scripts" / "run_p2_group_balanced_raw_residual_20260901_v8.py"
            ),
        },
    }
    atomic_json(ARTIFACT / "result.json", result)
    REPORT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT / "result.json", result)
    atomic_json(REPORT / "v7-readiness-audit.json", v7_audit)
    write_report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight or --execute")
    payload = preflight() if args.preflight else run()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
