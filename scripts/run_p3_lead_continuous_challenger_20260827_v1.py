"""One-shot local P3 lead-continuous structural challenger.

This runner is deliberately local-only.  It consumes three pinned training/OOF
artifacts, never opens anonymous evaluation inputs, and never creates a prediction
upload.  The single preregistered arm is a six-parameter smooth shrink surface:

    candidate = incumbent + clip((persistence - incumbent) * w(lead, regime))

where ``w`` is continuous in lead time, vanishes at lead zero, and uses only causal
48-hour-context summaries.  Every outer fold is predicted from strictly earlier
fold truth; the first fold is an exact identity/no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXPERIMENT_ID: Final = "p3_lead_continuous_challenger_20260827_v1"
SCHEMA_VERSION: Final = "p3.lead_continuous_challenger.20260827.v1"
FOLD_ORDER: Final = ("2024_h2_storm", "winter_transition", "2025_h1")
LEADS: Final = (3, 6, 9, 12, 18, 24)
REGIME_FEATURES: Final = (
    "hs_delta_3h",
    "hs_std_6h",
    "wspd_delta_3h",
    "caph_delta_6h",
)
BASIS_NAMES: Final = (
    "lead_linear",
    "lead_quadratic",
    "lead_x_wave_trend",
    "lead_x_wave_volatility",
    "lead_x_wind_trend",
    "lead_x_pressure_fall",
)
RIDGE_ALPHA: Final = 16.0
ROBUST_Z_CLIP: Final = 3.0
WEIGHT_MIN: Final = -0.25
WEIGHT_MAX: Final = 0.50
CORRECTION_LIMIT_M: Final = 0.15
PREDICTION_MIN_M: Final = 0.0
PREDICTION_MAX_M: Final = 30.0
REGIME_THRESHOLD_M: Final = 0.20
BOOTSTRAP_REPLICATES: Final = 5_000
CASE_BOOTSTRAP_SEED: Final = 2026082701
DAY_BOOTSTRAP_SEED: Final = 2026082702

PROMOTE_OVERALL_DELTA_M: Final = -0.003
PROMOTE_ACTIVE_DELTA_M: Final = -0.005
PROMOTE_MAX_SLICE_REGRESSION_M: Final = 0.015
REJECT_MAX_SLICE_REGRESSION_M: Final = 0.030

OOF_RELATIVE: Final = Path(
    "artifacts/p3_corrected_repeated_forward_catboost_v2/oof.parquet"
)
ANCHORS_RELATIVE: Final = Path("artifacts/p3/features_all20_v1/train_anchors.parquet")
FEATURES_RELATIVE: Final = Path("artifacts/p3/features_all20_v1/train_features.parquet")
OUTPUT_RELATIVE: Final = Path("artifacts/structural_challenger_20260827_v1/p3")
SCRIPT_RELATIVE: Final = Path("scripts/run_p3_lead_continuous_challenger_20260827_v1.py")

PINNED_SHA256: Final = {
    OOF_RELATIVE.as_posix(): "eb0af75ec29210254da0d13d1bb8164c0d6b427f4ad5853622144a11fe795f7e",
    ANCHORS_RELATIVE.as_posix(): "07452389a19efd63121f4465a9c08cf7f9ef9e58cf1e3ea1f577e2dca5d8611a",
    FEATURES_RELATIVE.as_posix(): "f974e7951ed9490e68b96154f89afd69ee98e4ed2d27c179fc898779a4aec388",
}

OOF_COLUMNS: Final = (
    "anchor_id",
    "station",
    "lead_h",
    "current_hs",
    "target_hs",
    "fold",
    "persistence",
    "final_prediction",
)
ANCHOR_COLUMNS: Final = ("anchor_id", "station", "anchor_time")
FEATURE_COLUMNS: Final = ("anchor_id", "station", "hs_current", *REGIME_FEATURES)


class ContractError(RuntimeError):
    """Raised when the frozen local-only experiment contract differs."""


@dataclass(frozen=True)
class SmoothShrinkModel:
    medians: np.ndarray
    robust_scales: np.ndarray
    basis_scales: np.ndarray
    coefficients: np.ndarray

    def receipt(self) -> dict[str, Any]:
        return {
            "class": "ContinuousLeadRegimePersistenceAxisRidge",
            "regime_features": list(REGIME_FEATURES),
            "basis_names": list(BASIS_NAMES),
            "ridge_alpha": RIDGE_ALPHA,
            "medians": self.medians.tolist(),
            "robust_scales": self.robust_scales.tolist(),
            "basis_scales": self.basis_scales.tolist(),
            "coefficients": self.coefficients.tolist(),
            "weight_bounds": [WEIGHT_MIN, WEIGHT_MAX],
            "correction_limit_m": CORRECTION_LIMIT_M,
        }


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: Any) -> None:
    data = _canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("exclusive JSON write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git_state(root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "head": head.stdout.strip() if head.returncode == 0 else "unknown",
        "dirty": bool(dirty),
        "dirty_entry_count": len(dirty),
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _verify_inputs(root: Path) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in PINNED_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise ContractError(f"pinned local input missing: {relative}")
        observed = _sha256(path)
        if observed != expected:
            raise ContractError(f"pinned local input SHA differs: {relative}")
        verified[relative] = {"bytes": path.stat().st_size, "sha256": observed}
    schemas = {
        OOF_RELATIVE.as_posix(): tuple(pq.ParquetFile(root / OOF_RELATIVE).schema_arrow.names),
        ANCHORS_RELATIVE.as_posix(): tuple(
            pq.ParquetFile(root / ANCHORS_RELATIVE).schema_arrow.names
        ),
        FEATURES_RELATIVE.as_posix(): tuple(
            pq.ParquetFile(root / FEATURES_RELATIVE).schema_arrow.names
        ),
    }
    for required, relative in (
        (OOF_COLUMNS, OOF_RELATIVE),
        (ANCHOR_COLUMNS, ANCHORS_RELATIVE),
        (FEATURE_COLUMNS, FEATURES_RELATIVE),
    ):
        missing = set(required).difference(schemas[relative.as_posix()])
        if missing:
            raise ContractError(f"pinned local schema missing columns: {sorted(missing)}")
    return verified


def _preregistration(root: Path, verified: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_VERSION}.preregistration",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": _now_kst(),
        "hypothesis_count": 1,
        "hypothesis": (
            "A six-parameter continuous lead-by-causal-regime shrink surface, constrained "
            "to the incumbent/persistence axis and fit only on earlier blocked folds, "
            "improves the frozen local incumbent without a material station, regime, or "
            "short/long-horizon regression."
        ),
        "nonduplication": {
            "no_lead_one_hot": True,
            "no_tree_neural_state_space_or_spectral_refit": True,
            "not_a_constant_long_lead_persistence_weight": True,
            "correction_restricted_to_incumbent_persistence_axis": True,
            "continuous_lead_coordinate_vanishes_at_zero": True,
            "prior_failures_audited": {
                "event_balanced_trajectory_delta_m": 0.13570947773470388,
                "causal_forcing_sequence_delta_m": 0.03785402015984651,
                "causal_spectral_kernel_delta_m": 0.042782536910337954,
                "dense72_delta_m": 0.06729476403367496,
                "state_space_full_delta_m": 0.04281365149503025,
                "fixed_25pct_long_shrink_delta_m": -0.00044396004699287506,
                "gen6_result": "NO_LOCAL_CURVE_QUALIFICATION_RESEARCH_ONLY",
            },
        },
        "surface": {
            "expected_cases": 181,
            "expected_rows": 1086,
            "fold_order": list(FOLD_ORDER),
            "leads_h": list(LEADS),
            "anchor_spacing_h": 78,
            "context_h": 48,
            "target_h": 24,
            "buffer_h": 6,
            "comparison_keys": ["fold", "anchor_id", "station", "lead_h"],
            "incumbent_column": "final_prediction",
            "persistence_column": "persistence",
        },
        "model": {
            "basis": list(BASIS_NAMES),
            "regime_features": list(REGIME_FEATURES),
            "lead_coordinate": "lead_h / 24",
            "ridge_alpha": RIDGE_ALPHA,
            "robust_center_scale": "history-only median and 1.4826*MAD",
            "robust_z_clip": ROBUST_Z_CLIP,
            "weight_bounds": [WEIGHT_MIN, WEIGHT_MAX],
            "correction_limit_m": CORRECTION_LIMIT_M,
            "prediction_range_m": [PREDICTION_MIN_M, PREDICTION_MAX_M],
            "hyperparameter_candidates": 1,
            "search_runs": 0,
            "posthoc_tuning_allowed": False,
        },
        "prequential_rule": {
            "first_fold": "EXACT_IDENTITY_NO_PRIOR_TRUTH",
            "later_folds": "fit once on all strictly earlier folds and predict current fold",
            "current_fold_target_used_for_fit": False,
            "full_surface_no_op_arm": "frozen incumbent",
        },
        "uncertainty": {
            "replicates": BOOTSTRAP_REPLICATES,
            "complete_case_seed": CASE_BOOTSTRAP_SEED,
            "utc_anchor_day_seed": DAY_BOOTSTRAP_SEED,
            "interval": "paired percentile 90%",
        },
        "decision_gate": {
            "PROMOTE": {
                "overall_delta_at_most_m": PROMOTE_OVERALL_DELTA_M,
                "active_delta_at_most_m": PROMOTE_ACTIVE_DELTA_M,
                "active_day_ci90_upper_below_zero": True,
                "both_active_folds_improve": True,
                "worst_eligible_slice_regression_at_most_m": (
                    PROMOTE_MAX_SLICE_REGRESSION_M
                ),
            },
            "REJECT": {
                "active_delta_nonnegative_or_day_ci90_lower_positive": True,
                "both_active_folds_nonimproving": True,
                "worst_eligible_slice_regression_above_m": (
                    REJECT_MAX_SLICE_REGRESSION_M
                ),
            },
            "otherwise": "INCONCLUSIVE",
        },
        "read_scope": {
            "pinned_inputs": verified,
            "anonymous_evaluation_inputs_allowed": False,
            "current_or_round_submission_inputs_allowed": False,
            "external_data_allowed": False,
            "raw_row_output_allowed": False,
        },
        "implementation": {
            "script": SCRIPT_RELATIVE.as_posix(),
            "script_sha256": _sha256(root / SCRIPT_RELATIVE),
        },
    }


def _load_surface(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    oof = pd.read_parquet(root / OOF_RELATIVE, columns=list(OOF_COLUMNS)).copy()
    oof.insert(0, "_row_order", np.arange(len(oof), dtype=np.int64))
    anchors = pd.read_parquet(root / ANCHORS_RELATIVE, columns=list(ANCHOR_COLUMNS))
    features = pd.read_parquet(root / FEATURES_RELATIVE, columns=list(FEATURE_COLUMNS))

    keys = ["fold", "anchor_id", "station", "lead_h"]
    if len(oof) != 1_086 or oof.duplicated(keys).any() or oof[keys].isna().any().any():
        raise ContractError("OOF row/key contract differs")
    if tuple(dict.fromkeys(oof["fold"].astype(str))) != FOLD_ORDER:
        raise ContractError("OOF chronological fold order differs")
    if set(oof["lead_h"].astype(int)) != set(LEADS):
        raise ContractError("OOF lead contract differs")
    lead_contract = oof.groupby(
        ["fold", "anchor_id", "station"], sort=False, observed=True
    )["lead_h"].agg(lambda value: tuple(sorted(value.astype(int))))
    if len(lead_contract) != 181 or not lead_contract.map(lambda value: value == LEADS).all():
        raise ContractError("OOF does not contain 181 complete six-lead cases")
    numeric = oof[
        ["current_hs", "target_hs", "persistence", "final_prediction"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ContractError("OOF scoring values are non-finite")
    if float(oof["current_hs"].min()) < 1.5 - 1e-12:
        raise ContractError("OOF anchor does not mimic the high-wave condition")

    case_keys = oof[["fold", "anchor_id", "station"]].drop_duplicates()
    anchor_view = anchors.merge(
        case_keys[["anchor_id", "station"]],
        on=["anchor_id", "station"],
        how="inner",
        validate="one_to_one",
    )
    feature_view = features.merge(
        case_keys[["anchor_id", "station"]],
        on=["anchor_id", "station"],
        how="inner",
        validate="one_to_one",
    )
    if len(anchor_view) != 181 or len(feature_view) != 181:
        raise ContractError("anchor/feature join does not cover the exact OOF cases")
    case_meta = case_keys.merge(
        anchor_view, on=["anchor_id", "station"], how="left", validate="one_to_one"
    ).merge(
        feature_view,
        on=["anchor_id", "station"],
        how="left",
        validate="one_to_one",
    )
    if case_meta["anchor_time"].isna().any():
        raise ContractError("OOF anchor times are incomplete")
    if not np.allclose(
        case_meta["hs_current"].to_numpy(float),
        case_meta["hs_current"].to_numpy(float),
        equal_nan=True,
    ):
        raise ContractError("unexpected non-reflexive current-Hs values")

    frame = oof.merge(
        case_meta[
            ["fold", "anchor_id", "station", "anchor_time", "hs_current", *REGIME_FEATURES]
        ],
        on=["fold", "anchor_id", "station"],
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("_row_order", kind="stable")
    if not np.allclose(
        frame["current_hs"].to_numpy(float),
        frame["hs_current"].to_numpy(float),
        atol=0.0,
        rtol=0.0,
    ):
        raise ContractError("feature-cache current Hs differs from OOF current Hs")

    unique_cases = frame[
        ["fold", "anchor_id", "station", "anchor_time"]
    ].drop_duplicates()
    station_gaps: dict[str, float] = {}
    for station, part in unique_cases.groupby("station", sort=True, observed=True):
        gap = (
            part.sort_values("anchor_time")["anchor_time"]
            .diff()
            .dt.total_seconds()
            .div(3600.0)
            .dropna()
        )
        station_gaps[str(station)] = float(gap.min())
    minimum_gap = min(station_gaps.values())
    if minimum_gap < 78.0 - 1e-12:
        raise ContractError("blocked OOF anchor gap is below 78 hours")
    chronology: list[dict[str, Any]] = []
    for index, fold in enumerate(FOLD_ORDER):
        part = unique_cases.loc[unique_cases["fold"].astype(str).eq(fold)]
        if part.empty:
            raise ContractError(f"blocked fold is empty: {fold}")
        prior = unique_cases.loc[
            unique_cases["fold"].astype(str).isin(FOLD_ORDER[:index])
        ]
        separation = None
        station_boundary_separation: dict[str, float] = {}
        if not prior.empty:
            for station in sorted(set(part["station"]).intersection(prior["station"])):
                current_station = part.loc[part["station"].eq(station), "anchor_time"]
                prior_station = prior.loc[prior["station"].eq(station), "anchor_time"]
                station_boundary_separation[str(station)] = float(
                    (current_station.min() - prior_station.max()).total_seconds() / 3600.0
                )
            separation = min(station_boundary_separation.values())
            if separation < 78.0 - 1e-12:
                raise ContractError("prequential fold boundary is below the 78-hour contract")
        chronology.append(
            {
                "fold": fold,
                "cases": int(len(part)),
                "prior_to_current_min_boundary_h": separation,
                "prior_to_current_station_boundary_h": station_boundary_separation,
            }
        )
    audit = {
        "cases": 181,
        "rows": 1086,
        "keys_unique": True,
        "complete_six_lead_cases": True,
        "station_minimum_gap_h": station_gaps,
        "minimum_gap_h": minimum_gap,
        "minimum_context_target_buffer_h": minimum_gap - 72.0,
        "fold_chronology": chronology,
        "official_condition_mimic": {
            "past_context_h": 48,
            "future_target_h": 24,
            "buffer_h": 6,
            "current_hs_at_least_1p5": True,
        },
    }
    return frame.reset_index(drop=True), audit


def _robust_parameters(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, REGIME_FEATURES].to_numpy(dtype=np.float64)
    medians = np.nanmedian(values, axis=0)
    if not np.isfinite(medians).all():
        raise ContractError("a regime feature has no finite historical values")
    filled = np.where(np.isfinite(values), values, medians)
    mad = np.median(np.abs(filled - medians), axis=0)
    scales = 1.4826 * mad
    fallback = np.std(filled, axis=0)
    scales = np.where(scales > 1e-6, scales, fallback)
    scales = np.where(scales > 1e-6, scales, 1.0)
    return medians, scales


def _raw_basis(
    frame: pd.DataFrame, medians: np.ndarray, robust_scales: np.ndarray
) -> np.ndarray:
    values = frame.loc[:, REGIME_FEATURES].to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(values), values, medians)
    z = np.clip((values - medians) / robust_scales, -ROBUST_Z_CLIP, ROBUST_Z_CLIP)
    lead = frame["lead_h"].to_numpy(dtype=np.float64) / 24.0
    basis = np.column_stack(
        [
            lead,
            np.square(lead),
            lead * z[:, 0],
            lead * z[:, 1],
            lead * z[:, 2],
            lead * (-z[:, 3]),
        ]
    )
    if basis.shape[1] != len(BASIS_NAMES) or not np.isfinite(basis).all():
        raise ContractError("continuous lead-regime basis is invalid")
    return basis


def _fit_model(history: pd.DataFrame) -> SmoothShrinkModel:
    if history[["anchor_id", "station"]].drop_duplicates().shape[0] < 24:
        raise ContractError("insufficient prior complete cases for fixed smooth model")
    medians, robust_scales = _robust_parameters(history)
    basis = _raw_basis(history, medians, robust_scales)
    basis_scales = np.sqrt(np.mean(np.square(basis), axis=0))
    basis_scales = np.where(basis_scales > 1e-8, basis_scales, 1.0)
    standardized = basis / basis_scales
    gap = (
        history["persistence"].to_numpy(float)
        - history["final_prediction"].to_numpy(float)
    )
    design = gap[:, None] * standardized
    target = (
        history["target_hs"].to_numpy(float)
        - history["final_prediction"].to_numpy(float)
    )
    gram = design.T @ design
    coefficients = np.linalg.solve(
        gram + RIDGE_ALPHA * np.eye(gram.shape[0], dtype=np.float64),
        design.T @ target,
    )
    if not np.isfinite(coefficients).all():
        raise ContractError("smooth shrink coefficients are non-finite")
    return SmoothShrinkModel(medians, robust_scales, basis_scales, coefficients)


def _predict_model(model: SmoothShrinkModel, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    basis = _raw_basis(frame, model.medians, model.robust_scales) / model.basis_scales
    raw_weight = basis @ model.coefficients
    weight = np.clip(raw_weight, WEIGHT_MIN, WEIGHT_MAX)
    gap = (
        frame["persistence"].to_numpy(float)
        - frame["final_prediction"].to_numpy(float)
    )
    raw_correction = gap * weight
    correction = np.clip(raw_correction, -CORRECTION_LIMIT_M, CORRECTION_LIMIT_M)
    prediction = np.clip(
        frame["final_prediction"].to_numpy(float) + correction,
        PREDICTION_MIN_M,
        PREDICTION_MAX_M,
    )
    if not np.isfinite(prediction).all():
        raise ContractError("challenger prediction is non-finite")
    return prediction, {
        "raw_weight_min": float(np.min(raw_weight)),
        "raw_weight_max": float(np.max(raw_weight)),
        "clipped_weight_fraction": float(np.mean(raw_weight != weight)),
        "maximum_absolute_correction_m": float(np.max(np.abs(correction))),
        "correction_limit_hit_fraction": float(np.mean(raw_correction != correction)),
    }


def _prequential_prediction(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    prediction = frame["final_prediction"].to_numpy(float).copy()
    receipts: list[dict[str, Any]] = []
    for index, fold in enumerate(FOLD_ORDER):
        current = frame["fold"].astype(str).eq(fold).to_numpy()
        history = frame["fold"].astype(str).isin(FOLD_ORDER[:index]).to_numpy()
        if not current.any():
            raise ContractError(f"current fold is empty: {fold}")
        if index == 0:
            if history.any():
                raise ContractError("first fold unexpectedly has prior truth")
            receipts.append(
                {
                    "fold": fold,
                    "decision": "EXACT_IDENTITY_NO_PRIOR_TRUTH",
                    "history_cases": 0,
                    "current_cases": int(
                        frame.loc[current, ["anchor_id", "station"]].drop_duplicates().shape[0]
                    ),
                    "current_fold_target_used_for_fit": False,
                    "identity_max_abs_m": 0.0,
                }
            )
            continue
        model = _fit_model(frame.loc[history])
        fold_prediction, prediction_audit = _predict_model(model, frame.loc[current])
        prediction[current] = fold_prediction
        receipts.append(
            {
                "fold": fold,
                "decision": "APPLY_SINGLE_PREREGISTERED_SMOOTH_SURFACE",
                "history_folds": list(FOLD_ORDER[:index]),
                "history_cases": int(
                    frame.loc[history, ["anchor_id", "station"]].drop_duplicates().shape[0]
                ),
                "current_cases": int(
                    frame.loc[current, ["anchor_id", "station"]].drop_duplicates().shape[0]
                ),
                "current_fold_target_used_for_fit": False,
                "model": model.receipt(),
                "prediction_audit": prediction_audit,
            }
        )
    first = frame["fold"].astype(str).eq(FOLD_ORDER[0]).to_numpy()
    identity_error = float(
        np.max(
            np.abs(
                prediction[first] - frame.loc[first, "final_prediction"].to_numpy(float)
            )
        )
    )
    if identity_error != 0.0:
        raise ContractError("first-fold exact no-op arm changed")
    audit = {
        "first_fold_identity_max_abs_m": identity_error,
        "first_fold_identity_bytes_equal": bool(
            prediction[first].tobytes()
            == frame.loc[first, "final_prediction"].to_numpy(float).tobytes()
        ),
        "assigned_rows": int(np.isfinite(prediction).sum()),
        "single_preregistered_arm": True,
        "hyperparameter_search_runs": 0,
        "posthoc_tuning_runs": 0,
    }
    return prediction, receipts, audit


def _rmse(truth: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    left = np.asarray(truth, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(left - right))))


def _metric(frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame["target_hs"].to_numpy(float)
    incumbent = frame["final_prediction"].to_numpy(float)
    candidate = frame["candidate_prediction"].to_numpy(float)
    persistence = frame["persistence"].to_numpy(float)
    incumbent_rmse = _rmse(truth, incumbent)
    candidate_rmse = _rmse(truth, candidate)
    return {
        "rows": int(len(frame)),
        "cases": int(frame[["anchor_id", "station"]].drop_duplicates().shape[0]),
        "incumbent_rmse_m": incumbent_rmse,
        "candidate_rmse_m": candidate_rmse,
        "persistence_rmse_m": _rmse(truth, persistence),
        "delta_candidate_minus_incumbent_m": candidate_rmse - incumbent_rmse,
    }


def _group_metrics(frame: pd.DataFrame, group: str) -> dict[str, dict[str, Any]]:
    return {
        str(value): _metric(part)
        for value, part in frame.groupby(group, sort=True, observed=True, dropna=False)
    }


def _bootstrap(
    frame: pd.DataFrame,
    block_columns: list[str],
    *,
    seed: int,
) -> dict[str, Any]:
    work = frame[
        [*block_columns, "target_hs", "final_prediction", "candidate_prediction"]
    ].copy()
    work["incumbent_se"] = np.square(
        work["final_prediction"].to_numpy(float) - work["target_hs"].to_numpy(float)
    )
    work["candidate_se"] = np.square(
        work["candidate_prediction"].to_numpy(float) - work["target_hs"].to_numpy(float)
    )
    blocks = (
        work.groupby(block_columns, sort=True, observed=True, dropna=False)
        .agg(rows=("target_hs", "size"), incumbent_sse=("incumbent_se", "sum"), candidate_sse=("candidate_se", "sum"))
        .reset_index(drop=True)
    )
    if len(blocks) < 12:
        raise ContractError("paired bootstrap has fewer than 12 blocks")
    rows = blocks["rows"].to_numpy(dtype=np.float64)
    incumbent_sse = blocks["incumbent_sse"].to_numpy(dtype=np.float64)
    candidate_sse = blocks["candidate_sse"].to_numpy(dtype=np.float64)
    generator = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = generator.integers(0, len(blocks), len(blocks))
        denominator = float(np.sum(rows[sampled]))
        deltas[index] = math.sqrt(float(np.sum(candidate_sse[sampled])) / denominator) - math.sqrt(
            float(np.sum(incumbent_sse[sampled])) / denominator
        )
    return {
        "blocks": int(len(blocks)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "delta_candidate_minus_incumbent_ci90_m": [
            float(np.quantile(deltas, 0.05)),
            float(np.quantile(deltas, 0.95)),
        ],
        "median_delta_m": float(np.median(deltas)),
        "probability_candidate_improves_descriptive": float(np.mean(deltas < 0.0)),
    }


def _evaluate(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame.copy()
    trend = work["hs_delta_3h"].to_numpy(float)
    work["regime"] = np.where(
        ~np.isfinite(trend),
        "unknown",
        np.where(
            trend < -REGIME_THRESHOLD_M,
            "falling",
            np.where(trend > REGIME_THRESHOLD_M, "rising", "stable"),
        ),
    )
    work["horizon_group"] = np.where(work["lead_h"].astype(int) <= 9, "short_3_9", "long_12_24")
    work["anchor_day_utc"] = work["anchor_time"].dt.floor("D")
    active = work.loc[~work["fold"].astype(str).eq(FOLD_ORDER[0])].copy()
    if active.empty:
        raise ContractError("active prequential surface is empty")

    breakdowns = {
        "by_fold": _group_metrics(work, "fold"),
        "by_station": _group_metrics(work, "station"),
        "by_lead": _group_metrics(work, "lead_h"),
        "by_horizon_group": _group_metrics(work, "horizon_group"),
        "by_regime": _group_metrics(work, "regime"),
    }
    active_breakdowns = {
        "by_fold": _group_metrics(active, "fold"),
        "by_station": _group_metrics(active, "station"),
        "by_lead": _group_metrics(active, "lead_h"),
        "by_horizon_group": _group_metrics(active, "horizon_group"),
        "by_regime": _group_metrics(active, "regime"),
    }
    uncertainty = {
        "all_complete_case": _bootstrap(
            work, ["anchor_id", "station"], seed=CASE_BOOTSTRAP_SEED
        ),
        "all_utc_anchor_day": _bootstrap(
            work, ["anchor_day_utc"], seed=DAY_BOOTSTRAP_SEED
        ),
        "active_complete_case": _bootstrap(
            active, ["anchor_id", "station"], seed=CASE_BOOTSTRAP_SEED + 10
        ),
        "active_utc_anchor_day": _bootstrap(
            active, ["anchor_day_utc"], seed=DAY_BOOTSTRAP_SEED + 10
        ),
    }
    critical: list[dict[str, Any]] = []
    for family in ("by_station", "by_lead", "by_horizon_group", "by_regime"):
        for name, values in active_breakdowns[family].items():
            if int(values["cases"]) >= 12:
                critical.append(
                    {
                        "family": family,
                        "slice": name,
                        "cases": int(values["cases"]),
                        "delta_m": float(values["delta_candidate_minus_incumbent_m"]),
                    }
                )
    worst = max((item["delta_m"] for item in critical), default=math.inf)
    overall = _metric(work)
    active_metric = _metric(active)
    active_fold_delta = {
        fold: values["delta_candidate_minus_incumbent_m"]
        for fold, values in active_breakdowns["by_fold"].items()
    }
    if set(active_fold_delta) != set(FOLD_ORDER[1:]):
        raise ContractError("active fold diagnostics are incomplete")
    active_day_ci = uncertainty["active_utc_anchor_day"][
        "delta_candidate_minus_incumbent_ci90_m"
    ]
    promote_checks = {
        "overall_delta_at_most_minus_0p003m": overall[
            "delta_candidate_minus_incumbent_m"
        ]
        <= PROMOTE_OVERALL_DELTA_M,
        "active_delta_at_most_minus_0p005m": active_metric[
            "delta_candidate_minus_incumbent_m"
        ]
        <= PROMOTE_ACTIVE_DELTA_M,
        "active_day_ci90_upper_below_zero": active_day_ci[1] < 0.0,
        "both_active_folds_improve": all(value < 0.0 for value in active_fold_delta.values()),
        "worst_eligible_slice_regression_at_most_0p015m": worst
        <= PROMOTE_MAX_SLICE_REGRESSION_M,
    }
    reject_checks = {
        "active_delta_nonnegative": active_metric["delta_candidate_minus_incumbent_m"] >= 0.0,
        "active_day_ci90_lower_positive": active_day_ci[0] > 0.0,
        "both_active_folds_nonimproving": all(
            value >= 0.0 for value in active_fold_delta.values()
        ),
        "worst_eligible_slice_regression_above_0p030m": worst
        > REJECT_MAX_SLICE_REGRESSION_M,
    }
    if all(promote_checks.values()):
        verdict = "PROMOTE"
    elif any(reject_checks.values()):
        verdict = "REJECT"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "overall": overall,
        "active_prequential_folds": active_metric,
        "exact_no_op_arm": {
            "definition": "frozen final_prediction on all 181 cases",
            "metrics": {
                "rmse_m": overall["incumbent_rmse_m"],
                "rows": overall["rows"],
                "cases": overall["cases"],
            },
        },
        "breakdowns": breakdowns,
        "active_breakdowns": active_breakdowns,
        "uncertainty": uncertainty,
        "critical_slices": critical,
        "worst_eligible_slice_regression_m": worst,
        "decision": {
            "verdict": verdict,
            "promote_checks": promote_checks,
            "reject_checks": reject_checks,
            "interpretation": (
                "One preregistered structural screen only; no coefficient, basis, gate, "
                "or slice was changed after scores were observed."
            ),
        },
    }


def _execute(root: Path, verified: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = root / OUTPUT_RELATIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    preregistration = _preregistration(root, verified)
    _write_exclusive(output / "preregistered_design.json", preregistration)

    frame, blocked_audit = _load_surface(root)
    prediction, fit_receipts, no_op_audit = _prequential_prediction(frame)
    frame["candidate_prediction"] = prediction
    evaluation = _evaluate(frame)
    metrics = {
        "schema_version": f"{SCHEMA_VERSION}.metrics",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": _now_kst(),
        "status": "ONE_SHOT_SCREEN_COMPLETE",
        "verdict": evaluation["decision"]["verdict"],
        "hypothesis_count": 1,
        "screen_count": 1,
        "fit_receipts": fit_receipts,
        "blocked_surface_audit": blocked_audit,
        "no_op_audit": no_op_audit,
        "evaluation": evaluation,
        "access_audit": {
            "pinned_training_or_oof_files_read": 3,
            "anonymous_evaluation_files_read": 0,
            "current_or_round_prediction_files_read": 0,
            "external_data_files_read": 0,
            "raw_rows_written": 0,
            "prediction_upload_files_created": 0,
            "uploads": 0,
        },
        "environment": _environment(),
        "git": _git_state(root),
    }
    _write_exclusive(output / "metrics.json", metrics)
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.manifest",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": _now_kst(),
        "verdict": metrics["verdict"],
        "implementation": {
            SCRIPT_RELATIVE.as_posix(): {
                "bytes": (root / SCRIPT_RELATIVE).stat().st_size,
                "sha256": _sha256(root / SCRIPT_RELATIVE),
            }
        },
        "inputs": verified,
        "outputs": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in ("preregistered_design.json", "metrics.json")
        },
        "no_row_level_artifact": True,
        "official_or_upload_artifact_created": False,
    }
    _write_exclusive(output / "manifest.json", manifest)
    return {
        "status": metrics["status"],
        "verdict": metrics["verdict"],
        "output": OUTPUT_RELATIVE.as_posix(),
        "pooled_delta_m": evaluation["overall"]["delta_candidate_minus_incumbent_m"],
        "active_delta_m": evaluation["active_prequential_folds"][
            "delta_candidate_minus_incumbent_m"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="metadata/hash preflight only")
    group.add_argument("--execute", action="store_true", help="run the one-shot local screen")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (root / SCRIPT_RELATIVE).resolve() != Path(__file__).resolve():
        raise ContractError("runner is not at its canonical repository-relative path")
    verified = _verify_inputs(root)
    output = root / OUTPUT_RELATIVE
    if output.exists():
        raise FileExistsError(f"one-shot output already exists: {OUTPUT_RELATIVE.as_posix()}")
    if args.check:
        print(
            json.dumps(
                {
                    "status": "CHECK_ONLY_PASS",
                    "experiment_id": EXPERIMENT_ID,
                    "pinned_inputs": len(verified),
                    "expected_cases": 181,
                    "expected_rows": 1086,
                    "output_absent": True,
                },
                sort_keys=True,
            )
        )
        return 0
    result = _execute(root, verified)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
