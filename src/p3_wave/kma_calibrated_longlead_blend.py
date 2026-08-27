"""Fixed KMA residual calibration and long-lead blend for P3 generation v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

LEADS = (3, 6, 9, 12, 18, 24)
ACTIVE_LEADS = (18, 24)
NO_OP_LEADS = (3, 6, 9, 12)
STATIONS = ("G-ORS", "I-ORS", "S-ORS")
ALPHA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4)
PAIR_KEYS = ("fold", "anchor_id", "station", "lead_h")
INNER_COLUMNS = (
    "fold",
    "anchor_id",
    "station",
    "lead_h",
    "current_hs",
    "target_hs",
    "source_prediction",
    "calibrated_source",
    "control_single_prediction",
    "control_final",
    "selected_alpha",
    "candidate_final",
)
OUTER_COLUMNS = (
    "fold",
    "anchor_id",
    "station",
    "lead_h",
    "current_hs",
    "incumbent_final",
    "source_prediction",
    "calibrated_source",
    "selected_alpha",
    "candidate_final",
)


class KMALongLeadError(ValueError):
    """Raised when the fixed v2 experiment contract is violated."""


@dataclass(frozen=True)
class RidgeAffineCalibrator:
    lead_h: int
    ridge_alpha: float
    fit_intercept: bool
    solver: str
    design_columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    fit_rows: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["design_columns"] = list(self.design_columns)
        result["coefficients"] = list(self.coefficients)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RidgeAffineCalibrator:
        return cls(
            lead_h=int(payload["lead_h"]),
            ridge_alpha=float(payload["ridge_alpha"]),
            fit_intercept=bool(payload["fit_intercept"]),
            solver=str(payload["solver"]),
            design_columns=tuple(str(item) for item in payload["design_columns"]),
            coefficients=tuple(float(item) for item in payload["coefficients"]),
            fit_rows=int(payload["fit_rows"]),
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_preregistration(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != "p3_kma_calibrated_longlead_blend_v2":
        raise KMALongLeadError("experiment id changed")
    source = config["source_reuse"]
    if source.get("fit_new_source_model") is not False:
        raise KMALongLeadError("v2 must reuse the sealed v1 source model")
    calibrator = config["calibrator"]
    expected_calibrator = {
        "family": "Ridge",
        "active_leads": list(ACTIVE_LEADS),
        "fit_intercept": False,
        "ridge_alpha": 10.0,
        "solver": "cholesky",
        "standardize": False,
        "hyperparameter_grid_size": 0,
        "period_features": 0,
        "missingness_features": 0,
    }
    for key, value in expected_calibrator.items():
        if calibrator.get(key) != value:
            raise KMALongLeadError(f"fixed calibrator contract changed: {key}")
    expected_design = ["source_residual", *[f"station_{station}" for station in STATIONS]]
    if calibrator.get("design_columns") != expected_design:
        raise KMALongLeadError("ridge design columns changed")
    blend = config["blend"]
    if tuple(float(value) for value in blend.get("alpha_grid", ())) != ALPHA_GRID:
        raise KMALongLeadError("alpha grid changed")
    if blend.get("active_leads") != list(ACTIVE_LEADS):
        raise KMALongLeadError("blend active leads changed")
    if blend.get("no_op_leads") != list(NO_OP_LEADS):
        raise KMALongLeadError("blend no-op leads changed")
    if blend.get("alpha_selected_separately_inside_each_outer_fold") is not True:
        raise KMALongLeadError("alpha must be selected fold-locally")
    if blend.get("fold_alpha_zero_is_an_honest_no_op_and_is_allowed") is not True:
        raise KMALongLeadError("fold-local alpha zero must remain allowed")
    if blend.get("deployment_alpha") != "median_of_three_inner_selected_fold_alphas":
        raise KMALongLeadError("deployment alpha rule changed")
    proxy = config["inner_control_proxy"]
    expected_proxy = {
        "base_feature_count": 591,
        "iterations": 700,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.2,
        "random_seed": 20260817,
        "loss_function": "RMSE",
        "task_type": "CPU",
        "persistence_shrink_active_leads": [12, 18, 24],
        "persistence_shrink_weight": 0.2,
        "model_or_iteration_search": 0,
    }
    for key, value in expected_proxy.items():
        if proxy.get(key) != value:
            raise KMALongLeadError(f"inner control proxy changed: {key}")
    validation = config["validation"]
    if validation.get("outer_membership") != "frozen_incumbent_oof_keys_only":
        raise KMALongLeadError("outer membership must remain frozen")
    if validation.get("embargo_hours") != 78 or validation.get("inner_validation_days") != 45:
        raise KMALongLeadError("fold timing contract changed")
    gate = validation["inner_gate"]
    if gate.get("minimum_pooled_full_six_lead_delta_m_exclusive") != 0.0:
        raise KMALongLeadError("inner pooled gate changed")
    if gate.get("minimum_strictly_improved_folds") != 2:
        raise KMALongLeadError("inner fold gate changed")
    if gate.get("alpha_zero_fold_allowed") is not True:
        raise KMALongLeadError("inner alpha-zero policy changed")
    rolling = validation["rolling_origin_label_scope"]
    expected_rolling = {
        "current_fold_validation_targets_excluded_from_that_fold_training_inner_fit_and_inner_calibration": True,
        "earlier_fold_validation_targets_allowed_only_as_later_fold_training_history": True,
        "future_fold_validation_targets_forbidden_from_earlier_fold_training": True,
        "global_process_level_zero_outer_target_exposure_before_blind_seal_claimed": False,
        "designated_scoring_read_occurs_after_global_blind_seal": True,
    }
    if rolling != expected_rolling:
        raise KMALongLeadError("rolling-origin label scope changed")
    if validation.get("one_shot_no_rerun") is not True:
        raise KMALongLeadError("one-shot rule changed")
    promotion = config["promotion_gate"]
    if promotion.get("applies_to") != "candidate_final_vs_exact_incumbent_prediction_only":
        raise KMALongLeadError("outer promotion comparison changed")
    execution = config["execution"]
    if execution.get("actual_authorized") is not False:
        raise KMALongLeadError("canonical config must remain dry-only")
    if execution.get("actual_authorization_mechanism") != (
        "separate_O_EXCL_amendment_bound_to_exact_dry_receipt_and_implementation_SHA"
    ):
        raise KMALongLeadError("authorization mechanism changed")
    prohibitions = config["prohibitions"]
    required_prohibitions = (
        "new_source_model_fit",
        "direct_source_target_row_concatenation",
        "period_or_missingness_features",
        "test_context_or_test_index_read",
        "outer_membership_regeneration",
        "current_or_future_fold_validation_target_in_that_fold_training_or_inner_selection",
        "router_or_incumbent_weight_reselection",
        "submission_write_or_upload",
        "v1_or_frozen_artifact_mutation",
    )
    if not all(prohibitions.get(name) is True for name in required_prohibitions):
        raise KMALongLeadError("a required prohibition is missing")


def load_preregistration(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_preregistration(config)
    return config


def _require_frame(frame: pd.DataFrame, columns: Sequence[str], *, role: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KMALongLeadError(f"{role} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise KMALongLeadError(f"{role} is empty")


def _design(frame: pd.DataFrame) -> np.ndarray:
    _require_frame(frame, ["station", "current_hs", "source_prediction"], role="ridge frame")
    station = frame["station"].astype(str)
    if not station.isin(STATIONS).all():
        raise KMALongLeadError("ridge frame contains an unknown station")
    source_residual = frame["source_prediction"].to_numpy(dtype=np.float64) - frame[
        "current_hs"
    ].to_numpy(dtype=np.float64)
    columns = [source_residual]
    columns.extend(station.eq(name).to_numpy(dtype=np.float64) for name in STATIONS)
    design = np.column_stack(columns)
    if not np.isfinite(design).all():
        raise KMALongLeadError("ridge design contains non-finite values")
    return design


def fit_ridge_affine(frame: pd.DataFrame, *, lead_h: int) -> RidgeAffineCalibrator:
    if int(lead_h) not in ACTIVE_LEADS:
        raise KMALongLeadError("calibrator may only fit 18h or 24h")
    _require_frame(
        frame,
        ["anchor_id", "lead_h", "station", "current_hs", "source_prediction", "target_hs"],
        role="ridge fit frame",
    )
    selected = frame.loc[frame["lead_h"].eq(int(lead_h))].copy()
    if selected.empty or selected["anchor_id"].duplicated().any():
        raise KMALongLeadError("ridge fit lead has empty or duplicate cases")
    design = _design(selected)
    target = selected["target_hs"].to_numpy(dtype=np.float64) - selected["current_hs"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(target).all():
        raise KMALongLeadError("ridge target contains non-finite values")
    model = Ridge(alpha=10.0, fit_intercept=False, solver="cholesky")
    model.fit(design, target)
    coefficients = tuple(float(value) for value in np.asarray(model.coef_).reshape(-1))
    if len(coefficients) != 4 or not np.isfinite(coefficients).all():
        raise KMALongLeadError("ridge coefficients are invalid")
    return RidgeAffineCalibrator(
        lead_h=int(lead_h),
        ridge_alpha=10.0,
        fit_intercept=False,
        solver="cholesky",
        design_columns=("source_residual", *[f"station_{station}" for station in STATIONS]),
        coefficients=coefficients,
        fit_rows=int(len(selected)),
    )


def fit_ridge_pair(frame: pd.DataFrame) -> dict[int, RidgeAffineCalibrator]:
    return {lead: fit_ridge_affine(frame, lead_h=lead) for lead in ACTIVE_LEADS}


def apply_ridge_affine(frame: pd.DataFrame, calibrator: RidgeAffineCalibrator) -> np.ndarray:
    if calibrator.lead_h not in ACTIVE_LEADS:
        raise KMALongLeadError("stored calibrator lead is invalid")
    expected_columns = ("source_residual", *[f"station_{station}" for station in STATIONS])
    if calibrator.design_columns != expected_columns:
        raise KMALongLeadError("stored calibrator design changed")
    if (
        calibrator.ridge_alpha != 10.0
        or calibrator.fit_intercept
        or calibrator.solver != "cholesky"
    ):
        raise KMALongLeadError("stored calibrator parameters changed")
    design = _design(frame)
    coefficient = np.asarray(calibrator.coefficients, dtype=np.float64)
    residual = design @ coefficient
    prediction = frame["current_hs"].to_numpy(dtype=np.float64) + residual
    prediction = np.clip(prediction, 0.0, 30.0)
    if not np.isfinite(prediction).all():
        raise KMALongLeadError("calibrated source prediction is non-finite")
    return prediction


def add_calibrated_source(
    frame: pd.DataFrame, calibrators: Mapping[int, RidgeAffineCalibrator]
) -> pd.DataFrame:
    _require_frame(
        frame,
        ["lead_h", "station", "current_hs", "source_prediction"],
        role="calibration apply frame",
    )
    if set(calibrators) != set(ACTIVE_LEADS):
        raise KMALongLeadError("both fixed long-lead calibrators are required")
    result = frame.copy()
    result["calibrated_source"] = result["source_prediction"].to_numpy(dtype=np.float64)
    for lead in ACTIVE_LEADS:
        mask = result["lead_h"].eq(lead)
        if not mask.any():
            raise KMALongLeadError(f"calibration frame lacks lead {lead}")
        result.loc[mask, "calibrated_source"] = apply_ridge_affine(
            result.loc[mask], calibrators[lead]
        )
    return result


def apply_fixed_control_shrink(
    control_single: np.ndarray, current_hs: np.ndarray, lead_h: np.ndarray
) -> np.ndarray:
    single = np.asarray(control_single, dtype=np.float64)
    current = np.asarray(current_hs, dtype=np.float64)
    leads = np.asarray(lead_h, dtype=np.int64)
    if single.shape != current.shape or single.shape != leads.shape:
        raise KMALongLeadError("control shrink arrays differ in shape")
    weight = np.where(np.isin(leads, [12, 18, 24]), 0.2, 0.0)
    prediction = (1.0 - weight) * single + weight * current
    if not np.isfinite(prediction).all():
        raise KMALongLeadError("control proxy contains non-finite values")
    return prediction


def blend_long_leads(
    control_final: np.ndarray,
    calibrated_source: np.ndarray,
    lead_h: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    if float(alpha) not in ALPHA_GRID:
        raise KMALongLeadError("blend alpha is outside the frozen grid")
    control = np.asarray(control_final, dtype=np.float64)
    source = np.asarray(calibrated_source, dtype=np.float64)
    leads = np.asarray(lead_h, dtype=np.int64)
    if control.shape != source.shape or control.shape != leads.shape:
        raise KMALongLeadError("blend arrays differ in shape")
    if not np.isin(leads, LEADS).all():
        raise KMALongLeadError("blend frame contains an unknown lead")
    candidate = control.copy()
    if float(alpha) != 0.0:
        active = np.isin(leads, ACTIVE_LEADS)
        candidate[active] = (1.0 - float(alpha)) * control[active] + float(alpha) * source[active]
    if not np.array_equal(
        candidate[np.isin(leads, NO_OP_LEADS)], control[np.isin(leads, NO_OP_LEADS)]
    ):
        raise AssertionError("short-lead no-op is not byte exact")
    if not np.isfinite(candidate).all() or (candidate < 0.0).any() or (candidate > 30.0).any():
        raise KMALongLeadError("blended prediction is invalid")
    return candidate


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(prediction_array - truth_array))))


def select_fold_alpha(inner: pd.DataFrame) -> tuple[float, pd.DataFrame, list[dict[str, float]]]:
    _require_frame(
        inner,
        [*PAIR_KEYS, "target_hs", "control_final", "calibrated_source"],
        role="inner alpha frame",
    )
    if inner["fold"].nunique() != 1:
        raise KMALongLeadError("alpha selection must receive exactly one outer fold")
    if inner.duplicated(list(PAIR_KEYS)).any():
        raise KMALongLeadError("inner alpha frame contains duplicate keys")
    truth = inner["target_hs"].to_numpy(dtype=np.float64)
    control = inner["control_final"].to_numpy(dtype=np.float64)
    source = inner["calibrated_source"].to_numpy(dtype=np.float64)
    leads = inner["lead_h"].to_numpy(dtype=np.int64)
    scores: list[dict[str, float]] = []
    predictions: dict[float, np.ndarray] = {}
    for alpha in ALPHA_GRID:
        candidate = blend_long_leads(control, source, leads, alpha=alpha)
        predictions[alpha] = candidate
        scores.append({"alpha": alpha, "full_six_lead_rmse": _rmse(truth, candidate)})
    best_rmse = min(row["full_six_lead_rmse"] for row in scores)
    selected = next(
        row
        for row in scores
        if np.isclose(row["full_six_lead_rmse"], best_rmse, rtol=0.0, atol=1e-15)
    )
    alpha = float(selected["alpha"])
    result = inner.copy()
    result["selected_alpha"] = alpha
    result["candidate_final"] = predictions[alpha]
    if alpha == 0.0 and not np.array_equal(
        result["candidate_final"].to_numpy(dtype=np.float64), control
    ):
        raise AssertionError("alpha-zero fold is not an exact no-op")
    return alpha, result, scores


def evaluate_inner_gate(selected: pd.DataFrame) -> dict[str, Any]:
    _require_frame(
        selected,
        [
            "fold",
            "anchor_id",
            "lead_h",
            "target_hs",
            "control_final",
            "candidate_final",
            "selected_alpha",
        ],
        role="selected inner frame",
    )
    if selected["fold"].nunique() != 3:
        raise KMALongLeadError("inner gate requires exactly three outer-fold blocks")
    fold_alphas = selected.groupby("fold", observed=True)["selected_alpha"].nunique()
    if not fold_alphas.eq(1).all():
        raise KMALongLeadError("each fold must have exactly one selected alpha")
    control = _rmse(selected["target_hs"], selected["control_final"])
    candidate = _rmse(selected["target_hs"], selected["candidate_final"])
    delta_by_fold = {
        str(name): _rmse(group["target_hs"], group["candidate_final"])
        - _rmse(group["target_hs"], group["control_final"])
        for name, group in selected.groupby("fold", observed=True)
    }
    improved = int(sum(value < 0.0 for value in delta_by_fold.values()))
    alphas = {
        str(name): float(group["selected_alpha"].iloc[0])
        for name, group in selected.groupby("fold", observed=True)
    }
    deployment = float(np.median(np.asarray(list(alphas.values()), dtype=np.float64)))
    delta = float(candidate - control)
    checks = {
        "pooled_full_six_lead_delta_below_zero": delta < 0.0,
        "minimum_two_strictly_improved_folds": improved >= 2,
    }
    return {
        "pooled_control_rmse": control,
        "pooled_candidate_rmse": candidate,
        "pooled_delta_rmse": delta,
        "delta_by_fold": delta_by_fold,
        "strictly_improved_folds": improved,
        "selected_alpha_by_fold": alphas,
        "alpha_zero_folds": sorted(name for name, value in alphas.items() if value == 0.0),
        "deployment_alpha_median": deployment,
        "checks": checks,
        "pass": all(checks.values()),
    }


def validate_inner_predictions(frame: pd.DataFrame) -> dict[str, int]:
    if list(frame.columns) != list(INNER_COLUMNS):
        raise KMALongLeadError("inner prediction schema changed")
    if frame.duplicated(list(PAIR_KEYS)).any():
        raise KMALongLeadError("inner prediction keys are duplicated")
    numeric = frame[
        [
            "current_hs",
            "target_hs",
            "source_prediction",
            "calibrated_source",
            "control_single_prediction",
            "control_final",
            "selected_alpha",
            "candidate_final",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise KMALongLeadError("inner prediction contains non-finite values")
    lead_sets = frame.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if not lead_sets.map(lambda values: values == LEADS).all():
        raise KMALongLeadError("inner prediction case does not contain all six leads")
    for fold, group in frame.groupby("fold", observed=True):
        alpha_values = group["selected_alpha"].unique()
        if len(alpha_values) != 1 or float(alpha_values[0]) not in ALPHA_GRID:
            raise KMALongLeadError(f"inner fold alpha is invalid: {fold}")
        expected = blend_long_leads(
            group["control_final"].to_numpy(dtype=np.float64),
            group["calibrated_source"].to_numpy(dtype=np.float64),
            group["lead_h"].to_numpy(dtype=np.int64),
            alpha=float(alpha_values[0]),
        )
        if not np.array_equal(group["candidate_final"].to_numpy(dtype=np.float64), expected):
            raise KMALongLeadError("inner candidate does not reconstruct from sealed inputs")
    return {"rows": int(len(frame)), "cases": int(frame["anchor_id"].nunique())}


def validate_outer_blind(frame: pd.DataFrame) -> dict[str, int]:
    if "target_hs" in frame.columns:
        raise KMALongLeadError("outer truth leaked into blind prediction")
    if list(frame.columns) != list(OUTER_COLUMNS):
        raise KMALongLeadError("outer blind schema changed")
    if frame.duplicated(list(PAIR_KEYS)).any():
        raise KMALongLeadError("outer blind keys are duplicated")
    numeric = frame[
        [
            "current_hs",
            "incumbent_final",
            "source_prediction",
            "calibrated_source",
            "selected_alpha",
            "candidate_final",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise KMALongLeadError("outer blind prediction contains non-finite values")
    lead_sets = frame.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if not lead_sets.map(lambda values: values == LEADS).all():
        raise KMALongLeadError("outer blind case does not contain all six leads")
    for fold, group in frame.groupby("fold", observed=True):
        alpha_values = group["selected_alpha"].unique()
        if len(alpha_values) != 1 or float(alpha_values[0]) not in ALPHA_GRID:
            raise KMALongLeadError(f"outer fold alpha is invalid: {fold}")
        expected = blend_long_leads(
            group["incumbent_final"].to_numpy(dtype=np.float64),
            group["calibrated_source"].to_numpy(dtype=np.float64),
            group["lead_h"].to_numpy(dtype=np.int64),
            alpha=float(alpha_values[0]),
        )
        if not np.array_equal(group["candidate_final"].to_numpy(dtype=np.float64), expected):
            raise KMALongLeadError("outer candidate does not reconstruct from sealed inputs")
    return {"rows": int(len(frame)), "cases": int(frame["anchor_id"].nunique())}


def _metric_slices(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rmse": _rmse(frame["target_hs"], frame[prediction_column]),
        "rows": int(len(frame)),
    }
    for group_column, output_name in (
        ("fold", "by_fold"),
        ("station", "by_station"),
        ("lead_h", "by_lead"),
    ):
        result[output_name] = {
            str(int(key)) if group_column == "lead_h" else str(key): _rmse(
                group["target_hs"], group[prediction_column]
            )
            for key, group in frame.groupby(group_column, observed=True)
        }
    return result


def _paired_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    grouped = list(frame.groupby(["fold", "anchor_id"], observed=True, sort=True))
    if not grouped:
        raise KMALongLeadError("outer bootstrap received no cases")
    incumbent_sse = np.asarray(
        [np.square(group["incumbent_final"] - group["target_hs"]).sum() for _, group in grouped],
        dtype=np.float64,
    )
    candidate_sse = np.asarray(
        [np.square(group["candidate_final"] - group["target_hs"]).sum() for _, group in grouped],
        dtype=np.float64,
    )
    counts = np.asarray([len(group) for _, group in grouped], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    delta = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        draw = rng.integers(0, len(grouped), size=len(grouped))
        denominator = counts[draw].sum()
        incumbent = np.sqrt(incumbent_sse[draw].sum() / denominator)
        candidate = np.sqrt(candidate_sse[draw].sum() / denominator)
        delta[index] = candidate - incumbent
    return {
        "replicates": int(replicates),
        "case_count": int(len(grouped)),
        "delta_rmse_mean": float(delta.mean()),
        "ci90_lower": float(np.quantile(delta, 0.05)),
        "ci90_upper": float(np.quantile(delta, 0.95)),
    }


def evaluate_outer_promotion(frame: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    _require_frame(
        frame,
        [*OUTER_COLUMNS, "target_hs"],
        role="outer evaluated frame",
    )
    incumbent = _metric_slices(frame, "incumbent_final")
    candidate = _metric_slices(frame, "candidate_final")
    delta = float(candidate["rmse"] - incumbent["rmse"])
    by_fold = {
        key: float(candidate["by_fold"][key] - incumbent["by_fold"][key])
        for key in incumbent["by_fold"]
    }
    by_station = {
        key: float(candidate["by_station"][key] - incumbent["by_station"][key])
        for key in incumbent["by_station"]
    }
    by_lead = {
        key: float(candidate["by_lead"][key] - incumbent["by_lead"][key])
        for key in incumbent["by_lead"]
    }
    validation = config["validation"]
    bootstrap = _paired_bootstrap(
        frame,
        replicates=int(validation["bootstrap_replicates"]),
        seed=int(validation["bootstrap_seed"]),
    )
    gate = config["promotion_gate"]
    checks = {
        "pooled_rmse_improvement": delta <= -float(gate["minimum_pooled_rmse_improvement_meters"]),
        "bootstrap_ci90_upper_below_zero": bootstrap["ci90_upper"] < 0.0,
        "minimum_improved_outer_folds": sum(value < 0.0 for value in by_fold.values())
        >= int(gate["minimum_improved_outer_folds"]),
        "station_degradation_limit": max(by_station.values())
        <= float(gate["maximum_any_station_rmse_degradation_meters"]),
        "lead_degradation_limit": max(by_lead.values())
        <= float(gate["maximum_any_lead_rmse_degradation_meters"]),
        "lead_18_non_degrading": by_lead["18"] <= 0.0,
        "lead_24_non_degrading": by_lead["24"] <= 0.0,
    }
    return {
        "incumbent": incumbent,
        "candidate": candidate,
        "candidate_minus_incumbent_rmse": delta,
        "delta_by_fold": by_fold,
        "delta_by_station": by_station,
        "delta_by_lead": by_lead,
        "paired_case_bootstrap": bootstrap,
        "checks": checks,
        "decision": "GO_TO_INTEGRATION" if all(checks.values()) else "NO_GO_EXACT_INCUMBENT",
    }


__all__ = [
    "ACTIVE_LEADS",
    "ALPHA_GRID",
    "INNER_COLUMNS",
    "KMALongLeadError",
    "LEADS",
    "NO_OP_LEADS",
    "OUTER_COLUMNS",
    "PAIR_KEYS",
    "RidgeAffineCalibrator",
    "STATIONS",
    "add_calibrated_source",
    "apply_fixed_control_shrink",
    "apply_ridge_affine",
    "blend_long_leads",
    "evaluate_inner_gate",
    "evaluate_outer_promotion",
    "fit_ridge_affine",
    "fit_ridge_pair",
    "load_preregistration",
    "select_fold_alpha",
    "sha256_file",
    "validate_inner_predictions",
    "validate_outer_blind",
    "validate_preregistration",
]
