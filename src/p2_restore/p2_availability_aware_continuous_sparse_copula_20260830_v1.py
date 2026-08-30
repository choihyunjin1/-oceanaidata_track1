"""Sealed P2 availability-aware continuous sparse Gaussian-rank experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, BinaryIO

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from scripts import run_p2_state_conditioned_copula_20260830_v1 as legacy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "p2_availability_aware_continuous_sparse_copula_20260830_v1"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RUNNER_PATH = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
SEALED_CONFIG_CANONICAL_SHA256 = "e501fc89433061c13ef05b8157a6e1a9186dd0aee7841a7f538eb4bfae55bc41"
KST = "Asia/Seoul"
CONDITIONERS = (
    "temp_contrast_signed",
    "psal_contrast_signed",
    "thermal_change_24h_signed",
)
RESPONSES = ("residual_l2", "residual_l3", "residual_l4")
STATE_FEATURES = (
    "thermal_contrast_abs",
    "thermal_change_12h_signed",
    "thermal_change_24h_signed",
)
EXPOSED_EDGES = (
    "temp_contrast_signed__residual_l2",
    "temp_contrast_signed__residual_l3",
    "temp_contrast_signed__residual_l4",
    "psal_contrast_signed__residual_l4",
    "thermal_change_24h_signed__residual_l4",
    "residual_l2__residual_l4",
    "residual_l3__residual_l4",
)
VARIABLES = CONDITIONERS + RESPONSES


class ExperimentContractError(RuntimeError):
    """Raised when the sealed experiment contract is violated."""


class ModelGuardError(ExperimentContractError):
    """Raised when the continuous dependence model must be an exact no-op."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(payload)


class SourceAccessLedger:
    """Open only observations.csv below the explicit source directory."""

    def __init__(self, root: Path, allowed_basenames: list[str]) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ExperimentContractError("--p2-dir must resolve to a directory")
        self.allowed = frozenset(allowed_basenames)
        if self.allowed != {"observations.csv"}:
            raise ExperimentContractError("source allowlist must contain only observations.csv")
        self.open_counts = {"observations.csv": 0}

    def open_binary(self, basename: str) -> BinaryIO:
        if basename not in self.allowed:
            raise ExperimentContractError(f"source basename is not allowlisted: {basename}")
        path = (self.root / basename).resolve(strict=True)
        if path.parent != self.root or path.name != basename or not path.is_file():
            raise ExperimentContractError(f"source escaped --p2-dir: {basename}")
        self.open_counts[basename] += 1
        return path.open("rb")


def _validate_pinned_path(path: Path, record: dict[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or _sha256_file(path) != record["sha256"]
    ):
        raise ExperimentContractError(f"immutable input changed: {path}")


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if _canonical_sha256(config) != SEALED_CONFIG_CANONICAL_SHA256:
        raise ExperimentContractError("preregistered config changed")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ExperimentContractError("experiment id changed")
    if tuple(config["dependence"]["stage0_exposed_edges"]) != EXPOSED_EDGES:
        raise ExperimentContractError("Stage-0 exposed edge set changed")
    if tuple(config["dependence"]["conditioners"]) != CONDITIONERS:
        raise ExperimentContractError("conditioner set changed")
    if tuple(config["dependence"]["responses"]) != RESPONSES:
        raise ExperimentContractError("response set changed")
    if tuple(config["state"]["continuous_features"]) != STATE_FEATURES:
        raise ExperimentContractError("continuous state basis changed")
    if float(config["dependence"]["diagonal_shrinkage"]) != 0.8:
        raise ExperimentContractError("fixed diagonal shrinkage changed")
    if float(config["dependence"]["continuous_state_slope_ridge"]) != 1.0:
        raise ExperimentContractError("fixed state ridge changed")
    if config["dependence"]["ridge_selected_or_tuned"]:
        raise ExperimentContractError("state ridge selection was enabled")
    if config["dependence"]["shrinkage_selected_or_tuned"]:
        raise ExperimentContractError("dependence shrinkage selection was enabled")
    if config["dependence"]["nearest_psd_projection_allowed"]:
        raise ExperimentContractError("nearest-PSD repair was enabled")
    closed = config["closed_family_exclusion"]
    if any(value for key, value in closed.items() if key != "closed_experiment_id"):
        raise ExperimentContractError("closed-family exclusion changed")
    policy = config["execution_policy"]
    forbidden = (
        policy["result_based_tuning"],
        policy["result_based_retry"],
        policy["technical_failure_retry"],
        policy["official_interface_reads_allowed"],
        policy["query_support_reads_allowed"],
        policy["csv_output_allowed"],
        policy["submission_generation_allowed"],
        policy["upload_allowed"],
    )
    if any(forbidden) or int(policy["maximum_executions"]) != 1:
        raise ExperimentContractError("execution policy changed")
    if not policy["real_training_execution_authorized"] or not policy["aggregate_json_only"]:
        raise ExperimentContractError("one-shot aggregate execution is not authorized")
    decision = config["primary_decision"]
    if any(
        decision[name]
        for name in (
            "minimum_improved_windows_is_hard_veto",
            "worst_season_cap_is_hard_veto",
            "all_layers_nonworse_is_hard_veto",
            "support_is_hard_veto_after_level0_validity",
            "correction_magnitude_is_hard_veto",
        )
    ):
        raise ExperimentContractError("a diagnostic slice was promoted to a hard veto")
    resources = config["resource_contract"]
    if (
        int(resources["outer_dependence_model_fits"]) != 3
        or int(resources["inner_selection_fits"]) != 0
        or int(resources["hpo_trials"]) != 0
        or int(resources["continuous_edge_estimations"]) != 21
    ):
        raise ExperimentContractError("fit or search budget changed")
    correction = config["correction"]
    if (
        float(correction["structural_minimum_c"]) != -0.2
        or float(correction["structural_maximum_c"]) != 0.2
        or correction["magnitude_is_promotion_veto"]
    ):
        raise ExperimentContractError("structural correction contract changed")
    _validate_pinned_path(ROOT / config["governing_policy"]["path"], config["governing_policy"])
    for record in config["immutable_training_inputs"].values():
        _validate_pinned_path(ROOT / record["path"], record)
    for relative, record in config["code_lineage"].items():
        _validate_pinned_path(ROOT / relative, record)
    governing = json.loads(
        (ROOT / config["governing_policy"]["path"]).read_text(encoding="utf-8")
    )
    if governing.get("status") != "GOVERNING_FUTURE_RESEARCH_POLICY_NO_OFFICIAL_ACTION":
        raise ExperimentContractError("governing policy status changed")
    stage0_record = config["immutable_training_inputs"]["stage0_result"]
    stage0_result = json.loads((ROOT / stage0_record["path"]).read_text(encoding="utf-8"))
    passing = tuple(stage0_result["kendall_heterogeneity"]["passing_edges"])
    if stage0_result.get("status") != "TRAIN_ONLY_ZERO_FIT_PREFLIGHT_PASS":
        raise ExperimentContractError("Stage-0 no longer passes")
    if set(passing) != set(EXPOSED_EDGES) or len(passing) != len(EXPOSED_EDGES):
        raise ExperimentContractError("Stage-0 passing edges changed")
    return config


def _read_training_source(
    p2_dir: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], SourceAccessLedger]:
    source = config["source_contract"]
    access = SourceAccessLedger(p2_dir, source["allowed_basenames"])
    with access.open_binary("observations.csv") as handle:
        payload = handle.read()
    record = source["observations"]
    if len(payload) != int(record["bytes"]) or _sha256_bytes(payload) != record["sha256"]:
        raise ExperimentContractError("source pin changed: observations.csv")
    frame = pd.read_csv(
        io.BytesIO(payload),
        dtype={"station": "string", "time": "string"},
    )
    if list(frame.columns) != source["observation_schema"]:
        raise ExperimentContractError("observations.csv schema changed")
    frame["time"] = pd.to_datetime(frame["time"], errors="raise", utc=True)
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    for coordinate in ("temp", "psal", "depth", "nominal_depth"):
        frame[coordinate] = pd.to_numeric(frame[coordinate], errors="coerce")
    if frame.duplicated(["station", "time", "layer"]).any():
        raise ExperimentContractError("observation keys duplicate")
    receipt = {
        "basename": "observations.csv",
        "rows": int(len(frame)),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }
    return frame, receipt, access


def _public_state_table(observations: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    layers = list(map(int, config["state"]["public_endpoint_layers"]))
    source = observations.loc[
        observations["layer"].isin(layers),
        ["station", "time", "layer", "temp", "psal"],
    ]
    wide = source.pivot(
        index=["station", "time"],
        columns="layer",
        values=["temp", "psal"],
    ).sort_index()
    profile = wide.index.to_frame(index=False)
    for coordinate in ("temp", "psal"):
        for layer in layers:
            key = (coordinate, layer)
            profile[f"{coordinate}_l{layer}"] = (
                pd.to_numeric(wide[key], errors="coerce").to_numpy()
                if key in wide.columns
                else np.nan
            )
    top, bottom = layers
    profile["temp_contrast_signed"] = profile[f"temp_l{top}"] - profile[f"temp_l{bottom}"]
    profile["psal_contrast_signed"] = profile[f"psal_l{top}"] - profile[f"psal_l{bottom}"]
    profile["thermal_contrast_abs"] = np.abs(profile["temp_contrast_signed"])
    for hours in map(int, config["state"]["lag_hours"]):
        lag_name = f"thermal_contrast_abs_lag_{hours}h"
        lag = profile[["station", "time", "thermal_contrast_abs"]].copy()
        lag["time"] = lag["time"] + pd.Timedelta(hours=hours)
        lag = lag.rename(columns={"thermal_contrast_abs": lag_name})
        profile = profile.merge(lag, on=["station", "time"], how="left", validate="one_to_one")
        profile[f"thermal_change_{hours}h_signed"] = (
            profile["thermal_contrast_abs"] - profile[lag_name]
        )
        profile = profile.drop(columns=lag_name)
    profile["thermal_contrast_missing"] = ~np.isfinite(profile["thermal_contrast_abs"])
    for hours in map(int, config["state"]["lag_hours"]):
        profile[f"thermal_change_{hours}h_missing"] = ~np.isfinite(
            profile[f"thermal_change_{hours}h_signed"]
        )
    profile["conditioner_missing"] = ~np.isfinite(
        profile[list(CONDITIONERS)].to_numpy(dtype=np.float64)
    ).all(axis=1)
    return profile


def _robust_state_parameters(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    values = frame[list(STATE_FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ModelGuardError("state normalization received nonfinite values")
    center = np.median(values, axis=0)
    scale = (np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)) / (
        1.3489795003921634
    )
    if (scale < float(config["state"]["minimum_robust_scale"])).any():
        raise ModelGuardError("continuous state scale collapsed")
    return center, scale


def _state_design(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(values, dtype=np.float64) - center) / scale
    return np.column_stack([np.ones(len(standardized)), standardized])


@dataclass(frozen=True)
class PredictionResult:
    correction: np.ndarray
    active: np.ndarray
    missing: np.ndarray
    ood: np.ndarray
    matrix_guard: np.ndarray


@dataclass(frozen=True)
class ContinuousSparseModel:
    x_support: tuple[np.ndarray, ...]
    state_center: np.ndarray
    state_scale: np.ndarray
    conditioner_lower: np.ndarray
    conditioner_upper: np.ndarray
    state_lower: np.ndarray
    state_upper: np.ndarray
    edge_coefficients: np.ndarray
    nuisance_correlations: np.ndarray
    response_scale: np.ndarray
    raw_edge_clip: float
    diagonal_shrinkage: float
    maximum_absolute_latent_mean: float
    minimum_eigenvalue: float
    maximum_condition_number: float
    receipt: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> PredictionResult:
        x = frame[list(CONDITIONERS)].to_numpy(dtype=np.float64)
        state = frame[list(STATE_FEATURES)].to_numpy(dtype=np.float64)
        finite = np.isfinite(x).all(axis=1) & np.isfinite(state).all(axis=1)
        in_bounds = (
            (x >= self.conditioner_lower).all(axis=1)
            & (x <= self.conditioner_upper).all(axis=1)
            & (state >= self.state_lower).all(axis=1)
            & (state <= self.state_upper).all(axis=1)
        )
        missing = ~finite
        ood = finite & ~in_bounds
        eligible = finite & in_bounds
        active = np.zeros(len(frame), dtype=bool)
        matrix_guard = np.zeros(len(frame), dtype=bool)
        correction = np.zeros((len(frame), len(RESPONSES)), dtype=np.float64)
        if not eligible.any():
            return PredictionResult(correction, active, missing, ood, matrix_guard)
        selected = np.flatnonzero(eligible)
        x_selected = x[selected]
        state_selected = state[selected]
        x_score = np.column_stack(
            [
                legacy._empirical_to_normal(support, x_selected[:, column])
                for column, support in enumerate(self.x_support)
            ]
        )
        design = _state_design(state_selected, self.state_center, self.state_scale)
        raw_edges = np.clip(
            design @ self.edge_coefficients.T,
            -self.raw_edge_clip,
            self.raw_edge_clip,
        )
        edge_values = (1.0 - self.diagonal_shrinkage) * raw_edges
        matrices = np.broadcast_to(
            np.eye(len(VARIABLES), dtype=np.float64),
            (len(selected), len(VARIABLES), len(VARIABLES)),
        ).copy()
        variable_index = {name: index for index, name in enumerate(VARIABLES)}
        for left_index, right_index in combinations(range(len(CONDITIONERS)), 2):
            value = self.nuisance_correlations[left_index, right_index]
            matrices[:, left_index, right_index] = value
            matrices[:, right_index, left_index] = value
        for edge_index, edge in enumerate(EXPOSED_EDGES):
            left, right = edge.split("__", maxsplit=1)
            left_index, right_index = variable_index[left], variable_index[right]
            matrices[:, left_index, right_index] = edge_values[:, edge_index]
            matrices[:, right_index, left_index] = edge_values[:, edge_index]
        eigenvalues = np.linalg.eigvalsh(matrices)
        condition_numbers = eigenvalues[:, -1] / eigenvalues[:, 0]
        valid = (eigenvalues[:, 0] >= self.minimum_eigenvalue) & (
            condition_numbers <= self.maximum_condition_number
        )
        matrix_guard[selected[~valid]] = True
        if valid.any():
            valid_selected = selected[valid]
            valid_matrices = matrices[valid]
            sigma_xx = valid_matrices[:, : len(CONDITIONERS), : len(CONDITIONERS)]
            sigma_yx = valid_matrices[:, len(CONDITIONERS) :, : len(CONDITIONERS)]
            solved = np.linalg.solve(sigma_xx, np.swapaxes(sigma_yx, 1, 2))
            beta = np.swapaxes(solved, 1, 2)
            latent = np.einsum("nij,nj->ni", beta, x_score[valid])
            latent = np.clip(
                latent,
                -self.maximum_absolute_latent_mean,
                self.maximum_absolute_latent_mean,
            )
            correction[valid_selected] = latent * self.response_scale
            active[valid_selected] = True
        return PredictionResult(correction, active, missing, ood, matrix_guard)


def _training_profiles(
    training: pd.DataFrame, state_table: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = training[["station", "time", "block"]].drop_duplicates()
    if keys.duplicated(["station", "time"]).any():
        raise ExperimentContractError("training profile block keys duplicate")
    state = keys.merge(state_table, on=["station", "time"], how="left", validate="one_to_one")
    response = training.pivot(
        index=["station", "time"], columns="layer", values="residual"
    ).reindex(columns=legacy.TARGET_LAYERS)
    response.columns = list(RESPONSES)
    profiles = state.merge(
        response.reset_index(), on=["station", "time"], how="left", validate="one_to_one"
    )
    required = list(CONDITIONERS + STATE_FEATURES + RESPONSES)
    complete = np.isfinite(profiles[required].to_numpy(dtype=np.float64)).all(axis=1)
    profiles["kst_day"] = profiles["time"].dt.tz_convert(KST).dt.strftime("%Y-%m-%d")
    missing_receipt = {
        "candidate_training_profiles": int(len(profiles)),
        "complete_training_profiles": int(complete.sum()),
        "missing_training_profiles": int((~complete).sum()),
        "missing_by_mask": {
            name: int(profiles[name].fillna(True).astype(bool).sum())
            for name in (
                "thermal_contrast_missing",
                "thermal_change_12h_missing",
                "thermal_change_24h_missing",
                "conditioner_missing",
            )
        },
    }
    return profiles.loc[complete].copy(), missing_receipt


def _fit_continuous_model(
    profiles: pd.DataFrame, config: dict[str, Any]
) -> ContinuousSparseModel:
    values = profiles[list(VARIABLES)].to_numpy(dtype=np.float64)
    state_values = profiles[list(STATE_FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(state_values).all():
        raise ModelGuardError("continuous model fit received nonfinite values")
    center, scale = _robust_state_parameters(profiles, config)
    design = _state_design(state_values, center, scale)
    normal_scores = np.empty_like(values)
    supports: list[np.ndarray] = []
    for column in range(values.shape[1]):
        support = np.sort(values[:, column])
        supports.append(support)
        normal_scores[:, column] = legacy._empirical_to_normal(support, values[:, column])
    ridge = float(config["dependence"]["continuous_state_slope_ridge"])
    penalty = np.diag([0.0, ridge, ridge, ridge])
    gram = design.T @ design / len(design) + penalty
    variable_index = {name: index for index, name in enumerate(VARIABLES)}
    coefficients = np.zeros((len(EXPOSED_EDGES), design.shape[1]), dtype=np.float64)
    edge_receipts: dict[str, Any] = {}
    for edge_index, edge in enumerate(EXPOSED_EDGES):
        left, right = edge.split("__", maxsplit=1)
        target = normal_scores[:, variable_index[left]] * normal_scores[:, variable_index[right]]
        coefficient = np.linalg.solve(gram, design.T @ target / len(design))
        coefficients[edge_index] = coefficient
        fitted = design @ coefficient
        edge_receipts[edge] = {
            "coefficients": coefficient.tolist(),
            "raw_fitted_min": float(fitted.min()),
            "raw_fitted_max": float(fitted.max()),
        }
    shrinkage = float(config["dependence"]["diagonal_shrinkage"])
    nuisance = np.eye(len(CONDITIONERS), dtype=np.float64)
    nuisance_receipts: dict[str, Any] = {}
    for left_index, right_index in combinations(range(len(CONDITIONERS)), 2):
        tau, latent = legacy._latent_correlation(
            values[:, left_index], values[:, right_index]
        )
        shrunk = (1.0 - shrinkage) * latent
        nuisance[left_index, right_index] = shrunk
        nuisance[right_index, left_index] = shrunk
        edge = f"{CONDITIONERS[left_index]}__{CONDITIONERS[right_index]}"
        nuisance_receipts[edge] = {
            "kendall_tau_b": tau,
            "latent_correlation": latent,
            "shrunk_correlation": shrunk,
        }
    response_values = values[:, len(CONDITIONERS) :]
    response_scale = (
        np.quantile(response_values, 0.75, axis=0)
        - np.quantile(response_values, 0.25, axis=0)
    ) / 1.3489795003921634
    if (response_scale < float(config["dependence"]["minimum_response_scale_c"])).any():
        raise ModelGuardError("residual response scale collapsed")
    lower_quantile = float(config["state"]["ood_lower_quantile"])
    upper_quantile = float(config["state"]["ood_upper_quantile"])
    conditioner_lower = np.quantile(values[:, : len(CONDITIONERS)], lower_quantile, axis=0)
    conditioner_upper = np.quantile(values[:, : len(CONDITIONERS)], upper_quantile, axis=0)
    state_lower = np.quantile(state_values, lower_quantile, axis=0)
    state_upper = np.quantile(state_values, upper_quantile, axis=0)
    receipt = {
        "profiles": int(len(profiles)),
        "kst_days": int(profiles["kst_day"].nunique()),
        "training_blocks": int(profiles["block"].nunique()),
        "state_center": center.tolist(),
        "state_scale": scale.tolist(),
        "state_ood_lower": state_lower.tolist(),
        "state_ood_upper": state_upper.tolist(),
        "conditioner_ood_lower": conditioner_lower.tolist(),
        "conditioner_ood_upper": conditioner_upper.tolist(),
        "edge_receipts": edge_receipts,
        "nuisance_edge_receipts": nuisance_receipts,
        "response_scale_c": response_scale.tolist(),
        "continuous_edge_estimations": len(EXPOSED_EDGES),
        "degrees_of_freedom_per_edge": int(config["state"]["degrees_of_freedom_per_edge"]),
        "ridge": ridge,
        "diagonal_shrinkage": shrinkage,
        "nearest_psd_projection_applied": False,
    }
    return ContinuousSparseModel(
        x_support=tuple(supports[: len(CONDITIONERS)]),
        state_center=center,
        state_scale=scale,
        conditioner_lower=conditioner_lower,
        conditioner_upper=conditioner_upper,
        state_lower=state_lower,
        state_upper=state_upper,
        edge_coefficients=coefficients,
        nuisance_correlations=nuisance,
        response_scale=response_scale,
        raw_edge_clip=float(config["dependence"]["raw_edge_correlation_clip"]),
        diagonal_shrinkage=shrinkage,
        maximum_absolute_latent_mean=float(
            config["dependence"]["maximum_absolute_latent_mean"]
        ),
        minimum_eigenvalue=float(config["dependence"]["minimum_eigenvalue"]),
        maximum_condition_number=float(config["dependence"]["maximum_condition_number"]),
        receipt=receipt,
    )


def _fit_outer_model(
    training: pd.DataFrame, state_table: pd.DataFrame, config: dict[str, Any]
) -> tuple[ContinuousSparseModel | None, dict[str, Any]]:
    profiles, missing_receipt = _training_profiles(training, state_table)
    support = {
        "profiles": int(len(profiles)),
        "kst_days": int(profiles["kst_day"].nunique()),
        "training_blocks": int(profiles["block"].nunique()),
    }
    state = config["state"]
    support["passes_support_diagnostic"] = (
        support["profiles"] >= int(state["minimum_training_profiles"])
        and support["kst_days"] >= int(state["minimum_kst_days"])
        and support["training_blocks"] >= int(state["minimum_training_blocks"])
    )
    receipt: dict[str, Any] = {
        "availability": missing_receipt,
        "support": support,
        "outer_dependence_model_fits": 1,
        "inner_selection_fits": 0,
        "hpo_trials": 0,
    }
    if not support["passes_support_diagnostic"]:
        receipt["model_guard_exact_noop"] = True
        receipt["model_guard_reason"] = "insufficient preregistered continuous-state support"
        receipt["continuous_edge_estimations"] = 0
        return None, receipt
    try:
        model = _fit_continuous_model(profiles, config)
    except ModelGuardError as error:
        receipt["model_guard_exact_noop"] = True
        receipt["model_guard_reason"] = str(error)
        receipt["continuous_edge_estimations"] = 0
        return None, receipt
    receipt["model_guard_exact_noop"] = False
    receipt["continuous_edge_estimations"] = len(EXPOSED_EDGES)
    receipt["model"] = model.receipt
    return model, receipt


def _predict_outer(
    *,
    fold: str,
    fold_spec: dict[str, Any],
    config: dict[str, Any],
    observations: pd.DataFrame,
    state_table: pd.DataFrame,
    profile_flags: pd.DataFrame,
    anchor_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    query, reference, training, _endpoints, reference_receipt = legacy._reference_frames(
        fold=fold,
        fold_spec=fold_spec,
        config=config,
        observations=observations,
        anchor_path=anchor_path,
    )
    model, model_receipt = _fit_outer_model(training, state_table, config)
    profile = query[["station", "time"]].drop_duplicates()
    counts = query.groupby(["station", "time"], sort=False)["layer"].nunique().rename(
        "target_layer_count"
    )
    profile = profile.merge(
        counts.reset_index(), on=["station", "time"], how="left", validate="one_to_one"
    )
    profile = profile.merge(
        state_table, on=["station", "time"], how="left", validate="one_to_one"
    )
    complete_target = profile["target_layer_count"].eq(len(legacy.TARGET_LAYERS)).to_numpy()
    raw_profile = np.zeros((len(profile), len(RESPONSES)), dtype=np.float64)
    active_profile = np.zeros(len(profile), dtype=bool)
    no_op_reasons = {
        "incomplete_target_profile": int((~complete_target).sum()),
        "missing_public_state": 0,
        "ood_public_state": 0,
        "correlation_matrix_guard": 0,
        "outer_model_guard": 0,
    }
    if model is None:
        no_op_reasons["outer_model_guard"] = int(complete_target.sum())
    else:
        prediction = model.predict(profile)
        eligible_active = prediction.active & complete_target
        raw_profile[eligible_active] = prediction.correction[eligible_active]
        active_profile[eligible_active] = True
        no_op_reasons["missing_public_state"] = int(
            (prediction.missing & complete_target).sum()
        )
        no_op_reasons["ood_public_state"] = int((prediction.ood & complete_target).sum())
        no_op_reasons["correlation_matrix_guard"] = int(
            (prediction.matrix_guard & complete_target).sum()
        )
    profile_correction = profile[["station", "time"]].copy()
    for column, response in enumerate(RESPONSES):
        profile_correction[response] = raw_profile[:, column]
    profile_correction["active_profile"] = active_profile
    long = profile_correction.melt(
        id_vars=["station", "time", "active_profile"],
        value_vars=list(RESPONSES),
        var_name="response",
        value_name="raw_correction",
    )
    long["layer"] = long["response"].str.removeprefix("residual_l").astype(int)
    row_prediction = query[["station", "time", "layer", "current_blend50"]].copy()
    row_prediction["reference"] = reference
    row_prediction = row_prediction.merge(
        long.drop(columns="response"),
        on=["station", "time", "layer"],
        how="left",
        validate="one_to_one",
    )
    row_prediction["active_profile"] = row_prediction["active_profile"].fillna(False)
    row_prediction["raw_correction"] = row_prediction["raw_correction"].fillna(0.0)
    lower = float(config["correction"]["structural_minimum_c"])
    upper = float(config["correction"]["structural_maximum_c"])
    correction = np.clip(
        row_prediction["raw_correction"].to_numpy(dtype=np.float64), lower, upper
    )
    active_rows = row_prediction["active_profile"].to_numpy(dtype=bool)
    correction[~active_rows] = 0.0
    candidate = reference + correction
    if not np.array_equal(candidate[~active_rows], reference[~active_rows]):
        raise ExperimentContractError("exact no-op rows changed")
    if not np.isfinite(candidate).all() or ((candidate < -5.0) | (candidate > 45.0)).any():
        raise ExperimentContractError("candidate left the finite physical temperature domain")
    row_prediction["candidate"] = candidate
    row_prediction["correction"] = candidate - reference
    realized = row_prediction["correction"].to_numpy(dtype=np.float64)
    if np.max(realized) > upper + 1e-12 or np.min(realized) < lower - 1e-12:
        raise ExperimentContractError("structural correction bound failed")
    receipt = {
        "fold": fold,
        "training_blocks": list(fold_spec["training_blocks"]),
        "training_rows": int(len(training)),
        "query_rows": int(len(query)),
        "query_profiles": int(len(profile)),
        "model": model_receipt,
        "reference": reference_receipt,
        "no_op_profile_counts": no_op_reasons,
        "active_profiles": int(active_profile.sum()),
        "inactive_profiles": int((~active_profile).sum()),
        "maximum_absolute_inactive_correction": float(
            np.max(np.abs(realized[~active_rows])) if (~active_rows).any() else 0.0
        ),
        "structural_correction_bound_c": [lower, upper],
        "correction_rms_c_diagnostic": float(np.sqrt(np.mean(np.square(realized)))),
        "correction_p99_c_diagnostic": float(np.quantile(np.abs(realized), 0.99)),
        "outlier_diagnostic": legacy._outlier_receipt(profile_flags, training),
        "prediction_sha256": legacy._prediction_hash(row_prediction),
    }
    return row_prediction, receipt


def _moving_block_bootstrap(
    scored: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    interval = config["primary_decision"]["paired_interval"]
    replicates = int(interval["replicates"])
    block_length = int(interval["block_length_days"])
    confidence = float(interval["confidence"])
    rng = np.random.default_rng(int(interval["seed"]))
    daily = scored[["window", "time", "truth", "reference", "candidate"]].copy()
    daily["kst_day"] = daily["time"].dt.tz_convert(KST).dt.strftime("%Y-%m-%d")
    daily["reference_sq"] = np.square(daily["truth"] - daily["reference"])
    daily["candidate_sq"] = np.square(daily["truth"] - daily["candidate"])
    grouped = (
        daily.groupby(["window", "kst_day"], sort=True)
        .agg(
            reference_sq=("reference_sq", "sum"),
            candidate_sq=("candidate_sq", "sum"),
            rows=("truth", "size"),
        )
        .reset_index()
    )
    windows = {
        str(window): group.sort_values("kst_day").reset_index(drop=True)
        for window, group in grouped.groupby("window", sort=True)
    }
    if any(len(group) < block_length for group in windows.values()):
        raise ExperimentContractError("moving-block bootstrap window is too short")
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        reference_sum = 0.0
        candidate_sum = 0.0
        rows_sum = 0
        for group in windows.values():
            days = len(group)
            starts = rng.integers(0, days, size=int(np.ceil(days / block_length)))
            indices = (
                starts[:, None] + np.arange(block_length, dtype=np.int64)[None, :]
            ) % days
            indices = indices.ravel()[:days]
            reference_sum += float(group["reference_sq"].to_numpy()[indices].sum())
            candidate_sum += float(group["candidate_sq"].to_numpy()[indices].sum())
            rows_sum += int(group["rows"].to_numpy()[indices].sum())
        deltas[replicate] = np.sqrt(candidate_sum / rows_sum) - np.sqrt(
            reference_sum / rows_sum
        )
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": interval["method"],
        "unit": interval["independent_unit"],
        "block_length_days": block_length,
        "replicates": replicates,
        "seed": int(interval["seed"]),
        "confidence": confidence,
        "days": int(len(grouped)),
        "days_by_window": {key: int(len(value)) for key, value in windows.items()},
        "mean_delta_rmse": float(deltas.mean()),
        "ci90_low": float(np.quantile(deltas, alpha)),
        "ci90_high": float(np.quantile(deltas, 1.0 - alpha)),
        "probability_improved": float(np.mean(deltas < 0.0)),
        "layers_preserved_together_within_day": True,
        "windows_resampled_separately": True,
    }


def _tail_risk_diagnostic(scored: pd.DataFrame, block_length_days: int) -> dict[str, Any]:
    diagnostic = scored.copy()
    diagnostic["kst_day"] = diagnostic["time"].dt.tz_convert(KST).dt.strftime("%Y-%m-%d")
    day_lookup = (
        diagnostic[["window", "kst_day"]]
        .drop_duplicates()
        .sort_values(["window", "kst_day"])
    )
    day_lookup["day_index"] = day_lookup.groupby("window", sort=False).cumcount()
    day_lookup["diagnostic_block"] = day_lookup["day_index"] // int(block_length_days)
    diagnostic = diagnostic.merge(
        day_lookup[["window", "kst_day", "diagnostic_block"]],
        on=["window", "kst_day"],
        how="left",
        validate="many_to_one",
    )
    deltas: list[float] = []
    cells: list[dict[str, Any]] = []
    for (window, block, layer), group in diagnostic.groupby(
        ["window", "diagnostic_block", "layer"], sort=True
    ):
        metric = legacy._metric_record(group)
        delta = float(metric["delta_rmse"])
        deltas.append(delta)
        cells.append(
            {
                "window": str(window),
                "block": int(block),
                "layer": int(layer),
                "rows": int(metric["rows"]),
                "delta_rmse": delta,
            }
        )
    positive = np.maximum(np.asarray(deltas, dtype=np.float64), 0.0)
    threshold = float(np.quantile(positive, 0.8))
    tail = positive[positive >= threshold]
    return {
        "role": "DIAGNOSTIC_SENSITIVITY_ONLY_NOT_A_PROMOTION_GATE",
        "cell_definition": f"nonoverlapping_{block_length_days}_KST_day_by_layer",
        "cells": len(cells),
        "positive_part_cvar80_rmse_c": float(tail.mean()) if len(tail) else 0.0,
        "maximum_cell_regression_rmse_c": float(max(deltas)) if deltas else 0.0,
        "minimum_cell_delta_rmse_c": float(min(deltas)) if deltas else 0.0,
        "cell_records": cells,
    }


def _evidence_state(pooled_delta: float, bootstrap: dict[str, Any]) -> str:
    low = float(bootstrap["ci90_low"])
    high = float(bootstrap["ci90_high"])
    if pooled_delta < 0.0 and high < 0.0:
        return "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
    if pooled_delta < 0.0 and low <= 0.0 <= high:
        return "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
    if pooled_delta > 0.0 and low > 0.0:
        return "PRIMARY_HARM_RESEARCH_ONLY"
    return "INCONCLUSIVE_RESEARCH_ONLY"


def run(config: dict[str, Any], p2_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    observations, source_receipt, access = _read_training_source(p2_dir, config)
    state_table = _public_state_table(observations, config)
    block_observations = legacy._assign_blocks(observations, config)
    diagnostic_rows = block_observations.loc[block_observations["block"].notna()].copy()
    marked = legacy.stage0._mark_row_diagnostics(diagnostic_rows, config)
    profile_flags = legacy.stage0._profile_flag_table(marked, config)
    anchor_record = config["immutable_training_inputs"]["alpha50_oof_anchor"]
    anchor_path = ROOT / anchor_record["path"]
    predictions: dict[str, pd.DataFrame] = {}
    fold_receipts: dict[str, Any] = {}
    with threadpool_limits(limits=int(config["resource_contract"]["blas_threads"])):
        for fold, fold_spec in config["frozen_historical_windows"].items():
            prediction, receipt = _predict_outer(
                fold=fold,
                fold_spec=fold_spec,
                config=config,
                observations=observations,
                state_table=state_table,
                profile_flags=profile_flags,
                anchor_path=anchor_path,
            )
            predictions[fold] = prediction
            fold_receipts[fold] = receipt
            if time.perf_counter() - started > float(
                config["resource_contract"]["maximum_wall_seconds"]
            ):
                raise ExperimentContractError("bounded runtime exceeded before scoring")
    outer_fits = sum(
        int(receipt["model"]["outer_dependence_model_fits"])
        for receipt in fold_receipts.values()
    )
    edge_estimations = sum(
        int(receipt["model"]["continuous_edge_estimations"])
        for receipt in fold_receipts.values()
    )
    if outer_fits != int(config["resource_contract"]["outer_dependence_model_fits"]):
        raise ExperimentContractError("outer fit count drifted")
    if edge_estimations != int(config["resource_contract"]["continuous_edge_estimations"]):
        raise ExperimentContractError("continuous edge estimation count drifted")
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "truth_metrics_computed": False,
        "prediction_hashes": {
            fold: receipt["prediction_sha256"] for fold, receipt in fold_receipts.items()
        },
        "config_sha256": _sha256_file(CONFIG_PATH),
        "observations_sha256": source_receipt["sha256"],
        "outer_dependence_model_fits": outer_fits,
        "continuous_edge_estimations": edge_estimations,
    }
    commitment_sha256 = _canonical_sha256(commitment)
    scored_parts: list[pd.DataFrame] = []
    for fold, prediction in predictions.items():
        if legacy._prediction_hash(prediction) != fold_receipts[fold]["prediction_sha256"]:
            raise ExperimentContractError("in-memory prediction changed before truth binding")
        truth = legacy.base.block_anchor(anchor_path, fold, include_truth=True)[
            ["time", "layer", "truth"]
        ]
        scored = prediction.merge(
            truth, on=["time", "layer"], how="left", validate="one_to_one"
        )
        if scored["truth"].isna().any() or len(scored) != len(prediction):
            raise ExperimentContractError("late historical truth binding failed")
        scored["window"] = fold
        scored_parts.append(scored)
    scored = pd.concat(scored_parts, ignore_index=True)
    scored["season"] = legacy._season_labels(scored["time"], config)
    metrics = {
        "pooled": legacy._metric_record(scored),
        "by_window": {
            str(key): legacy._metric_record(group)
            for key, group in scored.groupby("window", sort=True)
        },
        "by_layer": {
            str(int(key)): legacy._metric_record(group)
            for key, group in scored.groupby("layer", sort=True)
        },
        "by_season": {
            str(key): legacy._metric_record(group)
            for key, group in scored.groupby("season", sort=True)
        },
    }
    bootstrap = _moving_block_bootstrap(scored, config)
    correction = scored["correction"].to_numpy(dtype=np.float64)
    pooled_delta = float(metrics["pooled"]["delta_rmse"])
    evidence_state = _evidence_state(pooled_delta, bootstrap)
    tail_diagnostic = _tail_risk_diagnostic(
        scored,
        int(config["primary_decision"]["paired_interval"]["block_length_days"]),
    )
    elapsed = time.perf_counter() - started
    if elapsed > float(config["resource_contract"]["maximum_wall_seconds"]):
        raise ExperimentContractError("bounded runtime exceeded")
    result = {
        "schema_version": "p2.availability_aware_continuous_sparse_copula.result.20260830.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": evidence_state,
        "classification": config["classification"],
        "governing_policy": config["governing_policy"],
        "comparator": config["comparator"],
        "comparator_disclosure": config["comparator_disclosure"],
        "closed_family_rerun": False,
        "stage0_exposed_edges": list(EXPOSED_EDGES),
        "config_sha256": _sha256_file(CONFIG_PATH),
        "config_canonical_sha256": SEALED_CONFIG_CANONICAL_SHA256,
        "implementation_hashes": {
            "core": _sha256_file(Path(__file__)),
            "runner": _sha256_file(RUNNER_PATH),
        },
        "source": source_receipt,
        "source_open_counts": access.open_counts,
        "source_basenames_opened": ["observations.csv"],
        "immutable_training_input_hashes": {
            name: record["sha256"]
            for name, record in config["immutable_training_inputs"].items()
        },
        "prediction_commitment_sha256": commitment_sha256,
        "fold_receipts": fold_receipts,
        "metrics": metrics,
        "dependence_aware_bootstrap": bootstrap,
        "primary_decision_receipt": {
            "metric": config["primary_decision"]["metric"],
            "pooled_point_favorable": pooled_delta < 0.0,
            "paired_interval_wholly_favorable": float(bootstrap["ci90_high"]) < 0.0,
            "evidence_state": evidence_state,
            "diagnostic_slice_hard_veto_count": 0,
        },
        "diagnostics": {
            "tail_risk": tail_diagnostic,
            "transport_slices_are_not_hard_vetoes": True,
            "support_is_not_a_performance_veto": True,
            "correction_magnitude_is_not_a_performance_veto": True,
        },
        "correction": {
            "structural_bound_c": [
                float(config["correction"]["structural_minimum_c"]),
                float(config["correction"]["structural_maximum_c"]),
            ],
            "rms_c_diagnostic": float(np.sqrt(np.mean(np.square(correction)))),
            "p99_absolute_c_diagnostic": float(np.quantile(np.abs(correction), 0.99)),
            "maximum_absolute_c": float(np.max(np.abs(correction))),
        },
        "fit_counts": {
            "outer_dependence_model_fits": outer_fits,
            "inner_selection_fits": 0,
            "hpo_trials": 0,
            "continuous_edge_estimations": edge_estimations,
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "maximum_wall_seconds": float(config["resource_contract"]["maximum_wall_seconds"]),
            "python": platform.python_version(),
            "maximum_total_threads": int(config["resource_contract"]["maximum_total_threads"]),
        },
        "access_receipt": {
            "historical_truth_rows_read_after_prediction_commitment": int(len(scored)),
            "official_interface_rows_read": 0,
            "query_support_rows_read": 0,
            "csv_output_count": 0,
            "submission_generated": False,
            "upload_count": 0,
            "hard_deleted_training_profiles": 0,
        },
        "execution_receipt": {
            "attempts": 1,
            "result_based_tuning": False,
            "result_based_retry": False,
            "technical_failure_retry": False,
            "aggregate_json_only": True,
        },
    }
    return result


def _write_result(
    result: dict[str, Any], output: Path, p2_dir: Path, config: dict[str, Any]
) -> Path:
    expected = (ROOT / config["artifact_path"]).resolve(strict=False)
    target = output.resolve(strict=False)
    if target != expected or target.suffix.lower() != ".json":
        raise ExperimentContractError("--output-json must equal the sealed artifact path")
    if target.is_relative_to(p2_dir.resolve(strict=True)):
        raise ExperimentContractError("output cannot be written inside --p2-dir")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    config = load_config()
    output = args.output_json.resolve(strict=False)
    if output.exists():
        raise FileExistsError(output)
    result = run(config, args.p2_dir)
    written = _write_result(result, args.output_json, args.p2_dir, config)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "decision": result["decision"],
                "output_json": str(written),
                "pooled_delta_rmse": result["metrics"]["pooled"]["delta_rmse"],
                "ci90_high": result["dependence_aware_bootstrap"]["ci90_high"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "official_interface_rows_read": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0
