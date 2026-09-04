"""Run the sealed one-shot P2 state-conditioned Gaussian-rank experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
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
from scipy.stats import kendalltau, norm  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _directory in (ROOT, SRC):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from p2_restore.depth_registered_cmfpca import build_layer_identity_panel  # noqa: E402
from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (  # noqa: E402
    bounded_profile_correction,
    paired_kst_day_bootstrap,
    rmse,
)
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p2_restore.supervised_rank1_functional_residual import TARGET_LAYERS  # noqa: E402
from scripts import (  # noqa: E402
    run_p2_alpha50_supervised_rank1_functional_residual_20260828_v1 as base,
)
from scripts import (  # noqa: E402
    run_p2_state_conditioned_copula_preflight_20260830_v1 as stage0,
)

EXPERIMENT_ID = "p2_state_conditioned_copula_20260830_v1"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
SEALED_CONFIG_CANONICAL_SHA256 = "934a5c0ae162ec01618d87719f57f3f2e21e456c71746162cf72faa1bc0a46dc"
KST = "Asia/Seoul"
CONDITIONERS = (
    "temp_contrast_signed",
    "psal_contrast_signed",
    "thermal_change_24h_signed",
)
RESPONSES = ("residual_l2", "residual_l3", "residual_l4")
EXPOSED_EDGES = (
    "temp_contrast_signed__residual_l2",
    "temp_contrast_signed__residual_l3",
    "temp_contrast_signed__residual_l4",
    "psal_contrast_signed__residual_l4",
    "thermal_change_24h_signed__residual_l4",
    "residual_l2__residual_l4",
    "residual_l3__residual_l4",
)


class ExperimentContractError(RuntimeError):
    """Raised when the immutable experiment contract changes."""


class CellModelGuardError(ExperimentContractError):
    """Raised when one preregistered cell must fall back to exact no-op."""


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
    """Open only the two allowlisted training files under one explicit directory."""

    def __init__(self, root: Path, allowed_basenames: list[str]) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ExperimentContractError("--p2-dir must resolve to a directory")
        self.allowed = frozenset(allowed_basenames)
        self.open_counts = {name: 0 for name in sorted(self.allowed)}

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
    if float(config["dependence"]["diagonal_shrinkage"]) != 0.8:
        raise ExperimentContractError("fixed shrinkage changed")
    closed = config["closed_family_exclusion"]
    excluded_switches = (
        closed["seasonal_empirical_residual_margins"],
        closed["shrinkage_grid_or_selection"],
        closed["inner_model_selection"],
        closed["gauss_hermite_quadrature"],
        closed["prior_copula_predictions_reused"],
        closed["exact_closed_recipe_rerun"],
    )
    if any(excluded_switches):
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
        raise ExperimentContractError("training-only aggregate execution is not authorized")
    resources = config["resource_contract"]
    if int(resources["outer_dependence_model_fits"]) != 3:
        raise ExperimentContractError("fit count changed")
    if int(resources["inner_selection_fits"]) != 0:
        raise ExperimentContractError("inner selection was enabled")
    for _relative, record in config["immutable_training_inputs"].items():
        _validate_pinned_path(ROOT / record["path"], record)
    for relative, record in config["code_lineage"].items():
        _validate_pinned_path(ROOT / relative, record)
    stage0_record = config["immutable_training_inputs"]["stage0_result"]
    stage0_result = json.loads((ROOT / stage0_record["path"]).read_text(encoding="utf-8"))
    passing = tuple(stage0_result["kendall_heterogeneity"]["passing_edges"])
    if stage0_result.get("status") != "TRAIN_ONLY_ZERO_FIT_PREFLIGHT_PASS":
        raise ExperimentContractError("Stage-0 no longer passes")
    if set(passing) != set(EXPOSED_EDGES) or len(passing) != len(EXPOSED_EDGES):
        raise ExperimentContractError("Stage-0 passing edges changed")
    supported = stage0_result["state_cell_support"]
    if len(supported) != 6 or not all(item["passes_overlap_support_gate"] for item in supported):
        raise ExperimentContractError("Stage-0 six-cell support changed")
    return config


def _read_training_source(
    p2_dir: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], SourceAccessLedger]:
    source = config["source_contract"]
    access = SourceAccessLedger(p2_dir, source["allowed_basenames"])
    with access.open_binary("README.md") as handle:
        readme_payload = handle.read()
    with access.open_binary("observations.csv") as handle:
        observations_payload = handle.read()
    try:
        readme_payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExperimentContractError("README.md is not UTF-8 decodable") from exc
    for name, payload, record in (
        ("README.md", readme_payload, source["readme"]),
        ("observations.csv", observations_payload, source["observations"]),
    ):
        if len(payload) != int(record["bytes"]) or _sha256_bytes(payload) != record["sha256"]:
            raise ExperimentContractError(f"source pin changed: {name}")
    frame = pd.read_csv(
        io.BytesIO(observations_payload),
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
        "observations": {
            "basename": "observations.csv",
            "rows": int(len(frame)),
            "bytes": len(observations_payload),
            "sha256": _sha256_bytes(observations_payload),
        },
        "readme": {
            "basename": "README.md",
            "bytes": len(readme_payload),
            "sha256": _sha256_bytes(readme_payload),
        },
    }
    return frame, receipt, access


def _reference_config(config: dict[str, Any]) -> dict[str, Any]:
    reference = config["reference"]
    return {
        "model": {
            "reference_alpha": float(reference["alpha"]),
            "season_bin_days": int(reference["season_bin_days"]),
            "season_window_days": float(reference["season_window_days"]),
            "minimum_season_rows": int(reference["minimum_season_rows"]),
            "fallback_nearest_complete_rows": int(
                reference["fallback_nearest_complete_rows"]
            ),
            "spline_ridge": float(reference["spline_ridge"]),
            "change_hours": list(reference["change_hours"]),
        }
    }


def _assign_blocks(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    assigned = frame.copy()
    assigned["block"] = pd.Series(pd.NA, index=assigned.index, dtype="string")
    overlap = np.zeros(len(assigned), dtype=np.int8)
    for block, bounds in config["block_bounds"].items():
        start, stop = base.utc(bounds[0]), base.utc(bounds[1])
        mask = assigned["time"].ge(start) & assigned["time"].lt(stop)
        overlap += mask.to_numpy(dtype=np.int8)
        assigned.loc[mask, "block"] = block
    if int((overlap > 1).sum()):
        raise ExperimentContractError("historical blocks overlap")
    return assigned


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
    lag = profile[["station", "time", "thermal_contrast_abs"]].copy()
    lag["time"] = lag["time"] + pd.Timedelta(hours=int(config["state"]["lag_hours"]))
    lag = lag.rename(columns={"thermal_contrast_abs": "thermal_contrast_abs_lag"})
    profile = profile.merge(lag, on=["station", "time"], how="left", validate="one_to_one")
    profile["thermal_change_24h_signed"] = (
        profile["thermal_contrast_abs"] - profile["thermal_contrast_abs_lag"]
    )
    profile["thermal_change_24h_abs"] = np.abs(profile["thermal_change_24h_signed"])
    return profile


def _expected_cells(config: dict[str, Any]) -> list[str]:
    state = config["state"]
    return [
        f"thermal_{thermal}__dynamic_{dynamic}"
        for thermal in state["thermal_labels"]
        for dynamic in state["dynamic_labels"]
    ]


def _state_thresholds(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    finite = frame[["thermal_contrast_abs", "thermal_change_24h_abs"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(finite) < 2:
        raise ExperimentContractError("state threshold support is empty")
    q1, q2 = finite["thermal_contrast_abs"].quantile(
        list(map(float, config["state"]["thermal_quantiles"]))
    )
    dynamic = finite["thermal_change_24h_abs"].quantile(
        float(config["state"]["dynamic_quantile"])
    )
    return {"thermal_q1": float(q1), "thermal_q2": float(q2), "dynamic_median": float(dynamic)}


def _assign_state_cells(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    assigned = frame.copy()
    thermal = np.where(
        assigned["thermal_contrast_abs"].le(thresholds["thermal_q1"]),
        "low",
        np.where(
            assigned["thermal_contrast_abs"].le(thresholds["thermal_q2"]),
            "middle",
            "high",
        ),
    )
    dynamic = np.where(
        assigned["thermal_change_24h_abs"].le(thresholds["dynamic_median"]),
        "steady",
        "active",
    )
    assigned["state_cell"] = "thermal_" + thermal + "__dynamic_" + dynamic
    incomplete = ~np.isfinite(
        assigned[["thermal_contrast_abs", "thermal_change_24h_abs"]].to_numpy(
            dtype=np.float64
        )
    ).all(axis=1)
    assigned.loc[incomplete, "state_cell"] = pd.NA
    return assigned


def _latent_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    tau = float(kendalltau(left, right, method="auto", variant="b").statistic)
    if not np.isfinite(tau):
        raise CellModelGuardError("Kendall tau became nonfinite")
    return tau, float(np.sin(0.5 * np.pi * np.clip(tau, -1.0, 1.0)))


def _empirical_to_normal(support: np.ndarray, query: np.ndarray) -> np.ndarray:
    ordered = np.asarray(support, dtype=np.float64)
    values = np.asarray(query, dtype=np.float64)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    rank = 0.5 * (left + right)
    probability = np.clip(
        (rank + 0.5) / len(ordered),
        0.5 / len(ordered),
        1.0 - 0.5 / len(ordered),
    )
    result = norm.ppf(probability)
    if not np.isfinite(result).all():
        raise ExperimentContractError("empirical normal score became nonfinite")
    return result


@dataclass(frozen=True)
class StateCellModel:
    state_cell: str
    x_support: tuple[np.ndarray, ...]
    ood_lower: np.ndarray
    ood_upper: np.ndarray
    beta: np.ndarray
    response_scale: np.ndarray
    receipt: dict[str, Any]

    def predict(
        self, x: np.ndarray, maximum_absolute_latent_mean: float
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        inside = finite & (values >= self.ood_lower).all(axis=1) & (
            values <= self.ood_upper
        ).all(axis=1)
        correction = np.zeros((len(values), len(RESPONSES)), dtype=np.float64)
        if inside.any():
            normal_score = np.column_stack(
                [
                    _empirical_to_normal(support, values[inside, column])
                    for column, support in enumerate(self.x_support)
                ]
            )
            latent = np.clip(
                normal_score @ self.beta.T,
                -float(maximum_absolute_latent_mean),
                float(maximum_absolute_latent_mean),
            )
            correction[inside] = latent * self.response_scale
        return correction, inside


def _fit_state_cell(
    state_cell: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> StateCellModel:
    variables = CONDITIONERS + RESPONSES
    values = frame[list(variables)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ExperimentContractError("cell fit received nonfinite values")
    raw_sparse = np.eye(len(variables), dtype=np.float64)
    edge_receipts: dict[str, dict[str, float]] = {}
    index = {name: position for position, name in enumerate(variables)}
    modeled_edges = list(config["dependence"]["stage0_exposed_edges"])
    nuisance_edges = [f"{left}__{right}" for left, right in combinations(CONDITIONERS, 2)]
    for edge in modeled_edges + nuisance_edges:
        left, right = edge.split("__", maxsplit=1)
        tau, latent = _latent_correlation(values[:, index[left]], values[:, index[right]])
        raw_sparse[index[left], index[right]] = latent
        raw_sparse[index[right], index[left]] = latent
        edge_receipts[edge] = {"kendall_tau_b": tau, "latent_correlation": latent}
    shrinkage = float(config["dependence"]["diagonal_shrinkage"])
    correlation = (1.0 - shrinkage) * raw_sparse + shrinkage * np.eye(len(variables))
    eigenvalues = np.linalg.eigvalsh(correlation)
    minimum_eigenvalue = float(eigenvalues.min())
    condition_number = float(np.linalg.cond(correlation))
    if minimum_eigenvalue < float(config["dependence"]["minimum_eigenvalue"]):
        raise CellModelGuardError(f"cell covariance is not PSD: {state_cell}")
    if condition_number > float(config["dependence"]["maximum_condition_number"]):
        raise CellModelGuardError(f"cell covariance is ill-conditioned: {state_cell}")
    x_count = len(CONDITIONERS)
    sigma_xx = correlation[:x_count, :x_count]
    sigma_yx = correlation[x_count:, :x_count]
    beta = np.linalg.solve(sigma_xx, sigma_yx.T).T
    y = values[:, x_count:]
    scale = (
        np.quantile(y, 0.75, axis=0) - np.quantile(y, 0.25, axis=0)
    ) / 1.3489795003921634
    if (scale < float(config["dependence"]["minimum_response_scale_c"])).any():
        raise CellModelGuardError(f"cell residual scale collapsed: {state_cell}")
    x = values[:, :x_count]
    lower = np.quantile(x, float(config["state"]["ood_lower_quantile"]), axis=0)
    upper = np.quantile(x, float(config["state"]["ood_upper_quantile"]), axis=0)
    return StateCellModel(
        state_cell=state_cell,
        x_support=tuple(np.sort(x[:, column]) for column in range(x_count)),
        ood_lower=lower,
        ood_upper=upper,
        beta=beta,
        response_scale=scale,
        receipt={
            "state_cell": state_cell,
            "rows": int(len(frame)),
            "kst_days": int(frame["kst_day"].nunique()),
            "training_blocks": int(frame["block"].nunique()),
            "diagonal_shrinkage": shrinkage,
            "minimum_eigenvalue": minimum_eigenvalue,
            "condition_number": condition_number,
            "nearest_psd_projection_applied": False,
            "response_scale_c": scale.tolist(),
            "ood_lower": lower.tolist(),
            "ood_upper": upper.tolist(),
            "edge_receipts": edge_receipts,
        },
    )


def _training_profiles(
    training: pd.DataFrame,
    state_table: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    keys = training[["station", "time", "block"]].drop_duplicates()
    if keys.duplicated(["station", "time"]).any():
        raise ExperimentContractError("training profile block keys duplicate")
    state = keys.merge(state_table, on=["station", "time"], how="left", validate="one_to_one")
    thresholds = _state_thresholds(state, config)
    state = _assign_state_cells(state, thresholds)
    response = training.pivot(
        index=["station", "time"], columns="layer", values="residual"
    ).reindex(columns=TARGET_LAYERS)
    response.columns = list(RESPONSES)
    response = response.reset_index()
    profiles = state.merge(response, on=["station", "time"], how="left", validate="one_to_one")
    required = list(CONDITIONERS + RESPONSES)
    eligible = np.isfinite(profiles[required].to_numpy(dtype=np.float64)).all(axis=1)
    profiles = profiles.loc[eligible & profiles["state_cell"].notna()].copy()
    profiles["kst_day"] = profiles["time"].dt.tz_convert(KST).dt.strftime("%Y-%m-%d")
    return profiles, thresholds


def _fit_outer_state_model(
    training: pd.DataFrame,
    state_table: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, StateCellModel], dict[str, Any]]:
    profiles, thresholds = _training_profiles(training, state_table, config)
    models: dict[str, StateCellModel] = {}
    support_receipts: list[dict[str, Any]] = []
    estimation_count = 0
    state = config["state"]
    for cell in _expected_cells(config):
        subset = profiles.loc[profiles["state_cell"].eq(cell)].copy()
        support = {
            "state_cell": cell,
            "profiles": int(len(subset)),
            "kst_days": int(subset["kst_day"].nunique()),
            "training_blocks": int(subset["block"].nunique()),
        }
        passes = (
            support["profiles"] >= int(state["minimum_profiles_per_cell"])
            and support["kst_days"] >= int(state["minimum_kst_days_per_cell"])
            and support["training_blocks"] >= int(state["minimum_training_blocks_per_cell"])
        )
        support["passes_support_gate"] = passes
        if passes:
            estimation_count += 1
            try:
                model = _fit_state_cell(cell, subset, config)
            except CellModelGuardError as error:
                support["model_guard_exact_noop"] = True
                support["model_guard_reason"] = str(error)
            else:
                models[cell] = model
                support["model_receipt"] = model.receipt
                support["model_guard_exact_noop"] = False
        else:
            support["unsupported_action"] = state["unsupported_cell_action"]
        support_receipts.append(support)
    return models, {
        "state_thresholds": thresholds,
        "training_profiles": int(len(profiles)),
        "state_cell_support": support_receipts,
        "support_eligible_state_cells": [
            item["state_cell"] for item in support_receipts if item["passes_support_gate"]
        ],
        "fitted_state_cells": sorted(models),
        "unsupported_state_cells": [
            item["state_cell"] for item in support_receipts if not item["passes_support_gate"]
        ],
        "exact_noop_state_cells": [
            cell for cell in _expected_cells(config) if cell not in models
        ],
        "state_cell_correlation_estimations": estimation_count,
    }


def _prediction_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["station", "time", "layer"]).reset_index(drop=True)
    digest = hashlib.sha256()
    for column in ("station",):
        digest.update("\n".join(ordered[column].astype(str)).encode("utf-8"))
    digest.update(pd.DatetimeIndex(ordered["time"]).as_unit("ns").asi8.astype("<i8").tobytes())
    digest.update(ordered["layer"].to_numpy(dtype="<i2").tobytes())
    for column in ("reference", "candidate", "correction"):
        digest.update(ordered[column].to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()


def _outlier_receipt(profile_flags: pd.DataFrame, training: pd.DataFrame) -> dict[str, Any]:
    keys = training[["station", "time"]].drop_duplicates()
    selected = keys.merge(profile_flags, on=["station", "time"], how="left", validate="one_to_one")
    boolean_columns = (
        "sensor_suspect",
        "coherent_multilayer_temp_event",
        "physical_extreme_any",
        "preserved_physical_extreme",
    )
    for column in boolean_columns:
        selected[column] = selected[column].fillna(False).astype(bool)
    selected["diagnostic_weight"] = selected["diagnostic_weight"].fillna(1.0)
    return {
        "profiles": int(len(selected)),
        "sensor_suspect_profiles": int(selected["sensor_suspect"].sum()),
        "coherent_multilayer_temp_event_profiles": int(
            selected["coherent_multilayer_temp_event"].sum()
        ),
        "physical_extreme_profiles": int(selected["physical_extreme_any"].sum()),
        "preserved_physical_extreme_profiles": int(
            selected["preserved_physical_extreme"].sum()
        ),
        "diagnostic_weight_sum": float(selected["diagnostic_weight"].sum()),
        "hard_deleted_profiles": 0,
        "weights_enter_fit_or_metric": False,
    }


def _reference_frames(
    *,
    fold: str,
    fold_spec: dict[str, Any],
    config: dict[str, Any],
    observations: pd.DataFrame,
    anchor_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    start, stop = base.utc(fold_spec["start"]), base.utc(fold_spec["stop"])
    masked = observations.copy()
    validation_mask = (
        masked["time"].ge(start)
        & masked["time"].lt(stop)
        & masked["layer"].isin(TARGET_LAYERS)
    )
    masked.loc[validation_mask, ["temp", "psal"]] = np.nan
    panel, _, _ = build_layer_identity_panel(masked)
    endpoints = public_endpoint_frame(masked)
    query = base.add_metadata(base.block_anchor(anchor_path, fold, include_truth=False), observations)
    model_config = _reference_config(config)
    reference, query_reference_receipts = base.alpha50_reference(
        panel=panel,
        endpoints=endpoints,
        query=query,
        train_stop=start,
        config=model_config,
    )
    training_parts: list[pd.DataFrame] = []
    training_reference_receipts: dict[str, Any] = {}
    for training_block in fold_spec["training_blocks"]:
        training = base.add_metadata(
            base.block_anchor(anchor_path, training_block, include_truth=True), observations
        )
        if not training["time"].lt(start).all():
            raise ExperimentContractError("training label crosses outer boundary")
        bounds = config["block_bounds"][training_block]
        training_reference, receipts = base.alpha50_reference(
            panel=panel,
            endpoints=endpoints,
            query=training,
            train_stop=start,
            config=model_config,
            exclude=(base.utc(bounds[0]), base.utc(bounds[1])),
        )
        training["reference"] = training_reference
        training["residual"] = training["truth"].to_numpy(dtype=np.float64) - training_reference
        training_parts.append(training)
        training_reference_receipts[training_block] = receipts
    return (
        query,
        reference,
        pd.concat(training_parts, ignore_index=True),
        endpoints,
        {
            "query_reference": query_reference_receipts,
            "training_reference": training_reference_receipts,
            "validation_target_temp_psal_masked_together": True,
        },
    )


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
    query, reference, training, endpoints, reference_receipt = _reference_frames(
        fold=fold,
        fold_spec=fold_spec,
        config=config,
        observations=observations,
        anchor_path=anchor_path,
    )
    models, model_receipt = _fit_outer_state_model(training, state_table, config)
    thresholds = model_receipt["state_thresholds"]
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
    profile = _assign_state_cells(profile, thresholds)
    raw_profile = np.zeros((len(profile), len(RESPONSES)), dtype=np.float64)
    active_profile = np.zeros(len(profile), dtype=bool)
    no_op_reasons = {
        "incomplete_target_profile": int((profile["target_layer_count"] != len(TARGET_LAYERS)).sum()),
        "missing_public_state": int(profile["state_cell"].isna().sum()),
        "unsupported_state_cell": 0,
        "model_guard_state_cell": 0,
        "ood_public_state": 0,
    }
    complete_profile = profile["target_layer_count"].eq(len(TARGET_LAYERS)).to_numpy()
    support_eligible = set(model_receipt["support_eligible_state_cells"])
    for cell in _expected_cells(config):
        cell_mask = profile["state_cell"].eq(cell).to_numpy() & complete_profile
        if not cell_mask.any():
            continue
        model = models.get(cell)
        if model is None:
            reason = "model_guard_state_cell" if cell in support_eligible else "unsupported_state_cell"
            no_op_reasons[reason] += int(cell_mask.sum())
            continue
        correction, inside = model.predict(
            profile.loc[cell_mask, list(CONDITIONERS)].to_numpy(dtype=np.float64),
            float(config["dependence"]["maximum_absolute_latent_mean"]),
        )
        indices = np.flatnonzero(cell_mask)
        raw_profile[indices] = correction
        active_profile[indices[inside]] = True
        no_op_reasons["ood_public_state"] += int((~inside).sum())
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
    bounded, cap_receipt = bounded_profile_correction(
        row_prediction["raw_correction"].to_numpy(dtype=np.float64),
        row_prediction["active_profile"].to_numpy(dtype=bool),
        rms_cap=float(config["correction_caps"]["maximum_rms_c"]),
        p99_cap=float(config["correction_caps"]["maximum_p99_absolute_c"]),
    )
    candidate = reference.copy()
    active_rows = row_prediction["active_profile"].to_numpy(dtype=bool)
    if active_rows.any():
        active_query = query.loc[active_rows].reset_index(drop=True)
        active_prediction = reference[active_rows] + bounded[active_rows]
        candidate[active_rows] = project_profiles_vectorized(
            active_query, active_prediction, endpoints
        ).prediction
    if not np.array_equal(candidate[~active_rows], reference[~active_rows]):
        raise ExperimentContractError("exact no-op rows changed")
    row_prediction["candidate"] = candidate
    row_prediction["correction"] = candidate - reference
    if not np.isfinite(row_prediction[["reference", "candidate", "correction"]]).all().all():
        raise ExperimentContractError("prediction became nonfinite")
    correction = row_prediction["correction"].to_numpy(dtype=np.float64)
    receipt = {
        "fold": fold,
        "training_blocks": list(fold_spec["training_blocks"]),
        "training_rows": int(len(training)),
        "query_rows": int(len(query)),
        "query_profiles": int(len(profile)),
        "outer_dependence_model_fits": 1,
        "model": model_receipt,
        "reference": reference_receipt,
        "no_op_profile_counts": no_op_reasons,
        "active_profiles": int(active_profile.sum()),
        "inactive_profiles": int((~active_profile).sum()),
        "maximum_absolute_inactive_correction": float(
            np.max(np.abs(correction[~active_rows])) if (~active_rows).any() else 0.0
        ),
        "preprojection_cap": cap_receipt,
        "postprojection_correction_rms_c": float(np.sqrt(np.mean(np.square(correction)))),
        "postprojection_correction_p99_c": float(np.quantile(np.abs(correction), 0.99)),
        "outlier_diagnostic": _outlier_receipt(profile_flags, training),
        "prediction_sha256": _prediction_hash(row_prediction),
    }
    return row_prediction, receipt


def _metric_record(frame: pd.DataFrame) -> dict[str, float | int]:
    reference = rmse(frame["truth"].to_numpy(), frame["reference"].to_numpy())
    candidate = rmse(frame["truth"].to_numpy(), frame["candidate"].to_numpy())
    return {
        "rows": int(len(frame)),
        "reference_rmse": reference,
        "candidate_rmse": candidate,
        "delta_rmse": candidate - reference,
    }


def _season_labels(times: pd.Series, config: dict[str, Any]) -> np.ndarray:
    month_to_season = {
        int(month): season
        for season, months in config["season_definition"].items()
        for month in months
    }
    months = pd.to_datetime(times, utc=True).dt.tz_convert(KST).dt.month
    return np.asarray([month_to_season[int(month)] for month in months], dtype=str)


def _gate_checks(
    metrics: dict[str, Any],
    bootstrap: dict[str, Any],
    correction: np.ndarray,
    config: dict[str, Any],
) -> dict[str, bool]:
    gate = config["promotion_gate"]
    fold_deltas = [float(item["delta_rmse"]) for item in metrics["by_window"].values()]
    layer_deltas = [float(item["delta_rmse"]) for item in metrics["by_layer"].values()]
    season_deltas = [float(item["delta_rmse"]) for item in metrics["by_season"].values()]
    correction_rms = float(np.sqrt(np.mean(np.square(correction))))
    correction_p99 = float(np.quantile(np.abs(correction), 0.99))
    return {
        "pooled_delta_rmse_lt_0": metrics["pooled"]["delta_rmse"]
        < float(gate["pooled_delta_rmse_max_c"]),
        "at_least_two_of_three_windows_improve": sum(delta < 0.0 for delta in fold_deltas)
        >= int(gate["minimum_improved_windows"]),
        "no_layer_worse_by_more_than_0_001_c": max(layer_deltas)
        <= float(gate["maximum_layer_regression_c"]),
        "worst_season_regression_lte_0_003_c": max(season_deltas)
        <= float(gate["maximum_worst_season_regression_c"]),
        "paired_kst_day_bootstrap_ci90_upper_lt_0": bootstrap["ci90_high"]
        < float(gate["bootstrap_ci90_upper_max_c"]),
        "correction_rms_lte_0_075_c": correction_rms
        <= float(gate["maximum_correction_rms_c"]) + 1e-12,
        "correction_p99_lte_0_2_c": correction_p99
        <= float(gate["maximum_correction_p99_c"]) + 1e-12,
    }


def run(config: dict[str, Any], p2_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    observations, source_receipt, access = _read_training_source(p2_dir, config)
    state_table = _public_state_table(observations, config)
    block_observations = _assign_blocks(observations, config)
    diagnostic_rows = block_observations.loc[block_observations["block"].notna()].copy()
    marked = stage0._mark_row_diagnostics(diagnostic_rows, config)
    profile_flags = stage0._profile_flag_table(marked, config)
    anchor_record = config["immutable_training_inputs"]["alpha50_oof_anchor"]
    anchor_path = ROOT / anchor_record["path"]
    predictions: dict[str, pd.DataFrame] = {}
    fold_receipts: dict[str, Any] = {}
    cell_estimations = 0
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
            cell_estimations += int(receipt["model"]["state_cell_correlation_estimations"])
            if time.perf_counter() - started > float(
                config["resource_contract"]["maximum_wall_seconds"]
            ):
                raise ExperimentContractError("bounded runtime exceeded before scoring")
    outer_fits = sum(
        int(receipt["outer_dependence_model_fits"]) for receipt in fold_receipts.values()
    )
    if outer_fits != int(config["resource_contract"]["outer_dependence_model_fits"]):
        raise ExperimentContractError("outer fit count drifted")
    if cell_estimations > int(
        config["resource_contract"]["maximum_state_cell_correlation_estimations"]
    ):
        raise ExperimentContractError("cell estimation budget exceeded")
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "truth_metrics_computed": False,
        "prediction_hashes": {
            fold: receipt["prediction_sha256"] for fold, receipt in fold_receipts.items()
        },
        "config_sha256": _sha256_file(CONFIG_PATH),
        "observations_sha256": source_receipt["observations"]["sha256"],
        "outer_dependence_model_fits": outer_fits,
        "state_cell_correlation_estimations": cell_estimations,
    }
    commitment_sha256 = _canonical_sha256(commitment)
    scored_parts: list[pd.DataFrame] = []
    for fold, prediction in predictions.items():
        if _prediction_hash(prediction) != fold_receipts[fold]["prediction_sha256"]:
            raise ExperimentContractError("in-memory prediction changed before truth binding")
        truth = base.block_anchor(anchor_path, fold, include_truth=True)[
            ["time", "layer", "truth"]
        ]
        scored = prediction.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
        if scored["truth"].isna().any() or len(scored) != len(prediction):
            raise ExperimentContractError("late historical truth binding failed")
        scored["window"] = fold
        scored_parts.append(scored)
    scored = pd.concat(scored_parts, ignore_index=True)
    scored["season"] = _season_labels(scored["time"], config)
    metrics = {
        "pooled": _metric_record(scored),
        "by_window": {
            str(key): _metric_record(group)
            for key, group in scored.groupby("window", sort=True)
        },
        "by_layer": {
            str(int(key)): _metric_record(group)
            for key, group in scored.groupby("layer", sort=True)
        },
        "by_season": {
            str(key): _metric_record(group)
            for key, group in scored.groupby("season", sort=True)
        },
    }
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(config["promotion_gate"]["bootstrap_replicates"]),
        seed=int(config["promotion_gate"]["bootstrap_seed"]),
    )
    correction = scored["correction"].to_numpy(dtype=np.float64)
    checks = _gate_checks(metrics, bootstrap, correction, config)
    elapsed = time.perf_counter() - started
    if elapsed > float(config["resource_contract"]["maximum_wall_seconds"]):
        raise ExperimentContractError("bounded runtime exceeded")
    result = {
        "schema_version": "p2.state_conditioned_copula.result.20260830.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": (
            "GO_RESEARCH_ONLY_NO_OFFICIAL_DEPLOYMENT"
            if all(checks.values())
            else "NO_GO_STATE_CONDITIONED_COPULA_STAGE1"
        ),
        "classification": config["classification"],
        "comparator": config["comparator"],
        "comparator_disclosure": config["comparator_disclosure"],
        "closed_family_rerun": False,
        "stage0_exposed_edges": list(EXPOSED_EDGES),
        "config_sha256": _sha256_file(CONFIG_PATH),
        "config_canonical_sha256": SEALED_CONFIG_CANONICAL_SHA256,
        "source": source_receipt,
        "source_open_counts": access.open_counts,
        "source_basenames_opened": sorted(
            name for name, count in access.open_counts.items() if count > 0
        ),
        "immutable_training_input_hashes": {
            name: record["sha256"]
            for name, record in config["immutable_training_inputs"].items()
        },
        "prediction_commitment_sha256": commitment_sha256,
        "fold_receipts": fold_receipts,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "correction": {
            "rms_c": float(np.sqrt(np.mean(np.square(correction)))),
            "p99_absolute_c": float(np.quantile(np.abs(correction), 0.99)),
            "maximum_absolute_c": float(np.max(np.abs(correction))),
        },
        "promotion_checks": checks,
        "fit_counts": {
            "outer_dependence_model_fits": outer_fits,
            "inner_selection_fits": 0,
            "state_cell_correlation_estimations": cell_estimations,
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


def _write_result(result: dict[str, Any], output: Path, p2_dir: Path, config: dict[str, Any]) -> Path:
    expected = (ROOT / config["artifact_path"]).resolve(strict=False)
    target = output.resolve(strict=False)
    if target != expected or target.suffix.lower() != ".json":
        raise ExperimentContractError("--output-json must equal the sealed artifact path")
    if target.is_relative_to(p2_dir.resolve(strict=True)):
        raise ExperimentContractError("output cannot be written inside --p2-dir")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "official_interface_rows_read": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
