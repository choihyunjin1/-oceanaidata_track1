"""Sealed, zero-fit, training-only preflight for a state-conditioned P2 copula.

The command deliberately has no environment-variable fallback and never lists the
source directory.  The only files it can open below ``--p2-dir`` are README.md and
observations.csv.  It emits aggregate support receipts, never row-level data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_state_conditioned_copula_preflight_20260830_v1"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
KST = "Asia/Seoul"
SEALED_CONFIG_CANONICAL_SHA256 = "0adf21b8df1c9789c428afacef6a309e136d22e40b10e4c3ba5c4fd9da1db86d"


class PreflightError(RuntimeError):
    """Raised when the sealed source or preflight contract changes."""


class SourceAccessLedger:
    """Open only explicitly allowlisted files below one immutable P2 directory."""

    def __init__(self, root: Path, allowed_basenames: list[str]) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise PreflightError("--p2-dir must resolve to a directory")
        self.allowed = frozenset(allowed_basenames)
        self.open_counts = {name: 0 for name in sorted(self.allowed)}

    def _path(self, basename: str) -> Path:
        if basename not in self.allowed:
            raise PreflightError(f"source basename is not allowlisted: {basename}")
        path = (self.root / basename).resolve(strict=True)
        if path.parent != self.root or path.name != basename or not path.is_file():
            raise PreflightError(f"source escaped --p2-dir: {basename}")
        return path

    def open_binary(self, basename: str) -> BinaryIO:
        path = self._path(basename)
        self.open_counts[basename] += 1
        return path.open("rb")

    def stat(self, basename: str) -> tuple[str, int]:
        path = self._path(basename)
        return path.name, path.stat().st_size


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_source(access: SourceAccessLedger, basename: str) -> str:
    digest = hashlib.sha256()
    with access.open_binary(basename) as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise PreflightError("config experiment id changed")
    source = config.get("source_contract", {})
    if source.get("explicit_cli_argument") != "--p2-dir":
        raise PreflightError("explicit --p2-dir contract changed")
    if source.get("allowed_basenames") != ["observations.csv", "README.md"]:
        raise PreflightError("source allowlist changed")
    if source.get("directory_listing_allowed") or source.get("environment_fallback_allowed"):
        raise PreflightError("source discovery must remain disabled")
    gates = config.get("gates", {})
    expected_gates = {
        "minimum_kendall_tau_span": 0.10,
        "minimum_profiles_per_state_cell": 500,
        "minimum_kst_days_per_state_cell": 30,
        "minimum_chronological_blocks_per_state_cell": 2,
        "minimum_heterogeneous_blocks_per_edge": 2,
        "minimum_passing_edges": 2,
    }
    for key, value in expected_gates.items():
        if float(gates.get(key, -1)) != float(value):
            raise PreflightError(f"sealed gate changed: {key}")
    state = config.get("state_definition", {})
    if int(state.get("degrees_of_freedom", -1)) != 2:
        raise PreflightError("state surface must remain two-dimensional and low-DF")
    outlier = config.get("outlier_diagnostic", {})
    if outlier.get("policy") != "AGGREGATE_FLAG_AND_DIAGNOSTIC_WEIGHT_ONLY_NEVER_DELETE":
        raise PreflightError("outlier policy changed")
    if not outlier.get("physical_extrema_are_never_hard_deleted"):
        raise PreflightError("physical extrema preservation changed")
    if outlier.get("threshold_tuning_allowed"):
        raise PreflightError("outlier thresholds must remain sealed")
    execution = config.get("execution_policy", {})
    if int(execution.get("model_fit_count", -1)) != 0:
        raise PreflightError("preflight must remain zero-fit")
    if not execution.get("aggregate_json_only"):
        raise PreflightError("output must remain aggregate JSON")
    if execution.get("submission_generation_allowed") or execution.get("upload_allowed"):
        raise PreflightError("submission or upload cannot be enabled")
    if execution.get("real_data_execution_authorized") is not True:
        raise PreflightError("training-only real-data authorization changed")
    canonical = json.dumps(
        config, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != SEALED_CONFIG_CANONICAL_SHA256:
        raise PreflightError("preregistered config changed")


def _read_sources(
    p2_dir: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], SourceAccessLedger]:
    access = SourceAccessLedger(p2_dir, config["source_contract"]["allowed_basenames"])
    with access.open_binary("README.md") as handle:
        readme_payload = handle.read()
    try:
        readme_payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PreflightError("README.md is not UTF-8 decodable") from exc
    if not readme_payload.strip():
        raise PreflightError("README.md is empty")

    columns = list(config["observation_schema"])
    try:
        with access.open_binary("observations.csv") as handle:
            frame = pd.read_csv(handle, usecols=columns, dtype={"station": "string"})
    except ValueError as exc:
        raise PreflightError("observations.csv schema changed") from exc
    if set(frame.columns) != set(columns):
        raise PreflightError("observations.csv schema changed")

    readme_name, readme_bytes = access.stat("README.md")
    observations_name, observations_bytes = access.stat("observations.csv")
    receipt = {
        "readme": {
            "basename": readme_name,
            "bytes": int(readme_bytes),
            "sha256": hashlib.sha256(readme_payload).hexdigest(),
            "utf8_decoded": True,
        },
        "observations": {
            "basename": observations_name,
            "bytes": int(observations_bytes),
            "sha256": _sha256_source(access, "observations.csv"),
            "rows": int(len(frame)),
        },
    }
    return frame, receipt, access


def _prepare_observations(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    prepared = frame.copy()
    try:
        prepared["year"] = pd.to_numeric(prepared["year"], errors="raise").astype(int)
        prepared["layer"] = pd.to_numeric(prepared["layer"], errors="raise").astype(int)
        prepared["time"] = pd.to_datetime(prepared["time"], errors="raise", utc=True).dt.tz_convert(KST)
    except (TypeError, ValueError) as exc:
        raise PreflightError("observation key parsing failed") from exc
    for coordinate in ("temp", "psal", "depth", "nominal_depth"):
        prepared[coordinate] = pd.to_numeric(prepared[coordinate], errors="coerce")
    duplicate_rows = int(prepared.duplicated(["station", "year", "layer", "time"]).sum())
    if duplicate_rows:
        raise PreflightError("observations contain duplicate station-year-layer-time rows")

    prepared["block"] = pd.Series(pd.NA, index=prepared.index, dtype="string")
    assignment_count = np.zeros(len(prepared), dtype=np.int8)
    for block, bounds in config["chronological_blocks"].items():
        start, stop = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        mask = prepared["time"].ge(start) & prepared["time"].lt(stop)
        assignment_count += mask.to_numpy(dtype=np.int8)
        prepared.loc[mask, "block"] = block
    if int((assignment_count > 1).sum()):
        raise PreflightError("chronological block definitions overlap")
    rows_in_blocks = int(prepared["block"].notna().sum())
    receipt = {
        "input_rows": int(len(prepared)),
        "rows_in_preregistered_blocks": rows_in_blocks,
        "rows_outside_preregistered_blocks": int(len(prepared) - rows_in_blocks),
        "duplicate_rows": duplicate_rows,
    }
    return prepared.loc[prepared["block"].notna()].copy(), receipt


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 1e-6
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    return max(scale, 1e-6)


def _isolated_temporal_spikes(
    frame: pd.DataFrame,
    coordinate: str,
    *,
    robust_z: float,
    maximum_gap_minutes: float,
) -> pd.Series:
    flags = pd.Series(False, index=frame.index, dtype=bool)
    groups = frame.groupby(["station", "layer", "block"], sort=False, observed=True)
    for _, group in groups:
        ordered = group.sort_values("time")
        values = ordered[coordinate].to_numpy(dtype=np.float64)
        if len(values) < 3:
            continue
        times = ordered["time"]
        previous = np.roll(values, 1)
        following = np.roll(values, -1)
        central = np.abs(values - 0.5 * (previous + following))
        differences = np.diff(values)
        scale = _robust_scale(differences)
        previous_gap = (times - times.shift(1)).dt.total_seconds().to_numpy() / 60.0
        following_gap = (times.shift(-1) - times).dt.total_seconds().to_numpy() / 60.0
        valid = (
            np.isfinite(values)
            & np.isfinite(previous)
            & np.isfinite(following)
            & np.isfinite(previous_gap)
            & np.isfinite(following_gap)
            & (previous_gap <= maximum_gap_minutes)
            & (following_gap <= maximum_gap_minutes)
        )
        current = valid & (central / scale >= robust_z)
        flags.loc[ordered.index] = current
    return flags


def _mark_row_diagnostics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    marked = frame.copy()
    policy = config["outlier_diagnostic"]
    robust_z = float(policy["temporal_isolated_robust_z"])
    maximum_gap = float(policy["temporal_neighbor_max_gap_minutes"])
    marked["temp_temporal_spike"] = _isolated_temporal_spikes(
        marked, "temp", robust_z=robust_z, maximum_gap_minutes=maximum_gap
    )
    marked["psal_temporal_spike"] = _isolated_temporal_spikes(
        marked, "psal", robust_z=robust_z, maximum_gap_minutes=maximum_gap
    )
    lower_q = float(policy["physical_extreme_lower_quantile"])
    upper_q = float(policy["physical_extreme_upper_quantile"])
    grouped = marked.groupby(["station", "layer", "block"], sort=False, observed=True)["temp"]
    lower = grouped.transform(lambda values: values.quantile(lower_q))
    upper = grouped.transform(lambda values: values.quantile(upper_q))
    marked["physical_temp_extreme"] = marked["temp"].notna() & (
        marked["temp"].le(lower) | marked["temp"].ge(upper)
    )
    marked["joint_ts_temporal_discontinuity"] = (
        marked["temp_temporal_spike"] & marked["psal_temporal_spike"]
    )
    return marked


def _profile_outlier_policy(profile_flags: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = profile_flags.copy()
    result["sensor_suspect"] = result["temp_spike_layer_count"].eq(1)
    result["coherent_multilayer_temp_event"] = result["temp_spike_layer_count"].ge(2)
    result["coherent_physical_extreme"] = (
        result["physical_extreme_any"] & ~result["sensor_suspect"]
    )
    result["preserved_physical_extreme"] = result["physical_extreme_any"]
    suspect_weight = float(config["outlier_diagnostic"]["sensor_suspect_profile_weight"])
    result["diagnostic_weight"] = np.where(result["sensor_suspect"], suspect_weight, 1.0)
    return result


def _profile_flag_table(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    source = frame.copy()
    source["joint_ts_flag"] = source["joint_ts_temporal_discontinuity"]
    grouped = source.groupby(["station", "time"], sort=True, observed=True)
    flags = grouped.agg(
        temp_spike_layer_count=("temp_temporal_spike", "sum"),
        psal_spike_layer_count=("psal_temporal_spike", "sum"),
        joint_ts_discontinuity=("joint_ts_flag", "any"),
        physical_extreme_any=("physical_temp_extreme", "any"),
    ).reset_index()
    return _profile_outlier_policy(flags, config)


def _wide_column(wide: pd.DataFrame, coordinate: str, layer: int) -> pd.Series:
    key = (coordinate, layer)
    if key not in wide.columns:
        return pd.Series(np.nan, index=wide.index, dtype=np.float64)
    return pd.to_numeric(wide[key], errors="coerce")


def _effective_depth(wide: pd.DataFrame, layer: int) -> pd.Series:
    depth = _wide_column(wide, "depth", layer)
    nominal = _wide_column(wide, "nominal_depth", layer)
    return depth.where(np.isfinite(depth), nominal)


def _build_profile_table(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    flags = _profile_flag_table(frame, config)
    wide = frame.pivot(
        index=["station", "time"],
        columns="layer",
        values=["temp", "psal", "depth", "nominal_depth"],
    ).sort_index()
    profiles = wide.index.to_frame(index=False)
    profile_index = wide.index
    profiles["block"] = (
        frame.groupby(["station", "time"], sort=True, observed=True)["block"].first().to_numpy()
    )
    temp_1 = _wide_column(wide, "temp", 1)
    temp_5 = _wide_column(wide, "temp", 5)
    psal_1 = _wide_column(wide, "psal", 1)
    psal_5 = _wide_column(wide, "psal", 5)
    depth_1 = _effective_depth(wide, 1)
    depth_5 = _effective_depth(wide, 5)
    denominator = depth_5 - depth_1
    profiles["temp_contrast_signed"] = (temp_1 - temp_5).to_numpy()
    profiles["thermal_contrast_abs"] = np.abs(profiles["temp_contrast_signed"])
    profiles["psal_contrast_signed"] = (psal_1 - psal_5).to_numpy()
    for layer in map(int, config["target_layers"]):
        target_depth = float(config["target_nominal_depth_m"][str(layer)])
        weight = (target_depth - depth_1) / denominator
        interpolation = temp_1 + weight * (temp_5 - temp_1)
        profiles[f"residual_l{layer}"] = (
            _wide_column(wide, "temp", layer) - interpolation
        ).to_numpy()

    lag_hours = int(config["state_definition"]["lag_hours"])
    lag = profiles[["station", "time", "block", "thermal_contrast_abs"]].copy()
    lag["time"] = lag["time"] + pd.Timedelta(hours=lag_hours)
    lag = lag.rename(
        columns={
            "block": "lag_block",
            "thermal_contrast_abs": "thermal_contrast_abs_lag",
        }
    )
    profiles = profiles.merge(lag, on=["station", "time"], how="left", validate="one_to_one")
    same_block = profiles["block"].eq(profiles["lag_block"])
    profiles.loc[~same_block, "thermal_contrast_abs_lag"] = np.nan
    profiles["thermal_change_24h_signed"] = (
        profiles["thermal_contrast_abs"] - profiles["thermal_contrast_abs_lag"]
    )
    profiles["thermal_change_24h_abs"] = np.abs(profiles["thermal_change_24h_signed"])
    profiles = profiles.merge(flags, on=["station", "time"], how="left", validate="one_to_one")

    required = [name for edge in config["dependence_edges"] for name in edge]
    required.extend(["thermal_contrast_abs", "thermal_change_24h_abs"])
    required = sorted(set(required))
    eligible = profiles[required].notna().all(axis=1) & np.isfinite(
        profiles[required].to_numpy(dtype=np.float64)
    ).all(axis=1)
    receipt = {
        "profiles_before_complete_filter": int(len(profiles)),
        "profiles_with_complete_public_state_and_target_response": int(eligible.sum()),
        "profiles_excluded_by_complete_filter": int((~eligible).sum()),
        "response_uses_target_temperature_only": True,
        "conditioner_uses_public_layers_only": True,
        "target_salinity_used_as_conditioner": False,
        "public_endpoint_layers": [1, 5],
        "profile_index_rows": int(len(profile_index)),
    }
    return profiles.loc[eligible].reset_index(drop=True), receipt


def _expected_state_cells(config: dict[str, Any]) -> list[str]:
    state = config["state_definition"]
    return [
        f"thermal_{thermal}__dynamic_{dynamic}"
        for thermal in state["thermal_labels"]
        for dynamic in state["dynamic_labels"]
    ]


def _assign_state_cells(
    profiles: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    assigned = profiles.copy()
    state = config["state_definition"]
    thermal_quantiles = list(map(float, state["thermal_quantiles"]))
    if assigned.empty:
        assigned["state_cell"] = pd.Series(dtype="string")
        assigned["kst_day"] = pd.Series(dtype="string")
        return assigned, {
            "thermal_q1": None,
            "thermal_q2": None,
            "dynamic_median": None,
        }
    thermal_q1, thermal_q2 = assigned["thermal_contrast_abs"].quantile(thermal_quantiles).tolist()
    dynamic_median = float(assigned["thermal_change_24h_abs"].quantile(float(state["dynamic_quantile"])))
    thermal = np.where(
        assigned["thermal_contrast_abs"].le(thermal_q1),
        state["thermal_labels"][0],
        np.where(
            assigned["thermal_contrast_abs"].le(thermal_q2),
            state["thermal_labels"][1],
            state["thermal_labels"][2],
        ),
    )
    dynamic = np.where(
        assigned["thermal_change_24h_abs"].le(dynamic_median),
        state["dynamic_labels"][0],
        state["dynamic_labels"][1],
    )
    assigned["state_cell"] = "thermal_" + thermal + "__dynamic_" + dynamic
    assigned["kst_day"] = assigned["time"].dt.strftime("%Y-%m-%d")
    return assigned, {
        "thermal_q1": float(thermal_q1),
        "thermal_q2": float(thermal_q2),
        "dynamic_median": dynamic_median,
    }


def _state_support_receipts(profiles: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    gates = config["gates"]
    for cell in _expected_state_cells(config):
        subset = profiles.loc[profiles["state_cell"].eq(cell)]
        profile_count = int(len(subset))
        kst_days = int(subset["kst_day"].nunique())
        chronological_blocks = int(subset["block"].nunique())
        profiles_pass = profile_count >= int(gates["minimum_profiles_per_state_cell"])
        days_pass = kst_days >= int(gates["minimum_kst_days_per_state_cell"])
        blocks_pass = chronological_blocks >= int(
            gates["minimum_chronological_blocks_per_state_cell"]
        )
        receipts.append(
            {
                "state_cell": cell,
                "profiles": profile_count,
                "kst_days": kst_days,
                "chronological_blocks": chronological_blocks,
                "block_names": sorted(subset["block"].dropna().astype(str).unique().tolist()),
                "diagnostic_weight_sum": float(subset["diagnostic_weight"].sum()),
                "sensor_suspect_profiles": int(subset["sensor_suspect"].sum()),
                "preserved_physical_extreme_profiles": int(
                    subset["preserved_physical_extreme"].sum()
                ),
                "profiles_gte_500": profiles_pass,
                "kst_days_gte_30": days_pass,
                "chronological_blocks_gte_2": blocks_pass,
                "passes_overlap_support_gate": profiles_pass and days_pass and blocks_pass,
            }
        )
    return receipts


def _kendall_receipt(left: pd.Series, right: pd.Series, minimum_pairs: int) -> dict[str, Any]:
    x = left.to_numpy(dtype=np.float64)
    y = right.to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    pairs = int(finite.sum())
    if pairs < minimum_pairs or np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
        return {"pairs": pairs, "kendall_tau": None, "latent_gaussian_correlation": None}
    tau = float(kendalltau(x[finite], y[finite], method="auto", variant="b").statistic)
    if not np.isfinite(tau):
        return {"pairs": pairs, "kendall_tau": None, "latent_gaussian_correlation": None}
    return {
        "pairs": pairs,
        "kendall_tau": tau,
        "latent_gaussian_correlation": float(np.sin(0.5 * np.pi * tau)),
    }


def _kendall_heterogeneity(
    profiles: pd.DataFrame,
    config: dict[str, Any],
    supported_cells: set[str] | None = None,
) -> dict[str, Any]:
    expected_cells = _expected_state_cells(config)
    if supported_cells is None:
        support = _state_support_receipts(profiles, config)
        supported_cells = {
            item["state_cell"] for item in support if item["passes_overlap_support_gate"]
        }
    cells = [cell for cell in expected_cells if cell in supported_cells]
    no_op_cells = [cell for cell in expected_cells if cell not in supported_cells]
    blocks = list(config["chronological_blocks"])
    edges = [tuple(edge) for edge in config["dependence_edges"]]
    minimum_pairs = int(config["gates"]["minimum_pairs_for_block_cell_tau"])
    minimum_span = float(config["gates"]["minimum_kendall_tau_span"])
    minimum_blocks = int(config["gates"]["minimum_heterogeneous_blocks_per_edge"])
    pooled_receipts: list[dict[str, Any]] = []
    block_cell_receipts: list[dict[str, Any]] = []
    heterogeneity: list[dict[str, Any]] = []

    for left, right in edges:
        pooled_by_cell: dict[str, float] = {}
        block_by_cell: dict[str, dict[str, float]] = {block: {} for block in blocks}
        for cell in cells:
            subset = profiles.loc[profiles["state_cell"].eq(cell)]
            receipt = _kendall_receipt(subset[left], subset[right], minimum_pairs)
            pooled_receipts.append({"edge": f"{left}__{right}", "state_cell": cell, **receipt})
            if receipt["kendall_tau"] is not None:
                pooled_by_cell[cell] = float(receipt["kendall_tau"])
            for block in blocks:
                block_subset = subset.loc[subset["block"].eq(block)]
                block_receipt = _kendall_receipt(
                    block_subset[left], block_subset[right], minimum_pairs
                )
                block_cell_receipts.append(
                    {
                        "edge": f"{left}__{right}",
                        "block": block,
                        "state_cell": cell,
                        **block_receipt,
                    }
                )
                if block_receipt["kendall_tau"] is not None:
                    block_by_cell[block][cell] = float(block_receipt["kendall_tau"])

        pooled_span = (
            float(max(pooled_by_cell.values()) - min(pooled_by_cell.values()))
            if len(pooled_by_cell) >= 2
            else None
        )
        block_spans: dict[str, float | None] = {}
        for block, values in block_by_cell.items():
            span = float(max(values.values()) - min(values.values())) if len(values) >= 2 else None
            block_spans[block] = span

        pair_receipts: list[dict[str, Any]] = []
        passing_pairs: list[str] = []
        passing_pair_blocks: set[str] = set()
        maximum_consistent_blocks = 0
        for first_cell, second_cell in combinations(cells, 2):
            pooled_difference = (
                float(pooled_by_cell[first_cell] - pooled_by_cell[second_cell])
                if first_cell in pooled_by_cell and second_cell in pooled_by_cell
                else None
            )
            block_differences: dict[str, float | None] = {}
            consistent_blocks: list[str] = []
            for block in blocks:
                values = block_by_cell[block]
                difference = (
                    float(values[first_cell] - values[second_cell])
                    if first_cell in values and second_cell in values
                    else None
                )
                block_differences[block] = difference
                if (
                    pooled_difference is not None
                    and abs(pooled_difference) >= minimum_span
                    and difference is not None
                    and abs(difference) >= minimum_span
                    and np.sign(difference) == np.sign(pooled_difference)
                ):
                    consistent_blocks.append(block)
            pair_name = f"{first_cell}__vs__{second_cell}"
            pair_passes = len(consistent_blocks) >= minimum_blocks
            if pair_passes:
                passing_pairs.append(pair_name)
                passing_pair_blocks.update(consistent_blocks)
            maximum_consistent_blocks = max(maximum_consistent_blocks, len(consistent_blocks))
            pair_receipts.append(
                {
                    "state_cell_pair": pair_name,
                    "pooled_tau_difference": pooled_difference,
                    "block_tau_differences": block_differences,
                    "consistent_heterogeneous_blocks": consistent_blocks,
                    "consistent_heterogeneous_block_count": len(consistent_blocks),
                    "passes_repeated_same_pair_gate": pair_passes,
                }
            )
        passes = bool(passing_pairs)
        heterogeneity.append(
            {
                "edge": f"{left}__{right}",
                "pooled_tau_span": pooled_span,
                "block_tau_spans": block_spans,
                "same_state_cell_pair_receipts": pair_receipts,
                "passing_state_cell_pairs": passing_pairs,
                "heterogeneous_blocks": sorted(passing_pair_blocks),
                "heterogeneous_block_count": maximum_consistent_blocks,
                "passes_predeclared_heterogeneity_gate": passes,
            }
        )
    return {
        "evaluated_supported_state_cells": cells,
        "exact_no_op_unsupported_state_cells": no_op_cells,
        "unsupported_state_cell_action": config["overlap_policy"][
            "unsupported_state_cell_action"
        ],
        "pooled_state_cell_tau": pooled_receipts,
        "chronological_block_state_cell_tau": block_cell_receipts,
        "edge_heterogeneity": heterogeneity,
        "passing_edges": [
            item["edge"] for item in heterogeneity if item["passes_predeclared_heterogeneity_gate"]
        ],
    }


def _aggregate_outlier_diagnostics(profiles: pd.DataFrame) -> dict[str, Any]:
    def summarize(subset: pd.DataFrame) -> dict[str, Any]:
        profiles_count = int(len(subset))
        return {
            "profiles": profiles_count,
            "sensor_suspect_profiles": int(subset["sensor_suspect"].sum()),
            "joint_ts_discontinuity_profiles": int(subset["joint_ts_discontinuity"].sum()),
            "coherent_multilayer_temp_event_profiles": int(
                subset["coherent_multilayer_temp_event"].sum()
            ),
            "physical_extreme_profiles": int(subset["physical_extreme_any"].sum()),
            "coherent_physical_extreme_profiles": int(
                subset["coherent_physical_extreme"].sum()
            ),
            "preserved_physical_extreme_profiles": int(
                subset["preserved_physical_extreme"].sum()
            ),
            "diagnostic_weight_sum": float(subset["diagnostic_weight"].sum()),
            "mean_diagnostic_weight": (
                float(subset["diagnostic_weight"].mean()) if profiles_count else None
            ),
            "hard_deleted_profiles": 0,
        }

    return {
        "policy": "AGGREGATE_FLAG_AND_DIAGNOSTIC_WEIGHT_ONLY_NEVER_DELETE",
        "rank_based_dependence": "ordinary unweighted Kendall tau-b with sin(pi*tau/2) mapping",
        "diagnostic_weights_enter_state_cutpoints": False,
        "diagnostic_weights_enter_support_counts": False,
        "diagnostic_weights_enter_kendall_tau": False,
        "physical_extrema_remain_in_estimand": True,
        "global": summarize(profiles),
        "by_block": [
            {"block": str(block), **summarize(subset)}
            for block, subset in profiles.groupby("block", sort=True, observed=True)
        ],
        "by_state_cell": [
            {"state_cell": str(cell), **summarize(subset)}
            for cell, subset in profiles.groupby("state_cell", sort=True, observed=True)
        ],
    }


def _gate_checks(
    support: list[dict[str, Any]], heterogeneity: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    supported = [item for item in support if item["passes_overlap_support_gate"]]
    minimum_supported = int(
        config["overlap_policy"]["minimum_supported_state_cells_for_tau_span"]
    )
    enough_supported = len(supported) >= minimum_supported
    passing_edges = len(heterogeneity["passing_edges"])
    return {
        "at_least_two_supported_state_cells": enough_supported,
        "evaluated_state_cells_profiles_gte_500": enough_supported
        and all(item["profiles_gte_500"] for item in supported),
        "evaluated_state_cells_kst_days_gte_30": enough_supported
        and all(item["kst_days_gte_30"] for item in supported),
        "evaluated_state_cells_chronological_blocks_gte_2": enough_supported
        and all(item["chronological_blocks_gte_2"] for item in supported),
        "kendall_tau_span_gte_0_10_in_at_least_two_blocks": passing_edges
        >= int(config["gates"]["minimum_passing_edges"]),
    }


def audit(p2_dir: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    sealed = _load_config() if config is None else config
    _validate_config(sealed)
    raw, source_receipt, access = _read_sources(p2_dir, sealed)
    observations, observation_receipt = _prepare_observations(raw, sealed)
    marked = _mark_row_diagnostics(observations, sealed)
    profiles, profile_receipt = _build_profile_table(marked, sealed)
    profiles, state_thresholds = _assign_state_cells(profiles, sealed)
    support = _state_support_receipts(profiles, sealed)
    supported_cells = {
        item["state_cell"] for item in support if item["passes_overlap_support_gate"]
    }
    heterogeneity = _kendall_heterogeneity(profiles, sealed, supported_cells)
    checks = _gate_checks(support, heterogeneity, sealed)
    outliers = _aggregate_outlier_diagnostics(profiles)
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "TRAIN_ONLY_ZERO_FIT_PREFLIGHT_PASS"
            if all(checks.values())
            else "NO_GO_STATE_CONDITIONED_COPULA_PREFLIGHT"
        ),
        "config_sha256": _sha256_path(CONFIG_PATH),
        "source": source_receipt,
        "source_open_counts": access.open_counts,
        "source_basenames_opened": sorted(
            name for name, count in access.open_counts.items() if count > 0
        ),
        "observation_receipt": observation_receipt,
        "profile_receipt": profile_receipt,
        "state_definition": {
            "degrees_of_freedom": int(sealed["state_definition"]["degrees_of_freedom"]),
            "thresholds": state_thresholds,
            "expected_cells": _expected_state_cells(sealed),
        },
        "scientific_estimand_contract": sealed["scientific_estimand_contract"],
        "state_cell_support": support,
        "kendall_heterogeneity": heterogeneity,
        "outlier_weight_diagnostic": outliers,
        "checks": checks,
        "model_fit_count": 0,
        "official_input_rows_read": 0,
        "csv_output_count": 0,
        "submission_generated": False,
        "upload_count": 0,
        "aggregate_json_only": True,
    }


def _write_aggregate_json(result: dict[str, Any], output: Path, p2_dir: Path) -> Path:
    source_root = p2_dir.resolve(strict=True)
    target = output.resolve(strict=False)
    if target.suffix.lower() != ".json":
        raise PreflightError("--output-json must end in .json")
    if target.is_relative_to(source_root):
        raise PreflightError("aggregate output cannot be written inside --p2-dir")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.p2_dir)
    output = _write_aggregate_json(result, args.output_json, args.p2_dir)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": result["status"],
                "output_json": str(output),
                "model_fit_count": 0,
                "official_input_rows_read": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
