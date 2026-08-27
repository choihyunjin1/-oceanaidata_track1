"""Preregistered ERA5 mixing features for the P2 two-expert convex gate.

This module contains feature construction and invariants only.  It does not
load P2 labels, fit a gate, inspect test data, or write a submission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

WINDOW_HOURS = (6, 24, 72, 168)
HOUR_SECONDS = 3600.0

U10_COLUMN = "10m_u_component_of_wind"
V10_COLUMN = "10m_v_component_of_wind"
TAU_EAST_COLUMN = "eastward_turbulent_surface_stress"
TAU_NORTH_COLUMN = "northward_turbulent_surface_stress"
SSR_COLUMN = "surface_net_solar_radiation"
STR_COLUMN = "surface_net_thermal_radiation"
SLHF_COLUMN = "surface_latent_heat_flux"
SSHF_COLUMN = "surface_sensible_heat_flux"
LSM_COLUMN = "land_sea_mask"

ERA5_VALUE_COLUMNS = (
    U10_COLUMN,
    V10_COLUMN,
    TAU_EAST_COLUMN,
    TAU_NORTH_COLUMN,
    SSR_COLUMN,
    STR_COLUMN,
    SLHF_COLUMN,
    SSHF_COLUMN,
    LSM_COLUMN,
)

ERA5_BASE_FEATURES = (
    "era5_u10_ms",
    "era5_v10_ms",
    "era5_tau_mag_nm2",
    "era5_tau_dir_sin",
    "era5_tau_dir_cos",
    "era5_qnet_native_wm2",
)


def _window_feature_names(window: int) -> tuple[str, ...]:
    return (
        f"era5_u10_mean_{window}h_ms",
        f"era5_v10_mean_{window}h_ms",
        f"era5_tau_mag_impulse_{window}h_nsm2",
        f"era5_qnet_energy_{window}h_jm2",
        f"era5_tau_mag_trend_{window}h_nm2_per_h",
        f"era5_qnet_trend_{window}h_wm2_per_h",
    )


ERA5_MIXING_FEATURES = ERA5_BASE_FEATURES + tuple(
    name for window in WINDOW_HOURS for name in _window_feature_names(window)
)


def validate_preregistered_feature_contract(feature_names: Sequence[str]) -> None:
    """Fail closed when the config and implementation feature orders diverge."""

    if tuple(feature_names) != ERA5_MIXING_FEATURES:
        raise ValueError("ERA5 mixing feature contract differs from the preregistered order")


def validate_manifest_units_and_signs(
    manifest: Mapping[str, object],
    *,
    expected_units: Mapping[str, str],
    expected_signs: Mapping[str, str],
) -> None:
    """Pin source units and native signs before any derived field is built."""

    units = manifest.get("units")
    signs = manifest.get("sign_semantics")
    if not isinstance(units, Mapping) or not isinstance(signs, Mapping):
        raise ValueError("ERA5 manifest is missing units or sign semantics")
    if dict(units) != dict(expected_units):
        raise ValueError("ERA5 manifest units changed")
    if dict(signs) != dict(expected_signs):
        raise ValueError("ERA5 manifest sign semantics changed")


def validate_era5_source_frame(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    expected_blocks: Sequence[str] | None = None,
    expected_grid_points: int | None = None,
) -> dict[str, object]:
    """Validate the independent ERA5 table and return aggregate-only diagnostics."""

    required = {
        "chunk_id",
        "block",
        "time_utc",
        "time_kst",
        "latitude",
        "longitude",
        *ERA5_VALUE_COLUMNS,
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"ERA5 source schema is missing {sorted(missing_columns)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError("ERA5 source row count changed")

    utc = pd.to_datetime(frame["time_utc"], utc=True, errors="raise")
    kst = pd.to_datetime(frame["time_kst"], utc=True, errors="raise").dt.tz_convert(
        "Asia/Seoul"
    )
    if not (utc.array == kst.array).all():
        raise ValueError("ERA5 UTC and KST columns do not represent the same instants")
    wall_clock_offset = (
        kst.dt.tz_localize(None) - utc.dt.tz_localize(None)
    ).dt.total_seconds().div(60)
    if not wall_clock_offset.eq(540.0).all():
        raise ValueError("ERA5 KST wall-clock offset is not exactly +09:00")

    key_columns = ["time_utc", "latitude", "longitude"]
    duplicates = int(frame.duplicated(key_columns).sum())
    if duplicates:
        raise ValueError("ERA5 source contains duplicate hour/grid keys")
    numeric = frame.loc[:, ["latitude", "longitude", *ERA5_VALUE_COLUMNS]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("ERA5 source contains missing or non-finite numeric values")
    lsm = frame[LSM_COLUMN].to_numpy(float)
    if np.min(lsm) < 0.0 or np.max(lsm) > 1.0:
        raise ValueError("ERA5 land-sea mask is outside [0, 1]")

    blocks = tuple(sorted(frame["block"].astype(str).unique()))
    if expected_blocks is not None and blocks != tuple(sorted(expected_blocks)):
        raise ValueError("ERA5 block labels changed")
    grid_points = int(frame[["latitude", "longitude"]].drop_duplicates().shape[0])
    if expected_grid_points is not None and grid_points != expected_grid_points:
        raise ValueError("ERA5 grid-point count changed")

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "chunk_count": int(frame["chunk_id"].nunique()),
        "block_count": len(blocks),
        "blocks": list(blocks),
        "unique_hour_count": int(utc.nunique()),
        "unique_grid_point_count": grid_points,
        "duplicate_key_count": duplicates,
        "missing_value_count": int(frame.loc[:, ERA5_VALUE_COLUMNS].isna().sum().sum()),
        "time_start_utc": utc.min().isoformat(),
        "time_end_utc": utc.max().isoformat(),
        "time_start_kst": kst.min().isoformat(),
        "time_end_kst": kst.max().isoformat(),
        "utc_to_kst_wall_clock_offset_minutes": 540,
        "land_sea_mask_minimum": float(np.min(lsm)),
        "land_sea_mask_maximum": float(np.max(lsm)),
    }


def _rolling_ols_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(centered @ centered)

    def slope(values: np.ndarray) -> float:
        return float(centered @ values / denominator)

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def build_hourly_ocean_mixing_features(
    frame: pd.DataFrame,
    *,
    ocean_lsm_maximum: float = 0.5,
    expected_ocean_cells_per_hour: int | None = 9,
) -> pd.DataFrame:
    """Build the exact causal, block-reset ERA5 feature panel at hourly grain."""

    required = {"block", "time_utc", "latitude", "longitude", *ERA5_VALUE_COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError("ERA5 mixing input schema is incomplete")
    selected = frame.loc[
        frame[LSM_COLUMN].le(ocean_lsm_maximum),
        ["block", "time_utc", *ERA5_VALUE_COLUMNS],
    ].copy()
    selected["time_utc"] = pd.to_datetime(selected["time_utc"], utc=True, errors="raise")
    if selected.empty:
        raise ValueError("ERA5 ocean-cell mask selected no rows")
    counts = selected.groupby(["block", "time_utc"], sort=True).size()
    if expected_ocean_cells_per_hour is not None and not counts.eq(
        expected_ocean_cells_per_hour
    ).all():
        raise ValueError("ERA5 ocean-cell count per hour changed")

    aggregate_columns = [name for name in ERA5_VALUE_COLUMNS if name != LSM_COLUMN]
    hourly = (
        selected.groupby(["block", "time_utc"], sort=True, as_index=False)[aggregate_columns]
        .mean()
        .sort_values(["block", "time_utc"], kind="stable")
        .reset_index(drop=True)
    )
    panels: list[pd.DataFrame] = []
    for block, current in hourly.groupby("block", sort=True):
        current = current.sort_values("time_utc", kind="stable").reset_index(drop=True)
        cadence = current["time_utc"].diff().dropna()
        if not cadence.eq(pd.Timedelta(hours=1)).all():
            raise ValueError(f"ERA5 hourly cadence changed within block {block}")

        tau_east = current[TAU_EAST_COLUMN].to_numpy(float) / HOUR_SECONDS
        tau_north = current[TAU_NORTH_COLUMN].to_numpy(float) / HOUR_SECONDS
        tau_magnitude = np.hypot(tau_east, tau_north)
        qnet_native_jm2 = current[[SSR_COLUMN, STR_COLUMN, SLHF_COLUMN, SSHF_COLUMN]].sum(
            axis=1
        )
        panel = current.loc[:, ["block", "time_utc"]].copy()
        panel["era5_u10_ms"] = current[U10_COLUMN].to_numpy(float)
        panel["era5_v10_ms"] = current[V10_COLUMN].to_numpy(float)
        panel["era5_tau_mag_nm2"] = tau_magnitude
        panel["era5_tau_dir_sin"] = np.divide(
            tau_north,
            tau_magnitude,
            out=np.zeros_like(tau_magnitude),
            where=tau_magnitude > 0.0,
        )
        panel["era5_tau_dir_cos"] = np.divide(
            tau_east,
            tau_magnitude,
            out=np.zeros_like(tau_magnitude),
            where=tau_magnitude > 0.0,
        )
        panel["era5_qnet_native_wm2"] = qnet_native_jm2.to_numpy(float) / HOUR_SECONDS

        tau_series = panel["era5_tau_mag_nm2"]
        qnet_flux = panel["era5_qnet_native_wm2"]
        for window in WINDOW_HOURS:
            panel[f"era5_u10_mean_{window}h_ms"] = (
                panel["era5_u10_ms"].rolling(window, min_periods=window).mean()
            )
            panel[f"era5_v10_mean_{window}h_ms"] = (
                panel["era5_v10_ms"].rolling(window, min_periods=window).mean()
            )
            panel[f"era5_tau_mag_impulse_{window}h_nsm2"] = (
                tau_series.rolling(window, min_periods=window).sum() * HOUR_SECONDS
            )
            panel[f"era5_qnet_energy_{window}h_jm2"] = qnet_native_jm2.rolling(
                window, min_periods=window
            ).sum()
            panel[f"era5_tau_mag_trend_{window}h_nm2_per_h"] = _rolling_ols_slope(
                tau_series, window
            )
            panel[f"era5_qnet_trend_{window}h_wm2_per_h"] = _rolling_ols_slope(
                qnet_flux, window
            )
        panels.append(panel.loc[:, ["block", "time_utc", *ERA5_MIXING_FEATURES]])

    result = pd.concat(panels, ignore_index=True)
    if result[["block", "time_utc"]].duplicated().any():
        raise ValueError("ERA5 hourly feature panel contains duplicate keys")
    return result


def align_mixing_features_to_oof_keys(
    hourly_features: pd.DataFrame,
    keys: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Causally align hourly ERA5 features to 10-minute OOF keys within block."""

    required_keys = {"time", "layer", "block"}
    if not required_keys.issubset(keys.columns):
        raise ValueError("OOF key schema is incomplete")
    if tuple(name for name in ERA5_MIXING_FEATURES if name not in hourly_features.columns):
        raise ValueError("hourly ERA5 feature panel is incomplete")
    if keys[["time", "layer", "block"]].duplicated().any():
        raise ValueError("OOF keys are not unique")

    left = keys.loc[:, ["time", "layer", "block"]].copy()
    left["_row_order"] = np.arange(len(left), dtype=np.int64)
    left["_key_time_utc"] = pd.to_datetime(left["time"], utc=True, errors="raise")
    left["_era5_hour_utc"] = left["_key_time_utc"].dt.floor("h")
    right = hourly_features.rename(columns={"time_utc": "_era5_hour_utc"})
    merged = left.merge(
        right,
        on=["block", "_era5_hour_utc"],
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("_row_order", kind="stable")
    feature_missing = merged.loc[:, ERA5_MIXING_FEATURES].isna().any(axis=1)
    matched_rows = int((~feature_missing).sum())
    coverage = matched_rows / len(merged) if len(merged) else 0.0
    age_minutes = (
        merged["_key_time_utc"] - merged["_era5_hour_utc"]
    ).dt.total_seconds().div(60)
    if (age_minutes < 0).any() or (age_minutes >= 60).any():
        raise ValueError("ERA5 alignment is not a causal backward hour join")
    if matched_rows != len(merged):
        raise ValueError("ERA5 features do not cover every OOF key")

    result = merged.loc[:, ["time", "layer", "block", *ERA5_MIXING_FEATURES]].reset_index(
        drop=True
    )
    audit = {
        "rows": int(len(result)),
        "matched_rows": matched_rows,
        "join_coverage": float(coverage),
        "feature_count": len(ERA5_MIXING_FEATURES),
        "missing_feature_values": int(result.loc[:, ERA5_MIXING_FEATURES].isna().sum().sum()),
        "minimum_alignment_age_minutes": float(age_minutes.min()),
        "maximum_alignment_age_minutes": float(age_minutes.max()),
        "future_era5_rows": 0,
    }
    return result, audit


def convex_two_expert_blend(
    deep_prediction: np.ndarray,
    physical_prediction: np.ndarray,
    physical_weight: np.ndarray,
) -> np.ndarray:
    """Apply the only allowed final form: a non-extrapolating convex blend."""

    deep = np.asarray(deep_prediction, dtype=np.float64)
    physical = np.asarray(physical_prediction, dtype=np.float64)
    weight = np.asarray(physical_weight, dtype=np.float64)
    if deep.shape != physical.shape or deep.shape != weight.shape:
        raise ValueError("two-expert blend arrays are not aligned")
    if not np.isfinite(deep).all() or not np.isfinite(physical).all():
        raise ValueError("two-expert predictions contain non-finite values")
    if not np.isfinite(weight).all() or np.any((weight < 0.0) | (weight > 1.0)):
        raise ValueError("physical expert weight is outside [0, 1]")
    prediction = (1.0 - weight) * deep + weight * physical
    lower = np.minimum(deep, physical)
    upper = np.maximum(deep, physical)
    if np.any(prediction < lower - 1e-12) or np.any(prediction > upper + 1e-12):
        raise AssertionError("convex blend extrapolated beyond its two experts")
    return prediction
