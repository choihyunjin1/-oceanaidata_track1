"""Credential, runtime, field, and causal-alignment preflight checks for ERA5."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from p2_restore.era5_request import (
    ANCILLARY_VARIABLES,
    AREA_3X3,
    ERA5_VARIABLES,
    SORS_LATITUDE,
    SORS_LONGITUDE,
)

CDS_API_URL_DEFAULT = "https://cds.climate.copernicus.eu/api"
REQUIRED_SETTING_NAMES = ("CDSAPI_KEY", "ERA5_CDS_TERMS_ACCEPTED")
OPTIONAL_SETTING_NAMES = ("CDSAPI_URL",)
TRUE_VALUES = {"1", "true", "yes", "accepted"}


@dataclass(frozen=True)
class CredentialPreflight:
    status: str
    token_present: bool
    terms_accepted: bool
    api_url_valid: bool
    cdsapi_installed: bool
    grib_reader_available: bool
    netcdf_reader_available: bool
    required_setting_names: tuple[str, ...] = REQUIRED_SETTING_NAMES
    optional_setting_names: tuple[str, ...] = OPTIONAL_SETTING_NAMES

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["ready"] = self.ready
        return value


@dataclass(frozen=True)
class Era5Field:
    values: np.ndarray
    units: str
    standard_name: str


def _available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def credential_preflight(environment: Mapping[str, str] | None = None) -> CredentialPreflight:
    env = os.environ if environment is None else environment
    token_present = bool(env.get("CDSAPI_KEY", "").strip())
    terms_accepted = env.get("ERA5_CDS_TERMS_ACCEPTED", "").strip().lower() in TRUE_VALUES
    api_url = env.get("CDSAPI_URL", CDS_API_URL_DEFAULT).strip().rstrip("/")
    api_url_valid = api_url == CDS_API_URL_DEFAULT
    cdsapi_installed = _available("cdsapi")
    grib_reader = _available("cfgrib") and _available("eccodes")
    netcdf_reader = _available("xarray") and any(
        _available(name) for name in ("netCDF4", "h5netcdf", "scipy")
    )
    if not token_present or not terms_accepted or not api_url_valid:
        status = "awaiting_credential"
    elif not cdsapi_installed or not (grib_reader or netcdf_reader):
        status = "awaiting_runtime"
    else:
        status = "ready"
    return CredentialPreflight(
        status=status,
        token_present=token_present,
        terms_accepted=terms_accepted,
        api_url_valid=api_url_valid,
        cdsapi_installed=cdsapi_installed,
        grib_reader_available=grib_reader,
        netcdf_reader_available=netcdf_reader,
    )


def preferred_format_order(preflight: CredentialPreflight) -> tuple[str, ...]:
    order: list[str] = []
    if preflight.grib_reader_available:
        order.append("grib")
    if preflight.netcdf_reader_available:
        order.append("netcdf")
    return tuple(order)


def _normalized_unit(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "")
        .replace("**", "")
        .replace("^", "")
        .replace("{", "")
        .replace("}", "")
        .replace("(", "")
        .replace(")", "")
        .replace("−", "-")
    )


def _validate_units(name: str, units: str) -> None:
    accepted = {
        "10m_u_component_of_wind": {"ms-1", "m/s"},
        "10m_v_component_of_wind": {"ms-1", "m/s"},
        "eastward_turbulent_surface_stress": {"nm-2s"},
        "northward_turbulent_surface_stress": {"nm-2s"},
        "surface_net_solar_radiation": {"jm-2", "j/m2"},
        "surface_net_thermal_radiation": {"jm-2", "j/m2"},
        "surface_latent_heat_flux": {"jm-2", "j/m2"},
        "surface_sensible_heat_flux": {"jm-2", "j/m2"},
        "land_sea_mask": {"1", "0-1", "dimensionless"},
    }
    if _normalized_unit(units) not in accepted[name]:
        raise ValueError(f"unexpected ERA5 units for {name}: {units}")


def _three_dimensional(values: np.ndarray, time_count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and name in ANCILLARY_VARIABLES:
        array = np.broadcast_to(array, (time_count, *array.shape))
    if array.shape != (time_count, 3, 3):
        raise ValueError(f"ERA5 {name} must have shape (time, 3, 3)")
    return array


def validate_era5_fields(
    fields: Mapping[str, Era5Field],
    *,
    times_utc: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    require_24h_smoke: bool = False,
) -> dict[str, Any]:
    expected_fields = {*ERA5_VARIABLES, *ANCILLARY_VARIABLES}
    if set(fields) != expected_fields:
        raise ValueError("ERA5 field set differs from eight variables plus land-mask ancillary")
    time = pd.DatetimeIndex(times_utc)
    if time.tz is None:
        raise ValueError("ERA5 valid times must be explicitly UTC")
    time = time.tz_convert("UTC")
    if time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError("ERA5 valid times must be unique and sorted")
    if len(time) > 1 and not (time.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("ERA5 valid times are not hourly")
    if require_24h_smoke and (
        len(time) != 24 or tuple(time.hour) != tuple(range(24)) or len(set(time.date)) != 1
    ):
        raise ValueError("ERA5 smoke must contain one UTC day with exactly 24 hours")
    latitude = np.asarray(latitudes, dtype=float)
    longitude = np.asarray(longitudes, dtype=float)
    expected_latitude = np.array([AREA_3X3[0], (AREA_3X3[0] + AREA_3X3[2]) / 2, AREA_3X3[2]])
    expected_longitude = np.array([AREA_3X3[1], (AREA_3X3[1] + AREA_3X3[3]) / 2, AREA_3X3[3]])
    if latitude.shape != (3,) or not np.allclose(np.sort(latitude), np.sort(expected_latitude)):
        raise ValueError("ERA5 latitude coordinates are not the fixed S-ORS 3-cell axis")
    if longitude.shape != (3,) or not np.allclose(np.sort(longitude), expected_longitude):
        raise ValueError("ERA5 longitude coordinates are not the fixed S-ORS 3-cell axis")

    expected_standard_names = {
        "10m_u_component_of_wind": "",
        "10m_v_component_of_wind": "",
        "eastward_turbulent_surface_stress": "surface_downward_eastward_stress",
        "northward_turbulent_surface_stress": "surface_downward_northward_stress",
        "surface_net_solar_radiation": "surface_net_downward_shortwave_flux",
        "surface_net_thermal_radiation": "surface_net_upward_longwave_flux",
        "surface_latent_heat_flux": "surface_upward_latent_heat_flux",
        "surface_sensible_heat_flux": "surface_upward_sensible_heat_flux",
        "land_sea_mask": "land_binary_mask",
    }
    arrays: dict[str, np.ndarray] = {}
    for name in (*ERA5_VARIABLES, *ANCILLARY_VARIABLES):
        _validate_units(name, fields[name].units)
        if fields[name].standard_name != expected_standard_names[name]:
            raise ValueError(f"unexpected ERA5 sign convention metadata for {name}")
        arrays[name] = _three_dimensional(fields[name].values, len(time), name)
    land = arrays["land_sea_mask"]
    if not np.isfinite(land).all() or np.nanmin(land) < 0 or np.nanmax(land) > 1:
        raise ValueError("ERA5 land-sea mask is outside [0, 1]")
    latitude_index = int(np.argmin(np.abs(latitude - SORS_LATITUDE)))
    longitude_index = int(np.argmin(np.abs(longitude - SORS_LONGITUDE)))
    center_land_fraction = float(np.mean(land[:, latitude_index, longitude_index]))
    if center_land_fraction >= 0.5:
        raise ValueError("ERA5 grid cell nearest S-ORS is not classified as ocean")

    for name in ERA5_VARIABLES:
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"ERA5 {name} contains non-finite values")
    bounds = {
        "10m_u_component_of_wind": (-100.0, 100.0),
        "10m_v_component_of_wind": (-100.0, 100.0),
        "eastward_turbulent_surface_stress": (-100_000.0, 100_000.0),
        "northward_turbulent_surface_stress": (-100_000.0, 100_000.0),
        "surface_net_solar_radiation": (-1e-3, 500_000_000.0),
        "surface_net_thermal_radiation": (-500_000_000.0, 500_000_000.0),
        "surface_latent_heat_flux": (-500_000_000.0, 500_000_000.0),
        "surface_sensible_heat_flux": (-500_000_000.0, 500_000_000.0),
    }
    for name, (lower, upper) in bounds.items():
        if np.nanmin(arrays[name]) < lower or np.nanmax(arrays[name]) > upper:
            raise ValueError(f"ERA5 {name} violates the fixed physical/sign bounds")
    return {
        "passed": True,
        "time_count": len(time),
        "grid_shape": [3, 3],
        "variable_count": len(ERA5_VARIABLES),
        "validation_ancillary_count": len(ANCILLARY_VARIABLES),
        "center_land_fraction": center_land_fraction,
        "semantics": {
            "10m_u_component_of_wind": "positive eastward",
            "10m_v_component_of_wind": "positive northward",
            "eastward_turbulent_surface_stress": "positive eastward accumulated N m-2 s",
            "northward_turbulent_surface_stress": "positive northward accumulated N m-2 s",
            "surface_net_solar_radiation": "net downward accumulated J m-2",
            "surface_net_thermal_radiation": "net upward accumulated J m-2",
            "surface_latent_heat_flux": "upward accumulated J m-2",
            "surface_sensible_heat_flux": "upward accumulated J m-2",
        },
    }


def causal_align_utc_to_kst(
    hourly: pd.DataFrame,
    keys: pd.DataFrame,
    *,
    tolerance_minutes: int = 70,
) -> pd.DataFrame:
    if not isinstance(hourly.index, pd.DatetimeIndex) or hourly.index.tz is None:
        raise ValueError("ERA5 hourly index must be timezone-aware UTC")
    source = hourly.copy()
    source.index = source.index.tz_convert("UTC")
    if source.index.duplicated().any() or not source.index.is_monotonic_increasing:
        raise ValueError("ERA5 hourly index must be unique and sorted")
    if "time" not in keys:
        raise ValueError("P2 alignment keys must contain time")
    source = source.reset_index(names="source_time_utc")
    left = keys.reset_index(drop=True).copy()
    left["_row"] = np.arange(len(left), dtype=np.int64)
    left["key_time_utc"] = pd.to_datetime(left["time"], utc=True, errors="raise")
    merged = pd.merge_asof(
        left.sort_values("key_time_utc"),
        source.sort_values("source_time_utc"),
        left_on="key_time_utc",
        right_on="source_time_utc",
        direction="backward",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
        allow_exact_matches=True,
    ).sort_values("_row")
    if merged["source_time_utc"].isna().any():
        raise ValueError("ERA5 causal alignment left uncovered P2 rows")
    if (merged["source_time_utc"] > merged["key_time_utc"]).any():
        raise AssertionError("ERA5 causal alignment used a future atmospheric value")
    merged["source_time_kst"] = merged["source_time_utc"].dt.tz_convert("Asia/Seoul")
    merged["source_lag_minutes"] = (
        merged["key_time_utc"] - merged["source_time_utc"]
    ).dt.total_seconds() / 60
    return merged.drop(columns="_row").reset_index(drop=True)
