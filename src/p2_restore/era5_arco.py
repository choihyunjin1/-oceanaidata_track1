"""Anonymous Google Research ARCO-ERA5 metadata and bounded smoke access."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from p2_restore.era5_preflight import Era5Field, validate_era5_fields
from p2_restore.era5_request import (
    ANCILLARY_VARIABLES,
    AREA_3X3,
    ERA5_VARIABLES,
)

ARCO_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ARCO_GCS_PATH = ARCO_URI.removeprefix("gs://")
ARCO_OFFICIAL_REPOSITORY = "https://github.com/google-research/arco-era5"
GOOGLE_PUBLIC_DATASETS_URL = "https://docs.cloud.google.com/storage/docs/public-datasets"
COPERNICUS_LICENSE_URL = (
    "https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf"
)
ERA5_DOI = "10.24381/cds.adbb2d47"
SMOKE_DAY_UTC = "2024-09-01"
MAX_SMOKE_GIB = 0.75
CATALOG_SHA256 = "08645af1c6238fff256d60580f99154ac89070d655bff5e70ca11925c1cb52a8"


@dataclass(frozen=True)
class ArcoMetadataReport:
    metadata_sha256: str
    metadata_generation: str | None
    valid_time_start: str
    valid_time_stop: str
    last_updated: str
    variable_count: int
    ancillary_count: int
    time_index: int
    compressed_one_hour_bytes: int
    estimated_smoke_bytes: int
    estimated_smoke_gib: float
    chunk_shape: tuple[int, int, int]
    passed: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "metadata_sha256": self.metadata_sha256,
            "metadata_generation": self.metadata_generation,
            "valid_time_start": self.valid_time_start,
            "valid_time_stop": self.valid_time_stop,
            "last_updated": self.last_updated,
            "variable_count": self.variable_count,
            "ancillary_count": self.ancillary_count,
            "time_index": self.time_index,
            "compressed_one_hour_bytes": self.compressed_one_hour_bytes,
            "estimated_smoke_bytes": self.estimated_smoke_bytes,
            "estimated_smoke_gib": self.estimated_smoke_gib,
            "chunk_shape": list(self.chunk_shape),
            "passed": self.passed,
        }


def _time_index(day_utc: str) -> int:
    start = np.datetime64("1900-01-01T00", "h")
    target = np.datetime64(f"{day_utc}T00", "h")
    return int((target - start) / np.timedelta64(1, "h"))


def _expected_standard_names() -> dict[str, str]:
    return {
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


def validate_arco_metadata(
    raw_metadata: bytes,
    *,
    object_sizes: Mapping[str, int],
    metadata_generation: str | None,
    smoke_day_utc: str = SMOKE_DAY_UTC,
) -> ArcoMetadataReport:
    payload = json.loads(raw_metadata)
    metadata = payload["metadata"]
    root = metadata[".zattrs"]
    target = pd.Timestamp(smoke_day_utc, tz="UTC")
    if not (
        pd.Timestamp(root["valid_time_start"], tz="UTC")
        <= target
        <= pd.Timestamp(root["valid_time_stop"], tz="UTC")
    ):
        raise ValueError("ARCO stable ERA5 range does not cover the smoke day")
    variables = (*ERA5_VARIABLES, *ANCILLARY_VARIABLES)
    standards = _expected_standard_names()
    for name in variables:
        array = metadata.get(f"{name}/.zarray")
        attributes = metadata.get(f"{name}/.zattrs")
        if array is None or attributes is None:
            raise ValueError(f"ARCO variable is absent: {name}")
        if array.get("chunks") != [1, 721, 1440] or array.get("dtype") != "<f4":
            raise ValueError(f"ARCO chunk/dtype contract changed: {name}")
        if attributes.get("_ARRAY_DIMENSIONS") != ["time", "latitude", "longitude"]:
            raise ValueError(f"ARCO dimensions changed: {name}")
        if standards[name] and attributes.get("standard_name") != standards[name]:
            raise ValueError(f"ARCO sign convention metadata changed: {name}")
        if name == "10m_u_component_of_wind" and (
            attributes.get("short_name") != "u10"
            or attributes.get("long_name") != "10 metre U wind component"
        ):
            raise ValueError("ARCO eastward wind identity metadata changed")
        if name == "10m_v_component_of_wind" and (
            attributes.get("short_name") != "v10"
            or attributes.get("long_name") != "10 metre V wind component"
        ):
            raise ValueError("ARCO northward wind identity metadata changed")
        if name not in object_sizes or int(object_sizes[name]) <= 0:
            raise ValueError(f"ARCO smoke object size is unavailable: {name}")
    index = _time_index(smoke_day_utc)
    time_shape = int(metadata["time/.zarray"]["shape"][0])
    if index < 0 or index + 23 >= time_shape:
        raise ValueError("ARCO smoke time index is outside the store")
    feature_hour_bytes = sum(int(object_sizes[name]) for name in ERA5_VARIABLES)
    ancillary_bytes = sum(int(object_sizes[name]) for name in ANCILLARY_VARIABLES)
    estimated = feature_hour_bytes * 24 + ancillary_bytes
    estimated_gib = estimated / 2**30
    if estimated_gib > MAX_SMOKE_GIB:
        raise ValueError("ARCO one-day smoke exceeds the fixed 0.75 GiB read gate")
    return ArcoMetadataReport(
        metadata_sha256=hashlib.sha256(raw_metadata).hexdigest(),
        metadata_generation=metadata_generation,
        valid_time_start=str(root["valid_time_start"]),
        valid_time_stop=str(root["valid_time_stop"]),
        last_updated=str(root["last_updated"]),
        variable_count=len(ERA5_VARIABLES),
        ancillary_count=len(ANCILLARY_VARIABLES),
        time_index=index,
        compressed_one_hour_bytes=feature_hour_bytes,
        estimated_smoke_bytes=estimated,
        estimated_smoke_gib=estimated_gib,
        chunk_shape=(1, 721, 1440),
        passed=True,
    )


def inspect_anonymous_arco(smoke_day_utc: str = SMOKE_DAY_UTC) -> ArcoMetadataReport:
    import gcsfs

    filesystem = gcsfs.GCSFileSystem(token="anon")
    metadata_path = f"{ARCO_GCS_PATH}/.zmetadata"
    raw_metadata = filesystem.cat(metadata_path)
    index = _time_index(smoke_day_utc)
    names = (*ERA5_VARIABLES, *ANCILLARY_VARIABLES)
    object_sizes = {
        name: int(filesystem.info(f"{ARCO_GCS_PATH}/{name}/{index}.0.0")["size"]) for name in names
    }
    metadata_generation = str(filesystem.info(metadata_path).get("generation") or "") or None
    return validate_arco_metadata(
        raw_metadata,
        object_sizes=object_sizes,
        metadata_generation=metadata_generation,
        smoke_day_utc=smoke_day_utc,
    )


def read_anonymous_arco_smoke(
    report: ArcoMetadataReport,
    *,
    smoke_day_utc: str = SMOKE_DAY_UTC,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not report.passed or report.estimated_smoke_gib > MAX_SMOKE_GIB:
        raise ValueError("ARCO metadata gate did not authorize the one-day smoke")
    import gcsfs
    import zarr

    filesystem = gcsfs.GCSFileSystem(token="anon")
    group = zarr.open_group(
        filesystem.get_mapper(ARCO_GCS_PATH),
        mode="r",
        use_consolidated=True,
    )
    latitude = np.asarray(group["latitude"][:], dtype=float)
    longitude = np.asarray(group["longitude"][:], dtype=float)
    requested_latitude = np.array([AREA_3X3[0], 37.50, AREA_3X3[2]])
    requested_longitude = np.array([AREA_3X3[1], 124.75, AREA_3X3[3]])
    latitude_index = np.array(
        [int(np.argmin(np.abs(latitude - value))) for value in requested_latitude]
    )
    longitude_index = np.array(
        [int(np.argmin(np.abs(longitude - value))) for value in requested_longitude]
    )
    selected_latitude = latitude[latitude_index]
    selected_longitude = longitude[longitude_index]
    if not np.allclose(selected_latitude, requested_latitude) or not np.allclose(
        selected_longitude, requested_longitude
    ):
        raise ValueError("ARCO grid does not contain the fixed S-ORS 3x3 coordinates")
    time_index = np.arange(report.time_index, report.time_index + 24, dtype=np.int64)
    times = pd.date_range(f"{smoke_day_utc}T00:00:00Z", periods=24, freq="h")
    fields: dict[str, Era5Field] = {}
    values: dict[str, np.ndarray] = {}
    standards = _expected_standard_names()
    for name in ERA5_VARIABLES:
        array = group[name]
        selected = np.asarray(
            array.get_orthogonal_selection((time_index, latitude_index, longitude_index)),
            dtype=np.float64,
        )
        values[name] = selected
        fields[name] = Era5Field(selected, str(array.attrs["units"]), standards[name])
    land_array = group[ANCILLARY_VARIABLES[0]]
    land_single = np.asarray(
        land_array.get_orthogonal_selection(
            (np.array([report.time_index]), latitude_index, longitude_index)
        ),
        dtype=np.float64,
    )[0]
    land = np.broadcast_to(land_single, (24, 3, 3)).copy()
    values["land_sea_mask"] = land
    fields["land_sea_mask"] = Era5Field(
        land,
        str(land_array.attrs["units"]),
        standards["land_sea_mask"],
    )
    validation = validate_era5_fields(
        fields,
        times_utc=times,
        latitudes=selected_latitude,
        longitudes=selected_longitude,
        require_24h_smoke=True,
    )
    index = pd.MultiIndex.from_product(
        [times, selected_latitude, selected_longitude],
        names=["time_utc", "latitude", "longitude"],
    )
    frame = pd.DataFrame(
        {name: array.reshape(-1) for name, array in values.items()},
        index=index,
    ).reset_index()
    validation["rows"] = len(frame)
    validation["source"] = ARCO_URI
    validation["anonymous_access"] = True
    return frame, validation
