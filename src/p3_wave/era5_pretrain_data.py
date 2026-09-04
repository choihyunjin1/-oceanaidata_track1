"""Quarantined, pre-2024 ERA5 retrieval and processing for P3 research.

The module never reads P3 competition inputs.  It first requests one UTC day on
a 3 x 3 ERA5 grid around each official station coordinate, selects the nearest
cell that is ocean and has sufficiently complete fields, and only then permits
single-cell monthly transport requests for 2014--2023.

Network access is fail-closed: :func:`retrieve_cds_request` cannot instantiate a
CDS client unless ``execute_download=True`` is passed by the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from calendar import monthrange
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

DATASET_ID: Final = "reanalysis-era5-single-levels"
DATASET_URL: Final = (
    "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview"
)
DATASET_DOI: Final = "10.24381/cds.adbb2d47"
LICENSE_NAME: Final = "Copernicus Products Licence"
LICENSE_URL: Final = (
    "https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf"
)
ATTRIBUTION: Final = (
    "Contains modified Copernicus Climate Change Service information; neither the "
    "European Commission nor ECMWF is responsible for downstream use."
)

GRID_DEGREES: Final = 0.25
SMOKE_DAY_UTC: Final = date(2023, 12, 30)
START_YEAR: Final = 2014
END_YEAR: Final = 2023
CUTOFF_EXCLUSIVE_UTC: Final = pd.Timestamp("2023-12-31T15:00:00Z")
LAST_ELIGIBLE_ANCHOR_UTC: Final = pd.Timestamp("2023-12-30T14:00:00Z")
OCEAN_LSM_MAXIMUM: Final = 0.5
MINIMUM_FINITE_FRACTION: Final = 0.90
MAX_YEAR_REQUEST_FIELD_HOURS: Final = 7_440  # 31 days x 24 hours x 10 fields
QUARANTINE_RELATIVE: Final = Path("external_data/quarantine/era5_p3_context_pretrain_v1")

# Public short name -> CDS request name.  Order is part of the frozen contract.
VARIABLES: Final = {
    "swh": "significant_height_of_combined_wind_waves_and_swell",
    "mwp": "mean_wave_period",
    "hmax": "maximum_individual_wave_height",
    "mwd": "mean_wave_direction",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "land_sea_mask": "land_sea_mask",
}
DYNAMIC_VARIABLES: Final = tuple(name for name in VARIABLES if name != "land_sea_mask")


class Era5PretrainError(RuntimeError):
    """Base exception for the isolated P3 ERA5 path."""


class DownloadAuthorizationError(Era5PretrainError):
    """Raised before CDS client creation when explicit authorization is absent."""


class Era5SchemaError(Era5PretrainError):
    """Raised when downloaded ERA5 content violates the fixed contract."""


@dataclass(frozen=True)
class StationPoint:
    station: str
    latitude: float
    longitude: float

    def public_dict(self) -> dict[str, object]:
        return {
            "station": self.station,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "coordinate_source": "KIOST official KORS station introduction",
            "crs": "WGS84 (EPSG:4326)",
        }


STATIONS: Final = {
    "G-ORS": StationPoint("G-ORS", 33.9428, 124.5919),
    "I-ORS": StationPoint("I-ORS", 32.1228, 125.1822),
    "S-ORS": StationPoint("S-ORS", 37.4231, 124.7380),
}


def _nearest_grid_coordinate(value: float) -> float:
    # Avoid Python's banker rounding at exact half-grid positions.
    return round(math.floor(value / GRID_DEGREES + 0.5) * GRID_DEGREES, 6)


def smoke_area(station: StationPoint) -> tuple[float, float, float, float]:
    """Return a CDS north/west/south/east area containing exactly 3 x 3 cells."""

    center_latitude = _nearest_grid_coordinate(station.latitude)
    center_longitude = _nearest_grid_coordinate(station.longitude)
    return (
        round(center_latitude + GRID_DEGREES, 6),
        round(center_longitude - GRID_DEGREES, 6),
        round(center_latitude - GRID_DEGREES, 6),
        round(center_longitude + GRID_DEGREES, 6),
    )


@dataclass(frozen=True)
class Era5Request:
    request_id: str
    station: str
    purpose: str
    area: tuple[float, float, float, float]
    year: int
    months: tuple[int, ...]
    days: tuple[int, ...]
    hours: tuple[int, ...]
    requested_start_utc: str
    requested_end_utc: str

    def __post_init__(self) -> None:
        if self.station not in STATIONS:
            raise ValueError(f"unknown P3 station: {self.station}")
        if self.purpose not in {"smoke_3x3", "selected_cell_year"}:
            raise ValueError(f"unsupported ERA5 request purpose: {self.purpose}")
        if not START_YEAR <= self.year <= END_YEAR:
            raise ValueError("ERA5 P3 requests must stay in 2014--2023")
        if not self.months or not self.days or not self.hours:
            raise ValueError("ERA5 request time axes must not be empty")
        if min(self.months) < 1 or max(self.months) > 12:
            raise ValueError("ERA5 request contains an invalid month")
        if min(self.days) < 1 or max(self.days) > 31:
            raise ValueError("ERA5 request contains an invalid day")
        if min(self.hours) < 0 or max(self.hours) > 23:
            raise ValueError("ERA5 request contains an invalid UTC hour")
        start = pd.Timestamp(self.requested_start_utc)
        end = pd.Timestamp(self.requested_end_utc)
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("ERA5 request boundaries must be timezone-aware")
        start = start.tz_convert("UTC")
        end = end.tz_convert("UTC")
        if start > end or end >= CUTOFF_EXCLUSIVE_UTC:
            raise ValueError("ERA5 request crossed the approved UTC cutoff")
        if start.year != self.year or end.year != self.year:
            raise ValueError("one ERA5 request must stay within one calendar year")
        north, west, south, east = self.area
        if north < south or east < west:
            raise ValueError("ERA5 CDS area has invalid bounds")
        if self.purpose == "smoke_3x3":
            if not (
                math.isclose(north - south, 2 * GRID_DEGREES)
                and math.isclose(east - west, 2 * GRID_DEGREES)
            ):
                raise ValueError("smoke request must be exactly 3 x 3 at 0.25 degrees")
        elif not (math.isclose(north, south) and math.isclose(west, east)):
            raise ValueError("selected-cell request must target exactly one cell")

    @property
    def start_utc(self) -> pd.Timestamp:
        return pd.Timestamp(self.requested_start_utc).tz_convert("UTC")

    @property
    def end_utc(self) -> pd.Timestamp:
        return pd.Timestamp(self.requested_end_utc).tz_convert("UTC")

    def request(self) -> dict[str, Any]:
        if self.end_utc >= CUTOFF_EXCLUSIVE_UTC:
            raise AssertionError("ERA5 request crossed the pre-2024 cutoff")
        return {
            "product_type": ["reanalysis"],
            "variable": list(VARIABLES.values()),
            "year": [f"{self.year:04d}"],
            "month": [f"{value:02d}" for value in self.months],
            "day": [f"{value:02d}" for value in self.days],
            "time": [f"{value:02d}:00" for value in self.hours],
            "area": list(self.area),
            "grid": [GRID_DEGREES, GRID_DEGREES],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "station": self.station,
            "purpose": self.purpose,
            "time_start_utc": self.start_utc.isoformat(),
            "time_end_utc": self.end_utc.isoformat(),
            "request": self.request(),
        }


def build_smoke_plan() -> tuple[Era5Request, ...]:
    result = []
    for station in STATIONS.values():
        result.append(
            Era5Request(
                request_id=f"smoke_{station.station.lower().replace('-', '_')}_20231230",
                station=station.station,
                purpose="smoke_3x3",
                area=smoke_area(station),
                year=SMOKE_DAY_UTC.year,
                months=(SMOKE_DAY_UTC.month,),
                days=(SMOKE_DAY_UTC.day,),
                hours=tuple(range(24)),
                requested_start_utc="2023-12-30T00:00:00Z",
                requested_end_utc="2023-12-30T23:00:00Z",
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class SelectedCell:
    station: str
    station_latitude: float
    station_longitude: float
    latitude: float
    longitude: float
    distance_km: float
    mean_land_sea_mask: float
    finite_fraction: Mapping[str, float]

    def __post_init__(self) -> None:
        expected = STATIONS.get(self.station)
        if expected is None:
            raise ValueError(f"unknown P3 station selection: {self.station}")
        if not math.isclose(self.station_latitude, expected.latitude, abs_tol=1e-8):
            raise ValueError("selected-cell station latitude changed")
        if not math.isclose(self.station_longitude, expected.longitude, abs_tol=1e-8):
            raise ValueError("selected-cell station longitude changed")
        if not 0.0 <= self.mean_land_sea_mask <= OCEAN_LSM_MAXIMUM:
            raise ValueError("selected ERA5 cell is not ocean")
        if set(self.finite_fraction) != set(DYNAMIC_VARIABLES):
            raise ValueError("selected ERA5 cell finite-fraction schema changed")
        if min(self.finite_fraction.values()) < MINIMUM_FINITE_FRACTION:
            raise ValueError("selected ERA5 cell lacks complete smoke fields")
        if not (
            math.isclose(self.latitude / GRID_DEGREES, round(self.latitude / GRID_DEGREES))
            and math.isclose(self.longitude / GRID_DEGREES, round(self.longitude / GRID_DEGREES))
        ):
            raise ValueError("selected ERA5 coordinates are not on the 0.25-degree grid")
        north, west, south, east = smoke_area(expected)
        if not (south <= self.latitude <= north and west <= self.longitude <= east):
            raise ValueError("selected ERA5 coordinates escape the station smoke grid")

    def public_dict(self) -> dict[str, Any]:
        return {
            "station": self.station,
            "station_latitude": self.station_latitude,
            "station_longitude": self.station_longitude,
            "selected_latitude": self.latitude,
            "selected_longitude": self.longitude,
            "distance_km": self.distance_km,
            "mean_land_sea_mask": self.mean_land_sea_mask,
            "finite_fraction": dict(self.finite_fraction),
            "selection_rule": (
                "minimum geodesic distance among smoke cells with mean land_sea_mask "
                "<=0.5 and >=90% finite values for every dynamic field"
            ),
        }

    @classmethod
    def from_public_dict(cls, payload: Mapping[str, Any]) -> SelectedCell:
        return cls(
            station=str(payload["station"]),
            station_latitude=float(payload["station_latitude"]),
            station_longitude=float(payload["station_longitude"]),
            latitude=float(payload["selected_latitude"]),
            longitude=float(payload["selected_longitude"]),
            distance_km=float(payload["distance_km"]),
            mean_land_sea_mask=float(payload["mean_land_sea_mask"]),
            finite_fraction={
                str(key): float(value) for key, value in dict(payload["finite_fraction"]).items()
            },
        )


def _distance_km(point: StationPoint, latitude: float, longitude: float) -> float:
    radius_km = 6371.0088
    latitude_one = math.radians(point.latitude)
    latitude_two = math.radians(latitude)
    delta_latitude = latitude_two - latitude_one
    delta_longitude = math.radians(longitude - point.longitude)
    haversine = (
        math.sin(delta_latitude / 2.0) ** 2
        + math.cos(latitude_one) * math.cos(latitude_two) * math.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(min(1.0, math.sqrt(haversine)))


def _field_cube(
    values: np.ndarray,
    *,
    time_count: int,
    latitude_count: int,
    longitude_count: int,
    variable: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (latitude_count, longitude_count):
        array = np.broadcast_to(array, (time_count, latitude_count, longitude_count))
    if array.shape != (time_count, latitude_count, longitude_count):
        raise Era5SchemaError(f"ERA5 {variable} shape must be time x latitude x longitude")
    return array


def select_nearest_valid_ocean_cell(
    station_name: str,
    *,
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    fields: Mapping[str, np.ndarray],
) -> SelectedCell:
    """Select one deterministic ocean cell from a validated 24-hour 3 x 3 smoke cube."""

    station = STATIONS.get(station_name)
    if station is None:
        raise ValueError(f"unknown P3 station: {station_name}")
    latitude_values = np.asarray(latitudes, dtype=np.float64)
    longitude_values = np.asarray(longitudes, dtype=np.float64)
    if latitude_values.shape != (3,) or longitude_values.shape != (3,):
        raise Era5SchemaError("ERA5 smoke coordinates must be exactly 3 x 3")
    if not np.isfinite(latitude_values).all() or not np.isfinite(longitude_values).all():
        raise Era5SchemaError("ERA5 smoke coordinates contain non-finite values")
    if set(fields) != set(VARIABLES):
        raise Era5SchemaError("ERA5 smoke variables differ from the frozen ten-field contract")

    candidate_arrays: dict[str, np.ndarray] = {}
    inferred_time_counts = {
        np.asarray(value).shape[0] for value in fields.values() if np.asarray(value).ndim == 3
    }
    if inferred_time_counts != {24}:
        raise Era5SchemaError("ERA5 smoke must contain exactly 24 hourly rows")
    for name, values in fields.items():
        candidate_arrays[name] = _field_cube(
            values,
            time_count=24,
            latitude_count=3,
            longitude_count=3,
            variable=name,
        )

    candidates: list[tuple[float, float, float, float, dict[str, float]]] = []
    land = candidate_arrays["land_sea_mask"]
    for latitude_index, latitude in enumerate(latitude_values):
        for longitude_index, longitude in enumerate(longitude_values):
            land_values = land[:, latitude_index, longitude_index]
            if not np.isfinite(land_values).any():
                continue
            mean_land = float(np.nanmean(land_values))
            finite_fraction = {
                name: float(np.isfinite(values[:, latitude_index, longitude_index]).mean())
                for name, values in candidate_arrays.items()
                if name in DYNAMIC_VARIABLES
            }
            if mean_land > OCEAN_LSM_MAXIMUM:
                continue
            if min(finite_fraction.values()) < MINIMUM_FINITE_FRACTION:
                continue
            candidates.append(
                (
                    _distance_km(station, float(latitude), float(longitude)),
                    float(latitude),
                    float(longitude),
                    mean_land,
                    finite_fraction,
                )
            )
    if not candidates:
        raise Era5SchemaError(f"no valid ocean cell found for {station_name}")
    distance, latitude, longitude, mean_land, finite_fraction = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return SelectedCell(
        station=station.station,
        station_latitude=station.latitude,
        station_longitude=station.longitude,
        latitude=latitude,
        longitude=longitude,
        distance_km=distance,
        mean_land_sea_mask=mean_land,
        finite_fraction=finite_fraction,
    )


def build_year_plan(selections: Mapping[str, SelectedCell]) -> tuple[Era5Request, ...]:
    if set(selections) != set(STATIONS):
        raise ValueError("monthly ERA5 plan requires one selected cell for every P3 station")
    result: list[Era5Request] = []
    for station_name in STATIONS:
        selection = selections[station_name]
        if selection.station != station_name:
            raise ValueError("monthly ERA5 selection key/station mismatch")
        area = (
            selection.latitude,
            selection.longitude,
            selection.latitude,
            selection.longitude,
        )
        for year in range(START_YEAR, END_YEAR):
            for month in range(1, 13):
                last_day = monthrange(year, month)[1]
                result.append(
                    Era5Request(
                        request_id=(
                            f"year_{station_name.lower().replace('-', '_')}_{year:04d}_m{month:02d}"
                        ),
                        station=station_name,
                        purpose="selected_cell_year",
                        area=area,
                        year=year,
                        months=(month,),
                        days=tuple(range(1, 32)),
                        hours=tuple(range(24)),
                        requested_start_utc=(f"{year:04d}-{month:02d}-01T00:00:00Z"),
                        requested_end_utc=(f"{year:04d}-{month:02d}-{last_day:02d}T23:00:00Z"),
                    )
                )
        # Keep 2023 requests monthly through November, then split December so no
        # 2023-12-31 15:00--23:00 UTC values are requested or stored.
        segments = []
        for month in range(1, 12):
            last_day = monthrange(2023, month)[1]
            segments.append(
                (
                    f"m{month:02d}",
                    (month,),
                    tuple(range(1, 32)),
                    tuple(range(24)),
                    f"2023-{month:02d}-01T00:00:00Z",
                    f"2023-{month:02d}-{last_day:02d}T23:00:00Z",
                )
            )
        segments.extend(
            (
                (
                    "dec01_30",
                    (12,),
                    tuple(range(1, 31)),
                    tuple(range(24)),
                    "2023-12-01T00:00:00Z",
                    "2023-12-30T23:00:00Z",
                ),
                (
                    "dec31_h00_14",
                    (12,),
                    (31,),
                    tuple(range(15)),
                    "2023-12-31T00:00:00Z",
                    "2023-12-31T14:00:00Z",
                ),
            )
        )
        for suffix, months, days, hours, start_utc, end_utc in segments:
            result.append(
                Era5Request(
                    request_id=(f"year_{station_name.lower().replace('-', '_')}_2023_{suffix}"),
                    station=station_name,
                    purpose="selected_cell_year",
                    area=area,
                    year=2023,
                    months=months,
                    days=days,
                    hours=hours,
                    requested_start_utc=start_utc,
                    requested_end_utc=end_utc,
                )
            )
    for station_name in STATIONS:
        station_requests = [value for value in result if value.station == station_name]
        if len(station_requests) != 121:
            raise AssertionError("ERA5 transport plan must have 121 chunks per station")
        starts = pd.DatetimeIndex(value.start_utc for value in station_requests)
        ends = pd.DatetimeIndex(value.end_utc for value in station_requests)
        if starts.min() != pd.Timestamp("2014-01-01T00:00:00Z"):
            raise AssertionError("ERA5 monthly plan start changed")
        if ends.max() != pd.Timestamp("2023-12-31T14:00:00Z"):
            raise AssertionError("ERA5 monthly plan cutoff changed")
        for previous, following in zip(ends[:-1], starts[1:], strict=True):
            if following != previous + pd.Timedelta(hours=1):
                raise AssertionError("ERA5 monthly plan has a gap or overlap")
        for request in station_requests:
            if len(request.months) != 1:
                raise AssertionError("ERA5 transport request must contain one month")
            duration_hours = int((request.end_utc - request.start_utc) / pd.Timedelta(hours=1)) + 1
            if duration_hours * len(VARIABLES) > MAX_YEAR_REQUEST_FIELD_HOURS:
                raise AssertionError("ERA5 transport request exceeds the cost ceiling")
    return tuple(result)


@dataclass(frozen=True)
class QuarantineLayout:
    root: Path

    @classmethod
    def from_repo_root(cls, repo_root: str | Path) -> QuarantineLayout:
        repository = Path(repo_root).expanduser().resolve()
        return cls((repository / QUARANTINE_RELATIVE).resolve())

    @property
    def raw_smoke(self) -> Path:
        return self.root / "raw/smoke"

    @property
    def raw_years(self) -> Path:
        return self.root / "raw/yearly"

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    @property
    def derived_years(self) -> Path:
        return self.derived / "yearly"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def ensure(self) -> None:
        for path in (
            self.raw_smoke,
            self.raw_years,
            self.derived,
            self.derived_years,
            self.manifests,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def assert_inside(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError("ERA5 raw/derived output must stay inside its quarantine")
        return resolved

    def raw_path(self, request: Era5Request) -> Path:
        parent = self.raw_smoke if request.purpose == "smoke_3x3" else self.raw_years
        return self.assert_inside(parent / f"{request.request_id}.nc")

    def derived_year_path(self, request: Era5Request) -> Path:
        if request.purpose != "selected_cell_year":
            raise ValueError("only selected-cell requests have derived segment paths")
        return self.assert_inside(self.derived_years / f"{request.request_id}.parquet")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileReceipt:
    request_id: str
    role: str
    relative_path: str
    bytes: int
    sha256: str
    time_start_utc: str
    time_end_utc: str
    row_count: int | None = None

    def public_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "role": self.role,
            "relative_path": self.relative_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "time_start_utc": self.time_start_utc,
            "time_end_utc": self.time_end_utc,
            "row_count": self.row_count,
        }


def file_receipt(
    path: str | Path,
    *,
    request: Era5Request,
    role: str,
    layout: QuarantineLayout,
    row_count: int | None = None,
) -> FileReceipt:
    source = layout.assert_inside(path)
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"ERA5 artifact is absent or empty: {source.name}")
    return FileReceipt(
        request_id=request.request_id,
        role=role,
        relative_path=source.relative_to(layout.root).as_posix(),
        bytes=source.stat().st_size,
        sha256=sha256_file(source),
        time_start_utc=request.start_utc.isoformat(),
        time_end_utc=request.end_utc.isoformat(),
        row_count=row_count,
    )


def retrieve_cds_request(
    request: Era5Request,
    *,
    target: str | Path,
    layout: QuarantineLayout,
    execute_download: bool,
    client_factory: Callable[[], Any] | None = None,
) -> FileReceipt:
    """Retrieve one request atomically, only after explicit download authorization."""

    target_path = layout.assert_inside(target)
    if not execute_download:
        raise DownloadAuthorizationError("explicit --execute-download authorization is absent")
    if target_path.exists() or target_path.with_suffix(target_path.suffix + ".partial").exists():
        raise FileExistsError(f"ERA5 target already exists: {target_path.name}")
    factory = client_factory or _default_cds_client

    try:
        client = factory()
    except Era5PretrainError:
        raise
    except Exception:
        raise Era5PretrainError("CDS client initialization failed") from None
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial = target_path.with_suffix(target_path.suffix + ".partial")
    try:
        client.retrieve(DATASET_ID, request.request(), str(partial))
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise OSError("CDS returned an empty ERA5 file")
        partial.replace(target_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise Era5PretrainError(f"CDS retrieval failed for {request.request_id}") from None
    return file_receipt(target_path, request=request, role="raw_cds_netcdf", layout=layout)


def _default_cds_client() -> Any:
    """Create a CDS client without logging or persisting environment credentials."""

    key = os.environ.get("CDSAPI_KEY")
    kwargs: dict[str, Any] = {"quiet": True, "debug": False}
    if key:
        url = os.environ.get(
            "CDSAPI_URL",
            "https://cds.climate.copernicus.eu/api",
        ).strip()
        parsed = urlsplit(url)
        if (
            parsed.scheme.casefold() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise Era5PretrainError("CDSAPI_URL must be a credential-free HTTPS endpoint")
        kwargs.update({"url": url, "key": key})

    import cdsapi

    return cdsapi.Client(**kwargs)


def _find_variable(dataset: Any, short_name: str) -> Any | None:
    aliases = ("lsm",) if short_name == "land_sea_mask" else ()
    candidates = (short_name, *aliases, VARIABLES[short_name])
    for candidate in candidates:
        if candidate in dataset.data_vars:
            return dataset[candidate]
    return None


def _collapse_field(
    data_array: Any,
    *,
    time_name: str,
    time_count: int,
    latitude_count: int,
    longitude_count: int,
    variable: str,
) -> np.ndarray:
    required = {time_name, "latitude", "longitude"}
    if variable == "land_sea_mask" and time_name not in data_array.dims:
        spatial = {"latitude", "longitude"}
        if not spatial.issubset(data_array.dims):
            raise Era5SchemaError("ERA5 land_sea_mask omits latitude/longitude")
        extras = [dimension for dimension in data_array.dims if dimension not in spatial]
        if any(dimension not in {"expver", "number"} for dimension in extras):
            raise Era5SchemaError("ERA5 land_sea_mask has an unexpected dimension")
        ordered = data_array.transpose(*extras, "latitude", "longitude")
        values = np.asarray(ordered.values, dtype=np.float64).reshape(
            (-1, latitude_count, longitude_count)
        )
        collapsed = np.full((latitude_count, longitude_count), np.nan, dtype=np.float64)
        for candidate in values:
            missing = ~np.isfinite(collapsed)
            collapsed[missing] = candidate[missing]
        return np.broadcast_to(collapsed, (time_count, latitude_count, longitude_count)).copy()
    if not required.issubset(data_array.dims):
        raise Era5SchemaError(f"ERA5 {variable} dimensions omit time/latitude/longitude")
    extras = [dimension for dimension in data_array.dims if dimension not in required]
    if any(dimension not in {"expver", "number"} for dimension in extras):
        raise Era5SchemaError(f"ERA5 {variable} has an unexpected dimension")
    ordered = data_array.transpose(*extras, time_name, "latitude", "longitude")
    values = np.asarray(ordered.values, dtype=np.float64)
    expected_tail = (time_count, latitude_count, longitude_count)
    if values.shape[-3:] != expected_tail:
        raise Era5SchemaError(f"ERA5 {variable} coordinate shape changed")
    flattened = values.reshape((-1, *expected_tail))
    result = np.full(expected_tail, np.nan, dtype=np.float64)
    for candidate in flattened:
        missing = ~np.isfinite(result)
        result[missing] = candidate[missing]
    return result


def _load_xarray_member(
    path: Path,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:  # pragma: no cover - environment preflight
        raise RuntimeError("xarray is required to process ERA5 NetCDF files") from exc

    with xr.open_dataset(path, decode_times=True) as dataset:
        time_name = "valid_time" if "valid_time" in dataset.coords else "time"
        if time_name not in dataset.coords:
            raise Era5SchemaError("ERA5 NetCDF lacks a valid UTC time coordinate")
        if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
            raise Era5SchemaError("ERA5 NetCDF lacks latitude/longitude coordinates")
        times = pd.DatetimeIndex(pd.to_datetime(dataset[time_name].values, utc=True))
        latitudes = np.asarray(dataset["latitude"].values, dtype=np.float64)
        longitudes = np.asarray(dataset["longitude"].values, dtype=np.float64)
        fields: dict[str, np.ndarray] = {}
        for short_name in VARIABLES:
            variable = _find_variable(dataset, short_name)
            if variable is None:
                continue
            fields[short_name] = _collapse_field(
                variable,
                time_name=time_name,
                time_count=len(times),
                latitude_count=len(latitudes),
                longitude_count=len(longitudes),
                variable=short_name,
            )
        if not fields:
            raise Era5SchemaError("ERA5 NetCDF member contains no requested variables")
    return times, latitudes, longitudes, fields


def _safe_zip_members(source: Path, target_directory: Path) -> tuple[Path, ...]:
    try:
        with ZipFile(source) as archive:
            infos = archive.infolist()
            if not 1 <= len(infos) <= 8:
                raise Era5SchemaError("ERA5 ZIP member count is outside the fixed limit")
            if len({value.filename for value in infos}) != len(infos):
                raise Era5SchemaError("ERA5 ZIP contains duplicate member names")
            paths: list[Path] = []
            for index, info in enumerate(infos):
                normalized = info.filename.replace("\\", "/")
                parts = PurePosixPath(normalized).parts
                if (
                    info.is_dir()
                    or len(parts) != 1
                    or ".." in parts
                    or not normalized.casefold().endswith(".nc")
                    or info.flag_bits & 0x1
                ):
                    raise Era5SchemaError("ERA5 ZIP contains an unsafe NetCDF member")
                if info.file_size <= 0 or info.file_size > 512 * 1024 * 1024:
                    raise Era5SchemaError("ERA5 ZIP member violates the size limit")
                if info.file_size / max(info.compress_size, 1) > 200.0:
                    raise Era5SchemaError("ERA5 ZIP member violates the compression-ratio limit")
                target = target_directory / f"member_{index:02d}.nc"
                with archive.open(info) as source_stream, target.open("xb") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
                if target.stat().st_size != info.file_size:
                    raise Era5SchemaError("ERA5 ZIP member length changed while reading")
                paths.append(target)
            return tuple(paths)
    except BadZipFile as exc:
        raise Era5SchemaError("ERA5 CDS response is a corrupt ZIP") from exc


def _load_member_set(
    paths: Sequence[Path],
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    reference_times: pd.DatetimeIndex | None = None
    reference_latitudes: np.ndarray | None = None
    reference_longitudes: np.ndarray | None = None
    fields: dict[str, np.ndarray] = {}
    for path in paths:
        times, latitudes, longitudes, member_fields = _load_xarray_member(path)
        if reference_times is None:
            reference_times = times
            reference_latitudes = latitudes
            reference_longitudes = longitudes
        elif (
            not times.equals(reference_times)
            or not np.array_equal(latitudes, reference_latitudes)
            or not np.array_equal(longitudes, reference_longitudes)
        ):
            raise Era5SchemaError("ERA5 NetCDF members have different coordinates")
        overlap = set(fields).intersection(member_fields)
        if overlap:
            raise Era5SchemaError(f"ERA5 NetCDF members duplicate variables: {sorted(overlap)}")
        fields.update(member_fields)
    if reference_times is None or reference_latitudes is None or reference_longitudes is None:
        raise Era5SchemaError("ERA5 response contains no NetCDF members")
    if set(fields) != set(VARIABLES):
        raise Era5SchemaError("ERA5 response differs from the frozen ten-field contract")
    return reference_times, reference_latitudes, reference_longitudes, fields


def load_netcdf_cube(
    path: str | Path,
    *,
    expected_request: Era5Request,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read direct or ZIP-wrapped CDS NetCDF members into one validated UTC grid."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"ERA5 NetCDF is absent: {source.name}")
    with source.open("rb") as handle:
        is_zip = handle.read(4) == b"PK\x03\x04"
    if is_zip:
        with tempfile.TemporaryDirectory(prefix=".era5_members_", dir=source.parent) as temporary:
            paths = _safe_zip_members(source, Path(temporary))
            times, latitudes, longitudes, fields = _load_member_set(paths)
    else:
        times, latitudes, longitudes, fields = _load_member_set((source,))

    if times.empty or times.has_duplicates or not times.is_monotonic_increasing:
        raise Era5SchemaError("ERA5 UTC time coordinate is empty, duplicated, or unordered")
    if times.max() >= CUTOFF_EXCLUSIVE_UTC:
        raise Era5SchemaError("ERA5 NetCDF crossed the approved UTC cutoff")
    expected_times = pd.date_range(
        expected_request.start_utc,
        expected_request.end_utc,
        freq="h",
    )
    if not times.equals(expected_times):
        raise Era5SchemaError("ERA5 NetCDF UTC hours differ from the request")
    expected_spatial = 3 if expected_request.purpose == "smoke_3x3" else 1
    if latitudes.shape != (expected_spatial,) or longitudes.shape != (expected_spatial,):
        raise Era5SchemaError("ERA5 NetCDF spatial shape differs from the request")
    return times, latitudes, longitudes, fields


def select_cell_from_smoke_file(
    path: str | Path,
    *,
    expected_request: Era5Request,
) -> SelectedCell:
    if expected_request.purpose != "smoke_3x3":
        raise ValueError("cell selection requires a smoke request")
    _, latitudes, longitudes, fields = load_netcdf_cube(path, expected_request=expected_request)
    return select_nearest_valid_ocean_cell(
        expected_request.station,
        latitudes=latitudes,
        longitudes=longitudes,
        fields=fields,
    )


def derive_selected_cell_frame(
    *,
    station: str,
    selection: SelectedCell,
    times_utc: Sequence[Any],
    values: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Convert one selected ERA5 cell to P3-oriented units and forcing variables."""

    if station != selection.station or station not in STATIONS:
        raise ValueError("selected-cell station mismatch")
    if set(values) != set(VARIABLES):
        raise Era5SchemaError("selected-cell values differ from the ten-field contract")
    times = pd.DatetimeIndex(pd.to_datetime(list(times_utc), utc=True))
    if times.empty or times.has_duplicates or not times.is_monotonic_increasing:
        raise Era5SchemaError("selected-cell UTC timestamps are invalid")
    if times.max() >= CUTOFF_EXCLUSIVE_UTC:
        raise Era5SchemaError("selected-cell data crossed the pre-2024 cutoff")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}
    if any(array.shape != (len(times),) for array in arrays.values()):
        raise Era5SchemaError("selected-cell field length differs from its UTC time axis")

    u10 = arrays["u10"]
    v10 = arrays["v10"]
    wind_speed = np.hypot(u10, v10)
    # Meteorological convention: degrees clockwise from north, direction wind comes from.
    wind_direction = (np.degrees(np.arctan2(-u10, -v10)) + 360.0) % 360.0
    air_c = arrays["t2m"] - 273.15
    dewpoint_c = arrays["d2m"] - 273.15
    with np.errstate(over="ignore", invalid="ignore"):
        relative_humidity = 100.0 * np.exp(
            (17.625 * dewpoint_c) / (243.04 + dewpoint_c) - (17.625 * air_c) / (243.04 + air_c)
        )
    relative_humidity = np.clip(relative_humidity, 0.0, 100.0)
    return pd.DataFrame(
        {
            "station": station,
            "time_utc": times,
            "latitude": selection.latitude,
            "longitude": selection.longitude,
            "swh_m": arrays["swh"],
            "mwp_s": arrays["mwp"],
            "hmax_m": arrays["hmax"],
            "mwd_deg": arrays["mwd"] % 360.0,
            "u10_m_s": u10,
            "v10_m_s": v10,
            "wspd10_m_s": wind_speed,
            "wdir10_from_deg": wind_direction,
            "msl_hpa": arrays["msl"] / 100.0,
            "t2m_c": air_c,
            "d2m_c": dewpoint_c,
            "relh2m_pct": relative_humidity,
            "land_sea_mask": arrays["land_sea_mask"],
        }
    )


def process_year_file(
    raw_path: str | Path,
    *,
    request: Era5Request,
    selection: SelectedCell,
    output_path: str | Path,
    layout: QuarantineLayout,
) -> FileReceipt:
    if request.purpose != "selected_cell_year":
        raise ValueError("segment processing requires a selected-cell request")
    source = layout.assert_inside(raw_path)
    destination = layout.assert_inside(output_path)
    if destination.exists() or destination.with_suffix(".parquet.partial").exists():
        raise FileExistsError(f"ERA5 derived target already exists: {destination.name}")
    times, latitudes, longitudes, fields = load_netcdf_cube(source, expected_request=request)
    if not (
        math.isclose(float(latitudes[0]), selection.latitude, abs_tol=1e-6)
        and math.isclose(float(longitudes[0]), selection.longitude, abs_tol=1e-6)
    ):
        raise Era5SchemaError("segment ERA5 cell differs from the smoke-selected cell")
    values = {name: array[:, 0, 0] for name, array in fields.items()}
    frame = derive_selected_cell_frame(
        station=request.station,
        selection=selection,
        times_utc=times,
        values=values,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.partial")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return file_receipt(
        destination,
        request=request,
        role="derived_selected_cell_hourly_parquet",
        layout=layout,
        row_count=len(frame),
    )


COMBINED_FILE_NAME: Final = "era5_p3_context_pretrain_2014_2023.parquet"
DERIVED_COLUMNS: Final = (
    "station",
    "time_utc",
    "latitude",
    "longitude",
    "swh_m",
    "mwp_s",
    "hmax_m",
    "mwd_deg",
    "u10_m_s",
    "v10_m_s",
    "wspd10_m_s",
    "wdir10_from_deg",
    "msl_hpa",
    "t2m_c",
    "d2m_c",
    "relh2m_pct",
    "land_sea_mask",
)

# Frozen, deliberately broad hard bounds. Missing ERA5 values remain admissible;
# finite-fraction completeness is a separate runner-level quality gate.
PHYSICAL_HARD_BOUNDS: Final = {
    # column: (minimum, maximum, minimum_inclusive, maximum_inclusive)
    "swh_m": (0.0, 50.0, True, True),
    "mwp_s": (0.0, 60.0, False, True),
    "hmax_m": (0.0, 100.0, True, True),
    "mwd_deg": (0.0, 360.0, True, False),
    "u10_m_s": (-150.0, 150.0, True, True),
    "v10_m_s": (-150.0, 150.0, True, True),
    "wspd10_m_s": (0.0, 200.0, True, True),
    "wdir10_from_deg": (0.0, 360.0, True, False),
    "msl_hpa": (500.0, 1200.0, True, True),
    "t2m_c": (-120.0, 80.0, True, True),
    "d2m_c": (-120.0, 80.0, True, True),
    "relh2m_pct": (0.0, 100.0, True, True),
    "land_sea_mask": (0.0, 1.0, True, True),
}


def _validate_physical_hard_bounds(frame: pd.DataFrame) -> None:
    for column, (
        minimum,
        maximum,
        include_minimum,
        include_maximum,
    ) in PHYSICAL_HARD_BOUNDS.items():
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & numeric.isna()).any():
            raise Era5SchemaError(f"derived ERA5 {column} contains non-numeric values")
        values = numeric.to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise Era5SchemaError(f"derived ERA5 {column} contains infinite values")
        finite = values[np.isfinite(values)]
        below = finite < minimum if include_minimum else finite <= minimum
        above = finite > maximum if include_maximum else finite >= maximum
        if below.any() or above.any():
            brackets = (
                "[" if include_minimum else "(",
                "]" if include_maximum else ")",
            )
            raise Era5SchemaError(
                f"derived ERA5 {column} violates frozen physical hard bound "
                f"{brackets[0]}{minimum}, {maximum}{brackets[1]}"
            )


def _validate_derived_segment_frame(
    frame: pd.DataFrame,
    *,
    request: Era5Request,
    selection: SelectedCell,
) -> None:
    if selection.station != request.station:
        raise Era5SchemaError("derived ERA5 selection station differs from its request")
    if tuple(frame.columns) != DERIVED_COLUMNS:
        raise Era5SchemaError("derived ERA5 segment schema changed")
    _validate_physical_hard_bounds(frame)
    if frame.empty or frame[["station", "time_utc"]].duplicated().any():
        raise Era5SchemaError("derived ERA5 segment is empty or has duplicate keys")
    if set(frame["station"].astype(str)) != {request.station}:
        raise Era5SchemaError("derived ERA5 segment station differs from its request")
    times = pd.DatetimeIndex(pd.to_datetime(frame["time_utc"], utc=True, errors="raise"))
    if times.max() >= CUTOFF_EXCLUSIVE_UTC:
        raise Era5SchemaError("derived ERA5 segment crossed the approved cutoff")
    expected_times = pd.date_range(request.start_utc, request.end_utc, freq="h")
    if not times.equals(expected_times):
        raise Era5SchemaError("derived ERA5 segment time axis differs from its request")
    latitudes = pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(float)
    longitudes = pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(float)
    if not (
        np.isfinite(latitudes).all()
        and np.isfinite(longitudes).all()
        and np.allclose(latitudes, selection.latitude, rtol=0.0, atol=1e-8)
        and np.allclose(longitudes, selection.longitude, rtol=0.0, atol=1e-8)
    ):
        raise Era5SchemaError("derived ERA5 segment collides with smoke-selected coordinates")


def load_validated_derived_year_file(
    path: str | Path,
    *,
    request: Era5Request,
    selection: SelectedCell,
    layout: QuarantineLayout,
) -> tuple[pd.DataFrame, FileReceipt]:
    source = layout.assert_inside(path)
    if not source.is_file():
        raise FileNotFoundError(f"derived ERA5 segment is absent: {source.name}")
    frame = pd.read_parquet(source)
    try:
        _validate_derived_segment_frame(frame, request=request, selection=selection)
    except (Era5SchemaError, ValueError, TypeError) as exc:
        raise FileExistsError(
            "existing derived ERA5 segment collision; refusing overwrite: "
            f"{request.request_id}: {exc}"
        ) from exc
    receipt = file_receipt(
        source,
        request=request,
        role="derived_selected_cell_hourly_parquet",
        layout=layout,
        row_count=len(frame),
    )
    return frame, receipt


def _validate_combined_frame(
    frame: pd.DataFrame,
    *,
    selections: Mapping[str, SelectedCell],
) -> dict[str, object]:
    if set(selections) != set(STATIONS):
        raise Era5SchemaError("combined ERA5 validation lacks selected cells")
    if tuple(frame.columns) != DERIVED_COLUMNS:
        raise Era5SchemaError("combined ERA5 derived schema changed")
    _validate_physical_hard_bounds(frame)
    if frame.empty or frame[["station", "time_utc"]].duplicated().any():
        raise Era5SchemaError("combined ERA5 data is empty or has duplicate station-hours")
    times = pd.to_datetime(frame["time_utc"], utc=True, errors="raise")
    if times.max() >= CUTOFF_EXCLUSIVE_UTC:
        raise Era5SchemaError("combined ERA5 data crossed the approved cutoff")
    if set(frame["station"].astype(str)) != set(STATIONS):
        raise Era5SchemaError("combined ERA5 data does not contain exactly three stations")
    expected_times = pd.date_range(
        pd.Timestamp("2014-01-01T00:00:00Z"),
        CUTOFF_EXCLUSIVE_UTC - pd.Timedelta(hours=1),
        freq="h",
    )
    for station in STATIONS:
        keep = frame["station"].astype(str).eq(station)
        observed = pd.DatetimeIndex(times.loc[keep]).sort_values()
        if not observed.equals(expected_times):
            raise Era5SchemaError(f"combined ERA5 hourly coverage changed for {station}")
        latitude = pd.to_numeric(frame.loc[keep, "latitude"], errors="coerce").to_numpy(float)
        longitude = pd.to_numeric(frame.loc[keep, "longitude"], errors="coerce").to_numpy(float)
        selection = selections[station]
        if not (
            np.isfinite(latitude).all()
            and np.isfinite(longitude).all()
            and np.allclose(latitude, selection.latitude, rtol=0.0, atol=1e-8)
            and np.allclose(longitude, selection.longitude, rtol=0.0, atol=1e-8)
        ):
            raise Era5SchemaError(
                f"combined ERA5 coordinates collide with selected cell for {station}"
            )
    return {
        "row_count": int(len(frame)),
        "observed_start": times.min().isoformat(),
        "observed_end": times.max().isoformat(),
        "station_count": int(frame["station"].nunique()),
        "rows_per_station": int(len(expected_times)),
    }


def load_validated_combined_file(
    layout: QuarantineLayout,
    selections: Mapping[str, SelectedCell],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the canonical combined Parquet after reapplying its full contract."""

    source = layout.assert_inside(layout.derived / COMBINED_FILE_NAME)
    if not source.is_file():
        raise FileNotFoundError(f"combined ERA5 file is absent: {source.name}")
    try:
        frame = pd.read_parquet(source)
        summary = _validate_combined_frame(frame, selections=selections)
    except (Era5SchemaError, OSError, ValueError, TypeError) as exc:
        raise FileExistsError(f"existing combined ERA5 collision; refusing reuse: {exc}") from exc
    return frame, summary


def combine_derived_year_files(
    *,
    layout: QuarantineLayout,
    requests: Sequence[Era5Request],
    selections: Mapping[str, SelectedCell],
) -> tuple[Path, FileReceipt, dict[str, object]]:
    """Combine the 363 validated segment Parquets into one canonical pretraining file."""

    if len(requests) != 363 or any(request.purpose != "selected_cell_year" for request in requests):
        raise ValueError("combined ERA5 input must be the frozen 363-segment monthly plan")
    destination = layout.assert_inside(layout.derived / COMBINED_FILE_NAME)
    if destination.is_file():
        combined, summary = load_validated_combined_file(layout, selections)
    else:
        frames: list[pd.DataFrame] = []
        for request in requests:
            path = layout.derived_year_path(request)
            frame, _ = load_validated_derived_year_file(
                path,
                request=request,
                selection=selections[request.station],
                layout=layout,
            )
            frames.append(frame)
        combined = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["station", "time_utc"], kind="stable")
            .reset_index(drop=True)
        )
        summary = _validate_combined_frame(combined, selections=selections)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.partial")
        try:
            combined.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    summary = _validate_combined_frame(combined, selections=selections)
    receipt = FileReceipt(
        request_id="combined_era5_p3_2014_2023",
        role="final_combined_selected_cell_hourly_parquet",
        relative_path=destination.relative_to(layout.root).as_posix(),
        bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        time_start_utc=str(summary["observed_start"]),
        time_end_utc=str(summary["observed_end"]),
        row_count=int(summary["row_count"]),
    )
    return destination, receipt, summary


def write_selected_cells(layout: QuarantineLayout, selections: Mapping[str, SelectedCell]) -> Path:
    if set(selections) != set(STATIONS):
        raise ValueError("selected-cell file requires all three P3 stations")
    payload = {
        "schema_version": "era5_p3_selected_cells.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "smoke_day_utc": SMOKE_DAY_UTC.isoformat(),
        "cells": [selections[name].public_dict() for name in STATIONS],
    }
    path = layout.assert_inside(layout.derived / "selected_cells.json")
    if path.is_file():
        existing = read_selected_cells(layout)
        if existing == dict(selections):
            return path
        derived_exists = any(layout.derived_years.glob("*.parquet"))
        combined_exists = (layout.derived / COMBINED_FILE_NAME).is_file()
        if derived_exists or combined_exists:
            raise FileExistsError(
                "smoke-selected cell collision with existing derived/combined ERA5; "
                "refusing to overwrite selected_cells.json"
            )
    _write_json_atomic(path, payload)
    return path


def read_selected_cells(layout: QuarantineLayout) -> dict[str, SelectedCell]:
    path = layout.assert_inside(layout.derived / "selected_cells.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "era5_p3_selected_cells.v1":
        raise Era5SchemaError("selected-cell schema version changed")
    if payload.get("smoke_day_utc") != SMOKE_DAY_UTC.isoformat():
        raise Era5SchemaError("selected-cell smoke day changed")
    cells = {
        str(value["station"]): SelectedCell.from_public_dict(value)
        for value in payload.get("cells", [])
    }
    if set(cells) != set(STATIONS):
        raise Era5SchemaError("selected-cell file does not cover all P3 stations")
    return cells


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_manifest(
    *,
    stage: str,
    smoke_requests: Sequence[Era5Request],
    year_requests: Sequence[Era5Request] = (),
    selections: Mapping[str, SelectedCell] | None = None,
    files: Iterable[FileReceipt] = (),
    network_action_taken: bool,
) -> dict[str, Any]:
    if stage not in {"plan", "smoke", "years", "combine"}:
        raise ValueError(f"unsupported ERA5 P3 manifest stage: {stage}")
    receipts = tuple(files)
    combined = [
        value for value in receipts if value.role == "final_combined_selected_cell_hourly_parquet"
    ]
    if len(combined) > 1:
        raise ValueError("ERA5 manifest received multiple combined files")
    final = combined[0] if combined else None
    if stage in {"years", "combine"} and final is None:
        raise ValueError("completed ERA5 manifest requires a validated combined file")
    if stage in {"plan", "smoke"} and final is not None:
        raise ValueError("plan/smoke receipts cannot claim a completed combined file")
    transformation_steps = [
        "Request 2023-12-30 UTC on a 0.25-degree 3x3 grid per station.",
        (
            "Select the nearest cell with mean land_sea_mask <= 0.5 and at least "
            "90% finite smoke values for every dynamic field."
        ),
        (
            "Request 2014--2022 as individual calendar months; request 2023 Jan--Nov "
            "monthly, then Dec 1--30 and the final partial day. Never request a valid "
            "time at or after 2023-12-31 15:00 UTC."
        ),
        (
            "Retain wave fields; derive wind speed/direction from u10/v10, convert "
            "msl Pa to hPa and t2m/d2m K to degC, and derive 2m relative humidity."
        ),
        (
            "Reject non-numeric, infinite, or finite values outside frozen broad "
            "physical hard bounds; retain legitimate missing ERA5 values."
        ),
        "Write raw NetCDF and derived Parquet only inside the fixed quarantine root.",
        "Combine 363 station-time segments into one key-unique hourly Parquet.",
    ]
    payload = {
        # Generic external-data preflight surface.
        "schema_version": "1.0",
        "source_id": "era5_pre2024",
        "local_file": (
            None if final is None else (QUARANTINE_RELATIVE / final.relative_path).as_posix()
        ),
        "file_sha256": None if final is None else final.sha256,
        "observed_start": None if final is None else final.time_start_utc,
        "observed_end": None if final is None else final.time_end_utc,
        "row_count": 0 if final is None else final.row_count,
        "variables": list(VARIABLES),
        "transformation_log": " ".join(transformation_steps),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "research_only": True,
        "official_test_or_submission_accessed": False,
        "detail": {
            "schema_version": "era5_p3_context_pretrain_manifest.v1",
            "quarantine_relative": QUARANTINE_RELATIVE.as_posix(),
            "transformation_steps": transformation_steps,
        },
        "source": {
            "dataset": "ERA5 hourly data on single levels",
            "dataset_id": DATASET_ID,
            "url": DATASET_URL,
            "doi": DATASET_DOI,
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "attribution": ATTRIBUTION,
        },
        "boundary": {
            "start_year": START_YEAR,
            "end_year": END_YEAR,
            "cutoff_exclusive_utc": CUTOFF_EXCLUSIVE_UTC.isoformat(),
            "maximum_valid_time_utc": "2023-12-31T14:00:00+00:00",
            "last_eligible_24h_anchor_utc": LAST_ELIGIBLE_ANCHOR_UTC.isoformat(),
            "approval_cutoff_kst": "2023-12-31T23:59:59+09:00",
        },
        "stations": [STATIONS[name].public_dict() for name in STATIONS],
        "variable_mapping": [
            {"short_name": short_name, "cds_name": cds_name}
            for short_name, cds_name in VARIABLES.items()
        ],
        "requests": {
            "smoke_3x3_one_day": [value.public_dict() for value in smoke_requests],
            "selected_single_cell_years": [value.public_dict() for value in year_requests],
        },
        "selected_cells": (
            [] if selections is None else [selections[name].public_dict() for name in STATIONS]
        ),
        "files": [value.public_dict() for value in receipts],
        "checksums_sha256": {value.relative_path: value.sha256 for value in receipts},
        "time_coverage": [
            {
                "request_id": value.request_id,
                "start_utc": value.time_start_utc,
                "end_utc": value.time_end_utc,
            }
            for value in receipts
        ],
        "network_action_taken": bool(network_action_taken),
        "download_requires_explicit_execute_download": True,
    }
    referenced_paths = " ".join(value.relative_path for value in receipts).casefold()
    if any(token in referenced_paths for token in ("test_context", "test_index", "submission")):
        raise AssertionError("forbidden P3 evaluation path entered the ERA5 manifest")
    return payload


def _validate_completed_manifest_payload(
    layout: QuarantineLayout,
    payload: Mapping[str, Any],
    *,
    combined_receipt: FileReceipt | None = None,
) -> None:
    required = {
        "local_file": str,
        "file_sha256": str,
        "observed_start": str,
        "observed_end": str,
    }
    if any(not isinstance(payload.get(key), kind) for key, kind in required.items()):
        raise ValueError("canonical ERA5 manifest lacks completed file provenance")
    if not isinstance(payload.get("row_count"), int) or payload["row_count"] <= 0:
        raise ValueError("canonical ERA5 manifest lacks a positive row count")
    expected_local = (QUARANTINE_RELATIVE / "derived" / COMBINED_FILE_NAME).as_posix()
    if payload["local_file"] != expected_local:
        raise ValueError("canonical ERA5 manifest final path changed")
    combined_path = layout.assert_inside(layout.derived / COMBINED_FILE_NAME)
    if not combined_path.is_file():
        raise FileNotFoundError("canonical ERA5 manifest final file is absent")
    actual_sha256 = sha256_file(combined_path)
    if payload["file_sha256"] != actual_sha256:
        raise ValueError("canonical ERA5 manifest SHA-256 differs from final file")
    final_entries = [
        value
        for value in payload.get("files", [])
        if isinstance(value, Mapping)
        and value.get("role") == "final_combined_selected_cell_hourly_parquet"
    ]
    if len(final_entries) != 1:
        raise ValueError("canonical ERA5 manifest lacks one final-file receipt")
    entry = final_entries[0]
    expected_relative = f"derived/{COMBINED_FILE_NAME}"
    if (
        entry.get("relative_path") != expected_relative
        or entry.get("sha256") != actual_sha256
        or entry.get("bytes") != combined_path.stat().st_size
        or entry.get("row_count") != payload["row_count"]
        or entry.get("time_start_utc") != payload["observed_start"]
        or entry.get("time_end_utc") != payload["observed_end"]
    ):
        raise ValueError("canonical ERA5 manifest final-file receipt is inconsistent")
    if combined_receipt is not None and (
        combined_receipt.relative_path != expected_relative
        or combined_receipt.sha256 != actual_sha256
        or combined_receipt.bytes != combined_path.stat().st_size
        or combined_receipt.row_count != payload["row_count"]
        or combined_receipt.time_start_utc != payload["observed_start"]
        or combined_receipt.time_end_utc != payload["observed_end"]
    ):
        raise ValueError("existing canonical ERA5 manifest differs from reused combined file")


def validate_existing_canonical_manifest(
    layout: QuarantineLayout,
    *,
    combined_receipt: FileReceipt,
) -> bool:
    """Validate an existing canonical manifest before reusing a combined file."""

    path = layout.assert_inside(layout.manifests / "manifest.json")
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError("canonical ERA5 manifest schema version changed")
        if payload.get("source_id") != "era5_pre2024":
            raise ValueError("canonical ERA5 manifest source changed")
        if payload.get("stage") not in {"years", "combine"}:
            raise ValueError("canonical ERA5 manifest is not a completed stage")
        if payload.get("variables") != list(VARIABLES):
            raise ValueError("canonical ERA5 manifest variables changed")
        _validate_completed_manifest_payload(
            layout,
            payload,
            combined_receipt=combined_receipt,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise FileExistsError(
            f"existing canonical ERA5 manifest collision; refusing reuse: {exc}"
        ) from exc
    return True


def write_manifest(
    layout: QuarantineLayout,
    payload: Mapping[str, Any],
    *,
    stage: str,
) -> Path:
    """Atomically write a stage receipt or completed canonical manifest.

    Planning and smoke-selection evidence deliberately use distinct filenames so a
    later diagnostic invocation cannot replace a completed pretraining manifest.
    """

    filenames = {
        "plan": "plan_receipt.json",
        "smoke": "smoke_receipt.json",
        "years": "manifest.json",
        "combine": "manifest.json",
    }
    if stage not in filenames or payload.get("stage") != stage:
        raise ValueError("ERA5 manifest stage/filename contract changed")
    if stage in {"years", "combine"}:
        _validate_completed_manifest_payload(layout, payload)
    elif payload.get("local_file") is not None or payload.get("row_count") != 0:
        raise ValueError("plan/smoke receipt cannot publish completed-file metadata")
    path = layout.assert_inside(layout.manifests / filenames[stage])
    _write_json_atomic(path, payload)
    return path
