"""Fixed, credential-free ERA5 request construction for the P2 primary path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

DATASET_ID = "reanalysis-era5-single-levels"
GRID_DEGREES = 0.25
SORS_LATITUDE = 37.4231333
SORS_LONGITUDE = 124.7380389
AREA_3X3 = (37.75, 124.50, 37.25, 125.00)  # north, west, south, east
ERA5_VARIABLES = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "eastward_turbulent_surface_stress",
    "northward_turbulent_surface_stress",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
)
ANCILLARY_VARIABLES = ("land_sea_mask",)
FORMAT_ORDER = ("grib", "netcdf")
OOF_SHA256 = "dab52579e99a20cc0444bf13bc3a1400191024a10303cb996ba59a89509c9cb4"
OOF_ROWS = 69_850
OOF_RANGES_UTC = {
    "2024_sep_oct": (
        "2024-08-31T15:00:00+00:00",
        "2024-10-31T14:50:00+00:00",
    ),
    "2025_jul_aug": (
        "2025-06-30T15:00:00+00:00",
        "2025-08-31T14:50:00+00:00",
    ),
    "2025_nov_dec": (
        "2025-10-31T15:00:00+00:00",
        "2025-12-09T18:00:00+00:00",
    ),
}


@dataclass(frozen=True)
class RequestChunk:
    chunk_id: str
    block: str
    dates: tuple[date, ...]
    hours_utc: tuple[int, ...]

    def timestamps_utc(self) -> pd.DatetimeIndex:
        values = [
            pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=hour)
            for day in self.dates
            for hour in self.hours_utc
        ]
        return pd.DatetimeIndex(values).sort_values()

    def request(self, data_format: str = "grib") -> dict[str, Any]:
        if data_format not in FORMAT_ORDER:
            raise ValueError(f"unsupported ERA5 format: {data_format}")
        years = {day.year for day in self.dates}
        months = {day.month for day in self.dates}
        if len(years) != 1 or len(months) != 1:
            raise ValueError("one ERA5 chunk must stay within one calendar month")
        return {
            "product_type": ["reanalysis"],
            "variable": [*ERA5_VARIABLES, *ANCILLARY_VARIABLES],
            "year": [f"{next(iter(years)):04d}"],
            "month": [f"{next(iter(months)):02d}"],
            "day": [f"{day.day:02d}" for day in self.dates],
            "time": [f"{hour:02d}:00" for hour in self.hours_utc],
            "area": list(AREA_3X3),
            "data_format": data_format,
            "download_format": "unarchived",
        }

    def target_name(self, data_format: str) -> str:
        suffix = ".grib" if data_format == "grib" else ".nc"
        return f"{self.chunk_id}{suffix}"

    def public_dict(self) -> dict[str, Any]:
        timestamps = self.timestamps_utc()
        return {
            "chunk_id": self.chunk_id,
            "block": self.block,
            "start_utc": timestamps.min().isoformat(),
            "end_utc": timestamps.max().isoformat(),
            "hour_count": len(timestamps),
            "grid_point_count": 9,
            "variable_count": len(ERA5_VARIABLES),
            "validation_ancillary": list(ANCILLARY_VARIABLES),
            "request_grib": self.request("grib"),
            "netcdf_fallback_changes": {"data_format": "netcdf"},
        }


def _chunk_id(block: str, dates: tuple[date, ...], hours: tuple[int, ...]) -> str:
    first = dates[0].strftime("%Y%m%d")
    last = dates[-1].strftime("%Y%m%d")
    date_part = first if first == last else f"{first}_{last}"
    hour_part = "h00_23" if hours == tuple(range(24)) else f"h{hours[0]:02d}_{hours[-1]:02d}"
    return f"{block}_{date_part}_{hour_part}"


def _exact_month_chunks(block: str, start: pd.Timestamp, end: pd.Timestamp) -> list[RequestChunk]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("ERA5 chunk boundaries must be timezone-aware")
    start_utc = start.tz_convert("UTC").floor("h")
    end_utc = end.tz_convert("UTC").floor("h")
    expected = pd.date_range(start_utc, end_utc, freq="h")
    by_day: dict[date, tuple[int, ...]] = {}
    for day, values in pd.Series(expected.hour, index=expected.date).groupby(level=0):
        by_day[day] = tuple(int(value) for value in values)

    grouped: list[tuple[list[date], tuple[int, ...]]] = []
    for day in sorted(by_day):
        hours = by_day[day]
        if (
            grouped
            and grouped[-1][1] == hours
            and grouped[-1][0][-1] + timedelta(days=1) == day
            and grouped[-1][0][-1].year == day.year
            and grouped[-1][0][-1].month == day.month
        ):
            grouped[-1][0].append(day)
        else:
            grouped.append(([day], hours))
    chunks = [
        RequestChunk(
            chunk_id=_chunk_id(block, tuple(days), hours),
            block=block,
            dates=tuple(days),
            hours_utc=hours,
        )
        for days, hours in grouped
    ]
    reconstructed = pd.DatetimeIndex(
        sorted(timestamp for chunk in chunks for timestamp in chunk.timestamps_utc())
    )
    if not reconstructed.equals(expected):
        raise AssertionError("ERA5 chunking added or omitted an OOF context hour")
    return chunks


def validate_frozen_oof(frame: pd.DataFrame) -> None:
    required = {"time", "layer", "block", "truth", "prediction"}
    if not required.issubset(frame.columns) or len(frame) != OOF_ROWS:
        raise ValueError("frozen incumbent OOF schema or row count changed")
    if frame[["time", "layer"]].duplicated().any():
        raise ValueError("frozen incumbent OOF contains duplicate keys")
    time = pd.to_datetime(frame["time"], utc=True, errors="raise")
    for block, (expected_start, expected_end) in OOF_RANGES_UTC.items():
        keep = frame["block"].eq(block)
        if not keep.any():
            raise ValueError(f"frozen incumbent OOF block missing: {block}")
        observed = (time[keep].min().isoformat(), time[keep].max().isoformat())
        if observed != (expected_start, expected_end):
            raise ValueError(f"frozen incumbent OOF time range changed: {block}")
    if set(frame["block"].astype(str)) != set(OOF_RANGES_UTC):
        raise ValueError("frozen incumbent OOF block set changed")


def build_oof_chunk_plan(frame: pd.DataFrame, pad_days: int = 7) -> tuple[RequestChunk, ...]:
    """Validate the frozen OOF artifact, then return the preregistered request plan."""

    validate_frozen_oof(frame)
    return build_registered_chunk_plan(pad_days=pad_days)


def build_registered_chunk_plan(pad_days: int = 7) -> tuple[RequestChunk, ...]:
    """Build the exact-hour plan from frozen aggregate ranges without reading OOF values."""

    if pad_days != 7:
        raise ValueError("ERA5 primary plan is fixed to exactly seven context days")
    chunks: list[RequestChunk] = []
    for block, (raw_start, raw_end) in OOF_RANGES_UTC.items():
        start = pd.Timestamp(raw_start) - pd.Timedelta(days=pad_days)
        end = pd.Timestamp(raw_end) + pd.Timedelta(days=pad_days)
        chunks.extend(_exact_month_chunks(block, start, end))
    result = tuple(chunks)
    timestamps = pd.DatetimeIndex(
        sorted(timestamp for chunk in result for timestamp in chunk.timestamps_utc())
    )
    if len(result) != 17 or len(timestamps) != 4_900 or timestamps.duplicated().any():
        raise AssertionError("registered ERA5 request-plan cardinality changed")
    return result


def build_smoke_chunk() -> RequestChunk:
    smoke_date = date(2024, 9, 1)
    return RequestChunk(
        chunk_id="smoke_sors_3x3_20240901_24h",
        block="smoke",
        dates=(smoke_date,),
        hours_utc=tuple(range(24)),
    )
