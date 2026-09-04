"""Read-only validation for ZIP-wrapped ERA5 CDS NetCDF smoke responses."""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from p2_restore.era5_preflight import Era5Field, validate_era5_fields
from p2_restore.era5_request import (
    ANCILLARY_VARIABLES,
    ERA5_VARIABLES,
    RequestChunk,
)

MAX_SMOKE_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_NETCDF_MEMBER_BYTES = 8 * 1024 * 1024
MAX_CHUNK_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_CHUNK_NETCDF_MEMBER_BYTES = 64 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200.0
EXPECTED_MEMBER_COUNT = 2

SHORT_TO_PUBLIC = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "ewss": "eastward_turbulent_surface_stress",
    "nsss": "northward_turbulent_surface_stress",
    "ssr": "surface_net_solar_radiation",
    "str": "surface_net_thermal_radiation",
    "slhf": "surface_latent_heat_flux",
    "sshf": "surface_sensible_heat_flux",
    "lsm": "land_sea_mask",
}
STEP_VARIABLES = {
    "instant": frozenset(("u10", "v10", "lsm")),
    "accum": frozenset(("ewss", "nsss", "ssr", "str", "slhf", "sshf")),
}
CANONICAL_STANDARD_NAMES = {
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
WIND_LONG_NAMES = {
    "u10": "10 metre U wind component",
    "v10": "10 metre V wind component",
}
ALLOWED_AUXILIARY_VARIABLES = {
    "number",
    "valid_time",
    "latitude",
    "longitude",
    "expver",
}
NETCDF_MAGICS = (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n")


@dataclass(frozen=True)
class CdsMemberSummary:
    name: str
    bytes: int
    compressed_bytes: int
    crc32: str
    step_type: str
    short_variables: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bytes": self.bytes,
            "compressed_bytes": self.compressed_bytes,
            "crc32": self.crc32,
            "step_type": self.step_type,
            "short_variables": list(self.short_variables),
        }


@dataclass(frozen=True)
class CdsSmokeReport:
    file_name: str
    file_sha256: str
    file_bytes: int
    members: tuple[CdsMemberSummary, ...]
    time_start_utc: str
    time_end_utc: str
    latitudes: tuple[float, ...]
    longitudes: tuple[float, ...]
    units: dict[str, str]
    finite_value_counts: dict[str, int]
    validation: dict[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "passed": True,
            "file_name": self.file_name,
            "file_sha256": self.file_sha256,
            "file_bytes": self.file_bytes,
            "container_format": "zip",
            "member_count": len(self.members),
            "members": [member.public_dict() for member in self.members],
            "time_start_utc": self.time_start_utc,
            "time_end_utc": self.time_end_utc,
            "latitudes": list(self.latitudes),
            "longitudes": list(self.longitudes),
            "feature_variables": list(ERA5_VARIABLES),
            "validation_ancillary": list(ANCILLARY_VARIABLES),
            "units": dict(self.units),
            "finite_value_counts": dict(self.finite_value_counts),
            "validation": dict(self.validation),
        }


@dataclass(frozen=True)
class _ParsedMember:
    summary: CdsMemberSummary
    times_utc: pd.DatetimeIndex
    latitudes: np.ndarray
    longitudes: np.ndarray
    fields: dict[str, Era5Field]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(info: ZipInfo, *, maximum_member_bytes: int) -> None:
    normalized = info.filename.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if (
        info.is_dir()
        or not normalized
        or normalized.startswith("/")
        or len(parts) != 1
        or ".." in parts
        or "\x00" in normalized
        or not normalized.lower().endswith(".nc")
    ):
        raise ValueError(f"unsafe ERA5 CDS archive member: {info.filename!r}")
    if info.flag_bits & 0x1:
        raise ValueError("encrypted ERA5 CDS archive members are forbidden")
    if info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
        raise ValueError("unsupported ERA5 CDS ZIP compression method")
    if info.file_size <= 0 or info.file_size > maximum_member_bytes:
        raise ValueError("ERA5 CDS NetCDF member violates the fixed size limit")
    ratio = info.file_size / max(info.compress_size, 1)
    if ratio > MAX_COMPRESSION_RATIO:
        raise ValueError("ERA5 CDS archive violates the compression-ratio limit")


def _read_archive(
    path: Path,
    *,
    maximum_archive_bytes: int,
    maximum_member_bytes: int,
) -> tuple[tuple[ZipInfo, bytes], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"ERA5 CDS smoke file is absent: {path.name}")
    size = path.stat().st_size
    if size <= 0 or size > maximum_archive_bytes:
        raise ValueError("ERA5 CDS archive violates the fixed size limit")
    with path.open("rb") as handle:
        if handle.read(4) != b"PK\x03\x04":
            raise ValueError("ERA5 CDS NetCDF response is not the expected ZIP container")
    try:
        with ZipFile(path, mode="r") as archive:
            infos = archive.infolist()
            if len(infos) != EXPECTED_MEMBER_COUNT:
                raise ValueError("ERA5 CDS smoke ZIP must contain exactly two NetCDF members")
            if len({info.filename for info in infos}) != len(infos):
                raise ValueError("ERA5 CDS smoke ZIP contains duplicate member names")
            result: list[tuple[ZipInfo, bytes]] = []
            for info in infos:
                _safe_member(info, maximum_member_bytes=maximum_member_bytes)
                payload = archive.read(info)
                if len(payload) != info.file_size:
                    raise ValueError("ERA5 CDS NetCDF member length changed while reading")
                if not any(payload.startswith(magic) for magic in NETCDF_MAGICS):
                    raise ValueError("ERA5 CDS ZIP member is not a NetCDF payload")
                if zlib.crc32(payload) & 0xFFFFFFFF != info.CRC:
                    raise ValueError("ERA5 CDS ZIP member failed CRC validation")
                result.append((info, payload))
            return tuple(result)
    except BadZipFile as exc:
        raise ValueError("ERA5 CDS response is a corrupt ZIP container") from exc


def _numeric_values(variable: Any, name: str) -> np.ndarray:
    values = np.ma.asarray(variable[:])
    try:
        return np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ERA5 CDS {name} is not numeric") from exc


def _decode_times(dataset: Any, num2date: Any) -> pd.DatetimeIndex:
    variable = dataset.variables["valid_time"]
    if variable.dimensions != ("valid_time",):
        raise ValueError("ERA5 CDS valid_time coordinate has unexpected dimensions")
    raw = _numeric_values(variable, "valid_time")
    if not np.isfinite(raw).all():
        raise ValueError("ERA5 CDS valid_time contains non-finite values")
    units = str(getattr(variable, "units", ""))
    calendar = str(getattr(variable, "calendar", "standard"))
    try:
        decoded = num2date(
            raw,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        return pd.DatetimeIndex(pd.to_datetime(list(decoded), utc=True))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ERA5 CDS valid_time metadata cannot be decoded") from exc


def _parse_member(info: ZipInfo, payload: bytes, *, expected_time_count: int) -> _ParsedMember:
    try:
        from netCDF4 import Dataset, num2date
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by runtime preflight
        raise RuntimeError("netCDF4 is required to validate ERA5 CDS NetCDF files") from exc

    try:
        dataset = Dataset(info.filename, mode="r", memory=payload)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"ERA5 CDS member cannot be opened as NetCDF: {info.filename}") from exc
    try:
        dimensions = {name: len(value) for name, value in dataset.dimensions.items()}
        if dimensions != {
            "valid_time": expected_time_count,
            "latitude": 3,
            "longitude": 3,
        }:
            raise ValueError("ERA5 CDS member dimensions differ from the exact-hour 3x3 request")
        variables = set(dataset.variables)
        required_coordinates = {"valid_time", "latitude", "longitude"}
        if not required_coordinates.issubset(variables):
            raise ValueError("ERA5 CDS NetCDF member is missing required coordinates")
        unexpected = variables - ALLOWED_AUXILIARY_VARIABLES - set(SHORT_TO_PUBLIC)
        if unexpected:
            raise ValueError(f"unexpected ERA5 CDS NetCDF variables: {sorted(unexpected)}")
        short_variables = frozenset(variables).intersection(SHORT_TO_PUBLIC)
        matching_steps = [
            step_type
            for step_type, expected in STEP_VARIABLES.items()
            if short_variables == expected
        ]
        if len(matching_steps) != 1:
            raise ValueError("ERA5 CDS instant/accum variable partition changed")
        step_type = matching_steps[0]

        times = _decode_times(dataset, num2date)
        latitude_variable = dataset.variables["latitude"]
        longitude_variable = dataset.variables["longitude"]
        if (
            latitude_variable.dimensions != ("latitude",)
            or str(getattr(latitude_variable, "units", "")) != "degrees_north"
        ):
            raise ValueError("ERA5 CDS latitude coordinate metadata changed")
        if (
            longitude_variable.dimensions != ("longitude",)
            or str(getattr(longitude_variable, "units", "")) != "degrees_east"
        ):
            raise ValueError("ERA5 CDS longitude coordinate metadata changed")
        latitudes = _numeric_values(latitude_variable, "latitude")
        longitudes = _numeric_values(longitude_variable, "longitude")
        if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
            raise ValueError("ERA5 CDS spatial coordinates contain non-finite values")

        fields: dict[str, Era5Field] = {}
        for short_name in sorted(short_variables):
            public_name = SHORT_TO_PUBLIC[short_name]
            variable = dataset.variables[short_name]
            if variable.dimensions != ("valid_time", "latitude", "longitude"):
                raise ValueError(f"ERA5 CDS {short_name} dimensions changed")
            observed_step = str(getattr(variable, "GRIB_stepType", ""))
            if observed_step != step_type:
                raise ValueError(f"ERA5 CDS {short_name} step type changed")
            standard_name = str(getattr(variable, "standard_name", ""))
            expected_standard = CANONICAL_STANDARD_NAMES[public_name]
            if short_name in WIND_LONG_NAMES:
                if (
                    standard_name not in {"", "unknown"}
                    or str(getattr(variable, "long_name", "")) != WIND_LONG_NAMES[short_name]
                ):
                    raise ValueError(f"ERA5 CDS {short_name} identity metadata changed")
            elif standard_name != expected_standard:
                raise ValueError(f"ERA5 CDS {short_name} sign convention metadata changed")
            fields[public_name] = Era5Field(
                values=_numeric_values(variable, short_name),
                units=str(getattr(variable, "units", "")),
                standard_name=expected_standard,
            )
        summary = CdsMemberSummary(
            name=info.filename,
            bytes=info.file_size,
            compressed_bytes=info.compress_size,
            crc32=f"{info.CRC:08x}",
            step_type=step_type,
            short_variables=tuple(sorted(short_variables)),
        )
        return _ParsedMember(summary, times, latitudes, longitudes, fields)
    finally:
        dataset.close()


def _validate_cds_archive(
    path: str | Path,
    *,
    expected_chunk: RequestChunk,
    require_24h_smoke: bool,
) -> tuple[
    CdsSmokeReport,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    dict[str, Era5Field],
]:
    source = Path(path)
    expected_times = expected_chunk.timestamps_utc()
    maximum_archive_bytes = (
        MAX_SMOKE_ARCHIVE_BYTES if require_24h_smoke else MAX_CHUNK_ARCHIVE_BYTES
    )
    maximum_member_bytes = (
        MAX_NETCDF_MEMBER_BYTES if require_24h_smoke else MAX_CHUNK_NETCDF_MEMBER_BYTES
    )
    parsed = tuple(
        _parse_member(info, payload, expected_time_count=len(expected_times))
        for info, payload in _read_archive(
            source,
            maximum_archive_bytes=maximum_archive_bytes,
            maximum_member_bytes=maximum_member_bytes,
        )
    )
    if {member.summary.step_type for member in parsed} != set(STEP_VARIABLES):
        raise ValueError("ERA5 CDS ZIP must contain one instant and one accum member")

    reference = parsed[0]
    for member in parsed[1:]:
        if not member.times_utc.equals(reference.times_utc):
            raise ValueError("ERA5 CDS NetCDF members have different valid times")
        if not np.array_equal(member.latitudes, reference.latitudes) or not np.array_equal(
            member.longitudes, reference.longitudes
        ):
            raise ValueError("ERA5 CDS NetCDF members have different spatial coordinates")
    if reference.times_utc.tolist() != expected_times.tolist():
        raise ValueError("ERA5 CDS valid times differ from the frozen request")

    fields: dict[str, Era5Field] = {}
    for member in parsed:
        overlap = set(fields).intersection(member.fields)
        if overlap:
            raise ValueError(f"duplicate ERA5 CDS variables across members: {sorted(overlap)}")
        fields.update(member.fields)
    validation = validate_era5_fields(
        fields,
        times_utc=reference.times_utc,
        latitudes=reference.latitudes,
        longitudes=reference.longitudes,
        require_24h_smoke=require_24h_smoke,
    )
    validation["expected_chunk_id"] = expected_chunk.chunk_id
    validation["archive_member_count"] = len(parsed)
    units = {name: fields[name].units for name in (*ERA5_VARIABLES, *ANCILLARY_VARIABLES)}
    finite_counts = {
        name: int(np.isfinite(fields[name].values).sum())
        for name in (*ERA5_VARIABLES, *ANCILLARY_VARIABLES)
    }
    report = CdsSmokeReport(
        file_name=source.name,
        file_sha256=_sha256(source),
        file_bytes=source.stat().st_size,
        members=tuple(sorted((member.summary for member in parsed), key=lambda value: value.name)),
        time_start_utc=reference.times_utc.min().isoformat(),
        time_end_utc=reference.times_utc.max().isoformat(),
        latitudes=tuple(float(value) for value in reference.latitudes),
        longitudes=tuple(float(value) for value in reference.longitudes),
        units=units,
        finite_value_counts=finite_counts,
        validation=validation,
    )
    return (
        report,
        reference.times_utc,
        reference.latitudes,
        reference.longitudes,
        fields,
    )


def validate_cds_smoke(path: str | Path, *, expected_chunk: RequestChunk) -> CdsSmokeReport:
    """Validate a CDS one-day 3x3 response without extracting or modifying it."""

    report, *_ = _validate_cds_archive(
        path,
        expected_chunk=expected_chunk,
        require_24h_smoke=True,
    )
    return report


def load_cds_chunk_frame(
    path: str | Path,
    *,
    expected_chunk: RequestChunk,
) -> tuple[CdsSmokeReport, pd.DataFrame]:
    """Validate one exact-hour CDS chunk and return its immutable 3x3 field table."""

    report, times, latitudes, longitudes, fields = _validate_cds_archive(
        path,
        expected_chunk=expected_chunk,
        require_24h_smoke=False,
    )
    index = pd.MultiIndex.from_product(
        [times, latitudes, longitudes],
        names=["time_utc", "latitude", "longitude"],
    )
    frame = pd.DataFrame(
        {
            name: np.asarray(fields[name].values, dtype=np.float64).reshape(-1)
            for name in (*ERA5_VARIABLES, *ANCILLARY_VARIABLES)
        },
        index=index,
    ).reset_index()
    frame.insert(
        1,
        "time_kst",
        pd.to_datetime(frame["time_utc"], utc=True).dt.tz_convert("Asia/Seoul"),
    )
    frame.insert(0, "chunk_id", expected_chunk.chunk_id)
    frame.insert(1, "block", expected_chunk.block)
    return report, frame
