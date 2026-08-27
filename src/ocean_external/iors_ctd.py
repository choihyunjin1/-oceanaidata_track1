"""Read and audit the official KIOST I-ORS OceanSITES CTD archive.

The module is deliberately independent from :mod:`p1_qc`.  It only accepts
OceanSITES values whose variable-specific flag equals ``1`` and it applies the
fixed KST cutoff before a profile can reach an experiment.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class IorsCtdError(RuntimeError):
    """Raised when source integrity or OceanSITES structure is invalid."""


@dataclass(frozen=True)
class YearProfile:
    """One year on the fixed P1 depth grid; non-QC1 values are NaN."""

    year: int
    time_utc: np.ndarray
    target_layers: np.ndarray
    target_depths: np.ndarray
    temp: np.ndarray
    psal: np.ndarray
    depth: np.ndarray
    depth_qc1: np.ndarray
    mapping: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class LooDataset:
    """Aggregate leave-one-layer-out feature matrix kept only in memory."""

    x: np.ndarray
    y: np.ndarray
    baseline: np.ndarray
    layer: np.ndarray
    year: np.ndarray
    feature_names: tuple[str, ...]
    group_counts: dict[str, int]


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IorsCtdError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _as_text(value.reshape(-1)[0])
    return str(value)


def _aware_datetime(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IorsCtdError(f"{field} must be ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise IorsCtdError(f"{field} must include a timezone offset")
    return parsed


def validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed before downloading or decoding observational values."""

    if manifest.get("schema_version") != "1.0":
        raise IorsCtdError("source manifest schema_version must be 1.0")
    if manifest.get("source_id") != "i_ors_ctd_2014_2023":
        raise IorsCtdError("unexpected source_id")
    archive = manifest.get("archive")
    coverage = manifest.get("coverage")
    license_value = manifest.get("license")
    if not all(isinstance(item, Mapping) for item in (archive, coverage, license_value)):
        raise IorsCtdError("manifest archive, coverage, and license must be objects")
    assert isinstance(archive, Mapping)
    assert isinstance(coverage, Mapping)
    assert isinstance(license_value, Mapping)
    if not str(manifest.get("record_url", "")).startswith("https://sciwatch.kiost.ac.kr/"):
        raise IorsCtdError("record_url must be the official KIOST HTTPS host")
    if not str(archive.get("url", "")).startswith("https://sciwatch.kiost.ac.kr/"):
        raise IorsCtdError("archive URL must be the official KIOST HTTPS host")
    expected_sha = str(archive.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise IorsCtdError("archive SHA256 is invalid")
    if int(archive.get("size_bytes", 0)) <= 0 or int(archive.get("member_count", 0)) != 10:
        raise IorsCtdError("archive size/member contract is invalid")
    if license_value.get("spdx") != "CC-BY-4.0":
        raise IorsCtdError("only the verified CC-BY-4.0 source is accepted")
    cutoff_kst = _aware_datetime(str(coverage.get("hard_cutoff_kst")), field="hard_cutoff_kst")
    cutoff_utc = _aware_datetime(str(coverage.get("hard_cutoff_utc")), field="hard_cutoff_utc")
    if cutoff_kst.astimezone(UTC) != cutoff_utc.astimezone(UTC):
        raise IorsCtdError("KST and UTC cutoffs describe different instants")
    if cutoff_kst.isoformat() != "2023-12-31T23:50:00+09:00":
        raise IorsCtdError("the fixed pre-2024 cutoff must not be relaxed")
    if list(coverage.get("years", [])) != list(range(2014, 2024)):
        raise IorsCtdError("source years must be exactly 2014..2023")
    if int(manifest.get("variables", {}).get("accepted_qc", -1)) != 1:
        raise IorsCtdError("only OceanSITES QC=1 may be accepted")


def verify_official_record(manifest: Mapping[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    """Verify DOI, licence, and version markers on the official item page."""

    validate_source_manifest(manifest)
    request = urllib.request.Request(
        str(manifest["record_url"]),
        headers={"User-Agent": "OceanAI-external-repro-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        body = response.read()
        status = int(getattr(response, "status", 200))
        content_type = str(response.headers.get("Content-Type", ""))
        charset = response.headers.get_content_charset() or "utf-8"
    if status != 200:
        raise IorsCtdError(f"official record returned HTTP {status}")
    text = body.decode(charset, "replace")
    markers = list(manifest["verification"]["record_page_required_markers"])
    missing = [marker for marker in markers if str(marker) not in text]
    if missing:
        raise IorsCtdError(f"official record is missing markers: {missing}")
    return {
        "url": str(manifest["record_url"]),
        "http_status": status,
        "content_type": content_type,
        "body_sha256": _sha256_bytes(body),
        "required_markers": markers,
        "markers_verified": True,
    }


def archive_path(manifest: Mapping[str, Any], quarantine_root: str | Path) -> Path:
    validate_source_manifest(manifest)
    return (
        Path(quarantine_root)
        / str(manifest["source_id"])
        / f"v{manifest['version']}"
        / str(manifest["archive"]["filename"])
    )


def ensure_archive(
    manifest: Mapping[str, Any],
    quarantine_root: str | Path,
    *,
    allow_download: bool,
    timeout: float = 90.0,
) -> Path:
    """Download atomically into quarantine and verify byte size and SHA256."""

    path = archive_path(manifest, quarantine_root)
    expected_size = int(manifest["archive"]["size_bytes"])
    expected_sha = str(manifest["archive"]["sha256"]).lower()
    if path.is_file():
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != expected_size or actual_sha != expected_sha:
            raise IorsCtdError(
                f"existing quarantine archive failed integrity: size={actual_size}, sha={actual_sha}"
            )
        return path
    if not allow_download:
        raise IorsCtdError(f"verified archive is absent and download is disabled: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise IorsCtdError(f"partial download already exists; inspect before retrying: {partial}")
    request = urllib.request.Request(
        str(manifest["archive"]["url"]),
        headers={"User-Agent": "OceanAI-external-repro-audit/1.0"},
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            partial.open("xb") as out,
        ):  # noqa: S310
            if int(getattr(response, "status", 200)) != 200:
                raise IorsCtdError(f"archive download returned HTTP {response.status}")
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    except Exception:
        # Preserve a partial file for forensic inspection; never silently overwrite it.
        raise
    if size != expected_size or digest.hexdigest() != expected_sha:
        raise IorsCtdError(f"download integrity failed: size={size}, sha256={digest.hexdigest()}")
    partial.replace(path)
    return path


def _require_h5py() -> Any:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise IorsCtdError(
            "h5py is optional; install requirements-external.txt into an isolated target"
        ) from exc
    return h5py


def _member_year(name: str) -> int:
    match = re.fullmatch(r"OS_I-ORS_(\d{4})_D_ocean_CTD\.nc", name)
    if match is None:
        raise IorsCtdError(f"unexpected archive member: {name}")
    return int(match.group(1))


def verify_archive(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify ZIP topology and per-member OceanSITES provenance attributes."""

    validate_source_manifest(manifest)
    archive_value = manifest["archive"]
    archive_file = Path(path)
    actual_size = archive_file.stat().st_size
    actual_sha = sha256_file(archive_file)
    if actual_size != int(archive_value["size_bytes"]):
        raise IorsCtdError("archive size mismatch")
    if actual_sha != str(archive_value["sha256"]).lower():
        raise IorsCtdError("archive SHA256 mismatch")
    expected_years = list(manifest["coverage"]["years"])
    required = manifest["verification"]["archive_required_global_attributes"]
    h5py = _require_h5py()
    members: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_file) as bundle:
        infos = [info for info in bundle.infolist() if not info.is_dir()]
        if len(infos) != int(archive_value["member_count"]):
            raise IorsCtdError("archive member count mismatch")
        if sum(info.file_size for info in infos) != int(archive_value["uncompressed_size_bytes"]):
            raise IorsCtdError("archive uncompressed size mismatch")
        years = sorted(_member_year(info.filename) for info in infos)
        if years != expected_years:
            raise IorsCtdError(f"archive years mismatch: {years}")
        for info in sorted(infos, key=lambda item: item.filename):
            payload = bundle.read(info.filename)
            year = _member_year(info.filename)
            with h5py.File(io.BytesIO(payload), "r") as dataset:
                attrs = {key: _as_text(value) for key, value in dataset.attrs.items()}
                checks = {
                    "platform_code": str(required["platform_code_contains"])
                    in attrs.get("platform_code", ""),
                    "citation": str(required["citation_contains"]) in attrs.get("citation", ""),
                    "license": str(required["license_contains"]) in attrs.get("license", ""),
                    "conventions": str(required["conventions_contains"])
                    in attrs.get("Conventions", ""),
                    "time_units": _as_text(dataset["TIME"].attrs.get("units", ""))
                    == str(manifest["variables"]["time_units"]),
                    "temp_qc_pairs": all(
                        f"{name}_QC" in dataset
                        for name in dataset.keys()
                        if re.fullmatch(r"TEMP\d+", name)
                    ),
                }
                if not all(checks.values()):
                    raise IorsCtdError(
                        f"OceanSITES provenance failed for {info.filename}: {checks}"
                    )
                members.append(
                    {
                        "year": year,
                        "name": info.filename,
                        "size_bytes": info.file_size,
                        "sha256": _sha256_bytes(payload),
                        "time_count": int(dataset["TIME"].shape[0]),
                        "temperature_series": sum(
                            bool(re.fullmatch(r"TEMP\d+", name)) for name in dataset.keys()
                        ),
                        "time_coverage_start": attrs.get("time_coverage_start"),
                        "time_coverage_end": attrs.get("time_coverage_end"),
                        "checks": checks,
                    }
                )
    return {
        "path": str(archive_file.resolve()),
        "size_bytes": actual_size,
        "sha256": actual_sha,
        "member_count": len(members),
        "uncompressed_size_bytes": sum(item["size_bytes"] for item in members),
        "members": members,
        "integrity_verified": True,
    }


def _decode_oceansites_time(days: np.ndarray, units: str) -> np.ndarray:
    if units != "days since 1950-01-01T00:00:00Z":
        raise IorsCtdError(f"unsupported TIME units: {units}")
    if days.ndim != 1 or not np.isfinite(days).all():
        raise IorsCtdError("TIME must be a finite one-dimensional array")
    seconds = np.rint(days.astype(np.float64) * 86400.0).astype(np.int64)
    return np.datetime64("1950-01-01T00:00:00", "s") + seconds.astype("timedelta64[s]")


def _nominal_depth(dataset: Any) -> float:
    comment = _as_text(dataset.attrs.get("comment", ""))
    match = re.search(r"target\s+depth\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*m", comment, re.I)
    if match is None:
        raise IorsCtdError(f"cannot parse nominal depth from {comment!r}")
    return float(match.group(1))


def _good_values(value: np.ndarray, qc: np.ndarray, *, low: float, high: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    qc = np.asarray(qc, dtype=np.float64)
    if value.shape != qc.shape:
        raise IorsCtdError("value/QC shape mismatch")
    good = (qc == 1.0) & np.isfinite(value) & (value >= low) & (value <= high)
    return np.where(good, value, np.nan)


def _map_series_to_layers(
    series: Sequence[dict[str, Any]],
    target_layers: np.ndarray,
    target_depths: np.ndarray,
    *,
    max_distance_m: float,
) -> dict[int, dict[str, Any]]:
    pairs: list[tuple[float, int, int]] = []
    for source_position, item in enumerate(series):
        for target_position, target_depth in enumerate(target_depths):
            distance = abs(float(item["mapping_depth_m"]) - float(target_depth))
            if distance <= max_distance_m:
                pairs.append((distance, source_position, target_position))
    mapped: dict[int, dict[str, Any]] = {}
    used_sources: set[int] = set()
    for distance, source_position, target_position in sorted(pairs):
        layer = int(target_layers[target_position])
        if source_position in used_sources or layer in mapped:
            continue
        item = dict(series[source_position])
        item["target_position"] = target_position
        item["target_layer"] = layer
        item["target_depth_m"] = float(target_depths[target_position])
        item["mapping_distance_m"] = float(distance)
        mapped[layer] = item
        used_sources.add(source_position)
    return mapped


def read_year_profile(
    archive_file: str | Path,
    manifest: Mapping[str, Any],
    *,
    year: int,
    target_depth_by_layer: Mapping[int, float],
    max_mapping_distance_m: float,
) -> YearProfile:
    """Decode one member, apply cutoff/QC1, then map by actual deployment depth."""

    validate_source_manifest(manifest)
    if year not in manifest["coverage"]["years"]:
        raise IorsCtdError(f"year is not in source manifest: {year}")
    target_layers = np.asarray(
        sorted(int(value) for value in target_depth_by_layer), dtype=np.int16
    )
    target_depths = np.asarray(
        [float(target_depth_by_layer[int(layer)]) for layer in target_layers], dtype=np.float64
    )
    member = f"OS_I-ORS_{year}_D_ocean_CTD.nc"
    h5py = _require_h5py()
    with zipfile.ZipFile(archive_file) as bundle:
        try:
            payload = bundle.read(member)
        except KeyError as exc:
            raise IorsCtdError(f"archive member is missing: {member}") from exc
    with h5py.File(io.BytesIO(payload), "r") as dataset:
        time = _decode_oceansites_time(
            np.asarray(dataset["TIME"][...]), _as_text(dataset["TIME"].attrs.get("units", ""))
        )
        cutoff = _aware_datetime(
            str(manifest["coverage"]["hard_cutoff_utc"]), field="hard_cutoff_utc"
        ).astimezone(UTC)
        cutoff_value = np.datetime64(cutoff.replace(tzinfo=None), "s")
        keep = time <= cutoff_value
        kept_time = time[keep]
        if kept_time.size == 0:
            raise IorsCtdError(f"no rows remain after cutoff for {year}")
        if np.any(np.diff(kept_time).astype("timedelta64[s]").astype(np.int64) != 600):
            raise IorsCtdError(f"{year} TIME grid is not strictly 10-minute")

        source_series: list[dict[str, Any]] = []
        for name in sorted(dataset.keys()):
            match = re.fullmatch(r"TEMP(\d+)", name)
            if match is None:
                continue
            suffix = match.group(1)
            depth_name = f"DEPTH{suffix}"
            depth_qc_name = f"{depth_name}_QC"
            if depth_name not in dataset or depth_qc_name not in dataset:
                raise IorsCtdError(f"depth/QC datasets are missing for {name}")
            nominal = _nominal_depth(dataset[name])
            depth_good = _good_values(
                dataset[depth_name][...], dataset[depth_qc_name][...], low=0.0, high=100.0
            )
            finite_depth = depth_good[np.isfinite(depth_good)]
            median_depth = float(np.median(finite_depth)) if finite_depth.size else nominal
            source_series.append(
                {
                    "suffix": suffix,
                    "temp_name": name,
                    "psal_name": f"PSAL{suffix}",
                    "depth_name": depth_name,
                    "nominal_depth_m": nominal,
                    "median_qc1_depth_m": median_depth,
                    "mapping_depth_m": median_depth,
                    "depth_qc1_count": int(finite_depth.size),
                }
            )

        mapping = _map_series_to_layers(
            source_series,
            target_layers,
            target_depths,
            max_distance_m=max_mapping_distance_m,
        )
        shape = (kept_time.size, target_layers.size)
        temp = np.full(shape, np.nan, dtype=np.float64)
        psal = np.full(shape, np.nan, dtype=np.float64)
        depth = np.full(shape, np.nan, dtype=np.float64)
        depth_qc1 = np.zeros(shape, dtype=bool)
        mapping_audit: list[dict[str, Any]] = []
        for layer in target_layers:
            layer_value = int(layer)
            if layer_value not in mapping:
                continue
            item = mapping[layer_value]
            position = int(item["target_position"])
            suffix = str(item["suffix"])
            temp_values = _good_values(
                dataset[item["temp_name"]][...],
                dataset[f"{item['temp_name']}_QC"][...],
                low=-3.0,
                high=45.0,
            )[keep]
            temp[:, position] = temp_values
            psal_name = str(item["psal_name"])
            if psal_name in dataset and f"{psal_name}_QC" in dataset:
                psal[:, position] = _good_values(
                    dataset[psal_name][...], dataset[f"{psal_name}_QC"][...], low=0.0, high=50.0
                )[keep]
            depth_name = str(item["depth_name"])
            raw_depth = np.asarray(dataset[depth_name][...], dtype=np.float64)[keep]
            raw_depth_qc = np.asarray(dataset[f"{depth_name}_QC"][...], dtype=np.float64)[keep]
            good_depth = (
                (raw_depth_qc == 1.0)
                & np.isfinite(raw_depth)
                & (raw_depth >= 0.0)
                & (raw_depth <= 100.0)
            )
            depth_qc1[:, position] = good_depth
            depth[:, position] = np.where(good_depth, raw_depth, float(item["median_qc1_depth_m"]))
            mapping_audit.append(
                {
                    "target_layer": layer_value,
                    "target_depth_m": float(item["target_depth_m"]),
                    "source_temperature": str(item["temp_name"]),
                    "nominal_depth_m": float(item["nominal_depth_m"]),
                    "median_qc1_depth_m": float(item["median_qc1_depth_m"]),
                    "mapping_distance_m": float(item["mapping_distance_m"]),
                    "temp_qc1_count_after_cutoff": int(np.isfinite(temp_values).sum()),
                    "psal_qc1_count_after_cutoff": int(np.isfinite(psal[:, position]).sum()),
                    "depth_qc1_count_after_cutoff": int(good_depth.sum()),
                }
            )
    audit = {
        "year": year,
        "member": member,
        "time_rows_raw": int(time.size),
        "time_rows_after_cutoff": int(kept_time.size),
        "time_rows_dropped_by_cutoff": int((~keep).sum()),
        "time_start_utc": str(kept_time.min()),
        "time_end_utc": str(kept_time.max()),
        "cadence_minutes": 10,
        "irregular_steps": 0,
        "source_series_count": len(source_series),
        "mapped_layer_count": len(mapping_audit),
        "mapping": mapping_audit,
    }
    return YearProfile(
        year=year,
        time_utc=kept_time,
        target_layers=target_layers,
        target_depths=target_depths,
        temp=temp,
        psal=psal,
        depth=depth,
        depth_qc1=depth_qc1,
        mapping=tuple(mapping_audit),
        audit=audit,
    )


def depth_linear_baseline(profile: YearProfile, target_position: int) -> np.ndarray:
    """Interpolate the masked target from the nearest valid depths on each side."""

    temp = np.asarray(profile.temp, dtype=np.float64).copy()
    temp[:, target_position] = np.nan
    depth = np.asarray(profile.depth, dtype=np.float64)
    target_depth = np.where(
        np.isfinite(depth[:, target_position]),
        depth[:, target_position],
        profile.target_depths[target_position],
    )
    valid = np.isfinite(temp) & np.isfinite(depth)
    delta = depth - target_depth[:, None]
    left_distance = np.where(valid & (delta < 0.0), -delta, np.inf)
    right_distance = np.where(valid & (delta > 0.0), delta, np.inf)
    left_position = np.argmin(left_distance, axis=1)
    right_position = np.argmin(right_distance, axis=1)
    rows = np.arange(temp.shape[0])
    has_left = np.isfinite(left_distance[rows, left_position])
    has_right = np.isfinite(right_distance[rows, right_position])
    left_temp = temp[rows, left_position]
    right_temp = temp[rows, right_position]
    left_depth = depth[rows, left_position]
    right_depth = depth[rows, right_position]
    result = np.full(temp.shape[0], np.nan, dtype=np.float64)
    both = has_left & has_right & (right_depth > left_depth)
    weight = np.divide(
        target_depth - left_depth,
        right_depth - left_depth,
        out=np.zeros_like(target_depth),
        where=right_depth > left_depth,
    )
    result[both] = left_temp[both] + weight[both] * (right_temp[both] - left_temp[both])
    only_left = has_left & ~has_right
    only_right = has_right & ~has_left
    result[only_left] = left_temp[only_left]
    result[only_right] = right_temp[only_right]
    return result


def _feature_names(layer_count: int) -> tuple[str, ...]:
    fixed = (
        "target_layer",
        "target_nominal_depth",
        "target_actual_depth",
        "target_depth_qc1",
        "doy_sin",
        "doy_cos",
        "minute_sin",
        "minute_cos",
        "target_psal",
        "target_psal_mask",
        "peer_count",
        "peer_median",
        "peer_std",
        "peer_range",
        "depth_linear_baseline",
    )
    blocks = []
    for prefix in ("peer_temp", "peer_depth", "peer_temp_mask", "psal", "psal_mask"):
        blocks.extend(f"{prefix}_layer_{index + 1}" for index in range(layer_count))
    return fixed + tuple(blocks)


def _even_sample(index: np.ndarray, maximum: int | None) -> np.ndarray:
    if maximum is None or index.size <= maximum:
        return index
    positions = np.linspace(0, index.size - 1, num=maximum, dtype=np.int64)
    return index[positions]


def build_loo_dataset(
    profiles: Sequence[YearProfile],
    *,
    min_peer_temperatures: int,
    max_rows_per_year_layer: int | None,
) -> LooDataset:
    """Mask each target TEMP while retaining only same-time QC1 peer values."""

    if not profiles:
        raise IorsCtdError("at least one YearProfile is required")
    layer_count = int(profiles[0].target_layers.size)
    feature_names = _feature_names(layer_count)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    layer_parts: list[np.ndarray] = []
    year_parts: list[np.ndarray] = []
    group_counts: dict[str, int] = {}
    for profile in profiles:
        if profile.target_layers.size != layer_count:
            raise IorsCtdError("all profiles must use the same target grid")
        time_kst = profile.time_utc + np.timedelta64(9, "h")
        day = time_kst.astype("datetime64[D]")
        year_start = time_kst.astype("datetime64[Y]")
        day_index = (day - year_start).astype(np.int64)
        minute_index = (time_kst - day).astype("timedelta64[m]").astype(np.int64)
        annual_angle = 2.0 * math.pi * (day_index + minute_index / 1440.0) / 365.2425
        minute_angle = 2.0 * math.pi * minute_index / 1440.0
        for target_position, layer in enumerate(profile.target_layers):
            masked_temp = profile.temp.copy()
            masked_temp[:, target_position] = np.nan
            peer_mask = np.isfinite(masked_temp)
            peer_count = peer_mask.sum(axis=1)
            baseline = depth_linear_baseline(profile, target_position)
            eligible = (
                np.isfinite(profile.temp[:, target_position])
                & (peer_count >= min_peer_temperatures)
                & np.isfinite(baseline)
            )
            index = _even_sample(np.flatnonzero(eligible), max_rows_per_year_layer)
            if index.size == 0:
                continue
            selected_temp = masked_temp[index]
            selected_mask = peer_mask[index]
            peer_median = np.nanmedian(selected_temp, axis=1)
            peer_std = np.nanstd(selected_temp, axis=1)
            peer_max = np.max(np.where(selected_mask, selected_temp, -np.inf), axis=1)
            peer_min = np.min(np.where(selected_mask, selected_temp, np.inf), axis=1)
            peer_range = peer_max - peer_min
            target_psal = profile.psal[:, target_position]
            target_actual_depth = profile.depth[:, target_position]
            fixed = np.column_stack(
                [
                    np.full(index.size, int(layer), dtype=np.float64),
                    np.full(index.size, profile.target_depths[target_position], dtype=np.float64),
                    target_actual_depth[index],
                    profile.depth_qc1[index, target_position].astype(np.float64),
                    np.sin(annual_angle[index]),
                    np.cos(annual_angle[index]),
                    np.sin(minute_angle[index]),
                    np.cos(minute_angle[index]),
                    target_psal[index],
                    np.isfinite(target_psal[index]).astype(np.float64),
                    peer_count[index].astype(np.float64),
                    peer_median,
                    peer_std,
                    peer_range,
                    baseline[index],
                ]
            )
            x = np.column_stack(
                [
                    fixed,
                    selected_temp,
                    profile.depth[index],
                    selected_mask.astype(np.float64),
                    profile.psal[index],
                    np.isfinite(profile.psal[index]).astype(np.float64),
                ]
            ).astype(np.float32, copy=False)
            if x.shape[1] != len(feature_names):
                raise AssertionError("feature width contract failed")
            # The only target-temperature copy becomes y; the matching peer slot is forced NaN.
            target_peer_column = len(fixed[0]) + target_position
            if np.isfinite(x[:, target_peer_column]).any():
                raise AssertionError("target temperature leaked into peer features")
            group = f"{profile.year}:layer_{int(layer)}"
            group_counts[group] = int(index.size)
            x_parts.append(x)
            y_parts.append(profile.temp[index, target_position].astype(np.float32))
            baseline_parts.append(baseline[index].astype(np.float32))
            layer_parts.append(np.full(index.size, int(layer), dtype=np.int16))
            year_parts.append(np.full(index.size, profile.year, dtype=np.int16))
    if not x_parts:
        raise IorsCtdError("no eligible leave-one-layer-out rows")
    return LooDataset(
        x=np.concatenate(x_parts, axis=0),
        y=np.concatenate(y_parts),
        baseline=np.concatenate(baseline_parts),
        layer=np.concatenate(layer_parts),
        year=np.concatenate(year_parts),
        feature_names=feature_names,
        group_counts=group_counts,
    )
