"""Credential-safe CDS downloader with GRIB-first, NetCDF-fallback behavior."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from p2_restore.era5_preflight import (
    CDS_API_URL_DEFAULT,
    credential_preflight,
    preferred_format_order,
)
from p2_restore.era5_request import DATASET_ID, RequestChunk


class Era5DownloadBlocked(RuntimeError):
    """Raised before any network operation when a preflight gate is closed."""


class Era5DownloadFailed(RuntimeError):
    """Raised without embedding CDS client exceptions, which may contain secrets."""


@dataclass(frozen=True)
class DownloadedChunk:
    path: Path
    data_format: str
    bytes: int
    chunk_id: str

    def public_dict(self) -> dict[str, object]:
        return {
            "file_name": self.path.name,
            "data_format": self.data_format,
            "bytes": self.bytes,
            "chunk_id": self.chunk_id,
        }


def download_cds_chunk(
    chunk: RequestChunk,
    target_directory: Path,
    *,
    execute_download: bool,
    environment: Mapping[str, str] | None = None,
) -> DownloadedChunk:
    env = os.environ if environment is None else environment
    preflight = credential_preflight(env)
    if preflight.status == "awaiting_credential":
        raise Era5DownloadBlocked("CDS credential or terms acknowledgement is absent")
    if preflight.status == "awaiting_runtime":
        raise Era5DownloadBlocked("CDS client or a supported ERA5 reader is absent")
    if not execute_download:
        raise Era5DownloadBlocked("explicit execute_download authorization is absent")
    formats = preferred_format_order(preflight)
    if not formats:
        raise Era5DownloadBlocked("neither GRIB nor NetCDF reader is available")

    import cdsapi

    target_directory.mkdir(parents=True, exist_ok=True)
    api_url = env.get("CDSAPI_URL", CDS_API_URL_DEFAULT).strip().rstrip("/")
    api_key = env["CDSAPI_KEY"]
    try:
        client = cdsapi.Client(url=api_url, key=api_key, quiet=True, debug=False)
    except Exception:
        raise Era5DownloadFailed("CDS client initialization failed") from None
    for data_format in formats:
        target = target_directory / chunk.target_name(data_format)
        partial = target.with_suffix(target.suffix + ".partial")
        if target.exists() or partial.exists():
            raise FileExistsError(f"ERA5 target already exists: {target.name}")
        try:
            client.retrieve(DATASET_ID, chunk.request(data_format), str(partial))
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise OSError("empty CDS response")
            partial.replace(target)
            return DownloadedChunk(target, data_format, target.stat().st_size, chunk.chunk_id)
        except Exception:
            partial.unlink(missing_ok=True)
            continue
    raise Era5DownloadFailed(f"CDS GRIB and NetCDF retrieval failed for {chunk.chunk_id}")
