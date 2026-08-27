"""Secret-free provenance receipts for ERA5 request plans and smoke subsets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from p2_restore.era5_arco import (
    ARCO_OFFICIAL_REPOSITORY,
    ARCO_URI,
    CATALOG_SHA256,
    COPERNICUS_LICENSE_URL,
    ERA5_DOI,
    GOOGLE_PUBLIC_DATASETS_URL,
    ArcoMetadataReport,
)
from p2_restore.era5_request import ANCILLARY_VARIABLES, DATASET_ID, ERA5_VARIABLES, RequestChunk

CDS_DATASET_URL = (
    "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview"
)
CDS_API_GUIDE_URL = "https://cds.climate.copernicus.eu/en/how-to-api"
ECMWF_ARCO_PUG_URL = "https://confluence.ecmwf.int/pages/viewpage.action?pageId=704944501"
FORBIDDEN_KEY_PARTS = ("token", "password", "secret", "authorization", "credential")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_keys(nested)


def assert_secret_free(payload: Mapping[str, Any], secret_values: Iterable[str] = ()) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for key in _walk_keys(payload):
        if any(part in key.lower() for part in FORBIDDEN_KEY_PARTS):
            raise ValueError(f"secret-like key is forbidden in ERA5 receipts: {key}")
    for secret in secret_values:
        if secret and secret in serialized:
            raise ValueError("secret value entered an ERA5 receipt")


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    assert_secret_free(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_provenance() -> dict[str, Any]:
    return {
        "dataset": "ERA5 hourly data on single levels",
        "dataset_id": DATASET_ID,
        "doi": ERA5_DOI,
        "cds_dataset_url": CDS_DATASET_URL,
        "cds_api_guide_url": CDS_API_GUIDE_URL,
        "ecmwf_arco_product_guide": ECMWF_ARCO_PUG_URL,
        "google_research_arco_repository": ARCO_OFFICIAL_REPOSITORY,
        "google_cloud_public_datasets": GOOGLE_PUBLIC_DATASETS_URL,
        "anonymous_store": ARCO_URI,
        "licence": "Copernicus licence; attribution required",
        "licence_url": COPERNICUS_LICENSE_URL,
        "catalog_sha256": CATALOG_SHA256,
        "attribution": (
            "Contains modified Copernicus Climate Change Service information; neither the "
            "European Commission nor ECMWF is responsible for downstream use."
        ),
    }


def build_request_plan_receipt(
    smoke: RequestChunk,
    chunks: Iterable[RequestChunk],
    *,
    frozen_oof_sha256: str,
) -> dict[str, Any]:
    planned = tuple(chunks)
    payload = {
        "schema_version": "1.0",
        "experiment_id": "p2_era5_primary_scaffold_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "research_only": True,
        "upload_allowed": False,
        "source": source_provenance(),
        "fixed_feature_variables": list(ERA5_VARIABLES),
        "validation_ancillary": list(ANCILLARY_VARIABLES),
        "format_order": ["grib", "netcdf"],
        "smoke": smoke.public_dict(),
        "oof_context": {
            "frozen_oof_sha256": frozen_oof_sha256,
            "padding_days": 7,
            "chunks": [chunk.public_dict() for chunk in planned],
            "chunk_count": len(planned),
            "hour_count": sum(len(chunk.timestamps_utc()) for chunk in planned),
        },
        "network_action_taken": False,
    }
    assert_secret_free(payload)
    return payload


def build_arco_smoke_receipt(
    frame_path: Path,
    frame: pd.DataFrame,
    metadata: ArcoMetadataReport,
    validation: Mapping[str, Any],
    *,
    dependency_versions: Mapping[str, str],
    code_sha256: Mapping[str, str],
) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "experiment_id": "p2_era5_arco_anonymous_smoke_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "research_only": True,
        "upload_allowed": False,
        "source": source_provenance(),
        "access": {
            "anonymous": True,
            "store_protocol": "Google Cloud Storage Zarr with token=anon in memory only",
            "metadata": metadata.public_dict(),
        },
        "subset": {
            "feature_variables": list(ERA5_VARIABLES),
            "validation_ancillary": list(ANCILLARY_VARIABLES),
            "rows": len(frame),
            "time_start_utc": pd.to_datetime(frame["time_utc"], utc=True).min().isoformat(),
            "time_end_utc": pd.to_datetime(frame["time_utc"], utc=True).max().isoformat(),
            "latitudes": sorted(frame["latitude"].unique().tolist()),
            "longitudes": sorted(frame["longitude"].unique().tolist()),
            "file_name": frame_path.name,
            "file_sha256": sha256(frame_path),
            "file_bytes": frame_path.stat().st_size,
        },
        "validation": dict(validation),
        "runtime": dict(dependency_versions),
        "code_sha256": dict(code_sha256),
        "hidden_target_values_used": False,
        "model_or_submission_modified": False,
    }
    assert_secret_free(payload)
    return payload


def build_cds_smoke_validation_receipt(
    validation: Mapping[str, Any],
    *,
    dependency_versions: Mapping[str, str],
    code_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build an aggregate-only receipt for a pre-existing local CDS smoke file."""

    payload = {
        "schema_version": "1.0",
        "experiment_id": "p2_era5_cds_smoke_validation_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "research_only": True,
        "upload_allowed": False,
        "source": source_provenance(),
        "smoke_validation": dict(validation),
        "runtime": dict(dependency_versions),
        "code_sha256": dict(code_sha256),
        "validation_network_action_taken": False,
        "raw_file_modified": False,
        "hidden_target_values_used": False,
        "model_or_submission_modified": False,
    }
    assert_secret_free(payload)
    return payload


def build_arco_metadata_receipt(
    metadata: ArcoMetadataReport,
    transfer_gate: Mapping[str, Any],
    *,
    dependency_versions: Mapping[str, str],
    code_sha256: Mapping[str, str],
) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "experiment_id": "p2_era5_arco_anonymous_metadata_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "research_only": True,
        "upload_allowed": False,
        "source": source_provenance(),
        "access": {
            "anonymous": True,
            "metadata_only": True,
            "store_protocol": "Google Cloud Storage Zarr metadata with anonymous access",
            "metadata": metadata.public_dict(),
        },
        "transfer_gate": dict(transfer_gate),
        "decision": "NO_GO_ANONYMOUS_ARCO_TRANSFER",
        "data_array_read": False,
        "hidden_target_values_used": False,
        "model_or_submission_modified": False,
        "runtime": dict(dependency_versions),
        "code_sha256": dict(code_sha256),
    }
    assert_secret_free(payload)
    return payload
