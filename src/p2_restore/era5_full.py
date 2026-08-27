"""Fail-closed full ERA5 retrieval and aggregate-only validation for P2 OOF blocks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p2_restore.era5_arco import CATALOG_SHA256
from p2_restore.era5_cds import load_cds_chunk_frame
from p2_restore.era5_downloader import download_cds_chunk
from p2_restore.era5_manifest import (
    assert_secret_free,
    sha256,
    source_provenance,
    write_receipt,
)
from p2_restore.era5_preflight import credential_preflight
from p2_restore.era5_request import (
    ANCILLARY_VARIABLES,
    AREA_3X3,
    ERA5_VARIABLES,
    OOF_SHA256,
    build_registered_chunk_plan,
)

EXPERIMENT_ID = "p2_era5_full_retrieval_v1"
EXPECTED_CHUNK_COUNT = 17
EXPECTED_HOUR_COUNT = 4_900
EXPECTED_GRID_POINT_COUNT = 9
EXPECTED_ROW_COUNT = EXPECTED_HOUR_COUNT * EXPECTED_GRID_POINT_COUNT
P2_MAXIMUM_TIME_KST = pd.Timestamp("2025-12-31T23:59:59+09:00")
REFERENCE_TRANSFER_MIB_PER_SECOND = 2.0
VARIABLE_COLUMNS = (*ERA5_VARIABLES, *ANCILLARY_VARIABLES)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class FullProgress:
    """Experiment-local gauge that never records paths, requests, or access settings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()

    def update(
        self,
        *,
        progress: float,
        phase: str,
        detail: str,
        status: str = "running",
        completed_chunks: int = 0,
        downloaded_bytes: int = 0,
    ) -> None:
        elapsed = time.perf_counter() - self.started
        bounded = min(max(float(progress), 0.1), 100.0)
        remaining = elapsed * (100.0 - bounded) / bounded if bounded < 100.0 else 0.0
        payload = {
            "title": "P2 ERA5 exact-hour full retrieval",
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "progress": bounded,
            "phase": phase,
            "detail": detail,
            "completed_chunks": int(completed_chunks),
            "total_chunks": EXPECTED_CHUNK_COUNT,
            "downloaded_bytes": int(downloaded_bytes),
            "elapsed_seconds": elapsed,
            "eta": (
                datetime.now().astimezone() + timedelta(seconds=max(remaining, 0.0))
            ).isoformat(),
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        _write_json_atomic(self.path, payload)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("cdsapi", "numpy", "pandas", "pyarrow", "xarray", "netCDF4"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _inside_repo(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError("ERA5 output must stay inside the repository quarantine") from exc
    return relative.as_posix()


def _validate_registered_inputs(
    *,
    repo_root: Path,
    request_plan_path: Path,
    catalog_path: Path,
    permission_path: Path,
    smoke_receipt_path: Path,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    for path in (request_plan_path, catalog_path, permission_path, smoke_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"required ERA5 gate artifact is absent: {path.name}")

    if sha256(catalog_path) != CATALOG_SHA256:
        raise ValueError("external-data catalog SHA changed")
    chunks = build_registered_chunk_plan(pad_days=7)
    plan = _read_json_object(request_plan_path)
    assert_secret_free(plan)
    context = plan.get("oof_context", {})
    expected_public_chunks = [chunk.public_dict() for chunk in chunks]
    if (
        plan.get("schema_version") != "1.0"
        or plan.get("research_only") is not True
        or plan.get("upload_allowed") is not False
        or plan.get("fixed_feature_variables") != list(ERA5_VARIABLES)
        or plan.get("validation_ancillary") != list(ANCILLARY_VARIABLES)
        or context.get("frozen_oof_sha256") != OOF_SHA256
        or context.get("padding_days") != 7
        or context.get("chunk_count") != EXPECTED_CHUNK_COUNT
        or context.get("hour_count") != EXPECTED_HOUR_COUNT
        or context.get("chunks") != expected_public_chunks
    ):
        raise ValueError("saved ERA5 request plan differs from the preregistered exact-hour plan")

    permission = _read_json_object(permission_path)
    assert_secret_free(permission)
    cutoff = pd.Timestamp(permission.get("cutoff_by_problem", {}).get("P2", ""))
    if (
        permission.get("status") != "approved"
        or "era5_pre2024" not in permission.get("allowed_sources", [])
        or "P2" not in permission.get("allowed_problems", [])
        or "feature_design" not in permission.get("allowed_purposes", [])
        or cutoff.tzinfo is None
        or cutoff.tz_convert("Asia/Seoul") < P2_MAXIMUM_TIME_KST
    ):
        raise ValueError(
            "official external-data permission receipt does not cover this P2 ERA5 path"
        )
    evidence_name = permission.get("evidence_file")
    if not isinstance(evidence_name, str):
        raise ValueError("official permission evidence path is absent")
    evidence_path = repo_root / evidence_name
    if not evidence_path.is_file() or sha256(evidence_path) != permission.get("evidence_sha256"):
        raise ValueError("official external-data evidence hash changed")

    smoke = _read_json_object(smoke_receipt_path)
    assert_secret_free(smoke)
    validation = smoke.get("smoke_validation", {})
    if (
        validation.get("passed") is not True
        or validation.get("feature_variables") != list(ERA5_VARIABLES)
        or validation.get("validation_ancillary") != list(ANCILLARY_VARIABLES)
        or validation.get("validation", {}).get("grid_shape") != [3, 3]
        or validation.get("validation", {}).get("variable_count") != len(ERA5_VARIABLES)
        or smoke.get("hidden_target_values_used") is not False
        or smoke.get("model_or_submission_modified") is not False
    ):
        raise ValueError("validated ERA5 CDS smoke receipt changed or is incomplete")

    expected_times = pd.DatetimeIndex(
        sorted(timestamp for chunk in chunks for timestamp in chunk.timestamps_utc())
    )
    if (
        len(chunks) != EXPECTED_CHUNK_COUNT
        or len(expected_times) != EXPECTED_HOUR_COUNT
        or expected_times.duplicated().any()
        or expected_times.max().tz_convert("Asia/Seoul") > P2_MAXIMUM_TIME_KST
    ):
        raise ValueError("registered ERA5 timestamps violate cardinality, uniqueness, or cutoff")

    preflight = credential_preflight()
    if not preflight.ready:
        raise RuntimeError(f"ERA5 CDS access runtime is not ready: {preflight.status}")

    smoke_bytes = int(validation.get("file_bytes", 0))
    estimated_bytes = int(round(smoke_bytes / 24 * EXPECTED_HOUR_COUNT))
    estimated_seconds = estimated_bytes / (REFERENCE_TRANSFER_MIB_PER_SECOND * 2**20)
    gate = {
        "passed": True,
        "chunk_count": len(chunks),
        "unique_hour_count": len(expected_times),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "planned_start_utc": expected_times.min().isoformat(),
        "planned_end_utc": expected_times.max().isoformat(),
        "planned_start_kst": expected_times.min().tz_convert("Asia/Seoul").isoformat(),
        "planned_end_kst": expected_times.max().tz_convert("Asia/Seoul").isoformat(),
        "maximum_allowed_time_kst": P2_MAXIMUM_TIME_KST.isoformat(),
        "estimated_transfer_bytes_from_smoke": estimated_bytes,
        "estimated_wire_seconds_at_2_mib_s": estimated_seconds,
        "request_plan_sha256": sha256(request_plan_path),
        "catalog_sha256": sha256(catalog_path),
        "permission_receipt_sha256": sha256(permission_path),
        "permission_evidence_sha256": sha256(evidence_path),
        "smoke_receipt_sha256": sha256(smoke_receipt_path),
        "frozen_oof_sha256_from_saved_plan": OOF_SHA256,
        "frozen_oof_read": False,
        "source_scope_note": (
            "P2 uses ERA5 meteorology-only covariates under the official P2 cutoff; "
            "no ERA5 temperature, salinity, or ocean-profile target is requested."
        ),
        "catalog_legacy_scope_caveat": (
            "The catalog source id retains its initial pre2024/P3 name; later P2 README and "
            "official permission receipt authorize independent ERA5 covariates through the P2 cutoff."
        ),
    }
    return chunks, gate


def _validate_combined_frame(
    frame: pd.DataFrame,
    *,
    chunks: tuple[Any, ...],
    cutoff_kst: pd.Timestamp,
) -> dict[str, Any]:
    expected_columns = [
        "chunk_id",
        "block",
        "time_utc",
        "time_kst",
        "latitude",
        "longitude",
        *VARIABLE_COLUMNS,
    ]
    if frame.columns.tolist() != expected_columns:
        raise ValueError("combined ERA5 column order changed")
    if len(frame) != EXPECTED_ROW_COUNT:
        raise ValueError("combined ERA5 row count changed")
    utc = pd.to_datetime(frame["time_utc"], utc=True, errors="raise")
    kst = pd.to_datetime(frame["time_kst"], utc=True, errors="raise").dt.tz_convert("Asia/Seoul")
    key_columns = ["time_utc", "latitude", "longitude"]
    if frame[key_columns].duplicated().any():
        raise ValueError("combined ERA5 contains duplicate time/grid keys")
    expected_times = pd.DatetimeIndex(
        sorted(timestamp for chunk in chunks for timestamp in chunk.timestamps_utc())
    )
    if not pd.DatetimeIndex(sorted(utc.unique())).equals(expected_times):
        raise ValueError("combined ERA5 hour coverage differs from the exact registered plan")
    observed_grid = (
        frame[["latitude", "longitude"]].drop_duplicates().sort_values(["latitude", "longitude"])
    )
    expected_grid = (
        pd.MultiIndex.from_product(
            [
                [AREA_3X3[2], (AREA_3X3[0] + AREA_3X3[2]) / 2, AREA_3X3[0]],
                [AREA_3X3[1], (AREA_3X3[1] + AREA_3X3[3]) / 2, AREA_3X3[3]],
            ],
            names=["latitude", "longitude"],
        )
        .to_frame(index=False)
        .sort_values(["latitude", "longitude"])
    )
    if not np.allclose(observed_grid.to_numpy(), expected_grid.to_numpy()):
        raise ValueError("combined ERA5 grid differs from the fixed S-ORS 3x3 grid")
    expected_kst = utc.dt.tz_convert("Asia/Seoul")
    if not (kst.array == expected_kst.array).all():
        raise ValueError("ERA5 UTC-to-KST conversion changed instants")
    wall_clock_offset = (
        kst.dt.tz_localize(None) - utc.dt.tz_localize(None)
    ).dt.total_seconds() / 60
    if not (wall_clock_offset == 540.0).all():
        raise ValueError("ERA5 KST wall-clock offset is not exactly +09:00")
    if kst.max() > cutoff_kst:
        raise ValueError("ERA5 P2 frame exceeds the approved time cutoff")

    numeric = frame.loc[:, VARIABLE_COLUMNS]
    missing = numeric.isna().sum()
    if int(missing.sum()) != 0 or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("combined ERA5 fields contain missing or non-finite values")
    land = frame["land_sea_mask"].to_numpy(dtype=float)
    if land.min() < 0.0 or land.max() > 1.0:
        raise ValueError("combined ERA5 land-sea mask lies outside [0, 1]")

    return {
        "passed": True,
        "rows": len(frame),
        "columns": len(frame.columns),
        "chunk_count": int(frame["chunk_id"].nunique()),
        "block_count": int(frame["block"].nunique()),
        "unique_hour_count": int(utc.nunique()),
        "unique_grid_point_count": len(observed_grid),
        "duplicate_key_count": 0,
        "missing_values_by_variable": {name: int(missing[name]) for name in VARIABLE_COLUMNS},
        "time_start_utc": utc.min().isoformat(),
        "time_end_utc": utc.max().isoformat(),
        "time_start_kst": kst.min().isoformat(),
        "time_end_kst": kst.max().isoformat(),
        "utc_to_kst_wall_clock_offset_minutes": 540,
        "maximum_allowed_time_kst": cutoff_kst.isoformat(),
        "land_sea_mask_minimum": float(land.min()),
        "land_sea_mask_maximum": float(land.max()),
    }


def run_full_retrieval(
    *,
    repo_root: Path,
    request_plan_path: Path,
    catalog_path: Path,
    permission_path: Path,
    smoke_receipt_path: Path,
    raw_output_directory: Path,
    artifact_directory: Path,
    status_path: Path,
    execute_download: bool,
) -> dict[str, Any]:
    """Retrieve all 17 chunks, validate them, and publish only after all gates pass."""

    progress = FullProgress(status_path)
    progress.update(progress=2, phase="preflight", detail="고정 계획·권한·smoke를 검증 중")
    if not execute_download:
        raise RuntimeError("explicit execute_download authorization is absent")

    raw_final = raw_output_directory.resolve()
    artifact_final = artifact_directory.resolve()
    raw_stage = raw_final.with_name(raw_final.name + ".staging")
    artifact_stage = artifact_final.with_name(artifact_final.name + ".staging")
    for path in (raw_final, raw_stage, artifact_final, artifact_stage):
        if path.exists():
            raise FileExistsError(f"ERA5 full output collision: {path.name}")

    chunks, gate = _validate_registered_inputs(
        repo_root=repo_root,
        request_plan_path=request_plan_path,
        catalog_path=catalog_path,
        permission_path=permission_path,
        smoke_receipt_path=smoke_receipt_path,
    )
    progress.update(
        progress=5,
        phase="ready",
        detail="17개 exact-hour chunk 다운로드 준비 완료",
    )
    raw_stage.mkdir(parents=True)
    artifact_stage.mkdir(parents=True)

    frames: list[pd.DataFrame] = []
    chunk_receipts: list[dict[str, Any]] = []
    reference_units: dict[str, str] | None = None
    reference_semantics: dict[str, str] | None = None
    downloaded_bytes = 0
    for number, chunk in enumerate(chunks, start=1):
        progress.update(
            progress=5 + 75 * (number - 1) / len(chunks),
            phase="download_validate",
            detail=f"chunk {number}/{len(chunks)} 다운로드·검증 중",
            completed_chunks=number - 1,
            downloaded_bytes=downloaded_bytes,
        )
        downloaded = download_cds_chunk(
            chunk,
            raw_stage,
            execute_download=True,
        )
        if downloaded.data_format != "netcdf":
            raise ValueError("full ERA5 path requires validated NetCDF responses")
        report, frame = load_cds_chunk_frame(downloaded.path, expected_chunk=chunk)
        public = report.public_dict()
        units = public["units"]
        semantics = public["validation"]["semantics"]
        if reference_units is None:
            reference_units = dict(units)
            reference_semantics = dict(semantics)
        elif units != reference_units or semantics != reference_semantics:
            raise ValueError("ERA5 units or sign metadata changed across chunks")
        downloaded_bytes += downloaded.bytes
        chunk_receipts.append(
            {
                "chunk_id": chunk.chunk_id,
                "block": chunk.block,
                "request_sha256": _json_sha256(chunk.request("netcdf")),
                "file_name": downloaded.path.name,
                "file_sha256": sha256(downloaded.path),
                "file_bytes": downloaded.bytes,
                "time_start_utc": public["time_start_utc"],
                "time_end_utc": public["time_end_utc"],
                "hour_count": public["validation"]["time_count"],
                "row_count": public["validation"]["time_count"] * EXPECTED_GRID_POINT_COUNT,
                "member_count": public["member_count"],
                "finite_value_counts": public["finite_value_counts"],
            }
        )
        frames.append(frame)

    progress.update(
        progress=82,
        phase="combine_validate",
        detail="4,900시간 3x3 자료를 결합·검증 중",
        completed_chunks=len(chunks),
        downloaded_bytes=downloaded_bytes,
    )
    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["time_utc", "latitude", "longitude"], kind="stable")
        .reset_index(drop=True)
    )
    validation = _validate_combined_frame(
        combined,
        chunks=chunks,
        cutoff_kst=P2_MAXIMUM_TIME_KST,
    )

    output_name = "era5_sors_3x3_p2_oof_exact17.parquet"
    parquet_stage = raw_stage / output_name
    combined.to_parquet(parquet_stage, index=False, compression="zstd")
    output_sha256 = sha256(parquet_stage)
    roundtrip = pd.read_parquet(parquet_stage)
    roundtrip_validation = _validate_combined_frame(
        roundtrip,
        chunks=chunks,
        cutoff_kst=P2_MAXIMUM_TIME_KST,
    )
    if roundtrip_validation != validation or sha256(parquet_stage) != output_sha256:
        raise ValueError("ERA5 Parquet roundtrip or output hash validation failed")

    progress.update(
        progress=94,
        phase="manifest",
        detail="aggregate-only provenance manifest를 작성 중",
        completed_chunks=len(chunks),
        downloaded_bytes=downloaded_bytes,
    )
    if reference_units is None or reference_semantics is None:
        raise AssertionError("ERA5 chunk validation produced no units or semantics")
    code_paths = {
        "runner": repo_root / "scripts/run_p2_era5_primary_scaffold.py",
        "full_retrieval": Path(__file__).resolve(),
        "request_builder": repo_root / "src/p2_restore/era5_request.py",
        "downloader": repo_root / "src/p2_restore/era5_downloader.py",
        "archive_validator": repo_root / "src/p2_restore/era5_cds.py",
        "field_validator": repo_root / "src/p2_restore/era5_preflight.py",
        "manifest": repo_root / "src/p2_restore/era5_manifest.py",
    }
    manifest = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "research_only": True,
        "upload_allowed": False,
        "source": source_provenance(),
        "policy_and_plan_gate": gate,
        "scope": {
            "problem": "P2",
            "purpose": "independent meteorology-only covariate ablation",
            "feature_variables": list(ERA5_VARIABLES),
            "validation_ancillary": list(ANCILLARY_VARIABLES),
            "hidden_target_temperature_or_salinity_used": False,
            "competition_source_data_read": False,
            "frozen_oof_read": False,
            "model_or_submission_modified": False,
        },
        "retrieval": {
            "chunk_count": len(chunk_receipts),
            "raw_total_bytes": downloaded_bytes,
            "chunks": chunk_receipts,
        },
        "output": {
            "file": _inside_repo(raw_final / output_name, repo_root),
            "sha256": output_sha256,
            "bytes": parquet_stage.stat().st_size,
            "format": "Parquet zstd",
            "grain": "one UTC hour by one point in the fixed S-ORS 3x3 ERA5 grid",
        },
        "validation": validation,
        "units": reference_units,
        "sign_semantics": reference_semantics,
        "runtime": _versions(),
        "code_sha256": {name: sha256(path) for name, path in code_paths.items()},
    }
    secret_value = os.environ.get("CDSAPI_KEY", "")
    assert_secret_free(manifest, secret_values=(secret_value,))
    write_receipt(artifact_stage / "manifest.json", manifest)

    raw_stage.replace(raw_final)
    artifact_stage.replace(artifact_final)
    manifest_path = artifact_final / "manifest.json"
    progress.update(
        progress=100,
        phase="complete",
        detail="17개 chunk와 aggregate validation 완료; 모델에는 아직 미사용",
        status="complete",
        completed_chunks=len(chunks),
        downloaded_bytes=downloaded_bytes,
    )
    return {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "chunk_count": len(chunks),
        "unique_hour_count": EXPECTED_HOUR_COUNT,
        "row_count": EXPECTED_ROW_COUNT,
        "raw_total_bytes": downloaded_bytes,
        "processed_bytes": (raw_final / output_name).stat().st_size,
        "processed_sha256": output_sha256,
        "manifest": _inside_repo(manifest_path, repo_root),
        "manifest_sha256": sha256(manifest_path),
        "hidden_target_values_used": False,
        "frozen_oof_read": False,
        "model_or_submission_modified": False,
    }
