"""Build the P2 ERA5 request plan and run only explicitly authorized smoke access."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from p2_restore.era5_arco import (
    CATALOG_SHA256,
    inspect_anonymous_arco,
    read_anonymous_arco_smoke,
)
from p2_restore.era5_downloader import Era5DownloadBlocked, download_cds_chunk
from p2_restore.era5_full import run_full_retrieval
from p2_restore.era5_manifest import (
    build_arco_metadata_receipt,
    build_arco_smoke_receipt,
    build_request_plan_receipt,
    sha256,
    write_receipt,
)
from p2_restore.era5_preflight import credential_preflight
from p2_restore.era5_request import (
    OOF_SHA256,
    build_oof_chunk_plan,
    build_smoke_chunk,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_era5_primary_scaffold_v1"
OOF_PATH = REPO_ROOT / "artifacts/p2_extrapolated_soft_gate_v2/oof.parquet"
CATALOG_PATH = REPO_ROOT / "configs/external_data/catalog.toml"
OUTPUT_DIR = REPO_ROOT / "artifacts/p2_era5_primary_scaffold_v1"
STATUS_PATH = REPO_ROOT / "artifacts/status/p2_era5_primary_preflight.json"
FULL_STATUS_PATH = REPO_ROOT / "artifacts/status/p2_era5_full_retrieval_v1.json"
ARCO_RAW_DIR = REPO_ROOT / "external_data/quarantine/era5_arco_sors_smoke"
CDS_RAW_DIR = REPO_ROOT / "external_data/quarantine/era5_cds_sors_smoke"
CDS_FULL_RAW_DIR = REPO_ROOT / "external_data/quarantine/era5_cds_sors_p2_oof_exact17_v1"
CDS_FULL_ARTIFACT_DIR = REPO_ROOT / "artifacts/p2_era5_full_retrieval_v1"
PERMISSION_PATH = REPO_ROOT / "configs/external_data/official_faq_permission.json"
CDS_SMOKE_RECEIPT_PATH = OUTPUT_DIR / "cds_smoke_validation_receipt.json"
MAX_FULL_TRANSFER_GIB = 5.0
MAX_FULL_TRANSFER_HOURS = 2.0
REFERENCE_THROUGHPUT_MIB_PER_SECOND = 6.25


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Progress:
    def __init__(
        self, path: Path, *, title: str = "P2 ERA5 primary credential-safe scaffold"
    ) -> None:
        self.path = path
        self.title = title
        self.started = time.perf_counter()

    def update(
        self,
        progress: float,
        phase: str,
        detail: str,
        *,
        status: str = "running",
        extra: dict[str, Any] | None = None,
    ) -> None:
        elapsed = time.perf_counter() - self.started
        bounded = min(max(float(progress), 0.1), 100.0)
        remaining = elapsed * (100.0 - bounded) / bounded if bounded < 100 else 0.0
        payload: dict[str, Any] = {
            "title": self.title,
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "progress": bounded,
            "phase": phase,
            "detail": detail,
            "elapsed_seconds": elapsed,
            "eta": (datetime.now().astimezone() + timedelta(seconds=max(remaining, 0.0))).strftime(
                "%Y-%m-%d %H:%M:%S KST"
            ),
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        if extra:
            payload.update(extra)
        _write_json(self.path, payload)


def _fixed_plan(progress: Progress) -> tuple[object, tuple[object, ...], Path]:
    progress.update(8, "frozen_inputs", "catalog와 frozen OOF SHA를 검사 중")
    if sha256(CATALOG_PATH) != CATALOG_SHA256:
        raise ValueError("external-data catalog SHA changed")
    if sha256(OOF_PATH) != OOF_SHA256:
        raise ValueError("frozen P2 incumbent OOF SHA changed")
    oof = pd.read_parquet(OOF_PATH)
    smoke = build_smoke_chunk()
    chunks = build_oof_chunk_plan(oof, pad_days=7)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT_DIR / "request_plan.json"
    plan = build_request_plan_receipt(smoke, chunks, frozen_oof_sha256=OOF_SHA256)
    write_receipt(plan_path, plan)
    return smoke, chunks, plan_path


def _versions() -> dict[str, str]:
    values: dict[str, str] = {}
    for name in ("python", "numpy", "pandas", "xarray", "zarr", "gcsfs", "dask"):
        if name == "python":
            import platform

            values[name] = platform.python_version()
            continue
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = "not-installed"
    return values


def _code_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "request_builder": REPO_ROOT / "src/p2_restore/era5_request.py",
        "preflight": REPO_ROOT / "src/p2_restore/era5_preflight.py",
        "arco": REPO_ROOT / "src/p2_restore/era5_arco.py",
        "manifest": REPO_ROOT / "src/p2_restore/era5_manifest.py",
        "downloader": REPO_ROOT / "src/p2_restore/era5_downloader.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def _anonymous_transfer_gate(metadata: object, chunks: tuple[object, ...]) -> dict[str, Any]:
    planned_hours = sum(len(chunk.timestamps_utc()) for chunk in chunks)
    estimated_full_bytes = int(metadata.compressed_one_hour_bytes) * planned_hours
    estimated_full_gib = estimated_full_bytes / 2**30
    estimated_full_hours = estimated_full_bytes / (
        REFERENCE_THROUGHPUT_MIB_PER_SECOND * 2**20 * 3600
    )
    return {
        "planned_unique_hours": planned_hours,
        "estimated_full_bytes": estimated_full_bytes,
        "estimated_full_gib": estimated_full_gib,
        "estimated_full_hours_at_50mbps": estimated_full_hours,
        "maximum_full_transfer_gib": MAX_FULL_TRANSFER_GIB,
        "maximum_full_transfer_hours": MAX_FULL_TRANSFER_HOURS,
        "passed": estimated_full_gib <= MAX_FULL_TRANSFER_GIB
        and estimated_full_hours <= MAX_FULL_TRANSFER_HOURS,
    }


def run(args: argparse.Namespace, progress: Progress) -> dict[str, Any]:
    if args.mode == "cds-full":
        return run_full_retrieval(
            repo_root=REPO_ROOT,
            request_plan_path=OUTPUT_DIR / "request_plan.json",
            catalog_path=CATALOG_PATH,
            permission_path=PERMISSION_PATH,
            smoke_receipt_path=CDS_SMOKE_RECEIPT_PATH,
            raw_output_directory=CDS_FULL_RAW_DIR,
            artifact_directory=CDS_FULL_ARTIFACT_DIR,
            status_path=progress.path,
            execute_download=args.execute_download,
        )
    smoke, chunks, plan_path = _fixed_plan(progress)
    cds = credential_preflight()
    base = {
        "experiment_id": EXPERIMENT_ID,
        "mode": args.mode,
        "request_plan": str(plan_path.relative_to(REPO_ROOT)),
        "planned_oof_chunks": len(chunks),
        "cds_preflight": cds.public_dict(),
        "required_setting_names": list(cds.required_setting_names),
        "optional_setting_names": list(cds.optional_setting_names),
        "network_action_taken": False,
    }
    if args.mode == "preflight":
        status = cds.status
        result = {**base, "status": status}
        _write_json(OUTPUT_DIR / "preflight.json", result)
        progress.update(
            100,
            "cds_preflight",
            "CDS 설정 대기; 익명 ARCO smoke는 별도 명시 실행 가능",
            status=status,
            extra={
                "required_setting_names": list(cds.required_setting_names),
                "optional_setting_names": list(cds.optional_setting_names),
            },
        )
        return result

    if args.mode == "cds-smoke":
        if cds.status == "awaiting_credential":
            result = {**base, "status": "awaiting_credential"}
            _write_json(OUTPUT_DIR / "preflight.json", result)
            progress.update(
                100,
                "cds_preflight",
                "CDS token/약관 설정이 없어 다운로드하지 않음",
                status="awaiting_credential",
                extra={"required_setting_names": list(cds.required_setting_names)},
            )
            return result
        progress.update(40, "cds_smoke", "명시 승인된 CDS 24시간 smoke를 실행 중")
        try:
            downloaded = download_cds_chunk(
                smoke,
                CDS_RAW_DIR,
                execute_download=args.execute_download,
            )
        except Era5DownloadBlocked as exc:
            result = {**base, "status": "blocked", "reason": str(exc)}
            _write_json(OUTPUT_DIR / "preflight.json", result)
            progress.update(100, "cds_smoke", str(exc), status="blocked")
            return result
        result = {
            **base,
            "status": "downloaded_unvalidated",
            "network_action_taken": True,
            "download": downloaded.public_dict(),
        }
        _write_json(OUTPUT_DIR / "cds_smoke.json", result)
        progress.update(
            100, "cds_smoke", "CDS smoke 다운로드 완료; 해석 검증 전", status="complete"
        )
        return result

    if not args.execute_anonymous_smoke:
        result = {**base, "status": "ready_no_download"}
        _write_json(OUTPUT_DIR / "preflight.json", result)
        progress.update(
            100,
            "arco_smoke",
            "익명 smoke 명시 실행 플래그가 없어 네트워크를 사용하지 않음",
            status="ready_no_download",
        )
        return result
    progress.update(25, "arco_metadata", "anonymous ARCO metadata·변수·chunk read 상한 검사 중")
    metadata = inspect_anonymous_arco()
    transfer_gate = _anonymous_transfer_gate(metadata, chunks)
    if not transfer_gate["passed"]:
        overall_status = (
            "awaiting_credential" if cds.status == "awaiting_credential" else cds.status
        )
        metadata_receipt_path = OUTPUT_DIR / "arco_metadata_receipt.json"
        write_receipt(
            metadata_receipt_path,
            build_arco_metadata_receipt(
                metadata,
                transfer_gate,
                dependency_versions=_versions(),
                code_sha256=_code_hashes(),
            ),
        )
        result = {
            **base,
            "status": overall_status,
            "anonymous_arco_status": "no_go_anonymous_arco_transfer",
            "anonymous_arco_metadata": metadata.public_dict(),
            "anonymous_arco_transfer_gate": transfer_gate,
            "anonymous_arco_metadata_receipt": str(metadata_receipt_path.relative_to(REPO_ROOT)),
            "network_action_taken": True,
            "data_array_read": False,
        }
        _write_json(OUTPUT_DIR / "preflight.json", result)
        progress.update(
            100,
            "arco_transfer_gate",
            "anonymous 경로는 비용 NO-GO; CDS token/약관 설정 대기",
            status=overall_status,
            extra={
                "estimated_full_gib": transfer_gate["estimated_full_gib"],
                "required_setting_names": list(cds.required_setting_names),
            },
        )
        return result
    progress.update(
        45,
        "arco_smoke",
        f"24시간 3x3 smoke 읽는 중 (예상 {metadata.estimated_smoke_gib:.3f} GiB)",
    )
    frame, validation = read_anonymous_arco_smoke(metadata)
    ARCO_RAW_DIR.mkdir(parents=True, exist_ok=True)
    frame_path = ARCO_RAW_DIR / "era5_arco_sors_3x3_20240901_24h.csv"
    if frame_path.exists():
        raise FileExistsError("anonymous ARCO smoke output already exists")
    frame.to_csv(frame_path, index=False, encoding="utf-8", lineterminator="\n")
    receipt = build_arco_smoke_receipt(
        frame_path,
        frame,
        metadata,
        validation,
        dependency_versions=_versions(),
        code_sha256=_code_hashes(),
    )
    receipt_path = OUTPUT_DIR / "arco_smoke_receipt.json"
    write_receipt(receipt_path, receipt)
    result = {
        **base,
        "status": "complete",
        "network_action_taken": True,
        "anonymous_arco_smoke": {
            "metadata": metadata.public_dict(),
            "validation": validation,
            "data_file": str(frame_path.relative_to(REPO_ROOT)),
            "receipt": str(receipt_path.relative_to(REPO_ROOT)),
        },
    }
    _write_json(OUTPUT_DIR / "preflight.json", result)
    progress.update(
        100,
        "complete",
        "anonymous ARCO 24시간 3x3 smoke 검증 완료; 모델에는 아직 사용하지 않음",
        status="complete",
        extra={
            "arco_smoke_receipt": str(receipt_path),
            "required_setting_names": list(cds.required_setting_names),
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "arco-smoke", "cds-smoke", "cds-full"),
        default="preflight",
    )
    parser.add_argument("--execute-anonymous-smoke", action="store_true")
    parser.add_argument("--execute-download", action="store_true")
    parser.add_argument("--status-file")
    return parser


def main() -> None:
    args = _parser().parse_args()
    default_status = FULL_STATUS_PATH if args.mode == "cds-full" else STATUS_PATH
    status_path = Path(args.status_file) if args.status_file else default_status
    if not status_path.is_absolute():
        status_path = REPO_ROOT / status_path
    title = (
        "P2 ERA5 exact-hour full retrieval"
        if args.mode == "cds-full"
        else "P2 ERA5 primary credential-safe scaffold"
    )
    progress = Progress(status_path, title=title)
    try:
        run(args, progress)
    except Exception as exc:
        progress.update(
            100,
            "failed",
            f"{type(exc).__name__}: {exc}",
            status="failed",
        )
        raise


if __name__ == "__main__":
    main()
