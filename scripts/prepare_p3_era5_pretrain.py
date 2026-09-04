"""Plan or explicitly execute quarantined pre-2024 ERA5 retrieval for P3."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from p3_wave.era5_pretrain_data import (
    DownloadAuthorizationError,
    FileReceipt,
    QuarantineLayout,
    SelectedCell,
    build_manifest,
    build_smoke_plan,
    build_year_plan,
    combine_derived_year_files,
    file_receipt,
    load_validated_derived_year_file,
    process_year_file,
    read_selected_cells,
    retrieve_cds_request,
    select_cell_from_smoke_file,
    validate_existing_canonical_manifest,
    write_manifest,
    write_selected_cells,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
YEAR_DOWNLOAD_WORKERS = 4


def _existing_or_download(
    request: Any,
    *,
    layout: QuarantineLayout,
    execute_download: bool,
    client_factory: Callable[[], Any] | None,
) -> tuple[FileReceipt, bool]:
    target = layout.raw_path(request)
    if target.is_file():
        return (
            file_receipt(
                target,
                request=request,
                role="raw_cds_netcdf",
                layout=layout,
            ),
            False,
        )
    if not execute_download:
        raise DownloadAuthorizationError(
            f"{request.request_id} is absent; pass --execute-download to contact CDS"
        )
    return (
        retrieve_cds_request(
            request,
            target=target,
            layout=layout,
            execute_download=True,
            client_factory=client_factory,
        ),
        True,
    )


def _run_smoke(
    *,
    layout: QuarantineLayout,
    execute_download: bool,
    client_factory: Callable[[], Any] | None,
) -> tuple[dict[str, SelectedCell], list[FileReceipt], int]:
    selections: dict[str, SelectedCell] = {}
    receipts: list[FileReceipt] = []
    download_count = 0
    for request in build_smoke_plan():
        receipt, downloaded = _existing_or_download(
            request,
            layout=layout,
            execute_download=execute_download,
            client_factory=client_factory,
        )
        receipts.append(receipt)
        download_count += int(downloaded)
        selections[request.station] = select_cell_from_smoke_file(
            layout.raw_path(request),
            expected_request=request,
        )
    write_selected_cells(layout, selections)
    return selections, receipts, download_count


def _run_years(
    *,
    layout: QuarantineLayout,
    execute_download: bool,
    client_factory: Callable[[], Any] | None,
) -> tuple[dict[str, SelectedCell], list[FileReceipt], int]:
    selections = read_selected_cells(layout)
    requests = build_year_plan(selections)
    receipts: list[FileReceipt] = []
    download_count = 0
    raw_targets = [layout.raw_path(request) for request in requests]
    if len(set(raw_targets)) != len(raw_targets):
        raise AssertionError("ERA5 monthly plan contains duplicate raw targets")

    with ThreadPoolExecutor(max_workers=YEAR_DOWNLOAD_WORKERS) as executor:
        futures = [
            executor.submit(
                _existing_or_download,
                request,
                layout=layout,
                execute_download=execute_download,
                client_factory=client_factory,
            )
            for request in requests
        ]
        try:
            raw_results = [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    for request, (raw_receipt, downloaded) in zip(requests, raw_results, strict=True):
        download_count += int(downloaded)
        receipts.append(raw_receipt)
        output = layout.derived_year_path(request)
        if output.is_file():
            _, derived_receipt = load_validated_derived_year_file(
                output,
                request=request,
                selection=selections[request.station],
                layout=layout,
            )
            receipts.append(derived_receipt)
        else:
            receipts.append(
                process_year_file(
                    layout.raw_path(request),
                    request=request,
                    selection=selections[request.station],
                    output_path=output,
                    layout=layout,
                )
            )
    _, combined_receipt, _ = combine_derived_year_files(
        layout=layout,
        requests=requests,
        selections=selections,
    )
    receipts.append(combined_receipt)
    return selections, receipts, download_count


def run(
    *,
    stage: str,
    execute_download: bool,
    repo_root: str | Path = REPO_ROOT,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded stage; only smoke/years with authorization can contact CDS."""

    if stage not in {"plan", "smoke", "years", "combine"}:
        raise ValueError(f"unsupported ERA5 P3 stage: {stage}")
    layout = QuarantineLayout.from_repo_root(repo_root)
    layout.ensure()
    smoke_requests = build_smoke_plan()
    selections: dict[str, SelectedCell] | None = None
    receipts: list[FileReceipt] = []
    download_count = 0

    if stage == "smoke":
        selections, receipts, download_count = _run_smoke(
            layout=layout,
            execute_download=execute_download,
            client_factory=client_factory,
        )
    elif stage == "years":
        selections, receipts, download_count = _run_years(
            layout=layout,
            execute_download=execute_download,
            client_factory=client_factory,
        )
    elif stage == "combine":
        selections = read_selected_cells(layout)
        _, combined_receipt, _ = combine_derived_year_files(
            layout=layout,
            requests=build_year_plan(selections),
            selections=selections,
        )
        receipts.append(combined_receipt)

    year_requests = () if selections is None else build_year_plan(selections)
    manifest = build_manifest(
        stage=stage,
        smoke_requests=smoke_requests,
        year_requests=year_requests,
        selections=selections,
        files=receipts,
        network_action_taken=download_count > 0,
    )
    if stage in {"years", "combine"}:
        combined_receipt = next(
            value
            for value in receipts
            if value.role == "final_combined_selected_cell_hourly_parquet"
        )
        validate_existing_canonical_manifest(
            layout,
            combined_receipt=combined_receipt,
        )
    manifest_path = write_manifest(layout, manifest, stage=stage)
    return {
        "status": "planned" if stage == "plan" else "complete",
        "stage": stage,
        "network_action_taken": download_count > 0,
        "download_count": download_count,
        "smoke_request_count": len(smoke_requests),
        "year_request_count": len(year_requests),
        "file_receipt_count": len(receipts),
        "manifest": manifest_path.relative_to(Path(repo_root).resolve()).as_posix(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("plan", "smoke", "years", "combine"),
        default="plan",
        help="plan is network-free; smoke and years reuse existing files or require authorization",
    )
    parser.add_argument(
        "--execute-download",
        action="store_true",
        help="explicitly authorize CDS network retrieval for missing raw files",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run(stage=args.stage, execute_download=args.execute_download)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
