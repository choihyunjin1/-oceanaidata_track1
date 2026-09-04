"""Prepare quarantined pre-2024 KMA buoy observations for P3.

The credential is accepted only through the KMA_API_KEY process environment
variable.  It is never accepted as a command-line argument, printed, or written
to a receipt.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from p3_wave.kma_external import (
    DEFAULT_STATION_EPOCHS,
    LAST_NATIVE_OBSERVATION,
    SOURCE_FLOOR,
    KMAExternalError,
    prepare_kma_external,
)


def _aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime must include a timezone offset")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("status", "smoke", "full"),
        default="status",
        help="status never uses the network; smoke requests six hours; full requests monthly chunks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("external_data/p3_kma_buoy_pre2024"),
        help="ignored quarantine directory, relative to the repository unless absolute",
    )
    parser.add_argument(
        "--start",
        type=_aware_datetime,
        default=SOURCE_FLOOR,
        help="inclusive ISO-8601 start on a KST-equivalent 00/30-minute boundary",
    )
    parser.add_argument(
        "--end",
        type=_aware_datetime,
        default=LAST_NATIVE_OBSERVATION,
        help="inclusive ISO-8601 end, never later than 2023-12-31 23:30 KST",
    )
    parser.add_argument(
        "--station-id",
        action="append",
        type=int,
        choices=tuple(DEFAULT_STATION_EPOCHS),
        dest="station_ids",
        help="stable KMA station ID; repeat to override the four-station default",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    try:
        result = prepare_kma_external(
            repo_root=repo_root,
            output_dir=output_dir,
            mode=args.mode,
            station_ids=args.station_ids,
            start=args.start,
            end=args.end,
        )
    except (KMAExternalError, PermissionError, ValueError) as exc:
        failure = {
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "model_trained": False,
            "submission_written": False,
        }
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
