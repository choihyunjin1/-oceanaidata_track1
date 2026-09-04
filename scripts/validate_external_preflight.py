from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ocean_external.policy import audit_catalog, preflight_external_use


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit or preflight quarantined external data")
    parser.add_argument("--catalog", default="configs/external_data/catalog.toml")
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument(
        "--approval",
        help="permission receipt backed by an official organizer FAQ or direct answer",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--problem", choices=("P1", "P2", "P3"))
    parser.add_argument(
        "--purpose",
        choices=("pretraining", "feature_design", "normalization", "augmentation", "fine_tuning"),
    )
    parser.add_argument("--source-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.catalog_only:
        print(json.dumps(asdict(audit_catalog(args.catalog)), ensure_ascii=False, indent=2))
        return 0
    missing = [
        name
        for name in ("approval", "manifest", "problem", "purpose", "source_id")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            f"preflight requires: {', '.join('--' + x.replace('_', '-') for x in missing)}"
        )
    result = preflight_external_use(
        catalog_path=Path(args.catalog),
        approval_receipt_path=Path(args.approval),
        manifest_path=Path(args.manifest),
        problem=args.problem,
        source_id=args.source_id,
        purpose=args.purpose,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
