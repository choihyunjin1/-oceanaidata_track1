"""Reproduce and retrain the frozen P1/P2/P3 deployment candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocean_reproduce import run_all

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-data-dir", type=Path, required=True)
    parser.add_argument("--p2-data-dir", type=Path, required=True)
    parser.add_argument("--p3-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("saved", "retrain", "both"), default="both")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_all(
            project_root=PROJECT_ROOT,
            p1_data_dir=args.p1_data_dir,
            p2_data_dir=args.p2_data_dir,
            p3_data_dir=args.p3_data_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
