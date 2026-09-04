"""CLI for the clean train-only P3 fractional-change residual cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from p3_wave.clean_fractional_change_residual_20260901_c1 import (
    canonical_json_bytes,
    environment_paths,
    execute,
    preflight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        parser.error("choose exactly one of --preflight or --execute")
    return args


def main() -> None:
    args = parse_args()
    env_root, env_data = environment_paths()
    root = args.root or env_root
    data_dir = args.data_dir or env_data
    if args.preflight:
        payload = preflight(root=root, data_dir=data_dir)
        print(canonical_json_bytes(payload).decode("utf-8"), end="")
    else:
        payload = execute(root=root, data_dir=data_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
