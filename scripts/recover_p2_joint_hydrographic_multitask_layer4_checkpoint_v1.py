"""CLI for append-only P2 checkpoint_v1 evaluation recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from p2_restore.joint_hydrographic_multitask_layer4_checkpoint_v1_recovery import (  # noqa: E402
    execute_recovery,
    preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="append evaluation/manifest/seal only; default is read-only recovery preflight",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve(strict=True)
    if args.execute:
        result = execute_recovery(root=root, data_dir=args.data_dir)
    else:
        result = preflight(root=root, data_dir=args.data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
