"""CLI for the append-only P2 Layer-4 checkpoint-policy experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from p2_restore.joint_hydrographic_multitask_layer4_checkpoint_v1 import (  # noqa: E402
    CONFIG_RELATIVE,
    execute,
    preflight,
)


def _progress(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(CONFIG_RELATIVE))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the one-shot GPU experiment; default is read-only preflight",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve(strict=True)
    config = args.config if args.config.is_absolute() else root / args.config
    if args.execute:
        result = execute(
            root=root,
            data_dir=args.data_dir,
            config_path=config,
            progress=_progress,
        )
    else:
        result = preflight(root=root, data_dir=args.data_dir, config_path=config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
