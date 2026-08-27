"""Seal or execute the P1 improvement-cycle recovery."""

from __future__ import annotations

import argparse
import json

from p1_qc.improvement_cycle_resume import resume_cycle, seal_resume


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/p1_full_improvement_cycle_v1.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seal", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = seal_resume(args.config) if args.seal else resume_cycle(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
