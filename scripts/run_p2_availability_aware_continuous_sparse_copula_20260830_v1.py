"""CLI for the sealed P2 availability-aware continuous sparse copula experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _directory in (ROOT, SRC):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from p2_restore.p2_availability_aware_continuous_sparse_copula_20260830_v1 import (  # noqa: E402
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
