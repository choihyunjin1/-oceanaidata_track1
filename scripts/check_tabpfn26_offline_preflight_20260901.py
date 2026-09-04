"""Print the non-mutating TabPFN-2.6 offline readiness report."""

from __future__ import annotations

import json
from pathlib import Path

from ocean_tabpfn26.offline import inspect_preflight

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    report = inspect_preflight(workspace=ROOT)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
