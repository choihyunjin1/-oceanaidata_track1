from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_consumed_namespace_has_lifecycle_safe_independent_qa() -> None:
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/qa_p1_v6_metric_consistency_preflight_20260901_v1.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert '"verdict": "PASS"' in process.stdout
