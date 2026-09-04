"""Logging-only recovery wrapper for P1 v11.

The original v11 completed all 18 fits before JSON rejected an infinity sentinel.
This recovery changes only the artifact namespace and uses the corrected native
JSON conversion; the v11 config, candidates, gates, and fit schedule are unchanged.
"""

from __future__ import annotations

import run_p1_public_transport_repair_cycle_20260831_v11 as cycle

cycle.EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v11r1"
cycle.ARTIFACT = cycle.ROOT / "artifacts" / cycle.EXPERIMENT_ID
cycle.REPORT = cycle.ROOT / "reports" / cycle.EXPERIMENT_ID


if __name__ == "__main__":
    raise SystemExit(cycle.main())
