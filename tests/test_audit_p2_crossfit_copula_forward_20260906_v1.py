"""No source values: distinguish missing grid from missing target support."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_p2_crossfit_copula_forward_20260906_v1 as audit  # noqa: E402


def test_empty_tail_reason_does_not_invent_source_end():
    records = []
    for stamp in (
        "2024-12-10T00:00:00+09:00",
        "2025-12-10T00:00:00+09:00",
        "2025-12-20T00:00:00+09:00",
    ):
        for layer in (1, 2, 3, 4, 5, 6):
            records.append(
                {
                    "station": "S-ORS",
                    "time": stamp,
                    "layer": layer,
                    "temp": np.nan if "12-20" in stamp and layer in (2, 3, 4) else 10.0,
                }
            )
    _, contract, _ = audit.m.settings()
    result = audit.summarize(pd.DataFrame(records), contract)
    assert result[0]["reason"] == "source_grid_ends_before_fixed_outage"
    assert result[1]["reason"] == "source_grid_exists_but_finite_supported_target_labels_absent"
    assert all(row["status"] == "NOT_ESTIMABLE_NO_ROWS" and not row["date_moved"] for row in result)
