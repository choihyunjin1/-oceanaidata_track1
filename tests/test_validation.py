from __future__ import annotations

import numpy as np
import pandas as pd

from p1_qc.validation import normal_station_layer_day_fp, paired_block_bootstrap


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["S-ORS"] * 4,
            "layer": [1] * 4,
            # First two rows straddle a UTC date but belong to one KST day.
            "time": [
                "2025-01-01T23:50:00+09:00",
                "2025-01-02T00:00:00+09:00",
                "2025-01-02T23:50:00+09:00",
                "2025-01-03T00:00:00+09:00",
            ],
        }
    )


def test_paired_bootstrap_uses_kst_normal_day_blocks() -> None:
    result = paired_block_bootstrap(
        np.zeros(4, dtype=np.int8),
        np.asarray([1, 0, 0, 0], dtype=np.int8),
        np.zeros(4, dtype=np.int8),
        _metadata(),
        replicates=8,
        seed=7,
        normal_day_timezone="Asia/Seoul",
    )
    assert result["positive_event_blocks"] == 0
    assert result["normal_day_blocks"] == 3
    assert result["normal_day_timezone"] == "Asia/Seoul"


def test_paired_bootstrap_preserves_historical_utc_default() -> None:
    result = paired_block_bootstrap(
        np.zeros(4, dtype=np.int8),
        np.asarray([1, 0, 0, 0], dtype=np.int8),
        np.zeros(4, dtype=np.int8),
        _metadata(),
        replicates=8,
        seed=7,
    )
    assert result["normal_day_blocks"] == 2
    assert result["normal_day_timezone"] == "UTC"


def test_normal_day_fp_uses_public_argument_order() -> None:
    result = normal_station_layer_day_fp(
        np.zeros(4, dtype=np.int8),
        np.asarray([1, 0, 0, 0], dtype=np.int8),
        np.zeros(4, dtype=np.int8),
        _metadata(),
    )
    assert result["normal_station_layer_days"] == 3
    assert result["candidate"]["false_positive_rows"] == 1
    assert result["baseline"]["false_positive_rows"] == 0
