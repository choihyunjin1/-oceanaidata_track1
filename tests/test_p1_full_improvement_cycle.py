from __future__ import annotations

import numpy as np
import pandas as pd

from p1_qc.improvement_cycle import binary_metrics, causal_event_rescue


def test_binary_metrics_reconciles() -> None:
    result = binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1
    assert result["f1"] == 0.5


def test_causal_event_rescue_adds_whole_event() -> None:
    frame = pd.DataFrame(
        {
            "station": ["A"] * 6,
            "layer": [1] * 6,
            "time": pd.date_range("2025-01-01", periods=6, freq="10min", tz="Asia/Seoul").astype(
                str
            ),
        }
    )
    result = causal_event_rescue(
        frame,
        np.array([0, 0, 1, 1, 0, 0]),
        np.array([0, 1, 1, 1, 1, 0]),
        np.array([0.0, 0.06, 0.8, 0.8, 0.01, 0.0]),
        np.array([0.0, 0.7, 0.9, 0.9, 0.7, 0.0]),
        causal_floor=0.65,
        incumbent_floor=0.05,
    )
    assert result.tolist() == [0, 1, 1, 1, 1, 0]


def test_causal_event_rescue_respects_physical_gap() -> None:
    frame = pd.DataFrame(
        {
            "station": ["A"] * 4,
            "layer": [1] * 4,
            "time": [
                "2025-01-01 00:00:00+09:00",
                "2025-01-01 00:10:00+09:00",
                "2025-01-01 02:00:00+09:00",
                "2025-01-01 02:10:00+09:00",
            ],
        }
    )
    result = causal_event_rescue(
        frame,
        np.zeros(4, dtype=np.int8),
        np.ones(4, dtype=np.int8),
        np.array([0.1, 0.1, 0.0, 0.0]),
        np.array([0.8, 0.8, 0.8, 0.8]),
        causal_floor=0.7,
        incumbent_floor=0.05,
    )
    assert result.tolist() == [1, 1, 0, 0]
