from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_oof_failures import (
    build_event_context,
    confusion_by_dimension,
    duration_bucket,
    main,
    oracle_diagnostics,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["A"] * 6 + ["B"] * 2,
            "year": [2025] * 8,
            "layer": [1] * 8,
            "time": [
                "2025-04-01T00:00:00+09:00",
                "2025-04-01T00:10:00+09:00",
                "2025-04-01T00:20:00+09:00",
                "2025-04-01T00:30:00+09:00",
                "2025-04-01T01:00:00+09:00",
                "2025-04-01T01:10:00+09:00",
                "2025-04-01T00:00:00+09:00",
                "2025-04-01T00:10:00+09:00",
            ],
            "label": [0, 1, 1, 0, 1, 1, 1, 0],
            "anomaly_signature": [
                "normal",
                "offset",
                "offset+drift",
                "normal",
                "noise",
                "noise",
                "spike",
                "normal",
            ],
            "station_layer": ["A/L1"] * 6 + ["B/L1"] * 2,
        }
    )


def test_duration_bucket_respects_official_landmarks() -> None:
    values = duration_bucket(np.array([1, 2, 17, 18, 47, 48, 143, 144, 287, 288]))
    assert list(values.astype(str)) == [
        "10m",
        "20m-<3h",
        "20m-<3h",
        "3h-<8h",
        "3h-<8h",
        "8h-<24h",
        "8h-<24h",
        "24h-<48h",
        "24h-<48h",
        ">=48h",
    ]


def test_event_context_never_crosses_time_gap_or_station_and_unions_composites() -> None:
    result = build_event_context(_frame())
    positive = result.loc[result["label"].eq(1)]
    assert positive["true_event_id"].nunique() == 3
    first_event = positive.loc[positive["time"].isin(_frame()["time"].iloc[1:3])]
    assert set(first_event["true_event_signature"]) == {"offset+drift"}
    assert set(first_event["event_composite_state"]) == {"composite"}
    assert set(positive["true_event_duration_bucket"]) == {"10m", "20m-<3h"}


def test_confusion_dimension_is_additive() -> None:
    frame = _frame()
    prediction = np.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int8)
    records = confusion_by_dimension(frame, prediction, "station_layer")
    assert sum(record["tp"] for record in records) == 3
    assert sum(record["fp"] for record in records) == 1
    assert sum(record["fn"] for record in records) == 2
    assert sum(record["tn"] for record in records) == 2
    assert abs(sum(record["fp_share_of_model"] for record in records) - 1.0) < 1e-12


def test_oracle_is_label_aware_and_explicitly_marked_unimplementable() -> None:
    label = np.array([1, 1, 0, 0], dtype=np.int8)
    predictions = {
        "a": np.array([1, 0, 1, 0], dtype=np.int8),
        "b": np.array([0, 1, 0, 1], dtype=np.int8),
    }
    result = oracle_diagnostics(label, predictions)
    assert result["per_row_label_oracle_unimplementable"]["f1"] == 1.0
    assert "RESEARCH DIAGNOSTIC ONLY" in result["warning"]


def test_cli_requires_research_only_acknowledgement() -> None:
    assert main([]) == 2
