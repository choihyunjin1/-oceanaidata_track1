from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_restore.external_meteorology import (
    POWER_PARAMETERS,
    append_power_features,
    build_power_features,
    finite_coverage,
    load_power_hourly,
    summarize_power_quality,
)
from p2_restore.features import FeatureTable


def _fixture(path: Path, start: str = "2024010100", hours: int = 200) -> Path:
    timestamps = pd.date_range(
        pd.to_datetime(start, format="%Y%m%d%H", utc=True), periods=hours, freq="h"
    )
    values = {}
    for number, parameter in enumerate(POWER_PARAMETERS):
        values[parameter] = {
            time.strftime("%Y%m%d%H"): float(number + row / 100)
            for row, time in enumerate(timestamps)
        }
    path.write_text(
        json.dumps({"properties": {"parameter": values}}),
        encoding="utf-8",
    )
    return path


def test_load_quality_and_cutoff(tmp_path: Path) -> None:
    path = _fixture(tmp_path / "power.json")
    frame = load_power_hourly([path], cutoff="2024-01-05T23:00:00Z")
    quality = summarize_power_quality(frame)

    assert quality.rows == 120
    assert quality.missing_values == 0
    assert quality.non_hourly_gaps == 0
    assert quality.maximum_gap_hours == 1.0


def test_overlapping_files_are_rejected(tmp_path: Path) -> None:
    first = _fixture(tmp_path / "a.json")
    second = _fixture(tmp_path / "b.json")
    with pytest.raises(ValueError, match="overlapping"):
        load_power_hourly([first, second])


def test_build_features_preserves_row_order_and_uses_public_layers(tmp_path: Path) -> None:
    hourly = load_power_hourly([_fixture(tmp_path / "power.json")])
    keys = pd.DataFrame(
        {
            "time": [
                "2024-01-08T12:10:00+09:00",
                "2024-01-07T03:50:00+09:00",
            ],
            "temp_1": [12.0, 10.0],
            "temp_5": [8.0, 9.0],
        }
    )
    features = build_power_features(hourly, keys)

    assert len(features) == len(keys)
    assert "ext_power_wind_energy_24h" in features
    assert "ext_power_surface_air_delta" in features
    assert "ext_power_wind_stratification" in features
    assert finite_coverage(features) == 1.0
    assert features.iloc[0]["ext_power_ws10m"] != features.iloc[1]["ext_power_ws10m"]


def test_append_features_does_not_read_targets(tmp_path: Path) -> None:
    hourly = load_power_hourly([_fixture(tmp_path / "power.json")])
    frame = pd.DataFrame(
        {
            "time": ["2024-01-05T12:10:00+09:00"],
            "temp_1": [12.0],
            "temp_5": [8.0],
            "target": [99.0],
            "residual": [77.0],
        }
    )
    table = FeatureTable(frame, ("temp_1", "temp_5"))
    result = append_power_features(table, hourly)

    assert result.frame["target"].item() == 99.0
    assert not any(name in {"target", "residual"} for name in result.feature_columns)
    assert np.isfinite(result.frame.loc[:, result.feature_columns]).all().all()
