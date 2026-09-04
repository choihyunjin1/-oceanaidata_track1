"""Raw 48-hour sequence extraction for P3 deep models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import P3Data, build_training_grid

RAW_COLUMNS = ("hs", "tp", "hmax", "wvdir", "wspd", "gust", "wdir", "airt", "relh", "caph")
CONTEXT_ROWS = 289


@dataclass(frozen=True)
class RawSequences:
    values: np.ndarray
    station_code: np.ndarray


def build_train_sequences(data: P3Data, anchors: pd.DataFrame) -> RawSequences:
    grid = build_training_grid(data)
    values = np.empty((len(anchors), CONTEXT_ROWS, len(RAW_COLUMNS)), dtype=np.float32)
    station_code = np.empty(len(anchors), dtype=np.int64)
    stations = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    for station, group in anchors.groupby("station", sort=False, observed=True):
        source = (
            grid.loc[grid["station"].eq(station)]
            .sort_values("time")[list(RAW_COLUMNS)]
            .to_numpy(dtype=np.float32)
        )
        for row in group.itertuples(index=False):
            stop = int(row.grid_position) + 1
            start = stop - CONTEXT_ROWS
            if start < 0:
                raise ValueError("anchor lacks a complete sequence")
            values[int(row.anchor_id)] = source[start:stop]
            station_code[int(row.anchor_id)] = stations[str(station)]
    return RawSequences(values, station_code)


def build_test_sequences(data: P3Data) -> RawSequences:
    cases = data.test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    values = np.empty((len(cases), CONTEXT_ROWS, len(RAW_COLUMNS)), dtype=np.float32)
    station_code = np.empty(len(cases), dtype=np.int64)
    stations = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    context_lookup = {
        str(case_id): group.sort_values("step_minute")
        for case_id, group in data.test_context.groupby("case_id", sort=False, observed=True)
    }
    for number, row in enumerate(cases.itertuples(index=False)):
        context = context_lookup[str(row.case_id)]
        if len(context) != CONTEXT_ROWS:
            raise ValueError("test sequence row count mismatch")
        values[number] = context[list(RAW_COLUMNS)].to_numpy(dtype=np.float32)
        station_code[number] = stations[str(row.station)]
    return RawSequences(values, station_code)
