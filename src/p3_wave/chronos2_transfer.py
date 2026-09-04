"""Train-only helpers for the preregistered P3 Chronos-2 research probe."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import LEADS

CONTEXT_STEPS_20MIN = 145
PREDICTION_STEPS_20MIN = 72
LEAD_INDICES = np.asarray([8, 17, 26, 35, 53, 71], dtype=np.int64)
RAW_INDEX = {
    "hs": 0,
    "tp": 1,
    "hmax": 2,
    "wvdir": 3,
    "wspd": 4,
    "gust": 5,
    "wdir": 6,
    "airt": 7,
    "relh": 8,
    "caph": 9,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(destination)


def _history_fill(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    index = np.arange(values.size)
    return np.interp(index, index[finite], values[finite]).astype(np.float32)


def _past_covariates(sequence_20min: np.ndarray) -> dict[str, np.ndarray]:
    def column(name: str) -> np.ndarray:
        return _history_fill(sequence_20min[:, RAW_INDEX[name]])

    wave_direction = np.deg2rad(column("wvdir"))
    wind_direction = np.deg2rad(column("wdir"))
    return {
        "tp": column("tp"),
        "hmax": column("hmax"),
        "wvdir_sin": np.sin(wave_direction).astype(np.float32),
        "wvdir_cos": np.cos(wave_direction).astype(np.float32),
        "wspd": column("wspd"),
        "gust": column("gust"),
        "wdir_sin": np.sin(wind_direction).astype(np.float32),
        "wdir_cos": np.cos(wind_direction).astype(np.float32),
        "airt": column("airt"),
        "relh": column("relh"),
        "caph": column("caph"),
    }


def prepare_context_inputs(
    raw_values: np.ndarray, anchor_ids: Sequence[int]
) -> list[dict[str, object]]:
    """Convert cached 10-minute train contexts to 20-minute Chronos inputs."""

    output: list[dict[str, object]] = []
    for anchor_id in np.asarray(anchor_ids, dtype=np.int64):
        sequence = np.asarray(raw_values[int(anchor_id), ::2, :], dtype=np.float32)
        if sequence.shape != (CONTEXT_STEPS_20MIN, len(RAW_INDEX)):
            raise ValueError(f"unexpected train context shape: {sequence.shape}")
        target = _history_fill(sequence[:, RAW_INDEX["hs"]])
        if not np.isfinite(target[-1]):
            raise ValueError("anchor hs is not finite")
        output.append({"target": target, "past_covariates": _past_covariates(sequence)})
    return output


def prepare_training_episodes(
    raw_values: np.ndarray,
    anchors: pd.DataFrame,
    anchor_ids: Sequence[int],
    train_wave_path: str | Path,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Build exact 48h+24h train-only episodes without opening any test file."""

    required = {"station", "time", "hs"}
    wave = pd.read_csv(train_wave_path, usecols=sorted(required))
    if set(wave.columns) != required:
        raise ValueError("train_wave schema mismatch")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    lookups = {
        str(station): group.set_index("time")["hs"].sort_index()
        for station, group in wave.groupby("station", observed=True)
    }
    anchor_lookup = anchors.set_index("anchor_id")
    episodes: list[dict[str, object]] = []
    kept: list[int] = []
    for anchor_id in np.asarray(anchor_ids, dtype=np.int64):
        row = anchor_lookup.loc[int(anchor_id)]
        anchor_time = pd.Timestamp(row["anchor_time"])
        future_index = pd.date_range(
            start=anchor_time + pd.Timedelta(minutes=20),
            periods=PREDICTION_STEPS_20MIN,
            freq="20min",
        )
        future = lookups[str(row["station"])].reindex(future_index).to_numpy(dtype=np.float32)
        if future.shape != (PREDICTION_STEPS_20MIN,) or not np.isfinite(future).all():
            continue
        sequence = np.asarray(raw_values[int(anchor_id), ::2, :], dtype=np.float32)
        target = np.concatenate([_history_fill(sequence[:, RAW_INDEX["hs"]]), future])
        covariates = {
            name: np.concatenate(
                [values, np.full(PREDICTION_STEPS_20MIN, np.nan, dtype=np.float32)]
            )
            for name, values in _past_covariates(sequence).items()
        }
        episodes.append({"target": target, "past_covariates": covariates})
        kept.append(int(anchor_id))
    return episodes, np.asarray(kept, dtype=np.int64)


def point_predictions(
    pipeline: Any,
    inputs: list[dict[str, object]],
    *,
    batch_size: int,
) -> np.ndarray:
    forecasts = pipeline.predict(
        inputs,
        prediction_length=PREDICTION_STEPS_20MIN,
        context_length=CONTEXT_STEPS_20MIN,
        batch_size=batch_size,
    )
    quantiles = np.asarray(pipeline.quantiles, dtype=float)
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))
    if abs(float(quantiles[median_index]) - 0.5) > 1e-8:
        raise ValueError("Chronos-2 checkpoint does not expose a median quantile")
    rows: list[np.ndarray] = []
    for forecast in forecasts:
        values = forecast.detach().float().cpu().numpy()
        if values.shape[0] != 1 or values.shape[2] != PREDICTION_STEPS_20MIN:
            raise ValueError(f"unexpected Chronos prediction shape: {values.shape}")
        rows.append(values[0, median_index, LEAD_INDICES])
    result = np.clip(np.stack(rows).astype(float), 0.0, 30.0)
    if result.shape != (len(inputs), len(LEADS)) or not np.isfinite(result).all():
        raise ValueError("invalid Chronos point predictions")
    return result


def prediction_frame(
    anchors: pd.DataFrame,
    anchor_ids: Sequence[int],
    predictions: np.ndarray,
    *,
    fold: str,
) -> pd.DataFrame:
    lookup = anchors.set_index("anchor_id")
    records: list[dict[str, object]] = []
    for row_number, anchor_id in enumerate(np.asarray(anchor_ids, dtype=np.int64)):
        row = lookup.loc[int(anchor_id)]
        for lead_number, lead in enumerate(LEADS):
            records.append(
                {
                    "fold": fold,
                    "anchor_id": int(anchor_id),
                    "station": str(row["station"]),
                    "lead_h": int(lead),
                    "current_hs": float(row["current_hs"]),
                    "target_hs": float(row[f"target_{lead}"]),
                    "prediction": float(predictions[row_number, lead_number]),
                }
            )
    return pd.DataFrame.from_records(records)


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(truth)))))


def score_frame(frame: pd.DataFrame) -> dict[str, object]:
    truth = frame["target_hs"].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    persistence = frame["current_hs"].to_numpy(dtype=float)
    return {
        "rows": int(len(frame)),
        "cases": int(frame["anchor_id"].nunique()),
        "rmse_m": rmse(truth, prediction),
        "persistence_rmse_m": rmse(truth, persistence),
        "delta_vs_persistence_m": rmse(truth, prediction) - rmse(truth, persistence),
        "by_fold_rmse_m": {
            str(name): rmse(group["target_hs"], group["prediction"])
            for name, group in frame.groupby("fold", observed=True)
        },
        "by_lead_rmse_m": {
            str(int(name)): rmse(group["target_hs"], group["prediction"])
            for name, group in frame.groupby("lead_h", observed=True)
        },
        "by_station_rmse_m": {
            str(name): rmse(group["target_hs"], group["prediction"])
            for name, group in frame.groupby("station", observed=True)
        },
    }
