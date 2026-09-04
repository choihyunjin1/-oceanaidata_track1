"""NASA POWER meteorology features for leakage-safe P2 ablations.

Only public atmospheric covariates are parsed.  The module never reads hidden
target-layer temperature or salinity and all timestamps are normalized to UTC
before an as-of join to the distributed P2 keys.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.features import FeatureTable

POWER_PARAMETERS = (
    "WS10M",
    "WD10M",
    "U10M",
    "V10M",
    "PS",
    "T2M",
    "RH2M",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
)
ROLLING_HOURS = (6, 24, 72, 168)


@dataclass(frozen=True)
class PowerQuality:
    rows: int
    start_utc: str
    end_utc: str
    duplicate_timestamps: int
    missing_values: int
    non_hourly_gaps: int
    maximum_gap_hours: float


def _parameter_frame(path: Path) -> pd.DataFrame:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        parameters = payload["properties"]["parameter"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid NASA POWER JSON: {path.name}") from exc
    if set(parameters) != set(POWER_PARAMETERS):
        raise ValueError(f"unexpected NASA POWER parameters: {sorted(parameters)}")
    key_sets = [set(parameters[name]) for name in POWER_PARAMETERS]
    if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("NASA POWER parameter timestamp keys differ")
    keys = sorted(key_sets[0])
    time = pd.to_datetime(keys, format="%Y%m%d%H", utc=True, errors="raise")
    frame = pd.DataFrame(
        {
            name: pd.to_numeric([parameters[name][key] for key in keys], errors="coerce")
            for name in POWER_PARAMETERS
        },
        index=time,
    )
    return frame.replace(-999.0, np.nan)


def load_power_hourly(
    paths: Sequence[str | Path], *, cutoff: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    """Load one or more unmodified NASA POWER JSON responses at hourly grain."""

    files = tuple(Path(path) for path in paths)
    if not files:
        raise ValueError("at least one NASA POWER JSON path is required")
    frame = pd.concat([_parameter_frame(path) for path in files]).sort_index()
    if frame.index.duplicated().any():
        raise ValueError("NASA POWER files contain overlapping timestamps")
    if cutoff is not None:
        limit = pd.Timestamp(cutoff)
        limit = limit.tz_localize("UTC") if limit.tzinfo is None else limit.tz_convert("UTC")
        frame = frame.loc[frame.index <= limit]
    if frame.empty:
        raise ValueError("NASA POWER cutoff removed every row")
    frame.index.name = "time_utc"
    return frame


def summarize_power_quality(frame: pd.DataFrame) -> PowerQuality:
    required = set(POWER_PARAMETERS)
    if not isinstance(frame.index, pd.DatetimeIndex) or not required.issubset(frame.columns):
        raise ValueError("NASA POWER frame has an invalid schema")
    difference = frame.index.to_series().diff().dt.total_seconds().div(3600)
    finite_gap = difference[np.isfinite(difference)]
    return PowerQuality(
        rows=len(frame),
        start_utc=frame.index.min().isoformat(),
        end_utc=frame.index.max().isoformat(),
        duplicate_timestamps=int(frame.index.duplicated().sum()),
        missing_values=int(frame.loc[:, POWER_PARAMETERS].isna().sum().sum()),
        non_hourly_gaps=int((finite_gap != 1.0).sum()),
        maximum_gap_hours=float(finite_gap.max()) if len(finite_gap) else 0.0,
    )


def _hourly_feature_bank(hourly: pd.DataFrame) -> pd.DataFrame:
    quality = summarize_power_quality(hourly)
    if quality.duplicate_timestamps or quality.non_hourly_gaps:
        raise ValueError("NASA POWER hourly series is not contiguous and unique")
    direct = hourly.loc[:, POWER_PARAMETERS].rename(
        columns={name: f"ext_power_{name.lower()}" for name in POWER_PARAMETERS}
    )
    direct["ext_power_wind_cubic"] = direct["ext_power_ws10m"].clip(lower=0) ** 3
    derived: dict[str, pd.Series] = {}
    for hours in ROLLING_HOURS:
        minimum = max(2, hours // 2)
        derived[f"ext_power_ws_mean_{hours}h"] = (
            direct["ext_power_ws10m"].rolling(hours, min_periods=minimum).mean()
        )
        derived[f"ext_power_ws_max_{hours}h"] = (
            direct["ext_power_ws10m"].rolling(hours, min_periods=minimum).max()
        )
        derived[f"ext_power_wind_energy_{hours}h"] = (
            direct["ext_power_wind_cubic"].rolling(hours, min_periods=minimum).mean()
        )
        derived[f"ext_power_precip_sum_{hours}h"] = (
            direct["ext_power_prectotcorr"].rolling(hours, min_periods=minimum).sum()
        )
        derived[f"ext_power_solar_mean_{hours}h"] = (
            direct["ext_power_allsky_sfc_sw_dwn"].rolling(hours, min_periods=minimum).mean()
        )
        derived[f"ext_power_air_mean_{hours}h"] = (
            direct["ext_power_t2m"].rolling(hours, min_periods=minimum).mean()
        )
    for hours in (6, 24, 72):
        derived[f"ext_power_pressure_delta_{hours}h"] = direct["ext_power_ps"].diff(hours)
        derived[f"ext_power_air_delta_{hours}h"] = direct["ext_power_t2m"].diff(hours)
    return pd.concat([direct, pd.DataFrame(derived, index=direct.index)], axis=1)


def build_power_features(hourly: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Align causal hourly meteorology to P2 rows without changing row order."""

    if "time" not in keys:
        raise ValueError("P2 keys must contain time")
    if {"target", "residual"}.intersection(hourly.columns):
        raise AssertionError("target-like columns are forbidden in external meteorology")
    bank = _hourly_feature_bank(hourly).reset_index()
    left = pd.DataFrame(
        {
            "_row": np.arange(len(keys), dtype=np.int64),
            "time_utc": pd.to_datetime(keys["time"], utc=True, errors="raise"),
        }
    ).sort_values("time_utc")
    merged = pd.merge_asof(
        left,
        bank.sort_values("time_utc"),
        on="time_utc",
        direction="backward",
        tolerance=pd.Timedelta(minutes=70),
        allow_exact_matches=True,
    ).sort_values("_row")
    features = merged.drop(columns=["_row", "time_utc"]).reset_index(drop=True)
    if len(features) != len(keys):
        raise ValueError("NASA POWER alignment changed P2 row count")
    if "temp_1" in keys:
        features["ext_power_surface_air_delta"] = pd.to_numeric(
            keys["temp_1"], errors="coerce"
        ).to_numpy(float) - features["ext_power_t2m"].to_numpy(float)
    if {"temp_1", "temp_5"}.issubset(keys.columns):
        stratification = np.abs(
            pd.to_numeric(keys["temp_1"], errors="coerce").to_numpy(float)
            - pd.to_numeric(keys["temp_5"], errors="coerce").to_numpy(float)
        )
        features["ext_power_wind_stratification"] = (
            features["ext_power_wind_energy_24h"].to_numpy(float) * stratification
        )
    return features.astype(np.float32)


def append_power_features(table: FeatureTable, hourly: pd.DataFrame) -> FeatureTable:
    additions = build_power_features(hourly, table.frame)
    if set(additions).intersection(table.frame.columns):
        raise ValueError("NASA POWER feature names collide with the P2 table")
    frame = pd.concat([table.frame.reset_index(drop=True), additions], axis=1)
    columns = tuple((*table.feature_columns, *additions.columns))
    if len(columns) != len(set(columns)):
        raise ValueError("NASA POWER feature columns are not unique")
    return FeatureTable(frame, columns)


def finite_coverage(features: pd.DataFrame, columns: Iterable[str] | None = None) -> float:
    names = tuple(columns) if columns is not None else tuple(features.columns)
    if not names:
        return 0.0
    values = features.loc[:, names].to_numpy(float)
    return float(np.isfinite(values).mean())
