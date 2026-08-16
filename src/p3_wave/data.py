"""P3 input resolution, immutable loading, and structural auditing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEADS = (3, 6, 9, 12, 18, 24)
STATIONS = ("G-ORS", "I-ORS", "S-ORS")
REQUIRED_FILES = (
    "README.md",
    "score.py",
    "train_wave.csv",
    "train_atmos.csv",
    "test_context.parquet",
    "test_index.csv",
    "sample_submission.csv",
    "baseline_persistence.csv",
)


@dataclass(frozen=True)
class P3Data:
    wave: pd.DataFrame
    atmos: pd.DataFrame
    test_context: pd.DataFrame
    test_index: pd.DataFrame
    sample_submission: pd.DataFrame
    baseline: pd.DataFrame


def _is_data_dir(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)


def resolve_p3_data_dir(path: str | Path | None = None, *, search_root: Path | None = None) -> Path:
    """Resolve the unique immutable P3 source directory."""

    explicit = path or os.environ.get("P3_DATA_DIR")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not _is_data_dir(candidate):
            raise FileNotFoundError(f"P3_DATA_DIR is missing required files: {candidate}")
        return candidate

    root = (search_root or Path.cwd()).resolve()
    candidates = sorted(
        {p.parent.resolve() for p in root.rglob("test_context.parquet") if _is_data_dir(p.parent)}
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one P3 source directory below {root}; found {len(candidates)}"
        )
    return candidates[0]


def load_p3_data(path: str | Path | None = None) -> P3Data:
    root = resolve_p3_data_dir(path)
    wave = pd.read_csv(root / "train_wave.csv")
    atmos = pd.read_csv(root / "train_atmos.csv")
    context = pd.read_parquet(root / "test_context.parquet")
    test_index = pd.read_csv(root / "test_index.csv")
    sample = pd.read_csv(root / "sample_submission.csv")
    baseline = pd.read_csv(root / "baseline_persistence.csv")
    for frame in (wave, atmos):
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    return P3Data(wave, atmos, context, test_index, sample, baseline)


def audit_p3_data(data: P3Data) -> dict[str, Any]:
    """Fail closed on the public P3 structural contract and return aggregates only."""

    wave = data.wave
    atmos = data.atmos
    context = data.test_context
    keys = ["case_id", "station", "lead_h"]
    expected_wave = ["station", "time", "hs", "tp", "hmax", "wvdir"]
    expected_atmos = ["station", "time", "wspd", "gust", "wdir", "airt", "relh", "caph"]
    expected_context = [
        "case_id",
        "station",
        "step_minute",
        "hs",
        "tp",
        "hmax",
        "wvdir",
        "wspd",
        "gust",
        "wdir",
        "airt",
        "relh",
        "caph",
    ]
    if list(wave.columns) != expected_wave:
        raise ValueError(f"train_wave schema mismatch: {list(wave.columns)}")
    if list(atmos.columns) != expected_atmos:
        raise ValueError(f"train_atmos schema mismatch: {list(atmos.columns)}")
    if list(context.columns) != expected_context:
        raise ValueError(f"test_context schema mismatch: {list(context.columns)}")
    if list(data.test_index.columns) != keys:
        raise ValueError("test_index schema mismatch")
    if list(data.sample_submission.columns) != keys + ["hs_pred"]:
        raise ValueError("sample_submission schema mismatch")
    if list(data.baseline.columns) != keys + ["hs_pred"]:
        raise ValueError("baseline_persistence schema mismatch")

    if len(wave) != 118_152 or len(atmos) != 130_896:
        raise ValueError("training row count mismatch")
    if len(context) != 57_800 or len(data.test_index) != 1_200:
        raise ValueError("test row count mismatch")
    if set(wave["station"].unique()) != set(STATIONS):
        raise ValueError("train_wave station set mismatch")
    if set(context["station"].unique()) != set(STATIONS):
        raise ValueError("test_context station set mismatch")
    if wave.duplicated(["station", "time"]).any() or atmos.duplicated(["station", "time"]).any():
        raise ValueError("duplicate training key")
    if context.duplicated(["case_id", "step_minute"]).any():
        raise ValueError("duplicate context key")
    if data.test_index.duplicated(keys).any():
        raise ValueError("duplicate test key")

    expected_steps = np.arange(-2880, 1, 10)
    sizes = context.groupby("case_id", observed=True).size()
    if len(sizes) != 200 or not sizes.eq(289).all():
        raise ValueError("each test case must contain exactly 289 rows")
    for _, group in context.groupby("case_id", sort=False, observed=True):
        if not np.array_equal(group["step_minute"].to_numpy(), expected_steps):
            raise ValueError("test context step grid mismatch or wrong order")
        if group["station"].nunique() != 1:
            raise ValueError("a case spans multiple stations")

    lead_counts = data.test_index["lead_h"].value_counts().to_dict()
    if set(lead_counts) != set(LEADS) or any(lead_counts[h] != 200 for h in LEADS):
        raise ValueError("test lead distribution mismatch")
    per_case_leads = data.test_index.groupby("case_id", observed=True)["lead_h"].agg(tuple)
    if not per_case_leads.map(lambda value: tuple(value) == LEADS).all():
        raise ValueError("lead order within test case mismatch")
    if not data.test_index[keys].equals(data.sample_submission[keys]):
        raise ValueError("sample key/order mismatch")
    if not data.test_index[keys].equals(data.baseline[keys]):
        raise ValueError("baseline key/order mismatch")

    current = context.loc[context["step_minute"].eq(0), ["case_id", "station", "hs"]]
    if len(current) != 200 or current["hs"].isna().any() or current["hs"].lt(1.5).any():
        raise ValueError("test anchor hs contract mismatch")
    merged = data.test_index.merge(current, on=["case_id", "station"], validate="many_to_one")
    merged = merged.merge(data.baseline, on=keys, validate="one_to_one")
    persistence_error = np.max(np.abs(merged["hs"].to_numpy() - merged["hs_pred"].to_numpy()))
    if persistence_error != 0.0:
        raise ValueError("baseline is not exact persistence")

    wave_slot = context["step_minute"].mod(20).eq(0)
    if context.loc[~wave_slot, ["hs", "tp", "hmax", "wvdir"]].notna().any().any():
        raise ValueError("wave values found on structural 10-minute intermediate rows")

    cadence: dict[str, dict[str, float | int]] = {}
    for name, frame, expected in (("wave", wave, 20), ("atmos", atmos, 10)):
        cadence[name] = {}
        for station, group in frame.groupby("station", observed=True):
            delta = group.sort_values("time")["time"].diff().dt.total_seconds().div(60).dropna()
            if not delta.eq(expected).all():
                raise ValueError(f"{name} cadence mismatch at {station}")
            cadence[name][station] = int(len(group))

    numeric_context = [c for c in expected_context if c not in {"case_id", "station"}]
    if not all(np.issubdtype(context[c].dtype, np.number) for c in numeric_context):
        raise ValueError("non-numeric context measurement")

    return {
        "rows": {
            "train_wave": int(len(wave)),
            "train_atmos": int(len(atmos)),
            "test_context": int(len(context)),
            "test_index": int(len(data.test_index)),
        },
        "cases": int(context["case_id"].nunique()),
        "cases_by_station": {
            str(k): int(v)
            for k, v in context.groupby("station", observed=True)["case_id"].nunique().items()
        },
        "current_hs": {
            "min": float(current["hs"].min()),
            "median": float(current["hs"].median()),
            "max": float(current["hs"].max()),
        },
        "missing_fraction": {c: float(context[c].isna().mean()) for c in expected_context[3:]},
        "cadence_rows": cadence,
        "persistence_exact": True,
    }


def build_training_grid(data: P3Data) -> pd.DataFrame:
    """Align each station to the public 10-minute test-context grid."""

    pieces: list[pd.DataFrame] = []
    for station in STATIONS:
        wave = data.wave.loc[data.wave["station"].eq(station)].drop(columns="station")
        atmos = data.atmos.loc[data.atmos["station"].eq(station)].drop(columns="station")
        start = wave["time"].min()
        end = wave["time"].max() + pd.Timedelta(minutes=10)
        index = pd.date_range(start=start, end=end, freq="10min", tz="UTC")
        part = pd.DataFrame({"time": index})
        part = part.merge(wave, on="time", how="left", validate="one_to_one")
        part = part.merge(atmos, on="time", how="left", validate="one_to_one")
        part.insert(0, "station", station)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def build_anchor_table(grid: pd.DataFrame, *, dense_spacing_minutes: int = 60) -> pd.DataFrame:
    """Build eligible train anchors and six future targets without using test labels."""

    if dense_spacing_minutes % 20:
        raise ValueError("dense spacing must be a multiple of the 20-minute wave cadence")
    records: list[pd.DataFrame] = []
    for station, group in grid.groupby("station", sort=False, observed=True):
        group = group.sort_values("time").reset_index(drop=True)
        eligible = group["hs"].ge(1.5)
        targets: dict[int, pd.Series] = {}
        for lead in LEADS:
            target = group["hs"].shift(-(lead * 6))
            targets[lead] = target
            eligible &= target.notna()
        eligible &= group["time"].ge(group["time"].min() + pd.Timedelta(hours=48))
        wave_number = np.arange(len(group)) // 2
        stride = dense_spacing_minutes // 20
        dense = eligible & ((wave_number % stride) == 0)
        idx = np.flatnonzero(dense.to_numpy())
        frame = pd.DataFrame(
            {
                "station": station,
                "anchor_time": group.loc[idx, "time"].to_numpy(),
                "grid_position": idx,
                "current_hs": group.loc[idx, "hs"].to_numpy(),
            }
        )
        for lead in LEADS:
            frame[f"target_{lead}"] = targets[lead].iloc[idx].to_numpy()
        records.append(frame)
    anchors = pd.concat(records, ignore_index=True)
    anchors.insert(0, "anchor_id", np.arange(len(anchors), dtype=np.int64))
    return anchors


def select_independent_validation(
    dense_anchors: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    gap_hours: int = 78,
) -> np.ndarray:
    """Greedily reproduce the problem's first-eligible independent-case structure."""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    chosen: list[int] = []
    for _, group in dense_anchors.groupby("station", sort=True, observed=True):
        eligible = group.loc[
            group["anchor_time"].ge(start_ts) & group["anchor_time"].lt(end_ts)
        ].sort_values("anchor_time")
        next_time: pd.Timestamp | None = None
        for row in eligible.itertuples(index=False):
            timestamp = pd.Timestamp(row.anchor_time)
            if next_time is None or timestamp >= next_time:
                chosen.append(int(row.anchor_id))
                next_time = timestamp + pd.Timedelta(hours=gap_hours)
    return np.asarray(sorted(chosen), dtype=np.int64)
