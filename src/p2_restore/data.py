"""Immutable P2 ingestion and contract checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_FILES = (
    "observations.csv",
    "test_index.csv",
    "sample_submission.csv",
    "baseline_interp.csv",
    "README.md",
    "score.py",
)
KEYS = ["station", "layer", "time"]


@dataclass(frozen=True)
class P2Data:
    observations: pd.DataFrame
    test_index: pd.DataFrame
    sample_submission: pd.DataFrame
    baseline: pd.DataFrame


def resolve_data_dir(path: str | Path | None = None) -> Path:
    raw = path or os.environ.get("P2_DATA_DIR")
    if not raw:
        raise FileNotFoundError("set P2_DATA_DIR or pass --data-dir")
    root = Path(raw).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P2 data directory is missing: {missing}")
    return root


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"station": "string", "time": "string"})


def load_p2_data(path: str | Path | None = None) -> P2Data:
    root = resolve_data_dir(path)
    data = P2Data(
        observations=_read(root / "observations.csv"),
        test_index=_read(root / "test_index.csv"),
        sample_submission=_read(root / "sample_submission.csv"),
        baseline=_read(root / "baseline_interp.csv"),
    )
    audit_p2_data(data)
    return data


def audit_p2_data(data: P2Data) -> dict[str, object]:
    obs, idx, sample, baseline = (
        data.observations,
        data.test_index,
        data.sample_submission,
        data.baseline,
    )
    if list(obs.columns) != [
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "nominal_depth",
    ]:
        raise ValueError("unexpected observations schema")
    if list(idx.columns) != KEYS + ["nominal_depth"]:
        raise ValueError("unexpected test_index schema")
    if list(sample.columns) != KEYS + ["temp"] or list(baseline.columns) != KEYS + ["temp"]:
        raise ValueError("unexpected submission/baseline schema")
    if len(idx) != 26_061 or len(sample) != len(idx) or len(baseline) != len(idx):
        raise ValueError("unexpected P2 row count")
    if obs.duplicated(["station", "year", "layer", "time"]).any() or idx.duplicated(KEYS).any():
        raise ValueError("duplicate P2 keys")
    if not idx[KEYS].equals(sample[KEYS]) or not idx[KEYS].equals(baseline[KEYS]):
        raise ValueError("test/sample/baseline key order differs")
    if idx[KEYS].isna().any().any() or not idx["time"].str.endswith("+09:00").all():
        raise ValueError("invalid P2 key or timezone")
    if not np.isfinite(baseline["temp"]).all():
        raise ValueError("baseline contains non-finite values")
    t = pd.to_datetime(obs["time"], utc=True).dt.tz_convert("Asia/Seoul")
    hidden = (
        t.ge(pd.Timestamp("2025-09-01", tz="Asia/Seoul"))
        & t.lt(pd.Timestamp("2025-11-01", tz="Asia/Seoul"))
        & obs["layer"].isin([2, 3, 4])
    )
    if int(hidden.sum()) != 26_352:
        raise ValueError("hidden grid does not contain 26,352 rows")
    if not obs.loc[hidden, ["temp", "psal"]].isna().all().all():
        raise ValueError("hidden target temp/psal are unexpectedly populated")
    return {
        "observation_rows": len(obs),
        "test_rows": len(idx),
        "hidden_grid_rows": int(hidden.sum()),
        "layer_counts": idx["layer"].value_counts().sort_index().to_dict(),
    }
