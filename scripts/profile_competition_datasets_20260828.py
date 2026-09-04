"""Aggregate-only profile of the three distributed competition datasets.

This script never reads hidden answers and never attempts to reconstruct P3
anonymous absolute timestamps. It emits only aggregate counts, rates and robust
distribution summaries needed to choose validation and model structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_quantiles(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    q = np.quantile(values, [0.1, 0.5, 0.9])
    return {"n": int(len(values)), "q10": float(q[0]), "median": float(q[1]), "q90": float(q[2])}


def missing_rates(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {column: float(frame[column].isna().mean()) for column in columns if column in frame}


def time_summary(series: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(series, utc=True, errors="raise")
    ordered = parsed.sort_values().drop_duplicates()
    deltas = ordered.diff().dropna().dt.total_seconds().div(60.0)
    return {
        "minimum": parsed.min().isoformat(),
        "maximum": parsed.max().isoformat(),
        "distinct": int(parsed.nunique()),
        "median_distinct_step_minutes": float(deltas.median()) if len(deltas) else None,
    }


def event_summary(train: pd.DataFrame) -> dict[str, Any]:
    keys = ["station", "year", "layer"]
    work = train.loc[train["label"].eq(1), keys + ["time", "anomaly_type"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True, errors="raise")
    work = work.sort_values(keys + ["time"], kind="mergesort")
    gap = work.groupby(keys, observed=True)["time"].diff().dt.total_seconds().div(60.0)
    type_change = work["anomaly_type"].ne(
        work.groupby(keys, observed=True)["anomaly_type"].shift()
    )
    work["event_start"] = gap.ne(10.0) | type_change
    work["event_id"] = work.groupby(keys, observed=True)["event_start"].cumsum()
    events = (
        work.groupby(keys + ["event_id", "anomaly_type"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    result: dict[str, Any] = {}
    for kind, block in events.groupby("anomaly_type", observed=True):
        hours = block["rows"].to_numpy(dtype=np.float64) / 6.0
        result[str(kind)] = {
            "events": int(len(block)),
            "rows": int(block["rows"].sum()),
            "duration_hours_median": float(np.median(hours)),
            "duration_hours_p90": float(np.quantile(hours, 0.9)),
            "duration_hours_max": float(hours.max()),
        }
    return result


def profile_p1(root: Path) -> dict[str, Any]:
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    key = ["station", "year", "layer", "time"]
    values = ["temp", "psal", "depth"]
    positive = train["label"].eq(1)
    return {
        "files": {
            "train_sha256": sha256_file(train_path),
            "test_sha256": sha256_file(test_path),
        },
        "grain": "station-year-layer-time row",
        "train": {
            "rows": int(len(train)),
            "columns": list(train.columns),
            "duplicate_keys": int(train.duplicated(key).sum()),
            "time": time_summary(train["time"]),
            "stations": train["station"].value_counts().sort_index().astype(int).to_dict(),
            "years": train["year"].value_counts().sort_index().astype(int).to_dict(),
            "layers": train["layer"].value_counts().sort_index().astype(int).to_dict(),
            "missing_rates": missing_rates(train, values),
            "positive_rows": int(positive.sum()),
            "positive_rate": float(positive.mean()),
            "positive_by_type": (
                train.loc[positive, "anomaly_type"].value_counts().sort_index().astype(int).to_dict()
            ),
            "positive_rate_by_station": (
                train.groupby("station", observed=True)["label"].mean().sort_index().to_dict()
            ),
            "positive_rate_by_layer": (
                train.groupby("layer", observed=True)["label"].mean().sort_index().to_dict()
            ),
            "events": event_summary(train),
        },
        "test": {
            "rows": int(len(test)),
            "columns": list(test.columns),
            "duplicate_keys": int(test.duplicated(key).sum()),
            "time": time_summary(test["time"]),
            "stations": test["station"].value_counts().sort_index().astype(int).to_dict(),
            "years": test["year"].value_counts().sort_index().astype(int).to_dict(),
            "layers": test["layer"].value_counts().sort_index().astype(int).to_dict(),
            "missing_rates": missing_rates(test, values),
            "depth_missing_by_station": (
                test.groupby("station", observed=True)["depth"].apply(lambda x: float(x.isna().mean())).sort_index().to_dict()
            ),
        },
        "support": {
            "train_station_layers": int(train[["station", "layer"]].drop_duplicates().shape[0]),
            "test_station_layers": int(test[["station", "layer"]].drop_duplicates().shape[0]),
            "test_only_station_layers": int(
                len(
                    test[["station", "layer"]]
                    .drop_duplicates()
                    .merge(train[["station", "layer"]].drop_duplicates(), how="left", indicator=True)
                    .query("_merge == 'left_only'")
                )
            ),
        },
    }


def profile_p2(root: Path) -> dict[str, Any]:
    observations_path = root / "observations.csv"
    index_path = root / "test_index.csv"
    observations = pd.read_csv(observations_path)
    index = pd.read_csv(index_path)
    key = ["station", "layer", "time"]
    observations["parsed_time"] = pd.to_datetime(observations["time"], utc=True, errors="raise")
    index["parsed_time"] = pd.to_datetime(index["time"], utc=True, errors="raise")
    target_times = index["parsed_time"].drop_duplicates()
    target_start = target_times.min()
    target_end = target_times.max()
    in_target = observations["parsed_time"].between(target_start, target_end)
    target_observations = observations.loc[in_target].copy()
    public_layers = target_observations[target_observations["layer"].isin([1, 5, 6, 7, 8])]
    public_temp_count = public_layers.groupby("parsed_time", observed=True)["temp"].count()

    seasonal = observations[
        observations["parsed_time"].dt.month.isin([9, 10])
        & observations["layer"].isin([1, 5, 6, 7, 8])
    ].copy()
    seasonal["year_utc"] = seasonal["parsed_time"].dt.year
    seasonal_summary: dict[str, Any] = {}
    for year, block in seasonal.groupby("year_utc", observed=True):
        seasonal_summary[str(int(year))] = {
            "rows": int(len(block)),
            "temp": finite_quantiles(block["temp"]),
            "psal": finite_quantiles(block["psal"]),
            "temp_missing_rate": float(block["temp"].isna().mean()),
            "psal_missing_rate": float(block["psal"].isna().mean()),
        }

    return {
        "files": {
            "observations_sha256": sha256_file(observations_path),
            "test_index_sha256": sha256_file(index_path),
        },
        "grain": "station-layer-time observation row",
        "observations": {
            "rows": int(len(observations)),
            "columns": [column for column in observations.columns if column != "parsed_time"],
            "duplicate_keys": int(observations.duplicated(key).sum()),
            "time": time_summary(observations["time"]),
            "stations": observations["station"].value_counts().sort_index().astype(int).to_dict(),
            "years": observations["year"].value_counts().sort_index().astype(int).to_dict(),
            "layers": observations["layer"].value_counts().sort_index().astype(int).to_dict(),
            "missing_rates": missing_rates(observations, ["temp", "psal", "depth"]),
        },
        "target_index": {
            "rows": int(len(index)),
            "columns": [column for column in index.columns if column != "parsed_time"],
            "duplicate_keys": int(index.duplicated(key).sum()),
            "distinct_times": int(index["time"].nunique()),
            "time_minimum": index["parsed_time"].min().isoformat(),
            "time_maximum": index["parsed_time"].max().isoformat(),
            "rows_by_layer": index["layer"].value_counts().sort_index().astype(int).to_dict(),
            "target_layers_temp_missing_rate_in_observations": (
                target_observations[target_observations["layer"].isin([2, 3, 4])]
                .groupby("layer", observed=True)["temp"]
                .apply(lambda x: float(x.isna().mean()))
                .sort_index()
                .to_dict()
            ),
            "target_layers_psal_missing_rate_in_observations": (
                target_observations[target_observations["layer"].isin([2, 3, 4])]
                .groupby("layer", observed=True)["psal"]
                .apply(lambda x: float(x.isna().mean()))
                .sort_index()
                .to_dict()
            ),
            "public_temperature_layers_available_per_time": {
                str(int(k)): int(v)
                for k, v in public_temp_count.value_counts().sort_index().items()
            },
        },
        "seasonal_public_layer_shift": seasonal_summary,
    }


def profile_p3(root: Path) -> dict[str, Any]:
    wave_path = root / "train_wave.csv"
    atmos_path = root / "train_atmos.csv"
    context_path = root / "test_context.parquet"
    index_path = root / "test_index.csv"
    wave = pd.read_csv(wave_path)
    atmos = pd.read_csv(atmos_path)
    context = pd.read_parquet(context_path)
    index = pd.read_csv(index_path)
    wave_key = ["station", "time"]
    atmos_key = ["station", "time"]
    context_key = ["case_id", "step_minute"]
    index_key = ["case_id", "station", "lead_h"]
    current = context.loc[context["step_minute"].eq(0)].copy()
    observed_context_wave = context.loc[context["hs"].notna()].copy()
    train_hs = pd.to_numeric(wave["hs"], errors="coerce")
    train_q90 = float(train_hs.quantile(0.9))
    context_wave_columns = ["hs", "tp", "hmax", "wvdir"]
    atmos_columns = ["wspd", "gust", "wdir", "airt", "relh", "caph"]

    by_station: dict[str, Any] = {}
    for station, block in current.groupby("station", observed=True):
        station_train = wave.loc[wave["station"].eq(station), "hs"]
        by_station[str(station)] = {
            "cases": int(len(block)),
            "current_hs": finite_quantiles(block["hs"]),
            "train_hs": finite_quantiles(station_train),
        }

    expected_steps = np.arange(-2880, 1, 10, dtype=np.int64)
    step_contract_pass = all(
        np.array_equal(np.sort(block["step_minute"].to_numpy(dtype=np.int64)), expected_steps)
        for _, block in context.groupby("case_id", sort=False, observed=True)
    )
    wave_slot = context["step_minute"].mod(20).eq(0)

    return {
        "files": {
            "train_wave_sha256": sha256_file(wave_path),
            "train_atmos_sha256": sha256_file(atmos_path),
            "test_context_sha256": sha256_file(context_path),
            "test_index_sha256": sha256_file(index_path),
        },
        "grain": {
            "train_wave": "station-time wave observation",
            "train_atmos": "station-time atmosphere observation",
            "test_context": "anonymous case-local relative-time row",
            "test_index": "case-station-lead prediction key",
        },
        "train_wave": {
            "rows": int(len(wave)),
            "columns": list(wave.columns),
            "duplicate_keys": int(wave.duplicated(wave_key).sum()),
            "time": time_summary(wave["time"]),
            "stations": wave["station"].value_counts().sort_index().astype(int).to_dict(),
            "missing_rates": missing_rates(wave, context_wave_columns),
            "hs": finite_quantiles(wave["hs"]),
        },
        "train_atmos": {
            "rows": int(len(atmos)),
            "columns": list(atmos.columns),
            "duplicate_keys": int(atmos.duplicated(atmos_key).sum()),
            "time": time_summary(atmos["time"]),
            "stations": atmos["station"].value_counts().sort_index().astype(int).to_dict(),
            "missing_rates": missing_rates(atmos, atmos_columns),
        },
        "test_context": {
            "rows": int(len(context)),
            "columns": list(context.columns),
            "duplicate_keys": int(context.duplicated(context_key).sum()),
            "cases": int(context["case_id"].nunique()),
            "rows_per_case_contract": bool(step_contract_pass),
            "station_cases": current["station"].value_counts().sort_index().astype(int).to_dict(),
            "wave_missing_rate_on_expected_20min_slots": missing_rates(
                context.loc[wave_slot], context_wave_columns
            ),
            "wave_missing_rate_on_intermediate_10min_slots": missing_rates(
                context.loc[~wave_slot], context_wave_columns
            ),
            "atmos_missing_rates": missing_rates(context, atmos_columns),
            "observed_wave_hs": finite_quantiles(observed_context_wave["hs"]),
            "current_hs": finite_quantiles(current["hs"]),
            "current_hs_at_or_above_1_5_fraction": float(current["hs"].ge(1.5).mean()),
            "current_hs_above_train_q90_fraction": float(current["hs"].gt(train_q90).mean()),
            "by_station": by_station,
        },
        "test_index": {
            "rows": int(len(index)),
            "duplicate_keys": int(index.duplicated(index_key).sum()),
            "cases": int(index["case_id"].nunique()),
            "leads": index["lead_h"].value_counts().sort_index().astype(int).to_dict(),
            "station_rows": index["station"].value_counts().sort_index().astype(int).to_dict(),
            "context_index_case_station_alignment": bool(
                current[["case_id", "station"]]
                .reset_index(drop=True)
                .equals(index[["case_id", "station"]].drop_duplicates().reset_index(drop=True))
            ),
        },
        "absolute_time_reconstruction_attempted": False,
        "hidden_truth_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-dir", type=Path, required=True)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--p3-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": "competition_dataset_profile.20260828.v1",
        "p1": profile_p1(args.p1_dir.resolve()),
        "p2": profile_p2(args.p2_dir.resolve()),
        "p3": profile_p3(args.p3_dir.resolve()),
        "hidden_answer_files_read": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
