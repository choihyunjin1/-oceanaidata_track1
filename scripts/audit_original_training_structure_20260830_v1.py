"""Aggregate-only re-audit of the immutable P1/P2/P3 training sources.

This audit deliberately opens only the distributed training tables and their
README files.  It does not enumerate or open official test, sample,
submission, baseline, score, or hidden-answer files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p3_wave.data import P3Data, build_anchor_table, build_training_grid

P3_LEADS_H = (3, 6, 9, 12, 18, 24)
P3_STATIONS = ("G-ORS", "I-ORS", "S-ORS")
P3_DENSE_SPACING_MINUTES = 60
P3_WAVE_CADENCE_MINUTES = 20
P2_TARGET_LAYERS = (2, 3, 4)
P2_PUBLIC_LAYERS = (1, 5, 6, 7, 8)


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(root: Path, name: str) -> Path:
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / name).resolve(strict=True)
    if path.parent != resolved_root or path.name != name or not path.is_file():
        raise AuditError(f"training source escaped immutable root: {name}")
    return path


def finite_summary(values: pd.Series | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "q10": None, "median": None, "q90": None}
    q10, median, q90 = np.quantile(array, [0.1, 0.5, 0.9])
    return {
        "count": int(len(array)),
        "q10": float(q10),
        "median": float(median),
        "q90": float(q90),
    }


def audit_p1(root: Path) -> dict[str, Any]:
    readme = require_file(root, "README.md")
    source = require_file(root, "train.csv")
    frame = pd.read_csv(source)
    expected = {
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "label",
        "anomaly_type",
    }
    if set(frame.columns) != expected or len(frame) != 776_706:
        raise AuditError("P1 training schema or row count changed")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    keys = ["station", "year", "layer"]
    frame = frame.sort_values(keys + ["time"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(keys + ["time"]).any():
        raise AuditError("P1 duplicate training keys")
    positive = frame["label"].eq(1)
    if not frame["label"].isin([0, 1]).all():
        raise AuditError("P1 labels are not binary")
    same_series = pd.Series(True, index=frame.index)
    for key in keys:
        same_series &= frame[key].eq(frame[key].shift())
    gap_min = frame.groupby(keys, observed=True)["time"].diff().dt.total_seconds().div(60)
    binary_start = positive & (
        ~positive.shift(fill_value=False) | ~same_series | gap_min.ne(10.0)
    )
    binary_id = binary_start.cumsum().where(positive)
    anomaly = frame["anomaly_type"].fillna("").astype(str)
    typed_start = binary_start | (positive & anomaly.ne(anomaly.shift()))
    typed_id = typed_start.cumsum().where(positive)

    binary_events = (
        frame.loc[positive]
        .assign(binary_event_id=binary_id.loc[positive].astype(np.int64))
        .groupby("binary_event_id", observed=True)
        .agg(
            rows=("label", "size"),
            type_count=("anomaly_type", lambda x: int(x.dropna().astype(str).nunique())),
        )
    )
    typed_events = (
        frame.loc[positive]
        .assign(typed_event_id=typed_id.loc[positive].astype(np.int64))
        .groupby(["typed_event_id", "anomaly_type"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    by_type: dict[str, Any] = {}
    for anomaly_type, part in typed_events.groupby("anomaly_type", observed=True):
        by_type[str(anomaly_type)] = {
            "events": int(len(part)),
            "rows": int(part["rows"].sum()),
            "duration_hours": finite_summary((part["rows"] - 1) / 6.0),
        }
    return {
        "source": {
            "filename": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "readme_sha256": sha256_file(readme),
        },
        "rows": int(len(frame)),
        "positive_rows": int(positive.sum()),
        "positive_rate": float(positive.mean()),
        "binary_contiguous_event_count": int(len(binary_events)),
        "typed_contiguous_event_count": int(len(typed_events)),
        "multi_type_binary_event_count": int(binary_events["type_count"].gt(1).sum()),
        "by_type": by_type,
        "event_definition_note": (
            "binary events split only on station/year/layer, 10-minute continuity, and label; "
            "typed events additionally split when anomaly_type changes"
        ),
    }


def _p2_timestamp_features(frame: pd.DataFrame) -> pd.DataFrame:
    public = frame.loc[frame["layer"].isin(P2_PUBLIC_LAYERS)].copy()
    grouped = public.groupby(["station", "time"], sort=True, observed=True)
    features = grouped.agg(
        public_temp_count=("temp", "count"),
        public_psal_count=("psal", "count"),
        public_temp_min=("temp", "min"),
        public_temp_max=("temp", "max"),
        public_temp_mean=("temp", "mean"),
        public_psal_min=("psal", "min"),
        public_psal_max=("psal", "max"),
        public_psal_mean=("psal", "mean"),
    ).reset_index()
    features["public_temp_range"] = features["public_temp_max"] - features["public_temp_min"]
    features["public_psal_range"] = features["public_psal_max"] - features["public_psal_min"]
    for hours in (6, 24, 72):
        periods = hours * 6
        features[f"temp_mean_change_{hours}h"] = features.groupby(
            "station", observed=True
        )["public_temp_mean"].diff(periods)
        features[f"temp_range_change_{hours}h"] = features.groupby(
            "station", observed=True
        )["public_temp_range"].diff(periods)
    return features


def audit_p2(root: Path) -> dict[str, Any]:
    readme = require_file(root, "README.md")
    source = require_file(root, "observations.csv")
    columns = ["station", "layer", "time", "temp", "psal", "depth", "nominal_depth"]
    frame = pd.read_csv(source, usecols=columns)
    if list(frame.columns) != columns or len(frame) != 789_408:
        raise AuditError("P2 training schema or row count changed")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if frame.duplicated(["station", "layer", "time"]).any():
        raise AuditError("P2 duplicate training keys")
    complete_temp_psal = (
        frame.assign(complete=frame[["temp", "psal"]].notna().all(axis=1))
        .groupby(["station", "time"], observed=True)["complete"]
        .all()
    )
    complete_with_depth = (
        frame.assign(complete=frame[["temp", "psal", "depth"]].notna().all(axis=1))
        .groupby(["station", "time"], observed=True)["complete"]
        .all()
    )
    feature = _p2_timestamp_features(frame)
    state_columns = [
        "public_temp_count",
        "public_psal_count",
        "public_temp_range",
        "public_psal_range",
        "temp_mean_change_6h",
        "temp_mean_change_24h",
        "temp_mean_change_72h",
        "temp_range_change_6h",
        "temp_range_change_24h",
        "temp_range_change_72h",
    ]
    finite_state = np.isfinite(feature[state_columns].to_numpy(dtype=np.float64)).all(axis=1)
    target = frame.loc[frame["layer"].isin(P2_TARGET_LAYERS)].copy()
    target_complete = (
        target.assign(complete=target[["temp", "psal", "depth"]].notna().all(axis=1))
        .groupby(["station", "time"], observed=True)["complete"]
        .all()
    )
    return {
        "source": {
            "filename": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "readme_sha256": sha256_file(readme),
        },
        "rows": int(len(frame)),
        "layers": sorted(int(value) for value in frame["layer"].unique()),
        "complete_temp_psal_all_layer_timestamps": int(complete_temp_psal.sum()),
        "complete_temp_psal_depth_all_layer_timestamps": int(
            complete_with_depth.sum()
        ),
        "complete_target_profile_timestamps": int(target_complete.sum()),
        "public_state_timestamp_count": int(len(feature)),
        "finite_public_state_72h_count": int(finite_state.sum()),
        "finite_public_state_72h_fraction": float(finite_state.mean()),
        "public_layer_availability": {
            "temperature_count": {
                str(int(key)): int(value)
                for key, value in feature["public_temp_count"].value_counts().sort_index().items()
            },
            "salinity_count": {
                str(int(key)): int(value)
                for key, value in feature["public_psal_count"].value_counts().sort_index().items()
            },
        },
        "state_features": {
            column: finite_summary(feature[column]) for column in state_columns
        },
        "candidate_support_note": (
            "public T/S profile state plus 6/24/72-hour changes can be formed without "
            "opening query, test, sample, baseline, score, or submission files"
        ),
    }


def _greedy_78h(frame: pd.DataFrame) -> pd.DataFrame:
    selected: list[int] = []
    for _, station in frame.groupby("station", sort=True, observed=True):
        next_allowed: pd.Timestamp | None = None
        for row in station.sort_values("time").itertuples():
            timestamp = pd.Timestamp(row.time)
            if next_allowed is None or timestamp >= next_allowed:
                selected.append(int(row.Index))
                next_allowed = timestamp + pd.Timedelta(hours=78)
    return frame.loc[selected].sort_values(["station", "time"]).reset_index(drop=True)


def audit_p3(root: Path) -> dict[str, Any]:
    readme = require_file(root, "README.md")
    wave_path = require_file(root, "train_wave.csv")
    atmos_path = require_file(root, "train_atmos.csv")
    wave = pd.read_csv(wave_path)
    atmos = pd.read_csv(atmos_path)
    if len(wave) != 118_152 or len(atmos) != 130_896:
        raise AuditError("P3 training row count changed")
    if list(wave.columns) != ["station", "time", "hs", "tp", "hmax", "wvdir"]:
        raise AuditError("P3 train_wave schema changed")
    if list(atmos.columns) != [
        "station",
        "time",
        "wspd",
        "gust",
        "wdir",
        "airt",
        "relh",
        "caph",
    ]:
        raise AuditError("P3 train_atmos schema changed")
    for frame in (wave, atmos):
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
        frame.sort_values(["station", "time"], inplace=True, kind="mergesort")
        frame.reset_index(drop=True, inplace=True)
        if frame.duplicated(["station", "time"]).any():
            raise AuditError("P3 duplicate training keys")
    if set(wave["station"].unique()) != set(P3_STATIONS):
        raise AuditError("P3 station set changed")

    pieces: list[pd.DataFrame] = []
    for station, part in wave.groupby("station", sort=True, observed=True):
        indexed = part.set_index("time").sort_index()
        base = indexed[["hs"]].rename(columns={"hs": "current_hs"}).copy()
        dense_stride = P3_DENSE_SPACING_MINUTES // P3_WAVE_CADENCE_MINUTES
        base["canonical_60min_slot"] = (
            np.arange(len(base), dtype=np.int64) % dense_stride == 0
        )
        base["history_48h_elapsed"] = base.index >= (
            base.index.min() + pd.Timedelta(hours=48)
        )
        base["hs_minus_12h"] = indexed["hs"].reindex(
            base.index - pd.Timedelta(hours=12)
        ).to_numpy()
        base["hs_minus_48h"] = indexed["hs"].reindex(
            base.index - pd.Timedelta(hours=48)
        ).to_numpy()
        for lead in P3_LEADS_H:
            base[f"target_{lead}h"] = indexed["hs"].reindex(
                base.index + pd.Timedelta(hours=lead)
            ).to_numpy()
        base["station"] = str(station)
        base["time"] = base.index
        pieces.append(base.reset_index(drop=True))
    anchor = pd.concat(pieces, ignore_index=True)
    required = ["current_hs", "hs_minus_12h", "hs_minus_48h"] + [
        f"target_{lead}h" for lead in P3_LEADS_H
    ]
    complete = anchor.loc[
        anchor["canonical_60min_slot"] & anchor["history_48h_elapsed"]
    ].dropna(subset=required).copy()
    eligible = complete.loc[complete["current_hs"].ge(1.5)].copy()
    eligible["rise_12h"] = eligible["current_hs"] - eligible["hs_minus_12h"]
    matched = eligible.loc[
        eligible["current_hs"].lt(2.2) & eligible["rise_12h"].gt(0.2)
    ].copy()
    strict_independent = _greedy_78h(matched)

    # The canonical P3 builder requires 48 hours to have elapsed and complete
    # future targets, but it does not require the single Hs value at t-48h to
    # be finite.  Keep the stricter diagnostic above, while deriving promotion
    # support from the exact shared builder used by the modelling pipeline.
    empty = pd.DataFrame()
    canonical_grid = build_training_grid(
        P3Data(
            wave=wave,
            atmos=atmos,
            test_context=empty,
            test_index=empty,
            sample_submission=empty,
            baseline=empty,
        )
    )
    canonical = build_anchor_table(canonical_grid, dense_spacing_minutes=60).copy()
    canonical["hs_minus_12h"] = np.nan
    for station in P3_STATIONS:
        grid_part = (
            canonical_grid.loc[canonical_grid["station"].eq(station)]
            .sort_values("time")
            .reset_index(drop=True)
        )
        station_index = canonical.index[canonical["station"].eq(station)]
        positions = canonical.loc[station_index, "grid_position"].to_numpy(dtype=np.int64)
        canonical.loc[station_index, "hs_minus_12h"] = grid_part["hs"].to_numpy(
            dtype=np.float64
        )[positions - 12 * 6]
    canonical["rise_12h"] = canonical["current_hs"] - canonical["hs_minus_12h"]
    canonical_matched = canonical.loc[
        canonical["current_hs"].lt(2.2)
        & canonical["rise_12h"].gt(0.2)
        & canonical["hs_minus_12h"].notna()
    ].copy()
    canonical_independent = _greedy_78h(
        canonical_matched.rename(columns={"anchor_time": "time"})
    )

    by_lead: dict[str, Any] = {}
    for lead in P3_LEADS_H:
        residual = canonical_independent[f"target_{lead}"] - canonical_independent[
            "current_hs"
        ]
        by_lead[str(lead)] = {
            "persistence_rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "future_change": finite_summary(residual),
        }
    atmos_numeric = ["wspd", "gust", "wdir", "airt", "relh", "caph"]
    return {
        "source": {
            "train_wave": {
                "bytes": wave_path.stat().st_size,
                "sha256": sha256_file(wave_path),
            },
            "train_atmos": {
                "bytes": atmos_path.stat().st_size,
                "sha256": sha256_file(atmos_path),
            },
            "readme_sha256": sha256_file(readme),
        },
        "rows": {"train_wave": int(len(wave)), "train_atmos": int(len(atmos))},
        "official_leads_hours": list(P3_LEADS_H),
        "canonical_dense_spacing_minutes": P3_DENSE_SPACING_MINUTES,
        "strict_complete_hs_lag48_and_six_lead_anchor_count": int(len(complete)),
        "canonical_elapsed_48h_high_state_anchor_count": int(len(canonical)),
        "selection_matched_dense_anchor_count": int(len(canonical_matched)),
        "selection_matched_78h_independent_count": int(len(canonical_independent)),
        "selection_matched_78h_by_station": {
            str(key): int(value)
            for key, value in canonical_independent["station"]
            .value_counts()
            .sort_index()
            .items()
        },
        "selection_matched_current_hs": finite_summary(
            canonical_independent["current_hs"]
        ),
        "selection_matched_rise_12h": finite_summary(
            canonical_independent["rise_12h"]
        ),
        "selection_matched_persistence_by_lead": by_lead,
        "strict_hs_lag48_selection_diagnostic": {
            "dense_anchor_count": int(len(matched)),
            "independent_78h_count": int(len(strict_independent)),
            "independent_78h_by_station": {
                str(key): int(value)
                for key, value in strict_independent["station"]
                .value_counts()
                .sort_index()
                .items()
            },
            "interpretation": (
                "Diagnostic-only subset requiring finite Hs exactly at t-48h; the canonical "
                "pipeline requires 48h elapsed and supports masked missing history."
            ),
        },
        "atmos_complete_fraction_by_column": {
            column: float(atmos[column].notna().mean()) for column in atmos_numeric
        },
        "prior_audit_contract_correction": {
            "prior_horizons_hours": [12, 24, 36, 48, 60, 72],
            "correct_horizons_hours": list(P3_LEADS_H),
            "prior_horizon_vector_matches_official": False,
        },
    }


def run(p1_dir: Path, p2_dir: Path, p3_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "original_training_structure_reaudit.20260830.v1",
        "p1": audit_p1(p1_dir),
        "p2": audit_p2(p2_dir),
        "p3": audit_p3(p3_dir),
        "data_access": {
            "opened_training_files": [
                "P1/README.md",
                "P1/train.csv",
                "P2/README.md",
                "P2/observations.csv",
                "P3/README.md",
                "P3/train_wave.csv",
                "P3/train_atmos.csv",
            ],
            "test_rows_read": 0,
            "test_index_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_rows_read": 0,
            "score_rows_read": 0,
            "submission_rows_read": 0,
            "hidden_label_rows_read": 0,
            "csv_created": 0,
            "uploads": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-dir", type=Path, required=True)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--p3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.p1_dir, args.p2_dir, args.p3_dir)
    rendered = json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output.resolve())
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
