"""Sealed zero-fit, train-only P3 selection-matched cohort preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p3_wave.data import LEADS, STATIONS, P3Data, build_anchor_table, build_training_grid
from p3_wave.validation import DEFAULT_WINDOWS

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_selection_matched_cohort_preflight_20260830_v1"
DEFAULT_CONFIG = (
    ROOT / "configs/experiments/p3_selection_matched_cohort_preflight_20260830_v1.json"
)
ALLOWED_SOURCE_BASENAMES = ("README.md", "train_wave.csv", "train_atmos.csv")
WAVE_COLUMNS = ("station", "time", "hs", "tp", "hmax", "wvdir")
ATMOS_COLUMNS = ("station", "time", "wspd", "gust", "wdir", "airt", "relh", "caph")
EXPECTED_TRAIN_ROWS = {"train_wave": 118_152, "train_atmos": 130_896}
EXPECTED_ROWS_BY_STATION = {
    "train_wave": {"G-ORS": 39_384, "I-ORS": 39_384, "S-ORS": 39_384},
    "train_atmos": {"G-ORS": 78_768, "I-ORS": 26_064, "S-ORS": 26_064},
}
EXPECTED_NATIVE_CADENCE = {
    "train_wave": {"G-ORS": 20, "I-ORS": 20, "S-ORS": 20},
    "train_atmos": {"G-ORS": 10, "I-ORS": 10, "S-ORS": 10},
}


class CohortPreflightError(ValueError):
    """Raised when the sealed train-only preflight contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CohortPreflightError(message)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_config(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "p3.selection_matched_cohort_preflight.preregistration.v1",
        "preflight config schema changed",
    )
    _require(config.get("experiment_id") == EXPERIMENT_ID, "experiment ID changed")

    boundary = config["data_boundary"]
    _require(boundary["explicit_p3_dir_required"] is True, "explicit P3 directory is required")
    _require(
        boundary["environment_or_repository_search_allowed"] is False,
        "source-directory search must remain forbidden",
    )
    _require(
        tuple(boundary["allowed_source_basenames"]) == ALLOWED_SOURCE_BASENAMES,
        "allowed source basenames changed",
    )
    _require(boundary["load_p3_data_allowed"] is False, "broad P3 loader must remain forbidden")
    _require(boundary["raw_rows_in_receipt_allowed"] is False, "raw receipt rows forbidden")
    _require(boundary["source_mutation_allowed"] is False, "source mutation forbidden")

    source = config["source_contract"]
    _require(tuple(source["stations"]) == STATIONS, "station contract changed")
    for name, columns in (("train_wave", WAVE_COLUMNS), ("train_atmos", ATMOS_COLUMNS)):
        contract = source[name]
        _require(contract["rows"] == EXPECTED_TRAIN_ROWS[name], f"{name} row contract changed")
        _require(tuple(contract["columns"]) == columns, f"{name} column contract changed")
        _require(
            contract["rows_by_station"] == EXPECTED_ROWS_BY_STATION[name],
            f"{name} station row contract changed",
        )
        _require(
            contract["native_cadence_minutes_by_station"] == EXPECTED_NATIVE_CADENCE[name],
            f"{name} cadence contract changed",
        )

    cohort = config["cohort_contract"]
    _require(tuple(cohort["official_leads_hours"]) == LEADS, "official lead vector changed")
    _require(cohort["canonical_grid_minutes"] == 10, "canonical grid must remain 10 minutes")
    _require(
        cohort["canonical_dense_anchor_minutes"] == 60,
        "canonical dense anchor spacing must remain 60 minutes",
    )
    _require(cohort["wave_cadence_minutes"] == 20, "wave cadence changed")
    _require(cohort["history_hours"] == 48, "history footprint changed")
    _require(cohort["current_hs_min_inclusive_m"] == 1.5, "lower Hs bound changed")
    _require(cohort["current_hs_max_exclusive_m"] == 2.2, "upper Hs bound changed")
    _require(cohort["rise_lookback_hours"] == 12, "rise lookback changed")
    _require(cohort["rise_min_exclusive_m"] == 0.2, "rise threshold changed")
    _require(cohort["station_global_gap_hours"] == 78, "station gap changed")

    frozen_windows = tuple(
        (
            row["name"],
            pd.Timestamp(row["validation_start_utc"]).strftime("%Y-%m-%d"),
            pd.Timestamp(row["validation_end_utc"]).strftime("%Y-%m-%d"),
        )
        for row in config["forward_windows"]
    )
    _require(frozen_windows == DEFAULT_WINDOWS, "forward validation windows changed")
    _require(
        all(row["train_cutoff_hours_before_start"] == 78 for row in config["forward_windows"]),
        "forward train cutoff changed",
    )

    support = config["support_gates"]
    _require(
        support["minimum_global_independent_cases_per_station"] == 30,
        "global station support gate changed",
    )
    _require(
        support["minimum_independent_cases_per_complete_historical_window"] == 20,
        "historical window support gate changed",
    )
    _require(
        support["minimum_scientifically_applicable_complete_windows"] == 2,
        "minimum applicable complete-window gate changed",
    )

    sensor = config["sensor_error_flags"]
    _require(sensor["aggregate_only"] is True, "sensor flags must remain aggregate-only")
    _require(sensor["delete_or_mask_rows"] is False, "sensor flags may not delete rows")
    _require(
        sensor["use_flags_for_cohort_membership"] is False,
        "sensor flags may not alter cohort membership",
    )

    execution = config["execution_contract"]
    for key in ("model_fit_count", "prediction_row_count", "official_row_count", "csv_output_count"):
        _require(execution[key] == 0, f"zero-execution contract changed: {key}")
    _require(execution["submission_or_upload_allowed"] is False, "submission must be forbidden")
    _require(execution["aggregate_json_receipt_only"] is True, "only JSON receipt is allowed")

    correction = config["prior_audit_correction"]
    _require(
        tuple(correction["invalid_prior_horizons_hours"]) == (12, 24, 36, 48, 60, 72),
        "prior invalid horizon evidence changed",
    )
    _require(
        tuple(correction["corrected_horizons_hours"]) == LEADS,
        "corrected horizon evidence changed",
    )
    closed = config["closed_family_boundary"]["hierarchical_residual_basis_dense72"]
    _require(closed["status"] == "CLOSED_EXACT_45_FIT_FAMILY", "dense72 family reopened")
    _require(abs(float(closed["full_delta_m"]) - 0.067295) < 1e-12, "dense72 evidence changed")
    _require(closed["all_three_folds_worse"] is True, "dense72 fold evidence changed")


def resolve_train_only_source_paths(p3_dir: Path) -> dict[str, Path]:
    root = p3_dir.expanduser().resolve(strict=True)
    _require(root.is_dir(), f"explicit P3 directory is not a directory: {root}")
    paths = {name: (root / name).resolve(strict=True) for name in ALLOWED_SOURCE_BASENAMES}
    for name, path in paths.items():
        _require(path.is_file(), f"required train-only source is not a file: {name}")
        _require(path.parent == root, f"train-only source escapes explicit P3 directory: {name}")
        _require(path.name == name, f"source basename changed: {name}")
    return {"root": root, **paths}


def _validate_training_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: tuple[str, ...],
    config: dict[str, Any],
) -> pd.DataFrame:
    contract = config["source_contract"][name]
    _require(tuple(frame.columns) == columns, f"{name} schema mismatch")
    _require(len(frame) == contract["rows"], f"{name} row count mismatch")
    parsed = frame.copy()
    parsed["time"] = pd.to_datetime(parsed["time"], utc=True, errors="raise")
    _require(set(parsed["station"].astype(str).unique()) == set(STATIONS), f"{name} stations changed")
    counts = {
        str(key): int(value)
        for key, value in parsed["station"].value_counts().sort_index().items()
    }
    _require(counts == contract["rows_by_station"], f"{name} station row counts changed")
    parsed.sort_values(["station", "time"], inplace=True, kind="mergesort")
    parsed.reset_index(drop=True, inplace=True)
    return parsed


def load_train_only_sources(
    paths: dict[str, Path], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    readme_bytes = paths["README.md"].read_bytes()
    _require(bool(readme_bytes.strip()), "source README is empty")
    wave = pd.read_csv(paths["train_wave.csv"])
    atmos = pd.read_csv(paths["train_atmos.csv"])
    wave = _validate_training_frame(
        wave, name="train_wave", columns=WAVE_COLUMNS, config=config
    )
    atmos = _validate_training_frame(
        atmos, name="train_atmos", columns=ATMOS_COLUMNS, config=config
    )
    source_receipt = {
        "README.md": {
            "bytes": int(len(readme_bytes)),
            "sha256": hashlib.sha256(readme_bytes).hexdigest(),
        },
        "train_wave.csv": {
            "bytes": int(paths["train_wave.csv"].stat().st_size),
            "sha256": sha256_file(paths["train_wave.csv"]),
            "rows": int(len(wave)),
        },
        "train_atmos.csv": {
            "bytes": int(paths["train_atmos.csv"].stat().st_size),
            "sha256": sha256_file(paths["train_atmos.csv"]),
            "rows": int(len(atmos)),
        },
    }
    return wave, atmos, source_receipt


def _cadence_summary(
    frame: pd.DataFrame, *, expected_by_station: dict[str, int]
) -> dict[str, Any]:
    by_station: dict[str, Any] = {}
    all_pass = True
    for station in STATIONS:
        part = frame.loc[frame["station"].eq(station)].sort_values("time")
        delta = part["time"].diff().dt.total_seconds().div(60).dropna().to_numpy(float)
        unique = sorted(float(value) for value in np.unique(delta))
        expected = int(expected_by_station[station])
        passed = bool(len(delta) > 0 and np.all(delta == expected))
        all_pass &= passed
        by_station[station] = {
            "rows": int(len(part)),
            "expected_minutes": expected,
            "observed_unique_minutes": unique,
            "exact_expected": passed,
        }
    return {"all_stations_pass": bool(all_pass), "by_station": by_station}


def build_canonical_train_only_surface(
    wave: pd.DataFrame, atmos: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame()
    training_only = P3Data(
        wave=wave,
        atmos=atmos,
        test_context=empty,
        test_index=empty,
        sample_submission=empty,
        baseline=empty,
    )
    grid = build_training_grid(training_only)
    anchors = build_anchor_table(grid, dense_spacing_minutes=60)
    return grid, anchors


def enrich_and_check_anchor_footprints(
    grid: pd.DataFrame, anchors: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    enriched = anchors.copy()
    enriched["hs_minus_12h"] = np.nan
    dense_slot_pass = True
    history_pass = True
    target_pass = True
    current_pass = True
    grid_cadence_pass = True

    for station in STATIONS:
        group = (
            grid.loc[grid["station"].eq(station)].sort_values("time").reset_index(drop=True)
        )
        grid_delta = group["time"].diff().dt.total_seconds().div(60).dropna()
        grid_cadence_pass &= bool(len(grid_delta) > 0 and grid_delta.eq(10).all())
        station_index = enriched.index[enriched["station"].eq(station)]
        position = enriched.loc[station_index, "grid_position"].to_numpy(dtype=np.int64)
        hs = group["hs"].to_numpy(dtype=float)
        dense_slot_pass &= bool(np.all(position % 6 == 0))
        history_pass &= bool(np.all(position >= 48 * 6))
        current = hs[position]
        expected_current = enriched.loc[station_index, "current_hs"].to_numpy(dtype=float)
        current_pass &= bool(np.array_equal(current, expected_current, equal_nan=True))
        prior_position = position - 12 * 6
        enriched.loc[station_index, "hs_minus_12h"] = hs[prior_position]
        for lead in LEADS:
            target_position = position + lead * 6
            target_pass &= bool(np.all(target_position < len(group)))
            expected = enriched.loc[station_index, f"target_{lead}"].to_numpy(dtype=float)
            target_pass &= bool(
                np.array_equal(hs[target_position], expected, equal_nan=True)
                and np.isfinite(expected).all()
            )

    checks = {
        "canonical_grid_exact_10_minutes": bool(grid_cadence_pass),
        "canonical_dense_anchor_exact_60_minutes": bool(dense_slot_pass),
        "history_48h_elapsed_before_every_anchor": bool(history_pass),
        "current_hs_matches_grid": bool(current_pass),
        "official_six_targets_match_grid_and_are_finite": bool(target_pass),
        "official_leads_hours": list(LEADS),
        "maximum_target_horizon_hours": max(LEADS),
    }
    _require(all(value is True for key, value in checks.items() if key not in {"official_leads_hours", "maximum_target_horizon_hours"}), "canonical anchor footprint check failed")
    return enriched, checks


def build_selection_matched_cohort(
    anchors: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    cohort = config["cohort_contract"]
    matched = anchors.copy()
    matched["rise_12h"] = matched["current_hs"] - matched["hs_minus_12h"]
    mask = (
        matched["current_hs"].ge(cohort["current_hs_min_inclusive_m"])
        & matched["current_hs"].lt(cohort["current_hs_max_exclusive_m"])
        & matched["rise_12h"].gt(cohort["rise_min_exclusive_m"])
        & matched["hs_minus_12h"].notna()
    )
    return matched.loc[mask].sort_values(["station", "anchor_time"]).reset_index(drop=True)


def select_station_global_independent(
    frame: pd.DataFrame, *, gap_hours: int = 78
) -> pd.DataFrame:
    selected: list[int] = []
    ordered = frame.sort_values(["station", "anchor_time"]).reset_index(drop=True)
    for _, part in ordered.groupby("station", sort=True, observed=True):
        next_allowed: pd.Timestamp | None = None
        for row in part.itertuples(index=True):
            timestamp = pd.Timestamp(row.anchor_time)
            if next_allowed is None or timestamp >= next_allowed:
                selected.append(int(row.Index))
                next_allowed = timestamp + pd.Timedelta(hours=gap_hours)
    return ordered.loc[selected].sort_values(["station", "anchor_time"]).reset_index(drop=True)


def _counts_by_station(frame: pd.DataFrame) -> dict[str, int]:
    observed = frame["station"].value_counts().to_dict() if len(frame) else {}
    return {station: int(observed.get(station, 0)) for station in STATIONS}


def _minimum_station_gap_hours(frame: pd.DataFrame) -> float | None:
    gaps: list[float] = []
    for _, part in frame.groupby("station", sort=True, observed=True):
        delta = (
            part.sort_values("anchor_time")["anchor_time"]
            .diff()
            .dt.total_seconds()
            .div(3600)
            .dropna()
        )
        gaps.extend(float(value) for value in delta)
    return min(gaps) if gaps else None


def _frame_contract_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, pd.Timestamp):
                values.append(value.tz_convert("UTC").isoformat())
            elif isinstance(value, (float, np.floating)):
                values.append(float(value).hex())
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            else:
                values.append(str(value))
        digest.update(("|".join(values) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _robust_sigma(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0
    center = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - center)))


def _jump_return_flags(wave: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    contract = config["sensor_error_flags"]["jump_return"]
    by_station: dict[str, Any] = {}
    total = 0
    for station in STATIONS:
        part = wave.loc[wave["station"].eq(station)].sort_values("time").reset_index(drop=True)
        hs = part["hs"].to_numpy(dtype=float)
        times = part["time"]
        first_delta = hs[1:] - hs[:-1]
        sigma = _robust_sigma(first_delta)
        jump_threshold = max(
            float(contract["absolute_jump_floor_m"]),
            float(contract["robust_sigma_multiplier"]) * sigma,
        )
        return_tolerance = max(
            float(contract["absolute_return_tolerance_floor_m"]),
            float(contract["return_tolerance_sigma_multiplier"]) * sigma,
        )
        if len(part) < 3:
            flags = np.zeros(len(part), dtype=bool)
        else:
            before = hs[1:-1] - hs[:-2]
            after = hs[2:] - hs[1:-1]
            neighbor_return = np.abs(hs[2:] - hs[:-2]) <= return_tolerance
            opposite = before * after < 0
            large = (np.abs(before) >= jump_threshold) & (np.abs(after) >= jump_threshold)
            cadence_before = (
                times.diff().dt.total_seconds().div(60).iloc[1:-1].to_numpy(float)
                == contract["cadence_minutes"]
            )
            cadence_after = (
                times.diff().dt.total_seconds().div(60).iloc[2:].to_numpy(float)
                == contract["cadence_minutes"]
            )
            interior = (
                np.isfinite(before)
                & np.isfinite(after)
                & neighbor_return
                & opposite
                & large
                & cadence_before
                & cadence_after
            )
            flags = np.pad(interior, (1, 1), constant_values=False)
        count = int(flags.sum())
        total += count
        by_station[station] = {
            "flag_count": count,
            "robust_sigma_m": sigma,
            "jump_threshold_m": jump_threshold,
            "return_tolerance_m": return_tolerance,
        }
    return {"total_flag_count": total, "by_station": by_station}


def sensor_error_flag_aggregates(
    wave: pd.DataFrame, atmos: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    direction_low, direction_high = config["sensor_error_flags"][
        "direction_bounds_inclusive_degrees"
    ]
    humidity_low, humidity_high = config["sensor_error_flags"][
        "relative_humidity_bounds_inclusive_percent"
    ]

    def out_of_bounds(series: pd.Series, low: float, high: float) -> int:
        finite = series.notna()
        return int((finite & (series.lt(low) | series.gt(high))).sum())

    wave_duplicate = wave.duplicated(["station", "time"], keep=False)
    atmos_duplicate = atmos.duplicated(["station", "time"], keep=False)
    return {
        "negative_hs_rows": int((wave["hs"].notna() & wave["hs"].lt(0)).sum()),
        "negative_period_rows": int((wave["tp"].notna() & wave["tp"].lt(0)).sum()),
        "wave_direction_out_of_bounds_rows": out_of_bounds(
            wave["wvdir"], direction_low, direction_high
        ),
        "wind_direction_out_of_bounds_rows": out_of_bounds(
            atmos["wdir"], direction_low, direction_high
        ),
        "relative_humidity_out_of_bounds_rows": out_of_bounds(
            atmos["relh"], humidity_low, humidity_high
        ),
        "duplicate_station_time": {
            "train_wave_rows": int(wave_duplicate.sum()),
            "train_wave_keys": int(
                wave.loc[wave_duplicate, ["station", "time"]].drop_duplicates().shape[0]
            ),
            "train_atmos_rows": int(atmos_duplicate.sum()),
            "train_atmos_keys": int(
                atmos.loc[atmos_duplicate, ["station", "time"]].drop_duplicates().shape[0]
            ),
        },
        "jump_return_hs": _jump_return_flags(wave, config),
        "rows_deleted_or_masked": 0,
        "flags_used_for_cohort_membership": False,
        "high_hs_or_storm_extreme_is_an_error_flag": False,
    }


def summarize_support(
    grid: pd.DataFrame,
    anchors: pd.DataFrame,
    matched: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gap_hours = int(config["cohort_contract"]["station_global_gap_hours"])
    global_independent = select_station_global_independent(matched, gap_hours=gap_hours)
    global_counts = _counts_by_station(global_independent)
    station_minimum = int(config["support_gates"]["minimum_global_independent_cases_per_station"])
    global_gate = {
        station: {
            "count": count,
            "minimum": station_minimum,
            "pass": bool(count >= station_minimum),
        }
        for station, count in global_counts.items()
    }

    validation_mask = pd.Series(False, index=matched.index)
    for window in config["forward_windows"]:
        start = pd.Timestamp(window["validation_start_utc"])
        end = pd.Timestamp(window["validation_end_utc"])
        validation_mask |= matched["anchor_time"].ge(start) & matched["anchor_time"].lt(end)
    validation_pool = matched.loc[validation_mask].copy()
    validation_independent = select_station_global_independent(
        validation_pool, gap_hours=gap_hours
    )

    coverage = {
        station: {
            "minimum": pd.Timestamp(part["time"].min()),
            "maximum": pd.Timestamp(part["time"].max()),
        }
        for station, part in grid.groupby("station", sort=True, observed=True)
    }
    fold_minimum = int(
        config["support_gates"]["minimum_independent_cases_per_complete_historical_window"]
    )
    applicable_window_minimum = int(
        config["support_gates"]["minimum_scientifically_applicable_complete_windows"]
    )
    windows: dict[str, Any] = {}
    fold_required_passes: list[bool] = []
    for window in config["forward_windows"]:
        name = window["name"]
        start = pd.Timestamp(window["validation_start_utc"])
        end = pd.Timestamp(window["validation_end_utc"])
        cutoff = start - pd.Timedelta(hours=window["train_cutoff_hours_before_start"])
        history_start = start - pd.Timedelta(hours=config["cohort_contract"]["history_hours"])
        target_end = end + pd.Timedelta(hours=max(LEADS))
        applicable_by_station = {
            station: bool(
                coverage[station]["minimum"] <= history_start
                and coverage[station]["maximum"] >= target_end
            )
            for station in STATIONS
        }
        applicable = all(applicable_by_station.values())
        dense_validation = matched.loc[
            matched["anchor_time"].ge(start) & matched["anchor_time"].lt(end)
        ]
        independent_validation = validation_independent.loc[
            validation_independent["anchor_time"].ge(start)
            & validation_independent["anchor_time"].lt(end)
        ]
        canonical_train = anchors.loc[anchors["anchor_time"].lt(cutoff)]
        matched_train = matched.loc[matched["anchor_time"].lt(cutoff)]
        independent_train = select_station_global_independent(
            matched_train, gap_hours=gap_hours
        )
        count = int(len(independent_validation))
        if applicable:
            passed = count >= fold_minimum
            status = (
                config["support_gates"]["passed_support_status"]
                if passed
                else config["support_gates"]["failed_support_status"]
            )
            fold_required_passes.append(bool(passed))
        else:
            passed = None
            status = config["support_gates"]["incomplete_historical_window_status"]
        max_train_time = canonical_train["anchor_time"].max() if len(canonical_train) else None
        min_validation_time = (
            independent_validation["anchor_time"].min() if len(independent_validation) else None
        )
        train_cutoff_pass = bool(max_train_time is None or pd.Timestamp(max_train_time) < cutoff)
        separation_hours = (
            None
            if max_train_time is None or min_validation_time is None
            else float(
                (pd.Timestamp(min_validation_time) - pd.Timestamp(max_train_time)).total_seconds()
                / 3600
            )
        )
        windows[name] = {
            "validation_start_utc": start.isoformat(),
            "validation_end_utc": end.isoformat(),
            "train_cutoff_utc": cutoff.isoformat(),
            "scientifically_applicable_complete_footprint": applicable,
            "complete_footprint_by_station": applicable_by_station,
            "canonical_train_anchor_count": int(len(canonical_train)),
            "selection_matched_train_dense_count": int(len(matched_train)),
            "selection_matched_train_independent_count": int(len(independent_train)),
            "validation_selection_matched_dense_count": int(len(dense_validation)),
            "validation_selection_matched_dense_by_station": _counts_by_station(dense_validation),
            "validation_selection_matched_independent_count": count,
            "validation_selection_matched_independent_by_station": _counts_by_station(
                independent_validation
            ),
            "minimum_independent_cases_if_applicable": fold_minimum,
            "support_gate_pass": passed,
            "support_gate_status": status,
            "train_cutoff_strictly_respected": train_cutoff_pass,
            "nearest_train_to_selected_validation_anchor_gap_hours": separation_hours,
            "membership_note": (
                "Counts include only anchors inside this frozen historical window after the "
                "Hs/rise filter and one station-global 78-hour greedy pass across all three "
                "validation windows; incomplete window coverage is reported as not applicable, "
                "never widened after observing support."
            ),
        }

    validation_min_gap = _minimum_station_gap_hours(validation_independent)
    applicable_window_count = len(fold_required_passes)
    minimum_applicable_windows_pass = applicable_window_count >= applicable_window_minimum
    applicable_window_support_pass = bool(
        minimum_applicable_windows_pass and all(fold_required_passes)
    )
    support = {
        "selection_matched_dense_count": int(len(matched)),
        "selection_matched_global_independent_count": int(len(global_independent)),
        "selection_matched_global_independent_by_station": global_counts,
        "global_station_support_gate": global_gate,
        "global_station_minimum_gap_hours": _minimum_station_gap_hours(global_independent),
        "validation_union_independent_count": int(len(validation_independent)),
        "validation_union_independent_by_station": _counts_by_station(validation_independent),
        "validation_union_minimum_station_gap_hours": validation_min_gap,
        "scientifically_applicable_complete_window_count": applicable_window_count,
        "minimum_scientifically_applicable_complete_windows": applicable_window_minimum,
        "overall_window_support_gate_status": (
            config["support_gates"]["passed_support_status"]
            if applicable_window_support_pass
            else config["support_gates"]["failed_support_status"]
        ),
        "forward_windows": windows,
    }
    gates = {
        "all_global_station_support_gates_pass": all(
            item["pass"] for item in global_gate.values()
        ),
        "minimum_scientifically_applicable_complete_windows_pass": (
            minimum_applicable_windows_pass
        ),
        "all_scientifically_applicable_window_support_gates_pass": (
            applicable_window_support_pass
        ),
        "station_global_78h_spacing_pass": bool(
            (validation_min_gap is None or validation_min_gap >= gap_hours)
            and (
                support["global_station_minimum_gap_hours"] is None
                or support["global_station_minimum_gap_hours"] >= gap_hours
            )
        ),
        "all_train_cutoffs_pass": all(
            item["train_cutoff_strictly_respected"] for item in windows.values()
        ),
    }
    return support, gates


def run_preflight(p3_dir: Path, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    paths = resolve_train_only_source_paths(p3_dir)
    wave, atmos, source_receipt = load_train_only_sources(paths, config)
    sensor_flags = sensor_error_flag_aggregates(wave, atmos, config)
    duplicate = sensor_flags["duplicate_station_time"]
    _require(
        duplicate["train_wave_rows"] == 0 and duplicate["train_atmos_rows"] == 0,
        "duplicate station-time keys prevent canonical no-deletion construction",
    )
    wave_cadence = _cadence_summary(
        wave, expected_by_station=EXPECTED_NATIVE_CADENCE["train_wave"]
    )
    atmos_cadence = _cadence_summary(
        atmos, expected_by_station=EXPECTED_NATIVE_CADENCE["train_atmos"]
    )
    _require(wave_cadence["all_stations_pass"], "wave cadence contract failed")
    _require(atmos_cadence["all_stations_pass"], "atmos cadence contract failed")

    grid, base_anchors = build_canonical_train_only_surface(wave, atmos)
    anchors, footprint_checks = enrich_and_check_anchor_footprints(grid, base_anchors)
    matched = build_selection_matched_cohort(anchors, config)
    support, support_gates = summarize_support(grid, anchors, matched, config)
    gap_hours = int(config["cohort_contract"]["station_global_gap_hours"])
    global_independent = select_station_global_independent(matched, gap_hours=gap_hours)

    extreme_count = int(anchors["current_hs"].ge(2.2).sum())
    contract_columns = [
        "anchor_id",
        "station",
        "anchor_time",
        "grid_position",
        "current_hs",
        *[f"target_{lead}" for lead in LEADS],
    ]
    payload: dict[str, Any] = {
        "schema_version": "p3.selection_matched_cohort_preflight.receipt.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PREFLIGHT_COMPLETE_ZERO_FIT",
        "provenance": {
            "config_path": config_path.resolve().relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(config_path.resolve()),
            "runner_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "source": source_receipt,
        },
        "data_access": {
            "explicit_p3_dir_used": True,
            "opened_source_basenames": list(ALLOWED_SOURCE_BASENAMES),
            "forbidden_source_basenames_opened": [],
            "official_test_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_rows_read": 0,
            "submission_rows_read": 0,
            "hidden_or_answer_rows_read": 0,
            "source_rows_modified_or_deleted": 0,
        },
        "source_contract": {
            "rows": {"train_wave": int(len(wave)), "train_atmos": int(len(atmos))},
            "wave_cadence": wave_cadence,
            "atmos_cadence": atmos_cadence,
        },
        "cohort_contract": {
            "official_leads_hours": list(LEADS),
            "history_hours": 48,
            "canonical_grid_minutes": 10,
            "canonical_dense_anchor_minutes": 60,
            "current_hs_bounds_m": {"minimum_inclusive": 1.5, "maximum_exclusive": 2.2},
            "rise_12h_minimum_exclusive_m": 0.2,
            "station_global_gap_hours": gap_hours,
            "canonical_anchor_count": int(len(anchors)),
            "canonical_anchor_contract_sha256": _frame_contract_sha256(
                anchors, contract_columns
            ),
            "selection_matched_global_independent_sha256": _frame_contract_sha256(
                global_independent, ["anchor_id", "station", "anchor_time"]
            ),
        },
        "footprint_checks": footprint_checks,
        "support": support,
        "sensor_error_flags": {
            **sensor_flags,
            "canonical_hs_ge_2_2_anchor_count_preserved": extreme_count,
            "extreme_storm_rows_deleted": 0,
            "selection_filter_is_diagnostic_membership_not_source_deletion": True,
        },
        "gates": {
            **support_gates,
            "canonical_footprint_pass": all(
                value is True
                for key, value in footprint_checks.items()
                if key not in {"official_leads_hours", "maximum_target_horizon_hours"}
            ),
            "sensor_flag_no_deletion_pass": sensor_flags["rows_deleted_or_masked"] == 0,
            "zero_fit_pass": True,
            "zero_prediction_pass": True,
            "zero_official_access_pass": True,
        },
        "execution": {
            "model_fit_count": 0,
            "prediction_row_count": 0,
            "csv_output_count": 0,
            "submission_or_upload_attempted": False,
            "closed_family_reopened": False,
        },
        "prior_audit_correction": config["prior_audit_correction"],
        "closed_family_boundary": config["closed_family_boundary"],
    }
    payload["gates"]["overall_preflight_pass"] = all(
        bool(value) for value in payload["gates"].values()
    )
    payload["seal"] = {
        "algorithm": "sha256",
        "payload_without_seal_sha256": _payload_sha256(payload),
    }
    return payload


def _write_receipt(path: Path, payload: dict[str, Any], *, source_root: Path) -> None:
    destination = path.expanduser().resolve()
    _require(
        destination != source_root and source_root not in destination.parents,
        "receipt may not be written inside the immutable P3 source directory",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False, suffix=".tmp"
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic create-if-absent: a repeated or concurrent run cannot replace a receipt.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or ROOT / config["implementation"]["default_output"]
    payload = run_preflight(args.p3_dir, args.config)
    source_root = args.p3_dir.expanduser().resolve(strict=True)
    _write_receipt(output, payload, source_root=source_root)
    print(output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
