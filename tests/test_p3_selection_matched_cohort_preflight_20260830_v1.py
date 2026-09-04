from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_wave.data import LEADS, STATIONS, P3Data, build_anchor_table, build_training_grid
from p3_wave.validation import DEFAULT_WINDOWS

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/experiments/p3_selection_matched_cohort_preflight_20260830_v1.json"
)
RUNNER_PATH = ROOT / "scripts/run_p3_selection_matched_cohort_preflight_20260830_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_selection_matched_preflight_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _small_training_frames(days: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp("2024-01-01T00:00:00+00:00")
    wave_rows: list[pd.DataFrame] = []
    atmos_rows: list[pd.DataFrame] = []
    for station_number, station in enumerate(STATIONS):
        wave_time = pd.date_range(start, periods=days * 72, freq="20min")
        wave_index = np.arange(len(wave_time), dtype=float)
        wave_rows.append(
            pd.DataFrame(
                {
                    "station": station,
                    "time": wave_time,
                    "hs": 1.55 + 0.5 * ((wave_index % 72) / 71.0),
                    "tp": 7.0 + station_number,
                    "hmax": 2.1 + 0.5 * ((wave_index % 72) / 71.0),
                    "wvdir": 90.0 + 10.0 * station_number,
                }
            )
        )
        atmos_time = pd.date_range(start, periods=days * 144, freq="10min")
        atmos_rows.append(
            pd.DataFrame(
                {
                    "station": station,
                    "time": atmos_time,
                    "wspd": 5.0,
                    "gust": 7.0,
                    "wdir": 120.0,
                    "airt": 18.0,
                    "relh": 70.0,
                    "caph": 1010.0,
                }
            )
        )
    return pd.concat(wave_rows, ignore_index=True), pd.concat(atmos_rows, ignore_index=True)


@pytest.fixture(scope="module")
def full_synthetic_p3_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_p3_train_only")
    (root / "README.md").write_text(
        "Synthetic train-only P3 contract fixture. No official rows.\n", encoding="utf-8"
    )
    start = pd.Timestamp("2024-01-01T00:00:00+00:00")
    wave_time = pd.date_range(start, periods=39_384, freq="20min")
    wave_index = np.arange(len(wave_time), dtype=float)
    phase = (wave_index % 72) / 71.0
    wave_parts: list[pd.DataFrame] = []
    for station_number, station in enumerate(STATIONS):
        wave_parts.append(
            pd.DataFrame(
                {
                    "station": station,
                    "time": wave_time,
                    "hs": 1.5 + 0.6 * phase,
                    "tp": 7.0 + station_number,
                    "hmax": 2.0 + 0.6 * phase,
                    "wvdir": 80.0 + 20.0 * station_number,
                }
            )
        )
    pd.concat(wave_parts, ignore_index=True).to_csv(root / "train_wave.csv", index=False)

    atmos_parts: list[pd.DataFrame] = []
    for station_number, station in enumerate(STATIONS):
        if station == "G-ORS":
            time = pd.date_range(start, periods=78_768, freq="10min")
        else:
            time = pd.date_range(start, periods=26_064, freq="10min")
        atmos_parts.append(
            pd.DataFrame(
                {
                    "station": station,
                    "time": time,
                    "wspd": 5.0 + station_number,
                    "gust": 7.0 + station_number,
                    "wdir": 120.0,
                    "airt": 18.0,
                    "relh": 70.0,
                    "caph": 1010.0,
                }
            )
        )
    pd.concat(atmos_parts, ignore_index=True).to_csv(root / "train_atmos.csv", index=False)
    return root


def test_preregistration_freezes_target_contract_and_closed_family() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    assert tuple(config["cohort_contract"]["official_leads_hours"]) == LEADS
    assert config["cohort_contract"]["canonical_dense_anchor_minutes"] == 60
    assert config["cohort_contract"]["history_hours"] == 48
    assert config["cohort_contract"]["station_global_gap_hours"] == 78
    assert tuple(
        (
            row["name"],
            pd.Timestamp(row["validation_start_utc"]).strftime("%Y-%m-%d"),
            pd.Timestamp(row["validation_end_utc"]).strftime("%Y-%m-%d"),
        )
        for row in config["forward_windows"]
    ) == DEFAULT_WINDOWS
    assert config["support_gates"]["minimum_global_independent_cases_per_station"] == 30
    assert (
        config["support_gates"]["minimum_independent_cases_per_complete_historical_window"]
        == 20
    )
    assert config["support_gates"]["minimum_scientifically_applicable_complete_windows"] == 2
    closed = config["closed_family_boundary"]["hierarchical_residual_basis_dense72"]
    assert closed["status"] == "CLOSED_EXACT_45_FIT_FAMILY"
    assert closed["full_delta_m"] == pytest.approx(0.067295)
    assert closed["all_three_folds_worse"] is True


def test_runner_has_fixed_train_only_read_surface() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "load_p3_data(" not in source
    assert "read_parquet" not in source
    assert tuple(RUNNER.ALLOWED_SOURCE_BASENAMES) == (
        "README.md",
        "train_wave.csv",
        "train_atmos.csv",
    )


def test_train_only_loader_ignores_present_forbidden_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wave, atmos = _small_training_frames(days=2)
    (tmp_path / "README.md").write_text("synthetic\n", encoding="utf-8")
    wave.to_csv(tmp_path / "train_wave.csv", index=False)
    atmos.to_csv(tmp_path / "train_atmos.csv", index=False)
    for name in ("test_index.csv", "sample_submission.csv", "baseline_persistence.csv"):
        (tmp_path / name).write_text("must,not,be,opened\n", encoding="utf-8")

    config = copy.deepcopy(RUNNER.load_config(CONFIG_PATH))
    for name, frame in (("train_wave", wave), ("train_atmos", atmos)):
        config["source_contract"][name]["rows"] = len(frame)
        config["source_contract"][name]["rows_by_station"] = {
            station: int(frame["station"].eq(station).sum()) for station in STATIONS
        }
    paths = RUNNER.resolve_train_only_source_paths(tmp_path)
    observed: list[str] = []
    original = RUNNER.pd.read_csv

    def recording_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        observed.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(RUNNER.pd, "read_csv", recording_read_csv)
    loaded_wave, loaded_atmos, receipt = RUNNER.load_train_only_sources(paths, config)
    assert len(loaded_wave) == len(wave)
    assert len(loaded_atmos) == len(atmos)
    assert observed == ["train_wave.csv", "train_atmos.csv"]
    assert set(receipt) == {"README.md", "train_wave.csv", "train_atmos.csv"}


def test_canonical_surface_is_exactly_shared_implementation() -> None:
    wave, atmos = _small_training_frames(days=5)
    actual_grid, actual_anchors = RUNNER.build_canonical_train_only_surface(wave, atmos)
    empty = pd.DataFrame()
    direct_data = P3Data(wave, atmos, empty, empty, empty, empty)
    expected_grid = build_training_grid(direct_data)
    expected_anchors = build_anchor_table(expected_grid, dense_spacing_minutes=60)
    pd.testing.assert_frame_equal(actual_grid, expected_grid)
    pd.testing.assert_frame_equal(actual_anchors, expected_anchors)
    enriched, checks = RUNNER.enrich_and_check_anchor_footprints(actual_grid, actual_anchors)
    assert len(enriched) == len(actual_anchors)
    assert checks["canonical_grid_exact_10_minutes"] is True
    assert checks["canonical_dense_anchor_exact_60_minutes"] is True
    assert checks["history_48h_elapsed_before_every_anchor"] is True
    assert checks["official_six_targets_match_grid_and_are_finite"] is True


def test_station_global_spacing_and_frozen_window_support() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2024-01-01T00:00:00+00:00")
    end = pd.Timestamp("2025-06-30T23:00:00+00:00")
    for station in STATIONS:
        for anchor_id, timestamp in enumerate(pd.date_range(start, end, freq="80h")):
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "station": station,
                    "anchor_time": timestamp,
                    "current_hs": 1.8,
                    "hs_minus_12h": 1.5,
                }
            )
    matched = pd.DataFrame(rows)
    anchors = matched.copy()
    grid = pd.concat(
        [
            pd.DataFrame(
                {
                    "station": station,
                    "time": pd.date_range(start - pd.Timedelta(hours=48), periods=3, freq="275D"),
                }
            )
            for station in STATIONS
        ],
        ignore_index=True,
    )
    support, gates = RUNNER.summarize_support(grid, anchors, matched, config)
    assert support["global_station_minimum_gap_hours"] >= 78
    assert support["validation_union_minimum_station_gap_hours"] >= 78
    assert gates["station_global_78h_spacing_pass"] is True
    assert gates["all_global_station_support_gates_pass"] is True
    assert support["scientifically_applicable_complete_window_count"] == 3
    assert gates["minimum_scientifically_applicable_complete_windows_pass"] is True
    assert gates["all_scientifically_applicable_window_support_gates_pass"] is True
    assert support["overall_window_support_gate_status"] == "PASS_SUPPORT"
    for window in support["forward_windows"].values():
        if window["scientifically_applicable_complete_footprint"]:
            assert window["validation_selection_matched_independent_count"] >= 20
            assert window["support_gate_pass"] is True
        assert window["train_cutoff_strictly_respected"] is True


def test_fewer_than_two_applicable_windows_cannot_pass_support() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    start = pd.Timestamp("2024-06-29T00:00:00+00:00")
    end = pd.Timestamp("2024-11-03T00:00:00+00:00")
    rows: list[dict[str, object]] = []
    for station in STATIONS:
        for anchor_id, timestamp in enumerate(
            pd.date_range("2024-07-01T00:00:00+00:00", "2024-10-31T23:00:00+00:00", freq="80h")
        ):
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "station": station,
                    "anchor_time": timestamp,
                    "current_hs": 1.8,
                    "hs_minus_12h": 1.5,
                }
            )
    matched = pd.DataFrame(rows)
    grid = pd.concat(
        [
            pd.DataFrame({"station": station, "time": [start, end]})
            for station in STATIONS
        ],
        ignore_index=True,
    )
    support, gates = RUNNER.summarize_support(grid, matched.copy(), matched, config)
    assert gates["all_global_station_support_gates_pass"] is True
    assert support["forward_windows"]["2024_h2_storm"]["support_gate_pass"] is True
    assert support["scientifically_applicable_complete_window_count"] == 1
    assert gates["minimum_scientifically_applicable_complete_windows_pass"] is False
    assert gates["all_scientifically_applicable_window_support_gates_pass"] is False
    assert support["overall_window_support_gate_status"] == "FAIL_SUPPORT"


def test_sensor_error_flags_are_aggregate_only_and_do_not_delete() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    start = pd.Timestamp("2024-01-01T00:00:00+00:00")
    wave_parts: list[pd.DataFrame] = []
    atmos_parts: list[pd.DataFrame] = []
    for station in STATIONS:
        hs = (
            [1.5, 1.51, 1.52, 1.53, 5.0, 1.54, 1.55, 1.56, 1.57]
            if station == "G-ORS"
            else [1.5 + 0.01 * index for index in range(9)]
        )
        tp = [7.0] * 9
        if station == "I-ORS":
            tp[1] = -1.0
        direction = [90.0] * 9
        if station == "S-ORS":
            direction[1] = 361.0
        wave_parts.append(
            pd.DataFrame(
                {
                    "station": station,
                    "time": pd.date_range(start, periods=9, freq="20min"),
                    "hs": hs,
                    "tp": tp,
                    "hmax": np.asarray(hs) + 0.5,
                    "wvdir": direction,
                }
            )
        )
        wind_direction = [120.0] * 9
        humidity = [70.0] * 9
        if station == "G-ORS":
            wind_direction[0] = -1.0
        if station == "I-ORS":
            humidity[1] = 101.0
        atmos_parts.append(
            pd.DataFrame(
                {
                    "station": station,
                    "time": pd.date_range(start, periods=9, freq="10min"),
                    "wspd": 5.0,
                    "gust": 7.0,
                    "wdir": wind_direction,
                    "airt": 18.0,
                    "relh": humidity,
                    "caph": 1010.0,
                }
            )
        )
    wave = pd.concat(wave_parts, ignore_index=True)
    atmos = pd.concat(atmos_parts, ignore_index=True)
    duplicate = atmos.iloc[[1]].copy()
    atmos = pd.concat([atmos, duplicate], ignore_index=True)
    wave_before = wave.copy(deep=True)
    atmos_before = atmos.copy(deep=True)
    result = RUNNER.sensor_error_flag_aggregates(wave, atmos, config)
    pd.testing.assert_frame_equal(wave, wave_before)
    pd.testing.assert_frame_equal(atmos, atmos_before)
    assert result["negative_period_rows"] == 1
    assert result["wave_direction_out_of_bounds_rows"] == 1
    assert result["wind_direction_out_of_bounds_rows"] == 1
    assert result["relative_humidity_out_of_bounds_rows"] == 1
    assert result["duplicate_station_time"]["train_atmos_rows"] == 2
    assert result["jump_return_hs"]["total_flag_count"] == 1
    assert result["rows_deleted_or_masked"] == 0
    assert result["flags_used_for_cohort_membership"] is False
    assert result["high_hs_or_storm_extreme_is_an_error_flag"] is False


def test_selection_membership_does_not_delete_storm_extremes() -> None:
    config = RUNNER.load_config(CONFIG_PATH)
    anchors = pd.DataFrame(
        {
            "anchor_id": [1, 2],
            "station": ["G-ORS", "G-ORS"],
            "anchor_time": pd.to_datetime(
                ["2024-07-01T00:00:00+00:00", "2024-07-04T06:00:00+00:00"], utc=True
            ),
            "current_hs": [1.8, 4.0],
            "hs_minus_12h": [1.5, 3.5],
        }
    )
    before = anchors.copy(deep=True)
    matched = RUNNER.build_selection_matched_cohort(anchors, config)
    pd.testing.assert_frame_equal(anchors, before)
    assert matched["anchor_id"].tolist() == [1]
    assert anchors.loc[anchors["anchor_id"].eq(2), "current_hs"].item() == 4.0


def test_full_synthetic_zero_fit_preflight_and_seal(full_synthetic_p3_dir: Path) -> None:
    result = RUNNER.run_preflight(full_synthetic_p3_dir, CONFIG_PATH)
    assert result["status"] == "PREFLIGHT_COMPLETE_ZERO_FIT"
    assert result["data_access"]["opened_source_basenames"] == [
        "README.md",
        "train_wave.csv",
        "train_atmos.csv",
    ]
    assert result["data_access"]["forbidden_source_basenames_opened"] == []
    assert result["data_access"]["official_test_rows_read"] == 0
    assert result["execution"]["model_fit_count"] == 0
    assert result["execution"]["prediction_row_count"] == 0
    assert result["execution"]["csv_output_count"] == 0
    assert result["execution"]["closed_family_reopened"] is False
    assert result["cohort_contract"]["official_leads_hours"] == list(LEADS)
    assert result["footprint_checks"]["canonical_dense_anchor_exact_60_minutes"] is True
    assert result["gates"]["overall_preflight_pass"] is True
    assert result["sensor_error_flags"]["rows_deleted_or_masked"] == 0
    seal = result["seal"]["payload_without_seal_sha256"]
    without_seal = copy.deepcopy(result)
    without_seal.pop("seal")
    assert seal == RUNNER._payload_sha256(without_seal)


def test_receipt_is_exclusive_and_cannot_overwrite(
    tmp_path: Path, full_synthetic_p3_dir: Path
) -> None:
    output = tmp_path / "receipt.json"
    RUNNER._write_receipt(output, {"status": "first"}, source_root=full_synthetic_p3_dir)
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
    with pytest.raises(FileExistsError):
        RUNNER._write_receipt(
            output, {"status": "replacement"}, source_root=full_synthetic_p3_dir
        )
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
