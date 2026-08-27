from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock, get_ident
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pandas as pd
import pytest

import p3_wave.era5_pretrain_data as era5_pretrain
from p3_wave.era5_pretrain_data import (
    COMBINED_FILE_NAME,
    CUTOFF_EXCLUSIVE_UTC,
    DATASET_DOI,
    DATASET_URL,
    DYNAMIC_VARIABLES,
    LAST_ELIGIBLE_ANCHOR_UTC,
    LICENSE_NAME,
    MAX_YEAR_REQUEST_FIELD_HOURS,
    QUARANTINE_RELATIVE,
    STATIONS,
    VARIABLES,
    DownloadAuthorizationError,
    Era5PretrainError,
    Era5Request,
    Era5SchemaError,
    FileReceipt,
    QuarantineLayout,
    SelectedCell,
    build_manifest,
    build_smoke_plan,
    build_year_plan,
    combine_derived_year_files,
    derive_selected_cell_frame,
    load_netcdf_cube,
    load_validated_combined_file,
    load_validated_derived_year_file,
    read_selected_cells,
    retrieve_cds_request,
    select_nearest_valid_ocean_cell,
    sha256_file,
    validate_existing_canonical_manifest,
    write_manifest,
    write_selected_cells,
)


def _selection(station: str) -> SelectedCell:
    point = STATIONS[station]
    return SelectedCell(
        station=station,
        station_latitude=point.latitude,
        station_longitude=point.longitude,
        latitude=round(point.latitude * 4) / 4,
        longitude=round(point.longitude * 4) / 4,
        distance_km=0.0,
        mean_land_sea_mask=0.0,
        finite_fraction={name: 1.0 for name in DYNAMIC_VARIABLES},
    )


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "prepare_p3_era5_pretrain.py"
    spec = importlib.util.spec_from_file_location("prepare_p3_era5_pretrain", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_netcdf_member(
    path: Path,
    *,
    request: Era5Request,
    variable_names: tuple[str, ...],
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    north, west, south, east = request.area
    latitudes = np.array([north, (north + south) / 2.0, south], dtype=np.float32)
    longitudes = np.array([west, (west + east) / 2.0, east], dtype=np.float32)
    with netcdf4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("valid_time", 24)
        dataset.createDimension("latitude", 3)
        dataset.createDimension("longitude", 3)
        time = dataset.createVariable("valid_time", "i8", ("valid_time",))
        time.units = "hours since 2023-12-30 00:00:00"
        time.calendar = "standard"
        time[:] = np.arange(24)
        dataset.createVariable("latitude", "f4", ("latitude",))[:] = latitudes
        dataset.createVariable("longitude", "f4", ("longitude",))[:] = longitudes
        for index, short_name in enumerate(variable_names, start=1):
            netcdf_name = "lsm" if short_name == "land_sea_mask" else short_name
            if short_name == "land_sea_mask":
                variable = dataset.createVariable(netcdf_name, "f4", ("latitude", "longitude"))
                variable[:] = np.zeros((3, 3), dtype=np.float32)
            else:
                variable = dataset.createVariable(
                    netcdf_name,
                    "f4",
                    ("valid_time", "latitude", "longitude"),
                )
                variable[:] = np.full((24, 3, 3), index, dtype=np.float32)


def _derived_frame(request: Era5Request, selection: SelectedCell) -> pd.DataFrame:
    times = pd.date_range(request.start_utc, request.end_utc, freq="h")
    values = {name: np.ones(len(times), dtype=float) for name in VARIABLES}
    values.update(
        {
            "msl": np.full(len(times), 101325.0),
            "t2m": np.full(len(times), 283.15),
            "d2m": np.full(len(times), 278.15),
            "land_sea_mask": np.zeros(len(times), dtype=float),
        }
    )
    return derive_selected_cell_frame(
        station=request.station,
        selection=selection,
        times_utc=times,
        values=values,
    )


def test_official_coordinates_and_variable_contract_are_exact() -> None:
    assert {name: (value.latitude, value.longitude) for name, value in STATIONS.items()} == {
        "G-ORS": (33.9428, 124.5919),
        "I-ORS": (32.1228, 125.1822),
        "S-ORS": (37.4231, 124.7380),
    }
    assert tuple(VARIABLES) == (
        "swh",
        "mwp",
        "hmax",
        "mwd",
        "u10",
        "v10",
        "msl",
        "t2m",
        "d2m",
        "land_sea_mask",
    )
    assert {value.public_dict()["coordinate_source"] for value in STATIONS.values()} == {
        "KIOST official KORS station introduction"
    }


def test_smoke_plan_is_three_fixed_3x3_one_day_pre2024_requests() -> None:
    plan = build_smoke_plan()
    assert len(plan) == 3
    assert {value.station for value in plan} == set(STATIONS)
    for value in plan:
        request = value.request()
        north, west, south, east = request["area"]
        assert north - south == pytest.approx(0.5)
        assert east - west == pytest.approx(0.5)
        assert request["grid"] == [0.25, 0.25]
        assert request["time"] == [f"{hour:02d}:00" for hour in range(24)]
        assert request["variable"] == list(VARIABLES.values())
        assert value.start_utc == pd.Timestamp("2023-12-30T00:00:00Z")
        assert value.end_utc == pd.Timestamp("2023-12-30T23:00:00Z")
        assert value.end_utc < CUTOFF_EXCLUSIVE_UTC


def test_actual_xarray_direct_netcdf_accepts_lsm_short_name(tmp_path: Path) -> None:
    pytest.importorskip("xarray")
    request = build_smoke_plan()[0]
    source = tmp_path / "direct-smoke.nc"
    _write_netcdf_member(
        source,
        request=request,
        variable_names=tuple(VARIABLES),
    )

    times, latitudes, longitudes, fields = load_netcdf_cube(source, expected_request=request)

    assert times.equals(pd.date_range(request.start_utc, request.end_utc, freq="h"))
    assert latitudes.shape == (3,)
    assert longitudes.shape == (3,)
    assert set(fields) == set(VARIABLES)
    assert fields["land_sea_mask"].shape == (24, 3, 3)
    assert np.count_nonzero(fields["land_sea_mask"]) == 0


def test_actual_xarray_zip_multimember_smoke_merges_all_fields(tmp_path: Path) -> None:
    pytest.importorskip("xarray")
    request = build_smoke_plan()[0]
    wave_member = tmp_path / "wave.nc"
    atmosphere_member = tmp_path / "atmosphere.nc"
    _write_netcdf_member(
        wave_member,
        request=request,
        variable_names=("swh", "mwp", "hmax", "mwd"),
    )
    _write_netcdf_member(
        atmosphere_member,
        request=request,
        variable_names=("u10", "v10", "msl", "t2m", "d2m", "land_sea_mask"),
    )
    response = tmp_path / "zip-wrapped-response.nc"
    with ZipFile(response, "w", compression=ZIP_STORED) as archive:
        archive.write(wave_member, arcname="wave.nc")
        archive.write(atmosphere_member, arcname="atmosphere.nc")

    times, latitudes, longitudes, fields = load_netcdf_cube(response, expected_request=request)

    assert times.equals(pd.date_range(request.start_utc, request.end_utc, freq="h"))
    assert latitudes.shape == longitudes.shape == (3,)
    assert set(fields) == set(VARIABLES)
    assert not list(tmp_path.glob(".era5_members_*"))


def test_nearest_valid_ocean_cell_rejects_land_and_incomplete_cells() -> None:
    shape = (24, 3, 3)
    fields = {name: np.ones(shape, dtype=float) for name in VARIABLES}
    fields["land_sea_mask"][:] = 1.0
    fields["land_sea_mask"][:, 2, 2] = 0.0
    fields["swh"][:4, 2, 2] = np.nan  # 20/24 finite is below the fixed 90% threshold.
    fields["land_sea_mask"][:, 0, 0] = 0.0

    selected = select_nearest_valid_ocean_cell(
        "G-ORS",
        latitudes=[34.25, 34.00, 33.75],
        longitudes=[124.25, 124.50, 124.75],
        fields=fields,
    )

    assert (selected.latitude, selected.longitude) == (34.25, 124.25)
    assert selected.mean_land_sea_mask == 0.0
    assert min(selected.finite_fraction.values()) == 1.0


def test_no_valid_ocean_cell_fails_closed() -> None:
    fields = {name: np.ones((24, 3, 3), dtype=float) for name in VARIABLES}
    with pytest.raises(Era5SchemaError, match="no valid ocean cell"):
        select_nearest_valid_ocean_cell(
            "I-ORS",
            latitudes=[32.25, 32.00, 31.75],
            longitudes=[125.00, 125.25, 125.50],
            fields=fields,
        )


def test_selected_cells_expand_to_cutoff_safe_single_cell_year_chunks() -> None:
    selections = {station: _selection(station) for station in STATIONS}
    plan = build_year_plan(selections)
    assert len(plan) == 363
    assert len({value.request_id for value in plan}) == 363
    assert {value.year for value in plan} == set(range(2014, 2024))
    assert {value.station for value in plan} == set(STATIONS)
    assert max(value.end_utc for value in plan) == pd.Timestamp("2023-12-31T14:00:00Z")
    assert max(value.end_utc for value in plan) < CUTOFF_EXCLUSIVE_UTC
    assert LAST_ELIGIBLE_ANCHOR_UTC == pd.Timestamp("2023-12-30T14:00:00Z")
    field_hour_costs = []
    for value in plan:
        north, west, south, east = value.request()["area"]
        assert north == south
        assert west == east
        duration_hours = int((value.end_utc - value.start_utc) / pd.Timedelta(hours=1)) + 1
        field_hour_costs.append(duration_hours * len(VARIABLES))
        assert field_hour_costs[-1] <= MAX_YEAR_REQUEST_FIELD_HOURS
        assert len(value.months) == 1
        assert value.request()["month"] == [f"{value.months[0]:02d}"]
        if value.year < 2023 or value.months[0] < 12:
            assert value.request()["day"] == [f"{day:02d}" for day in range(1, 32)]
    assert max(field_hour_costs) == MAX_YEAR_REQUEST_FIELD_HOURS
    for station in STATIONS:
        station_plan = [value for value in plan if value.station == station]
        assert len(station_plan) == 121
        assert station_plan[0].request_id.endswith("2014_m01")
        assert station_plan[107].request_id.endswith("2022_m12")
        assert station_plan[108].request_id.endswith("2023_m01")
        assert station_plan[118].request_id.endswith("2023_m11")
        assert station_plan[119].request_id.endswith("2023_dec01_30")
        assert station_plan[-1].request_id.endswith("2023_dec31_h00_14")
        assert station_plan[0].start_utc == pd.Timestamp("2014-01-01T00:00:00Z")
        assert station_plan[-1].end_utc == pd.Timestamp("2023-12-31T14:00:00Z")
        for previous, following in zip(station_plan[:-1], station_plan[1:], strict=True):
            assert following.start_utc == previous.end_utc + pd.Timedelta(hours=1)
    tail = [value for value in plan if value.request_id.endswith("dec31_h00_14")]
    assert len(tail) == 3
    assert all(value.request()["time"][-1] == "14:00" for value in tail)


def test_download_is_blocked_before_client_creation_without_explicit_flag(
    tmp_path: Path,
) -> None:
    layout = QuarantineLayout.from_repo_root(tmp_path)
    request = build_smoke_plan()[0]
    calls = 0

    def forbidden_factory():
        nonlocal calls
        calls += 1
        raise AssertionError("client must not be constructed")

    with pytest.raises(DownloadAuthorizationError, match="execute-download"):
        retrieve_cds_request(
            request,
            target=layout.raw_path(request),
            layout=layout,
            execute_download=False,
            client_factory=forbidden_factory,
        )
    assert calls == 0
    assert not layout.root.exists()


def test_explicit_fake_download_is_atomic_and_receipted_inside_quarantine(
    tmp_path: Path,
) -> None:
    layout = QuarantineLayout.from_repo_root(tmp_path)
    request = build_smoke_plan()[0]

    class FakeClient:
        def retrieve(self, dataset: str, payload: dict[str, object], target: str) -> None:
            assert dataset == "reanalysis-era5-single-levels"
            assert payload == request.request()
            Path(target).write_bytes(b"fixture-netcdf")

    receipt = retrieve_cds_request(
        request,
        target=layout.raw_path(request),
        layout=layout,
        execute_download=True,
        client_factory=FakeClient,
    )

    assert receipt.bytes == len(b"fixture-netcdf")
    assert len(receipt.sha256) == 64
    assert (layout.root / receipt.relative_path).read_bytes() == b"fixture-netcdf"
    assert not list(layout.raw_smoke.glob("*.partial"))
    with pytest.raises(PermissionError, match="quarantine"):
        layout.assert_inside(tmp_path / "outside.nc")


@pytest.mark.parametrize(
    ("configured_url", "expected_url"),
    [
        (None, "https://cds.climate.copernicus.eu/api"),
        ("https://example.invalid/custom-cds", "https://example.invalid/custom-cds"),
    ],
)
def test_default_client_uses_environment_credential_without_printing_or_storing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    configured_url: str | None,
    expected_url: str,
) -> None:
    secret = "uid:regression-secret-value"
    client_kwargs: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            client_kwargs.append(kwargs)

        def retrieve(self, dataset: str, payload: dict[str, object], target: str) -> None:
            assert dataset == "reanalysis-era5-single-levels"
            assert payload == request.request()
            Path(target).write_bytes(b"environment-client-fixture")

    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=FakeClient))
    monkeypatch.setenv("CDSAPI_KEY", secret)
    if configured_url is None:
        monkeypatch.delenv("CDSAPI_URL", raising=False)
    else:
        monkeypatch.setenv("CDSAPI_URL", configured_url)
    layout = QuarantineLayout.from_repo_root(tmp_path)
    request = build_smoke_plan()[0]

    receipt = retrieve_cds_request(
        request,
        target=layout.raw_path(request),
        layout=layout,
        execute_download=True,
    )

    assert client_kwargs == [{"url": expected_url, "key": secret, "quiet": True, "debug": False}]
    assert receipt.bytes == len(b"environment-client-fixture")
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert secret.encode() not in (layout.root / receipt.relative_path).read_bytes()


def test_default_client_without_environment_key_uses_cds_config_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_kwargs: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            client_kwargs.append(kwargs)

    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=FakeClient))
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.setenv("CDSAPI_URL", "http://ignored-without-an-environment-key.invalid")

    client = era5_pretrain._default_cds_client()

    assert isinstance(client, FakeClient)
    assert client_kwargs == [{"quiet": True, "debug": False}]


def test_non_https_environment_url_fails_before_client_creation_without_key_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "uid:must-not-appear"
    calls = 0

    class ForbiddenClient:
        def __init__(self, **kwargs: object) -> None:
            nonlocal calls
            calls += 1

    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=ForbiddenClient))
    monkeypatch.setenv("CDSAPI_KEY", secret)
    monkeypatch.setenv("CDSAPI_URL", "http://cds.example.invalid/api")

    with pytest.raises(Era5PretrainError, match="credential-free HTTPS") as caught:
        era5_pretrain._default_cds_client()
    assert calls == 0
    assert secret not in str(caught.value)


def test_default_client_initialization_failure_does_not_leak_environment_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "uid:constructor-failure-secret"

    class FailingClient:
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError(f"upstream accidentally echoed {kwargs['key']}")

    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=FailingClient))
    monkeypatch.setenv("CDSAPI_KEY", secret)
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    layout = QuarantineLayout.from_repo_root(tmp_path)
    request = build_smoke_plan()[0]

    with pytest.raises(Era5PretrainError, match="initialization failed") as caught:
        retrieve_cds_request(
            request,
            target=layout.raw_path(request),
            layout=layout,
            execute_download=True,
        )

    captured = capsys.readouterr()
    assert secret not in str(caught.value)
    assert secret not in captured.out + captured.err
    assert not layout.raw_path(request).exists()


def test_selected_cell_processing_converts_units_and_rejects_2024() -> None:
    selection = _selection("S-ORS")
    times = pd.date_range("2023-12-30T21:00:00Z", periods=3, freq="h")
    values = {name: np.ones(3, dtype=float) for name in VARIABLES}
    values.update(
        {
            "u10": np.full(3, 3.0),
            "v10": np.full(3, 4.0),
            "msl": np.full(3, 101325.0),
            "t2m": np.full(3, 293.15),
            "d2m": np.full(3, 283.15),
            "mwd": np.full(3, 370.0),
            "land_sea_mask": np.zeros(3),
        }
    )
    frame = derive_selected_cell_frame(
        station="S-ORS",
        selection=selection,
        times_utc=times,
        values=values,
    )
    assert frame["wspd10_m_s"].tolist() == [5.0, 5.0, 5.0]
    assert frame["msl_hpa"].tolist() == [1013.25, 1013.25, 1013.25]
    assert frame["t2m_c"].tolist() == pytest.approx([20.0, 20.0, 20.0])
    assert frame["mwd_deg"].tolist() == [10.0, 10.0, 10.0]
    assert frame["relh2m_pct"].between(0.0, 100.0).all()

    with pytest.raises(Era5SchemaError, match="pre-2024"):
        derive_selected_cell_frame(
            station="S-ORS",
            selection=selection,
            times_utc=["2024-01-01T00:00:00Z"],
            values={name: np.ones(1) for name in VARIABLES},
        )


def test_existing_derived_reuse_revalidates_full_contract(tmp_path: Path) -> None:
    layout = QuarantineLayout.from_repo_root(tmp_path)
    layout.ensure()
    selections = {station: _selection(station) for station in STATIONS}
    request = next(
        value
        for value in build_year_plan(selections)
        if value.request_id == "year_g_ors_2023_dec31_h00_14"
    )
    selection = selections[request.station]
    valid = _derived_frame(request, selection)
    output = layout.derived_year_path(request)
    valid.to_parquet(output, index=False)

    loaded, receipt = load_validated_derived_year_file(
        output,
        request=request,
        selection=selection,
        layout=layout,
    )
    assert len(loaded) == 15
    assert receipt.row_count == 15

    wrong_station = valid.copy()
    wrong_station["station"] = "I-ORS"
    wrong_time = valid.copy()
    wrong_time.loc[0, "time_utc"] += pd.Timedelta(minutes=30)
    wrong_coordinates = valid.copy()
    wrong_coordinates["longitude"] += 0.25
    crossed_cutoff = valid.copy()
    crossed_cutoff.loc[crossed_cutoff.index[-1], "time_utc"] = CUTOFF_EXCLUSIVE_UTC
    negative_wave = valid.copy()
    negative_wave.loc[0, "swh_m"] = -0.01
    zero_period = valid.copy()
    zero_period.loc[0, "mwp_s"] = 0.0
    wrapped_direction = valid.copy()
    wrapped_direction.loc[0, "mwd_deg"] = 360.0
    invalid_mask = valid.copy()
    invalid_mask.loc[0, "land_sea_mask"] = 1.01
    invalid_humidity = valid.copy()
    invalid_humidity.loc[0, "relh2m_pct"] = 100.01
    extreme_pressure = valid.copy()
    extreme_pressure.loc[0, "msl_hpa"] = 1500.0
    extreme_temperature = valid.copy()
    extreme_temperature.loc[0, "t2m_c"] = 100.0
    infinite_wind = valid.copy()
    infinite_wind.loc[0, "u10_m_s"] = np.inf
    corruptions = {
        "schema": valid.drop(columns="swh_m"),
        "station": wrong_station,
        "request time": wrong_time,
        "selected coordinates": wrong_coordinates,
        "cutoff": crossed_cutoff,
        "negative wave": negative_wave,
        "zero period": zero_period,
        "direction endpoint": wrapped_direction,
        "land-sea mask": invalid_mask,
        "relative humidity": invalid_humidity,
        "pressure": extreme_pressure,
        "temperature": extreme_temperature,
        "infinite wind": infinite_wind,
    }
    for corrupted in corruptions.values():
        corrupted.to_parquet(output, index=False)
        with pytest.raises(FileExistsError, match="collision; refusing overwrite"):
            load_validated_derived_year_file(
                output,
                request=request,
                selection=selection,
                layout=layout,
            )

    legitimate_missing = valid.copy()
    legitimate_missing.loc[0, "swh_m"] = np.nan
    legitimate_missing.to_parquet(output, index=False)
    missing_frame, _ = load_validated_derived_year_file(
        output,
        request=request,
        selection=selection,
        layout=layout,
    )
    assert pd.isna(missing_frame.loc[0, "swh_m"])


def test_changed_smoke_selection_cannot_overwrite_existing_derived(
    tmp_path: Path,
) -> None:
    layout = QuarantineLayout.from_repo_root(tmp_path)
    layout.ensure()
    selections = {station: _selection(station) for station in STATIONS}
    selected_path = write_selected_cells(layout, selections)
    (layout.derived_years / "existing.parquet").write_bytes(b"derived-sentinel")
    changed = dict(selections)
    changed["G-ORS"] = replace(
        selections["G-ORS"],
        latitude=selections["G-ORS"].latitude + 0.25,
        distance_km=1.0,
    )

    with pytest.raises(FileExistsError, match="collision.*refusing"):
        write_selected_cells(layout, changed)

    assert read_selected_cells(layout) == selections
    assert selected_path.is_file()


def test_existing_combined_reuse_revalidates_full_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = QuarantineLayout.from_repo_root(tmp_path)
    layout.ensure()
    selections = {station: _selection(station) for station in STATIONS}
    requests = build_year_plan(selections)
    short_cutoff = pd.Timestamp("2014-01-01T03:00:00Z")
    monkeypatch.setattr(era5_pretrain, "CUTOFF_EXCLUSIVE_UTC", short_cutoff)
    times = pd.date_range("2014-01-01T00:00:00Z", periods=3, freq="h")
    frames = []
    for station, selection in selections.items():
        values = {name: np.ones(3, dtype=float) for name in VARIABLES}
        values.update(
            {
                "msl": np.full(3, 101325.0),
                "t2m": np.full(3, 283.15),
                "d2m": np.full(3, 278.15),
                "land_sea_mask": np.zeros(3, dtype=float),
            }
        )
        frames.append(
            derive_selected_cell_frame(
                station=station,
                selection=selection,
                times_utc=times,
                values=values,
            )
        )
    valid = pd.concat(frames, ignore_index=True)
    output = layout.derived / COMBINED_FILE_NAME
    valid.to_parquet(output, index=False)

    loaded, loaded_summary = load_validated_combined_file(layout, selections)
    assert len(loaded) == loaded_summary["row_count"] == 9
    assert loaded_summary["station_count"] == 3
    assert loaded_summary["rows_per_station"] == 3

    _, receipt, summary = combine_derived_year_files(
        layout=layout,
        requests=requests,
        selections=selections,
    )
    assert receipt.row_count == summary["row_count"] == 9

    wrong_station = valid.copy()
    wrong_station.loc[0, "station"] = "unknown"
    wrong_time = valid.copy()
    wrong_time.loc[0, "time_utc"] += pd.Timedelta(minutes=30)
    wrong_coordinates = valid.copy()
    wrong_coordinates.loc[0, "latitude"] += 0.25
    crossed_cutoff = valid.copy()
    crossed_cutoff.loc[0, "time_utc"] = short_cutoff
    impossible_wave = valid.copy()
    impossible_wave.loc[0, "hmax_m"] = -1.0
    corruptions = (
        valid.drop(columns="swh_m"),
        wrong_station,
        wrong_time,
        wrong_coordinates,
        crossed_cutoff,
        impossible_wave,
    )
    for corrupted in corruptions:
        corrupted.to_parquet(output, index=False)
        with pytest.raises(FileExistsError, match="collision; refusing reuse"):
            combine_derived_year_files(
                layout=layout,
                requests=requests,
                selections=selections,
            )
        with pytest.raises(FileExistsError, match="collision; refusing reuse"):
            load_validated_combined_file(layout, selections)


def test_manifest_records_source_requests_checksums_time_and_transform(
    tmp_path: Path,
) -> None:
    layout = QuarantineLayout.from_repo_root(tmp_path)
    request = build_smoke_plan()[0]

    class FakeClient:
        def retrieve(self, dataset: str, payload: dict[str, object], target: str) -> None:
            Path(target).write_bytes(b"manifest-fixture")

    receipt = retrieve_cds_request(
        request,
        target=layout.raw_path(request),
        layout=layout,
        execute_download=True,
        client_factory=FakeClient,
    )
    manifest = build_manifest(
        stage="smoke",
        smoke_requests=build_smoke_plan(),
        files=[receipt],
        network_action_taken=True,
    )

    assert manifest["source"]["url"] == DATASET_URL
    assert manifest["source"]["doi"] == DATASET_DOI
    assert manifest["source"]["license"] == LICENSE_NAME
    assert manifest["requests"]["smoke_3x3_one_day"][0]["request"]
    assert manifest["checksums_sha256"][receipt.relative_path] == receipt.sha256
    assert manifest["time_coverage"][0]["end_utc"] < "2024-01-01"
    assert manifest["boundary"]["cutoff_exclusive_utc"] == "2023-12-31T15:00:00+00:00"
    assert manifest["boundary"]["maximum_valid_time_utc"] == "2023-12-31T14:00:00+00:00"
    assert isinstance(manifest["transformation_log"], str)
    assert len(manifest["detail"]["transformation_steps"]) >= 5
    assert manifest["official_test_or_submission_accessed"] is False
    assert manifest["schema_version"] == "1.0"
    assert manifest["source_id"] == "era5_pre2024"
    assert manifest["local_file"] is None
    assert manifest["file_sha256"] is None
    assert manifest["observed_start"] is None
    assert manifest["observed_end"] is None
    assert manifest["row_count"] == 0
    assert manifest["variables"] == list(VARIABLES)
    assert manifest["stage"] == "smoke"


def test_final_manifest_local_file_is_repo_relative_and_generic_preflight_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = QuarantineLayout.from_repo_root(tmp_path)
    layout.ensure()
    combined_path = layout.derived / COMBINED_FILE_NAME
    combined_path.write_bytes(b"combined-fixture")
    receipt = FileReceipt(
        request_id="combined_era5_p3_2014_2023",
        role="final_combined_selected_cell_hourly_parquet",
        relative_path=combined_path.relative_to(layout.root).as_posix(),
        bytes=combined_path.stat().st_size,
        sha256=sha256_file(combined_path),
        time_start_utc="2014-01-01T00:00:00+00:00",
        time_end_utc="2023-12-31T14:00:00+00:00",
        row_count=262_917,
    )
    selections = {station: _selection(station) for station in STATIONS}
    year_requests = build_year_plan(selections)
    manifest = build_manifest(
        stage="combine",
        smoke_requests=build_smoke_plan(),
        year_requests=year_requests,
        selections=selections,
        files=[receipt],
        network_action_taken=False,
    )
    monkeypatch.chdir(tmp_path)
    assert (
        manifest["local_file"] == (QUARANTINE_RELATIVE / "derived" / COMBINED_FILE_NAME).as_posix()
    )
    assert Path(manifest["local_file"]).is_file()
    assert manifest["file_sha256"] == sha256_file(combined_path)
    assert manifest["observed_end"] == "2023-12-31T14:00:00+00:00"
    assert manifest["row_count"] == 262_917
    assert isinstance(manifest["transformation_log"], str)
    assert len(manifest["requests"]["selected_single_cell_years"]) == 363
    assert "Combine 363 station-time segments" in manifest["transformation_log"]
    manifest_path = write_manifest(layout, manifest, stage="combine")
    assert manifest_path == layout.manifests / "manifest.json"
    assert not manifest_path.with_suffix(".json.partial").exists()
    assert validate_existing_canonical_manifest(
        layout,
        combined_receipt=receipt,
    )
    combined_path.write_bytes(b"tampered-combined-fixture")
    with pytest.raises(FileExistsError, match="manifest collision; refusing reuse"):
        validate_existing_canonical_manifest(
            layout,
            combined_receipt=receipt,
        )


def test_plan_and_smoke_receipts_cannot_clobber_canonical_manifest(
    tmp_path: Path,
) -> None:
    layout = QuarantineLayout.from_repo_root(tmp_path)
    layout.ensure()
    canonical = layout.manifests / "manifest.json"
    canonical.write_bytes(b"completed-canonical-sentinel")

    for stage, expected_name in (
        ("plan", "plan_receipt.json"),
        ("smoke", "smoke_receipt.json"),
    ):
        payload = build_manifest(
            stage=stage,
            smoke_requests=build_smoke_plan(),
            network_action_taken=False,
        )
        receipt_path = write_manifest(layout, payload, stage=stage)
        assert receipt_path.name == expected_name
        assert canonical.read_bytes() == b"completed-canonical-sentinel"

    with pytest.raises(ValueError, match="stage/filename"):
        write_manifest(layout, payload, stage="years")


def test_year_runner_parallelizes_only_raw_transport_with_deterministic_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    layout = QuarantineLayout.from_repo_root(tmp_path)
    layout.ensure()
    selections = {station: _selection(station) for station in STATIONS}
    requests = build_year_plan(selections)[:8]
    lock = Lock()
    all_workers_entered = Event()
    main_thread = get_ident()
    active = 0
    maximum_active = 0
    raw_completed = 0
    processed: list[str] = []

    def receipt(request: Era5Request, role: str, relative_path: str) -> FileReceipt:
        return FileReceipt(
            request_id=request.request_id,
            role=role,
            relative_path=relative_path,
            bytes=1,
            sha256="a" * 64,
            time_start_utc=request.start_utc.isoformat(),
            time_end_utc=request.end_utc.isoformat(),
            row_count=None,
        )

    def fake_existing_or_download(
        request: Era5Request,
        *,
        layout: QuarantineLayout,
        execute_download: bool,
        client_factory: object,
    ) -> tuple[FileReceipt, bool]:
        nonlocal active, maximum_active, raw_completed
        assert execute_download
        assert layout.root.is_relative_to(tmp_path)
        assert client_factory is None
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 4:
                all_workers_entered.set()
        assert all_workers_entered.wait(timeout=5)
        with lock:
            active -= 1
            raw_completed += 1
        return (
            receipt(
                request,
                "raw_cds_netcdf",
                f"raw/yearly/{request.request_id}.nc",
            ),
            True,
        )

    def fake_process_year_file(
        raw_path: Path,
        *,
        request: Era5Request,
        selection: SelectedCell,
        output_path: Path,
        layout: QuarantineLayout,
    ) -> FileReceipt:
        assert get_ident() == main_thread
        assert raw_completed == len(requests)
        assert request.station == selection.station
        assert raw_path == layout.raw_path(request)
        assert output_path == layout.derived_year_path(request)
        processed.append(request.request_id)
        return receipt(
            request,
            "derived_selected_cell_hourly_parquet",
            f"derived/yearly/{request.request_id}.parquet",
        )

    def fake_combine(**kwargs: object) -> tuple[Path, FileReceipt, dict[str, object]]:
        assert kwargs["requests"] == requests
        combined_receipt = FileReceipt(
            request_id="combined_era5_p3_2014_2023",
            role="final_combined_selected_cell_hourly_parquet",
            relative_path=f"derived/{COMBINED_FILE_NAME}",
            bytes=1,
            sha256="b" * 64,
            time_start_utc="2014-01-01T00:00:00+00:00",
            time_end_utc="2023-12-31T14:00:00+00:00",
            row_count=262_917,
        )
        return layout.derived / COMBINED_FILE_NAME, combined_receipt, {}

    monkeypatch.setattr(runner, "read_selected_cells", lambda _: selections)
    monkeypatch.setattr(runner, "build_year_plan", lambda _: requests)
    monkeypatch.setattr(runner, "_existing_or_download", fake_existing_or_download)
    monkeypatch.setattr(runner, "process_year_file", fake_process_year_file)
    monkeypatch.setattr(runner, "combine_derived_year_files", fake_combine)

    _, receipts, download_count = runner._run_years(
        layout=layout,
        execute_download=True,
        client_factory=None,
    )

    assert runner.YEAR_DOWNLOAD_WORKERS == maximum_active == 4
    assert download_count == len(requests)
    assert processed == [request.request_id for request in requests]
    assert [value.request_id for value in receipts[:-1]] == [
        request.request_id for request in requests for _ in range(2)
    ]
    assert [value.role for value in receipts[:-1:2]] == ["raw_cds_netcdf"] * len(requests)
    assert [value.role for value in receipts[1:-1:2]] == [
        "derived_selected_cell_hourly_parquet"
    ] * len(requests)


def test_parallel_raw_failure_cleans_partials_and_preserves_completed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    layout = QuarantineLayout.from_repo_root(tmp_path)
    layout.ensure()
    selections = {station: _selection(station) for station in STATIONS}
    requests = build_year_plan(selections)[:8]

    class FakeClient:
        def retrieve(self, dataset: str, payload: dict[str, object], target: str) -> None:
            assert dataset == "reanalysis-era5-single-levels"
            target_path = Path(target)
            target_path.write_bytes(b"completed-raw-fixture")
            if "2014_m03" in target_path.name:
                raise RuntimeError("synthetic transport failure")

    monkeypatch.setattr(runner, "read_selected_cells", lambda _: selections)
    monkeypatch.setattr(runner, "build_year_plan", lambda _: requests)

    with pytest.raises(Era5PretrainError, match="retrieval failed"):
        runner._run_years(
            layout=layout,
            execute_download=True,
            client_factory=FakeClient,
        )

    assert not list(layout.raw_years.glob("*.partial"))
    completed = sorted(layout.raw_years.glob("*.nc"))
    assert completed
    assert not list(layout.derived_years.glob("*.parquet"))
    reusable_request = next(
        request for request in requests if layout.raw_path(request) == completed[0]
    )
    factory_calls = 0

    def forbidden_factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("completed raw file must be reused")

    _, downloaded = runner._existing_or_download(
        reusable_request,
        layout=layout,
        execute_download=True,
        client_factory=forbidden_factory,
    )
    assert downloaded is False
    assert factory_calls == 0


def test_default_runner_is_network_free_and_writes_only_fixed_quarantine(
    tmp_path: Path,
) -> None:
    runner = _runner_module()
    calls = 0

    def forbidden_factory():
        nonlocal calls
        calls += 1
        raise AssertionError("plan stage must never construct a client")

    layout = QuarantineLayout.from_repo_root(tmp_path)
    layout.ensure()
    canonical = layout.manifests / "manifest.json"
    canonical.write_bytes(b"completed-canonical-sentinel")

    result = runner.run(
        stage="plan",
        execute_download=False,
        repo_root=tmp_path,
        client_factory=forbidden_factory,
    )
    assert calls == 0
    assert result["network_action_taken"] is False
    assert result["smoke_request_count"] == 3
    assert result["year_request_count"] == 0
    assert result["manifest"].endswith("/manifests/plan_receipt.json")
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert files
    expected_root = (tmp_path / QUARANTINE_RELATIVE).resolve()
    assert all(path.resolve().is_relative_to(expected_root) for path in files)
    manifest_path = tmp_path / result["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["download_requires_explicit_execute_download"] is True
    assert manifest["stage"] == "plan"
    assert canonical.read_bytes() == b"completed-canonical-sentinel"
    assert runner._parser().parse_args(["--stage", "combine"]).stage == "combine"
