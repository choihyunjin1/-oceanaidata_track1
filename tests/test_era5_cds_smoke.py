from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest

from p2_restore.era5_cds import load_cds_chunk_frame, validate_cds_smoke
from p2_restore.era5_request import ERA5_VARIABLES, build_smoke_chunk

netcdf4 = pytest.importorskip("netCDF4")

VARIABLES = {
    "instant": {
        "u10": ("m s**-1", "unknown", "10 metre U wind component", 1.0),
        "v10": ("m s**-1", "unknown", "10 metre V wind component", -1.0),
        "lsm": ("(0 - 1)", "land_binary_mask", "Land-sea mask", 0.0),
    },
    "accum": {
        "ewss": (
            "N m**-2 s",
            "surface_downward_eastward_stress",
            "Time-integrated eastward turbulent surface stress",
            100.0,
        ),
        "nsss": (
            "N m**-2 s",
            "surface_downward_northward_stress",
            "Time-integrated northward turbulent surface stress",
            -100.0,
        ),
        "ssr": (
            "J m**-2",
            "surface_net_downward_shortwave_flux",
            "Surface net short-wave (solar) radiation",
            1000.0,
        ),
        "str": (
            "J m**-2",
            "surface_net_upward_longwave_flux",
            "Surface net long-wave (thermal) radiation",
            -500.0,
        ),
        "slhf": (
            "J m**-2",
            "surface_upward_latent_heat_flux",
            "Time-integrated surface latent heat net flux",
            -600.0,
        ),
        "sshf": (
            "J m**-2",
            "surface_upward_sensible_heat_flux",
            "Time-integrated surface sensible heat net flux",
            -100.0,
        ),
    },
}


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "validate_era5_cds_smoke.py"
    spec = importlib.util.spec_from_file_location("validate_era5_cds_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_member(
    path: Path,
    step_type: str,
    *,
    hour_offset: int = 0,
    nonfinite_variable: str | None = None,
) -> None:
    with netcdf4.Dataset(path, mode="w", format="NETCDF4") as dataset:
        dataset.createDimension("valid_time", 24)
        dataset.createDimension("latitude", 3)
        dataset.createDimension("longitude", 3)

        valid_time = dataset.createVariable("valid_time", "i8", ("valid_time",))
        valid_time.units = "seconds since 1970-01-01"
        valid_time.calendar = "proleptic_gregorian"
        valid_time.standard_name = "time"
        valid_time[:] = 1_725_148_800 + hour_offset * 3600 + np.arange(24) * 3600

        latitude = dataset.createVariable("latitude", "f8", ("latitude",))
        latitude.units = "degrees_north"
        latitude[:] = [37.75, 37.50, 37.25]
        longitude = dataset.createVariable("longitude", "f8", ("longitude",))
        longitude.units = "degrees_east"
        longitude[:] = [124.50, 124.75, 125.00]

        for name, (units, standard_name, long_name, value) in VARIABLES[step_type].items():
            variable = dataset.createVariable(
                name,
                "f4",
                ("valid_time", "latitude", "longitude"),
                fill_value=np.nan,
            )
            variable.units = units
            variable.standard_name = standard_name
            variable.long_name = long_name
            variable.GRIB_stepType = step_type
            values = np.full((24, 3, 3), value, dtype=np.float32)
            if name == nonfinite_variable:
                values[0, 0, 0] = np.nan
            variable[:] = values


def _archive(
    tmp_path: Path,
    *,
    nonfinite_variable: str | None = None,
    accum_hour_offset: int = 0,
    unsafe_name: bool = False,
) -> Path:
    instant = tmp_path / "instant-source.nc"
    accum = tmp_path / "accum-source.nc"
    _write_member(instant, "instant", nonfinite_variable=nonfinite_variable)
    _write_member(
        accum,
        "accum",
        hour_offset=accum_hour_offset,
        nonfinite_variable=nonfinite_variable,
    )
    target = tmp_path / "smoke.nc"
    with ZipFile(target, mode="w", compression=ZIP_STORED) as archive:
        archive.write(
            instant,
            arcname="../instant.nc" if unsafe_name else "data_stepType-instant.nc",
        )
        archive.write(accum, arcname="data_stepType-accum.nc")
    return target


def test_zip_wrapped_instant_and_accum_members_are_validated(tmp_path: Path) -> None:
    report = validate_cds_smoke(_archive(tmp_path), expected_chunk=build_smoke_chunk())
    public = report.public_dict()

    assert public["passed"] is True
    assert public["container_format"] == "zip"
    assert public["member_count"] == 2
    assert {member["step_type"] for member in public["members"]} == {"instant", "accum"}
    assert public["feature_variables"] == list(ERA5_VARIABLES)
    assert public["time_start_utc"] == "2024-09-01T00:00:00+00:00"
    assert public["time_end_utc"] == "2024-09-01T23:00:00+00:00"
    assert public["validation"]["grid_shape"] == [3, 3]
    assert set(public["finite_value_counts"].values()) == {216}


def test_validated_chunk_frame_has_unique_utc_grid_keys_and_exact_kst_alignment(
    tmp_path: Path,
) -> None:
    report, frame = load_cds_chunk_frame(
        _archive(tmp_path),
        expected_chunk=build_smoke_chunk(),
    )

    assert report.validation["time_count"] == 24
    assert len(frame) == 24 * 9
    assert not frame[["time_utc", "latitude", "longitude"]].duplicated().any()
    utc = frame["time_utc"].dt.tz_localize(None)
    kst = frame["time_kst"].dt.tz_localize(None)
    assert ((kst - utc).dt.total_seconds() == 9 * 3600).all()


def test_nonfinite_field_is_rejected(tmp_path: Path) -> None:
    path = _archive(tmp_path, nonfinite_variable="ssr")
    with pytest.raises(ValueError, match="contains non-finite"):
        validate_cds_smoke(path, expected_chunk=build_smoke_chunk())


def test_members_must_share_the_frozen_hour_axis(tmp_path: Path) -> None:
    path = _archive(tmp_path, accum_hour_offset=1)
    with pytest.raises(ValueError, match="different valid times"):
        validate_cds_smoke(path, expected_chunk=build_smoke_chunk())


def test_archive_member_paths_are_never_extracted_or_trusted(tmp_path: Path) -> None:
    path = _archive(tmp_path, unsafe_name=True)
    with pytest.raises(ValueError, match="unsafe ERA5 CDS archive member"):
        validate_cds_smoke(path, expected_chunk=build_smoke_chunk())


def test_offline_runner_preserves_raw_file_and_writes_aggregate_receipt(tmp_path: Path) -> None:
    source = _archive(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    before = source.read_bytes()

    result = _runner_module().run(source, receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert source.read_bytes() == before
    assert result["network_action_taken"] is False
    assert result["raw_file_modified"] is False
    assert receipt["validation_network_action_taken"] is False
    assert receipt["raw_file_modified"] is False
    assert receipt["smoke_validation"]["passed"] is True
    assert "finite_value_counts" in receipt["smoke_validation"]
