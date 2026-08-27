from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from p2_restore.era5_arco import MAX_SMOKE_GIB, validate_arco_metadata
from p2_restore.era5_downloader import Era5DownloadBlocked, download_cds_chunk
from p2_restore.era5_manifest import assert_secret_free
from p2_restore.era5_preflight import (
    Era5Field,
    causal_align_utc_to_kst,
    credential_preflight,
    validate_era5_fields,
)
from p2_restore.era5_request import (
    ANCILLARY_VARIABLES,
    AREA_3X3,
    ERA5_VARIABLES,
    _exact_month_chunks,
    build_registered_chunk_plan,
    build_smoke_chunk,
)


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "run_p2_era5_primary_scaffold.py"
    spec = importlib.util.spec_from_file_location("run_p2_era5_primary_scaffold", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata_bytes() -> bytes:
    standards = {
        "10m_u_component_of_wind": "",
        "10m_v_component_of_wind": "",
        "eastward_turbulent_surface_stress": "surface_downward_eastward_stress",
        "northward_turbulent_surface_stress": "surface_downward_northward_stress",
        "surface_net_solar_radiation": "surface_net_downward_shortwave_flux",
        "surface_net_thermal_radiation": "surface_net_upward_longwave_flux",
        "surface_latent_heat_flux": "surface_upward_latent_heat_flux",
        "surface_sensible_heat_flux": "surface_upward_sensible_heat_flux",
        "land_sea_mask": "land_binary_mask",
    }
    metadata: dict[str, object] = {
        ".zattrs": {
            "valid_time_start": "1940-01-01",
            "valid_time_stop": "2026-04-30",
            "last_updated": "2026-08-21T00:00:00Z",
        },
        "time/.zarray": {"shape": [1_323_648]},
    }
    for name in (*ERA5_VARIABLES, *ANCILLARY_VARIABLES):
        metadata[f"{name}/.zarray"] = {
            "chunks": [1, 721, 1440],
            "dtype": "<f4",
        }
        metadata[f"{name}/.zattrs"] = {
            "_ARRAY_DIMENSIONS": ["time", "latitude", "longitude"],
            "standard_name": standards[name],
        }
    metadata["10m_u_component_of_wind/.zattrs"].update(
        {"short_name": "u10", "long_name": "10 metre U wind component"}
    )
    metadata["10m_v_component_of_wind/.zattrs"].update(
        {"short_name": "v10", "long_name": "10 metre V wind component"}
    )
    return json.dumps({"metadata": metadata}).encode()


def _valid_fields() -> dict[str, Era5Field]:
    shape = (24, 3, 3)
    standards = {
        "10m_u_component_of_wind": "",
        "10m_v_component_of_wind": "",
        "eastward_turbulent_surface_stress": "surface_downward_eastward_stress",
        "northward_turbulent_surface_stress": "surface_downward_northward_stress",
        "surface_net_solar_radiation": "surface_net_downward_shortwave_flux",
        "surface_net_thermal_radiation": "surface_net_upward_longwave_flux",
        "surface_latent_heat_flux": "surface_upward_latent_heat_flux",
        "surface_sensible_heat_flux": "surface_upward_sensible_heat_flux",
        "land_sea_mask": "land_binary_mask",
    }
    units = {
        "10m_u_component_of_wind": "m s**-1",
        "10m_v_component_of_wind": "m s**-1",
        "eastward_turbulent_surface_stress": "N m**-2 s",
        "northward_turbulent_surface_stress": "N m**-2 s",
        "surface_net_solar_radiation": "J m**-2",
        "surface_net_thermal_radiation": "J m**-2",
        "surface_latent_heat_flux": "J m**-2",
        "surface_sensible_heat_flux": "J m**-2",
        "land_sea_mask": "(0 - 1)",
    }
    values = {
        "10m_u_component_of_wind": np.full(shape, -3.0),
        "10m_v_component_of_wind": np.full(shape, 4.0),
        "eastward_turbulent_surface_stress": np.full(shape, -100.0),
        "northward_turbulent_surface_stress": np.full(shape, 120.0),
        "surface_net_solar_radiation": np.full(shape, 1000.0),
        "surface_net_thermal_radiation": np.full(shape, -500.0),
        "surface_latent_heat_flux": np.full(shape, -600.0),
        "surface_sensible_heat_flux": np.full(shape, -100.0),
        "land_sea_mask": np.zeros(shape),
    }
    return {
        name: Era5Field(values[name], units[name], standards[name])
        for name in (*ERA5_VARIABLES, *ANCILLARY_VARIABLES)
    }


def test_request_is_fixed_3x3_24h_and_grib_first() -> None:
    chunk = build_smoke_chunk()
    request = chunk.request("grib")
    assert request["area"] == list(AREA_3X3)
    assert request["time"] == [f"{hour:02d}:00" for hour in range(24)]
    assert request["variable"] == [*ERA5_VARIABLES, *ANCILLARY_VARIABLES]
    assert request["data_format"] == "grib"
    assert chunk.request("netcdf")["data_format"] == "netcdf"


def test_exact_chunks_do_not_expand_partial_boundary_days() -> None:
    start = pd.Timestamp("2024-08-31T15:00:00Z")
    end = pd.Timestamp("2024-09-02T14:50:00Z")
    chunks = _exact_month_chunks("fixture", start, end)
    observed = pd.DatetimeIndex(
        sorted(timestamp for chunk in chunks for timestamp in chunk.timestamps_utc())
    )
    expected = pd.date_range(start, end.floor("h"), freq="h")
    assert observed.equals(expected)
    assert len(observed) == 48


def test_registered_full_plan_is_17_chunks_and_4900_unique_hours_without_oof() -> None:
    chunks = build_registered_chunk_plan()
    observed = pd.DatetimeIndex(
        sorted(timestamp for chunk in chunks for timestamp in chunk.timestamps_utc())
    )

    assert len(chunks) == 17
    assert len(observed) == 4_900
    assert not observed.duplicated().any()


def test_missing_cds_settings_block_before_network(tmp_path: Path) -> None:
    state = credential_preflight({})
    public = state.public_dict()
    assert state.status == "awaiting_credential"
    assert public["token_present"] is False
    assert "CDSAPI_KEY" in public["required_setting_names"]
    with pytest.raises(Era5DownloadBlocked, match="credential"):
        download_cds_chunk(
            build_smoke_chunk(),
            tmp_path,
            execute_download=True,
            environment={},
        )
    assert list(tmp_path.iterdir()) == []


def test_receipt_secret_guard_rejects_names_and_values() -> None:
    assert_secret_free({"required_setting_names": ["CDSAPI_KEY"]}, ["actual-secret"])
    with pytest.raises(ValueError, match="secret-like key"):
        assert_secret_free({"authorization_header": "redacted"})
    with pytest.raises(ValueError, match="secret value"):
        assert_secret_free({"setting": "actual-secret"}, ["actual-secret"])


def test_arco_metadata_gate_checks_variables_and_read_size() -> None:
    sizes = {name: 1_000_000 for name in (*ERA5_VARIABLES, *ANCILLARY_VARIABLES)}
    report = validate_arco_metadata(
        _metadata_bytes(),
        object_sizes=sizes,
        metadata_generation="fixture",
    )
    assert report.passed
    assert report.variable_count == 8
    assert report.estimated_smoke_gib < MAX_SMOKE_GIB

    too_large = {name: 10_000_000 for name in (*ERA5_VARIABLES, *ANCILLARY_VARIABLES)}
    with pytest.raises(ValueError, match="0.75 GiB"):
        validate_arco_metadata(
            _metadata_bytes(),
            object_sizes=too_large,
            metadata_generation="fixture",
        )


def test_units_sign_metadata_land_mask_and_smoke_shape_are_validated() -> None:
    fields = _valid_fields()
    report = validate_era5_fields(
        fields,
        times_utc=pd.date_range("2024-09-01", periods=24, freq="h", tz="UTC"),
        latitudes=[37.75, 37.50, 37.25],
        longitudes=[124.50, 124.75, 125.00],
        require_24h_smoke=True,
    )
    assert report["passed"]
    assert report["center_land_fraction"] == 0.0

    bad = dict(fields)
    bad["land_sea_mask"] = Era5Field(
        np.ones((24, 3, 3)),
        "(0 - 1)",
        "land_binary_mask",
    )
    with pytest.raises(ValueError, match="not classified as ocean"):
        validate_era5_fields(
            bad,
            times_utc=pd.date_range("2024-09-01", periods=24, freq="h", tz="UTC"),
            latitudes=[37.75, 37.50, 37.25],
            longitudes=[124.50, 124.75, 125.00],
            require_24h_smoke=True,
        )


def test_utc_to_kst_alignment_is_strictly_causal() -> None:
    hourly = pd.DataFrame(
        {"u10": [1.0, 2.0]},
        index=pd.date_range("2024-09-01T00:00:00Z", periods=2, freq="h"),
    )
    keys = pd.DataFrame(
        {
            "time": [
                "2024-09-01T09:10:00+09:00",
                "2024-09-01T10:50:00+09:00",
            ]
        }
    )
    aligned = causal_align_utc_to_kst(hourly, keys)
    assert aligned["u10"].tolist() == [1.0, 2.0]
    assert aligned["source_lag_minutes"].tolist() == [10.0, 50.0]
    assert str(aligned["source_time_kst"].dt.tz) == "Asia/Seoul"
    assert (aligned["source_time_utc"] <= aligned["key_time_utc"]).all()


def test_full_anonymous_transfer_gate_stops_large_family() -> None:
    runner = _runner_module()
    metadata = SimpleNamespace(compressed_one_hour_bytes=24_759_485)
    chunks = (build_smoke_chunk(),) * 200
    gate = runner._anonymous_transfer_gate(metadata, chunks)
    assert gate["estimated_full_gib"] > 5.0
    assert gate["passed"] is False
