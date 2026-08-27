from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_restore.era5_mixing_gate import (
    ERA5_MIXING_FEATURES,
    WINDOW_HOURS,
    align_mixing_features_to_oof_keys,
    build_hourly_ocean_mixing_features,
    convex_two_expert_blend,
    validate_era5_source_frame,
    validate_preregistered_feature_contract,
)
from p2_restore.regime_gate import STATE_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/experiments/p2_era5_mixing_gate_v1.json"


def _synthetic_era5(hours: int = 200, cells: int = 2) -> pd.DataFrame:
    times = pd.date_range("2025-06-23T15:00:00Z", periods=hours, freq="h")
    rows: list[dict[str, object]] = []
    for hour, time in enumerate(times):
        for cell in range(cells):
            rows.append(
                {
                    "chunk_id": "synthetic_chunk",
                    "block": "2025_jul_aug",
                    "time_utc": time,
                    "time_kst": time.tz_convert("Asia/Seoul"),
                    "latitude": 37.25 + cell * 0.25,
                    "longitude": 124.5,
                    "10m_u_component_of_wind": 2.0 + hour + cell,
                    "10m_v_component_of_wind": -1.0 + 0.5 * cell,
                    "eastward_turbulent_surface_stress": 3600.0 * (1.0 + 0.01 * hour),
                    "northward_turbulent_surface_stress": 0.0,
                    "surface_net_solar_radiation": 3600.0 * (2.0 + hour),
                    "surface_net_thermal_radiation": -360.0,
                    "surface_latent_heat_flux": -180.0,
                    "surface_sensible_heat_flux": 90.0,
                    "land_sea_mask": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_preregistered_config_and_implementation_have_exact_feature_order() -> None:
    contract = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_preregistered_feature_contract(contract["era5_mixing_features"])
    assert tuple(contract["public_state_features"]) == STATE_FEATURES
    assert len(ERA5_MIXING_FEATURES) == 30
    assert tuple(contract["era5_source_contract"]["windows_hours"]) == WINDOW_HOURS
    assert contract["gate_model"]["parameter_grid"] == []
    assert contract["gate_model"]["parameter_trials"] == 0
    assert contract["authorization"]["inner_fit_or_label_access"] is False
    assert contract["authorization"]["outer_fit_or_label_access"] is False
    assert contract["authorization"]["model_read_or_write"] is False
    assert contract["authorization"]["test_read"] is False
    assert contract["authorization"]["submission_read_or_write"] is False


def test_source_validation_checks_utc_kst_and_duplicate_grain() -> None:
    source = _synthetic_era5(hours=8)
    audit = validate_era5_source_frame(
        source,
        expected_rows=16,
        expected_blocks=("2025_jul_aug",),
        expected_grid_points=2,
    )
    assert audit["rows"] == 16
    assert audit["unique_hour_count"] == 8
    assert audit["utc_to_kst_wall_clock_offset_minutes"] == 540

    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_era5_source_frame(duplicate)


def test_hourly_features_use_native_qnet_sum_and_reset_complete_windows() -> None:
    source = _synthetic_era5()
    features = build_hourly_ocean_mixing_features(
        source, expected_ocean_cells_per_hour=2
    )
    assert len(features) == 200
    assert tuple(features.columns[2:]) == ERA5_MIXING_FEATURES
    assert features.loc[0, "era5_u10_ms"] == pytest.approx(2.5)
    assert features.loc[0, "era5_tau_mag_nm2"] == pytest.approx(1.0)
    assert features.loc[0, "era5_tau_dir_cos"] == pytest.approx(1.0)
    assert features.loc[0, "era5_tau_dir_sin"] == pytest.approx(0.0)
    expected_qnet = (7200.0 - 360.0 - 180.0 + 90.0) / 3600.0
    assert features.loc[0, "era5_qnet_native_wm2"] == pytest.approx(expected_qnet)
    assert pd.isna(features.loc[166, "era5_qnet_energy_168h_jm2"])
    assert np.isfinite(features.loc[167, "era5_qnet_energy_168h_jm2"])
    assert features.loc[167, "era5_tau_mag_trend_168h_nm2_per_h"] == pytest.approx(
        0.01
    )
    assert features.loc[167, "era5_qnet_trend_168h_wm2_per_h"] == pytest.approx(1.0)


def test_causal_hour_alignment_has_full_10_minute_key_coverage() -> None:
    source = _synthetic_era5()
    hourly = build_hourly_ocean_mixing_features(source, expected_ocean_cells_per_hour=2)
    key_time = pd.Timestamp("2025-06-30T15:10:00Z")
    keys = pd.DataFrame(
        {
            "time": [key_time, key_time + pd.Timedelta(minutes=20)],
            "layer": [2, 4],
            "block": ["2025_jul_aug", "2025_jul_aug"],
        }
    )
    aligned, audit = align_mixing_features_to_oof_keys(hourly, keys)
    assert len(aligned) == 2
    assert audit["join_coverage"] == 1.0
    assert audit["minimum_alignment_age_minutes"] == 10.0
    assert audit["maximum_alignment_age_minutes"] == 30.0
    assert audit["future_era5_rows"] == 0
    assert np.isfinite(aligned.loc[:, ERA5_MIXING_FEATURES].to_numpy()).all()


def test_two_expert_prediction_is_convex_and_rejects_extrapolating_weight() -> None:
    deep = np.array([1.0, 3.0, 2.0])
    physical = np.array([3.0, 1.0, 2.5])
    prediction = convex_two_expert_blend(deep, physical, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(prediction, [1.0, 2.0, 2.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        convex_two_expert_blend(deep, physical, np.array([0.0, 1.01, 1.0]))
