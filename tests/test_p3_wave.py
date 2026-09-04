from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from p3_wave.data import LEADS, select_independent_validation
from p3_wave.deep import build_model, derived_channels, fit_channel_statistics
from p3_wave.event_phase import summarize_event_phase
from p3_wave.features import summarize_context
from p3_wave.submission import build_submission, validate_submission


def _context() -> pd.DataFrame:
    step = np.arange(-2880, 1, 10)
    wave_slot = step % 20 == 0
    frame = pd.DataFrame(
        {
            "step_minute": step,
            "hs": np.where(wave_slot, 1.0 + np.linspace(0, 0.7, len(step)), np.nan),
            "tp": np.where(wave_slot, 6.0, np.nan),
            "hmax": np.where(wave_slot, 2.0, np.nan),
            "wvdir": np.where(wave_slot, np.where(step < -1440, 359.0, 1.0), np.nan),
            "wspd": np.linspace(4.0, 12.0, len(step)),
            "gust": np.linspace(5.0, 14.0, len(step)),
            "wdir": np.where(step < -1440, 358.0, 2.0),
            "airt": 12.0,
            "relh": 80.0,
            "caph": np.linspace(1020.0, 1005.0, len(step)),
        }
    )
    return frame


def test_context_features_are_case_local_and_direction_safe() -> None:
    first = summarize_context(_context())
    second_frame = _context()
    second_frame.insert(0, "absolute_time", pd.date_range("2030-01-01", periods=289, freq="10min"))
    second = summarize_context(second_frame)
    assert first == second
    assert first["hs_current"] == pytest.approx(1.7)
    assert abs(first["wvdir_sin_mean_48h"]) < 0.1
    assert first["wvdir_cos_mean_48h"] > 0.9
    assert first["hs_valid_48h"] == pytest.approx(145 / 289)


def test_context_length_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="289"):
        summarize_context(_context().iloc[:-1])


def test_independent_validation_uses_78_hour_gap_per_station() -> None:
    times = pd.date_range("2025-01-01", periods=12, freq="20h", tz="UTC")
    anchors = pd.DataFrame(
        {
            "anchor_id": np.arange(24),
            "station": ["G-ORS"] * 12 + ["I-ORS"] * 12,
            "anchor_time": list(times) * 2,
        }
    )
    chosen = select_independent_validation(
        anchors, start="2025-01-01", end="2025-02-01", gap_hours=78
    )
    result = anchors.set_index("anchor_id").loc[chosen]
    assert len(result) == 6
    for _, group in result.groupby("station"):
        assert group["anchor_time"].sort_values().diff().dropna().ge(pd.Timedelta(hours=78)).all()


def test_deep_models_return_six_residuals() -> None:
    raw = np.stack(
        [
            _context()[
                ["hs", "tp", "hmax", "wvdir", "wspd", "gust", "wdir", "airt", "relh", "caph"]
            ].to_numpy(dtype=np.float32),
            _context()[
                ["hs", "tp", "hmax", "wvdir", "wspd", "gust", "wdir", "airt", "relh", "caph"]
            ].to_numpy(dtype=np.float32),
        ]
    )
    derived = derived_channels(raw)
    assert derived.shape == (2, 289, 12)
    center, scale = fit_channel_statistics(raw)
    station = torch.tensor([0, 2])
    for architecture in ("gru", "tcn"):
        model = build_model(architecture, center, scale)
        output = model(torch.from_numpy(raw), station)
        assert output.shape == (2, len(LEADS))
        assert torch.isfinite(output).all()


def test_submission_contract_is_exact() -> None:
    index = pd.DataFrame(
        {
            "case_id": np.repeat([f"C{i:04d}" for i in range(1, 201)], 6),
            "station": np.repeat(["G-ORS"] * 70 + ["I-ORS"] * 70 + ["S-ORS"] * 60, 6),
            "lead_h": list(LEADS) * 200,
        }
    )
    submission = build_submission(index, np.full(1_200, 1.7))
    validate_submission(submission, index)
    with pytest.raises(ValueError, match="order"):
        validate_submission(submission.iloc[::-1].reset_index(drop=True), index)


def test_event_phase_exposes_threshold_age_and_peak_recency() -> None:
    raw = _context()[
        ["hs", "tp", "hmax", "wvdir", "wspd", "gust", "wdir", "airt", "relh", "caph"]
    ].to_numpy(dtype=np.float32)
    phase = summarize_event_phase(raw)
    assert phase["event_hs_run_above_1p5_hours"] > 0
    assert phase["event_hs_hours_since_peak_6h"] == 0
    assert phase["event_hs_drop_from_peak_6h"] == pytest.approx(0)
    assert 0 <= phase["event_hs_fft_power_6_12h"] <= 1
