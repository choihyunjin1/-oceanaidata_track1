from __future__ import annotations

import numpy as np

from p3_wave.wind_wave_memory import MEMORY_FEATURES, summarize_wind_wave_memory_context


def test_aligned_wind_has_finite_positive_memory_features() -> None:
    raw = np.full((289, 10), np.nan, dtype=np.float64)
    raw[:, 0] = np.linspace(1.5, 2.2, 289)
    raw[:, 1] = 8.0
    raw[:, 3] = 45.0
    raw[:, 4] = 12.0
    raw[:, 6] = 45.0
    result = summarize_wind_wave_memory_context(raw)
    assert tuple(result) == MEMORY_FEATURES
    assert result["wwm_aligned_wind_power_ewma6"] > 0.0
    assert result["wwm_phase_speed_proxy_last_valid"] == 1.0


def test_direction_opposed_wind_is_clipped_to_zero_not_wrapped_median() -> None:
    raw = np.full((289, 10), np.nan, dtype=np.float64)
    raw[:, 0] = 2.0
    raw[:, 1] = 7.0
    raw[:, 3] = 0.0
    raw[:, 4] = 10.0
    raw[:, 6] = 180.0
    result = summarize_wind_wave_memory_context(raw)
    assert result["wwm_aligned_wind_power_ewma24"] == 0.0
    assert result["wwm_phase_speed_proxy_last"] == 0.0
