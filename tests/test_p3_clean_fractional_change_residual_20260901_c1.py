from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p3_wave.clean_fractional_change_residual_20260901_c1 import (
    blend_with_clean_fallback,
    evaluate_gate,
    fractional_target,
    restore_delta,
)


def test_fractional_target_round_trip() -> None:
    delta = np.asarray([-0.5, 0.0, 1.25])
    current = np.asarray([1.5, 2.0, 3.5])
    fraction = fractional_target(delta, current, offset_m=0.5)
    restored = restore_delta(fraction, current, offset_m=0.5)
    np.testing.assert_allclose(restored, delta, rtol=0.0, atol=1e-15)


def test_fractional_target_rejects_bad_contract() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        fractional_target(np.ones(2), np.ones(3), offset_m=0.5)
    with pytest.raises(ValueError, match="positive"):
        fractional_target(np.ones(2), np.ones(2), offset_m=0.0)


def test_fixed_blend() -> None:
    base = np.asarray([1.0, 2.0])
    challenger = np.asarray([3.0, 4.0])
    result = blend_with_clean_fallback(base, challenger, challenger_weight=0.25)
    np.testing.assert_allclose(result, [1.5, 2.5])


def test_gate_requires_pooled_and_transport_checks() -> None:
    rows = []
    for fold in ("a", "b", "c"):
        for station in ("G-ORS", "I-ORS", "S-ORS"):
            for lead in (3, 18, 24):
                rows.append(
                    {
                        "fold": fold,
                        "station": station,
                        "lead_h": lead,
                        "target_hs": 2.0,
                        "clean_fallback_prediction": 2.2,
                        "candidate_prediction": 2.1,
                    }
                )
    frame = pd.DataFrame(rows)
    config = {
        "gate": {
            "minimum_improved_folds": 2,
            "maximum_station_degradation_m": 0.015,
            "maximum_long_lead_degradation_m": 0.015,
            "prediction_clip_m": [0.0, 30.0],
        }
    }
    result = evaluate_gate(frame, config)
    assert result["passed"] is True
    assert result["checks"]["strict_pooled_improvement"] is True


def test_gate_rejects_non_improving_candidate() -> None:
    frame = pd.DataFrame(
        {
            "fold": ["a", "b", "c"],
            "station": ["G-ORS", "I-ORS", "S-ORS"],
            "lead_h": [18, 18, 24],
            "target_hs": [2.0, 2.0, 2.0],
            "clean_fallback_prediction": [2.1, 2.1, 2.1],
            "candidate_prediction": [2.2, 2.2, 2.2],
        }
    )
    config = {
        "gate": {
            "minimum_improved_folds": 2,
            "maximum_station_degradation_m": 0.015,
            "maximum_long_lead_degradation_m": 0.015,
            "prediction_clip_m": [0.0, 30.0],
        }
    }
    assert evaluate_gate(frame, config)["passed"] is False
