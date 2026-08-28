import numpy as np
import pandas as pd

from scripts.run_p3_atmosphere_complete_residual_pilot_20260828_v1 import (
    LEADS,
    add_station_indicators,
    bootstrap_case_delta,
    evaluate_model,
    rmse,
)


def test_rmse() -> None:
    assert rmse(np.array([0.0, 1.0]), np.array([0.0, 2.0])) == np.sqrt(0.5)


def test_station_indicators_are_fixed_width() -> None:
    frame = pd.DataFrame({"station": ["G-ORS", "S-ORS"], "x": [1.0, 2.0]})
    result = add_station_indicators(frame, ["x"])
    assert result.columns.tolist() == ["x", "station__G-ORS", "station__I-ORS", "station__S-ORS"]
    assert result["station__I-ORS"].sum() == 0.0


def test_bootstrap_zero_delta_is_exact() -> None:
    error = np.arange(18, dtype=float).reshape(3, 6) / 10.0
    result = bootstrap_case_delta(error, error, replicates=200, confidence=0.90, seed=7)
    assert result["ci_lower"] == 0.0
    assert result["ci_upper"] == 0.0
    assert result["probability_improved"] == 0.0


def test_evaluate_model_passes_clear_improvement() -> None:
    rows = []
    stations = ["G-ORS", "I-ORS", "S-ORS"]
    folds = ["a", "b", "c"]
    for index, (station, fold) in enumerate(zip(stations, folds)):
        for lead in LEADS:
            rows.append(
                {
                    "anchor_id": index,
                    "station": station,
                    "fold": fold,
                    "lead_h": lead,
                    "target_hs": 1.0,
                    "champion_prediction": 1.2,
                    "candidate_prediction": 1.0,
                }
            )
    result = evaluate_model(
        pd.DataFrame(rows),
        {"replicates": 200, "confidence": 0.90, "seed": 9},
        {
            "overall_delta_below": 0.0,
            "bootstrap_ci90_upper_below": 0.0,
            "minimum_non_degrading_stations": 2,
            "maximum_station_degradation_rmse_m": 0.01,
            "minimum_non_degrading_folds": 2,
            "maximum_fold_degradation_rmse_m": 0.02,
            "long_lead_delta_below": 0.0,
        },
    )
    assert result["delta_rmse"] < 0.0
    assert result["promotion_gate_pass"] is True
