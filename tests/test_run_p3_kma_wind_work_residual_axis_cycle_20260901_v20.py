from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_kma_wind_work_residual_axis_cycle_20260901_v20.py"
CONFIG = ROOT / "configs/experiments/p3_kma_wind_work_residual_axis_cycle_20260901_v20.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("p3_v20", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_sealed_and_official_access_is_zero() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "SEALED_BEFORE_OUTER_SCORING"
    assert config["candidate"]["grid_or_search"] is False
    assert config["candidate"]["result_based_tuning"] is False
    assert config["validation"]["outer_model_fits"] == 6
    assert all(value == 0 for value in config["official_policy"].values())


def test_raw_transform_has_declared_log_clip_and_eligibility() -> None:
    runner = load_runner()
    frame = pd.DataFrame(
        {
            "wind_input_proxy_current": [-3.0, 3.0, np.nan],
            "wind_wave_alignment_current": [2.0, -2.0, np.nan],
            "wspd_change_6h": [np.nan, 1.0, np.nan],
            "caph_change_6h": [np.nan, 2.0, 4.0],
            "gust_excess_current": [np.nan, 3.0, np.nan],
        }
    )
    raw, eligible = runner.transformed_raw_features(frame)
    assert raw[0, 0] == 0.0
    assert raw[1, 0] == np.log1p(3.0)
    assert raw[0, 1] == 1.0
    assert raw[1, 1] == -1.0
    assert eligible.tolist() == [True, True, False]


def test_transform_is_finite_and_train_centered() -> None:
    runner = load_runner()
    raw = np.array(
        [
            [0.0, 1.0, np.nan, 2.0, 3.0],
            [1.0, np.nan, 0.0, 4.0, 5.0],
            [2.0, -1.0, 1.0, np.nan, 7.0],
        ],
        dtype=np.float64,
    )
    state = runner.fit_transformer(raw)
    transformed = runner.apply_transformer(raw, state)
    assert transformed.shape == (3, 10)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed.mean(axis=0), 0.0)


class _FixedModel:
    coef_ = np.array([1.0] * 10, dtype=np.float64)


def test_prediction_preserves_short_leads_and_bounds_alpha() -> None:
    runner = load_runner()
    frame = pd.DataFrame(
        {
            "lead_h": [3, 18, 24],
            "base": [1.0, 1.0, 1.0],
            "delta": [0.5, 0.5, 0.5],
            "v19_alpha": [0.425, 0.2, 0.6],
            "v19_prediction": [1.2125, 1.1, 1.3],
            "wind_input_proxy_current": [1.0, 1.0, np.nan],
            "wind_wave_alignment_current": [0.0, 0.0, np.nan],
            "wspd_change_6h": [0.0, 0.0, np.nan],
            "caph_change_6h": [0.0, 0.0, 1.0],
            "gust_excess_current": [0.0, 0.0, np.nan],
        }
    )
    raw, _ = runner.transformed_raw_features(frame)
    state = runner.fit_transformer(raw[:2])
    prediction, alpha, eligible = runner.predict_residual_axis(frame, _FixedModel(), state)
    assert prediction[0] == frame.loc[0, "v19_prediction"]
    assert alpha[0] == frame.loc[0, "v19_alpha"]
    assert eligible.tolist() == [True, True, False]
    assert alpha[2] == frame.loc[2, "v19_alpha"]
    assert np.all(alpha >= runner.ALPHA_MIN)
    assert np.all(alpha <= runner.ALPHA_MAX)


def test_runner_contains_no_official_or_csv_materialization_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    forbidden = ("official_frame(", "read_csv(", "to_csv(", "submission.csv", "load_hidden")
    assert all(token not in source for token in forbidden)
