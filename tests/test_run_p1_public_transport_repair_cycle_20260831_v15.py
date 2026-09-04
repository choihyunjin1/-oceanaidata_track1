from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v15 as cycle  # noqa: E402


def test_contract_has_exact_ten_features_and_two_fits() -> None:
    config = cycle.load_contract()
    assert config["features"]["names"] == list(cycle.SOURCE_FEATURES)
    assert len(config["features"]["names"]) == 10
    assert config["fit_budget"]["maximum"] == 2
    assert config["model"]["probability_threshold_inclusive"] == 0.9


def test_singleton_weight_only_for_bounded_training_positive() -> None:
    frame = pd.DataFrame(
        {
            "station": ["S"] * 5,
            "layer": [1] * 5,
            "time": pd.date_range("2025-01-01", periods=5, freq="10min", tz="UTC"),
        }
    )
    labels = np.array([0, 1, 0, 1, 1], dtype=np.int8)
    weights, count = cycle.prefix_singleton_weights(frame, labels, np.ones(5, dtype=bool))
    assert count == 1
    assert weights.tolist() == [1.0, 0.5, 1.0, 1.0, 1.0]


def test_additive_model_is_interaction_free_and_finite() -> None:
    rng = np.random.default_rng(9)
    values = rng.normal(size=(80, 10))
    values[0, 0] = np.nan
    labels = np.array([0, 1] * 40, dtype=np.int8)
    model = cycle.fit_additive(values, labels, np.ones(80), cycle.load_contract())
    probability = model.predict_probability(values[:7])
    assert probability.shape == (7,)
    assert np.isfinite(probability).all()
    assert model.classifier.coef_.shape[0] == 1
