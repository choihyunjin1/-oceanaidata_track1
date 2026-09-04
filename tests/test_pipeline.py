from __future__ import annotations

import numpy as np
import pandas as pd

from p1_qc.config import load_config
from p1_qc.features import build_features
from p1_qc.pipeline import (
    TabularEncoder,
    _augmented_fit_data,
    apply_postprocess,
    tune_postprocess,
)


def _frame() -> pd.DataFrame:
    count = 30
    return pd.DataFrame(
        {
            "station": ["S-ORS"] * count,
            "year": [2025] * count,
            "layer": [1] * count,
            "time": pd.date_range(
                "2025-01-01", periods=count, freq="10min", tz="Asia/Seoul"
            ).astype(str),
            "temp": np.r_[np.arange(10), np.repeat(10.0, 6), np.arange(14)],
            "psal": [30.0] * count,
            "depth": [5.0] * count,
            "label": np.r_[np.zeros(10), np.ones(6), np.zeros(14)].astype(int),
            "anomaly_type": [""] * 10 + ["flatline"] * 6 + [""] * 14,
        }
    )


def test_tabular_encoder_handles_unknown_category() -> None:
    frame = _frame()
    bundle = build_features(frame, mode="causal")
    encoder = TabularEncoder().fit(bundle, np.arange(20))
    matrix = encoder.transform(bundle)
    assert matrix.shape == (len(frame), len(bundle.feature_columns))


def test_postprocess_tuning_preserves_plateau() -> None:
    frame = _frame()
    probability = np.full(len(frame), 0.01)
    plateau = np.r_[np.zeros(10), np.ones(6), np.zeros(14)].astype(bool)
    spike = np.zeros(len(frame), dtype=bool)
    config = load_config()
    parameters, prediction, diagnostics = tune_postprocess(
        frame,
        probability,
        frame["label"].to_numpy(),
        plateau,
        spike,
        config,
    )
    reapplied = apply_postprocess(frame, probability, plateau, spike, parameters)
    assert np.array_equal(prediction, reapplied)
    assert diagnostics["plateau"]["precision"] == 1.0


def test_augmentation_is_fold_local_and_source_is_unchanged() -> None:
    count = 800
    frame = pd.DataFrame(
        {
            "station": ["S-ORS"] * count,
            "year": [2025] * count,
            "layer": [1] * count,
            "time": pd.date_range(
                "2025-01-01", periods=count, freq="10min", tz="Asia/Seoul"
            ).astype(str),
            "temp": 10 + np.sin(np.arange(count) / 20),
            "psal": [30.0] * count,
            "depth": [5.0] * count,
            "label": np.zeros(count, dtype=np.int8),
            "anomaly_type": [""] * count,
        }
    )
    original = frame.copy(deep=True)
    config = load_config()
    bundle = build_features(frame, config=config)
    indices = np.arange(700)
    _, matrix, target, audit = _augmented_fit_data(
        frame, bundle, indices, config, seed=7, enabled=True
    )
    pd.testing.assert_frame_equal(frame, original)
    assert matrix.shape[0] == len(indices)
    assert target.sum() == audit["injected_rows"]
    assert audit["injected_rows"] > 0
