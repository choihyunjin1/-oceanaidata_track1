from __future__ import annotations

import numpy as np

from p1_qc.mstcn_partial_pooling_calibrator import (
    PartialPoolingState,
    fit_partial_pooling,
    predict_partial_pooling,
)


def test_partial_pooling_round_trip_and_group_offset() -> None:
    score = np.asarray([0.1, 0.2, 0.7, 0.8] * 30, dtype=np.float32)
    station = np.asarray(["A", "A", "A", "A", "B", "B", "B", "B"] * 15)
    layer = np.ones(len(score), dtype=np.int16)
    target = np.asarray([0, 0, 1, 1, 0, 1, 1, 1] * 15, dtype=np.int8)
    state = fit_partial_pooling(score, station, layer, target, regularization_c=1.0)
    restored = PartialPoolingState.from_dict(state.as_dict())
    probability = predict_partial_pooling(restored, score, station, layer)
    assert probability.shape == score.shape
    assert probability.dtype == np.float32
    assert np.isfinite(probability).all()
    # B has more positives at the same repeating score distribution.
    assert float(probability[station == "B"].mean()) > float(probability[station == "A"].mean())


def test_unseen_group_uses_global_fallback() -> None:
    score = np.asarray([0.1, 0.2, 0.8, 0.9] * 20, dtype=np.float32)
    station = np.asarray(["A"] * len(score))
    layer = np.ones(len(score), dtype=np.int16)
    target = np.asarray([0, 0, 1, 1] * 20, dtype=np.int8)
    state = fit_partial_pooling(score, station, layer, target)
    unseen = predict_partial_pooling(
        state,
        np.asarray([0.2, 0.8], dtype=np.float32),
        np.asarray(["Z", "Z"]),
        np.asarray([9, 9]),
    )
    assert unseen[1] > unseen[0]
