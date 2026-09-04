from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from p1_qc.synthetic_context_tcn import (
    SyntheticContextTCN,
    decode_long_components,
    fit_robust_scale,
    inject_synthetic_event,
    transform_robust,
    union_diagnostics,
    window_rows,
)


def test_robust_scale_is_finite_and_shape_preserving() -> None:
    values = np.asarray([[1.0, np.nan], [2.0, 4.0], [100.0, 4.0]], dtype=np.float32)
    state = fit_robust_scale(values[:2])
    transformed = transform_robust(values, state)
    assert transformed.shape == values.shape
    assert np.isfinite(transformed).all()


def test_synthetic_event_has_registered_minimum_support() -> None:
    window = np.zeros((288, 6), dtype=np.float32)
    augmented, mask, kind = inject_synthetic_event(
        window,
        np.random.default_rng(7),
        event_min_rows=19,
        event_max_rows=96,
        primary_channels=(0, 1, 2),
        difference_channels=(3, 4),
        donor=np.ones_like(window),
    )
    assert augmented.shape == window.shape
    assert int(mask.sum()) >= 19
    assert kind in {"offset", "drift", "noise", "flatline", "coe"}


def test_decoder_keeps_long_components_and_bridges_tiny_gap() -> None:
    rows = 60
    keys = pd.DataFrame(
        {
            "station": ["S-ORS"] * rows,
            "year": [2025] * rows,
            "layer": [2] * rows,
            "time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"),
        }
    )
    scores = np.zeros(rows, dtype=np.float32)
    scores[10:20] = 0.9
    scores[22:31] = 0.9
    decoded = decode_long_components(
        scores,
        keys,
        np.arange(rows),
        threshold=0.5,
        minimum_rows=19,
        bridge_rows=2,
    )
    assert decoded[10:31].all()
    assert int(decoded.sum()) == 21


def test_union_diagnostics_never_removes_anchor_rows() -> None:
    truth = np.asarray([1, 0, 1, 0], dtype=np.int8)
    anchor = np.asarray([1, 0, 0, 0], dtype=np.int8)
    additions = np.asarray([0, 0, 1, 0], dtype=np.int8)
    metrics = union_diagnostics(truth, anchor, additions, np.arange(4))
    assert metrics["delta_f1"] > 0
    assert metrics["anchor_positive_removed_rows"] == 0


def test_tcn_emits_one_logit_per_row() -> None:
    model = SyntheticContextTCN(8, width=16, dilations=(1, 2), dropout=0.0)
    output = model(torch.zeros(3, 32, 8))
    assert output.shape == (3, 32)


def test_short_inference_segment_is_left_padded() -> None:
    segment = np.arange(7, dtype=np.int64)
    windows = window_rows([segment], 12, 4, pad_short=True)
    assert len(windows) == 1
    assert len(windows[0]) == 12
    assert np.array_equal(windows[0][-7:], segment)
