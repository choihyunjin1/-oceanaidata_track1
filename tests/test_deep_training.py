from __future__ import annotations

import numpy as np

from p1_qc.deep_training import (
    balanced_window_specs,
    inference_window_specs,
    predefined_search_space,
    robust_fit,
    robust_transform,
)


def test_windows_never_cross_segments() -> None:
    labels = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    segments = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    specifications = balanced_window_specs(
        labels, segments, window_steps=4, negative_ratio=1.0, seed=7
    )
    assert specifications
    assert all(len(set(segments[item.start : item.stop])) == 1 for item in specifications)
    inference = inference_window_specs(segments, window_steps=3, stride_steps=2)
    assert all(len(set(segments[item.start : item.stop])) == 1 for item in inference)


def test_fold_robust_normalization_is_finite() -> None:
    values = np.array([[1.0, np.nan], [2.0, np.inf], [100.0, 5.0]], dtype=np.float32)
    center, scale = robust_fit(values)
    transformed = robust_transform(values, center, scale)
    assert np.isfinite(transformed).all()


def test_each_deep_search_space_has_twelve_configs() -> None:
    assert len(predefined_search_space("tcn", causal=False)) == 12
    assert len(predefined_search_space("patch_transformer", causal=False)) == 12
